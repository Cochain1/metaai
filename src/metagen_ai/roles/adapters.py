# src/metagen_ai/roles/adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os, json

from metagen_ai.roles.schema import RoleProfile

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

@dataclass
class LoRAHyper:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None   # e.g., ["q_proj","v_proj"]

@dataclass
class AdapterTrainingCfg:
    base_model: str = "Qwen2.5-1.5B-Instruct"   # freely replace with llama/mistral/qwen that you have
    output_dir: str = "data/adapters"
    lr: float = 2e-4
    bs: int = 2
    grad_accum: int = 8
    max_steps: int = 200
    fp16: bool = True
    logging_steps: int = 10

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _format_sft_examples(role: RoleProfile, examples: List[Dict[str, Any]]):
    out = []
    for ex in examples:
        task = ex.get("task") or {}
        inputs = ex.get("inputs") or {}
        expected = ex.get("expected") or ""
        system = role.system_template
        user = role.user_template.format(task=task, inputs=inputs, prev_summary="")
        out.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(expected)}
        ]})
    return out

def fit_lora_adapter(role: RoleProfile,
                     train_examples: List[Dict[str, Any]],
                     lora: LoRAHyper = LoRAHyper(),
                     cfg: AdapterTrainingCfg = AdapterTrainingCfg()) -> str:
    """
    Train a real LoRA adapter and save it to output_dir/role.name.
    """
    if not train_examples:
        raise ValueError("train_examples is empty")

    save_dir = os.path.join(cfg.output_dir, role.name)
    _ensure_dir(save_dir)

    dataset = _format_sft_examples(role, train_examples)

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, trust_remote_code=True)
    peft_conf = LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                           bias="none", task_type="CAUSAL_LM", target_modules=lora.target_modules)
    model = get_peft_model(model, peft_conf)

    class _Dataset:
        def __init__(self, arr): self.arr = arr
        def __len__(self): return len(self.arr)
        def __getitem__(self, i): return self.arr[i]

    def _collate(batch):
        # Simple chat templating → single sequence with masked prompt tokens.
        import torch
        input_ids_list, labels_list = [], []
        for item in batch:
            msgs = item["messages"]
            prompt = ""
            for m in msgs[:-1]:
                prompt += f"<|{m['role']}|>: {m['content']}\n"
            prompt += "<|assistant|>: "
            out_text = msgs[-1]["content"]

            ids = tok(prompt, return_tensors="pt").input_ids[0]
            lbl = tok(out_text, return_tensors="pt").input_ids[0]
            input_ids = torch.cat([ids, lbl], dim=0)
            labels = torch.cat([torch.full_like(ids, -100), lbl], dim=0)

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        max_len = max(t.size(0) for t in input_ids_list)
        def pad(t):
            import torch
            pad_len = max_len - t.size(0)
            return torch.cat([torch.full((pad_len,), tok.pad_token_id, dtype=torch.long), t], dim=0)

        input_ids = pad(input_ids_list[0]).unsqueeze(0)
        labels = pad(labels_list[0]).unsqueeze(0)
        for i in range(1, len(input_ids_list)):
            input_ids = torch.cat([input_ids, pad(input_ids_list[i]).unsqueeze(0)], dim=0)
            labels = torch.cat([labels, pad(labels_list[i]).unsqueeze(0)], dim=0)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": (input_ids != tok.pad_token_id).long()}

    args = TrainingArguments(
        output_dir=save_dir, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.bs, gradient_accumulation_steps=cfg.grad_accum,
        max_steps=cfg.max_steps, logging_steps=cfg.logging_steps,
        fp16=cfg.fp16, save_steps=cfg.max_steps, report_to=[])

    trainer = Trainer(model=model, args=args, train_dataset=_Dataset(dataset), data_collator=_collate)
    trainer.train()
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    # Side manifest（用于推理端检查）
    with open(os.path.join(save_dir, "adapter_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"role": role.name, "base_model": cfg.base_model}, f, ensure_ascii=False, indent=2)
    return save_dir

def attach_lora_to_role(role: RoleProfile, adapter_dir: str) -> None:
    role.description += f" [adapter:{adapter_dir}]"
