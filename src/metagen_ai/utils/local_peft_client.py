# src/metagen_ai/utils/local_peft_client.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class LocalPEFTClient:
    """
    Minimal chat client that loads base model + PEFT LoRA and runs generation on GPU.
    """
    def __init__(self, base_model: str, adapter_dir: str, device: Optional[str] = None, dtype: str = "bfloat16"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        # dtype cast
        if self.device == "cuda":
            if dtype == "float16":
                self.model = self.model.half().to(self.device)
            elif dtype == "bfloat16":
                self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
            else:
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _render(self, messages: List[Dict[str, str]]) -> str:
        # Simple chat prompt rendering
        s = ""
        for m in messages:
            s += f"<|{m['role']}|>: {m['content']}\n"
        s += "<|assistant|>: "
        return s

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 256) -> Dict[str, Any]:
        prompt = self._render(messages)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        gen = self.model.generate(
            **ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(0.01, temperature),
            top_p=0.9 if temperature > 0.0 else 1.0,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id
        )
        out = self.tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        usage = {"prompt_tokens": int(ids["input_ids"].numel()),
                 "completion_tokens": int(gen[0].numel() - ids["input_ids"].numel()),
                 "total_tokens": int(gen[0].numel())}
        return {"text": out, "usage": usage}
