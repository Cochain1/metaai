# scripts/peft_train_and_run.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.roles.schema import RoleProfile
from metagen_ai.roles.adapters import fit_lora_adapter, attach_lora_to_role, AdapterTrainingCfg, LoRAHyper
from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.graph_ops.softmask import SoftMaskManager, SoftMaskConfig
from metagen_ai.feedback.soft_textual import build_softmask_textual_hook
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.feedback.param_textual import build_parametric_hooks
from metagen_ai.pruning.one_shot import prune_once

def main():
    cfg = bootstrap("configs/default.yaml")

    # 1) 定义一个要 LoRA 化的新角色（用少量样本）
    rp = RoleProfile(
        name="retrieval_specialist",
        description="Retrieve and quote concise, relevant facts from the task/context.",
        system_template="You retrieve concise, relevant snippets; do not speculate.",
        user_template="Task: {task[question]}\nUpstream: {inputs}\nReturn 2-3 short relevant facts if available."
    )
    examples = [
        {"task":{"question":"What is 12 + 21?"}, "inputs":{}, "expected":"12 and 21 are the addends. The sum is 33."},
        {"task":{"question":"Add 19 and 24"}, "inputs":{}, "expected":"19 + 24 = 43."},
    ]
    path = fit_lora_adapter(rp, examples, LoRAHyper(r=8, alpha=16, dropout=0.05),
                            AdapterTrainingCfg(base_model=cfg["llm"]["model"],
                                               output_dir="data/adapters",
                                               max_steps=20, bs=2, grad_accum=8, lr=2e-4, fp16=True))
    attach_lora_to_role(rp, path)

    # 2) 将该角色注入角色库，并切换 LLM provider 为 peft_local
    role_lib = dict(BUILTIN_ROLES)
    role_lib[rp.name] = rp

    cfg["llm"]["provider"] = "peft_local"
    cfg["llm"]["adapter_dir"] = path

    # 3) 构图 + 成本感知 gating
    task = {"question": "Plan, compute and verify 19 + 24. Output the final number.", "answer": "43"}
    program = build_task_graph(task, cfg, role_library=role_lib)
    program = sample_architecture_costaware(program, task, cfg)

    # 4) 结构软门 + 参数化联动
    smm = SoftMaskManager(program, SoftMaskConfig(tau_node=0.8, tau_edge=0.8, lr_node=0.25, lr_edge=0.25))
    smm.forward(write_active=True)
    softmask_hook = build_softmask_textual_hook(smm)
    pm = ParamManager()
    param_hooks = build_parametric_hooks(program, pm)

    class Combined:
        def before_node(self, node_id, ctx):
            if hasattr(param_hooks, "before_node"):
                param_hooks.before_node(node_id, ctx)
        def textual_gradient_hook(self, payload):
            if hasattr(param_hooks, "textual_gradient_hook"):
                param_hooks.textual_gradient_hook(payload)
            softmask_hook(payload)
    hooks = Combined()

    # 5) 多轮在线自进化 + 一拍裁剪
    for r in range(4):
        res = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
        print(f"[Round {r}] Final={res['final']}")

    program = prune_once(program, task, cfg)
    resF = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
    print("[Post-Prune] Final:", resF["final"])

if __name__ == "__main__":
    main()
