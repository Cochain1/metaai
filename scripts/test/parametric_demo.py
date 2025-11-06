# scripts/parametric_demo.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.feedback.param_textual import build_parametric_hooks

def toy_task():
    return {"question": "Compute 12 + 21. Output only the final number.", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    # 1) Build a task-adaptive program
    program = build_task_graph(task, cfg)
    # 2) Cost-aware gating (activate subgraph + exit)
    program = sample_architecture_costaware(program, task, cfg)

    # 3) Create ParamManager and hook it into the runner
    pm = ParamManager()
    hooks = build_parametric_hooks(program, pm)

    # 4) Run multiple rounds to see parameters evolve
    for r in range(5):
        res = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
        print(f"[Round {r}] Final={res['final']}")
        # Observe current temps and hints for key roles
        for role_name in ["math_simplifier", "calculator", "evaluator"]:
            if role_name in program.role_library:
                rp = program.role_library[role_name]
                print(f"  - {role_name}: T={rp.temperature} hints={pm.prompt_hints.get(role_name, [])}")

if __name__ == "__main__":
    main()
