# scripts/role_space_demo.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.roles.generative import generate_and_register_roles
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.sampler import sample_architecture
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook

def task():
    return {"question": "Plan, compute, and verify 12 + 21. Output the final number.", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    t = task()

    # Start from builtin roles, then invent new ones
    role_lib = dict(BUILTIN_ROLES)
    role_lib, selected = generate_and_register_roles(t, cfg, role_library=role_lib)

    print("New roles added:", [s.profile.name for s in selected])
    for s in selected:
        print(f" - {s.profile.name}: diversity={s.diversity_score:.3f} | {s.profile.description}")

    # Build a task-adaptive graph using the augmented role library
    program = build_task_graph(t, cfg, role_library=role_lib)

    # Apply gating (still useful when role set grows)
    program = sample_architecture(program, t, cfg)

    # Optional textual gradient on each round
    tg_hook = build_textual_gradient_hook(program, cfg)

    # Run before pruning
    res1 = program.run(task=t, rounds=1, early_exit=True, hooks=None)
    print("[Before Pruning] Final:", res1["final"])

    # Prune and run again with textual-gradient
    program = prune_once(program, t, cfg)
    res2 = program.run(task=t, rounds=1, early_exit=True,
                       hooks=type("H", (), {"textual_gradient_hook": tg_hook})())
    print("[After  Pruning ] Final:", res2["final"])

if __name__ == "__main__":
    main()
