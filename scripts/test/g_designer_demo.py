# scripts/g_designer_demo.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.sampler import sample_architecture
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook

def toy_task():
    # You can change this to test different graphs (reasoning, retrieval, coding, etc.)
    return {"question": "Add 12 and 21. Provide the final result.", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    program = build_task_graph(task, cfg)  # <- G-Designer constructor
    # Optional gating (often redundant if max_nodes is already small)
    program = sample_architecture(program, task, cfg)

    # Optional textual gradient on each round
    tg_hook = build_textual_gradient_hook(program, cfg)

    # Run before pruning
    res1 = program.run(task=task, rounds=1, early_exit=True, hooks=None)
    print("[G-Designer] Final (before pruning):", res1["final"])

    # One-shot prune
    program = prune_once(program, task, cfg)

    # Run after pruning with textual-gradient applied
    # NOTE: use staticmethod to avoid binding the function as a method (no extra 'self')
    hooks = type("H", (), {
        "textual_gradient_hook": staticmethod(tg_hook),
        # (optional) you can add no-op hooks to be extra safe:
        # "before_node": staticmethod(lambda node_id, ctx: None),
        # "after_node": staticmethod(lambda node_id, out: None),
    })()

    res2 = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
    print("[G-Designer] Final (after pruning):", res2["final"])
    print("Kept edges:", program.G.graph.get("kept_edges"))

if __name__ == "__main__":
    main()
