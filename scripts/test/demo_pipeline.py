# scripts/demo_pipeline.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.seed import seed_graph
from metagen_ai.controller.sampler import sample_architecture
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook

def toy_task():
    return {"question": "What is 12 + 21?", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    # 1) Initialize program from architect
    program = seed_graph(task, cfg)

    # 2) MoE gating + mark exit
    program = sample_architecture(program, task, cfg)

    # 3) Attach textual-gradient hook
    tg_hook = build_textual_gradient_hook(program, cfg)

    # 4) First run (pre-pruning)
    result_before = program.run(task=task, rounds=1, early_exit=True, hooks=None)
    print("[Before Pruning] Final:", result_before["final"])
    print("[Before Pruning] Selected nodes:", program.G.graph.get("selected_nodes"))

    # 5) One-shot pruning (uses warmup internally)
    program = prune_once(program, task, cfg)

    # 6) Second run with textual-gradient hook to slightly update temps/topology
    result_after = program.run(task=task, rounds=1, early_exit=True,
                               hooks=type("H", (), {"textual_gradient_hook": tg_hook})())

    print("[After  Pruning] Final:", result_after["final"])
    print("[Diag] Kept edges:", program.G.graph.get("kept_edges"))
    print("[Diag] Edge scores (truncated):",
          dict(list(program.G.graph.get("edge_scores", {}).items())[:5]))

if __name__ == "__main__":
    main()
