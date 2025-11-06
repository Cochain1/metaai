from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.seed import seed_graph
from metagen_ai.controller.sampler import sample_architecture

def toy_task():
    return {"question": "What is 12 + 21?", "answer": "33"}

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    program = seed_graph(task, cfg)
    # Apply MoE gating to activate a minimal subgraph and set exit node
    program = sample_architecture(program, task, cfg)

    # Run with early_exit enabled
    result = program.run(task=task, rounds=1, early_exit=True)

    print("Final:", result["final"])
    print("Usage:", result["usage"])
    print("Selected nodes:", program.G.graph.get("selected_nodes"))
    print("Scores:", program.G.graph.get("gating_scores"))
    for t in result["traces"]:
        print(f"[{t.node_id}::{t.role}] -> {t.output_preview}")

if __name__ == "__main__":
    main()
