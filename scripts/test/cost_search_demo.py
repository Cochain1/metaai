# scripts/cost_search_demo.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.roles.generative import generate_and_register_roles
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.utils.metrics import summarize_usage, accuracy_from_final

def task():
    return {"question": "Compute 12 + 21 and provide only the number.", "answer": "33"}

def run_once(cfg, lambda_cost: float, threshold: float):
    # Copy cfg and set parameters
    cfg = dict(cfg)
    cfg.setdefault("controller", {})
    cfg["controller"] = dict(cfg["controller"])
    cfg["controller"]["cost_weight"] = lambda_cost
    cfg["controller"]["threshold"] = threshold

    t = task()
    # Start with builtin roles, optionally add generative roles to stress the controller
    role_lib = dict(BUILTIN_ROLES)
    role_lib, _ = generate_and_register_roles(t, cfg, role_library=role_lib)

    program = build_task_graph(t, cfg, role_library=role_lib)
    program = sample_architecture_costaware(program, t, cfg)

    # Optional pruning for extra frugality
    if (cfg.get("pruning", {}) or {}).get("enable", True):
        program = prune_once(program, t, cfg)

    res = program.run(task=t, rounds=1, early_exit=True)
    acc = accuracy_from_final(res["final"], t["answer"])
    usage = res["usage"]
    return acc, usage, program.G.graph.get("selected_nodes"), program.G.graph.get("gating_combined")

def main():
    cfg = bootstrap("configs/default.yaml")
    lambdas = [0.0, 0.2, 0.4, 0.6]      # stronger penalty → smaller subgraph
    thresholds = [0.5, 0.7, 0.9]        # higher → keep more nodes

    results = []
    for lam in lambdas:
        for th in thresholds:
            acc, usage, picked, comb = run_once(cfg, lam, th)
            results.append((lam, th, acc, usage.get("total_tokens", 0), picked))

    # Print a compact table
    print("lambda\tthr\tacc\ttokens\tselected_nodes")
    for lam, th, acc, toks, picked in results:
        print(f"{lam:.2f}\t{th:.2f}\t{acc:.2f}\t{toks}\t{picked}")

if __name__ == "__main__":
    main()
