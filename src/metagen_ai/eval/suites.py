# src/metagen_ai/eval/suites.py
from __future__ import annotations
from typing import Dict, List, Callable
import csv, os, time

from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.param_textual import build_parametric_hooks
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.utils.metrics import accuracy_from_final

Task = Dict[str, str]

def gsm8k_tiny() -> List[Task]:
    # A tiny hand-crafted slice to avoid external downloads (replace with real set later).
    return [
        {"question": "What is 7 + 9?", "answer": "16"},
        {"question": "What is 12 + 21?", "answer": "33"},
        {"question": "What is 30 + 14?", "answer": "44"},
        {"question": "What is 19 + 24?", "answer": "43"},
    ]

def basic_arith_n(a: int = 10, b: int = 50) -> List[Task]:
    arr = []
    for x in range(a, a+4):
        for y in range(b, b+4):
            arr.append({"question": f"What is {x} + {y}?", "answer": str(x+y)})
    return arr

def run_suite(tasks: List[Task], cfg: Dict, csv_path: str, apply_prune: bool = True) -> Dict[str, float]:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pm = ParamManager()  # parametric roles hook
    # Evaluate
    total, correct, tokens = 0, 0, 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["idx", "question", "answer", "final", "acc", "tokens", "selected_nodes"])
        for i, t in enumerate(tasks):
            program = build_task_graph(t, cfg)
            program = sample_architecture_costaware(program, t, cfg)
            hooks = build_parametric_hooks(program, pm)  # light parametric evolution during eval
            if apply_prune and (cfg.get("pruning", {}) or {}).get("enable", True):
                program = prune_once(program, t, cfg)
            res = program.run(task=t, rounds=1, early_exit=True, hooks=hooks)
            acc = accuracy_from_final(res["final"], t["answer"])
            total += 1; correct += int(acc); tokens += int(res["usage"].get("total_tokens", 0))
            w.writerow([i, t["question"], t["answer"], res["final"], float(acc), res["usage"].get("total_tokens", 0),
                        program.G.graph.get("selected_nodes")])
    return {"accuracy": correct/max(1,total), "avg_tokens": tokens/max(1,total)}
