# src/metagen_ai/training/loop.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json, os, time

from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook, apply_textual_update
from metagen_ai.utils.metrics import accuracy_from_final

@dataclass
class TrainerConfig:
    steps: int = 200
    prune_every: int = 5            # how often to run one-shot pruning
    early_exit: bool = True
    log_dir: str = "logs"
    log_file: str = "train_self_evolve.jsonl"
    save_every: int = 50            # how often to flush summaries
    # controller knobs can be adjusted in configs/default.yaml
    # feedback knobs live in cfg["feedback"]

class SelfEvolveTrainer:
    """
    Minimal trainer that iterates over a task stream and updates the meta-parameters
    via rule-based textual gradients (temperature/topology nudges).
    """
    def __init__(self, cfg: Dict[str, Any], dataset, role_library=None):
        self.cfg = cfg
        self.dataset = dataset
        self.role_library = role_library  # optional pre-augmented role set

        # Prepare log path
        log_dir = cfg.get("logging", {}).get("dir", None) or cfg.get("runtime", {}).get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, cfg.get("training", {}).get("log_file", "train_self_evolve.jsonl"))

        self.stats = {
            "steps": 0,
            "correct": 0,
            "tokens": 0,
            "calls": 0,   # approx: number of node outputs per step if tracked
        }

    def step(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Build task-adaptive program (roles may be augmented upstream)
        program = build_task_graph(task, self.cfg, role_library=self.role_library)

        # 2) Cost-aware gating (activates subgraph + exit)
        program = sample_architecture_costaware(program, task, self.cfg)

        # 3) Optional periodic one-shot pruning
        if self.cfg.get("pruning", {}).get("enable", True) and (self.stats["steps"] % max(1, self.cfg.get("training", {}).get("prune_every", 5)) == 0):
            program = prune_once(program, task, self.cfg)

        # 4) Attach textual-gradient hook for per-round updates
        tg_hook = build_textual_gradient_hook(program, self.cfg)

        # 5) Execute a single round
        result = program.run(task=task, rounds=1, early_exit=self.cfg.get("controller", {}).get("early_exit", True),
                             hooks=type("H", (), {"textual_gradient_hook": tg_hook})())

        # 6) Optionally apply another textual update using final result (idempotent)
        apply_textual_update(program, result, self.cfg)

        # 7) Metrics
        acc = accuracy_from_final(result.get("final", ""), task.get("answer", ""))
        usage = result.get("usage", {})
        record = {
            "step": int(self.stats["steps"]),
            "question": task.get("question"),
            "answer": task.get("answer"),
            "final": result.get("final"),
            "acc": float(acc),
            "usage": usage,
            "selected_nodes": program.G.graph.get("selected_nodes"),
            "lambda_cost": program.G.graph.get("lambda_cost"),
            "pruning_kept_edges": [(u, v) for (u, v) in (program.G.graph.get("kept_edges") or [])],
        }
        # Update running stats
        self.stats["steps"] += 1
        self.stats["correct"] += int(acc)
        self.stats["tokens"] += int(usage.get("total_tokens", 0))
        self.stats["calls"] += 1

        return record

    def train(self, tcfg: Optional[TrainerConfig] = None) -> Dict[str, Any]:
        tcfg = tcfg or TrainerConfig(
            steps=int(self.cfg.get("training", {}).get("steps", 200)),
            prune_every=int(self.cfg.get("training", {}).get("prune_every", 5)),
            early_exit=bool(self.cfg.get("controller", {}).get("early_exit", True)),
            log_dir=self.cfg.get("runtime", {}).get("log_dir", "logs"),
            log_file=self.cfg.get("training", {}).get("log_file", "train_self_evolve.jsonl"),
            save_every=int(self.cfg.get("training", {}).get("save_every", 50)),
        )
        stream = self.dataset.stream()
        # Ensure directory exists
        os.makedirs(tcfg.log_dir, exist_ok=True)
        log_fp = open(self.log_path, "a", encoding="utf-8")

        since = time.time()
        for _ in range(tcfg.steps):
            task = next(stream)
            rec = self.step(task)

            log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if self.stats["steps"] % tcfg.save_every == 0:
                log_fp.flush()

        log_fp.flush()
        log_fp.close()

        elapsed = time.time() - since
        summary = {
            "steps": self.stats["steps"],
            "acc@mean": round(self.stats["correct"] / max(1, self.stats["steps"]), 4),
            "tokens@sum": int(self.stats["tokens"]),
            "elapsed_sec": round(elapsed, 2),
            "log_path": self.log_path,
        }
        return summary
