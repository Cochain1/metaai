# scripts/run_baselines.py
from __future__ import annotations
import argparse, csv, os, time, random
from typing import Dict, Any, Callable

from tqdm import tqdm

from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.utils.llm import build_llm_from_cfg
from metagen_ai.datasets.loader import load_dataset
from metagen_ai.eval.metrics import judge_correct

from metagen_ai.baselines.core import (
    run_cot, run_self_consistency, run_debate, run_tot, run_star_lite
)

BASELINES: Dict[str, Callable] = {
    "cot": run_cot,
    "selfcons": run_self_consistency,
    "debate": run_debate,
    "tot": run_tot,
    "star": run_star_lite,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="gsm8k_test")
    ap.add_argument("--baseline", default="cot", choices=list(BASELINES.keys()))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=-1)
    # 通用超参
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=512)
    # 自一致
    ap.add_argument("--sc_k", type=int, default=5)
    # debate
    ap.add_argument("--debate_rounds", type=int, default=2)
    # ToT
    ap.add_argument("--tot_breadth", type=int, default=3)
    ap.add_argument("--tot_depth", type=int, default=2)
    # 输出
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = bootstrap(args.config)
    os.makedirs("logs/metrics", exist_ok=True)
    out_csv = args.out or f"logs/metrics/baseline_{args.baseline}.csv"

    tasks_all = list(load_dataset(args.dataset))
    tasks = tasks_all if args.max_examples <= 0 else tasks_all[: args.max_examples]
    if not tasks:
        print(f"[WARN] dataset empty: {args.dataset}")
        return

    rows = []
    for seed in range(args.seeds):
        random.seed(seed)
        llm = build_llm_from_cfg(cfg)

        # 选 baseline 函数并封装参数
        def run_one(task: Dict[str,Any]):
            if args.baseline == "cot":
                return run_cot(llm, task, temperature=args.temperature, max_tokens=args.max_tokens)
            if args.baseline == "selfcons":
                return run_self_consistency(llm, task, k=args.sc_k, temperature=max(args.temperature, 0.5), max_tokens=args.max_tokens)
            if args.baseline == "debate":
                return run_debate(llm, task, rounds=args.debate_rounds, temperature=max(args.temperature, 0.5), max_tokens=args.max_tokens)
            if args.baseline == "tot":
                return run_tot(llm, task, breadth=args.tot_breadth, depth=args.tot_depth, temperature=max(args.temperature, 0.4), max_tokens=args.max_tokens)
            if args.baseline == "star":
                return run_star_lite(llm, task, temperature=max(args.temperature, 0.3), max_tokens=args.max_tokens)
            raise ValueError("unknown baseline")

        n_ok, tot_tok, lat_sum = 0, 0, 0.0
        bar = tqdm(total=len(tasks), desc=f"{args.dataset} {args.baseline} seed={seed}", unit="ex")
        for i, task in enumerate(tasks, 1):
            out, usage, dt = run_one(task)
            ok = bool(judge_correct(task, out))
            n_ok += int(ok); tot_tok += usage.get("total_tokens", 0); lat_sum += dt
            bar.set_postfix({"acc": f"{(n_ok/i):.3f}",
                             "avg_tok": f"{(tot_tok/max(1,i)):.1f}",
                             "avg_lat": f"{(lat_sum/max(1,i)):.2f}s"})
            bar.update(1)
        bar.close()

        rows.append({
            "baseline": args.baseline,
            "dataset": args.dataset,
            "seed": seed,
            "accuracy": n_ok / max(1, len(tasks)),
            "avg_tokens": tot_tok / max(1, len(tasks)),
            "avg_latency_s": lat_sum / max(1, len(tasks)),
            "count": len(tasks),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved -> {out_csv}")

if __name__ == "__main__":
    main()
