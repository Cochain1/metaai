# scripts/run_paper_baseline.py
from __future__ import annotations
import argparse, os, csv, json, random
from tqdm import tqdm

from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.utils.llm import build_llm_from_cfg
from metagen_ai.datasets.loader import load_dataset
from metagen_ai.eval.metrics import judge_correct
from metagen_ai.baselines.paperflow import load_paperflow_from_yaml, PaperFlowRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="gsm8k_test")
    ap.add_argument("--flow", required=True, help="configs/paperflows/*.yaml")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--dump_dir", default="logs/gens/paper")
    args = ap.parse_args()

    cfg = bootstrap(args.config)
    os.makedirs("logs/metrics", exist_ok=True)
    out_csv = args.out_csv or f"logs/metrics/paperflow_{os.path.splitext(os.path.basename(args.flow))[0]}.csv"

    flow = load_paperflow_from_yaml(args.flow)

    rows = []
    for seed in range(args.seeds):
        random.seed(seed)
        tasks_all = list(load_dataset(args.dataset))
        tasks = tasks_all if args.max_examples <= 0 else tasks_all[: args.max_examples]
        llm = build_llm_from_cfg(cfg)
        runner = PaperFlowRunner(flow, llm)

        ok, tot_tok, lat = 0, 0, 0.0
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_path = os.path.join(args.dump_dir, f"{os.path.splitext(os.path.basename(args.flow))[0]}_{args.dataset}_seed{seed}.jsonl")
        with open(dump_path, "w", encoding="utf-8") as dumpf:
            bar = tqdm(total=len(tasks), desc=f"{args.dataset} flow={os.path.basename(args.flow)} seed={seed}", unit="ex")
            for i, task in enumerate(tasks, 1):
                final, usage, dt, dbg = runner.run_one(task)
                is_ok = bool(judge_correct(task, final))
                ok += int(is_ok); tot_tok += usage.get("total_tokens",0); lat += dt

                dumpf.write(json.dumps({
                    "idx": i-1, "question": task.get("question",""), "gold": task.get("answer",""),
                    "final": final, "usage": usage, "latency": dt,
                    "history": {str(k): v for k, v in dbg["history"].items()}
                }, ensure_ascii=False) + "\n")

                bar.set_postfix({"acc": f"{(ok/i):.3f}",
                                 "avg_tok": f"{(tot_tok/max(1,i)):.1f}",
                                 "avg_lat": f"{(lat/max(1,i)):.2f}s"})
                bar.update(1)
            bar.close()

        rows.append({
            "flow": os.path.basename(args.flow),
            "dataset": args.dataset,
            "seed": seed,
            "accuracy": ok / max(1, len(tasks)),
            "avg_tokens": tot_tok / max(1, len(tasks)),
            "avg_latency_s": lat / max(1, len(tasks)),
            "count": len(tasks),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved metrics -> {out_csv}")
    print(f"[OK] raw generations -> {dump_path}")

if __name__ == "__main__":
    main()

'''
python scripts/run_paper_baseline.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --flow configs/paperflows/mac_community.yaml \
  --seeds 1 \
  --max_examples -1 \
  --out_csv logs/metrics/paper_mac_community_gsm8k_test.csv

python scripts/run_paper_baseline.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --flow configs/paperflows/star_community.yaml \
  --seeds 1 \
  --max_examples -1 \
  --out_csv logs/metrics/paper_star_community_gsm8k_test.csv
'''