# scripts/single_agent_baseline.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.utils.llm import build_llm_from_cfg, render_system_user

def run_single_agent_benchmark(cfg, datasets, seeds=3):
    import random, csv, os
    llm = build_llm_from_cfg(cfg)
    rows = []
    for seed in range(seeds):
        random.seed(seed)
        for ds in datasets:
            tasks = load_dataset(ds)  # 复用你的加载器
            acc, tokens, calls = 0, 0, 0
            for task in tasks:
                q = task["question"]
                gold = str(task.get("answer", "")).strip()
                msgs = render_system_user(
                    "You are a precise math problem solver. Output strictly 'Final answer: <number>'.",
                    f"{q}\nOnly output one line: Final answer: <number>."
                )
                resp = llm.chat(msgs, temperature=0.1, max_tokens=64)
                text = resp["text"]
                ok = judge_correct(task, text)      # 复用你的判分
                usage = resp.get("usage", {})
                acc += int(ok)
                tokens += usage.get("total_tokens", 0)
                calls += 1
            n = max(1, len(list(load_dataset(ds))))
            rows.append({
                "mode": "single-agent",
                "dataset": ds,
                "seed": seed,
                "accuracy": acc / n,
                "avg_tokens": tokens / max(1, calls),
                "calls": calls
            })

    os.makedirs("logs/metrics", exist_ok=True)
    out_csv = "logs/metrics/single_agent.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode","dataset","seed","accuracy","avg_tokens","calls"])
        w.writeheader(); w.writerows(rows)
    print(f"[single-agent] Saved: {out_csv}")

# ---- 依赖你已有的工具函数：按你的项目名对齐 ----
def load_dataset(name):
    from metagen_ai.datasets.loader import load_dataset as _ld
    return _ld(name)

def judge_correct(task, text):
    from metagen_ai.eval.metrics import judge_correct as _jc
    return _jc(task, text)
