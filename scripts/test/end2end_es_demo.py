# scripts/end2end_es_demo.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.optim.blackbox_es import es_step, ESConfig

def toy_task():
    return {"question": "What is 23 + 18? Output only the number.", "answer": "41"}

def main():
    cfg = bootstrap("configs/default.yaml")
    es = ESConfig(sigma=0.4, lr=0.3, population=8, lam_cost=1e-3)

    # Perform a handful of ES steps to show improvement signal
    task = toy_task()
    for step in range(5):
        out = es_step(task, cfg, es)
        print(f"[ES step {step}] reward={out['reward']:.4f} final={out['final']} active={out['active_counts']}")

if __name__ == "__main__":
    main()
