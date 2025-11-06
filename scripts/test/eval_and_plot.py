# scripts/eval_and_plot.py
# Runs two suites and produces a CSV and a simple matplotlib plot (default colors).
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.eval.suites import gsm8k_tiny, basic_arith_n, run_suite

import matplotlib.pyplot as plt
import csv, os

def main():
    cfg = bootstrap("configs/default.yaml")

    out_dir = "logs/metrics"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Tiny GSM8K-like slice
    csv_gsm = os.path.join(out_dir, "gsm8k_tiny.csv")
    m1 = run_suite(gsm8k_tiny(), cfg, csv_gsm)

    # 2) Basic arithmetic grid
    csv_arith = os.path.join(out_dir, "arith_grid.csv")
    m2 = run_suite(basic_arith_n(10, 20), cfg, csv_arith)

    # Plot a simple bar chart (accuracy)
    labels = ["gsm8k_tiny", "arith_grid"]
    accs = [m1["accuracy"], m2["accuracy"]]
    plt.figure()
    plt.bar(labels, accs)  # default colors, single plot, no style set
    plt.title("Accuracy by Suite")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    fig_path = os.path.join(out_dir, "accuracy_bar.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print("Saved:", csv_gsm, csv_arith, fig_path)

if __name__ == "__main__":
    main()
