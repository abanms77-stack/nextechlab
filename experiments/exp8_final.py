"""Exp 8 final - continuous similarity across models (Qwen + gpt2-medium).

Produces:
    results/figure_exp8_models.png   accuracy vs continuous similarity, 2 models
    results/exp8_metric_table.csv    metric comparison per model

Usage:
    python experiments/exp8_final.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = {
    "Qwen2.5-0.5B": "exp8b_qwen_d4.csv",
    "gpt2-medium": "exp8b_gpt2-medium_d4.csv",
}


def main():
    met = pd.read_csv(RESULTS / "exp8_similarity_metrics.csv")
    rows = []
    series = {}
    for mname, fname in MODELS.items():
        obs = pd.read_csv(RESULTS / fname)
        acc = (obs.groupby("label", as_index=False)
               .agg(n=("correct", "size"), acc=("correct", "mean")))
        df = acc.merge(met, on="label")
        series[mname] = df
        for metric in ["cos_sim", "jaccard", "shared_prefix"]:
            r = pearsonr(df[metric], df["acc"])
            rho = spearmanr(df[metric], df["acc"])
            rows.append({"model": mname, "metric": metric,
                         "pearson": round(r.statistic, 3),
                         "spearman": round(rho.statistic, 3)})

    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "exp8_metric_table.csv", index=False)
    print("=== metric comparison (continuous similarity -> accuracy) ===")
    print(tab.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {"Qwen2.5-0.5B": "#1f4e79", "gpt2-medium": "#b5651d"}
    for mname, df in series.items():
        df = df.sort_values("jaccard")
        ax.scatter(df["jaccard"], df["acc"], s=45, label=mname,
                   color=colors[mname])
    ax.set_xlabel("lexical Jaccard similarity (target sentence vs decoy)")
    ax.set_ylabel("accuracy")
    ax.set_title("Continuous similarity predicts interference on 2 model families "
                 "(D=4)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_exp8_models.png", dpi=120)
    print("saved figure_exp8_models.png")


if __name__ == "__main__":
    main()