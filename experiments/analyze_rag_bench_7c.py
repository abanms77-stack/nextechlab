"""Phase 7C - choice attribution benchmark (analyzer).

Reads results/rag7c_bench.csv and reports the PREREGISTERED comparisons:

  A1  CONTEXTLAB - RANDOM    accuracy        (better choice?)
  A2  CONTEXTLAB - RANDOM    gold_survival   (less likely to delete the gold)
  A3  CONTEXTLAB - BASELINE  accuracy
  A4  CONTEXTLAB - BASELINE  ctx_tokens
  A5  RANDOM     - BASELINE  accuracy        (does random removal hurt?)

Paired per query with a bootstrap CI on the mean delta. Also reports the
per-condition gold-survival rate and a Pareto figure.

Usage:
    python experiments/analyze_rag_bench_7c.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COND_ORDER = ["BASELINE", "RANDOM", "CONTEXTLAB", "ORACLE"]
COLORS = {"BASELINE": "#888888", "RANDOM": "#c0392b",
          "CONTEXTLAB": "#1f4e79", "ORACLE": "#27ae60"}


def boot_ci(x, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    stats = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    path = RESULTS / "rag7c_bench.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run experiments/run_rag_bench_7c.py first")
    df = pd.read_csv(path)
    print("=== Phase 7C: choice attribution benchmark ===")
    print(f"queries = {df['qid'].nunique()}  rows = {len(df)}\n")

    summ = (df.groupby("condition")
            .agg(accuracy=("correct", "mean"),
                 gold_survival=("gold_survived", "mean"),
                 distractor_survival=("distractor_survived", "mean"),
                 ctx_tokens=("ctx_tokens", "mean"),
                 n_passages=("n_passages", "mean"),
                 n_removed=("n_removed", "mean"),
                 latency_s=("seconds", "mean"))
            .reindex(COND_ORDER))
    print(summ.round(3).to_string())

    piv_acc = df.pivot_table(index="qid", columns="condition",
                             values="correct")
    piv_tok = df.pivot_table(index="qid", columns="condition",
                             values="ctx_tokens")
    piv_gold = df.pivot_table(index="qid", columns="condition",
                              values="gold_survived")

    comps = [
        ("A1", "CONTEXTLAB", "RANDOM", "accuracy", piv_acc),
        ("A2", "CONTEXTLAB", "RANDOM", "gold_survival", piv_gold),
        ("A3", "CONTEXTLAB", "BASELINE", "accuracy", piv_acc),
        ("A4", "CONTEXTLAB", "BASELINE", "ctx_tokens", piv_tok),
        ("A5", "RANDOM", "BASELINE", "accuracy", piv_acc),
    ]
    rows = []
    print("\n=== preregistered comparisons (paired, n=40) ===")
    for cid, a, b, metric, piv in comps:
        d = (piv[a] - piv[b]).dropna()
        lo, hi = boot_ci(d, np.mean)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        rows.append({"id": cid, "a": a, "b": b, "metric": metric,
                     "delta": round(float(d.mean()), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "sig": sig, "n": int(len(d))})
        print(f"  {cid} {a:<10}-{b:<10} {metric:<14} = {d.mean():+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    pd.DataFrame(rows).to_csv(RESULTS / "rag7c_comparisons.csv", index=False)
    summ.to_csv(RESULTS / "rag7c_summary.csv")
    print(f"\nsaved {RESULTS / 'rag7c_comparisons.csv'}")
    print(f"saved {RESULTS / 'rag7c_summary.csv'}")

    # ---- Pareto figure (accuracy vs tokens) ------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cond in COND_ORDER:
        ax.scatter(summ.loc[cond, "ctx_tokens"], summ.loc[cond, "accuracy"],
                   s=90, color=COLORS[cond], label=cond, zorder=3)
    ax.set_xlabel("ctx tokens")
    ax.set_ylabel("accuracy")
    ax.set_title("Phase 7C Pareto: choice attribution (gold always present)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_rag7c_pareto.png", dpi=120)
    print(f"saved {RESULTS / 'figure_rag7c_pareto.png'}")


if __name__ == "__main__":
    main()
