"""Phase 7B - REDUNDANCY intervention benchmark (analyzer).

Reads results/rag7b_bench.csv and reports the PREREGISTERED comparisons:

  C1  CONTEXTLAB - BASELINE  accuracy
  C2  CONTEXTLAB - BASELINE  ctx_tokens
  C3  CONTEXTLAB - RANDOM    accuracy
  C4  CONTEXTLAB - ORACLE    accuracy
  C5  CONTEXTLAB - ORACLE    ctx_tokens
  W   all of the above WITHIN each context type (clean/redundant/
      contradictory/mixed)

Paired per (query, context_type) with a bootstrap CI on the mean delta.
Also emits the Pareto frontier (accuracy vs ctx_tokens) and a figure.

Usage:
    python experiments/analyze_rag_bench_7b.py
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
CT_ORDER = ["clean", "redundant", "contradictory", "mixed"]
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
    path = RESULTS / "rag7b_bench.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run experiments/run_rag_bench_7b.py first")
    df = pd.read_csv(path)
    n_q = df["qid"].nunique()
    print("=== Phase 7B: redundancy intervention benchmark ===")
    print(f"queries = {n_q}  rows = {len(df)}  "
          f"(context types x conditions x queries)\n")

    # ---- per (context_type, condition) summary ---------------------------
    summ = (df.groupby(["context_type", "condition"])
            .agg(accuracy=("correct", "mean"),
                 ctx_tokens=("ctx_tokens", "mean"),
                 n_passages=("n_passages", "mean"),
                 n_removed=("n_removed", "mean"),
                 latency_s=("seconds", "mean"))
            .reindex(pd.MultiIndex.from_product(
                [CT_ORDER, COND_ORDER], names=["context_type", "condition"])))
    print(summ.round(3).to_string())

    # ---- preregistered comparisons ---------------------------------------
    piv_acc = df.pivot_table(index=["qid", "context_type"],
                             columns="condition", values="correct")
    piv_tok = df.pivot_table(index=["qid", "context_type"],
                             columns="condition", values="ctx_tokens")

    comps = [
        ("C1", "CONTEXTLAB", "BASELINE", "accuracy", piv_acc),
        ("C2", "CONTEXTLAB", "BASELINE", "ctx_tokens", piv_tok),
        ("C3", "CONTEXTLAB", "RANDOM", "accuracy", piv_acc),
        ("C4", "CONTEXTLAB", "ORACLE", "accuracy", piv_acc),
        ("C5", "CONTEXTLAB", "ORACLE", "ctx_tokens", piv_tok),
    ]
    rows = []
    print("\n=== preregistered comparisons (paired, all context types pool) ===")
    for cid, a, b, metric, piv in comps:
        d = (piv[a] - piv[b]).dropna()
        lo, hi = boot_ci(d, np.mean)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        rows.append({"id": cid, "scope": "pooled", "metric": metric,
                     "a": a, "b": b, "delta": round(float(d.mean()), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "sig": sig, "n": int(len(d))})
        print(f"  {cid} {a:<10}-{b:<10} {metric:<11} = {d.mean():+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}]  {sig}  (n={len(d)})")

    # ---- within each context type ----------------------------------------
    print("\n=== within each context type ===")
    for ct in CT_ORDER:
        print(f"-- {ct} --")
        for cid, a, b, metric, piv in comps:
            d = (piv[a] - piv[b]).xs(ct, level="context_type").dropna()
            lo, hi = boot_ci(d, np.mean)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            rows.append({"id": cid, "scope": ct, "metric": metric,
                         "a": a, "b": b, "delta": round(float(d.mean()), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "sig": sig, "n": int(len(d))})
            print(f"   {cid} {a:<10}-{b:<10} {metric:<11} = {d.mean():+.3f}"
                  f" CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    pd.DataFrame(rows).to_csv(RESULTS / "rag7b_comparisons.csv", index=False)
    summ.to_csv(RESULTS / "rag7b_summary.csv")
    print(f"\nsaved {RESULTS / 'rag7b_comparisons.csv'}")
    print(f"saved {RESULTS / 'rag7b_summary.csv'}")

    # ---- Pareto frontier figure ------------------------------------------
    fig, axes = plt.subplots(1, len(CT_ORDER), figsize=(4 * len(CT_ORDER), 4),
                             sharey=True)
    for ax, ct in zip(axes, CT_ORDER):
        sub = summ.xs(ct, level="context_type")
        for cond in COND_ORDER:
            ax.scatter(sub.loc[cond, "ctx_tokens"], sub.loc[cond, "accuracy"],
                       s=70, color=COLORS[cond], label=cond, zorder=3)
        ax.set_title(ct)
        ax.set_xlabel("ctx tokens")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase 7B Pareto: accuracy vs context tokens")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_rag7b_pareto.png", dpi=120)
    print(f"saved {RESULTS / 'figure_rag7b_pareto.png'}")


if __name__ == "__main__":
    main()
