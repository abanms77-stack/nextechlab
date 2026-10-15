"""Phase 7 - REAL RAG intervention benchmark (analyzer).

Reads results/rag_bench.csv (produced by run_rag_bench.py) and reports, per
frozen condition (BASELINE | CONTEXTLAB | RANDOM):

    accuracy, context tokens, mean latency, passages retained
and the headline deltas:
    CONTEXTLAB - BASELINE   (accuracy and tokens)
    CONTEXTLAB - RANDOM     (accuracy)

Paired comparison per query (same query across conditions), with a bootstrap CI
on the accuracy delta. No number is invented here.

Usage:
    python experiments/analyze_rag_bench.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def boot_ci(x, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    stats = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    path = RESULTS / "rag_bench.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run experiments/run_rag_bench.py first")
    df = pd.read_csv(path)

    print("=== REAL RAG intervention benchmark ===")
    print(f"queries = {df['qid'].nunique()}  rows = {len(df)}")
    print(f"model answers restricted to the 12-word candidate set (K-way)\n")

    summary = (df.groupby("condition")
               .agg(accuracy=("correct", "mean"),
                    ctx_tokens=("ctx_tokens", "mean"),
                    latency_s=("seconds", "mean"),
                    n_passages=("n_passages", "mean"),
                    n_removed=("n_removed", "mean"))
               .reindex(["BASELINE", "CONTEXTLAB", "RANDOM"]))
    print(summary.round(4).to_string())

    piv = df.pivot_table(index="qid", columns="condition",
                         values=["correct", "ctx_tokens"])
    base_acc = piv[("correct", "BASELINE")]
    cl_acc = piv[("correct", "CONTEXTLAB")]
    rn_acc = piv[("correct", "RANDOM")]
    base_tok = piv[("ctx_tokens", "BASELINE")]
    cl_tok = piv[("ctx_tokens", "CONTEXTLAB")]

    d_acc = (cl_acc - base_acc).dropna()
    d_acc_rn = (cl_acc - rn_acc).dropna()
    d_tok = (cl_tok - base_tok).dropna()

    print("\n=== headline deltas (paired per query) ===")
    print(f"ContextLab - Baseline  acc = {d_acc.mean():+.3f} "
          f"95% CI {boot_ci(d_acc, np.mean)}")
    print(f"ContextLab - Random    acc = {d_acc_rn.mean():+.3f} "
          f"95% CI {boot_ci(d_acc_rn, np.mean)}")
    tok_pct = (d_tok / base_tok.replace(0, np.nan)).dropna() * 100
    print(f"ContextLab - Baseline  ctx tokens = {d_tok.mean():+.1f} "
          f"({tok_pct.mean():+.1f}%)  "
          f"95% CI {boot_ci(d_tok, np.mean)}")

    print("\n=== per-condition accuracy (mean over queries) ===")
    for c in ["BASELINE", "CONTEXTLAB", "RANDOM"]:
        acc = df[df.condition == c]["correct"].mean()
        tok = df[df.condition == c]["ctx_tokens"].mean()
        print(f"  {c:<11} acc={acc:.3f}  ctx_tokens={tok:.1f}")

    summary.to_csv(RESULTS / "rag_bench_summary.csv")
    print(f"\nsaved {RESULTS / 'rag_bench_summary.csv'}")


if __name__ == "__main__":
    main()
