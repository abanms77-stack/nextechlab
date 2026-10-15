"""Phase 7D - conflict-resolution policies (analyzer).

Reads results/rag7d_bench.csv and reports the PREREGISTERED comparisons:

  P1  DROP_WEAKER - PROTECT  accuracy
  P2  ORACLE      - PROTECT  accuracy   (headroom / upper bound)
  P3  SURFACE     - PROTECT  accuracy
  P4  DROP_WEAKER - PROTECT  ctx_tokens
pooled and per context type, with bootstrap CIs and gold-survival per policy.

Usage:
    python experiments/analyze_rag_bench_7d.py
"""

import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POL_ORDER = ["PROTECT", "DROP_WEAKER", "SURFACE", "ORACLE"]
CT_ORDER = ["contradictory", "mixed"]


def boot_ci(x, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    stats = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    path = RESULTS / "rag7d_bench.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run experiments/run_rag_bench_7d.py first")
    df = pd.read_csv(path)
    print("=== Phase 7D: conflict-resolution policies ===")
    print(f"queries = {df['qid'].nunique()}  rows = {len(df)}\n")

    summ = (df.groupby("policy")
            .agg(accuracy=("correct", "mean"),
                 gold_survival=("gold_survived", "mean"),
                 ctx_tokens=("ctx_tokens", "mean"),
                 n_passages=("n_passages", "mean"),
                 n_removed=("n_removed", "mean"),
                 latency_s=("seconds", "mean"))
            .reindex(POL_ORDER))
    print(summ.round(3).to_string())

    print("\nper context type (accuracy):")
    per_ct = df.pivot_table(index="context_type", columns="policy",
                            values="correct", aggfunc="mean").reindex(
        CT_ORDER)[POL_ORDER]
    print(per_ct.round(3).to_string())

    piv_acc = df.pivot_table(index=["qid", "context_type"], columns="policy",
                             values="correct")
    piv_tok = df.pivot_table(index=["qid", "context_type"], columns="policy",
                             values="ctx_tokens")

    comps = [
        ("P1", "DROP_WEAKER", "PROTECT", "accuracy", piv_acc),
        ("P2", "ORACLE", "PROTECT", "accuracy", piv_acc),
        ("P3", "SURFACE", "PROTECT", "accuracy", piv_acc),
        ("P4", "DROP_WEAKER", "PROTECT", "ctx_tokens", piv_tok),
    ]
    rows = []
    print("\n=== preregistered comparisons (pooled, n=80) ===")
    for cid, a, b, metric, piv in comps:
        d = (piv[a] - piv[b]).dropna()
        lo, hi = boot_ci(d, np.mean)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        rows.append({"id": cid, "scope": "pooled", "metric": metric, "a": a,
                     "b": b, "delta": round(float(d.mean()), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "sig": sig, "n": int(len(d))})
        print(f"  {cid} {a:<12}-{b:<8} {metric:<11} = {d.mean():+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n=== within each context type ===")
    for ct in CT_ORDER:
        print(f"-- {ct} --")
        for cid, a, b, metric, piv in comps:
            d = (piv[a] - piv[b]).xs(ct, level="context_type").dropna()
            lo, hi = boot_ci(d, np.mean)
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            rows.append({"id": cid, "scope": ct, "metric": metric, "a": a,
                         "b": b, "delta": round(float(d.mean()), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "sig": sig, "n": int(len(d))})
            print(f"   {cid} {a:<12}-{b:<8} {metric:<11} = {d.mean():+.3f} "
                  f"CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    pd.DataFrame(rows).to_csv(RESULTS / "rag7d_comparisons.csv", index=False)
    summ.to_csv(RESULTS / "rag7d_summary.csv")
    print(f"\nsaved {RESULTS / 'rag7d_comparisons.csv'}")
    print(f"saved {RESULTS / 'rag7d_summary.csv'}")


if __name__ == "__main__":
    main()
