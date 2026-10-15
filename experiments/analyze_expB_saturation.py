"""Experiment B - analyze the saturation sweep (D-grid characterization).

Reads results/expB_dsweep_factorial.csv (Qwen2.5-0.5B, L=256, S=medium,
D in {0,1,2,3,4,5,6,8,10,12,16}) and asks whether accuracy actually saturates,
and whether D_SATURATION = 5 is empirically supported.

Compares candidate count-term models via a trial-level logistic fit on
[I, I*S, log2L] where I = min(1, D/k):
    (a) linear count:  I = D          (k -> inf)
    (b) configured:    I = min(1, D/5)
    (c) free k:        best k by grid search on held-out CV accuracy

Usage:
    python experiments/analyze_expB_saturation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
S_W = {"low": 0.1, "medium": 0.5, "high": 1.0}


def cv_acc_for_k(df, k):
    D = df["n_distractors"].to_numpy(float)
    S = df["similarity"].map(S_W).to_numpy(float)
    L = np.log2(df["length"].to_numpy(float))
    I = np.minimum(1.0, D / k) if k < 1e8 else D
    X = np.column_stack([I, I * S, L])
    y = df["correct"].to_numpy(int)
    clf = LogisticRegression(max_iter=4000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())


def main():
    path = RESULTS / "expB_dsweep_factorial.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run experiments/run_expB.py first")

    df = pd.read_csv(path)
    df = df[df["sweep"] == "expB_d"] if "sweep" in df.columns else df
    cells = (df.groupby(["length", "similarity", "n_distractors"], as_index=False)
             .agg(n=("correct", "size"), acc=("correct", "mean")))
    print("=== saturation sweep (L=256, S=medium) ===")
    print(cells.to_string(index=False))

    c = cells.sort_values("n_distractors")
    ds = c["n_distractors"].to_numpy()
    acc = c["acc"].to_numpy()
    print("\naccuracy vs D (with step delta):")
    for i, (d, a) in enumerate(zip(ds, acc)):
        dl = "" if i == 0 else f"  (delta {acc[i-1]-a:+.2f})"
        print(f"  D={d:<3} acc={a:.3f}{dl}")

    # Non-monotonicity check: any increase as D grows?
    nonmono = [(int(ds[i]), float(acc[i]), float(acc[i + 1]))
               for i in range(len(acc) - 1) if acc[i + 1] > acc[i] + 1e-9]
    print("\nnon-monotonic steps (acc rises as D grows):")
    if nonmono:
        for d0, a0, a1 in nonmono:
            print(f"  D={d0} acc={a0:.2f} -> D>={d0} acc={a1:.2f}  <-- INVERSION")
    else:
        print("  none")

    # Saturation knee: first step with |delta| < 0.02.
    knee = None
    for i in range(len(acc) - 1):
        if abs(acc[i] - acc[i + 1]) < 0.02:
            knee = int(ds[i])
            break
    print(f"\nfirst |delta| < 0.02 at D = {knee}  (empirical saturation knee)")
    print(f"accuracy at largest D ({int(ds[-1])}) = {acc[-1]:.3f}"
          "  -> does NOT flatten if still well above chance (1/12=0.083)")

    print("\n=== count-term model comparison (5-fold CV accuracy) ===")
    lin = cv_acc_for_k(df, 1e9)
    cfg = cv_acc_for_k(df, 5.0)
    print(f"  linear D (no saturation)   CV acc = {lin:.3f}")
    print(f"  D_SATURATION = 5           CV acc = {cfg:.3f}")
    best_k, best_acc = None, -1.0
    for k in np.arange(1.0, 41.0, 1.0):
        a = cv_acc_for_k(df, k)
        if a > best_acc:
            best_k, best_acc = k, a
    print(f"  best-fit k                 CV acc = {best_acc:.3f}  (k={best_k})")

    print(f"\nVERDICT: configured D_SATURATION=5 vs best-fit k={best_k}"
          f"  (linear CV={lin:.3f})")
    if abs(best_k - 5.0) <= 1.0:
        print("  -> 5 is empirically supported (within +/-1 of best fit).")
    elif lin >= max(cfg, best_acc) - 0.005:
        print("  -> data do NOT saturate: linear count fits as well or better.")
        print("     D_SATURATION=5 is NOT supported by this sweep.")
    else:
        print(f"  -> 5 is NOT the best fit; data prefer k={best_k}.")

    cells.to_csv(RESULTS / "expB_dsweep_cells.csv", index=False)
    print(f"\nsaved {RESULTS / 'expB_dsweep_cells.csv'}")


if __name__ == "__main__":
    main()
