"""Phase A analysis - does D x S survive on the counting task?

Fits P = f(D, S, L) on counting-task data and compares the D*S coefficient with
the retrieval-task fit. Also correlates accuracy with the joint predictor on
both tasks (a cross-task generalisation check).

Usage:
    python experiments/analyze_taskA.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# measured S per level for each task family
S_RETRIEVAL = {"low": 0.1, "medium": 0.5, "high": 1.0}   # as used in Exp 6
S_COUNTING = {"low": 0.184, "medium": 0.626, "high": 0.836}  # RGB-based


def feats(df, smap):
    L = np.log2(df["length"].to_numpy(float))
    D = df["n_distractors"].to_numpy(float)
    S = df["similarity"].map(smap).to_numpy(float)
    return np.column_stack([L, D, S, D * S])


def main():
    count = pd.read_csv(RESULTS / "expA_counting.csv")
    print(f"counting rows={len(count)}  cells={count.groupby(['length','similarity','n_distractors']).ngroups}")

    cell = (count.groupby(["length", "similarity", "n_distractors"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean"),
                 S=("interference_load", "first")))
    print("\n=== counting-task accuracy (D x S) ===")
    piv = cell.pivot_table(index=["similarity", "n_distractors"], columns="length",
                           values="acc")
    print(piv.round(2).to_string())
    cell.to_csv(RESULTS / "summary_expA.csv", index=False)

    # fit on counting data
    y = count["correct"].to_numpy(int)
    Xc = feats(count, S_COUNTING)
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(Xc, y)
    print("\n=== counting-task fit P=f(log2L, D, S, D*S) ===")
    print("  " + "  ".join(f"{n}={v:+.3f}" for n, v in
                           zip(["log2L", "D", "S", "D*S"], clf.coef_[0])))

    # retrieval fit for comparison
    ret = pd.read_csv(RESULTS / "exp6_factorial.csv")
    clf_r = LogisticRegression(max_iter=4000, C=1.0).fit(
        feats(ret, S_RETRIEVAL), ret["correct"].to_numpy(int))
    print("  retrieval D*S = %+.3f   counting D*S = %+.3f"
          % (clf_r.coef_[0][3], clf.coef_[0][3]))

    # cross-task: predict counting cells using the RETRIEVAL-fitted law
    pred = clf_r.predict_proba(Xc)[:, 1]
    pcell = np.array([pred[count["length"].to_numpy() == L].mean()
                      for L in [256, 1024]])  # placeholder
    r_all = pearsonr(pred, y).statistic
    print(f"\ncross-task (retrieval law -> counting trials) corr = {r_all:+.3f}")

    # figure
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for lvl, g in cell.groupby("similarity"):
        g = g.sort_values("n_distractors")
        ax.plot(g["n_distractors"], g["acc"], "o-",
                label=f"S={lvl} (S_meas~{S_COUNTING[lvl]:.2f})")
    ax.set_xlabel("number of similar-colour items (D)")
    ax.set_ylabel("accuracy (count = three)")
    ax.set_title("Cross-task: counting task, interference by D and S")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_taskA_counting.png", dpi=120)
    print("saved figure_taskA_counting.png")


if __name__ == "__main__":
    main()