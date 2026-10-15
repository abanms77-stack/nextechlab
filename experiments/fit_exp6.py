"""Fit a law  P(correct) = f(D, S, L)  with interaction terms (Experiment 6).

Reads results/exp6_factorial.csv and:
  1. prints a per-cell accuracy table
  2. fits a logistic regression with main effects + interactions
  3. reports 5-fold cross-validated accuracy
  4. runs out-of-distribution tests (hold out a similarity level, a decoy
     count, or a length) and reports held-out accuracy
  5. saves summary tables + a predicted-vs-observed figure

Usage:
    python experiments/fit_exp6.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
S_W = {"low": 0.1, "medium": 0.5, "high": 1.0}
FEATS = ["log2L", "D", "S", "D*S", "D*log2L", "S*log2L", "D*S*log2L"]


def make_features(df):
    L = np.log2(df["length"].to_numpy(float))
    D = df["n_distractors"].to_numpy(float)
    S = df["similarity"].map(S_W).to_numpy(float)
    return np.column_stack([L, D, S, D * S, D * L, S * L, D * S * L])


def fit_and_score(train, test, tag):
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(make_features(train), train["correct"].to_numpy(int))
    pred = clf.predict(make_features(test))
    actual = test["correct"].to_numpy(int)
    acc = float((pred == actual).mean())
    print(f"  OOD [{tag:<28}] n_train={len(train):<4} n_test={len(test):<4} "
          f"held-out acc={acc:.3f}")
    return acc


def main():
    df = pd.read_csv(RESULTS / "exp6_factorial.csv")
    print(f"rows={len(df)}  cells={df.groupby(['length','similarity','n_distractors']).ngroups}")

    cell = (df.groupby(["length", "similarity", "n_distractors"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))
    print("\n=== per-cell accuracy ===")
    print(cell.to_string(index=False))
    cell.to_csv(RESULTS / "summary_exp6_cells.csv", index=False)

    X = make_features(df)
    y = df["correct"].to_numpy(int)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    clf.fit(X, y)
    print("\n=== logistic fit P = f(log2L, D, S, interactions) ===")
    print("  coefficients:", {k: round(v, 3) for k, v in zip(FEATS, clf.coef_[0])})
    print(f"  intercept: {clf.intercept_[0]:.3f}")
    print(f"  5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")

    print("\n=== out-of-distribution tests ===")
    res = {}
    res["hold out S=medium"] = fit_and_score(
        df[df.similarity != "medium"], df[df.similarity == "medium"],
        "train low+high -> medium")
    res["hold out S=high"] = fit_and_score(
        df[df.similarity != "high"], df[df.similarity == "high"],
        "train low+medium -> high")
    res["hold out D=8"] = fit_and_score(
        df[df.n_distractors != 8], df[df.n_distractors == 8],
        "train D!=8 -> D=8")
    res["hold out L=1024"] = fit_and_score(
        df[df.length == 256], df[df.length == 1024],
        "train L=256 -> L=1024")

    # predicted vs observed per cell
    cell2 = cell.copy()
    cell2["pred"] = clf.predict_proba(make_features(cell2))[:, 1]
    r = float(np.corrcoef(cell2["acc"], cell2["pred"])[0, 1])
    mae = float(np.abs(cell2["acc"] - cell2["pred"]).mean())
    print(f"\n=== per-cell fit quality ===\n  pearson r = {r:.3f}   MAE = {mae:.3f}")
    print(cell2.to_string(index=False))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(cell2["acc"], cell2["pred"], s=45, c=cell2["length"], cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("observed accuracy (cell)")
    ax.set_ylabel("predicted accuracy (model)")
    ax.set_title("P(correct) = f(L, D, S): predicted vs observed")
    ax.grid(alpha=0.3)
    for _, r in cell2.iterrows():
        ax.text(r["acc"], r["pred"], f"L{int(r['length'])}/{r['similarity'][:1]}/D{int(r['n_distractors'])}",
                fontsize=5.5, alpha=0.75)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_exp6_fit.png", dpi=120)
    cell2.to_csv(RESULTS / "summary_exp6_fit.csv", index=False)
    print("\nsaved figure_exp6_fit.png")


if __name__ == "__main__":
    main()
