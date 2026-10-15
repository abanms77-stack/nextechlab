"""Exp 8 step 3 - which measurable similarity metric predicts interference?

Merges the pre-registered similarity metrics with the measured accuracy and
tests each candidate metric (cos_sim, jaccard, shared_prefix) as a predictor of
accuracy. Reports Pearson r, Spearman rho and the R^2 of a simple monotone fit
per (metric, length), then fits the accuracy-vs-cos_sim curve.

Usage:
    python experiments/exp8_analyze.py [tag]     # tag in {qwen, llama, ...}
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    # Prefer the D=4 run (D=1 was below the detection floor).
    d4 = RESULTS / f"exp8b_{tag}_d4.csv"
    if d4.exists() and len(pd.read_csv(d4)) > 50:
        obs = pd.read_csv(d4)
        obs["length"] = obs["length"].astype(int)
        grid = ["label", "length"]
        dval = int(obs["n_copies"].iloc[0])
        print(f"(using D={dval} run: {d4.name})")
    else:
        obs = pd.read_csv(RESULTS / f"exp8_{tag}.csv")
        grid = ["label", "length"]
        dval = 1
    met = pd.read_csv(RESULTS / "exp8_similarity_metrics.csv")

    acc = (obs.groupby(grid, as_index=False)
           .agg(n=("correct", "size"), acc=("correct", "mean"),
                p_correct=("prob_correct", "mean")))
    df = acc.merge(met, on="label", how="left")

    # single decoy setups: minimum interference density I = D*w(S)/L with D=1
    print(f"=== Exp 8 ({tag}): which similarity metric predicts accuracy? ===")
    for L in sorted(df["length"].unique()):
        sub = df[df.length == L]
        print(f"\n-- L={L} --")
        for m in ["cos_sim", "jaccard", "shared_prefix"]:
            r = pearsonr(sub[m], sub["acc"])
            rho = spearmanr(sub[m], sub["acc"])
            print(f"  {m:<14} pearson={r.statistic:+.3f} (p={r.pvalue:.1e})  "
                  f"spearman={rho.statistic:+.3f}")

    # fit accuracy vs cos_sim
    df.to_csv(RESULTS / f"exp8_summary_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.3))
    for L, g in df.groupby("length"):
        g = g.sort_values("cos_sim")
        ax.scatter(g["cos_sim"], g["acc"], s=40, label=f"L={L}")
        try:
            popt, _ = curve_fit(sigmoid, g["cos_sim"], g["acc"],
                                p0=[10, 0.7], maxfev=20000)
            xs = np.linspace(-0.05, 1.02, 200)
            ax.plot(xs, sigmoid(xs, *popt), "--", alpha=0.8,
                    label=f"fit L={L}: k={popt[0]:.1f}, x0={popt[1]:.2f}")
        except Exception:
            pass
    ax.set_xlabel("cosine similarity  cos(target sentence, decoy sentence)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Continuous similarity vs interference ({tag})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"figure_exp8_similarity_{tag}.png", dpi=120)
    print(f"\nsaved figure_exp8_similarity_{tag}.png")

    # held-out decoy test: fit acc = a + b*metric on all-but-k decoys, predict k
    print("\n=== held-out-decoy CV: does the metric generalise across decoys? ===")
    from sklearn.linear_model import LinearRegression
    for m in ["cos_sim", "jaccard"]:
        errs = []
        idx = np.arange(len(df))
        for test_i in idx:  # leave-one-decoy-out
            tr, te = df.iloc[idx != test_i], df.iloc[[test_i]]
            lr = LinearRegression().fit(tr[[m]], tr["acc"])
            errs.append(abs(float(lr.predict(te[[m]])[0]) - te["acc"].iloc[0]))
        print(f"  {m:<10} leave-one-decoy-out MAE = {np.mean(errs):.3f}")

    # correlation of the law with the CONTINUOUS S = max(0, cos_sim)
    df["S_cont"] = df["cos_sim"].clip(lower=0.0)
    print("\n=== replacement of categorical S with continuous cos_sim ===")
    for L in sorted(df["length"].unique()):
        sub = df[df.length == L]
        print(f"  L={L}: corr(S_cont, acc) = {pearsonr(sub['S_cont'], sub['acc']).statistic:+.3f}")


if __name__ == "__main__":
    main()
