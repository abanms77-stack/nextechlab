"""Phase D - statistical finalization.

Gathers publication-grade statistics for the main claims, all from the saved
CSVs, and clearly separates measured quantities from inferred interpretation:

  measured : benchmark accuracies, similarity metrics, fit correlations,
             transfer metrics, intervention gain, calibration
  inferred : the "attention budget" mechanism (stated as an interpretation)

Produces results/final_stats.csv and results/figure_calibration.png.

Usage:
    python experiments/analyze_final.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_exp6 import FEATS, make_features  # noqa: E402

S_RET = {"low": 0.1, "medium": 0.5, "high": 1.0}
ROWS = []


def boot_ci(vals, fn, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(v), len(v))
        try:
            stats.append(fn(v[idx]))
        except Exception:
            pass
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def add(claim, value, lo, hi, kind="measured"):
    ROWS.append({"claim": claim, "value": round(value, 4),
                 "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                 "kind": kind})


def ece(prob, y, bins=10):
    prob, y = np.asarray(prob, float), np.asarray(y, float)
    e = 0.0
    for i in range(bins):
        hi = i == bins - 1
        m = (prob >= i / bins) & ((prob <= (i + 1) / bins) if hi
                                  else (prob < (i + 1) / bins))
        if m.sum():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return e


def main():
    # ---- Exp 6 fit quality (retrieval, Qwen), bootstrap over cells ----
    q = pd.read_csv(RESULTS / "exp6_factorial.csv")
    y = q["correct"].to_numpy(int)
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(make_features(q), y)
    cell = (q.groupby(["length", "similarity", "n_distractors"], as_index=False)
            .agg(acc=("correct", "mean")))
    cell["pred"] = clf.predict_proba(make_features(cell))[:, 1]
    r = pearsonr(cell["pred"], cell["acc"]).statistic
    lo, hi = boot_ci(cell[["pred", "acc"]].to_numpy(), lambda a: pearsonr(a[:, 0], a[:, 1]).statistic)
    add("Exp6 cell fit Pearson r (Qwen)", r, lo, hi)
    # bootstrap the D*S coefficient (resample samples, refit)
    Xq = make_features(q)
    rng = np.random.default_rng(1)
    bcoef = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        try:
            bcoef.append(LogisticRegression(max_iter=3000, C=1.0)
                         .fit(Xq[idx], y[idx]).coef_[0][3])
        except Exception:
            pass
    add("Exp6 D*S coefficient", clf.coef_[0][3],
        float(np.percentile(bcoef, 2.5)), float(np.percentile(bcoef, 97.5)))

    # ---- Transfer (frozen Qwen -> Llama), bootstrap over cells ----
    frozen = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    coefs = np.array([frozen["coefs"][f] for f in FEATS])
    pred = pd.read_csv(RESULTS / "pred_qwen_on_llama.csv")
    obs = pd.read_csv(RESULTS / "exp7_llama_factorial.csv")
    oc = obs.groupby(["length", "similarity", "n_distractors"], as_index=False).agg(
        acc=("correct", "mean"))
    m = pred.merge(oc, on=["length", "similarity", "n_distractors"])
    r = pearsonr(m["pred_qwen"], m["acc"]).statistic
    lo, hi = boot_ci(m[["pred_qwen", "acc"]].to_numpy(),
                     lambda a: pearsonr(a[:, 0], a[:, 1]).statistic)
    add("Cross-model transfer Pearson r", r, lo, hi)
    mae = float((m["pred_qwen"] - m["acc"]).abs().mean())
    lo, hi = boot_ci((m["pred_qwen"] - m["acc"]).abs().to_numpy(), np.mean)
    add("Cross-model transfer MAE", mae, lo, hi)
    p = 1 / (1 + np.exp(-(make_features(obs) @ coefs + frozen["intercept"])))
    yl = obs["correct"].to_numpy(int)
    rng2 = np.random.default_rng(2)
    bauc = []
    for _ in range(400):
        idx = rng2.integers(0, len(yl), len(yl))
        if len(set(yl[idx])) > 1:
            bauc.append(roc_auc_score(yl[idx], p[idx]))
    add("Cross-model AUC (trials)", roc_auc_score(yl, p),
        float(np.percentile(bauc, 2.5)), float(np.percentile(bauc, 97.5)))
    add("Cross-model ECE", ece(p, yl), np.nan, np.nan)

    # ---- Exp8 continuous similarity correlation ----
    met = pd.read_csv(RESULTS / "exp8_similarity_metrics.csv")
    acc8 = pd.read_csv(RESULTS / "exp8b_qwen_d4.csv").groupby(
        "label", as_index=False).agg(acc=("correct", "mean"))
    d8 = acc8.merge(met, on="label")
    rr = pearsonr(d8["jaccard"], d8["acc"]).statistic
    lo, hi = boot_ci(d8[["jaccard", "acc"]].to_numpy(),
                     lambda a: pearsonr(a[:, 0], a[:, 1]).statistic)
    add("Continuous similarity (jaccard) r", rr, lo, hi)

    # ---- Intervention gain ----
    b = pd.read_csv(RESULTS / "expB_intervention.csv")
    piv = b.pivot_table(index="sample", columns="condition", values="correct")
    gain = (piv["SIM"] - piv["RANDOM"]).to_numpy()
    add("Intervention SIM-RANDOM accuracy gain", gain.mean(),
        *boot_ci(gain, np.mean))

    out = pd.DataFrame(ROWS)
    out.to_csv(RESULTS / "final_stats.csv", index=False)
    print("=== FINAL STATISTICS (measured) ===")
    print(out.to_string(index=False))

    # ---- calibration figure ----
    fig, ax = plt.subplots(figsize=(5.4, 5))
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for i in range(10):
        sel = (p >= bins[i]) & (p < bins[i + 1])
        if sel.sum() >= 5:
            xs.append(p[sel].mean())
            ys.append(yl[sel].mean())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    ax.plot(xs, ys, "o-", color="#1f4e79", label="frozen Qwen law on Llama")
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Calibration (Llama trials), ECE={ece(p, yl):.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_calibration.png", dpi=120)
    print("saved figure_calibration.png")


if __name__ == "__main__":
    main()