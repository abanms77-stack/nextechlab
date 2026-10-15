"""Cross-model transfer evaluation: frozen Qwen law -> Llama observations.

Protocol (pre-registered): the Qwen law was frozen (results/frozen_law_qwen.json)
and cell predictions saved (results/pred_qwen_on_llama.csv) BEFORE the Llama
experiment ran. This script only compares.

Reports:
    Pearson r, Spearman rho, MAE (cell level)
    ECE and AUC (sample level, frozen coefficients, no refit)
    residuals by D, S, L
    Llama-only refit vs Qwen coefficients
    combined fit: shared slopes + model-specific intercept

Usage:
    python experiments/transfer_eval.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_exp6 import FEATS, make_features  # noqa: E402


def ece(prob, y, bins=10):
    prob = np.asarray(prob, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        hi_inclusive = i == bins - 1
        m = (prob >= edges[i]) & ((prob <= edges[i + 1]) if hi_inclusive
                                  else (prob < edges[i + 1]))
        if m.sum():
            out += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(out)


def main():
    frozen = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    pred = pd.read_csv(RESULTS / "pred_qwen_on_llama.csv")
    obs = pd.read_csv(RESULTS / "exp7_llama_factorial.csv")

    cell = (obs.groupby(["length", "similarity", "n_distractors"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean"),
                 p_correct=("prob_correct", "mean")))
    m = pred.merge(cell, on=["length", "similarity", "n_distractors"], how="inner")
    print(f"cell predictions matched: {len(m)} of {len(pred)}")

    r = float(np.corrcoef(m["pred_qwen"], m["acc"])[0, 1])
    rho = float(spearmanr(m["pred_qwen"], m["acc"]).correlation)
    mae = float((m["pred_qwen"] - m["acc"]).abs().mean())

    # sample-level metrics from the FROZEN coefficients (no refit)
    coefs = np.array([frozen["coefs"][f] for f in FEATS])
    p = 1.0 / (1.0 + np.exp(-(make_features(obs) @ coefs + frozen["intercept"])))
    y = obs["correct"].to_numpy(int)
    e = ece(p, y)
    auc = float(roc_auc_score(y, p))

    print("\n=== TRANSFER: frozen Qwen law -> Llama ===")
    print(f"  Pearson r  = {r:.3f}")
    print(f"  Spearman r = {rho:.3f}")
    print(f"  MAE (cells)= {mae:.3f}")
    print(f"  ECE        = {e:.3f}")
    print(f"  AUC        = {auc:.3f}")

    m["resid"] = m["acc"] - m["pred_qwen"]
    print("\nmean residual by D:", m.groupby("n_distractors")["resid"].mean().round(3).to_dict())
    print("mean residual by S:", m.groupby("similarity")["resid"].mean().round(3).to_dict())
    print("mean residual by L:", m.groupby("length")["resid"].mean().round(3).to_dict())
    m.to_csv(RESULTS / "transfer_pairs.csv", index=False)

    # Llama-only refit
    clf_l = LogisticRegression(max_iter=3000, C=1.0).fit(make_features(obs), y)
    print("\n=== coefficient comparison ===")
    for f, cq, cl in zip(FEATS, coefs, clf_l.coef_[0]):
        print(f"  {f:<10} Qwen={cq:+.3f}  Llama={cl:+.3f}  delta={cl-cq:+.3f}")
    print(f"  intercept  Qwen={frozen['intercept']:+.3f}  "
          f"Llama={clf_l.intercept_[0]:+.3f}")

    # combined fit: shared slopes + model-specific intercept
    qwen = pd.read_csv(RESULTS / "exp6_factorial.csv")
    qwen["is_llama"] = 0.0
    o2 = obs.copy()
    o2["is_llama"] = 1.0
    comb = pd.concat([qwen, o2], ignore_index=True)
    Xc = np.column_stack([make_features(comb), comb["is_llama"].to_numpy(float)])
    clf_c = LogisticRegression(max_iter=4000, C=1.0).fit(
        Xc, comb["correct"].to_numpy(int))
    names = FEATS + ["is_llama"]
    print("\n=== combined fit (shared slopes + model intercept) ===")
    print("  " + "  ".join(f"{n}={v:+.3f}" for n, v in
                           zip(names, clf_c.coef_[0])))
    print(f"  is_llama coefficient = {clf_c.coef_[0][-1]:+.3f} "
          f"(model-specific attention budget)")

    # figure
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for L, g in m.groupby("length"):
        ax.scatter(g["pred_qwen"], g["acc"], s=50, label=f"L={L}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("predicted accuracy (frozen Qwen law)")
    ax.set_ylabel("observed accuracy (Llama-3.2-1B)")
    ax.set_title(f"Cross-model transfer:  r={r:.3f}  MAE={mae:.3f}")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figure_transfer.png", dpi=120)
    print("\nsaved figure_transfer.png")


if __name__ == "__main__":
    main()
