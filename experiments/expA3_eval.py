"""Experiment A (third model) - zero-refit evaluation: frozen Qwen law vs gpt2-medium.

Compares the pre-registered predictions (results/pred_qwen_on_gpt2medium.csv)
against the observed gpt2-medium cells, with the Qwen law FROZEN (no refit).

Reports: Pearson r, Spearman rho, MAE, RMSE, ECE, monotonicity per family,
per-similarity and per-length breakdowns.

Usage:
    python experiments/expA3_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def ece(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return total


def monotone(series):
    vals = list(series)
    return all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def main():
    pred = pd.read_csv(RESULTS / "pred_qwen_on_gpt2medium.csv")
    obs = pd.read_csv(RESULTS / "expA3_gpt2medium_factorial.csv")
    frozen = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    print(f"Frozen Qwen law sha256 = {frozen['sha256']}")

    cells = (obs.groupby(["length", "similarity", "n_distractors"])["correct"]
             .agg(["mean", "count"]).reset_index())
    cells.columns = ["length", "similarity", "n_distractors", "acc", "n"]
    m = pred.merge(cells, on=["length", "similarity", "n_distractors"], how="inner")
    print(f"\nmatched cells: {len(m)} (expected 30)")

    y, p = m["acc"].to_numpy(), m["pred_qwen"].to_numpy()
    r = stats.pearsonr(y, p)[0]
    rho = stats.spearmanr(y, p)[0]
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    e = ece(y, p)
    print(f"\nZERO-REFIT  r={r:.3f}  rho={rho:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  ECE={e:.3f}")

    print("\nmonotonicity (accuracy as D grows), per S x L family:")
    viol = 0
    for (L, S), g in m.groupby(["length", "similarity"]):
        g = g.sort_values("n_distractors")
        om = monotone(g["acc"])
        pm = monotone(g["pred_qwen"])
        if not pm:
            viol += 1
        print(f"  L={L:<5} S={S:<7} obs={om!s:<5} pred={pm!s:<5} "
              f"acc={[round(v,2) for v in g['acc']]}")
    print(f"  pred monotonicity violations: {viol}")

    print("\nper-similarity:")
    for S, g in m.groupby("similarity"):
        print(f"  {S:<7} obs={g['acc'].mean():.3f} pred={g['pred_qwen'].mean():.3f} "
              f"r={stats.pearsonr(g['acc'], g['pred_qwen'])[0]:.3f}")
    print("\nper-length:")
    for L, g in m.groupby("length"):
        print(f"  L={L:<5} obs={g['acc'].mean():.3f} pred={g['pred_qwen'].mean():.3f} "
              f"r={stats.pearsonr(g['acc'], g['pred_qwen'])[0]:.3f}")

    out = m.copy()
    out["resid"] = out["acc"] - out["pred_qwen"]
    out.to_csv(RESULTS / "expA3_gpt2medium_transfer_pairs.csv", index=False)
    print(f"\nsaved {RESULTS / 'expA3_gpt2medium_transfer_pairs.csv'}")


if __name__ == "__main__":
    main()
