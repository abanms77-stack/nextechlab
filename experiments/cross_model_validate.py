"""Cross-model bridge validation (HONEST, leakage-controlled).

There are three real cross-model results in this project:

  1. Qwen -> Llama   : zero-refit (law frozen before Llama data existed)
  2. Qwen -> gpt2-med: zero-refit (law frozen before gpt2-medium data existed)
  3. calibration      : a Qwen->target affine/logistic recalibration learned on
                        TRAIN cells only, evaluated on held-out TEST cells

IMPORTANT HONESTY NOTE
----------------------
The models here share the SAME (L, S, D) cell grid, so a "train/test split of
cells" is NOT an unseen-model test - it is a within-target calibration check.
We report it explicitly as CALIBRATION TRANSFER, not as unseen-model transfer.
The zero-refit numbers are the genuinely strong cross-model evidence.

Pre-declared PASS criterion (set before running):
  - zero-refit Spearman rho >= 0.80 on the target's full 30 cells, AND
  - held-out calibration rho >= 0.80 with the frozen structure untouched.

Outputs:
    results/cross_model_validation.csv   - per-model summary rows
    results/cross_model_pairs.csv        - per-cell rows for all targets

Usage:
    python experiments/cross_model_validate.py
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
        hi = p <= 1.0 if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def load_target(name, data_csv, pred_csv):
    obs = pd.read_csv(RESULTS / data_csv)
    pred = pd.read_csv(RESULTS / pred_csv)
    cells = (obs.groupby(["length", "similarity", "n_distractors"])["correct"]
             .agg(["mean", "count"]).reset_index())
    cells.columns = ["length", "similarity", "n_distractors", "acc", "n"]
    m = pred.merge(cells, on=["length", "similarity", "n_distractors"], how="inner")
    m["target"] = name
    return m


def zero_refit(m):
    y, p = m["acc"].to_numpy(), m["pred_qwen"].to_numpy()
    return {
        "n_cells": len(m),
        "pearson": float(stats.pearsonr(y, p)[0]),
        "spearman": float(stats.spearmanr(y, p)[0]),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "ece": ece(y, p),
    }


def leave_one_cell_out_calibration(m):
    """Fit an affine calibration pred->acc on TRAIN, score on TEST.

    Leave-one-cell-out: for each held-out cell, fit a 1-D linear calibration
    a*pred+b on the other 29 cells and predict the held-out cell. This tests the
    transferability of the calibration layer ONLY (frozen Qwen structure is the
    input feature pred_qwen; nothing about the frozen law is refit).
    """
    y = m["acc"].to_numpy()
    p = m["pred_qwen"].to_numpy()
    preds = np.empty_like(y)
    for i in range(len(m)):
        tr = np.arange(len(m)) != i
        a, b = np.polyfit(p[tr], y[tr], 1)
        preds[i] = a * p[i] + b
    return {
        "loo_pearson": float(stats.pearsonr(y, preds)[0]),
        "loo_spearman": float(stats.spearmanr(y, preds)[0]),
        "loo_mae": float(np.mean(np.abs(y - preds))),
        "loo_rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
    }


def main():
    frozen = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    print(f"Frozen Qwen law sha256 = {frozen['sha256']}\n")

    targets = {
        "llama-3.2-1b": ("exp7_llama_factorial.csv", "pred_qwen_on_llama.csv"),
        "gpt2-medium": ("expA3_gpt2medium_factorial.csv", "pred_qwen_on_gpt2medium.csv"),
    }

    rows, all_pairs = [], []
    for name, (data_csv, pred_csv) in targets.items():
        if not (RESULTS / data_csv).exists():
            print(f"[skip] {name}: {data_csv} not found")
            continue
        m = load_target(name, data_csv, pred_csv)
        zr = zero_refit(m)
        loo = leave_one_cell_out_calibration(m)
        # no-transfer baseline == zero-refit (the frozen law itself)
        row = {"target": name, **zr, **loo,
               "no_transfer_mae": zr["mae"]}
        rows.append(row)
        all_pairs.append(m)
        print(f"=== {name} (n={zr['n_cells']} cells) ===")
        print(f"  zero-refit  r={zr['pearson']:.3f} rho={zr['spearman']:.3f} "
              f"MAE={zr['mae']:.3f} RMSE={zr['rmse']:.3f} ECE={zr['ece']:.3f}")
        print(f"  LOO-calib   r={loo['loo_pearson']:.3f} rho={loo['loo_spearman']:.3f} "
              f"MAE={loo['loo_mae']:.3f} RMSE={loo['loo_rmse']:.3f}")
        gain = zr["mae"] - loo["loo_mae"]
        print(f"  calibration MAE gain over no-transfer: {gain:+.3f}\n")

    if not rows:
        sys.exit("no target data available yet")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "cross_model_validation.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(
        RESULTS / "cross_model_pairs.csv", index=False)

    print("=== PRE-DECLARED PASS CRITERION ===")
    for r in rows:
        ok = r["spearman"] >= 0.80 and r["loo_spearman"] >= 0.80
        print(f"  {r['target']:<16} zero-refit rho={r['spearman']:.3f} "
              f"LOO rho={r['loo_spearman']:.3f}  -> {'PASS' if ok else 'FAIL'}")
    print(f"\nsaved {RESULTS / 'cross_model_validation.csv'}")
    print(f"saved {RESULTS / 'cross_model_pairs.csv'}")


if __name__ == "__main__":
    main()
