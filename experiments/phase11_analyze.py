"""Phase 11 - position effects (analyzer + figure).

Reads results/phase11_transfer.csv and results/phase11_factorial.csv and
produces (a) an accuracy-vs-position curve for each D (observed) and (b) a
predicted-vs-observed scatter comparing the position-blind base against the
position-aware extension on held-out test positions. Prints the
preregistered answers. All numbers are copied from the evaluator, not
recomputed.

Usage:
  python experiments/phase11_analyze.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    tr = pd.read_csv(RESULTS / "phase11_transfer.csv")
    fac = pd.read_csv(RESULTS / "phase11_factorial.csv")

    print("=== Phase 11 transfer metrics (zero-refit, held-out positions) ===")
    print(tr.to_string(index=False))

    cell = (fac.groupby(["n_distractors", "position"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    # Left: accuracy vs position, one line per D (observed)
    ax = axes[0]
    for d, sub in cell.groupby("n_distractors"):
        sub = sub.sort_values("position")
        ax.plot(sub["position"], sub["acc"], marker="o",
                label=f"D={int(d)}")
    ax.set_xlabel("evidence position P (fraction of context)")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Position effect (L=256, S=medium)")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: held-out test observed vs predicted (base vs pos-aware)
    ax = axes[1]
    test_pos = [0.15, 0.50, 0.85]
    base = pd.read_csv(RESULTS / "expB_dsweep_cells.csv")
    # rebuild the cell-level predictions exactly as the evaluator does
    import numpy as np
    from scipy.optimize import minimize
    Dref = base["n_distractors"].to_numpy(float)
    yref = base["acc"].to_numpy(float)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def nll(p):
        a, b = p
        ph = np.clip(sigmoid(a + b * np.log1p(Dref)), 1e-6, 1 - 1e-6)
        return -np.sum(yref * np.log(ph) + (1 - yref) * np.log(1 - ph))

    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    a0, b0 = res.x
    c_edge = 0.0929  # from evaluator output (fit-split)
    te = cell[cell["position"].isin(test_pos)].copy()
    basep = sigmoid(a0 + b0 * np.log1p(te["n_distractors"].to_numpy(float)))
    posp = np.clip(basep + c_edge * np.abs(2 * te["position"].to_numpy(float) - 1),
                   1e-6, 1 - 1e-6)
    ax.scatter(basep, te["acc"], marker="o", label="base (position-blind)",
               color="#1f77b4")
    ax.scatter(posp, te["acc"], marker="x", label="pos-aware (held-out)",
               color="#d62728")
    lim = [0.3, 1.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Held-out test positions")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS / "figure_phase11_position.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")

    # ---- preregistered answers (from evaluator CSV) ----------------------
    test = tr[tr.target == "test"].set_index("form")
    fit = tr[tr.target == "fit"].set_index("form")
    base_mae = float(test.loc["base_log", "MAE"])
    edge_mae = float(test.loc["pos_edge", "MAE"])
    gain = base_mae - edge_mae
    rho = float(test.loc["pos_edge", "Spearman"])
    d_r2 = float(fit.loc["pos_edge", "R2"] - fit.loc["base_log", "R2"])

    print("\n=== Phase 11 preregistered answers ===")
    print(f"Does position explain residual error after D, S, L are known?")
    print(f"  held-out test: base MAE={base_mae:.3f} -> pos_edge MAE={edge_mae:.3f}"
          f"  (gain {gain:.3f})")
    print(f"  pos_edge Spearman on test = {rho:.3f}")
    print(f"  delta R2 on fit-split = {d_r2:+.3f}")
    if abs(rho) >= 0.8 and gain >= 0.03:
        v = "A - position adds information (extend law to +P)"
    elif abs(rho) >= 0.8:
        v = "B - position ranks cells, calibration gain below 0.03 threshold"
    else:
        v = "C - position redundant beyond D/S/L"
    print(f"  verdict: {v}")
    print("\nNOTE: preliminary (n=10/cell, 7 positions, 2 D). Position = "
          "evidence position; decoy positions randomized as noise.")


if __name__ == "__main__":
    main()
