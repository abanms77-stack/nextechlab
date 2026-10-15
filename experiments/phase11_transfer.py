"""Phase 11 - position zero-refit transfer (evaluator).

Runs the frozen protocol in results/phase11_protocol.json. One evaluation,
with position handled out-of-sample:

  BASELINE (position-blind), calibrated ONLY on the reference position=0.5
      D-sweep cells (results/expB_dsweep_cells.csv), then used to predict
      every cell in the new position grid. This is the "D/S/L known, position
      ignored" model.

  POSITION-AWARE extension: base_log + c*g(P), where the position coefficient
      c is fit ONLY on the preregistered FIT-positions {0.05,0.30,0.70,0.95}
      and then evaluated on the held-out TEST-positions {0.15,0.50,0.85}.
      Test positions are never seen during fitting (true zero-refit).

  PRIMARY TEST (the box):
      Does position explain residual error after D, S, L are known?
      => compare held-out MAE/ECE of base vs position-aware on TEST-positions,
         plus delta-R2 vs base on the fit-split.

Preregistered metrics: Spearman, MAE, RMSE, ECE (primary); Pearson, calib
slope/intercept, delta_R2_vs_base, n_params (secondary).

Verdict (declared in advance, applies to the held-out TEST-positions):
  A - position adds information: |Spearman| >= 0.8 AND pos-aware beats base
      by >= 0.03 MAE on test positions
  B - position ranks cells but does not improve calibration (|Spearman| >= 0.8,
      MAE gain < 0.03)
  C - position redundant (fails to beat base and |Spearman| < 0.8)

Output:
  results/phase11_transfer.csv   (per-form metrics on fit + test splits)

Usage:
  python experiments/phase11_transfer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACT = RESULTS / "phase11_factorial.csv"
PROTOCOL = RESULTS / "phase11_protocol.json"

L_FIXED = 256
S_MEDIUM = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_ab(x, y):
    def nll(p):
        a, b = p
        ph = np.clip(sigmoid(a + b * x), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(ph) + (1 - y) * np.log(1 - ph))
    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def metrics_block(y, pred, form="", target="", nparams=0):
    y = np.asarray(y, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    bins = np.clip((pred * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(pred[m].mean() - y[m].mean())
    rho = np.nan
    if len(y) >= 3 and np.std(y) > 1e-9 and np.std(pred) > 1e-9:
        rho = float(spearmanr(y, pred).correlation)
    pear = np.nan
    if len(y) >= 3 and np.std(pred) > 1e-9:
        pear = float(pearsonr(y, pred)[0])
    slope, icept = np.nan, np.nan
    if len(y) >= 3 and np.std(pred) > 1e-9:
        slope, icept = map(float, np.polyfit(pred, y, 1))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan
    return {"form": form, "target": target, "n_params": nparams,
            "Spearman": rho, "MAE": mae, "RMSE": rmse, "ECE": ece,
            "Pearson": pear, "calib_slope": slope, "calib_intercept": icept,
            "R2": r2}


def ref_curve():
    c = pd.read_csv(RESULTS / "expB_dsweep_cells.csv").sort_values("n_distractors")
    return c["n_distractors"].to_numpy(float), c["acc"].to_numpy(float)


def base_log_params():
    """(a,b) for log(1+D) frozen from the reference position=0.5 curve."""
    D, y = ref_curve()
    a, b = fit_ab(np.log1p(D), y)
    return {"a": a, "b": b}


def base_log_pred(params, D):
    return sigmoid(params["a"] + params["b"] * np.log1p(np.asarray(D, float)))


def base_law_pred(D, L, S):
    """Frozen 7-feature law: feats log2L, D, S, D*S, D*log2L, S*log2L,
    D*S*log2L (as in Phase 8/10)."""
    path = RESULTS / "frozen_law_qwen.json"
    if not path.exists():
        return None
    law = json.loads(path.read_text(encoding="utf-8"))
    c = law["coefs"]
    Lv, Sv = np.log2(L), S
    D = np.asarray(D, float)
    z = (c["log2L"] * Lv + c["D"] * D + c["S"] * Sv + c["D*S"] * D * Sv
         + c["D*log2L"] * D * Lv + c["S*log2L"] * Sv * Lv
         + c["D*S*log2L"] * D * Sv * Lv + law["intercept"])
    return sigmoid(z)
def g_edge(P):
    """Position feature: distance from centre (edge strength)."""
    return np.abs(2.0 * np.asarray(P, float) - 1.0)


def g_quad(P):
    """Position feature: quadratic edge strength."""
    return (2.0 * np.asarray(P, float) - 1.0) ** 2





def main():
    if not FACT.exists():
        sys.exit(f"missing {FACT}; run experiments/run_phase11.py first")
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fit_pos = proto["split"]["fit_positions"]
    test_pos = proto["split"]["test_positions"]

    df = pd.read_csv(FACT)
    cell = (df.groupby(["n_distractors", "position"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))

    base = base_log_params()
    print(f"base_log frozen (a,b) = ({base['a']:.4f}, {base['b']:.4f}) "
          f"[from reference position=0.5 cells]")
    cell["base"] = base_log_pred(base, cell["n_distractors"])

    fit_mask = cell["position"].isin(fit_pos)
    test_mask = cell["position"].isin(test_pos)

    def fit_c(gfn):
        y = cell.loc[fit_mask, "acc"].to_numpy(float)
        b = cell.loc[fit_mask, "base"].to_numpy(float)
        g = gfn(cell.loc[fit_mask, "position"].to_numpy(float))

        def loss(p):
            pred = np.clip(b + p[0] * g, 1e-6, 1 - 1e-6)
            return float(np.mean((pred - y) ** 2))

        res = minimize(loss, [0.0], method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
        return float(res.x[0])

    c_edge = fit_c(g_edge)
    c_quad = fit_c(g_quad)
    print(f"position coeff (fit-split only): c_edge={c_edge:.4f} "
          f"c_quad={c_quad:.4f}")

    cell["pos_edge"] = np.clip(cell["base"] + c_edge * g_edge(cell["position"]),
                               1e-6, 1 - 1e-6)
    cell["pos_quad"] = np.clip(cell["base"] + c_quad * g_quad(cell["position"]),
                               1e-6, 1 - 1e-6)

    rows = []
    for split, mask in [("fit", fit_mask), ("test", test_mask)]:
        sub = cell[mask]
        y = sub["acc"].to_numpy(float)
        for form, col, np_ in [("base_log", "base", 2),
                               ("pos_edge", "pos_edge", 3),
                               ("pos_quad", "pos_quad", 3)]:
            rows.append(metrics_block(y, sub[col].to_numpy(float),
                                      form=form, target=split, nparams=np_))
        lawp = base_law_pred(sub["n_distractors"].to_numpy(float), L_FIXED,
                             S_MEDIUM)
        if lawp is not None:
            rows.append(metrics_block(y, lawp, form="base_law",
                                      target=split, nparams=7))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phase11_transfer.csv", index=False)
    print(f"\nsaved {RESULTS / 'phase11_transfer.csv'}")
    print("\n=== Phase 11 transfer metrics ===")
    print(out.to_string(index=False))

    fit_rows = out[out.target == "fit"].set_index("form")
    d_r2_edge = float(fit_rows.loc["pos_edge", "R2"]
                      - fit_rows.loc["base_log", "R2"])

    test_rows = out[out.target == "test"].set_index("form")
    base_mae = float(test_rows.loc["base_log", "MAE"])
    edge, quad = test_rows.loc["pos_edge"], test_rows.loc["pos_quad"]
    best = edge if edge["MAE"] <= quad["MAE"] else quad
    best_name = "pos_edge" if best is edge else "pos_quad"
    gain = base_mae - float(best["MAE"])
    rho = float(best["Spearman"]) if not np.isnan(best["Spearman"]) else 0.0

    print("\n=== preregistered outcome (held-out TEST positions) ===")
    print(f"base_log:   MAE={base_mae:.3f} "
          f"ECE={float(test_rows.loc['base_log', 'ECE']):.3f}")
    print(f"{best_name}: MAE={float(best['MAE']):.3f} "
          f"ECE={float(best['ECE']):.3f} rho={rho:.3f} "
          f"(gain vs base = {gain:.3f} MAE)")
    print(f"delta_R2 (fit-split, pos_edge vs base) = {d_r2_edge:+.3f}")

    if abs(rho) >= 0.8 and gain >= 0.03:
        verdict = ("A (position adds information: extend to "
                   "R=F(task,domain,f(D),S,L,P))")
    elif abs(rho) >= 0.8:
        verdict = "B (position ranks cells but does not improve calibration)"
    else:
        verdict = "C (position redundant beyond D/S/L)"
    print(f"verdict: {verdict}")
    print("\nNOTE: verdicts preliminary (n=10/cell, 7 positions, 2 D); "
          "decoy positions randomized as noise.")


if __name__ == "__main__":
    main()

