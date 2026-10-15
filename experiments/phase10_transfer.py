"""Phase 10 - cross-domain + cross-length zero-refit transfer (evaluator).

Runs the frozen protocol in results/phase10_protocol.json. Two evaluations,
both with ZERO refit on the target:

  TARGET 1 - domain-B cells (NEW model run, filler = general English):
      Compare frozen log / linear / min_sat (params fit on the wiki-prose
      reference curve) and the frozen 7-feature law against observed accuracy
      on D in {0,4,8}. Primary comparison: log vs frozen law.

  TARGET 2 - cross-length (existing Exp6 data, no new runs):
      The L-side analogue of Phase 9's task transfer: fit each form on the
      Exp6 L=256 cells, then evaluate zero-refit on the L=1024 cells across
      all S levels.

Preregistered metrics: Spearman, MAE, RMSE, ECE (primary); Pearson,
mono_viol, calib slope/intercept, n_params (secondary).

Verdict (declared in advance, applies to TARGET 1):
  A - |Spearman| >= 0.8 and MAE <= 0.15
  B - |Spearman| >= 0.8 and MAE > 0.15 (rank yes, calibration no)
  C - |Spearman| < 0.8

Output:
  results/phase10_transfer.csv   (per-form metrics on both targets)

Usage:
  python experiments/phase10_transfer.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOMAINB = RESULTS / "phase10_domainB_factorial.csv"
EXP6 = RESULTS / "exp6_factorial.csv"
PROTOCOL = RESULTS / "phase10_protocol.json"

S_MEDIUM = 0.5
L_FIXED = 256
S_W = {"low": 0.1, "medium": 0.5, "high": 1.0}


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


def fit_k(form_fn, D, y):
    best_k, best = None, np.inf
    for k in np.arange(0.5, 40.0, 0.5):
        x = form_fn(D, {"k": k})
        a, b = fit_ab(x, y)
        mae = float(np.mean(np.abs(sigmoid(a + b * x) - y)))
        if mae < best:
            best, best_k = mae, k
    return best_k


def make_form(name):
    if name == "linear":
        return (lambda D, p=None: D), None
    if name == "min_sat":
        return (lambda D, p=None: np.minimum(1.0, D / p["k"])), "k"
    if name == "log":
        return (lambda D, p=None: np.log1p(D)), None
    raise ValueError(name)


def ref_curve():
    c = pd.read_csv(RESULTS / "expB_dsweep_cells.csv").sort_values("n_distractors")
    return c["n_distractors"].to_numpy(float), c["acc"].to_numpy(float)


def freeze_forms():
    D, y = ref_curve()
    frozen = {}
    for name in ["linear", "min_sat", "log"]:
        f, kmode = make_form(name)
        if kmode == "k":
            k = fit_k(f, D, y)
            a, b = fit_ab(f(D, {"k": k}), y)
            frozen[name] = {"a": a, "b": b, "k": k}
        else:
            a, b = fit_ab(f(D, None), y)
            frozen[name] = {"a": a, "b": b}
    return frozen


def sigmoid_pred(name, params, D):
    f, kmode = make_form(name)
    x = f(D, params) if kmode == "k" else f(D, None)
    return sigmoid(params["a"] + params["b"] * x)


def ece(pred, y, bins=5):
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (pred >= lo) & (pred < hi + (1e-9 if i == bins - 1 else 0))
        if m.sum():
            e += m.mean() * abs(pred[m].mean() - y[m].mean())
    return float(e)


def mono_viol(D, pred):
    o = np.argsort(D)
    return int(np.sum(np.diff(pred[o]) > 1e-6))


def calib_slope_int(y, pred):
    if len(set(pred)) < 2:
        return float("nan"), float("nan")
    A = np.vstack([pred, np.ones(len(pred))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def frozen_law_pred(D, L, S):
    law = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    c = law["coefs"]
    Lg = np.log2(L)
    z = (c["log2L"] * Lg + c["D"] * D + c["S"] * S + c["D*S"] * D * S
         + c["D*log2L"] * D * Lg + c["S*log2L"] * S * Lg
         + c["D*S*log2L"] * D * S * Lg + law["intercept"])
    return sigmoid(z)


def metrics_block(y, pred, D, form, target, n_params):
    rho = float(spearmanr(y, pred).statistic) if len(set(pred)) > 1 else 0.0
    r = float(pearsonr(y, pred).statistic) if len(set(pred)) > 1 else 0.0
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ec = ece(pred, y)
    slope, icept = calib_slope_int(y, pred)
    return {
        "target": target, "form": form, "Spearman": round(rho, 4),
        "Pearson": round(r, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "ECE": round(ec, 4), "mono_viol": mono_viol(D, pred),
        "calib_slope": round(slope, 4), "calib_intercept": round(icept, 4),
        "n_params": n_params,
    }


# --- part 2: main() appended below ---


def main():
    if not PROTOCOL.exists():
        sys.exit("missing phase10_protocol.json; run phase10_freeze.py first")
    json.loads(PROTOCOL.read_text(encoding="utf-8"))  # validate protocol present
    frozen = freeze_forms()

    rows = []
    nparams = {"log": 2, "linear": 2, "min_sat": 3}

    # ---------------- TARGET 1: domain-B cells ---------------------------
    if not DOMAINB.exists():
        sys.exit(f"missing {DOMAINB}; run run_phase10.py first")
    db = pd.read_csv(DOMAINB)
    cell = (db.groupby("n_distractors", as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))
    D1 = cell["n_distractors"].to_numpy(float)
    y1 = cell["acc"].to_numpy(float)
    print("=== TARGET 1: domain-B cells (filler = general English) ===")
    print(f"  D={list(map(int, D1))}  observed acc={list(np.round(y1, 3))}")
    for name in ["log", "linear", "min_sat"]:
        pred = sigmoid_pred(name, frozen[name], D1)
        rows.append(metrics_block(y1, pred, D1, name, "domainB",
                                  nparams[name]))
    bp = frozen_law_pred(D1, L_FIXED, S_MEDIUM)
    rows.append(metrics_block(y1, bp, D1, "frozen_law", "domainB", 7))
    for r in rows:
        if r["target"] == "domainB":
            print(f"  {r['form']:10s} rho={r['Spearman']:.3f} MAE={r['MAE']:.3f} "
                  f"ECE={r['ECE']:.3f} slope={r['calib_slope']:.2f}")

    # ---------------- TARGET 2: cross-length (Exp6) ----------------------
    print("\n=== TARGET 2: cross-length (train L=256 -> test L=1024, zero-refit) ===")
    e6 = pd.read_csv(EXP6)
    Ls = sorted(e6["length"].unique())
    L_train, L_test = int(Ls[0]), int(Ls[-1])

    tr = e6[e6.length == L_train]
    cell_tr = (tr.groupby("n_distractors", as_index=False)
               .agg(acc=("correct", "mean")))
    Dtr = cell_tr["n_distractors"].to_numpy(float)
    ytr = cell_tr["acc"].to_numpy(float)
    fit2 = {}
    for name in ["log", "linear", "min_sat"]:
        f, kmode = make_form(name)
        if kmode == "k":
            k = fit_k(f, Dtr, ytr)
            a, b = fit_ab(f(Dtr, {"k": k}), ytr)
            fit2[name] = {"a": a, "b": b, "k": k}
        else:
            a, b = fit_ab(f(Dtr, None), ytr)
            fit2[name] = {"a": a, "b": b}

    te = e6[e6.length == L_test]
    for s in ["low", "medium", "high"]:
        sel = te[te.similarity == s]
        cell_t = (sel.groupby("n_distractors", as_index=False)
                  .agg(acc=("correct", "mean")))
        Dte = cell_t["n_distractors"].to_numpy(float)
        yte = cell_t["acc"].to_numpy(float)
        for name in ["log", "linear", "min_sat"]:
            pred = sigmoid_pred(name, fit2[name], Dte)
            rows.append(metrics_block(yte, pred, Dte, name,
                                      f"crosslen_S{s}", nparams[name]))
        bp = frozen_law_pred(Dte, L_test, S_W[s])
        rows.append(metrics_block(yte, bp, Dte, "frozen_law",
                                  f"crosslen_S{s}", 7))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phase10_transfer.csv", index=False)
    print(f"\nsaved {RESULTS / 'phase10_transfer.csv'}")

    # ---- verdict (TARGET 1, preregistered) -------------------------------
    log_row = out[(out.target == "domainB") & (out.form == "log")].iloc[0]
    law_row = out[(out.target == "domainB") & (out.form == "frozen_law")].iloc[0]
    rho, mae = log_row["Spearman"], log_row["MAE"]
    if abs(rho) >= 0.8 and mae <= 0.15:
        verdict = "A (cross-domain transfer works)"
    elif abs(rho) >= 0.8:
        verdict = "B (rank transfers, calibration does not)"
    else:
        verdict = "C (fails on domain-B cells)"
    print("\n=== preregistered outcome (TARGET 1: log form) ===")
    print(f"log:        Spearman={rho:.3f} MAE={mae:.3f} ECE={log_row['ECE']:.3f}")
    print(f"frozen_law: Spearman={law_row['Spearman']:.3f} MAE={law_row['MAE']:.3f} "
          f"ECE={law_row['ECE']:.3f}")
    print(f"verdict: {verdict}")
    print("\nNOTE: verdicts are preliminary (n=10/cell, 3 D points).")


if __name__ == "__main__":
    main()
