"""Phase 8 - D functional-form study (fitter / evaluator).

Runs the frozen protocol in results/phase8_protocol.json:

  * Builds the D -> accuracy curve from results/expB_dsweep_cells.csv.
  * For each candidate f(D) it fits P = sigma(a + b*f(D)) by LEAVE-ONE-D-OUT
    (fit on all D except the held-out one; predict it), for both interpolation
    (each interior D) and extrapolation (D=16).
  * Reports MAE, RMSE, Spearman, ECE, monotonicity violations, param count.
  * Evaluates the FROZEN 7-feature ContextLab law on the same held-out D.

Usage:
    python experiments/phase8_fit.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
S_MEDIUM = 0.5
L_FIXED = 256


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_curve():
    c = pd.read_csv(RESULTS / "expB_dsweep_cells.csv")
    c = c.sort_values("n_distractors")
    D = c["n_distractors"].to_numpy(float)
    # cell-level empirical accuracy (n per cell is in the file)
    n = c["n"].to_numpy(float) if "n" in c.columns else np.full(len(c), 20.0)
    y = c["acc"].to_numpy(float)
    return D, y, n


def f_linear(D, params):
    return D


def f_min_sat(D, params):
    k = params.get("k", 5.0)
    return np.minimum(1.0, D / k)


def f_exp_sat(D, params):
    k = params.get("k", 5.0)
    return 1.0 - np.exp(-D / k)


def f_log(D, params):
    return np.log1p(D)


FORMS = {
    "linear": f_linear,
    "min_sat": f_min_sat,
    "exp_sat": f_exp_sat,
    "log": f_log,
}


def fit_ab(x, y):
    """Fit P = sigmoid(a + b*x) by weighted least squares on the logit-ish
    target. Use scipy on negative log-likelihood with y as soft targets."""
    def nll(p):
        a, b = p
        phat = np.clip(sigmoid(a + b * x), 1e-6, 1 - 1e-6)
        # soft cross-entropy using empirical accuracy as probability
        return -np.sum(y * np.log(phat) + (1 - y) * np.log(1 - phat))
    res = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def fit_k(form_fn, D, y):
    """Grid-search k for saturating forms on the training cells."""
    best_k, best = None, np.inf
    for k in np.arange(0.5, 40.0, 0.5):
        x = form_fn(D, {"k": k})
        a, b = fit_ab(x, y)
        phat = sigmoid(a + b * x)
        mae = float(np.mean(np.abs(phat - y)))
        if mae < best:
            best, best_k = mae, k
    return best_k


def ece(pred, y, bins=5):
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (pred >= lo) & (pred < hi + (1e-9 if i == bins - 1 else 0))
        if m.sum():
            e += m.mean() * abs(pred[m].mean() - y[m].mean())
    return float(e)


def mono_violations(x, pred):
    """Count steps where a larger D yields a LARGER predicted probability
    (accuracy should be non-increasing in D)."""
    order = np.argsort(x)
    p = pred[order]
    return int(np.sum(np.diff(p) > 1e-6))


def frozen_law_pred(D):
    """Evaluate the frozen 7-feature ContextLab law at S=medium, L=256."""
    law = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    c = law["coefs"]
    L = np.log2(L_FIXED)
    S = S_MEDIUM
    z = (c["log2L"] * L + c["D"] * D + c["S"] * S + c["D*S"] * D * S
         + c["D*log2L"] * D * L + c["S*log2L"] * S * L
         + c["D*S*log2L"] * D * S * L + law["intercept"])
    return sigmoid(z)


def evaluate_loo(D, y, form_name):
    """Leave-one-D-out predictions for one functional form."""
    preds = np.zeros(len(D))
    params_list = []
    for i in range(len(D)):
        tr = np.ones(len(D), bool)
        tr[i] = False
        Dtr, ytr = D[tr], y[tr]
        if form_name in ("min_sat", "exp_sat"):
            k = fit_k(FORMS[form_name], Dtr, ytr)
            x_tr = FORMS[form_name](Dtr, {"k": k})
            a, b = fit_ab(x_tr, ytr)
            x_te = FORMS[form_name](D[i:i + 1], {"k": k})
            params_list.append({"k": k, "a": a, "b": b})
        elif form_name == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(Dtr, ytr)
            preds[i] = float(iso.predict([D[i]])[0])
            params_list.append({"n_params": len(Dtr)})
            continue
        else:
            x_tr = FORMS[form_name](Dtr, {})
            a, b = fit_ab(x_tr, ytr)
            x_te = FORMS[form_name](D[i:i + 1], {})
            params_list.append({"a": a, "b": b})
        preds[i] = float(sigmoid(params_list[-1]["a"]
                                 + params_list[-1]["b"] * x_te[0]))
    return preds, params_list


def metrics(y, pred):
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rho = float(spearmanr(y, pred).statistic) if len(set(pred)) > 1 else 0.0
    return mae, rmse, rho


def n_params_for(form_name, params_list):
    if form_name == "isotonic":
        return int(np.mean([p["n_params"] for p in params_list]))
    if "k" in params_list[0]:
        return 3  # a, b, k
    return 2      # a, b


def main():
    D, y, n = load_curve()
    print("=== Phase 8: D functional-form study ===")
    print(f"D grid = {[int(d) for d in D]}")
    print(f"empirical acc = {[round(v, 3) for v in y]}\n")

    # ---- baseline: frozen 7-feature law ----
    base_pred = frozen_law_pred(D)
    b_mae, b_rmse, b_rho = metrics(y, base_pred)
    base_ece = ece(base_pred, y)
    base_mono = mono_violations(D, base_pred)
    print("frozen 7-feature ContextLab law (baseline):")
    print(f"  MAE={b_mae:.3f} RMSE={b_rmse:.3f} rho={b_rho:.3f} "
          f"ECE={base_ece:.3f} mono_viol={base_mono} params=7\n")

    rows = []
    loo_rows = []
    for form in ["linear", "min_sat", "exp_sat", "log", "isotonic"]:
        preds, plist = evaluate_loo(D, y, form)
        mae, rmse, rho = metrics(y, preds)
        ec = ece(preds, y)
        mv = mono_violations(D, preds)
        np_ = n_params_for(form, plist)
        rows.append({"form": form, "MAE": round(mae, 4),
                     "RMSE": round(rmse, 4), "Spearman": round(rho, 4),
                     "ECE": round(ec, 4), "mono_viol": mv, "n_params": np_,
                     "beats_baseline_MAE": bool(mae < b_mae)})
        for d, yy, pp in zip(D, y, preds):
            loo_rows.append({"form": form, "D": int(d), "observed": yy,
                             "pred": round(float(pp), 4)})
        print(f"{form:10s} MAE={mae:.3f} RMSE={rmse:.3f} rho={rho:.3f} "
              f"ECE={ec:.3f} mono_viol={mv} params={np_} "
              f"{'<-- beats baseline' if mae < b_mae else ''}")

    forms_df = pd.DataFrame(rows)
    forms_df.to_csv(RESULTS / "phase8_forms.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(RESULTS / "phase8_loo.csv", index=False)
    print(f"\nsaved {RESULTS / 'phase8_forms.csv'}")
    print(f"saved {RESULTS / 'phase8_loo.csv'}")

    # ---- preregistered questions ----
    best = forms_df.sort_values("MAE").iloc[0]
    print("\n=== preregistered answers ===")
    print(f"Q1 best interpolation form by MAE: {best['form']} "
          f"(MAE={best['MAE']:.3f})")
    print(f"Q3 any candidate beats frozen law MAE ({b_mae:.3f})? "
          f"{'YES' if (forms_df['MAE'] < b_mae).any() else 'NO'}")
    print("Q4 monotonicity violations:",
          dict(zip(forms_df["form"], forms_df["mono_viol"])))

    # ---- extrapolation: hold out the largest D ----
    idx = int(np.argmax(D))
    Dtr = np.delete(D, idx)
    ytr = np.delete(y, idx)
    y_te = y[idx]
    print(f"\n=== extrapolation: hold out D={int(D[idx])} ===")
    print(f"observed acc at D={int(D[idx])} = {y_te:.3f}")
    b_te = float(frozen_law_pred(np.array([D[idx]]))[0])
    print(f"  frozen law      pred={b_te:.3f}  err={abs(b_te-y_te):.3f}")
    for form in ["linear", "min_sat", "exp_sat", "log"]:
        if form in ("min_sat", "exp_sat"):
            k = fit_k(FORMS[form], Dtr, ytr)
            a, b = fit_ab(FORMS[form](Dtr, {"k": k}), ytr)
            x_te = FORMS[form](np.array([D[idx]]), {"k": k})
        else:
            a, b = fit_ab(FORMS[form](Dtr, {}), ytr)
            x_te = FORMS[form](np.array([D[idx]]), {})
        p_te = float(sigmoid(a + b * x_te[0]))
        print(f"  {form:10s}      pred={p_te:.3f}  err={abs(p_te-y_te):.3f}")


if __name__ == "__main__":
    main()


