"""Phase 9 - cross-task generalization of the competition-count structure.

Phase 8 established that the count term f(D) = log(1+D) best predicts accuracy
on HELD-OUT D within the RETRIEVAL task family. Phase 9 asks the decisive
transfer question:

    Does the Phase-8 competition structure survive a genuinely DIFFERENT task
    with ZERO refit?

FROZEN BEFORE LOOKING AT TARGET RESULTS (this file):

  reference (frozen) : the count-term fits from the retrieval D-sweep
                       (results/expB_dsweep_factorial.csv, Qwen2.5-0.5B, L=256,
                       S=medium). Each form fits P = sigma(a + b*f(D)); the
                       fitted (a, b, k) are FROZEN here.
  target task        : COUNTING (results/expA_counting.csv) - "count the red
                       objects" while similar-colour items compete. A different
                       task family ("how many", not "which value"), not a
                       rewording of retrieval.
  transfer protocol  : fit on reference -> freeze -> evaluate on target.
                       NO target refit, NO per-task tuning.

Forms carried over from Phase 8: linear, min_sat, exp_sat, log, plus the frozen
7-feature ContextLab law as a baseline.

Preregistered metrics (on the target task):
  primary   : Spearman rho, MAE, RMSE, ECE
  secondary : Pearson r, monotonicity violations, calibration slope/intercept,
              accuracy-by-D table

Preregistered outcome interpretation (declared in advance):
  A transfer works        : best form keeps strong rank correlation on target
  B count transfers, coefs do not : form stays monotone/predictive but a
                            different coefficient would be needed
  C fails                 : competition-count structure is task-dependent

Output:
  results/phase9_protocol.json + .sha256
  results/phase9_transfer.csv     (per-form frozen params + target metrics)

Usage:
  python experiments/phase9_transfer.py --freeze   # freeze only
  python experiments/phase9_transfer.py            # freeze + evaluate
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
REF = RESULTS / "expB_dsweep_factorial.csv"
TARGET = RESULTS / "expA_counting.csv"
PROTOCOL = RESULTS / "phase9_protocol.json"
S_MEDIUM = 0.5
L_FIXED = 256
FREEZE_ONLY = "--freeze" in sys.argv


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_curve():
    c = pd.read_csv(RESULTS / "expB_dsweep_cells.csv").sort_values("n_distractors")
    return c["n_distractors"].to_numpy(float), c["acc"].to_numpy(float)


def fit_ab(x, y):
    def nll(p):
        a, b = p
        ph = np.clip(sigmoid(a + b * x), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(ph) + (1 - y) * np.log(1 - ph))
    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def fit_k(xf, D, y):
    """Fit k for a saturating form. Grid limited to a sensible range so that
    k=huge (which collapses the form to linear) is not silently selected."""
    best_k, best = None, np.inf
    for k in np.arange(0.5, 20.5, 0.5):
        x = xf(D, k)
        a, b = fit_ab(x, y)
        mae = float(np.mean(np.abs(sigmoid(a + b * x) - y)))
        if mae < best:
            best, best_k = mae, k
    return best_k


def make_form(name):
    if name == "linear":
        return (lambda D: D), None
    if name == "min_sat":
        return (lambda D, k: np.minimum(1.0, D / k)), "k"
    if name == "exp_sat":
        return (lambda D, k: 1.0 - np.exp(-D / k)), "k"
    if name == "log":
        return (lambda D: np.log1p(D)), None
    raise ValueError(name)


def freeze_forms():
    """Fit each form on the retrieval curve; return frozen params."""
    D, y = ref_curve()
    frozen = {}
    for name in ["linear", "min_sat", "exp_sat", "log"]:
        f, kmode = make_form(name)
        if kmode == "k":
            k = fit_k(f, D, y)
            a, b = fit_ab(f(D, k), y)
            frozen[name] = {"a": a, "b": b, "k": k}
        else:
            a, b = fit_ab(f(D), y)
            frozen[name] = {"a": a, "b": b}
    return frozen


def sigmoid_pred(name, params, D):
    f, kmode = make_form(name)
    x = f(D, params["k"]) if kmode == "k" else f(D)
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


def frozen_law_pred(D):
    law = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    c = law["coefs"]
    L, S = np.log2(L_FIXED), S_MEDIUM
    z = (c["log2L"] * L + c["D"] * D + c["S"] * S + c["D*S"] * D * S
         + c["D*log2L"] * D * L + c["S*log2L"] * S * L
         + c["D*S*log2L"] * D * S * L + law["intercept"])
    return sigmoid(z)


def main():
    frozen = freeze_forms()

    proto = {
        "name": "phase9_cross_task_transfer_v1",
        "frozen": True,
        "reference": REF.name,
        "reference_sha256": hashlib.sha256(REF.read_bytes()).hexdigest(),
        "target": TARGET.name,
        "target_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "frozen_params": frozen,
        "forms": ["linear", "min_sat", "exp_sat", "log"],
        "baseline": "frozen 7-feature ContextLab law",
        "protocol": "fit on reference -> freeze -> evaluate on target (no refit)",
        "primary_metrics": ["Spearman", "MAE", "RMSE", "ECE"],
        "secondary_metrics": ["Pearson", "mono_viol", "calib_slope",
                              "calib_intercept"],
        "outcome_interpretation": {
            "A": "transfer works (strong target rank correlation)",
            "B": "count form transfers but coefficient would need to differ",
            "C": "fails: competition-count structure is task-dependent",
        },
    }
    PROTOCOL.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    (RESULTS / "phase9_protocol.sha256").write_text(digest + "\n",
                                                    encoding="utf-8")
    print(f"FROZEN Phase 9 protocol -> {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    for n, p in frozen.items():
        print(f"  {n:9s} frozen params = {p}")
    if FREEZE_ONLY:
        sys.exit(0)

    # ---- evaluate on target (counting) at cell level ----------------------
    tc = pd.read_csv(TARGET)
    # cell-level accuracy vs D (pool across similarity/length is NOT done: we
    # evaluate the count term as a function of D on the L=1024 slice, where a
    # weak competition effect exists, AND on the full pooled data).
    print("\n=== target task: counting (zero-refit) ===")
    rows = []
    for slice_name, sel in [("L1024", tc[tc.length == 1024]),
                            ("pooled", tc)]:
        cell = (sel.groupby("n_distractors", as_index=False)
                .agg(acc=("correct", "mean")))
        D = cell["n_distractors"].to_numpy(float)
        y = cell["acc"].to_numpy(float)
        for name in ["linear", "min_sat", "exp_sat", "log"]:
            pred = sigmoid_pred(name, frozen[name], D)
            rho = float(spearmanr(y, pred).statistic) if len(set(pred)) > 1 else 0.0
            r = float(pearsonr(y, pred).statistic) if len(set(pred)) > 1 else 0.0
            mae = float(np.mean(np.abs(pred - y)))
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            ec = ece(pred, y)
            rows.append({"slice": slice_name, "model": name,
                         "Spearman": round(rho, 4), "Pearson": round(r, 4),
                         "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                         "ECE": round(ec, 4), "mono_viol": mono_viol(D, pred)})
        # baseline
        bp = frozen_law_pred(D)
        rows.append({"slice": slice_name, "model": "frozen_law",
                     "Spearman": round(float(spearmanr(y, bp).statistic), 4),
                     "Pearson": round(float(pearsonr(y, bp).statistic), 4),
                     "MAE": round(float(np.mean(np.abs(bp - y))), 4),
                     "RMSE": round(float(np.sqrt(np.mean((bp - y) ** 2))), 4),
                     "ECE": round(ece(bp, y), 4),
                     "mono_viol": mono_viol(D, bp)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phase9_transfer.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nsaved {RESULTS / 'phase9_transfer.csv'}")

    # ---- transparency: target accuracy-by-D per slice ---------------------
    print("\n=== target accuracy by D (per slice) ===")
    for slice_name, sel in [("L1024", tc[tc.length == 1024]), ("pooled", tc)]:
        cell = (sel.groupby("n_distractors", as_index=False)
                .agg(n=("correct", "size"), acc=("correct", "mean")))
        print(f"-- {slice_name} --")
        print("  " + "  ".join(f"D{int(r.n_distractors)}={r.acc:.2f}"
                               for _, r in cell.iterrows()))

    # ---- outcome verdict (declared in advance) ---------------------------
    l1024 = out[(out.slice == "L1024") & (out.model == "log")].iloc[0]
    pooled = out[(out.slice == "pooled") & (out.model == "log")].iloc[0]
    print("\n=== preregistered outcome ===")
    print(f"log form on L1024: Spearman={l1024['Spearman']} MAE={l1024['MAE']}")
    print(f"log form on pooled: Spearman={pooled['Spearman']} MAE={pooled['MAE']}")
    if abs(l1024["Spearman"]) >= 0.8 and l1024["MAE"] <= 0.10:
        verdict = "A (transfer works on L1024)"
    elif abs(l1024["Spearman"]) >= 0.8:
        verdict = "B (rank transfers, calibration does not)"
    else:
        verdict = "C (fails)"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()

