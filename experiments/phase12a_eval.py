"""Phase 12A - long-context screening evaluator (applies the FROZEN rule).

Reads the 45-sample screening (results/phase12a_factorial.csv) plus the
existing short-L baselines (exp6 S=medium, n=20), computes the preregistered
AGGREGATE statistics (harm(L), deltaInt), fits the Model A/B/C comparison,
and applies the DECISION RULE frozen in phase12a_protocol.json - NO
threshold shopping, no redesign after seeing results.

Reports:
  * accuracy grid  L in {2048,4096,8192} x D in {0,4,8}   (screening, n=5)
  * short-L baseline grid from exp6 (zero-refit anchor)
  * meanAtL(D) / harm(L) / deltaInt  (the preregistered statistics)
  * Model A vs B vs C  (does L add predictive power? does L x D?)
  * Stage-A INTERESTING/NOT + implied outcome A/B/C  (PRELIMINARY, n=5)

Usage:
  python experiments/phase12a_eval.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACT = RESULTS / "phase12a_factorial.csv"
PROTOCOL = RESULTS / "phase12a_protocol.json"
OUT = RESULTS / "phase12a_eval.csv"

LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
S_FIXED = "medium"
BASELINE_L = 256
D_HARM = [4, 8]           # aggregate over these D for harm/deltaInt

OUT = RESULTS / "phase12a_eval.csv"


def load_cells():
    if not FACT.exists():
        sys.exit(f"missing {FACT}; run experiments/run_phase12a.py first")
    df = pd.read_csv(FACT)
    return (df.groupby(["length", "n_distractors"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))


def mean_at_L(cells, L, dlist):
    sub = cells[(cells.length == L) & (cells.n_distractors.isin(dlist))]
    return float(sub.acc.mean()) if len(sub) else float("nan")


def fit_form(features, y):
    """Fit P = sigmoid(X @ w) by MLE; returns (w, avg log-loss)."""
    X = np.hstack([np.ones((len(features[0]), 1))] +
                  [f.reshape(-1, 1) for f in features])
    y = np.asarray(y, float)

    def nll(w):
        p = np.clip(sigmoid(X @ w), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, np.zeros(X.shape[1]), method="BFGS",
                   options={"maxiter": 2000})
    p = np.clip(sigmoid(X @ res.x), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    return res.x, X, p, brier


def main():
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    stored = (RESULTS / "phase12a_protocol.sha256").read_text().strip()
    actual = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert stored == actual, "protocol hash mismatch - protocol changed!"
    rule = proto["decision_rule"]

    cells = load_cells()
    exp6 = pd.read_csv(RESULTS / "exp6_factorial.csv")
    base = exp6[exp6.similarity == S_FIXED].groupby(
        ["length", "n_distractors"])["correct"].agg(["mean", "count"]).reset_index()

    print("=== Phase 12A screening cells (n=5) ===")
    print(cells.to_string(index=False))

    # preregistered aggregate statistics
    mA = {L: mean_at_L(cells, L, D_HARM) for L in LENGTHS}
    base256 = float(base[(base.length == BASELINE_L) &
                         (base.n_distractors.isin(D_HARM))]["mean"].mean())
    harm = {L: base256 - mA[L] for L in LENGTHS}
    delta_int = mA[8192] - mA[2048]

    print("\n=== preregistered statistics (aggregate over D in {4,8}) ===")
    print(f"  meanAtL(256) baseline = {base256:.3f}   (exp6, n=20)")
    for L in LENGTHS:
        print(f"  meanAtL({L}) = {mA[L]:.3f}   harm({L}) = {harm[L]:+.3f}")
    print(f"  deltaInt = meanAtL(8192) - meanAtL(2048) = {delta_int:+.3f}")

    # ---- Model A / B / C comparison (zero-refit, on screening cells) ----
    # pool screening cells; predict acc with log(1+D), L, interaction
    y = cells.acc.to_numpy(float)
    ld = np.log1p(cells.n_distractors.to_numpy(float))
    lg = np.log2(cells.length.to_numpy(float))
    zL = (lg - lg.mean()) / lg.std()
    mA_fit = fit_form([ld], y)
    mB_fit = fit_form([ld, zL], y)
    mC_fit = fit_form([ld, zL, ld * zL], y)
    rows = []
    for name, (w, X, p, brier) in [
            ("M_A  F(log(1+D), S)", mA_fit),
            ("M_B  F(log(1+D), S, L)", mB_fit),
            ("M_C  F(log(1+D), S, L, log(1+D)xL)", mC_fit)]:
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        rho = spearmanr(y, p).statistic if len(set(np.round(p, 6))) > 1 else np.nan
        rows.append({"model": name, "params": len(w), "logloss": ll,
                     "brier": brier, "spearman": rho})
    cmp_df = pd.DataFrame(rows)
    print("\n=== Model comparison (screening cells) ===")
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT, index=False)

    # ---- apply FROZEN decision rule ----
    thresh = 0.15
    interesting = (max(harm[4096], harm[8192]) >= thresh) or (delta_int <= -thresh)
    confirmed = (harm[8192] >= 0.20) and (delta_int <= -0.05)

    # implied outcome A/B/C from the SCREENING (preliminary)
    d0_flat = all(mean_at_L(cells, L, [0]) >= 0.9 for L in LENGTHS)
    if confirmed or (delta_int <= -0.05 and harm[8192] >= 0.15):
        outcome = "A (long-context penalty exists: L x D negative)"
    elif d0_flat and harm[8192] > 0.15:
        outcome = "B (length harms ONLY under competition; D=0 stays flat)"
    else:
        outcome = "C (length barely matters; composition dominates)"

    print("\n=== Phase 12A preregistered decision ===")
    print(f"  STAGE-A INTERESTING rule : {rule['stage_a_interesting']}")
    print(f"      => harm4096={harm[4096]:+.3f} harm8192={harm[8192]:+.3f} "
          f"deltaInt={delta_int:+.3f}")
    print(f"  STAGE-A verdict: {'INTERESTING -> run Phase 12B (n=10-20)' if interesting else 'NOT INTERESTING -> stop (no 12B)'}")
    print(f"  STAGE-B CONFIRMED rule   : {rule['stage_b_confirmed']} "
          f"-> {'met' if confirmed else 'NOT met (needs 12B at n>=10-20)'}")
    print(f"  implied outcome (PRELIMINARY, n=5): {outcome}")
    print("\nNOTE: n=5 screening - single-cell diffs are noise (SE~0.2); "
          "only the aggregate + Stage-B confirmation are meaningful.")


if __name__ == "__main__":
    main()

