"""Phase 12B - long-context penalty CONFIRMATION evaluator (frozen rule).

Reads the 180-sample INDEPENDENT confirmation (results/phase12b_factorial.csv,
k in [5,25) disjoint from 12A) plus the SAME short-L baseline (exp6 S=medium,
n=20), computes the preregistered AGGREGATE statistics (harm(L), deltaInt),
fits the Model A/B/C comparison, and applies the decision thresholds FROZEN in
phase12b_protocol.json (= 12A's thresholds, UNCHANGED) - NO threshold shopping.

This is the genuine confirmation: Stage-B passes (or not) on FRESH n=20 data at
the SAME thresholds 12A used. If Stage-B fails, 12A stays screening-only.

Usage:
  python experiments/phase12b_eval.py
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
FACT = RESULTS / "phase12b_factorial.csv"
PROTOCOL = RESULTS / "phase12b_protocol.json"
OUT = RESULTS / "phase12b_eval.csv"

LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
S_FIXED = "medium"
BASELINE_L = 256
D_HARM = [4, 8]           # aggregate over these D for harm/deltaInt
EXPECTED = 180


def load_cells():
    if not FACT.exists():
        sys.exit(f"missing {FACT}; run experiments/run_phase12b.py first")
    df = pd.read_csv(FACT)
    return (df.groupby(["length", "n_distractors"], as_index=False)
            .agg(n=("correct", "size"), acc=("correct", "mean")))


def mean_at_L(cells, L, dlist):
    sub = cells[(cells.length == L) & (cells.n_distractors.isin(dlist))]
    return float(sub.acc.mean()) if len(sub) else float("nan")


def fit_form(features, y):
    """Fit P = sigmoid(X @ w) by MLE; returns (w, X, p, brier)."""
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
    stored = (RESULTS / "phase12b_protocol.sha256").read_text().strip()
    actual = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert stored == actual, "protocol hash mismatch - protocol changed!"
    rule = proto["decision_rule"]

    if not FACT.exists():
        sys.exit(f"missing {FACT}; run experiments/run_phase12b.py first")
    raw = pd.read_csv(FACT)
    n_rows = len(raw)
    t_min, t_max = int(raw["sample"].min()), int(raw["sample"].max())
    seed_lo = int(raw["seed"].min())

    cells = load_cells()
    exp6 = pd.read_csv(RESULTS / "exp6_factorial.csv")
    base = exp6[exp6.similarity == S_FIXED].groupby(
        ["length", "n_distractors"])["correct"].agg(["mean", "count"]).reset_index()

    print("=== Phase 12B INDEPENDENT confirmation (n=20/cell) ===")
    print(f"rows={n_rows}/{EXPECTED} | true sample index t in [{t_min},{t_max}] "
          f"(disjoint from 12A's [0,5): {t_min >= 5}) | min seed={seed_lo}")
    print(cells.to_string(index=False))

    # preregistered aggregate statistics (UNCHANGED definitions)
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

    # ---- Model A / B / C comparison (spec UNCHANGED from 12A) ----
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
    print("\n=== Model comparison (12B confirmation cells) ===")
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(OUT, index=False)

    # ---- apply FROZEN decision rule (thresholds UNCHANGED from 12A) ----
    thresh = 0.15
    interesting = (max(harm[4096], harm[8192]) >= thresh) or (delta_int <= -thresh)
    confirmed = (harm[8192] >= 0.20) and (delta_int <= -0.05)

    d0_flat = all(mean_at_L(cells, L, [0]) >= 0.9 for L in LENGTHS)
    if confirmed:
        outcome = "A (long-context penalty CONFIRMED: Stage-B passes on fresh n=20)"
    elif d0_flat and harm[8192] > 0.15:
        outcome = "B (length harms ONLY under competition; D=0 stays flat) - Stage-B NOT met"
    else:
        outcome = "C (penalty does NOT replicate / composition dominates) - 12A screening-only"

    print("\n=== Phase 12B preregistered confirmation ===")
    print(f"  STAGE-A INTERESTING rule : {rule['stage_a_interesting']}")
    print(f"      => harm4096={harm[4096]:+.3f} harm8192={harm[8192]:+.3f} "
          f"deltaInt={delta_int:+.3f}")
    print(f"  STAGE-A verdict: {'INTERESTING' if interesting else 'NOT INTERESTING'}")
    print(f"  STAGE-B CONFIRMED rule   : {rule['stage_b_confirmed']}")
    print(f"      => harm8192={harm[8192]:+.3f} (>=0.20? {harm[8192] >= 0.20}) "
          f"deltaInt={delta_int:+.3f} (<=-0.05? {delta_int <= -0.05})")
    verdict = ("CONFIRMED at the preregistered threshold" if confirmed
               else "NOT CONFIRMED (12A stays screening-only)")
    print(f"  STAGE-B CONFIRMATION verdict: {verdict}")
    print(f"  implied outcome: {outcome}")
    print("\nNOTE: n=20 independent confirmation - aggregates over D in {4,8} "
          "are the meaningful statistics; thresholds UNCHANGED from 12A.")


if __name__ == "__main__":
    main()
