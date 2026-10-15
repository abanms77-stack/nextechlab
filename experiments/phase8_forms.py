"""Phase 8 - D functional-form study (pre-registration / freeze).

The product's count term is I(D) = min(1, D / D_SATURATION) with
D_SATURATION = 5. Phase 1B-6 showed this value is NOT recovered by the D-sweep
(best-fit k ~= 9). Phase 8 asks the sharper question: WHICH functional form of
the count term f(D) best predicts accuracy on HELD-OUT D values?

This file freezes the protocol BEFORE any fit:

  data        : results/expB_dsweep_factorial.csv (Qwen2.5-0.5B, L=256,
                S=medium, D in {0,1,2,3,4,5,6,8,10,12,16}, n=20 per cell)
  target      : P(correct) as a function of D
  candidates  : f(D) in { linear, min-saturation, exponential-saturation,
                          logarithmic, learned-monotonic(isotonic) }
  model       : P = sigma(a + b*f(D))                (S, L fixed in the sweep)
  protocol    : leave-out-D cross-validation. For each held-out D the model is
                fit ONLY on the other D values, then predicts the held-out D.
                Interpolation splits (hold out interior D) and extrapolation
                splits (hold out the largest D) are reported separately.
  metrics     : held-out MAE, RMSE, Spearman rho, ECE (calibration),
                monotonicity violations, parameter count
  baseline    : the FROZEN 7-feature ContextLab law (results/frozen_law_qwen.json)
                evaluated on the same held-out D (its S and L terms fixed).

Frozen comparison (preregistered): for each metric, is there a candidate whose
functional form beats the frozen law on held-out D? Declared before fitting.

Output:
  results/phase8_protocol.json + .sha256
  results/phase8_forms.csv        (fits + held-out metrics)
  results/phase8_loo.csv          (per-held-out-D predictions)

Usage:
  python experiments/phase8_forms.py            # freeze + fit + report
  python experiments/phase8_forms.py --freeze   # freeze only
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC = RESULTS / "expB_dsweep_factorial.csv"
FROZEN_LAW = RESULTS / "frozen_law_qwen.json"
PROTOCOL = RESULTS / "phase8_protocol.json"

S_MEDIUM = 0.5
L_FIXED = 256

CANDIDATES = ["linear", "min_sat", "exp_sat", "log", "isotonic"]
FREEZE_ONLY = "--freeze" in sys.argv


def freeze():
    if not SRC.exists():
        sys.exit(f"missing {SRC}; run experiments/run_expB.py first")
    df = pd.read_csv(SRC)
    df = df[df["sweep"] == "expB_d"] if "sweep" in df.columns else df
    dvals = sorted(int(d) for d in df["n_distractors"].unique())

    proto = {
        "name": "phase8_d_functional_form_v1",
        "frozen": True,
        "source": SRC.name,
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "model": "Qwen/Qwen2.5-0.5B",
        "length": L_FIXED,
        "similarity": "medium",
        "S_weight": S_MEDIUM,
        "target": "P(correct | D)",
        "model_form": "P = sigma(a + b * f(D))",
        "candidates": CANDIDATES,
        "candidate_defs": {
            "linear": "f(D) = D",
            "min_sat": "f(D) = min(1, D/k); k fit by held-out",
            "exp_sat": "f(D) = 1 - exp(-D/k); k fit by held-out",
            "log": "f(D) = log(1 + D)",
            "isotonic": "f(D) = learned monotone step function (isotonic)",
        },
        "protocol": "leave-one-D-out; fit on all other D, predict held-out D",
        "splits": {
            "interpolation": "hold out each interior D (1..12), train on the rest",
            "extrapolation": "hold out D=16 (largest), train on D<=12",
        },
        "metrics": ["MAE", "RMSE", "Spearman", "ECE", "mono_violations",
                    "n_params"],
        "baseline": {
            "name": "frozen 7-feature ContextLab law",
            "source": "results/frozen_law_qwen.json",
            "feats": ["log2L", "D", "S", "D*S", "D*log2L", "S*log2L",
                      "D*S*log2L"],
        },
        "preregistered_questions": [
            "Q1 which f(D) minimises held-out MAE (interpolation)?",
            "Q2 which f(D) survives extrapolation to D=16?",
            "Q3 does any candidate beat the frozen 7-feature law on held-out D?",
            "Q4 how many monotonicity violations does each candidate make?",
        ],
        "d_grid": dvals,
    }
    PROTOCOL.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    (RESULTS / "phase8_protocol.sha256").write_text(digest + "\n",
                                                    encoding="utf-8")
    print(f"FROZEN Phase 8 protocol -> {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    print(f"D grid = {dvals}")
    print(f"candidates = {' | '.join(CANDIDATES)}")
    return proto


if __name__ == "__main__":
    freeze()
    if FREEZE_ONLY:
        sys.exit(0)
    print("NOTE: run the full study with the fitting code (phase8_fit.py).")
