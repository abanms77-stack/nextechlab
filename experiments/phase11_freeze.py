"""Phase 11 - position effects (pre-registration / freeze).

Phases 8-10 fixed position=0.5. Phase 11 asks the clean isolation question:

    Does the SAME context produce different reliability SOLELY because the
    evidence needle is placed at a different position P?  And - the crucial
    test - does position explain residual error AFTER D, S, L are known?

The protocol is frozen BEFORE any new cell is run.

  FROZEN (held fixed across all cells)
    task=retrieval, query text, filler corpus (wiki prose) + exact filler
    token count, L=256, S=medium, D grid, passage/needle content.

  MANIPULATED (the only free variable)
    evidence position  P in {0.05,0.15,0.30,0.50,0.70,0.85,0.95}  (= P/T)

  NEW DATA (only model run of this phase)
    factorial cells: retrieval, L=256, S=medium, D in {0,4},
                     P in the 7-point grid, n=10/cell  => 140 cells.
                     D in {0,4} lets us test position x D interaction.

  EXISTING-DATA REFERENCE (zero-refit baseline, no new runs)
    the frozen position=0.5 D-sweep cells (results/expB_dsweep_factorial.csv)
    calibrate the base reliability model; position is then evaluated
    held-out (zero-refit on position).

  MODELS (zero-refit, NO position tuning for the base)
    base_log    P = sigma(a + b*log(1+D))            - frozen from reference
    base_law    7-feature ContextLab law (frozen)    - frozen from reference
    pos_edge    base + edge term  c*|2P-1|           - position extension
    pos_quad    base + quadratic  c*(2P-1)^2         - position extension
    (position coefficients are fit on a PREREGISTERED SPLIT - see below)

  PRIMARY TEST (the box):
    Does position explain residual error after D, S, L are known?
    Baseline (position-blind) held-out MAE/ECE on the position grid vs
    position-aware model MAE/ECE, evaluated on a HELD-OUT position split.

  SPLIT (preregistered, so position is evaluated out-of-sample):
    fit   positions {0.05,0.30,0.70,0.95}   (edges + near-centre extremes)
    test  positions {0.15,0.50,0.85}         (unseen positions)
    => position coefficient is estimated on the fit half only, then
       judged on unseen positions (zero-refit to test positions).

Preregistered metrics: Spearman, MAE, RMSE, ECE (primary); Pearson, calib
slope/intercept, delta-R2 vs baseline, params (secondary).

Preregistered outcome (declared BEFORE running):
  A - position adds information: |Spearman| >= 0.8 on held-out positions AND
      pos-aware model beats base by >= 0.03 MAE (extend law to
      R = F(task, domain, f(D), S, L, P))
  B - position ranks cells but does not improve calibration (rank yes,
      MAE gain < 0.03)
  C - position is redundant: position-aware model fails to beat base AND
      |Spearman| < 0.8 (position adds nothing beyond D/S/L)
Honesty notes frozen in advance:
  * position = EVIDENCE (needle) position only. Decoy positions are
    independently randomized per seed (as in all prior phases) and are
    controlled as random noise across cells - NOT a second manipulated axis.
  * n=10/cell, 7 positions, 2 D => verdicts preliminary.
  * position coefficients fit ONLY on the preregistered fit-split; test
    positions never seen during fitting (true zero-refit on position).
  * filler token count exact (same corpus), so position is not confounded
    with amount of filler; needle passage length identical across positions.

Output:
  results/phase11_protocol.json + .sha256

Usage:
  python experiments/phase11_freeze.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC_REF = RESULTS / "expB_dsweep_factorial.csv"
PROTOCOL = RESULTS / "phase11_protocol.json"

L_FIXED = 256
S_FIXED = "medium"
D_GRID = [0, 4]
POSITIONS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
N_SAMPLES = 10
FIT_POS = [0.05, 0.30, 0.70, 0.95]
TEST_POS = [0.15, 0.50, 0.85]

FORMS = ["base_log", "base_law", "pos_edge", "pos_quad"]
OUT = RESULTS / "phase11_factorial.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for p in (SRC_REF,):
        if not p.exists():
            sys.exit(f"missing {p}")

    ref = pd.read_csv(SRC_REF)
    ref_d = sorted(int(d) for d in ref[ref.sweep == "expB_d"]
                   ["n_distractors"].unique())

    # --- part 2 appended below ---

    proto = {
        "name": "phase11_position_v1",
        "frozen": True,
        "reference_source": SRC_REF.name,
        "reference_sha256": _sha(SRC_REF),
        "reference_conditions": {
            "domain": "wiki prose filler (data/wiki_train.txt)",
            "length": L_FIXED, "similarity": S_FIXED,
            "position": 0.5, "D_grid": ref_d, "n_per_cell": 20,
        },
        "target_new": {
            "file": OUT.name,
            "domain": "wiki prose filler (same corpus as reference)",
            "length": L_FIXED, "similarity": S_FIXED,
            "D_grid": D_GRID, "positions": POSITIONS,
            "n_per_cell": N_SAMPLES,
            "model": "Qwen/Qwen2.5-0.5B",
        },
        "manipulated_variable": "evidence (needle) position P only",
        "held_fixed": [
            "task=retrieval", "query text", "filler corpus + token count",
            "length=256", "similarity=medium", "needle passage content",
        ],
        "split": {
            "fit_positions": FIT_POS,
            "test_positions": TEST_POS,
            "note": "position coefficients fit ONLY on fit_positions; "
                    "test_positions never seen during fitting (zero-refit)",
        },
        "models_compared": FORMS,
        "model_defs": {
            "base_log": "P = sigma(a + b*log(1+D)); (a,b) frozen from reference",
            "base_law": "7-feature ContextLab law (frozen_law_qwen.json)",
            "pos_edge": "base_log + c*|2P-1|; (a,b) frozen, c fit on fit-split",
            "pos_quad": "base_log + c*(2P-1)^2; (a,b) frozen, c fit on fit-split",
        },
        "primary_comparison": "base_log vs pos-aware models on held-out positions",
        "primary_test": "does position explain residual error after D, S, L "
                        "are known?",
        "protocol": "calibrate base on position=0.5 reference -> freeze -> "
                    "evaluate position extension held-out (zero-refit on "
                    "test positions)",
        "primary_metrics": ["Spearman", "MAE", "RMSE", "ECE"],
        "secondary_metrics": ["Pearson", "calib_slope", "calib_intercept",
                              "delta_R2_vs_base", "n_params"],
        "outcome_interpretation": {
            "A": "position adds information (|Spearman| >= 0.8 on held-out "
                 "positions AND pos-aware beats base by >= 0.03 MAE)",
            "B": "position ranks cells but does not improve calibration "
                 "(|Spearman| >= 0.8, MAE gain < 0.03)",
            "C": "position redundant (fails to beat base and |Spearman| < 0.8)",
        },
        "honesty_notes": [
            "position = evidence (needle) position only; decoy positions "
            "independently randomized per seed, controlled as random noise",
            "n=10/cell, 7 positions, 2 D: verdicts preliminary",
            "position coefficients fit only on fit-split; test positions "
            "never seen during fitting (true zero-refit on position)",
            "filler token count exact (same corpus); needle passage length "
            "identical across positions",
        ],
    }
    PROTOCOL.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    digest = _sha(PROTOCOL)
    (RESULTS / "phase11_protocol.sha256").write_text(digest + "\n",
                                                     encoding="utf-8")
    print(f"FROZEN Phase 11 protocol -> {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    print(f"positions={POSITIONS} D={D_GRID} n={N_SAMPLES}/cell "
          f"=> {len(POSITIONS)*len(D_GRID)*N_SAMPLES} cells")
    print(f"fit={FIT_POS} test={TEST_POS}")


if __name__ == "__main__":
    main()
