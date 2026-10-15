"""Phase 12B - long-context PENALTY CONFIRMATION (independent, preregistered).

Phase 12A (n=5/cell) STAGE-A = INTERESTING and STAGE-B = CONFIRMED at the
preregistered screening threshold (harm8192 = +0.375, deltaInt = -0.400), and
called for this confirmation run. 12B is a GENUINELY INDEPENDENT replication:
it re-samples the SAME L x D x S design with a FRESH, DISJOINT set of random
seeds and larger n, and applies the decision thresholds and model spec FROZEN
in 12A UNCHANGED - no threshold shopping, no redesign after seeing 12A.

  QUESTION (the box):
      Does the competition-dependent long-context penalty (harm growing with L,
      D=0 flat) REPLICATE on fresh samples at higher n?

  FROZEN (identical to 12A)
      task=retrieval, wiki-prose filler, S=medium, position=0.5, needle+decoys,
      12-word answer set; exp6 S=medium L=256 baseline (zero new runs).

  MANIPULATED (only independence dimensions differ from 12A)
      length  L in {2048, 4096, 8192}          (identical grid)
      decoys  D in {0, 4, 8}                   (identical grid)
      n = 20 / cell   (independent samples)     (12A was n=5)
      sample index k in [5, 25)  => FRESH, DISJOINT seeds vs 12A's k in [0,5).
      => 3 L x 3 D x 1 S x 20 = 180 samples (90 new beyond a 12A-matched 5).

  WHY k IN [5,25) GIVES GENUINELY INDEPENDENT SAMPLES
      runner.py seed = round(pos*1000)*1000 + k + L  =  500000 + k + L (pos 0.5).
      12A used k in [0,5) -> seeds 502053.. ; 12B uses k in [5,25) -> seeds
      502053+5 .. ; the two ranges are DISJOINT (verified: no collisions), so
      every 12B context is a different filler draw / answer word / decoy
      placement than 12A. Yet answer_word = rng.choice(candidates) depends only
      on the seed (not on D), and length is token-exact, so the candidate
      distribution and context length are IDENTICAL to 12A - independent draws
      from the SAME population (an independent replication, not a new design).

  DECISION RULE (COPIED VERBATIM FROM phase12a_protocol.json - UNCHANGED):
      meanAtL(D) = mean acc over D in {4,8} at length L
      harm(L)    = meanAtL(256) - meanAtL(L)      (exp6 baseline, same as 12A)
      deltaInt   = meanAtL(8192) - meanAtL(2048)
      STAGE-A "INTERESTING": max(harm4096, harm8192) >= 0.15 OR deltaInt <= -0.15
      STAGE-B "CONFIRMED"  : harm(8192) >= 0.20 AND deltaInt <= -0.05
      These are the SAME thresholds 12A used; 12B tests them on fresh data at
      n=20 (SE ~0.10 per cell, ~0.05 per aggregate), so a Stage-B pass here is
      the genuine confirmation 12A's n=5 could only screen for.

  MODEL COMPARISON (spec COPIED VERBATIM from 12A - UNCHANGED):
      M_A  F(log(1+D), S)                 2 params
      M_B  F(log(1+D), S, L)              3 params
      M_C  F(log(1+D), S, L, log(1+D)xL)  4 params
      evaluated with log-loss / Brier / Spearman; M_A->M_B should carry the
      predictive gain if the penalty is length-under-competition.

  PRE-DECLARED OUTCOMES A / B / C: as frozen in 12A (A = L x D penalty
      replicates; B = length harms ONLY under competition, D=0 flat; C = length
      barely matters, composition dominates).

  HONESTY NOTES frozen in advance:
    * "Confirmed" means the UNCHANGED Stage-B rule passes on FRESH n=20 data.
      If it fails, 12A's result stands as screening-only; we do NOT lower the
      threshold to rescue it.
    * Seed ranges k in [5,25) overlap no 12A sample (k in [0,5)); all 180 rows
      are written to a SEPARATE CSV so the independence is auditable.
    * n=20 resolves ~0.10/cell; aggregates over D in {4,8} are the meaningful
      statistics (per-cell diffs are still not over-read).
    * S=medium only (S robustness deferred to Phase 13), same as 12A.
    * length token-exact => filler/query/evidence/decoys matched; L is the only
      variable across the length axis.

Output:
  results/phase12b_protocol.json + .sha256

Usage:
  python experiments/phase12b_freeze.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = RESULTS / "phase12b_protocol.json"

# ---- design (identical grid to 12A; only n and k-range differ) ----
LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
S_FIXED = "medium"
POSITION = 0.5
N_SAMPLES = 20            # independent samples / cell (12A was 5)
K_START, K_END = 5, 25    # fresh, DISJOINT sample indices vs 12A's [0,5)
BASELINE_L = 256          # exp6 short-L anchor (unchanged)
D_HARM = [4, 8]           # aggregate over these D for harm/deltaInt (unchanged)

# ---- decision thresholds + model spec: COPIED VERBATIM from 12A ----
INTEREST_THRESHOLD = 0.15   # Stage-A interesting
CONFIRM_HARM = 0.20         # Stage-B: harm8192 >=
CONFIRM_DELTA_INT = -0.05   # Stage-B: deltaInt <=

SEED_FORMULA = "round(pos*1000)*1000 + k + length  (= 500000 + k + L at pos 0.5)"


def main():
    # Cross-check: thresholds must match 12A's frozen protocol exactly (no
    # post-hoc tampering). If 12A's protocol is present, assert equality; the
    # constants above were copied from it.
    a12a = RESULTS / "phase12a_protocol.json"
    if a12a.exists():
        pa = json.loads(a12a.read_text(encoding="utf-8"))
        ra = pa["decision_rule"]
        assert (f"harm(8192) >= {CONFIRM_HARM} AND deltaInt <= "
                f"{CONFIRM_DELTA_INT}") == ra["stage_b_confirmed"], \
            "12B Stage-B rule differs from 12A frozen rule!"
        assert (f"max(harm(4096), harm(8192)) >= {INTEREST_THRESHOLD} OR "
                f"deltaInt <= -{INTEREST_THRESHOLD}") == ra["stage_a_interesting"], \
            "12B Stage-A rule differs from 12A frozen rule!"
        print("verified: 12B decision thresholds == 12A frozen thresholds (unchanged)")

    proto = {
        "phase": "12B",
        "title": ("long-context penalty CONFIRMATION (independent, "
                  "preregistered)"),
        "frozen_before_run": True,
        "extends": ("phase12a_protocol.json (thresholds + model spec COPIED "
                    "UNCHANGED; only n and seed range differ)"),
        "task": "retrieval",
        "domain": "wiki-prose",
        "filler": "data/wiki_train.txt",
        "similarity": S_FIXED,
        "position": POSITION,
        "answer_set_size": 12,
        "manipulated": {"length": LENGTHS, "n_distractors": D_GRID,
                        "similarity": [S_FIXED]},
        "target_new": {
            "n_samples_per_cell": N_SAMPLES,
            "n_cells": len(LENGTHS) * len(D_GRID),
            "n_samples_total": len(LENGTHS) * len(D_GRID) * N_SAMPLES,
            "sample_index_range": [K_START, K_END],
            "seed_formula": SEED_FORMULA,
            "independence": (
                "k in [5,25) is disjoint from 12A's k in [0,5); seeds are "
                "500000+k+L, verified collision-free vs 12A and within 12B. "
                "answer_word depends only on seed (not D) and length is "
                "token-exact, so 12B draws are independent samples from the "
                "SAME population as 12A."),
        },
        "baselines": {
            "exp6_factorial.csv": ("S=medium, n=20, L in {256,1024} - the "
                                   "short-L anchor; ZERO new baseline runs"),
            "exp4_length_decoys.csv": "S=medium, n=4, L in {256,512,1024,2048}",
        },
        "models": [
            {"name": "M_A  F(log(1+D), S)", "params": 2,
             "desc": "competition only (length never harms)"},
            {"name": "M_B  F(log(1+D), S, L)", "params": 3,
             "desc": "additive length term (length harms regardless of D)"},
            {"name": "M_C  F(log(1+D), S, L, log(1+D) x L)", "params": 4,
             "desc": "competition x length interaction (competition more "
                     "damaging as L grows)"},
        ],
        "comparison": ("M_A vs M_B (does L add predictive power?) vs M_C (does "
                       "L x D add more?), evaluated on fresh 12B cells using "
                       "log-loss/Brier + Spearman - spec identical to 12A."),
        "decision_rule": {
            "stats": {
                "meanAtL(D)": "mean acc over D in {4,8} at length L",
                "harm(L)": "meanAtL(256) - meanAtL(L), from exp6 baseline",
                "deltaInt": "meanAtL(8192) - meanAtL(2048)",
            },
            "stage_a_interesting": (
                f"max(harm(4096), harm(8192)) >= {INTEREST_THRESHOLD} OR "
                f"deltaInt <= -{INTEREST_THRESHOLD}"),
            "stage_b_confirmed": (
                f"harm(8192) >= {CONFIRM_HARM} AND "
                f"deltaInt <= {CONFIRM_DELTA_INT}"),
            "threshold_justification": (
                "UNCHANGED from 12A's frozen rule. 12A set these before "
                "seeing its own data; 12B tests them on FRESH n=20 samples "
                "(SE ~0.10/cell, ~0.05/aggregate), so a Stage-B pass here is "
                "the genuine confirmation 12A could only screen for."),
        },
        "outcomes": {
            "A": ("Long-context penalty REPLICATES: L x D negative and Stage-B "
                  "passes on fresh n=20 -> 'effective context' thesis "
                  "confirmed (R = F(task, domain, log(1+D), S, L) with a "
                  "harmful L interaction)."),
            "B": ("Length matters ONLY under competition: D=0 stays flat at "
                  "long L but D>0 degrades -> large context becomes unreliable "
                  "only when it contains competing evidence."),
            "C": ("Length barely matters / penalty does NOT replicate -> "
                  "context COMPOSITION, not length, is the dominant reliability "
                  "variable; 12A stays screening-only."),
        },
        "metrics": ["accuracy", "log-loss", "Brier", "MAE", "Spearman"],
        "honesty": [
            "'Confirmed' = UNCHANGED Stage-B rule passes on FRESH n=20 data; "
            "if it fails, 12A remains screening-only (we do NOT lower the "
            "threshold).",
            "k in [5,25) overlaps no 12A sample; all 180 rows go to a SEPARATE "
            "CSV so independence is auditable.",
            "n=20 resolves ~0.10/cell; aggregates over D in {4,8} are the "
            "meaningful statistics (per-cell diffs not over-read).",
            "S=medium only (S robustness deferred to Phase 13).",
            "length token-exact => L is the only variable across the L axis.",
        ],
    }

    blob = json.dumps(proto, indent=2, sort_keys=True).encode()
    PROTOCOL.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (RESULTS / "phase12b_protocol.sha256").write_text(digest + "\n",
                                                      encoding="utf-8")
    total = proto["target_new"]["n_samples_total"]
    print(f"FROZEN {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    print(f"lengths={LENGTHS} D={D_GRID} S={S_FIXED} n={N_SAMPLES}/cell "
          f"k=[{K_START},{K_END}) => {total} samples "
          f"({total - len(LENGTHS) * len(D_GRID) * 5} new beyond a 12A-matched 5)")
    print("stage-A interesting rule (unchanged): "
          f"{proto['decision_rule']['stage_a_interesting']}")
    print("stage-B confirmed rule   (unchanged): "
          f"{proto['decision_rule']['stage_b_confirmed']}")


if __name__ == "__main__":
    main()
