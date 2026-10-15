"""Phase 12A - long-context scaling (PRE-REGISTRATION / SCREENING FREEZE).

Phases 8-11 fixed relatively short contexts. Phase 12A reconnects to the
original thesis - how much of a context window is actually usable? - with a
cheap, honest, RESUMABLE screening run rather than an 8-hour full factorial.

  QUESTION (the box):
      Does increasing context length create a REPRODUCIBLE reliability
      penalty, particularly when competing evidence is present?
      i.e. does the L x D interaction turn negative (competition becomes more
      damaging as contexts get long), stay flat, or reverse?

  FROZEN (held fixed across all cells)
      task=retrieval, query text, filler corpus (wiki prose), S=medium,
      position=0.5, needle + decoy content, 12-word answer set.

  MANIPULATED (the only free variables)
      length  L in {2048, 4096, 8192}      (token-exact body length)
      decoys  D in {0, 4, 8}
      n = 5 / cell   (screening; NOT confirmatory power)
      => 3 L x 3 D x 1 S x 5 = 45 samples (the 45-run screening).

  WHY ONLY THESE LENGTHS / n=5
      Existing exp4/exp6 already cover L in {256,512,1024} (exp6 at n=20).
      The genuinely unmeasured regime is 2048-8192, so screening runs ONLY
      there. Cost is ~linear in L (measured: 9.7s@1024, 18s@2048, 33s@4096,
      ~60s@8192 per sample) so 45 runs ~ 1.6h median, not 7-8h. n=5 is a
      SCREENING device (is there an effect worth confirming?), explicitly NOT
      final inference - binomial SE at n=5 is ~0.18-0.22, so single-cell
      differences are NOT read as effects.

  BASELINES (existing data, ZERO new runs - the short-L reference)
      exp6_factorial.csv  S=medium, n=20, L in {256,1024}
      exp4_length_decoys  S=medium, n=4,  L in {256,512,1024,2048}

  DECISION RULE (frozen BEFORE any cell runs -> what triggers 12B):
      Screening STATISTICS are AGGREGATES so the n=5 noise averages down:
        meanAtL(D) = mean accuracy over D in {4,8} at length L
        harm(L)    = meanAtL(256) - meanAtL(L)   (from exp6 baseline)
                     > 0 => competition more damaging at L than at 256
        deltaInt   = meanAtL(8192) - meanAtL(2048)
                     < 0 => penalty GROWS with length (L x D interaction)
      STAGE-A "INTERESTING" if max(harm(4096),harm(8192)) >= 0.15 OR
                               deltaInt <= -0.15   => run Phase 12B (n=10-20).
      STAGE-B "CONFIRMED"   if harm(8192) >= 0.20 AND deltaInt <= -0.05.
      Threshold justification (frozen): binomial diff-of-means SE ~0.32 at
      n=5, so 0.15 is a GO/NO-GO on an AGGREGATE (not a per-cell claim);
      0.20 sits above noise and is resolvable at n>=10-20 (the 12B
      confirmation). A negative interaction >= 0.05 is the smallest change we
      commit to confirm.

  PRE-DECLARED OUTCOMES A / B / C (see MODEL COMPARISON below).

  HONESTY NOTES frozen in advance:
    * n=5 screening => Stage-A verdicts PRELIMINARY; only 12B confirms.
    * S=medium only in 12A (S robustness = Phase 13); S held fixed, so this
      phase does NOT estimate an S term.
    * We do NOT assume longer is worse - D=4 already rises 0.50->0.75 from
    L=256->1024 (exp4), so A, B and C are all live.
  * length is token-exact by construction (generate() builds exactly L body
    tokens), so filler amount, query, evidence and decoys are matched and L
    is the ONLY thing that changes across the L axis.

Output:
  results/phase12a_protocol.json + .sha256

Usage:
  python experiments/phase12a_freeze.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC_EXP6 = RESULTS / "exp6_factorial.csv"
SRC_EXP4 = RESULTS / "exp4_length_decoys.csv"
PROTOCOL = RESULTS / "phase12a_protocol.json"

LENGTHS = [2048, 4096, 8192]
D_GRID = [0, 4, 8]
S_FIXED = "medium"
POSITION = 0.5
N_SAMPLES = 5
SWEEP = "phase12a_screen"
OUT = RESULTS / "phase12a_factorial.csv"

BASELINE_L = 256
INTEREST_THRESHOLD = 0.15     # Stage-A -> 12B
CONFIRM_HARM = 0.20           # Stage-B harm(8192)
CONFIRM_DELTA_INT = -0.05     # Stage-B interaction must not exceed this


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for p in (SRC_EXP6, SRC_EXP4):
        if not p.exists():
            sys.exit(f"missing {p}")

    e6 = pd.read_csv(SRC_EXP6)
    ref = e6[e6.similarity == S_FIXED]
    baseline = (ref.groupby(["length", "n_distractors"])["correct"].mean()
                .reset_index())

    proto = {
        "name": "phase12a_longctx_screen_v1",
        "frozen": True,
        "stage": "A (screening)",
        "question": ("Does increasing context length create a reproducible "
                     "reliability penalty, especially under competition "
                     "(L x D interaction)?"),
        "reference_source": [SRC_EXP6.name, SRC_EXP4.name],
        "reference_sha256": {SRC_EXP6.name: _sha(SRC_EXP6),
                             SRC_EXP4.name: _sha(SRC_EXP4)},
        "baseline_exp6_medium": baseline.to_dict(orient="records"),
        "target_new": {
            "file": OUT.name,
            "domain": "wiki prose filler (data/wiki_train.txt)",
            "similarity": S_FIXED, "position": POSITION,
            "lengths": LENGTHS, "D_grid": D_GRID, "n_per_cell": N_SAMPLES,
            "n_samples_total": len(LENGTHS) * len(D_GRID) * N_SAMPLES,
            "model": "Qwen/Qwen2.5-0.5B",
            "max_position_embeddings": 32768,
            "position_note": ("max_context passed as 8192; adapter never "
                              "truncates and model allows 32768, so 8192 is "
                              "safe."),
        },
        "models": [
            {"name": "M_A  F(log(1+D), S)",
             "desc": "competition only; L ignored (length never harms)"},
            {"name": "M_B  F(log(1+D), S, L)",
             "desc": "additive length term (length harms regardless of D)"},
            {"name": "M_C  F(log(1+D), S, L, log(1+D) x L)",
             "desc": "competition x length interaction (competition more "
                     "damaging as L grows)"},
        ],
        "comparison": ("M_A vs M_B (does L add predictive power?) vs M_C (does "
                       "L x D add more?), evaluated zero-refit on held-out "
                       "long-L cells using log-loss/Brier/MAE + Spearman."),
        "decision_rule": {
            "stats": {
                "meanAtL(D)": "mean acc over D in {4,8} at length L",
                "harm(L)": "meanAtL(256) - meanAtL(L), from exp6 baseline",
                "deltaInt": "meanAtL(8192) - meanAtL(2048)",
            },
            "stage_a_interesting": ("max(harm(4096), harm(8192)) >= "
                                    f"{INTEREST_THRESHOLD} OR "
                                    f"deltaInt <= -{INTEREST_THRESHOLD}"),
            "stage_b_confirmed": (f"harm(8192) >= {CONFIRM_HARM} AND "
                                  f"deltaInt <= {CONFIRM_DELTA_INT}"),
            "threshold_justification": (
                "binomial diff-of-means SE ~0.32 at n=5, so 0.15 is a "
                "GO/NO-GO on an AGGREGATE (not per-cell); 0.20 sits above "
                "noise and is resolvable at n>=10-20 (the 12B confirmation)."),
        },
        "outcomes": {
            "A": ("Long-context penalty exists: L x D term negative and "
                  "confirmed -> supports 'effective context' thesis (R = F("
                  "task, domain, log(1+D), S, L) with a harmful L "
                  "interaction)."),
            "B": ("Length matters ONLY under competition: D=0 stays flat at "
                  "long L but D>0 degrades -> large context is not inherently "
                  "unreliable; it becomes unreliable when it contains "
                  "competing evidence."),
            "C": ("Length barely matters: no L term / interaction survives "
                  "-> context COMPOSITION, not length, is the dominant "
                  "reliability variable."),
        },
        "metrics": ["accuracy", "log-loss", "Brier", "MAE", "Spearman"],
        "honesty": [
            "n=5 screening => Stage-A verdicts PRELIMINARY; only 12B confirms.",
            "S=medium only (S robustness deferred to Phase 13).",
            "We do NOT assume longer is worse; A, B and C are all live.",
            "length token-exact => filler/query/evidence/decoys matched; L is "
            "the only variable across the L axis.",
        ],
    }

    blob = json.dumps(proto, indent=2, sort_keys=True).encode()
    PROTOCOL.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (RESULTS / "phase12a_protocol.sha256").write_text(digest + "\n",
                                                      encoding="utf-8")
    print(f"FROZEN {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    print(f"lengths={LENGTHS} D={D_GRID} S={S_FIXED} n={N_SAMPLES}/cell "
          f"=> {proto['target_new']['n_samples_total']} samples")
    print(f"stage-A interesting rule: max(harm4096,harm8192) >= "
          f"{INTEREST_THRESHOLD} OR deltaInt <= -{INTEREST_THRESHOLD}")
    print(f"stage-B confirmed rule:  harm8192 >= {CONFIRM_HARM} AND "
          f"deltaInt <= {CONFIRM_DELTA_INT}")
    print("\nexp6 S=medium baseline (zero-refit):")
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()

