"""Phase 10 - cross-domain generalization (pre-registration / freeze).

Phase 9 showed the count-term FORM transfers across tasks but the
coefficients are task-dependent. Phase 10 asks:

    Does the zero-refit retrieval law survive a change of CONTEXT DOMAIN
    (different filler prose)?

The protocol is frozen BEFORE any new cell is run:

  NEW DATA (only model run of this phase)
    domain-B cells  : retrieval D-sweep at L=256, S=medium, position=0.5,
                      D in {0,4,8}, n=10/cell, filler = GENERAL ENGLISH
                      prose (NLTK raw corpora, URL-stripped) instead of the
                      wiki prose used for the reference cells.
    reference cells : existing wiki-prose D-sweep cells at the same
                      conditions (results/expB_dsweep_cells.csv).

  EXISTING-DATA TRANSFER (no new model runs)
    cross-length ROI: frozen 7-feature law + Phase-8 forms, fit on Exp6
                      L=256 cells, evaluated zero-refit on L=1024 cells.

  MODELS COMPARED (zero-refit, NO target tuning)
    log        P = sigma(a + b*log(1+D))   - Phase-8 winner, 2 params
    linear     P = sigma(a + b*D)          - 2 params
    min_sat    P = sigma(a + b*min(1,D/k)) - 3 params
    frozen     7-feature ContextLab law    - 7 params (baseline)

  PRIMARY COMPARISON: log vs frozen law on the new-domain cells.

Preregistered metrics: Spearman, MAE, RMSE, ECE (primary); Pearson,
calibration slope/intercept, monotonicity violations, params (secondary).

Preregistered outcome (declared BEFORE running):
  A - |Spearman| >= 0.8 and MAE <= 0.15 on the new-domain cells
  B - |Spearman| >= 0.8 but MAE > 0.15 (rank yes, calibration no)
  C - |Spearman| < 0.8 (fails)
Honesty notes frozen in advance:
  * n=10/cell, 3 D points => Spearman verdicts are preliminary.
  * No intercept recalibration anywhere (true zero-refit).
  * Domain-B filler matched to reference clean tokens within 1% of corpus
    size, so domain is not confounded with filler amount.

Output:
  results/phase10_protocol.json + .sha256

Usage:
  python experiments/phase10_freeze.py           # freeze only
  python experiments/phase10_freeze.py --build   # freeze + build domain-B filler
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SRC_REF = RESULTS / "expB_dsweep_factorial.csv"
SRC_EXP6 = RESULTS / "exp6_factorial.csv"
PROTOCOL = RESULTS / "phase10_protocol.json"

S_MEDIUM = 0.5
L_FIXED = 256
D_GRID = [0, 4, 8]
N_SAMPLES = 10
POSITION = 0.5

FORMS = ["log", "linear", "min_sat"]

OUT_DOMAINB = RESULTS / "phase10_domainB_factorial.csv"
FILLER_DOMAINB = DATA / "filler_general.txt"

FREEZE_ONLY = "--build" not in sys.argv


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- part 2: build_domain_filler() and main() appended below ---



def build_domain_filler() -> dict:
    """Build the domain-B filler corpus (general English, URL-free).

    Source: NeelNanda/pile-10k (HF datasets, already in local cache) - a
    general-English forum/blog sample of The Pile, genuinely different in
    domain from the wiki prose reference, with no URL-like text after
    stripping. Offline: reads from the local HF cache, no network needed.

    The corpus is TRUNCATED TO MATCH THE REFERENCE'S CLEAN-TOKEN COUNT
    (345,590 tokens, computed by the same ProseFiller recipe) within +/-1%,
    rather than matching raw character count - token count is the quantity
    that determines filler-pool size, so this holds "filler amount" fixed
    while letting the DOMAIN (register/style/vocabulary) differ.
    Returns a manifest dict written into the protocol.
    """
    import re

    from datasets import load_dataset

    cand = ["apple", "river", "tiger", "cloud", "stone", "candle",
            "silver", "garden", "planet", "forest", "harbor", "lantern"]
    from transformers import AutoTokenizer

    import numpy as np

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cand_ids = np.asarray(sorted(
        tok.encode(" " + w, add_special_tokens=False)[0] for w in cand))

    ref_txt = (DATA / "wiki_train.txt").read_text(
        encoding="utf-8", errors="ignore")[:1_500_000]
    ref_ids = np.asarray(tok.encode(ref_txt, add_special_tokens=False))
    n_ref = int((~np.isin(ref_ids, cand_ids)).sum())
    target_tokens = n_ref

    def clean_count(txt: str) -> int:
        ids = np.asarray(tok.encode(txt, add_special_tokens=False))
        return int((~np.isin(ids, cand_ids)).sum())

    # Stream pile-10k, keeping accumulating text until clean tokens ~ target.
    ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
    budget_chars = 4_000_000  # generous cap; we stop on token count below
    chunks, total_chars = [], 0
    buf, buf_tokens = [], 0
    for rec in ds:
        t = rec.get("text") or ""
        if not t or re.search(r"https?://|www\.", t, re.I):
            continue
        take = t[: budget_chars - total_chars]
        if not take:
            break
        chunks.append(take)
        total_chars += len(take)
        # recount on the whole accumulated text periodically (cheap enough)
        if total_chars % 500_000 < len(take):
            text = "".join(chunks)
            text = "\n".join(ln for ln in text.splitlines()
                             if not re.search(r"https?://|www\.", ln, re.I))
            buf_tokens = clean_count(text)
            if buf_tokens >= target_tokens:
                break
        if total_chars >= budget_chars:
            break
    text = "\n".join(ln for ln in "".join(chunks).splitlines()
                     if not re.search(r"https?://|www\.", ln, re.I))

    # Trim to target token count by dropping trailing lines.
    lines = text.splitlines()
    while lines and clean_count("\n".join(lines)) > int(target_tokens * 1.01):
        cut = max(1, len(lines) // 50)
        lines = lines[:-cut]
    text = "\n".join(lines)
    while clean_count(text) < int(target_tokens * 0.99) and total_chars < budget_chars:
        sys.exit("domain-B filler fell under token budget; adjust stream logic")

    if re.search(r"https?://|www\.", text, re.I):
        sys.exit("domain-B filler still contains URL-like text")
    FILLER_DOMAINB.write_text(text, encoding="utf-8")

    n_clean = clean_count(text)
    diff_pct = 100.0 * abs(n_clean - n_ref) / n_ref
    if diff_pct > 1.0:
        sys.exit(f"domain-B filler off by {diff_pct:.2f}% vs reference (>1%)")

    return {
        "path": str(FILLER_DOMAINB.relative_to(ROOT)),
        "sha256": _sha(FILLER_DOMAINB),
        "clean_tokens": n_clean,
        "reference_clean_tokens": n_ref,
        "pct_diff": round(diff_pct, 3),
        "sources": "hf:NeelNanda/pile-10k (streamed from local cache, url-stripped, "
                   "truncated to match reference clean-token count +/-1%)",
    }



def main():
    for p in (SRC_REF, SRC_EXP6):
        if not p.exists():
            sys.exit(f"missing {p}")

    ref = pd.read_csv(SRC_REF)
    ref_d = sorted(int(d) for d in ref[ref.sweep == "expB_d"]
                   ["n_distractors"].unique())
    e6 = pd.read_csv(SRC_EXP6)
    e6_l = sorted(int(x) for x in e6["length"].unique())
    e6_d = sorted(int(x) for x in e6["n_distractors"].unique())

    filler_manifest = None
    if not FREEZE_ONLY:
        filler_manifest = build_domain_filler()

    proto = {
        "name": "phase10_cross_domain_v1",
        "frozen": True,
        "reference_source": SRC_REF.name,
        "reference_sha256": _sha(SRC_REF),
        "reference_conditions": {
            "domain": "wiki prose filler (data/wiki_train.txt)",
            "length": L_FIXED, "similarity": "medium",
            "position": POSITION, "D_grid": ref_d, "n_per_cell": 20,
        },
        "target_new": {
            "file": OUT_DOMAINB.name,
            "domain": "general-English filler (data/filler_general.txt)",
            "length": L_FIXED, "similarity": "medium",
            "position": POSITION, "D_grid": D_GRID,
            "n_per_cell": N_SAMPLES,
            "model": "Qwen/Qwen2.5-0.5B",
            "filler_manifest": filler_manifest,
        },
        "target_cross_length": {
            "source": SRC_EXP6.name,
            "sha256": _sha(SRC_EXP6),
            "train": f"length={e6_l[0]} cells",
            "test": f"length={e6_l[-1]} cells",
            "D_grid": e6_d, "S_grid": ["low", "medium", "high"],
        },
        "models_compared": FORMS + ["frozen_law"],
        "model_defs": {
            "log": "P = sigma(a + b*log(1+D)); (a,b) frozen from reference curve",
            "linear": "P = sigma(a + b*D); (a,b) frozen from reference curve",
            "min_sat": "P = sigma(a + b*min(1,D/k)); (a,b,k) frozen from reference",
            "frozen_law": "7-feature ContextLab law (frozen_law_qwen.json, sha256 cf79eeae...)",
        },
        "primary_comparison": "log vs frozen_law on target_new (domain-B cells)",
        "protocol": "fit on reference -> freeze -> evaluate on target (no refit)",
        "primary_metrics": ["Spearman", "MAE", "RMSE", "ECE"],
        "secondary_metrics": ["Pearson", "mono_viol", "calib_slope",
                              "calib_intercept", "n_params"],
        "outcome_interpretation": {
            "A": "cross-domain transfer works (|Spearman| >= 0.8, MAE <= 0.15)",
            "B": "rank transfers, calibration does not (|Spearman| >= 0.8, MAE > 0.15)",
            "C": "fails (|Spearman| < 0.8 on domain-B cells)",
        },
        "honesty_notes": [
            "n=10/cell and 3 D points: Spearman verdicts are preliminary",
            "no intercept recalibration anywhere (true zero-refit)",
            "domain-B filler matched to reference clean tokens within 1%",
            "cross-length target is cell-level (20 samples/cell, existing data)",
        ],
    }
    PROTOCOL.write_text(json.dumps(proto, indent=2), encoding="utf-8")
    digest = _sha(PROTOCOL)
    (RESULTS / "phase10_protocol.sha256").write_text(digest + "\n",
                                                     encoding="utf-8")
    print(f"FROZEN Phase 10 protocol -> {PROTOCOL.name}")
    print(f"sha256 = {digest}")
    if filler_manifest:
        print(f"domain-B filler: {filler_manifest['clean_tokens']} clean tokens "
              f"({filler_manifest['pct_diff']}% off reference)")
    else:
        print("(freeze only; run with --build to construct the domain-B filler)")


if __name__ == "__main__":
    main()
