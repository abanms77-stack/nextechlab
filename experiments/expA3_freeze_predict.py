"""Experiment A (third model) - pre-register predictions BEFORE collecting data.

The Qwen law is ALREADY frozen (results/frozen_law_qwen.json, sha256 recorded).
This script uses that frozen law - with ZERO refitting - to emit predictions
for every cell of the distilgpt2 factorial. The output file is written and
hashed BEFORE experiments/expA3_run.py produces any distilgpt2 data, so the
prediction is provably pre-registered.

Model choice: gpt2-medium (355M, 24 layers) - a different family from Qwen2.5
(0.5B) and Llama-3.2 (1B). We FIRST tried distilgpt2 (82M) but it sat at the
guess-rate floor (~0.15, chance = 1/12 = 0.083) with no interference signal, so
it has no dynamic range for a transfer test. See results/expA3_distilgpt2_*
(kept as a documented capacity-floor finding). gpt2-medium demonstrably has the
dynamic range required (Exp 8: 1.0 -> 0.2 -> 0.1 as similarity rose).

gpt2-medium has n_positions = 1024, so body length is capped at 768.

Outputs:
    results/pred_qwen_on_distilgpt2.csv   - pre-registered cell predictions
    results/pred_qwen_on_distilgpt2.sha256 - hash of that file

Usage:
    python experiments/expA3_freeze_predict.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_exp6 import FEATS, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The distilgpt2 factorial grid - MUST match expA3_run.py exactly.
# distilgpt2 has n_positions = 1024, so the body length is capped well below
# that (a 1024-token body + query suffix overflows GPT-2's position embedding).
LENGTHS = [256, 768]
DECOYS = [0, 1, 2, 4, 8]
SIMILARITIES = ["low", "medium", "high"]


def sigmoid(z):
    return 1.0 / (1.0 + pow(2.718281828459045, -z))


def main():
    frozen = json.loads((RESULTS / "frozen_law_qwen.json").read_text())
    coefs = frozen["coefs"]
    intercept = frozen["intercept"]
    print(f"USING FROZEN Qwen law sha256={frozen['sha256']}")
    print(f"  coefs={coefs}")

    rows = [
        {"length": L, "similarity": s, "n_distractors": d}
        for L in LENGTHS for s in SIMILARITIES for d in DECOYS
    ]
    grid = pd.DataFrame(rows)

    X = make_features(grid)  # column order == FEATS
    z = intercept + sum(coefs[f] * X[:, i] for i, f in enumerate(FEATS))
    grid["pred_qwen"] = [round(sigmoid(v), 4) for v in z]
    grid["frozen_sha256"] = frozen["sha256"]

    out = RESULTS / "pred_qwen_on_gpt2medium.csv"
    grid.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "pred_qwen_on_gpt2medium.sha256").write_text(digest + "\n")

    print(f"\nPRE-REGISTERED predictions written to {out.name}")
    print(f"  file sha256 = {digest}")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
