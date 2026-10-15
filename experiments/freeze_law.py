"""Freeze the Qwen interference law and predict all Llama cells BEFORE running.

This enforces the correct OOD protocol:
    1. fit the law on Qwen data only,
    2. hash + timestamp it (so the prediction is provably pre-registered),
    3. emit predictions for every cell of the Llama factorial,
    4. then (separately) run Llama and compare.

Outputs:
    results/frozen_law_qwen.json   - coefficients + sha256 + timestamp
    results/pred_qwen_on_llama.csv - pre-registered cell predictions

Usage:
    python experiments/freeze_law.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_exp6 import FEATS, make_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The Llama factorial grid - MUST match run_exp7.py exactly.
LENGTHS = [256, 1024]
DECOYS = [0, 1, 2, 4, 8]
SIMILARITIES = ["low", "medium", "high"]


def main():
    qwen = pd.read_csv(RESULTS / "exp6_factorial.csv")
    X = make_features(qwen)
    y = qwen["correct"].to_numpy(int)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X, y)

    coefs = [round(float(c), 6) for c in clf.coef_[0]]
    intercept = round(float(clf.intercept_[0]), 6)
    payload = json.dumps(
        {"coefs": coefs, "intercept": intercept, "feats": FEATS}, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()

    frozen = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_data": "exp6_factorial.csv",
        "source_rows": int(len(qwen)),
        "feats": FEATS,
        "coefs": dict(zip(FEATS, coefs)),
        "intercept": intercept,
        "sha256": digest,
    }
    (RESULTS / "frozen_law_qwen.json").write_text(json.dumps(frozen, indent=2))
    print(f"FROZEN Qwen law  sha256={digest}")
    print(f"  coefficients: {frozen['coefs']}  intercept={intercept}")

    rows = [
        {"length": L, "similarity": s, "n_distractors": d}
        for L in LENGTHS for s in SIMILARITIES for d in DECOYS
    ]
    grid = pd.DataFrame(rows)
    grid["pred_qwen"] = clf.predict_proba(make_features(grid))[:, 1].round(4)
    grid.to_csv(RESULTS / "pred_qwen_on_llama.csv", index=False)
    print("\nPRE-REGISTERED predictions for Llama cells:")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
