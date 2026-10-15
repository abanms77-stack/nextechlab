"""Exp 8 step 1 - PRE-REGISTER candidate similarity metrics for the decoy grid.

Computes, for every decoy sentence in exp8_grid.DECOYS (with {word} = "apple"):
    cos_sim     - cosine similarity of sentence embeddings (MiniLM)
    jaccard     - token Jaccard overlap with the target sentence
    shared_prefix - shared word-prefix length with the target sentence

The candidate metrics are fixed here, BEFORE the LLM is run, and saved to
results/exp8_similarity_metrics.csv. The LLM run (exp8_run.py) then measures
accuracy per decoy without reference to these values.

Usage:
    python experiments/exp8_make_prediction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp8_grid import DECOYS, TARGET_SENTENCE  # noqa: E402

WORD = "apple"


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def shared_prefix(a, b):
    ta, tb = a.lower().split(), b.lower().split()
    n = 0
    for x, y in zip(ta, tb):
        if x == y:
            n += 1
        else:
            break
    return float(n)


def main():
    target = TARGET_SENTENCE.format(word=WORD)
    sents = [t.format(word=WORD) for _, t in DECOYS]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([target] + sents, normalize_embeddings=True)
    cos = emb[1:] @ emb[0]

    rows = []
    for (label, tmpl), c, s in zip(DECOYS, cos, sents):
        rows.append({
            "label": label,
            "template": tmpl,
            "cos_sim": round(float(c), 4),
            "jaccard": round(jaccard(target, s), 4),
            "shared_prefix": shared_prefix(target, s),
        })
    df = pd.DataFrame(rows).sort_values("cos_sim")
    df.to_csv(ROOT / "results" / "exp8_similarity_metrics.csv", index=False)
    print("PRE-REGISTERED candidate similarity metrics (decoy, {word}=apple):")
    print(df[["label", "cos_sim", "jaccard", "shared_prefix"]].to_string(index=False))


if __name__ == "__main__":
    main()
