"""Experiment B - saturation characterization of the interference count term.

The product's count term is I_count(D) = min(1, D / D_SATURATION) with
D_SATURATION = 5. That value was never empirically justified: the original
factorial only sampled D in {0,1,2,4,8}. This experiment measures a wider grid

    D in {0,1,2,3,4,5,6,8,10,12,16}

at fixed L = 256 and S = medium, on Qwen2.5-0.5B, to test whether accuracy
actually saturates, and where.

Note on the decoy pool: medium-similarity decoys draw from _DECOY_NOUNS (10
entries) and answer words from the 12 candidate words, so for D > 10 the key
nouns begin to repeat. This is recorded as a documented limitation; the sweep
still exercises a genuinely wider range than the original factorial.

Output:
    results/expB_dsweep_factorial.csv
    results/log_expB.txt

Usage:
    python experiments/run_expB.py
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
WIKI = "data/wiki_train.txt"
OUT = "results/expB_dsweep_factorial.csv"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
D_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
LENGTH = 256
SIMILARITY = "medium"
N_SAMPLES = 20
POSITION = 0.5


def main():
    t0 = time.time()
    model = LocalHFModel(MODEL, threads=8, max_context=2048)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, WIKI, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={MODEL} filler={filler.kind} tokens={filler.size}", flush=True)

    for d in D_GRID:
        run_grid(model, gen, [(LENGTH, POSITION)],
                 n_samples=N_SAMPLES, sweep="expB_d", n_distractors=d,
                 similarity=SIMILARITY, out_csv=OUT)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
