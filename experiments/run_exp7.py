"""Experiment 7 - the SAME factorial on a different model (Llama-3.2-1B).

This is the cross-model out-of-distribution test. The Qwen law was frozen
(see experiments/freeze_law.py) BEFORE this script is run; predictions for
every cell below already exist in results/pred_qwen_on_llama.csv.

Grid (must match freeze_law.py):
    L in {256, 1024}, D in {0,1,2,4,8}, S in {low, medium, high}, n = 20.

Resumable: re-run to continue after any interruption.

Usage:
    python experiments/run_exp7.py
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
WIKI = "data/wiki_train.txt"
OUT = "results/exp7_llama_factorial.csv"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
LENGTHS = [256, 1024]
DECOYS = [0, 1, 2, 4, 8]
SIMILARITIES = ["low", "medium", "high"]
N_SAMPLES = 20
POSITION = 0.5


def main():
    t0 = time.time()
    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, WIKI, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={MODEL} filler={filler.kind} tokens={filler.size}", flush=True)

    for sim in SIMILARITIES:
        for d in DECOYS:
            run_grid(model, gen, [(L, POSITION) for L in LENGTHS],
                     n_samples=N_SAMPLES, sweep="exp7", n_distractors=d,
                     similarity=sim, out_csv=OUT)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
