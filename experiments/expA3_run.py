"""Experiment A (third model) - run the SAME factorial on gpt2-medium.

gpt2-medium (355M, 24 layers) is a different family from Qwen2.5-0.5B and
Llama-3.2-1B. (distilgpt2 was tried first but sat at the guess-rate floor with
no dynamic range; that run is kept as a documented finding.) The Qwen law was
frozen and predictions pre-registered (results/pred_qwen_on_gpt2medium.csv +
.sha256) BEFORE this script runs.

Grid (must match expA3_freeze_predict.py): L in {256,768}, D in {0,1,2,4,8},
S in {low,medium,high}, n = 20, position 0.5.

Resumable: re-run to continue after any interruption.

Usage:
    python experiments/expA3_run.py
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "gpt2-medium"
WIKI = "data/wiki_train.txt"
OUT = "results/expA3_gpt2medium_factorial.csv"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
# distilgpt2 has n_positions = 1024; the body must leave room for the query
# suffix. We therefore use L in {256, 768} rather than {256, 1024}.
LENGTHS = [256, 768]
DECOYS = [0, 1, 2, 4, 8]
SIMILARITIES = ["low", "medium", "high"]
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

    for sim in SIMILARITIES:
        for d in DECOYS:
            run_grid(model, gen, [(L, POSITION) for L in LENGTHS],
                     n_samples=N_SAMPLES, sweep="expA3", n_distractors=d,
                     similarity=sim, out_csv=OUT)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
