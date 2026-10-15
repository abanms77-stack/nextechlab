"""Phase 1B - Experiment 4 (real-prose interference x length) and Experiment 5
(similarity sweep).

Real WikiText prose replaces the small tiled sentence bank, removing the
documented repetition confound. Stages:

    main       : lengths 256..2048 x decoys {0,1,2,4,8}, similarity=medium
    L4096      : length 4096 x decoys {0,1,2,4,8} (fewer samples)
    similarity : length 1024, 4 decoys, similarity in {low, medium, high}
    all        : main + L4096 + similarity

Usage:
    python experiments/run_exp4.py main
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
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
LENGTHS = [256, 512, 1024, 2048]
DECOYS = [0, 1, 2, 4, 8]
POSITION = 0.5


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    t0 = time.time()

    model = LocalHFModel(MODEL, threads=12)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, WIKI, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"filler={filler.kind} tokens={filler.size}", flush=True)

    if stage in ("main", "all"):
        for d in DECOYS:
            run_grid(model, gen, [(L, POSITION) for L in LENGTHS], n_samples=4,
                     sweep="exp4", n_distractors=d, similarity="medium",
                     out_csv="results/exp4_length_decoys.csv")

    if stage in ("L4096", "rest", "all"):
        for d in DECOYS:
            run_grid(model, gen, [(4096, POSITION)], n_samples=2,
                     sweep="exp4", n_distractors=d, similarity="medium",
                     out_csv="results/exp4b_L4096.csv")

    if stage in ("similarity", "rest", "all"):
        for sim in ("low", "medium", "high"):
            run_grid(model, gen, [(1024, POSITION)], n_samples=6,
                     sweep=f"sim_{sim}", n_distractors=4, similarity=sim,
                     out_csv="results/exp5_similarity.csv")

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
