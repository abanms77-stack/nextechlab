"""Main Phase-1 experiments.

Stages:
    position : accuracy vs needle position, at L=512, for d=0 and d=4 decoys
    length   : accuracy vs context length, at position=0.5, for d=0 and d=4

Each result row is appended to CSV immediately (see runner), so runs can be
monitored live and never lose progress.

Usage:
    python experiments/run_main.py position
    python experiments/run_main.py length
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import (  # noqa: E402
    run_length_sweep,
    run_position_sweep,
    summarize,
)
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
POSITIONS = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
LENGTHS = [256, 512, 1024, 2048]
L_FIXED = 512


def get_model():
    return LocalHFModel(MODEL, threads=12)


def stage_position(model, gen):
    for d in (0, 4):
        run_position_sweep(
            model, gen, length=L_FIXED, positions=POSITIONS, n_samples=8,
            out_csv=f"results/exp1_position_d{d}.csv", n_distractors=d,
        )


def stage_length(model, gen):
    for d in (0, 4):
        run_length_sweep(
            model, gen, lengths=LENGTHS, position=0.5, n_samples=4,
            out_csv=f"results/exp2_length_d{d}.csv", n_distractors=d,
        )


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    model = get_model()
    gen = NeedleGenerator(model.tokenizer)
    if stage in ("position", "all"):
        stage_position(model, gen)
    if stage in ("length", "all"):
        stage_length(model, gen)
    print(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
