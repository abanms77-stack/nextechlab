"""Phase 11 - position benchmark runner (the only model run of Phase 11).

Runs the retrieval sweep at L=256, S=medium, D in {0,4}, across the
preregistered 7-position grid, n=10/cell, using the SAME wiki-prose filler
as the reference cells. Position is the ONLY manipulated variable - task,
query, filler token count, length, similarity and needle content are frozen.
This is the only new model run in Phase 11.

Resumable: re-running skips any (sweep, L, pos, D, S, sample) already in the
output CSV.

Output:
  results/phase11_factorial.csv

Usage:
  python experiments/run_phase11.py [model] [--limit=N]
"""

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
FILLER_PATH = "data/wiki_train.txt"  # same domain as reference
OUT = "results/phase11_factorial.csv"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]
D_GRID = [0, 4]
POSITIONS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
LENGTH = 256
SIMILARITY = "medium"
N_SAMPLES = 10
SWEEP = "phase11_position"


def main():
    t0 = time.time()
    model = LocalHFModel(MODEL, threads=8, max_context=2048)
    cand_ids = [
        model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in CANDIDATES
    ]
    filler = ProseFiller(model.tokenizer, FILLER_PATH, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={MODEL} filler={filler.kind} size={filler.size} "
          f"path={FILLER_PATH}", flush=True)

    cells = [(LENGTH, p) for p in POSITIONS]
    for d in D_GRID:
        run_grid(model, gen, cells,
                 n_samples=N_SAMPLES, sweep=SWEEP, n_distractors=d,
                 similarity=SIMILARITY, out_csv=OUT)
        print(f"  D={d} done, elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
