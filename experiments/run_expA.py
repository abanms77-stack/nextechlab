"""Phase A - cross-task validation on a COUNTING task family.

Tests whether the interference structure P = f(D, S, L) survives a task that is
NOT key retrieval: count the "red" objects while excluding items of a similar
colour. D = number of similar-colour items; S = measured RGB similarity between
the target colour and the competitor colour (rescan-free, deterministic).

Grid: L in {256, 1024}, D in {0,2,4,8}, S in {low,medium,high}, n = 10.
Resumable.

Usage:
    python experiments/run_expA.py [model]
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.task_families import CountingTask  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
OUT = "results/expA_counting.csv"
LENGTHS = [256, 1024]
DECOYS = [0, 2, 4, 8]
LEVELS = ["low", "medium", "high"]
N = 10
POS = 0.5


def main():
    t0 = time.time()
    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    task = CountingTask(model.tokenizer)
    print(f"model={MODEL} task=counting", flush=True)
    for lvl in LEVELS:
        for d in DECOYS:
            run_grid(model, task, [(L, POS) for L in LENGTHS], n_samples=N,
                     sweep="expA", n_distractors=d, similarity=lvl,
                     out_csv=OUT)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()