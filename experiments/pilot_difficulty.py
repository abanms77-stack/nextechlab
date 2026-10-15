"""Pilot: find a difficulty regime by varying the number of decoy passages.

We need accuracy in a *sensitive* range (neither 0% nor 100%) so that context
length / position actually has room to matter.
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid, summarize  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
LENGTH = 512
POSITION = 0.5
DISTRACTORS = [0, 3, 6, 9, 12]
N_SAMPLES = 6

model = LocalHFModel(MODEL, threads=12)
gen = NeedleGenerator(model.tokenizer)

t0 = time.time()
print(f"pilot: model={MODEL} L={LENGTH} pos={POSITION} samples={N_SAMPLES}")
for d in DISTRACTORS:
    df = run_grid(
        model, gen, [(LENGTH, POSITION)], n_samples=N_SAMPLES,
        sweep=f"d{d}", n_distractors=d, out_csv="results/pilot_difficulty.csv",
    )
    s = summarize(df)
    print(
        f"  distractors={d:<3} acc={float(s['accuracy'].iloc[0]):.2f} "
        f"p_correct={float(s['mean_prob_correct'].iloc[0]):.3f}"
    )
print(f"done in {time.time()-t0:.0f}s")
