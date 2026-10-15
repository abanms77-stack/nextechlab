"""Phase C - robustness checks.

Tests whether the interference result survives changes in construction that
were not part of the main experiment:

    * target position  p in {0.1, 0.5, 0.9}
    * filler source    (real WikiText vs the small sentence bank)
    * random seeds     (n = 12 per cell)

Fixed: retrieval task, L = 1024, D = 4, similarity = medium. Resumable.

Usage:
    python experiments/run_expC.py
"""

import sys
import time

sys.path.insert(0, ".")

from contextlab.experiments.runner import run_grid  # noqa: E402
from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = "results/expC_robustness.csv"
L = 1024
D = 4
N = 12
POSITIONS = [0.1, 0.5, 0.9]
CAND = ["apple", "river", "tiger", "cloud", "stone", "candle",
        "silver", "garden", "planet", "forest", "harbor", "lantern"]


def main():
    t0 = time.time()
    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                for w in CAND]
    prose = ProseFiller(model.tokenizer, "data/wiki_train.txt", cand_ids)

    # (label, filler) - prose vs the legacy sentence bank
    gen_prose = NeedleGenerator(model.tokenizer, filler=prose)
    gen_bank = NeedleGenerator(model.tokenizer)

    run_grid(model, gen_prose, [(L, p) for p in POSITIONS], n_samples=N,
             sweep="pos_prose", n_distractors=D, similarity="medium",
             out_csv=OUT)
    run_grid(model, gen_bank, [(L, p) for p in POSITIONS], n_samples=N,
             sweep="pos_bank", n_distractors=D, similarity="medium",
             out_csv=OUT)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()