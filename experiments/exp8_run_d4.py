"""Exp 8b - continuous similarity at D=4 (the detection floor was D=1).

At D=1 accuracy was 1.00 for every decoy (below threshold). This rerun places
the SAME decoy 4 times (D=4) at L=1024, sweeping the 20 pre-registered
similarity levels, so the accuracy-vs-similarity curve has dynamic range.

Usage:
    python experiments/exp8_run_d4.py [model]
"""

import csv
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402
from exp8_grid import DECOYS  # noqa: E402

WIKI = "data/wiki_train.txt"
DEFAULT_L = 1024   # override via argv[2] (gpt2-family caps at 1024 total)
D = 4
N = 10
POS = 0.5
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_L
    tag = "qwen" if "Qwen" in model_name else (
        "llama" if "Llama" in model_name else model_name.split("/")[-1].lower())
    out = f"results/exp8b_{tag}_d4.csv"
    done = set()
    if os.path.exists(out) and os.path.getsize(out) > 0:
        import pandas as pd
        for _, r in pd.read_csv(out).iterrows():
            done.add((r["label"], int(r["sample"])))

    model = LocalHFModel(model_name, threads=8, max_context=4096)
    cand_ids = [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                for w in CANDIDATES]
    filler = ProseFiller(model.tokenizer, WIKI, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={model_name} D={D} L={L} done={len(done)}", flush=True)

    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["label", "length", "n_copies", "sample", "seed",
                    "answer_word", "decoy_word", "correct", "prob_correct",
                    "seconds"])
    t0 = time.time()
    for label, tmpl in DECOYS:
        for k in range(N):
            if (label, k) in done:
                continue
            s = gen.generate_with_decoy(L, tmpl, position=POS,
                                        decoy_position=POS, seed=2000 + k,
                                        n_copies=D)
            tt = time.time()
            correct, prob, _, _ = model.predict(
                s.input_ids, s.candidate_ids, s.answer_token_id)
            w.writerow([label, L, D, k, 2000 + k, s.answer_word,
                        s.meta["decoy_word"], correct, round(prob, 6),
                        round(time.time() - tt, 3)])
            fh.flush()
        print(f"[exp8b] {label:<26} done elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()