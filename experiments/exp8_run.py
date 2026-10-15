"""Exp 8 step 2 - run the LLM over the continuous-similarity decoy grid.

For each decoy in exp8_grid.DECOYS and each length, generate n samples and
record correctness. Resumable. Real WikiText filler. Position 0.5.

Usage:
    python experiments/exp8_run.py [model]
"""

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402
from exp8_grid import DECOYS, LENGTHS, N_SAMPLES, POSITION  # noqa: E402

WIKI = "data/wiki_train.txt"
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    tag = "qwen" if "Qwen" in model_name else (
        "llama" if "Llama" in model_name else model_name.split("/")[-1].lower())
    out = f"results/exp8_{tag}.csv"

    import csv
    import os
    done = set()
    if os.path.exists(out) and os.path.getsize(out) > 0:
        import pandas as pd
        ex = pd.read_csv(out)
        for _, r in ex.iterrows():
            done.add((r["label"], int(r["length"]), int(r["sample"])))

    model = LocalHFModel(model_name, threads=8, max_context=4096)
    cand_ids = [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                for w in CANDIDATES]
    filler = ProseFiller(model.tokenizer, WIKI, cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={model_name} filler={filler.kind} tokens={filler.size} "
          f"done={len(done)}", flush=True)

    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["label", "length", "sample", "seed", "answer_word",
                    "decoy_word", "correct", "prob_correct", "seconds"])
    t0 = time.time()
    for label, tmpl in DECOYS:
        for L in LENGTHS:
            for k in range(N_SAMPLES):
                if (label, L, k) in done:
                    continue
                seed = 1000 + k
                s = gen.generate_with_decoy(L, tmpl, position=POSITION,
                                            decoy_position=POSITION, seed=seed)
                tt = time.time()
                correct, prob, _, _ = model.predict(
                    s.input_ids, s.candidate_ids, s.answer_token_id)
                w.writerow([label, L, k, seed, s.answer_word,
                            s.meta["decoy_word"], correct,
                            round(prob, 6), round(time.time() - tt, 3)])
                fh.flush()
            print(f"[exp8] {label:<26} L={L:<5} done elapsed={time.time()-t0:.0f}s",
                  flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
