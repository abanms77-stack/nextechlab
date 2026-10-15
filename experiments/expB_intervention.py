"""Phase B - intervention experiment (does pruning improve accuracy?).

We build a retrieval context containing the needle plus 8 decoys of mixed
measured similarity, then compare accuracy under four conditions:

    FULL     : all 8 decoys                     (no intervention)
    NONE     : 0 decoys                         (upper bound)
    RANDOM   : remove 4 decoys at random        (naive pruning)
    SIM      : remove the 4 highest-S decoys    (profiler-guided pruning)

If SIM > RANDOM, the *measured interference* is actionable: removing the most
similar competing passages recovers accuracy. Resumable, real prose filler.

Usage:
    python experiments/expB_intervention.py [model]
"""

import csv
import os
import random
import sys
import time

sys.path.insert(0, ".")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from contextlab.tasks.filler import ProseFiller  # noqa: E402
from contextlab.tasks.generator import NeedleGenerator  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
OUT = "results/expB_intervention.csv"
L = 1024
N = 20
POS = 0.5
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]

# decoy templates with measured cosine similarity to the target sentence
# (values from results/exp8_similarity_metrics.csv)
DECOYS = [
    ("sim_identical", " Important: the vault password is {word}.", 1.00),
    ("sim_note", " Note: the vault password is {word}.", 0.96),
    ("sim_strongbox", " Important: the strongbox password is {word}.", 0.69),
    ("sim_safe", " Important: the safe password is {word}.", 0.76),
    ("mid_backup", " The backup password is {word}.", 0.53),
    ("mid_admin", " The admin password is {word}.", 0.51),
    ("low_market", " The market opened early on Monday.", 0.01),
    ("low_weather", " The weather was warm today.", -0.04),
]
PRUNE_K = 4


def build_ids(gen, decoys, answer_word, decoy_words, rng):
    needle = gen._needle_tokens[answer_word]
    segs = [(POS, needle)]
    for i, (_, tmpl, _) in enumerate(decoys):
        segs.append((0.05 + 0.9 * rng.random(),
                     gen._enc(tmpl.format(word=decoy_words[i]))))
    return gen._assemble(L, segs, rng) + gen._query_ids


def main():
    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((r["condition"], int(r["sample"])))
    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                for w in CANDIDATES]
    filler = ProseFiller(model.tokenizer, "data/wiki_train.txt", cand_ids)
    gen = NeedleGenerator(model.tokenizer, filler=filler)
    print(f"model={MODEL} L={L} done={len(done)}", flush=True)

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["condition", "sample", "n_decoys", "removed_sim",
                    "kept_sim", "answer_word", "correct", "prob_correct",
                    "seconds"])
    t0 = time.time()
    for k in range(N):
        seed = 7000 + k
        rng = random.Random(seed)
        answer_word = rng.choice(CANDIDATES)
        pool = [w0 for w0 in CANDIDATES if w0 != answer_word]
        rng.shuffle(pool)
        # 8 decoy words (reuse pool cyclically)
        decoy_words = [pool[i % len(pool)] for i in range(len(DECOYS))]

        conditions = {}
        conditions["NONE"] = []
        conditions["FULL"] = list(range(len(DECOYS)))
        rnd_idx = list(range(len(DECOYS)))
        rng.shuffle(rnd_idx)
        conditions["RANDOM"] = sorted(rnd_idx[:len(DECOYS) - PRUNE_K])
        by_sim = sorted(range(len(DECOYS)),
                        key=lambda i: DECOYS[i][2], reverse=True)
        conditions["SIM"] = sorted(by_sim[PRUNE_K:])

        for cond, keep in conditions.items():
            if (cond, k) in done:
                continue
            decoys = [DECOYS[i] for i in keep]
            r2 = random.Random(seed)  # identical filler for fair comparison
            ids = build_ids(gen, decoys, answer_word, decoy_words, r2)
            tt = time.time()
            correct, prob, _, _ = model.predict(ids, cand_ids,
                                                gen._enc(" " + answer_word)[0])
            removed = sorted(DECOYS[i][2] for i in range(len(DECOYS))
                             if i not in keep)
            kept = sorted(DECOYS[i][2] for i in keep)
            w.writerow([cond, k, len(decoys), "|".join(map(str, removed)),
                        "|".join(map(str, kept)), answer_word, correct,
                        round(prob, 6), round(time.time() - tt, 3)])
            fh.flush()
        print(f"[expB] sample {k} done elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()