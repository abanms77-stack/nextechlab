"""Phase 7E - DECOUPLED conflict resolution (runner).

Applies six frozen policies to the conflicting passage group of each query:
  PROTECT / DROP_MORE_SPECIFIC / DROP_LESS_SPECIFIC / DROP_WEAKER / RANDOM /
  ORACLE. Specificity (token count) is decoupled from correctness and the correct
  passage's position is randomized (verified at freeze time).

Usage:
    python experiments/run_rag_bench_7e.py [model] [--limit=N]
"""

import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from rag_engine import (Passage, estimate_tokens, profile,  # noqa: E402
                        query_coverage)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "rag7e_bench.csv")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
LIMIT = None
for _a in sys.argv[2:]:
    if _a.startswith("--limit="):
        LIMIT = int(_a.split("=", 1)[1])

PROMPT = (
    "Context:\n{ctx}\n\n"
    "Question: {q}\n"
    "Answer with a single word from the allowed set.\nAnswer:"
)


def load_spec():
    with open(os.path.join(RESULTS, "rag7e_queries.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_candidate_ids(model, candidates):
    return [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in candidates]


def score(model, passages, question, cands, cand_ids, answer_id):
    ctx = "\n".join(p.text for p in passages)
    ids = model.tokenizer.encode(PROMPT.format(ctx=ctx, q=question),
                                 add_special_tokens=False)
    correct, prob, argmax, _ = model.predict(ids, cand_ids, answer_id)
    pred = cands[cand_ids.index(argmax)] if argmax in cand_ids else "?"
    return correct, prob, pred


def pick_confident(policy, conflicts, question, rng):
    """Return the index (into `conflicts`) the policy keeps, or None to keep all."""
    if policy == "PROTECT":
        return None
    if policy == "ORACLE":
        return next(i for i, c in enumerate(conflicts) if c["correct"])
    if policy == "RANDOM":
        return rng.randrange(len(conflicts))
    if policy == "DROP_MORE_SPECIFIC":
        return max(range(len(conflicts)), key=lambda i: conflicts[i]["tokens"])
    if policy == "DROP_LESS_SPECIFIC":
        return min(range(len(conflicts)), key=lambda i: conflicts[i]["tokens"])
    if policy == "DROP_WEAKER":
        return max(range(len(conflicts)),
                   key=lambda i: query_coverage(question, conflicts[i]["text"]))
    raise ValueError(policy)


def main():
    spec = load_spec()
    cands = spec["candidates"]
    queries = spec["queries"]
    if LIMIT:
        queries = queries[:LIMIT]
    policies = spec["conditions"]

    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = build_candidate_ids(model, cands)
    ans_id = {w: i for w, i in zip(cands, cand_ids)}

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((str(r["qid"]), str(r["policy"])))

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["qid", "policy", "gold_answer", "pred_word", "correct",
                    "prob_correct", "n_passages", "n_removed", "ctx_tokens",
                    "survivor_correct", "survivor_specificity", "seconds"])
    t0 = time.time()

    for qi, q in enumerate(queries):
        conflicts = q["conflicts"]
        irr = q["irrelevant"]
        base = [Passage(id=f"c{i}", index=i, text=c["text"],
                        tokens=estimate_tokens(c["text"]), relevance=0.0)
                for i, c in enumerate(conflicts)]
        base += [Passage(id=f"r{j}", index=len(conflicts) + j, text=t,
                         tokens=estimate_tokens(t), relevance=0.0)
                 for j, t in enumerate(irr)]
        base, feats = profile(q["question"], base)

        for policy in policies:
            if (q["qid"], policy) in done:
                continue
            tt = time.time()
            rng = random.Random(spec["random_seed"] + qi * 97
                                + (hash(policy) % 97))
            keep_i = pick_confident(policy, conflicts, q["question"], rng)

            kept_idx = (list(range(len(conflicts))) if keep_i is None
                        else [keep_i])
            kept = [base[i] for i in kept_idx]
            kept += [base[len(conflicts) + j] for j in range(len(irr))]
            kept = sorted(kept, key=lambda p: (-p.relevance, -p.competition,
                                               -p.nearest_similarity))
            n_rm = len(base) - len(kept)
            if keep_i is None:
                surv = next(c for c in conflicts if c["correct"])
            else:
                surv = conflicts[keep_i]
            correct, prob, pred = score(
                model, kept, q["question"], cands, cand_ids,
                ans_id[q["gold_answer"]])
            w.writerow([
                q["qid"], policy, q["gold_answer"], pred, correct,
                round(prob, 6), len(kept), n_rm,
                sum(p.tokens for p in kept), int(surv["correct"]),
                surv["specificity"], round(time.time() - tt, 3),
            ])
            fh.flush()
        print(f"[rag7e] {q['qid']} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
