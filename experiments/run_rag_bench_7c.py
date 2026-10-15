"""Phase 7C - HARDER redundancy / CHOICE ATTRIBUTION (runner).

Only the `gold` passage yields the answer. ContextLab prunes the redundant
duplicates of the WEAK roles (partial_dup, indirect_dup) and never the gold;
RANDOM removes the same count at random and will sometimes delete the gold.

Conditions (frozen): BASELINE | RANDOM | CONTEXTLAB | ORACLE.
Scoring is frozen (single token, K-way over the 12-word candidate set).

Queries are pre-registered (results/rag7c_queries.json) BEFORE this runs.

Usage:
    python experiments/run_rag_bench_7c.py [model] [--limit=N]
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
from rag_engine import Passage, estimate_tokens, profile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "rag7c_bench.csv")
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
    with open(os.path.join(RESULTS, "rag7c_queries.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_candidate_ids(model, candidates):
    return [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in candidates]


def assemble(passages, question):
    ctx = "\n".join(p.text for p in passages)
    return PROMPT.format(ctx=ctx, q=question)


def score(model, passages, question, cands, cand_ids, answer_id):
    ids = model.tokenizer.encode(assemble(passages, question),
                                 add_special_tokens=False)
    correct, prob, argmax, _ = model.predict(ids, cand_ids, answer_id)
    pred = cands[cand_ids.index(argmax)] if argmax in cand_ids else "?"
    return correct, prob, pred


def main():
    spec = load_spec()
    cands = spec["candidates"]
    queries = spec["queries"]
    if LIMIT:
        queries = queries[:LIMIT]

    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = build_candidate_ids(model, cands)
    ans_id = {w: i for w, i in zip(cands, cand_ids)}

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((str(r["qid"]), str(r["condition"])))

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["qid", "condition", "gold_answer", "pred_word", "correct",
                    "prob_correct", "n_passages", "n_removed", "ctx_tokens",
                    "gold_survived", "distractor_survived", "competitors",
                    "competitor_similarity", "interference", "seconds"])
    t0 = time.time()

    for qi, q in enumerate(queries):
        items = q["passages"]
        roles = [p["role"] for p in items]
        passages = [Passage(id=f"p{i}", index=i, text=p["text"],
                            tokens=estimate_tokens(p["text"]), relevance=0.0)
                    for i, p in enumerate(items)]
        passages, feats = profile(q["question"], passages)
        removed = [p for p in passages if not p.keep]
        kept = [p for p in passages if p.keep]
        kept = sorted(kept, key=lambda p: (-p.relevance, -p.competition,
                                           -p.nearest_similarity))
        n_rm = len(removed)

        rrng = random.Random(spec["random_seed"] + qi * 131)
        rand_keep = set(range(len(passages)))
        if n_rm > 0:
            for i in rrng.sample(range(len(passages)), n_rm):
                rand_keep.discard(i)
        rand_kept = [passages[i] for i in sorted(rand_keep)]

        # ORACLE: keep the gold (first gold-role passage) + best support only
        gold_idx = roles.index("gold")
        others = [i for i in range(len(passages)) if i != gold_idx]
        others.sort(key=lambda i: (-passages[i].relevance,
                                   -passages[i].nearest_similarity))
        oracle_idx = [gold_idx] + others[:1]
        oracle = [passages[i] for i in oracle_idx]
        o_rm = len(passages) - len(oracle)

        def survival(ps):
            gs = any(p.index == gold_idx for p in ps)
            ds = any(roles[p.index] == "distractor" for p in ps)
            return int(gs), int(ds)

        conditions = {
            "BASELINE": (passages, 0),
            "RANDOM": (rand_kept, n_rm),
            "CONTEXTLAB": (kept, n_rm),
            "ORACLE": (oracle, o_rm),
        }
        for cond, (ps, n_rem) in conditions.items():
            if (q["qid"], cond) in done:
                continue
            tt = time.time()
            gs, ds = survival(ps)
            correct, prob, pred = score(
                model, ps, q["question"], cands, cand_ids,
                ans_id[q["gold_answer"]])
            w.writerow([
                q["qid"], cond, q["gold_answer"], pred, correct,
                round(prob, 6), len(ps), n_rem, sum(p.tokens for p in ps),
                gs, ds, feats["competitors"],
                round(feats["competitor_similarity"], 4),
                round(feats["interference"], 4),
                round(time.time() - tt, 3),
            ])
            fh.flush()
        print(f"[rag7c] {q['qid']} n_rm={n_rm} elapsed={time.time()-t0:.0f}s",
              flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
