"""Phase 7B - REDUNDANCY intervention benchmark (runner).

Controlled end-to-end RAG: for every query and every context type we feed a
KNOWN passage set to the reader (so redundancy/contradiction structure is
exactly as constructed), under four frozen conditions:

    BASELINE   : all passages, original order
    RANDOM     : remove a random subset (matched to ContextLab's removed count)
    CONTEXTLAB : profiler prune (redundancy only) + reorder (strongest first)
    ORACLE     : keep gold fact + one best supporting passage only

Scoring is frozen: single token from the frozen 12-word candidate set (K-way
argmax), same as the rest of the project.

Queries/contexts are pre-registered (results/rag7b_queries.json) BEFORE this.

Usage:
    python experiments/run_rag_bench_7b.py [model] [--limit=N]
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
OUT = os.path.join(RESULTS, "rag7b_bench.csv")
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
    with open(os.path.join(RESULTS, "rag7b_queries.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_candidate_ids(model, candidates):
    return [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in candidates]


def assemble(passages, question):
    ctx = "\n".join(p.text for p in passages)
    return PROMPT.format(ctx=ctx, q=question)


def score(model, passages, question, cands, cand_ids, answer_id):
    text = assemble(passages, question)
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    correct, prob, argmax, _ = model.predict(ids, cand_ids, answer_id)
    pred = cands[cand_ids.index(argmax)] if argmax in cand_ids else "?"
    return correct, prob, pred, len(ids)


def oracle_select(passages, gold_text):
    """Ideal selector: the gold fact + the single best supporting passage.

    'Best supporting' = highest relevance among non-gold passages. Returns the
    gold first (strongest position) then the best support.
    """
    gold = [p for p in passages if p.text == gold_text]
    others = [p for p in passages if p.text != gold_text]
    others.sort(key=lambda p: (-p.relevance, -p.nearest_similarity))
    kept = gold[:1] + others[:1]
    if not gold:  # fallback (should not happen: gold is always present)
        kept = others[:2]
    return kept, len(passages) - len(kept)


def main():
    spec = load_spec()
    cands = spec["candidates"]
    queries = spec["queries"]
    if LIMIT:
        queries = queries[:LIMIT]
    ctypes = spec["context_types"]

    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = build_candidate_ids(model, cands)
    ans_id = {w: i for w, i in zip(cands, cand_ids)}

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((str(r["qid"]), str(r["context_type"]), str(r["condition"])))

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["qid", "context_type", "condition", "gold_answer",
                    "pred_word", "correct", "prob_correct", "n_passages",
                    "n_removed", "ctx_tokens", "competitors",
                    "competitor_similarity", "interference", "seconds"])
    t0 = time.time()

    for qi, q in enumerate(queries):
        for ct in ctypes:
            texts = q["contexts"][ct]
            passages = [Passage(id=f"p{i}", index=i, text=t,
                                tokens=estimate_tokens(t), relevance=0.0)
                        for i, t in enumerate(texts)]
            passages, feats = profile(q["question"], passages)
            removed = [p for p in passages if not p.keep]
            kept = [p for p in passages if p.keep]
            kept = sorted(kept, key=lambda p: (-p.relevance, -p.competition,
                                               -p.nearest_similarity))
            n_rm = len(removed)

            rrng = random.Random(spec["random_seed"] + qi * 97
                                 + (hash(ct) % 97))
            rand_keep = set(range(len(passages)))
            if n_rm > 0:
                for i in rrng.sample(range(len(passages)), n_rm):
                    rand_keep.discard(i)
            rand_kept = [passages[i] for i in sorted(rand_keep)]

            oracle, o_rm = oracle_select(passages, q["gold_fact"])

            conditions = {
                "BASELINE": (passages, 0),
                "RANDOM": (rand_kept, n_rm),
                "CONTEXTLAB": (kept, n_rm),
                "ORACLE": (oracle, o_rm),
            }
            for cond, (ps, n_rem) in conditions.items():
                if (q["qid"], ct, cond) in done:
                    continue
                tt = time.time()
                ctx_tokens = sum(p.tokens for p in ps)
                correct, prob, pred, _ = score(
                    model, ps, q["question"], cands, cand_ids,
                    ans_id[q["gold_answer"]])
                w.writerow([
                    q["qid"], ct, cond, q["gold_answer"], pred, correct,
                    round(prob, 6), len(ps), n_rem, ctx_tokens,
                    feats["competitors"],
                    round(feats["competitor_similarity"], 4),
                    round(feats["interference"], 4),
                    round(time.time() - tt, 3),
                ])
                fh.flush()
        print(f"[rag7b] {q['qid']} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
