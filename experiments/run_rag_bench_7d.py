"""Phase 7D - CONFLICT-RESOLUTION policies (runner).

Applies four frozen policies (PROTECT / DROP_WEAKER / SURFACE / ORACLE) to the
contradictory and mixed contexts of the frozen 7B queries, and scores the reader
with the frozen K-way single-token rule.

Usage:
    python experiments/run_rag_bench_7d.py [model] [--limit=N]
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402
from rag_engine import Passage, estimate_tokens, profile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "rag7d_bench.csv")
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
    with open(os.path.join(RESULTS, "rag7d_queries.json"), encoding="utf-8") as fh:
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


def apply_policy(policy, passages, gold_text):
    """Return (kept_passages, n_removed). Passages carry .contradict flags."""
    conflicts = [p for p in passages if p.contradict]
    others = [p for p in passages if not p.contradict]

    if policy == "PROTECT":
        return passages, 0

    if policy == "SURFACE":
        kept = sorted(passages, key=lambda p: (-p.relevance, -p.competition,
                                               -p.nearest_similarity))
        return kept, 0

    if policy == "DROP_WEAKER":
        if not conflicts:
            return passages, 0
        # keep only the highest-token (most specific) conflicting passage
        best = max(conflicts, key=lambda p: (p.tokens, p.relevance))
        kept = others + [best]
        kept = sorted(kept, key=lambda p: (-p.relevance, -p.competition,
                                           -p.nearest_similarity))
        return kept, len(passages) - len(kept)

    if policy == "ORACLE":
        gold = [p for p in passages if p.text == gold_text]
        if not gold:
            gold = others[:1]
        return gold, len(passages) - len(gold)

    raise ValueError(policy)


def main():
    spec = load_spec()
    cands = spec["candidates"]
    queries = spec["queries"]
    if LIMIT:
        queries = queries[:LIMIT]
    ctypes = spec["context_types"]
    policies = spec["policies"]

    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cand_ids = build_candidate_ids(model, cands)
    ans_id = {w: i for w, i in zip(cands, cand_ids)}

    done = set()
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        import pandas as pd
        for _, r in pd.read_csv(OUT).iterrows():
            done.add((str(r["qid"]), str(r["context_type"]), str(r["policy"])))

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["qid", "context_type", "policy", "gold_answer", "pred_word",
                    "correct", "prob_correct", "n_passages", "n_removed",
                    "ctx_tokens", "n_conflicts", "gold_survived", "seconds"])
    t0 = time.time()

    for qi, q in enumerate(queries):
        for ct in ctypes:
            texts = q["contexts"][ct]
            base = [Passage(id=f"p{i}", index=i, text=t,
                            tokens=estimate_tokens(t), relevance=0.0)
                    for i, t in enumerate(texts)]
            base, feats = profile(q["question"], base)
            for policy in policies:
                if (q["qid"], ct, policy) in done:
                    continue
                tt = time.time()
                kept, n_rm = apply_policy(policy, base, q["gold_fact"])
                n_conf = sum(1 for p in base if p.contradict)
                gold_surv = int(any(p.text == q["gold_fact"] for p in kept))
                correct, prob, pred = score(
                    model, kept, q["question"], cands, cand_ids,
                    ans_id[q["gold_answer"]])
                w.writerow([
                    q["qid"], ct, policy, q["gold_answer"], pred, correct,
                    round(prob, 6), len(kept), n_rm,
                    sum(p.tokens for p in kept), n_conf, gold_surv,
                    round(time.time() - tt, 3),
                ])
                fh.flush()
        print(f"[rag7d] {q['qid']} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
