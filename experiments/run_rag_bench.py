"""Phase 7 - REAL RAG intervention benchmark (runner).

End-to-end: real retriever (MiniLM+FAISS over local wiki) -> three frozen
conditions -> real LLM (Qwen2.5-0.5B) -> real answers.

    A BASELINE   : top-k passages -> LLM
    B CONTEXTLAB : top-k -> profiler prune/reorder -> LLM
    C RANDOM     : top-k -> random prune (same # removed) -> LLM

Scoring is frozen: the LLM must answer with a single candidate word; we restrict
the next-token distribution to the 12 candidate tokens and take the argmax
(same clean K-way scoring as the rest of the project).

Queries are pre-registered (results/rag_queries.json) BEFORE this runs.

Usage:
    python experiments/run_rag_bench.py [model]
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
from rag_retriever import Retriever  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
WIKI = os.path.join(ROOT, "data", "wiki_train.txt")
OUT = os.path.join(RESULTS, "rag_bench.csv")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"

PROMPT = (
    "Context:\n{ctx}\n\n"
    "Question: {q}\n"
    "Answer with a single word from the allowed set.\nAnswer:"
)


def load_spec():
    with open(os.path.join(RESULTS, "rag_queries.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_candidate_ids(model, candidates):
    return [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in candidates]


def assemble(passages, question):
    ctx = "\n".join(p.text for p in passages)
    return PROMPT.format(ctx=ctx, q=question)


def answer_ids_for(model, passages, question, cand_ids, answer_id):
    text = assemble(passages, question)
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    correct, prob, argmax, probs = model.predict(ids, cand_ids, answer_id)
    return correct, prob, argmax, len(ids)


def main():
    spec = load_spec()
    cands = spec["candidates"]
    top_k = spec["top_k"]
    rng_seed = spec["random_seed"]
    queries = spec["queries"]

    # ---- build the corpus with all gold facts + decoys injected -----------
    extra = []
    for q in queries:
        extra.append(q["gold_fact"])
        for d in q["decoys"]:
            extra.append(d["text"])
    print(f"building retriever over {WIKI} (+{len(extra)} injected facts)",
          flush=True)
    retr = Retriever(spec["embedder"]).build(
        __import__("pathlib").Path(WIKI), extra)
    print(f"corpus chunks = {len(retr.texts)}", flush=True)

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
                    "competitors", "competitor_similarity", "interference",
                    "retrieval_score_max", "seconds"])
    t0 = time.time()

    for qi, q in enumerate(queries):
        hit = retr.search(q["question"], top_k)
        # profiler scoring over the retrieved passages
        passages = [Passage(id=f"p{i}", index=h["index"], text=h["text"],
                            tokens=estimate_tokens(h["text"]), relevance=0.0)
                    for i, h in enumerate(hit)]
        passages, feats = profile(q["question"], passages)
        removed = [p for p in passages if not p.keep]
        kept = [p for p in passages if p.keep]
        # ContextLab also REORDERS: strongest evidence first (relevance, then
        # competition). Frozen policy - declared in rag_preregister.py.
        kept = sorted(kept, key=lambda p: (-p.relevance, -p.competition,
                                           -p.nearest_similarity))

        # deterministic random control: remove the SAME number, random choice
        rrng = random.Random(rng_seed + qi)
        n_rm = len(removed)
        rand_keep_idx = set(range(len(passages)))
        if n_rm > 0:
            for i in rrng.sample(range(len(passages)), n_rm):
                rand_keep_idx.discard(i)
        rand_kept = [passages[i] for i in sorted(rand_keep_idx)]

        conditions = {
            "BASELINE": (passages, 0),
            "CONTEXTLAB": (kept, n_rm),
            "RANDOM": (rand_kept, n_rm),
        }
        for cond, (ps, n_rem) in conditions.items():
            if (q["qid"], cond) in done:
                continue
            tt = time.time()
            correct, prob, argmax, ntokens = answer_ids_for(
                model, ps, q["question"], cand_ids, ans_id[q["gold_answer"]])
            pred = cands[cand_ids.index(argmax)] if argmax in cand_ids else "?"
            w.writerow([
                q["qid"], cond, q["gold_answer"], pred, correct,
                round(prob, 6), len(ps), n_rem, ntokens,
                feats["competitors"], round(feats["competitor_similarity"], 4),
                round(feats["interference"], 4),
                round(hit[0]["retrieval_score"], 4),
                round(time.time() - tt, 3),
            ])
            fh.flush()
        print(f"[rag] {q['qid']} gold={q['gold_answer']} "
              f"n_rm={n_rm} elapsed={time.time()-t0:.0f}s", flush=True)
    fh.close()
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
