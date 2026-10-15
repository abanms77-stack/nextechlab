"""Diagnostic 2: what does the FULL pipeline actually feed the model?

Checks, per query, whether the gold fact is in the top-k retrieved passages,
and what the context looks like. Explains the gap between the clean diagnostic
(0.65 with gold+decoys) and the benchmark (~0.10).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from rag_retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "results" / "rag_queries.json").read_text())

extra = []
for q in spec["queries"]:
    extra.append(q["gold_fact"])
    extra += [d["text"] for d in q["decoys"]]
retr = Retriever(spec["embedder"]).build(ROOT / "data" / "wiki_train.txt", extra)

k = spec["top_k"]
gold_in = 0
pos = []
for q in spec["queries"]:
    hits = retr.search(q["question"], k)
    texts = [h["text"] for h in hits]
    if q["gold_fact"] in texts:
        gold_in += 1
        pos.append(texts.index(q["gold_fact"]))
print(f"gold in top-{k}: {gold_in}/{len(spec['queries'])}")
if pos:
    print(f"gold rank: mean={sum(pos)/len(pos):.2f}  (0 = first)")
    print(f"  ranks: {sorted(pos)}")

# show one example context
q = spec["queries"][0]
hits = retr.search(q["question"], k)
print(f"\nexample q: {q['question']}  gold={q['gold_answer']}")
for i, h in enumerate(hits):
    tag = "GOLD" if h["text"] == q["gold_fact"] else ("decoy" if "code is" in h["text"] else "wiki")
    print(f"  [{i}] {h['retrieval_score']:.3f} {tag:<5} {h['text'][:60]}")
