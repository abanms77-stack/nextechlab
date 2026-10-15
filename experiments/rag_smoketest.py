"""Smoke test for the RAG benchmark (no LLM): verifies retrieval + profiler."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from rag_engine import Passage, estimate_tokens, profile  # noqa: E402
from rag_retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "results" / "rag_queries.json").read_text())
q = spec["queries"][0]

extra = [q["gold_fact"]] + [d["text"] for d in q["decoys"]]
retr = Retriever(spec["embedder"], max_chunks=4000).build(
    ROOT / "data" / "wiki_train.txt", extra)
print("chunks:", len(retr.texts))

hit = retr.search(q["question"], spec["top_k"])
print(f"\nquestion: {q['question']}  gold={q['gold_answer']}")
print("top-k retrieved:")
for h in hit:
    gold = "  <-- GOLD" if h["text"] == q["gold_fact"] else ""
    print(f"  {h['retrieval_score']:.3f}  {h['text'][:70]}{gold}")

passages = [Passage(id=f"p{i}", index=h["index"], text=h["text"],
                    tokens=estimate_tokens(h["text"]), relevance=0.0)
            for i, h in enumerate(hit)]
passages, feats = profile(q["question"], passages)
print("\nprofiler features:", feats)
print("kept:", sum(p.keep for p in passages), "/", len(passages))
for p in passages:
    if not p.keep:
        print("  PRUNED:", p.text[:70])
