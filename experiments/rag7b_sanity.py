"""Sanity-check 7B: does the engine prune redundancy and protect contradictions?"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from rag_engine import Passage, estimate_tokens, profile  # noqa: E402

s = json.load(open("results/rag7b_queries.json", encoding="utf-8"))
print(f"{'qid':5s} {'type':14s} {'n':>2s} {'removed':>7s} {'comp':>4s} {'sim':>5s}")
for q in s["queries"][:5]:
    for ct in s["context_types"]:
        ps = [Passage(id=f"p{i}", index=i, text=t, tokens=estimate_tokens(t),
                      relevance=0.0) for i, t in enumerate(q["contexts"][ct])]
        ps, f = profile(q["question"], ps)
        print(f"{q['qid']:5s} {ct:14s} {f['passages']:2d} {f['removed']:7d} "
              f"{f['competitors']:4d} {f['competitor_similarity']:5.3f}")

# aggregate
import collections
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0])
for q in s["queries"]:
    for ct in s["context_types"]:
        ps = [Passage(id=f"p{i}", index=i, text=t, tokens=estimate_tokens(t),
                      relevance=0.0) for i, t in enumerate(q["contexts"][ct])]
        ps, f = profile(q["question"], ps)
        a = agg[ct]
        a[0] += f["passages"]; a[1] += f["removed"]
        a[2] += f["competitors"]; a[3] += f["competitor_similarity"]
n = len(s["queries"])
print("\n=== mean over 40 queries ===")
for ct in s["context_types"]:
    a = agg[ct]
    print(f"{ct:14s} passages={a[0]/n:.1f} removed={a[1]/n:.2f} "
          f"competitors={a[2]/n:.1f} sim={a[3]/n:.3f}")
