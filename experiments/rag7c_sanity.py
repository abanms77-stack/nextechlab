"""Sanity-check 7C: does ContextLab preferentially retain the GOLD passage?"""
import collections
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from rag_engine import Passage, estimate_tokens, profile  # noqa: E402

s = json.load(open("results/rag7c_queries.json", encoding="utf-8"))
removed = collections.Counter()
kept = collections.Counter()
gold_survives = gold_dup_survives = distractor_survives = nrm = 0
for q in s["queries"]:
    items = q["passages"]
    ps = [Passage(id=f"p{i}", index=i, text=p["text"],
                  tokens=estimate_tokens(p["text"]), relevance=0.0)
          for i, p in enumerate(items)]
    ps, f = profile(q["question"], ps)
    nrm += f["removed"]
    for p, prof in zip(items, ps):
        (kept if prof.keep else removed)[p["role"]] += 1
    gk = {p["role"]: prof.keep for p, prof in zip(items, ps)}
    gold_survives += int(gk.get("gold", False))
    gold_dup_survives += int(gk.get("gold_dup", False))
    distractor_survives += int(gk.get("distractor", False))

n = len(s["queries"])
print("=== role retention by ContextLab (n=40) ===")
print(f"{'role':13s} {'kept':>5s} {'removed':>8s}")
for r in dict.fromkeys(p["role"] for p in s["queries"][0]["passages"]):
    print(f"{r:13s} {kept[r]:5d} {removed[r]:8d}")
print(f"\ngold survives:       {gold_survives}/{n} ({gold_survives/n:.2f})")
print(f"gold_dup survives:   {gold_dup_survives}/{n} ({gold_dup_survives/n:.2f})")
print(f"distractor survives: {distractor_survives}/{n} "
      f"({distractor_survives/n:.2f})")
print(f"removed per query:   {nrm/n:.2f}")
