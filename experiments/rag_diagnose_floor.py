"""Diagnostic: is the NEGATIVE RAG result engine-bound or reader-bound?

For each frozen query, ask the SAME model with:
  (a) gold fact ONLY            (clean, no competition)
  (b) gold + 7 decoys           (competing)
If (a) is also near chance, the reader is capacity-bound (floor), and no
pruning policy could help. If (a) is high and (b) collapses, the reader works
and the engine/context is the bottleneck.
"""

import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")

from contextlab.models.local_hf import LocalHFModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "results", "rag_queries.json"),
                     encoding="utf-8"))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"

PROMPT = ("Context:\n{ctx}\n\nQuestion: {q}\n"
          "Answer with a single word from the allowed set.\nAnswer:")


def main():
    model = LocalHFModel(MODEL, threads=8, max_context=4096)
    cands = SPEC["candidates"]
    cand_ids = [model.tokenizer.encode(" " + w, add_special_tokens=False)[0]
                for w in cands]
    acc_clean = acc_comp = 0
    n = len(SPEC["queries"])
    for q in SPEC["queries"]:
        for label, ctx in (("clean", q["gold_fact"]),
                           ("comp", "\n".join([q["gold_fact"]] +
                                              [d["text"] for d in q["decoys"]]))):
            text = PROMPT.format(ctx=ctx, q=q["question"])
            ids = model.tokenizer.encode(text, add_special_tokens=False)
            correct, prob, argmax, _ = model.predict(
                ids, cand_ids, cand_ids[cands.index(q["gold_answer"])])
            if label == "clean":
                acc_clean += correct
            else:
                acc_comp += correct
    print(f"\nmodel = {MODEL}")
    print(f"clean (gold only)      acc = {acc_clean/n:.3f}")
    print(f"competing (gold+decoys) acc = {acc_comp/n:.3f}")
    print(f"chance (1/{len(cands)})         = {1/len(cands):.3f}")


if __name__ == "__main__":
    main()
