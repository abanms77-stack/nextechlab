"""Phase 7D - CONFLICT-RESOLUTION policies (pre-registration).

Phase 7B showed the current conflict policy ("protect the conflict") caused an
accuracy regression on contradictory context (0.350 -> 0.250) while ORACLE
(drop conflicts) reached 0.525 - i.e. detection is not resolution. Phase 7D
tests explicit resolution policies on the SAME contradictory/mixed contexts.

Frozen policies (applied to the retrieved passage set):
  PROTECT      keep every conflicting passage (current product policy)
  DROP_WEAKER  among a conflicting group, keep only the highest-quality member
               (quality = token count, i.e. most specific evidence)
  SURFACE      keep every conflicting passage but order strongest-first
  ORACLE       keep the gold passage only (upper bound; not deployable)

Context types reused from Phase 7B: `contradictory` and `mixed` (the two that
contain conflicts). All four policies are applied to every query x context.

Preregistered comparisons (paired, bootstrap CI; pooled and per context type):
  P1  DROP_WEAKER - PROTECT   accuracy
  P2  ORACLE      - PROTECT   accuracy   (upper bound / headroom)
  P3  SURFACE     - PROTECT   accuracy
  P4  DROP_WEAKER - PROTECT   ctx_tokens
and the gold-accuracy at each policy (does the policy discard conflicting noise
without discarding the gold?).

Frozen before any LLM run.

Output:
  results/rag7d_queries.json + .sha256, results/rag7d_meta.json

Usage:
  python experiments/rag_preregister_7d.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ENGINE_VERSION = "profiler-2.1"
RANDOM_SEED = 20261005
CONTEXT_TYPES = ["contradictory", "mixed"]
POLICIES = ["PROTECT", "DROP_WEAKER", "SURFACE", "ORACLE"]

# Reuse the frozen Phase 7B queries/contexts (same sha256 discipline).
SRC = RESULTS / "rag7b_queries.json"


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    if not src.get("frozen"):
        raise SystemExit("Phase 7B queries are not frozen; run 7B prereg first")
    queries = []
    for q in src["queries"]:
        queries.append({
            "qid": q["qid"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "gold_fact": q["gold_fact"],
            "contexts": {ct: q["contexts"][ct] for ct in CONTEXT_TYPES},
        })
    if len(queries) != 40:
        raise SystemExit(f"expected 40 queries, got {len(queries)}")

    payload = {
        "protocol": "rag_conflict_resolution_v1",
        "frozen": True,
        "engine_version": ENGINE_VERSION,
        "source": "results/rag7b_queries.json",
        "source_sha256": (RESULTS / "rag7b_queries.sha256").read_text().strip(),
        "random_seed": RANDOM_SEED,
        "candidates": src["candidates"],
        "context_types": CONTEXT_TYPES,
        "policies": POLICIES,
        "policy_defs": {
            "PROTECT": "keep every conflicting passage (current product policy)",
            "DROP_WEAKER": "in a conflicting group keep only the highest-token "
                           "member (most specific evidence)",
            "SURFACE": "keep all conflicts, order strongest-first",
            "ORACLE": "keep the gold passage only (upper bound, not deployable)",
        },
        "preregistered_comparisons": [
            {"id": "P1", "a": "DROP_WEAKER", "b": "PROTECT", "metric": "accuracy"},
            {"id": "P2", "a": "ORACLE", "b": "PROTECT", "metric": "accuracy"},
            {"id": "P3", "a": "SURFACE", "b": "PROTECT", "metric": "accuracy"},
            {"id": "P4", "a": "DROP_WEAKER", "b": "PROTECT",
             "metric": "ctx_tokens"},
        ],
        "queries": queries,
    }

    out = RESULTS / "rag7d_queries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "rag7d_queries.sha256").write_text(digest + "\n", encoding="utf-8")
    meta = {"n_queries": len(queries), "context_types": CONTEXT_TYPES,
            "policies": POLICIES, "source_sha256": payload["source_sha256"]}
    (RESULTS / "rag7d_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
    print(f"FROZEN {len(queries)} queries x {len(CONTEXT_TYPES)} context types "
          f"x {len(POLICIES)} policies")
    print(f"sha256 = {digest}")
    print(f"policies = {' | '.join(POLICIES)}")


if __name__ == "__main__":
    main()
