"""Phase 7 - REAL RAG intervention benchmark (pre-registration).

This is the "killer" product experiment: build a genuine retrieval pipeline over
the local Wikipedia corpus, then compare THREE frozen conditions end-to-end:

    A  BASELINE   : top-k retrieved passages -> LLM
    B  CONTEXTLAB : top-k -> profiler prune/reorder -> LLM
    C  RANDOM     : top-k -> random prune (same # removed) -> LLM

Frozen-then-run discipline (matches the research protocol):
  1. engine/policy frozen (this file - constants below)
  2. queries frozen (written + sha256 hashed here, BEFORE any LLM runs)
  3. scoring frozen (functions below)
  4. pruning policy frozen (mirrors website/lib/profile.ts)
  5..8. run baseline, run ContextLab, run random, then analyze

No number is invented: every metric is produced by experiments/run_rag_bench.py
and experiments/analyze_rag_bench.py from real model outputs.

Output:
    results/rag_queries.json          - frozen held-out queries (this script)
    results/rag_queries.sha256
    results/rag_corpus_meta.json      - frozen corpus/retriever config

Usage:
    python experiments/rag_preregister.py
"""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WIKI = ROOT / "data" / "wiki_train.txt"

# ---- frozen engine policy (mirror of website/lib/profile.ts) ---------------
COMPETITION_FLOOR = 0.35
RELEVANCE_FLOOR = 0.15
COMPETITION_WEIGHT = 0.6
D_SATURATION = 5
NEAR_DUPLICATE_FLOOR = 0.8
ENGINE_VERSION = "profiler-2.1"

# ---- frozen retriever config ----------------------------------------------
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8
RANDOM_SEED = 20261004
N_QUERIES = 40

# ---- frozen answer space (single-token answers for clean scoring) ----------
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]

# Frozen query templates: each injects a *fact* passage with a candidate-word
# answer into the corpus, then asks for it. Ground truth is the injected answer.
FACT_TEMPLATES = [
    ("The {noun} access code is {word}.",
     "What is the {noun} access code?"),
    ("The {noun} recovery phrase is {word}.",
     "What is the {noun} recovery phrase?"),
    ("The {noun} backup key is {word}.",
     "What is the {noun} backup key?"),
    ("The {noun} override token is {word}.",
     "What is the {noun} override token?"),
]
NOUNS = [
    "vault", "storage", "archive", "gateway", "mailbox",
    "intranet", "admin", "backup", "guest", "media",
    "portal", "ledger", "registry", "cabinet", "locker",
    "terminal", "beacon", "sensor", "relay", "tunnel",
    "channel", "matrix", "cluster", "keyring", "keystone",
    "quorum", "shard", "bucket", "buffer", "socket",
    "socket2", "dossier", "manifest", "catalog", "index",
    "cache", "journal", "archive2", "vault2", "storage2",
]


def build_noun_pool(need: int) -> list:
    """Deterministically build a noun pool of at least `need` unique, natural
    looking tokens (prefix + base), so every injected passage can use a private
    noun and every injected string is globally unique.
    """
    bases = NOUNS
    prefixes = ["", "primary", "secondary", "backup", "auxiliary",
                "internal", "external", "legacy", "shadow", "secure",
                "north", "south", "east", "west", "central"]
    pool = []
    for p in prefixes:
        for b in bases:
            pool.append(b if not p else f"{p} {b}")
    assert len(pool) >= need, f"noun pool {len(pool)} < need {need}"
    return pool[:need]


DECOYS_EACH = 8


def build_queries(rng: random.Random):
    """Deterministically build N_QUERIES held-out query specs.

    Each spec: a UNIQUE injected gold fact passage, a natural-language question,
    and the single-token gold answer. Decoy facts (same template, other nouns)
    are ALSO injected so retrieval genuinely surfaces competing passages - the
    interference the profiler is meant to detect.

    IMPORTANT: every injected passage string must be UNIQUE across the whole
    benchmark (gold and decoys alike). Otherwise duplicate strings flood the
    FAISS index and retrieval returns k copies of one passage - a data bug that
    invalidates the benchmark. We enumerate the 4 templates x 10 nouns grid in a
    fixed order so all N_QUERIES = 40 (template, noun) pairs are distinct.
    """
    slots = [(t, n) for t in range(len(FACT_TEMPLATES)) for n in NOUNS]
    assert len(slots) >= N_QUERIES, "template x noun grid too small"
    gold_nouns = [n for _, n in slots[:N_QUERIES]]
    # Build a decoy pool with ALL gold nouns removed, so a decoy can never
    # reproduce a gold passage string.
    full = build_noun_pool(N_QUERIES * DECOYS_EACH + len(gold_nouns))
    pool = [n for n in full if n not in set(gold_nouns)]
    specs = []
    for i in range(N_QUERIES):
        ti, noun = slots[i]
        tmpl_fact, tmpl_q = FACT_TEMPLATES[ti]
        answer = CANDIDATES[rng.randrange(len(CANDIDATES))]
        qid = f"q{i:03d}"
        # Private decoy nouns: each query draws a disjoint slice of the pool, so
        # no injected string repeats anywhere in the benchmark.
        base = i * DECOYS_EACH
        decoy_nouns = pool[base:base + DECOYS_EACH]
        decoys = []
        pool_words = [w for w in CANDIDATES if w != answer]
        for j, n2 in enumerate(decoy_nouns):
            decoys.append({
                "noun": n2,
                "text": tmpl_fact.format(noun=n2,
                                         word=pool_words[j % len(pool_words)]),
            })
        specs.append({
            "qid": qid,
            "template_id": ti,
            "noun": noun,
            "question": tmpl_q.format(noun=noun),
            "gold_answer": answer,
            "gold_fact": tmpl_fact.format(noun=noun, word=answer),
            "decoys": decoys,
        })
    # Guard: no injected passage string may repeat across the benchmark.
    seen: dict = {}
    for q in specs:
        for t in [q["gold_fact"]] + [d["text"] for d in q["decoys"]]:
            seen[t] = seen.get(t, 0) + 1
    dups = {t: n for t, n in seen.items() if n > 1}
    if dups:
        raise SystemExit(
            f"data bug: {len(dups)} duplicate injected strings "
            f"(e.g. {list(dups.items())[:3]}) -> would flood the index")
    return specs


def main():
    rng = random.Random(RANDOM_SEED)
    specs = build_queries(rng)

    payload = {
        "protocol": "rag_intervention_v1",
        "frozen": True,
        "engine_version": ENGINE_VERSION,
        "competition_floor": COMPETITION_FLOOR,
        "relevance_floor": RELEVANCE_FLOOR,
        "competition_weight": COMPETITION_WEIGHT,
        "d_saturation": D_SATURATION,
        "near_duplicate_floor": NEAR_DUPLICATE_FLOOR,
        "embedder": EMBEDDER,
        "top_k": TOP_K,
        "random_seed": RANDOM_SEED,
        "candidates": CANDIDATES,
        "conditions": ["BASELINE", "CONTEXTLAB", "RANDOM"],
        "contextlab_policy": {
            "prune": "remove redundant near-duplicates (keep strongest); "
                     "NEVER delete contradictions (surface them)",
            "reorder": "kept passages sorted by (-relevance, -competition, "
                       "-nearest_similarity) - strongest evidence first",
        },
        "queries": specs,
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rag_queries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "rag_queries.sha256").write_text(digest + "\n", encoding="utf-8")

    meta = {
        "corpus": str(WIKI.name),
        "corpus_bytes": WIKI.stat().st_size,
        "embedder": EMBEDDER,
        "top_k": TOP_K,
        "n_queries": len(specs),
        "chunk_words": 80,
        "chunk_stride": 80,
    }
    (RESULTS / "rag_corpus_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"FROZEN {len(specs)} held-out queries -> {out.name}")
    print(f"sha256 = {digest}")
    print(f"conditions = BASELINE | CONTEXTLAB | RANDOM")
    print(f"example: {specs[0]['question']}  ->  {specs[0]['gold_answer']}")


if __name__ == "__main__":
    main()
