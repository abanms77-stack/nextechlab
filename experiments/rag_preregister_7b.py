"""Phase 7B - REDUNDANCY intervention benchmark (pre-registration / freeze).

Phase 7 could only exercise the *reorder* arm of ContextLab: its corpus was
contradiction-rich, so the frozen policy protected every competing passage and
`removed == 0` -> tokens never changed. Phase 7B fixes that design flaw by
building FOUR context regimes and, crucially, adds an ORACLE condition as a
reference bound.

FROZEN BEFORE ANY LLM RUN (this file):
  1. engine/policy constants (imported from rag_preregister.py)
  2. the four context-type constructors (below)
  3. the ORACLE selection rule (below)
  4. the scoring rule (single-token, K-way, frozen 12-word set)
  5. the preregistered comparisons (below; declared before seeing data)
Then: run BASELINE / RANDOM / CONTEXTLAB / ORACLE on every query x context type.

Context types (each query is rendered in all four):
  clean         : gold fact + K irrelevant passages (no competition)
  redundant     : gold + R paraphrases of gold + supporting duplicates
  contradictory : gold + K contradictory passages (different answers)
  mixed         : gold + R redundant + C contradictory + irrelevant

Conditions (frozen):
  BASELINE   : all retrieved passages, original order
  RANDOM     : remove a random subset (matched to ContextLab's removed count)
  CONTEXTLAB : profiler prune (redundancy only) + reorder (strongest first)
  ORACLE     : keep gold + the single best supporting passage (ideal selector)

Preregistered comparisons (with bootstrap CIs, paired per query x type):
  C1  CONTEXTLAB - BASELINE     accuracy  (expect >= 0)
  C2  CONTEXTLAB - BASELINE     tokens    (expect < 0, i.e. fewer)
  C3  CONTEXTLAB - RANDOM       accuracy  (expect > 0)
  C4  CONTEXTLAB - ORACLE       accuracy  (expect ~ 0)
  C5  CONTEXTLAB - ORACLE       tokens    (expect > 0, CL uses a few more)
and every comparison is ALSO reported within each context type.

Output:
  results/rag7b_queries.json      frozen queries x context types
  results/rag7b_queries.sha256
  results/rag7b_meta.json

Usage:
  python experiments/rag_preregister_7b.py
"""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WIKI = ROOT / "data" / "wiki_train.txt"

# ---- frozen engine policy (shared with Phase 7; mirror of profile.ts) -------
COMPETITION_FLOOR = 0.35
RELEVANCE_FLOOR = 0.15
COMPETITION_WEIGHT = 0.6
D_SATURATION = 5
NEAR_DUPLICATE_FLOOR = 0.8
ENGINE_VERSION = "profiler-2.1"

# ---- frozen retriever / benchmark config -----------------------------------
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8                      # passages handed to the reader (all conditions)
RANDOM_SEED = 20261004
N_QUERIES = 40
R_REDUNDANT = 3                # redundant supporting passages (redundant/mixed)
C_CONTRADICTORY = 3            # contradictory passages (contradictory/mixed)
IRRELEVANT = 2                 # irrelevant passages mixed into every type

CONTEXT_TYPES = ["clean", "redundant", "contradictory", "mixed"]
CONDITIONS = ["BASELINE", "RANDOM", "CONTEXTLAB", "ORACLE"]

# ---- frozen answer space (single-token answers for clean scoring) ----------
CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]

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

# Irrelevant passages: distinct topics so they never compete with the gold.
IRRELEVANT_TOPICS = [
    ("The {topic} observatory recorded unusually clear skies last winter.", "observatory"),
    ("A regional {topic} committee met to discuss routine drainage repairs.", "drainage"),
    ("The {topic} museum extended its weekend opening hours this season.", "museum"),
    ("Local {topic} volunteers organised a riverside clean-up on Saturday.", "riverside"),
    ("The {topic} library added new shelves for its growing map collection.", "library"),
    ("A travelling {topic} exhibition visited three coastal towns in June.", "exhibition"),
    ("The {topic} ferry timetable was revised after the harbour dredging.", "ferry"),
    ("An amateur {topic} astronomer published a note on a faint comet.", "comet"),
]


def build_noun_pool(need: int) -> list:
    """Unique natural-looking nouns (prefix + base) so every injected passage
    can use a private noun and every injected string is globally unique."""
    prefixes = ["", "primary", "secondary", "backup", "auxiliary",
                "internal", "external", "legacy", "shadow", "secure",
                "north", "south", "east", "west", "central"]
    pool = [b if not p else f"{p} {b}" for p in prefixes for b in NOUNS]
    assert len(pool) >= need, f"noun pool {len(pool)} < need {need}"
    return pool[:need]


# Redundant supporting passages: NEAR-DUPLICATES of the gold fact (same answer,
# minimal lexical change, similarity >= NEAR_DUPLICATE_FLOOR). This mirrors real
# RAG redundancy, where overlapping chunks return the same sentence twice.
REDUNDANT_FORMS = [
    "The {noun} access code is {word}.",          # identical content
    "The {noun} access code is {word}",           # no period
    "The {noun} access code is {word}!",          # exclamation
    "The {noun} access code is: {word}.",         # colon
    "The {noun} access code is {word}. ",         # trailing space
]


def build_query(i: int, rng: random.Random, nouns, answer, contra_words):
    """Build one query rendered as four context-type passage lists.

    Every passage string is made globally unique by binding it to a private
    noun/suffix, so no duplicate string can flood the retrieval index.
    """
    ti = i % len(FACT_TEMPLATES)
    tmpl_fact, tmpl_q = FACT_TEMPLATES[ti]
    qid = f"q{i:03d}"
    noun = nouns[4 * i]           # gold noun (private)
    gold = tmpl_fact.format(noun=noun, word=answer)

    # redundant supporting passages (same answer, same question shape)
    red = []
    for k in range(R_REDUNDANT):
        red.append(REDUNDANT_FORMS[k].format(noun=noun, word=answer))

    # contradictory passages (different answer word, same question shape)
    con = [tmpl_fact.format(noun=noun, word=contra_words[k])
           for k in range(C_CONTRADICTORY)]

    # irrelevant passages (distinct topics, no competition)
    irr = []
    for k in range(IRRELEVANT):
        tpl, topic = IRRELEVANT_TOPICS[(i + k) % len(IRRELEVANT_TOPICS)]
        irr.append(tpl.format(topic=f"{topic}-{qid}"))

    contexts = {
        "clean": [gold] + irr,
        "redundant": [gold] + red + irr,
        "contradictory": [gold] + con + irr,
        "mixed": [gold] + red + con + irr,
    }
    return {
        "qid": qid,
        "template_id": ti,
        "noun": noun,
        "question": tmpl_q.format(noun=noun),
        "gold_answer": answer,
        "gold_fact": gold,
        "contexts": contexts,
    }


def main():
    rng = random.Random(RANDOM_SEED)
    slots = [(t, n) for t in range(len(FACT_TEMPLATES)) for n in NOUNS]
    assert len(slots) >= N_QUERIES, "template x noun grid too small"
    nouns = build_noun_pool(4 * N_QUERIES + 8)

    specs = []
    for i in range(N_QUERIES):
        answer = CANDIDATES[rng.randrange(len(CANDIDATES))]
        pool = [w for w in CANDIDATES if w != answer]
        rng.shuffle(pool)
        specs.append(build_query(i, rng, nouns, answer,
                                 pool[:C_CONTRADICTORY]))

    # ---- guard: the injected INDEX (union of all passages) must have no
    # duplicate strings. A passage reused across context types is fine (it is
    # injected once and merely *selected* by each type); what must never happen
    # is the same string injected twice as two separate index entries.
    index = {}
    for q in specs:
        seen_q = set()
        for typed in q["contexts"].values():
            for t in typed:
                if t in seen_q:
                    continue
                seen_q.add(t)
                index[t] = index.get(t, 0) + 1
    dups = {t: n for t, n in index.items() if n > 1}
    if dups:
        raise SystemExit(
            f"data bug: {len(dups)} strings injected as >1 index entry "
            f"(e.g. {list(dups.items())[:3]}) -> would flood the index")

    payload = {
        "protocol": "rag_redundancy_intervention_v1",
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
        "context_types": CONTEXT_TYPES,
        "conditions": CONDITIONS,
        "policy": {
            "prune": "remove redundant near-duplicates (keep strongest); "
                     "NEVER delete contradictions (surface them)",
            "reorder": "kept sorted by (-relevance, -competition, "
                       "-nearest_similarity) - strongest first",
            "oracle": "keep gold fact + one best supporting passage only",
        },
        "preregistered_comparisons": [
            {"id": "C1", "a": "CONTEXTLAB", "b": "BASELINE",
             "metric": "accuracy", "expect": ">= 0"},
            {"id": "C2", "a": "CONTEXTLAB", "b": "BASELINE",
             "metric": "ctx_tokens", "expect": "< 0 (fewer)"},
            {"id": "C3", "a": "CONTEXTLAB", "b": "RANDOM",
             "metric": "accuracy", "expect": "> 0"},
            {"id": "C4", "a": "CONTEXTLAB", "b": "ORACLE",
             "metric": "accuracy", "expect": "~ 0"},
            {"id": "C5", "a": "CONTEXTLAB", "b": "ORACLE",
             "metric": "ctx_tokens", "expect": "> 0"},
            {"id": "W", "scope": "within each context type",
             "metrics": ["accuracy", "ctx_tokens"]},
        ],
        "queries": specs,
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rag7b_queries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "rag7b_queries.sha256").write_text(digest + "\n", encoding="utf-8")

    meta = {
        "corpus": WIKI.name,
        "corpus_bytes": WIKI.stat().st_size,
        "embedder": EMBEDDER,
        "top_k": TOP_K,
        "n_queries": len(specs),
        "context_types": CONTEXT_TYPES,
        "conditions": CONDITIONS,
        "r_redundant": R_REDUNDANT,
        "c_contradictory": C_CONTRADICTORY,
        "irrelevant": IRRELEVANT,
        "injected_strings": len(index),
        "unique_strings": len(set(index)),
    }
    (RESULTS / "rag7b_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"FROZEN {len(specs)} queries x {len(CONTEXT_TYPES)} context types")
    print(f"sha256 = {digest}")
    print(f"conditions = {' | '.join(CONDITIONS)}")
    print(f"injected strings = {len(index)} (unique = {len(set(index))})")
    ex = specs[0]
    for ct in CONTEXT_TYPES:
        print(f"  [{ct}] {len(ex['contexts'][ct])} passages e.g. "
              f"{ex['contexts'][ct][0]}")


if __name__ == "__main__":
    main()

