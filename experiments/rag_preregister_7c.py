"""Phase 7C - HARDER redundancy / CHOICE ATTRIBUTION (pre-registration).

Phase 7B proved "pruning can save tokens" but not "ContextLab chooses WHICH
passage to remove better than random" - because the gold answer was replicated,
so any surviving passage sufficed. Phase 7C removes that crutch: the supporting
passages are NOT identical and only ONE passage (the gold) carries the answer;
the others are graded partial/indirect/related/distractor evidence.

Construction per query (6 passages, one of each role):
    gold        directly answers (question shape + answer word)
    partial     same topic/answer, weaker phrasing
    indirect    mentions the topic, no answer
    related     related but incomplete (different noun)
    distractor  plausible competing answer word (same topic)
    irrelevant  off-topic

Because only `gold` yields the answer, RANDOM removal will sometimes delete the
gold -> accuracy falls; ContextLab should PREFERENTIALLY RETAIN the gold.

Conditions (frozen):
    BASELINE   : all 6 passages
    RANDOM     : remove k random passages (k = ContextLab's removed count)
    CONTEXTLAB : profiler prune + reorder (strongest evidence first)
    ORACLE     : keep gold fact + one best supporting only

Preregistered comparisons (paired, bootstrap CI):
    A1  CONTEXTLAB - RANDOM  accuracy   (expect > 0: better choice)
    A2  CONTEXTLAB - RANDOM  gold_survival rate  (expect > 0)
    A3  CONTEXTLAB - BASELINE accuracy  (expect >= 0)
    A4  CONTEXTLAB - BASELINE tokens    (expect < 0)
    A5  RANDOM - BASELINE accuracy      (expect <= 0: random removal can hurt)
    plus gold-survival and removal-choice distributions per condition.

Output:
    results/rag7c_queries.json + .sha256, results/rag7c_meta.json

Usage:
    python experiments/rag_preregister_7c.py
"""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WIKI = ROOT / "data" / "wiki_train.txt"

COMPETITION_FLOOR = 0.35
RELEVANCE_FLOOR = 0.15
COMPETITION_WEIGHT = 0.6
D_SATURATION = 5
NEAR_DUPLICATE_FLOOR = 0.8
ENGINE_VERSION = "profiler-2.1"

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 20261005
N_QUERIES = 40

CONDITIONS = ["BASELINE", "RANDOM", "CONTEXTLAB", "ORACLE"]
ROLES = ["gold", "partial", "indirect", "related", "distractor", "irrelevant"]

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
    "dossier", "manifest", "catalog", "index", "cache",
    "journal", "archive2", "vault2", "storage2", "gateway2",
]


def role_passages(i, tmpl_fact, noun, answer, other_answer, other_noun):
    """Return an ordered list of (role, text) with graded evidence strength.

    Only `gold` contains the answer in the question shape. The remaining
    passages come in REDUNDANT PAIRS (near-duplicate wording) around weak roles,
    so the profiler's redundancy prune fires - and RANDOM removal of the same
    count risks deleting the gold. Text is globally unique via `qid`.
    """
    qid = f"q{i:03d}"

    def weak_partial(tag):
        return tmpl_fact.replace(f"is {answer}.", f"is {answer}").replace(
            "The ", f"The ").rstrip(".") + f" ({tag} record)."

    return [
        ("gold", tmpl_fact.format(noun=noun, word=answer)),
        ("gold_dup", tmpl_fact.format(noun=noun, word=answer).rstrip(".")
         + f" ({qid})"),
        ("partial", f"A {noun} note suggests the {noun} code resembles "
                    f"{answer[:2]}... incomplete {qid}"),
        ("partial_dup", f"A {noun} note suggests the {noun} code resembles "
                        f"{answer[:2]}... incomplete {qid} (copy)"),
        ("indirect", f"Records for the {noun} system note access is restricted "
                     f"but omit the value ({qid})."),
        ("indirect_dup", f"Records for the {noun} system note access is "
                         f"restricted but omit the value ({qid} copy)."),
        ("related", f"The {other_noun} process is documented separately and is "
                    f"not the {noun} value ({qid})."),
        ("distractor", tmpl_fact.format(noun=noun, word=other_answer)),
        ("irrelevant", f"The {noun} annexe grounds hosted a regional gardening "
                       f"fair last spring ({qid})."),
    ]


def build_query(i, rng, nouns):
    ti = i % len(FACT_TEMPLATES)
    tmpl_fact, tmpl_q = FACT_TEMPLATES[ti]
    qid = f"q{i:03d}"
    noun = nouns[2 * i]
    other_noun = nouns[2 * i + 1]
    answer = CANDIDATES[rng.randrange(len(CANDIDATES))]
    pool = [w for w in CANDIDATES if w != answer]
    rng.shuffle(pool)
    other_answer = pool[0]
    pairs = role_passages(i, tmpl_fact, noun, answer, other_answer, other_noun)
    return {
        "qid": qid,
        "template_id": ti,
        "noun": noun,
        "question": tmpl_q.format(noun=noun),
        "gold_answer": answer,
        "gold_fact": pairs[0][1],
        "passages": [{"role": r, "text": t} for r, t in pairs],
    }


def main():
    rng = random.Random(RANDOM_SEED)
    names = ["", "primary", "secondary", "tertiary", "auxiliary", "internal",
             "external", "legacy", "shadow", "secure"]
    pool = [f"{b}" if not p else f"{p} {b}" for p in names for b in NOUNS]
    nouns = pool[:2 * N_QUERIES + 4]

    specs = [build_query(i, rng, nouns) for i in range(N_QUERIES)]

    seen = {}
    for q in specs:
        for p in q["passages"]:
            seen[p["text"]] = seen.get(p["text"], 0) + 1
    dups = {t: n for t, n in seen.items() if n > 1}
    if dups:
        raise SystemExit(f"data bug: {len(dups)} duplicate strings "
                         f"(e.g. {list(dups.items())[:3]})")

    payload = {
        "protocol": "rag_choice_attribution_v1",
        "frozen": True,
        "engine_version": ENGINE_VERSION,
        "competition_floor": COMPETITION_FLOOR,
        "relevance_floor": RELEVANCE_FLOOR,
        "competition_weight": COMPETITION_WEIGHT,
        "d_saturation": D_SATURATION,
        "near_duplicate_floor": NEAR_DUPLICATE_FLOOR,
        "embedder": EMBEDDER,
        "random_seed": RANDOM_SEED,
        "candidates": CANDIDATES,
        "roles": [r for r, _ in role_passages(0, FACT_TEMPLATES[0][0],
                                              "x", "apple", "river", "y")],
        "conditions": CONDITIONS,
        "preregistered_comparisons": [
            {"id": "A1", "a": "CONTEXTLAB", "b": "RANDOM", "metric": "accuracy",
             "expect": "> 0 (better choice)"},
            {"id": "A2", "a": "CONTEXTLAB", "b": "RANDOM",
             "metric": "gold_survival", "expect": "> 0"},
            {"id": "A3", "a": "CONTEXTLAB", "b": "BASELINE", "metric": "accuracy",
             "expect": ">= 0"},
            {"id": "A4", "a": "CONTEXTLAB", "b": "BASELINE", "metric": "ctx_tokens",
             "expect": "< 0"},
            {"id": "A5", "a": "RANDOM", "b": "BASELINE", "metric": "accuracy",
             "expect": "<= 0"},
        ],
        "queries": specs,
    }

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rag7c_queries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "rag7c_queries.sha256").write_text(digest + "\n", encoding="utf-8")

    meta = {"corpus": WIKI.name, "embedder": EMBEDDER, "n_queries": len(specs),
            "conditions": CONDITIONS, "injected_strings": len(seen),
            "unique_strings": len(set(seen))}
    (RESULTS / "rag7c_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")

    print(f"FROZEN {len(specs)} queries x {len(specs[0]['passages'])} passages "
          f"x {len(CONDITIONS)} conditions")
    print(f"sha256 = {digest}")
    print(f"injected strings = {len(seen)} (unique = {len(set(seen))})")
    ex = specs[0]
    print(f"example q: {ex['question']} -> {ex['gold_answer']}")
    for p in ex["passages"]:
        print(f"  [{p['role']:12s}] {p['text']}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()

