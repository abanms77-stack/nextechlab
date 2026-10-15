"""Phase 7E - DECOUPLED conflict resolution (pre-registration).

Phase 7D found DROP_WEAKER reached 100% accuracy, but that result is confounded:
the conflicting passages all shared one template (identical token counts) and the
gold passage was always at index 0, so "keep the highest-token conflict" only
ever resolved ties by picking the FIRST passage - i.e. it won by POSITION, not by
any real signal.

Phase 7E removes BOTH confounds:
  * SPECIFICITY is decoupled from CORRECTNESS: a passage's length (specificity)
    is drawn independently of whether it carries the true answer. We VERIFY the
    anti-leakage property before running: P(correct | specificity) ~= P(correct).
  * POSITION is randomized: the correct passage sits at a random index.

Construction per query: 4 conflicting passages, all the same question shape,
each with an independent (correct?, specificity) label:
    exactly one passage is correct (the answer word),
    the other three carry wrong answer words,
    specificity in {low, medium, high} assigned independently.
Plus irrelevant passages. Anti-leakage is asserted at freeze time.

Frozen policies (applied to the conflicting group):
  PROTECT             keep every conflicting passage (current product policy)
  DROP_MORE_SPECIFIC  keep the highest-token conflicting passage
  DROP_LESS_SPECIFIC  keep the lowest-token conflicting passage
  DROP_WEAKER         keep the conflicting passage most lexically relevant to
                      the query (content-based)
  RANDOM              keep a random conflicting passage
  ORACLE              keep the correct passage

Preregistered comparisons (paired, bootstrap CI):
  E1 DROP_WEAKER        - PROTECT  accuracy
  E2 DROP_MORE_SPECIFIC - PROTECT  accuracy
  E3 ORACLE             - PROTECT  accuracy   (headroom)
  E4 RANDOM             - PROTECT  accuracy   (does any resolution help?)
  E5 DROP_WEAKER        - RANDOM   accuracy   (content signal vs chance)
  E6 DROP_WEAKER        - PROTECT  ctx_tokens
plus survived-evidence correctness per policy.

Output:
  results/rag7e_queries.json + .sha256, results/rag7e_meta.json

Usage:
  python experiments/rag_preregister_7e.py
"""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ENGINE_VERSION = "profiler-2.1"
RANDOM_SEED = 20261006
N_QUERIES = 40
N_CONFLICTS = 4                     # conflicting passages per query
CONDITIONS = ["PROTECT", "DROP_MORE_SPECIFIC", "DROP_LESS_SPECIFIC",
              "DROP_WEAKER", "RANDOM", "ORACLE"]
IRRELEVANT = 2

CANDIDATES = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]

FACT_TEMPLATES = [
    ("The {noun} access code is {word}.", "What is the {noun} access code?"),
    ("The {noun} recovery phrase is {word}.",
     "What is the {noun} recovery phrase?"),
    ("The {noun} backup key is {word}.", "What is the {noun} backup key?"),
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

# Specificity = amount of independent detail padding added to a passage.
# 3 levels; the padding is content-neutral (it never reveals the answer).
SPEC_PAD = {
    "low": "",
    "medium": " This entry was revised during the routine quarterly audit.",
    "high": (" This entry was revised during the routine quarterly audit and "
             "cross-checked against the earlier signed manuscript record."),
}
SPEC_ORDER = ["low", "medium", "high"]


def build_query(i, rng, nouns, cwp):
    """One query with N_CONFLICTS conflicting passages.

    Each conflicting passage has an independent (correct?, specificity) label;
    exactly one is correct. The correct passage's position is randomized.
    """
    ti = i % len(FACT_TEMPLATES)
    tmpl_fact, tmpl_q = FACT_TEMPLATES[ti]
    qid = f"q{i:03d}"
    noun = nouns[i]
    answer = cwp[i]["answer"]
    wrongs = cwp[i]["wrongs"]

    specs = [rng.choice(SPEC_ORDER) for _ in range(N_CONFLICTS)]
    correct_slot = rng.randrange(N_CONFLICTS)

    conflicts = []
    wi = 0
    for s in range(N_CONFLICTS):
        correct = (s == correct_slot)
        word = answer if correct else wrongs[wi]
        if not correct:
            wi += 1
        text = tmpl_fact.format(noun=noun, word=word) + SPEC_PAD[specs[s]]
        conflicts.append({"text": text, "correct": correct,
                          "specificity": specs[s], "tokens": len(text.split())})

    irr = [f"The {noun} annexe hosted a regional fair (record {k})."
           for k in range(IRRELEVANT)]
    return {
        "qid": qid,
        "template_id": ti,
        "noun": noun,
        "question": tmpl_q.format(noun=noun),
        "gold_answer": answer,
        "conflicts": conflicts,
        "irrelevant": irr,
    }


def main():
    rng = random.Random(RANDOM_SEED)
    names = ["", "primary", "secondary", "tertiary", "auxiliary", "internal",
             "external", "legacy", "shadow", "secure"]
    pool = [f"{b}" if not p else f"{p} {b}" for p in names for b in NOUNS]
    nouns = pool[:N_QUERIES + 2]

    cwp = []
    for _ in range(N_QUERIES):
        w = list(CANDIDATES)
        rng.shuffle(w)
        cwp.append({"answer": w[0], "wrongs": w[1:1 + N_CONFLICTS]})

    specs = [build_query(i, rng, nouns, cwp) for i in range(N_QUERIES)]

    # ---- anti-leakage check: P(correct | specificity) ~= P(correct) ---------
    def p_correct_given(spec):
        rows = [c for q in specs for c in q["conflicts"]
                if c["specificity"] == spec]
        return sum(c["correct"] for c in rows) / len(rows)

    all_rows = [c for q in specs for c in q["conflicts"]]
    p_correct = sum(c["correct"] for c in all_rows) / len(all_rows)
    by_spec = {s: p_correct_given(s) for s in SPEC_ORDER}
    max_dev = max(abs(v - p_correct) for v in by_spec.values())
    if max_dev > 0.12:
        raise SystemExit(
            f"anti-leakage FAILED: P(correct|spec)={by_spec} vs "
            f"P(correct)={p_correct:.3f} (max dev {max_dev:.3f})")

    # ---- uniqueness + position-balance checks ------------------------------
    seen = {}
    for q in specs:
        for c in q["conflicts"]:
            seen[c["text"]] = seen.get(c["text"], 0) + 1
        for t in q["irrelevant"]:
            seen[t] = seen.get(t, 0) + 1
    dups = {t: n for t, n in seen.items() if n > 1}
    if dups:
        raise SystemExit(f"data bug: {len(dups)} duplicate strings")
    pos_counts = [sum(1 for q in specs if q["conflicts"][k]["correct"])
                  for k in range(N_CONFLICTS)]

    payload = {
        "protocol": "rag_decoupled_conflict_v1",
        "frozen": True,
        "engine_version": ENGINE_VERSION,
        "random_seed": RANDOM_SEED,
        "candidates": CANDIDATES,
        "conditions": CONDITIONS,
        "n_conflicts": N_CONFLICTS,
        "specificity_levels": SPEC_ORDER,
        "specificity_pad": SPEC_PAD,
        "anti_leakage": {"p_correct": p_correct, "by_specificity": by_spec,
                         "max_deviation": max_dev},
        "correct_slot_counts": pos_counts,
        "policy_defs": {
            "PROTECT": "keep every conflicting passage",
            "DROP_MORE_SPECIFIC": "keep the highest-token conflicting passage",
            "DROP_LESS_SPECIFIC": "keep the lowest-token conflicting passage",
            "DROP_WEAKER": "keep the conflicting passage most lexically "
                           "relevant to the query (content-based)",
            "RANDOM": "keep a random conflicting passage",
            "ORACLE": "keep the correct passage",
        },
        "preregistered_comparisons": [
            {"id": "E1", "a": "DROP_WEAKER", "b": "PROTECT", "metric": "accuracy"},
            {"id": "E2", "a": "DROP_MORE_SPECIFIC", "b": "PROTECT",
             "metric": "accuracy"},
            {"id": "E3", "a": "ORACLE", "b": "PROTECT", "metric": "accuracy"},
            {"id": "E4", "a": "RANDOM", "b": "PROTECT", "metric": "accuracy"},
            {"id": "E5", "a": "DROP_WEAKER", "b": "RANDOM", "metric": "accuracy"},
            {"id": "E6", "a": "DROP_WEAKER", "b": "PROTECT",
             "metric": "ctx_tokens"},
        ],
        "queries": specs,
    }

    out = RESULTS / "rag7e_queries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (RESULTS / "rag7e_queries.sha256").write_text(digest + "\n", encoding="utf-8")
    meta = {"n_queries": len(specs), "conditions": CONDITIONS,
            "n_conflicts": N_CONFLICTS, "anti_leakage": payload["anti_leakage"],
            "correct_slot_counts": pos_counts,
            "injected_strings": len(seen), "unique_strings": len(set(seen))}
    (RESULTS / "rag7e_meta.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")

    print(f"FROZEN {len(specs)} queries x {N_CONFLICTS} conflicts "
          f"x {len(CONDITIONS)} conditions")
    print(f"sha256 = {digest}")
    print(f"anti-leakage: P(correct)={p_correct:.3f} by_spec={by_spec} "
          f"max_dev={max_dev:.3f} (PASS if <= 0.12)")
    print(f"correct-slot counts (position balance): {pos_counts}")
    print(f"injected strings = {len(seen)} (unique = {len(set(seen))})")
    ex = specs[0]
    print(f"example q: {ex['question']} -> {ex['gold_answer']}")
    for c in ex["conflicts"]:
        print(f"  correct={int(c['correct'])} spec={c['specificity']:6s} "
              f"tok={c['tokens']:2d} | {c['text'][:70]}")


if __name__ == "__main__":
    main()

