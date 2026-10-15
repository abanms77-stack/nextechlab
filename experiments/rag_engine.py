"""ContextLab profiler engine - Python mirror of website/lib/profile.ts.

This reproduces the PRODUCT's measured-similarity layer and pruning policy in
Python so the RAG benchmark intervenes with the SAME logic the product uses.
Constants are imported from experiments/rag_preregister.py (frozen).

Similarity is MEASURED (stemmed-token Jaccard blended with char-trigram
Jaccard). Interference is competition BETWEEN passages, contradiction-aware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from rag_preregister import (
    COMPETITION_FLOOR,
    COMPETITION_WEIGHT,
    D_SATURATION,
    NEAR_DUPLICATE_FLOOR,
    RELEVANCE_FLOOR,
)

_STOP = set(
    "a an the and or of to in is are was were be been being for on at by with "
    "from as that this these those it its what which who whom how when where "
    "why do does did can could should would may might will shall not no".split()
)


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _raw_tokens(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", _normalize(s))
            if len(w) > 1 and w not in _STOP]


def _stem(w: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def stem_set(s: str):
    return {_stem(w) for w in _raw_tokens(s)}


def _trigrams(s: str):
    x = " " + _normalize(s) + " "
    return {x[i:i + 3] for i in range(max(0, len(x) - 2))}


def _jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def text_similarity(a: str, b: str) -> float:
    tok = _jaccard(stem_set(a), stem_set(b))
    tri = _jaccard(_trigrams(a), _trigrams(b))
    return min(1.0, COMPETITION_WEIGHT * tok + (1 - COMPETITION_WEIGHT) * tri)


def query_coverage(query: str, passage: str) -> float:
    q = stem_set(query)
    if not q:
        return 0.0
    p = stem_set(passage)
    return len(q & p) / len(q)


@dataclass
class Passage:
    id: str
    index: int
    text: str
    tokens: int
    relevance: float
    nearest_similarity: float = 0.0
    competition: float = 0.0
    redundant: bool = False
    contradict: bool = False
    keep: bool = True


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.33))


def _extract_answer(text: str):
    """Crude answer-key extractor (mirror of the product's relations layer)."""
    m = re.search(r"is\s+([a-z0-9]+)", text.lower())
    return m.group(1) if m else None


def accept_redundant(a: Passage, b: Passage) -> bool:
    """A redundant pair is one whose texts are near-duplicate in meaning."""
    return text_similarity(a.text, b.text) >= NEAR_DUPLICATE_FLOOR


def _prune(passages: List[Passage]) -> int:
    """Engineered policy: keep the strongest evidence, never delete a conflict.

    * CONTRADICTION group -> keep EVERY member (surface the conflict).
    * REDUNDANT group     -> keep the highest-relevance member, prune the rest.
    * Unrelated           -> keep.
    Returns number removed.
    """
    parent = {p.id: p.id for p in passages}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            a, b = passages[i], passages[j]
            if a.contradict or b.contradict:
                continue
            if (a.redundant and b.redundant
                    and text_similarity(a.text, b.text) >= NEAR_DUPLICATE_FLOOR):
                union(a.id, b.id)

    groups = {}
    for p in passages:
        groups.setdefault(find(p.id), []).append(p)

    removed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        if any(m.contradict for m in members):
            continue  # protected
        keep = max(members, key=lambda m: (m.relevance, -m.tokens))
        for m in members:
            if m is not keep:
                m.keep = False
                removed += 1
    return removed


def profile(query: str, passages: List[Passage]):
    """Compute per-passage competition and the engineered prune decision.

    Returns (passages_with_scores, features_dict).
    """
    for p in passages:
        p.relevance = query_coverage(query, p.text)
        p.contradict = False
        p.redundant = False
        p.competition = 0.0
        p.nearest_similarity = 0.0

    ans = {p.id: _extract_answer(p.text) for p in passages}

    competing = 0
    sims = []
    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            a, b = passages[i], passages[j]
            s = text_similarity(a.text, b.text)
            a.nearest_similarity = max(a.nearest_similarity, s)
            b.nearest_similarity = max(b.nearest_similarity, s)
            related = (a.relevance >= RELEVANCE_FLOOR
                       and b.relevance >= RELEVANCE_FLOOR)
            if not related or s < COMPETITION_FLOOR:
                continue
            aa, ab = ans[a.id], ans[b.id]
            contradict = (aa is not None and ab is not None and aa != ab
                          and s >= COMPETITION_FLOOR)
            if contradict or accept_redundant(a, b):
                a.competition = max(a.competition, s)
                b.competition = max(b.competition, s)
                if contradict:
                    a.contradict = True
                    b.contradict = True
                else:
                    a.redundant = True
                    b.redundant = True
                competing += 1
                sims.append(s)

    removed = _prune(passages)
    mean_s = (sum(sims) / len(sims)) if sims else 0.0
    features = {
        "passages": len(passages),
        "competitors": competing,
        "competitor_similarity": mean_s,
        "interference": min(1.0, competing / D_SATURATION) * mean_s,
        "removed": removed,
    }
    return passages, features

