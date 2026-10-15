"""Filler providers for ContextLab.

A filler supplies the neutral tokens that surround the needle. The original
version tiled a tiny ~388-token bank of sentences (a documented confound:
long contexts repeated the same text). This module adds a real-prose filler
(WikiText) so long contexts are made of genuine, non-repeating language.

Both providers guarantee that no *candidate* token (i.e. no possible answer
word) appears in the filler, so the true answer occurs exactly once per prompt.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class SentenceBankFiller:
    """Legacy filler: tiles a small bank of neutral sentences."""

    def __init__(self, tokenizer, sentences: Sequence[str], candidate_ids):
        cand = set(int(c) for c in candidate_ids)
        pool: List[int] = []
        for s in sentences:
            ids = tokenizer.encode(" " + s, add_special_tokens=False)
            if cand.isdisjoint(ids):
                pool.extend(ids)
        if len(pool) < 64:
            raise ValueError("not enough clean filler tokens available")
        self._pool = pool
        self.raw = pool
        self.kind = "sentence-bank"
        self.size = len(pool)

    def sample(self, n: int, rng) -> List[int]:
        out: List[int] = []
        pool = self._pool
        while len(out) < n:
            start = rng.randrange(0, len(pool))
            out.extend(pool[start:])
            out.extend(pool[:start])
        return out[:n]


class ProseFiller:
    """Real prose filler (e.g. WikiText) read from a local text file."""

    def __init__(self, tokenizer, path: str, candidate_ids,
                 max_chars: int = 1_500_000):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()[:max_chars]
        ids = tokenizer.encode(text, add_special_tokens=False)
        arr = np.asarray(ids, dtype=np.int64)
        cand = np.asarray(sorted(set(int(c) for c in candidate_ids)),
                          dtype=np.int64)
        keep = ~np.isin(arr, cand)
        self.tokens = arr[keep]
        self.raw = self.tokens
        self.kind = "prose"
        self.size = int(self.tokens.shape[0])

    def sample(self, n: int, rng) -> List[int]:
        m = self.size
        if n >= m:
            raise ValueError("prose pool smaller than requested context")
        start = rng.randrange(0, m - n)
        return self.tokens[start:start + n].tolist()
