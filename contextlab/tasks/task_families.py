"""Additional task families for cross-task validation.

The original benchmark is single-fact *retrieval* (the vault password). To test
whether the interference structure P = f(D, S, L) is task-specific, we add a
family with a genuinely different computation:

    CountingTask - aggregate and count: "how many red objects were recorded?"
                   The context also contains items of a *similar* colour
                   (green / orange / crimson), which must be EXCLUDED. This is a
                   counting/aggregation task, not a key lookup.

The task exposes the same `generate(...)` interface as NeedleGenerator, and S is
the *measured* embedding similarity between the target colour and the competitor
colour (not a handcrafted label).
"""

from __future__ import annotations

import random
from typing import Dict, List

from .filler import SentenceBankFiller
from .generator import TaskSample

# ------------------------------------------------------------------ counting
# Answer is a WORD-number (a single token in the Qwen BPE), e.g. "three".
_COUNT_QUERY = "\nQuestion: How many red objects were recorded?\nAnswer:"
_COUNT_CANDIDATES = ["one", "two", "three", "four", "five", "six",
                     "seven", "eight", "nine"]
_COUNT_ITEM = " Object {i}: a {colour} item was recorded."

# Canonical RGB values for colour names (standard web colours). Colour
# confusion is objectively about distance in colour space, so for the counting
# task we use an RGB-distance similarity. This is a *measured*, reproducible
# quantity (unlike a handcrafted low/medium/high label), and unlike word
# embeddings it orders colour similarity sensibly.
_RGB = {
    "red": (255, 0, 0), "crimson": (220, 20, 60), "scarlet": (255, 36, 0),
    "ruby": (224, 17, 95), "vermilion": (227, 66, 52), "orange": (255, 165, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "purple": (128, 0, 128), "brown": (165, 42, 42), "pink": (255, 192, 203),
    "violet": (238, 130, 238), "maroon": (128, 0, 0), "cyan": (0, 255, 255),
}
_MAX_RGB = (3 * 255 ** 2) ** 0.5


def colour_similarity(a: str, b: str) -> float:
    """Measured RGB-distance similarity in [0, 1] between two colour names."""
    import math
    d = math.dist(_RGB[a], _RGB[b])
    return float(1.0 - d / _MAX_RGB)


_EMBEDDER = None


def word_similarity(a: str, b: str) -> float:
    """Measured cosine similarity between two words (MiniLM), cached model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
    v = _EMBEDDER.encode([a, b], normalize_embeddings=True)
    return float(v[0] @ v[1])


class _FamilyBase:
    """Shared machinery: token-exact assembly + filler + measured similarity."""

    def __init__(self, tokenizer, filler=None):
        self.tokenizer = tokenizer
        self._enc = lambda t: tokenizer.encode(t, add_special_tokens=False)
        self.filler = filler or SentenceBankFiller(
            tokenizer, self._filler_sentences(), self._answer_tokens())
        self._sim_cache: Dict[str, float] = {}

    def _sim(self, target: str, other: str) -> float:
        key = f"{target}|{other}"
        if key not in self._sim_cache:
            self._sim_cache[key] = word_similarity(target, other)
        return self._sim_cache[key]

    def _assemble(self, body_len, segments, rng):
        total = sum(len(s) for _, s in segments)
        filler_total = body_len - total
        if filler_total < 0:
            raise ValueError("length too small for segments")
        out: List[int] = []
        prev = 0
        for t, seg in sorted(segments, key=lambda x: x[0]):
            tgt = int(round(t * filler_total))
            out.extend(self.filler.sample(max(0, tgt - prev), rng))
            out.extend(seg)
            prev = tgt
        out.extend(self.filler.sample(filler_total - prev, rng))
        return out


class CountingTask(_FamilyBase):
    """Count objects of one colour while excluding a *similar* colour."""

    LEVELS = {"low": "blue", "medium": "orange", "high": "crimson"}
    TARGET = "red"
    ANSWER = "three"
    N_TARGET = 3

    def __init__(self, tokenizer, filler=None):
        self._count_ids = [
            tokenizer.encode(" " + w, add_special_tokens=False)[0]
            for w in _COUNT_CANDIDATES]
        super().__init__(tokenizer, filler)

    def _answer_tokens(self):
        return (self._enc(" " + " ".join(_COUNT_CANDIDATES))
                + self._enc("red item recorded object"))

    def _filler_sentences(self):
        return [
            " The committee met on Tuesday.", " Officials announced a schedule.",
            " The train was delayed again.", " Researchers presented findings.",
            " The bakery opens early.", " Attendance was higher than expected.",
            " Sales rose after the cold snap.", " The library extended its hours.",
            " A bridge will connect the districts.", " The factory reported output.",
            " The market moved indoors.", " Demand for vehicles continued to rise.",
        ]

    def generate(self, length, position=0.5, seed=0, task="counting",
                 n_distractors=0, similarity="medium", level=None):
        key = level if level is not None else similarity
        comp_colour = self.LEVELS.get(key, self.LEVELS["medium"])
        rng = random.Random(seed)
        s_val = colour_similarity(self.TARGET, comp_colour)

        segs = []
        for i in range(self.N_TARGET):  # target colour items (count = 3)
            segs.append((0.15 + 0.6 * i / max(1, self.N_TARGET - 1),
                         self._enc(_COUNT_ITEM.format(i=i, colour=self.TARGET))))
        for j in range(n_distractors):  # similar-colour competitors
            segs.append((0.05 + 0.9 * rng.random(),
                         self._enc(_COUNT_ITEM.format(i=100 + j,
                                                      colour=comp_colour))))
        body = self._assemble(length, segs, rng)
        input_ids = body + self._enc(_COUNT_QUERY)
        ans_id = self.tokenizer.encode(" " + self.ANSWER,
                                       add_special_tokens=False)[0]
        return TaskSample(
            input_ids=input_ids, answer_token_id=ans_id,
            candidate_ids=list(self._count_ids), answer_word=self.ANSWER,
            needle_positions=[position],
            meta={
                "task": "counting", "target_length": length,
                "n_distractors": n_distractors, "similarity": key,
                "target_colour": self.TARGET, "competitor_colour": comp_colour,
                "S_measured": round(s_val, 4),
                "interference_load": n_distractors * s_val,
                "context_tokens": len(input_ids),
                "position": position, "seed": seed,
            },
        )