"""Controllable context-stress task generator for ContextLab.

Generates needle-in-a-haystack style prompts whose properties are known
exactly and can be varied independently:

    * context length  (in *tokens*, not characters)
    * needle position (fraction of the context body, 0.0 = start, 1.0 = end)
    * number of needles
    * task type ("retrieval" for now)

Everything is assembled at the *token* level so the measured context length is
exact and reproducible. The answer is restricted to a fixed vocabulary of
single-token words, which turns "did the model use the information?" into a
clean, deterministic K-way classification over candidate tokens.

Honesty notes:
    * Filler is neutral prose. Any sentence containing a candidate token is
      filtered out at construction time, so the only occurrence of the answer
      word in the prompt is the needle itself.
    * ``distractor_similarity`` is a placeholder for a later, harder version.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .filler import ProseFiller, SentenceBankFiller

# Verified (Qwen2.5 BPE) to tokenise to a SINGLE token with a leading space.
DEFAULT_CANDIDATES: List[str] = [
    "apple", "river", "tiger", "cloud", "stone", "candle",
    "silver", "garden", "planet", "forest", "harbor", "lantern",
]

_NEEDLE_TEMPLATE = " Important: the vault password is {word}."
_DECOY_TEMPLATE = " Note: the {noun} password is {word}."
_QUERY = "\nQuestion: What is the vault password?\nAnswer:"

# Decoy key-nouns used to create binding interference: the model must retrieve
# the value bound to the *vault* key, not any of the other keys.
_DECOY_NOUNS: List[str] = [
    "storage", "archive", "mail", "gate", "access",
    "backup", "intranet", "guest", "admin", "media",
]

# Near-synonyms of the target key ("vault") for HIGH-similarity decoys.
_NEAR_NOUNS: List[str] = [
    "safe", "strongbox", "lockbox", "strongroom", "vault door",
    "vault backup", "safe deposit", "ledger",
]
# Unrelated sentences for LOW-similarity decoys (attention load, no key clash).
_LOW_SIM_SENTENCES: List[str] = [
    " The weather in the northern region was unusually warm this week.",
    " The council postponed the meeting until further notice.",
    " A new bakery opened near the station last month.",
    " The orchestra rehearsed late into the evening.",
    " Ticket sales rose sharply after the announcement.",
    " The bridge will be closed for repairs on Sunday.",
]
_HIGH_DECOY_TEMPLATE = " Important: the {noun} password is {word}."

# Similarity weights used to compute the scalar interference load I = D * w(S).
_SIM_WEIGHT: Dict[str, float] = {"low": 0.1, "medium": 0.5, "high": 1.0}

# Neutral filler prose, deliberately unrelated to the answer vocabulary.
_FILLER_SENTENCES: List[str] = [
    "The committee met on Tuesday to review the quarterly budget.",
    "Local officials announced a new schedule for road maintenance.",
    "The train service was delayed due to signal upgrades on the north line.",
    "Researchers presented their findings at the annual conference.",
    "The bakery on the corner opens early every weekday morning.",
    "Attendance at the town hall meeting was higher than expected.",
    "The software update fixed several long-standing bugs reported by users.",
    "Sales of winter coats rose sharply after the first cold snap.",
    "The library extended its opening hours during the exam period.",
    "A new bridge will connect the two sides of the industrial district.",
    "The factory reported steady output for the third month in a row.",
    "Volunteers cleaned up the park over the weekend.",
    "The orchestra rehearsed a difficult passage for several hours.",
    "Heating costs depend heavily on the age of the building.",
    "The museum acquired a small collection of historical maps.",
    "Ticket prices will remain unchanged for the coming season.",
    "The hospital added two new wards to handle patient demand.",
    "Many commuters switched to cycling after the fuel price increase.",
    "The council debated the proposal for nearly three hours.",
    "A regional airline added a direct route to the coastal city.",
    "The university opened applications for the next academic year.",
    "Farmers reported a good harvest despite the dry summer.",
    "The accounting team migrated to a new reporting system.",
    "Rainfall totals were slightly below the long-term average.",
    "The startup hired ten engineers in a single quarter.",
    "Membership at the sports club grew steadily over the year.",
    "The publisher released three novels in the same month.",
    "Engineers tested the prototype under extreme temperatures.",
    "The market moved its stalls indoors for the winter months.",
    "A documentary about the region aired on national television.",
    "The clinic introduced an online booking system for patients.",
    "Several shops on the high street closed for renovation.",
    "The conference attracted visitors from more than twenty countries.",
    "New safety rules took effect at the beginning of the month.",
    "Demand for electric vehicles continued to rise across the region.",
    "The archive digitised thousands of documents from the last century.",
]


@dataclass
class TaskSample:
    """One generated prompt with exact ground truth."""

    input_ids: List[int]
    answer_token_id: int
    candidate_ids: List[int]
    answer_word: str
    needle_positions: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_tokens(self) -> int:
        return len(self.input_ids)


class NeedleGenerator:
    """Build controllable needle-retrieval prompts at the token level."""

    def __init__(
        self,
        tokenizer,
        candidates: Sequence[str] = DEFAULT_CANDIDATES,
        filler_sentences: Sequence[str] = _FILLER_SENTENCES,
        filler: Optional[object] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.candidates = list(candidates)

        def enc(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        self._enc = enc
        self.candidate_ids = [enc(" " + w)[0] for w in self.candidates]

        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate words do not map to distinct tokens")

        self.filler = filler or SentenceBankFiller(
            tokenizer, filler_sentences, self.candidate_ids
        )
        self._filler_ids = self.filler.raw  # kept for backward compatibility

        cand_set = set(self.candidate_ids)
        self._low_sim_tokens = [
            enc(s) for s in _LOW_SIM_SENTENCES if cand_set.isdisjoint(enc(s))
        ]
        if not self._low_sim_tokens:
            raise ValueError("no clean low-similarity decoys available")

        self._needle_tokens = {
            w: enc(_NEEDLE_TEMPLATE.format(word=w)) for w in self.candidates
        }
        self._query_ids = enc(_QUERY)

    def _fill(self, n_tokens: int, rng: random.Random) -> List[int]:
        """Return n_tokens of filler token ids from the configured provider."""
        return self.filler.sample(n_tokens, rng)

    def _assemble(self, body_len: int, segments, rng: random.Random) -> List[int]:
        """Interleave special segments into the filler at target fractions."""
        total_special = sum(len(seg) for _, seg in segments)
        filler_total = body_len - total_special
        if filler_total < 0:
            raise ValueError("length too small for requested segments")
        out: List[int] = []
        prev = 0
        for t, seg in sorted(segments, key=lambda s: s[0]):
            target = int(round(t * filler_total))
            out.extend(self._fill(max(0, target - prev), rng))
            out.extend(seg)
            prev = target
        out.extend(self._fill(filler_total - prev, rng))
        return out

    def generate_with_decoy(
        self,
        length: int,
        decoy_template: str,
        answer_word: Optional[str] = None,
        decoy_word: Optional[str] = None,
        position: float = 0.5,
        decoy_position: float = 0.5,
        seed: int = 0,
        n_copies: int = 1,
    ) -> TaskSample:
        """Build a sample with an *arbitrary* decoy sentence.

        `decoy_template` must contain a `{word}` placeholder, which is filled
        with a candidate word different from the answer. `n_copies` repeats the
        decoy (spread across the context) to control the decoy count D while
        holding similarity S fixed. This allows decoys of continuously varying
        similarity (used for the continuous-S experiment).
        """
        rng = random.Random(seed)
        if answer_word is None:
            answer_word = rng.choice(self.candidates)
        needle = self._needle_tokens[answer_word]
        answer_id = self._enc(" " + answer_word)[0]
        if decoy_word is None:
            decoy_word = rng.choice(
                [w for w in self.candidates if w != answer_word])
        decoy = self._enc(decoy_template.format(word=decoy_word))
        segments = [(position, needle)]
        if n_copies <= 1:
            segments.append((decoy_position, decoy))
        else:
            for i in range(n_copies):
                frac = 0.1 + 0.8 * (i / (n_copies - 1))
                segments.append((frac, decoy))
        body = self._assemble(length, segments, rng)
        input_ids = body + self._query_ids
        return TaskSample(
            input_ids=input_ids,
            answer_token_id=answer_id,
            candidate_ids=list(self.candidate_ids),
            answer_word=answer_word,
            needle_positions=[position],
            meta={
                "task": "retrieval_custom",
                "target_length": length,
                "decoy_template": decoy_template,
                "answer_word": answer_word,
                "decoy_word": decoy_word,
                "n_copies": n_copies,
                "context_tokens": len(input_ids),
                "body_tokens": len(body),
            },
        )

    def generate(
        self,
        length: int,
        position: float = 0.5,
        seed: int = 0,
        task: str = "retrieval",
        n_distractors: int = 0,
        similarity: str = "medium",
    ) -> TaskSample:
        """Generate one sample.

        Args:
            length:        target body length in tokens (excl. query suffix).
            position:      needle start as a fraction of the body, in [0, 1].
            seed:          RNG seed (controls filler order and answer word).
            task:          "retrieval" (only supported task for now).
            n_distractors: number of decoy "{noun} password is {word}" passages
                           inserted to create binding interference.
        """
        if task != "retrieval":
            raise ValueError(f"unsupported task: {task}")
        if not 0.0 <= position <= 1.0:
            raise ValueError("position must be in [0, 1]")
        if length < 16:
            raise ValueError("length too small")

        rng = random.Random(seed)
        answer_word = rng.choice(self.candidates)
        needle = self._needle_tokens[answer_word]
        answer_id = self._enc(" " + answer_word)[0]

        body_len = length

        if similarity not in _SIM_WEIGHT:
            raise ValueError(f"unknown similarity: {similarity}")

        # Special segments: the true needle + optional decoys. Each is placed
        # at a target fraction of the *filler* stream (needle at `position`).
        segments = [(position, needle)]
        if n_distractors > 0:
            words = [w for w in self.candidates if w != answer_word]
            rng.shuffle(words)
            nouns: List[str] = []
            if similarity == "medium":
                nouns = list(_DECOY_NOUNS)
                rng.shuffle(nouns)
            elif similarity == "high":
                nouns = list(_NEAR_NOUNS)
                rng.shuffle(nouns)
            for i in range(n_distractors):
                if similarity == "low":
                    seg = rng.choice(self._low_sim_tokens)
                else:
                    w = words[i % len(words)]
                    noun = nouns[i % len(nouns)]
                    tmpl = (_DECOY_TEMPLATE if similarity == "medium"
                            else _HIGH_DECOY_TEMPLATE)
                    seg = self._enc(tmpl.format(noun=noun, word=w))
                segments.append((rng.random(), seg))

        segments.sort(key=lambda s: s[0])
        total_special = sum(len(seg) for _, seg in segments)
        filler_total = body_len - total_special
        if filler_total < 0:
            raise ValueError("length too small for the requested distractors")

        body: List[int] = []
        prev_filler = 0
        for t, seg in segments:
            target = int(round(t * filler_total))
            body.extend(self._fill(max(0, target - prev_filler), rng))
            body.extend(seg)
            prev_filler = target
        body.extend(self._fill(filler_total - prev_filler, rng))

        input_ids = body + self._query_ids
        return TaskSample(
            input_ids=input_ids,
            answer_token_id=answer_id,
            candidate_ids=list(self.candidate_ids),
            answer_word=answer_word,
            needle_positions=[position],
            meta={
                "task": task,
                "target_length": body_len,
                "position": position,
                "seed": seed,
                "answer_word": answer_word,
                "n_distractors": n_distractors,
                "similarity": similarity,
                "interference_load": n_distractors * _SIM_WEIGHT[similarity],
                "context_tokens": len(input_ids),
                "body_tokens": len(body),
            },
        )
