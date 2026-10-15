"""Unit tests for the ContextLab generator and metrics (no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlab.metrics.ecc import (
    effective_context_capacity,
    effective_context_mass,
)
from contextlab.tasks.generator import DEFAULT_CANDIDATES, NeedleGenerator


class FakeTokenizer:
    """Deterministic char-level tokenizer (one token per whitespace word)."""

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(w)) % 100000 for w in text.split()]


def test_candidate_tokens_distinct():
    gen = NeedleGenerator(FakeTokenizer())
    assert len(set(gen.candidate_ids)) == len(DEFAULT_CANDIDATES)


def test_length_is_controlled():
    gen = NeedleGenerator(FakeTokenizer())
    for L in (64, 128, 256):
        s = gen.generate(length=L, position=0.5, seed=1)
        # body = L tokens, plus the query suffix
        assert s.meta["body_tokens"] == L
        assert s.context_tokens > L


def test_position_monotonic():
    gen = NeedleGenerator(FakeTokenizer())
    p_low = gen.generate(length=128, position=0.05, seed=2).needle_positions[0]
    p_high = gen.generate(length=128, position=0.95, seed=2).needle_positions[0]
    assert p_low < p_high


def test_answer_only_in_needle():
    """The answer word must appear at the query position only via the needle."""
    gen = NeedleGenerator(FakeTokenizer())
    s = gen.generate(length=256, position=0.5, seed=3)
    # filler must not contain any candidate token id  (guaranteed by filtering)
    filler = set(gen._filler_ids)
    assert filler.isdisjoint(set(gen.candidate_ids))


def test_ecc_finds_crossing():
    lengths = [1, 2, 4, 8, 16, 32]
    accs = [0.9, 0.9, 0.88, 0.85, 0.7, 0.5]
    out = effective_context_capacity(lengths, accs, tau=0.9)
    assert 4 < out["ecc"] < 16
    assert out["censored"] == 0.0


def test_ecc_censored_when_flat():
    lengths = [1, 2, 4, 8]
    accs = [1.0, 1.0, 1.0, 1.0]
    out = effective_context_capacity(lengths, accs, tau=0.9)
    assert out["censored"] == 1.0
    assert out["ecc"] == 8


def test_ecm_in_unit_range():
    out = effective_context_mass([1, 2, 4, 8], [1.0, 0.9, 0.7, 0.4])
    assert 0.0 < out["ecm"] <= 1.0
