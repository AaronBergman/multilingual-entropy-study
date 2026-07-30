"""Transparent add-alpha byte n-gram language models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

BYTE_VOCABULARY_SIZE = 256
BOS_SYMBOL = 256


@dataclass
class ByteNGram:
    """A fixed-order byte Markov model with add-alpha smoothing.

    The model predicts UTF-8 bytes. Sentence starts are represented by repeated
    out-of-vocabulary BOS symbols, but only actual text bytes are scored.
    """

    order: int
    alpha: float = 0.1
    counts: dict[tuple[int, ...], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    totals: Counter[tuple[int, ...]] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    @property
    def context_width(self) -> int:
        return self.order - 1

    def _events(self, text: str) -> Iterable[tuple[tuple[int, ...], int]]:
        encoded = text.encode("utf-8")
        history = [BOS_SYMBOL] * self.context_width + list(encoded)
        for offset, target in enumerate(encoded):
            start = offset
            context = tuple(history[start : start + self.context_width])
            yield context, target

    def fit(self, texts: Iterable[str]) -> ByteNGram:
        for text in texts:
            for context, target in self._events(text):
                self.counts[context][target] += 1
                self.totals[context] += 1
        return self

    def score_bits(self, text: str) -> float:
        bits = 0.0
        smoothing_mass = self.alpha * BYTE_VOCABULARY_SIZE
        for context, target in self._events(text):
            numerator = self.counts[context][target] + self.alpha
            denominator = self.totals[context] + smoothing_mass
            bits -= math.log2(numerator / denominator)
        return bits

    def probability(self, context: tuple[int, ...], target: int) -> float:
        if len(context) != self.context_width:
            raise ValueError("context has the wrong width")
        if not 0 <= target < BYTE_VOCABULARY_SIZE:
            raise ValueError("target must be a byte")
        return (self.counts[context][target] + self.alpha) / (
            self.totals[context] + self.alpha * BYTE_VOCABULARY_SIZE
        )
