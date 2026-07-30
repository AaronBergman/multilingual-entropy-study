from __future__ import annotations

import gzip
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

BYTE_ALPHABET_SIZE = 256


def normalize_lines(lines: Iterable[str], form: str = "NFC") -> list[str]:
    """Normalize Unicode without trimming meaningful whitespace."""
    return [unicodedata.normalize(form, line) for line in lines]


def lines_to_bytes(lines: Sequence[str]) -> bytes:
    """Encode a sentence sequence as one newline-delimited UTF-8 corpus."""
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def gzip_bytes(data: bytes, level: int = 9) -> bytes:
    """Create a deterministic gzip member."""
    return gzip.compress(data, compresslevel=level, mtime=0)


def gzip_empty_overhead(level: int = 9) -> int:
    """Measured size of an empty gzip member for this runtime."""
    return len(gzip_bytes(b"", level=level))


@dataclass(frozen=True)
class GzipMeasurement:
    original_bytes: int
    compressed_bytes: int
    empty_stream_bytes: int

    @property
    def raw_bits_per_byte(self) -> float:
        if not self.original_bytes:
            return math.nan
        return 8.0 * self.compressed_bytes / self.original_bytes

    @property
    def corrected_compressed_bytes(self) -> int:
        return max(0, self.compressed_bytes - self.empty_stream_bytes)

    @property
    def corrected_bits_per_byte(self) -> float:
        if not self.original_bytes:
            return math.nan
        return 8.0 * self.corrected_compressed_bytes / self.original_bytes


def measure_gzip(data: bytes, level: int = 9) -> GzipMeasurement:
    return GzipMeasurement(
        original_bytes=len(data),
        compressed_bytes=len(gzip_bytes(data, level=level)),
        empty_stream_bytes=gzip_empty_overhead(level=level),
    )


class ByteNgramModel:
    """Add-alpha byte n-gram language model over the full 256-byte alphabet."""

    def __init__(self, order: int, alpha: float = 0.1) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.ngram_counts: Counter[bytes] = Counter()
        self.context_counts: Counter[bytes] = Counter()

    def fit(self, data: bytes) -> ByteNgramModel:
        context_width = self.order - 1
        if context_width == 0:
            self.ngram_counts.update(bytes([value]) for value in data)
            self.context_counts[b""] = len(data)
            return self

        for index in range(context_width, len(data)):
            context = data[index - context_width : index]
            self.context_counts[context] += 1
            self.ngram_counts[context + data[index : index + 1]] += 1
        return self

    def cross_entropy_bits_per_byte(self, data: bytes) -> float:
        context_width = self.order - 1
        if len(data) <= context_width:
            return math.nan

        negative_log_likelihood = 0.0
        observations = 0
        for index in range(context_width, len(data)):
            context = data[index - context_width : index] if context_width else b""
            ngram = context + data[index : index + 1]
            numerator = self.ngram_counts[ngram] + self.alpha
            denominator = (
                self.context_counts[context] + self.alpha * BYTE_ALPHABET_SIZE
            )
            negative_log_likelihood -= math.log2(numerator / denominator)
            observations += 1
        return negative_log_likelihood / observations


def byte_ngram_cross_entropy(
    train_data: bytes, test_data: bytes, order: int, alpha: float = 0.1
) -> float:
    return (
        ByteNgramModel(order=order, alpha=alpha)
        .fit(train_data)
        .cross_entropy_bits_per_byte(test_data)
    )
