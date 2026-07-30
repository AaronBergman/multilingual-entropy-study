import math

import pytest

from multilingual_entropy.ngram import BYTE_VOCABULARY_SIZE, ByteNGram


def test_unseen_context_is_uniform() -> None:
    model = ByteNGram(order=3, alpha=0.1)
    assert model.probability((999, 999), ord("a")) == pytest.approx(
        1 / BYTE_VOCABULARY_SIZE
    )


def test_training_increases_seen_event_probability() -> None:
    model = ByteNGram(order=1, alpha=0.1).fit(["aaaa"])
    assert model.probability((), ord("a")) > model.probability((), ord("b"))


def test_score_counts_utf8_bytes_not_characters() -> None:
    model = ByteNGram(order=1, alpha=0.1)
    assert model.score_bits("é") == pytest.approx(2 * math.log2(BYTE_VOCABULARY_SIZE))


def test_invalid_hyperparameters_fail() -> None:
    with pytest.raises(ValueError):
        ByteNGram(order=0)
    with pytest.raises(ValueError):
        ByteNGram(order=2, alpha=0)
