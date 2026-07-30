import math

from gzip_arm.metrics import (
    byte_ngram_cross_entropy,
    gzip_empty_overhead,
    lines_to_bytes,
    measure_gzip,
    normalize_lines,
)


def test_normalization_and_utf8_serialization_are_explicit() -> None:
    composed = normalize_lines(["Cafe\u0301"], form="NFC")
    assert composed == ["Café"]
    assert lines_to_bytes(composed) == "Café\n".encode()


def test_empty_stream_overhead_is_subtracted_exactly() -> None:
    measurement = measure_gzip(b"abc")
    assert measurement.empty_stream_bytes == gzip_empty_overhead()
    assert measurement.corrected_compressed_bytes == (
        measurement.compressed_bytes - measurement.empty_stream_bytes
    )
    assert math.isclose(
        measurement.raw_bits_per_byte - measurement.corrected_bits_per_byte,
        8 * measurement.empty_stream_bytes / measurement.original_bytes,
    )


def test_repetition_is_easier_for_gzip() -> None:
    repeated = b"a" * 20_000
    varied = bytes(range(256)) * 78 + bytes(range(32))
    assert len(repeated) == len(varied)
    assert (
        measure_gzip(repeated).corrected_bits_per_byte
        < measure_gzip(varied).corrected_bits_per_byte
    )


def test_ngram_cross_entropy_rewards_seen_structure() -> None:
    train = (b"abababab\n" * 2_000)
    familiar = b"abababab\n" * 200
    unfamiliar = b"azbycxdw\n" * 200
    familiar_entropy = byte_ngram_cross_entropy(train, familiar, order=3)
    unfamiliar_entropy = byte_ngram_cross_entropy(train, unfamiliar, order=3)
    assert familiar_entropy < unfamiliar_entropy


def test_ngram_metrics_are_finite_for_all_orders() -> None:
    train = b"the quick brown fox\n" * 50
    test = b"the quick blue fox\n" * 5
    for order in range(1, 6):
        assert math.isfinite(byte_ngram_cross_entropy(train, test, order=order))
