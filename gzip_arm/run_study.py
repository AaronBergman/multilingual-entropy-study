from __future__ import annotations

import csv
import json
import platform
import sys
import unicodedata
from pathlib import Path

import numpy as np
import scipy
from gzip_arm.metrics import (
    byte_ngram_cross_entropy,
    gzip_empty_overhead,
    lines_to_bytes,
    measure_gzip,
    normalize_lines,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, rankdata, spearmanr
from tqdm import tqdm

# Configuration: intentionally deterministic; scripts take no required arguments.
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "NTREX"
CORPUS_DIR = DATASET_DIR / "NTREX-128"
RESULTS_DIR = PROJECT_ROOT / "results"
UNICODE_NORMALIZATION = "NFC"
TRAIN_FRACTION = 0.70
NGRAM_ORDERS = (1, 2, 3, 4, 5)
NGRAM_ALPHA = 0.1
GZIP_LEVEL = 9
PREFIX_SENTENCE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 250, 500)
PRIMARY_NGRAM_ORDER = 4
EXPECTED_SENTENCES = 1997
SOURCE_COMMIT = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"


def load_language_names() -> dict[str, str]:
    names: dict[str, str] = {"eng": "English"}
    for line in (DATASET_DIR / "LANGUAGES.tsv").read_text(encoding="utf-8").splitlines():
        code, name = line.split(maxsplit=1)
        names[code] = name
    return names


def corpus_files() -> dict[str, Path]:
    files = {"eng": CORPUS_DIR / "newstest2019-src.eng.txt"}
    for path in sorted(CORPUS_DIR.glob("newstest2019-ref.*.txt")):
        code = path.name.removeprefix("newstest2019-ref.").removesuffix(".txt")
        files[code] = path
    return files


def load_lines(path: Path) -> list[str]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != EXPECTED_SENTENCES:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_SENTENCES} lines, got {len(raw_lines)}"
        )
    return normalize_lines(raw_lines, form=UNICODE_NORMALIZATION)


def sentencewise_gzip(lines: list[str]) -> tuple[float, float]:
    total_original = 0
    total_compressed = 0
    total_corrected = 0
    for line in lines:
        measurement = measure_gzip(lines_to_bytes([line]), level=GZIP_LEVEL)
        total_original += measurement.original_bytes
        total_compressed += measurement.compressed_bytes
        total_corrected += measurement.corrected_compressed_bytes
    return (
        8.0 * total_compressed / total_original,
        8.0 * total_corrected / total_original,
    )


def byte_distribution_js_divergence(train_data: bytes, test_data: bytes) -> float:
    """Jensen-Shannon divergence (bits) between train/test byte marginals."""
    train_counts = np.bincount(
        np.frombuffer(train_data, dtype=np.uint8), minlength=256
    )
    test_counts = np.bincount(
        np.frombuffer(test_data, dtype=np.uint8), minlength=256
    )
    return float(jensenshannon(train_counts, test_counts, base=2.0) ** 2)


def analyze_language(code: str, name: str, path: Path) -> tuple[dict, list[dict]]:
    lines = load_lines(path)
    split_index = int(len(lines) * TRAIN_FRACTION)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    train_bytes = lines_to_bytes(train_lines)
    test_bytes = lines_to_bytes(test_lines)
    gzip_measurement = measure_gzip(test_bytes, level=GZIP_LEVEL)
    sentence_raw_bpb, sentence_corrected_bpb = sentencewise_gzip(test_lines)

    codepoints = len("".join(test_lines)) + len(test_lines)
    row: dict[str, str | int | float] = {
        "language_code": code,
        "language_name": name,
        "sentences_total": len(lines),
        "sentences_train": len(train_lines),
        "sentences_test": len(test_lines),
        "train_bytes": len(train_bytes),
        "test_bytes": len(test_bytes),
        "test_codepoints_including_newlines": codepoints,
        "utf8_bytes_per_codepoint": len(test_bytes) / codepoints,
        "gzip_compressed_bytes": gzip_measurement.compressed_bytes,
        "gzip_empty_stream_bytes": gzip_measurement.empty_stream_bytes,
        "gzip_raw_bits_per_byte": gzip_measurement.raw_bits_per_byte,
        "gzip_corrected_bits_per_byte": gzip_measurement.corrected_bits_per_byte,
        "gzip_corrected_bits_per_codepoint": (
            gzip_measurement.corrected_bits_per_byte * len(test_bytes) / codepoints
        ),
        "sentencewise_gzip_raw_bits_per_byte": sentence_raw_bpb,
        "sentencewise_gzip_corrected_bits_per_byte": sentence_corrected_bpb,
        "train_test_byte_js_divergence_bits": byte_distribution_js_divergence(
            train_bytes, test_bytes
        ),
    }
    for order in NGRAM_ORDERS:
        bpb = byte_ngram_cross_entropy(
            train_bytes, test_bytes, order=order, alpha=NGRAM_ALPHA
        )
        row[f"ngram_{order}_bits_per_byte"] = bpb
        row[f"ngram_{order}_bits_per_codepoint"] = (
            bpb * len(test_bytes) / codepoints
        )

    overhead_rows = []
    for sentence_count in PREFIX_SENTENCE_COUNTS:
        if sentence_count > len(test_lines):
            continue
        prefix_bytes = lines_to_bytes(test_lines[:sentence_count])
        prefix_measurement = measure_gzip(prefix_bytes, level=GZIP_LEVEL)
        overhead_rows.append(
            {
                "language_code": code,
                "language_name": name,
                "sentence_count": sentence_count,
                "original_bytes": prefix_measurement.original_bytes,
                "compressed_bytes": prefix_measurement.compressed_bytes,
                "empty_stream_bytes": prefix_measurement.empty_stream_bytes,
                "raw_bits_per_byte": prefix_measurement.raw_bits_per_byte,
                "corrected_bits_per_byte": (
                    prefix_measurement.corrected_bits_per_byte
                ),
                "overhead_bits_per_byte": (
                    prefix_measurement.raw_bits_per_byte
                    - prefix_measurement.corrected_bits_per_byte
                ),
            }
        )
    return row, overhead_rows


def add_rank_columns(rows: list[dict]) -> None:
    gzip_values = np.array(
        [float(row["gzip_corrected_bits_per_byte"]) for row in rows]
    )
    primary_values = np.array(
        [float(row[f"ngram_{PRIMARY_NGRAM_ORDER}_bits_per_byte"]) for row in rows]
    )
    gzip_codepoint_values = np.array(
        [float(row["gzip_corrected_bits_per_codepoint"]) for row in rows]
    )
    primary_codepoint_values = np.array(
        [
            float(row[f"ngram_{PRIMARY_NGRAM_ORDER}_bits_per_codepoint"])
            for row in rows
        ]
    )
    gzip_ranks = rankdata(gzip_values, method="average")
    primary_ranks = rankdata(primary_values, method="average")
    gzip_codepoint_ranks = rankdata(gzip_codepoint_values, method="average")
    primary_codepoint_ranks = rankdata(primary_codepoint_values, method="average")
    for row, gzip_rank, primary_rank, gzip_cp_rank, primary_cp_rank in zip(
        rows,
        gzip_ranks,
        primary_ranks,
        gzip_codepoint_ranks,
        primary_codepoint_ranks,
        strict=True,
    ):
        row["gzip_rank"] = float(gzip_rank)
        row[f"ngram_{PRIMARY_NGRAM_ORDER}_rank"] = float(primary_rank)
        row["gzip_minus_ngram_rank"] = float(gzip_rank - primary_rank)
        row["absolute_rank_shift"] = float(abs(gzip_rank - primary_rank))
        row["gzip_minus_ngram_bits_per_byte"] = float(
            row["gzip_corrected_bits_per_byte"]
            - row[f"ngram_{PRIMARY_NGRAM_ORDER}_bits_per_byte"]
        )
        row["gzip_codepoint_rank"] = float(gzip_cp_rank)
        row[f"ngram_{PRIMARY_NGRAM_ORDER}_codepoint_rank"] = float(primary_cp_rank)
        row["gzip_minus_ngram_codepoint_rank"] = float(
            gzip_cp_rank - primary_cp_rank
        )
        row["absolute_codepoint_rank_shift"] = float(
            abs(gzip_cp_rank - primary_cp_rank)
        )


def correlation_summary(rows: list[dict], unit: str = "byte") -> list[dict]:
    if unit not in {"byte", "codepoint"}:
        raise ValueError(f"Unsupported unit: {unit}")
    gzip_values = np.array(
        [float(row[f"gzip_corrected_bits_per_{unit}"]) for row in rows]
    )
    summaries = []
    for order in NGRAM_ORDERS:
        ngram_values = np.array(
            [float(row[f"ngram_{order}_bits_per_{unit}"]) for row in rows]
        )
        spearman = spearmanr(gzip_values, ngram_values)
        kendall = kendalltau(gzip_values, ngram_values)
        summaries.append(
            {
                "ngram_order": order,
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p_value": float(kendall.pvalue),
            }
        )
    return summaries


def rank_shift_summary(rows: list[dict], column: str) -> dict:
    shifts = np.array([float(row[column]) for row in rows])
    return {
        "median": float(np.median(shifts)),
        "percentile_90": float(np.percentile(shifts, 90)),
        "maximum": float(np.max(shifts)),
        "languages_with_shift_at_most_5": int(np.sum(shifts <= 5)),
        "languages_with_shift_at_most_10": int(np.sum(shifts <= 10)),
    }


def overhead_summary(overhead_rows: list[dict]) -> list[dict]:
    summaries = []
    for count in PREFIX_SENTENCE_COUNTS:
        matches = [
            row for row in overhead_rows if int(row["sentence_count"]) == count
        ]
        summaries.append(
            {
                "sentence_count": count,
                "median_original_bytes": float(
                    np.median([row["original_bytes"] for row in matches])
                ),
                "median_raw_bits_per_byte": float(
                    np.median([row["raw_bits_per_byte"] for row in matches])
                ),
                "median_corrected_bits_per_byte": float(
                    np.median([row["corrected_bits_per_byte"] for row in matches])
                ),
                "median_empty_stream_correction_bits_per_byte": float(
                    np.median([row["overhead_bits_per_byte"] for row in matches])
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit("Dataset missing. Run `uv run python download_data.py` first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = load_language_names()
    files = corpus_files()
    rows: list[dict] = []
    overhead_rows: list[dict] = []

    for code, path in tqdm(files.items(), desc="Analyzing languages", unit="lang"):
        row, language_overhead = analyze_language(
            code=code, name=names.get(code, code), path=path
        )
        rows.append(row)
        overhead_rows.extend(language_overhead)

    add_rank_columns(rows)
    rows.sort(key=lambda row: str(row["language_code"]))
    correlations = correlation_summary(rows, unit="byte")
    codepoint_correlations = correlation_summary(rows, unit="codepoint")
    write_csv(RESULTS_DIR / "language_metrics.csv", rows)
    write_csv(RESULTS_DIR / "gzip_overhead_by_prefix.csv", overhead_rows)
    write_csv(RESULTS_DIR / "rank_correlations.csv", correlations)
    write_csv(
        RESULTS_DIR / "rank_correlations_codepoint.csv", codepoint_correlations
    )

    primary_correlation = next(
        item
        for item in correlations
        if item["ngram_order"] == PRIMARY_NGRAM_ORDER
    )
    primary_codepoint_correlation = next(
        item
        for item in codepoint_correlations
        if item["ngram_order"] == PRIMARY_NGRAM_ORDER
    )
    rank_shifts = sorted(
        rows, key=lambda row: float(row["absolute_rank_shift"]), reverse=True
    )
    summary = {
        "research_question": (
            "Do cross-language complexity rankings change between byte n-gram "
            "cross-entropy and gzip compression?"
        ),
        "dataset": {
            "name": "NTREX-128",
            "source_commit": SOURCE_COMMIT,
            "languages_including_english_source": len(rows),
            "sentences_per_language": EXPECTED_SENTENCES,
            "aligned_train_sentences": int(EXPECTED_SENTENCES * TRAIN_FRACTION),
            "aligned_test_sentences": (
                EXPECTED_SENTENCES - int(EXPECTED_SENTENCES * TRAIN_FRACTION)
            ),
        },
        "text_processing": {
            "unicode_normalization": UNICODE_NORMALIZATION,
            "encoding": "UTF-8",
            "separator": "LF after every sentence including final sentence",
        },
        "gzip": {
            "implementation": "Python gzip.compress",
            "level": GZIP_LEVEL,
            "mtime": 0,
            "empty_stream_bytes": gzip_empty_overhead(level=GZIP_LEVEL),
            "primary_metric": "header-corrected compressed bits per original byte",
            "interpretation": (
                "finite-sample compression proxy; not probabilistic cross-entropy"
            ),
        },
        "ngram": {
            "unit": "raw UTF-8 bytes",
            "orders": list(NGRAM_ORDERS),
            "alpha": NGRAM_ALPHA,
            "alphabet_size": 256,
            "primary_order": PRIMARY_NGRAM_ORDER,
            "primary_metric": "held-out cross-entropy in bits per byte",
        },
        "primary_correlation": primary_correlation,
        "secondary_codepoint_correlation": primary_codepoint_correlation,
        "rank_shift_summary": {
            "bits_per_byte": rank_shift_summary(rows, "absolute_rank_shift"),
            "bits_per_codepoint": rank_shift_summary(
                rows, "absolute_codepoint_rank_shift"
            ),
        },
        "encoding_diagnostics": {
            "gzip_bpb_vs_utf8_bytes_per_codepoint_spearman": float(
                spearmanr(
                    [row["gzip_corrected_bits_per_byte"] for row in rows],
                    [row["utf8_bytes_per_codepoint"] for row in rows],
                ).statistic
            ),
            "ngram_4_bpb_vs_utf8_bytes_per_codepoint_spearman": float(
                spearmanr(
                    [row["ngram_4_bits_per_byte"] for row in rows],
                    [row["utf8_bytes_per_codepoint"] for row in rows],
                ).statistic
            ),
        },
        "distribution_shift_diagnostic": {
            "absolute_rank_shift_vs_train_test_byte_js_divergence_spearman": (
                float(
                    spearmanr(
                        [row["absolute_rank_shift"] for row in rows],
                        [
                            row["train_test_byte_js_divergence_bits"]
                            for row in rows
                        ],
                    ).statistic
                )
            ),
            "largest_byte_distribution_shifts": [
                {
                    "language_code": row["language_code"],
                    "language_name": row["language_name"],
                    "js_divergence_bits": row[
                        "train_test_byte_js_divergence_bits"
                    ],
                }
                for row in sorted(
                    rows,
                    key=lambda row: float(
                        row["train_test_byte_js_divergence_bits"]
                    ),
                    reverse=True,
                )[:10]
            ],
        },
        "small_sample_overhead": overhead_summary(overhead_rows),
        "largest_absolute_rank_shifts": [
            {
                "language_code": row["language_code"],
                "language_name": row["language_name"],
                "gzip_rank": row["gzip_rank"],
                f"ngram_{PRIMARY_NGRAM_ORDER}_rank": row[
                    f"ngram_{PRIMARY_NGRAM_ORDER}_rank"
                ],
                "gzip_minus_ngram_rank": row["gzip_minus_ngram_rank"],
            }
            for row in rank_shifts[:15]
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "unicode_database": unicodedata.unidata_version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote study outputs to {RESULTS_DIR}")
    print(
        "Primary Spearman rho "
        f"(gzip vs {PRIMARY_NGRAM_ORDER}-gram): "
        f"{primary_correlation['spearman_rho']:.3f}"
    )


if __name__ == "__main__":
    main()
