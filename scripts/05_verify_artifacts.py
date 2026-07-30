"""Verify that the committed study artifacts are complete and internally consistent."""

from __future__ import annotations

import json

import pandas as pd

from multilingual_entropy.config import (
    FIGURES_DIR,
    LANGUAGES,
    LLM_MODELS,
    NGRAM_ORDERS,
    RESULTS_DIR,
)

EVALUATION_SENTENCES = 1_012
NGRAM_SCOPES = 2


def require_file(path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty artifact: {path}")


def main() -> None:
    ngram_path = RESULTS_DIR / "ngram_sentence_scores.parquet"
    llm_path = RESULTS_DIR / "llm_sentence_scores.parquet"
    require_file(ngram_path)
    require_file(llm_path)

    ngram = pd.read_parquet(ngram_path)
    llm = pd.read_parquet(llm_path)
    expected_ngram = (
        len(LANGUAGES) * EVALUATION_SENTENCES * len(NGRAM_ORDERS) * NGRAM_SCOPES
    )
    expected_llm = len(LANGUAGES) * EVALUATION_SENTENCES * len(LLM_MODELS)
    if len(ngram) != expected_ngram:
        raise ValueError(f"Expected {expected_ngram:,} n-gram rows, found {len(ngram):,}")
    if len(llm) != expected_llm:
        raise ValueError(f"Expected {expected_llm:,} LLM rows, found {len(llm):,}")
    if ngram.duplicated(["metric", "language", "sentence_id"]).any():
        raise ValueError("Duplicate n-gram sentence scores")
    if llm.duplicated(["metric", "language", "sentence_id"]).any():
        raise ValueError("Duplicate LLM sentence scores")

    for model in LLM_MODELS:
        model_counts = llm[llm["model"] == model].groupby("language").size()
        if len(model_counts) != len(LANGUAGES) or not (
            model_counts == EVALUATION_SENTENCES
        ).all():
            raise ValueError(f"Incomplete model/language coverage for {model}")

    required_results = (
        "aggregate_metrics.csv",
        "metric_correlations.csv",
        "model_size_deltas.csv",
        "ngram_order_sensitivity.csv",
        "rank_changes.csv",
        "summary.json",
        "llm_provenance.json",
        "memory_benchmark.json",
        "mlx_scoring_validation.json",
        "ngram_provenance.json",
    )
    for filename in required_results:
        require_file(RESULTS_DIR / filename)
    for filename in (
        "cross_language_heatmap.png",
        "ngram_llm_scatter.png",
        "model_scale_dumbbell.png",
        "tokenizer_fertility.png",
    ):
        require_file(FIGURES_DIR / filename)

    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["n_languages"] != len(LANGUAGES):
        raise ValueError("Summary language count disagrees with the fixed panel")
    benchmark = json.loads(
        (RESULTS_DIR / "memory_benchmark.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (RESULTS_DIR / "mlx_scoring_validation.json").read_text(encoding="utf-8")
    )
    if not benchmark["all_passed"] or not validation["all_passed"]:
        raise ValueError("Required MLX memory or numerical validation did not pass")

    gzip_summary = RESULTS_DIR.parent / "gzip_arm" / "results" / "summary.json"
    require_file(gzip_summary)
    print(
        "Artifact verification passed: "
        f"{len(ngram):,} n-gram rows, {len(llm):,} LLM rows, "
        f"{len(LANGUAGES)} languages, {len(LLM_MODELS)} LLMs."
    )


if __name__ == "__main__":
    main()
