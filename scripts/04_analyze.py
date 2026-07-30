"""Aggregate scores, quantify rank changes, and generate publication-ready figures."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

from multilingual_entropy.config import (
    FIGURES_DIR,
    LANGUAGE_BY_CODE,
    LLM_MODELS,
    NGRAM_ALPHA,
    NGRAM_ORDERS,
    PRIMARY_NGRAM_ORDER,
    RESULTS_DIR,
    TABLES_DIR,
)
from multilingual_entropy.stats import (
    delete_one_jackknife_delta,
    delete_one_jackknife_ratio,
)

NGRAM_PATH = RESULTS_DIR / "ngram_sentence_scores.parquet"
LLM_PATH = RESULTS_DIR / "llm_sentence_scores.parquet"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
AGGREGATE_PATH = RESULTS_DIR / "aggregate_metrics.csv"
CORRELATION_PATH = RESULTS_DIR / "metric_correlations.csv"
DELTA_PATH = RESULTS_DIR / "model_size_deltas.csv"
RANK_CHANGE_PATH = RESULTS_DIR / "rank_changes.csv"
SENSITIVITY_PATH = RESULTS_DIR / "ngram_order_sensitivity.csv"
PRIMARY_NGRAM = f"ngram_monolingual_o{PRIMARY_NGRAM_ORDER}_a{NGRAM_ALPHA:g}"
BALANCED_NGRAM = f"ngram_multilingual_balanced_o{PRIMARY_NGRAM_ORDER}_a{NGRAM_ALPHA:g}"
MODEL_METRICS = {model: model.replace("/", "__") for model in LLM_MODELS}
PRIMARY_METRICS = [PRIMARY_NGRAM, BALANCED_NGRAM, *MODEL_METRICS.values()]
SCATTER_LABEL_OFFSETS = {
    "jpn_Jpan": (4, 10),
    "spa_Latn": (4, -12),
}


def metric_label(metric: str) -> str:
    labels = {
        PRIMARY_NGRAM: f"Monolingual byte {PRIMARY_NGRAM_ORDER}-gram",
        BALANCED_NGRAM: f"Balanced multilingual byte {PRIMARY_NGRAM_ORDER}-gram",
        MODEL_METRICS[LLM_MODELS[0]]: "Qwen3 0.6B (4-bit)",
        MODEL_METRICS[LLM_MODELS[1]]: "Qwen3 4B (4-bit)",
    }
    return labels.get(metric, metric)


def aggregate(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, language), group in scores.groupby(["metric", "language"], sort=False):
        byte_stats = delete_one_jackknife_ratio(
            group["bits"].to_numpy(), group["bytes"].to_numpy()
        )
        character_stats = delete_one_jackknife_ratio(
            group["bits"].to_numpy(), group["characters"].to_numpy()
        )
        language_info = LANGUAGE_BY_CODE[language]
        rows.append(
            {
                "metric": metric,
                "metric_label": metric_label(metric),
                "language": language,
                "language_name": language_info.name,
                "script": language_info.script,
                "family": language_info.family,
                "bits_per_byte": byte_stats["estimate"],
                "bpb_se": byte_stats["standard_error"],
                "bpb_ci_low": byte_stats["ci_low"],
                "bpb_ci_high": byte_stats["ci_high"],
                "bits_per_character": character_stats["estimate"],
                "bpc_ci_low": character_stats["ci_low"],
                "bpc_ci_high": character_stats["ci_high"],
                "n_sentences": byte_stats["n_sentences"],
                "tokens_per_byte": (
                    group["tokens"].sum() / group["bytes"].sum()
                    if "tokens" in group and group["tokens"].notna().all()
                    else np.nan
                ),
                "tokens_per_character": (
                    group["tokens"].sum() / group["characters"].sum()
                    if "tokens" in group and group["tokens"].notna().all()
                    else np.nan
                ),
            }
        )
    aggregate_frame = pd.DataFrame(rows)
    aggregate_frame["rank_bpb"] = aggregate_frame.groupby("metric")["bits_per_byte"].rank(
        method="average"
    )
    aggregate_frame["z_bpb"] = aggregate_frame.groupby("metric")["bits_per_byte"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=1)
    )
    return aggregate_frame


def correlations(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    pivot = aggregate_frame.pivot(
        index="language", columns="metric", values="bits_per_byte"
    )
    rows: list[dict[str, object]] = []
    metrics = [metric for metric in PRIMARY_METRICS if metric in pivot]
    for index, first in enumerate(metrics):
        for second in metrics[index + 1 :]:
            spearman = spearmanr(pivot[first], pivot[second])
            kendall = kendalltau(pivot[first], pivot[second])
            rows.append(
                {
                    "metric_1": first,
                    "metric_1_label": metric_label(first),
                    "metric_2": second,
                    "metric_2_label": metric_label(second),
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "kendall_tau": kendall.statistic,
                    "kendall_p": kendall.pvalue,
                    "n_languages": len(pivot),
                }
            )
    return pd.DataFrame(rows)


def ngram_order_sensitivity(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indexed = aggregate_frame.set_index(["metric", "language"])
    for scope in ("monolingual", "multilingual_balanced"):
        for order in NGRAM_ORDERS:
            ngram_metric = f"ngram_{scope}_o{order}_a{NGRAM_ALPHA:g}"
            for model_id, model_metric in MODEL_METRICS.items():
                for unit, value_column in (
                    ("UTF-8 byte", "bits_per_byte"),
                    ("Unicode character", "bits_per_character"),
                ):
                    pair = pd.concat(
                        [
                            indexed.loc[ngram_metric][value_column].rename("ngram"),
                            indexed.loc[model_metric][value_column].rename("model"),
                        ],
                        axis=1,
                    ).sort_index()
                    spearman = spearmanr(pair["ngram"], pair["model"])
                    kendall = kendalltau(pair["ngram"], pair["model"])
                    rows.append(
                        {
                            "ngram_scope": scope,
                            "ngram_order": order,
                            "model": model_id,
                            "model_label": metric_label(model_metric),
                            "normalization": unit,
                            "spearman_rho": spearman.statistic,
                            "spearman_p": spearman.pvalue,
                            "kendall_tau": kendall.statistic,
                            "kendall_p": kendall.pvalue,
                            "n_languages": len(pair),
                        }
                    )
    return pd.DataFrame(rows)


def model_size_deltas(llm_scores: pd.DataFrame) -> pd.DataFrame:
    small_model, large_model = LLM_MODELS
    small = llm_scores[llm_scores["model"] == small_model]
    large = llm_scores[llm_scores["model"] == large_model]
    merged = small.merge(
        large,
        on=["language", "sentence_id"],
        suffixes=("_small", "_large"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for language, group in merged.groupby("language"):
        stats = delete_one_jackknife_delta(
            group["bits_small"].to_numpy(),
            group["bits_large"].to_numpy(),
            group["bytes_small"].to_numpy(),
        )
        rows.append(
            {
                "language": language,
                "language_name": LANGUAGE_BY_CODE[language].name,
                "large_minus_small_bpb": stats["estimate"],
                "improvement_bpb": -stats["estimate"],
                "improvement_ci_low": -stats["ci_high"],
                "improvement_ci_high": -stats["ci_low"],
                "n_sentences": stats["n_sentences"],
            }
        )
    return pd.DataFrame(rows).sort_values("improvement_bpb", ascending=False)


def rank_changes(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (PRIMARY_NGRAM, MODEL_METRICS[LLM_MODELS[0]]),
        (PRIMARY_NGRAM, MODEL_METRICS[LLM_MODELS[1]]),
        (BALANCED_NGRAM, MODEL_METRICS[LLM_MODELS[1]]),
        (MODEL_METRICS[LLM_MODELS[0]], MODEL_METRICS[LLM_MODELS[1]]),
    ]
    indexed = aggregate_frame.set_index(["metric", "language"])
    rows: list[dict[str, object]] = []
    for first, second in comparisons:
        for language in sorted(aggregate_frame["language"].unique()):
            first_row = indexed.loc[(first, language)]
            second_row = indexed.loc[(second, language)]
            rank_change = float(second_row["rank_bpb"] - first_row["rank_bpb"])
            rows.append(
                {
                    "metric_1": first,
                    "metric_1_label": metric_label(first),
                    "metric_2": second,
                    "metric_2_label": metric_label(second),
                    "language": language,
                    "language_name": first_row["language_name"],
                    "metric_1_bpb": first_row["bits_per_byte"],
                    "metric_2_bpb": second_row["bits_per_byte"],
                    "metric_1_rank": first_row["rank_bpb"],
                    "metric_2_rank": second_row["rank_bpb"],
                    "metric_2_minus_metric_1_rank": rank_change,
                    "absolute_rank_change": abs(rank_change),
                }
            )
    return pd.DataFrame(rows)


def save_heatmap(aggregate_frame: pd.DataFrame) -> None:
    selected = aggregate_frame[aggregate_frame["metric"].isin(PRIMARY_METRICS)]
    pivot = selected.pivot(index="language_name", columns="metric_label", values="z_bpb")
    pivot = pivot[[metric_label(metric) for metric in PRIMARY_METRICS]]
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Within-metric z-score (bits/byte)"},
    )
    plt.title("Relative cross-language difficulty changes across metrics")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_language_heatmap.png", dpi=220)
    plt.savefig(FIGURES_DIR / "cross_language_heatmap.pdf")
    plt.close()


def save_ngram_scatter(aggregate_frame: pd.DataFrame) -> None:
    pivot = aggregate_frame.pivot(
        index=["language", "language_name"], columns="metric", values="bits_per_byte"
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, model in zip(axes, MODEL_METRICS.values(), strict=True):
        sns.regplot(
            data=pivot,
            x=PRIMARY_NGRAM,
            y=model,
            ax=axis,
            scatter_kws={"s": 55},
            ci=None,
        )
        for _, row in pivot.iterrows():
            offset = SCATTER_LABEL_OFFSETS.get(row["language"], (4, 3))
            axis.annotate(
                row["language"],
                (row[PRIMARY_NGRAM], row[model]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(metric_label(PRIMARY_NGRAM) + " (bits/byte)")
        axis.set_ylabel(metric_label(model) + " (bits/byte)")
        axis.set_title(f"N-gram vs {metric_label(model)}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ngram_llm_scatter.png", dpi=220)
    plt.savefig(FIGURES_DIR / "ngram_llm_scatter.pdf")
    plt.close()


def save_scale_plot(aggregate_frame: pd.DataFrame) -> None:
    small_metric = MODEL_METRICS[LLM_MODELS[0]]
    large_metric = MODEL_METRICS[LLM_MODELS[1]]
    selected = aggregate_frame[aggregate_frame["metric"].isin([small_metric, large_metric])]
    pivot = selected.pivot(
        index=["language", "language_name"], columns="metric", values="bits_per_byte"
    ).reset_index()
    pivot = pivot.sort_values(small_metric)
    figure, axis = plt.subplots(figsize=(9, 7))
    y_positions = np.arange(len(pivot))
    for y_position, (_, row) in zip(y_positions, pivot.iterrows(), strict=True):
        small_value = row[small_metric]
        large_value = row[large_metric]
        axis.plot([large_value, small_value], [y_position, y_position], color="#b9c0c8")
        axis.scatter(large_value, y_position, color="#2563eb", s=45, label=None)
        axis.scatter(small_value, y_position, color="#dc2626", s=45, label=None)
    axis.set_yticks(y_positions, pivot["language_name"])
    axis.set_xlabel("Bits per UTF-8 byte (lower is better)")
    axis.set_title("Every language's loss under the small and larger Qwen3 models")
    axis.scatter([], [], color="#dc2626", label="Qwen3 0.6B")
    axis.scatter([], [], color="#2563eb", label="Qwen3 4B")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_scale_dumbbell.png", dpi=220)
    plt.savefig(FIGURES_DIR / "model_scale_dumbbell.pdf")
    plt.close()


def save_fertility_plot(aggregate_frame: pd.DataFrame) -> None:
    llm = aggregate_frame[aggregate_frame["metric"].isin(MODEL_METRICS.values())]
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=llm,
        x="tokens_per_character",
        y="bits_per_byte",
        hue="metric_label",
        style="metric_label",
        s=75,
    )
    for _, row in llm.iterrows():
        plt.annotate(
            row["language"],
            (row["tokens_per_character"], row["bits_per_byte"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Tokenizer fertility (tokens per Unicode character)")
    plt.ylabel("Bits per UTF-8 byte")
    plt.title("Tokenization and measured LLM loss are related but not identical")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tokenizer_fertility.png", dpi=220)
    plt.savefig(FIGURES_DIR / "tokenizer_fertility.pdf")
    plt.close()


def build_summary(
    aggregate_frame: pd.DataFrame,
    correlation_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
    rank_change_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
) -> dict[str, object]:
    dispersion = (
        aggregate_frame[aggregate_frame["metric"].isin(PRIMARY_METRICS)]
        .groupby(["metric", "metric_label"])["bits_per_byte"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    small_metric = MODEL_METRICS[LLM_MODELS[0]]
    large_metric = MODEL_METRICS[LLM_MODELS[1]]
    llm_correlation = correlation_frame[
        (correlation_frame["metric_1"] == small_metric)
        & (correlation_frame["metric_2"] == large_metric)
    ].iloc[0]
    ngram_large = correlation_frame[
        (correlation_frame["metric_1"] == PRIMARY_NGRAM)
        & (correlation_frame["metric_2"] == large_metric)
    ].iloc[0]
    primary_character_large = sensitivity_frame[
        (sensitivity_frame["ngram_scope"] == "monolingual")
        & (sensitivity_frame["ngram_order"] == PRIMARY_NGRAM_ORDER)
        & (sensitivity_frame["model"] == LLM_MODELS[1])
        & (sensitivity_frame["normalization"] == "Unicode character")
    ].iloc[0]
    rank_change_summaries: list[dict[str, object]] = []
    for (first, second), group in rank_change_frame.groupby(
        ["metric_1", "metric_2"], sort=False
    ):
        largest = group.sort_values(
            ["absolute_rank_change", "language"], ascending=[False, True]
        ).iloc[0]
        rank_change_summaries.append(
            {
                "metric_1": first,
                "metric_1_label": metric_label(first),
                "metric_2": second,
                "metric_2_label": metric_label(second),
                "median_absolute_rank_change": float(
                    group["absolute_rank_change"].median()
                ),
                "maximum_absolute_rank_change": float(
                    group["absolute_rank_change"].max()
                ),
                "largest_mover_language": largest["language"],
                "largest_mover_language_name": largest["language_name"],
                "largest_mover_signed_rank_change": float(
                    largest["metric_2_minus_metric_1_rank"]
                ),
            }
        )
    tokenizer_diagnostics: list[dict[str, object]] = []
    for model_id, model_metric in MODEL_METRICS.items():
        model_rows = aggregate_frame[aggregate_frame["metric"] == model_metric]
        tokenizer_diagnostics.append(
            {
                "model": model_id,
                "tokens_per_character_vs_bpb_spearman": float(
                    spearmanr(
                        model_rows["tokens_per_character"],
                        model_rows["bits_per_byte"],
                    ).statistic
                ),
                "tokens_per_character_vs_bpc_spearman": float(
                    spearmanr(
                        model_rows["tokens_per_character"],
                        model_rows["bits_per_character"],
                    ).statistic
                ),
            }
        )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_languages": int(aggregate_frame["language"].nunique()),
        "n_sentences_per_language": int(aggregate_frame["n_sentences"].min()),
        "primary_metrics": PRIMARY_METRICS,
        "larger_model": {
            "languages_with_lower_bpb": int((delta_frame["large_minus_small_bpb"] < 0).sum()),
            "languages_total": int(len(delta_frame)),
            "mean_language_improvement_bpb": float(delta_frame["improvement_bpb"].mean()),
            "min_language_improvement_bpb": float(delta_frame["improvement_bpb"].min()),
            "max_language_improvement_bpb": float(delta_frame["improvement_bpb"].max()),
            "languages_with_positive_improvement_ci": int(
                (delta_frame["improvement_ci_low"] > 0).sum()
            ),
            "small_large_spearman_rho": float(llm_correlation["spearman_rho"]),
            "small_large_kendall_tau": float(llm_correlation["kendall_tau"]),
        },
        "ngram_vs_large_llm": {
            "spearman_rho": float(ngram_large["spearman_rho"]),
            "spearman_p": float(ngram_large["spearman_p"]),
            "kendall_tau": float(ngram_large["kendall_tau"]),
            "kendall_p": float(ngram_large["kendall_p"]),
            "character_normalized_spearman_rho": float(
                primary_character_large["spearman_rho"]
            ),
            "character_normalized_kendall_tau": float(
                primary_character_large["kendall_tau"]
            ),
        },
        "rank_changes": rank_change_summaries,
        "tokenizer_diagnostics": tokenizer_diagnostics,
        "cross_language_dispersion": dispersion.to_dict(orient="records"),
    }


def main() -> None:
    ngram_scores = pd.read_parquet(NGRAM_PATH)
    llm_scores = pd.read_parquet(LLM_PATH)
    for frame in (ngram_scores, llm_scores):
        if frame.duplicated(["metric", "language", "sentence_id"]).any():
            raise ValueError("Duplicate sentence scores detected")
    scores = pd.concat([ngram_scores, llm_scores], ignore_index=True, sort=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    aggregate_frame = aggregate(scores)
    correlation_frame = correlations(aggregate_frame)
    delta_frame = model_size_deltas(llm_scores)
    rank_change_frame = rank_changes(aggregate_frame)
    sensitivity_frame = ngram_order_sensitivity(aggregate_frame)

    aggregate_frame.to_csv(AGGREGATE_PATH, index=False)
    correlation_frame.to_csv(CORRELATION_PATH, index=False)
    delta_frame.to_csv(DELTA_PATH, index=False)
    rank_change_frame.to_csv(RANK_CHANGE_PATH, index=False)
    sensitivity_frame.to_csv(SENSITIVITY_PATH, index=False)
    aggregate_frame[aggregate_frame["metric"].isin(PRIMARY_METRICS)].to_csv(
        TABLES_DIR / "primary_language_metrics.csv", index=False
    )

    save_heatmap(aggregate_frame)
    save_ngram_scatter(aggregate_frame)
    save_scale_plot(aggregate_frame)
    save_fertility_plot(aggregate_frame)

    summary = build_summary(
        aggregate_frame,
        correlation_frame,
        delta_frame,
        rank_change_frame,
        sensitivity_frame,
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote aggregate results to {AGGREGATE_PATH}")
    print(f"Wrote study summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
