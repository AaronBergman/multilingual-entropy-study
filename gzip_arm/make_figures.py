from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
PRIMARY_NGRAM_ORDER = 4
ANNOTATED_LANGUAGES = 4
RANK_SHIFT_LANGUAGES = 20
FIGURE_DPI = 180


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def comparison_scatter(rows: list[dict[str, str]]) -> None:
    x = np.array(
        [float(row[f"ngram_{PRIMARY_NGRAM_ORDER}_bits_per_byte"]) for row in rows]
    )
    y = np.array([float(row["gzip_corrected_bits_per_byte"]) for row in rows])
    shifts = np.array([float(row["absolute_rank_shift"]) for row in rows])
    annotated = np.argsort(shifts)[-ANNOTATED_LANGUAGES:]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    scatter = ax.scatter(x, y, c=shifts, cmap="viridis", s=32, alpha=0.82)
    for index in annotated:
        ax.annotate(
            rows[index]["language_code"],
            (x[index], y[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    correlation = np.corrcoef(x, y)[0, 1]
    ax.set(
        xlabel=f"Byte {PRIMARY_NGRAM_ORDER}-gram held-out cross-entropy (bits/byte)",
        ylabel="gzip compressed size (header-corrected bits/original byte)",
        title="Cross-language byte n-gram vs gzip measurements",
    )
    ax.text(
        0.02,
        0.98,
        f"Pearson r = {correlation:.2f}\nColor = absolute rank shift",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, label="Absolute rank shift")
    ax.grid(alpha=0.2)
    save_figure(fig, "ngram4_vs_gzip")


def rank_shift_chart(rows: list[dict[str, str]]) -> None:
    selected = sorted(
        rows, key=lambda row: float(row["absolute_rank_shift"]), reverse=True
    )[:RANK_SHIFT_LANGUAGES]
    selected.reverse()
    values = [float(row["gzip_minus_ngram_rank"]) for row in selected]
    labels = [row["language_code"] for row in selected]
    colors = ["#c44e52" if value > 0 else "#4c72b0" for value in values]

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="gzip rank − byte 4-gram rank (positive = harder under gzip)",
        ylabel="Language code",
        title="Largest changes in cross-language ranking",
    )
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "largest_rank_shifts")


def correlation_chart(rows: list[dict[str, str]]) -> None:
    orders = [int(row["ngram_order"]) for row in rows]
    spearman = [float(row["spearman_rho"]) for row in rows]
    kendall = [float(row["kendall_tau"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(orders, spearman, marker="o", label="Spearman ρ")
    ax.plot(orders, kendall, marker="s", label="Kendall τ")
    ax.set(
        xlabel="Byte n-gram order",
        ylabel="Rank correlation with gzip",
        title="Agreement with gzip depends on n-gram order",
        xticks=orders,
        ylim=(-0.05, 1.0),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, "rank_correlation_by_ngram_order")


def overhead_chart(rows: list[dict[str, str]]) -> None:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"raw": [], "corrected": [], "overhead": []}
    )
    for row in rows:
        count = int(row["sentence_count"])
        grouped[count]["raw"].append(float(row["raw_bits_per_byte"]))
        grouped[count]["corrected"].append(float(row["corrected_bits_per_byte"]))
        grouped[count]["overhead"].append(float(row["overhead_bits_per_byte"]))

    counts = sorted(grouped)
    raw = [float(np.median(grouped[count]["raw"])) for count in counts]
    corrected = [
        float(np.median(grouped[count]["corrected"])) for count in counts
    ]
    overhead = [float(np.median(grouped[count]["overhead"])) for count in counts]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(counts, raw, marker="o", label="Raw gzip bits/byte")
    ax.plot(counts, corrected, marker="s", label="Subtract empty-member size")
    ax.plot(counts, overhead, marker="^", label="Empty-member correction")
    ax.set_xscale("log")
    ax.set(
        xlabel="Held-out prefix length (sentences, log scale)",
        ylabel="Median across languages (bits/original byte)",
        title="gzip fixed overhead dominates very small samples",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, "gzip_small_sample_overhead")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(RESULTS_DIR / "language_metrics.csv")
    comparison_scatter(metrics)
    rank_shift_chart(metrics)
    correlation_chart(read_csv(RESULTS_DIR / "rank_correlations.csv"))
    overhead_chart(read_csv(RESULTS_DIR / "gzip_overhead_by_prefix.csv"))
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
