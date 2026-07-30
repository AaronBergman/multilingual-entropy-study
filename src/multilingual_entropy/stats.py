"""Deterministic estimators and uncertainty summaries."""

from __future__ import annotations

import numpy as np

NORMAL_95 = 1.959963984540054


def ratio_of_sums(numerators: np.ndarray, denominators: np.ndarray) -> float:
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if numerators.shape != denominators.shape:
        raise ValueError("numerators and denominators must have matching shapes")
    if denominators.sum() <= 0:
        raise ValueError("denominator sum must be positive")
    return float(numerators.sum() / denominators.sum())


def delete_one_jackknife_ratio(
    numerators: np.ndarray, denominators: np.ndarray
) -> dict[str, float]:
    """Ratio-of-sums estimate and normal jackknife interval.

    This avoids stochastic bootstrap seeds while retaining sentence-cluster
    uncertainty for the aggregate bits-per-unit estimator.
    """

    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if numerators.ndim != 1 or denominators.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(numerators) != len(denominators):
        raise ValueError("inputs must have matching lengths")
    if len(numerators) < 3:
        raise ValueError("at least three observations are required")

    total_numerator = numerators.sum()
    total_denominator = denominators.sum()
    estimate = total_numerator / total_denominator
    leave_one_out = (total_numerator - numerators) / (total_denominator - denominators)
    leave_one_mean = leave_one_out.mean()
    standard_error = np.sqrt(
        ((len(numerators) - 1) / len(numerators))
        * np.square(leave_one_out - leave_one_mean).sum()
    )
    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "ci_low": float(estimate - NORMAL_95 * standard_error),
        "ci_high": float(estimate + NORMAL_95 * standard_error),
        "n_sentences": int(len(numerators)),
    }


def delete_one_jackknife_delta(
    first: np.ndarray, second: np.ndarray, denominators: np.ndarray
) -> dict[str, float]:
    """Jackknife interval for the difference of two ratio-of-sums estimates."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if first.shape != second.shape or first.shape != denominators.shape:
        raise ValueError("all inputs must have matching shapes")
    return delete_one_jackknife_ratio(second - first, denominators)
