import numpy as np
import pytest

from multilingual_entropy.stats import delete_one_jackknife_delta, delete_one_jackknife_ratio


def test_jackknife_ratio_exact_for_constant_rates() -> None:
    denominators = np.array([2.0, 3.0, 4.0, 5.0])
    numerators = denominators * 1.5
    result = delete_one_jackknife_ratio(numerators, denominators)
    assert result["estimate"] == pytest.approx(1.5)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-12)


def test_jackknife_delta_direction() -> None:
    denominators = np.array([2.0, 3.0, 4.0, 5.0])
    small = denominators * 4.0
    large = denominators * 3.0
    result = delete_one_jackknife_delta(small, large, denominators)
    assert result["estimate"] == pytest.approx(-1.0)


def test_jackknife_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        delete_one_jackknife_ratio(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
