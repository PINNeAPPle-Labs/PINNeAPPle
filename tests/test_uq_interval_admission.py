import numpy as np
import pytest

from pinneapple_analysis.uncertainty.interval_admission import assess_intervals


def assess(y, lo, hi):
    return assess_intervals(y, lo, hi, min_coverage=0.9,
                            max_mean_width=2., min_samples=30)


def test_adequate_intervals():
    y = np.zeros(100)
    result = assess(y, y - 0.5, y + 0.5)
    assert result.approved
    assert result.coverage == 1
    assert 0.9 < result.coverage_lower_bound < 1


def test_small_sample_cannot_claim_coverage():
    y = np.zeros(5)
    assert not assess(y, y - 0.5, y + 0.5).approved


def test_wide_or_undercovering_intervals():
    y = np.zeros(100)
    assert not assess(y, y - 100, y + 100).approved
    assert not assess(y, y + 0.1, y + 0.2).approved


@pytest.mark.parametrize("lo,hi", [([1.], [0.]), ([float("nan")], [1.])])
def test_invalid_intervals(lo, hi):
    with pytest.raises(ValueError):
        assess([0.], lo, hi)
