"""The root locus, against the two plants whose answer is analytic (#2324).

Every check here is a closed form, not a stored number: the first-order loop has a pole at exactly ``-a - K``, the
double integrator crosses at ``K = 1/4`` on ``±j/2``, and the discrete first-order loop leaves the unit circle at
``K = 1.5`` on ``z = -1``. The degenerate cases are the ones that decide whether a plot lies: a numerator of zero
(the gain never reaches the loop), an improper loop (more zeros than poles), and a sweep that never crosses.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.control import root_locus


def test_a_first_order_plant_puts_its_pole_exactly_at_minus_a_minus_k():
    """``1 + K/(s+a) = 0`` gives ``s = -a - K`` -- the analytic path every plot of this plant is drawn against."""
    a = 0.7
    gains = np.linspace(0.0, 10.0, 51)
    locus = root_locus([1.0], [1.0, a], gains)
    assert locus["poles"].shape == (51, 1)
    assert np.allclose(locus["poles"][:, 0].real, -a - gains, atol=1e-12)
    assert np.allclose(locus["poles"][:, 0].imag, 0.0, atol=1e-12)


def test_a_third_order_plant_crosses_at_the_routh_gain():
    """``1 + K/(s(s+1)(s+2)) = 0`` is ``s^3 + 3s^2 + 2s + K = 0``; Routh puts the crossing at ``K = 6``, where the
    polynomial factors as ``(s+3)(s^2+2)`` -- so the poles are on ``+-j*sqrt(2)``."""
    gains = np.linspace(0.0, 10.0, 1001)
    locus = root_locus([1.0], [1.0, 3.0, 2.0, 0.0], gains)
    assert locus["crossing_gain"] == pytest.approx(6.0, abs=1e-6)
    assert locus["crossing_pole"] == pytest.approx(1j * np.sqrt(2.0), abs=1e-6)
    assert locus["boundary"] == "imaginary axis"


def test_a_discrete_plant_leaves_the_unit_circle_at_the_analytic_gain():
    """``1 + K/(z - 1/2) = 0`` gives ``z = 1/2 - K``, so ``|z| = 1`` at ``K = 3/2``, on ``z = -1``."""
    gains = np.linspace(0.0, 3.0, 301)
    locus = root_locus([1.0], [1.0, -0.5], gains, discrete=True)
    assert locus["crossing_gain"] == pytest.approx(1.5, abs=1e-9)
    assert locus["crossing_pole"] == pytest.approx(-1.0, abs=1e-6)
    assert locus["boundary"] == "unit circle"


def test_a_sweep_that_never_crosses_says_so_instead_of_inventing_a_gain():
    locus = root_locus([1.0], [1.0, 1.0], np.linspace(0.0, 5.0, 51))
    assert locus["crossing_gain"] is None and locus["crossing_pole"] is None
    assert any("never crosses" in n for n in locus["notes"])


def test_a_numerator_of_zero_means_the_gain_does_not_reach_the_loop():
    locus = root_locus([0.0], [1.0, 2.0, 1.0], np.linspace(0.0, 4.0, 21))
    assert all(np.allclose(row, locus["open_loop_poles"], atol=1e-12) for row in locus["poles"])
    assert any("numerator is zero" in n for n in locus["notes"])


def test_an_improper_loop_keeps_every_closed_loop_pole_and_declares_it():
    """More zeros than poles: the closed loop has more poles than the plant, and they are all returned."""
    locus = root_locus([1.0, 0.0, 0.0], [1.0, 1.0], np.linspace(0.0, 2.0, 11))
    assert locus["poles"].shape == (11, 2)
    assert np.isnan(locus["poles"][0, -1].real), "a pole at infinity must be NaN, not zero"
    assert any("improper" in n for n in locus["notes"])
    assert any("changes along the sweep" in n for n in locus["notes"])


def test_the_branches_are_the_cheapest_continuation_of_the_row_before():
    """No zigzag, stated exactly: at every gain, the ordering of the poles is the permutation of the next row that
    moves the branches the least. ``np.roots`` gives no ordering at all, so without this the plot swaps branches
    between two adjacent gains that never cross."""
    from itertools import permutations

    gains = np.linspace(0.0, 10.0, 201)
    locus = root_locus([1.0], [1.0, 3.0, 2.0, 0.0], gains)
    poles = locus["poles"]
    for i in range(len(gains) - 1):
        chosen = float(np.abs(poles[i + 1] - poles[i]).sum())
        best = min(float(np.abs(poles[i + 1][list(perm)] - poles[i]).sum())
                   for perm in permutations(range(poles.shape[1])))
        assert chosen <= best + 1e-9, f"row {i} is not the cheapest continuation ({chosen:.4f} > {best:.4f})"


def test_a_negative_gain_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="non-negative"):
        root_locus([1.0], [1.0, 1.0], [-1.0, 0.0, 1.0])


def test_an_empty_sweep_or_a_zero_denominator_is_refused():
    with pytest.raises(ValueError, match="at least one gain"):
        root_locus([1.0], [1.0, 1.0], [])
    with pytest.raises(ValueError, match="non-zero polynomial"):
        root_locus([1.0], [0.0, 0.0], [0.0, 1.0])
