"""Measured checks of the adaptive sampler: the budget buys more accuracy than Sobol' at the same cost (#209).

The claim of the issue — "adaptive sampling beats Sobol' at equal budget on a documented benchmark case" — is only
worth writing down if it is counted, so the tests here run :func:`benchmark_adaptive` and compare the surrogate's RMSE
on held-out points at the same budget, and they also run the case where adaptive sampling does *not* win, which is
part of the documentation.

Measured on this machine (``2026-11-25``), budget 64, ``n_initial = 16``, ``batch = 8``, 6 repeats, 4,000 test points,
signals ``uncertainty + gradient + diversity``:

===============  ==================  =======================  ==============
problem          median RMSE Sobol'   median RMSE adaptive      median ratio
===============  ==================  =======================  ==============
boundary_layer   2.8e-3              1.1e-3                   0.41
smooth           7.0e-4              4.1e-4                   0.57
buried_bump      5.6e-2              6.4e-2                   1.14
===============  ==================  =======================  ==============

Adding ``boundary_distance`` on the boundary layer: 0.27 (worst seed 0.40). The two signals' ablation on the same
problem, at 4 repeats: ``boundary_distance + uncertainty`` alone 6.93, plus ``diversity`` 0.39 — the distance term is
what stops the batch from collapsing onto the wall.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pinneapple_data.adaptive_sampling import (  # noqa: E402
    AdaptiveDesign,
    GPSurrogate,
    Signal,
    _builtin_signals,
    adapt,
    boundary_layer,
    benchmark_adaptive,
    smooth,
)

BOX = [(0.0, 1.0), (0.0, 1.0)]


def test_adapt_spends_exactly_the_budget_and_reports_its_rounds():
    r = adapt(boundary_layer, BOX, budget=40, n_initial=12, batch=7, seed=0)
    assert isinstance(r, AdaptiveDesign)
    assert r.n_evals == 40 and r.X.shape == (40, 2) and r.y.shape == (40,)
    assert r.rounds[0]["kind"] == "initial" and r.rounds[0]["n"] == 12
    assert sum(x["n"] for x in r.rounds) == 40
    assert r.report()["signals"] == ["uncertainty", "gradient", "diversity"]
    assert r.report()["min_distance"] > 0
    json.dumps(r.to_dict())


def test_a_caller_signal_moves_the_design_where_it_says():
    """A residual-like signal brought by the caller (not the surrogate) has to steer the batch: that is the whole
    point of the multi-signal engine for PINNs and inverse problems."""
    sharp = adapt(boundary_layer, BOX, budget=40, n_initial=12, batch=7, seed=0,
                  signals=["uncertainty", {"pde_residual": lambda X, st: -((X[:, 0] - 0.8) ** 2) * 100}])
    flat = adapt(boundary_layer, BOX, budget=40, n_initial=12, batch=7, seed=0,
                 signals=["uncertainty", {"pde_residual": lambda X, st: -((X[:, 0] - 0.2) ** 2) * 100}])
    assert np.median(sharp.X[12:, 0]) > np.median(flat.X[12:, 0]) + 0.25
    assert "pde_residual" in sharp.report()["signals"]


def test_adaptive_beats_sobol_at_equal_budget_on_the_boundary_layer():
    """The acceptance criterion of #209, counted: same budget, same surrogate, RMSE on held-out points."""
    r = benchmark_adaptive("boundary_layer", budget=64, repeats=6)
    assert r["rmse_adaptive"] < r["rmse_sobol"]
    assert r["median_ratio"] < 0.65, r
    assert r["worst_ratio"] < 1.6, r                     # one seed may lose; the median is the claim
    assert all(row["rmse_sobol"] > 0 for row in r["rows"])


def test_the_boundary_signal_makes_it_better_still():
    plain = benchmark_adaptive("boundary_layer", budget=64, repeats=6)
    edged = benchmark_adaptive("boundary_layer", budget=64, repeats=6,
                               signals=("uncertainty", "gradient", "diversity", "boundary_distance"))
    assert edged["median_ratio"] < 0.45, edged
    assert edged["worst_ratio"] < 0.7, edged
    assert edged["median_ratio"] < plain["median_ratio"]


def test_the_diversity_term_is_what_keeps_the_batch_apart():
    """Without it, every candidate at the wall scores the same and the batch lands on one line — measured, not argued."""
    no_div = benchmark_adaptive("boundary_layer", budget=64, repeats=4,
                                signals=("boundary_distance", "uncertainty"))
    with_div = benchmark_adaptive("boundary_layer", budget=64, repeats=4,
                                  signals=("boundary_distance", "uncertainty", "diversity"))
    assert no_div["median_ratio"] > 3.0, no_div
    assert with_div["median_ratio"] < 1.0, with_div


def test_adaptive_does_not_hurt_a_function_with_nothing_to_find():
    r = benchmark_adaptive("smooth", budget=64, repeats=6)
    assert r["median_ratio"] < 1.2, r


def test_a_feature_the_surrogate_smoothed_away_is_not_found_and_the_benchmark_says_so():
    """The honest limit: a buried bump reports neither uncertainty nor gradient, so no gain (and no loss of note)."""
    r = benchmark_adaptive("buried_bump", budget=64, repeats=6)
    assert r["median_ratio"] > 0.9, r
    assert r["median_ratio"] < 2.5, r


def test_the_boundary_signal_peaks_at_the_faces():
    lo, hi = np.zeros(2), np.ones(2)
    st = {"lo": lo, "hi": hi}
    sig = _builtin_signals(["boundary_distance"], {})[0]
    X = np.array([[0.0, 0.5], [0.02, 0.5], [0.5, 0.5], [1.0, 0.1]])
    v = sig.values(X, st)
    assert v[0] < v[1] < v[2] and v[3] < v[1]                 # 0 at a face, 0.5 at the centre


def test_a_signal_that_returns_the_wrong_length_is_refused():
    sig = Signal("bad", lambda X, st: np.zeros(3))
    with pytest.raises(ValueError, match="returned 3 values"):
        sig.values(np.zeros((10, 2)), {})


def test_the_contract_rejects_what_it_cannot_do():
    with pytest.raises(ValueError, match="unknown signal"):
        adapt(smooth, BOX, budget=16, n_initial=8, signals=["no_such_signal"])
    with pytest.raises(ValueError, match="leaves nothing to adapt"):
        adapt(smooth, BOX, budget=8, n_initial=8)
    with pytest.raises(ValueError, match="increasing"):
        adapt(smooth, [(1.0, 0.0)], budget=16, n_initial=8)
    with pytest.raises(ValueError, match="returned"):
        adapt(lambda X: np.zeros(2), BOX, budget=16, n_initial=8)


def test_the_boxes_are_read_from_the_library_parameter_forms():
    from pinneapple_data.contracts import Parameter, ParameterSpace
    ps = ParameterSpace(name="box", parameters=[Parameter(name="a", low=0.0, high=1.0),
                                                Parameter(name="b", low=-2.0, high=2.0)])
    r = adapt(smooth, ps, budget=16, n_initial=8, seed=0)
    assert r.names == ["a", "b"]
    assert r.X[:, 0].min() >= 0.0 and r.X[:, 1].max() <= 2.0 and r.X[:, 1].min() >= -2.0
    r2 = adapt(smooth, {"a": (0.0, 1.0), "b": (-2.0, 2.0)}, budget=16, n_initial=8, seed=0)
    assert r2.names == ["a", "b"] and r2.hi.tolist() == [1.0, 2.0]
    r3 = adapt(smooth, [{"name": "x", "low": 0.0, "high": 1.0}, {"name": "y", "low": 0.0, "high": 1.0}],
               budget=16, n_initial=8, seed=0)
    assert r3.names == ["x", "y"]


def test_a_surrogate_without_gradients_still_runs():
    """The engine must accept the minimal surrogate contract: ``fit`` and ``predict -> (mean, std)``."""

    class Plain:
        def fit(self, X, y):
            self.m = GPSurrogate(2, seed=0).fit(X, y)
            return self

        def predict(self, X):
            return self.m.predict(X)

    r = adapt(smooth, BOX, budget=24, n_initial=8, batch=4, seed=0, surrogate=Plain())
    assert r.n_evals == 24
