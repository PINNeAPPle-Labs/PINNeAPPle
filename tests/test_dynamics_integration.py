"""Fokker-Planck survival (first passage from a PDE solution), on closed forms and a pp.solve-like object."""
from __future__ import annotations

import types

import numpy as np
import pytest

from pinneapple_analysis import dynamics as dyn
from pinneapple_analysis.dynamics.first_passage import (
    CARD_FOKKER_PLANCK,
    brownian_exit_survival,
    survival_from_density,
    survival_from_solution,
)

D = 0.1


def _density(x, t):
    """Exact density of Brownian motion (sigma^2 = 2 D) killed at 0 and 1, started from (pi/2) sin(pi x)."""
    return 0.5 * np.pi * np.sin(np.pi * x)[None, :] * np.exp(-D * np.pi ** 2 * t)[:, None]


def test_survival_from_density_matches_the_closed_form():
    x, t = np.linspace(0, 1, 401), np.linspace(0, 2, 81)
    r = survival_from_density(t, x, _density(x, t))
    exact = np.exp(-D * np.pi ** 2 * t)
    assert np.max(np.abs(r.estimate["S"] - exact)) < 1e-5
    assert r.estimate["median"] == pytest.approx(np.log(2) / (D * np.pi ** 2), rel=1e-4)
    assert r.estimate["restricted_mean"] == pytest.approx((1 - exact[-1]) / (D * np.pi ** 2), rel=1e-4)
    assert np.nanmax(np.abs(r.estimate["hazard"] - D * np.pi ** 2)) < 2e-3      # constant hazard of one mode
    assert r.estimate["mean_kind"] == "restricted"
    assert {d.name: d.status for d in r.diagnostics}["initial_mass"] == "ok"
    assert r.method == CARD_FOKKER_PLANCK.name


def test_point_start_agrees_with_the_backward_series():
    # density from x0 = 0.3 by the eigenfunction expansion of the forward equation; S from the backward series
    x, t, x0 = np.linspace(0, 1, 801), np.linspace(0.05, 1.5, 60), 0.3
    n = np.arange(1, 200)
    p = 2 * np.einsum("n,xn,tn->tx", np.sin(n * np.pi * x0), np.sin(np.outer(x, n) * np.pi),
                      np.exp(-D * np.outer(t, n * n) * np.pi ** 2))
    r = survival_from_density(t, x, p)
    ref = brownian_exit_survival(t, x0, 0.0, 1.0, 0.0, np.sqrt(2 * D))
    assert np.max(np.abs(r.estimate["S"] - ref)) < 1e-5


def test_survival_from_a_pp_solve_like_solution_and_against_monte_carlo():
    problem = types.SimpleNamespace(coords=("x", "t"), fields=("u",))
    sol = types.SimpleNamespace(problem=problem,
                                predict=lambda P: (0.5 * np.pi * np.sin(np.pi * P[:, 0])
                                                   * np.exp(-D * np.pi ** 2 * P[:, 1]))[:, None],
                                metadata=lambda: {"method": "analytic", "problem": "exit"})
    x, t = np.linspace(0, 1, 101), np.linspace(0, 1, 11)
    r = survival_from_solution(sol, x, t)
    assert r.details["solution"]["method"] == "analytic" and r.details["field"] == "u"
    rng = np.random.default_rng(0)
    x0 = np.arccos(1 - 2 * rng.uniform(size=20_000)) / np.pi
    mc = dyn.FirstPassageAnalyzer(dyn.BoundaryCondition.interval(0, 1, name="exit")).monte_carlo(
        dyn.SDE.brownian(0, np.sqrt(2 * D)), x0[:, None], t_max=1.0, dt=1e-3, n_paths=x0.size, seed=0)
    S_mc = mc.distributions["exit"].survival(t)
    assert np.max(np.abs(S_mc - r.estimate["S"])) < 0.015                    # MC sd <= 0.0035 per point


def test_problems_are_reported_or_refused():
    x, t = np.linspace(0, 1, 51), np.linspace(0, 1, 11)
    p = _density(x, t)
    grow = p * np.exp(2 * D * np.pi ** 2 * t)[:, None]      # mass entering: S grows
    diag = {d.name: d.status for d in survival_from_density(t, x, grow).diagnostics}
    assert diag["max_increase_of_S"] == "warn"
    diag = {d.name: d.status for d in survival_from_density(t, x, 2 * p).diagnostics}
    assert diag["initial_mass"] == "warn"
    assert survival_from_density(t, x, 2 * p, normalise=True).estimate["S"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        survival_from_density(t, x, p[:, :-1])
    with pytest.raises(ValueError):
        survival_from_density(t[::-1], x, p)
    bad = types.SimpleNamespace(problem=types.SimpleNamespace(coords=("x", "y", "t"), fields=("u",)))
    with pytest.raises(ValueError):
        survival_from_solution(bad, x, t)
