"""Set-based encounter feasibility (research only): closed-form checks and the research-only gate."""
import math
import warnings

import numpy as np
import pytest

from pinneapple_analysis.uncertainty.reachability import BodyState, encounter_feasibility, time_to_reach_region
from pinneapple_neural._licencas import ResearchOnlyError, ResearchOnlyWarning


@pytest.fixture(autouse=True)
def _research_mode(monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    warnings.simplefilter("ignore", ResearchOnlyWarning)


def test_rendezvous_time_matches_closed_form():
    """Chaser at rest, target region of radius r at distance D: 0.5 a t^2 = D - r."""
    a_max, D, r = 0.2, 100.0, 5.0
    t = time_to_reach_region(BodyState([0, 0, 0], [0, 0, 0], max_accel=a_max), [D, 0, 0], r, horizon=100)
    assert t == pytest.approx(math.sqrt(2 * (D - r) / a_max), rel=1e-6)


def test_head_on_uncertain_bodies_first_overlap_time():
    """Centres close at speed V from D; each region radius r0 + 0.5 a t^2 -> D - V t = 2 r0 + a t^2."""
    D, V, r0, a = 1000.0, 20.0, 10.0, 0.5
    A = BodyState([0, 0], [V / 2, 0], pos_uncertainty=r0, max_accel=a)
    B = BodyState([D, 0], [-V / 2, 0], pos_uncertainty=r0, max_accel=a)
    res = encounter_feasibility(A, B, horizon=60)
    t_exact = (-V + math.sqrt(V * V + 4 * a * (D - 2 * r0))) / (2 * a)
    assert res["possible"] and res["t_first"] == pytest.approx(t_exact, rel=1e-6)


def test_diverging_bodies_never_meet_and_latency_delays_contact():
    A = BodyState([0, 0], [-1, 0], pos_uncertainty=1)
    B = BodyState([50, 0], [1, 0], pos_uncertainty=1)
    assert not encounter_feasibility(A, B, horizon=100)["possible"]
    fast = time_to_reach_region(BodyState([0, 0], [0, 0], max_accel=1.0), [50, 0], horizon=100)
    slow = time_to_reach_region(BodyState([0, 0], [0, 0], max_accel=1.0, latency=3.0), [50, 0], horizon=100)
    assert slow == pytest.approx(fast + 3.0, rel=1e-6)


def test_research_only_gate(monkeypatch):
    warnings.simplefilter("default", ResearchOnlyWarning)
    with pytest.warns(ResearchOnlyWarning, match="RESEARCH ONLY"):
        encounter_feasibility(BodyState([0], [0]), BodyState([1], [0]), 1.0)
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    with pytest.raises(ResearchOnlyError):
        encounter_feasibility(BodyState([0], [0]), BodyState([1], [0]), 1.0)


def test_fast_flyby_between_grid_points_is_caught():
    # regression: the closest approach (distance 0.05 < radius 0.1 at t = 0.50025) falls between two of the 2001
    # samples, where the margin is 0.40; the overlap lasted 1e-4 s and contact was reported impossible
    A = BodyState([-1000.5, 0.05], [2000.0, 0.0])
    B = BodyState([0.0, 0.0], [0.0, 0.0], pos_uncertainty=0.1)
    res = encounter_feasibility(A, B, horizon=1.0)
    assert res["possible"]
    assert res["min_margin"] == pytest.approx(-0.05, abs=1e-9)
    t_exact = (1000.5 - math.sqrt(0.1 ** 2 - 0.05 ** 2)) / 2000.0
    assert res["t_first"] == pytest.approx(t_exact, rel=1e-9)


def test_feasibility_matches_a_dense_brute_force_reference():
    # near misses: the centres pass within 0..3 of each other at a random time, at speeds of up to ~280 per unit time,
    # so n = 51 samples are far too coarse and the answer rests on the closest-approach search
    rng = np.random.default_rng(3)
    dense = np.linspace(0.0, 1.0, 2_000_001)
    hits = misses = 0
    for _ in range(40):
        t0, P, miss = rng.uniform(0.05, 0.95), rng.uniform(-50, 50, 2), rng.uniform(-3, 3, 2)
        va, vb = rng.uniform(-200, 200, 2), rng.uniform(-200, 200, 2)
        A = BodyState(P - va * t0, va, pos_uncertainty=rng.uniform(0, 1), vel_uncertainty=rng.uniform(0, 0.5),
                      max_accel=rng.uniform(0, 2), latency=rng.uniform(0, 0.5))
        B = BodyState(P + miss - vb * t0, vb, pos_uncertainty=rng.uniform(0, 1))
        ref = (np.linalg.norm(A.center(dense) - B.center(dense), axis=1) - A.radius(dense) - B.radius(dense)).min()
        if abs(ref) > 1e-3:
            assert encounter_feasibility(A, B, horizon=1.0, n=51)["possible"] == (ref <= 0)
            hits, misses = hits + (ref <= 0), misses + (ref > 0)
    assert hits >= 5 and misses >= 5


@pytest.mark.parametrize("kw", [{"pos_uncertainty": -1.0}, {"max_accel": -0.1}, {"latency": -1.0}])
def test_negative_uncertainty_is_rejected(kw):
    with pytest.raises(ValueError, match="non-negative"):
        encounter_feasibility(BodyState([0, 0], [0, 0], **kw), BodyState([1, 0], [0, 0]), 1.0)
    with pytest.raises(ValueError, match="horizon"):
        encounter_feasibility(BodyState([0, 0], [0, 0]), BodyState([1, 0], [0, 0]), 0.0)
