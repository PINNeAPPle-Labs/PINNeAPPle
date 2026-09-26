"""Set-based encounter feasibility (research only): closed-form checks and the research-only gate."""
import math
import warnings

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
