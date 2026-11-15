"""Offline scientific regression tests of three external inspiration projects."""
import math
import pytest

from pinneapple_catalog.public_physics import list_public_physics_sources
from pinneapple_analysis.retrieval.arxiv_trends import canonical_arxiv_id, monthly_trends
from pinneapple_physics.pipe_pressure_reference import straight_pipe_pressure_drop, generate_pipe_reference


def test_curated_sources_are_distinct_and_not_preapproved():
    sources = list_public_physics_sources()
    assert len(sources) == 10
    assert len({s["id"] for s in sources}) == 10
    assert all(s["role"] == "discovery_only" for s in sources)
    assert len(list_public_physics_sources(domain="astrophysics")) == 3


def test_pipe_laminar_hagen_poiseuille_exact():
    rho, mu, v, d, length = 998.0, 0.001, 0.01, 0.05, 2.0
    r = straight_pipe_pressure_drop(v, d, length, rho, mu)
    assert r.valid and r.regime == "laminar"
    assert r.pressure_drop_pa == pytest.approx(32 * mu * v * length / d**2)


def test_pipe_blasius_reference_and_guard():
    r = straight_pipe_pressure_drop(1.0, 0.05)
    assert r.valid and r.regime == "turbulent_smooth_blasius"
    assert r.darcy_friction_factor == pytest.approx(0.3164 / r.reynolds**0.25)
    transitional = straight_pipe_pressure_drop(0.06, 0.05)
    assert not transitional.valid and transitional.regime == "transitional"
    assert math.isnan(transitional.pressure_drop_pa)
    assert not straight_pipe_pressure_drop(3.0, 0.05).valid


def test_invalid_physical_inputs_and_lengths():
    with pytest.raises(ValueError):
        straight_pipe_pressure_drop(-1, 0.1)
    with pytest.raises(ValueError):
        straight_pipe_pressure_drop(1, float("nan"))
    with pytest.raises(ValueError):
        generate_pipe_reference([1, 2], [0.1])


def test_arxiv_trends_dedup_versions_and_fill_gaps():
    papers = [
        {"link": "https://arxiv.org/abs/2401.01234v1", "data_publicacao": "2024-01-01"},
        {"link": "https://arxiv.org/abs/2401.01234v2", "data_publicacao": "2024-01-01"},
        {"link": "https://arxiv.org/abs/2403.01234v1", "data_publicacao": "2024-03-11"},
    ]
    assert canonical_arxiv_id(papers[0]["link"]) == "2401.01234"
    t = monthly_trends(papers)
    assert t["total_unique_papers"] == 2
    assert [x["count"] for x in t["months"]] == [1, 0, 1]
    assert monthly_trends(papers, exclude_boundary_months=True)["months"] == [{"month": "2024-02", "count": 0}]
