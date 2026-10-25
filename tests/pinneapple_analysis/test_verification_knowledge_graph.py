"""Real tests for the Physics Knowledge Graph (core/knowledge_graph.py),
roadmap point 16. Every assertion here is cross-checked by hand against
the actual current state of
``pinneapple_physics.pde_environment.presets.registry`` and the actual
``tests/test_*_validation.py`` / ``tests/test_manufactured_solutions.py``
files in this repo -- not against a fixture invented for this test file.
If PINNeAPPle's preset library changes, this file's counts may need
updating (that's the point: the graph is live, not a frozen snapshot)."""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.knowledge_graph import (
    build_knowledge_graph,
    explain_preset,
    find_presets_for_phenomenon,
    find_verified_equations,
    graph_stats,
)
from pinneapple_physics.pde_environment.presets.registry import list_presets


@pytest.fixture(scope="module")
def graph():
    return build_knowledge_graph()


# ---------------------------------------------------------------------------
# Structural sanity: node/edge counts are real and non-trivial
# ---------------------------------------------------------------------------

def test_graph_has_one_preset_node_per_real_registered_preset(graph):
    real_names = set(list_presets())
    preset_nodes = {d["name"] for _, d in graph.nodes(data=True) if d.get("type") == "Preset"}
    # Every registered preset either appears in the graph, or is recorded
    # as a real, explained skip (a preset that genuinely needs kwargs).
    skipped = set(graph_stats(graph)["presets_skipped_during_graph_build"])
    assert preset_nodes | skipped == real_names
    # As of this writing every single registered preset instantiates with
    # no overrides -- verified directly against the live registry, not
    # assumed -- so nothing should actually be skipped.
    assert skipped == set()
    assert len(preset_nodes) == len(real_names)
    assert len(preset_nodes) >= 50, "preset count looks implausibly small for the real registry"


def test_graph_has_a_plausible_number_of_distinct_equations(graph):
    equation_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Equation"]
    # Many presets share a pde_kind (e.g. several elasticity presets all
    # compile to "linear_elasticity_plane_stress"), so distinct equations
    # must be fewer than presets, but there are still dozens of them.
    preset_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Preset"]
    assert 20 <= len(equation_nodes) < len(preset_nodes)


def test_graph_has_multiple_phenomenon_categories(graph):
    phenomena = {d["name"] for _, d in graph.nodes(data=True) if d.get("type") == "Phenomenon"}
    # At minimum the domains this session's own work has repeatedly
    # touched must show up as real phenomenon nodes.
    for expected in ("astrophysics", "fluid dynamics", "heat transfer", "solid mechanics", "terramechanics", "turbomachinery"):
        assert expected in phenomena, f"expected phenomenon {expected!r} missing: {sorted(phenomena)}"


def test_every_equation_node_has_at_least_one_implementing_preset(graph):
    # By construction (see build_knowledge_graph's docstring) this graph
    # only creates an Equation node for a pde_kind actually reachable
    # from a real preset -- so no Equation should be an orphan.
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "Equation":
            continue
        implementers = [
            src for src, _tgt, edata in graph.in_edges(node_id, data=True)
            if edata.get("relation") == "implements"
        ]
        assert implementers, f"equation node {node_id} has no implementing preset"


def test_every_preset_edge_types_are_well_formed(graph):
    valid_relations = {"implements", "belongs_to", "has_parameter", "cites", "verified_by"}
    for _u, _v, edata in graph.edges(data=True):
        assert edata.get("relation") in valid_relations


# ---------------------------------------------------------------------------
# explain_preset: cross-checked by hand against the real source files
# ---------------------------------------------------------------------------

def test_explain_lane_emden_polytrope_matches_astrophysics_py_by_hand():
    """Cross-checked against
    pinneapple_physics/pde_environment/presets/astrophysics.py's
    lane_emden_polytrope() factory: default n=1.0, a real Chandrasekhar
    1939 citation, and a real MMS/exact-solution test in both
    test_astrophysics_validation.py and
    test_lane_emden_numerical_validation.py."""
    g = build_knowledge_graph()
    info = explain_preset(g, "lane_emden_polytrope")
    assert info["pde_kind"] == "lane_emden_polytrope"
    assert info["phenomena"] == ["astrophysics"]
    assert info["parameters"] == {"n": 1.0}
    assert len(info["references"]) == 1
    assert "Chandrasekhar" in info["references"][0]
    assert "1939" in info["references"][0]
    assert info["verified"] is True
    verified_files = {v["file"] for v in info["verification_methods"]}
    assert "test_astrophysics_validation.py" in verified_files
    assert "test_lane_emden_numerical_validation.py" in verified_files
    assert "Chandrasekhar" in info["llm_context"]


def test_explain_kepler_two_body_orbit_matches_astrophysics_py_by_hand():
    """Cross-checked against astrophysics.py's kepler_two_body_orbit():
    default mu=398600.4418 (Earth GM, Vallado), a real Vallado 2013
    citation, and a real exact-solution residual test in
    test_astrophysics_validation.py."""
    g = build_knowledge_graph()
    info = explain_preset(g, "kepler_two_body_orbit")
    assert info["pde_kind"] == "kepler_two_body_orbit"
    assert info["phenomena"] == ["astrophysics"]
    assert info["parameters"] == pytest.approx({"mu": 398600.4418})
    assert len(info["references"]) == 1
    assert "Vallado" in info["references"][0]
    assert info["verified"] is True
    assert any(v["file"] == "test_astrophysics_validation.py" for v in info["verification_methods"])


def test_explain_burgers_1d_matches_academics_py_by_hand():
    """Cross-checked against
    pinneapple_physics/pde_environment/presets/academics.py: burgers_1d
    default nu=0.01, pde_kind "burgers" (not "burgers_1d"), and its
    "references" tuple is just a descriptive caption ("1D viscous
    Burgers template.") with no real author/year -- so this preset must
    show zero real references. Since the source app was written, the plain
    "burgers" pde_kind gained a validation test in this repository
    (tests/test_pinn_validation.py, Cole-Hopf reference), so it is now
    verified, through that file."""
    g = build_knowledge_graph()
    info = explain_preset(g, "burgers_1d")
    assert info["pde_kind"] == "burgers"
    assert info["parameters"] == pytest.approx({"nu": 0.01})
    assert info["references"] == []
    assert info["verified"] is True
    assert {m["file"] for m in info["verification_methods"]} == {"test_pinn_validation.py"}


def test_explain_unknown_preset_raises_keyerror(graph):
    with pytest.raises(KeyError):
        explain_preset(graph, "not_a_real_preset_name")


# ---------------------------------------------------------------------------
# find_presets_for_phenomenon / find_verified_equations
# ---------------------------------------------------------------------------

def test_find_presets_for_phenomenon_astrophysics_matches_real_module(graph):
    presets = find_presets_for_phenomenon(graph, "astrophysics")
    # Every genuinely astrophysics.py-sourced preset must be present.
    for expected in (
        "kepler_two_body_orbit", "space_debris_cw_relative_motion",
        "satellite_j2_perturbation", "spacecraft_attitude_euler_rotation",
        "lane_emden_polytrope", "nfw_dark_matter_potential",
        "sod_shock_tube_astro", "cr3bp_planar_synodic",
        "schwarzschild_light_bending_weak_field", "shakura_sunyaev_accretion_disk",
    ):
        assert expected in presets, f"{expected} missing from astrophysics phenomenon: {presets}"


def test_find_presets_for_phenomenon_unknown_phenomenon_returns_empty(graph):
    assert find_presets_for_phenomenon(graph, "not_a_real_phenomenon") == []


def test_find_verified_equations_is_real_nonempty_and_correct_by_inspection(graph):
    verified = find_verified_equations(graph)
    assert verified, "expected at least one real verified equation"
    # Positive controls: these pde_kinds are directly, manually confirmed
    # (by reading the actual test files) to have a real exact-solution
    # check.
    for expected in (
        "laplace", "kepler_two_body_orbit", "lane_emden_polytrope",
        "bekker_wong_terramechanics", "shakura_sunyaev_disk_1d",
        "cr3bp_planar_synodic", "schwarzschild_light_bending_weak_field",
    ):
        assert expected in verified, f"{expected} should be verified: {verified}"
    # Negative controls: these pde_kinds are directly, manually confirmed
    # to have NO dedicated exact-solution/MMS test anywhere in this repo
    # today -- the thermoelasticity/plain-elasticity engineering presets
    # (burgers was in this list in the source app; it has a test now, see
    # test_explain_burgers_1d_matches_academics_py_by_hand).
    for not_expected in ("axisymmetric_linear_elasticity", "thermoelasticity_2d"):
        assert not_expected not in verified, f"{not_expected} should NOT be verified: {verified}"


# ---------------------------------------------------------------------------
# graph_stats: the honest audit numbers
# ---------------------------------------------------------------------------

def test_graph_stats_reports_plausible_real_fractions(graph):
    stats = graph_stats(graph)
    assert stats["total_presets"] == stats["total_registered_presets_in_pinneapple"]
    assert stats["total_presets"] >= 50

    # The source app pinned the counts of its day (31/57 presets with a
    # citation, 42 references, 7 verification files). The registry has
    # grown since (72 presets, 32 cited, 43 references, 8 files on
    # 2026-10-24), so this checks lower bounds and consistency instead.
    assert 31 <= stats["presets_with_citation"] < stats["total_presets"]
    assert stats["distinct_references"] >= 42
    assert stats["distinct_verification_methods"] >= 7
    assert 25 <= len(stats["verified_equations"]) <= stats["distinct_equations"]
    assert stats["presets_with_verified_equation"] >= 25

    # The fraction strings must actually match the raw counts (no silent
    # drift between the two representations).
    n, d = stats["presets_with_citation_fraction"].split(" ")[0].split("/")
    assert int(n) == stats["presets_with_citation"]
    assert int(d) == stats["total_presets"]


def test_graph_stats_is_deterministic_across_rebuilds():
    """Rebuilding the graph from scratch twice must give identical
    audit numbers -- this is what makes it safe to call
    build_knowledge_graph() fresh on every request instead of caching a
    stale copy."""
    g1 = build_knowledge_graph()
    g2 = build_knowledge_graph()
    s1, s2 = graph_stats(g1), graph_stats(g2)
    assert s1 == s2
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
