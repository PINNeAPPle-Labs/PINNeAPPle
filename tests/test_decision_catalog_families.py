"""Extra model families (KPI regressor, POD ROM, point-cloud operator, hybrid) in the decision engine."""
import pytest

from pinneapple_analysis.verification.architecture_recommendation import recommend_architecture
from pinneapple_decision import ModelSelector, decide


def _cfd04(**kw):
    p = {"n_simulations": 32, "needs_parameter_generalization": True, "representation": "unstructured_mesh",
         "time_dependent": False, "target_kind": "kpi", "has_solver": True, "geometry_varies": True,
         "is_inverse": False, "dimension": 3}
    p.update(kw)
    return p


def test_defaults_keep_the_old_recommendation():
    assert recommend_architecture(n_high_fidelity_simulations=100, needs_parameter_generalization=True
                                  ).recommended[0] == "fno"


def test_kpi_target_recommends_kpi_regressor():
    rec = recommend_architecture(n_high_fidelity_simulations=32, has_analytical_solver=True,
                                 needs_parameter_generalization=True, geometry_varies=True, target_kind="kpi")
    assert rec.recommended[0] == "kpi_regressor" and "Gaussian" in rec.reasoning


def test_engine_scores_kpi_regressor_for_a_kpi_problem():
    d = decide(_cfd04(), choice=ModelSelector.choice(["kpi_regressor", "mesh_graph_net", "deeponet"]))
    assert d.selected == "kpi_regressor" and "kpi_regressor" not in d.excluded


def test_pod_only_with_fixed_topology_and_fields():
    rec = recommend_architecture(n_high_fidelity_simulations=200, needs_parameter_generalization=True,
                                 target_kind="field", fixed_topology=True)
    assert "pod_rom" in rec.recommended
    d = decide(_cfd04(target_kind="field"), choice=ModelSelector.choice(["pod_rom", "mesh_graph_net"]))
    assert d.excluded["pod_rom"].startswith("must_support_geometry")   # geometry varies -> POD excluded


def test_point_cloud_operator_is_research_only_in_commercial_mode(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    d = decide(_cfd04(target_kind="field", n_simulations=300),
               choice=ModelSelector.choice(["point_cloud_operator", "mesh_graph_net"]))
    assert d.excluded["point_cloud_operator"].startswith("commercial_use_allowed")
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "0")
    d = decide(_cfd04(target_kind="field", n_simulations=300),
               choice=ModelSelector.choice(["point_cloud_operator", "mesh_graph_net"]))
    assert "point_cloud_operator" not in d.excluded


def test_hybrid_needs_the_physics_postprocessor_fact():
    rec = recommend_architecture(n_high_fidelity_simulations=32, target_kind="kpi", has_physics_postprocessor=True)
    assert "hybrid_surrogate_physics" in rec.recommended
    info = ModelSelector.option_info()["hybrid_surrogate_physics"]
    assert info.implementation == "pinneapple_neural.workflows.hybrid_surrogate_physics.HybridSurrogatePhysics"
    import importlib

    mod, _, sym = info.implementation.rpartition(".")
    assert hasattr(importlib.import_module(mod), sym)


def test_adapter_reads_the_new_facts():
    from pinneapple_decision import adapt_problem
    a = adapt_problem({"same_mesh": True, "physics_postprocessor": "yes"})
    assert a.problem["fixed_topology"] is True and a.problem["has_physics_postprocessor"] is True
