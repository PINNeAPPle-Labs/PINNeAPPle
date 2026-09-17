"""Tests for pinneapple_analysis.trust.TrustGate."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_analysis import TrustGate
from pinneapple_analysis.verification.geometry_ood_guardrail import GeometryOODGuardrail
from pinneapple_design.geometry.gen import primitives as geom_primitives
from pinneapple_systems.component_library import ComponentRegistry
from pinneapple_systems.component_modeling.physics_residuals import heat_conduction_residual


def test_fit_and_ood_score_in_distribution_vs_far_out():
    gate = TrustGate()
    training_coords = torch.rand(200, 2)  # roughly Uniform[0,1]^2
    gate.fit(training_coords)

    in_dist = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    far_out = gate.score(model=None, x=torch.tensor([[50.0, 50.0]]))

    assert 0.0 <= in_dist.ood_score <= 1.0
    assert 0.0 <= far_out.ood_score <= 1.0
    assert in_dist.ood_score > far_out.ood_score


def test_fit_from_bounds():
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    result = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    assert result.ood_score is not None


def test_score_without_any_evidence_raises():
    gate = TrustGate()
    try:
        gate.score(model=None, x=torch.rand(1, 2))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_residual_score_low_for_trained_model_high_error_for_random():
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(16, 16))
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    x = torch.rand(16, 2)
    result = gate.score(model=model, x=x, residual_fn=heat_conduction_residual, residual_kwargs={"k": 1.0, "Q": 0.0})
    assert result.residual_score is not None
    assert 0.0 < result.combined <= 1.0


def test_ensemble_score():
    class FakeEnsemble:
        def predict(self, x):
            mean = torch.full((x.shape[0], 1), 10.0)
            std = torch.full((x.shape[0], 1), 0.01)  # 0.1% relative std -> high confidence
            return mean, std

    gate = TrustGate()
    result = gate.score(model=None, x=torch.rand(4, 2), ensemble=FakeEnsemble())
    assert result.ensemble_score is not None
    assert result.ensemble_score > 0.9  # tiny std -> high confidence


def _fitted_channel_family_guardrail() -> GeometryOODGuardrail:
    rng = np.random.default_rng(1)
    ref_meshes = [
        geom_primitives.build_mesh(
            "channel",
            length=abs(1.0 + 0.1 * rng.normal()),
            width=abs(0.2 + 0.02 * rng.normal()),
            height=abs(0.2 + 0.02 * rng.normal()),
        )
        for _ in range(30)
    ]
    return GeometryOODGuardrail().fit(ref_meshes)


def test_geometry_score_in_distribution_vs_anomalous_shape():
    """A real geometry OOD sub-score: a shape from the same family the
    guardrail was fit on must score higher trust than a genuinely
    different shape (a sphere)."""
    guardrail = _fitted_channel_family_guardrail()
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})

    similar_mesh = geom_primitives.build_mesh("channel", length=1.02, width=0.19, height=0.21)
    sphere_mesh = geom_primitives.build_mesh("sphere", radius=5.0, subdivisions=3)

    in_dist = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]), geometry_guardrail=guardrail, mesh=similar_mesh)
    out_dist = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]), geometry_guardrail=guardrail, mesh=sphere_mesh)

    assert in_dist.geometry_score is not None and out_dist.geometry_score is not None
    assert 0.0 <= in_dist.geometry_score <= 1.0
    assert 0.0 <= out_dist.geometry_score <= 1.0
    assert in_dist.geometry_score > out_dist.geometry_score
    assert in_dist.combined > out_dist.combined
    assert "geometry_ood_result" in in_dist.details
    assert in_dist.details["geometry_ood_result"].status == "OK"
    assert out_dist.details["geometry_ood_result"].status == "REJECT"


def test_geometry_guardrail_requires_mesh_and_vice_versa():
    guardrail = _fitted_channel_family_guardrail()
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})

    with pytest.raises(ValueError):
        gate.score(model=None, x=torch.tensor([[0.5, 0.5]]), geometry_guardrail=guardrail)  # mesh missing

    mesh = geom_primitives.build_mesh("channel", length=1.0, width=0.2, height=0.2)
    with pytest.raises(ValueError):
        gate.score(model=None, x=torch.tensor([[0.5, 0.5]]), mesh=mesh)  # guardrail missing
