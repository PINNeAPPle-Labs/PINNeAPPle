"""Stage 5's other backend: a PhysicsNeMo model behind the same fit/predict calls, on the same designs."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinneapple_neural.lpm_stages import training


def _designs(n: int = 3, points: int = 20, dim: int = 3):
    from pinneapple_neural.lpm import Design

    torch.manual_seed(0)
    out = []
    for i in range(n):
        th = torch.linspace(0, 2 * math.pi, points + 1)[:-1]
        r = 1.0 + 0.1 * i
        ring = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.zeros(points)], 1)[:, :dim]
        normal = torch.stack([torch.cos(th), torch.sin(th), torch.zeros(points)], 1)[:, :dim]
        target = (1.0 - torch.cos(th))[:, None]
        out.append(Design(points=ring, normals=normal, query=ring, params=torch.tensor([0.4 + 0.1 * i]),
                         target=target))
    return out


def test_the_backends_report_what_they_are_rather_than_failing_at_import():
    facts = training.availability()
    assert "torch" in facts and "mps" in facts
    assert isinstance(facts["domino"], bool) and isinstance(facts["x_meshgraphnet"], bool)
    # X-MeshGraphNet needs both graph packages, and the answer is the conjunction, not a guess
    assert facts["x_meshgraphnet"] == (facts["torch_geometric"] and facts["torch_scatter"])


@pytest.mark.skipif(not training.availability()["domino"], reason="physicsnemo is not installed")
def test_domino_answers_on_a_point_cloud_design_in_the_target_units():
    designs = _designs(3)
    member = training.DoMINOMember(n_params=1, n_fields=1, seed=0, device_name="cpu")
    member.fit(designs, steps=2)
    pred = member.predict(designs[0])
    assert pred.shape == (20, 1)
    assert np.isfinite(pred).all()
    described = member.describe()
    assert described["budget"].startswith("2 steps of DoMINO")
    assert "point cloud, not a mesh" in described["inputs"]
    assert described["framework"].startswith("physicsnemo")


@pytest.mark.skipif(not training.availability()["domino"], reason="physicsnemo is not installed")
def test_a_two_dimensional_design_is_embedded_in_the_three_dimensional_grid():
    designs = _designs(2, dim=2)
    member = training.DoMINOMember(n_params=1, n_fields=1, seed=0, device_name="cpu")
    bundle = member.bundle(designs[0])
    assert tuple(bundle["surface_mesh_centers"].shape) == (1, 20, 3)   # padded, not refused
    assert tuple(bundle["global_params_values"].shape) == (1, 1, 1)


def test_a_framework_member_is_scored_in_physical_units_like_any_other():
    """The conversion is the whole point of the wrapper: a member answers in the target's units, a row is physical."""
    from benchmarks.lpm_compare import TrainingMember

    class _Stub:
        def predict(self, design):
            return np.full((int(design.query.shape[0]), 1), 2.0)

        def describe(self):
            return {"model": "stub"}

    family = type("F", (), {"scales": np.array([5.0])})()
    candidate = TrainingMember(_Stub(), family, name="stub", budget="0 steps")
    assert float(candidate.predict(_designs(1)[0])[0, 0]) == pytest.approx(10.0)
    assert candidate.describe() == {"model": "stub"}
