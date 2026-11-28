"""Stage 5's other backend: a PhysicsNeMo model behind the same fit/predict calls, on the same designs."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinneapple_neural.lpm_pipeline import FieldFamily
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


# ------------------------------------------------------------------ stage 5: the warm start
class _Bar(FieldFamily):
    """A tiny design space, so the warm start has two datasets to move between."""

    name = "pp_bar"
    dim = 2
    field_names = ("cp",)
    field_units = ("-",)
    geometry_params = {"length_m": (0.5, 1.5)}
    operating_params = {"alpha_deg": (-4.0, 12.0)}

    def points_and_normals(self, spec):
        n = 20
        x = np.linspace(0.0, spec.geometry["length_m"], n)
        return np.stack([x, np.zeros(n)], axis=1), np.tile(np.array([[0.0, 1.0]]), (n, 1))

    def field(self, spec):
        x = np.linspace(0.0, 1.0, 20)
        return (spec.operating["alpha_deg"] * x ** 2)[:, None]


def test_a_warm_start_keeps_the_weights_it_transfers_and_refuses_another_architecture(tmp_path, monkeypatch):
    import pinneapple_neural.lpm_pipeline as pipeline
    from pinneapple_neural.lpm_stages import training

    family = _Bar()
    # the test family is not in the shipped registry, and ``train`` resolves the family by name for the published
    # split and the field scale; the two routes are the same object here
    monkeypatch.setattr(pipeline, "_family_by_name", lambda name: family)
    generate_dataset, load_model, train = pipeline.generate_dataset, pipeline.load_model, pipeline.train
    first = tmp_path / "a"
    second = tmp_path / "b"
    # the first run still stands: the fine-tune reads its weights and `train` refuses a target whose run has no
    # model/model.pt, so the source path is the run directory, exactly as a caller would keep it
    generate_dataset(family, first / "dataset", n_geometries=4, conditions_per_geometry=1, seed=0)
    source = train(first / "dataset", first, epochs=6, batch_designs=2, width=24, code=16, n_heads=2,
                   attn_layers=1, attn_heads=2, radii=(0.2, 0.8), seed=0, device="cpu", verbose=False)

    generate_dataset(family, second / "dataset", n_geometries=4, conditions_per_geometry=1, seed=1)
    warm_dir = tmp_path / "c"
    warm = training.finetune(first, second / "dataset", warm_dir, epochs=1, batch_designs=2, width=24, code=16,
                             n_heads=2, attn_layers=1, attn_heads=2, radii=(0.2, 0.8), seed=0, device="cpu")

    def weights(run):
        return load_model(run).state_dict()

    trained = weights(first)
    started = weights(warm_dir)
    # one epoch at the default learning rate moves the weights a little; it does not replace them
    cosine = float(torch.dot(torch.cat([v.flatten().float() for v in trained.values()]),
                             torch.cat([v.flatten().float() for v in started.values()])))
    norms = float(torch.cat([v.flatten().float() for v in trained.values()]).norm()
                  * torch.cat([v.flatten().float() for v in started.values()]).norm())
    assert cosine / max(norms, 1e-12) > 0.99
    assert warm["family"] == source["family"]
    assert any("warm-started" in note for note in warm["notes"])

    # a source built for another design space is refused before a single weight is copied
    other = tmp_path / "d"
    # the dataset's own radii win over the argument (the neighbourhood cache is built for them), so the mismatch has to
    # be produced where it can happen: a dataset written for another radii set
    generate_dataset(_Bar(), other / "dataset", n_geometries=3, conditions_per_geometry=1, seed=2,
                     radii=(0.3, 0.9))
    with pytest.raises(ValueError, match="another design space"):
        train(other / "dataset", tmp_path / "e", epochs=1, batch_designs=2, width=24, code=16, n_heads=2,
              attn_layers=1, attn_heads=2, seed=0, device="cpu", init_from=first)
