"""Regressions found while writing the module reference examples."""
import numpy as np
import torch

from pinneapple_data import CollocationSampler
from pinneapple_design.geometry.sample import sample_uniform_box
from pinneapple_registry import ArtifactRegistry


def test_uniform_box_any_dimension():
    rng = np.random.default_rng(0)
    for d in (1, 2, 3, 4):
        lo, hi = np.zeros(d), np.arange(1, d + 1, dtype=float)
        pts = sample_uniform_box(lo, hi, 500, rng=rng)
        assert pts.shape == (500, d)
        assert (pts >= lo).all() and (pts <= hi).all()


def test_collocation_uniform_strategy_in_2d():
    s = CollocationSampler.from_bounds({"x": (0.0, 1.0), "y": (2.0, 3.0)}, strategy="uniform", seed=1)
    pts = s.sample(n_col=200, n_bc=0)["x_col"]
    assert pts.shape == (200, 2)
    assert pts[:, 1].min() >= 2.0 and pts[:, 1].max() <= 3.0


def test_registry_saves_in_the_same_second_keep_both_versions(tmp_path):
    reg = ArtifactRegistry(str(tmp_path))
    for width in (4, 8, 16):
        reg.models.save("p", torch.nn.Linear(2, width), metadata={"width": width})
    versions = reg.models.versions("p")
    assert len(versions) == 3
    assert reg.models.latest("p") == versions[-1]
    assert reg.models.metadata("p")["width"] == 16
    assert [reg.models.metadata("p", v)["width"] for v in versions] == [4, 8, 16]
