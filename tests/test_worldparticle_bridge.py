"""CPU-only tests: upstream model dependency is intentionally optional."""
import numpy as np
import pytest

from pinneapple_simulation.external_solvers.worldparticle import (
    WorldParticleState, WorldParticleBackend, load_worldparticle_npz,
    save_worldparticle_npz, worldparticle_to_upd, validate_worldparticle_rollout,
)


def test_npz_roundtrip(tmp_path):
    p = np.zeros((4, 5, 3))
    v = np.ones_like(p)
    path = save_worldparticle_npz(tmp_path / "traj.npz", p, v, feats=np.ones((5, 2)))
    loaded = load_worldparticle_npz(path)
    assert np.array_equal(loaded["position"], p)
    assert loaded["feats"].shape == (5, 2)
    assert np.array_equal(loaded["velocity"], v)


def test_invalid_csr():
    with pytest.raises(ValueError, match="CSR"):
        WorldParticleState(
            np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 1)),
            neighbors_index=np.array([5]), neighbors_row_splits=np.array([0, 1, 1]),
        ).validate()


def test_upd_and_metrics():
    p = np.zeros((4, 3, 3))
    v = np.zeros_like(p)
    sample = worldparticle_to_upd(p, v, dt=0.01)
    assert sample.domain_type() == "points"
    assert sample.domain["coords"]["time"][-1] == pytest.approx(0.03)
    assert validate_worldparticle_rollout(p, v, dt=0.01)["kinematic_rmse"] == 0.0


def test_torch_model_rollout_cpu():
    torch = pytest.importorskip("torch")
    class ConstantVelocity:
        def __call__(self, pos, vel, feats, box, box_feats, **kwargs):
            return pos + vel * 0.2, vel
    state = WorldParticleState(np.zeros((6, 3)), np.ones((6, 3)), np.ones((6, 1)))
    p, v = WorldParticleBackend(ConstantVelocity()).rollout(state, steps=3)
    assert p.shape == (4, 6, 3)
    assert p[-1, 0, 0] == pytest.approx(0.6)
    assert validate_worldparticle_rollout(p, v, dt=0.2)["kinematic_rmse"] < 1e-6


def test_nonfinite_trajectory_rejected(tmp_path):
    p = np.zeros((2, 1, 3))
    p[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_worldparticle_npz(tmp_path / "bad.npz", p, p)
