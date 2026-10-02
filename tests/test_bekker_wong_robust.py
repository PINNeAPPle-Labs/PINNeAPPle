"""Robustness guarantees of the Bekker-Wong solver (pinneapple_simulation.numerical_solvers.bekker_wong)."""
import numpy as np
import pytest
import torch

from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver, bekker_wong_forces_torch
from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams


def test_gauss_legendre_matches_adaptive_quadrature():
    s = BekkerWongSolver()
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(0, 0.75, 40), rng.uniform(0.002, 0.058, 40)]
    ref = np.array([s.forces(*p) for p in P])
    gl = s.forces_batch(P[:, 0], P[:, 1], n_gl=48)
    assert np.max(np.abs(gl - ref) / (np.abs(ref).max(0) + 1e-9)) < 1e-6


def test_non_integer_sinkage_exponent_converges():
    s = BekkerWongSolver(soil=SoilParams(n=0.8))
    ref = np.array(s.forces(0.3, 0.02))
    err = [np.abs(s.forces_batch(0.3, 0.02, n_gl=k)[0] - ref).max() for k in (16, 64)]
    assert err[1] < err[0] and err[1] < 1e-3 * np.abs(ref).max()


def test_invalid_inputs_raise():
    s = BekkerWongSolver()
    for slip, z in ((0.2, 0.0), (0.2, 0.2), (1.5, 0.01)):
        with pytest.raises(ValueError):
            s.forces(slip, z)
    with pytest.raises(ValueError):
        BekkerWongSolver(soil=SoilParams(K=-1.0))


def test_braking_shear_is_bounded():
    s = BekkerWongSolver()
    fx, fz, my = s.forces(-0.5, 0.02)
    assert np.isfinite([fx, fz, my]).all() and my < 0  # braking torque, not an exponential blow-up


def test_sinkage_from_load_inverts_normal_force():
    s = BekkerWongSolver()
    z = s.sinkage_from_load(10.8, 0.2)
    assert abs(s.forces(0.2, z)[1] - 10.8) < 1e-6
    with pytest.raises(ValueError):
        s.sinkage_from_load(1e9, 0.2)


def test_dataset_never_fabricates_values():
    s = BekkerWongSolver()
    rep = {}
    X, Y = s.generate_dataset(n_slip=5, n_sink=5, n_lhs=10, strict=False, report=rep)
    assert rep["n_failed"] == 0 and len(X) == len(Y) == 35 and np.all(Y[:, 1] > 0)


def test_torch_quadrature_is_differentiable_in_soil_parameters():
    c = torch.tensor(1400.0, dtype=torch.float64, requires_grad=True)
    out = bekker_wong_forces_torch(torch.tensor([0.3], dtype=torch.float64), torch.tensor([0.02], dtype=torch.float64),
                                   R=0.125, b=0.06, c=c, tan_phi=np.tan(np.radians(30)), K=0.018, k_c=1370.0,
                                   k_phi=814000.0, n=1.0)
    out[0, 0].backward()
    assert c.grad is not None and c.grad > 0  # more cohesion -> more drawbar pull
