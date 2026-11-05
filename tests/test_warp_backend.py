"""The Warp backend is held to two measurements: FEM convergence order and gradients against autograd.

Everything here skips cleanly when ``warp-lang`` is not installed: the module imports (Warp is imported lazily), so
the checks that do not need Warp still run.
"""
import numpy as np
import pytest

from pinneapple_tools.compute_backends import warp_backend as wb

requires_warp = pytest.mark.skipif(not wb.warp_available(), reason="warp-lang is not installed")


# ---------------------------------------------------------------------------
# Without Warp: the lazy import and its message
# ---------------------------------------------------------------------------


def test_module_imports_and_reports_availability():
    assert isinstance(wb.warp_available(), bool)
    device = wb.warp_device()
    assert device is None or device == "cpu" or device.startswith("cuda")


def test_install_message_when_warp_is_absent(monkeypatch):
    import sys

    monkeypatch.setattr(wb, "_WARP", None)
    monkeypatch.setitem(sys.modules, "warp", None)      # `import warp` now raises ImportError
    with pytest.raises(ImportError, match="pip install warp-lang"):
        wb._wp()


# ---------------------------------------------------------------------------
# Arguments are checked before any kernel runs
# ---------------------------------------------------------------------------


def test_poisson_rejects_impossible_arguments():
    with pytest.raises(ValueError, match="resolution"):
        wb.fem_poisson(1)
    with pytest.raises(ValueError, match="degree"):
        wb.fem_poisson(8, degree=7)


def test_manufactured_solution_satisfies_its_own_laplacian():
    """``manufactured_f`` is -laplace(u) of ``manufactured_u``, checked against finite differences."""
    h = 1e-5
    x = np.array([0.31, 0.5, 0.77])
    y = np.array([0.22, 0.6, 0.44])
    u = wb.manufactured_u
    lap = (u(x + h, y) - 2 * u(x, y) + u(x - h, y)) / h ** 2 + \
          (u(x, y + h) - 2 * u(x, y) + u(x, y - h)) / h ** 2
    assert np.allclose(-lap, wb.manufactured_f(x, y), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Zero-copy tensors
# ---------------------------------------------------------------------------


@requires_warp
def test_torch_tensor_and_warp_array_share_one_buffer():
    import torch

    t = torch.arange(12, dtype=torch.float64).reshape(4, 3).contiguous()
    a = wb.to_warp(t)
    back = wb.to_torch(a)
    assert back.data_ptr() == t.data_ptr()
    back[0, 0] = 7.0
    assert t[0, 0].item() == 7.0                    # the write went to the torch buffer, no copy


@requires_warp
def test_to_warp_refuses_a_non_contiguous_tensor_instead_of_copying():
    import torch

    t = torch.arange(12, dtype=torch.float64).reshape(4, 3).T     # transposed: not contiguous
    with pytest.raises(ValueError, match="contiguous"):
        wb.to_warp(t)


# ---------------------------------------------------------------------------
# warp.fem: the assembled Poisson problem against its manufactured solution
# ---------------------------------------------------------------------------


@requires_warp
def test_fem_poisson_q2_converges_at_third_order():
    rows = wb.poisson_convergence((8, 16, 32), degree=2)
    assert rows[0]["order"] is None
    assert all(row["l2_error"] > 0.0 for row in rows)
    assert rows[-1]["order"] > 2.5, rows                       # Q2 in L2 is ~h^3
    assert rows[-1]["l2_error"] < rows[0]["l2_error"] / 50.0    # 4x refinement at order 3 is a factor ~64
    assert rows[-1]["n_dofs"] == (2 * 32 + 1) ** 2


@requires_warp
def test_fem_poisson_q1_converges_at_second_order():
    rows = wb.poisson_convergence((8, 16, 32), degree=1)
    assert rows[-1]["order"] > 1.8, rows                       # Q1 in L2 is ~h^2
    assert rows[-1]["n_dofs"] == (32 + 1) ** 2


@requires_warp
def test_fem_poisson_solves_the_linear_system_it_assembles():
    """The CG residual is reported, and it is the residual of the system: not "0" because nobody measured it."""
    sol = wb.fem_poisson(16)
    assert sol.cg_iterations > 1
    assert sol.max_residual < 1e-10
    assert sol.dof_values.shape == (sol.n_dofs,)
    assert np.isfinite(sol.dof_values).all()


# ---------------------------------------------------------------------------
# The differentiable step, against PyTorch autograd on the module the library ships
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
@requires_warp
def test_step_gradients_agree_with_torch(dim):
    report = wb.WarpBackend(dim).gradients_against_torch(dt=0.02, n_steps=4, n_bodies=3)
    assert report["gradients_agree_1e_6"], report["gradient_max_rel_diff"]
    assert set(report["gradient_max_rel_diff"]) == {"pos0", "vel0", "force", "torque"}
    for value in report["value_max_abs_diff"].values():
        assert value < 1e-9, report["value_max_abs_diff"]


@pytest.mark.parametrize("dim", [2, 3])
@requires_warp
def test_the_step_returns_a_new_state_and_leaves_the_input_alone(dim):
    backend = wb.WarpBackend(dim)
    wp = backend._wp
    vec = wp.vec2d if dim == 2 else wp.vec3d
    state = backend.state_arrays(n_bodies=2)
    state["vel"] = wp.array(np.array([[1.0, 0.0, 0.0][:dim], [0.0, 1.0, 0.0][:dim]]), dtype=vec,
                            requires_grad=True)
    before = state["vel"].numpy().copy()
    force = wp.array(np.zeros((2, dim)), dtype=vec)
    torque = wp.array(np.zeros(dim if dim == 2 else (2, 3)), dtype=vec if dim == 3 else wp.float64)
    new = backend.step(state, force=force, torque=torque, mass=np.array([1.0, 1.0]), dt=0.5,
                       inertia=np.array([1.0, 1.0]) if dim == 2 else np.tile(np.eye(3), (2, 1, 1)),
                       inertia_inv=np.array([1.0, 1.0]) if dim == 2 else np.tile(np.eye(3), (2, 1, 1)))
    assert np.allclose(state["vel"].numpy(), before)            # in place would have overwritten this
    assert np.allclose(new["pos"].numpy(), np.asarray(before) * 0.5)   # x = x0 + dt v with x0 = 0
    assert new["pos"] is not state["pos"]
