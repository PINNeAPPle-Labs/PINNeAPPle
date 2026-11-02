"""Core coverage (#1129, toward 1.0) for the PINN compiler's differential operators, the trainer's audit helpers and
the PDE-family capabilities. The operators are compared with the exact derivatives of polynomials, so the expected
values are analytic."""
import json

import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.trainer import audit as A  # noqa: E402
from pinneapple_physics.pde_environment import capabilities as C  # noqa: E402
from pinneapple_physics.pinn_solver.compiler import autograd_ops as O  # noqa: E402


def _x():
    torch.manual_seed(0)
    return torch.rand(16, 3, dtype=torch.float64, requires_grad=True)          # columns (t, x, y)


# ------------------------------------------------------------------------------------------------ operators
def test_divergence_time_derivative_and_normal_derivative_are_exact_on_polynomials():
    X = _x()
    t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    V = torch.cat([x ** 2 * y, x * y ** 3], dim=1)
    div = O.divergence(V, X, coord_indices=[1, 2])
    assert torch.allclose(div, 2 * x * y + 3 * x * y ** 2)                        # d(x^2 y)/dx + d(x y^3)/dy
    u = t ** 2 * x + y
    assert torch.allclose(O.time_derivative(u, X, 0), 2 * t * x)                 # analytic
    n = torch.tensor([[0.0, 0.6, 0.8]], dtype=torch.float64).expand(16, 3)
    assert torch.allclose(O.norm_dot_grad(u, X, n), 0.6 * t ** 2 + 0.8)          # grad u . n
    W = torch.cat([t * x, x * y, y * t], dim=1)
    assert torch.allclose(O.divergence(W, X), x + y + t)                          # all columns paired by index


def test_ensure_tensor_and_mse():
    y = torch.ones(3, 1)

    class Out:
        pass
    o = Out()
    o.y = y
    assert O.ensure_tensor(o) is y and O.ensure_tensor(y) is y
    assert O.mse(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0])).item() == pytest.approx(5.0)


# ------------------------------------------------------------------------------------------------ audit
def test_set_seed_makes_every_generator_repeat():
    import random

    import numpy as np
    A.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    A.set_seed(123)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first


def test_set_deterministic_switches_torch_to_deterministic_algorithms():
    before = torch.are_deterministic_algorithms_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        A.set_deterministic(False)
        assert torch.are_deterministic_algorithms_enabled() == before             # False: nothing changes
        A.set_deterministic(True)
        assert torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(before)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def test_env_fingerprint_and_run_logger(tmp_path):
    fp = A.env_fingerprint()
    assert fp["torch"] == torch.__version__ and set(fp) >= {"python", "platform", "cuda_available"}
    log = A.RunLogger(out_dir=str(tmp_path / "runs"), run_name="r1")
    log.log({"epoch": 0, "loss": 1.5})
    log.log({"epoch": 1, "loss": 0.5, "ts_unix": 7})
    rows = [json.loads(s) for s in open(log.path)]
    assert [r["loss"] for r in rows] == [1.5, 0.5] and rows[1]["ts_unix"] == 7 and "ts_unix" in rows[0]
    log.save_config({"lr": 1e-3})
    assert json.load(open(tmp_path / "runs" / "r1.config.json")) == {"lr": 1e-3}
    log.save_artifact("meta.json", [1, 2])
    log.save_artifact("w.pt", torch.ones(2))
    assert json.load(open(tmp_path / "runs" / "meta.json")) == [1, 2]
    assert torch.equal(torch.load(tmp_path / "runs" / "w.pt"), torch.ones(2))


# ------------------------------------------------------------------------------------------------ capabilities
def test_pde_families_lookup_and_identification():
    kinds = C.list_pde_families()
    assert kinds == sorted(kinds) and {"burgers", "heat_equation", "navier_stokes", "poisson", "wave"} <= set(kinds)
    fam = C.get_pde_family("Viscous Burgers")                                     # an alias, any case
    assert isinstance(fam, C.PDEFamily) and fam.kind == "burgers" and fam.default_params["nu"] > 0
    assert C.get_pde_family("not a pde") is None
    ranked = C.identify_pde("heat equation in a rod")
    assert ranked[0][0] == "heat_equation" and [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_suggest_problem_spec_uses_the_best_family_or_falls_back():
    s = C.suggest_problem_spec("viscous burgers shock")
    assert s["kind"] == "burgers" and s["coords"] == ["x", "t"] and s["fields"] == ["u"] and s["confidence"] > 0
    none = C.suggest_problem_spec("qwxz")
    assert none["kind"] == "custom" and none["confidence"] == 0.0
