"""Tests of the neutral parts of the cross-framework benchmark (pinneapple_tools.benchmark_suite.crossframework) and
smoke tests of its runners (benchmarks/vs_frameworks).

Always run: references against independent computations, the finite-difference residual of the exact solutions, the
evaluator and statistics against hand-computed values, the point sets and shared weights. Runner smoke tests need the
framework installed (``pytest.importorskip``) and train a few iterations; the Darcy / Allen-Cahn ones are behind
``PINNEAPPLE_SLOW=1`` (data generation or a fine spectral reference).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_tools.benchmark_suite.crossframework import evaluate as E
from pinneapple_tools.benchmark_suite.crossframework import problems as P
from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework import weights as W

ROOT = Path(__file__).resolve().parents[1]
SLOW = pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="set PINNEAPPLE_SLOW=1")


# --------------------------------------------------------------------------- references
def _burgers_adaptive(x, t, nu):
    """Cole-Hopf by adaptive quadrature in the original variable (independent of the Gauss-Hermite reference)."""
    from scipy.integrate import quad
    s = math.sqrt(4 * nu * t)
    lim = 12 * s
    es = np.linspace(-lim, lim, 4001)
    m = np.max(-np.cos(np.pi * (x - es)) / (2 * np.pi * nu) - es ** 2 / (4 * nu * t))

    def w(e):
        return math.exp(-math.cos(math.pi * (x - e)) / (2 * math.pi * nu) - e * e / (4 * nu * t) - m)
    num = quad(lambda e: math.sin(math.pi * (x - e)) * w(e), -lim, lim, limit=400, points=[0.0])[0]
    den = quad(w, -lim, lim, limit=400, points=[0.0])[0]
    return -num / den


def test_burgers_reference_matches_adaptive_quadrature():
    p = P.get_problem("burgers")
    pts = [(0.01, 0.99), (-0.005, 0.8), (0.5, 0.5), (0.002, 0.4), (0.9, 0.1), (0.0005, 0.99)]
    X = np.array(pts)
    got = p.exact(X)
    want = np.array([_burgers_adaptive(x, t, p.params["nu"]) for x, t in pts])
    assert np.max(np.abs(got - want)) < 1e-10


def test_burgers_reference_quadrature_converged():
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact
    p = P.get_problem("burgers")
    X = P.test_points(p)
    a = burgers_sine_exact(X[:, 0], X[:, 1], p.params["nu"], n_nodes=120)
    b = burgers_sine_exact(X[:, 0], X[:, 1], p.params["nu"], n_nodes=250)
    assert np.max(np.abs(a - b)) < 1e-12


def test_allen_cahn_reference_fast_checks():
    """Coarse grid (fast): the initial state, the bounds of the solution and dt convergence."""
    x, t, u = P.allen_cahn_reference(n=256, dt=1e-3, n_save=100)
    assert np.allclose(u[0], x ** 2 * np.cos(np.pi * x), atol=1e-12)
    assert u.min() >= -1.0 - 1e-6 and u.max() <= 1.0 + 1e-6          # u = +-1 are the stable states
    _, _, u2 = P.allen_cahn_reference(n=256, dt=5e-4, n_save=100)
    assert np.max(np.abs(u - u2)) < 1e-6                             # ETDRK4: fourth order in time


@SLOW
def test_allen_cahn_reference_resolution():
    """The reference used by the benchmark (2048 modes) against 4096 modes, and the interpolation in t."""
    _, _, a = P.allen_cahn_reference()
    _, _, b = P.allen_cahn_reference(n=4096)
    assert np.max(np.abs(b[:, ::2] - a)) < 3e-4
    mid = 0.5 * (a[:-2:2] + a[2::2])
    assert np.max(np.abs(mid - a[1:-1:2])) < 1e-5


@pytest.mark.parametrize("name", ["burgers", "poisson", "helmholtz"])
def test_fd_residual_of_exact_solution_is_small(name):
    p = P.get_problem(name)
    X = P.residual_points(p, n=400)
    vals = p.exact(P.stencil(p, X).reshape(-1, 2))
    r = P.residual_from_stencil(p, X, vals)
    scale = {"burgers": 1.0, "poisson": 2 * np.pi ** 2, "helmholtz": 17 * np.pi ** 2}[name]
    # second-order differences: the truncation error h^2 / 12 * u'''' relative to the size of the terms. For Burgers
    # the front near x = 0 (width ~ nu) is not resolved by h = 1e-3: the exact solution itself has an RMS residual of
    # about 0.027 there, the floor of this metric (reported next to the measured residuals).
    assert np.sqrt(np.mean(r ** 2)) / scale < {"burgers": 4e-2, "poisson": 1e-4, "helmholtz": 2e-3}[name]


def test_fd_residual_detects_a_wrong_solution():
    p = P.get_problem("poisson")
    X = P.residual_points(p, n=400)
    vals = 1.1 * p.exact(P.stencil(p, X).reshape(-1, 2))
    r = P.residual_from_stencil(p, X, vals)
    assert np.sqrt(np.mean(r ** 2)) > 1.0


def test_sources_match_the_manufactured_solutions():
    """f = laplacian(u) (Poisson) and q = laplacian(u) + k^2 u (Helmholtz), by finite differences of exact()."""
    for name in ("poisson", "helmholtz"):
        p = P.get_problem(name)
        X = P.residual_points(p, n=200)
        h = 1e-4
        u = p.exact
        lap = sum((u(X + h * e) - 2 * u(X) + u(X - h * e)) / h ** 2 for e in (np.array([1, 0]), np.array([0, 1])))
        k2 = p.params.get("k", 0.0) ** 2 if name == "helmholtz" else 0.0
        assert np.max(np.abs(lap + k2 * u(X) - p.source(X))) < 1e-3 * np.max(np.abs(p.source(X)))


# --------------------------------------------------------------------------- point sets and weights
def test_point_set_is_reproducible_and_on_the_faces(tmp_path):
    p = P.get_problem("burgers")
    a = P.make_point_set(p, n_interior=100, n_boundary=40, seed=1)
    b = P.make_point_set(p, n_interior=100, n_boundary=40, seed=1)
    assert P.point_set_digest(a) == P.point_set_digest(b)
    assert np.all(np.isin(a["X_b"][:, 0], [-1.0, 1.0])) and np.all(a["X_0"][:, 1] == 0.0)
    assert np.allclose(a["u_0"][:, 0], -np.sin(np.pi * a["X_0"][:, 0]), atol=1e-6)
    path = tmp_path / "p.npz"
    assert P.save_point_set(a, path) == P.point_set_digest(P.load_point_set(path))
    q = P.make_point_set(P.get_problem("helmholtz"), n_interior=10, n_boundary=40, seed=1)
    on_edge = np.isclose(np.abs(q["X_b"]), 1.0).any(axis=1)
    assert on_edge.all() and len(q["X_b"]) == 40


def test_shared_xavier_weights():
    sizes = W.layer_sizes()
    w0 = W.xavier_uniform_init(sizes, 0)
    assert [w.shape for w, _ in w0] == [(50, 2), (50, 50), (50, 50), (50, 50), (1, 50)]
    for (w, b), fi, fo in zip(w0, sizes[:-1], sizes[1:], strict=False):
        assert np.abs(w).max() <= math.sqrt(6 / (fi + fo)) and not b.any()
    w1 = W.xavier_uniform_init(sizes, 0)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(w0, w1, strict=False))
    torch = pytest.importorskip("torch")
    lins = [torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=False)]
    W.load_into_linears(lins, w0)
    assert np.array_equal(lins[0].weight.detach().numpy(), w0[0][0])


# --------------------------------------------------------------------------- evaluator and statistics
def test_evaluator_metrics_by_hand():
    p = P.get_problem("poisson")
    X = P.test_points(p)
    ref = p.exact(X)
    m = E.evaluate_pinn(p, ref)
    assert m["rel_l2"] == 0.0 and m["max_abs"] == 0.0
    m = E.evaluate_pinn(p, 1.02 * ref)
    assert abs(m["rel_l2"] - 0.02) < 1e-12 and abs(m["max_abs"] - 0.02 * np.abs(ref).max()) < 1e-12
    bad = E.evaluate_pinn(p, np.full(len(X), np.nan))
    assert bad["finite"] is False and math.isnan(bad["rel_l2"])
    with pytest.raises(ValueError):
        E.evaluate_pinn(p, ref[:10])


def test_evaluator_residual_uses_the_stencil_export():
    p = P.get_problem("helmholtz")
    exp = proto.export_predictions(p, lambda Y: p.exact(np.asarray(Y, dtype=np.float64)))
    m = E.evaluate_pinn(p, exp["u_test"], exp["u_stencil"])
    assert m["rel_l2"] < 1e-6 and m["residual_rms"] < 0.5 and m["n_residual"] == 2000


def test_operator_and_inverse_metrics():
    ref = np.ones((3, 4, 4))
    pred = ref.copy()
    pred[0] *= 1.1
    m = E.evaluate_operator(pred, ref)
    assert abs(m["rel_l2_mean"] - 0.1 / 3) < 1e-12 and abs(m["rel_l2_max"] - 0.1) < 1e-12
    inv = E.evaluate_inverse(0.0033, 0.003, np.ones(5), np.ones(5))
    assert abs(inv["nu_rel_err"] - 0.1) < 1e-12 and inv["rel_l2"] == 0.0


def test_seed_statistics_and_sign_test():
    s = E.seed_stats([3.0, 1.0, float("nan"), 2.0])
    assert s == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 4, "n_failed": 1}
    assert E.sign_test_p(5, 5) == 0.0625 and E.sign_test_p(0, 5) == 0.0625
    assert abs(E.sign_test_p(3, 5) - 1.0) < 1e-12
    r = E.paired_ratio({0: 1.0, 1: 2.0, 2: 3.0}, {0: 2.0, 1: 2.0, 2: 1.0})
    assert r["n"] == 3 and r["median"] == 1.0 and r["a_better"] == 1


def test_inverse_observations():
    from pinneapple_tools.benchmark_suite.crossframework import inverse as INV
    o = INV.observations(n=500)
    noise = o["u"] - o["u_clean"]
    assert o["X"].shape == (500, 2) and abs(float(noise.std()) / float(o["sigma"]) - 1) < 0.15
    assert abs(float(o["sigma"]) - 0.01 * float(o["u_clean"].std())) < 1e-6


# --------------------------------------------------------------------------- PINNeAPPle change made for track A1
def test_solve_pde_fixed_collocation_points():
    """solve_pde(x_col=...) trains on the given points every step (identical runs give identical losses), and the
    default (fresh samples) is unchanged."""
    torch = pytest.importorskip("torch")
    from benchmarks.vs_frameworks.track_a_pinneapple import builder_problem
    from pinneapple_physics import solve_pde
    from pinneapple_physics.pinn_solver.compiler.loss import LossWeights

    p = P.get_problem("poisson")
    pts = P.make_point_set(p, n_interior=64, n_boundary=40, seed=3)
    spec = builder_problem(p, weight=1.0, single_bc=True)
    ctx = {"source_fn": lambda X, ctx=None: p.source(np.asarray(X, dtype=np.float64))[:, None]}

    def run(**kw):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        out = solve_pde(spec, net, epochs=5, lr=1e-3, seed=0, weights=LossWeights(1, 1, 1, 1), ctx=ctx,
                        x_bc=torch.as_tensor(pts["X_b"]), y_bc=torch.as_tensor(pts["u_b"]), **kw)
        return out["history"]["loss"]

    a, b = run(x_col=pts["X_f"]), run(x_col=pts["X_f"])
    assert a == b
    # the first loss is the plain mean squared residual + boundary error on those points
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    X = torch.tensor(pts["X_f"], requires_grad=True)
    u = net(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i] for i in range(2))
    r = lap - torch.as_tensor(p.source(pts["X_f"].astype(np.float64)), dtype=torch.float32)
    want = float((r ** 2).mean() + ((net(torch.as_tensor(pts["X_b"])) - torch.as_tensor(pts["u_b"])) ** 2).mean())
    assert abs(a[0] - want) < 1e-4 * max(1.0, abs(want))
    c = run()                                                         # default: fresh points each step
    assert c != a
    with pytest.raises(ValueError):
        run(x_col=pts["X_f"][:, :1])


def test_solve_pde_evaluates_the_collocation_points_once_per_step():
    """solve_pde used to compute ``y_hat = model(x_col)`` every step, which the compiled loss never reads (it
    evaluates the model on its own gradient-tracking copy): one wasted forward pass per step."""
    torch = pytest.importorskip("torch")
    from benchmarks.vs_frameworks.track_a_pinneapple import builder_problem
    from pinneapple_physics import solve_pde

    p = P.get_problem("poisson")
    pts = P.make_point_set(p, n_interior=50, n_boundary=40, seed=3)
    spec = builder_problem(p, weight=1.0, single_bc=True)
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    calls = []
    net.register_forward_hook(lambda m, inp, out: calls.append(inp[0].shape[0]))
    solve_pde(spec, net, epochs=4, lr=1e-3, x_col=pts["X_f"], x_bc=torch.as_tensor(pts["X_b"]),
              y_bc=torch.as_tensor(pts["u_b"]),
              ctx={"source_fn": lambda X, ctx=None: p.source(np.asarray(X, dtype=np.float64))[:, None]})
    assert calls.count(50) == 4                      # one collocation forward per step (was 8)


# --------------------------------------------------------------------------- runner smoke tests
def _run_runner(module, tmp_path, *extra):
    cmd = [sys.executable, "-m", f"benchmarks.vs_frameworks.{module}", "--root", str(tmp_path), "--out",
           str(tmp_path / "runs"), "--run-id", "smoke", *extra]
    env = dict(os.environ, DDE_BACKEND="pytorch", OMP_NUM_THREADS="1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, timeout=1200)
    meta = json.loads((tmp_path / "runs" / "smoke.json").read_text())
    assert meta["status"] == "ok", meta.get("error")
    return meta, dict(np.load(tmp_path / "runs" / "smoke.npz"))


@pytest.mark.parametrize("module,pkg", [("track_a_pinneapple", "torch"), ("track_a_deepxde", "deepxde"),
                                        ("track_a_physicsnemo_sym", "physicsnemo.sym")])
def test_track_a1_runner_smoke(module, pkg, tmp_path):
    pytest.importorskip(pkg)
    meta, arr = _run_runner(module, tmp_path, "--problem", "poisson", "--mode", "a1", "--iters", "3")
    assert meta["iterations"] == 3 and meta["n_parameters"] == 7851
    m = E.evaluate_pinn(P.get_problem("poisson"), arr["u_test"], arr["u_stencil"])
    assert m["finite"]


@SLOW
@pytest.mark.parametrize("module,pkg", [("track_a_pinneapple", "torch"), ("track_a_deepxde", "deepxde"),
                                        ("track_a_physicsnemo_sym", "physicsnemo.sym")])
def test_track_a2_runner_smoke(module, pkg, tmp_path):
    pytest.importorskip(pkg)
    meta, _ = _run_runner(module, tmp_path, "--problem", "burgers", "--mode", "a2", "--cap-s", "5")
    assert meta["iterations"] > 0 and meta["cpu_s"] < 60


@pytest.mark.parametrize("module,pkg", [("track_c_pinneapple", "torch"), ("track_c_deepxde", "deepxde"),
                                        ("track_c_physicsnemo_sym", "physicsnemo.sym")])
def test_track_c_runner_smoke(module, pkg, tmp_path):
    pytest.importorskip(pkg)
    meta, arr = _run_runner(module, tmp_path, "--problem", "burgers_inverse", "--mode", "c", "--iters", "3")
    assert np.isfinite(float(arr["nu_hat"])) and arr["u_test"].shape == (25600,)


@SLOW
@pytest.mark.parametrize("module,pkg", [("track_b_pinneapple", "torch"), ("track_b_physicsnemo", "physicsnemo"),
                                        ("track_b_neuraloperator", "neuralop")])
def test_track_b_runner_smoke(module, pkg, tmp_path):
    pytest.importorskip(pkg)
    meta, arr = _run_runner(module, tmp_path, "--problem", "darcy", "--mode", "b", "--epochs", "1",
                            "--n-train", "40")
    assert arr["pred_test"].shape == (200, 64, 64) and 2_300_000 < meta["n_parameters_real"] < 2_400_000
