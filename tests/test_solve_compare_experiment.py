"""pp.metrics, pp.solve, pp.compare, pp.Experiment, the Burgers closed form, and condition sampling."""
import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics import metrics
from pinneapple_physics.closed_form.burgers import burgers_sine_exact
from pinneapple_physics.pde_environment.condition_sampling import sample_condition_points
from pinneapple_physics.solving import MethodNotAvailable, Solution

NU = 0.01 / np.pi
SMALL_PINN = {"epochs": 30, "n_collocation": 256, "hidden_dim": 16, "n_layers": 2}


# --------------------------------------------------------------------------- metrics
def test_metrics_known_values_and_per_field_convention():
    true = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    pred = true + np.array([[0.1, 1.0], [0.0, 0.0], [-0.1, -1.0]])
    rel = metrics.relative_l2(pred, true)
    assert rel == pytest.approx([np.sqrt(0.02) / np.sqrt(14), np.sqrt(2) / np.sqrt(140000)])
    assert metrics.rmse(pred, true) == pytest.approx([np.sqrt(0.02 / 3), np.sqrt(2 / 3)])
    assert metrics.max_abs(pred, true) == pytest.approx([0.1, 1.0])
    assert metrics.mae(pred, true) == pytest.approx([0.2 / 3, 2 / 3])
    assert metrics.r2(true, true) == pytest.approx([1.0, 1.0])
    assert metrics.per_field(metrics.max_abs, pred, true, ("u", "p")) == pytest.approx({"u": 0.1, "p": 1.0})
    s = metrics.summary(pred, true, ("u", "p"))
    assert set(s) == {"relative_l2", "rmse", "max_abs"} and set(s["rmse"]) == {"u", "p"}


def test_metrics_edge_cases():
    assert np.isnan(metrics.relative_l2([1.0, 1.0], [0.0, 0.0]))       # undefined, not huge or zero
    assert metrics.relative_l2([1.0, 2.0], [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        metrics.rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        metrics.rmse([], [])
    with pytest.raises(KeyError):
        metrics.summary([1.0], [1.0], ("u",), names=("bogus",))
    torch = pytest.importorskip("torch")
    assert metrics.rmse(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])) == pytest.approx(np.sqrt(2))


# --------------------------------------------------------------------------- Burgers closed form
def test_burgers_closed_form_satisfies_the_pde_ic_and_bc():
    assert burgers_sine_exact(np.linspace(-1, 1, 9), 0.0, NU) == pytest.approx(-np.sin(np.pi * np.linspace(-1, 1, 9)))
    assert burgers_sine_exact(np.array([-1.0, 1.0]), np.array([0.5, 0.5]), NU) == pytest.approx([0.0, 0.0], abs=1e-12)
    X, T = np.meshgrid(np.linspace(-0.9, 0.9, 61), np.linspace(0.1, 0.9, 21), indexing="ij")
    h = 1e-4
    u = burgers_sine_exact(X, T, NU)
    ut = (burgers_sine_exact(X, T + h, NU) - burgers_sine_exact(X, T - h, NU)) / (2 * h)
    ux = (burgers_sine_exact(X + h, T, NU) - burgers_sine_exact(X - h, T, NU)) / (2 * h)
    uxx = (burgers_sine_exact(X + h, T, NU) - 2 * u + burgers_sine_exact(X - h, T, NU)) / h ** 2
    assert np.sqrt(np.mean((ut + u * ux - NU * uxx) ** 2)) < 1e-3
    assert np.isfinite(burgers_sine_exact(0.0, 1.0, 1e-3))          # no overflow at small viscosity


# --------------------------------------------------------------------------- solve
def test_solve_analytic_exact_and_custom_methods():
    sol = pp.solve("burgers_1d", "analytic", preset_kwargs={"nu": NU})
    assert sol.predict([[0.5, 0.75]])[0, 0] == pytest.approx(burgers_sine_exact(0.5, 0.75, NU))
    ex = pp.solve("burgers_1d", "exact", fn=lambda X: np.zeros((len(X), 1)))
    assert ex.predict(np.zeros((3, 2))).shape == (3, 1)

    def constant(problem, value=1.0):
        return Solution(problem, "constant", lambda X: np.full((len(X), 1), value), 0.0)

    assert pp.solve("burgers_1d", constant, value=2.0).predict([[0.0, 0.0]])[0, 0] == 2.0
    with pytest.raises(KeyError, match="unknown method"):
        pp.solve("burgers_1d", "nope")
    with pytest.raises(ValueError, match="x must have shape"):
        sol.predict(np.zeros((2, 3)))


def test_solve_refuses_an_inconsistent_problem():
    with pytest.raises(ValueError, match="not consistent"):
        pp.solve("industrial_furnace_thermal", "pinn")


def test_reference_method_refuses_a_partial_grid():
    """The preset's FDM path returns a grid over x only (time dropped); comparing it would be wrong."""
    with pytest.raises(MethodNotAvailable, match="partial grid"):
        pp.solve("burgers_1d", "reference")


def test_analytic_is_unavailable_when_no_closed_form_matches():
    with pytest.raises(MethodNotAvailable, match="no closed-form"):
        pp.solve("laplace_2d", "analytic")


# --------------------------------------------------------------------------- compare
def test_compare_scores_methods_on_the_same_points_and_reports_failures():
    def broken(problem):
        raise RuntimeError("solver crashed")

    exact = lambda X: burgers_sine_exact(X[:, 0], X[:, 1], NU)[:, None]  # noqa: E731
    c = pp.compare("burgers_1d", ["exact", broken], reference="analytic", options={"exact": {"fn": exact}},
                   preset_kwargs={"nu": NU}, n_points=500)
    ok, failed = c.rows
    assert ok["status"] == "ok" and ok["metrics"]["relative_l2"]["u"] < 1e-12
    assert failed["status"] == "failed" and "solver crashed" in failed["error"]
    assert c.best() == "exact" and c.n_points == 500
    assert "relL2(u)" in str(c) and c.to_dict()["fingerprint"]


# --------------------------------------------------------------------------- boundary conditions are enforced
def test_condition_sampling_finds_points_on_faces():
    spec = pp.get_preset("burgers_1d")
    rng = np.random.default_rng(0)
    for cond in spec.conditions:
        pts = sample_condition_points(lambda X, c=cond: c.mask(X, {}), spec.domain_bounds, spec.coords, 128, rng)
        assert len(pts) == 128, cond.name
    from pinneapple_simulation.numerical_solvers.problem_runner import _sample_callable_condition
    ic = spec.conditions[0]
    pts = _sample_callable_condition(ic.selector, spec.domain_bounds, spec.coords, 64, rng, {})
    assert np.all(pts[:, 1] == 0.0) and pts[:, 0].std() > 0.3    # on t = 0, spread in x (not the origin)


def test_solve_pde_now_enforces_the_initial_condition():
    """Before the fix, every boundary/initial condition selected zero interior points and was dropped, so the
    network learned u ~ 0. A short run must already follow u(x, 0) = -sin(pi x)."""
    sol = pp.solve("burgers_1d", "pinn", preset_kwargs={"nu": NU}, epochs=300, n_collocation=1024)
    x = np.linspace(-0.8, 0.8, 9)
    pred = sol.predict(np.c_[x, np.zeros_like(x)])[:, 0]
    assert np.max(np.abs(pred + np.sin(np.pi * x))) < 0.15


# --------------------------------------------------------------------------- Experiment
def test_experiment_is_reproducible_and_records_provenance(tmp_path):
    exp = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), reference="analytic", seed=3, n_eval=256)
    exp.problem = exp.problem.with_parameters(nu=NU)
    r1, r2 = exp.run(), exp.run()
    pts = np.random.default_rng(0).random((50, 2)) * [2, 1] - [1, 0]
    assert np.array_equal(r1.solution.predict(pts), r2.solution.predict(pts))
    assert r1.metrics == r2.metrics and set(r1.metrics) == {"relative_l2", "rmse", "max_abs"}
    assert r1.problem_fingerprint == exp.problem.fingerprint() and r1.environment["pinneapple"]
    other = pp.Experiment(exp.problem, method="pinn", options=dict(SMALL_PINN), seed=4).run()
    assert not np.array_equal(other.solution.predict(pts), r1.solution.predict(pts))
    path = r1.save(str(tmp_path / "run"))
    record = pp.ExperimentResult.load_record(str(tmp_path / "run"))
    assert path.endswith("result.json") and (tmp_path / "run" / "model.pt").exists()
    assert record["config"]["seed"] == 3 and record["metrics"] == r1.metrics


def test_front_door_exports():
    for name in ("solve", "compare", "Experiment", "ExperimentResult", "Solution", "metrics", "list_methods"):
        assert hasattr(pp, name), name
    assert {"pinn", "analytic", "exact", "reference"} <= set(pp.list_methods())
