"""From an inverse model to an equation (#400): the law of a model that implements one is read back with the
coefficients right; the identifiability table keeps the terms the replicates keep and drops the noise term; a
coefficient interval covers the fits that *had* that term; the front of complexity against error is monotone; an
optional back end that is not installed says so and the others carry on; and a heat equation read out of a surface
fitted on a short time window is integrated past that window, where the surface itself diverges.

Every number asserted here was measured by running the code; the guide ``docs/guides/symbolic_distillation.md``
carries the same numbers with their units and how they were measured.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from pinneapple_analysis.inverse_problems.symbolic_distillation import (
    Dataset,
    DistillationConfig,
    OptionalBackendMissing,
    PolynomialSurface,
    Support,
    distill,
    field_dataset,
    module_dataset,
    outside_support,
    simulate_evolution,
    verify,
)

# ---------------------------------------------------------------------------
# the models the equations are read out of
# ---------------------------------------------------------------------------


class _Oscillator(torch.nn.Module):
    """``f = -0.5 x + 2 y^2``: a law with one linear and one quadratic term, so the library also offers three terms
    the fit has to leave at zero."""

    def forward(self, X):
        x, y = X[:, 0], X[:, 1]
        return torch.stack([-0.5 * x + 2.0 * y * y], dim=1)


def _oscillator_dataset(n: int = 400, seed: int = 0) -> tuple[Dataset, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n, 2))
    data = module_dataset(_Oscillator(), points, names=["x", "y"], target_name="f", name="oscillator")
    return data, points


def _noisy_oscillator() -> tuple[Dataset, np.ndarray]:
    """The same model, with the target carrying noise that a spurious term can explain in part: the sample on which
    the identifiability and the front are measured (``x*x`` is the term that comes and goes)."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, size=(600, 2))
    data = module_dataset(_Oscillator(), points, names=["x", "y"], target_name="f", name="oscillator")
    data.target = data.target + 0.12 * np.std(data.target) * rng.standard_normal(data.n) * (1 + points[:, 0] * 2)
    return data, points


def _sindy(n_bootstrap: int = 40, **options) -> DistillationConfig:
    return DistillationConfig(backends=("sindy",), n_bootstrap=n_bootstrap, poly_order=2, lhs="f", **options)


# ---------------------------------------------------------------------------
# the law of a model, read back
# ---------------------------------------------------------------------------


def test_the_right_hand_side_of_a_model_is_recovered_with_its_coefficients():
    data, _points = _oscillator_dataset()
    report = distill(data, config=_sindy(30))

    assert report.equation() == "f = -0.5*(x) + 2*(y*y)"
    assert [row["term"] for row in report.terms] == ["x", "y*y"]
    assert report.terms[0]["coefficient"] == pytest.approx(-0.5, abs=1e-9)
    assert report.terms[1]["coefficient"] == pytest.approx(2.0, abs=1e-9)
    assert report.held_out_error < 1e-10                   # the model implements the law exactly
    assert report.fit_error < 1e-10
    assert report.n_samples == 400 and report.backend == "sindy"
    assert all(check["status"] == "pass" for check in report.checks)
    # the equation as SymPy, as LaTeX, as Python: the same law in the three forms the report hands out
    assert report.sympy_expression == "-0.5*x + 2*y**2"
    assert report.latex == r"f = - 0.5 x + 2 y^{2}"
    namespace: dict = {}
    exec(report.code, namespace)                                        # noqa: S102 - the code the report wrote
    x = np.linspace(-1.0, 1.0, 7)
    y = np.linspace(1.0, -1.0, 7)
    assert np.allclose(namespace["f"](x, y), -0.5 * x + 2.0 * y * y)
    # evaluating the report on named columns gives the model's own values back
    features = {"x": x, "y": y}
    assert np.allclose(report.predict(features), -0.5 * x + 2.0 * y * y)
    assert set(report.support["bounds"]) == {"x", "y"}


def test_the_identifiability_table_keeps_the_law_and_drops_the_noise_term():
    data, _points = _noisy_oscillator()
    report = distill(data, config=_sindy(60))

    table = report.identifiability["terms"]
    assert set(table) == {"x", "y*y", "x*x"}
    assert report.identifiability["stable"] == ["x", "y*y"]
    assert report.identifiability["unstable"] == ["x*x"]
    assert table["x"]["inclusion"] == 1.0 and table["y*y"]["inclusion"] == 1.0
    assert table["x*x"]["inclusion"] < report.identifiability["threshold"]
    assert report.identifiability["n_replicates"] == 60
    assert "x*x did not survive 80% of the replicates" in report.identifiability["reason"]
    # the unstable term is not in the equation, and the report says where it went
    assert [row["term"] for row in report.terms] == ["x", "y*y"]
    assert report.terms[0]["coefficient"] == pytest.approx(-0.5, abs=0.02)
    assert report.terms[1]["coefficient"] == pytest.approx(2.0, abs=0.02)
    assert any("x*x did not survive the replicates" in note for note in report.notes)
    # the term the law really has keeps an interval that holds its fitted value
    for label, row in (("x", report.terms[0]), ("y*y", report.terms[1])):
        low, high = table[label]["interval95"]
        assert low < row["coefficient"] < high


def test_a_coefficient_interval_covers_the_fits_that_had_the_term():
    """The interval is over the fits that *included* the term: a replicate that dropped it is not a coefficient of
    zero, and mixing those zeros in makes an interval that spans 0 while every fit that had the term agrees on its
    sign -- the two statements cannot both be true (this is the defect the interval used to have)."""
    data, _points = _noisy_oscillator()
    report = distill(data, config=_sindy(60))
    entry = report.identifiability["terms"]["x*x"]

    assert 0.0 < entry["inclusion"] < 0.98                 # present in some fits, dropped in others
    assert entry["same_sign"] is True
    low, high = entry["interval95"]
    assert low < entry["coefficient"] < high
    assert low * high > 0.0                                # one sign only: the zero of a dropped fit is not in it


def test_the_front_of_complexity_against_error_is_monotone():
    """A front is one point per complexity, with the error falling as the model grows: a longer model that fits worse
    is dominated and does not belong on it, and neither does the same model reported twice."""
    data, _points = _noisy_oscillator()
    report = distill(data, config=_sindy(30))
    front = report.front

    assert len(front) >= 2
    complexities = [entry["complexity"] for entry in front]
    errors = [entry["relative_error"] for entry in front]
    assert complexities == sorted(set(complexities))       # one point per complexity, ascending
    assert all(later < earlier for earlier, later in list(zip(errors, errors[1:], strict=False)))
    for entry in front:
        assert sum(1 for other in front
                   if other["complexity"] <= entry["complexity"]
                   and other["relative_error"] <= entry["relative_error"]) >= 1     # nothing dominates it
    assert front[0]["complexity"] == 2 and set(front[0]["expression"].split(" + ")) == {"-0.496656*(x)",
                                                                                      "2.00074*(y*y)"}


def test_a_model_with_nothing_to_trade_has_a_front_of_one_point():
    data, _points = _oscillator_dataset()
    report = distill(data, config=_sindy(20))
    assert len(report.front) == 1
    assert report.front[0]["complexity"] == 2
    assert report.front[0]["relative_error"] < 1e-4


# ---------------------------------------------------------------------------
# the back ends
# ---------------------------------------------------------------------------


def test_an_optional_back_end_that_is_missing_says_how_to_get_it(monkeypatch):
    """PySR and Julia are an extra, never a requirement: absent, the arm is recorded with the fix in its message and
    the arms that need nothing still deliver the law."""
    data, _points = _oscillator_dataset(200)
    monkeypatch.setitem(sys.modules, "pysr", None)          # ``import pysr`` now raises ImportError

    with pytest.raises(OptionalBackendMissing) as alone:
        distill(data, config=DistillationConfig(backends=("pysr",), n_bootstrap=0))
    assert "pip install pysr" in str(alone.value) and "no back end could be run" in str(alone.value)

    report = distill(data, config=DistillationConfig(backends=("sindy", "pysr"), n_bootstrap=10, poly_order=2,
                                                    lhs="f"))
    assert report.backend == "sindy"
    assert "pip install pysr" in report.unavailable["backend:pysr"]
    assert any("the 'pysr' back end was not run" in note for note in report.notes)
    assert [row["term"] for row in report.terms] == ["x", "y*y"]


# ---------------------------------------------------------------------------
# verification: the equation against the model that produced it
# ---------------------------------------------------------------------------


def test_an_out_of_distribution_case_with_an_exact_network_is_still_graded():
    """A network that reproduces the reference exactly has an error of 0.0, which is a value and not a missing one:
    the out-of-distribution case it belongs to has to be graded, and the verdict said so (it used to be demoted to
    the held-out rows because 0.0 is falsy)."""
    data, points = _noisy_oscillator()
    report = distill(data, config=_sindy(10))
    case = {"name": "the network's own values", "kind": "out_of_distribution",
            "features": {"x": points[:, 0], "y": points[:, 1]}, "target": data.target, "reference": data.target}
    report = verify(report, [case])

    row = report.verification["cases"][0]
    assert row["network_error"] == 0.0 and row["winner"] == "network"
    assert report.verification["reference"] == "out_of_distribution"
    assert report.verification["verdict"] == "network_wins"
    assert "out of distribution" in report.verification["detail"]
    check = next(check for check in report.checks if check["code"] == "distill.verification")
    assert check["status"] == "fail"                        # the equation loses to a network that is the truth


def test_an_ensemble_member_is_fitted_and_a_different_law_is_not_stable():
    """The replicates of the genetic-programming arm are the resamples *and* the ensemble members, each fitted: a
    member that found another law is evidence against the equation, and a member that found the same one is evidence
    for it (the member used to be recorded as the string "0" instead of being fitted)."""

    class _Square(torch.nn.Module):
        def forward(self, X):
            x = X[:, 0]
            return torch.stack([0.5 * x * x], dim=1)

    class _Line(torch.nn.Module):
        def forward(self, X):
            x = X[:, 0]
            return torch.stack([0.5 * x], dim=1)

    rng = np.random.default_rng(5)
    points = rng.uniform(-1.0, 1.0, size=(60, 1))
    data = module_dataset(_Square(), points, names=["x"], target_name="f")
    other = module_dataset(_Line(), points, names=["x"], target_name="f")
    config = DistillationConfig(backends=("gp",), n_bootstrap=0, population=60, generations=6, operators=(),
                                max_nodes=6, seed=0, lhs="f")

    agree = distill(data, config=config, ensemble=[Dataset(features=dict(data.features), target=data.target.copy())])
    disagree = distill(data, config=config, ensemble=[other])

    assert agree.equation() == "f = (0.5 * (x)**2)" and agree.backend == "gp"
    assert agree.held_out_error == 0.0                     # the law, on the rows the search did not see
    assert [(entry["complexity"], round(entry["relative_error"], 4)) for entry in agree.front] == [(1, 0.643),
                                                                                                   (4, 0.0)]
    assert agree.front[-1]["expression"] == agree.expression          # the model it reports is on its own front
    assert agree.identifiability["n_replicates"] == 1
    assert agree.identifiability["stable"] == [agree.expression]
    assert disagree.identifiability["n_replicates"] == 1
    assert disagree.identifiability["stable"] == []
    assert disagree.identifiability["unstable"] == [disagree.expression]
    assert disagree.identifiability["terms"][disagree.expression]["inclusion"] == 0.0


def test_without_a_replicate_the_report_says_so_instead_of_inventing_a_reason():
    """``n_bootstrap=0`` and no ensemble: no term has an inclusion probability, and the report says that rather than
    reporting the equation as unstable."""
    data, _points = _oscillator_dataset(120)
    report = distill(data, config=DistillationConfig(backends=("sindy",), n_bootstrap=0, poly_order=2, lhs="f"))

    assert report.identifiability["n_replicates"] == 0
    assert report.identifiability["terms"] == {} and report.identifiability["unstable"] == []
    assert "no replicate was run" in report.unavailable["identifiability"]
    check = next(check for check in report.checks if check["code"] == "distill.stable_terms")
    assert check["status"] == "fail" and "no replicate was run" in check["detail"]


# ---------------------------------------------------------------------------
# units: what a term is, and what a coefficient would have to be
# ---------------------------------------------------------------------------


def test_the_units_constraint_keeps_the_term_whose_dimension_matches_the_left_hand_side():
    """``u_t = -1.5 u_x`` with ``u`` dimensionless and ``u_x`` a rate: only a term of unit 1/s can be multiplied by
    a dimensionless coefficient, so ``1`` and ``u`` leave the library before the regression and the report names
    them and the unit their coefficient would need."""
    class _Rate(torch.nn.Module):
        def forward(self, X):
            return torch.stack([-1.5 * X[:, 1]], dim=1)

    rng = np.random.default_rng(2)
    points = rng.uniform(-1.0, 1.0, size=(300, 2))
    data = module_dataset(_Rate(), points, names=["u", "u_x"], target_name="f")
    report = distill(data, config=DistillationConfig(backends=("sindy",), n_bootstrap=10, poly_order=1, seed=0,
                                                     lhs="f", units={"u": "1", "u_x": "1/s"},
                                                     target_unit="1/s", dimensionless_coefficients=True))

    assert [row["term"] for row in report.terms] == ["u_x"]
    assert report.terms[0]["coefficient"] == pytest.approx(-1.5, abs=1e-9)
    assert report.terms[0]["unit"] == "1"                  # dimensionless, as the filter required
    note = next(note for note in report.notes if "units constraint" in note)
    assert "1: the term is 1, the left-hand side is 1/s" in note
    assert "u: the term is 1, the left-hand side is 1/s" in note


def test_a_function_term_the_expression_language_cannot_read_is_refused_with_the_reason():
    """A term like ``tan(u)`` is neither a quantity nor a variable: the units filter refuses it and says why, instead
    of reporting that "the unit of tan(u) is not declared" (a name that is not a variable) or crashing on it."""
    class _Rate(torch.nn.Module):
        def forward(self, X):
            return torch.stack([-1.5 * X[:, 1]], dim=1)

    rng = np.random.default_rng(2)
    points = rng.uniform(-1.0, 1.0, size=(200, 2))
    data = module_dataset(_Rate(), points, names=["u", "u_x"], target_name="f")
    report = distill(data, config=DistillationConfig(
        backends=("sindy",), n_bootstrap=5, poly_order=1, seed=0, lhs="f",
        units={"u": "1", "u_x": "1/s"}, target_unit="1/s",
        custom_terms=[("tan(x0)", lambda X: np.tan(X[:, 0])), ("u^2", lambda X: X[:, 0] ** 2)]))
    note = next(note for note in report.notes if "units constraint" in note)
    # ``^`` is not a power in the expression language and ``tan`` is not a function it has: both are refused by name
    assert "the term 'tan(u)' is not one the expression language reads" in note
    assert "the term 'u^2' is not one the expression language reads" in note
    assert "BitXor is not part of the expression language" in note
    assert [row["term"] for row in report.terms] == ["u_x"]


# ---------------------------------------------------------------------------
# the heat equation: read from a surface fitted on a short window, integrated past it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def heat_study():
    """``u_t = D u_xx`` distilled from a Chebyshev surface fitted to its solution on ``t`` in [0, 0.4], together with
    the analytic solution it has to reproduce on ``t`` in [0, 4], where the surface was never fitted."""
    diffusivity, nx, nt = 0.5, 41, 25
    x = np.linspace(0.0, 2 * np.pi, nx, endpoint=False)
    t_fit = np.linspace(0.0, 0.4, nt)
    X, T = np.meshgrid(x, t_fit, indexing="ij")
    u0 = np.sin(x) + 0.5 * np.sin(2 * x)
    U = np.exp(-diffusivity * T) * np.sin(X) + 0.5 * np.exp(-4 * diffusivity * T) * np.sin(2 * X)
    points = np.column_stack([X.ravel(), T.ravel()])
    surface = PolynomialSurface.fit(points, U.ravel(), degree=(14, 10))
    data = field_dataset(surface, points, coords=["x", "t"],
                         columns={"u": ("u", ()), "u_x": ("u", ("x",)), "u_xx": ("u", ("x", "x")),
                                  "u_t": ("u", ("t",))},
                         target="u_t", support=Support.from_reference(points, names=["x", "t"]), name="heat")
    report = distill(data, config=DistillationConfig(backends=("sindy",), n_bootstrap=30, poly_order=1, seed=0,
                                                     lhs="u_t", units={"u": "K", "u_x": "K/m", "u_xx": "K/m^2"},
                                                     target_unit="K/s"))
    return {"report": report, "surface": surface, "x": x, "u0": u0, "diffusivity": diffusivity,
            "residual": surface.residual(points, U.ravel())}


def _heat_exact(x: np.ndarray, t: np.ndarray, diffusivity: float) -> np.ndarray:
    xr, tr = np.meshgrid(x, t, indexing="ij")
    return np.exp(-diffusivity * tr) * np.sin(xr) + 0.5 * np.exp(-4 * diffusivity * tr) * np.sin(2 * xr)


def test_a_heat_law_is_read_out_of_a_surface_and_survives_where_the_surface_does_not(heat_study):
    report, surface, x = heat_study["report"], heat_study["surface"], heat_study["x"]
    diffusivity = heat_study["diffusivity"]
    assert heat_study["residual"] < 1e-5                   # the surface is a faithful reading inside the window

    assert report.equation() == "u_t = 0.499945*(u_xx)"
    assert [row["term"] for row in report.terms] == ["u_xx"]
    assert report.terms[0]["coefficient"] == pytest.approx(diffusivity, rel=2e-4)
    assert report.terms[0]["unit"] == "m^2/s"              # the unit the diffusivity of heat has
    assert report.identifiability["stable"] == ["u_xx"] and report.identifiability["unstable"] == []
    assert all(check["status"] == "pass" for check in report.checks)

    # the equation integrated from the initial condition over ten times the window it was read from
    t_out = np.linspace(0.0, 4.0, 81)
    exact = _heat_exact(x, t_out, diffusivity)
    scale = float(np.sqrt(np.mean(exact ** 2)))
    rolled = simulate_evolution(report.to_text("u"), x=x, t=t_out, u0=heat_study["u0"])
    equation_error = float(np.sqrt(np.mean((rolled - exact) ** 2)) / scale)
    grid = np.meshgrid(x, t_out, indexing="ij")
    extrapolated = surface.predict(np.column_stack([grid[0].ravel(), grid[1].ravel()])).reshape(exact.shape)
    network_error = float(np.sqrt(np.mean((extrapolated - exact) ** 2)) / scale)

    assert equation_error < 1e-3                           # the equation reproduces the held-out trajectory
    assert network_error > 1.0                             # the surface it was read from does not: it diverges
    assert equation_error < network_error / 1000

    # and that is the verdict verify() writes, the simulator replacing the pointwise evaluation
    xr, tr = np.meshgrid(x, t_out, indexing="ij")
    features = {"u": exact.ravel(),
                "u_x": (np.exp(-diffusivity * tr) * np.cos(xr)
                        + np.exp(-4 * diffusivity * tr) * np.cos(2 * xr)).ravel(),
                "u_xx": (-np.exp(-diffusivity * tr) * np.sin(xr)
                         - 2 * np.exp(-4 * diffusivity * tr) * np.sin(2 * xr)).ravel(),
                "u_t": (-diffusivity * np.exp(-diffusivity * tr) * np.sin(xr)
                        - 2 * diffusivity * np.exp(-4 * diffusivity * tr) * np.sin(2 * xr)).ravel()}
    verified = verify(report, [{
        "name": "time beyond the window the surface was fitted on", "kind": "out_of_distribution",
        "features": features, "target": extrapolated.ravel(), "reference": exact.ravel(),
        "simulator": lambda rep, _case: simulate_evolution(rep.to_text("u"), x=x, t=t_out,
                                                           u0=heat_study["u0"]).ravel()}])
    row = verified.verification["cases"][0]
    assert row["winner"] == "equation"
    assert row["equation_error"] == pytest.approx(equation_error)
    assert row["network_error"] == pytest.approx(network_error)
    assert verified.verification["verdict"] == "equation_wins"
    assert verified.verification["reference"] == "out_of_distribution"
    check = next(check for check in verified.checks if check["code"] == "distill.verification")
    assert check["status"] == "pass"
    assert "verification" not in verified.unavailable


def test_the_equation_goes_back_to_the_solver_and_to_the_pinn(heat_study):
    report = heat_study["report"]
    assert report.to_text("u") == "u_t = 0.499945*(u_xx)"
    system = report.to_system()
    assert system.pdes[0].text == "u_t = 0.499945*(u_xx)"
    assert list(system.coords) == ["x", "t"] and list(system.fields) == ["u"]
    with pytest.raises(ValueError, match="not one of the equation's features"):
        report.to_text("v")
    # the record round-trips through JSON without losing the equation
    restored = type(report).from_dict(report.to_dict())
    assert restored.equation() == report.equation()
    assert restored.summary().startswith(report.equation())


# ---------------------------------------------------------------------------
# where the model may be asked, and what it is asked through
# ---------------------------------------------------------------------------


def test_a_query_outside_the_support_is_refused_before_it_reaches_the_fit():
    rng = np.random.default_rng(4)
    points = rng.uniform(-1.0, 1.0, size=(200, 2))
    support = Support.from_reference(points, names=["x", "y"])
    queried = np.vstack([points, [[9.0, 9.0]] * 5])
    data = field_dataset(_Oscillator(), queried, coords=["x", "y"], columns={"f": ("f", ())}, target="f",
                         support=support)
    assert data.refused == 5 and data.n == 200
    assert any("were inside the support" in note for note in data.notes)

    far = np.array([[9.0, 9.0], [0.0, 0.0]])
    assert not support.contains(far)[0] and support.contains(far)[1]
    assert support.inside_box(far).tolist() == [False, True]     # the box alone would have let the far query in
    assert outside_support(far, points, names=["x", "y"]).tolist() == [True, False]
    assert set(Support.from_dict(support.to_dict()).bounds) == {"x", "y"}


def test_a_learned_term_network_is_read_through_the_same_dataset():
    """A :class:`~pinneapple_analysis.inverse_problems.missing_term.NeuralTermDiscovery` answers ``predict_term`` on
    plain coordinates: that is a model this module reads directly, so its learned term is distilled by the same
    pipeline (and a plain numpy callable is read the same way)."""

    class _Term:
        """What ``NeuralTermDiscovery.predict_term`` answers: plain coordinates in, the learned term out."""

        def predict_term(self, points):
            return 0.25 * np.asarray(points, dtype=float)[:, 0]

    rng = np.random.default_rng(6)
    points = rng.uniform(-1.0, 1.0, size=(200, 1))
    from_term = module_dataset(_Term(), points, names=["u"], target_name="tau")
    from_callable = module_dataset(lambda p: 0.25 * np.asarray(p, dtype=float)[:, 0], points, names=["u"],
                                   target_name="tau")
    assert np.allclose(from_term.target, from_callable.target)
    assert from_callable.n == 200 and from_callable.names == ["u"]
    report = distill(from_term, config=DistillationConfig(backends=("sindy",), n_bootstrap=10, poly_order=1,
                                                          lhs="tau"))
    assert report.equation() == "tau = 0.25*(u)"
    assert report.terms[0]["coefficient"] == pytest.approx(0.25, abs=1e-9)


def test_a_near_tie_goes_to_the_simpler_arm_and_the_note_names_both_models(monkeypatch):
    """Two arms whose held-out errors are within 2 % of each other: the simpler one wins, and the note that records it
    names the complexity of the arm it beat -- not its own twice."""
    from pinneapple_analysis.inverse_problems import symbolic_distillation as module

    def arm(complexity, error):
        def fit(dataset, fit_rows, held_out, config):
            return module._Fit(backend="arm", coefficients={"f": 1.0}, expression="1*(x)", front=[],
                               complexity=complexity, held_out_error=error, fit_error=error)
        return fit

    monkeypatch.setitem(module._BACKENDS, "costlier", arm(6, 0.100))
    monkeypatch.setitem(module._BACKENDS, "simpler", arm(3, 0.101))
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, size=(100, 1))
    data = module_dataset(lambda p: np.asarray(p, dtype=float)[:, 0], points, names=["x"], target_name="f")
    report = distill(data, config=DistillationConfig(backends=("costlier", "simpler"), n_bootstrap=0))

    assert report.backend == "simpler"
    note = next(note for note in report.notes if "near-tie" in note)
    assert "(3 against 6)" in note
    assert "held-out error (10.100%)" in note and "(10.000%)" in note
