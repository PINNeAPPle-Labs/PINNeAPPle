"""One problem, three training backends (#75).

The acceptance lines under test: **the same problem trains in PyTorch and JAX, and PhysicsNeMo runs Benchmark case
02.** So the tests here do not check that two engines produce the same weights -- they cannot -- but that the same
:class:`~pinneapple_physics.physical_problem.PhysicalProblem` object goes through ``problem.backend(...)`` on each
engine, that what comes out is scored against the case's closed-form solution, and that every artifact reloads
through the one door (``pinneapple_neural.model_provider.load``) with the provenance naming the engine that wrote it.

A backend whose package is not installed is checked for its *refusal* (the message names the package), so the file
stays meaningful on a machine without JAX or PhysicsNeMo. The budgets are deliberately small: the published protocol
numbers are in ``benchmarks/cases/results.json`` and ``docs/benchmark_cases.md``, not in a unit test.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_neural.model_provider import ModelNotLoadable, load
from pinneapple_neural.problem_backends import (
    JAX_KINDS,
    BackendNotInstalled,
    ProblemBackend,
    UnsupportedProblem,
    available_backends,
    backends,
    jax_function_ref,
    problem_backend,
)
from pinneapple_neural.physics_model import PhysicsModel
from pinneapple_tools.benchmark_suite.cases import _score, get_case

CASE_ID = "02"                                   # heat_2d, the case the issue names


def _skip_unless(backend: str) -> None:
    ok, detail = available_backends()[backend]["available"], available_backends()[backend]["detail"]
    if not ok:
        pytest.skip(f"{backend} cannot run here: {detail}")


@pytest.fixture(scope="module")
def case_02():
    """The benchmark case (its own evaluation grid and its own in-domain/extrapolation split)."""
    case = get_case(CASE_ID)
    problem, X, exact = case.build()
    in_domain = X[:, list(problem.coords).index(case.split_coord)] <= case.split_at
    return problem, X, np.asarray(exact), in_domain


# ---------------------------------------------------------------------------------------- the registry
def test_the_three_backends_are_registered_and_say_when_a_package_is_missing():
    assert backends() == ["jax", "physicsnemo", "pytorch"]
    described = available_backends()
    assert described["pytorch"]["available"] and described["pytorch"]["artifact"] == ".pt"
    assert described["jax"]["artifact"] == ".npz" and described["physicsnemo"]["artifact"] == ".mdlus"
    for name, block in described.items():
        if not block["available"]:                # a missing package is named, with the way to install it
            assert name in block["detail"] and "pip install" in block["detail"]


def test_an_unknown_backend_is_refused_with_the_registered_ones():
    with pytest.raises(KeyError, match="unknown backend"):
        problem_backend(object(), "tensorflow")


def test_the_problem_hands_out_the_trainer_and_it_is_a_physics_model(case_02):
    problem, X, _exact, _mask = case_02
    train = problem.backend("pytorch")
    assert isinstance(train, ProblemBackend) and isinstance(train, PhysicsModel)
    assert train.family == "pinn" and train.backend_name == "pytorch"
    described = train.describe()
    assert described["backend"] == "pytorch" and described["coords"] == list(problem.coords)
    assert train.problem is problem


def test_a_missing_package_is_a_refusal_that_names_it(monkeypatch, case_02):
    """Not a ``ModuleNotFoundError`` from inside a train: the reason arrives before any work is done."""
    import pinneapple_neural.problem_backends as module

    problem, *_ = case_02
    monkeypatch.setattr(module, "_installed", lambda name: (False, f"{name} is not installed (pip install {name})"))
    with pytest.raises(BackendNotInstalled, match="jax is not installed"):
        problem.backend("jax").fit(epochs=1)
    assert problem.backend("jax").available()[0] is False


def test_uncertainty_is_refused_with_the_reason_instead_of_a_zero_spread(case_02):
    from pinneapple_neural.physics_model import UncertaintyNotAvailable

    problem, X, _exact, _mask = case_02
    with pytest.raises(UncertaintyNotAvailable, match="no sampler|sampler"):
        problem.backend("pytorch").uncertainty(X[:4])


# ---------------------------------------------------------------------------------------- the two engines
def test_the_same_problem_trains_on_pytorch_and_on_jax(case_02):
    """The acceptance line: one problem object, two engines, each scored against the closed form.

    The budgets are equal (same epochs, same collocation count, same seed) and the bar is the same for both: far
    better than the trivial zero field (relative L2 1.0) on the window the case trains in. The engines build
    different networks, so the two errors are reported apart rather than compared for equality.
    """
    _skip_unless("pytorch")
    _skip_unless("jax")
    problem, X, exact, in_domain = case_02
    errors = {}
    for name in ("pytorch", "jax"):
        train = problem.backend(name)
        report = train.fit(epochs=60, n_collocation=256, seed=0, hidden_dim=32, n_layers=3)
        assert report.family == "pinn" and report.method == "gradient"
        assert report.n_params > 0 and report.seconds > 0
        assert report.improved() is not None and report.improved() > 1.0, f"{name}: the loss did not fall"
        predicted = np.asarray(train.predict(X).mean.detach(), dtype=np.float64)
        assert predicted.shape == exact.shape
        errors[name] = _score(predicted, exact, in_domain)["in_domain"]["relative_l2"]
    for name, error in errors.items():
        assert np.isfinite(error) and error < 0.9, f"{name}: relative L2 {error:.3f} is no better than predicting 0"


def test_the_jax_backend_builds_the_kinds_it_declares_and_refuses_the_rest(case_02):
    """A residual this module does not write is refused by name -- never trained with a term silently missing."""
    _skip_unless("jax")
    import dataclasses

    problem, *_ = case_02
    navier_stokes = dataclasses.replace(problem, pde=dataclasses.replace(problem.pde, kind="navier_stokes_incompressible"))
    with pytest.raises(UnsupportedProblem) as err:
        navier_stokes.backend("jax").fit(epochs=1)
    assert "navier_stokes_incompressible" in str(err.value) and "heat_equation" in str(err.value)


def test_the_jax_backend_refuses_a_condition_that_needs_a_boundary_normal(case_02):
    _skip_unless("jax")
    import dataclasses

    from pinneapple_physics.pde_environment.conditions import NeumannBC

    problem, *_ = case_02
    with_neumann = dataclasses.replace(
        problem, conditions=problem.conditions + (NeumannBC("flux_out", ("T",), "callable",
                                                            lambda X, ctx: X[:, 1] >= 1.0 - 1e-6),))
    with pytest.raises(UnsupportedProblem, match="boundary normal"):
        with_neumann.backend("jax").fit(epochs=1)


def test_the_jax_backend_refuses_data_instead_of_silently_ignoring_it(case_02):
    _skip_unless("jax")
    problem, *_ = case_02
    with pytest.raises(TypeError, match="data"):
        problem.backend("jax").fit({"x": np.zeros((2, 3)), "y": np.zeros((2, 1))}, epochs=1)


def test_a_trained_jax_model_gives_a_solution_a_comparison_can_score(case_02):
    _skip_unless("jax")
    problem, X, exact, in_domain = case_02
    train = problem.backend("jax")
    train.fit(epochs=30, n_collocation=128, seed=0, hidden_dim=16, n_layers=2)
    solution = train.to_solution()
    assert solution.kind == "neural" and solution.info["backend"] == "jax"
    from_solution = solution.predict(X)
    assert from_solution.shape == exact.shape
    assert np.allclose(from_solution, train.predict_numpy(X), atol=0.0)


# ---------------------------------------------------------------------------------------- the one door
def _roundtrip(problem, name: str, tmp_path, **budget):
    train = problem.backend(name)
    train.fit(**budget)
    before = np.asarray(train.predict(problem_X(problem)).mean.detach())
    path = train.save(tmp_path / f"model{train.suffix}")
    assert path.is_file(), f"{name}: save() wrote no {train.suffix}"
    loaded = load(path)
    after = np.asarray(loaded.predict(np.asarray(problem_X(problem), dtype=np.float32)))
    assert loaded.provenance.provider == name
    assert after.shape == before.shape
    assert np.allclose(after, before, atol=1e-5), f"{name}: the reloaded artifact predicts something else"
    return loaded


def problem_X(problem):
    """A few points inside the problem's box, in ``problem.coords`` order."""
    rng = np.random.default_rng(7)
    lo = np.array([problem.domain_bounds[c][0] for c in problem.coords], dtype=np.float64)
    hi = np.array([problem.domain_bounds[c][1] for c in problem.coords], dtype=np.float64)
    return lo + (hi - lo) * rng.random((16, len(lo)))


def test_each_backend_artifact_reloads_through_the_same_door(case_02, tmp_path):
    """PyTorch, JAX and PhysicsNeMo write three different files; ``model_provider.load`` reads all three."""
    problem, *_ = case_02
    budget = dict(epochs=20, n_collocation=128, seed=0, hidden_dim=16, n_layers=2)
    for name in ("pytorch", "jax", "physicsnemo"):
        _skip_unless(name)
        loaded = _roundtrip(problem, name, tmp_path, **budget)
        assert loaded.provenance.framework, f"{name}: the provenance does not say what ran it"
        assert loaded.provenance.sha256, f"{name}: the artifact has no checksum in its provenance"


def test_a_file_the_jax_backend_wrote_can_be_read_by_a_second_process(case_02, tmp_path):
    """The artifact carries ``function: module:attribute`` -- an import path, so reloading is not a pickle of this
    process's state."""
    _skip_unless("jax")
    problem, *_ = case_02
    train = problem.backend("jax")
    train.fit(epochs=5, n_collocation=64, seed=0, hidden_dim=8, n_layers=1)
    path = train.save(tmp_path / "small.npz")
    with np.load(path, allow_pickle=True) as saved:
        assert str(saved["function"]) == jax_function_ref()
        assert str(saved["function"]).endswith(":jax_forward")
        assert "layer0_w" in saved.files
    assert load(path).provenance.provider == "jax"


def test_the_physicsnemo_provider_reads_the_case_02_checkpoint(case_02):
    """PhysicsNeMo on Benchmark case 02: a ``.mdlus`` module that scores at the level of the published table and
    reloads with ``physicsnemo.Module.from_checkpoint`` behind the provider."""
    _skip_unless("physicsnemo")
    problem, X, exact, in_domain = case_02
    train = problem.backend("physicsnemo")
    report = train.fit(epochs=300, n_collocation=512, seed=0, hidden_dim=32, n_layers=3)
    assert report.notes.startswith("physicsnemo")
    predicted = np.asarray(train.predict(X).mean.detach(), dtype=np.float64)
    error = _score(predicted, exact, in_domain)["in_domain"]["relative_l2"]
    # the published torch row for this case is 0.022; the same architecture in a PhysicsNeMo container lands there
    assert error < 0.10, f"case 02 in-domain relative L2 {error:.4f} is not a fit"


def test_the_door_unwraps_the_family_output_instead_of_returning_the_dataclass(tmp_path):
    """Regression: ``LoadedModel.predict`` of a library model returned a 0-d object array holding ``ModelOutput``."""
    import pinneapple_neural.architectures  # noqa: F401 - registers the model zoo
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_neural.physics_model import NotAPhysicsModel

    model = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden_dim=8, n_layers=2)
    path = tmp_path / "registry.pt"
    model.save_checkpoint(str(path), {"who": "test"})
    loaded = load(path)
    out = loaded.predict(np.zeros((5, 2), dtype=np.float32))
    assert isinstance(out, np.ndarray) and out.shape == (5, 1) and out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert loaded.physics_model.family == "pinn"          # the door hands it to the family interface

    other = ModelRegistry.build("modified_mlp", in_dim=2, out_dim=1, hidden_dim=8, n_layers=2)
    other.save_checkpoint(str(tmp_path / "other.pt"), {})
    # a registry architecture that belongs to no family is refused by name, not wrapped in an empty adapter
    with pytest.raises(NotAPhysicsModel, match="ModifiedMLP"):
        load(tmp_path / "other.pt").physics_model
