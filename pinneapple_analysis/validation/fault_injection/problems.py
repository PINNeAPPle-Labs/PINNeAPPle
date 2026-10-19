"""Benchmark problems with a verified reference solver, as sources of correct cases for the fault-injection suite.

A :class:`Case` is one input/output pair of an operator-learning problem: the inputs the model is given, the field it
returned on a fixed grid, and the parameters of the equation. The two problems wrap the verified solvers of
:mod:`pinneapple_simulation.numerical_solvers.operator_benchmarks` (checked in ``tests/test_operator_benchmarks.py``):

* :class:`DarcyProblem`: ``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary, ``a`` a thresholded
  Gaussian random field (Li et al. 2021); the field is ``u`` at the ``n x n`` cell centres.
* :class:`BurgersProblem`: ``u_t + (u^2/2)_x = nu u_xx`` on the periodic interval [0, 1); the field is the space-time
  solution ``u(t_k, x_j)`` at ``n_t`` snapshots from 0 to ``t_end`` (what a space-time PINN or an autoregressive
  operator returns).

Every case is reproducible from its ``seed`` and its ``variation`` (the knobs of the false-alarm search: coefficient
contrast, feature size, amplitude, viscosity, solver resolution), so a stored regression case is a few numbers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Case", "Dataset", "BenchmarkProblem", "DarcyProblem", "BurgersProblem", "get_problem", "PROBLEMS"]


@dataclass
class Case:
    """One input/output pair. ``field``: the output on the problem's grid; ``inputs``: what the model was given;
    ``params``: the equation's parameters for this case; ``seed`` and ``variation`` regenerate the correct case."""
    problem: str
    inputs: dict[str, np.ndarray]
    field: np.ndarray
    params: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    variation: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def with_field(self, values: np.ndarray, **meta: Any) -> Case:
        return replace(self, field=np.asarray(values, dtype=np.float64), meta={**self.meta, **meta})


@dataclass
class Dataset:
    """A training and a test split of cases (what a datapipe hands to training)."""
    train: list[Case]
    test: list[Case]

    @property
    def cases(self) -> list[Case]:
        return [*self.train, *self.test]


class BenchmarkProblem:
    """A problem with a verified reference solver. Subclasses define ``name``, ``defaults`` (options),
    ``variation_space`` (the ranges of the false-alarm search, all inside the training envelope) and
    ``_solve(seed, variation)``."""
    name = ""
    title = ""
    defaults: dict[str, Any] = {}
    variation_space: dict[str, tuple[float, float]] = {}

    def __init__(self, **options: Any):
        unknown = set(options) - set(self.defaults)
        if unknown:
            raise ValueError(f"unknown options for {self.name}: {sorted(unknown)}")
        self.options = {**self.defaults, **options}
        self._envelope = None

    # ---- cases
    def case(self, seed: int, **variation: float) -> Case:
        """The correct case of ``seed`` (the input drawn from the training law, perturbed by ``variation``)."""
        bad = set(variation) - set(self.variation_space)
        if bad:
            raise ValueError(f"unknown variation for {self.name}: {sorted(bad)}; use {sorted(self.variation_space)}")
        return self._solve(int(seed), {k: float(v) for k, v in variation.items()})

    def sample(self, n: int, seed: int = 0, **variation: float) -> list[Case]:
        """``n`` correct cases with seeds derived from ``seed``."""
        seeds = np.random.default_rng(seed).integers(0, 2 ** 31 - 1, size=n)
        return [self.case(int(s), **variation) for s in seeds]

    def sample_varied(self, n: int, seed: int = 0) -> list[Case]:
        """``n`` correct cases whose ``variation`` is drawn uniformly from ``variation_space`` (unusual but
        correct: the clean population a threshold must not flag)."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            var = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.variation_space.items()}
            out.append(self.case(int(rng.integers(0, 2 ** 31 - 1)), **var))
        return out

    def _solve(self, seed: int, variation: dict[str, float]) -> Case:
        raise NotImplementedError

    def resolve(self, case: Case, **changes: Any) -> np.ndarray:
        """The reference field for ``case`` with some of its inputs or parameters changed (a model that solves a
        nearby problem exactly: the physics-consistent faults)."""
        raise NotImplementedError

    def predict_reference(self, inputs: Mapping[str, np.ndarray], params: Mapping[str, float]) -> np.ndarray:
        """The reference solver as a model ``predict(inputs, params) -> field`` (the ``audit_model`` interface)."""
        c = Case(self.name, dict(inputs), np.empty(0), dict(params))
        return self.resolve(c)

    # ---- what the checks need
    def coords(self, case: Case) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def features(self, case: Case) -> dict[str, float]:
        """Scalar descriptors of the input (for the training envelope)."""
        raise NotImplementedError

    def input_case(self, seed: int, **variation: float) -> Case:
        """The inputs of ``case(seed, **variation)`` without solving (``field`` is empty)."""
        inputs, params = self._inputs(int(seed), {k: float(v) for k, v in variation.items()})
        return Case(self.name, inputs, np.empty(0), params, int(seed), dict(variation))

    def _inputs(self, seed: int, variation: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        raise NotImplementedError

    def envelope(self):
        """The training envelope (``physics_audit.Envelope``) over the input features of ``n_train`` inputs drawn
        from the training law (``variation_space`` uniformly, seed ``envelope_seed``)."""
        if self._envelope is None:
            from pinneapple_physics.physics_audit import Envelope
            rng = np.random.default_rng(int(self.options["envelope_seed"]))
            feats = []
            for _ in range(int(self.options["n_train"])):
                var = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.variation_space.items()}
                feats.append(self.features(self.input_case(int(rng.integers(0, 2 ** 31 - 1)), **var)))
            self._envelope = Envelope.from_samples(feats)
        return self._envelope

    def true_error(self, case: Case, reference: Case | np.ndarray) -> float:
        """Relative L2 error of ``case.field`` against the correct field."""
        ref = reference.field if isinstance(reference, Case) else np.asarray(reference)
        return float(np.linalg.norm(case.field - ref) / max(np.linalg.norm(ref), 1e-300))

    def describe(self) -> dict[str, Any]:
        return {"problem": self.name, "options": dict(self.options)}


# ====================================================================== Darcy 2-D
class DarcyProblem(BenchmarkProblem):
    """``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary.

    Options: ``n`` (cells per side), ``high``/``low`` (the two permeabilities), ``alpha``/``tau`` (law of the
    underlying Gaussian random field), ``f`` (uniform source). Variation (all within the training envelope):
    ``contrast`` (multiplies ``high / low`` around their geometric mean, 0.8 .. 1.25), ``tau`` (feature size),
    ``level`` (threshold of the random field: the fraction of high-permeability cells), ``refine`` (the reference
    solved on a 1x, 2x or 3x finer grid and averaged onto the ``n x n`` cells: a different but consistent
    discretisation).
    """
    name = "darcy_2d"
    title = "Darcy 2-D"
    defaults = {"n": 32, "high": 12.0, "low": 3.0, "alpha": 2.0, "tau": 3.0, "f": 1.0, "n_train": 256,
                "envelope_seed": 7}
    variation_space = {"contrast": (0.8, 1.25), "tau": (2.5, 4.0), "level": (-0.3, 0.3), "refine": (1.0, 3.99)}

    def _coefficient(self, seed: int, variation: Mapping[str, float], n: int) -> np.ndarray:
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import periodic_grf
        o = self.options
        tau = variation.get("tau", o["tau"])
        g = periodic_grf(1, (n, n), alpha=o["alpha"], tau=tau, sigma=1.0, seed=seed)[0]
        g = g / max(float(g.std()), 1e-300)
        c = variation.get("contrast", 1.0)
        mid = np.sqrt(o["high"] * o["low"])
        hi, lo = mid * (o["high"] / mid) ** c, mid * (o["low"] / mid) ** c
        return np.where(g >= variation.get("level", 0.0), hi, lo)

    def _pool(self, z: np.ndarray, r: int) -> np.ndarray:
        n = int(self.options["n"])
        return z.reshape(n, r, n, r).mean(axis=(1, 3)) if r > 1 else z

    def _inputs(self, seed, variation):
        r = int(np.floor(variation.get("refine", 1.0)))
        a_fine = self._coefficient(seed, variation, int(self.options["n"]) * r)
        return {"a": self._pool(a_fine, r)}, {"f": float(self.options["f"])}

    def _solve(self, seed: int, variation: dict[str, float]) -> Case:
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d
        o = self.options
        r = int(np.floor(variation.get("refine", 1.0)))
        a_fine = self._coefficient(seed, variation, int(o["n"]) * r)
        u = self._pool(darcy_2d(a_fine, o["f"]), r)
        return Case(self.name, {"a": self._pool(a_fine, r)}, u, {"f": float(o["f"])}, seed, dict(variation))

    def resolve(self, case: Case, *, a: np.ndarray | None = None, f: float | np.ndarray | None = None,
                axis_weights: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d
        return darcy_2d(case.inputs["a"] if a is None else a, case.params["f"] if f is None else f,
                        axis_weights=axis_weights)

    def coords(self, case: Case) -> dict[str, np.ndarray]:
        n = case.field.shape[0]
        xc = (np.arange(n) + 0.5) / n
        return {"x": xc, "y": xc}

    def features(self, case: Case) -> dict[str, float]:
        from pinneapple_analysis.benchmark.validation import field_features
        a = case.inputs["a"]
        return {**field_features(a), "a_min": float(a.min()), "a_max": float(a.max())}


# ====================================================================== Burgers 1-D
class BurgersProblem(BenchmarkProblem):
    """``u_t + (u^2/2)_x = nu u_xx`` on the periodic interval [0, 1), ``u(x, 0) = u0(x)``.

    Options: ``n`` (points), ``n_t`` (snapshots from 0 to ``t_end``, both included), ``nu``, ``dt`` (solver time
    step), ``alpha``/``tau``/``sigma`` (law of the initial condition, a periodic Gaussian random field). Variation
    (within the training envelope): ``amplitude`` (multiplies ``u0``, 0.7 .. 1.3), ``tau`` (feature size),
    ``nu_factor`` (viscosity of the case, 0.8 .. 1.25 times ``nu``: the model takes ``nu`` as an input),
    ``refine`` (reference solved on a 1x, 2x or 3x finer grid with a smaller time step and sampled back).
    """
    name = "burgers_1d"
    title = "Burgers 1-D"
    defaults = {"n": 128, "n_t": 21, "t_end": 0.5, "nu": 0.01, "dt": 1e-3, "alpha": 2.5, "tau": 7.0, "sigma": 1.0,
                "n_train": 256, "envelope_seed": 7}
    variation_space = {"amplitude": (0.7, 1.3), "tau": (6.0, 9.0), "nu_factor": (0.8, 1.25), "refine": (1.0, 3.99)}

    def _u0(self, seed: int, variation: Mapping[str, float], n: int) -> np.ndarray:
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import periodic_grf
        o = self.options
        g = periodic_grf(1, (n,), alpha=o["alpha"], tau=variation.get("tau", o["tau"]), sigma=1.0, seed=seed)[0]
        return o["sigma"] * variation.get("amplitude", 1.0) * g / max(float(g.std()), 1e-300)

    def _run(self, u0: np.ndarray, nu: float, dt: float) -> np.ndarray:
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
        o = self.options
        _, u = burgers_1d(u0, nu, o["t_end"], dt=dt, n_save=int(o["n_t"]) - 1)
        return u[0]

    def _inputs(self, seed, variation):
        r = int(np.floor(variation.get("refine", 1.0)))
        u0 = self._u0(seed, variation, int(self.options["n"]) * r)[::r].copy()
        return {"u0": u0}, {"nu": float(self.options["nu"] * variation.get("nu_factor", 1.0))}

    def _solve(self, seed: int, variation: dict[str, float]) -> Case:
        o = self.options
        n = int(o["n"])
        r = int(np.floor(variation.get("refine", 1.0)))
        nu = o["nu"] * variation.get("nu_factor", 1.0)
        # the coarse input is the fine one sampled back: band-limited GRF, so the coarse grid sees the same u0
        u0_fine = self._u0(seed, variation, n * r)
        u = self._run(u0_fine, nu, o["dt"] / r)[:, ::r]
        return Case(self.name, {"u0": u0_fine[::r].copy()}, u, {"nu": float(nu)}, seed, dict(variation))

    def resolve(self, case: Case, *, u0: np.ndarray | None = None, nu: float | None = None) -> np.ndarray:
        return self._run(case.inputs["u0"] if u0 is None else u0, case.params["nu"] if nu is None else nu,
                         self.options["dt"])

    def coords(self, case: Case) -> dict[str, np.ndarray]:
        nt, n = case.field.shape
        return {"t": np.linspace(0.0, self.options["t_end"], nt), "x": np.arange(n) / n}

    def features(self, case: Case) -> dict[str, float]:
        from pinneapple_analysis.benchmark.validation import field_features
        u0 = case.inputs["u0"]
        return {**field_features(u0), "u0_max": float(np.abs(u0).max()), "nu": float(case.params["nu"])}


PROBLEMS: dict[str, type[BenchmarkProblem]] = {"darcy_2d": DarcyProblem, "burgers_1d": BurgersProblem}


def get_problem(problem: str | BenchmarkProblem, **options: Any) -> BenchmarkProblem:
    """A problem by name (``"darcy_2d"``, ``"burgers_1d"``; ``"darcy"``/``"burgers"`` work too) or as given."""
    if isinstance(problem, BenchmarkProblem):
        return problem
    key = {"darcy": "darcy_2d", "burgers": "burgers_1d"}.get(problem, problem)
    if key not in PROBLEMS:
        raise KeyError(f"unknown problem {problem!r}; use one of {sorted(PROBLEMS)}")
    return PROBLEMS[key](**options)
