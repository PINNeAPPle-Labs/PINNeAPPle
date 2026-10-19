"""Red-team generators: parametrised families of wrong-but-plausible outputs that the evasion search explores.

A generator maps a correct case and a parameter vector to ``(faulty_case, correct_field)``. Parameters are named,
with bounds; the search works on the unit box and ``Generator.params`` maps back. Families:

* ``mutant``: the mutation operators of the fault library combined (bias, scale, smoothing, noise), the model-side
  operators of mutation testing for deep learning (DeepCrime, Humbatova et al. 2021; MuFF, arXiv:2501.09846).
* ``smooth_modes``: a smooth distortion that keeps the boundary or initial condition exactly (Darcy: sine modes
  vanishing on the boundary; Burgers: periodic modes growing from zero at t = 0).
* physics-consistent attacks, outputs that solve a nearby problem exactly, so the PDE residual stays small:
  Darcy ``coefficient_field`` (a smooth change of log a) and ``source_field`` (a smooth change of f); Burgers
  ``viscosity`` (another nu), ``hard_ic_evolution`` (the exact initial condition at t = 0, then the evolution of a
  smoothed and rescaled one, what a hard-constrained PINN with spectral bias returns) and ``time_warp``
  (u(x, s t)).
* ``fault:<name>``: one fault of the fault library at a given magnitude (escapes found by the mutation score are
  stored with these).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..fault_injection.faults import get_fault
from ..fault_injection.matrix import fault_rng
from ..fault_injection.problems import BenchmarkProblem, BurgersProblem, Case

__all__ = ["Generator", "generators_for", "get_generator", "GENERATORS"]


@dataclass(frozen=True)
class Generator:
    name: str
    bounds: dict[str, tuple[float, float]]
    fn: Callable[[BenchmarkProblem, Case, dict[str, float]], tuple[Case, np.ndarray]]
    problems: tuple[str, ...]
    physics_consistent: bool
    description: str

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def params(self, z: np.ndarray) -> dict[str, float]:
        """Unit-box point -> named parameters."""
        return {k: float(lo + (hi - lo) * zi) for (k, (lo, hi)), zi in zip(self.bounds.items(), z, strict=True)}

    def apply(self, problem: BenchmarkProblem, case: Case, params: dict[str, float]) -> tuple[Case, np.ndarray]:
        faulty, ref = self.fn(problem, case, dict(params))
        return replace(faulty, meta={**faulty.meta, "generator": self.name, "params": dict(params)}), ref


def _rng(case: Case, params: dict[str, float]) -> np.random.Generator:
    return np.random.default_rng([int(case.seed) % (2 ** 32), int(params.get("rng_seed", 0))])


# ====================================================================== generic
def _mutant(p, c, q):
    rng = _rng(c, q)
    out = c
    for name in ("smoothing", "scale", "bias", "noise"):
        m = q[name]
        if abs(m) > 1e-12 and not (name == "smoothing" and m < 0.05):
            out, _ = get_fault(name).apply(p, out, m, rng)
    return out, c.field


def _smooth_modes(p, c, q):
    co = p.coords(c)
    if isinstance(p, BurgersProblem):                  # periodic modes, zero at t = 0, growing linearly in time
        t, x = np.meshgrid(co["t"], co["x"], indexing="ij")
        amp = float(np.abs(c.inputs["u0"]).max())
        d = sum(q[f"c{k}"] * np.sin(2 * np.pi * k * x + q[f"phase{k}"]) for k in (1, 2, 3))
        return c.with_field(c.field + amp * d * t / co["t"][-1]), c.field
    x, y = np.meshgrid(co["x"], co["y"], indexing="ij")    # Darcy: u (1 + sum c_kl sin(k pi x) sin(l pi y))
    d = sum(q[f"c{k}{j}"] * np.sin(k * np.pi * x) * np.sin(j * np.pi * y) for k in (1, 2) for j in (1, 2))
    return c.with_field(c.field * (1 + d)), c.field


# ====================================================================== physics-consistent, Darcy
def _darcy_modes(c, q, prefix):
    n = c.field.shape[0]
    xc = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(xc, xc, indexing="ij")
    return sum(q[f"{prefix}{k}{j}"] * np.cos(k * np.pi * x) * np.cos(j * np.pi * y) for k in (0, 1) for j in (0, 1))


def _coefficient_field(p, c, q):
    a = c.inputs["a"] * np.exp(_darcy_modes(c, q, "g"))
    return c.with_field(p.resolve(c, a=a)), c.field


def _source_field(p, c, q):
    f = c.params["f"] * (1 + _darcy_modes(c, q, "s"))
    return c.with_field(p.resolve(c, f=f)), c.field


# ====================================================================== physics-consistent, Burgers
def _viscosity(p, c, q):
    return c.with_field(p.resolve(c, nu=c.params["nu"] * float(np.exp(q["log_factor"])))), c.field


def _hard_ic_evolution(p, c, q):
    from scipy.ndimage import gaussian_filter1d
    u0 = q["amplitude"] * gaussian_filter1d(c.inputs["u0"], q["sigma"], mode="wrap") if q["sigma"] > 0.05 \
        else q["amplitude"] * c.inputs["u0"]
    out = p.resolve(c, u0=u0)
    out[0] = c.inputs["u0"]
    return c.with_field(out), c.field


def _time_warp(p, c, q):
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
    o = p.options
    _, u = burgers_1d(c.inputs["u0"], c.params["nu"], o["t_end"] * q["s"], dt=o["dt"], n_save=int(o["n_t"]) - 1)
    return c.with_field(u[0]), c.field


GENERATORS: dict[str, Generator] = {g.name: g for g in [
    Generator("mutant", {"bias": (-0.15, 0.15), "scale": (-0.3, 0.3), "smoothing": (0.0, 4.0), "noise": (0.0, 0.05)},
              _mutant, ("darcy_2d", "burgers_1d"), False,
              "fault-library mutation operators combined: smoothing, scale, bias, noise"),
    Generator("smooth_modes", {**{f"c{k}{j}": (-0.4, 0.4) for k in (1, 2) for j in (1, 2)}}, _smooth_modes,
              ("darcy_2d",), False, "u (1 + sum c_kl sin(k pi x) sin(l pi y)): boundary condition kept exactly"),
    Generator("smooth_modes_periodic", {**{f"c{k}": (-0.3, 0.3) for k in (1, 2, 3)},
                                        **{f"phase{k}": (0.0, 2 * np.pi) for k in (1, 2, 3)}},
              _smooth_modes, ("burgers_1d",), False,
              "u + max|u0| (t / T) sum c_k sin(2 pi k x + phi_k): initial condition and periodicity kept"),
    Generator("coefficient_field", {f"g{k}{j}": (-0.6, 0.6) for k in (0, 1) for j in (0, 1)}, _coefficient_field,
              ("darcy_2d",), True, "exact solution for a' = a exp(sum g_kl cos(k pi x) cos(l pi y))"),
    Generator("source_field", {f"s{k}{j}": (-0.5, 0.5) for k in (0, 1) for j in (0, 1)}, _source_field,
              ("darcy_2d",), True, "exact solution for f' = f (1 + sum s_kl cos(k pi x) cos(l pi y))"),
    Generator("viscosity", {"log_factor": (-2.0, 2.0)}, _viscosity, ("burgers_1d",), True,
              "exact solution for nu' = nu exp(log_factor)"),
    Generator("hard_ic_evolution", {"sigma": (0.0, 4.0), "amplitude": (0.6, 1.4)}, _hard_ic_evolution,
              ("burgers_1d",), True,
              "u(0) = u0 exactly, then the exact evolution of amplitude * smooth(u0, sigma cells)"),
    Generator("time_warp", {"s": (0.5, 1.6)}, _time_warp, ("burgers_1d",), True,
              "u(x, s t): the exact solution at rescaled times"),
]}


def _fault_generator(name: str) -> Generator:
    F = get_fault(name)

    def fn(p, c, q):                                   # the same random stream as run_fault_injection(seed=rng_seed)
        return F.apply(p, c, q["magnitude"], fault_rng(int(q.get("rng_seed", 0)), name, q["magnitude"], c.seed))
    return Generator(f"fault:{name}", {"magnitude": (min(F.magnitudes), max(F.magnitudes))}, fn,
                     ("darcy_2d", "burgers_1d"), False, F.description)


def get_generator(name: str) -> Generator:
    if name.startswith("fault:"):
        return _fault_generator(name.split(":", 1)[1])
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; use one of {sorted(GENERATORS)} or 'fault:<name>'")
    return GENERATORS[name]


def generators_for(problem: BenchmarkProblem) -> list[Generator]:
    return [g for g in GENERATORS.values() if problem.name in g.problems]

