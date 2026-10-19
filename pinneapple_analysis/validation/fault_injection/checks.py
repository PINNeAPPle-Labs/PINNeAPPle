"""The checks under test, as ``physics_audit.monitor.ValidityCheck`` objects scoring a :class:`~.problems.Case`.

A check is ``fn(field, case) -> value`` with a tolerance; it flags a case when ``value > tolerance`` (or the value is
not finite). The values are continuous, so the same check can be thresholded by hand (the defaults here), calibrated
(:mod:`pinneapple_analysis.validation.arena`) or attacked (an optimiser sees how close a case is to being flagged).

Building blocks reused, not reimplemented: the finite-difference provider and relative residual of
:mod:`pinneapple_check.numerics` (``GridD``, ``rel_rms``), the conservative Darcy operator of the verified solver
(``operator_benchmarks.darcy_operator``) and its Burgers integrator (``burgers_1d``), the spectral shells of
``validation.field_metrics``, the Darcy consistency and the duplicate and leakage detectors of the dataset audit
(``pinneapple_data.audit``), and the bounds and training-envelope checks of ``pinneapple_physics.physics_audit.monitor``.

Two check sets per problem:

* ``default_checks(problem)``: what a user would run today (a finite-difference PDE residual, boundary and bounds
  checks, the training envelope), with hand-set tolerances. This is round 0 of the arena.
* ``check_library(problem)``: every check available, including stronger ones (the conservative residual on the
  solver's own stencil, a one-step re-integration, conservation, spectral content, invariants). The blue team of
  the arena picks from it.

Dataset checks (``dataset_checks``) score a whole :class:`~.problems.Dataset`: near-duplicate inputs, train/test
leakage, and the worst per-sample value of each case check.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .problems import BenchmarkProblem, BurgersProblem, Case, DarcyProblem, Dataset

__all__ = ["score", "flags", "default_checks", "check_library", "dataset_checks", "TORSION_MAX"]

#: max of w with -lap(w) = 1 on the unit square, w = 0 on the boundary (Timoshenko & Goodier, Theory of
#: Elasticity, torsion of a square bar: 0.0737 a^2 for side a; series value 0.07367): the comparison bound of Darcy.
TORSION_MAX = 0.07367


def _vc():
    from pinneapple_physics.physics_audit import monitor
    return monitor


def score(check, case: Case) -> float:
    """The value of ``check`` on ``case`` (``inf`` when it cannot be computed or the field is not finite)."""
    if not np.all(np.isfinite(case.field)):
        return float("inf")
    try:
        v = float(check.fn(case.field, case))
    except Exception:  # noqa: BLE001 - a check that crashes on a case is a flag, not a pass
        return float("inf")
    return v if np.isfinite(v) else float("inf")


def flags(check, case: Case, tolerance: float | None = None) -> bool:
    """True when ``check`` rejects ``case`` (``tolerance`` overrides the check's own)."""
    return score(check, case) > (check.tolerance if tolerance is None else tolerance)


def _check(name: str, kind: str, fn: Callable[[np.ndarray, Case], float], tol: float, describe: str):
    m = _vc()
    c = m.ValidityCheck(name, kind, fn, float(tol))
    c.description = describe  # type: ignore[attr-defined]
    return c


# ====================================================================== Darcy
def _darcy_fd_residual(u: np.ndarray, case: Case, problem: DarcyProblem) -> float:
    from pinneapple_check.numerics import GridD, rel_rms
    xy = problem.coords(case)
    a = case.inputs["a"]
    D = GridD(xy, {"u": u})
    F = GridD(xy, {"qx": a * D.d("u", "x"), "qy": a * D.d("u", "y")})
    dqx, dqy = F.d("qx", "x"), F.d("qy", "y")
    f = np.broadcast_to(np.asarray(case.params["f"], float), u.shape)
    return rel_rms(-(dqx + dqy) - f, [dqx, dqy, f], D.interior)


def _darcy_consistency(u: np.ndarray, case: Case) -> dict[str, float]:
    # pinneapple_data.audit (#700): the residual of the solver's own stencil in the A^-1 norm (|u - A^-1 f| / |A^-1 f|)
    # and the net finite-volume boundary flux against the source
    from pinneapple_data.audit import darcy_consistency
    return darcy_consistency(case.inputs["a"], u, case.params["f"])


def _darcy_boundary(u: np.ndarray, case: Case) -> float:
    # cell-centred values extrapolated linearly to the four faces, where u = 0
    faces = [1.5 * u[0] - 0.5 * u[1], 1.5 * u[-1] - 0.5 * u[-2], 1.5 * u[:, 0] - 0.5 * u[:, 1],
             1.5 * u[:, -1] - 0.5 * u[:, -2]]
    return float(max(np.abs(fc).max() for fc in faces) / max(np.abs(u).max(), 1e-300))


def _darcy_bound(case: Case) -> float:
    f = np.asarray(case.params["f"], float)
    return TORSION_MAX * float(np.abs(f).max()) / float(case.inputs["a"].min())


def _high_frequency(u: np.ndarray, ndim: int, cut: float) -> float:
    from pinneapple_analysis.validation.field_metrics import spectral_error_shells
    shells = spectral_error_shells(u, np.zeros_like(u), ndim)
    shells = shells.reshape(-1, shells.shape[-1]).sum(axis=0)
    k0 = int(np.ceil(cut * min(u.shape[-ndim:])))
    return float(shells[k0:].sum() / max(shells.sum(), 1e-300))


def _darcy_library(p: DarcyProblem) -> dict[str, Any]:
    m = _vc()
    env = m.envelope_check(p.envelope(), p.features, min_score=0.5, name="training envelope")
    env.description = "1 - physics_audit.Envelope score of the input features (amplitude, spectral centroid, a range)"
    bounds = m.bounds_check(0.0, 1.0, name="bounds (0 <= u <= max torsion / a_min)",
                            select=lambda u, c: u / _darcy_bound(c), critical=False)
    bounds.description = "fraction of cells outside [0, 0.0737 max f / min a] (minimum and comparison principles)"
    cons = _check("conservation (outflow = source)", "conservation",
                  lambda u, c: _darcy_consistency(u, c)["flux"], 0.05,
                  "net finite-volume outflow through the boundary against the integral of the source "
                  "(pinneapple_data.audit.darcy_consistency)")
    return {
        "residual (finite differences)": _check(
            "residual (finite differences)", "residual", lambda u, c: _darcy_fd_residual(u, c, p), 0.6,
            "pinneapple_check.numerics GridD: -d(a u_x)/dx - d(a u_y)/dy - f, rms over the interior / rms of |terms|"),
        "residual (conservative, solver stencil)": _check(
            "residual (conservative, solver stencil)", "residual", lambda u, c: _darcy_consistency(u, c)["solve"], 0.08,
            "residual of the solver's two-point-flux stencil in the A^-1 norm, |A^-1 (A u - f)| / |A^-1 f| "
            "(pinneapple_data.audit.darcy_consistency)"),
        "boundary condition (u = 0)": _check(
            "boundary condition (u = 0)", "boundary", _darcy_boundary, 0.1,
            "max |u| linearly extrapolated to the faces / max |u|"),
        bounds.name: bounds,
        cons.name: cons,
        "spectral content (k > n/4)": _check(
            "spectral content (k > n/4)", "invariant", lambda u, c: _high_frequency(u, 2, 0.25), 1e-3,
            "fraction of the energy of u above a quarter of the grid wavenumbers (field_metrics shells)"),
        env.name: env,
    }


# ====================================================================== Burgers
def _burgers_fd_residual(u: np.ndarray, case: Case, problem: BurgersProblem) -> float:
    from pinneapple_check.numerics import GridD, rel_rms
    D = GridD(problem.coords(case), {"u": u})
    ut, ux, uxx = D.dt("u"), D.d("u", "x"), D.dd("u", "x", "x")
    nu = case.params["nu"]
    return rel_rms(ut + u * ux - nu * uxx, [ut, u * ux, nu * uxx], D.interior)


def _burgers_one_step(u: np.ndarray, case: Case, problem: BurgersProblem) -> float:
    # re-integrate every snapshot interval with the verified integrator and compare with the next snapshot
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
    o = problem.options
    dt_snap = o["t_end"] / (u.shape[0] - 1)
    with np.errstate(all="ignore"):                    # a wild field may blow the integrator up: that is a flag
        _, v = burgers_1d(u[:-1], case.params["nu"], dt_snap, dt=min(o["dt"], dt_snap))
    err = np.linalg.norm(v[:, -1] - u[1:], axis=1)
    return float(err.max() / max(np.linalg.norm(u[0]), 1e-300))


def _burgers_ic(u: np.ndarray, case: Case) -> float:
    u0 = case.inputs["u0"]
    return float(np.abs(u[0] - u0).max() / max(np.abs(u0).max(), 1e-300))


def _burgers_periodicity(u: np.ndarray, case: Case, w: int = 4) -> float:
    # the periodic second difference at the two points next to x = 0 against the largest one in the w points
    # around them: a solution continued across x = 0 looks like its neighbourhood (a front there too), a jump does not
    d2 = np.abs(np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1))
    wrap = d2[:, [0, -1]].max(axis=1)
    near = np.concatenate([d2[:, 1:1 + w], d2[:, -1 - w:-1]], axis=1).max(axis=1)
    scale = 1e-3 * float(np.abs(np.diff(u, axis=1)).max())
    return float((wrap / np.maximum(near, scale + 1e-300)).max())


def _burgers_mass(u: np.ndarray, case: Case) -> float:
    u0 = case.inputs["u0"]
    return float(np.abs(u.mean(axis=1) - u0.mean()).max() / max(np.abs(u0).mean(), 1e-300))


def _burgers_energy(u: np.ndarray, case: Case) -> float:
    e = (u ** 2).mean(axis=1)
    return float(max(np.diff(e).max(), 0.0) / max(e[0], 1e-300))


def _burgers_library(p: BurgersProblem) -> dict[str, Any]:
    m = _vc()
    env = m.envelope_check(p.envelope(), p.features, min_score=0.5, name="training envelope")
    env.description = "1 - physics_audit.Envelope score of the input features (amplitude, spectral centroid, max |u0|, nu)"
    bounds = m.bounds_check(None, 1.0 + 1e-9, name="bounds (max |u| <= max |u0|)",
                            select=lambda u, c: np.abs(u) / np.abs(c.inputs["u0"]).max(), critical=False)
    bounds.description = "fraction of values above max |u0| (maximum principle of viscous Burgers)"
    return {
        "residual (finite differences)": _check(
            "residual (finite differences)", "residual", lambda u, c: _burgers_fd_residual(u, c, p), 0.3,
            "pinneapple_check.numerics GridD: u_t + u u_x - nu u_xx over the interior / rms of |terms|"),
        "residual (one-step re-integration)": _check(
            "residual (one-step re-integration)", "residual", lambda u, c: _burgers_one_step(u, c, p), 0.02,
            "each snapshot advanced to the next with operator_benchmarks.burgers_1d: max L2 gap / L2 of u(0)"),
        "initial condition": _check(
            "initial condition", "boundary", _burgers_ic, 0.02, "max |u(x, 0) - u0| / max |u0|"),
        "periodicity": _check(
            "periodicity", "boundary", _burgers_periodicity, 4.0,
            "|periodic second difference at x = 0| / max |second difference| of the 4 points on each side"),
        bounds.name: bounds,
        "mass conservation": _check(
            "mass conservation", "conservation", _burgers_mass, 0.01, "max_t |mean u(t) - mean u0| / mean |u0|"),
        "energy decay": _check(
            "energy decay", "invariant", _burgers_energy, 1e-3, "largest increase of mean u^2 between snapshots / mean u0^2"),
        "spectral content (k > n/3)": _check(
            "spectral content (k > n/3)", "invariant", lambda u, c: _high_frequency(u[1:], 1, 1 / 3), 1e-4,
            "energy above the 2/3-rule cutoff at t > 0 / total (the solver dealiases)"),
        env.name: env,
    }


# ====================================================================== sets
_DEFAULT = {
    "darcy_2d": ("residual (finite differences)", "boundary condition (u = 0)",
                 "bounds (0 <= u <= max torsion / a_min)", "training envelope"),
    "burgers_1d": ("residual (finite differences)", "initial condition", "bounds (max |u| <= max |u0|)",
                   "training envelope"),
}


def check_library(problem: BenchmarkProblem) -> dict[str, Any]:
    """Every check available for ``problem`` (name -> ``ValidityCheck`` with a hand-set tolerance)."""
    if isinstance(problem, DarcyProblem):
        return _darcy_library(problem)
    if isinstance(problem, BurgersProblem):
        return _burgers_library(problem)
    raise TypeError(f"no checks for {type(problem).__name__}")


def default_checks(problem: BenchmarkProblem) -> list:
    """The checks a user runs today (round 0 of the arena)."""
    lib = check_library(problem)
    return [lib[n] for n in _DEFAULT[problem.name]]


# ====================================================================== dataset checks
def _input_matrix(cases: list[Case]) -> np.ndarray:
    return np.stack([np.concatenate([np.ravel(v) for _, v in sorted(c.inputs.items())]) for c in cases])


def _duplicates(ds: Dataset) -> float:
    from pinneapple_data.audit import near_duplicates
    r = near_duplicates(_input_matrix(ds.cases))
    return float(max(r["exact"], r["near"]) / max(r["n"], 1))


def _leakage(ds: Dataset) -> float:
    from pinneapple_data.audit import leakage
    if not ds.train or not ds.test:
        return 0.0
    return float(leakage(_input_matrix(ds.train), _input_matrix(ds.test))["fraction"])


def dataset_checks(problem: BenchmarkProblem, case_checks: list | None = None) -> list:
    """Checks of a whole dataset: near-duplicate inputs and test inputs that copy training inputs (the detectors of
    ``pinneapple_data.audit``), and the worst per-sample value of every case check (default: the whole
    ``check_library`` but the training envelope, which the data defines)."""
    out = [
        _check("duplicated or non-diverse samples", "data", lambda _u, ds: _duplicates(ds), 0.0,
               "fraction of samples with an exact or near copy (pinneapple_data.audit.near_duplicates)"),
        _check("train/test leakage", "data", lambda _u, ds: _leakage(ds), 0.0,
               "fraction of test inputs that copy a training input (pinneapple_data.audit.leakage)"),
    ]
    if case_checks is None:
        case_checks = [c for c in check_library(problem).values() if c.kind != "envelope"]
    for c in case_checks:
        out.append(_check(f"every sample: {c.name}", "data", _worst_sample(c), c.tolerance,
                          f"worst sample of: {getattr(c, 'description', c.name)}"))
    return out


def _worst_sample(check, cache_size: int = 1024):
    # datasets share their correct samples: remember the last values by object (the case is kept alive with them)
    cache: dict[int, tuple[Case, float]] = {}

    def fn(_u, ds: Dataset) -> float:
        worst = -np.inf
        for s in ds.cases:
            hit = cache.get(id(s))
            if hit is None or hit[0] is not s:
                if len(cache) >= cache_size:
                    cache.clear()
                hit = cache[id(s)] = (s, score(check, s))
            worst = max(worst, hit[1])
        return float(worst)
    return fn
