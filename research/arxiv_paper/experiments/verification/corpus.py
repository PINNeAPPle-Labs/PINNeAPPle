"""Controlled corpus of valid and invalid predictions, used to measure the error-detection
behaviour of the verification layer (``pinneapple check``).

The corpus is deliberately *not* a set of trained models. Each case is a function on the same
test grid as the cross-framework benchmark (:mod:`pinneapple_tools.benchmark_suite.crossframework`),
built from the benchmark's own reference solutions, so that the ground truth is known exactly and
any machine can rebuild it in seconds:

* ``valid`` cases are the reference plus a smooth, boundary-vanishing perturbation scaled to hit a
  prescribed relative L2 error. They are the negative controls: no error check should fire.
* ``invalid`` cases each inject exactly one defect that a physics check is supposed to catch, and
  that defect is named in the case record (``defect``).

Nothing here depends on a random number generator that is not seeded, on wall-clock time, or on
the machine; running it twice gives byte-identical arrays.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PI = math.pi

#: Relative L2 levels of the valid approximations (negative controls).
VALID_LEVELS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)

#: The one defect each invalid case injects.
DEFECTS = ("bc_offset", "wrong_coefficient", "over_smoothed", "unit_scale", "nonfinite", "zero_field")

#: Tolerance declared for the regression-against-a-stored-reference check (PC701). 5% is a
#: plausible engineering acceptance level; it is declared once, here, before any result is read.
REFERENCE_REL_L2 = 0.05


@dataclass(frozen=True)
class Grid:
    """Axes of a test grid, in the order the checker reads them."""

    axes: tuple[str, ...]
    values: tuple[np.ndarray, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self.values)

    def mesh(self) -> tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.values, indexing="ij"))


@dataclass
class Case:
    case_id: str
    problem: str
    family: str                       # "valid" | "invalid"
    defect: str                       # "none" for valid cases
    target_rel_l2: float              # the level a valid case was built to; measured for invalid ones
    measured_rel_l2: float
    field: np.ndarray
    nans: int = 0
    extra: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- problems
class Problem:
    """One benchmark problem: its grid, its reference solution and its checker declaration.

    ``expressible`` says whether the verification layer's equation set can represent the governing
    equation *exactly*. It is False when the equation (Helmholtz, Burgers, Allen-Cahn) or one of its
    terms (a spatially varying source in the heat equation) has no counterpart in the checker, and in
    that case only the equation-agnostic checks run, plus the tool-independent residual of
    :mod:`run`. Measuring this gap is one of the results, not a limitation of the corpus.
    """

    name: str
    physics: str                      # pinneapple check ``physics`` the case is declared with
    expressible: bool
    grid: Grid

    def exact(self, X: tuple[np.ndarray, ...]) -> np.ndarray:
        raise NotImplementedError

    def checker_parameters(self) -> dict[str, object]:
        return {}

    def checker_units(self) -> dict[str, str]:
        return {}

    def checker_bc(self) -> dict[str, dict[str, float]]:
        return {}

    def checker_ic(self) -> dict[str, object]:
        return {}

    def checker_bounds(self) -> dict[str, list[float]]:
        return {}

    def checker_tolerances(self) -> dict[str, float]:
        return {}

    # a perturbation that vanishes on every declared boundary and at the declared initial time
    def perturbation(self, X: tuple[np.ndarray, ...]) -> np.ndarray:
        raise NotImplementedError

    def wrong_coefficient(self, X: tuple[np.ndarray, ...]) -> np.ndarray:
        """The reference solution of the *same* problem with one physical coefficient changed."""
        raise NotImplementedError

    def write_reference(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{a: v for a, v in zip(self.grid.axes, self.grid.values, strict=False)},
                            T=self.exact(self.grid.mesh()))


def _axes(*spec: tuple[str, float, float, int]) -> Grid:
    return Grid(tuple(s[0] for s in spec), tuple(np.linspace(s[1], s[2], s[3]) for s in spec))


class Poisson2D(Problem):
    """``u_xx + u_yy = f`` on the unit square, ``u = sin(pi x) sin(pi y)`` (the benchmark's sanity case).

    **Not expressible.** Mapped to steady heat conduction ``k lap T + q = 0`` with ``k = 1``, which
    is exact only if ``q = -f`` can be declared; the checker accepts a ``q`` that is a number, not an
    expression of the coordinates, so the closest declaration is ``q = 0``. The residual check then
    fires on the exact solution. The case is kept precisely to measure that, and the residual of
    :mod:`run` supplies the physics answer the tool cannot.
    """

    name = "poisson"
    physics = "heat"
    expressible = False
    grid = _axes(("x", 0.0, 1.0, 101), ("y", 0.0, 1.0, 101))

    def exact(self, X):
        x, y = X
        return np.sin(PI * x) * np.sin(PI * y)

    def source(self, X):
        """The benchmark's ``f`` of ``u_xx + u_yy = f`` at ``X``."""
        x, y = X
        return -2.0 * PI ** 2 * np.sin(PI * x) * np.sin(PI * y)

    def checker_parameters(self):
        return {"k": 1.0, "q": 0.0}          # the closest the checker can represent; see the class note

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": 0.0}, "x=max": {"T": 0.0}, "y=min": {"T": 0.0}, "y=max": {"T": 0.0}}

    def checker_bounds(self):
        return {"T": [-1.5, 1.5]}

    def perturbation(self, X):
        x, y = X
        return np.sin(3 * PI * x) * np.sin(2 * PI * y)

    def wrong_coefficient(self, X):
        x, y = X
        return np.sin(PI * x) * np.sin(PI * y) + 0.18 * np.sin(2 * PI * x) * np.sin(2 * PI * y)


class SteadyHeat2D(Problem):
    """Steady conduction with a **constant** source: ``lap T + 1 = 0`` on the unit square, ``T = (x - x^2)/2``.

    Exactly expressible (``heat`` with ``k = 1`` and ``q = 1``, no time axis), so the residual check
    runs on the true equation. The field does not vary in ``y``, which is why only the ``x`` faces
    carry a boundary condition -- that is the problem, not an omission.
    """

    name = "steady_heat"
    physics = "heat"
    expressible = True
    grid = _axes(("x", 0.0, 1.0, 101), ("y", 0.0, 1.0, 41))

    def exact(self, X):
        x, _ = X
        return 0.5 * (x - x ** 2)

    def checker_parameters(self):
        return {"k": 1.0, "q": 1.0}

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": 0.0}, "x=max": {"T": 0.0}}

    def checker_bounds(self):
        return {"T": [-0.25, 0.25]}

    def perturbation(self, X):
        x, y = X
        return np.sin(3 * PI * x) * np.sin(2 * PI * y)

    def wrong_coefficient(self, X):
        # the solution for q = 1.25 instead of q = 1: right boundary data, wrong source strength
        x, _ = X
        return 0.5 * 1.25 * (x - x ** 2)


class Heat2DTransient(Problem):
    """Manufactured transient conduction, ``dT/dt = alpha lap T`` with ``T = sin(pi x) sin(pi y) e^{-2 pi^2 t}``.

    ``alpha = k / rho_cp = 1``, declared to the checker as ``heat`` with a time axis, so the check
    exercises the ``rho_cp dT/dt`` term and not only the Laplacian.
    """

    name = "heat2d"
    physics = "heat"
    expressible = True
    grid = _axes(("x", 0.0, 1.0, 101), ("y", 0.0, 1.0, 101), ("t", 0.0, 0.1, 21))

    def exact(self, X):
        x, y, t = X
        return np.sin(PI * x) * np.sin(PI * y) * np.exp(-2 * PI ** 2 * t)

    def checker_parameters(self):
        return {"k": 1.0, "rho_cp": 1.0, "q": 0.0}

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": 0.0}, "x=max": {"T": 0.0}, "y=min": {"T": 0.0}, "y=max": {"T": 0.0}}

    def checker_ic(self):
        return {"T": "sin(pi*x)*sin(pi*y)"}

    def checker_bounds(self):
        return {"T": [-1.5, 1.5]}

    def perturbation(self, X):
        x, y, t = X
        return np.sin(3 * PI * x) * np.sin(2 * PI * y) * np.sin(PI * t / 0.1)

    def wrong_coefficient(self, X):
        x, y, t = X
        # the same initial and boundary data, cooling 1.6x faster: right shape, wrong conductivity
        return np.sin(PI * x) * np.sin(PI * y) * np.exp(-2 * PI ** 2 * 1.6 * t)


class Helmholtz2D(Problem):
    """``u_xx + u_yy + k^2 u = q`` on ``[-1,1]^2`` (Wang, Teng & Perdikaris 2021).

    Not expressible in the checker's equation set (the ``+k^2 u`` term has no counterpart), so this
    problem carries only the equation-agnostic checks. That is a measured property of the tool, and
    the paper reports it as such.
    """

    name = "helmholtz"
    physics = "none"
    expressible = False
    grid = _axes(("x", -1.0, 1.0, 101), ("y", -1.0, 1.0, 101))
    a1, a2, k = 1.0, 4.0, 1.0

    def _u(self, X, a2):
        x, y = X
        return np.sin(self.a1 * PI * x) * np.sin(a2 * PI * y)

    def exact(self, X):
        return self._u(X, self.a2)

    def source(self, X):
        """The benchmark's ``q`` of ``u_xx + u_yy + k^2 u = q`` at ``X``."""
        a1, a2, k = self.a1, self.a2, self.k
        x, y = X
        return (k ** 2 - (a1 * PI) ** 2 - (a2 * PI) ** 2) * np.sin(a1 * PI * x) * np.sin(a2 * PI * y)

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": 0.0}, "x=max": {"T": 0.0}, "y=min": {"T": 0.0}, "y=max": {"T": 0.0}}

    def checker_bounds(self):
        return {"T": [-1.5, 1.5]}

    def perturbation(self, X):
        x, y = X
        return np.sin(2 * PI * (x + 1) / 2) * np.sin(6 * PI * (y + 1) / 2)

    def wrong_coefficient(self, X):
        return self._u(X, 2.0)


class Burgers1D(Problem):
    """``u_t + u u_x = nu u_xx``, ``nu = 0.01/pi`` (Raissi, Perdikaris & Karniadakis 2019).

    The reference is the benchmark's Cole-Hopf solution, so a wrong-viscosity case is the reference
    of the same problem at ``3 nu``: same initial and boundary data, different physics.
    """

    name = "burgers"
    physics = "none"
    expressible = False
    grid = _axes(("x", -1.0, 1.0, 256), ("t", 0.0, 1.0, 100))
    nu = 0.01 / PI

    def _u(self, X, nu):
        from pinneapple_physics.closed_form.burgers import burgers_sine_exact
        x, t = X
        return burgers_sine_exact(x, t, nu, n_nodes=120)

    def exact(self, X):
        return self._u(X, self.nu)

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": 0.0}, "x=max": {"T": 0.0}}

    def checker_ic(self):
        return {"T": "-sin(pi*x)"}

    def checker_bounds(self):
        return {"T": [-1.5, 1.5]}

    def perturbation(self, X):
        x, t = X
        return np.sin(2 * PI * (x + 1) / 2) * np.sin(PI * t)

    def wrong_coefficient(self, X):
        return self._u(X, 3 * self.nu)


class AllenCahn1D(Problem):
    """``u_t = d u_xx + 5u - 5u^3``, periodic, ``d = 1e-4`` (Raissi et al. 2019 data-set parameters)."""

    name = "allen_cahn"
    physics = "none"
    expressible = False
    # The benchmark tests on 512 x 201; here the time axis is thinned to 41 points, because this
    # corpus measures which checks fire, not the accuracy of a solution, and the reference
    # interpolation is the most expensive step of the whole corpus.
    grid = _axes(("x", -1.0, 1.0, 512), ("t", 0.0, 1.0, 41))
    d = 1e-4

    def exact(self, X):
        x, t = X
        return _allen_cahn_interp(x, t, self.d)

    def checker_units(self):
        return {"T": "1"}

    def checker_bc(self):
        return {"x=min": {"T": -1.0}, "x=max": {"T": -1.0}}

    def checker_ic(self):
        return {"T": "x**2*cos(pi*x)"}

    def checker_bounds(self):
        return {"T": [-1.5, 1.5]}

    def checker_tolerances(self):
        # The true condition is periodicity, which the checker cannot express; the declared
        # u(+-1) = -1 is what the reference itself satisfies, to 3.1e-3 of the boundary scale.
        # The tolerance is that measured uncertainty rounded up, so the declaration is not
        # stricter than the reference it is checked against.
        return {"bc": 5e-3}

    def perturbation(self, X):
        x, t = X
        return np.sin(2 * PI * (x + 1) / 2) * np.sin(PI * t)

    def wrong_coefficient(self, X):
        x, t = X
        return _allen_cahn_interp(x, t, 2 * self.d)


def _allen_cahn_interp(x: np.ndarray, t: np.ndarray, d: float) -> np.ndarray:
    """The benchmark's own reference interpolation, reused so this corpus and track A share a reference."""
    from pinneapple_tools.benchmark_suite.crossframework.problems import (
        _allen_cahn_interp as interp,
    )
    return interp(np.asarray(x, float).ravel(), np.asarray(t, float).ravel(), d).reshape(np.shape(x))


PROBLEMS: tuple[type[Problem], ...] = (Poisson2D, SteadyHeat2D, Heat2DTransient, Helmholtz2D, Burgers1D,
                                       AllenCahn1D)
BY_NAME = {p.name: p for p in PROBLEMS}


# --------------------------------------------------------------------------- perturbations
def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(np.asarray(a, float).ravel() - np.asarray(b, float).ravel())
    n = np.linalg.norm(np.asarray(b, float).ravel())
    return float(d / n) if n else float("inf")


def _boundary_distance(problem: Problem) -> np.ndarray:
    """Distance to the nearest declared boundary face, in axis fractions, on the problem grid."""
    axes = problem.grid.values
    names = problem.grid.axes
    d = np.full(problem.grid.shape, np.inf)
    spatial = [n for n in names if n != "t"]
    for n in spatial:
        k = names.index(n)
        lo, hi = axes[k][0], axes[k][-1]
        span = hi - lo
        frac = (axes[k] - lo) / span
        shape = [1] * len(names)
        shape[k] = len(axes[k])
        f = frac.reshape(shape)
        np.minimum(d, np.broadcast_to(np.minimum(f, 1 - f), problem.grid.shape), out=d)
    return d


def _smooth(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(a, float), sigma=2.0, mode="nearest")


def _over_smoothed(problem: Problem, ref: np.ndarray) -> np.ndarray:
    """The reference with its small scales removed and its boundary data untouched.

    A Gaussian filter alone also moves the values on the boundary face (``mode="nearest"`` only
    extends the edge value), which would add a boundary-condition defect on top of the spectral
    one. The declared boundary faces are therefore restored from the reference, so this case
    carries exactly one defect: missing high-wavenumber content.
    """
    out = _smooth(ref)
    names = problem.grid.axes
    faces = list(problem.checker_bc())
    if problem.checker_ic():                     # the initial time is data too, and it must survive too
        faces.append("t=min")
    for face in faces:
        axis, side = face.split("=")
        k = names.index(axis)
        sl = [slice(None)] * out.ndim
        sl[k] = 0 if side == "min" else -1
        out[tuple(sl)] = ref[tuple(sl)]
    return out


def build_cases(problem: Problem) -> list[Case]:
    """Every case of one problem: the negative controls and the six single-defect positives."""
    X = problem.grid.mesh()
    ref = np.asarray(problem.exact(X), float)
    norm = float(np.linalg.norm(ref.ravel()))
    cases: list[Case] = [Case(f"{problem.name}/valid_exact", problem.name, "valid", "none", 0.0, 0.0, ref)]

    p = np.asarray(problem.perturbation(X), float)
    p_norm = float(np.linalg.norm(p.ravel()))
    for level in VALID_LEVELS:
        cases.append(Case(f"{problem.name}/valid_{level:g}", problem.name, "valid", "none", level,
                          level, ref + p * (level * norm / p_norm)))

    d = _boundary_distance(problem)
    delta = 0.02
    offset = 0.05 * float(np.ptp(ref))
    cases.append(Case(f"{problem.name}/bc_offset", problem.name, "invalid", "bc_offset", 0.0, 0.0,
                      ref + offset * np.exp(-d / delta)))

    wc = np.asarray(problem.wrong_coefficient(X), float)
    cases.append(Case(f"{problem.name}/wrong_coefficient", problem.name, "invalid", "wrong_coefficient",
                      0.0, 0.0, wc))

    cases.append(Case(f"{problem.name}/over_smoothed", problem.name, "invalid", "over_smoothed", 0.0, 0.0,
                      _over_smoothed(problem, ref)))

    cases.append(Case(f"{problem.name}/unit_scale", problem.name, "invalid", "unit_scale", 0.0, 0.0,
                      ref * 1e3))

    nf = ref.copy()
    idx = np.linspace(0, nf.size - 1, 5).astype(int)
    nf.ravel()[idx] = np.nan
    cases.append(Case(f"{problem.name}/nonfinite", problem.name, "invalid", "nonfinite", 0.0, 0.0, nf,
                      nans=int(np.isnan(nf).sum())))

    cases.append(Case(f"{problem.name}/zero_field", problem.name, "invalid", "zero_field", 0.0, 0.0,
                      np.zeros_like(ref)))

    for c in cases:
        c.extra["reference_rel_l2"] = rel_l2(c.field, ref)
    return cases


# --------------------------------------------------------------------------- writing a case to disk
def checker_toml(problem: Problem, with_reference: bool) -> str:
    """The ``pinneapple-check.toml`` a practitioner would write for this problem."""
    lines = ['select = ["PC201", "PC202", "PC301", "PC401", "PC403", "PC701", "PC704"]', "",
             "[[results]]", 'path = "prediction.npz"', 'kind = "grid"', f'physics = "{problem.physics}"']
    params = problem.checker_parameters()
    if params:
        lines += ["", "[results.parameters]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in params.items()]
    units = problem.checker_units()
    if units:
        lines += ["", "[results.units]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in units.items()]
    bounds = problem.checker_bounds()
    if bounds:
        lines += ["", "[results.bounds]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in bounds.items()]
    bc = problem.checker_bc()
    if bc:
        lines += ["", "[results.bc]"]
        lines += [f"{_toml_key(face)} = {_toml_value(vals)}" for face, vals in bc.items()]
    ic = problem.checker_ic()
    if ic:
        lines += ["", "[results.ic]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in ic.items()]
    tolerances = problem.checker_tolerances()
    if tolerances:
        lines += ["", "[results.tolerances]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in tolerances.items()]
    if with_reference:
        lines += ["", "[results.reference]", 'path = "reference.npz"', f"rel_l2 = {REFERENCE_REL_L2}"]
    return "\n".join(lines) + "\n"


def _toml_key(k: str) -> str:
    """A TOML key, quoted when it is not a bare key (``x=min`` needs the quotes)."""
    import re
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def _toml_value(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def write_case(problem: Problem, case: Case, root: Path) -> Path:
    """Write one case as a directory the checker can be pointed at; returns the directory."""
    d = root / case.case_id.replace("/", "__")
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "prediction.npz",
                        **{a: v for a, v in zip(problem.grid.axes, problem.grid.values, strict=False)}, T=case.field)
    problem.write_reference(d / "reference.npz")
    (d / "pinneapple-check.toml").write_text(checker_toml(problem, with_reference=True))
    (d / "case.json").write_text(json.dumps({
        "case_id": case.case_id, "problem": case.problem, "family": case.family, "defect": case.defect,
        "target_rel_l2": case.target_rel_l2, "measured_rel_l2": case.extra["reference_rel_l2"],
        "nans": case.nans,
    }, indent=2, sort_keys=True) + "\n")
    return d
