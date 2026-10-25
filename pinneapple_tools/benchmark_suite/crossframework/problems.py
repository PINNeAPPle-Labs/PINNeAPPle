"""The forward and inverse PINN problems of the cross-framework benchmark, with their references.

Every framework gets the same problem definition from here: domain, coefficients, source terms, the exact or
reference solution, the shared training point sets, the test points and the fresh residual points. Nothing in this
module depends on a competitor framework.

Problems (``benchmarks/vs_frameworks/PROTOCOL.md``):

* ``burgers``: ``u_t + u u_x = nu u_xx`` on ``x in [-1, 1]``, ``t in [0, 1]``, ``nu = 0.01 / pi``,
  ``u(x, 0) = -sin(pi x)``, ``u(+-1, t) = 0`` (Raissi, Perdikaris, Karniadakis 2019). Reference: the Cole-Hopf
  solution by Gauss-Hermite quadrature, :func:`pinneapple_physics.closed_form.burgers.burgers_sine_exact`.
* ``poisson``: ``u_xx + u_yy = f`` on the unit square, ``u = 0`` on the boundary, manufactured
  ``u = sin(pi x) sin(pi y)``, ``f = -2 pi^2 u``.
* ``helmholtz``: ``u_xx + u_yy + k^2 u = q`` on ``[-1, 1]^2``, ``u = 0`` on the boundary, manufactured
  ``u = sin(a1 pi x) sin(a2 pi y)`` with ``a1 = 1``, ``a2 = 4``, ``k = 1`` (Wang, Teng, Perdikaris 2021, SIAM J. Sci.
  Comput. 43(5), A3055), ``q = (k^2 - (a1 pi)^2 - (a2 pi)^2) u``.
* ``allen_cahn``: ``u_t = d u_xx + 5 u - 5 u^3`` on ``x in [-1, 1]`` periodic, ``t in [0, 1]``, ``d = 1e-4``,
  ``u(x, 0) = x^2 cos(pi x)`` (Raissi et al. 2019, the discrete-time example's data set). Reference:
  :func:`allen_cahn_reference`, Fourier pseudo-spectral in space and ETDRK4 in time
  (:class:`pinneapple_simulation.numerical_solvers.etdrk4.ETDRK4`), checked by a resolution study in the tests.

The residual used by the neutral evaluator is computed by central finite differences from predictions on a
5-point stencil (:func:`residual_from_stencil`), so the same formula applies to every framework.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PINNProblem", "PROBLEMS", "get_problem", "make_point_set", "save_point_set", "load_point_set",
           "test_points", "residual_points", "stencil", "residual_from_stencil", "allen_cahn_reference",
           "point_set_digest", "callable_exact", "describe"]

PI = math.pi


@dataclass(frozen=True)
class PINNProblem:
    """One benchmark problem. ``coords`` are ``("x", "t")`` or ``("x", "y")``; ``bounds`` one ``(lo, hi)`` each."""
    name: str
    coords: tuple[str, str]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    params: dict[str, float]
    time_dependent: bool
    fd_step: float                      # stencil spacing of the finite-difference residual
    citation: str
    test_grid: tuple[int, int]          # points per coordinate of the test grid
    extra: dict[str, object] = field(default_factory=dict)

    # -------------------------------------------------------------- data of the problem
    def initial(self, x: np.ndarray) -> np.ndarray:
        """``u(x, 0)`` of a time-dependent problem."""
        if self.name == "burgers":
            return -np.sin(PI * x)
        if self.name == "allen_cahn":
            return x ** 2 * np.cos(PI * x)
        raise ValueError(f"{self.name} has no initial condition")

    def boundary_value(self, X: np.ndarray) -> np.ndarray:
        """Dirichlet value on the boundary points ``X`` (all problems here have homogeneous or constant data)."""
        if self.name in ("burgers", "poisson", "helmholtz"):
            return np.zeros(len(X))
        if self.name == "allen_cahn":
            # Periodic problem; u(+-1, t) = -1 holds to 1e-3 or better on the reference (see the tests), which is what
            # DeepXDE's official example imposes through its output transform.
            return -np.ones(len(X))
        raise KeyError(self.name)

    def source(self, X: np.ndarray) -> np.ndarray:
        """Right-hand side of the steady problems (``f`` of Poisson, ``q`` of Helmholtz) at ``X`` (N, 2)."""
        x, y = X[:, 0], X[:, 1]
        if self.name == "poisson":
            return -2.0 * PI ** 2 * np.sin(PI * x) * np.sin(PI * y)
        if self.name == "helmholtz":
            a1, a2, k = self.params["a1"], self.params["a2"], self.params["k"]
            return (k ** 2 - (a1 * PI) ** 2 - (a2 * PI) ** 2) * np.sin(a1 * PI * x) * np.sin(a2 * PI * y)
        raise ValueError(f"{self.name} has no source term")

    def exact(self, X: np.ndarray) -> np.ndarray:
        """Reference solution at ``X`` (N, 2), float64."""
        X = np.asarray(X, dtype=np.float64)
        a, b = X[:, 0], X[:, 1]
        if self.name == "burgers":
            from pinneapple_physics.closed_form.burgers import burgers_sine_exact
            return burgers_sine_exact(a, b, self.params["nu"], n_nodes=int(self.extra.get("hermite_nodes", 120)))
        if self.name == "poisson":
            return np.sin(PI * a) * np.sin(PI * b)
        if self.name == "helmholtz":
            return np.sin(self.params["a1"] * PI * a) * np.sin(self.params["a2"] * PI * b)
        if self.name == "allen_cahn":
            return _allen_cahn_interp(a, b, self.params["d"])
        raise KeyError(self.name)

    def residual(self, u: np.ndarray, d: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """PDE residual from the value ``u`` and the derivatives ``d`` (keys ``a``, ``b``, ``aa``, ``bb``; ``a`` is
        the first coordinate, ``b`` the second) at ``X``."""
        if self.name == "burgers":
            return d["b"] + u * d["a"] - self.params["nu"] * d["aa"]
        if self.name == "allen_cahn":
            return d["b"] - self.params["d"] * d["aa"] - 5.0 * u + 5.0 * u ** 3
        if self.name == "poisson":
            return d["aa"] + d["bb"] - self.source(X)
        if self.name == "helmholtz":
            return d["aa"] + d["bb"] + self.params["k"] ** 2 * u - self.source(X)
        raise KeyError(self.name)


PROBLEMS: dict[str, PINNProblem] = {
    "burgers": PINNProblem(
        "burgers", ("x", "t"), ((-1.0, 1.0), (0.0, 1.0)), {"nu": 0.01 / PI}, True, 1e-3,
        "Raissi M., Perdikaris P., Karniadakis G.E. (2019), J. Comput. Phys. 378, 686-707", (256, 100),
        {"hermite_nodes": 120}),
    "poisson": PINNProblem(
        "poisson", ("x", "y"), ((0.0, 1.0), (0.0, 1.0)), {}, False, 1e-2,
        "manufactured solution u = sin(pi x) sin(pi y) (sanity case)", (101, 101)),
    "helmholtz": PINNProblem(
        "helmholtz", ("x", "y"), ((-1.0, 1.0), (-1.0, 1.0)), {"a1": 1.0, "a2": 4.0, "k": 1.0}, False, 1e-2,
        "Wang S., Teng Y., Perdikaris P. (2021), SIAM J. Sci. Comput. 43(5), A3055-A3081", (101, 101)),
    "allen_cahn": PINNProblem(
        "allen_cahn", ("x", "t"), ((-1.0, 1.0), (0.0, 1.0)), {"d": 1e-4}, True, 2e-3,
        "Raissi M., Perdikaris P., Karniadakis G.E. (2019), J. Comput. Phys. 378, 686-707 (data set parameters)",
        (512, 201)),
}


def get_problem(name: str) -> PINNProblem:
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[name]


# --------------------------------------------------------------------------- point sets
def make_point_set(problem: PINNProblem, *, n_interior: int = 10_000, n_boundary: int = 400,
                   seed: int = 20260) -> dict[str, np.ndarray]:
    """The training points every framework loads (protocol A1): ``X_f`` interior collocation points (uniform),
    ``X_b``/``u_b`` boundary points and values, and for time-dependent problems ``X_0``/``u_0`` initial points.

    ``n_boundary`` counts boundary plus initial points. Time-dependent: half on ``t = 0`` and a quarter on each of
    ``x = -1`` and ``x = 1``. Steady: a quarter on each edge. Coordinates are uniform random on each face.
    """
    rng = np.random.default_rng(seed)
    (a0, a1), (b0, b1) = problem.bounds
    X_f = np.stack([rng.uniform(a0, a1, n_interior), rng.uniform(b0, b1, n_interior)], axis=1)
    out: dict[str, np.ndarray] = {"X_f": X_f}
    if problem.time_dependent:
        n0 = n_boundary // 2
        nb = n_boundary - n0
        x0 = rng.uniform(a0, a1, n0)
        X_0 = np.stack([x0, np.full(n0, b0)], axis=1)
        tb = rng.uniform(b0, b1, nb)
        side = np.where(np.arange(nb) < nb // 2, a0, a1)
        X_b = np.stack([side, tb], axis=1)
        out.update(X_0=X_0, u_0=problem.initial(x0)[:, None], X_b=X_b, u_b=problem.boundary_value(X_b)[:, None])
    else:
        q = n_boundary // 4
        sizes = [q, q, q, n_boundary - 3 * q]
        faces = []
        for i, n in enumerate(sizes):
            s = rng.uniform(0.0, 1.0, n)
            if i == 0:
                faces.append(np.stack([np.full(n, a0), b0 + (b1 - b0) * s], 1))
            elif i == 1:
                faces.append(np.stack([np.full(n, a1), b0 + (b1 - b0) * s], 1))
            elif i == 2:
                faces.append(np.stack([a0 + (a1 - a0) * s, np.full(n, b0)], 1))
            else:
                faces.append(np.stack([a0 + (a1 - a0) * s, np.full(n, b1)], 1))
        X_b = np.concatenate(faces)
        out.update(X_b=X_b, u_b=problem.boundary_value(X_b)[:, None], f_f=problem.source(X_f)[:, None])
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def point_set_digest(points: dict[str, np.ndarray]) -> str:
    """SHA-256 over the arrays (sorted by name) so every run can record which point set it trained on."""
    h = hashlib.sha256()
    for k in sorted(points):
        a = np.ascontiguousarray(points[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_point_set(points: dict[str, np.ndarray], path) -> str:
    """Write the point set to ``path`` (.npz) and return its digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **points)
    return point_set_digest(points)


def load_point_set(path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def test_points(problem: PINNProblem) -> np.ndarray:
    """The structured test grid (float64, N x 2): the same points for every framework."""
    (a0, a1), (b0, b1) = problem.bounds
    na, nb = problem.test_grid
    if problem.name == "burgers":
        b1 = 0.99                       # Raissi's grid: 256 x on [-1, 1], 100 t on [0, 0.99]
    if problem.name == "allen_cahn":
        a = np.linspace(a0, a1, na, endpoint=False)   # periodic grid
    else:
        a = np.linspace(a0, a1, na)
    b = np.linspace(b0, b1, nb)
    A, B = np.meshgrid(a, b, indexing="ij")
    return np.stack([A.ravel(), B.ravel()], axis=1)


def residual_points(problem: PINNProblem, n: int = 2000, seed: int = 777) -> np.ndarray:
    """Fresh interior points (never used in training; a different generator stream) for the residual, kept one
    stencil step away from the boundary."""
    rng = np.random.default_rng(seed)
    h = problem.fd_step
    (a0, a1), (b0, b1) = problem.bounds
    return np.stack([rng.uniform(a0 + h, a1 - h, n), rng.uniform(b0 + h, b1 - h, n)], axis=1)


def stencil(problem: PINNProblem, X: np.ndarray) -> np.ndarray:
    """The 5-point stencil around each residual point: (N, 5, 2) = center, a+h, a-h, b+h, b-h. A runner predicts on
    ``stencil(...).reshape(-1, 2)`` and exports the values; :func:`residual_from_stencil` turns them into the
    residual."""
    h = problem.fd_step
    off = np.array([[0, 0], [h, 0], [-h, 0], [0, h], [0, -h]], dtype=np.float64)
    return X[:, None, :] + off[None, :, :]


def residual_from_stencil(problem: PINNProblem, X: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Residual at ``X`` from the stencil values (N, 5) by second-order central differences (float64)."""
    v = np.asarray(values, dtype=np.float64).reshape(len(X), 5)
    h = problem.fd_step
    c = v[:, 0]
    d = {"a": (v[:, 1] - v[:, 2]) / (2 * h), "b": (v[:, 3] - v[:, 4]) / (2 * h),
         "aa": (v[:, 1] - 2 * c + v[:, 2]) / h ** 2, "bb": (v[:, 3] - 2 * c + v[:, 4]) / h ** 2}
    return problem.residual(c, d, X)


# --------------------------------------------------------------------------- Allen-Cahn reference
_AC_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def allen_cahn_reference(d: float = 1e-4, n: int = 2048, dt: float = 1e-4, t_end: float = 1.0,
                         n_save: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, t, u)`` of ``u_t = d u_xx + 5 u - 5 u^3``, periodic on ``[-1, 1)``, ``u(x, 0) = x^2 cos(pi x)``.

    Fourier pseudo-spectral in space (``n`` modes, the linear part ``-d k^2`` integrated exactly), ETDRK4 in time
    (Cox & Matthews 2002; Kassam & Trefethen 2005). ``u`` is (n_save + 1, n) at ``t = linspace(0, t_end, n_save + 1)``.
    """
    key = (float(d), int(n), float(dt), float(t_end), int(n_save))
    if key in _AC_CACHE:
        return _AC_CACHE[key]
    from pinneapple_simulation.numerical_solvers.etdrk4 import ETDRK4, fourier_wavenumbers

    x = np.linspace(-1.0, 1.0, n, endpoint=False)
    k = fourier_wavenumbers(n, 2.0)
    L = -d * k ** 2

    def nonlinear(uh, _t):
        u = np.real(np.fft.ifft(uh))
        return np.fft.fft(5.0 * u - 5.0 * u ** 3)

    steps = int(round(t_end / dt))
    every = steps // n_save
    if every * n_save != steps:
        raise ValueError("t_end / dt must be a multiple of n_save")
    res = ETDRK4(L, nonlinear, dt).solve(np.fft.fft(x ** 2 * np.cos(PI * x)), t_end, save_every=every)
    if res.blew_up is not None:
        raise FloatingPointError(str(res.blew_up))
    u = np.real(np.fft.ifft(res.u, axis=-1))
    out = (x, res.t, u)
    _AC_CACHE[key] = out
    return out


def _allen_cahn_interp(x: np.ndarray, t: np.ndarray, d: float) -> np.ndarray:
    """Reference at arbitrary points: trigonometric interpolation in x, linear interpolation in t between the
    1001 saved states (dt_save = 1e-3; the tests bound this error)."""
    xs, ts, U = allen_cahn_reference(d)
    n = len(xs)
    Uh = np.fft.fft(U, axis=1) / n                                     # (nt, n)
    k = np.fft.fftfreq(n, 1.0 / n)                                      # integer modes on a period of 2
    k[n // 2] = 0.0                                                     # drop the (tiny) Nyquist mode
    # u(x) = sum_k Uh_k exp(i pi k (x + 1))
    j = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
    w = (t - ts[j]) / (ts[j + 1] - ts[j])
    out = np.empty(len(x))
    chunk = 4096
    for s in range(0, len(x), chunk):
        e = np.exp(1j * PI * np.outer(x[s:s + chunk] + 1.0, k))         # (m, n)
        lo = np.real(np.sum(e * Uh[j[s:s + chunk]], axis=1))
        hi = np.real(np.sum(e * Uh[j[s:s + chunk] + 1], axis=1))
        out[s:s + chunk] = (1 - w[s:s + chunk]) * lo + w[s:s + chunk] * hi
    return out


def callable_exact(problem: PINNProblem) -> Callable[[np.ndarray], np.ndarray]:
    """``X -> u`` column vector, for APIs that take a reference callable (``pp.compare(..., reference="exact")``)."""
    return lambda X: problem.exact(np.asarray(X))[:, None]


def describe(problem: PINNProblem) -> dict[str, object]:
    """JSON-ready description (stored in every result file)."""
    return {"name": problem.name, "coords": list(problem.coords), "bounds": [list(b) for b in problem.bounds],
            "params": dict(problem.params), "citation": problem.citation, "fd_step": problem.fd_step,
            "test_grid": list(problem.test_grid)}
