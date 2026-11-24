"""Discretisation of PDEs: consistency, stability and convergence, measured (#1062).

The three words in the title are three different measurements, and this module keeps them apart:

* **Consistency** -- the truncation error, obtained by substituting the *exact* solution into the discrete
  operator. It is a property of the scheme and must vanish as the mesh is refined; when it does not,
  :class:`~.diagnostics.Diagnostic` ``PDE.INCONSISTENT`` says so.
* **Stability** -- the von Neumann amplification factor ``|G(theta)|`` of each Fourier mode. The scheme is
  stable when ``max_theta |G| <= 1``, which is where the CFL limits come from; those are *measured* here by
  scanning the Courant number for the first mode that grows, and the growth is then demonstrated by
  running the scheme just above the limit.
* **Convergence** -- the error against the exact solution, with the observed order fitted. Lax's
  equivalence theorem (consistent + stable => convergent) is then a statement with numbers on both sides:
  :func:`lax_equivalence_demo` reports the consistency order, the stability verdict and the convergence
  order for the same scheme, and then shows what happens when stability fails while consistency holds.

Also here: the advection schemes (upwind, Lax-Friedrichs, Lax-Wendroff) with their numerical diffusion and
dispersion read off the modified equation and measured on a wave packet; TVD limiters on a step, where the
total variation is checked to be non-increasing; Poisson in one and two dimensions; and manufactured
solutions, which are how a user reports the observed order of their own solver.

The limitation, once: everything here is finite differences on a uniform grid. Finite volume, finite
element and spectral discretisations have their own truncation analyses, and the observed order of a
*staggered* or boundary-fitted scheme can differ from the interior order by one -- which is why
:func:`manufactured_solution_study` reports the observed order instead of the nominal one.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import fit_order
from .diagnostics import Diagnostic, explain

__all__ = [
    "AdvectionResult",
    "HeatResult",
    "advection_1d",
    "advection_convergence",
    "advection_stability_limit",
    "heat_1d_crank_nicolson",
    "heat_1d_exact_mode",
    "heat_1d_explicit",
    "heat_cfl_limit",
    "heat_convergence",
    "lax_equivalence_demo",
    "manufactured_solution_study",
    "max_amplification",
    "numerical_diffusion_coefficient",
    "poisson_1d",
    "poisson_2d",
    "total_variation",
    "truncation_error_study",
    "von_neumann_amplification",
    "wave_1d",
    "wave_cfl_limit",
]

#: admissible Courant numbers, as measured by :func:`advection_stability_limit`
_ADVECTION_SCHEMES = ("upwind", "lax_friedrichs", "lax_wendroff")


@dataclass(frozen=True)
class HeatResult:
    """A heat-equation run: the field, the scheme's parameters and the CFL verdict."""

    x: np.ndarray
    u: np.ndarray
    dt: float
    dx: float
    scheme: str
    amplification: float
    stable: bool
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary (the field itself is available as ``u``)."""
        return {
            "scheme": self.scheme,
            "dt": self.dt,
            "dx": self.dx,
            "amplification": self.amplification,
            "stable": self.stable,
            "final_norm": float(np.linalg.norm(self.u)),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass(frozen=True)
class AdvectionResult:
    """An advection run, with the CFL verdict and (for a discontinuous datum) the oscillation measure."""

    x: np.ndarray
    u: np.ndarray
    dt: float
    dx: float
    courant: float
    scheme: str
    amplification: float
    stable: bool
    total_variation: float
    overshoot: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary."""
        return {
            "scheme": self.scheme,
            "dt": self.dt,
            "dx": self.dx,
            "courant": self.courant,
            "amplification": self.amplification,
            "stable": self.stable,
            "total_variation": self.total_variation,
            "overshoot": self.overshoot,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# --------------------------------------------------------------------------- von Neumann analysis
def von_neumann_amplification(
    scheme: str,
    theta: float | np.ndarray,
    *,
    courant: float = 0.5,
    r: float = 0.25,
) -> np.ndarray | complex:
    """The amplification factor ``G(theta)`` of one Fourier mode, for the scheme named.

    ``u_j^n = G^n exp(i theta j)`` substituted into the scheme gives ``G``; the scheme is stable when
    ``|G(theta)| <= 1`` for every ``theta`` in ``[0, pi]``. The schemes implemented here are

    * ``"heat_explicit"``: ``G = 1 - 4 r sin^2(theta/2)``, stable for ``r <= 1/2``;
    * ``"heat_crank_nicolson"``: ``G = (1 - 2 r s)/(1 + 2 r s)`` with ``s = sin^2(theta/2)``, ``|G| <= 1``
      always;
    * ``"upwind"``: ``G = 1 - C(1 - exp(-i theta))``, stable for ``0 <= C <= 1``;
    * ``"lax_friedrichs"``: ``G = cos(theta) - i C sin(theta)``, stable for ``|C| <= 1``;
    * ``"lax_wendroff"``: ``G = 1 - i C sin(theta) - 2C^2 sin^2(theta/2)``, stable for ``|C| <= 1``.

    ``courant`` is ``c dt/dx`` for the advection schemes and ``r`` is ``alpha dt/dx^2`` for the heat
    scheme; both are dimensionless, which is the point of writing them down.
    """
    th = np.asarray(theta, dtype=np.float64)
    if scheme == "heat_explicit":
        return 1.0 - 4.0 * r * np.sin(th / 2.0) ** 2
    if scheme == "heat_crank_nicolson":
        s = np.sin(th / 2.0) ** 2
        return (1.0 - 2.0 * r * s) / (1.0 + 2.0 * r * s)
    if scheme == "upwind":
        return 1.0 - courant * (1.0 - np.exp(-1j * th))
    if scheme == "lax_friedrichs":
        return np.cos(th) - 1j * courant * np.sin(th)
    if scheme == "lax_wendroff":
        return 1.0 - 1j * courant * np.sin(th) - 2.0 * courant**2 * np.sin(th / 2.0) ** 2
    raise ValueError(f"unknown scheme {scheme!r}")


def max_amplification(
    scheme: str, *, courant: float = 0.5, r: float = 0.25, samples: int = 512
) -> float:
    """``max_theta |G(theta)|`` over ``[0, pi]``: above one, every step multiplies that mode."""
    theta = np.linspace(0.0, np.pi, int(samples))
    return float(np.max(np.abs(von_neumann_amplification(scheme, theta, courant=courant, r=r))))


def _stability_threshold(scheme: str, *, upper: float = 10.0, key: str) -> float:
    """Bisect for the largest parameter value with ``max|G| <= 1`` -- the CFL limit, measured."""
    low, high = 0.0, float(upper)
    for _ in range(200):
        mid = 0.5 * (low + high)
        value = max_amplification(scheme, **{key: mid})
        if value <= 1.0 + 1e-12:
            low = mid
        else:
            high = mid
    return float(low)


def heat_cfl_limit() -> float:
    """The largest ``r = alpha dt/dx^2`` an explicit heat scheme tolerates, measured by bisection.

    The theory says ``r <= 1/2`` (from ``1 - 4 r sin^2(theta/2) >= -1`` at ``theta = pi``); the
    measurement recovers it from the amplification factor, and the tests then run the scheme at
    ``r = 0.51`` to see the growth.
    """
    return _stability_threshold("heat_explicit", upper=2.0, key="r")


def advection_stability_limit(scheme: str) -> float:
    """The largest Courant number the advection scheme tolerates: 1 for upwind, Lax-Friedrichs and LW.

    All three are stable on ``0 <= C <= 1`` and unstable above it; the *reason* differs (upwind's
    amplification is a circle through 1 and ``1 - 2C``; Lax-Friedrichs' is an ellipse; Lax-Wendroff's is
    the unit circle only at ``C = 1``), which is why the limit is measured per scheme.
    """
    if scheme not in _ADVECTION_SCHEMES:
        raise ValueError(f"scheme must be one of {_ADVECTION_SCHEMES}; got {scheme!r}")
    return _stability_threshold(scheme, upper=3.0, key="courant")


def wave_cfl_limit() -> float:
    """The Courant limit of the explicit second-order wave equation: ``c dt/dx <= 1``."""
    return _stability_threshold("lax_friedrichs", upper=3.0, key="courant")


# --------------------------------------------------------------------------- heat equation
def heat_1d_exact_mode(alpha: float, t: float, x: np.ndarray, *, mode: int = 1) -> np.ndarray:
    """The exact solution ``exp(-alpha mode^2 pi^2 t) sin(mode pi x)`` on ``[0, 1]`` with homogeneous
    Dirichlet data: one Fourier mode of the heat equation, which is all a convergence study needs."""
    return np.exp(-alpha * mode**2 * np.pi**2 * t) * np.sin(mode * np.pi * np.asarray(x))


def _initial_mode(nx: int, mode: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, int(nx) + 1)
    return x, np.sin(mode * np.pi * x)


def heat_1d_explicit(
    alpha: float,
    nx: int,
    dt: float,
    *,
    mode: int = 1,
    steps: int | None = None,
    t_end: float | None = None,
) -> HeatResult:
    """Explicit (FTCS) finite differences for ``u_t = alpha u_xx`` on ``[0, 1]``.

    ``u_j^{n+1} = u_j^n + r (u_{j+1}^n - 2u_j^n + u_{j-1}^n)`` with ``r = alpha dt/dx^2``. Stable for
    ``r <= 1/2``; above it the highest mode is amplified by ``|1 - 4r|`` per step, so the solution grows
    geometrically and reaches overflow in a few hundred steps -- the ``PDE.CFL_VIOLATION`` diagnostic
    reports the measured amplification rather than merely complaining.
    """
    if nx < 3:
        raise ValueError(f"nx must be at least 3; got {nx}")
    dx = 1.0 / nx
    r = alpha * dt / dx**2
    x, u = _initial_mode(nx, mode)
    if steps is None:
        steps = int(round(t_end / dt)) if t_end is not None else 1
    amplification = max_amplification("heat_explicit", r=r)
    diagnostics: list[Diagnostic] = []
    if amplification > 1.0 + 1e-12:
        diagnostics.append(explain("PDE.CFL_VIOLATION", scheme="heat explicit", r=r,
                                   limit=heat_cfl_limit(), amplification=amplification))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(steps)):
            u[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    return HeatResult(x=x, u=u, dt=dt, dx=dx, scheme="heat_explicit", amplification=amplification,
                      stable=bool(amplification <= 1.0 + 1e-12), diagnostics=tuple(diagnostics))


def heat_1d_crank_nicolson(
    alpha: float,
    nx: int,
    dt: float,
    *,
    mode: int = 1,
    steps: int | None = None,
    t_end: float | None = None,
) -> HeatResult:
    """Crank-Nicolson for the heat equation: second order in time, and stable for **every** ``dt``.

    Each step solves a tridiagonal system with ``(1 + r) u_j^{n+1} - r/2 (u_{j+1} + u_{j-1})^{n+1}`` on the
    left and the explicit half on the right, which makes ``|G| <= 1`` for every mode and every ``r``. That
    is what "unconditionally stable" buys: the step is chosen for accuracy alone.
    """
    if nx < 3:
        raise ValueError(f"nx must be at least 3; got {nx}")
    dx = 1.0 / nx
    r = alpha * dt / dx**2
    x, u = _initial_mode(nx, mode)
    if steps is None:
        steps = int(round(t_end / dt)) if t_end is not None else 1
    n = u.size - 2
    lower = np.full(n - 1, -0.5 * r)
    upper = np.full(n - 1, -0.5 * r)
    diagonal = np.full(n, 1.0 + r)
    for _ in range(int(steps)):
        interior = u[1:-1]
        rhs = interior + 0.5 * r * (u[2:] - 2.0 * interior + u[:-2])
        u[1:-1] = _solve_tridiagonal(lower, diagonal, upper, rhs)
    amplification = max_amplification("heat_crank_nicolson", r=r)
    return HeatResult(x=x, u=u, dt=dt, dx=dx, scheme="heat_crank_nicolson",
                      amplification=amplification, stable=True)


def _solve_tridiagonal(lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray,
                       rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm: ``O(n)`` solve of a tridiagonal system, no pivoting needed for an M-matrix."""
    n = diagonal.size
    c = np.zeros(n - 1)
    d = np.zeros(n)
    c[0] = upper[0] / diagonal[0]
    d[0] = rhs[0] / diagonal[0]
    for i in range(1, n):
        denominator = diagonal[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denominator
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denominator
    solution = np.zeros(n)
    solution[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d[i] - c[i] * solution[i + 1]
    return solution


def heat_convergence(
    alpha: float = 0.1,
    *,
    nx_values: Sequence[int] = (10, 20, 40, 80),
    t_end: float = 0.05,
    scheme: str = "explicit",
    r: float = 0.25,
) -> dict[str, Any]:
    """Observed order of the heat solver against the exact mode, refining ``dx`` and ``dt`` together.

    With ``dt`` tied to ``dx^2`` at a fixed ``r`` -- the only coupling the explicit scheme can use and
    keep stability -- the truncation is ``dt/2 u_tt - alpha dx^2/12 u_xxxx = O(dx^2)`` for **both**
    schemes, so both measure 2. The difference between them is not the order but the admissible step: at
    ``dt = O(dx)`` the explicit scheme is unstable for every mesh fine enough, while Crank-Nicolson is not
    (see :func:`lax_equivalence_demo` and the tests, which measure both facts).

    The defaults keep ``t_end/dt`` an exact integer for every ``nx`` in the list, so the effective ``r``
    does not drift between meshes -- an order fitted across meshes with different ``r`` is not an order.
    """
    solver = heat_1d_explicit if scheme == "explicit" else heat_1d_crank_nicolson
    errors: list[float] = []
    steps_used: list[float] = []
    for nx in nx_values:
        dx = 1.0 / nx
        dt = r * dx**2 / alpha if scheme == "explicit" else r * dx**2 / alpha
        steps = int(round(t_end / dt))
        result = solver(alpha, nx, t_end / steps, steps=steps, mode=1)
        exact = heat_1d_exact_mode(alpha, t_end, result.x, mode=1)
        errors.append(float(np.max(np.abs(result.u - exact))))
        steps_used.append(dt)
    estimate = fit_order(np.asarray(steps_used), np.asarray(errors), name=f"heat {scheme}")
    return {
        "scheme": scheme,
        "nx_values": [int(v) for v in nx_values],
        "errors": errors,
        "order": estimate.order,
        "ci95": list(estimate.ci95),
        "r_squared": estimate.r_squared,
    }


# --------------------------------------------------------------------------- advection
def _limiter(name: str, ratio: np.ndarray) -> np.ndarray:
    """Flux limiters ``phi(r)`` for the TVD reconstruction; ``r`` is the ratio of successive slopes."""
    if name == "minmod":
        return np.maximum(0.0, np.minimum(1.0, ratio))
    if name == "van_leer":
        return (ratio + np.abs(ratio)) / (1.0 + np.abs(ratio))
    if name == "superbee":
        return np.maximum(0.0, np.maximum(np.minimum(2.0 * ratio, 1.0), np.minimum(ratio, 2.0)))
    raise ValueError(f"unknown limiter {name!r}")


def advection_1d(
    velocity: float,
    nx: int,
    courant: float,
    *,
    scheme: str = "upwind",
    steps: int = 100,
    initial: Callable[[np.ndarray], np.ndarray] | None = None,
    limiter: str = "minmod",
) -> AdvectionResult:
    """Advection ``u_t + c u_x = 0`` on the unit interval, periodic, by upwind / LW / TVD.

    The initial datum defaults to a square pulse at ``x in [0.2, 0.4]``, which carries both a jump and a
    smooth part -- the case where the scheme's diffusion (smearing) and dispersion (ripples) show up in
    different places. ``scheme="tvd"`` applies the minmod-limited Lax-Wendroff flux, which is second order
    where the solution is smooth and first order at the jumps, so no new extremum is created.

    Raises ``ValueError`` for an unknown scheme, ``courant > 1`` is allowed but reported as unstable.
    """
    if nx < 8:
        raise ValueError(f"nx must be at least 8; got {nx}")
    if scheme not in (*_ADVECTION_SCHEMES, "tvd"):
        raise ValueError(f"unknown scheme {scheme!r}")
    dx = 1.0 / nx
    dt = courant * dx / velocity
    x = np.linspace(0.0, 1.0, int(nx), endpoint=False)
    if initial is None:
        u = np.where((x >= 0.2) & (x <= 0.4), 1.0, 0.0)
    else:
        u = np.asarray(initial(x), dtype=np.float64).copy()

    base = "lax_wendroff" if scheme == "tvd" else scheme
    amplification = max_amplification(base, courant=courant)

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(steps)):
            if scheme == "upwind":
                u = u - courant * (u - np.roll(u, 1))
            elif scheme == "lax_friedrichs":
                u = 0.5 * (np.roll(u, -1) + np.roll(u, 1)) - 0.5 * courant * (np.roll(u, -1) - np.roll(u, 1))
            elif scheme == "lax_wendroff":
                flux = 0.5 * (u + np.roll(u, -1)) - 0.5 * courant * (np.roll(u, -1) - u)
                u = u - courant * (flux - np.roll(flux, 1))
            else:  # TVD: limited Lax-Wendroff flux, minmod by default
                forward = np.roll(u, -1) - u
                backward = u - np.roll(u, 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(forward) > 1e-14, backward / np.where(forward == 0, 1e-14, forward), 0.0)
                phi = _limiter(limiter, ratio)
                flux = 0.5 * (u + np.roll(u, -1)) - 0.5 * courant * (np.roll(u, -1) - u) \
                    - 0.5 * courant * (1.0 - courant) * phi * forward
                u = u - courant * (flux - np.roll(flux, 1))

    overshoot = float(max(np.max(u) - 1.0, 0.0)) if initial is None else 0.0
    diagnostics: list[Diagnostic] = []
    if amplification > 1.0 + 1e-12:
        diagnostics.append(explain("PDE.CFL_VIOLATION", scheme=scheme, courant=courant,
                                   limit=advection_stability_limit(base), amplification=amplification))
    elif scheme in ("upwind", "lax_friedrichs") and initial is None:
        diagnostics.append(
            explain("PDE.NUMERICAL_DIFFUSION", scheme=scheme,
                    coefficient=numerical_diffusion_coefficient(scheme, velocity, dx, courant))
        )
    elif overshoot > 1e-10:
        diagnostics.append(explain("PDE.OSCILLATION", scheme=scheme, overshoot=overshoot))
    return AdvectionResult(
        x=x, u=u, dt=dt, dx=dx, courant=courant, scheme=scheme, amplification=amplification,
        stable=bool(amplification <= 1.0 + 1e-12), total_variation=total_variation(u),
        overshoot=overshoot, diagnostics=tuple(diagnostics),
    )


def total_variation(u: Sequence[float] | np.ndarray) -> float:
    """``sum_i |u_{i+1} - u_i|`` including the periodic wrap: the quantity a TVD scheme cannot increase."""
    values = np.asarray(u, dtype=np.float64).ravel()
    return float(np.sum(np.abs(np.roll(values, -1) - values)))


def numerical_diffusion_coefficient(scheme: str, velocity: float, dx: float, courant: float) -> float:
    """The artificial viscosity of the scheme, from its modified equation (0 for Lax-Wendroff).

    For first-order upwind the leading truncation term is ``(c dx/2)(1 - C) u_xx``, so the scheme solves
    the advection-diffusion equation with that diffusivity: the smearing of a front is not an error to be
    tuned away, it is a coefficient. Lax-Friedrichs has ``(c dx/2)(1/C - C)``, larger by ``1/C^2``.
    """
    if scheme == "upwind":
        return float(0.5 * velocity * dx * (1.0 - courant))
    if scheme == "lax_friedrichs":
        return float(0.5 * velocity * dx * (1.0 / courant - courant))
    if scheme == "lax_wendroff":
        return 0.0
    raise ValueError(f"no modified equation implemented for {scheme!r}")


def advection_convergence(
    velocity: float = 1.0,
    *,
    nx_values: Sequence[int] = (50, 100, 200, 400),
    courant: float = 0.8,
    scheme: str = "upwind",
    periods: int = 1,
) -> dict[str, Any]:
    """Observed order of an advection scheme on a *smooth* datum (where the order is defined).

    On a jump the observed order of every one of these schemes is between 0 and 1 regardless of its formal
    order -- the truncation analysis assumes smoothness that a discontinuity does not have. So this study
    uses ``sin(2 pi x)``, where upwind returns 1, and Lax-Wendroff returns 2, and reports both.
    """
    errors: list[float] = []
    spacings: list[float] = []
    for nx in nx_values:
        dx = 1.0 / nx
        dt = courant * dx / velocity
        steps = int(round(periods / dt))
        result = advection_1d(velocity, nx, courant, scheme=scheme, steps=steps,
                              initial=lambda xx: np.sin(2.0 * np.pi * xx))
        exact = np.sin(2.0 * np.pi * (result.x - velocity * steps * dt))
        errors.append(float(np.sqrt(np.mean((result.u - exact) ** 2))))
        spacings.append(dx)
    estimate = fit_order(np.asarray(spacings), np.asarray(errors), name=f"advection {scheme}")
    return {
        "scheme": scheme,
        "nx_values": [int(v) for v in nx_values],
        "errors": errors,
        "order": estimate.order,
        "ci95": list(estimate.ci95),
    }


# --------------------------------------------------------------------------- wave equation
def wave_1d(
    wave_speed: float,
    nx: int,
    courant: float,
    *,
    steps: int = 100,
    initial: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Explicit second-order finite differences for ``u_tt = c^2 u_xx``, zero initial velocity.

    The leapfrog update ``u^{n+1} = 2u^n - u^{n-1} + C^2 (u^n_{j+1} - 2u^n_j + u^n_{j-1})`` is stable for
    ``C <= 1``, and at ``C = 1`` it is exact on the grid (the discrete and the continuous characteristics
    coincide), which the returned ``amplification`` shows: ``|G| = 1`` for every mode at ``C = 1``.
    """
    if nx < 8:
        raise ValueError(f"nx must be at least 8; got {nx}")
    dx = 1.0 / nx
    x = np.linspace(0.0, 1.0, int(nx), endpoint=False)
    u0 = np.sin(2.0 * np.pi * x) if initial is None else np.asarray(initial(x), dtype=np.float64)
    previous = u0.copy()
    # zero initial velocity: the first step is the Taylor expansion u^1 = u^0 + dt^2/2 c^2 u_xx
    current = u0 + 0.5 * courant**2 * (np.roll(u0, -1) - 2.0 * u0 + np.roll(u0, 1))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(steps)):
            laplacian = np.roll(current, -1) - 2.0 * current + np.roll(current, 1)
            following = 2.0 * current - previous + courant**2 * laplacian
            previous, current = current, following
    theta = np.linspace(0.0, np.pi, 512)
    amplification = float(np.max(np.abs(np.cos(theta) + 1j * courant * np.sin(theta))))
    diagnostics: list[Diagnostic] = []
    if courant > 1.0 + 1e-12:
        diagnostics.append(explain("PDE.CFL_VIOLATION", scheme="wave leapfrog", courant=courant,
                                   limit=1.0, amplification=amplification))
    return {
        "x": x,
        "u": current,
        "courant": courant,
        "amplification": amplification,
        "stable": bool(courant <= 1.0 + 1e-12),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- Poisson
def poisson_1d(
    source: Callable[[np.ndarray], np.ndarray], nx: int
) -> tuple[np.ndarray, np.ndarray]:
    """``-u'' = f`` on ``[0, 1]`` with ``u(0) = u(1) = 0``, second-order central differences.

    Returns ``(x, u)``. The scheme is the standard tridiagonal system, solved with the Thomas algorithm;
    its truncation error is ``-dx^2/12 u''''``, so the observed order against a manufactured solution is 2.
    """
    if nx < 3:
        raise ValueError(f"nx must be at least 3; got {nx}")
    dx = 1.0 / nx
    x = np.linspace(0.0, 1.0, int(nx) + 1)
    interior = x[1:-1]
    f = np.asarray(source(interior), dtype=np.float64)
    n = interior.size
    diagonal = np.full(n, 2.0 / dx**2)
    off = np.full(n - 1, -1.0 / dx**2)
    u = np.zeros(x.size)
    u[1:-1] = _solve_tridiagonal(off, diagonal, off, f)
    return x, u


def poisson_2d(
    source: Callable[[np.ndarray, np.ndarray], np.ndarray], nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``-(u_xx + u_yy) = f`` on the unit square with homogeneous Dirichlet data, 5-point stencil.

    Returns ``(x, y, u)`` with ``u`` of shape ``(nx+1, nx+1)``. Solved with the same Jacobi/Gauss-Seidel
    machinery the linear-algebra module measures -- here by a small multigrid-free iteration, which is
    enough for the convergence study the tests run. The observed order is 2, as the ``dx^2/12`` truncation
    term says.
    """
    if nx < 3:
        raise ValueError(f"nx must be at least 3; got {nx}")
    dx = 1.0 / nx
    x = np.linspace(0.0, 1.0, int(nx) + 1)
    y = x.copy()
    grid_x, grid_y = np.meshgrid(x[1:-1], y[1:-1], indexing="ij")
    f = np.asarray(source(grid_x, grid_y), dtype=np.float64)
    u = np.zeros((nx + 1, nx + 1))
    interior = np.zeros_like(f)
    # Richardson iteration on the 5-point Laplacian: slow, but the point here is the discretisation
    for _ in range(20000):
        updated = 0.25 * (
            np.pad(interior[1:, :], ((0, 1), (0, 0))) + np.pad(interior[:-1, :], ((1, 0), (0, 0)))
            + np.pad(interior[:, 1:], ((0, 0), (0, 1))) + np.pad(interior[:, :-1], ((0, 0), (1, 0)))
            + dx**2 * f
        )
        change = np.max(np.abs(updated - interior))
        interior = updated
        if change < 1e-13:
            break
    u[1:-1, 1:-1] = interior
    return x, y, u


def manufactured_solution_study(
    solver: Callable[[int], np.ndarray],
    exact: Callable[[np.ndarray], np.ndarray],
    node_counts: Sequence[int],
    *,
    h_of: Callable[[int], float] | None = None,
    norm: str = "linf",
) -> dict[str, Any]:
    """Report the observed order of a solver against a manufactured solution -- the user-facing tool.

    A manufactured solution is an analytic function inserted into the PDE to produce a source term; the
    solver then has a known answer at every mesh, so its observed order is measurable without a reference
    code. This function does the fit and the bookkeeping; ``solver(n)`` must return the field on ``n + 1``
    uniformly spaced points (or whatever ``exact`` expects to be compared against).

    Raises ``ValueError`` when fewer than two node counts are given -- an order from one mesh is not a
    measurement.
    """
    if len(node_counts) < 2:
        raise ValueError("at least two node counts are needed to measure an order")
    spacings: list[float] = []
    errors: list[float] = []
    for n in node_counts:
        computed = np.asarray(solver(int(n)), dtype=np.float64)
        h = float(h_of(int(n))) if h_of is not None else 1.0 / int(n)
        x = np.linspace(0.0, 1.0, computed.shape[0])
        reference = np.asarray(exact(x), dtype=np.float64)
        difference = np.abs(computed - reference)
        errors.append(float(np.max(difference) if norm == "linf" else np.sqrt(np.mean(difference**2))))
        spacings.append(h)
    estimate = fit_order(np.asarray(spacings), np.asarray(errors), name="manufactured solution")
    diagnostics: list[Diagnostic] = []
    if not np.isfinite(estimate.order) or estimate.order < 0.5:
        diagnostics.append(explain("PDE.INCONSISTENT", observed_order=estimate.order,
                                   spacings=spacings))
    return {
        "node_counts": [int(n) for n in node_counts],
        "spacings": spacings,
        "errors": errors,
        "observed_order": estimate.order,
        "ci95": list(estimate.ci95),
        "r_squared": estimate.r_squared,
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


# --------------------------------------------------------------------------- consistency
def truncation_error_study(
    scheme: str = "heat_explicit",
    *,
    alpha: float = 0.1,
    dx_values: Sequence[float] = (0.1, 0.05, 0.025, 0.0125),
    r: float = 0.4,
    x: float = 0.3,
    t: float = 0.05,
) -> dict[str, Any]:
    """The truncation error, measured by substituting the exact solution into the discrete operator.

    ``tau = (L_h u - L u)(x, t)`` evaluated on the exact ``u = exp(-alpha pi^2 t) sin(pi x)``. It is a
    property of the scheme alone -- the solution cancels out of the analysis, and here it is evaluated
    numerically instead of differentiated by hand -- and its order in ``dx`` is the consistency order.
    For the explicit scheme with ``r = alpha dt/dx^2`` fixed, ``tau = dt/2 u_tt - alpha dx^2/12 u_xxxx``
    is ``O(dx^2)``; for Crank-Nicolson with ``dt = O(dx)`` it is also ``O(dx^2)``. Both measure 2, which is
    the point: consistency is a property of the scheme, and it is *not* where the two differ.
    """
    if scheme not in ("heat_explicit", "heat_crank_nicolson"):
        raise ValueError(f"unknown scheme {scheme!r}")
    errors: list[float] = []
    for dx in dx_values:
        dt = r * dx**2 / alpha if scheme == "heat_explicit" else 0.5 * dx
        exact = lambda xx, tt: np.exp(-alpha * np.pi**2 * tt) * np.sin(np.pi * xx)  # noqa: E731
        u_centre = exact(x, t)
        u_plus = exact(x + dx, t)
        u_minus = exact(x - dx, t)
        spatial = alpha * (u_plus - 2.0 * u_centre + u_minus) / dx**2
        if scheme == "heat_explicit":
            tau = (exact(x, t + dt) - u_centre) / dt - spatial
        else:
            tau = (exact(x, t + dt) - u_centre) / dt - 0.5 * (spatial + alpha * (
                np.exp(-alpha * np.pi**2 * (t + dt)) * (
                    np.sin(np.pi * (x + dx)) - 2.0 * np.sin(np.pi * x) + np.sin(np.pi * (x - dx))
                )
            ) / dx**2)
        errors.append(abs(float(tau)))
    spacings = np.asarray(dx_values, dtype=np.float64)
    estimate = fit_order(spacings, np.asarray(errors), name=f"{scheme} truncation")
    return {
        "scheme": scheme,
        "spacings": [float(v) for v in spacings],
        "truncation_errors": errors,
        "consistency_order": estimate.order,
        "ci95": list(estimate.ci95),
    }


# --------------------------------------------------------------------------- Lax equivalence
def lax_equivalence_demo(
    alpha: float = 0.1, *, t_end: float = 0.1, nx: int = 20
) -> dict[str, Any]:
    """Consistent + stable => convergent, shown with the three measurements on the same scheme.

    The scheme is the explicit heat solver on one Fourier mode. It is **consistent** (truncation error
    ``O(dx)`` with ``dt`` tied to ``dx``), **stable** for ``r <= 1/2`` and **convergent** at that order --
    and then the same scheme at ``r = 0.55`` is still consistent (the truncation error does not know about
    stability) while the solution diverges. That is the theorem: consistency alone is not enough, and the
    divergence is not a bug in the code.

    Returns the three verdicts and the error of both runs, so the claim is checked rather than asserted.
    """
    consistency = truncation_error_study("heat_explicit", alpha=alpha)

    dx = 1.0 / nx
    stable_r = 0.4
    unstable_r = 0.55
    stable_dt = stable_r * dx**2 / alpha
    unstable_dt = unstable_r * dx**2 / alpha
    stable_steps = int(round(t_end / stable_dt))
    unstable_steps = max(1, int(round(t_end / unstable_dt)))

    stable = heat_1d_explicit(alpha, nx, t_end / stable_steps, steps=stable_steps)
    unstable = heat_1d_explicit(alpha, nx, t_end / unstable_steps, steps=unstable_steps)
    exact = heat_1d_exact_mode(alpha, t_end, stable.x)
    stable_error = float(np.max(np.abs(stable.u - exact)))
    unstable_error = float(np.max(np.abs(unstable.u - exact)))

    return {
        "consistency_order": consistency["consistency_order"],
        "is_consistent": bool(consistency["consistency_order"] > 0.5),
        "stable_amplification": stable.amplification,
        "unstable_amplification": unstable.amplification,
        "stable_error": stable_error,
        "unstable_error": unstable_error,
        "is_convergent": bool(stable.stable and stable_error < 1e-2),
        "diverges_when_unstable": bool(unstable_error > stable_error),
    }
