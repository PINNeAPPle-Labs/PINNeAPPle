"""pinneapple_analysis.trust.benchmark — the five-family benchmark and one CFD surrogate the certificate is measured
on (#341).

:mod:`pinneapple_analysis.trust.certificate` decides, per law, whether a field satisfies the physics it claims. The
question this module answers is the one about the *decision*: does it separate a prediction made inside the model's
domain from one made outside it — on all five families, or only on some?

Five families, one exact case each
----------------------------------
Every family is one of the physics of :mod:`pinneapple_check.physics`, and every *in-domain* field is an analytic
solution of that physics, evaluated on a grid. Nothing here solves anything numerically: the residual the
certificate reports is then the prediction's and the discretisation's, never a solver's.

=========================  ==============================================================================
``advection-diffusion``    ``C = exp(-alpha t) sin(x - c t)`` on ``x in [0, 2 pi)``, ``t in [0, 1]``
``heat``                   ``T = exp(-(k / rho_cp) t) sin(x)`` (``q = 0``), the same axes
``incompressible-flow``    the Taylor-Green vortex, ``u = cos x sin y e^{-2 nu t}``, ``v = -sin x cos y e^{-2 nu t}``,
                           ``p = -(cos 2x + cos 2y) e^{-4 nu t} / 4`` (kinematic pressure): an exact solution of
                           div u = 0 and du/dt + (u.grad)u + grad p - nu lap u = 0
``shallow-water``          a steady vortex in gradient balance, ``v_theta = Omega r exp(-r^2 / 2 a^2)`` and
                           ``h = H0 + (Omega^2 a^2 / 2 g) (1 - exp(-r^2 / a^2))``, which solves
                           ``u.grad u + g grad h = 0`` and ``div(h u) = 0`` exactly
``elasticity``             plane strain with no body force, ``ux = c1 x y``, ``uy = c2 y^2 / 2 - c1 x^2 / 2`` with
                           ``c2 = -nu c1 / (1 - nu)``: ``div sigma = 0`` with the plane-strain Lame constants
=========================  ==============================================================================

Each family brings three in-domain fields -- the exact solution and two with white noise at a fraction of its own
standard deviation -- and three out-of-domain ones, in the three kinds a surrogate produces outside its regime:

``parameter``    the field of the same equation solved with another value of a parameter the equation uses (a
                 diffusivity, a viscosity, a gravity);
``other-equation``  the solution of a *different* family, plugged into this one;
``sign``         the exact solution with its sign inverted.

The noise level is a fraction of the field's own standard deviation, and it is **not** the same fraction in all five
families: a residual that is a second derivative amplifies white noise by ``(L / h)^2``, so 0.1 % of noise on a
50-point axis is a perturbation of the derivatives, not of the field. The level used for each family is the one that
keeps the in-domain residual at least ten times below the 5 % the cases are judged against, and both the level and
the residual it produces are in the report (:data:`NOISE`).

The three out-of-domain kinds are not equally visible, and that is part of the result: the residual of a **linear,
homogeneous** law is invariant under ``C -> -C``, so the sign case is undetectable by construction for
advection-diffusion and heat (and for elasticity, whose equilibrium with no body force is homogeneous too). The
benchmark says so instead of leaving the row out.

One CFD surrogate
-----------------
:func:`cfd_surrogate_cases` puts a real surrogate on the same question. The reference flow is the Taylor-Green vortex
above -- the case :func:`pinneapple_simulation.numerical_solvers.ns3d_spectral.taylor_green_ic` starts from -- and the
surrogate is :class:`pinneapple_simulation.cfd_validation.warm_start.OperatorSurrogate`: a POD of reference fields
solved at five viscosities with its coefficients interpolated in the parameter, which is what the repository means by
a field surrogate. The certificate is asked about four viscosities inside the fitted range and four outside it. The
case declares ``domain={"nu": (lo, hi)}``, so a query outside it also has to reach the domain gate.

>>> from pinneapple_analysis.trust.benchmark import five_family_cases, measure
>>> report = measure()                                    # doctest: +SKIP
>>> report["families"]["heat"]["detection"]               # doctest: +SKIP
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .certificate import Calibration, CaseResult, Problem, verify
from .record import GATE_FAIL, GATE_NO_SKILL

__all__ = ["FAMILIES", "NOISE", "TOLERANCE", "Case", "five_family_cases", "family_cases", "cfd_surrogate_cases",
           "measure", "table", "write", "SCHEMA"]

#: The five families, in the order the tables print them.
FAMILIES = ("advection-diffusion", "heat", "incompressible-flow", "shallow-water", "elasticity")

#: The relative tolerance every case is judged against, one entry per law the five families evaluate. The same 5 %
#: for all of them, so the table compares like with like; what changes between families is the noise floor.
TOLERANCE = 0.05

#: White noise as a fraction of the field's own standard deviation, per family, applied at half and at full level.
#: The rule: the largest level whose residual stays at or below a fifth of :data:`TOLERANCE` (1 %), so the
#: perturbation is not the thing being measured. The level that achieves it differs by two orders of magnitude
#: between the families, and that difference is itself part of the result -- a residual that is a second derivative
#: amplifies white noise by ``(L / h)^2``. Both the level and the residual it produces are in the report, so the choice
#: can be checked instead of trusted.
NOISE: dict[str, float] = {"advection-diffusion": 2.0e-4, "heat": 5.0e-5, "incompressible-flow": 8.0e-4,
                           "shallow-water": 1.0e-3, "elasticity": 4.0e-6}

#: What the JSON says it is.
SCHEMA = "pinneapple.certificate_benchmark/1"


@dataclass(frozen=True)
class Case:
    """One case of the benchmark: the field, the problem it claims, and the truth about it.

    ``in_domain`` is the truth the case was built with -- the field obeys the declared physics with the declared
    parameters -- and never something read from the certificate. ``kind`` says which of the four the case is
    (``exact``, ``noise``, ``parameter``, ``other-equation``, ``sign``), so a table can be read per kind.
    """

    label: str
    family: str
    fields: Mapping[str, np.ndarray]
    problem: Problem
    in_domain: bool
    kind: str
    note: str = ""

    def as_calibration_case(self) -> tuple[str, Any, Problem, bool]:
        return (self.label, dict(self.fields), self.problem, self.in_domain)


# ------------------------------------------------------------------------------------------------ the reference grids
def _advection_grid() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]          # periodic in x: the last point repeats the first
    t = np.linspace(0.0, 1.0, 41)
    return x, t


def _flow_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # a 48 x 48 x 11 space-time grid: the momentum residual is a second derivative in space, so the axis has to be
    # resolved enough for the discretisation floor to stay under the 5 % the cases are judged against
    x = y = np.linspace(0.0, 2.0 * np.pi, 48)
    t = np.linspace(0.0, 1.0, 11)
    return x, y, t


def _lake_grid() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, 80)


def _solid_grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, 48)


def _noisy(a: np.ndarray, level: float, seed: int) -> np.ndarray:
    """The field plus white noise at ``level`` times its own standard deviation."""
    if not level:
        return a
    s = float(np.std(a))
    return a + level * s * np.random.default_rng(seed).standard_normal(a.shape)


# ------------------------------------------------------------------------------------------------ the five families
def _advection_diffusion() -> list[Case]:
    """``dC/dt + c dC/dx = alpha d2C/dx2``: the exact solution, and the three ways a surrogate leaves it."""
    x, t = _advection_grid()
    X, T = np.meshgrid(x, t, indexing="ij")
    c, alpha = 1.0, 0.05

    def solution(c_adv: float = c, alpha_d: float = alpha) -> np.ndarray:
        return np.exp(-alpha_d * T) * np.sin(X - c_adv * T)

    problem = Problem(physics="advection-diffusion", axes={"x": x, "t": t},
                      parameters={"c": c, "alpha": alpha}, units={"C": "1", "x": "m", "t": "s"},
                      tolerance={"advection-diffusion": TOLERANCE},
                      notes=("exact solution of dC/dt + c dC/dx = alpha d2C/dx2 on a periodic x, t in [0, 1]",))
    exact = solution()
    lvl = NOISE["advection-diffusion"]
    heat_like = np.exp(-T) * np.sin(X)                          # the heat family's solution, claimed as C
    return [
        Case("advection-diffusion/exact", "advection-diffusion", {"C": exact}, problem, True, "exact",
             "the analytic solution: what a surrogate that is right looks like"),
        Case("advection-diffusion/noise-half", "advection-diffusion", {"C": _noisy(exact, lvl / 2, 11)}, problem,
             True, "noise", f"{lvl / 2:.1e} of the field's standard deviation"),
        Case("advection-diffusion/noise", "advection-diffusion", {"C": _noisy(exact, lvl, 12)}, problem, True,
             "noise", f"{lvl:.1e} of the field's standard deviation"),
        Case("advection-diffusion/parameter", "advection-diffusion", {"C": solution(c_adv=1.6)}, problem, False,
             "parameter", "the same equation advecting at 1.6 while the problem declares 1.0"),
        Case("advection-diffusion/other-equation", "advection-diffusion", {"C": heat_like}, problem, False,
             "other-equation", "the heat family's exact solution, claimed to solve advection-diffusion"),
        Case("advection-diffusion/sign", "advection-diffusion", {"C": -exact}, problem, False, "sign",
             "the exact solution with its sign inverted: the law is linear and homogeneous, so the residual cannot "
             "see it -- the expected result of this row is 'not flagged', and that is the finding"),
    ]


def _heat() -> list[Case]:
    """``rho_cp dT/dt = k lap T + q`` with ``q = 0``: the exact solution and the three departures from it."""
    x, t = _advection_grid()
    X, T = np.meshgrid(x, t, indexing="ij")
    k, rho_cp = 1.0, 1.0

    def solution(k_diff: float = k) -> np.ndarray:
        return np.exp(-(k_diff / rho_cp) * T) * np.sin(X)

    problem = Problem(physics="heat", axes={"x": x, "t": t}, parameters={"k": k, "rho_cp": rho_cp, "q": 0.0},
                      units={"T": "K", "x": "m", "t": "s"}, tolerance={"heat": TOLERANCE},
                      notes=("exact solution of rho_cp dT/dt = k lap T on a periodic x, t in [0, 1], q = 0",))
    exact = solution()
    lvl = NOISE["heat"]
    return [
        Case("heat/exact", "heat", {"T": exact}, problem, True, "exact", "the analytic solution"),
        Case("heat/noise-half", "heat", {"T": _noisy(exact, lvl / 2, 21)}, problem, True, "noise",
             f"{lvl / 2:.1e} of the field's standard deviation"),
        Case("heat/noise", "heat", {"T": _noisy(exact, lvl, 22)}, problem, True, "noise",
             f"{lvl:.1e} of the field's standard deviation"),
        Case("heat/parameter", "heat", {"T": solution(k_diff=3.0)}, problem, False, "parameter",
             "the field decaying three times too fast for the declared k"),
        Case("heat/other-equation", "heat", {"T": np.exp(-0.05 * T) * np.sin(X - T)}, problem, False,
             "other-equation", "the advection-diffusion solution, claimed to be a temperature"),
        Case("heat/sign", "heat", {"T": -exact}, problem, False, "sign",
             "sign inverted: q = 0 makes the law linear and homogeneous, so the residual is blind to it"),
    ]


def _taylor_green() -> dict[str, np.ndarray]:
    """The Taylor-Green vortex at ``nu`` on the flow grid: an exact solution of the incompressible equations."""
    x, y, t = _flow_grid()
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    return {"x": x, "y": y, "t": t, "X": X, "Y": Y, "T": T}


def _taylor_green_fields(nu: float, xs: np.ndarray | None = None, ys: np.ndarray | None = None,
                         ts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The Taylor-Green vortex at ``nu`` on the flow grid, or on the axes given (so another family can be asked to
    accept it)."""
    g = _taylor_green()
    X, Y, T = np.meshgrid(g["x"] if xs is None else xs, g["y"] if ys is None else ys,
                          g["t"] if ts is None else ts, indexing="ij")
    e, e2 = np.exp(-2.0 * nu * T), np.exp(-4.0 * nu * T)
    return {"u": np.cos(X) * np.sin(Y) * e, "v": -np.sin(X) * np.cos(Y) * e,
            "p": -(np.cos(2.0 * X) + np.cos(2.0 * Y)) / 4.0 * e2}


def _flow_problem(nu: float, *, domain: Mapping[str, tuple[float, float]] | None = None) -> Problem:
    g = _taylor_green()
    return Problem(physics="incompressible-flow", axes={"x": g["x"], "y": g["y"], "t": g["t"]},
                   parameters={"nu": nu, "rho": 1.0},
                   units={"u": "m/s", "v": "m/s", "p": "m^2/s^2", "x": "m", "y": "m", "t": "s"},
                   domain=dict(domain or {}), tolerance={"continuity": TOLERANCE, "momentum": TOLERANCE},
                   notes=("the Taylor-Green vortex, an exact solution of div u = 0 and "
                          "du/dt + (u.grad)u + grad p - nu lap u = 0",))


def _incompressible_flow() -> list[Case]:
    """The Taylor-Green vortex, and the three fields a surrogate returns when it leaves that regime."""
    nu = 0.02
    problem = _flow_problem(nu)
    exact = _taylor_green_fields(nu)
    lvl = NOISE["incompressible-flow"]
    g = _taylor_green()
    lake = _vortex_on(g["x"], g["y"])                        # another family's flow, claimed as a solution here
    other = {"u": lake["u"][:, :, None] * np.ones_like(g["T"]), "v": lake["v"][:, :, None] * np.ones_like(g["T"]),
             "p": exact["p"]}
    return [
        Case("incompressible-flow/exact", "incompressible-flow", exact, problem, True, "exact",
             "the analytic Taylor-Green vortex at nu = 0.02"),
        Case("incompressible-flow/noise-half", "incompressible-flow",
             {k: _noisy(v, lvl / 2, 31) for k, v in exact.items()}, problem, True, "noise",
             f"{lvl / 2:.1e} of each field's standard deviation"),
        Case("incompressible-flow/noise", "incompressible-flow",
             {k: _noisy(v, lvl, 32) for k, v in exact.items()}, problem, True, "noise",
             f"{lvl:.1e} of each field's standard deviation"),
        Case("incompressible-flow/parameter", "incompressible-flow", _taylor_green_fields(0.2), problem, False,
             "parameter", "the vortex decaying at nu = 0.2, ten times the 0.02 the problem declares"),
        Case("incompressible-flow/parameter-undamped", "incompressible-flow", _taylor_green_fields(0.0), problem,
             False, "parameter-undamped",
             "the vortex that does not decay at all, against nu = 0.02: the viscous term is small next to the "
             "advection at this Reynolds number, so this direction of the same error is NOT flagged -- the residual "
             "of the two fields differs by a factor of ten and only one of them crosses the limit"),
        Case("incompressible-flow/other-equation", "incompressible-flow", other, problem, False, "other-equation",
             "the shallow-water vortex's velocity, claimed to be an incompressible flow"),
        Case("incompressible-flow/sign", "incompressible-flow", {k: -v for k, v in exact.items()}, problem, False,
             "sign", "sign inverted: the convective term does not flip with the rest, so momentum sees it"),
    ]


_G = 9.80665                                                  # gravity, the one parameter of the shallow-water family
_H0, _OMEGA, _A = 1.0, 0.5, 1.0                               # still depth, angular rate and core radius of the vortex


def _lake_vortex_fields(g: float = _G, *, omega: float = _OMEGA) -> dict[str, np.ndarray]:
    """The steady vortex in gradient balance: ``v_theta = omega r exp(-r^2 / 2a^2)`` with the depth that holds it.

    ``g`` is the gravity the depth profile was built for; asking for another ``g`` returns a field whose free
    surface no longer balances the velocity it carries.
    """
    xs = _lake_grid()
    return _vortex_on(xs, xs, g, omega)


def _vortex_on(xs: np.ndarray, ys: np.ndarray, g: float = _G, omega: float = _OMEGA) -> dict[str, np.ndarray]:
    """The same vortex sampled on any grid, so another family can be asked to accept it."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X - np.pi, Y - np.pi                          # centred in the periodic box
    r2 = X ** 2 + Y ** 2
    vort = omega * np.exp(-r2 / (2.0 * _A ** 2))          # v_theta / r, finite at r = 0
    h = _H0 + (omega ** 2 * _A ** 2 / (2.0 * g)) * (1.0 - np.exp(-r2 / _A ** 2))
    return {"h": h, "u": -vort * Y, "v": vort * X}


def _lake_problem() -> Problem:
    xs = _lake_grid()
    return Problem(physics="shallow-water", axes={"x": xs, "y": xs}, parameters={"g": _G},
                   units={"h": "m", "u": "m/s", "v": "m/s", "x": "m", "y": "m"}, positivity=("h",),
                   tolerance={"continuity": TOLERANCE, "momentum": TOLERANCE},
                   notes=("a steady vortex in gradient balance: an exact solution of dh/dt + div(h u) = 0 and "
                          "d(hu)/dt + div(hu u) + g h grad h = 0",))


def _shallow_water() -> list[Case]:
    """The steady vortex, and the three fields that stop balancing it."""
    problem = _lake_problem()
    exact = _lake_vortex_fields()
    lvl = NOISE["shallow-water"]
    xs = _lake_grid()
    flow = _taylor_green_fields(0.02, xs, xs, np.zeros(1))   # a steady slice of the incompressible vortex
    other = {"h": np.full(flow["u"][..., 0].shape, _H0), "u": flow["u"][..., 0], "v": flow["v"][..., 0]}
    return [
        Case("shallow-water/exact", "shallow-water", exact, problem, True, "exact",
             "the vortex whose free surface balances its velocity"),
        Case("shallow-water/noise-half", "shallow-water",
             {k: _noisy(v, lvl / 2, 41) for k, v in exact.items()}, problem, True, "noise",
             f"{lvl / 2:.1e} of each field's standard deviation"),
        Case("shallow-water/noise", "shallow-water", {k: _noisy(v, lvl, 42) for k, v in exact.items()}, problem,
             True, "noise", f"{lvl:.1e} of each field's standard deviation"),
        Case("shallow-water/parameter", "shallow-water", _lake_vortex_fields(2.0 * _G), problem, False, "parameter",
             "the free surface built for 2g, judged against g: the gradient no longer balances the velocity"),
        Case("shallow-water/other-equation", "shallow-water", other, problem, False, "other-equation",
             "an incompressible vortex under a flat free surface, claimed to solve the shallow-water equations"),
        Case("shallow-water/sign", "shallow-water", {k: -v for k, v in exact.items()}, problem, False, "sign",
             "sign inverted: the depth goes negative, which positivity and momentum both refuse"),
    ]


def _elasticity_fields() -> dict[str, np.ndarray]:
    """Plane strain in equilibrium: ``ux = c1 x y``, ``uy = c2 y^2 / 2 - c1 x^2 / 2``, ``c2 = -nu c1 / (1 - nu)``."""
    xs = _solid_grid()
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    c1 = 1.0e-3
    c2 = -_POISSON / (1.0 - _POISSON) * c1
    return {"ux": c1 * X * Y, "uy": c2 * Y ** 2 / 2.0 - c1 * X ** 2 / 2.0 + c1 / 2.0}


_POISSON = 0.3
_YOUNG = 1.0e6


def _solid_problem(*, body_force: Sequence[float] | None = None) -> Problem:
    xs = _solid_grid()
    return Problem(physics="elasticity", axes={"x": xs, "y": xs},
                   parameters={"E": _YOUNG, "nu": _POISSON, "f": list(body_force or (0.0, 0.0))},
                   units={"ux": "m", "uy": "m", "x": "m", "y": "m"},
                   tolerance={"equilibrium": TOLERANCE},
                   notes=("plane strain with div sigma = 0, sigma = lambda tr(eps) I + 2 mu eps, and the "
                          "plane-strain Lame constants of E and nu",))


def _elasticity() -> list[Case]:
    """The equilibrium displacement, and the three fields that are not in equilibrium."""
    problem = _solid_problem()
    exact = _elasticity_fields()
    lvl = NOISE["elasticity"]
    xs = _solid_grid()
    flow = _taylor_green_fields(0.02, xs, xs, np.zeros(1))   # an incompressible vortex, claimed as a displacement
    return [
        Case("elasticity/exact", "elasticity", exact, problem, True, "exact",
             "the analytic equilibrium displacement, div sigma = 0"),
        Case("elasticity/noise-half", "elasticity", {k: _noisy(v, lvl / 2, 51) for k, v in exact.items()}, problem,
             True, "noise", f"{lvl / 2:.1e} of each field's standard deviation"),
        Case("elasticity/noise", "elasticity", {k: _noisy(v, lvl, 52) for k, v in exact.items()}, problem, True,
             "noise", f"{lvl:.1e} of each field's standard deviation"),
        Case("elasticity/parameter", "elasticity", exact, _solid_problem(body_force=(0.0, -1.0e4)), False,
             "parameter", "the same field, but the problem declares a body force it does not carry"),
        Case("elasticity/other-equation", "elasticity",
             {"ux": flow["u"][..., 0] * 1e-3, "uy": flow["v"][..., 0] * 1e-3}, problem, False, "other-equation",
             "an incompressible vortex, claimed to be a displacement field in equilibrium"),
        Case("elasticity/sign", "elasticity", {k: -v for k, v in exact.items()}, problem, False, "sign",
             "sign inverted: with no body force the equilibrium is linear and homogeneous, so the residual cannot "
             "see it -- the expected result of this row is 'not flagged'"),
    ]


_BUILDERS = {"advection-diffusion": _advection_diffusion, "heat": _heat,
             "incompressible-flow": _incompressible_flow, "shallow-water": _shallow_water,
             "elasticity": _elasticity}


def family_cases(family: str) -> list[Case]:
    """The six cases of one family: three in-domain and three out-of-domain."""
    try:
        return _BUILDERS[family]()
    except KeyError:
        raise KeyError(f"unknown family {family!r}; the benchmark has {', '.join(FAMILIES)}") from None


def five_family_cases() -> dict[str, list[Case]]:
    """Every case of the five-family benchmark, keyed by family (six cases each, three in and three out)."""
    return {family: family_cases(family) for family in FAMILIES}


# ------------------------------------------------------------------------------------------------ the CFD surrogate
#: The viscosities the surrogate is fitted on. A case between two of them is in the surrogate's envelope; a case
#: outside the range is a query the surrogate extrapolates to, which is what "out of domain" means for a field model.
CFD_TRAIN = (0.05, 0.1, 0.2, 0.4)
#: The queries: four inside the fitted range and four outside it.
CFD_INSIDE = (0.07, 0.15, 0.25, 0.35)
CFD_OUTSIDE = (0.02, 0.6, 0.8, 1.2)
#: The modes the POD keeps per field.
CFD_MODES = 8


def cfd_surrogate_cases() -> list[Case]:
    """The Taylor-Green vortex through the repository's POD surrogate, inside and outside its fitted envelope.

    The surrogate is :class:`pinneapple_simulation.cfd_validation.warm_start.OperatorSurrogate` fitted on the reference
    fields at :data:`CFD_TRAIN`; the certificate is asked about :data:`CFD_INSIDE` (interpolated) and
    :data:`CFD_OUTSIDE` (extrapolated beyond either end). The problem declares ``domain={"nu": (min, max)}`` of the
    fitted range, so an extrapolated query has to reach the domain gate as well as the residual checks.
    """
    from pinneapple_simulation.cfd_validation.warm_start import OperatorSurrogate

    sur = OperatorSurrogate(modes=CFD_MODES).fit({nu: _taylor_green_fields(nu) for nu in CFD_TRAIN})
    lo, hi = min(CFD_TRAIN), max(CFD_TRAIN)
    out: list[Case] = []
    for nu in CFD_INSIDE:
        out.append(Case(f"cfd-surrogate/nu={nu:g}", "cfd-surrogate", sur.predict(nu),
                        _flow_problem(nu, domain={"nu": (lo, hi)}), True, "inside",
                        f"nu inside the fitted range [{lo:g}, {hi:g}]: the surrogate interpolates"))
    for nu in CFD_OUTSIDE:
        out.append(Case(f"cfd-surrogate/nu={nu:g}", "cfd-surrogate", sur.predict(nu),
                        _flow_problem(nu, domain={"nu": (lo, hi)}), False, "outside",
                        f"nu outside the fitted range [{lo:g}, {hi:g}]: the surrogate extrapolates"))
    return out


# ------------------------------------------------------------------------------------------------ measuring
def _family_rates(cal: Calibration) -> dict[str, Any]:
    """The two rates of every gate, plus the one the certificate's own verdict carries.

    ``verdict`` is the certificate's badge written as a decision: ``rejected`` or ``qualified`` means the certificate
    did not certify the field, so the rate is the one a consumer reads -- "would this have been stopped?" -- while the
    per-gate rates say which check did it.
    """
    det, n_out = cal._rate("__verdict__", in_domain=False)
    fa, n_in = cal._rate("__verdict__", in_domain=True)
    return {"cases_in": n_in, "cases_out": n_out,
            "gates": {g: {"detection": cal.rate(g)[0], "false_alarm": cal.rate(g)[1]} for g in cal.gates()},
            "verdict": {"detection": det / n_out if n_out else None, "false_alarm": fa / n_in if n_in else None}}


def _verdict_flags(cases: Sequence[Case]) -> Calibration:
    """A :class:`Calibration` whose one gate is the certificate's verdict: flagged when it did not certify."""
    rows = []
    for case in cases:
        cert = verify(case.fields, case.problem)
        flags = {g.name: g.status in (GATE_FAIL, GATE_NO_SKILL) for g in cert.gates()}
        flags["__verdict__"] = not cert.certified()
        values = {c.name: c.value for c in cert.checks if c.value is not None}
        rows.append(CaseResult(case.label, case.in_domain, cert.verdict(), flags, values))
    return Calibration(tuple(rows))


def _cases_report(cases: Sequence[Case]) -> list[dict[str, Any]]:
    """Per case: what the certificate said, the number behind it, and what the case was."""
    out = []
    for case in cases:
        cert = verify(case.fields, case.problem)
        decided = {c.name: {"value": c.value, "threshold": c.threshold, "status": c.status}
                   for c in cert.checks if c.value is not None and (c.name.startswith("residual")
                                                                    or c.name in ("domain", "positivity:h"))}
        out.append({"label": case.label, "kind": case.kind, "in_domain": case.in_domain,
                    "verdict": cert.verdict(), "why": cert.separating(), "checks": decided,
                    "note": case.note})
    return out


def measure() -> dict[str, Any]:
    """Run the benchmark and return the report: the rates and every case behind them.

    Two trials, both reported: the five families, one calibration each (a family is one law, and the rates of a law
    over the cases of another law would not mean anything), and the CFD surrogate as a sixth group.
    """
    out: dict[str, Any] = {"schema": SCHEMA, "tolerance": TOLERANCE,
                           "noise": dict(NOISE), "families": {}, "cfd_surrogate": {}}
    for family in FAMILIES:
        cases = family_cases(family)
        cal = _verdict_flags(cases)
        rates = _family_rates(cal)
        out["families"][family] = {**rates, "cases": _cases_report(cases)}
    cfd = cfd_surrogate_cases()
    cal = _verdict_flags(cfd)
    out["cfd_surrogate"] = {**_family_rates(cal), "train": list(CFD_TRAIN), "modes": CFD_MODES,
                            "inside": list(CFD_INSIDE), "outside": list(CFD_OUTSIDE),
                            "cases": _cases_report(cfd)}
    return out


def table(report: Mapping[str, Any] | None = None) -> str:
    """The report as the text table the guide prints: one line per family, with the counts behind the rates.

    A family whose verdict rate is not 1 has a line under it naming the out-of-domain cases the certificate let
    through: the misses are the measurement, not an omission from the table.
    """
    rep = dict(report or measure())
    lines = [f"{'family':22s} {'cases':>12s} {'detection':>10s} {'false alarms':>13s}"]
    for name, body in list(rep["families"].items()) + [("cfd-surrogate", rep["cfd_surrogate"])]:
        v = body["verdict"]
        det = "n/a" if v["detection"] is None else f"{v['detection']:.3f}"
        fa = "n/a" if v["false_alarm"] is None else f"{v['false_alarm']:.3f}"
        cases = f"{body['cases_in']} in / {body['cases_out']} out"
        lines.append(f"{name:22s} {cases:>14s} {det:>10s} {fa:>13s}")
        missed = [c["label"] for c in body["cases"] if not c["in_domain"] and c["verdict"] == "certified"]
        if missed:
            lines.append(f"{'':22s} out of domain, certified: {', '.join(missed)}")
        wrong = [c["label"] for c in body["cases"] if c["in_domain"] and c["verdict"] != "certified"]
        if wrong:
            lines.append(f"{'':22s} in domain, flagged: {', '.join(wrong)}")
    return "\n".join(lines)


def write(path: Any = "results/certificate_five_families.json", *, report: Mapping[str, Any] | None = None) -> Path:
    """Write the report as JSON (``allow_nan=False``: a rate over no case is ``None``, never ``nan``)."""
    rep = json.loads(json.dumps(dict(report or measure()), allow_nan=False, default=_jsonable))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _jsonable(value: Any) -> Any:
    """numpy scalars and arrays, for the JSON dump (the rates are already plain floats)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")
