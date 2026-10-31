"""Fully developed pipe and channel flow: Darcy friction factors, pressure drop and near-wall cell sizing.

Darcy friction factor ``f`` is defined by ``dp = f (L / D) rho U^2 / 2`` with U the bulk (mean) velocity and D the
(hydraulic) diameter (White, *Fluid Mechanics*, 7th ed., 2011, §6.6). The correlations are empirical and only hold
in their range; outside it they emit a ``RuntimeWarning`` (non-physical inputs raise ``ValueError``):

========== ==================================== =========================================
function   formula                              range
========== ==================================== =========================================
laminar    64 / Re (Hagen-Poiseuille)           Re < 2300
blasius    0.316 Re^-1/4 (Blasius 1913)         smooth pipe, 4e3 <= Re <= 1e5
colebrook  implicit (Colebrook 1939)            turbulent, Re >= 4e3
haaland    explicit (Haaland 1983)              4e3 <= Re <= 1e8, eps/D <= 0.05
curved_    White (1929) laminar curved pipe,    11.6 < De < 2000 (De = Re sqrt(D / 2 Rc))
pipe_ratio fully developed f_curved / f_straight
========== ==================================== =========================================

References: C. F. Colebrook, "Turbulent flow in pipes...", J. Inst. Civil Eng. 11 (1939) 133-156;
S. E. Haaland, "Simple and explicit formulas for the friction factor in turbulent pipe flow", J. Fluids Eng. 105
(1983) 89-90; H. Blasius, "Das Aehnlichkeitsgesetz bei Reibungsvorgaengen in Fluessigkeiten", Forschungsheft 131,
VDI (1913); F. M. White, *Fluid Mechanics*, 7th ed. (2011), §6.4-6.8; C. M. White, "Streamline flow through
curved pipes", Proc. R. Soc. Lond. A 123 (1929) 645-663 (the correlation as given in P. Naphon and S. Wongwises,
"A review of flow and heat transfer characteristics in curved tubes", Renew. Sustain. Energy Rev. 10 (2006) 463-490).
"""
from __future__ import annotations

import math
import warnings

__all__ = ["laminar", "blasius", "colebrook", "haaland", "friction_factor", "darcy_pressure_drop",
           "parallel_plates_pressure_drop", "first_cell_height", "dean_number", "curved_pipe_ratio"]

RE_LAMINAR_MAX = 2300.0     # upper limit of laminar pipe flow (White §6.2)
RE_TURBULENT_MIN = 4000.0   # lower limit of the turbulent correlations (Moody chart)


def _check(Re: float, rel_rough: float = 0.0) -> None:
    if not Re > 0:
        raise ValueError(f"Re must be positive, got {Re}")
    if rel_rough < 0:
        raise ValueError(f"relative roughness must be non-negative, got {rel_rough}")


def _warn(name: str, msg: str) -> None:
    warnings.warn(f"{name}: {msg}", RuntimeWarning, stacklevel=3)


def laminar(Re: float) -> float:
    """Darcy friction factor of laminar fully developed pipe flow, ``f = 64 / Re`` (Hagen-Poiseuille; White
    eq. 6.49). Warns for ``Re >= 2300``.

    Returns
    -------
    f : float [-]
        Darcy friction factor.
    """
    _check(Re)
    if Re >= RE_LAMINAR_MAX:
        _warn("laminar", f"Re = {Re:g} is above the laminar limit {RE_LAMINAR_MAX:g}")
    return 64.0 / Re


def blasius(Re: float) -> float:
    """Blasius (1913) smooth-pipe correlation ``f = 0.316 Re^-0.25`` (White eq. 6.55). Warns outside
    ``4e3 <= Re <= 1e5``.

    Returns
    -------
    f : float [-]
        Darcy friction factor.
    """
    _check(Re)
    if not RE_TURBULENT_MIN <= Re <= 1e5:
        _warn("blasius", f"Re = {Re:g} is outside its range [4e3, 1e5]")
    return 0.316 * Re ** -0.25


def colebrook(Re: float, rel_rough: float = 0.0, tol: float = 1e-14, max_iter: int = 100) -> float:
    """Darcy friction factor from the Colebrook (1939) equation (White eq. 6.48),

        1 / sqrt(f) = -2 log10( (eps/D) / 3.7 + 2.51 / (Re sqrt(f)) ),

    solved by fixed-point iteration on ``1 / sqrt(f)`` (a contraction for turbulent Re) from ``f = 0.02`` until the
    update is below ``tol``. ``rel_rough`` = eps / D. Warns for ``Re < 4000`` (laminar / transitional flow) and
    ``eps / D > 0.05`` (beyond the Moody chart).

    Returns
    -------
    f : float [-]
        Darcy friction factor.
    """
    _check(Re, rel_rough)
    if Re < RE_TURBULENT_MIN:
        _warn("colebrook", f"Re = {Re:g} is below the turbulent range (Re >= {RE_TURBULENT_MIN:g})")
    if rel_rough > 0.05:
        _warn("colebrook", f"eps/D = {rel_rough:g} is beyond the Moody chart (<= 0.05)")
    x = 1.0 / math.sqrt(0.02)
    for _ in range(max_iter):
        x_new = -2.0 * math.log10(rel_rough / 3.7 + 2.51 * x / Re)
        if abs(x_new - x) < tol * x_new:
            x = x_new
            break
        x = x_new
    return 1.0 / (x * x)


def haaland(Re: float, rel_rough: float = 0.0) -> float:
    """Explicit Haaland (1983) approximation of Colebrook (White eq. 6.64), within about 2 % of it:

        1 / sqrt(f) = -1.8 log10( ((eps/D) / 3.7)^1.11 + 6.9 / Re ).

    Warns outside ``4e3 <= Re <= 1e8`` or ``eps / D > 0.05``.

    Returns
    -------
    f : float [-]
        Darcy friction factor.
    """
    _check(Re, rel_rough)
    if not RE_TURBULENT_MIN <= Re <= 1e8:
        _warn("haaland", f"Re = {Re:g} is outside its range [4e3, 1e8]")
    if rel_rough > 0.05:
        _warn("haaland", f"eps/D = {rel_rough:g} is outside its range (<= 0.05)")
    return (-1.8 * math.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def dean_number(Re: float, D: float, Rc: float) -> float:
    """Dean number ``De = Re sqrt(D / (2 Rc))`` of a pipe of diameter D bent to a centre-line radius Rc (same
    units).

    Returns
    -------
    De : float [-]
        Dean number.
    """
    if not (D > 0 and Rc > 0):
        raise ValueError("D and Rc must be positive")
    return Re * math.sqrt(D / (2.0 * Rc))


def curved_pipe_ratio(De: float) -> float:
    """Fully developed laminar friction factor of a curved pipe over that of a straight one at the same Re, White
    (1929): ``f_c / f_s = 1 / (1 - (1 - (11.6 / De)^0.45)^(1 / 0.45))``; 1 for De <= 11.6. Warns above De = 2000
    (beyond White's data).

    Returns
    -------
    ratio : float [-]
        ``f_c / f_s``.
    """
    if not De >= 0:
        raise ValueError(f"De must be non-negative, got {De}")
    if De <= 11.6:
        return 1.0
    if De > 2000:
        _warn("curved_pipe_ratio", f"De = {De:g} is above the range of White's data (<= 2000)")
    return 1.0 / (1.0 - (1.0 - (11.6 / De) ** 0.45) ** (1.0 / 0.45))


def friction_factor(Re: float, rel_rough: float = 0.0) -> float:
    """``laminar(Re)`` below Re = 2300, ``colebrook(Re, rel_rough)`` otherwise (the transitional band 2300-4000 is
    uncertain; Colebrook is used there without a warning).

    Returns
    -------
    f : float [-]
        Darcy friction factor.
    """
    if Re < RE_LAMINAR_MAX:
        return laminar(Re)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = colebrook(Re, rel_rough)
    if rel_rough > 0.05:
        _warn("friction_factor", f"eps/D = {rel_rough:g} is beyond the Moody chart (<= 0.05)")
    return f


def darcy_pressure_drop(f: float, L: float, D: float, rho: float, U: float) -> float:
    """Darcy-Weisbach pressure drop [Pa] ``dp = f (L / D) rho U^2 / 2`` (White eq. 6.10) over a length L [m] of a
    duct of (hydraulic) diameter D [m], density rho [kg/m^3], bulk velocity U [m/s].

    Returns
    -------
    dp : float [Pa]
        Pressure drop over ``L``.
    """
    return f * (L / D) * 0.5 * rho * U * U


def parallel_plates_pressure_drop(mu: float, L: float, h: float, U: float) -> float:
    """Laminar pressure drop [Pa] between infinite parallel plates a gap ``h`` [m] apart (plane Poiseuille flow),
    ``dp = 12 mu L U / h^2`` with U the mean velocity [m/s] and mu the dynamic viscosity [Pa s] (White §6.6).

    Returns
    -------
    dp : float [Pa]
        Pressure drop over ``L``.
    """
    return 12.0 * mu * L * U / (h * h)


def first_cell_height(y_plus: float, Re: float, D: float, U: float, nu: float, f: float | None = None,
                      rel_rough: float = 0.0) -> float:
    """Wall-normal height y [m] of the first cell such that ``y+ = y u_tau / nu`` equals ``y_plus`` (the usual
    y+ estimate; a cell-centred code puts the first centroid at y / 2).

    Wall shear from the friction factor of fully developed flow, ``tau_w = f rho U^2 / 8`` (White eq. 6.17), so the
    friction velocity is ``u_tau = U sqrt(f / 8)`` and ``y = y+ nu / u_tau``. ``f`` defaults to
    :func:`friction_factor` (laminar below Re = 2300, Colebrook with ``rel_rough`` above) at ``Re``; D [m] is only
    used to check that ``Re`` is consistent with ``U D / nu`` (a mismatch above 1 % warns).

    Returns
    -------
    y : float [m]
        Height of the first cell.
    """
    if not (y_plus > 0 and U > 0 and nu > 0 and D > 0):
        raise ValueError("y_plus, U, nu and D must be positive")
    if abs(U * D / nu - Re) > 0.01 * Re:
        _warn("first_cell_height", f"Re = {Re:g} does not match U D / nu = {U * D / nu:g}")
    if f is None:
        f = friction_factor(Re, rel_rough)
    u_tau = U * math.sqrt(f / 8.0)
    return y_plus * nu / u_tau
