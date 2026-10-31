"""Heat-exchanger rating: effectiveness-NTU relations, the LMTD correction factor, and tube- and shell-side
coefficients of a shell-and-tube exchanger.

* Effectiveness (Incropera, DeWitt, Bergman & Lavine, *Fundamentals of Heat and Mass Transfer*, 6th ed., Table
  11.3): counterflow, parallel flow, one shell pass with 2, 4, ... tube passes (TEMA E, eq. 11.30a), n shell passes
  (eq. 11.31a), crossflow both fluids unmixed (eq. 11.32, approximate), and C_r = 0 for every arrangement
  (eq. 11.35a).
* LMTD correction factor F of a 1-2 exchanger (Bowman, Mueller & Nagle 1940; Incropera Fig. 11S.1).
* Tube side: Gnielinski (1976), ``Nu = (f/8)(Re-1000)Pr / (1 + 12.7 sqrt(f/8)(Pr^(2/3)-1))`` with Petukhov's
  ``f = (0.790 ln Re - 1.64)^-2``, valid for 3000 <= Re <= 5e6 and 0.5 <= Pr <= 2000. Laminar: Nu = 3.66.
* Laminar thermal entry, constant wall temperature, developed velocity: Hausen (1943), mean
  ``Nu_D = 3.66 + 0.0668 Gz / (1 + 0.04 Gz^(2/3))``, ``Gz = (D / L) Re Pr`` (Incropera eq. 8.57).
* Shell side: Kern (1950), *Process Heat Transfer*, eq. 7.3, ``Nu = 0.36 Re^0.55 Pr^(1/3)`` (with the viscosity
  correction dropped), on the shell equivalent diameter. Valid for 2e3 < Re < 1e6 with 25 % segmental baffles.
  This is the standard first estimate. Bell-Delaware gives lower and more accurate values.
* Fouling resistances: from TEMA RGP-T-2.4 (stated by the caller).

:func:`one_two_exchanger_ode` integrates the 1-2 exchanger directly. It provides an independent check of the
closed form that does not use eq. 11.30a.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["effectiveness", "ntu_from_effectiveness", "lmtd_correction_1_2", "gnielinski", "kern_shell",
           "one_two_exchanger_ode", "hausen_mean_nusselt", "mean_nusselt_from_temperatures"]


def effectiveness(ntu: float, cr: float, arrangement: str = "counterflow", n_shell: int = 1) -> float:
    """epsilon(NTU, C_r) of the given arrangement ("counterflow", "parallel", "shell_tube", "crossflow").

    Returns
    -------
    eps : float [-]
        Effectiveness.
    """
    if ntu < 0 or not 0 <= cr <= 1:
        raise ValueError("NTU >= 0 and 0 <= C_r <= 1")
    if cr < 1e-12:
        return 1 - math.exp(-ntu)
    if arrangement == "counterflow":
        if abs(cr - 1) < 1e-12:
            return ntu / (1 + ntu)
        e = math.exp(-ntu * (1 - cr))
        return (1 - e) / (1 - cr * e)
    if arrangement == "parallel":
        return (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)
    if arrangement == "shell_tube":
        n1 = ntu / n_shell
        s = math.sqrt(1 + cr ** 2)
        x = math.exp(-n1 * s)
        e1 = 2 / (1 + cr + s * (1 + x) / (1 - x))
        if n_shell == 1:
            return e1
        if abs(cr - 1) < 1e-12:
            return n_shell * e1 / (1 + (n_shell - 1) * e1)
        a = ((1 - e1 * cr) / (1 - e1)) ** n_shell
        return (a - 1) / (a - cr)
    if arrangement == "crossflow":
        return 1 - math.exp(ntu ** 0.22 / cr * (math.exp(-cr * ntu ** 0.78) - 1))
    raise ValueError("arrangement: counterflow, parallel, shell_tube or crossflow")


def ntu_from_effectiveness(eps: float, cr: float, arrangement: str = "counterflow") -> float:
    """Inverse of :func:`effectiveness` (bisection; NTU up to 50).

    Returns
    -------
    ntu : float [-]
        Number of transfer units.
    """
    lo, hi = 0.0, 50.0
    if eps >= effectiveness(hi, cr, arrangement):
        raise ValueError("effectiveness not reachable with this arrangement")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if effectiveness(mid, cr, arrangement) < eps else (lo, mid)
    return 0.5 * (lo + hi)


def lmtd_correction_1_2(t_hot_in: float, t_hot_out: float, t_cold_in: float, t_cold_out: float) -> float:
    """F of a 1-2 exchanger (shell fluid either side), Bowman et al. (1940).

    Returns
    -------
    F : float [-]
        LMTD correction factor.
    """
    R = (t_hot_in - t_hot_out) / (t_cold_out - t_cold_in)
    P = (t_cold_out - t_cold_in) / (t_hot_in - t_cold_in)
    s = math.sqrt(R * R + 1)
    if abs(R - 1) < 1e-9:
        return s * P / ((1 - P) * math.log((2 - P * (2 - s)) / (2 - P * (2 + s))))
    num = s / (R - 1) * math.log((1 - P) / (1 - P * R))
    den = math.log((2 - P * (R + 1 - s)) / (2 - P * (R + 1 + s)))
    return num / den


def gnielinski(re: float, pr: float) -> dict[str, float]:
    """Tube-side Nusselt number (Gnielinski; laminar fully developed Nu = 3.66 below Re = 2300)."""
    if re < 2300:
        return {"nu": 3.66, "f": 64 / re, "regime": "laminar"}
    f = (0.790 * math.log(re) - 1.64) ** -2
    nu = (f / 8) * (re - 1000) * pr / (1 + 12.7 * math.sqrt(f / 8) * (pr ** (2 / 3) - 1))
    return {"nu": nu, "f": f, "regime": "turbulent" if re >= 3000 else "transitional"}


def hausen_mean_nusselt(re: float, pr: float, d_over_l: float) -> float:
    """Mean Nusselt number over a heated length L of a tube at constant wall temperature, laminar, velocity already
    developed (thermal entry): Hausen (1943), Incropera 6th ed. eq. 8.57. Tends to 3.66 for a long tube.

    Returns
    -------
    Nu_m : float [-]
        Mean Nusselt number.
    """
    if not (re > 0 and pr > 0 and d_over_l > 0):
        raise ValueError("re, pr and d_over_l must be positive")
    gz = d_over_l * re * pr
    return 3.66 + 0.0668 * gz / (1.0 + 0.04 * gz ** (2.0 / 3.0))


def mean_nusselt_from_temperatures(re: float, pr: float, d_over_l: float, t_wall: float, t_in: float,
                                   t_out: float) -> float:
    """Mean Nusselt number of a tube at constant wall temperature from its bulk inlet and outlet temperatures:
    the energy balance ``ln((Tw - Ti) / (Tw - To)) = 4 (L / D) Nu_m / (Re Pr)`` (Incropera eq. 8.41b with
    h_m = Nu_m k / D).

    Returns
    -------
    Nu_m : float [-]
        Mean Nusselt number.
    """
    return re * pr * d_over_l / 4.0 * math.log((t_wall - t_in) / (t_wall - t_out))


def kern_shell(m_dot: float, mu: float, k: float, cp: float, *, shell_id: float, baffle_spacing: float,
               tube_od: float, pitch: float, layout: str = "triangular") -> dict[str, float]:
    """Shell-side coefficient by Kern's method: crossflow area, equivalent diameter, Re, Nu, h."""
    a_s = shell_id * (pitch - tube_od) * baffle_spacing / pitch
    if layout == "triangular":
        de = 4 * (math.sqrt(3) / 4 * pitch ** 2 - math.pi * tube_od ** 2 / 8) / (math.pi * tube_od / 2)
    else:
        de = 4 * (pitch ** 2 - math.pi * tube_od ** 2 / 4) / (math.pi * tube_od)
    G = m_dot / a_s
    re = G * de / mu
    pr = cp * mu / k
    nu = 0.36 * re ** 0.55 * pr ** (1 / 3)
    return {"a_s": a_s, "de": de, "G": G, "re": re, "pr": pr, "nu": nu, "h": nu * k / de}


def one_two_exchanger_ode(ntu: float, cr: float, n: int = 4001, profiles: bool = False):
    """Effectiveness of a 1-2 exchanger by integrating its three energy equations directly. The shell fluid is
    well mixed across each section, and the two tube passes run in opposite directions (Nagle's model). This is
    solved as a boundary-value problem by finite differences, so it does not use the closed form. The shell fluid
    is C_min. ``profiles``: also return x and the dimensionless shell / pass-1 / pass-2 temperatures."""
    from scipy.integrate import solve_bvp
    # x in [0, 1] along the shell; shell temperature T, tube pass 1 t1 (co-current with the shell), pass 2 t2
    # dimensionless: theta = (T - t_in) / (T_in - t_in); NTU on the shell (C_min = shell) side
    a = ntu / 2

    def f(x, y):
        T, t1, t2 = y
        dT = -a * ((T - t1) + (T - t2))
        dt1 = a * cr * (T - t1)
        dt2 = -a * cr * (T - t2)
        return np.vstack([dT, dt1, dt2])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, ya[1] - 0.0, yb[1] - yb[2]])    # shell in at x=0, tubes in at x=0, U-turn

    x = np.linspace(0, 1, 201)
    y0 = np.vstack([1 - 0.5 * x, 0.3 * x, 0.3 + 0 * x])
    sol = solve_bvp(f, bc, x, y0, tol=1e-10, max_nodes=n * 10)
    if not sol.success:
        raise RuntimeError(sol.message)
    eps = float(1 - sol.sol(1.0)[0])
    if profiles:
        xs = np.linspace(0, 1, 101)
        T, t1, t2 = sol.sol(xs)
        return eps, {"x": xs, "shell": T, "pass1": t1, "pass2": t2}
    return eps
