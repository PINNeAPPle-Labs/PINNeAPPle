"""Coupled cases with closed-form solutions for the multiphysics workflow (#1209).

Each builder returns a :class:`CoupledCase`: the two participants (:mod:`.coupling`), the reference interface field
for the one-way solve, the quantities of interest, the interface and energy balances, and the closed-form
solution of the *coupled* problem the discrete result is checked against.

* :func:`joule_wire` (electro-thermal, two-way, nonlinear): a wire of length ``L`` carrying a current ``I``, ends
  held at ``T0``, resistivity ``rho(T) = rho0 (1 + alpha (T - T0))``. Electric participant: ``q = J^2 rho(T)``;
  thermal participant: ``-k T'' = q``. Closed form, with ``m^2 = J^2 rho0 alpha / k``:
  ``T - T0 = (cos(m (x - L/2)) / cos(m L / 2) - 1) / alpha`` and ``dV = rho0 J (2/m) tan(m L / 2)``; no steady
  state exists for ``m L >= pi`` (thermal runaway), which the builder refuses.
* :func:`bimetal_strip` (thermal-structural, one-way by nature): two bonded layers heated by ``dT``. Thermal
  participant: conduction through the thickness; structural participant: beam (laminate) theory, axial force and
  bending moment equilibrium for the mid-plane strain and the curvature. Closed form: Timoshenko (1925),
  ``kappa = 6 (a2 - a1) dT (1 + m)^2 / (h (3 (1 + m)^2 + (1 + m n) (m^2 + 1 / (m n))))``, ``m = t1/t2``,
  ``n = E1/E2``, tip deflection ``kappa L^2 / 2`` for a cantilever.
* :func:`conjugate_channel` (fluid-thermal, conjugate heat transfer): fluid in a channel (capacity rate
  ``C = m_dot c_p``, ``C T_f' = hP (T_w - T_f)``) against a wall with axial conduction heated by ``q'`` per unit
  length (``kA T_w'' - hP (T_w - T_f) + q' = 0``, insulated ends). The partition is the usual CHT one: the solid
  reads the fluid temperature (Robin), the fluid reads the wall temperature. Closed form: the exact solution of
  the linear constant-coefficient system (matrix exponential), and ``C (T_out - T_in) = q' L``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .coupling import Participant


@dataclass
class CoupledCase:
    name: str
    physics: str
    a: Participant
    b: Participant
    y_ref: np.ndarray
    qoi: Callable[[np.ndarray, np.ndarray], dict[str, float]]
    balances: Callable[[np.ndarray, np.ndarray], list[dict[str, Any]]]
    exact: Callable[[np.ndarray, np.ndarray], dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)
    feedback_note: str = ""


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# --------------------------------------------------------------------------------------------------------------
# electro-thermal: Joule-heated wire
# --------------------------------------------------------------------------------------------------------------

def joule_wire(*, length: float = 0.05, radius: float = 0.25e-3, current: float = 6.0, rho0: float = 5.6e-8,
               alpha: float = 4.5e-3, k: float = 170.0, t0: float = 300.0, n: int = 201) -> CoupledCase:
    """Defaults: a tungsten-like wire, 50 mm long, 0.5 mm diameter, at 6 A (``m L / pi`` about 0.6)."""
    area = np.pi * radius**2
    J = current / area
    m = np.sqrt(J**2 * rho0 * alpha / k)
    if m * length >= np.pi:
        raise ValueError(f"no steady state: m L = {m * length:.4g} >= pi (thermal runaway; the closed form "
                         f"diverges). Lower the current below {np.pi / length / np.sqrt(rho0 * alpha / k) * area:.4g} A")
    x = np.linspace(0.0, length, n)
    h = x[1] - x[0]
    main = np.full(n, 2.0 * k / h**2)
    off = np.full(n - 1, -k / h**2)
    K = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    for i in (0, n - 1):
        K[i, :] = 0.0
        K[i, i] = 1.0
    K = K.tocsc()
    lu = spla.splu(K)

    def heat(T):                                    # electric participant: Joule heat per volume
        return J**2 * rho0 * (1.0 + alpha * (T - t0))

    def rhs(q):
        b = q.copy()
        b[0] = b[-1] = t0
        return b

    q_scale = J**2 * rho0                           # residuals in units of the cold Joule heat

    elec = Participant("electric", "temperature", "joule_heat", solve=heat,
                       residual=lambda q, T: (q - heat(T)) / q_scale, output=lambda q: q)
    therm = Participant("thermal", "joule_heat", "temperature", solve=lambda q: lu.solve(rhs(q)),
                        residual=lambda T, q: (K @ T - rhs(q)) / q_scale, output=lambda T: T)

    def dv(T):
        return float(np.trapezoid(rho0 * (1 + alpha * (T - t0)) * J, x))

    def qoi(q, T):
        v = dv(T)
        return {"T_max_K": float(T.max()), "voltage_drop_V": v, "resistance_ohm": v / current}

    def balances(q, T):
        power = current * dv(T)
        deposited = float(np.trapezoid(q, x)) * area           # what the thermal model receives
        g0 = (-3 * T[0] + 4 * T[1] - T[2]) / (2 * h)
        gL = (3 * T[-1] - 4 * T[-2] + T[-3]) / (2 * h)
        out = k * area * (g0 - gL)                              # conducted out of both ends
        return [{"code": "interface.joule_power", "value": _rel(deposited, power),
                 "detail": f"electric power I dV = {power:.6g} W; heat received by the thermal model "
                           f"{deposited:.6g} W"},
                {"code": "energy.global", "value": _rel(out, power),
                 "detail": f"heat conducted out of the ends {out:.6g} W against I dV = {power:.6g} W"}]

    def exact(q, T):
        Te = t0 + (np.cos(m * (x - length / 2)) / np.cos(m * length / 2) - 1.0) / alpha
        ve = rho0 * J * 2 / m * np.tan(m * length / 2)
        return {"field_error": float(np.max(np.abs(T - Te)) / np.max(np.abs(Te - t0))),
                "qoi": {"T_max_K": float(Te.max()), "voltage_drop_V": float(ve), "resistance_ohm": float(ve / current)},
                "formula": "T - T0 = (cos(m(x - L/2))/cos(mL/2) - 1)/alpha, dV = rho0 J (2/m) tan(mL/2)",
                "mL_over_pi": float(m * length / np.pi)}

    return CoupledCase("joule_wire", "electro-thermal", elec, therm, np.full(n, t0), qoi, balances, exact,
                       {"length_m": length, "radius_m": radius, "current_A": current, "rho0_ohm_m": rho0,
                        "alpha_1_K": alpha, "k_W_mK": k, "T0_K": t0, "n": n, "mL_over_pi": float(m * length / np.pi)},
                       "two-way: the temperature changes the resistivity, which changes the Joule heat")


# --------------------------------------------------------------------------------------------------------------
# thermal-structural: bimetal strip
# --------------------------------------------------------------------------------------------------------------

def bimetal_strip(*, t1: float = 0.5e-3, t2: float = 0.5e-3, E1: float = 200e9, E2: float = 100e9,
                  alpha1: float = 12e-6, alpha2: float = 19e-6, k1: float = 50.0, k2: float = 110.0,
                  length: float = 0.05, t_ref: float = 293.15, t_bottom: float = 393.15, t_top: float = 393.15,
                  n: int = 41) -> CoupledCase:
    """Layer 1 (bottom, ``0 <= z <= t1``) steel-like, layer 2 (top) brass-like; stress-free at ``t_ref``."""
    hth = t1 + t2
    z = np.concatenate([np.linspace(0, t1, n), np.linspace(t1, hth, n)[1:]])
    N = z.size
    i_int = n - 1
    kz = np.where(np.arange(N - 1) < i_int, k1, k2)            # conductivity of each segment
    dz = np.diff(z)
    cond = kz / dz
    A = np.zeros((N, N))
    for i in range(1, N - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -cond[i - 1], cond[i - 1] + cond[i], -cond[i]
    A[0, 0] = A[-1, -1] = 1.0

    def therm_rhs():
        b = np.zeros(N)
        b[0], b[-1] = t_bottom, t_top
        return b
    T_lin = np.linalg.solve(A, therm_rhs())

    Eseg = np.where(np.arange(N - 1) < i_int, E1, E2)
    aseg = np.where(np.arange(N - 1) < i_int, alpha1, alpha2)

    def _seg_int(f0, f1, g0, g1, d):                # exact integral of the product of two linear functions
        return d / 6.0 * (2 * f0 * g0 + f0 * g1 + f1 * g0 + 2 * f1 * g1)

    def stiffness_and_load(T):
        th0, th1 = aseg * (T[:-1] - t_ref), aseg * (T[1:] - t_ref)
        one = np.ones_like(th0)
        z0, z1 = z[:-1], z[1:]
        A11 = np.sum(Eseg * _seg_int(one, one, one, one, dz))
        A12 = np.sum(Eseg * _seg_int(one, one, z0, z1, dz))
        A22 = np.sum(Eseg * _seg_int(z0, z1, z0, z1, dz))
        F1 = np.sum(Eseg * _seg_int(th0, th1, one, one, dz))
        F2 = np.sum(Eseg * _seg_int(th0, th1, z0, z1, dz))
        return np.array([[A11, A12], [A12, A22]]), np.array([F1, F2])

    def struct_solve(T):                             # state: [eps0 (strain at z = 0), kappa]
        Km, F = stiffness_and_load(T)
        return np.linalg.solve(Km, F)

    def struct_res(s, T):
        Km, F = stiffness_and_load(T)
        return (Km @ s - F) / max(np.abs(F).max(), 1e-300)

    therm = Participant("thermal", "curvature", "temperature", solve=lambda s: T_lin.copy(),
                        residual=lambda T, s: (A @ T - therm_rhs()) / max(t_bottom, t_top), output=lambda T: T)
    struct = Participant("structural", "temperature", "curvature", solve=struct_solve, residual=struct_res,
                         output=lambda s: s)

    def stress(s, T):
        return Eseg, (s[0] + s[1] * z[:-1] - aseg * (T[:-1] - t_ref)), (s[0] + s[1] * z[1:] - aseg * (T[1:] - t_ref))

    def qoi(T, s):
        return {"curvature_1_m": float(s[1]), "tip_deflection_mm": float(s[1] * length**2 / 2 * 1e3)}

    def balances(T, s):
        E, e0, e1 = stress(s, T)
        one = np.ones_like(e0)
        Nseg = E * _seg_int(e0, e1, one, one, dz)
        N1, N2 = float(Nseg[:i_int].sum()), float(Nseg[i_int:].sum())
        Mseg = E * _seg_int(e0, e1, z[:-1], z[1:], dz)
        M = float(Mseg.sum())
        Mref = float(np.abs(Mseg).sum())
        q1 = k1 * (T[i_int] - T[i_int - 1]) / dz[i_int - 1]
        q2 = k2 * (T[i_int + 1] - T[i_int]) / dz[i_int]
        sj = (E2 * e0[i_int] - E1 * e1[i_int - 1]) / 1e6
        return [{"code": "interface.axial_force", "value": abs(N1 + N2) / max(abs(N1), 1e-300),
                 "detail": f"layer forces {N1:.6g} N/m + {N2:.6g} N/m (free strip: they cancel); "
                           f"stress jump at the bond {sj:.4g} MPa"},
                {"code": "interface.bending_moment", "value": abs(M) / max(Mref, 1e-300),
                 "detail": f"resultant moment {M:.3g} N (free strip: zero) against sum |M_i| = {Mref:.4g}"},
                {"code": "interface.heat_flux", "value": abs(q1 - q2) / max(abs(q1), abs(q2), 1.0),
                 "detail": f"conducted flux below / above the bond {q1:.6g} / {q2:.6g} W/m^2"}]

    def exact(T, s):
        if t_bottom != t_top:
            return {"skipped": "Timoshenko's closed form is for uniform heating (t_bottom == t_top)"}
        mm, nn = t1 / t2, E1 / E2
        dT = t_bottom - t_ref
        kap = 6 * (alpha2 - alpha1) * dT * (1 + mm) ** 2 / (hth * (3 * (1 + mm) ** 2 + (1 + mm * nn) *
                                                                    (mm**2 + 1 / (mm * nn))))
        return {"field_error": _rel(float(s[1]), kap),
                "qoi": {"curvature_1_m": float(kap), "tip_deflection_mm": float(kap * length**2 / 2 * 1e3)},
                "formula": "Timoshenko (1925): kappa = 6 (a2 - a1) dT (1+m)^2 / (h (3(1+m)^2 + (1+mn)(m^2+1/(mn))))"}

    return CoupledCase("bimetal_strip", "thermal-structural", therm, struct, np.zeros(2), qoi, balances, exact,
                       {"t1_m": t1, "t2_m": t2, "E1_Pa": E1, "E2_Pa": E2, "alpha1_1_K": alpha1, "alpha2_1_K": alpha2,
                        "k1_W_mK": k1, "k2_W_mK": k2, "length_m": length, "T_ref_K": t_ref, "T_bottom_K": t_bottom,
                        "T_top_K": t_top, "n_per_layer": n},
                       "one-way by nature: the deformation does not change the temperature (thermoelastic "
                       "heating neglected)")


# --------------------------------------------------------------------------------------------------------------
# fluid-thermal: conjugate heat transfer in a heated channel
# --------------------------------------------------------------------------------------------------------------

def conjugate_channel(*, length: float = 1.0, capacity_rate: float = 41.8, hP: float = 50.0, kA: float = 2.0,
                      q_line: float = 1000.0, t_in: float = 300.0, n: int = 201) -> CoupledCase:
    """Defaults: water at 10 g/s (``C = 41.8 W/K``), ``hP = 50 W/(m K)``, a copper-like wall ``kA = 2 W m/K``,
    1 kW/m heater over 1 m."""
    x = np.linspace(0.0, length, n)
    h = x[1] - x[0]
    c = kA / h**2
    S = np.zeros((n, n))
    for i in range(n):
        lo, hi = (i - 1 if i > 0 else 1), (i + 1 if i < n - 1 else n - 2)   # ghost nodes: insulated ends
        S[i, lo] += c
        S[i, hi] += c
        S[i, i] -= 2 * c + hP
    S_lu = np.linalg.inv(S)

    def solid_res(Tw, Tf):
        return (S @ Tw + hP * Tf + q_line) / q_line

    def solid_solve(Tf):
        return S_lu @ (-(hP * Tf + q_line))

    beta = hP / capacity_rate

    def fluid_solve(Tw):
        Tf = np.empty(n)
        Tf[0] = t_in
        a = 1.0 / h + 0.5 * beta
        bco = 1.0 / h - 0.5 * beta
        for i in range(n - 1):
            Tf[i + 1] = (bco * Tf[i] + 0.5 * beta * (Tw[i] + Tw[i + 1])) / a
        return Tf

    def fluid_res(Tf, Tw):
        r = np.empty(n)
        r[0] = Tf[0] - t_in
        r[1:] = (Tf[1:] - Tf[:-1]) / h - beta * (0.5 * (Tw[1:] + Tw[:-1]) - 0.5 * (Tf[1:] + Tf[:-1]))
        return r / max(q_line / capacity_rate, 1e-300)

    solid = Participant("solid", "fluid_temperature", "wall_temperature", solve=solid_solve, residual=solid_res,
                        output=lambda Tw: Tw)
    fluid = Participant("fluid", "wall_temperature", "fluid_temperature", solve=fluid_solve, residual=fluid_res,
                        output=lambda Tf: Tf)

    def qoi(Tw, Tf):
        return {"T_out_K": float(Tf[-1]), "T_wall_max_K": float(Tw.max())}

    def balances(Tw, Tf):
        to_fluid = float(np.trapezoid(hP * (Tw - Tf), x))
        gain = capacity_rate * (Tf[-1] - t_in)
        heater = q_line * length
        return [{"code": "interface.heat_flux", "value": _rel(to_fluid, gain),
                 "detail": f"heat leaving the wall {to_fluid:.6g} W; heat gained by the fluid {gain:.6g} W"},
                {"code": "energy.global", "value": _rel(gain, heater),
                 "detail": f"C (T_out - T_in) = {gain:.6g} W against the heater q' L = {heater:.6g} W"}]

    def exact(Tw, Tf):
        from scipy.linalg import expm
        g = hP / kA
        M = np.array([[-beta, beta, 0, 0], [0, 0, 1, 0], [-g, g, 0, 0], [0, 0, 0, 0]], float)
        M[2, 3] = -q_line / kA
        P = expm(M * length)
        tw0 = -(P[2, 0] * t_in + P[2, 3]) / P[2, 1]
        s0 = np.array([t_in, tw0, 0.0, 1.0])
        sol = np.array([expm(M * xi) @ s0 for xi in x])
        Tfe, Twe = sol[:, 0], sol[:, 1]
        rise = float(np.max(Twe) - t_in)
        return {"field_error": float(max(np.abs(Tw - Twe).max(), np.abs(Tf - Tfe).max()) / rise),
                "qoi": {"T_out_K": float(Tfe[-1]), "T_wall_max_K": float(Twe.max())},
                "formula": "exact solution of the linear system [T_f, T_w, T_w', 1]' = M [...] (matrix exponential); "
                           "C (T_out - T_in) = q' L"}

    return CoupledCase("conjugate_channel", "fluid-thermal", solid, fluid, np.full(n, t_in), qoi, balances, exact,
                       {"length_m": length, "capacity_rate_W_K": capacity_rate, "hP_W_mK": hP, "kA_W_m_K": kA,
                        "q_line_W_m": q_line, "T_in_K": t_in, "n": n, "NTU": hP * length / capacity_rate},
                       "two-way: the wall heats the fluid, and the fluid temperature sets the wall's heat loss")


CASES = {"joule_wire": joule_wire, "bimetal_strip": bimetal_strip, "conjugate_channel": conjugate_channel}
