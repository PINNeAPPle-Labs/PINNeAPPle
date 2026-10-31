"""Thermodynamic potentials: Helmholtz energy builders, closed-form equations of state and their identities.

* :func:`helmholtz` names the tree of ``A(T, V)`` (molar or specific): ``S = -dA/dT``, ``p = -dA/dV``; responses
  ``U, H, G`` (depth 1) and ``c_v, c_p, kappa_T, alpha_p, w, (dp/dT)_V, (dp/dV)_T`` (depth 2).
* :func:`helmholtz_T_rho` does the same for a specific energy ``a(T, rho)`` (the variables of most reference
  equations of state): ``p = rho^2 da/drho``, ``s = -da/dT``.
* :func:`ideal_gas`, :func:`van_der_waals`, :func:`peng_robinson`: SymPy potentials with closed-form properties
  (symbolic and autodiff trees agree).
* :func:`gibbs`: ``G(T, p)`` by Legendre transform, with ``c_p = -T d2G/dT2``, ``kappa_T``, ``alpha_p``.
* :func:`identities_T_v`, :func:`identities_T_rho`: the Maxwell relation and the response identities used by
  :func:`~pinneapple_physics.potential.relations.consistency_check` on property-by-property models.

Units are SI; molar potentials in J/mol with ``V`` in m3/mol, specific ones in J/kg with ``v`` in m3/kg and ``rho``
in kg/m3. Sources: Callen, *Thermodynamics and an Introduction to Thermostatistics*, 2nd ed. (1985), ch. 3, 5 and 7
(potentials, Maxwell relations, response functions); Peng & Robinson, Ind. Eng. Chem. Fundam. 15 (1976) 59-64.
"""
from __future__ import annotations

import math

import torch

from .core import Potential, Response
from .legendre import LegendreResult, legendre
from .relations import Identity

__all__ = ["R_GAS", "helmholtz", "helmholtz_T_rho", "ideal_gas", "van_der_waals", "peng_robinson",
           "peng_robinson_constants", "vdw_critical_point", "vdw_spinodal", "gibbs", "identities_T_v",
           "identities_T_rho"]

#: molar gas constant, J/(mol K) (CODATA 2018, exact)
R_GAS = 8.314462618


def _helmholtz_responses(T: str, V: str, molar_mass: float, mass: bool) -> dict[str, Response]:
    e = "J/kg" if mass else "J/mol"
    c = "J/(kg K)" if mass else "J/(mol K)"

    def cv(d, x):
        return -x[T] * d[(T, T)]

    def cp(d, x):
        return -x[T] * d[(T, T)] + x[T] * d[(T, V)] ** 2 / d[(V, V)]

    def w(d, x):
        return torch.sqrt(x[V] ** 2 * (d[(V, V)] - d[(T, V)] ** 2 / d[(T, T)]) / molar_mass)

    return {
        "U": Response(lambda d, x: d[()] - x[T] * d[(T,)], 1, e, "internal energy A - T dA/dT"),
        "H": Response(lambda d, x: d[()] - x[T] * d[(T,)] - x[V] * d[(V,)], 1, e, "enthalpy U + p V"),
        "G": Response(lambda d, x: d[()] - x[V] * d[(V,)], 1, e, "Gibbs energy A + p V"),
        "cv": Response(cv, 2, c, "isochoric heat capacity -T d2A/dT2"),
        "cp": Response(cp, 2, c, "isobaric heat capacity c_v + T V alpha_p^2 / kappa_T"),
        "dpdT_V": Response(lambda d, x: -d[(T, V)], 2, "Pa/K", "(dp/dT)_V"),
        "dpdV_T": Response(lambda d, x: -d[(V, V)], 2, "Pa/(m3/" + ("kg)" if mass else "mol)"), "(dp/dV)_T"),
        "kappa_T": Response(lambda d, x: 1.0 / (x[V] * d[(V, V)]), 2, "1/Pa", "isothermal compressibility"),
        "alpha_p": Response(lambda d, x: -d[(T, V)] / (x[V] * d[(V, V)]), 2, "1/K", "thermal expansion"),
        "w": Response(w, 2, "m/s", "speed of sound"),
    }


def helmholtz(fn, variables=("T", "V"), *, molar_mass: float = 1.0, mass: bool = False, name: str = "A",
              expr=None, symbols=None) -> Potential:
    """A Helmholtz energy ``A(T, V)`` with its thermodynamic names and responses.

    ``fn(T=..., V=...)`` in torch (or pass a SymPy ``expr`` with ``symbols``). ``mass=True``: specific energy
    (J/kg, ``v`` in m3/kg); otherwise molar, and ``molar_mass`` (kg/mol) is needed for the speed of sound."""
    T, V = variables
    e = "J/kg" if mass else "J/mol"
    vol = "m3/kg" if mass else "m3/mol"
    names = {(T,): ("S", -1, "J/(kg K)" if mass else "J/(mol K)", "entropy"), (V,): ("p", -1, "Pa", "pressure")}
    kw = dict(name=name, unit=e, names=names, responses=_helmholtz_responses(T, V, 1.0 if mass else molar_mass, mass),
              relation_names={(T, V): "Maxwell relation of the Helmholtz energy"})
    if expr is not None:
        return Potential.from_sympy(expr, dict(zip(symbols, ("K", vol), strict=False)), **kw)
    return Potential(fn, {T: "K", V: vol}, **kw)


def helmholtz_T_rho(fn, *, name: str = "a", T: str = "T", rho: str = "rho") -> Potential:
    """A specific Helmholtz energy ``a(T, rho)`` (J/kg) with ``p = rho^2 da/drho``, ``s = -da/dT``, ``u``, ``h``,
    ``c_v``, ``c_p``, ``w`` and the pressure derivatives as responses."""

    def p(d, x):
        return x[rho] ** 2 * d[(rho,)]

    def dpdT(d, x):
        return x[rho] ** 2 * d[(T, rho)]

    def dpdr(d, x):
        return 2 * x[rho] * d[(rho,)] + x[rho] ** 2 * d[(rho, rho)]

    def cv(d, x):
        return -x[T] * d[(T, T)]

    def cp(d, x):
        return cv(d, x) + x[T] * dpdT(d, x) ** 2 / (x[rho] ** 2 * dpdr(d, x))

    def w(d, x):
        return torch.sqrt(dpdr(d, x) + x[T] * dpdT(d, x) ** 2 / (x[rho] ** 2 * cv(d, x)))

    responses = {
        "p": Response(p, 1, "Pa", "pressure rho^2 da/drho"),
        "u": Response(lambda d, x: d[()] - x[T] * d[(T,)], 1, "J/kg", "specific internal energy"),
        "h": Response(lambda d, x: d[()] - x[T] * d[(T,)] + x[rho] * d[(rho,)], 1, "J/kg", "specific enthalpy"),
        "cv": Response(cv, 2, "J/(kg K)", "isochoric heat capacity"),
        "cp": Response(cp, 2, "J/(kg K)", "isobaric heat capacity"),
        "dpdT_rho": Response(dpdT, 2, "Pa/K", "(dp/dT)_rho"),
        "dpdrho_T": Response(dpdr, 2, "Pa m3/kg", "(dp/drho)_T"),
        "w": Response(w, 2, "m/s", "speed of sound"),
    }
    return Potential(fn, {T: "K", rho: "kg/m3"}, name=name, unit="J/kg",
                     names={(T,): ("s", -1, "J/(kg K)", "specific entropy")}, responses=responses)


# ---------------------------------------------------------------------------------------------- closed forms
def _sym():
    import sympy as sp
    return sp, sp.Symbol("T", positive=True), sp.Symbol("V", positive=True)


def ideal_gas(c: float = 1.5, R: float = R_GAS, T0: float = 298.15, V0: float = 0.024789570296023,
              molar_mass: float = 1.0) -> Potential:
    """Ideal gas with constant ``c_v = c R`` (``c = 3/2`` monatomic): ``A = c R T (1 - ln(T/T0)) - R T ln(V/V0)``.
    Closed forms: ``p = R T / V``, ``S = c R ln(T/T0) + R ln(V/V0)``, ``c_p = (c + 1) R``, ``kappa_T = 1/p``,
    ``alpha_p = 1/T``, ``w = sqrt((c + 1)/c R T / M)``."""
    sp, T, V = _sym()
    A = c * R * T * (1 - sp.log(T / T0)) - R * T * sp.log(V / V0)
    return helmholtz(None, molar_mass=molar_mass, expr=A, symbols=(T, V))


def van_der_waals(a: float, b: float, c: float = 1.5, R: float = R_GAS, T0: float = 298.15,
                  V0: float = 0.024789570296023, molar_mass: float = 1.0) -> Potential:
    """Van der Waals fluid, ``A = A_ideal - R T ln(1 - b/V) - a/V``, so ``p = R T/(V - b) - a/V^2``."""
    sp, T, V = _sym()
    A = c * R * T * (1 - sp.log(T / T0)) - R * T * sp.log(V / V0) - R * T * sp.log(1 - b / V) - a / V
    return helmholtz(None, molar_mass=molar_mass, expr=A, symbols=(T, V))


def vdw_critical_point(a: float, b: float, R: float = R_GAS) -> dict[str, float]:
    """``T_c = 8a/(27 R b)``, ``V_c = 3b``, ``p_c = a/(27 b^2)``."""
    return {"T": 8 * a / (27 * R * b), "V": 3 * b, "p": a / (27 * b * b)}


def vdw_spinodal(a: float, b: float, T: float, R: float = R_GAS) -> tuple[float, float] | None:
    """The two volumes where ``(dp/dV)_T = 0`` (``R T V^3 = 2 a (V - b)^2``) for ``T < T_c``; None above."""
    import numpy as np

    roots = np.roots([R * T, -2 * a, 4 * a * b, -2 * a * b * b])
    r = sorted(float(x.real) for x in roots if abs(x.imag) < 1e-9 * abs(x) and x.real > b)
    return (r[0], r[1]) if len(r) >= 2 else None


#: Peng-Robinson Omega_a and Omega_b (roots of the critical conditions, Peng & Robinson 1976)
_PR_OMEGA_B = 0.07779607390388844
_PR_OMEGA_A = 0.45723552892138218


def peng_robinson_constants(Tc: float, pc: float, omega: float, R: float = R_GAS) -> dict[str, float]:
    """``a = Omega_a R^2 Tc^2 / pc``, ``b = Omega_b R Tc / pc``, ``kappa = 0.37464 + 1.54226 w - 0.26992 w^2``."""
    return {"a": _PR_OMEGA_A * R * R * Tc * Tc / pc, "b": _PR_OMEGA_B * R * Tc / pc,
            "kappa": 0.37464 + 1.54226 * omega - 0.26992 * omega * omega}


def peng_robinson(Tc: float, pc: float, omega: float, c: float = 2.5, R: float = R_GAS, T0: float = 298.15,
                  V0: float = 0.024789570296023, molar_mass: float = 1.0) -> Potential:
    """Peng-Robinson fluid with a constant ideal-gas ``c_v = c R``:
    ``A = A_ideal - R T ln(1 - b/V) - a alpha(T)/(2 sqrt2 b) ln((V + (1+sqrt2) b)/(V + (1-sqrt2) b))``,
    ``alpha = (1 + kappa (1 - sqrt(T/Tc)))^2``; ``p = R T/(V - b) - a alpha/(V^2 + 2 b V - b^2)``."""
    sp, T, V = _sym()
    k = peng_robinson_constants(Tc, pc, omega, R)
    a, b, kap = k["a"], k["b"], k["kappa"]
    s2 = math.sqrt(2.0)
    alpha = (1 + kap * (1 - sp.sqrt(T / Tc))) ** 2
    A = (c * R * T * (1 - sp.log(T / T0)) - R * T * sp.log(V / V0) - R * T * sp.log(1 - b / V)
         - a * alpha / (2 * s2 * b) * sp.log((V + (1 + s2) * b) / (V + (1 - s2) * b)))
    return helmholtz(None, molar_mass=molar_mass, expr=A, symbols=(T, V))


def gibbs(A: Potential, bounds=None, guess=None, method: str = "auto", **kw) -> LegendreResult:
    """``G(T, p) = A + p V`` by Legendre transform of a Helmholtz energy ``A(T, V)``, with ``S = -dG/dT``,
    ``V = dG/dp`` and responses ``cp = -T d2G/dT2``, ``kappa_T = -(d2G/dp2)/(dG/dp)``,
    ``alpha_p = (d2G/dTdp)/(dG/dp)``. ``bounds={"T": .., "V": ..}`` runs the convexity check in ``V``."""
    T, V = list(A.variables)
    names = {(T,): ("S", -1, A.names[(T,)].unit if (T,) in A.names else None, "entropy"),
             ("p",): (V, 1, A.variables[V].unit, "volume")}
    responses = {
        "cp": Response(lambda d, x: -x[T] * d[(T, T)], 2, A.responses["cp"].unit, "isobaric heat capacity"),
        "kappa_T": Response(lambda d, x: -d[("p", "p")] / d[("p",)], 2, "1/Pa", "isothermal compressibility"),
        "alpha_p": Response(lambda d, x: d[(T, "p")] / d[("p",)], 2, "1/K", "thermal expansion"),
    }
    return legendre(A, {V: "p"}, conjugate_sign=-1, name="G", bounds=bounds, guess=guess, method=method,
                    names=names, responses=responses,
                    relation_names={(T, "p"): "Maxwell relation of the Gibbs energy"}, **kw)


# ---------------------------------------------------------------------------------------------- identities
def identities_T_v() -> list[Identity]:
    """Identities between specific properties in ``(T, v)``: quantities ``p, s, u, cv, cp, w``."""
    return [
        Identity("Maxwell relation (Helmholtz)", "(ds/dv)_T = (dp/dT)_v",
                 lambda D: D("s", "v"), lambda D: D("p", "T"), ("s", "p")),
        Identity("c_v from the entropy", "c_v = T (ds/dT)_v",
                 lambda D: D("cv"), lambda D: D.x("T") * D("s", "T"), ("cv", "s")),
        Identity("energy equation", "(du/dv)_T = T (dp/dT)_v - p",
                 lambda D: D("u", "v"), lambda D: D.x("T") * D("p", "T") - D("p"), ("u", "p")),
        Identity("volume dependence of c_v", "(dc_v/dv)_T = T (d2p/dT2)_v",
                 lambda D: D("cv", "v"), lambda D: D.x("T") * D("p", "T", "T"), ("cv", "p"),
                 scale=lambda D: D("cv") / D.x("v")),
        Identity("Mayer relation", "c_p - c_v = -T (dp/dT)_v^2 / (dp/dv)_T",
                 lambda D: D("cp") - D("cv"), lambda D: -D.x("T") * D("p", "T") ** 2 / D("p", "v"),
                 ("cp", "cv", "p")),
        Identity("speed of sound", "w^2 = -v^2 [(dp/dv)_T - T (dp/dT)_v^2 / c_v]",
                 lambda D: D("w") ** 2,
                 lambda D: -D.x("v") ** 2 * (D("p", "v") - D.x("T") * D("p", "T") ** 2 / D("cv")),
                 ("w", "p", "cv")),
    ]


def identities_T_rho() -> list[Identity]:
    """Identities between specific properties in ``(T, rho)``: quantities ``p, s, u, cv, cp, w``."""
    return [
        Identity("Maxwell relation (Helmholtz)", "-rho^2 (ds/drho)_T = (dp/dT)_rho",
                 lambda D: -D.x("rho") ** 2 * D("s", "rho"), lambda D: D("p", "T"), ("s", "p")),
        Identity("c_v from the entropy", "c_v = T (ds/dT)_rho",
                 lambda D: D("cv"), lambda D: D.x("T") * D("s", "T"), ("cv", "s")),
        Identity("energy equation", "rho^2 (du/drho)_T = p - T (dp/dT)_rho",
                 lambda D: D.x("rho") ** 2 * D("u", "rho"), lambda D: D("p") - D.x("T") * D("p", "T"), ("u", "p")),
        Identity("density dependence of c_v", "-rho^2 (dc_v/drho)_T = T (d2p/dT2)_rho",
                 lambda D: -D.x("rho") ** 2 * D("cv", "rho"), lambda D: D.x("T") * D("p", "T", "T"), ("cv", "p"),
                 scale=lambda D: D("cv") * D.x("rho")),
        Identity("Mayer relation", "c_p - c_v = T (dp/dT)_rho^2 / (rho^2 (dp/drho)_T)",
                 lambda D: D("cp") - D("cv"),
                 lambda D: D.x("T") * D("p", "T") ** 2 / (D.x("rho") ** 2 * D("p", "rho")), ("cp", "cv", "p")),
        Identity("speed of sound", "w^2 = (dp/drho)_T + T (dp/dT)_rho^2 / (rho^2 c_v)",
                 lambda D: D("w") ** 2,
                 lambda D: D("p", "rho") + D.x("T") * D("p", "T") ** 2 / (D.x("rho") ** 2 * D("cv")),
                 ("w", "p", "cv")),
    ]
