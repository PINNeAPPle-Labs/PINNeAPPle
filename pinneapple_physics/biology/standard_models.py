"""The standard biological models of the bio workflow (#1205) as :class:`ODESystem` builders, each with the metadata
its checks need (``meta["nonnegative"]``, ``meta["bounded"]``, ``meta["mass_balance"]``).

* :func:`sir` and :func:`seir`: Kermack-McKendrick compartments with a contact-reduction intervention
  ``beta (1 - u)`` on ``t_on <= t < t_off``; ``S + (E) + I + R = N`` is conserved. Without intervention the final size
  obeys ``ln(S0 / S_inf) = R0 (N - R(0) - S_inf) / N`` (Kermack and McKendrick, Proc R Soc A 115:700, 1927).
* :func:`hodgkin_huxley`: the squid giant axon (Hodgkin and Huxley, J Physiol 117:500, 1952) in the modern
  convention (rest near -65 mV), with a rectangular current pulse and a sustained current ``I_ext`` (uA/cm^2).
* :func:`fitzhugh_nagumo`: the two-variable reduction (FitzHugh 1961; Nagumo et al. 1962) with the same pulse.
* :func:`pk_two_compartment`: linear two-compartment pharmacokinetics with first-order oral absorption, repeated
  doses, an effect compartment and the sigmoid Emax model (Holford and Sheiner 1981). ``A_elim`` and ``AUC`` are
  carried as states, so ``AUC(inf) = F D / CL`` and the mass balance are checked from the trajectory itself.
* :func:`michaelis_menten`: the elementary enzyme mechanism ``E + S <-> C -> E + P`` (mass action), whose totals
  ``E + C`` and ``S + C + P`` are conserved and whose quasi-steady state is the Michaelis-Menten rate.
"""
from __future__ import annotations

import sympy as sp

from .ode_system import TIME, ODESystem

__all__ = ["sir", "seir", "hodgkin_huxley", "fitzhugh_nagumo", "pk_two_compartment", "michaelis_menten",
           "HH_NOMINAL"]


def _s(n: str) -> sp.Symbol:
    return sp.Symbol(n, real=True)


def _window(t_on: float, t_off: float) -> sp.Expr:
    return sp.Heaviside(TIME - sp.Float(t_on), 1) - sp.Heaviside(TIME - sp.Float(t_off), 1)


def sir(*, beta: float = 0.3, gamma: float = 0.1, N: float = 1.0e6, I0: float = 10.0, R0_init: float = 0.0,
        u: float = 0.0, t_on: float = 0.0, t_off: float = 0.0) -> ODESystem:
    """SIR in persons per day; ``u`` is the fractional contact reduction on ``[t_on, t_off)``."""
    S, I, R = _s("S"), _s("I"), _s("R")
    P = {n: _s(n) for n in ("beta", "gamma", "N", "u")}
    b = P["beta"] * (1 - P["u"] * _window(t_on, t_off)) if t_off > t_on else P["beta"]
    inf = b * S * I / P["N"]
    rhs = {"S": -inf, "I": inf - P["gamma"] * I, "R": P["gamma"] * I}
    init = {"S": P["N"] - float(I0) - float(R0_init), "I": float(I0), "R": float(R0_init)}
    return ODESystem("sir", list(rhs), rhs, init, {"beta": beta, "gamma": gamma, "N": N, "u": u},
                     {"R0": P["beta"] / P["gamma"], "prevalence": I / P["N"]},
                     units={"t": "day", "S": "persons", "I": "persons", "R": "persons", "beta": "1/day",
                            "gamma": "1/day"},
                     meta={"nonnegative": ["S", "I", "R"],
                           "mass_balance": {"weights": {"S": 1, "I": 1, "R": 1}, "total": "N"},
                           "intervention": {"t_on": float(t_on), "t_off": float(t_off)},
                           "source": "Kermack and McKendrick 1927"})


def seir(*, beta: float = 0.5, sigma: float = 0.2, gamma: float = 0.1, N: float = 1.0e6, I0: float = 10.0,
         E0: float = 0.0, u: float = 0.0, t_on: float = 0.0, t_off: float = 0.0) -> ODESystem:
    """SEIR (latent period ``1/sigma`` days) with the same intervention as :func:`sir`."""
    S, E, I, R = _s("S"), _s("E"), _s("I"), _s("R")
    P = {n: _s(n) for n in ("beta", "sigma", "gamma", "N", "u")}
    b = P["beta"] * (1 - P["u"] * _window(t_on, t_off)) if t_off > t_on else P["beta"]
    inf = b * S * I / P["N"]
    rhs = {"S": -inf, "E": inf - P["sigma"] * E, "I": P["sigma"] * E - P["gamma"] * I, "R": P["gamma"] * I}
    init = {"S": P["N"] - float(I0) - float(E0), "E": float(E0), "I": float(I0), "R": 0.0}
    return ODESystem("seir", list(rhs), rhs, init,
                     {"beta": beta, "sigma": sigma, "gamma": gamma, "N": N, "u": u},
                     {"R0": P["beta"] / P["gamma"], "prevalence": I / P["N"]},
                     units={"t": "day", "S": "persons", "E": "persons", "I": "persons", "R": "persons"},
                     meta={"nonnegative": ["S", "E", "I", "R"],
                           "mass_balance": {"weights": {"S": 1, "E": 1, "I": 1, "R": 1}, "total": "N"},
                           "intervention": {"t_on": float(t_on), "t_off": float(t_off)},
                           "source": "Kermack and McKendrick 1927 with a latent class"})


HH_NOMINAL = {"C": 1.0, "gNa": 120.0, "gK": 36.0, "gL": 0.3, "ENa": 50.0, "EK": -77.0, "EL": -54.387,
              "I_ext": 0.0, "amp": 10.0}


def hodgkin_huxley(*, pulse: tuple[float, float] = (1.0, 1.0), **parameters: float) -> ODESystem:
    """V in mV, t in ms, currents in uA/cm^2. ``pulse`` = (start, width) of a rectangular stimulus of amplitude
    ``amp``; ``I_ext`` is a sustained current. Initial state: the resting equilibrium at ``I_ext = 0``."""
    p = {**HH_NOMINAL, **parameters}
    V, m, h, n = _s("V"), _s("m"), _s("h"), _s("n")
    S = {k: _s(k) for k in p}
    am = sp.Float(0.1) * (V + 40) / (1 - sp.exp(-(V + 40) / 10))
    bm = 4 * sp.exp(-(V + 65) / 18)
    ah = sp.Float(0.07) * sp.exp(-(V + 65) / 20)
    bh = 1 / (1 + sp.exp(-(V + 35) / 10))
    an = sp.Float(0.01) * (V + 55) / (1 - sp.exp(-(V + 55) / 10))
    bn = sp.Float(0.125) * sp.exp(-(V + 65) / 80)
    t0, w = map(float, pulse)
    I_stim = S["amp"] * _window(t0, t0 + w) + S["I_ext"]
    i_ion = (S["gNa"] * m ** 3 * h * (V - S["ENa"]) + S["gK"] * n ** 4 * (V - S["EK"]) + S["gL"] * (V - S["EL"]))
    rhs = {"V": (I_stim - i_ion) / S["C"], "m": am * (1 - m) - bm * m, "h": ah * (1 - h) - bh * h,
           "n": an * (1 - n) - bn * n}
    # resting state at I_ext = 0 (V = -65 mV gives the gating steady states; the small offset is relaxed away)
    v0 = -64.99637933
    inf = {g: float((a / (a + b)).subs(V, v0)) for g, a, b in (("m", am, bm), ("h", ah, bh), ("n", an, bn))}
    return ODESystem("hodgkin_huxley", list(rhs), rhs, {"V": v0, **inf}, p,
                     {"I_stim": I_stim},
                     units={"t": "ms", "V": "mV", "I_stim": "uA/cm^2"},
                     meta={"bounded": {"m": (0.0, 1.0), "h": (0.0, 1.0), "n": (0.0, 1.0)}, "nonnegative": [],
                           "pulse": [t0, w], "source": "Hodgkin and Huxley 1952 (modern sign convention)"})


def fitzhugh_nagumo(*, a: float = 0.7, b: float = 0.8, eps: float = 0.08, I_ext: float = 0.0, amp: float = 1.0,
                    pulse: tuple[float, float] = (5.0, 1.0)) -> ODESystem:
    """``dv/dt = v - v^3/3 - w + I``, ``dw/dt = eps (v + a - b w)`` (dimensionless), started at rest."""
    import numpy as np
    v, w = _s("v"), _s("w")
    S = {k: _s(k) for k in ("a", "b", "eps", "I_ext", "amp")}
    t0, wd = map(float, pulse)
    I_stim = S["amp"] * _window(t0, t0 + wd) + S["I_ext"]
    rhs = {"v": v - v ** 3 / 3 - w + I_stim, "w": S["eps"] * (v + S["a"] - S["b"] * w)}
    # rest: v - v^3/3 - (v + a)/b + I_ext = 0
    roots = np.roots([-1 / 3, 0, 1 - 1 / b, -a / b + I_ext])
    vr = float(min(r.real for r in roots if abs(r.imag) < 1e-9))
    return ODESystem("fitzhugh_nagumo", list(rhs), rhs, {"v": vr, "w": (vr + a) / b},
                     {"a": a, "b": b, "eps": eps, "I_ext": I_ext, "amp": amp}, {"I_stim": I_stim},
                     meta={"nonnegative": [], "pulse": [t0, wd], "source": "FitzHugh 1961; Nagumo et al. 1962"})


def pk_two_compartment(*, dose: float = 100.0, interval: float = 0.0, n_doses: int = 1, ka: float = 1.0,
                       CL: float = 2.0, V1: float = 20.0, Q: float = 10.0, V2: float = 40.0, F: float = 1.0,
                       ke0: float = 0.5, E0: float = 0.0, Emax: float = 100.0, EC50: float = 1.0,
                       hill: float = 1.0) -> ODESystem:
    """Oral doses of ``dose`` mg every ``interval`` h (``n_doses``), time in h, volumes in L, clearances in L/h.

    The first dose starts in the gut (``A_gut(0) = F dose``); later doses enter through their absorption input
    ``F dose ka exp(-ka (t - t_j))``. Algebraic: ``C`` (central, mg/L), ``E`` (effect), ``dosed`` (amount absorbed or
    still to be absorbed by t, mg) so that ``A_gut + A_c + A_p + A_elim + gut_pending = dosed``."""
    A_gut, A_c, A_p, Ce = _s("A_gut"), _s("A_c"), _s("A_p"), _s("Ce")
    S = {n: _s(n) for n in ("ka", "CL", "V1", "Q", "V2", "F", "ke0", "E0", "Emax", "EC50", "hill", "dose")}
    times = [j * float(interval) for j in range(int(n_doses))] if n_doses > 1 else [0.0]
    extra = sp.Integer(0)
    pending = sp.Integer(0)        # amount of the later doses still in the gut (implicit, carried by the input)
    dosed = S["F"] * S["dose"]
    for tj in times[1:]:
        H = sp.Heaviside(TIME - sp.Float(tj), 1)
        decay = sp.exp(-S["ka"] * (TIME - sp.Float(tj)))
        extra += S["F"] * S["dose"] * S["ka"] * decay * H
        pending += S["F"] * S["dose"] * decay * H
        dosed += S["F"] * S["dose"] * H
    C = A_c / S["V1"]
    rhs = {"A_gut": -S["ka"] * A_gut,
           "A_c": S["ka"] * A_gut + extra - (S["CL"] + S["Q"]) * C + S["Q"] * A_p / S["V2"],
           "A_p": S["Q"] * (C - A_p / S["V2"]),
           "A_elim": S["CL"] * C,
           "AUC": C,
           "Ce": S["ke0"] * (C - Ce)}
    assign = {"C": C, "E": S["E0"] + S["Emax"] * Ce ** S["hill"] / (S["EC50"] ** S["hill"] + Ce ** S["hill"]),
              "gut_pending": pending, "dosed": dosed}
    params = dict(ka=ka, CL=CL, V1=V1, Q=Q, V2=V2, F=F, ke0=ke0, E0=E0, Emax=Emax, EC50=EC50, hill=hill,
                  dose=dose)
    init = {"A_gut": S["F"] * S["dose"], "A_c": 0.0, "A_p": 0.0, "A_elim": 0.0, "AUC": 0.0, "Ce": 0.0}
    return ODESystem("pk_two_compartment", list(rhs), rhs, init, params, assign,
                     units={"t": "h", "A_c": "mg", "C": "mg/L", "AUC": "mg h/L", "dose": "mg", "CL": "L/h"},
                     meta={"nonnegative": ["A_gut", "A_c", "A_p", "A_elim", "AUC", "Ce"],
                           "mass_balance": {"weights": {"A_gut": 1, "A_c": 1, "A_p": 1, "A_elim": 1,
                                                        "gut_pending": 1}, "total": "dosed"},
                           "dose_times": times,
                           "source": "linear two-compartment PK, first-order absorption; sigmoid Emax PD "
                                     "(Holford and Sheiner 1981)"})


def michaelis_menten(*, k1: float = 1.0e-2, km1: float = 1.0, k2: float = 0.5, E0: float = 10.0,
                     S0: float = 1000.0) -> ODESystem:
    """``E + S <-> C -> E + P`` with mass action (concentrations in uM, t in s); ``Km = (km1 + k2) / k1`` and
    ``Vmax = k2 E0`` are the Michaelis-Menten constants of the quasi-steady state."""
    E, Sb, C, P = _s("E"), _s("S"), _s("C"), _s("P")
    K = {n: _s(n) for n in ("k1", "km1", "k2")}
    v1, v2, v3 = K["k1"] * E * Sb, K["km1"] * C, K["k2"] * C
    rhs = {"E": -v1 + v2 + v3, "S": -v1 + v2, "C": v1 - v2 - v3, "P": v3}
    return ODESystem("michaelis_menten", list(rhs), rhs, {"E": float(E0), "S": float(S0), "C": 0.0, "P": 0.0},
                     {"k1": k1, "km1": km1, "k2": k2},
                     {"Km": (K["km1"] + K["k2"]) / K["k1"], "v_qssa": K["k2"] * (E + C) * Sb
                      / ((K["km1"] + K["k2"]) / K["k1"] + Sb)},
                     units={"t": "s", "S": "uM"},
                     meta={"nonnegative": ["E", "S", "C", "P"], "source": "Michaelis and Menten 1913; Briggs and "
                           "Haldane 1925", "E0": float(E0), "S0": float(S0)})

