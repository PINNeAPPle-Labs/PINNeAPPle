"""Steady plug-flow (PFR, packed bed) and stirred-tank (CSTR) reactors with mole, energy and momentum balances.

PFR (Fogler, *Elements of Chemical Reaction Engineering*, 5th ed., ch. 11-12), along the reactor volume ``V``:

    dF_i/dV = sum_j nu_ij r_j
    dT/dV   = [Ua (Tc - T) + sum_j r_j (-dH_j(T))] / sum_i F_i cp_i
    dP/dV   = -(G / (rho Dp)) (1 - phi) / phi^3 [150 (1 - phi) mu / Dp + 1.75 G] / Ac      (Ergun, packed bed only)

``Ua`` is the heat-transfer coefficient times the wall area per reactor volume (4 U / D for a tube), ``Tc`` the
coolant temperature (constant: a boiling or well-mixed coolant). ``rho`` is the local gas density from the ideal
gas law, so the Ergun term follows the pressure, the temperature and the molar flow.

CSTR, with the volume ``V`` and the total heat-removal ``UA``:

    F_i0 - F_i + V sum_j nu_ij r_j = 0
    sum_i F_i0 cp_i (T0 - T) + UA (Tc - T) + V sum_j r_j (-dH_j(T)) = 0

Every steady state is found (an exothermic CSTR can have three): at each temperature the mole balances are solved,
which gives the heat generated ``G(T)`` and removed ``R(T)`` (van Heerden's construction); the steady states are the
roots of ``G - R``, refined by Brent's method. A state is stable only if ``dR/dT > dG/dT`` there (the slope
condition, necessary for stability). :func:`cstr_startup` integrates the transient from a reactor full of feed to
say which state a start-up reaches (liquid phase).

A liquid keeps its volumetric flow (its density is taken as constant along the reactor: state it when it is not).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

from .kinetics import R_GAS, Feed, Network, concentrations


@dataclass(frozen=True)
class Cooling:
    """Heat removal: ``Ua`` per reactor volume (W/(m^3 K), a PFR) or ``UA`` in total (W/K, a CSTR), coolant at
    ``Tc``. ``Ua = UA = 0`` is adiabatic."""
    Ua: float = 0.0
    Tc: float = 298.15
    UA: float | None = None


@dataclass(frozen=True)
class Bed:
    """A packed bed (Ergun pressure drop): tube ``diameter`` (m), particle diameter ``particle`` (m), ``voidage``
    (-), gas viscosity ``mu`` (Pa s). Needs the species' molar masses in the network."""
    diameter: float
    particle: float
    voidage: float
    mu: float

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4


@dataclass
class Profile:
    """A PFR solution along the volume: flows ``F`` (n, species), ``T``, ``P``, and the rates at each point."""
    network: Network
    feed: Feed
    V: np.ndarray
    F: np.ndarray
    T: np.ndarray
    P: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def conversion(self, key: str) -> np.ndarray:
        i = self.network.index(key)
        return 1.0 - self.F[:, i] / self.F[0, i]

    def rate_of(self, key: str) -> np.ndarray:
        """Net production rate of ``key`` along the reactor, mol/(m^3 s)."""
        i = self.network.index(key)
        out = []
        for F, T, P in zip(self.F, self.T, self.P, strict=True):
            c, p = concentrations(F, T, P, self.feed)
            out.append(float(self.network.nu[i] @ self.network.rates(c, p, T)))
        return np.array(out)

    def outlet(self) -> dict[str, Any]:
        return {"F": dict(zip(self.network.species, map(float, self.F[-1]), strict=True)), "T": float(self.T[-1]),
                "P": float(self.P[-1]), "T_max": float(self.T.max()),
                "V_at_T_max": float(self.V[int(self.T.argmax())])}


def _pfr_rhs(net: Network, feed: Feed, cooling: Cooling, bed: Bed | None):
    nu, cp = net.nu, net.cp_vec
    mass = None
    if bed is not None:
        if feed.phase != "gas":
            raise ValueError("a packed bed's Ergun pressure drop is written for a gas")
        miss = set(net.species) - set(net.molar_mass)
        if miss:
            raise ValueError(f"a packed bed needs the molar mass of {sorted(miss)}")
        mw = np.array([net.molar_mass[s] for s in net.species])
        mass = float(feed.vector(net) @ mw)                          # kg/s, constant along the bed

    def rhs(_V, y):
        F, T, P = y[:-2], y[-2], y[-1]
        c, p = concentrations(F, T, P, feed)
        r = net.rates(c, p, T)
        dF = nu @ r
        dT = (cooling.Ua * (cooling.Tc - T) + float(r @ -net.dH_at(T))) / float(F @ cp)
        dP = 0.0
        if bed is not None:
            G = mass / bed.area
            rho = P * mass / (max(float(F.sum()), 1e-300) * R_GAS * T)
            phi = bed.voidage
            dPdz = -(G / (rho * bed.particle)) * (1 - phi) / phi ** 3 * (150 * (1 - phi) * bed.mu / bed.particle
                                                                         + 1.75 * G)
            dP = dPdz / bed.area
        return np.concatenate([dF, [dT, dP]])
    return rhs


def pfr(net: Network, feed: Feed, V: float, cooling: Cooling | None = None, bed: Bed | None = None, *,
        n: int = 201, rtol: float = 1e-8, atol: float = 1e-12, stop: tuple[str, float] | None = None) -> Profile:
    """Integrate the PFR over ``V`` (m^3). ``stop=(key, X)`` ends at the volume where ``key`` reaches conversion ``X``
    (the PFR's size for that conversion; ``info["reached"]`` says whether it did within ``V``)."""
    cooling = cooling or Cooling()
    F0 = feed.vector(net)
    y0 = np.concatenate([F0, [feed.T, feed.P]])
    rhs = _pfr_rhs(net, feed, cooling, bed)
    events = []
    if stop is not None:
        i, X = net.index(stop[0]), float(stop[1])

        def ev(_V, y, i=i, X=X):
            return (1 - y[i] / F0[i]) - X
        ev.terminal, ev.direction = True, 1
        events.append(ev)
    if bed is not None:
        def low_p(_V, y):
            return y[-1] - 0.05 * feed.P                               # the bed has lost 95 % of its pressure
        low_p.terminal = True
        events.append(low_p)
    sol = solve_ivp(rhs, (0.0, float(V)), y0, method="LSODA", rtol=rtol, atol=atol, dense_output=True,
                    events=events or None)
    if not sol.success:
        raise RuntimeError(f"PFR integration failed: {sol.message}")
    V_end = float(sol.t[-1])
    grid = np.linspace(0.0, V_end, n)
    Y = sol.sol(grid).T
    info = {"V_end": V_end, "nfev": int(sol.nfev), "rtol": rtol,
            "pressure_lost": bool(bed is not None and len(sol.t_events[-1]))}
    if stop is not None:
        info["reached"] = bool(len(sol.t_events[0]))
    return Profile(net, feed, grid, Y[:, :-2], Y[:, -2], Y[:, -1], info)


# ------------------------------------------------------------------------------------------------------------ CSTR


@dataclass
class SteadyState:
    T: float
    F: dict[str, float]
    generated: float                 # W, heat released by the reactions
    removed: float                   # W, heat taken by the flow and the coolant
    slope_generated: float           # dG/dT, W/K
    slope_removed: float             # dR/dT, W/K
    stable: bool                     # slope condition: dR/dT > dG/dT

    def conversion(self, key: str, feed: Feed) -> float:
        return 1.0 - self.F[key] / feed.F[key]


def _cstr_moles_at(net: Network, feed: Feed, V: float, T: float, guess: np.ndarray) -> np.ndarray:
    F0, nu = feed.vector(net), net.nu
    scale = max(float(F0.sum()), 1e-30)

    def res(F):
        c, p = concentrations(np.maximum(F, 0.0), T, feed.P, feed)
        return (F0 - F + V * (nu @ net.rates(c, p, T))) / scale
    best = None
    for g in (guess, F0, 0.5 * F0 + 1e-12):
        sol, _info, ok, _msg = fsolve(res, g, full_output=True, xtol=1e-12)
        err = float(np.abs(res(sol)).max())
        if best is None or err < best[1]:
            best = (sol, err)
        if ok == 1 and err < 1e-10 and (sol >= -1e-9 * scale).all():
            return np.maximum(sol, 0.0)
    if best[1] > 1e-6:
        raise RuntimeError(f"CSTR mole balances did not converge at T = {T:.2f} K (residual {best[1]:.1e})")
    return np.maximum(best[0], 0.0)


def _heat(net: Network, feed: Feed, V: float, cooling: Cooling, T: float, F: np.ndarray) -> tuple[float, float]:
    c, p = concentrations(F, T, feed.P, feed)
    gen = V * float(net.rates(c, p, T) @ -net.dH_at(T))
    UA = cooling.UA if cooling.UA is not None else cooling.Ua * V
    rem = float(feed.vector(net) @ net.cp_vec) * (T - feed.T) + UA * (T - cooling.Tc)
    return gen, rem


def cstr(net: Network, feed: Feed, V: float, cooling: Cooling | None = None, *, T_range: tuple[float, float] | None
         = None, n_scan: int = 400) -> list[SteadyState]:
    """Every steady state of the CSTR of volume ``V`` (m^3), coldest first."""
    cooling = cooling or Cooling()
    F0 = feed.vector(net)
    if T_range is None:
        lo = min(feed.T, cooling.Tc) - 1.0
        heat = sum(max(-float(h), 0.0) * min((F0[net.index(s)] / -v for s, v in r.stoich.items() if v < 0),
                                              default=0.0) for r, h in zip(net.reactions, net.dH_at(feed.T),
                                                                           strict=True))
        T_range = (lo, max(feed.T, cooling.Tc) + 1.5 * heat / float(F0 @ net.cp_vec) + 50.0)
    Ts = np.linspace(*T_range, n_scan)
    guess = F0.copy()
    rows = []
    for T in Ts:
        F = _cstr_moles_at(net, feed, V, float(T), guess)
        guess = F
        g, r = _heat(net, feed, V, cooling, float(T), F)
        rows.append((float(T), F, g - r))
    states = []

    def diff(T, g0):
        F = _cstr_moles_at(net, feed, V, T, g0)
        g, r = _heat(net, feed, V, cooling, T, F)
        return g - r
    for (Ta, Fa, da), (Tb, _Fb, db) in zip(rows[:-1], rows[1:], strict=True):
        if da == 0 or da * db < 0:
            T = Ta if da == 0 else brentq(diff, Ta, Tb, args=(Fa,), xtol=1e-9)
            F = _cstr_moles_at(net, feed, V, T, Fa)
            g, r = _heat(net, feed, V, cooling, T, F)
            h = 1e-3
            gp, rp = _heat(net, feed, V, cooling, T + h, _cstr_moles_at(net, feed, V, T + h, F))
            gm, rm = _heat(net, feed, V, cooling, T - h, _cstr_moles_at(net, feed, V, T - h, F))
            sg, sr = (gp - gm) / (2 * h), (rp - rm) / (2 * h)
            states.append(SteadyState(T, dict(zip(net.species, map(float, F), strict=True)), g, r, sg, sr, sr > sg))
    return states


def cstr_startup(net: Network, feed: Feed, V: float, cooling: Cooling | None = None, *, T_start: float | None = None,
                 t_end_tau: float = 200.0) -> dict[str, Any]:
    """The transient of a liquid CSTR started full of feed at ``T_start`` (default: the feed temperature), to the
    steady state it settles on: ``{"T", "F", "t", "settled"}``. The contents' heat capacity is that of the species."""
    if feed.phase != "liquid":
        raise ValueError("the start-up transient is written for a liquid CSTR")
    cooling = cooling or Cooling()
    F0, nu, cp = feed.vector(net), net.nu, net.cp_vec
    c0 = F0 / feed.v
    tau = V / feed.v
    UA = cooling.UA if cooling.UA is not None else cooling.Ua * V

    def rhs(_t, y):
        c, T = np.maximum(y[:-1], 0.0), y[-1]
        r = net.rates(c, c * R_GAS * T, T)
        dc = (c0 - c) / tau + nu @ r
        heat = feed.v * float(c0 @ cp) * (feed.T - T) + UA * (cooling.Tc - T) + V * float(r @ -net.dH_at(T))
        return np.concatenate([dc, [heat / (V * float(c @ cp))]])
    y0 = np.concatenate([c0, [T_start if T_start is not None else feed.T]])
    sol = solve_ivp(rhs, (0, t_end_tau * tau), y0, method="LSODA", rtol=1e-8, atol=1e-10)
    yN = sol.y[:, -1]
    d = np.abs(rhs(0, yN))
    settled = bool(d[-1] * tau < 1e-6 * max(abs(yN[-1]), 1.0))
    return {"T": float(yN[-1]), "F": dict(zip(net.species, map(float, yN[:-1] * feed.v), strict=True)),
            "t": float(sol.t[-1]), "settled": settled}


def cstr_series(net: Network, feed: Feed, volumes: list[float], cooling: Cooling | None = None) -> list[dict]:
    """CSTRs in series: each tank's outlet feeds the next; per tank, every steady state and the one a start-up from
    the incoming stream reaches (liquid) or the coldest stable one (gas). Returns per tank ``{"V", "states",
    "chosen"}``."""
    out, f = [], feed
    for V in volumes:
        states = cstr(net, f, V, cooling)
        if f.phase == "liquid":
            s = cstr_startup(net, f, V, cooling)
            chosen = min(states, key=lambda st: abs(st.T - s["T"]))
        else:
            chosen = next(st for st in states if st.stable)
        out.append({"V": V, "states": states, "chosen": chosen})
        f = Feed(dict(chosen.F), chosen.T, f.phase, f.v, f.P)
    return out
