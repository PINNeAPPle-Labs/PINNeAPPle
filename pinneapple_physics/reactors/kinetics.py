"""Reaction networks: stoichiometry, Arrhenius power-law rates, heats of reaction with their temperature dependence.

A :class:`Reaction` has stoichiometric coefficients (negative for reactants), a rate

    r = A exp(-Ea / (R T)) prod_i c_i ** n_i          [mol/(m^3 s) of extent, per reactor volume]

on a concentration basis (``c_i`` in mol/m^3) or a partial-pressure basis (``c_i`` replaced by ``p_i`` in Pa), and a
heat of reaction ``dH`` [J/mol of extent] at ``T_ref`` (default 298.15 K). With constant species heat capacities the heat of reaction
at ``T`` is ``dH + dcp (T - T_ref)`` with ``dcp = sum_i nu_i cp_i`` (Kirchhoff's law), so the energy balances that
use it are exact for constant ``cp``, not the ``dcp = 0`` shortcut.

A rate on a catalyst-mass basis (mol/(kg s)) enters as ``A * rho_bed`` (the bulk density of the bed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314462618          # J/(mol K), CODATA 2018 (exact)
T_REF = 298.15               # K, reference temperature of the heats of reaction


@dataclass(frozen=True)
class Reaction:
    stoich: dict[str, float]               # species -> nu (negative: consumed)
    A: float                               # pre-exponential factor, units set by the orders and the basis
    Ea: float = 0.0                        # activation energy, J/mol
    orders: dict[str, float] | None = None  # species -> order (default: |nu| of each reactant, an elementary step)
    dH: float = 0.0                        # heat of reaction at T_ref, J/mol of extent (negative: exothermic)
    basis: str = "concentration"           # "concentration" (mol/m^3) or "pressure" (Pa)
    name: str = ""
    T_ref: float = T_REF                   # K, temperature of dH

    def order_of(self) -> dict[str, float]:
        if self.orders is not None:
            return dict(self.orders)
        return {s: -v for s, v in self.stoich.items() if v < 0}


@dataclass(frozen=True)
class Network:
    species: tuple[str, ...]
    reactions: tuple[Reaction, ...]
    cp: dict[str, float]                   # species -> molar heat capacity, J/(mol K) (constant)
    molar_mass: dict[str, float] = field(default_factory=dict)   # kg/mol (needed for a packed bed's pressure drop)
    source: str = ""

    def __post_init__(self):
        for r in self.reactions:
            bad = set(r.stoich) - set(self.species) | set(r.order_of()) - set(self.species)
            if bad:
                raise ValueError(f"reaction {r.name or r.stoich}: unknown species {sorted(bad)}")
            if r.basis not in ("concentration", "pressure"):
                raise ValueError(f"basis is 'concentration' or 'pressure', not {r.basis!r}")
        missing = set(self.species) - set(self.cp)
        if missing:
            raise ValueError(f"no heat capacity for {sorted(missing)}")

    @property
    def nu(self) -> np.ndarray:
        """Stoichiometric matrix, (species, reactions)."""
        return np.array([[r.stoich.get(s, 0.0) for r in self.reactions] for s in self.species], float)

    @property
    def cp_vec(self) -> np.ndarray:
        return np.array([self.cp[s] for s in self.species], float)

    def index(self, s: str) -> int:
        return self.species.index(s)

    def rates(self, c: np.ndarray, p: np.ndarray, T: float) -> np.ndarray:
        """Extent rates of every reaction, mol/(m^3 s), from concentrations (mol/m^3) and partial pressures (Pa)."""
        out = np.empty(len(self.reactions))
        for j, r in enumerate(self.reactions):
            x = c if r.basis == "concentration" else p
            v = r.A * math.exp(-r.Ea / (R_GAS * T))
            for s, n in r.order_of().items():
                xi = max(float(x[self.index(s)]), 0.0)          # no negative concentration inside a power law
                v *= xi ** n
            out[j] = v
        return out

    def dH_at(self, T: float) -> np.ndarray:
        """Heats of reaction at ``T`` (Kirchhoff, constant cp), J/mol of extent."""
        dcp = self.nu.T @ self.cp_vec
        return np.array([r.dH + d * (T - r.T_ref) for r, d in zip(self.reactions, dcp, strict=True)], float)


@dataclass(frozen=True)
class Feed:
    """Molar flows (mol/s) at temperature ``T`` (K). A liquid keeps its volumetric flow ``v`` (m^3/s); a gas is ideal
    at pressure ``P`` (Pa), so its concentrations follow the local temperature, pressure and total molar flow."""
    F: dict[str, float]
    T: float
    phase: str = "liquid"
    v: float | None = None
    P: float = 101325.0

    def vector(self, net: Network) -> np.ndarray:
        bad = set(self.F) - set(net.species)
        if bad:
            raise ValueError(f"feed species not in the network: {sorted(bad)}")
        return np.array([float(self.F.get(s, 0.0)) for s in net.species], float)

    def __post_init__(self):
        if self.phase not in ("liquid", "gas"):
            raise ValueError("phase is 'liquid' or 'gas'")
        if self.phase == "liquid" and not self.v:
            raise ValueError("a liquid feed needs its volumetric flow v (m^3/s)")


def concentrations(F: np.ndarray, T: float, P: float, feed: Feed) -> tuple[np.ndarray, np.ndarray]:
    """``(c, p)``: concentrations (mol/m^3) and partial pressures (Pa) of the flows ``F`` at ``T``, ``P``."""
    if feed.phase == "liquid":
        c = F / feed.v
        return c, c * R_GAS * T                       # p of a liquid is not used by a liquid-phase rate
    Ft = max(float(F.sum()), 1e-300)
    y = F / Ft
    return y * P / (R_GAS * T), y * P


def first_order_pfr_conversion(T: float, tau: float, A: float, Ea: float) -> dict[str, float]:
    """Conversion of an isothermal plug-flow reactor with a first-order Arrhenius reaction,
    ``X = 1 - exp(-A exp(-Ea / (R T)) tau)`` (Levenspiel, 3rd ed., eq. 5.17 with k(T) of Arrhenius): ``T`` in K,
    ``tau`` residence time in s, ``A`` in 1/s, ``Ea`` in J/mol."""
    k = A * math.exp(-Ea / (R_GAS * T))
    return {"X": 1.0 - math.exp(-k * tau), "k": k}
