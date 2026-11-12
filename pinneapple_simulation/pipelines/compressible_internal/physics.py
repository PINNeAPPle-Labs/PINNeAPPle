"""The gas and the conditions of a compressible internal-flow case — nothing about the geometry.

A gas here is what an equation of state plus a viscosity law needs and no more: critical
constants, molar mass, ideal-gas ``Cp``, conductivity, Sutherland coefficients. A condition set is
what the boundary conditions need: pressures, inlet temperature, mass flow, turbulence level,
scheme order. :class:`Gas` knows its own properties (density by Peng-Robinson, viscosity by
Sutherland, ideal-gas ``Cp``), so the same numbers reach the OpenFOAM dictionaries, the checks and
the tables from one place.

The real gas is the point of the module. At 70 bar and 300 K methane is at ``Tr = 1.57`` and
``Pr = 1.52``: the ideal gas puts it at 45.0 kg/m3, Peng-Robinson at 51.7 and GERG-2008 at 50.5.
A run with ``perfectGas`` is 13 % light in mass flow and, worse, has *no* Joule-Thomson cooling at
all — its enthalpy is a function of temperature only, so an isenthalpic throttle cannot change the
temperature. That is the single most common reason a valve CFD "shows no cooling", and it is the
reason this module computes the departure enthalpy itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Gas", "GASES", "Conditions", "peng_robinson", "sutherland_mu", "R_UNIVERSAL"]

R_UNIVERSAL = 8.31446261815324     # J/mol/K


@dataclass(frozen=True)
class Gas:
    """One gas: equation of state, viscosity law and ideal-gas caloric data."""

    name: str
    molWeight: float            # g/mol
    Tc: float                   # K
    Pc: float                   # Pa
    Vc: float                   # m3/kg
    omega: float                # acentric factor
    Cp_ig: float                # J/kg/K, ideal-gas Cp at the reference temperature
    kappa: float                # W/m/K, constant conductivity
    T_std: float = 300.0        # K, where Cp_ig is quoted
    #: Sutherland: mu = As T^1.5 / (T + Ts). ``As`` is set by :func:`with_sutherland`
    As: float = 0.0
    Ts: float = 0.0

    @property
    def Rs(self) -> float:
        """Specific gas constant, J/kg/K."""
        return R_UNIVERSAL / (self.molWeight * 1e-3)

    @property
    def gamma_ig(self) -> float:
        """Ideal-gas heat capacity ratio, for initialisation only."""
        return self.Cp_ig / (self.Cp_ig - self.Rs)

    def openfoam_dict(self) -> dict[str, float]:
        """The entries OpenFOAM's ``PengRobinsonGas`` reads."""
        return {"molWeight": self.molWeight, "Tc": self.Tc, "Pc": self.Pc, "Vc": self.Vc,
                "omega": self.omega}


#: Sutherland constants fitted so that each gas's low-pressure viscosity is right at 300 K.
GASES: dict[str, Gas] = {
    "methane": Gas("methane", 16.04303, 190.56, 4.599e6, 0.006146, 0.011, 2210.0, 0.0332,
                   As=9.72e-07, Ts=155.0),
    "natural_gas": Gas("natural_gas", 17.4, 198.0, 4.5e6, 0.0060, 0.020, 2260.0, 0.0360,
                       As=9.10e-07, Ts=170.0),
    "air": Gas("air", 28.96, 132.5, 3.786e6, 0.0031, 0.035, 1005.0, 0.0242,
               As=1.458e-06, Ts=110.4),
    "co2": Gas("co2", 44.01, 304.13, 7.377e6, 0.00214, 0.2236, 843.0, 0.0145,
               As=1.48e-06, Ts=240.0),
}


def peng_robinson(gas: Gas, p_pa: float, T_K: float) -> dict[str, float]:
    """Peng-Robinson: the compressibility root and the properties the checks need.

    Returns ``Z``, ``rho``, the departure ``h_dep`` (J/kg) and ``Cp`` (J/kg/K). The departure terms
    are the standard PR integrals; they are what makes an isenthalpic throttle cool the gas, and
    ``h_dep`` is compared with the library's GERG-2008 reference in :mod:`.checks`.
    """
    rs = gas.Rs
    a = 0.45724 * (rs * gas.Tc) ** 2 / gas.Pc
    b = 0.07780 * rs * gas.Tc / gas.Pc
    kappa = 0.37464 + 1.54226 * gas.omega - 0.26992 * gas.omega ** 2
    sqrt_tr = math.sqrt(T_K / gas.Tc)
    alpha = (1.0 + kappa * (1.0 - sqrt_tr)) ** 2
    A = a * alpha * p_pa / (rs * T_K) ** 2
    B = b * p_pa / (rs * T_K)
    roots = np.roots([1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B, -(A * B - B ** 2 - B ** 3)])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > B]
    if not real:
        raise ValueError(f"Peng-Robinson has no gas root at p = {p_pa:.4g} Pa, T = {T_K:g} K")
    Z = max(real)
    rho = p_pa / (Z * rs * T_K)
    root2 = math.sqrt(2.0)
    arg = (Z + (1.0 + root2) * B) / (Z + (1.0 - root2) * B)
    h_dep = rs * gas.Tc * (T_K / gas.Tc * (Z - 1.0)
                           - 2.078 * (1.0 + kappa) * math.sqrt(alpha) * math.log(arg))
    return {"Z": Z, "rho": rho, "h_dep": h_dep, "A": A, "B": B, "alpha": alpha, "log_arg": math.log(arg)}


def sutherland_mu(gas: Gas, T_K: float) -> float:
    """Low-pressure viscosity, Pa s. Sutherland, as the reference document uses."""
    return gas.As * T_K ** 1.5 / (T_K + gas.Ts)


@dataclass
class Conditions:
    """What a run imposes — the boundary conditions and the numerical settings, no geometry.

    The inlet is driven by **mass flow** and the outlet by **static pressure**. That is not the
    document's ``p_in = 70 bar`` wording, and the difference is deliberate: imposing static pressure
    at both ends of a 73 cm3 domain makes the first pressure correction take the whole start-up
    imbalance and overshoot by two orders of magnitude (measured: ``p`` went to 3.3e8 Pa in one
    iteration and the run died). With the flow imposed, the inlet pressure is an *output*, and
    :func:`pinneapple_simulation.pipelines.compressible_internal.study.solve_to_pressure_drop`
    iterates it onto the pressure the customer asked for.
    """

    p_out_bar: float = 60.0
    T_in_K: float = 300.0
    mass_flow_kg_s: float | None = None   # None -> estimate it from the target pressure drop
    p_in_target_bar: float | None = None     # the outer loop's target; None = one solve only
    turbulence_intensity: float = 0.05
    scheme: str = "linearUpwind"
    turbulence: str = "kOmegaSST"
    iterations: int = 3000
    write_interval: int = 1000
    #: the pressure equation's transonic branch. It is what the shipped tutorial uses and it is
    #: unstable at M ~ 0.5 in a throttling gap (measured on CFD-08: the continuity error went from
    #: 3e3 kg/s in one iteration to 1e-2 with it off).
    transonic: bool = True
    #: relaxation for U, h, k and omega (the pressure is relaxed by ``consistent`` when on)
    relaxation: float = 0.9
    n_non_orth: int = 1
    wall_thermal: str = "adiabatic"          # adiabatic | isothermal
    T_wall_K: float | None = None
    #: Solve in **physical time** (``rhoPimpleFoam``) instead of pseudo-time (``rhoSimpleFoam``).
    #: The steady formulation is the cheaper answer for a passage whose flow is steady; when it is
    #: not -- a throttling curtain at M ~ 0.5 with a shear layer that will not settle -- the steady
    #: run has no answer to converge to and this is the honest one: march in time for several
    #: acoustic transits and average over the last part of it.
    transient: bool = False
    #: physical end time, s. ``None`` means ``end_time_transits`` acoustic transits of the passage.
    end_time_s: float | None = None
    end_time_transits: float = 15.0
    max_co: float = 1.0
    #: faixa em que a correção de pressão do PIMPLE pode levar p a cada iteração. Um salto grande
    #: sobre um domínio parado faz a primeira correção sub-expandir o gás a 37 bar (medido), e com
    #: Peng-Robinson a inversão T(h) morre; apertar a faixa é o que segura a partida.
    p_min_factor: float = 0.1
    p_max_factor: float = 2.0
    n_outer: int = 2
    #: fraction of the run after which the time mean is accumulated (OpenFOAM ``fieldAverage``)
    average_from: float = 0.6

    def as_dict(self) -> dict[str, Any]:
        return {"p_out_bar": self.p_out_bar, "T_in_K": self.T_in_K,
                "mass_flow_kg_s": self.mass_flow_kg_s, "p_in_target_bar": self.p_in_target_bar,
                "turbulence_intensity": self.turbulence_intensity, "scheme": self.scheme,
                "turbulence": self.turbulence, "wall_thermal": self.wall_thermal, "iterations": self.iterations,
                "transonic": self.transonic, "relaxation": self.relaxation, "n_non_orth": self.n_non_orth,
                "transient": self.transient, "end_time_s": self.end_time_s, "max_co": self.max_co,
                "p_min_factor": self.p_min_factor, "p_max_factor": self.p_max_factor}


def resolve_gas(gas: Gas | str | None) -> Gas:
    if gas is None:
        return GASES["methane"]
    if isinstance(gas, Gas):
        return gas
    key = gas.strip().lower().replace(" ", "_")
    if key not in GASES:
        raise KeyError(f"unknown gas {gas!r}; known: {', '.join(sorted(GASES))}")
    return GASES[key]


def blend(gases: dict[str, float], name: str = "blend") -> Gas:
    """Kay's rule on the critical constants — enough for a first look, not a GERG replacement."""
    total = sum(gases.values())
    if total <= 0:
        raise ValueError("the composition sums to zero")
    parts = [(resolve_gas(g), w / total) for g, w in gases.items()]
    return Gas(
        name=name,
        molWeight=sum(g.molWeight * w for g, w in parts),
        Tc=sum(g.Tc * w for g, w in parts),
        Pc=sum(g.Pc * w for g, w in parts),
        Vc=sum(g.Vc * w for g, w in parts),
        omega=sum(g.omega * w for g, w in parts),
        Cp_ig=sum(g.Cp_ig * w for g, w in parts),
        kappa=sum(g.kappa * w for g, w in parts),
        As=sum(g.As * w for g, w in parts),
        Ts=sum(g.Ts * w for g, w in parts),
    )
