"""Two reactors from the textbooks, converted to SI, as starting points (replace the kinetics with your own measured
ones; every value says where it comes from).

* :func:`propylene_glycol_cstr`: hydrolysis of propylene oxide to propylene glycol in a cooled CSTR, water in excess
  and methanol as an inert (Fogler, *Elements of Chemical Reaction Engineering*, 5th ed., Examples 12-3 and 12-4;
  kinetics of Furusawa, Nishimura & Miyauchi, *J. Chem. Eng. Data* 14 (1969) 211). Liquid phase, first order in the
  oxide. It has several steady states, the case for the CSTR's multiplicity and start-up.
* :func:`phthalic_anhydride_bed`: partial oxidation of o-xylene to phthalic anhydride in a cooled multitubular
  packed bed, the classical parametric-sensitivity case (Froment & Bischoff, *Chemical Reactor Analysis and Design*,
  2nd ed., Example 11.7.A.1; Froment, *Ind. Eng. Chem.* 59 (1967) 18). The rate is written with both partial
  pressures (not pseudo-first-order: the oxygen consumed is followed); the Ergun pressure drop is added when the
  catalyst's particle size and voidage are given.
"""
from __future__ import annotations

import math

from .kinetics import R_GAS, Feed, Network, Reaction
from .models import Bed, Cooling

LBMOL_H = 0.45359237 / 3600 * 1000          # lbmol/h -> mol/s
FT3_H = 0.028316846592 / 3600                # ft^3/h -> m^3/s
BTU_LBMOL = 2.326                            # Btu/lbmol -> J/mol (exact, IT Btu)
BTU_LBMOL_F = 4.1868                         # Btu/(lbmol F) -> J/(mol K)
BTU_H_F = 1055.05585262 / 3600 * 1.8         # Btu/(h F) -> W/K
ATM = 101325.0


def _rankine(T_F: float) -> float:
    return (T_F + 459.67) * 5 / 9


def propylene_glycol_cstr() -> dict:
    """``{"network", "feed", "V", "cooling", "key", "desired", "source"}`` in SI."""
    fogler = "Fogler, 5th ed., Examples 12-3/12-4 (kinetics: Furusawa et al. 1969)"
    net = Network(
        ("PO", "W", "PG", "M"),
        (Reaction({"PO": -1, "W": -1, "PG": 1}, A=16.96e12 / 3600, Ea=32400 * BTU_LBMOL, orders={"PO": 1.0},
                  dH=-36400 * BTU_LBMOL, T_ref=_rankine(68.0), name="PO + H2O -> PG"),),
        cp={"PO": 35 * BTU_LBMOL_F, "W": 18 * BTU_LBMOL_F, "PG": 46 * BTU_LBMOL_F, "M": 19.5 * BTU_LBMOL_F},
        source=fogler)
    feed = Feed({"PO": 43.04 * LBMOL_H, "W": 802.8 * LBMOL_H, "M": 71.87 * LBMOL_H}, _rankine(75.0), "liquid",
                v=326.3 * FT3_H)
    return {"network": net, "feed": feed, "V": 40.1 * 0.028316846592,
            "cooling": Cooling(UA=16000 * BTU_H_F, Tc=_rankine(85.0)), "key": "PO", "desired": "PG",
            "source": fogler, "note": "the mixing temperature of the two feed streams (75 F) and the cooling "
                                      "(UA = 16 000 Btu/(h F), coolant at 85 F) follow the example; check them "
                                      "against your edition"}


def phthalic_anhydride_bed(p_xylene_atm: float = 0.011, T0: float = 625.0, bed: Bed | None = None) -> dict:
    """One tube of the bed: ``{"network", "feed", "V", "cooling", "bed", "key", "desired", "source", "assumptions"}``
    in SI. The example has no pressure drop: ``bed=None`` keeps it so and says so; pass a :class:`Bed` with the
    catalyst's particle size and voidage to add the Ergun term."""
    src = "Froment & Bischoff, 2nd ed., Example 11.7.A.1; Froment (1967)"
    rho_bed = 1300.0                                               # kg cat / m^3 bed
    D, L = 0.0254, 3.0
    G = 4684.0 / 3600                                              # kg/(m^2 s)
    U = 96.1 * 4186.8 / 3600                                       # kcal/(m^2 h K) -> W/(m^2 K)
    # ln k = 19.837 - 13636/T, k in kmol/(kg cat h atm^2) -> mol/(m^3 s Pa^2) per bed volume
    A = rho_bed * math.exp(19.837) * 1000 / 3600 / ATM ** 2
    mw = {"X": 0.10616, "O2": 0.031998, "N2": 0.028014, "PA": 0.14812, "H2O": 0.018015}
    # molar heat capacities near 650 K (approximate: ideal-gas tables); the mixture is mostly air
    cp = {"X": 250.0, "O2": 32.1, "N2": 30.1, "PA": 210.0, "H2O": 35.9}
    net = Network(("X", "O2", "N2", "PA", "H2O"),
                  (Reaction({"X": -1, "O2": -3, "PA": 1, "H2O": 3}, A=A, Ea=13636 * R_GAS,
                            orders={"X": 1.0, "O2": 1.0}, dH=-307e3 * 4.1868, basis="pressure", T_ref=T0,
                            name="o-xylene + 3 O2 -> PA + 3 H2O"),),
                  cp=cp, molar_mass=mw, source=src)
    y_x, y_o2 = p_xylene_atm, 0.208                               # mole fractions at 1 atm
    y_n2 = 1 - y_x - y_o2
    m_avg = y_x * mw["X"] + y_o2 * mw["O2"] + y_n2 * mw["N2"]
    F_tot = G * math.pi * D ** 2 / 4 / m_avg
    feed = Feed({"X": y_x * F_tot, "O2": y_o2 * F_tot, "N2": y_n2 * F_tot}, T0, "gas", P=ATM)
    return {"network": net, "feed": feed, "V": math.pi * D ** 2 / 4 * L, "cooling": Cooling(Ua=4 * U / D, Tc=T0),
            "bed": bed, "key": "X", "desired": "PA", "source": src,
            "assumptions": (["no pressure drop along the bed (as in the source example; pass bed=Bed(...) with the "
                             "catalyst's particle size and voidage for the Ergun term)"] if bed is None else [])
            + ["one reaction, as in the source example: the total oxidation to CO and CO2 (and so the selectivity "
               "loss) is not modelled; add those reactions with their kinetics for a selectivity study",
               "species heat capacities: approximate constant values near 650 K",
               "coolant (molten salt) at a uniform temperature equal to the inlet temperature"]}
