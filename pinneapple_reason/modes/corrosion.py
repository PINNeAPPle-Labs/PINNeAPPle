"""Corrosion: galvanic coupling of two metals in an aerated electrolyte.

When a less noble metal (anode) is connected to a more noble one (cathode) in water, the cathode reaction in aerated
near-neutral water is oxygen reduction, usually limited by oxygen diffusion: ``i_lim = n F D C / delta`` with n = 4
(Jones, *Principles and Prevention of Corrosion*, 2nd ed., 1996, ch. 3 and 6). The anode supplies that whole current
over its own area, so its dissolution current density is ``i_a = i_lim A_c / A_a`` and its wall loss rate follows
Faraday's law ``CR = i_a M / (z F rho)``. A small anode next to a large cathode corrodes fast; the classic remedies
are to break the electrical path (insulating kit) or to coat the cathode (never only the anode).
"""
from __future__ import annotations

from ..chain import tr
from ..library import Countermeasure, FailureMode, Need, register_countermeasure, register_mode
from ..props import FARADAY
from ._util import fmt, q, verdict

JONES = "Jones, Principles and Prevention of Corrosion, 2nd ed., Prentice Hall, 1996, ch. 3 (Faraday's law) and 6 " \
        "(galvanic corrosion, area effect)"
YEAR = 365.25 * 24 * 3600
# molar mass (kg/mol), valence, density (kg/m^3): CRC Handbook of Chemistry and Physics
METALS = {"steel": (55.845e-3, 2, 7874.0), "zinc": (65.38e-3, 2, 7140.0), "aluminium": (26.98e-3, 3, 2700.0)}


def faraday_rate(i: float, metal: str) -> float:
    """Uniform wall-loss rate (m/s) of ``metal`` dissolving at current density ``i`` (A/m^2)."""
    M, z, rho = METALS[metal]
    return i * M / (z * FARADAY * rho)


def _galvanic(p, limits):
    metal = p.get("anode_metal", "steel")
    insulated = "insulating_kit" in p.get("_elements", ())
    D, C, delta = float(p.get("o2_diffusivity", 2.1e-9)), float(p.get("o2_concentration", 0.25)), \
        float(p["diffusion_layer"])
    i_lim = 4 * FARADAY * D * C / delta
    ratio = 0.0 if insulated else float(p["area_ratio"]) * (1 - float(p.get("cathode_coating_efficiency", 0.0)))
    i_a = i_lim * ratio
    rate = faraday_rate(i_a, metal) + float(p.get("self_corrosion_rate", 0.0))
    loss = rate * YEAR * float(p["service_years"])
    allow = float(p["corrosion_allowance"])
    return verdict(demand=loss * 1e3, capacity=allow * 1e3, unit="mm",
                   demand_label=tr("perda de parede na vida útil", "wall loss over the service life"),
                   capacity_label=tr("sobre-espessura de corrosão", "corrosion allowance"),
                   mechanism=tr(f"O cátodo consome O₂ no limite de difusão ({fmt(i_lim)} A/m²); com área de cátodo "
                                f"{fmt(ratio)}× a do ânodo, o ânodo de {metal} dissolve a {fmt(i_a)} A/m² "
                                f"({fmt(rate * YEAR * 1e3)} mm/ano): {fmt(loss * 1e3)} mm em "
                                f"{fmt(p['service_years'])} anos contra {fmt(allow * 1e3)} mm de sobre-espessura.",
                                f"The cathode consumes O2 at the diffusion limit ({fmt(i_lim)} A/m^2); with a cathode "
                                f"{fmt(ratio)}x the anode area, the {metal} anode dissolves at {fmt(i_a)} A/m^2 "
                                f"({fmt(rate * YEAR * 1e3)} mm/yr): {fmt(loss * 1e3)} mm in {fmt(p['service_years'])} "
                                f"years against a {fmt(allow * 1e3)} mm allowance."),
                   method="oxygen-diffusion-limited cathode, area ratio, Faraday's law",
                   quantities={"i_limit": q(i_lim, "A/m^2", "O2 diffusion-limited current density"),
                               "i_anode": q(i_a, "A/m^2", "anode current density"),
                               "rate": q(rate * YEAR, "m/yr", "anode wall-loss rate")},
                   assumptions=[tr("acoplamento com diferença de potencial suficiente para o cátodo trabalhar no "
                                   "limite de difusão", "potential difference large enough for the cathode to run at "
                                   "the diffusion limit")],
                   sources=[JONES, "CRC Handbook (molar masses, densities, O2 diffusivity in water at 25 °C)"])


register_mode(FailureMode(
    "galvanic_corrosion", "corrosion", tr("corrosão galvânica", "galvanic corrosion"),
    tr("ânodo pequeno ligado a cátodo grande perde parede mais rápido que a sobre-espessura",
       "a small anode coupled to a large cathode loses wall faster than the allowance"), _galvanic,
    needs=(Need("area_ratio", "1", "cathode / anode wetted area"), Need("diffusion_layer", "m", "O2 boundary layer"),
           Need("service_years", "yr", "service life"), Need("corrosion_allowance", "m", "corrosion allowance")),
    sources=(JONES,), countermeasures=("insulating_kit", "coat_cathode"),
    uncertain={"diffusion_layer": 0.5, "o2_concentration": 0.2},
    known_answers=lambda: [{"params": {"area_ratio": 1.0, "diffusion_layer": 4 * FARADAY * 2.1e-9 * 0.25,
                                       "service_years": 1.0, "corrosion_allowance": 1e-3, "anode_metal": "steel"},
                            "quantity": "rate", "expected": 1.16e-3, "rtol": 5e-3,
                            "source": "Faraday: 1 A/m^2 on iron = 1.16 mm/yr (Jones 1996, ch. 3)"}]))

register_countermeasure(Countermeasure(
    "insulating_kit", "insulating_kit", tr("kit isolante (junta e buchas dielétricas)",
                                           "insulating kit (dielectric gasket and sleeves)"),
    tr("abre o circuito elétrico entre os metais: sem corrente galvânica", "opens the electrical path between the "
       "metals: no galvanic current"),
    ("galvanic_corrosion",), lambda p, v: ({}, (), ()), lambda b, a: (1.0, "kit", "insulating kit"),
    unit_cost=0.3, unit_cost_note="relative index per kit (premise)"))

register_countermeasure(Countermeasure(
    "coat_cathode", "cathode_coating", tr("revestir o cátodo", "coat the cathode"),
    tr("a área catódica efetiva cai com a eficiência do revestimento, e com ela a corrente no ânodo",
       "the effective cathode area falls with the coating efficiency, and so does the anode current"),
    ("galvanic_corrosion",), lambda p, v: ({"cathode_coating_efficiency": float(p.get("coating_efficiency", 0.95))},
                                           (), ()),
    lambda b, a: (float(b.get("cathode_area", 1.0)), "m2", "coated cathode area"), unit_cost=0.05,
    unit_cost_note="relative index per m2 of coating (premise)"))
