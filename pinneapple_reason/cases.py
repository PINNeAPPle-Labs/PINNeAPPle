"""Worked cases at different scales and in different fields, each starting from the naive concept.

* :func:`roof`: flat tiles -> pitch (rain through the laps) -> ceiling with a ventilated attic (condensation on a clear
  night) -> what the warm ceiling still needs (radiant heat on a sunny day).
* :func:`footbridge`: girders sized for strength only -> deeper girders (deflection) -> lateral bracing
  (lateral-torsional buckling) -> tuned mass damper or stiffer girders (resonance with footfall).
* :func:`power_board`: a MOSFET on a bare board -> heat sink (Joule heating, natural convection) -> forced air.
* :func:`pumped_main`: pump above the sump and a fast-closing valve on the main -> lower pump (cavitation) -> slow
  closure (water hammer, column separation).

Every value of a case is a premise (written in ``Concept.sources``) or a cited standard value; the results are what
the failure modes compute. No client data.
"""
from __future__ import annotations

from .chain import Concept, Requirement, tr

__all__ = ["roof", "footbridge", "power_board", "pumped_main", "CASES", "ROOFING", "get"]

P = "premise"


def _concept(name, description, values: dict, elements=()) -> Concept:
    params = {k: v[0] if isinstance(v, tuple) else v for k, v in values.items()}
    sources = {k: (v[1] if isinstance(v, tuple) else P) for k, v in values.items()}
    return Concept(name, params, list(elements), description, sources, 0)


NBR10844 = ("150 mm/h: intensity ABNT NBR 10844:1989 allows for small roofs without local rainfall data (premise; "
            "use the local 5-year intensity)")

#: illustrative lap geometries of roof coverings (premises, not catalogue values): lap, gap, wet contact angle,
#: piece length
ROOFING = {
    "ceramic_tile": {"label": tr("telha cerâmica", "clay tile"), "lap_length": 0.075, "lap_gap": 1.0e-3,
                     "contact_angle_deg": 20.0, "unit_length": 0.40},
    "concrete_tile": {"label": tr("telha de concreto", "concrete tile"), "lap_length": 0.075, "lap_gap": 1.5e-3,
                      "contact_angle_deg": 30.0, "unit_length": 0.42},
    "fibre_cement_sheet": {"label": tr("telha ondulada de fibrocimento", "corrugated fibre-cement sheet"),
                           "lap_length": 0.14, "lap_gap": 1.0e-3, "contact_angle_deg": 40.0, "unit_length": 1.83},
    "steel_sheet": {"label": tr("telha metálica trapezoidal", "trapezoidal steel sheet"), "lap_length": 0.20,
                    "lap_gap": 0.5e-3, "contact_angle_deg": 70.0, "unit_length": 6.0},
}


def roof():
    """A house roof in a warm, humid climate."""
    reqs = [
        Requirement("R1", tr("manter a chuva de projeto fora do ambiente", "keep the design rain out of the room"),
                    ["lap_water_ingress"]),
        Requirement("R2", tr("não pingar condensação no ambiente", "no condensation dripping into the room"),
                    ["underside_condensation"]),
        Requirement("R3", tr("teto sem desconforto radiante num dia de sol",
                             "no radiant discomfort from the ceiling on a sunny day"),
                    ["radiant_heat_gain"], source="ANSI/ASHRAE 55 (warm ceiling < 5 °C asymmetry)"),
    ]
    c = _concept("roof", tr("telhas cerâmicas assentadas planas sobre a estrutura, sem forro",
                            "clay tiles laid flat on the structure, no ceiling"), {
        "slope_deg": 0.0, **{k: v for k, v in ROOFING["ceramic_tile"].items() if k != "label"},
        "rain_intensity": (150e-3 / 3600, NBR10844), "roof_run": 5.0, "wind_speed_rain": 10.0, "wind_cp": 1.0,
        "storm_duration": 600.0,
        "tile_thickness": 0.012, "tile_k": 1.0, "tile_emissivity": 0.9, "tile_absorptance": 0.65,
        "t_out_night": 16.0, "rh_out_night": 0.9, "wind_night": 1.0, "t_room_night": 21.0, "rh_room_night": 0.75,
        "t_out_day": 32.0, "wind_day": 2.0, "solar_irradiance": 900.0, "t_room_day": 27.0,
        "ceiling_R": (0.05, "premise: 12.5 mm gypsum board, lambda 0.25 W/(m K)"), "ceiling_emissivity": 0.9,
        "attic_vent": (0.003, "premise: ~10 air changes per hour of a 1.1 m-high attic"),
        "barrier_emissivity": 0.05, "insulation_k": 0.04,
        "design_margin": (0.35, "premise: covers the spread of the lap gap, wind and rain premises"),
    }, ["tiles"])
    return reqs, c, tr("Telhado: da telha plana ao telhado que funciona", "Roof: from flat tiles to a working roof")


def footbridge():
    """A 30 m steel footbridge (two welded girders, 2.5 m deck)."""
    reqs = [
        Requirement("R1", tr("resistir à multidão com segurança (ELU)", "carry the crowd safely (ULS)"),
                    ["bending_strength"]),
        Requirement("R2", tr("não flechar a ponto de assustar (ELS)", "no alarming deflection (SLS)"),
                    ["deflection"]),
        Requirement("R3", tr("não flambar de lado", "no lateral buckling"), ["lateral_torsional_buckling"]),
        Requirement("R4", tr("não vibrar de forma desconfortável", "no uncomfortable vibration"),
                    ["pedestrian_resonance"]),
    ]
    c = _concept("footbridge", tr("duas vigas soldadas dimensionadas só pela resistência",
                                  "two welded girders sized for strength only"), {
        "span": 30.0, "deck_width": 2.5, "n_girders": 2,
        "deck_dead_load": (3000.0, "premise: 100 mm precast concrete planks (25 kN/m3, EN 1991-1-1 Table A.1) "
                                   "+ 0.5 kN/m2 finishes, on cross-beams"),
        "girder_h": 0.80, "girder_b": 0.25, "girder_tf": 0.020, "girder_tw": 0.010, "section_type": "welded",
        "fy": (355e6, "EN 10025-2 S355, t <= 16 mm (premise grade)"), "lateral_braces": 0,
        "damping_ratio": (0.005, "premise: welded steel footbridge"), "pedestrian_weight": (700.0, "Bachmann & "
                                                                                             "Ammann (1987)"),
        "brace_mass_per_m": (3.77, "premise: L50x50x5 angle"), "design_margin": 0.05,
    }, ["girders"])
    return reqs, c, tr("Passarela de 30 m: da viga que resiste à que não incomoda",
                       "30 m footbridge: from a girder that holds to one nobody minds walking on")


def power_board():
    """A 20 A MOSFET switch on a 4-layer board inside an enclosure at 40 °C."""
    reqs = [Requirement("R1", tr("conduzir 20 A sem passar do limite de junção",
                                 "carry 20 A without exceeding the junction limit"), ["joule_heating"],
                        {"tj_max": 125.0}, "premise: design limit 125 °C (derated from the 175 °C rating)")]
    c = _concept("power_board", tr("MOSFET D2PAK soldado numa placa de 4 camadas, sem dissipador",
                                   "D2PAK MOSFET soldered on a 4-layer board, no heat sink"), {
        "current": 20.0, "r_on_25": (0.020, "premise: datasheet-style R_DS(on) at 25 °C"),
        "r_on_tc": (0.006, "premise: R_DS(on) about x1.75 at 150 °C"), "t_ambient": 40.0, "tj_max": 125.0,
        "package": "TO-263 D2PAK", "theta_jb": (1.0, "premise: junction-to-board through the soldered tab"),
        "board_w_mm": 100.0, "board_d_mm": 80.0, "board_layers": 4, "thermal_vias": 0, "max_vias": 40,
        "theta_jc_tab": (0.8, "premise: junction to mounting tab, tab-mount package on the heat sink"),
        "tab_w": 0.010, "tab_d": 0.015, "tim_r": (1e-5, "premise: thin grease, K m^2/W"),
        "hs_max_w": 0.040, "hs_max_d": 0.040, "hs_max_h": 0.020, "hs_base_t": 0.003, "hs_fin_t": 0.0012,
        "hs_material": "al6063",
        "design_margin": 0.05,
    }, ["board"])
    return reqs, c, tr("Placa de potência: do componente na placa à convecção forçada",
                       "Power board: from a part on the board to forced air")


def pumped_main():
    """A pumping station lifting water from a sump into a tank, and the gravity main from the tank to a valve."""
    reqs = [
        Requirement("R1", tr("bombear sem cavitar", "pump without cavitation"), ["pump_cavitation"]),
        Requirement("R2", tr("fechar a válvula sem romper a adutora", "close the valve without bursting the main"),
                    ["water_hammer", "column_separation"]),
    ]
    c = _concept("pumped_main", tr("bomba 5 m acima do poço; válvula no fim da adutora fechando em 2 s",
                                   "pump 5 m above the sump; valve at the end of the main closing in 2 s"), {
        "altitude": 800.0, "t_water": 20.0,
        "suction_lift": 5.0, "suction_d": 0.15, "suction_length": 12.0,
        "suction_k": (3.0, "premise: foot valve with strainer, elbow and entrance"),
        "pump_q": 0.05, "npsh_required": (4.0, "premise: pump curve at the duty point"), "npsh_margin": 0.5,
        "pump_pit_area": 12.0,
        "main_length": 2400.0, "main_d": 0.25, "main_wall": 0.005, "main_q": 0.05, "h_reservoir": 60.0,
        "h_downstream": 0.0, "valve_close_time": 2.0, "closure_exponent": 1.0,
        "pressure_rating": (1.0e6, "premise: PN10 pipe class"),
        "design_margin": (0.15, "premise: covers the spread of the NPSH curve and losses"),
    }, ["pump", "main", "valve"])
    return reqs, c, tr("Estação de bombeamento e adutora: da bomba que cavita ao fechamento que não rompe",
                       "Pumping station and main: from a cavitating pump to a closure that holds")


CASES = {"roof": roof, "footbridge": footbridge, "power_board": power_board, "pumped_main": pumped_main}


def get(name: str):
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; known: {sorted(CASES)}")
    return CASES[name]()
