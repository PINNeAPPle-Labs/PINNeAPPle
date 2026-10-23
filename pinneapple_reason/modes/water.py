"""Water: rain getting through the laps of a roof covering (capillarity, wind pressure, ponding / falling film).

Lapped coverings (tiles, corrugated sheets) are not watertight joints: they are hydraulic barriers. Water reaches the
joint and three heads push it up the lap (Garden 1963 lists the forces that move rain through a joint: gravity,
kinetic energy, surface tension, air pressure difference):

* capillary rise in the gap ``w`` between the overlapping pieces, ``h_c = 2 sigma cos(theta) / (rho g w)`` (two
  parallel plates; de Gennes, Brochard-Wyart & Quere, *Capillarity and Wetting Phenomena*, 2004, ch. 1);
* the wind pressure across the covering during the storm, ``h_w = Cp 1/2 rho_air V^2 / (rho g)``;
* the water depth over the joint: on a slope the rain runs off as a laminar falling film,
  ``delta = (3 nu q / (g sin(alpha)))^(1/3)`` with ``q = i L`` per unit width (Nusselt film; Bird, Stewart &
  Lightfoot, *Transport Phenomena*, sec. 2.2); on a flat roof nothing drives the water off and the design storm
  ponds: ``h = i t``.

The lap resists with the height it rises: ``L_lap sin(alpha)``. The covering leaks when the heads exceed it.
"""
from __future__ import annotations

import math

from ..chain import tr
from ..library import (
    Countermeasure,
    FailureMode,
    Infeasible,
    Need,
    register_countermeasure,
    register_mode,
    validity,
)
from ..props import SRC, WATER, G, air
from ._util import fmt, q, target_margin, verdict

GARDEN = "Garden (1963), 'Rain penetration and its control', NRC Canada, Canadian Building Digest CBD-40"
CAPILLARY = "de Gennes, Brochard-Wyart & Quere (2004), Capillarity and Wetting Phenomena, ch. 1 (rise between plates)"
NUSSELT = "Bird, Stewart & Lightfoot, Transport Phenomena, 2nd ed., sec. 2.2 (laminar falling film)"
FILM_REGIMES = "Incropera et al., sec. 10.6: falling film wave-free for Re_f < 30, wavy-laminar up to 1800"


def lap_heads(p: dict) -> dict:
    """Heads acting on the lap (m of water) and the lap's rise, for the parameters ``p``."""
    rho, mu, sig = WATER["rho"], WATER["mu"], WATER["sigma"]
    nu = mu / rho
    alpha = math.radians(float(p["slope_deg"]))
    w = float(p["lap_gap"])
    theta = math.radians(float(p.get("contact_angle_deg", 0.0)))
    h_cap = 2 * sig * math.cos(theta) / (rho * G * w)
    rho_air = air(float(p.get("t_rain", 20.0)))["rho"]
    h_wind = float(p.get("wind_cp", 1.0)) * 0.5 * rho_air * float(p["wind_speed_rain"]) ** 2 / (rho * G)
    i = float(p["rain_intensity"])
    run = float(p["roof_run"])                        # horizontal projection of the slope length
    qf = i * run                                      # rain collected per unit width at the eaves, m^2/s
    if math.sin(alpha) > 1e-6:
        h_film = (3 * nu * qf / (G * math.sin(alpha))) ** (1 / 3)
        regime = "film"
    else:
        h_film = i * float(p["storm_duration"])
        regime = "ponding"
    re_f = 4 * qf / nu
    return {"h_cap": h_cap, "h_wind": h_wind, "h_film": h_film, "regime": regime, "re_f": re_f, "q": qf,
            "rise": float(p["lap_length"]) * math.sin(alpha), "alpha": alpha}


def _lap(p, limits):
    h = lap_heads(p)
    demand = h["h_cap"] + h["h_wind"] + h["h_film"]
    cap = h["rise"]
    mm = 1e3
    film_pt = (f"a lâmina d'água que escoa tem {fmt(h['h_film'] * mm)} mm" if h["regime"] == "film" else
               f"sem inclinação nada tira a água do telhado: a chuva de projeto empoça {fmt(h['h_film'] * mm)} mm")
    film_en = (f"the run-off film is {fmt(h['h_film'] * mm)} mm deep" if h["regime"] == "film" else
               f"with no slope nothing drives the water off: the design storm ponds {fmt(h['h_film'] * mm)} mm")
    head = f"{fmt(h['h_cap'] * mm)} + {fmt(h['h_wind'] * mm)} + {fmt(h['h_film'] * mm)} = {fmt(demand * mm)} mm"
    if demand > cap:
        mech = tr(f"A água sobe pela fresta de {fmt(p['lap_gap'] * mm)} mm da sobreposição por capilaridade até "
                  f"{fmt(h['h_cap'] * mm)} mm, o vento empurra mais {fmt(h['h_wind'] * mm)} mm e {film_pt}: a carga "
                  f"total ({head}) é maior que os {fmt(cap * mm)} mm que a sobreposição de "
                  f"{fmt(p['lap_length'] * mm)} mm levanta a {fmt(p['slope_deg'])}°. A água passa por cima da "
                  "sobreposição e pinga dentro.",
                  f"Water climbs the {fmt(p['lap_gap'] * mm)} mm gap of the lap by capillarity up to "
                  f"{fmt(h['h_cap'] * mm)} mm, the wind pushes it {fmt(h['h_wind'] * mm)} mm further and {film_en}: "
                  f"the total head ({head}) exceeds the {fmt(cap * mm)} mm the {fmt(p['lap_length'] * mm)} mm lap "
                  f"rises at {fmt(p['slope_deg'])}°. Water gets over the lap and drips inside.")
    else:
        mech = tr(f"A sobreposição levanta {fmt(cap * mm)} mm, acima da carga de capilaridade + vento + lâmina "
                  f"({head}): a água escorre antes de vencer a sobreposição.",
                  f"The lap rises {fmt(cap * mm)} mm, above the capillary + wind + film head ({head}): the water "
                  "runs off before it gets over the lap.")
    val = [validity("film Reynolds number 4q/nu (laminar film)", h["re_f"], None, 1800,
                    FILM_REGIMES)] if h["regime"] == "film" else []
    val.append(validity("gap much smaller than capillary length (plates model)", p["lap_gap"] / 2.7e-3, None, 1.0,
                        "capillary length of water sqrt(sigma/(rho g)) = 2.7 mm"))
    return verdict(
        demand=demand * mm, capacity=cap * mm, unit="mm", scale=max(demand * mm, 1e-3),
        demand_label=tr("carga capilar + vento + lâmina", "capillary + wind + film head"),
        capacity_label=tr("altura que a sobreposição levanta", "height the lap rises"),
        mechanism=mech,
        method="hydraulic barrier of a lapped joint: capillary rise between plates + wind head + Nusselt film / "
               "ponding vs lap rise",
        quantities={"h_capillary": q(h["h_cap"], "m", "capillary rise in the lap gap"),
                    "h_wind": q(h["h_wind"], "m", "wind pressure head"),
                    "h_film": q(h["h_film"], "m", "run-off film depth" if h["regime"] == "film" else
                                "ponded depth of the design storm"),
                    "lap_rise": q(cap, "m", "vertical rise of the lap"),
                    "runoff_per_width": q(h["q"], "m^2/s", "run-off per unit width at the eaves"),
                    "film_reynolds": q(h["re_f"], "1", "film Reynolds number 4q/nu")},
        assumptions=[tr("fresta da sobreposição como duas placas paralelas, molhamento pelo ângulo de contato dado",
                        "lap gap modelled as two parallel plates wetted at the given contact angle"),
                     tr("pressão do vento integral através da cobertura (Cp dado)",
                        "full wind pressure across the covering (given Cp)"),
                     tr("lâmina somada à carga sobre a junta (conservador)",
                        "film depth added to the head on the joint (conservative)")],
        sources=[GARDEN, CAPILLARY, NUSSELT, SRC["crc"]], validity=val)


register_mode(FailureMode(
    "lap_water_ingress", "hydraulics / capillarity",
    tr("infiltração de chuva pela sobreposição", "rain ingress through the lap"),
    tr("capilaridade + vento + lâmina d'água contra a altura que a sobreposição levanta",
       "capillarity + wind + water depth against the height the lap rises"),
    _lap,
    needs=(Need("slope_deg", "deg", "roof pitch"), Need("lap_length", "m", "head lap of the covering"),
           Need("lap_gap", "m", "gap between overlapping pieces"),
           Need("rain_intensity", "m/s", "design rain intensity"), Need("roof_run", "m", "horizontal run eaves-ridge"),
           Need("wind_speed_rain", "m/s", "wind speed during the storm"),
           Need("storm_duration", "s", "design storm duration (ponding on a flat roof)")),
    sources=(GARDEN, CAPILLARY, NUSSELT),
    countermeasures=("increase_slope", "longer_lap"),
    uncertain={"lap_gap": 0.3, "wind_speed_rain": 0.2, "rain_intensity": 0.3},
    known_answers=lambda: [
        {"params": {"slope_deg": 0.0, "lap_length": 0.07, "lap_gap": 1e-3, "contact_angle_deg": 0.0,
                    "rain_intensity": 150e-3 / 3600, "roof_run": 5.0, "wind_speed_rain": 0.0, "wind_cp": 1.0,
                    "storm_duration": 600.0},
         "quantity": "h_capillary", "expected": 2 * 0.0728 / (998.2 * 9.80665 * 1e-3), "rtol": 1e-9,
         "source": "h = 2 sigma / (rho g w) for w = 1 mm, sigma = 0.0728 N/m (14.87 mm)"},
        {"params": {"slope_deg": 0.0, "lap_length": 0.07, "lap_gap": 1e-3, "contact_angle_deg": 0.0,
                    "rain_intensity": 150e-3 / 3600, "roof_run": 5.0, "wind_speed_rain": 0.0, "wind_cp": 1.0,
                    "storm_duration": 600.0},
         "quantity": "h_film", "expected": 0.025, "rtol": 1e-9,
         "source": "flat roof: 150 mm/h for 10 min ponds 25 mm"},
    ],
))


def minimum_slope(p: dict, margin: float = 0.0) -> float:
    """Smallest pitch (deg) at which the lap holds ``(1 + margin)`` times the head, by bisection (the film depth
    falls as the slope rises). ``inf`` if not even 75° is enough."""
    def ok(a):
        h = lap_heads({**p, "slope_deg": a})
        return h["rise"] >= (1 + margin) * (h["h_cap"] + h["h_wind"] + h["h_film"])
    lo, hi = 0.01, 75.0
    if not ok(hi):
        return math.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def _increase_slope(p, v):
    need = target_margin(p)
    a = minimum_slope(p, need)
    if not math.isfinite(a):
        raise Infeasible("nem a 75° a sobreposição segura a carga", "not even at 75° does the lap hold the head")
    a = math.ceil(a)
    if a <= float(p["slope_deg"]):
        a = float(p["slope_deg"]) + 1
    return {"slope_deg": float(a)}, (), ()


def _slope_cost(before, after):
    a0, a1 = math.radians(before["slope_deg"]), math.radians(after["slope_deg"])
    return 1 / math.cos(a1) - 1 / math.cos(a0), "m2/m2", tr("+{pct} de área de telhado", "+{pct} roof area")


register_countermeasure(Countermeasure(
    "increase_slope", "pitch",
    tr("inclinar o telhado até a inclinação mínima do material", "pitch the roof to the material's minimum slope"),
    tr("a gravidade passa a tirar a água (lâmina fina em vez de poça) e a sobreposição ganha altura "
       "(L·sen α) contra a capilaridade e o vento",
       "gravity now drains the water (thin film instead of a pond) and the lap gains height (L sin(alpha)) "
       "against capillarity and wind"),
    ("lap_water_ingress",), _increase_slope, _slope_cost, unit_cost=1.0,
    unit_cost_note="relative index per m2 of covering (premise)", repeatable=True))


def _longer_lap(p, v):
    h = lap_heads(p)
    s = math.sin(h["alpha"])
    if s < 1e-6:
        raise Infeasible("sem inclinação nenhuma sobreposição levanta a água (L·sen 0 = 0)",
                         "with no slope no lap lifts the water (L sin 0 = 0)")
    need = (1 + target_margin(p)) * (h["h_cap"] + h["h_wind"] + h["h_film"])
    L = math.ceil(need / s * 1e3) / 1e3
    gauge = float(p.get("unit_length", 0.4))
    if L > 0.5 * gauge:
        raise Infeasible(f"precisaria de sobreposição {L * 1e3:.0f} mm, mais de metade da peça",
                         f"would need a {L * 1e3:.0f} mm lap, over half the piece")
    return {"lap_length": L}, (), ()


def _lap_cost(before, after):
    u = float(before.get("unit_length", 0.4))
    return u / (u - after["lap_length"]) - u / (u - before["lap_length"]), "m2/m2", tr("+{pct} de telhas", "+{pct} covering")


register_countermeasure(Countermeasure(
    "longer_lap", "lap",
    tr("aumentar a sobreposição das peças", "lengthen the lap of the pieces"),
    tr("a sobreposição sobe mais na mesma inclinação", "the lap rises higher at the same pitch"),
    ("lap_water_ingress",), _longer_lap, _lap_cost, unit_cost=1.0,
    unit_cost_note="relative index per m2 of covering (premise)", repeatable=True))
