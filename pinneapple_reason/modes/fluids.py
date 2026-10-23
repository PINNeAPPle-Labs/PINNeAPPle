"""Liquids in pipes: pump cavitation (NPSH), water hammer on valve closure, column separation.

* ``pump_cavitation``: net positive suction head available, ``NPSH_a = (p_atm - p_v)/(rho g) - z_s - h_f``, with the
  suction friction from the Colebrook friction factor (:mod:`pinneapple_physics.closed_form.pipe_flow`) plus minor
  losses, against the pump's required NPSH plus a margin (Karassik et al., *Pump Handbook*, 4th ed., ch. 2).
* ``water_hammer`` and ``column_separation``: transient after the valve at the end of a main closes, by the method of
  characteristics (:func:`pinneapple_simulation.numerical_solvers.water_hammer_moc.valve_closure_moc`), wave speed
  from Korteweg's formula; Joukowsky and Michaud as the closed-form references.
"""
from __future__ import annotations

import functools
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
from ..props import SRC, STEEL, WATER, G, atmospheric_pressure, saturation_pressure
from ._util import fmt, q, self_check, size_to_pass, target_margin, verdict

KARASSIK = "Karassik, Messina, Cooper & Heald, Pump Handbook, 4th ed., McGraw-Hill, 2008, ch. 2 (NPSH)"
COLEBROOK = "pinneapple_physics.closed_form.pipe_flow.colebrook (Colebrook 1939)"
MOC = "pinneapple_simulation.numerical_solvers.water_hammer_moc.valve_closure_moc (method of characteristics)"
JOUKOWSKY = "Joukowsky (1898) dH = a dV/g; Michaud (1878) for slow closure; Wylie & Streeter (1993) ch. 1-3"


def _friction(v: float, d: float, rough: float) -> tuple[float, float]:
    from pinneapple_physics.closed_form.pipe_flow import colebrook, laminar
    re = v * d * WATER["rho"] / WATER["mu"]
    return (laminar(re) if re < 2300 else colebrook(re, rough / d)), re


# ------------------------------------------------------------------------------------------------ NPSH
def npsh(p: dict) -> dict:
    rho = WATER["rho"]
    patm = atmospheric_pressure(float(p.get("altitude", 0.0)))
    pv = saturation_pressure(float(p.get("t_water", 20.0)))
    d = float(p["suction_d"])
    v = float(p["pump_q"]) / (math.pi * d ** 2 / 4)
    f, re = _friction(v, d, float(p.get("suction_roughness", 4.5e-5)))
    hf = (f * float(p["suction_length"]) / d + float(p.get("suction_k", 0.0))) * v ** 2 / (2 * G)
    npsha = (patm - pv) / (rho * G) - float(p["suction_lift"]) - hf
    return {"npsha": npsha, "patm": patm, "pv": pv, "hf": hf, "v": v, "f": f, "re": re,
            "h_atm": patm / (rho * G), "h_vap": pv / (rho * G)}


def _cavitation(p, limits):
    r = npsh(p)
    need = float(p["npsh_required"]) + float(limits.get("npsh_margin", p.get("npsh_margin", 0.5)))
    ok = r["npsha"] >= need
    mech = tr(f"Na sucção sobram {fmt(r['npsha'])} m de NPSH: {fmt(r['h_atm'])} m de pressão atmosférica a "
              f"{fmt(p.get('altitude', 0.0))} m de altitude, menos {fmt(r['h_vap'])} m de pressão de vapor, "
              f"{fmt(p['suction_lift'])} m de altura de sucção e {fmt(r['hf'])} m de perda na tubulação "
              f"({fmt(r['v'])} m/s)" + (f"; a bomba precisa de {fmt(need)} m (NPSH requerido + margem): a pressão na "
                                        "entrada do rotor cai abaixo da de vapor, formam-se bolhas que implodem — "
                                        "cavitação, ruído e erosão." if not ok else
                                        f", acima dos {fmt(need)} m que a bomba exige."),
              f"The suction leaves {fmt(r['npsha'])} m of NPSH: {fmt(r['h_atm'])} m of atmospheric pressure at "
              f"{fmt(p.get('altitude', 0.0))} m altitude, minus {fmt(r['h_vap'])} m of vapour pressure, "
              f"{fmt(p['suction_lift'])} m of suction lift and {fmt(r['hf'])} m of pipe loss ({fmt(r['v'])} m/s)"
              + (f"; the pump needs {fmt(need)} m (required NPSH + margin): the pressure at the impeller eye falls "
                 "below vapour pressure, bubbles form and collapse — cavitation, noise and erosion." if not ok else
                 f", above the {fmt(need)} m the pump needs."))
    return verdict(demand=need, capacity=r["npsha"], unit="m", scale=need,
                   demand_label=tr("NPSH requerido + margem", "required NPSH + margin"),
                   capacity_label=tr("NPSH disponível", "available NPSH"), mechanism=mech,
                   method="NPSH_a = (p_atm - p_v)/(rho g) - z_s - h_f, Darcy-Colebrook suction loss + minor losses",
                   quantities={"npsh_available": q(r["npsha"], "m", "available NPSH"),
                               "suction_loss": q(r["hf"], "m", "suction friction + minor losses"),
                               "suction_velocity": q(r["v"], "m/s", "suction velocity"),
                               "h_atm": q(r["h_atm"], "m", "atmospheric head"),
                               "h_vapour": q(r["h_vap"], "m", "vapour-pressure head")},
                   assumptions=[tr("NPSH requerido do catálogo da bomba no ponto de operação (premissa)",
                                   "required NPSH from the pump curve at the duty point (premise)")],
                   sources=[KARASSIK, COLEBROOK, SRC["iso2533"], SRC["magnus"]],
                   validity=[validity("suction Reynolds number (turbulent Colebrook)", r["re"], 4000, None)])


register_mode(FailureMode(
    "pump_cavitation", "hydraulics / cavitation", tr("cavitação na bomba", "pump cavitation"),
    tr("NPSH disponível menor que o requerido: a pressão na entrada do rotor cai abaixo da de vapor",
       "available NPSH below the required: the pressure at the impeller eye falls below vapour pressure"),
    _cavitation,
    needs=(Need("suction_lift", "m", "pump above the sump level"), Need("suction_d", "m", "suction diameter"),
           Need("suction_length", "m", "suction length"), Need("pump_q", "m^3/s", "flow"),
           Need("npsh_required", "m", "required NPSH")),
    sources=(KARASSIK, COLEBROOK), countermeasures=("lower_pump", "larger_suction_pipe"),
    uncertain={"npsh_required": 0.1, "suction_k": 0.3, "t_water": 0.25},
    known_answers=lambda: [{"params": {"suction_lift": 0.0, "suction_d": 0.2, "suction_length": 1e-9, "pump_q": 1e-9,
                                       "npsh_required": 1.0, "altitude": 0.0, "t_water": 20.0},
                            "quantity": "h_atm", "expected": 101325.0 / (998.2 * 9.80665), "rtol": 1e-9,
                            "source": "sea-level standard atmosphere 101325 Pa = 10.35 m of water"}]))


def _lower_pump(p, v):
    r = npsh(p)
    need = (float(p["npsh_required"]) + float(p.get("npsh_margin", 0.5))) * (1 + target_margin(p))
    z = math.floor((r["h_atm"] - r["h_vap"] - r["hf"] - need) * 10) / 10
    if z >= float(p["suction_lift"]):
        raise Infeasible("baixar a bomba não muda nada", "lowering the pump changes nothing")
    return {"suction_lift": z}, (), ()


register_countermeasure(Countermeasure(
    "lower_pump", "pump_level", tr("baixar a bomba (poço / sucção afogada)", "lower the pump (pit / flooded suction)"),
    tr("cada metro a menos de altura de sucção devolve um metro de NPSH disponível",
       "every metre less of suction lift gives back a metre of available NPSH"),
    ("pump_cavitation",), _lower_pump,
    lambda b, a: ((b["suction_lift"] - a["suction_lift"]) * float(b.get("pump_pit_area", 12.0)), "m3",
                  tr("{q} m³ de escavação do poço da bomba", "{q} m^3 of pump-pit excavation")),
    unit_cost=1.0, unit_cost_note="relative index per m3 of excavation and concrete (premise)", repeatable=True))

_NOMINAL = (0.05, 0.065, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def _larger_suction(p, v):
    d0 = float(p["suction_d"])
    sizes = [d for d in _NOMINAL if d0 < d <= 2 * d0]
    if not sizes:
        raise Infeasible("sem diâmetro maior disponível", "no larger diameter available")
    try:
        ch, _ = size_to_pass("pump_cavitation", p, sizes, lambda d: {"suction_d": d})
    except Infeasible:
        ch = {"suction_d": sizes[-1]}                 # the best it can do: the engine weighs the gain
    return ch, (), ()


register_countermeasure(Countermeasure(
    "larger_suction_pipe", "suction_pipe", tr("tubulação de sucção maior", "larger suction pipe"),
    tr("velocidade menor na sucção: a perda cai com ~V² (D^-5 para a mesma vazão)",
       "lower suction velocity: the loss falls with ~V^2 (D^-5 at the same flow)"),
    ("pump_cavitation",), _larger_suction,
    lambda b, a: (math.pi * (a["suction_d"] - b["suction_d"]) * float(b.get("suction_wall", 0.006))
                  * b["suction_length"] * STEEL["rho"], "kg", tr("+{q} kg de aço na sucção", "+{q} kg of suction-pipe steel")),
    unit_cost=1.0, unit_cost_note="relative index per kg of installed pipe (premise)", repeatable=True))


# ------------------------------------------------------------------------------------------------ water hammer
def main_wave_speed(p: dict) -> float:
    from pinneapple_simulation.numerical_solvers.water_hammer_moc import wave_speed
    return wave_speed(WATER["K"], WATER["rho"], float(p["main_d"]), float(p["main_wall"]),
                      float(p.get("main_E", STEEL["E"])), 1 - float(p.get("main_nu", STEEL["nu"])) ** 2)


@functools.lru_cache(maxsize=64)
def _moc(L, D, a, f, Q, HR, HD, tc, em, n):
    from pinneapple_simulation.numerical_solvers.water_hammer_moc import valve_closure_moc
    return valve_closure_moc(length=L, diameter=D, wave_speed=a, friction_factor=f, q0=Q, h_reservoir=HR,
                             h_downstream=HD, t_close=tc, closure_exponent=em, n_reaches=n)


def transient(p: dict, instantaneous_frictionless: bool = False) -> dict:
    D = float(p["main_d"])
    Q = float(p["main_q"])
    v = Q / (math.pi * D ** 2 / 4)
    f, re = _friction(v, D, float(p.get("main_roughness", 4.5e-5)))
    a = main_wave_speed(p)
    args = (float(p["main_length"]), D, a, 0.0 if instantaneous_frictionless else f, Q, float(p["h_reservoir"]),
            float(p.get("h_downstream", 0.0)), 0.0 if instantaneous_frictionless else float(p["valve_close_time"]),
            float(p.get("closure_exponent", 1.0)), int(p.get("moc_reaches", 40)))
    r = _moc(*args)
    return {**r, "a": a, "f": f, "re": re, "v": v}


def _surge_check(p):
    r = transient(p, instantaneous_frictionless=True)
    rise = r["max_head"] - r["valve_head_steady"]
    rel = abs(rise - r["joukowsky"]) / r["joukowsky"]
    return self_check("solver vs closed form: MOC instantaneous closure vs Joukowsky", rel < 1e-6, rel, 1e-6,
                      MOC + "; " + JOUKOWSKY)


def _decay_check(p, r):
    """Second law: with friction the surge at the valve after closure cannot grow from one wave period to the last."""
    import numpy as np
    t, h = np.asarray(r["t"]), np.asarray(r["h_valve"])
    tc, T = float(p["valve_close_time"]), r["period"]
    first = h[(t >= tc) & (t < tc + T)]
    last = h[t >= t[-1] - T]
    if len(first) == 0 or len(last) == 0:
        return self_check("energy: surge decays after closure (friction)", True, None, None, MOC, "not run: short "
                          "record")
    ref = r["h_steady"][0]
    a1, a2 = float(np.max(np.abs(first - ref))), float(np.max(np.abs(last - ref)))
    return self_check("energy: surge decays after closure (friction)", a2 <= a1 * (1 + 1e-9), a2 / max(a1, 1e-12),
                      1.0, MOC)


def _hammer(p, limits):
    r = transient(p)
    rho = WATER["rho"]
    pmax = rho * G * r["max_head"]
    rating = float(limits.get("pressure_rating", p["pressure_rating"]))
    ok = pmax <= rating
    tc = float(p["valve_close_time"])
    slow = tc > r["period"]
    mech = tr(f"Fechar a válvula em {fmt(tc)} s para {fmt(r['v'])} m/s de água numa tubulação onde a onda de pressão "
              f"corre a {fmt(r['a'])} m/s (ida e volta 2L/a = {fmt(r['period'])} s): "
              + ("o fechamento é mais rápido que a ida e volta da onda, e o golpe é o de Joukowsky, "
                 if not slow else "mesmo mais lento que 2L/a, a válvula corta a maior parte da vazão no fim do curso; ")
              + f"a carga máxima no MOC é {fmt(r['max_head'])} m ({fmt(pmax / 1e6)} MPa)"
              + (f", acima da classe de pressão do tubo ({fmt(rating / 1e6)} MPa): o tubo ou as juntas rompem."
                 if not ok else f", dentro da classe do tubo ({fmt(rating / 1e6)} MPa)."),
              f"Closing the valve in {fmt(tc)} s on {fmt(r['v'])} m/s of water in a main where the pressure wave "
              f"travels at {fmt(r['a'])} m/s (round trip 2L/a = {fmt(r['period'])} s): "
              + ("the closure is faster than the round trip, so the surge is Joukowsky's; " if not slow else
                 "even slower than 2L/a, the valve cuts most of the flow at the end of its stroke; ")
              + f"the MOC peak head is {fmt(r['max_head'])} m ({fmt(pmax / 1e6)} MPa)"
              + (f", above the pipe's pressure class ({fmt(rating / 1e6)} MPa): pipe or joints burst." if not ok else
                 f", within the pipe's class ({fmt(rating / 1e6)} MPa)."))
    return verdict(demand=pmax / 1e6, capacity=rating / 1e6, unit="MPa",
                   demand_label=tr("pressão máxima do transiente", "peak transient pressure"),
                   capacity_label=tr("classe de pressão do tubo", "pipe pressure class"), mechanism=mech,
                   method="method of characteristics, reservoir-pipe-valve, Korteweg wave speed, Colebrook friction",
                   quantities={"wave_speed": q(r["a"], "m/s", "pressure-wave speed"),
                               "period_2L_a": q(r["period"], "s", "wave round trip 2L/a"),
                               "max_head": q(r["max_head"], "m", "peak head (MOC)"),
                               "joukowsky": q(r["joukowsky"], "m", "Joukowsky surge a V/g"),
                               "michaud": q(r["michaud"], "m", "Michaud surge estimate"),
                               "velocity": q(r["v"], "m/s", "steady velocity")},
                   assumptions=[tr("tubo horizontal na cota da válvula; ancorado contra movimento axial",
                                   "horizontal pipe at valve level; anchored against axial movement"),
                                tr("lei de fechamento τ = (1 - t/t_c)^Em", "closure law tau = (1 - t/t_c)^Em")],
                   sources=[MOC, JOUKOWSKY, SRC["wylie"], COLEBROOK],
                   validity=[validity("main Reynolds number (turbulent Colebrook)", r["re"], 4000, None)],
                   self_checks=[_surge_check(p), _decay_check(p, r)],
                   series={"t": r["t"][::4].tolist(), "h_valve": r["h_valve"][::4].tolist(),
                           "x": r["x"].tolist(), "h_max": r["h_max"].tolist(), "h_min": r["h_min"].tolist()})


_MAIN_NEEDS = (Need("main_length", "m", "main length"), Need("main_d", "m", "main diameter"),
               Need("main_wall", "m", "wall thickness"), Need("main_q", "m^3/s", "steady flow"),
               Need("h_reservoir", "m", "reservoir head above the valve"),
               Need("valve_close_time", "s", "valve closure time"))

register_mode(FailureMode(
    "water_hammer", "hydraulic transients", tr("golpe de aríete", "water hammer"),
    tr("sobrepressão do fechamento da válvula acima da classe do tubo",
       "valve-closure surge above the pipe pressure class"), _hammer,
    needs=_MAIN_NEEDS + (Need("pressure_rating", "Pa", "pipe pressure class"),),
    sources=(MOC, JOUKOWSKY), countermeasures=("slow_valve_closure", "higher_pressure_class"),
    uncertain={"main_q": 0.1, "main_wall": 0.1},
    known_answers=lambda: [{"params": {"main_length": 1000.0, "main_d": 0.3, "main_wall": 0.006, "main_q": 0.1,
                                       "h_reservoir": 50.0, "valve_close_time": 1.0, "pressure_rating": 2.5e6},
                            "quantity": "wave_speed",
                            "expected": math.sqrt((2.18e9 / 998.2) / (1 + 0.91 * 2.18e9 * 0.3 / (210e9 * 0.006))),
                            "rtol": 1e-12, "source": "Korteweg wave speed, Wylie & Streeter eq. 2.29, c1 = 1 - nu^2"}]))


def _column(p, limits):
    r = transient(p)
    rho = WATER["rho"]
    patm = atmospheric_pressure(float(p.get("altitude", 0.0)))
    pv = saturation_pressure(float(p.get("t_water", 20.0)))
    pmin = patm + rho * G * r["min_head"]
    ok = pmin >= pv
    mech = tr(f"Depois do golpe a onda volta negativa do reservatório: a carga mínima no MOC é {fmt(r['min_head'])} m "
              f"(pressão absoluta {fmt(pmin / 1e3)} kPa) em x = {fmt(r['min_head_at'])} m"
              + (f", abaixo da pressão de vapor ({fmt(pv / 1e3)} kPa): a coluna d'água se separa, e o fechamento da "
                 "cavidade gera um segundo golpe maior que o primeiro." if not ok else
                 f", acima da pressão de vapor ({fmt(pv / 1e3)} kPa): sem separação de coluna."),
              f"After the surge the wave returns negative from the reservoir: the MOC minimum head is "
              f"{fmt(r['min_head'])} m (absolute pressure {fmt(pmin / 1e3)} kPa) at x = {fmt(r['min_head_at'])} m"
              + (f", below vapour pressure ({fmt(pv / 1e3)} kPa): the water column separates, and the cavity collapse "
                 "drives a second surge larger than the first." if not ok else
                 f", above vapour pressure ({fmt(pv / 1e3)} kPa): no column separation."))
    return verdict(demand=pv / 1e3, capacity=pmin / 1e3, unit="kPa", scale=patm / 1e3,
                   demand_label=tr("pressão de vapor", "vapour pressure"),
                   capacity_label=tr("pressão absoluta mínima do transiente", "minimum absolute transient pressure"),
                   mechanism=mech, method="method of characteristics (no cavity model: a minimum below vapour "
                                         "pressure flags column separation)",
                   quantities={"min_head": q(r["min_head"], "m", "minimum head (MOC)"),
                               "p_min_abs": q(pmin, "Pa", "minimum absolute pressure")},
                   sources=[MOC, SRC["wylie"], SRC["magnus"]])


register_mode(FailureMode(
    "column_separation", "hydraulic transients", tr("separação de coluna (cavitação no transiente)",
                                                    "column separation (transient cavitation)"),
    tr("a onda negativa leva a pressão abaixo da de vapor", "the negative wave takes the pressure below vapour "
       "pressure"), _column, needs=_MAIN_NEEDS, sources=(MOC,),
    countermeasures=("slow_valve_closure",), uncertain={"main_q": 0.1}))


def _slow_close(p, v):
    t0 = float(p["valve_close_time"])
    ch, _ = size_to_pass(v.mode, p, [t for t in (2, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240) if t > t0],
                         lambda t: {"valve_close_time": float(t)})
    return ch, (), ()


register_countermeasure(Countermeasure(
    "slow_valve_closure", "valve_actuator", tr("fechamento lento da válvula (atuador temporizado)",
                                               "slow valve closure (timed actuator)"),
    tr("a vazão cai durante várias idas e voltas da onda: as reflexões negativas do reservatório cancelam parte da "
       "sobrepressão (Michaud)",
       "the flow falls over several wave round trips: the negative reflections from the reservoir cancel part of the "
       "surge (Michaud)"),
    ("water_hammer", "column_separation"), _slow_close, lambda b, a: (1.0, "actuator", tr("1 atuador temporizado", "1 timed actuator")),
    unit_cost=40.0, unit_cost_note="relative index per actuator (premise)", repeatable=True))

_PN = (1.0e6, 1.6e6, 2.5e6, 4.0e6)


def _pn(p, v):
    r0 = float(p["pressure_rating"])
    e0 = float(p["main_wall"])
    ch, _ = size_to_pass("water_hammer", p, [r for r in _PN if r > r0],
                         lambda r: {"pressure_rating": r, "main_wall": round(e0 * r / r0, 4)})
    return ch, (), ()


register_countermeasure(Countermeasure(
    "higher_pressure_class", "pipe_class", tr("tubo de classe de pressão maior", "pipe of a higher pressure class"),
    tr("parede mais espessa resiste ao pico (Barlow), mas também enrijece o tubo e acelera a onda",
       "a thicker wall holds the peak (Barlow), but it also stiffens the pipe and speeds up the wave"),
    ("water_hammer",), _pn,
    lambda b, a: (math.pi * b["main_d"] * (a["main_wall"] - b["main_wall"]) * b["main_length"] * STEEL["rho"], "kg",
                  tr("+{q} kg de aço na parede da adutora", "+{q} kg of main pipe-wall steel")),
    unit_cost=1.0, unit_cost_note="relative index per kg of installed pipe (premise)"))
