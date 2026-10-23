"""Heat: condensation under a roof on a clear night, radiant heat from a sun-baked roof, Joule heating of a power
device.

Roof (``roof_network``): a steady one-dimensional network per m^2 of roof. Outer tile surface: absorbed sun, long-wave
exchange with the clear sky (Swinbank sky temperature), wind convection (Watmuff et al.). Tile: conduction. Without a
ceiling the tile underside faces the room; with a ceiling and a ventilated attic: tile underside <-> ceiling top by
radiation (parallel grey plates), both to the attic air by convection, attic air to outside by ventilation, ceiling
by conduction, ceiling underside to the room. Surface coefficients by heat-flow direction from ISO 6946 Annex C
(``h = h_c + eps 4 sigma T_m^3``). The nonlinear system is solved with ``scipy.optimize.fsolve``; the global energy
balance closes to round-off and is reported as a self-check.

Joule heating (``joule_heating``): conduction loss of a MOSFET, ``P = I^2 R_on(T_j)`` with
``R_on = R_25 (1 + a (T_j - 25))``, coupled to the thermal path until ``T_j`` is consistent (electrothermal fixed
point; the loop gain ``a I^2 R_25 R_th`` >= 1 means thermal runaway). Thermal paths: the board (3-D finite-volume
board model :mod:`pinneapple_physics.pcb`), or a plate-fin heat sink in natural or forced convection
(:mod:`pinneapple_physics.closed_form.plate_fin_heatsink`).
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
from ..props import ISO6946, SIGMA, SRC, air, dew_point, sky_temperature, wind_convection
from ._util import fmt, q, self_check, size_to_pass, verdict

ASHRAE55 = ("ANSI/ASHRAE Standard 55, Thermal Environmental Conditions for Human Occupancy: allowable radiant "
            "temperature asymmetry, warm ceiling < 5 °C")
GREY_PLATES = "Incropera et al., eq. 13.24 (radiation between large parallel grey plates)"
K0 = 273.15


# ------------------------------------------------------------------------------------------------ roof network
def _hc_down_facing(t_surf, t_air):
    """Convective coefficient of a downward-facing surface (ISO 6946 Annex C): hotter than the air below ->
    heat flows down (stable, 0.7); colder -> heat flows up (5.0)."""
    return ISO6946["hc_down"] if t_surf > t_air else ISO6946["hc_up"]


def _hc_up_facing(t_surf, t_air):
    return ISO6946["hc_up"] if t_surf > t_air else ISO6946["hc_down"]


def _hr(eps, t1, t2):
    tm = 0.5 * (t1 + t2)
    return eps * 4 * SIGMA * tm ** 3


def roof_network(p: dict, case: str) -> dict:
    """Steady temperatures (°C) and heat fluxes (W/m^2) of the roof for ``case`` = "night" or "day"."""
    from scipy.optimize import fsolve

    night = case == "night"
    t_out = (p["t_out_night"] if night else p["t_out_day"]) + K0
    t_room = (p["t_room_night"] if night else p["t_room_day"]) + K0
    wind = p["wind_night"] if night else p["wind_day"]
    gsol = 0.0 if night else float(p["solar_irradiance"])
    a_s = float(p.get("tile_absorptance", 0.6))
    e_t = float(p.get("tile_emissivity", 0.9))
    e_u = float(p.get("tile_under_emissivity", e_t))
    r_t = float(p["tile_thickness"]) / float(p["tile_k"])
    h_e = wind_convection(wind)
    t_sky = sky_temperature(t_out)
    ceiling = "ceiling" in p.get("_elements", ())
    e_c = float(p.get("ceiling_emissivity", 0.9))
    r_c = float(p.get("ceiling_R", 0.05))
    vent = float(p.get("attic_vent", 0.003))
    ac = air(t_out - K0)
    rcv = ac["rho"] * ac["cp"] * vent
    grey = 1 / (1 / e_u + 1 / e_c - 1)

    dirs = {"u": None, "ct": None, "cb": None}
    sol = None
    for _ in range(6):
        if not ceiling:
            def f(x, dirs=dirs):
                to, tu = x
                hu = dirs["u"] if dirs["u"] is not None else ISO6946["hc_up"]
                hi = hu + _hr(e_u, tu, t_room)
                return [a_s * gsol - e_t * SIGMA * (to ** 4 - t_sky ** 4) - h_e * (to - t_out) + (tu - to) / r_t,
                        hi * (t_room - tu) - (tu - to) / r_t]
            x0 = [t_out, 0.5 * (t_out + t_room)] if sol is None else sol
            sol = fsolve(f, x0, xtol=1e-12)
            to, tu = sol
            new = {"u": _hc_down_facing(tu, t_room), "ct": None, "cb": None}
        else:
            def f(x, dirs=dirs):
                to, tu, ta, tct, tcb = x
                hu = dirs["u"] if dirs["u"] is not None else ISO6946["hc_up"]
                hct = dirs["ct"] if dirs["ct"] is not None else ISO6946["hc_up"]
                hcb = dirs["cb"] if dirs["cb"] is not None else ISO6946["hc_up"]
                qr = grey * SIGMA * (tu ** 4 - tct ** 4)
                hi = hcb + _hr(e_c, tcb, t_room)
                return [a_s * gsol - e_t * SIGMA * (to ** 4 - t_sky ** 4) - h_e * (to - t_out) + (tu - to) / r_t,
                        (tu - to) / r_t + hu * (tu - ta) + qr,
                        hu * (ta - tu) + hct * (ta - tct) + rcv * (ta - t_out),
                        hct * (tct - ta) - qr + (tct - tcb) / r_c,
                        (tcb - tct) / r_c + hi * (tcb - t_room)]
            x0 = [t_out] * 3 + [t_room] * 2 if sol is None else sol
            sol = fsolve(f, x0, xtol=1e-12)
            to, tu, ta, tct, tcb = sol
            new = {"u": _hc_down_facing(tu, ta), "ct": _hc_up_facing(tct, ta), "cb": _hc_down_facing(tcb, t_room)}
        if new == dirs:
            break
        dirs = new
    out = {"t_sky": t_sky - K0, "h_e": h_e, "t_out": t_out - K0, "t_room": t_room - K0}
    absorbed = a_s * gsol
    q_sky = e_t * SIGMA * (to ** 4 - t_sky ** 4)
    q_conv = h_e * (to - t_out)
    if not ceiling:
        hi = dirs["u"] + _hr(e_u, tu, t_room)
        q_room = hi * (tu - t_room)
        out.update(t_tile_top=to - K0, t_tile_under=tu - K0, t_room_surface=tu - K0, q_room=q_room, q_vent=0.0)
    else:
        hi = dirs["cb"] + _hr(e_c, tcb, t_room)
        q_room = hi * (tcb - t_room)
        out.update(t_tile_top=to - K0, t_tile_under=tu - K0, t_attic=ta - K0, t_ceiling_top=tct - K0,
                   t_ceiling_under=tcb - K0, t_room_surface=tcb - K0, q_room=q_room, q_vent=rcv * (ta - t_out),
                   q_tile_to_ceiling_rad=grey * SIGMA * (tu ** 4 - tct ** 4))
    resid = absorbed - q_sky - q_conv - out["q_vent"] - q_room
    out.update(absorbed=absorbed, q_sky=q_sky, q_conv_out=q_conv, h_room=hi,
               balance_residual=resid, balance_scale=max(abs(absorbed), abs(q_sky), abs(q_conv), abs(q_room), 1.0))
    return out


def _roof_quantities(r: dict) -> dict:
    keys = {"t_tile_top": "outer tile surface", "t_tile_under": "tile underside", "t_attic": "attic air",
            "t_ceiling_top": "ceiling top surface", "t_ceiling_under": "ceiling underside",
            "t_sky": "effective sky temperature"}
    out = {k: q(r[k], "°C", lab) for k, lab in keys.items() if k in r}
    out["q_room"] = q(r["q_room"], "W/m^2", "heat flux into the room (negative: out of the room)")
    out["q_sky"] = q(r["q_sky"], "W/m^2", "net long-wave loss to the sky")
    out["absorbed_solar"] = q(r["absorbed"], "W/m^2", "absorbed solar flux")
    return out


def _balance_check(r):
    rel = abs(r["balance_residual"]) / r["balance_scale"]
    return self_check("roof energy balance closes", rel < 1e-6, rel, 1e-6, "steady network: sun = sky + wind + "
                      "ventilation + room", "")


_ROOF_ASSUMPTIONS = [
    tr("regime permanente, uma dimensão, por m² de telhado", "steady state, one-dimensional, per m^2 of roof"),
    tr("céu limpo (temperatura de céu de Swinbank)", "clear sky (Swinbank sky temperature)"),
    tr("ático ventilado com a vazão dada; placas paralelas cinzas entre telha e forro",
       "attic ventilated at the given rate; tile and ceiling as parallel grey plates"),
]


def _condensation(p, limits):
    r = roof_network(p, "night")
    td = dew_point(p["t_room_night"], p["rh_room_night"])
    ts = r["t_room_surface"]
    ceiling = "ceiling" in p.get("_elements", ())
    surf_pt, surf_en = ("a face inferior do forro", "the ceiling underside") if ceiling else \
        ("a face inferior da telha", "the tile underside")
    if ts < td:
        mech = tr(f"À noite de céu limpo a telha irradia para o céu a {fmt(r['t_sky'])} °C e esfria até "
                  f"{fmt(r['t_tile_top'])} °C; {surf_pt}, em contato com o ar do ambiente, fica a {fmt(ts)} °C, "
                  f"abaixo do ponto de orvalho do ar interno ({fmt(td)} °C a {fmt(p['t_room_night'])} °C e "
                  f"{fmt(100 * p['rh_room_night'])} %): o vapor condensa e pinga no ambiente.",
                  f"On a clear night the tile radiates to the sky at {fmt(r['t_sky'])} °C and cools to "
                  f"{fmt(r['t_tile_top'])} °C; {surf_en}, in contact with the room air, sits at {fmt(ts)} °C, below "
                  f"the dew point of the room air ({fmt(td)} °C at {fmt(p['t_room_night'])} °C and "
                  f"{fmt(100 * p['rh_room_night'])} %): vapour condenses and drips into the room.")
    else:
        mech = tr(f"{surf_pt.capitalize()} fica a {fmt(ts)} °C, acima do ponto de orvalho do ar interno "
                  f"({fmt(td)} °C): não condensa" + (f" (a telha, a {fmt(r['t_tile_under'])} °C, agora fica do "
                                                      "outro lado do forro, no ático ventilado)" if ceiling else ""),
                  f"{surf_en.capitalize()} sits at {fmt(ts)} °C, above the dew point of the room air ({fmt(td)} °C): "
                  "no condensation" + (f" (the tile, at {fmt(r['t_tile_under'])} °C, is now on the other side of the "
                                       "ceiling, in the ventilated attic)" if ceiling else ""))
    qs = _roof_quantities(r)
    qs["t_dew_room"] = q(td, "°C", "dew point of the room air")
    if ceiling:
        qs["t_dew_attic"] = q(dew_point(p["t_out_night"], p["rh_out_night"]), "°C",
                              "dew point of the attic (outdoor) air")
    return verdict(
        demand=td, capacity=ts, unit="°C", scale=max(p["t_room_night"] - td, 1.0),
        demand_label=tr("ponto de orvalho do ar interno", "dew point of the room air"),
        capacity_label=tr("temperatura da superfície voltada ao ambiente", "temperature of the surface facing the room"),
        mechanism=mech, method="steady roof heat-balance network (clear night) + Magnus dew point",
        quantities=qs, assumptions=_ROOF_ASSUMPTIONS + ([tr(
            "condensação no ático (telha abaixo do orvalho do ar externo) não avaliada: o requisito é o ambiente",
            "condensation inside the attic (tile below the outdoor dew point) not assessed: the requirement is the "
            "room")] if ceiling else []),
        sources=[SRC["swinbank"], SRC["watmuff"], SRC["iso6946"], SRC["magnus"], GREY_PLATES],
        validity=[validity("room temperature within Magnus range", p["t_room_night"], -40, 50)],
        self_checks=[_balance_check(r)])


register_mode(FailureMode(
    "underside_condensation", "hygrothermal",
    tr("condensação sob a cobertura", "condensation under the roof"),
    tr("superfície voltada ao ambiente abaixo do ponto de orvalho do ar interno numa noite de céu limpo",
       "surface facing the room below the dew point of the room air on a clear night"),
    _condensation,
    needs=(Need("tile_thickness", "m", "tile thickness"), Need("tile_k", "W/(m K)", "tile conductivity"),
           Need("t_out_night", "°C", "outdoor air, night"), Need("wind_night", "m/s", "wind, night"),
           Need("t_room_night", "°C", "room air, night"), Need("rh_room_night", "1", "room relative humidity")),
    sources=(SRC["swinbank"], SRC["iso6946"], SRC["magnus"]),
    countermeasures=("ceiling_ventilated_attic", "ceiling_insulation"),
    uncertain={"rh_room_night": 0.1, "wind_night": 0.5, "t_out_night": 0.1},
))


def _radiant(p, limits):
    r = roof_network(p, "day")
    lim = float(limits.get("radiant_asymmetry_max", p.get("radiant_asymmetry_max", 5.0)))
    dt = r["t_room_surface"] - r["t_room"]
    ceiling = "ceiling" in p.get("_elements", ())
    what_pt = "o forro" if ceiling else "a telha"
    what_en = "the ceiling" if ceiling else "the tile"
    if dt > lim:
        mech = tr(f"O sol de {fmt(p['solar_irradiance'])} W/m² aquece a telha até {fmt(r['t_tile_top'])} °C; "
                  f"{what_pt} sobre as pessoas fica a {fmt(r['t_room_surface'])} °C, {fmt(dt)} K acima do ar do "
                  f"ambiente ({fmt(r['t_room'])} °C), mais que os {fmt(lim)} K de assimetria radiante aceitáveis para "
                  f"teto quente: irradia {fmt(r['q_room'])} W/m² para dentro e incomoda.",
                  f"The {fmt(p['solar_irradiance'])} W/m^2 sun heats the tile to {fmt(r['t_tile_top'])} °C; "
                  f"{what_en} above the occupants sits at {fmt(r['t_room_surface'])} °C, {fmt(dt)} K above the room "
                  f"air ({fmt(r['t_room'])} °C), more than the {fmt(lim)} K of radiant asymmetry acceptable for a warm "
                  f"ceiling: it pours {fmt(r['q_room'])} W/m^2 into the room.")
    else:
        mech = tr(f"{what_pt.capitalize()} fica a {fmt(r['t_room_surface'])} °C, {fmt(dt)} K acima do ar "
                  f"(limite {fmt(lim)} K); entram {fmt(r['q_room'])} W/m².",
                  f"{what_en.capitalize()} sits at {fmt(r['t_room_surface'])} °C, {fmt(dt)} K above the air (limit "
                  f"{fmt(lim)} K); {fmt(r['q_room'])} W/m^2 enter.")
    return verdict(
        demand=dt, capacity=lim, unit="K", scale=lim,
        demand_label=tr("assimetria radiante do teto", "ceiling radiant asymmetry"),
        capacity_label=tr("limite de conforto (teto quente)", "comfort limit (warm ceiling)"),
        mechanism=mech, method="steady roof heat-balance network (clear day, sun on the roof)",
        quantities=_roof_quantities(r),
        assumptions=_ROOF_ASSUMPTIONS + [tr(
            "assimetria = temperatura da superfície do teto - ar (teto grande sobre o ocupante, piso à temperatura do "
            "ar)", "asymmetry = ceiling surface - air temperature (large ceiling over the occupant, floor at air "
            "temperature)")],
        sources=[ASHRAE55, SRC["swinbank"], SRC["watmuff"], SRC["iso6946"], GREY_PLATES],
        validity=[validity("solar irradiance on the plane", p["solar_irradiance"], 0, 1361)],
        self_checks=[_balance_check(r)])


register_mode(FailureMode(
    "radiant_heat_gain", "thermal comfort",
    tr("ganho térmico radiante do telhado", "radiant heat gain from the roof"),
    tr("teto quente sobre os ocupantes: assimetria radiante acima do limite de conforto",
       "warm ceiling over the occupants: radiant asymmetry above the comfort limit"),
    _radiant,
    needs=(Need("solar_irradiance", "W/m^2", "sun on the roof plane"), Need("t_out_day", "°C", "outdoor air, day"),
           Need("t_room_day", "°C", "room air, day"), Need("wind_day", "m/s", "wind, day")),
    sources=(ASHRAE55, SRC["swinbank"], SRC["iso6946"]),
    countermeasures=("ceiling_ventilated_attic", "radiant_barrier", "ceiling_insulation"),
    uncertain={"solar_irradiance": 0.1, "tile_absorptance": 0.15, "wind_day": 0.5},
))


def _ceiling(p, v):
    return {"ceiling_R": float(p.get("ceiling_R", 0.05)), "attic_vent": float(p.get("attic_vent", 0.003))}, \
        ("ceiling",), ()


register_countermeasure(Countermeasure(
    "ceiling_ventilated_attic", "ceiling",
    tr("forro com ático ventilado", "ceiling with a ventilated attic"),
    tr("o ar do ambiente deixa de tocar a telha fria (ou quente): entre os dois fica um ático ventilado e o forro, "
       "que recebe a radiação da telha e a devolve ao ambiente bem mais perto da temperatura do ar",
       "room air no longer touches the cold (or hot) tile: a ventilated attic and the ceiling sit in between, and "
       "the ceiling, fed by the tile's radiation, faces the room much closer to the air temperature"),
    ("underside_condensation", "radiant_heat_gain"), _ceiling,
    lambda b, a: (1.0, "m2/m2", "ceiling area per m2 of plan"), unit_cost=1.5,
    unit_cost_note="relative index per m2 of ceiling (premise; roof covering = 1)"))


def _barrier(p, v):
    return {"tile_under_emissivity": float(p.get("barrier_emissivity", 0.05))}, (), ()


register_countermeasure(Countermeasure(
    "radiant_barrier", "radiant_barrier",
    tr("barreira radiante (manta de baixa emissividade sob a telha)",
       "radiant barrier (low-emissivity foil under the tiles)"),
    tr("a face inferior da telha passa a ter emissividade ~0,05: a troca radiante telha-forro, a parcela dominante, "
       "cai uma ordem de grandeza",
       "the tile underside gets an emissivity of ~0.05: tile-to-ceiling radiation, the dominant path, drops by an "
       "order of magnitude"),
    ("radiant_heat_gain",), _barrier, lambda b, a: (1.0, "m2/m2", "foil per m2 of roof"), unit_cost=0.4,
    unit_cost_note="relative index per m2 of foil (premise)", requires=("ceiling",)))


def _insulation(p, v):
    lam = float(p.get("insulation_k", 0.04))
    r0 = float(p.get("ceiling_R", 0.05))
    ch, t = size_to_pass(v.mode, p, [0.025 * k for k in range(1, 9)],
                         lambda t: {"ceiling_R": r0 + t / lam, "insulation_thickness": t})
    return ch, (), ()


register_countermeasure(Countermeasure(
    "ceiling_insulation", "insulation",
    tr("isolante (lã mineral) sobre o forro", "insulation (mineral wool) on the ceiling"),
    tr("a resistência térmica do forro aumenta: a face inferior do forro acompanha o ar do ambiente",
       "the ceiling's thermal resistance rises: its underside follows the room air"),
    ("radiant_heat_gain", "underside_condensation"), _insulation,
    lambda b, a: (a.get("insulation_thickness", 0.0) - b.get("insulation_thickness", 0.0), "m3/m2",
                  "insulation volume per m2"),
    unit_cost=8.0, unit_cost_note="relative index per m3 of insulation (premise)", requires=("ceiling",)))


# ------------------------------------------------------------------------------------------------ Joule heating
def _p_loss(p, tj):
    return float(p["current"]) ** 2 * float(p["r_on_25"]) * (1 + float(p["r_on_tc"]) * (tj - 25.0))


def _board_path(p):
    from pinneapple_physics.pcb import Board, Component, Environment, solve
    board = Board(width_mm=p.get("board_w_mm", 100.0), depth_mm=p.get("board_d_mm", 80.0),
                  n_copper=int(p.get("board_layers", 4)))
    env = Environment(t_ambient_c=float(p["t_ambient"]), air_velocity_m_s=0.0,
                      orientation=p.get("board_orientation", "vertical"))

    def tj_of(P):
        comp = Component("Q1", board.width_mm / 2, board.depth_mm / 2, P, package=p.get("package", "TO-263 D2PAK"),
                         tj_max_c=float(p.get("tj_max", 125.0)), theta_jb=p.get("theta_jb"),
                         theta_jc=p.get("theta_jc"), vias=int(p.get("thermal_vias", 0)))
        r = solve(board, [comp], env, n_cells=int(p.get("board_cells", 36)))
        c = r["components"][0]
        return c["t_junction_c"], {"energy_balance": r["energy_balance_rel_error"],
                                   "h_top": r["h"]["top"], "warnings": r["correlation_warnings"]}
    return tj_of, "board: 3-D finite-volume PCB model (pinneapple_physics.pcb.solve), JEDEC two-resistor package"


def _heatsink_geometry(p):
    from pinneapple_physics.closed_form.plate_fin_heatsink import HeatSinkGeometry
    return HeatSinkGeometry(base_width=p["hs_base_w"], base_depth=p["hs_base_d"], base_thickness=p["hs_base_t"],
                            n_fins=int(p["hs_n_fins"]), fin_thickness=p["hs_fin_t"], fin_height=p["hs_fin_h"],
                            material=p.get("hs_material", "al6063"))


def _heatsink_path(p):
    from pinneapple_physics.closed_form.plate_fin_heatsink import OperatingPoint, evaluate
    g = _heatsink_geometry(p)
    v = float(p.get("fan_velocity", 0.0)) if "fan" in p.get("_elements", ()) else 0.0

    def tj_of(P):
        op = OperatingPoint(power_w=P, t_ambient_c=float(p["t_ambient"]), air_velocity_m_s=v,
                            source_width=p.get("tab_w", 0.010), source_depth=p.get("tab_d", 0.015),
                            tim_resistance_k_m2_w=p.get("tim_r", 1e-5))
        r = evaluate(g, op)
        return r["t_source_max_c"] + P * float(p["theta_jc_tab"]), {
            "r_heatsink": r["r_total_k_w"], "fan_power": r["fan_power_w"], "pressure_drop": r["pressure_drop_pa"],
            "mass": r["mass_kg"], "h": r["h_w_m2k"], "warnings": r["warnings"], "mode": r["mode"]}
    label = ("plate-fin heat sink, " + ("forced convection (Teertstra et al. 2000)" if v > 0 else
                                         "natural convection (Bar-Cohen & Rohsenow 1984)") +
             " (pinneapple_physics.closed_form.plate_fin_heatsink.evaluate)")
    return tj_of, label


def electrothermal(p: dict) -> dict:
    """Junction temperature with P = I^2 R_on(T_j) on the concept's thermal path (fixed point)."""
    tj_of, label = _heatsink_path(p) if "heatsink" in p.get("_elements", ()) else _board_path(p)
    ta = float(p["t_ambient"])
    tj, info, hist = ta + 20.0, {}, []
    runaway = False
    for _ in range(40):
        P = _p_loss(p, tj)
        new, info = tj_of(P)
        hist.append((P, new))
        if new > 2000 or not math.isfinite(new):
            runaway, tj = True, new
            break
        if abs(new - tj) < 1e-3:
            tj = new
            break
        tj = new
    P = _p_loss(p, tj)
    rth = (tj - ta) / P if P > 0 else math.nan
    gain = float(p["r_on_tc"]) * float(p["current"]) ** 2 * float(p["r_on_25"]) * rth
    if gain >= 1:
        runaway = True
    return {"tj": tj, "P": P, "rth": rth, "gain": gain, "runaway": runaway, "info": info, "path": label,
            "iterations": len(hist), "fixed_point_residual": abs(hist[-1][1] - tj) if hist else math.nan,
            "P25": _p_loss(p, 25.0)}


def _joule(p, limits):
    r = electrothermal(p)
    lim = float(limits.get("tj_max", p.get("tj_max", 125.0)))
    tj = math.inf if r["runaway"] else r["tj"]
    path_pt = ("dissipador com ventilação forçada" if "fan" in p.get("_elements", ()) else
               "dissipador em convecção natural" if "heatsink" in p.get("_elements", ()) else "só a placa")
    path_en = ("heat sink with forced air" if "fan" in p.get("_elements", ()) else
               "heat sink in natural convection" if "heatsink" in p.get("_elements", ()) else "the board alone")
    if r["runaway"]:
        mech = tr(f"Com {fmt(p['current'])} A o MOSFET dissipa I²·R_on; R_on cresce com a temperatura e o caminho "
                  f"térmico ({path_pt}, {fmt(r['rth'])} K/W) devolve mais aquecimento do que remove: ganho de "
                  f"realimentação eletrotérmica {fmt(r['gain'])} ≥ 1, fuga térmica.",
                  f"At {fmt(p['current'])} A the MOSFET dissipates I^2 R_on; R_on rises with temperature and the "
                  f"thermal path ({path_en}, {fmt(r['rth'])} K/W) feeds back more heating than it removes: "
                  f"electrothermal loop gain {fmt(r['gain'])} >= 1, thermal runaway.")
    elif tj > lim:
        mech = tr(f"Com {fmt(p['current'])} A o MOSFET dissipa {fmt(r['P'])} W (eram {fmt(r['P25'])} W a 25 °C: R_on "
                  f"sobe com a temperatura, ganho de realimentação {fmt(r['gain'])}); pelo caminho térmico ({path_pt}, "
                  f"{fmt(r['rth'])} K/W da junção ao ar) a junção chega a {fmt(tj)} °C, acima do limite de "
                  f"{fmt(lim)} °C" + (" (muito além do que o silício suporta: o número mede o tamanho do problema)"
                                      if tj > 200 else "") + ".",
                  f"At {fmt(p['current'])} A the MOSFET dissipates {fmt(r['P'])} W ({fmt(r['P25'])} W at 25 °C: "
                  f"R_on rises with temperature, loop gain {fmt(r['gain'])}); through the thermal path ({path_en}, "
                  f"{fmt(r['rth'])} K/W junction to air) the junction reaches {fmt(tj)} °C, above the {fmt(lim)} °C "
                  "limit" + (" (far beyond what silicon survives: the number measures the size of the problem)"
                             if tj > 200 else "") + ".")
    else:
        mech = tr(f"Pelo caminho térmico ({path_pt}, {fmt(r['rth'])} K/W) a junção fica em {fmt(tj)} °C com "
                  f"{fmt(r['P'])} W dissipados (limite {fmt(lim)} °C).",
                  f"Through the thermal path ({path_en}, {fmt(r['rth'])} K/W) the junction sits at {fmt(tj)} °C with "
                  f"{fmt(r['P'])} W dissipated (limit {fmt(lim)} °C).")
    info = r["info"]
    qs = {"t_junction": q(r["tj"], "°C", "junction temperature (electrothermal fixed point)"),
          "power": q(r["P"], "W", "conduction loss at T_j"), "power_25": q(r["P25"], "W", "conduction loss at 25 °C"),
          "r_th": q(r["rth"], "K/W", "junction-to-ambient resistance"),
          "loop_gain": q(r["gain"], "1", "electrothermal loop gain a I^2 R_25 R_th")}
    checks = [self_check("electrothermal fixed point converged", r["fixed_point_residual"] < 1e-2 or r["runaway"],
                         r["fixed_point_residual"], 1e-2, "T_j iteration")]
    if "energy_balance" in info:
        checks.append(self_check("board model energy balance", info["energy_balance"] < 1e-6, info["energy_balance"],
                                 1e-6, "pinneapple_physics.pcb.solve"))
    if "r_heatsink" in info:
        qs["r_heatsink"] = q(info["r_heatsink"], "K/W", "heat-sink resistance (TIM + spreading + fins)")
        qs["heatsink_mass"] = q(info["mass"], "kg", "heat-sink mass")
        qs["h_fins"] = q(info["h"], "W/(m^2 K)", "fin heat-transfer coefficient")
        if info["fan_power"]:
            qs["fan_air_power"] = q(info["fan_power"], "W", "air power through the fins (dp x flow)")
            qs["pressure_drop"] = q(info["pressure_drop"], "Pa", "fin-channel pressure drop the fan must supply")
    warns = info.get("warnings", [])
    val = [validity("junction below the silicon model limit", r["tj"], None, 200, "R_on(T) linear fit")]
    val += [validity(f"correlation note: {w[:80]}", 0.0, None, -1.0, w) for w in warns
            if "outside" in w or "turbulent" in w]
    return verdict(
        demand=tj, capacity=lim, unit="°C", scale=max(lim - float(p["t_ambient"]), 1.0),
        demand_label=tr("temperatura de junção", "junction temperature"),
        capacity_label=tr("limite de junção", "junction limit"),
        mechanism=mech, method="Joule loss I^2 R_on(T) coupled to the thermal path: " + r["path"],
        quantities=qs,
        assumptions=[tr("só perdas de condução (sem chaveamento)", "conduction losses only (no switching)"),
                     tr("R_on linear na temperatura", "R_on linear in temperature"),
                     tr("no dissipador, o caminho pela placa é desprezado (conservador)",
                        "on the heat sink, the path through the board is neglected (conservative)")],
        sources=["Incropera et al., ch. 3 and 9", "pinneapple_physics.pcb (3-D finite-volume board model)",
                 "pinneapple_physics.closed_form.plate_fin_heatsink (Teertstra et al. 2000; Bar-Cohen & Rohsenow "
                 "1984; Lee et al. 1995)"],
        validity=val, self_checks=checks)


register_mode(FailureMode(
    "joule_heating", "electrothermal",
    tr("aquecimento Joule da junção", "Joule heating of the junction"),
    tr("perda I²R_on(T) contra o caminho térmico até o ar: temperatura de junção acima do limite (ou fuga térmica)",
       "I^2 R_on(T) loss against the thermal path to the air: junction above its limit (or thermal runaway)"),
    _joule,
    needs=(Need("current", "A", "RMS current"), Need("r_on_25", "ohm", "on-resistance at 25 °C"),
           Need("r_on_tc", "1/K", "temperature coefficient of R_on"), Need("t_ambient", "°C", "air around the board")),
    sources=("pinneapple_physics.pcb", "pinneapple_physics.closed_form.plate_fin_heatsink"),
    countermeasures=("thermal_vias", "heatsink_natural", "forced_air"),
    uncertain={"r_on_tc": 0.2, "t_ambient": 0.1},
))


def _vias(p, v):
    return {"thermal_vias": int(p.get("max_vias", 40))}, (), ()


register_countermeasure(Countermeasure(
    "thermal_vias", "thermal_vias",
    tr("vias térmicas sob o componente", "thermal vias under the part"),
    tr("cobre em paralelo através do laminado leva o calor às camadas internas e à face inferior da placa",
       "copper in parallel through the laminate carries the heat to the inner planes and the bottom face"),
    ("joule_heating",), _vias, lambda b, a: (a["thermal_vias"] - b.get("thermal_vias", 0), "vias", "vias drilled"),
    unit_cost=0.005, unit_cost_note="relative index per via (premise)", excludes=("heatsink",)))


def _best_heatsink(p, v):
    from pinneapple_physics.closed_form.plate_fin_heatsink import OperatingPoint, evaluate
    W, D, H = p["hs_max_w"], p["hs_max_d"], p["hs_max_h"]
    base_t, fin_t = p.get("hs_base_t", 0.004), p.get("hs_fin_t", 0.0015)
    best = None
    for n in range(4, 31):
        g = {"hs_base_w": W, "hs_base_d": D, "hs_base_t": base_t, "hs_n_fins": n, "hs_fin_t": fin_t,
             "hs_fin_h": H - base_t}
        gap = (W - n * fin_t) / (n - 1)
        if gap < 1.5e-3:
            break
        r = evaluate(_heatsink_geometry({**p, **g}), OperatingPoint(
            power_w=_p_loss(p, 100.0), t_ambient_c=p["t_ambient"], air_velocity_m_s=0.0,
            source_width=p.get("tab_w", 0.010), source_depth=p.get("tab_d", 0.015)))
        if best is None or r["r_total_k_w"] < best[0]:
            best = (r["r_total_k_w"], g)
    if best is None:
        raise Infeasible("o envelope não comporta aletas", "the envelope does not fit any fins")
    return best[1], (), ()


def _hs_cost(b, a):
    return _heatsink_geometry(a).mass_kg(), "kg", "heat-sink mass"


register_countermeasure(Countermeasure(
    "heatsink_natural", "heatsink",
    tr("dissipador aletado (melhor número de aletas no envelope disponível), convecção natural",
       "finned heat sink (best fin count in the available envelope), natural convection"),
    tr("o calor sai pela aba do componente para uma área de aletas muito maior que a placa, e o ar sobe entre as "
       "aletas por empuxo",
       "the heat leaves through the device tab into a fin area far larger than the board, with air rising "
       "between the fins by buoyancy"),
    ("joule_heating",), _best_heatsink, _hs_cost, unit_cost=10.0,
    unit_cost_note="relative index per kg of extruded aluminium (premise)"))


def _fan(p, v):
    ch, _ = size_to_pass("joule_heating", p, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                         lambda u: {"fan_velocity": u}, add=("fan",))
    return ch, (), ()


def _fan_cost(b, a):
    return 1.0, "fan", f"one fan for {a['fan_velocity']:g} m/s through the fins"


register_countermeasure(Countermeasure(
    "forced_air", "fan",
    tr("ventilação forçada nas aletas", "forced air through the fins"),
    tr("o coeficiente de convecção nas aletas sobe várias vezes com a velocidade do ar (canais em desenvolvimento)",
       "the fin heat-transfer coefficient rises several-fold with the air speed (developing channel flow)"),
    ("joule_heating",), _fan, _fan_cost, unit_cost=0.5, unit_cost_note="relative index per fan (premise)",
    requires=("heatsink",)))


__all__ = ["roof_network", "electrothermal"]
