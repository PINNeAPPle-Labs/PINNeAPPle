"""Flight-dynamics nodes: the studies of ``examples/flight_dynamics`` (verification, trim and stability, take-off with
crosswind and gust cases, turbulence encounters, the Pratt check, the same 1-cosine gust for every aircraft) as
nodes on :mod:`pinneapple_simulation.flight`. Each study node takes the aircraft of a ``flight.aircraft`` node and
loops over them; the flights it keeps travel as a ``flight_run`` artifact to ``flight.scene`` (three.js viewer) and
the ``render.*`` nodes (Blender Cycles clips)."""
from __future__ import annotations

import importlib.util
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..types import register_port_type
from .examples import load_object, save_object

register_port_type("aircraft", "aircraft of pinneapple_simulation.flight.catalog (names), for the flight study nodes")
register_port_type("flight_run", "simulated flights: aircraft, case, flight history and wind field (object artifact)")

AIRCRAFT = ["light_ga", "regional_turboprop", "a320", "motor_glider"]
# take-off technique per aircraft class (pinneapple_simulation.flight.takeoff.default_schedule), as in
# examples/flight_dynamics/03_takeoff.py
TAKEOFF_KIND = {"light_ga": "light", "motor_glider": "light", "regional_turboprop": "transport", "a320": "transport"}


def _available():
    if importlib.util.find_spec("pinneapple_simulation.flight") is None:
        return False, "needs pinneapple_simulation.flight (the flight-dynamics library)"
    return True, ""


def _names(v) -> list[str]:
    if not isinstance(v, dict) or "aircraft" not in v:
        raise StepFailed("expected the aircraft of a flight.aircraft node")
    return list(v["aircraft"])


def _num(d: dict, *keys: str) -> dict[str, float]:
    return {k: float(d[k]) for k in keys if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)}


def _runs(ctx, runs: list[dict[str, Any]]) -> dict[str, Any]:
    a = save_object(ctx, "runs", runs)
    a["cases"] = [f"{r['aircraft']}_{r['case']}" for r in runs]
    return a


@node("flight.aircraft", "Aircraft (flight catalog)", category="Flight dynamics",
      outputs=[Port("aircraft", "aircraft"), Port("metrics", "metrics")],
      params=[Param("aircraft", "list", list(AIRCRAFT), help=f"names in pinneapple_simulation.flight.catalog: "
                                                              f"{', '.join(AIRCRAFT)}")],
      calls="pinneapple_simulation.flight.catalog.build", available=_available)
def aircraft(ctx, inputs, params):
    """The aircraft models (lattice wing and tail, fin, fuselage, gear, engines, mass properties with sources)."""
    from pinneapple_simulation.flight import catalog
    bad = [n for n in params["aircraft"] if n not in catalog.names()]
    if bad or not params["aircraft"]:
        raise StepFailed(f"unknown aircraft {bad}; known: {catalog.names()}")
    rows = []
    for n in params["aircraft"]:
        ac = catalog.build(n)
        rows.append([n, ac.info["type"], ac.mass, ac.af.wing_area, ac.af.span, ac.af.mac, ac.wing_loading])
        ctx.metric(f"{n}.mass_kg", ac.mass)
        ctx.metric(f"{n}.wing_loading_Pa", ac.wing_loading)
    ctx.table("aircraft", ["name", "type", "mass [kg]", "S [m^2]", "span [m]", "MAC [m]", "W/S [Pa]"], rows)
    ctx.summary(", ".join(params["aircraft"]))
    return {"aircraft": {"aircraft": list(params["aircraft"])}, "metrics": ctx.report["metrics"]}


@node("flight.verification", "Flight library verification", category="Flight dynamics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("parts", "list", ["turbulence", "trim", "conservation", "lattice"],
                    help="turbulence (MIL-F-8785C variances and PSDs), trim (10 s stick-fixed hold), conservation "
                         "(energy / angular momentum), lattice (in the loop vs steady)"),
              Param("seed", "int", 3, minimum=0, help="turbulence realisation")],
      calls="pinneapple_simulation.flight.verification.turbulence_checks / trim_checks / conservation_check / "
            "lattice_check; pinneapple_simulation.flight.figures.psd_figure", available=_available)
def verification(ctx, inputs, params):
    """The checks of examples/flight_dynamics/01_verification.py, metric names as in its manifest."""
    from pinneapple_simulation.flight import verification as V
    from pinneapple_simulation.flight.figures import psd_figure
    parts = set(params["parts"])
    fig = None
    if "turbulence" in parts:
        t = V.turbulence_checks(seed=int(params["seed"]))
        curves = t.pop("curves")
        for c in t["cases"]:
            ctx.metric(f"turbulence_sigma_{c['generator']}_{c['model']}_{c['component']}", c["sigma"],
                       f"generated sigma [m/s]; MIL-F-8785C target {c['sigma_target']:.4f}")
        ctx.table("turbulence: sigma and band PSD ratio (generated / MIL-F-8785C)",
                  ["generator", "model", "component", "sigma target", "sigma", "PSD ratio min", "PSD ratio max"],
                  [[c["generator"], c["model"], c["component"], c["sigma_target"], c["sigma"], c["psd_ratio_min"],
                    c["psd_ratio_max"]] for c in t["cases"]])
        p = psd_figure(curves[:3] + curves[-2:], str(ctx.path("turbulence_psd.png")))
        ctx.report["figures"].append({"path": p, "name": "turbulence_psd", "caption": "PSD against MIL-F-8785C"})
        from ..runtime import artifact
        fig = artifact(p, "png")
    if "trim" in parts:
        rows = V.trim_checks()
        for r in rows:
            ctx.metric(f"trim_residual_{r['aircraft']}", r["residual"], "trim residual [m/s^2]")
            ctx.metric(f"trim_hold_dV_max_{r['aircraft']}", r["dV_max"], "10 s stick fixed, max |dV| [m/s]")
        ctx.table("trim held 10 s stick fixed", ["aircraft", "V", "h", "alpha [deg]", "elevator [deg]", "residual",
                                                 "max |dV|", "max |dh|", "max |dtheta| [deg]"],
                  [[r["aircraft"], r["V"], r["altitude"], r["alpha_deg"], r["elevator_deg"], r["residual"],
                    r["dV_max"], r["dh_max"], r["dtheta_max_deg"]] for r in rows])
    if "conservation" in parts:
        for dt, c in V.conservation_check().items():
            ctx.metric(f"energy_rel_error_dt{dt}", c["energy_rel_error"], "20 s free flight, relative energy change")
            ctx.metric(f"momentum_rel_error_dt{dt}", c["momentum_rel_error"], "relative angular-momentum change")
    if "lattice" in parts:
        for r in V.lattice_check():
            ctx.metric(f"lattice_rel_diff_{r['aircraft']}_alpha{r['alpha_deg']}", r["rel_diff"],
                       "lattice-in-the-loop CL vs steady vortex lattice")
    ctx.summary(f"verification: {', '.join(sorted(parts))}")
    out = {"metrics": ctx.report["metrics"]}
    if fig is not None:
        out["figure"] = fig
    return out


@node("flight.trim_stability", "Trim, stability derivatives and modes", category="Flight dynamics",
      inputs=[Port("aircraft", "aircraft")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("condition", "enum", "cruise", choices=["cruise", "encounter"],
                    help="cruise: verification.CRUISE (V, h) per aircraft; encounter: ENCOUNTER_SPEED at 3000 m")],
      calls="pinneapple_simulation.flight.trim.trim + stability_derivatives + modes + classify_modes",
      available=_available)
def trim_stability(ctx, inputs, params):
    """Trim at the cruise point, non-dimensional derivatives of the full model and the eigenvalues of the 12-state
    linearisation (examples/flight_dynamics/02_trim_and_stability.py)."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.trim import classify_modes, modes, stability_derivatives, trim
    from pinneapple_simulation.flight.verification import CRUISE, ENCOUNTER_SPEED
    rows = []
    for n in _names(inputs["aircraft"]):
        ctx.check_stop()
        ac = catalog.build(n)
        V, h = CRUISE[n] if params["condition"] == "cruise" else (ENCOUNTER_SPEED[n], 3000.0)
        tr = trim(ac, V, h)
        d = stability_derivatives(ac, tr)
        m = classify_modes(modes(ac, tr)["eigenvalues"])
        for k, v in d.items():
            if isinstance(v, (int, float)):
                ctx.metric(f"{n}.{k}", v)
        osc = m["oscillatory"]
        for i, o in enumerate(osc):
            ctx.metric(f"{n}.mode{i}_wn", o["wn"], "rad/s, oscillatory mode of the linearisation")
            ctx.metric(f"{n}.mode{i}_zeta", o["zeta"])
        ctx.metric(f"{n}.trim_alpha_deg", tr.as_dict()["alpha_deg"])
        rows.append([n, V, h, d["CL_alpha"], d["static_margin"], d["Cm_q"], d["Cl_p"], d["Cn_beta"],
                     ", ".join(f"{o['wn']:.2f}/{o['zeta']:.2f}" for o in osc)])
    ctx.table("trim and stability", ["aircraft", "V [m/s]", "h [m]", "CL_alpha", "static margin", "Cm_q", "Cl_p",
                                     "Cn_beta", "oscillatory modes wn / zeta"], rows)
    ctx.summary(f"{len(rows)} aircraft trimmed and linearised")
    return {"metrics": ctx.report["metrics"]}


@node("flight.takeoff_cases", "Take-off: nominal, crosswind, gust at rotation", category="Flight dynamics",
      inputs=[Port("aircraft", "aircraft")],
      outputs=[Port("metrics", "metrics"), Port("runs", "flight_run"), Port("figure", "figure", required=False)],
      params=[Param("cases", "list", ["nominal"], help="nominal, crosswind, gust_at_rotation"),
              Param("crosswind_kt", "float", 15.0, minimum=0.0, maximum=60.0,
                    help="MIL-F-8785C log profile, speed at 20 ft, from the right (090)"),
              Param("gust_amplitude", "float", 4.0, help="1-cosine down-gust [m/s], met where the nominal run "
                                                         "reached V_R"),
              Param("gust_length", "float", 30.0, minimum=1.0, help="gust half-length [m]"),
              Param("record_gamma", "bool", True, help="keep the lattice circulation (wing loading in the scenes)")],
      calls="pinneapple_simulation.flight.takeoff.simulate_takeoff + takeoff_metrics + raymer_variants; "
            "turbulence.ShearWind / DiscreteGust; figures.takeoff_figure / lateral_figure", available=_available)
def takeoff_cases(ctx, inputs, params):
    """examples/flight_dynamics/03_takeoff.py: ground roll, rotation, lift-off, climb over the 35 ft screen,
    Raymer's closed form, a crosswind and a down-gust case."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.figures import lateral_figure, takeoff_figure
    from pinneapple_simulation.flight.takeoff import (
        raymer_variants,
        simulate_takeoff,
        takeoff_metrics,
    )
    from pinneapple_simulation.flight.turbulence import KT, DiscreteGust, ShearWind
    cases = set(params["cases"]) | {"nominal"}
    keep, nominal, cross, rows = [], {}, {}, []
    rg = bool(params["record_gamma"])
    for n in _names(inputs["aircraft"]):
        ctx.check_stop()
        ac = catalog.build(n)
        kind = TAKEOFF_KIND.get(n, "transport")
        h = simulate_takeoff(ac, kind=kind, record_gamma=rg)
        m = takeoff_metrics(ac, h)
        nominal[n] = (h, m)
        rv = raymer_variants(ac, m, kind)
        for k, v in _num(m, "ground_roll_m", "distance_35ft_m", "V_R_kt", "V_LOF_kt", "theta_LOF_deg",
                         "alpha_LOF_deg", "V_35ft_kt", "climb_gradient_pct", "tail_clearance_min_m",
                         "CL_max_reached", "t_35ft").items():
            ctx.metric(f"{n}.{k}", v)
        ctx.metric(f"{n}.raymer_1p1Vs_total_35ft_m", rv["raymer_1p1Vs"]["total_35ft_m"],
                   "Raymer sec. 17.8, V_LOF = 1.1 V_stall")
        ctx.metric(f"{n}.raymer_sim_VLOF_total_35ft_m", rv["raymer_sim_VLOF"]["total_35ft_m"],
                   "Raymer sec. 17.8 with the simulated V_LOF and rotation time")
        rows.append([n, "nominal", m["ground_roll_m"], m["distance_35ft_m"], rv["raymer_1p1Vs"]["total_35ft_m"],
                     m["V_LOF_kt"], m["tail_clearance_min_m"]])
        keep.append({"aircraft": n, "case": "takeoff", "hist": h, "wind": None,
                     "t_mark": m.get("t_35ft"), "title": f"{ac.info['type']}: take-off"})
        if "crosswind" in cases:
            w = ShearWind(w20=float(params["crosswind_kt"]) * KT, from_deg=90.0)
            hc = simulate_takeoff(ac, kind=kind, wind=w, record_gamma=rg)
            mc = takeoff_metrics(ac, hc)
            cross[n] = hc
            for k, v in _num(mc, "max_lateral_deviation_ground_m", "max_bank_deg", "max_rudder_deg",
                             "distance_35ft_m").items():
                ctx.metric(f"{n}.crosswind_{k}", v)
            rows.append([n, f"crosswind {params['crosswind_kt']:g} kt", mc["ground_roll_m"],
                         mc.get("distance_35ft_m"), None, mc["V_LOF_kt"], mc["tail_clearance_min_m"]])
            keep.append({"aircraft": n, "case": "crosswind", "hist": hc, "wind": w,
                         "title": f"{ac.info['type']}: {params['crosswind_kt']:g} kt crosswind take-off"})
        if "gust_at_rotation" in cases:
            xR = float(h.events["x_R"][0])
            g = DiscreteGust(amplitude=float(params["gust_amplitude"]), length=float(params["gust_length"]),
                             start=xR, shape="full", direction=(0.0, 0.0, 1.0))
            hg = simulate_takeoff(ac, kind=kind, wind=g, record_gamma=rg)
            mg = takeoff_metrics(ac, hg)
            for k, v in _num(mg, "distance_35ft_m", "tail_clearance_min_m").items():
                ctx.metric(f"{n}.gust_{k}", v)
            ctx.metric(f"{n}.gust_delta_distance_35ft_m", mg.get("distance_35ft_m", math.nan) - m["distance_35ft_m"])
            rows.append([n, "down-gust at rotation", mg["ground_roll_m"], mg.get("distance_35ft_m"), None,
                         mg["V_LOF_kt"], mg["tail_clearance_min_m"]])
            keep.append({"aircraft": n, "case": "gust", "hist": hg, "wind": g,
                         "title": f"{ac.info['type']}: down-gust at rotation"})
    ctx.table("take-off", ["aircraft", "case", "ground roll [m]", "to 35 ft [m]", "Raymer 1.1 V_s [m]",
                           "V_LOF [kt]", "min tail clearance [m]"], rows)
    p = takeoff_figure({k: v[0] for k, v in nominal.items()}, {k: v[1] for k, v in nominal.items()},
                       str(ctx.path("takeoff.png")))
    ctx.report["figures"].append({"path": p, "name": "takeoff", "caption": "take-off of each aircraft"})
    from ..runtime import artifact
    fig = artifact(p, "png")
    if cross:
        q = lateral_figure(cross, str(ctx.path("takeoff_crosswind.png")), f"{params['crosswind_kt']:g} kt crosswind")
        ctx.report["figures"].append({"path": q, "name": "takeoff_crosswind", "caption": "crosswind take-off"})
    ctx.summary("; ".join(f"{n}: 35 ft at {m['distance_35ft_m']:.0f} m" for n, (_, m) in nominal.items()))
    return {"metrics": ctx.report["metrics"], "runs": _runs(ctx, keep), "figure": fig}


def _encounter_job(name, V, altitude, severity, controller, seed, duration, field, keep):
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.encounter import turbulence_encounter
    from pinneapple_simulation.flight.turbulence import turbulence_spec
    ac = catalog.build(name)
    h, m, tr, d, wind = turbulence_encounter(ac, V, altitude, turbulence_spec(altitude, severity), duration=duration,
                                             controller=controller, seed=seed, field=field, record_gamma=keep)
    m.update(seed=seed, severity=severity)
    return m, ({"hist": h, "wind": wind} if keep else None)


_ENC_KEYS = ("dnz_cg_rms", "dnz_cg_max", "dnz_cg_min", "phi_rms_deg", "phi_peak_deg", "theta_rms_deg",
             "theta_peak_deg", "ny_cg_rms", "alpha_max_deg", "quasi_steady_dnz_rms", "wind_w_rms")


@node("flight.turbulence_encounter", "Turbulence encounter (MIL-F-8785C)", category="Flight dynamics",
      inputs=[Port("aircraft", "aircraft")],
      outputs=[Port("metrics", "metrics"), Port("runs", "flight_run"), Port("figure", "figure", required=False)],
      params=[Param("severity", "enum", "severe", choices=["light", "moderate", "severe"]),
              Param("altitude", "float", 3000.0, minimum=0.0),
              Param("controllers", "list", ["stick_fixed", "autopilot"], help="stick_fixed and / or autopilot"),
              Param("seeds", "list", [0, 1, 2], help="field realisations; metrics are their mean"),
              Param("duration", "float", 120.0, minimum=5.0, maximum=1200.0, help="seconds of flight per run"),
              Param("field", "enum", "3d", choices=["3d", "line"],
                    help="3d: frozen random-mode field (spanwise variation); line: the same gust across the span"),
              Param("keep", "enum", "autopilot_seed0", choices=["none", "autopilot_seed0", "first"],
                    help="which run to keep (with the lattice loading) for the scene and render nodes"),
              Param("workers", "int", 4, minimum=1, maximum=64, help="worker processes")],
      calls="pinneapple_simulation.flight.encounter.turbulence_encounter + turbulence.turbulence_spec; "
            "verification.ENCOUNTER_SPEED; figures.encounter_bars_figure", available=_available)
def turbulence_encounter(ctx, inputs, params):
    """examples/flight_dynamics/04_turbulence_encounters.py: each aircraft at its own speed through the same
    turbulence; dnz, roll and pitch excursions, cabin accelerations, as the mean over the realisations."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.figures import encounter_bars_figure
    from pinneapple_simulation.flight.verification import ENCOUNTER_SPEED
    names = _names(inputs["aircraft"])
    ctls = list(params["controllers"])
    seeds = [int(s) for s in params["seeds"]]
    if not ctls or not seeds or any(c not in ("stick_fixed", "autopilot") for c in ctls):
        raise StepFailed("controllers: stick_fixed and / or autopilot; seeds: at least one")
    jobs = []
    for n in names:
        for c in ctls:
            for s in seeds:
                k = params["keep"] == "first" and not jobs or (params["keep"] == "autopilot_seed0"
                                                                and c == "autopilot" and s == seeds[0])
                jobs.append((n, ENCOUNTER_SPEED[n], float(params["altitude"]), params["severity"], c, s,
                             float(params["duration"]), params["field"], bool(k)))
    t0 = time.perf_counter()
    workers = min(int(params["workers"]), len(jobs))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(_encounter_job, *zip(*jobs, strict=True)))
    else:
        done = [_encounter_job(*j) for j in jobs]
    ctx.metric("seconds", round(time.perf_counter() - t0, 1), f"{len(jobs)} runs on {workers} processes")
    keep, summary = [], []
    for (n, _V, _h, sev, *_), (_m, run) in zip(jobs, done, strict=True):
        if run:
            ac = catalog.build(n)
            keep.append({"aircraft": n, "case": f"turbulence_{sev}" if sev != "severe" else "turbulence", **run,
                         "title": f"{ac.info['type']}: {sev} turbulence, {params['altitude']:.0f} m"})
    for n in names:
        for c in ctls:
            rs = [m for (j, (m, _)) in zip(jobs, done, strict=True) if j[0] == n and j[4] == c]
            s = {"aircraft": n, "controller": c, "n": len(rs), "cabin": {
                st: {q: float(np.mean([r["cabin"][st][q] for r in rs])) for q in rs[0]["cabin"][st]}
                for st in rs[0]["cabin"]}}
            for k in _ENC_KEYS:
                v = np.array([r[k] for r in rs], float)
                s[k] = float(v.mean())
                ctx.metric(f"{n}.{c}.{k}", s[k], f"mean over seeds {seeds}, {params['duration']:g} s")
            ctx.metric(f"{n}.{c}.departed", int(any(r.get("departed") for r in rs)))
            if len(seeds) == 1:
                for k in _ENC_KEYS:
                    ctx.metric(f"{n}.{c}.seed{seeds[0]}.{k}", float(rs[0][k]))
            summary.append(s)
    ctx.table(f"{params['severity']} turbulence at {params['altitude']:.0f} m, {params['duration']:g} s, seeds {seeds}",
              ["aircraft", "controller", "dnz rms", "dnz max", "dnz min", "phi rms [deg]", "theta rms [deg]",
               "quasi-steady dnz rms"],
              [[s["aircraft"], s["controller"], s["dnz_cg_rms"], s["dnz_cg_max"], s["dnz_cg_min"], s["phi_rms_deg"],
                s["theta_rms_deg"], s["quasi_steady_dnz_rms"]] for s in summary])
    fig = None
    table = [s for s in summary if s["controller"] == ctls[0]]
    try:
        p = encounter_bars_figure(table, str(ctx.path("encounter_bars.png")))
        ctx.report["figures"].append({"path": p, "name": "encounter_bars",
                                      "caption": f"dnz rms at the CG and in the cabin, roll ({ctls[0]})"})
        from ..runtime import artifact
        fig = artifact(p, "png")
    except Exception as exc:  # noqa: BLE001 - a figure never fails the study
        ctx.warn(f"no bar figure: {exc}")
    ctx.summary(f"{len(jobs)} encounters, {params['severity']}, {params['duration']:g} s")
    out = {"metrics": ctx.report["metrics"], "runs": _runs(ctx, keep)}
    if fig is not None:
        out["figure"] = fig
    return out


@node("flight.pratt", "1-cosine gust against Pratt-Walker", category="Flight dynamics",
      inputs=[Port("aircraft", "aircraft")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("U_de", "float", 7.62, help="equivalent gust velocity [m/s] (25 ft/s)"),
              Param("altitude", "float", 3000.0, minimum=0.0)],
      calls="pinneapple_simulation.flight.encounter.pratt_check", available=_available)
def pratt(ctx, inputs, params):
    """Simulated peak dn for a 12.5-MAC 1-cosine gust against the Pratt-Walker formula (and unalleviated)."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.encounter import pratt_check
    from pinneapple_simulation.flight.verification import ENCOUNTER_SPEED
    rows = []
    for n in _names(inputs["aircraft"]):
        p = pratt_check(catalog.build(n), ENCOUNTER_SPEED[n], float(params["altitude"]), float(params["U_de"]))
        for k, v in _num(p, "dn_sim", "dn_pratt", "dn_unalleviated", "K_g", "mu", "ratio_sim_to_pratt").items():
            ctx.metric(f"{n}.{k}", v)
        rows.append([n, p["dn_sim"], p["dn_pratt"], p["dn_unalleviated"], p["K_g"], p["ratio_sim_to_pratt"]])
    ctx.table("peak load factor increment", ["aircraft", "simulated", "Pratt", "unalleviated", "K_g",
                                             "simulated / Pratt"], rows)
    return {"metrics": ctx.report["metrics"]}


@node("flight.same_gust", "The same 1-cosine gust for every aircraft", category="Flight dynamics",
      inputs=[Port("aircraft", "aircraft")],
      outputs=[Port("metrics", "metrics"), Port("runs", "flight_run")],
      params=[Param("amplitude", "float", 7.62, help="vertical up-gust [m/s]"),
              Param("half_length", "float", 50.0, minimum=1.0, help="H [m] (gust length 2H)"),
              Param("altitude", "float", 3000.0, minimum=0.0),
              Param("lead", "float", 1.5, minimum=0.0, help="seconds of level flight before the gust"),
              Param("dt", "float", 0.005, minimum=1e-4, maximum=0.05)],
      calls="pinneapple_simulation.flight.simulate.simulate (StickFixed) + turbulence.DiscreteGust + "
            "pratt_gust_load; encounter.gust_sensitivity; trim.trim + stability_derivatives", available=_available)
def same_gust(ctx, inputs, params):
    """examples/flight_dynamics/07_same_gust.py: peak load factor at the CG and in the cabin, pitch and height
    excursions, the quasi-steady and Pratt estimates for this gust."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.control import StickFixed
    from pinneapple_simulation.flight.encounter import gust_sensitivity
    from pinneapple_simulation.flight.simulate import simulate
    from pinneapple_simulation.flight.trim import stability_derivatives, trim
    from pinneapple_simulation.flight.turbulence import DiscreteGust, pratt_gust_load
    from pinneapple_simulation.flight.verification import ENCOUNTER_SPEED
    U, HALF, H_ALT, LEAD = (float(params[k]) for k in ("amplitude", "half_length", "altitude", "lead"))
    keep, rows = [], []
    for n in _names(inputs["aircraft"]):
        ctx.check_stop()
        ac = catalog.build(n)
        V = ENCOUNTER_SPEED[n]
        tr = trim(ac, V, H_ALT)
        d = stability_derivatives(ac, tr)
        g = DiscreteGust(amplitude=U, length=HALF, start=LEAD * V, shape="full")
        T = LEAD + 2 * HALF / V + 6.0
        h = simulate(ac, tr.x, StickFixed(tr.controls), T, float(params["dt"]), wind=g, record_gamma=True,
                     record_every=2)
        rho = ac.atmosphere(tr.x)[0]
        sig = rho / 1.225
        p = pratt_gust_load(ac.wing_loading, d["CL_alpha"], ac.af.mac, U * math.sqrt(sig), V * math.sqrt(sig),
                            rho=rho)
        dn = h.nz - h.nz[0]
        r = {"dnz_cg_max": float(dn.max()), "dnz_cg_min": float(dn.min()),
             "t_peak_s": float(h.t[int(np.argmax(dn))] - LEAD),
             "theta_max_deg": float(np.degrees(np.abs(h.euler[:, 1] - h.euler[0, 1]).max())),
             "dh_max_m": float(np.abs(h.altitude - H_ALT).max()),
             "quasi_steady_dn": gust_sensitivity(ac, V, rho, d["CL_alpha"]) * U, "pratt_dn_for_this_gust": p["dn"],
             "pratt_K_g": p["K_g"], "mass_ratio": p["mu"]}
        for k, v in r.items():
            ctx.metric(f"{n}.{k}", v)
        for st, v in h.cabin_nz.items():
            ctx.metric(f"{n}.cabin_dnz_max_{st}", float((v - v[0]).max()))
        rows.append([n, V, 2 * HALF / V, r["dnz_cg_max"], r["quasi_steady_dn"], p["dn"], r["theta_max_deg"]])
        keep.append({"aircraft": n, "case": "gust_same", "hist": h, "wind": g, "t_mark": LEAD + r["t_peak_s"],
                     "title": f"{ac.info['type']}: 1-cosine gust"})
    ctx.table(f"{U:g} m/s 1-cosine up-gust over {2 * HALF:g} m at {H_ALT:g} m, stick fixed",
              ["aircraft", "V [m/s]", "gust time [s]", "dnz max", "quasi-steady", "Pratt", "theta max [deg]"], rows)
    return {"metrics": ctx.report["metrics"], "runs": _runs(ctx, keep)}


@node("flight.scene", "Flight scene and three.js viewer", category="Flight dynamics",
      inputs=[Port("runs", "flight_run", multiple=True)],
      outputs=[Port("scene", "anim_scene"), Port("figure", "figure", required=False)],
      params=[Param("n_frames", "int", 200, minimum=4, maximum=2000, help="animation frames of the scene"),
              Param("only", "list", [], help="keep only these cases (aircraft_case, e.g. a320_takeoff); empty: all"),
              Param("arrows", "bool", True, help="the gust as moving arrows"),
              Param("wisps", "enum", "auto", choices=["auto", "on", "off"],
                    help="the gust as wisps along the corridor (auto: for flights in turbulence / gusts)"),
              Param("viewer", "bool", True, help="write the interactive three.js viewer of each scene"),
              Param("screenshot_frame", "int", 120, minimum=-1, help="viewer screenshot frame (-1: none; needs "
                                                                     "playwright + Chromium)")],
      calls="pinneapple_simulation.flight.visuals.flight_scene; pinneapple_tools.visualization.studio.web."
            "web_viewer + screenshot", available=_available)
def scene(ctx, inputs, params):
    """Animated 3-D scenes of the kept flights (aircraft, lattice loading, n_z trail, the gust field), as three.js
    viewers (examples/flight_dynamics/08_viewers.py) and as the input of the Blender render nodes."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.visuals import flight_scene
    from pinneapple_tools.visualization.studio.web import screenshot, web_viewer
    runs = [r for v in inputs["runs"] for r in load_object(v)]
    if params["only"]:
        runs = [r for r in runs if f"{r['aircraft']}_{r['case']}" in params["only"]]
    if not runs:
        raise StepFailed("no flights to show (check 'only')")
    scenes, shots, viewers = [], [], []
    for r in runs:
        ctx.check_stop()
        key = f"{r['aircraft']}_{r['case']}"
        flying = r["case"].startswith(("turbulence", "gust_same"))
        wisps = flying if params["wisps"] == "auto" else params["wisps"] == "on"
        ac = catalog.build(r["aircraft"])
        sc = flight_scene(ac, r["hist"], n_frames=int(params["n_frames"]), wind=r["wind"], runway=not flying,
                          arrows=bool(params["arrows"]), wisps=wisps, title=r.get("title"))
        entry = {"key": key, "scene": sc, "t_mark": r.get("t_mark"), "flying": flying, "title": r.get("title")}
        if params["viewer"]:
            d = ctx.path(f"viewer_{key}")
            web_viewer(sc, str(d))
            viewers.append({"key": key, "index": str(d / "index.html")})
            if int(params["screenshot_frame"]) >= 0:
                png = screenshot(str(d), str(ctx.path(f"viewer_{key}.png")),
                                 frame=min(int(params["screenshot_frame"]), sc.animation.n_frames - 1))
                if png:
                    shots.append(png)
                    ctx.report["figures"].append({"path": png, "name": f"viewer_{key}",
                                                  "caption": f"three.js viewer: {r.get('title') or key}"})
        scenes.append(entry)
    if viewers:
        ctx.view("viewers", viewers)
    ctx.metric("scenes", len(scenes))
    ctx.summary(", ".join(e["key"] for e in scenes))
    a = save_object(ctx, "scenes", scenes)
    a["keys"] = [e["key"] for e in scenes]
    out = {"scene": a}
    if shots:
        from ..runtime import artifact
        out["figure"] = artifact(shots[0], "png")
    return out


# ------------------------------------------------------------------------------------------- planner capabilities
_AC_WORDS = {"light_ga": ("cessna", "c172", "light aircraft", "light ga", "general aviation", "light single"),
             "regional_turboprop": ("atr", "turboprop", "regional"),
             "a320": ("a320", "airliner", "airbus", "single-aisle", "transport aircraft", "jet"),
             "motor_glider": ("glider", "planador", "stemme")}
_FLYING = ("aircraft", "airplane", "aeroplane", "plane", "flight", "avião", "aviao", "aeronave", "voo", "take-off",
           "takeoff", "airliner", "a320", "cessna", "glider", "turboprop")


def _named_aircraft(text: str) -> list[str]:
    t = text.lower()
    return [n for n, words in _AC_WORDS.items() if any(w in t for w in words)]


def _keep_named_aircraft(doc, text, det):
    """Only the aircraft the description names (all four when it names none)."""
    names = _named_aircraft(text)
    for n in doc["nodes"]:
        if n["type"] == "flight.aircraft" and names:
            n["params"]["aircraft"] = names
        if n["type"] == "examples.compare" and names:
            n["params"]["comparisons"] = [c for c in n["params"].get("comparisons", [])
                                          if not any(a in c["flow"] for a in AIRCRAFT if a not in names)]
    return doc


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    common = dict(domain="flight dynamics", example="flight_dynamics", adapt=_keep_named_aircraft,
                  inputs=("the aircraft (catalog: light GA, regional turboprop, A320-class, motor glider)",))
    register_capability(Capability(
        id="flight.takeoff", title="Aircraft take-off: ground roll, lift-off, 35 ft screen, crosswind, gust",
        keywords={"take-off": 3, "takeoff": 3, "take off": 3, "runway": 2, "decolagem": 3, "35 ft": 2,
                  "lift-off": 2, "crosswind": 1.5, "raymer": 2, "ground roll": 2}, min_score=3,
        nodes=("flight.aircraft", "flight.takeoff_cases", "flight.scene", "examples.compare"),
        downstream=("flight.scene", "render.blender_clip"), references=("raymer2018aircraft",),
        template="flight_dynamics/03_takeoff", quick=True,
        description="6-DOF take-off with the vortex lattice in the loop against Raymer's closed form "
                    "(examples/flight_dynamics/03_takeoff.py)", **common))
    register_capability(Capability(
        id="flight.turbulence", title="Aircraft in MIL-F-8785C turbulence: load factors, roll and pitch, cabin",
        keywords={"turbulence": 3, "turbulent air": 3, "turbulência": 3, "mil-f-8785": 3, "von karman": 2,
                  "von kármán": 2, "dryden": 2, "rough air": 2, "gust encounter": 2, "load factor": 1},
        requires_all=(_FLYING,), min_score=3,
        nodes=("flight.aircraft", "flight.turbulence_encounter", "flight.pratt", "flight.scene", "examples.compare"),
        downstream=("flight.scene", "render.blender_clip"),
        uncited="MIL-F-8785C is a military specification (no DOI); flight.verification checks the generated "
                "spectra and variances against it",
        template="flight_dynamics/04_turbulence_encounters", quick=True,
        description="the aircraft through the same severe turbulence, stick fixed and with the autopilot "
                    "(examples/flight_dynamics/04_turbulence_encounters.py)", **common))
    register_capability(Capability(
        id="flight.same_gust", title="The four aircraft through the same 1-cosine gust",
        keywords={"same gust": 4, "1-cosine": 3, "one-cosine": 3, "1 - cos": 3, "discrete gust": 3,
                  "side by side": 1, "pratt": 1.5}, requires_all=(_FLYING,), min_score=3,
        nodes=("flight.aircraft", "flight.same_gust", "flight.scene", "render.blender_clip", "render.clip_grid"),
        downstream=("render.clip_grid",), uncited="Pratt-Walker (NACA Report 1206) is quoted by the library "
                                                   "docs but not in the flow reference registry",
        template="flight_dynamics/07_same_gust", quick=True,
        description="peak load factors at the CG and in the cabin for one vertical gust "
                    "(examples/flight_dynamics/07_same_gust.py)", **common))
    register_capability(Capability(
        id="flight.trim_stability", title="Aircraft trim, stability derivatives and dynamic modes",
        keywords={"trim": 2.5, "stability derivative": 3, "static margin": 2.5, "dynamic modes": 2.5, "phugoid": 2,
                  "dutch roll": 2, "short period": 2, "eigenvalue": 1, "linearis": 1, "lineariz": 1},
        requires_all=(_FLYING,), min_score=2.5,
        nodes=("flight.aircraft", "flight.trim_stability", "examples.compare"),
        uncited="the derivatives come from the library's vortex lattice; checked by examples.compare",
        template="flight_dynamics/02_trim_and_stability",
        description="trim at cruise, non-dimensional derivatives, 12-state modes "
                    "(examples/flight_dynamics/02_trim_and_stability.py)", **common))
    register_capability(Capability(
        id="flight.verification", title="Verify the aircraft flight-dynamics library: conservation and turbulence spectra",
        keywords={"verification": 2, "verify": 1.5, "conservation": 1.5, "turbulence spectra": 2, "psd": 1.5},
        requires_all=(_FLYING,), min_score=3.5, nodes=("flight.verification", "examples.compare"),
        uncited="checks against MIL-F-8785C spectra, energy conservation and the steady lattice",
        template="flight_dynamics/01_verification",
        description="turbulence spectra, trim hold, conservation, lattice in the loop "
                    "(examples/flight_dynamics/01_verification.py)", domain="flight dynamics",
        example="flight_dynamics"))
    register_capability(Capability(
        id="flight.render", title="Blender clips and three.js viewers of flights",
        keywords={"blender": 2.5, "cycles": 2, "clip": 1.5, "render": 1.5, "video": 1, "three.js": 1.5, "viewer": 1},
        requires_all=(_FLYING,), min_score=3,
        nodes=("flight.scene", "render.blender_clip", "render.clip_grid"),
        uncited="visualisation (no physics claim)", template="flight_dynamics/05_render_clips", quick=True,
        description="Cycles clips of the take-off, the turbulence encounter and the same-gust grid "
                    "(examples/flight_dynamics/05_render_clips.py, 08_viewers.py)", **common))
    register_capability(Capability(
        id="flight.full_study", title="The whole flight-dynamics study of the four aircraft",
        keywords={"flight dynamics": 3, "flight-dynamics": 3, "whole study": 1.5, "full study": 1.5,
                  "complete study": 1.5, "dinâmica de voo": 3, "four aircraft": 1}, min_score=4,
        nodes=("flight.aircraft", "flight.verification", "flight.trim_stability", "flight.takeoff_cases",
               "flight.turbulence_encounter", "flight.pratt", "flight.same_gust", "flight.scene",
               "render.blender_clip", "render.clip_grid", "examples.compare"),
        references=("raymer2018aircraft",), template="flight_dynamics/00_full_study", quick=True,
        description="verification, trim, take-off, turbulence, Pratt, the same gust, viewers and clips on one "
                    "canvas", **common))


_register_capabilities()
