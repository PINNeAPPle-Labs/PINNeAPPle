"""The standard rover workflow (#1200): planet, terrain and mission -> mobility, power and thermal -> rover design ->
the traverse simulated on the mission profile.

* **Mobility** (Wong-Reece wheels, :class:`~pinneapple_simulation.terramechanics.WongReeceModel`): the grade the
  rover climbs at its slip limit (``n DP(s_max, m g cos a / n) = m g sin a``, bisection), the step a front wheel
  climbs when the other wheels push it (quasi-static, :func:`step_push_force`), static tip-over angles from the track,
  wheelbase and centre-of-mass height, static sinkage.
* **Power**: the electrical drive power at the mission speed on each slope (wheel torque at the slip that gives the
  needed drawbar pull, ``omega = v / (R (1 - s))``, over the drive efficiency), the hotel load, a solar array sized for
  the energy of a sol with a sinusoidal day profile (``E = A eta S tau t_day 2 / pi``, peak ``A eta S tau``) and a
  battery sized for the night.
* **Thermal**: a survival heater for the warm box, ``Q = eps_eff sigma A (T_in^4 - T_sink^4)`` less the internal
  dissipation, drawn from the battery at night.
* **Design**: wheel radius and width at minimum total mass (structure fraction, wheels, drive motors sized by peak
  torque, array, battery, insulation, avionics, payload; a fixed point because the loads scale with the mass), within
  the grade, step, stability, sinkage, power and thermal margins.
* **Re-verification**: :func:`traverse` time-steps the mission (sun, drive window, slope profile, wheel slip and
  power, battery state of charge) and reports the distance reached, the lowest state of charge, the largest slip and
  the energy closure.

**Published rover**: :func:`published_rover_check` runs the power and mobility model with Sojourner's published mass,
wheels, speed and array (``SOJOURNER``, each value with the source it was read from) and compares the peak array power
with the reported 16 W within ``SOJOURNER_BAND``. Every planet, mass-model and thermal default that is not from a
publication is marked "assumption" in ``ASSUMPTIONS``.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["PLANETS", "SOJOURNER", "SOJOURNER_BAND", "ASSUMPTIONS", "step_push_force", "max_step_height",
           "tip_over_angles", "grade_capability", "drive_power", "solar_array", "heater_power", "size_rover",
           "traverse", "published_rover_check", "rover_workflow"]

SIGMA_SB = 5.670374419e-8
#: g [m/s^2], top-of-atmosphere solar irradiance [W/m^2], day length [s], daylight fraction, transmission to the
#: surface at noon, night sink temperature [K]
PLANETS: dict[str, dict[str, Any]] = {
    "mars": {"g": 3.71, "solar_W_m2": 586.2, "day_s": 88775.0, "daylight_fraction": 0.5, "transmission": 0.6,
             "night_sink_K": 180.0, "soil": "JSC-1A",
             "sources": {"g": "secondary: NASA Mars fact sheet (3.71 m/s^2), not re-read here",
                         "solar_W_m2": "secondary: NASA Mars fact sheet, mean solar irradiance 586.2 W/m^2",
                         "day_s": "secondary: mean solar day 24.6597 h (NASA Mars fact sheet)",
                         "transmission": "assumption: global noon transmission 0.6 for a moderately dusty sky",
                         "night_sink_K": "assumption", "daylight_fraction": "assumption: equinox",
                         "soil": "assumption: no Mars soil in the database; JSC-1A as a proxy"}},
    "moon": {"g": 1.62, "solar_W_m2": 1361.0, "day_s": 29.53 * 86400.0, "daylight_fraction": 0.5,
             "transmission": 1.0, "night_sink_K": 100.0, "soil": "GRC-1",
             "sources": {"g": "secondary: NASA Moon fact sheet (1.62 m/s^2)",
                         "solar_W_m2": "secondary: total solar irradiance at 1 AU, about 1361 W/m^2",
                         "day_s": "secondary: synodic month 29.53 d", "transmission": "no atmosphere",
                         "night_sink_K": "assumption", "daylight_fraction": "equator", "soil": "database GRC-1"}},
}

#: Sojourner (Mars Pathfinder) as published; mass and power differ between sources, the one used is named
SOJOURNER = {"mass_kg": 10.5, "wheels": 6, "wheel_diameter_m": 0.13, "wheel_width_m": 0.079, "speed_m_s": 0.01,
             "array_m2": 0.22, "cell_efficiency": 0.18, "peak_power_W": 16.0,
             "sources": {"mass_kg": "JPL robotics, Mars Pathfinder page (10.5 kg; other sources give 10.6 and 11.5 kg)",
                         "wheel_diameter_m": "Wikipedia, Sojourner (rover): 13 cm diameter, 7.9 cm wide (secondary)",
                         "speed_m_s": "NASA Facts, Mars Pathfinder: 1 cm/s top speed (secondary)",
                         "array_m2": "Wikipedia, Sojourner (rover): 0.22 m^2 GaAs/Ge, about 18 % (secondary)",
                         "peak_power_W": "PDS rover description (mpfr-m-rvrcam-2-edr-v1.0 insthost.htm): peak panel "
                                         "production 16 W (other sources give 13 and 15 W)"}}
#: the reported peak power must be reproduced within this relative band (the sources disagree by -19 % .. 0 %)
SOJOURNER_BAND = 0.25

ASSUMPTIONS = {"structure_fraction": 0.25, "motor_N_m_per_kg": 8.0, "array_kg_per_m2": 2.0,
               "battery_Wh_per_kg": 100.0, "battery_dod": 0.6, "battery_efficiency": 0.95, "mli_kg_per_m2": 0.6,
               "drive_efficiency": 0.5, "avionics_kg": 3.0, "hotel_W": 4.0, "eps_eff": 0.03, "warm_box_K": 233.0,
               "wheelbase_per_d": 4.0, "track_per_d": 3.5, "cg_height_per_d": 1.5, "rock_friction": 0.6,
               "step_cap_per_d": {"rocker-bogie": 1.5, "rigid": 0.5}, "energy_margin": 1.2}
#: what each assumption stands for (shown with the result)
ASSUMPTION_NOTES = {
    "wheelbase_per_d": "Sojourner's 65 x 48 cm envelope on 13 cm wheels gives length / D 5.0 and width / D 3.7; the "
                       "wheel-centre wheelbase 4 D and track 3.5 D are taken inside that envelope",
    "step_cap_per_d": "suspension travel caps the climbable step (rocker-bogie 1.5 D, a rigid chassis R): a "
                      "design rule, not a measurement",
    "eps_eff": "effective emittance of the warm-box insulation", "other": "stated engineering values, not measured"}


# ------------------------------------------------------------------ mobility
def step_push_force(h: float, radius: float, load: float, mu: float) -> float:
    """Horizontal push [N] a driven wheel of ``radius`` carrying ``load`` needs to lift onto a step of height ``h``
    (quasi-static, ground reaction gone, friction ``mu`` at the step edge driven by the wheel's own torque).
    Contact at the edge at angle ``a`` from the vertical (``cos a = 1 - h / R``): ``N = W / (cos a + mu sin a)``,
    ``P = N (sin a - mu cos a)``; for ``h >= R`` the wheel climbs the face: ``P = W / mu``. 0 when the wheel climbs
    on its own torque."""
    if h <= 0:
        return 0.0
    if h >= radius:
        return load / mu
    a = math.acos(1.0 - h / radius)
    n = load / (math.cos(a) + mu * math.sin(a))
    return max(0.0, n * (math.sin(a) - mu * math.cos(a)))


def max_step_height(radius: float, load: float, push: float, mu: float, cap: float) -> float:
    """Largest step [m] (up to ``cap``) for which :func:`step_push_force` is within ``push`` [N] (bisection)."""
    if step_push_force(cap, radius, load, mu) <= push:
        return float(cap)
    lo, hi = 0.0, float(cap)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if step_push_force(mid, radius, load, mu) <= push else (lo, mid)
    return lo


def tip_over_angles(track: float, wheelbase: float, cg_height: float) -> dict[str, float]:
    """Static tip-over slope [deg]: lateral ``atan(track / 2 / h)`` and longitudinal ``atan(wheelbase / 2 / h)``."""
    return {"lateral_deg": math.degrees(math.atan(0.5 * track / cg_height)),
            "longitudinal_deg": math.degrees(math.atan(0.5 * wheelbase / cg_height))}


def _model(soil, radius, width, g, grousers=(0, 0.0)):
    from ..terramechanics import WongReeceModel
    from ..terramechanics.workflow import _wheel
    return WongReeceModel(soil, _wheel(radius, width, grousers[0], grousers[1]), g=g, n_gl=16)


def grade_capability(model, mass: float, g: float, n_wheels: int, max_slip: float, step_deg: float = 0.25) -> float:
    """Steepest grade [deg] the rover climbs at slip ``max_slip``: ``n DP = m g sin a`` with wheel load
    ``m g cos a / n`` (one vectorised solve on a ``step_deg`` grid of grades, then linear interpolation of the
    crossing)."""
    ang = np.arange(0.0, 60.0 + step_deg / 2, step_deg)
    a = np.radians(ang)
    w = mass * g * np.cos(a) / n_wheels
    ex = n_wheels * np.asarray(model.steady_state(w, np.full_like(w, max_slip))["drawbar_pull"]) - mass * g * np.sin(a)
    if ex[0] < 0:
        return 0.0
    neg = np.nonzero(ex < 0)[0]
    if not neg.size:
        return float(ang[-1])
    i = int(neg[0])
    return float(ang[i - 1] + (ang[i] - ang[i - 1]) * ex[i - 1] / (ex[i - 1] - ex[i]))


def _pull_curve(model, load: float, max_slip: float, n: int = 61):
    s = np.linspace(0.0, max_slip, n)
    r = model.steady_state(np.full_like(s, load), s)
    return s, np.asarray(r["drawbar_pull"]), np.asarray(r["torque"]), np.asarray(r["sinkage"])


def drive_power(model, mass: float, g: float, n_wheels: int, slope_deg: float, speed: float, efficiency: float,
                max_slip: float = 0.6) -> dict[str, float]:
    """Electrical drive power [W] at ``speed`` up ``slope_deg``: the slip where each wheel's drawbar pull is
    ``m g sin a / n`` (on the flat: zero, the wheel only overcomes its own compaction), the wheel torque there and
    ``omega = v / (R (1 - s))``. Down a slope the flat-ground power is used (no regeneration: conservative). The slip
    is interpolated on a 61-point slip curve (one vectorised solve)."""
    a = math.radians(max(slope_deg, 0.0))
    w = mass * g * math.cos(a) / n_wheels
    pull = mass * g * math.sin(a) / n_wheels
    s, dp, tq, z = _pull_curve(model, w, max_slip)
    if dp[-1] < pull:
        return {"power_W": float("nan"), "slip": float("nan"), "torque_Nm": float("nan"), "sinkage_m": float("nan")}
    i = int(np.argmax(dp >= pull))
    if i == 0:
        si, t, zz = 0.0, float(tq[0]), float(z[0])
    else:
        f = (pull - dp[i - 1]) / (dp[i] - dp[i - 1])
        si, t, zz = (float(v[i - 1] + f * (v[i] - v[i - 1])) for v in (s, tq, z))
    omega = speed / (model.wheel.radius * max(1.0 - si, 1e-6))
    return {"power_W": n_wheels * t * omega / efficiency, "slip": si, "torque_Nm": t, "sinkage_m": zz}


# ------------------------------------------------------------------ power and thermal
def solar_array(area: float, efficiency: float, planet: Mapping[str, Any]) -> dict[str, float]:
    """Noon peak power ``A eta S tau`` [W] and energy per day [W h] of a horizontal array with a sinusoidal day
    (mean over the daylight = peak x 2 / pi)."""
    peak = area * efficiency * planet["solar_W_m2"] * planet["transmission"]
    day = planet["day_s"] * planet["daylight_fraction"]
    return {"peak_W": peak, "energy_Wh": peak * day * 2.0 / math.pi / 3600.0, "daylight_s": day}


def heater_power(area: float, eps_eff: float, t_inside: float, t_sink: float, internal_W: float) -> float:
    """Survival heater [W]: radiated loss ``eps_eff sigma A (T_in^4 - T_sink^4)`` less the internal dissipation."""
    return max(0.0, eps_eff * SIGMA_SB * area * (t_inside ** 4 - t_sink ** 4) - internal_W)


def _geometry(radius, a):
    d = 2 * radius
    wb, tr, h = a["wheelbase_per_d"] * d, a["track_per_d"] * d, a["cg_height_per_d"] * d
    box = 2 * (0.8 * wb * 0.6 * tr + 0.8 * wb * d + 0.6 * tr * d)       # warm box: 0.8 wb x 0.6 track x D
    return {"wheelbase_m": wb, "track_m": tr, "cg_height_m": h, "box_area_m2": box}


def size_rover(*, radius: float, width: float, soil, planet: Mapping[str, Any], mission: Mapping[str, Any],
               n_wheels: int = 6, suspension: str = "rocker-bogie", max_slip: float = 0.3,
               cell_efficiency: float = 0.28, assumptions: Mapping[str, Any] | None = None, iters: int = 8,
               battery_min_wh: float = 0.0) -> dict[str, Any]:
    """Mass build-up, mobility, power and thermal of a rover with wheels ``radius`` x ``width`` for ``mission``
    (``payload_kg``, ``speed_m_s``, ``drive_hours``, ``max_slope_deg``, ``rock_m``), iterated to a fixed point."""
    from ..terramechanics.workflow import wheel_mass
    a = {**ASSUMPTIONS, **dict(assumptions or {})}
    g = planet["g"]
    geo = _geometry(radius, a)
    model = _model(soil, radius, width, g)
    m_wheel = wheel_mass(radius, width)
    mass = 3.0 * (mission["payload_kg"] + a["avionics_kg"])
    for _ in range(iters):
        w_static = mass * g / n_wheels
        flat = drive_power(model, mass, g, n_wheels, 0.0, mission["speed_m_s"], a["drive_efficiency"])
        up = drive_power(model, mass, g, n_wheels, mission["max_slope_deg"], mission["speed_m_s"],
                         a["drive_efficiency"], max_slip=max_slip)
        t_peak = up["torque_Nm"] if math.isfinite(up["torque_Nm"]) else float(
            model.steady_state(w_static, max_slip)["torque"])
        m_motor = n_wheels * t_peak / a["motor_N_m_per_kg"]
        p_drive = flat["power_W"] if math.isfinite(flat["power_W"]) else float("nan")
        heater = heater_power(geo["box_area_m2"], a["eps_eff"], a["warm_box_K"], planet["night_sink_K"],
                              a["hotel_W"])
        night_s = planet["day_s"] * (1 - planet["daylight_fraction"])
        e_night = (a["hotel_W"] + heater) * night_s / 3600.0
        e_day = (a["hotel_W"] * planet["day_s"] * planet["daylight_fraction"] + p_drive * mission["drive_hours"]
                 * 3600.0) / 3600.0
        e_need = e_day + e_night / a["battery_efficiency"]
        unit = solar_array(1.0, cell_efficiency, planet)
        area = a["energy_margin"] * e_need / unit["energy_Wh"]
        peak_need = (a["hotel_W"] + (up["power_W"] if math.isfinite(up["power_W"]) else p_drive))
        area = max(area, peak_need / unit["peak_W"])                       # drive on the slope from the array
        batt_wh = max(e_night / a["battery_dod"], float(battery_min_wh))
        parts = {"payload": mission["payload_kg"], "avionics": a["avionics_kg"], "wheels": n_wheels * m_wheel,
                 "motors": m_motor, "array": area * a["array_kg_per_m2"], "battery": batt_wh / a["battery_Wh_per_kg"],
                 "insulation": geo["box_area_m2"] * a["mli_kg_per_m2"]}
        new = sum(parts.values()) / (1.0 - a["structure_fraction"])
        if abs(new - mass) < 1e-4 * new:
            mass = new
            break
        mass = new
    parts["structure"] = a["structure_fraction"] * mass
    w_static = mass * g / n_wheels
    grade = grade_capability(model, mass, g, n_wheels, max_slip)
    push = (n_wheels - 1) * max(float(model.steady_state(w_static, max_slip)["drawbar_pull"]), 0.0)
    cap = a["step_cap_per_d"].get(suspension, 0.5) * 2 * radius
    step = max_step_height(radius, w_static, push, a["rock_friction"], cap)
    tip = tip_over_angles(geo["track_m"], geo["wheelbase_m"], geo["cg_height_m"])
    sink = float(model.steady_state(w_static, 0.0)["sinkage"])
    arr = solar_array(area, cell_efficiency, planet)
    return {"mass_kg": mass, "parts_kg": parts, "geometry": geo, "wheel": {"radius_m": radius, "width_m": width},
            "grade_deg": grade, "step_m": step, "step_cap_m": cap, "tip_over": tip, "static_sinkage_m": sink,
            "drive_flat": flat, "drive_slope": up, "heater_W": heater, "hotel_W": a["hotel_W"],
            "energy_need_Wh": e_need, "array_m2": area, "array": arr, "battery_Wh": batt_wh,
            "night_energy_Wh": e_night, "cell_efficiency": cell_efficiency, "assumptions": a, "model": model}


# ------------------------------------------------------------------ re-verification
def traverse(sized: Mapping[str, Any], *, planet: Mapping[str, Any], mission: Mapping[str, Any], n_wheels: int = 6,
             max_slip: float = 0.3, dt: float = 60.0, segment_m: float = 5.0, seed: int = 0) -> dict[str, Any]:
    """Time-step the mission: the sun (sinusoidal over the daylight), a drive window of ``drive_hours`` centred on
    noon, a slope profile (``segment_m`` segments, slopes uniform in +-``max_slope_deg``, seeded), the wheel slip and
    drive power on each slope, the hotel and heater loads, and the battery (charge efficiency, capacity,
    ``battery_dod`` floor). Stops when the distance is reached or after ``max_sols``."""
    a = sized["assumptions"]
    rng = np.random.default_rng(seed)
    g, m, model = planet["g"], sized["mass_kg"], sized["model"]
    slopes_grid = np.linspace(0.0, mission["max_slope_deg"], 7)
    tab = [drive_power(model, m, g, n_wheels, s, mission["speed_m_s"], a["drive_efficiency"], max_slip=max_slip)
           for s in slopes_grid]
    p_tab = np.array([t["power_W"] for t in tab])
    s_tab = np.array([t["slip"] for t in tab])
    n_seg = int(math.ceil(mission["distance_m"] / segment_m)) + 1
    profile = rng.uniform(-mission["max_slope_deg"], mission["max_slope_deg"], n_seg)
    cap = sized["battery_Wh"] * 3600.0
    floor = (1.0 - a["battery_dod"]) * cap
    e = cap
    day, light = planet["day_s"], planet["day_s"] * planet["daylight_fraction"]
    peak = sized["array"]["peak_W"]
    heater = sized["heater_W"]
    t, x = 0.0, 0.0
    e_min, slip_max = e, 0.0
    e_in = e_out = e_lost = clipped = 0.0
    drive_s = mission["drive_hours"] * 3600.0
    t_end = mission["max_sols"] * day
    while x < mission["distance_m"] and t < t_end:
        tod = t % day
        sun = peak * math.sin(math.pi * tod / light) if tod < light else 0.0
        load = a["hotel_W"] + (heater if tod >= light else 0.0)
        driving = abs(tod - light / 2) <= drive_s / 2 and tod < light
        if driving:
            sl = profile[min(int(x // segment_m), n_seg - 1)]
            p = float(np.interp(max(sl, 0.0), slopes_grid, p_tab))
            if not math.isfinite(p):
                return {"stuck": True, "reason": f"no slip within the limit gives the pull for a {sl:.1f} deg slope",
                        "distance_m": x}
            slip_max = max(slip_max, float(np.interp(max(sl, 0.0), slopes_grid, s_tab)))
            load += p
            x += mission["speed_m_s"] * dt
        net = (sun - load) * dt
        if net > 0:
            gain = net * a["battery_efficiency"]
            take = min(gain, cap - e)
            clipped += gain - take
            e_lost += net - gain
            e += take
        else:
            e += net
        e_in += sun * dt
        e_out += load * dt
        e_min = min(e_min, e)
        t += dt
    closure = abs((e - cap) - (e_in - e_out - e_lost - clipped)) / max(e_in, 1e-12)
    return {"stuck": False, "distance_m": x, "time_s": t, "sols": t / day, "soc_min": e_min / cap,
            "soc_floor": floor / cap, "slip_max": slip_max, "energy_in_Wh": e_in / 3600.0,
            "energy_out_Wh": e_out / 3600.0, "energy_closure": closure, "reached": x >= mission["distance_m"]}


# ------------------------------------------------------------------ published rover
def published_rover_check(soil=None, planet: str = "mars") -> dict[str, Any]:
    """Sojourner's published mass, wheels, speed and array through the power and mobility model: the peak array power
    against the reported 16 W (band ``SOJOURNER_BAND``) and the drive power at the reported 1 cm/s against it."""
    from ..terramechanics import get_soil
    p = PLANETS[planet]
    sl = soil or get_soil(p["soil"])
    s = SOJOURNER
    arr = solar_array(s["array_m2"], s["cell_efficiency"], p)
    model = _model(sl, s["wheel_diameter_m"] / 2, s["wheel_width_m"], p["g"])
    dp = drive_power(model, s["mass_kg"], p["g"], s["wheels"], 0.0, s["speed_m_s"], ASSUMPTIONS["drive_efficiency"])
    err = arr["peak_W"] / s["peak_power_W"] - 1.0
    return {"peak_power_W": arr["peak_W"], "reported_peak_W": s["peak_power_W"], "rel_error": err,
            "within_band": abs(err) <= SOJOURNER_BAND, "drive_power_W": dp["power_W"], "drive_slip": dp["slip"],
            "drivable_on_array": bool(dp["power_W"] < arr["peak_W"]), "soil": sl.name,
            "sources": s["sources"]}


# ------------------------------------------------------------------ workflow
DEFAULT_MISSION = {"distance_m": 100.0, "payload_kg": 2.0, "speed_m_s": 0.01, "drive_hours": 2.0,
                   "max_slope_deg": 15.0, "rock_m": 0.10, "max_sols": 30, "max_sinkage_m": 0.03}


def rover_workflow(*, planet: str = "mars", soil: str | None = None, mission: Mapping[str, Any] | None = None,
                   n_wheels: int = 6, suspension: str = "rocker-bogie", max_slip: float = 0.3,
                   radius_bounds=(0.05, 0.20), width_bounds=(0.04, 0.15), grid: int = 6,
                   cell_efficiency: float = 0.28, stability_factor: float = 1.5,
                   assumptions: Mapping[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
    """Size the lightest rover (wheel radius and width on a ``grid`` x ``grid`` box, array and battery sized inside)
    that climbs the mission's slope and rock, does not tip on ``stability_factor`` x the slope, stays within the
    sinkage limit and the power and thermal margins; then simulate the traverse and check the published rover."""
    from ..terramechanics import get_soil
    if planet not in PLANETS:
        raise ValueError(f"planet is one of {sorted(PLANETS)}")
    pl = PLANETS[planet]
    sl = get_soil(soil or pl["soil"])
    mi = {**DEFAULT_MISSION, **dict(mission or {})}
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None, status=None):
        st = status or ("pass" if ok is True else "fail" if ok is False else "inconclusive")
        checks.append({"code": code, "status": st, "detail": detail, "value": value})

    def margins(r):
        tip = min(r["tip_over"].values())
        return {"grade": r["grade_deg"] - mi["max_slope_deg"], "step": r["step_m"] - mi["rock_m"],
                "stability": tip - stability_factor * mi["max_slope_deg"],
                "sinkage": mi["max_sinkage_m"] - r["static_sinkage_m"],
                "power": 0.0 if math.isfinite(r["drive_slope"]["power_W"]) else -1.0}

    t0 = time.perf_counter()
    rows, best = [], None
    for R in np.linspace(*radius_bounds, grid):
        for b in np.linspace(*width_bounds, grid):
            r = size_rover(radius=float(R), width=float(b), soil=sl, planet=pl, mission=mi, n_wheels=n_wheels,
                           suspension=suspension, max_slip=max_slip, cell_efficiency=cell_efficiency,
                           assumptions=assumptions)
            mg = margins(r)
            ok = all(v >= 0 for v in mg.values()) and math.isfinite(r["mass_kg"])
            rows.append([float(R), float(b), r["mass_kg"], r["grade_deg"], r["step_m"], ok])
            if ok and (best is None or r["mass_kg"] < best["mass_kg"]):
                best = r
    seconds = time.perf_counter() - t0
    out: dict[str, Any] = {"planet": {k: v for k, v in pl.items()}, "soil": sl.name, "mission": mi,
                           "grid": rows, "search_seconds": seconds}
    n_ok = sum(1 for x in rows if x[5])
    if best is None:
        chk("design.feasible", False, f"none of the {len(rows)} wheel sizes meets the grade, step, stability, sinkage "
            "and power limits: widen the box or relax the mission")
    else:
        chk("design.feasible", True, f"{n_ok} of {len(rows)} wheel sizes meet every limit; the lightest is "
            f"{best['mass_kg']:.2f} kg")
        mg = margins(best)
        chk("mobility.grade", mg["grade"] >= 0, f"climbs {best['grade_deg']:.1f} deg at slip {max_slip:g} (mission "
            f"{mi['max_slope_deg']:g} deg)", best["grade_deg"])
        chk("mobility.step", mg["step"] >= 0, f"front wheel climbs a {100 * best['step_m']:.1f} cm step pushed by the "
            f"other wheels (cap {100 * best['step_cap_m']:.1f} cm; rock {100 * mi['rock_m']:.0f} cm)", best["step_m"])
        chk("mobility.stability", mg["stability"] >= 0, "tip-over at " + ", ".join(
            f"{k} {v:.1f} deg" for k, v in best["tip_over"].items()) + f" (>= {stability_factor:g} x "
            f"{mi['max_slope_deg']:g} deg)", min(best["tip_over"].values()))
        chk("mobility.sinkage", mg["sinkage"] >= 0, f"static sinkage {1e3 * best['static_sinkage_m']:.1f} mm "
            f"(limit {1e3 * mi['max_sinkage_m']:.0f} mm)", best["static_sinkage_m"])
        e_day = best["array"]["energy_Wh"]
        chk("power.energy_margin", e_day >= best["assumptions"]["energy_margin"] * best["energy_need_Wh"] * (1 - 1e-9),
            f"array {best['array_m2']:.3f} m^2 gives {e_day:.1f} W h per sol against {best['energy_need_Wh']:.1f} W h "
            f"needed", e_day / best["energy_need_Wh"])
        chk("thermal.heater", None, f"survival heater {best['heater_W']:.2f} W at night (warm box "
            f"{best['geometry']['box_area_m2']:.3f} m^2, eps_eff {best['assumptions']['eps_eff']:g}); battery "
            f"{best['battery_Wh']:.1f} W h for the night", best["heater_W"], status="info")
        tr = traverse(best, planet=pl, mission=mi, n_wheels=n_wheels, max_slip=max_slip, seed=seed)
        resized = []
        # closed loop: the night-only battery sizing misses the dawn and dusk deficits the traverse sees; grow the
        # battery by the measured depth of discharge and fly the mission again (the mass follows)
        for _ in range(4):
            if tr.get("stuck") or tr["soc_min"] >= tr["soc_floor"] - 1e-9:
                break
            used = (1.0 - tr["soc_min"]) * best["battery_Wh"]
            need = 1.02 * used / best["assumptions"]["battery_dod"]
            resized.append({"battery_Wh_from": best["battery_Wh"], "battery_Wh_to": need, "soc_min": tr["soc_min"]})
            best = size_rover(radius=best["wheel"]["radius_m"], width=best["wheel"]["width_m"], soil=sl, planet=pl,
                              mission=mi, n_wheels=n_wheels, suspension=suspension, max_slip=max_slip,
                              cell_efficiency=cell_efficiency, assumptions=assumptions, battery_min_wh=need)
            tr = traverse(best, planet=pl, mission=mi, n_wheels=n_wheels, max_slip=max_slip, seed=seed)
        out["traverse"] = tr
        out["battery_resized"] = resized
        chk("traverse.battery_resized", None, "the traverse needed no larger battery" if not resized else
            "battery grown by the traverse's measured depth of discharge: " + "; ".join(
                f"{x['battery_Wh_from']:.1f} -> {x['battery_Wh_to']:.1f} W h (lowest SOC was {x['soc_min']:.1%})"
                for x in resized) + f"; total mass now {best['mass_kg']:.2f} kg", len(resized), status="info")
        if tr.get("stuck"):
            chk("traverse.mobile", False, tr["reason"])
        else:
            chk("traverse.distance", tr["reached"], f"{tr['distance_m']:.1f} m of {mi['distance_m']:g} m in "
                f"{tr['sols']:.2f} sols (limit {mi['max_sols']})", tr["distance_m"])
            chk("traverse.battery", tr["soc_min"] >= tr["soc_floor"] - 1e-9, f"lowest state of charge "
                f"{tr['soc_min']:.1%} (floor {tr['soc_floor']:.0%})", tr["soc_min"])
            chk("traverse.slip", tr["slip_max"] <= max_slip, f"largest slip {tr['slip_max']:.3f} (limit {max_slip:g})",
                tr["slip_max"])
            chk("traverse.energy_closure", tr["energy_closure"] <= 1e-9, f"battery change = solar in - loads - "
                f"losses - clipped to {tr['energy_closure']:.1e} of the solar energy", tr["energy_closure"])
        out["design"] = {k: v for k, v in best.items() if k != "model"}
    pub = published_rover_check(planet="mars")
    out["published"] = pub
    chk("published.sojourner_power", pub["within_band"], f"Sojourner array (0.22 m^2, 18 %) on Mars: peak "
        f"{pub['peak_power_W']:.1f} W against the reported {pub['reported_peak_W']:g} W ({pub['rel_error']:+.1%}, band "
        f"+-{SOJOURNER_BAND:.0%}; transmission is an assumption)", pub["rel_error"])
    chk("published.sojourner_drive", pub["drivable_on_array"], f"drive power at the reported 1 cm/s "
        f"{pub['drive_power_W']:.2f} W (slip {pub['drive_slip']:.3f}, soil proxy {pub['soil']}) within the peak array "
        "power", pub["drive_power_W"])
    out["assumptions"] = {**ASSUMPTIONS, **dict(assumptions or {})}
    out["assumption_notes"] = ASSUMPTION_NOTES
    out["checks"] = checks
    bad = [c["code"] for c in checks if c["status"] == "fail"]
    out["verdict"] = "fail" if bad else "pass"
    out["failed"] = bad
    return out
