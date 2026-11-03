"""The standard satellite-design workflow (#1207): from the mission (ground sample distance, revisit, payload power
and data per day, lifetime) to the orbit, power, thermal, link and attitude budgets, the mass and power budget closed
at the least mass with margins, and an orbit propagation over the lifetime that re-verifies the margins.

1. **Orbit**: for each candidate altitude, the sun-synchronous inclination (J2 node drift = the Sun's mean motion),
   the beta angle over a year at the chosen local time of the node, the worst eclipse fraction, revisit and ground
   sample distance, the daily contact time with the ground station.
2. **Budgets** (:mod:`.budgets`): the solar array at end of life with degradation, the battery at the depth of
   discharge the cycle count allows, the coating and heater that keep the single node within its limits in the hot
   and cold cases, the data rate and transmitter power that close the link with margin, the disturbance torques
   (gravity gradient, drag at high solar flux, solar pressure, magnetic) with wheel and magnetorquer sizing.
3. **Lifetime**: orbit decay (:func:`pinneapple_physics.astro.decay.reference_decay`, NRLMSIS 2.0 when ``pymsis``
   is installed) for low, mean and high F10.7: the mission must survive the high-flux scenario and re-enter within
   the disposal limit in the low-flux one.
4. **Design**: the mass closed by fixed point (structure and harness as fractions, system margin), the least mass
   over the candidate altitudes that meets every constraint.
5. **Re-verification**: a Cowell J2 propagation (node drift and eclipse fraction measured on the trajectory), the
   decay to the end of life, and at the end-of-life altitude the power (battery state of charge stepped over the
   orbits), thermal (transient node bounded by the steady cases) and pointing margins (wheel momentum, torquer
   authority against the end-of-life drag).
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from . import budgets as B

__all__ = ["Mission", "satellite_design_workflow", "contact_time", "propagate_j2"]

RADIATOR_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


@dataclass(frozen=True)
class Mission:
    """An Earth-observation small satellite (representative 6U values, not a real mission)."""
    gsd_max_m: float = 6.0
    revisit_max_days: float = 35.0
    fov_deg: float = 10.0
    ifov_urad: float = 8.0
    payload_mass_kg: float = 2.0
    payload_power_w: float = 15.0
    payload_duty: float = 0.30                       # fraction of the sunlit time imaging
    data_gbit_per_day: float = 10.0
    lifetime_years: float = 3.0
    disposal_years: float = 25.0                     # passive re-entry limit (IADC guideline)
    ltan_h: float = 10.5
    dims_m: tuple[float, float, float] = (0.10, 0.226, 0.366)
    station_lat_deg: float = 60.0
    elev_min_deg: float = 10.0
    f_hz: float = 8.2e9
    g_tx_dbi: float = 6.0
    g_over_t_db_k: float = 18.0
    ebn0_req_db: float = 4.5
    link_margin_db: float = 3.0
    rates_mbps: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    tx_powers_w: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    t_min_c: float = -10.0
    t_max_c: float = 40.0
    #: constant-flux scenarios: the mission must survive "high"; disposal is judged at "mean" (a cycle average)
    f107: dict[str, float] = field(default_factory=lambda: {"low": 70.0, "mean": 150.0, "high": 200.0})
    bus_power_w: float = 3.0                         # computer, sensors, power conditioning
    power_margin: float = 0.20
    system_margin: float = 0.20
    structure_fraction: float = 0.20
    harness_fraction: float = 0.05


# --------------------------------------------------------------------------- geometry and propagation
def _orbit_frame(i_rad: float, raan: float = 0.0):
    p = np.array([math.cos(raan), math.sin(raan), 0.0])
    q = np.array([-math.sin(raan) * math.cos(i_rad), math.cos(raan) * math.cos(i_rad), math.sin(i_rad)])
    return p, q


def contact_time(h_km: float, i_deg: float, lat_deg: float, elev_min_deg: float, days: float = 3.0,
                 step_s: float = 5.0) -> dict[str, float]:
    """Mean daily time above ``elev_min`` from a station at ``lat`` (circular orbit, rotating spherical Earth)."""
    a = B.R_E + h_km * 1e3
    n = math.sqrt(B.MU / a ** 3)
    t = np.arange(0.0, days * 86400.0, step_s)
    p, q = _orbit_frame(math.radians(i_deg))
    r = a * (np.cos(n * t)[:, None] * p + np.sin(n * t)[:, None] * q)
    lat = math.radians(lat_deg)
    th = B.OMEGA_E * t
    g = B.R_E * np.stack([math.cos(lat) * np.cos(th), math.cos(lat) * np.sin(th), np.full_like(th, math.sin(lat))], 1)
    d = r - g
    up = g / B.R_E
    sin_el = np.sum(d * up, 1) / np.linalg.norm(d, axis=1)
    vis = sin_el >= math.sin(math.radians(elev_min_deg))
    passes = int(np.sum(np.diff(vis.astype(int)) == 1))
    return {"seconds_per_day": float(vis.mean() * 86400.0), "passes_per_day": passes / days}


def propagate_j2(h_km: float, i_deg: float, days: float, *, beta_deg: float = 0.0, n_out: int = 20000):
    """Cowell propagation with the J2 acceleration (DOP853, rtol 1e-11). Returns the node right ascension over
    time (unwrapped) and the eclipse fraction along the trajectory for a Sun fixed at ``beta`` to the initial plane,
    over the whole number of orbits nearest to ``days``."""
    from scipy.integrate import solve_ivp
    a = B.R_E + h_km * 1e3
    i = math.radians(i_deg)
    p, q = _orbit_frame(i)
    v = math.sqrt(B.MU / a)
    y0 = np.concatenate([a * p, v * q])

    def f(_t, y):
        r = y[:3]
        rn = np.linalg.norm(r)
        z2 = (r[2] / rn) ** 2
        k = 1.5 * B.J2 * B.MU * B.R_E ** 2 / rn ** 5
        acc = -B.MU * r / rn ** 3 + k * r * (5 * z2 - 1) - np.array([0.0, 0.0, 2 * k * r[2]])
        return np.concatenate([y[3:], acc])
    period = 2 * math.pi * math.sqrt(a ** 3 / B.MU)
    span = max(1, round(days * 86400.0 / period)) * period          # whole orbits: no partial-orbit bias
    t = np.linspace(0.0, span, n_out + 1)[:-1]
    sol = solve_ivp(f, (0.0, span), y0, t_eval=t, method="DOP853", rtol=1e-11, atol=1e-6)
    R, V = sol.y[:3].T, sol.y[3:].T
    hvec = np.cross(R, V)
    raan = np.unwrap(np.arctan2(hvec[:, 0], -hvec[:, 1]))
    nrm = np.cross(p, q)
    b = math.radians(beta_deg)
    s = math.cos(b) * p + math.sin(b) * nrm
    along = R @ s
    perp = np.linalg.norm(R - along[:, None] * s, axis=1)
    ecl = float(np.mean((along < 0) & (perp < B.R_E)))
    rate = float(np.polyfit(t, raan, 1)[0])
    alt = np.linalg.norm(R, axis=1) - B.R_E
    return {"raan_rate_rad_s": rate, "eclipse_fraction": ecl, "alt_km_min": float(alt.min() / 1e3),
            "alt_km_max": float(alt.max() / 1e3), "alt_km_mean": float(alt.mean() / 1e3), "success": bool(sol.success),
            "t": t}


def _lifetime_years(h_km: float, ballistic: float, f107: float, i_deg: float, max_years: float) -> tuple[float, Any]:
    from pinneapple_physics.astro.decay import DensityTable, reference_decay
    tab = DensityTable(f107=f107, f107a=f107, ap=15.0, inclination_deg=i_deg)
    t, h = reference_decay(h_km, ballistic, max_years * 365.25, tab, n=4000)
    k = np.flatnonzero(np.isnan(h))
    life = (t[k[0] - 1] / 365.25) if k.size else math.inf
    return float(life), (t, h, tab)


# --------------------------------------------------------------------------- one design point
def _evaluate(m: Mission, h_km: float) -> dict[str, Any]:
    i_deg = B.sso_inclination(h_km)
    rv = B.revisit(h_km, i_deg, m.fov_deg, m.ifov_urad)
    T = rv["period_s"]
    betas = B.beta_angles_sso(i_deg, m.ltan_h)
    f_e = max(B.eclipse_fraction(h_km, b) for b in betas)
    beta_worst = float(betas[np.argmin(np.abs(betas))])
    ct = contact_time(h_km, i_deg, m.station_lat_deg, m.elev_min_deg)
    need_bps = m.data_gbit_per_day * 1e9 / (0.8 * ct["seconds_per_day"])      # 80 % of the contact usable
    rate = next((r * 1e6 for r in m.rates_mbps if r * 1e6 >= need_bps), None)
    link, p_tx = None, None
    if rate is not None:
        for p in m.tx_powers_w:
            lb = B.link_budget(p_tx_w=p, f_hz=m.f_hz, h_km=h_km, elev_deg=m.elev_min_deg, g_tx_dbi=m.g_tx_dbi,
                               g_over_t_db_k=m.g_over_t_db_k, rate_bps=rate, ebn0_req_db=m.ebn0_req_db)
            if lb["margin_db"] >= m.link_margin_db:
                link, p_tx = lb, p
                break
    contact_frac = ct["seconds_per_day"] / 86400.0
    comms_w = ((p_tx or max(m.tx_powers_w)) / 0.35 + 2.0) * contact_frac + 0.5            # DC while transmitting
    # attitude and mass by fixed point (inertia and drag area depend on the mass and the arrays)
    from pinneapple_physics.astro.decay import orbit_mean_density
    rho_hi = float(orbit_mean_density([h_km], f107=m.f107["high"], f107a=m.f107["high"], inclination_deg=i_deg)[0])
    mass = 12.0
    a_, b_, c_ = sorted(m.dims_m)
    for _ in range(30):
        I = tuple(mass / 12 * (y * y + z * z) for y, z in ((b_, c_), (a_, c_), (a_, b_)))
        dist = B.disturbance_torques(h_km=h_km, rho_kg_m3=rho_hi, inertia=I, area_m2=b_ * c_, cp_offset_m=0.02,
                                     residual_dipole_am2=0.05)
        ad = B.wheel_and_torquer(dist["total"], T, dist["b_min_t"], I)
        adcs_w = ad["power_w"]
        p_day = (m.bus_power_w + adcs_w + comms_w + m.payload_power_w * m.payload_duty / (1 - f_e))
        th = None
        best_th = None
        for coat in (B.surface(c, f) for c in B.COATINGS for f in RADIATOR_FRACTIONS):
            th = B.thermal_cases(m.dims_m, h_km, coat, q_int_hot=p_day, q_int_cold=m.bus_power_w + adcs_w)
            if th["t_hot_c"] > m.t_max_c:
                continue
            tk = m.t_min_c + 273.15
            heater = max(0.0, coat.eps * B.SIGMA * th["area_m2"] * tk ** 4 - th["q_cold_w"])
            if best_th is None or heater < best_th[2]:
                best_th = (coat, th, heater)
        heater = best_th[2] if best_th else math.inf
        p_ecl = m.bus_power_w + adcs_w + comms_w + heater
        loads = (1 + m.power_margin)
        sa = B.solar_array(p_day * loads, p_ecl * loads, T, f_e, m.lifetime_years)
        bat = B.battery(p_ecl * loads, T, f_e, m.lifetime_years)
        comms_kg = 0.3 + 0.1 * (p_tx or 0.0)
        sub = {"payload": m.payload_mass_kg, "solar_array": sa["mass_kg"], "battery": bat["mass_kg"],
               "power_conditioning": 0.15 + 0.005 * sa["p_bol_w"], "adcs": ad["mass_kg"] + 0.2, "comms": comms_kg,
               "computer": 0.1, "thermal": 0.1}
        dry = sum(sub.values()) / (1 - m.structure_fraction - m.harness_fraction)
        new = dry * (1 + m.system_margin)
        if abs(new - mass) < 1e-9 * new:
            mass = new
            break
        mass = new if math.isfinite(new) else mass
        if not math.isfinite(new):
            break
    area_mean = (sum((a_ * b_, a_ * c_, b_ * c_)) / 3) + sa["area_m2"] / 2 if math.isfinite(sa["area_m2"]) else 0.05
    ballistic = 2.2 * area_mean / mass
    return {"h_km": h_km, "i_deg": i_deg, "revisit": rv, "beta_range_deg": [float(betas.min()), float(betas.max())],
            "beta_worst_deg": beta_worst, "eclipse_fraction": f_e, "contact": ct, "rate_bps": rate,
            "rate_needed_bps": need_bps, "link": link, "p_tx_w": p_tx, "comms_w": comms_w, "rho_high": rho_hi,
            "disturbance": dist, "adcs": ad, "inertia": I, "thermal": best_th[1] if best_th else th,
            "coating": asdict(best_th[0]) if best_th else None, "heater_w": heater, "p_day_w": p_day,
            "p_ecl_w": p_ecl, "solar_array": sa, "battery": bat, "subsystems_kg": sub, "dry_kg": dry,
            "mass_kg": mass, "ballistic_m2_kg": ballistic, "drag_area_m2": area_mean}


# --------------------------------------------------------------------------- the workflow
def satellite_design_workflow(*, mission: Mission | Mapping[str, Any] | None = None,
                              altitudes_km: Sequence[float] = tuple(range(400, 801, 25)), propagate_days: float = 1.0,
                              seed: int = 0) -> dict[str, Any]:
    """Run the workflow; see the module docstring. ``mission`` overrides :class:`Mission` fields."""
    t_start = time.perf_counter()
    m = mission if isinstance(mission, Mission) else replace(Mission(), **dict(mission or {}))
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        status = "skip" if ok == "skip" else ("inconclusive" if ok is None else ("pass" if ok else "fail"))
        checks.append({"code": code, "status": status, "detail": detail, "value": value})

    rows = []
    for h in altitudes_km:
        e = _evaluate(m, float(h))
        why = []
        if e["revisit"]["gsd_m"] > m.gsd_max_m:
            why.append(f"GSD {e['revisit']['gsd_m']:.2f} m")
        if e["revisit"]["revisit_days"] > m.revisit_max_days:
            why.append(f"revisit {e['revisit']['revisit_days']:.1f} d")
        if e["link"] is None:
            why.append("link does not close")
        if e["coating"] is None:
            why.append("no coating keeps the hot case under the limit")
        if not math.isfinite(e["mass_kg"]) or e["adcs"]["wheel"] is None or e["adcs"]["torquer"] is None:
            why.append("no wheel or torquer large enough")
        if not why:
            life_hi, _ = _lifetime_years(h, e["ballistic_m2_kg"], m.f107["high"], e["i_deg"], m.disposal_years * 4)
            life_lo, _ = _lifetime_years(h, e["ballistic_m2_kg"], m.f107["mean"], e["i_deg"], m.disposal_years * 4)
            e["lifetime_years"] = {"high": life_hi, "mean": life_lo}
            if life_hi < m.lifetime_years:
                why.append(f"decays in {life_hi:.2f} y at F10.7 {m.f107['high']:g}")
            if life_lo > m.disposal_years:
                why.append(f"stays {life_lo:.0f} y at F10.7 {m.f107['mean']:g} (> {m.disposal_years:g})")
        e["infeasible"] = why
        rows.append(e)
    feas = [e for e in rows if not e["infeasible"]]
    trade = [{"h_km": e["h_km"], "mass_kg": e["mass_kg"], "gsd_m": e["revisit"]["gsd_m"],
              "revisit_days": e["revisit"]["revisit_days"], "eclipse": e["eclipse_fraction"],
              "lifetime_high_years": (e.get("lifetime_years") or {}).get("high"), "infeasible": e["infeasible"]}
             for e in rows]
    if not feas:
        chk("design.feasible", False, "no candidate altitude meets every constraint: " + "; ".join(
            f"{e['h_km']:g} km: {', '.join(e['infeasible'])}" for e in rows))
        return {"mission": asdict(m), "trade": trade, "checks": checks, "failed": ["design.feasible"],
                "verdict": "fail", "summary": "no feasible design", "seconds": time.perf_counter() - t_start}
    d = min(feas, key=lambda e: e["mass_kg"])
    h, i_deg = d["h_km"], d["i_deg"]
    chk("design.feasible", True, f"{len(feas)} of {len(rows)} altitudes feasible; least mass {d['mass_kg']:.3f} kg at "
                                 f"{h:g} km (SSO {i_deg:.3f} deg, LTAN {m.ltan_h:g} h)", d["mass_kg"])

    # ---- verification of the closed forms
    a = B.R_E + h * 1e3
    rate = B.nodal_rate(a, 0.0, math.radians(i_deg))
    e_sso = abs(rate - B.OMEGA_SUN) / B.OMEGA_SUN
    chk("verify.sso_formula", e_sso < 1e-12, f"node drift at i = {i_deg:.4f} deg: {math.degrees(rate) * 86400:.6f} "
                                             f"deg/day = the Sun's {math.degrees(B.OMEGA_SUN) * 86400:.6f} ({e_sso:.1g})",
        e_sso)
    errs = [abs(B.eclipse_fraction(h, b) - B.eclipse_fraction_sampled(h, b)) for b in (0.0, d["beta_worst_deg"], 45.0,
                                                                                         60.0)]
    chk("verify.eclipse_geometry", max(errs) < 1e-4, f"closed-form eclipse fraction against the sampled cylindrical "
                                                     f"shadow at beta 0, {d['beta_worst_deg']:.1f}, 45, 60 deg: worst "
                                                     f"{max(errs):.1g}", max(errs))
    lb = d["link"]
    lin = (d["p_tx_w"] * 10 ** ((m.g_tx_dbi - 1.0) / 10) * (B.C_LIGHT / (4 * math.pi * lb["range_km"] * 1e3 * m.f_hz)) ** 2
           * 10 ** (-lb["other_losses_db"] / 10) * 10 ** (m.g_over_t_db_k / 10) / (1.380649e-23 * lb["rate_bps"]))
    e_link = abs(10 * math.log10(lin) - lb["ebn0_db"])
    chk("verify.link_linear", e_link < 1e-9, f"Eb/N0 {lb['ebn0_db']:.3f} dB from the dB chain = the linear-units "
                                             f"Friis product ({e_link:.1g} dB)", e_link)
    chk("link.margin", lb["margin_db"] >= m.link_margin_db,
        f"{d['p_tx_w']:g} W at {m.f_hz / 1e9:g} GHz, {lb['rate_bps'] / 1e6:g} Mbit/s, {lb['range_km']:.0f} km at "
        f"{m.elev_min_deg:g} deg: EIRP {lb['eirp_dbw']:.2f} dBW, path loss {lb['fspl_db']:.2f} dB, Eb/N0 "
        f"{lb['ebn0_db']:.2f} dB, margin {lb['margin_db']:.2f} dB (>= {m.link_margin_db:g})", lb["margin_db"])
    ct = d["contact"]
    chk("link.data_volume", lb["rate_bps"] * 0.8 * ct["seconds_per_day"] >= m.data_gbit_per_day * 1e9,
        f"{ct['passes_per_day']:.1f} passes, {ct['seconds_per_day'] / 60:.1f} min/day over {m.station_lat_deg:g} deg "
        f"latitude: {lb['rate_bps'] * 0.8 * ct['seconds_per_day'] / 1e9:.2f} Gbit/day (need "
        f"{m.data_gbit_per_day:g})")
    th = d["thermal"]
    chk("thermal.hot_cold", th["t_hot_c"] <= m.t_max_c and (th["t_cold_c"] >= m.t_min_c or d["heater_w"] > 0),
        f"{d['coating']['name']} (alpha {d['coating']['alpha']}, eps {d['coating']['eps']}): hot {th['t_hot_c']:.1f} C, "
        f"cold {th['t_cold_c']:.1f} C, heater {d['heater_w']:.2f} W in eclipse (limits {m.t_min_c:g} to "
        f"{m.t_max_c:g} C)")
    bat = d["battery"]
    chk("power.battery_dod", True, f"{bat['cycles']:.0f} cycles allow DoD {bat['dod']:.0%}: {bat['capacity_wh']:.1f} Wh"
                                   f" for {bat['eclipse_wh']:.2f} Wh per eclipse (worst eclipse "
                                   f"{d['eclipse_fraction']:.1%} of the orbit)")

    # ---- re-verification: propagation, lifetime, margins at end of life
    pj = propagate_j2(h, i_deg, propagate_days, beta_deg=d["beta_worst_deg"])
    e_raan = abs(pj["raan_rate_rad_s"] - B.OMEGA_SUN) / B.OMEGA_SUN
    chk("reverify.sso_propagated", pj["success"] and e_raan < 0.02,
        f"Cowell J2 propagation over {propagate_days:g} days: node drifts {math.degrees(pj['raan_rate_rad_s']) * 86400:.5f}"
        f" deg/day vs {math.degrees(B.OMEGA_SUN) * 86400:.5f} ({e_raan:.2%}; short-period J2 terms remain)", e_raan)
    f_mean = B.eclipse_fraction(pj["alt_km_mean"], d["beta_worst_deg"])
    e_ecl = abs(pj["eclipse_fraction"] - f_mean)
    chk("reverify.eclipse_propagated", e_ecl < 0.002, f"eclipse fraction on the propagated trajectory "
                                                       f"{pj['eclipse_fraction']:.4f} vs closed form {f_mean:.4f} at its "
                                                       f"mean altitude {pj['alt_km_mean']:.1f} km (J2 range "
                                                       f"{pj['alt_km_min']:.1f}-{pj['alt_km_max']:.1f} km; design value "
                                                       f"{d['eclipse_fraction']:.4f} at {h:g} km)", e_ecl)
    life = {}
    curves = {}
    for k, f in m.f107.items():
        life[k], (t_d, h_d, tab) = _lifetime_years(h, d["ballistic_m2_kg"], f, i_deg, m.disposal_years * 4)
        curves[k] = (t_d, h_d)
    t_hi, h_hi = curves["high"]
    h_eol = float(np.interp(m.lifetime_years * 365.25, t_hi[np.isfinite(h_hi)], h_hi[np.isfinite(h_hi)]))
    from pinneapple_physics.astro.decay import msis_available
    if not msis_available():
        chk("lifetime.flux_scenarios", None, "pymsis is not installed: the exponential atmosphere (mean solar "
                                             "activity) was used for every F10.7 scenario; pip install "
                                             "'pinneapple[sky]' for NRLMSIS 2.0")
    chk("lifetime.survives", life["high"] >= m.lifetime_years,
        f"lifetime {life['high']:.2f} y at F10.7 {m.f107['high']:g}, {life['mean']:.2f} y at {m.f107['mean']:g}, "
        f"{life['low']:.2f} y at {m.f107['low']:g} (mission {m.lifetime_years:g} y; B = {d['ballistic_m2_kg']:.4f} m^2/kg,"
        f" {tab.model})", life["high"])
    chk("lifetime.disposal", life["mean"] <= m.disposal_years,
        f"re-enters within {life['mean']:.1f} y at the mean flux {m.f107['mean']:g} (limit {m.disposal_years:g} y; "
        f"{life['low']:.1f} y if the flux stayed at {m.f107['low']:g})", life["mean"])
    # end-of-life point: power, thermal, pointing
    from pinneapple_physics.astro.decay import orbit_mean_density
    T_eol = 2 * math.pi * math.sqrt((B.R_E + h_eol * 1e3) ** 3 / B.MU)
    f_eol = max(B.eclipse_fraction(h_eol, b) for b in B.beta_angles_sso(i_deg, m.ltan_h))
    sa = d["solar_array"]
    p_avail = sa["area_m2"] * sa["p_eol_w_m2"]
    soc = _battery_soc(p_avail, d["p_day_w"], d["p_ecl_w"], T_eol, f_eol, bat["capacity_wh"], n_orbits=15)
    chk("reverify.power_eol", soc["min"] >= 1 - bat["dod"] - 1e-6 and soc["periodic"],
        f"at {h_eol:.1f} km after {m.lifetime_years:g} y (high flux): EOL array {p_avail:.2f} W, eclipse "
        f"{f_eol:.1%}; state of charge over 15 orbits min {soc['min']:.1%} (floor {1 - bat['dod']:.0%}), "
        f"recharged each orbit: {soc['periodic']}", soc["min"])
    rho_eol = float(orbit_mean_density([h_eol], f107=m.f107["high"], f107a=m.f107["high"], inclination_deg=i_deg)[0])
    a_, b_, c_ = sorted(m.dims_m)
    dist = B.disturbance_torques(h_km=h_eol, rho_kg_m3=rho_eol, inertia=d["inertia"], area_m2=b_ * c_,
                                 cp_offset_m=0.02, residual_dipole_am2=0.05)
    h_need = 0.707 * dist["total"] * T_eol / 4
    wheel, torq = d["adcs"]["wheel"], d["adcs"]["torquer"]
    chk("reverify.pointing_eol", h_need <= wheel["h_nms"] and torq["dipole_am2"] * dist["b_min_t"] >= dist["total"],
        f"EOL disturbance {dist['total'] * 1e6:.3g} uN m (drag {dist['aerodynamic'] * 1e6:.3g}): wheel momentum "
        f"{h_need * 1e3:.3g} mN m s of {wheel['h_nms'] * 1e3:.3g}; torquer {torq['dipole_am2'] * dist['b_min_t'] * 1e6:.3g}"
        f" uN m at the equator (>= the disturbance)", h_need / wheel["h_nms"])
    coat = B.Coating(**d["coating"])
    th_eol = B.thermal_cases(m.dims_m, h_eol, coat, q_int_hot=d["p_day_w"], q_int_cold=m.bus_power_w
                             + d["adcs"]["power_w"])
    tr = _thermal_transient(coat, th_eol, d["p_day_w"], m.bus_power_w + d["adcs"]["power_w"] + d["heater_w"],
                            d["mass_kg"], T_eol, f_eol, m)
    chk("reverify.thermal_eol", tr["t_max_c"] <= m.t_max_c + 1e-6 and tr["t_min_c"] >= m.t_min_c - 1e-6
        and tr["within_steady_bounds"],
        f"single-node transient over 20 orbits at EOL: {tr['t_min_c']:.1f} to {tr['t_max_c']:.1f} C, inside the steady "
        f"hot/cold cases ({tr['t_cold_steady_c']:.1f}, {th_eol['t_hot_c']:.1f} C) and the limits")

    failed = [c["code"] for c in checks if c["status"] == "fail"]
    summary = (f"{d['mass_kg']:.2f} kg at {h:g} km SSO {i_deg:.2f} deg: GSD {d['revisit']['gsd_m']:.2f} m, revisit "
               f"{d['revisit']['revisit_days']:.1f} d, array {sa['area_m2']:.3f} m^2 ({sa['p_bol_w']:.1f} W BOL), "
               f"battery {bat['capacity_wh']:.1f} Wh, {d['p_tx_w']:g} W at {lb['rate_bps'] / 1e6:g} Mbit/s "
               f"(margin {lb['margin_db']:.1f} dB), lifetime {life['high']:.1f} y at high flux; "
               f"{sum(c['status'] == 'pass' for c in checks)} checks pass, {len(failed)} fail"
               + (f" ({', '.join(failed)})" if failed else ""))
    clean = {k: v for k, v in d.items() if k not in ("infeasible",)}
    return {"mission": asdict(m), "design": clean, "trade": trade, "lifetime_years": life, "h_eol_km": h_eol,
            "propagation": {k: v for k, v in pj.items() if k != "t"}, "soc": soc, "thermal_transient": tr,
            "checks": checks, "failed": failed, "verdict": "pass" if not failed else "fail", "summary": summary,
            "density_model": tab.model, "seconds": time.perf_counter() - t_start, "seed": seed}


def _battery_soc(p_array: float, p_day: float, p_ecl: float, period: float, f_ecl: float, cap_wh: float, *,
                 n_orbits: int = 15, eta: float = 0.9, steps: int = 400) -> dict[str, Any]:
    """State of charge stepped over orbits: the array charges (efficiency ``eta``) up to full in sunlight, the
    battery supplies the eclipse load."""
    soc, cap = 1.0, cap_wh * 3600.0
    dt = period / steps
    out = []
    for _ in range(n_orbits):
        for k in range(steps):
            ecl = (k + 0.5) / steps >= 1 - f_ecl
            net = -p_ecl / eta if ecl else (p_array - p_day) * eta
            soc = min(1.0, soc + net * dt / cap)
            out.append(soc)
    out = np.asarray(out)
    last = out[-steps:]
    return {"min": float(out.min()), "end": float(out[-1]), "periodic": bool(abs(last[-1] - out[-1 - steps]) < 1e-9
                                                                              and last.max() >= 1.0 - 1e-12)}


def _thermal_transient(coat, th, q_day, q_ecl, mass_kg, period, f_ecl, m, *, c_j_kgk: float = 900.0,
                       n_orbits: int = 20, steps: int = 600) -> dict[str, Any]:
    """``m c dT/dt = Q_in(t) - eps sigma A T^4`` with the sunlit and eclipse inputs of the steady cases."""
    q_sun = th["q_hot_w"] - q_day
    q_earth = coat.eps * B.EARTH_IR * th["view_factor"] * th["a_nadir_m2"]
    C = mass_kg * c_j_kgk
    T = (th["t_hot_c"] + th["t_cold_c"]) / 2 + 273.15
    dt = period / steps
    temps = []
    for _ in range(n_orbits):
        for k in range(steps):
            ecl = (k + 0.5) / steps >= 1 - f_ecl
            q = q_earth + q_ecl if ecl else q_sun + q_day          # q_sun holds the Sun, albedo and Earth IR
            T += dt * (q - coat.eps * B.SIGMA * th["area_m2"] * T ** 4) / C
            temps.append(T - 273.15)
    last = np.asarray(temps[-steps:])
    t_cold = (max(q_earth + q_ecl, 1e-9) / (coat.eps * B.SIGMA * th["area_m2"])) ** 0.25 - 273.15
    return {"t_min_c": float(last.min()), "t_max_c": float(last.max()), "t_cold_steady_c": float(t_cold),
            "within_steady_bounds": bool(last.min() >= t_cold - 1e-6 and last.max() <= th["t_hot_c"] + 1e-6)}
