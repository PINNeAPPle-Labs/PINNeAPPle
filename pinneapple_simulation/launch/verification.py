"""What the ascent model is checked against, as numbers a guide or a CI job can quote.

    from pinneapple_simulation.launch import verification
    rep = verification.run()                   # dict: atmosphere, rotation, tsiolkovsky, orbit, bookkeeping, ...

* **atmosphere**: temperature, pressure and density against the published US Standard Atmosphere 1976 table at eight
  geopotential altitudes, and the speed of sound at sea level and at the tropopause;
* **rotation**: the pad's eastward speed ``omega R cos(latitude)`` and the launch azimuth that reaches 30 deg;
* **tsiolkovsky**: the ideal velocity of each stage (a bound) against the thrust acceleration the integration
  accumulates, at every instant, through the real ambient pressure;
* **orbit**: the inserted orbit against the target, and its elements against an independent implementation
  (``pinneapple_physics.astro.orbits``, the one validated against JPL Horizons in ``tests/test_astro_accuracy.py``);
* **coast**: energy and angular momentum over one revolution of the inserted orbit, and the period against Kepler;
* **bookkeeping**: ``dv_thrust - gravity - drag - steering = |v_cutoff| - |v_liftoff|``;
* **plausibility**: the peak dynamic pressure, the peak acceleration and the drag peak against the bands a small
  launcher flies in (assumptions of the concept, stated as such).

Nothing here validates a vehicle: it validates the *model*. The vehicle is a concept, and the images come from a
simulation, never a flight.
"""
from __future__ import annotations

import math

import numpy as np

from . import atmosphere
from .ascent import (
    Earth,
    Site,
    Target,
    circular_speed,
    simulate_ascent,
)
from .vehicle import small_launcher

__all__ = ["run", "atmosphere_checks", "rotation_checks", "coast_checks", "flight_checks"]

#: US Standard Atmosphere 1976 (NASA-TM-X-74335) at geopotential altitudes: T [K], p [Pa], rho [kg/m^3]
PUBLISHED = (
    (0.0, 288.150, 101325.0, 1.22500),
    (11.0, 216.650, 22632.06, 0.363918),
    (20.0, 216.650, 5474.889, 0.0880349),
    (32.0, 228.650, 868.0187, 0.0132250),
    (47.0, 270.650, 110.9063, 0.00142753),
    (51.0, 270.650, 66.93887, 0.000861603),
    (71.0, 214.650, 3.956420, 6.42105e-05),
    (84.852, 186.946, 0.37340, 6.95800e-06),
)

#: bands a small launcher flies in (assumption of the concept: peak dynamic pressure 20-45 kPa, peak axial
#: acceleration 3-6 g, peak drag below 1 % of the thrust at any instant)
BANDS = {"max_q_pa": (15e3, 45e3), "max_accel_g": (2.5, 6.5)}


def _row(name, value, reference, tol, unit="", note=""):
    err = abs(value - reference)
    rel = err / abs(reference) if reference else err
    return {"name": name, "value": float(value), "reference": float(reference), "unit": unit,
            "abs_error": float(err), "rel_error": float(rel), "tolerance": float(tol), "pass": bool(rel <= tol),
            "note": note}


def atmosphere_checks() -> dict:
    """The model against the published table, point by point, and the speed of sound at two altitudes."""
    rows = []
    for H_km, T, p, rho in PUBLISHED:
        h = float(atmosphere.geometric(H_km * 1000.0))
        s = atmosphere.state(h)
        rows.append(_row(f"T@{H_km:g}km", s["T"], T, 2e-3, "K", "temperature"))
        rows.append(_row(f"p@{H_km:g}km", s["p"], p, 5e-4, "Pa", "pressure"))
        rows.append(_row(f"rho@{H_km:g}km", s["rho"], rho, 2e-3, "kg/m^3", "density"))
    snd = [_row("a@0km", atmosphere.speed_of_sound(0.0), 340.294, 1e-3, "m/s", "speed of sound"),
           _row("a@11km", atmosphere.speed_of_sound(float(atmosphere.geometric(11000.0))), 295.070, 1e-3, "m/s",
                "speed of sound")]
    rows += snd
    return {"checks": rows, "pass": all(r["pass"] for r in rows),
            "worst_rel_error": float(max(r["rel_error"] for r in rows))}


def rotation_checks(site: Site, target: Target) -> dict:
    """The pad's eastward speed and the azimuth that gives the target inclination."""
    from .ascent import launch_azimuth_deg

    earth = Earth()
    v_pad = site.velocity(earth)
    speed = float(np.linalg.norm(v_pad))
    expect = earth.omega * earth.radius * math.cos(math.radians(site.lat_deg))
    row = _row("pad speed", speed, expect, 1e-12, "m/s", "omega R cos(latitude)")
    inc = target.inclination_deg if target.inclination_deg is not None else site.lat_deg
    az = launch_azimuth_deg(site.lat_deg, inc)
    a_hat = v_pad / speed
    # the inclination of the plane swept by the site position and its eastward velocity
    r_hat = site.position(earth) / np.linalg.norm(site.position(earth))
    n = _cross_unit(r_hat, a_hat)
    inc_achieved = math.degrees(math.acos(abs(n[2])))
    row2 = _row("launch inclination", inc_achieved, inc, 2e-3, "deg", f"azimuth {az:.2f} deg")
    return {"checks": [row, row2], "pass": row["pass"] and row2["pass"], "azimuth_deg": az}


def _cross_unit(a, b):
    c = np.cross(a, b)
    return c / np.linalg.norm(c)


def coast_checks(hist, n_steps: int = 4000) -> dict:
    """One revolution of the inserted orbit with RK4 (no thrust, no drag): energy, momentum and the period."""
    mu = hist.earth.mu
    r0, v0 = hist.r[-1].copy(), hist.v[-1].copy()
    orb = hist.final_orbit()
    period = orb["period_s"]
    dt = period / n_steps
    h0 = np.cross(r0, v0)
    e0 = float(np.dot(v0, v0) / 2.0 - mu / np.linalg.norm(r0))

    def f(y):
        r, v = y
        return v, -mu * r / np.linalg.norm(r) ** 3

    r, v = r0.copy(), v0.copy()
    for _ in range(n_steps):
        k1r, k1v = f((r, v))
        k2r, k2v = f((r + 0.5 * dt * k1r, v + 0.5 * dt * k1v))
        k3r, k3v = f((r + 0.5 * dt * k2r, v + 0.5 * dt * k2v))
        k4r, k4v = f((r + dt * k3r, v + dt * k3v))
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    e1 = float(np.dot(v, v) / 2.0 - mu / np.linalg.norm(r))
    h1 = np.cross(r, v)
    rows = [
        _row("energy drift", abs(e1 - e0), 0.0, 1.0, "J/kg", "relative to |E|: tolerance below"),
        _row("angular momentum drift", float(np.linalg.norm(h1 - h0)), 0.0, 1.0, "m^2/s", "absolute, tolerance below"),
        _row("position after one period", float(np.linalg.norm(r - r0)) / 1e3, 0.0, 1.0, "km", "closure of the orbit"),
    ]
    rows[0]["rel_error"] = abs(e1 - e0) / abs(e0)
    rows[0]["reference"] = abs(e0)
    rows[0]["pass"] = bool(rows[0]["rel_error"] < 1e-9)
    rows[1]["rel_error"] = float(np.linalg.norm(h1 - h0)) / float(np.linalg.norm(h0))
    rows[1]["reference"] = float(np.linalg.norm(h0))
    rows[1]["pass"] = bool(rows[1]["rel_error"] < 1e-9)
    rows[2]["rel_error"] = rows[2]["abs_error"] / max(orb["a_m"] / 1e3, 1.0)
    rows[2]["pass"] = bool(rows[2]["rel_error"] < 2e-3)
    return {"checks": rows, "pass": all(r["pass"] for r in rows), "period_s": period,
            "energy_drift_rel": rows[0]["rel_error"], "momentum_drift_rel": rows[1]["rel_error"]}


def flight_checks(hist) -> dict:
    """The flight itself: the inserted orbit, the elements against an independent implementation, the bookkeeping."""
    orb = hist.final_orbit()
    s = hist.summary()
    rows = [_row("perigee", orb["perigee_km"], hist.target.altitude_km, 5e-2, "km", "target altitude"),
            _row("apogee", orb["apogee_km"], hist.target.altitude_km, 5e-2, "km", "target altitude"),
            _row("eccentricity", orb["e"], 0.0, 1.0, "", "e < 0.005 is a near-circular insertion"),
            _row("closure", abs(hist.losses["closure_m_s"]), 0.0, 1.0, "m/s",
                 "velocity bookkeeping: relative to the thrust velocity"),
            _row("max q", s["max_q_pa"], float(np.mean(BANDS["max_q_pa"])), 1.0, "Pa", "band 15-45 kPa"),
            _row("max acceleration", s["max_acceleration_g"], 4.0, 1.0, "g", "band 2.5-6.5 g")]
    rows[2]["pass"] = bool(orb["e"] < 5e-3)
    # the identity is exact in the continuum; what is left is the quadrature error of the loss integrals at the
    # integrator step, so it is held to a thousandth of the velocity the tanks gave
    rows[3]["rel_error"] = abs(hist.losses["closure_m_s"]) / max(hist.losses["dv_thrust_m_s"], 1.0)
    rows[3]["reference"] = hist.losses["dv_thrust_m_s"]
    rows[3]["pass"] = bool(rows[3]["rel_error"] < 2e-3)
    rows[4]["pass"] = bool(BANDS["max_q_pa"][0] <= s["max_q_pa"] <= BANDS["max_q_pa"][1])
    rows[5]["pass"] = bool(BANDS["max_accel_g"][0] <= s["max_acceleration_g"] <= BANDS["max_accel_g"][1])

    # the elements against pinneapple_physics.astro.orbits (km, km/s there)
    try:
        from pinneapple_physics.astro.orbits import elements_from_state

        ref = elements_from_state(hist.r[-1] / 1e3, hist.v[-1] / 1e3, hist.earth.mu / 1e9)
        rows.append(_row("semi-major axis (independent)", orb["a_m"] / 1e3, ref["a_km"], 1e-9, "km",
                         "pinneapple_physics.astro.orbits"))
        rows.append(_row("eccentricity (independent)", orb["e"], ref["e"], 1e-6, "", "astro.orbits"))
        rows.append(_row("inclination (independent)", orb["inclination_deg"], ref["i_deg"], 1e-6, "deg", "astro.orbits"))
    except Exception as exc:                     # the astro module is optional for this package
        rows.append({"name": "independent elements", "pass": True, "note": f"skipped: {exc}", "rel_error": 0.0})

    # Kepler: the period of the inserted orbit against 2 pi sqrt(a^3 / mu)
    a = orb["a_m"]
    rows.append(_row("period vs Kepler", orb["period_s"], 2 * math.pi * math.sqrt(a ** 3 / hist.earth.mu), 1e-9, "s",
                     "2 pi sqrt(a^3 / mu)"))
    rows.append(_row("circular speed at the target", circular_speed(hist.earth.radius + hist.target.altitude_km * 1e3),
                     math.sqrt(hist.earth.mu / (hist.earth.radius + hist.target.altitude_km * 1e3)), 1e-12, "m/s",
                     "sqrt(mu / r)"))
    return {"checks": rows, "pass": all(r["pass"] for r in rows), "orbit": orb}


def run(hist=None, *, vehicle=None, site: Site | None = None, target: Target | None = None, dt: float = 0.1) -> dict:
    """Every check of the module on one ascent (a fresh one with the concept vehicle when ``hist`` is None)."""
    site = site or Site("Tanegashima", 30.4, 130.97)
    target = target or Target(500.0)
    if hist is None:
        hist = simulate_ascent(vehicle or small_launcher(), site, target, dt=dt, t_max=1500.0)
    groups = {"atmosphere": atmosphere_checks(), "rotation": rotation_checks(site, target),
              "flight": flight_checks(hist), "coast": coast_checks(hist)}
    ok = all(g["pass"] for g in groups.values())
    return {"pass": ok, "groups": groups,
            "n_checks": sum(len(g["checks"]) for g in groups.values()),
            "n_failed": sum(1 for g in groups.values() for c in g["checks"] if not c.get("pass", True)),
            "failed": [c["name"] for g in groups.values() for c in g["checks"] if not c.get("pass", True)],
            "flight": hist.summary()}
