"""Asteroids and comets: JPL Small-Body Database elements, a high-accuracy propagator in the field of the Sun, the
planets and the Moon from the DE421 ephemeris (with the Sun's relativistic term), close approaches to the Earth, and
a learned (neural) propagator checked against it.

    body = load_body("apophis")
    ca = close_approaches(body, jd_start, jd_end)          # [{jd, dist_au, v_rel_km_s}, ...]

Bundled elements and reference close approaches: JPL SBDB records (public domain) as shipped in the astroquery test
suite (BSD-3); Ceres reference vectors from JPL Horizons (DE441 + 16 perturbing asteroids) from the same source.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .ephemeris import AU_KM, C_KM_S, GM, GM_SUN, default_ephemeris
from .frames import ARCSEC, rot_x

DATA = os.path.join(os.path.dirname(__file__), "data", "smallbodies")
DAY = 86400.0
PERTURBERS = ["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]
# J2000 ecliptic (IAU 1976 obliquity, the SBDB convention) -> ICRF
ECL_TO_ICRF = rot_x(-84381.448 * ARCSEC)


@dataclass
class SmallBody:
    name: str
    epoch_jd: float                    # TDB
    a_au: float
    e: float
    i_deg: float
    node_deg: float
    peri_deg: float
    M_deg: float
    kind: str = ""
    neo: bool = False
    pha: bool = False
    H: float | None = None
    diameter_km: float | None = None
    ca_data: list = field(default_factory=list, repr=False)

    def state_icrf(self):
        """Heliocentric position (km) and velocity (km/s), ICRF axes, at the epoch."""
        from .orbits import state_from_elements
        el = {"a_km": self.a_au * AU_KM, "e": self.e, "i_deg": self.i_deg, "Omega_deg": self.node_deg,
              "omega_deg": self.peri_deg, "M_deg": self.M_deg}
        r, v = state_from_elements(el, GM_SUN, 0.0)
        return ECL_TO_ICRF @ np.ravel(r), ECL_TO_ICRF @ np.ravel(v)


def load_body(key: str) -> SmallBody:
    d = json.load(open(os.path.join(DATA, f"{key}.json"), encoding="utf-8"))
    el = {e["name"]: float(e["value"]) for e in d["orbit"]["elements"] if e["value"] is not None}
    phys = {p["name"]: p["value"] for p in d.get("phys_par") or []}
    return SmallBody(d["object"]["fullname"], float(d["orbit"]["epoch"]), el["a"], el["e"], el["i"], el["om"],
                     el["w"], el["ma"], d["object"].get("kind", ""), bool(d["object"].get("neo")),
                     bool(d["object"].get("pha")), float(phys["H"]) if phys.get("H") else None,
                     float(phys["diameter"]) if phys.get("diameter") else None, d.get("ca_data", []))


def available_bodies():
    """Keys of the bundled objects, the near-Earth asteroids first."""
    order = {"apophis": 0, "phaethon": 1}
    return sorted((f[:-5] for f in os.listdir(DATA) if f.endswith(".json")), key=lambda k: (order.get(k, 9), k))


def _accel(t_jd, r, v, eph, gr=True):
    """Barycentric acceleration (km/s^2) of a massless body at barycentric r (km), v (km/s)."""
    a = np.zeros(3)
    sun_r = sun_v = None
    for b in PERTURBERS:
        p, vv = eph.barycentric(b, np.array([t_jd]))
        d = p[0] - r
        a += GM[b] * d / np.linalg.norm(d) ** 3
        if b == "sun":
            sun_r, sun_v = p[0], vv[0]
    if gr:                                          # 1PN Schwarzschild term of the Sun (IERS Conventions eq. 10.12)
        x, u = r - sun_r, v - sun_v
        rr = np.linalg.norm(x)
        a += GM_SUN / (C_KM_S ** 2 * rr ** 3) * ((4 * GM_SUN / rr - u @ u) * x + 4 * (x @ u) * u)
    return a


class Trajectory:
    """One adaptive DOP853 integration from the epoch over [jd0, jd1] with dense output: positions and velocities
    anywhere in the span cost an interpolation, not a new integration."""

    def __init__(self, body: SmallBody, jd0: float, jd1: float, eph=None, rtol: float = 1e-11, gr: bool = True):
        from scipy.integrate import solve_ivp
        self.eph = eph or default_ephemeris()
        self.body = body
        r0, v0 = body.state_icrf()
        sp, sv = self.eph.barycentric("sun", np.array([body.epoch_jd]))
        y0 = np.concatenate([r0 + sp[0], v0 + sv[0]])

        def rhs(t, y):
            return np.concatenate([y[3:], _accel(body.epoch_jd + t / DAY, y[:3], y[3:], self.eph, gr)])
        self.sols = []
        for a, b in ((body.epoch_jd, max(jd1, body.epoch_jd)), (body.epoch_jd, min(jd0, body.epoch_jd))):
            if b != a:
                self.sols.append(solve_ivp(rhs, (0.0, (b - a) * DAY), y0, method="DOP853", rtol=rtol, atol=1e-6,
                                           dense_output=True))
        self.y0 = y0

    def barycentric(self, jd):
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        t = (jd - self.body.epoch_jd) * DAY
        out = np.tile(self.y0, (len(jd), 1))
        for sol in self.sols:
            lo, hi = sorted((sol.t[0], sol.t[-1]))
            m = (t >= lo) & (t <= hi) & (t != 0)
            if m.any():
                out[m] = sol.sol(t[m]).T
        return out[:, :3], out[:, 3:]

    def heliocentric(self, jd):
        r, v = self.barycentric(jd)
        s, sv = self.eph.barycentric("sun", np.atleast_1d(jd))
        return r - s, v - sv


def propagate(body: SmallBody, jd_out, eph=None, rtol: float = 1e-11, gr: bool = True):
    """Heliocentric ICRF positions (km) and velocities (km/s) at TDB Julian dates ``jd_out``."""
    jd_out = np.atleast_1d(np.asarray(jd_out, dtype=float))
    tr = Trajectory(body, float(jd_out.min()), float(jd_out.max()), eph, rtol, gr)
    return tr.heliocentric(jd_out)


def close_approaches(body: SmallBody, jd0: float, jd1: float, target: str = "earth", max_dist_au: float = 0.2,
                     step_days: float = 0.25, eph=None, traj: Trajectory | None = None):
    """Minima of the distance to ``target`` (geocentre by default), refined to ~1 s by golden section on the dense
    trajectory."""
    eph = eph or default_ephemeris()
    tr = traj or Trajectory(body, jd0, jd1, eph)

    def dist(jd):
        r, v = tr.heliocentric(jd)
        p, vt = eph.relative(target, "sun", np.atleast_1d(jd))
        return np.linalg.norm(r - p, axis=1), np.linalg.norm(v - vt, axis=1)
    jd = np.arange(jd0, jd1 + step_days, step_days)
    d, _ = dist(jd)
    out = []
    g = (math.sqrt(5) - 1) / 2
    for k in range(1, len(d) - 1):
        if d[k] <= d[k - 1] and d[k] < d[k + 1] and d[k] / AU_KM < max_dist_au:
            lo, hi = jd[k - 1], jd[k + 1]
            while (hi - lo) * DAY > 0.5:
                c1, c2 = hi - g * (hi - lo), lo + g * (hi - lo)
                if dist([c1])[0][0] < dist([c2])[0][0]:
                    hi = c2
                else:
                    lo = c1
            jm = 0.5 * (lo + hi)
            dm, vm = dist([jm])
            out.append({"jd_tdb": float(jm), "dist_km": float(dm[0]), "dist_au": float(dm[0] / AU_KM),
                        "v_rel_km_s": float(vm[0])})
    return out


def horizons_vectors(path=None):
    """Parse a JPL Horizons VECTORS table (CSV, AU-D, ecliptic J2000): jd, positions (AU), velocities (AU/d)."""
    path = path or os.path.join(DATA, "ceres_horizons_vectors_2022.txt")
    rows, on = [], False
    for line in open(path, encoding="utf-8"):
        if line.startswith("$$SOE"):
            on = True
            continue
        if line.startswith("$$EOE"):
            break
        if on:
            f = [x.strip() for x in line.split(",")]
            rows.append([float(f[0])] + [float(x) for x in f[2:8]])
    a = np.array(rows)
    return a[:, 0], a[:, 1:4], a[:, 4:7]


def cad_reference(body: SmallBody, year: int):
    """The JPL close-approach record (body = Earth) for a given year, with its 3-sigma distance range."""
    for c in body.ca_data:
        if c["body"] == "Earth" and c["cd"].startswith(str(year)):
            return {"cd": c["cd"], "jd_tdb": float(c["jd"]), "dist_au": float(c["dist"]),
                    "dist_min_au": float(c["dist_min"]), "dist_max_au": float(c["dist_max"]),
                    "v_rel_km_s": float(c["v_rel"]), "sigma_t_min": float(c["sigma_t"])}
    return None


def validation_summary(eph=None) -> dict:
    """The propagator against independent references: Ceres vs JPL Horizons vectors (2022, from 2018 elements) and
    the Apophis flyby of 2029-04-13 vs the JPL close-approach table (distance with its 3-sigma range, time, speed)."""
    eph = eph or default_ephemeris()
    c = load_body("ceres")
    jd, P, _V = horizons_vectors()
    r, _ = propagate(c, jd, eph)
    ceres_err = float(np.max(np.linalg.norm(r - (ECL_TO_ICRF @ P.T).T * AU_KM, axis=1)))
    ap = load_body("apophis")
    ref = cad_reference(ap, 2029)
    ca = close_approaches(ap, ref["jd_tdb"] - 2, ref["jd_tdb"] + 2, step_days=0.25, eph=eph)[0]
    d_err = abs(ca["dist_km"] - ref["dist_au"] * AU_KM)
    sig3 = (ref["dist_max_au"] - ref["dist_min_au"]) / 2 * AU_KM
    dt = (ca["jd_tdb"] - ref["jd_tdb"]) * DAY
    checks = [
        {"id": "ceres_horizons", "args": {"years": round((jd[0] - c.epoch_jd) / 365.25, 1)},
         "name": f"Ceres after {(jd[0] - c.epoch_jd) / 365.25:.1f} years vs JPL Horizons", "value": ceres_err,
         "unit": "km", "threshold": 100.0, "status": "pass" if ceres_err <= 100 else "fail"},
        {"id": "apophis_distance", "name": "Apophis 2029-04-13: distance vs JPL close approach (3-sigma range)", "value": d_err, "unit": "km",
         "threshold": sig3, "status": "pass" if d_err <= sig3 else "fail"},
        {"id": "apophis_time", "name": "Apophis 2029-04-13: time of closest approach vs JPL", "value": abs(dt), "unit": "s",
         "threshold": 120.0, "status": "pass" if abs(dt) <= 120 else "fail"},
    ]
    return {"ceres": {"max_err_km": ceres_err, "years": (jd[0] - c.epoch_jd) / 365.25,
                      "reference": "JPL Horizons (DE441 + 16 perturbing asteroids)"},
            "apophis_2029": {"model_dist_km": ca["dist_km"], "jpl_dist_km": ref["dist_au"] * AU_KM,
                             "jpl_3sigma_km": sig3, "dt_s": dt, "model_v_km_s": ca["v_rel_km_s"],
                             "jpl_v_km_s": ref["v_rel_km_s"], "jpl_cd": ref["cd"]},
            "checks": checks,
            "verdict": {"status": "trustworthy" if all(x["status"] == "pass" for x in checks) else "not_trustworthy"}}
