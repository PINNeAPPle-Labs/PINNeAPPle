"""Planetary defence: the momentum a kinetic impactor delivers, what it does to the orbit, and how far ahead the
miss can be guaranteed (#381).

The chain is short and every link is measurable:

1. **the impulse** — :func:`delta_v` from the impactor's mass and speed, the asteroid's mass and the momentum
   enhancement ``beta`` (the ejecta carry more momentum than the impactor brought). ``beta = 3.61`` is not a guess:
   it is what DART measured at Dimorphos, and :func:`dart_beta` recovers it from the published period change, so the
   model is held to the mission that flew it.
2. **the orbit** — an along-track impulse changes the velocity and, through the vis-viva relation, the semi-major
   axis and the period (:func:`period_change_fraction`: ``dP/P = 3 dv/v``). Nudged years before an encounter, a tiny
   ``dv`` moves the body off the resonant approach: :func:`deflect` applies the impulse at a date, rebuilds the
   osculating elements there and lets :func:`pinneapple_physics.astro.smallbodies.close_approaches` measure what it
   did to the miss distance.
3. **the uncertainty** — the orbit is not known to the metre: the Yarkovsky drift (:func:`yarkovsky_drift`) is the
   largest unmodelled acceleration and ``beta`` is uncertain too, so the answer is a distribution of miss distances,
   not a number. :func:`success_probability` is the fraction of that ensemble that clears the Earth, and
   :func:`trust_horizon` is the warning time beyond which the ensemble spread is wider than the Earth itself -- the
   age after which a "we will miss" cannot be guaranteed whatever the model says.
4. **the budget** — every acceleration the propagator carries is either measured or shown not to matter:
   :func:`perturbation_budget` drops each perturber in turn and switches solar radiation pressure on, and reports what
   the 2029 miss does. The numbers are what the guide quotes, and the small ones are named as such.

What is measured here against the public record, in the tests: the DART ``beta`` recovered from the 32-minute period
change, and the 2029 Apophis encounter (JD 2462240.41, 0.000256 au propagated / 0.000252 au published, 7.4 km/s) from
the bundled JPL elements, with the b-plane ellipse JPL publishes for it (:func:`jpl_miss_sigma_km`).
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .ephemeris import AU_KM, GM_SUN
from .orbits import elements_from_state
from .smallbodies import P_SRP, PERTURBERS, SmallBody

__all__ = ["BETA_DART", "DART", "APOPHIS_2029_JD", "GM_SUN_AU3_DAY2", "Terms", "delta_v", "period_change_fraction",
           "dart_beta", "dimorphos_period_hours", "yarkovsky_drift", "area_to_mass_ratio", "solar_radiation_pressure",
           "impulse_body", "deflect", "miss_distance", "deflection_table", "sample_ensemble", "success_probability",
           "trust_horizon", "impact_probability", "jpl_miss_sigma_km", "perturbation_budget", "ensemble_arcs"]

#: Gaussian gravitational constant squared, au^3/day^2: the mean motion of an orbit in the units the Yarkovsky ``A2``
#: parameter is published in. ``ephemeris.GM_SUN`` is km^3/s^2, which is the right constant for the propagator and the
#: wrong one here.
GM_SUN_AU3_DAY2 = 2.959122082855911e-4

#: Momentum enhancement measured by DART at Dimorphos (Cheng et al. 2023; ~3.6, the period change implies the ejecta
#: carried 2.6x the impactor's own momentum).
BETA_DART = 3.61

#: The published numbers the ``beta`` is recovered from: DART hit Dimorphos on 2022-09-26 with 570 kg at 6.1 km/s,
#: the moonlet's orbital period around Didymos fell from 11.92 h to 11.37 h (32 +/- 2 min), and Dimorphos' mass is
#: 4.8e9 kg (diameter ~151 m at ~2400 kg/m3).
DART = {"impactor_kg": 570.0, "speed_km_s": 6.1, "target_kg": 4.8e9,
        "period_before_s": 11.92 * 3600.0, "period_change_s": -32.0 * 60.0}

#: 2029-04-13, the Apophis encounter the ensemble is anchored on (JD TDB of closest approach as propagated here).
APOPHIS_2029_JD = 2462240.41


@dataclass(frozen=True)
class Terms:
    """The small accelerations a real impact solution carries, every one of them optional and off by default.

    ``a2_au_d2`` is the Yarkovsky transverse acceleration in JPL's ``A2`` units; ``area_to_mass_m2_kg`` with
    ``reflectivity`` switches solar radiation pressure on (:func:`solar_radiation_pressure` computes what it is worth);
    ``dropped`` removes perturbers from the field, which is how :func:`perturbation_budget` measures what each one does.
    """

    a2_au_d2: float = 0.0
    area_to_mass_m2_kg: float = 0.0
    reflectivity: float = 0.0
    dropped: tuple[str, ...] = field(default_factory=tuple)
    rtol: float = 1e-11

    @property
    def srp(self):
        """``(area_to_mass, reflectivity)`` for the propagator, or ``None`` when the pressure is off."""
        if self.area_to_mass_m2_kg <= 0.0:
            return None
        return (self.area_to_mass_m2_kg, max(self.reflectivity, 0.0))

    def kwargs(self, perturbers: Sequence[str] | None = None) -> dict:
        """What :class:`~pinneapple_physics.astro.smallbodies.Trajectory` needs to fly this set of terms."""
        keep = [p for p in (perturbers or PERTURBERS) if p not in self.dropped]
        return {"perturbers": keep, "a2_au_d2": self.a2_au_d2, "srp": self.srp, "rtol": self.rtol}


def _trajectory(body: SmallBody, jd0: float, jd1: float, terms: Terms | None = None, eph=None, **kw):
    from .ephemeris import default_ephemeris
    from .smallbodies import Trajectory

    kw = {**((terms or Terms()).kwargs()), **kw}
    return Trajectory(body, float(jd0), float(jd1), eph or default_ephemeris(), **kw)


def delta_v(m_impactor_kg: float, speed_km_s: float, m_target_kg: float, beta: float = 1.0) -> float:
    """Along-track velocity change (m/s) delivered by a kinetic impactor: ``beta * m_i * v / M``.

    ``beta = 1`` is pure momentum transfer; the ejecta of a real impact carry more, which is what DART measured.
    """
    if m_impactor_kg <= 0 or m_target_kg <= 0:
        raise ValueError("the impactor and the target need a positive mass")
    if speed_km_s <= 0:
        raise ValueError("the impactor needs a positive speed")
    if beta <= 0:
        raise ValueError("beta is at least 1 (the impactor's own momentum); below that the ejecta would push back")
    return beta * m_impactor_kg * speed_km_s * 1000.0 / m_target_kg


def period_change_fraction(delta_v_m_s: float, v_orbital_m_s: float) -> float:
    """``dP/P`` for an along-track impulse on a circular orbit: the radius is unchanged at the impulse, so
    ``da/a = 2 dv/v`` (vis-viva) and ``dP/P = 3 dv/v`` (Kepler's third law)."""
    if v_orbital_m_s <= 0:
        raise ValueError("the orbital speed must be positive")
    return 3.0 * delta_v_m_s / v_orbital_m_s


def dart_beta(period_change_s: float | None = None, *, impactor_kg: float | None = None,
              speed_km_s: float | None = None, target_kg: float | None = None,
              period_before_s: float | None = None) -> float:
    """The momentum enhancement the published period change implies -- the model held to the mission that flew it.

    Dimorphos orbits Didymos in almost a circle at ``v = 0.174 m/s``; the measured 32-minute shortening is
    ``dP/P = -0.0448``, so ``dv = (v/3) dP/P = -2.6 mm/s``. Dividing by the momentum the 570 kg impactor carried
    gives ``beta ~ 3.6``.
    """
    d = dict(DART)
    if period_change_s is not None:
        d["period_change_s"] = period_change_s
    if impactor_kg is not None:
        d["impactor_kg"] = impactor_kg
    if speed_km_s is not None:
        d["speed_km_s"] = speed_km_s
    if target_kg is not None:
        d["target_kg"] = target_kg
    if period_before_s is not None:
        d["period_before_s"] = period_before_s
    v = _circular_speed_dart()
    dv = (v / 3.0) * (d["period_change_s"] / d["period_before_s"])
    return abs(dv) / (d["impactor_kg"] * d["speed_km_s"] * 1000.0 / d["target_kg"])


def _circular_speed_dart(separation_m: float = 1190.0) -> float:
    """Orbital speed of Dimorphos about Didymos: ``sqrt(G M_didymos / a)`` with Didymos at 5.6e11 kg and a = 1.19 km.

    The period is the check: ``2 pi a / v`` has to come out at 11.92 h, which it does to 2 %.
    """
    return math.sqrt(6.674e-11 * 5.6e11 / separation_m)


def dimorphos_period_hours(separation_m: float = 1190.0) -> float:
    """The period the orbital speed implies, in hours -- the consistency check of :func:`_circular_speed_dart`."""
    return 2.0 * math.pi * separation_m / _circular_speed_dart(separation_m) / 3600.0


def yarkovsky_drift(a2_au_d2: float, a_au: float, e: float = 0.0) -> float:
    """Semi-major-axis drift from the Yarkovsky effect, au/day, from JPL's ``A2`` parameter.

    ``A2`` is the transverse acceleration in au/day^2 at 1 au, ``a_t = A2 (1 au / r)^2``. Gauss' equation for the
    transverse component is ``da/dt = 2 a_t / (n sqrt(1 - e^2))`` with ``n`` the mean motion: for Apophis
    (``a = 0.922 au``, ``e = 0.191``, ``A2 = -2.9e-14 au/day^2``) this is ``da/dt = -3.0e-12 au/day``, i.e.
    **-166 m a year**, against the ~170 m/yr published drift. The sign is the asteroid's: a prograde rotator (positive
    ``A2``) spirals out.
    """
    n = math.sqrt(GM_SUN_AU3_DAY2 / a_au ** 3)              # rad/day
    return 2.0 * a2_au_d2 / (n * math.sqrt(max(1.0 - e * e, 1e-12)))


def area_to_mass_ratio(diameter_km: float, mass_kg: float) -> float:
    """Cross-section over mass of a sphere, m^2/kg: ``(pi D^2 / 4) / M``.

    The number solar radiation pressure is multiplied by. For Apophis (0.325 km across, the bundled Herschel diameter;
    5.4e10 kg) it is **1.54e-6 m^2/kg** -- the same one a spacecraft of a few tonnes has, which is why the pressure
    that is negligible for a spacecraft hull is not obviously negligible for a rock.
    """
    if diameter_km <= 0 or mass_kg <= 0:
        raise ValueError("the diameter and the mass need to be positive")
    return math.pi * (diameter_km * 1000.0) ** 2 / 4.0 / mass_kg


def solar_radiation_pressure(area_to_mass_m2_kg: float, reflectivity: float = 0.0, r_au: float = 1.0) -> float:
    """Radial acceleration the sunlight gives the body, m/s^2: ``P (A/M) (1 + reflectivity) / r_au^2``.

    ``P = 4.56e-6 N/m^2`` at 1 au (the solar constant, 1361 W/m^2, over c), so a body at 1 au with 1 m^2/kg takes
    4.56e-6 m/s^2 before the reflecting term. ``reflectivity`` is the fraction reflected specularly (0 for a black
    body, 1 for a perfect mirror); a bare rock is nearer 0.1 and this module's default is 0, the conservative one.
    """
    if area_to_mass_m2_kg < 0 or reflectivity < 0:
        raise ValueError("the area-to-mass ratio and the reflectivity cannot be negative")
    if r_au <= 0:
        raise ValueError("the distance has to be positive")
    return P_SRP * area_to_mass_m2_kg * (1.0 + reflectivity) / r_au ** 2


def impulse_body(body: SmallBody, jd: float, dv_km_s: Sequence[float], eph=None,
                 terms: Terms | None = None) -> SmallBody:
    """The same body, with ``dv_km_s`` added to its heliocentric velocity at ``jd`` and its elements rebuilt there.

    The result is a :class:`~pinneapple_physics.astro.smallbodies.SmallBody` whose epoch is ``jd``, so every function
    that already propagates one (``close_approaches``, ``propagate``) works on the deflected orbit unchanged.

    The state comes out of the propagator in ICRF axes and the elements are stored in the ecliptic one
    (``SmallBody.state_icrf`` rotates them back), so the state is rotated here before the elements are computed: skip
    that and the obliquity is applied twice, which moves a flyby by millions of kilometres for a nudge of micrometres
    per second.

    ``terms`` is the trajectory the impulse is applied on: the offensive body arrives at ``jd`` with its own Yarkovsky
    and radiation-pressure history, and the deflected one inherits the state it had.
    """
    from .ephemeris import default_ephemeris
    from .smallbodies import ECL_TO_ICRF

    eph = eph or default_ephemeris()
    tr = _trajectory(body, jd, jd, terms, eph)
    r, v = tr.heliocentric(np.array([float(jd)]))
    v = v[0] + np.asarray(dv_km_s, float)
    r_ecl, v_ecl = ECL_TO_ICRF.T @ r[0], ECL_TO_ICRF.T @ v
    el = elements_from_state(r_ecl, v_ecl, GM_SUN)
    return SmallBody(body.name, float(jd), el["a_km"] / AU_KM, el["e"], el["i_deg"], el["Omega_deg"],
                     el["omega_deg"], el["M_deg"], body.kind, body.neo, body.pha, body.H, body.diameter_km,
                     body.ca_data)


def deflect(body: SmallBody, jd_impact: float, delta_v_m_s: float, *, prograde: bool = True, eph=None,
            terms: Terms | None = None) -> SmallBody:
    """Apply ``delta_v_m_s`` along the heliocentric velocity (retrograde to slow the body down) at ``jd_impact``."""
    from .ephemeris import default_ephemeris

    eph = eph or default_ephemeris()
    tr = _trajectory(body, jd_impact, jd_impact, terms, eph)
    _, v = tr.heliocentric(np.array([float(jd_impact)]))
    vhat = v[0] / np.linalg.norm(v[0])
    dv = (1.0 if prograde else -1.0) * delta_v_m_s * 1e-3 * vhat      # m/s -> km/s
    return impulse_body(body, jd_impact, dv, eph, terms)


def miss_distance(body: SmallBody, jd0: float, jd1: float, *, target: str = "earth", step_days: float = 2.0,
                  max_dist_au: float = 0.3, eph=None, terms: Terms | None = None, traj=None) -> dict:
    """The closest approach of ``body`` to ``target`` over ``[jd0, jd1]``: the minimum of
    :func:`~pinneapple_physics.astro.smallbodies.close_approaches`, or a dict saying there was none.

    ``max_dist_au`` is deliberately generous by default: a deflection that works moves the body *out* of the
    encounter, and a search capped at a small distance would report "no approach" for exactly the cases that matter.
    ``terms`` flies the small accelerations; ``traj`` reuses an integration the caller already has (the ensemble does:
    one trajectory per member feeds both the sampled arc and the closest approach).
    """
    from .smallbodies import close_approaches

    if traj is None and terms is not None:
        traj = _trajectory(body, jd0, jd1, terms, eph)
    ca = close_approaches(body, float(jd0), float(jd1), target=target, max_dist_au=max_dist_au,
                          step_days=step_days, eph=eph, traj=traj)
    if not ca:
        return {"jd_tdb": float("nan"), "dist_km": float("inf"), "dist_au": float("inf"), "v_rel_km_s": float("nan"),
                "found": False}
    best = min(ca, key=lambda c: c["dist_km"])
    return dict(best, found=True)


def deflection_table(body: SmallBody, jd_impact: float, jd_encounter: float, *, masses_kg: Iterable[float],
                     speed_km_s: float, target_kg: float, beta: float = BETA_DART, window_days: float = 120.0,
                     eph=None, terms: Terms | None = None) -> list[dict]:
    """Miss distance at ``jd_encounter`` for each impactor mass: the number the plot is drawn from.

    ``delta_v`` is applied along the orbit at ``jd_impact`` (the warning time), and the encounter is measured in a
    window of ``window_days`` around ``jd_encounter`` -- the encounter date itself moves when the orbit changes, so a
    fixed window is what keeps the comparison honest.
    """
    out = []
    for m in masses_kg:
        dv = delta_v(m, speed_km_s, target_kg, beta)
        nudged = deflect(body, jd_impact, dv, eph=eph, terms=terms)
        ca = miss_distance(nudged, jd_encounter - window_days, jd_encounter + window_days, eph=eph, terms=terms)
        out.append({"impactor_kg": float(m), "delta_v_m_s": dv, "warning_days": float(jd_encounter - jd_impact),
                    **{k: ca[k] for k in ("jd_tdb", "dist_km", "dist_au", "v_rel_km_s", "found")}})
    return out


def sample_ensemble(*, n: int = 32, a2_au_d2: float = 0.0, a2_sigma: float = 0.0, beta: float = BETA_DART,
                    beta_sigma: float = 0.0, area_to_mass_m2_kg: float = 0.0, am_sigma: float = 0.0,
                    seed: int = 0) -> list[dict]:
    """Draws of the three parameters the body is least known by: the Yarkovsky ``A2``, ``beta`` and the area-to-mass
    ratio that sets solar radiation pressure.

    A normal draw, seeded, so the ensemble is reproducible; the returned records are what
    :func:`success_probability`, :func:`trust_horizon` and :func:`ensemble_arcs` consume. ``am_sigma`` is relative
    (``0.5`` is 50 % on the pressure), which is the honest hand one can put on a rock's cross-section and reflectivity.
    """
    rng = np.random.default_rng(seed)
    a2 = rng.normal(a2_au_d2, a2_sigma, n) if a2_sigma > 0 else np.full(n, float(a2_au_d2))
    be = rng.normal(beta, beta_sigma, n) if beta_sigma > 0 else np.full(n, float(beta))
    am = rng.normal(area_to_mass_m2_kg, area_to_mass_m2_kg * am_sigma, n) if am_sigma > 0 else np.full(
        n, float(area_to_mass_m2_kg))
    return [{"a2_au_d2": float(a), "beta": float(b), "area_to_mass_m2_kg": float(max(c, 0.0))}
            for a, b, c in zip(a2, be, am, strict=True)]


def ensemble_arcs(body: SmallBody, jd_impact: float, *, jd_encounter: float, mass_kg: float, speed_km_s: float,
                  target_kg: float, draws: Sequence[dict], jd_out: Sequence[float], window_days: float = 30.0,
                  step_days: float = 0.5, reflectivity: float = 0.0, terms: Terms | None = None, eph=None) -> dict:
    """Fly the whole ensemble and keep the trajectories: the cloud, sampled, not a drawing of one.

    Each draw of ``beta`` gets its own impulse (:func:`delta_v`) on its own ``A2`` and area-to-mass history
    (:class:`Terms`), and one integration from the impulse to just past the encounter -- sampled at ``jd_out`` and
    searched for the closest approach with the same trajectory, so the arc the animation shows and the miss distance it
    reports come from the same numbers. The undeflected body is flown and sampled the same way and is the reference the
    cloud is measured against; ``terms`` is what that reference carries, which should be the centre of the draws
    (a reference flown with nothing while the members carry the Yarkovsky term would offset the whole cloud by it).

    Returns ``{"jd", "dist_earth_km", "nominal", "members"}``; ``dist_earth_km`` is the geocentric distance of the
    undeflected body at each ``jd_out`` and each member carries its own ``r_helio_km`` at the same dates.
    """
    from .ephemeris import default_ephemeris
    from .smallbodies import close_approaches

    eph = eph or default_ephemeris()
    jd = np.asarray(list(jd_out), float)
    jd1 = float(jd_encounter) + window_days
    jd0 = float(jd_impact)

    base_tr = _trajectory(body, jd0, jd1, terms, eph)
    r_base, _ = base_tr.heliocentric(jd)
    earth, _ = eph.relative("earth", "sun", jd)
    ca = close_approaches(body, float(jd_encounter) - window_days, jd1, max_dist_au=0.3, step_days=step_days, eph=eph,
                          traj=base_tr)
    base = min(ca, key=lambda c: c["dist_km"]) if ca else {"jd_tdb": float("nan"), "dist_km": float("inf")}

    members = []
    for d in draws:
        terms = Terms(a2_au_d2=d.get("a2_au_d2", 0.0), area_to_mass_m2_kg=d.get("area_to_mass_m2_kg", 0.0),
                      reflectivity=reflectivity)
        dv = delta_v(mass_kg, speed_km_s, target_kg, d["beta"])
        nudged = deflect(body, jd_impact, dv, eph=eph, terms=terms)
        tr = _trajectory(nudged, jd0, jd1, terms, eph)
        r, _ = tr.heliocentric(jd)
        ca = close_approaches(nudged, float(jd_encounter) - window_days, jd1, max_dist_au=0.3, step_days=step_days,
                              eph=eph, traj=tr)
        best = min(ca, key=lambda c: c["dist_km"]) if ca else {"jd_tdb": float("nan"), "dist_km": float("inf")}
        members.append({"draw": dict(d), "delta_v_m_s": dv, "miss_km": float(best["dist_km"]),
                        "jd_tdb": float(best["jd_tdb"]),
                        "r_helio_km": [[float(x) for x in row] for row in r],
                        "dist_earth_km": [float(np.linalg.norm(row - e)) for row, e in
                                          zip(r, earth, strict=True)]})
    return {"jd": [float(x) for x in jd], "dist_earth_km": [float(np.linalg.norm(r - e)) for r, e in
                                                            zip(r_base, earth, strict=True)],
            "nominal": {"miss_km": float(base["dist_km"]), "jd_tdb": float(base["jd_tdb"]),
                        "r_helio_km": [[float(x) for x in row] for row in r_base]},
            "members": members}


def success_probability(miss_km: Sequence[float], earth_radius_km: float = 6371.0,
                        capture_km: float = 0.0) -> float:
    """Fraction of the ensemble further from the Earth than ``earth_radius_km + capture_km``."""
    d = np.asarray(list(miss_km), float)
    if d.size == 0:
        raise ValueError("the ensemble is empty")
    return float(np.mean(d > earth_radius_km + capture_km))


def trust_horizon(*, warning_days: Sequence[float], spread_km: Sequence[float],
                  earth_radius_km: float = 6371.0) -> float:
    """The longest warning time for which the ensemble spread is still smaller than the Earth.

    ``spread_km`` is the standard deviation of the miss distance (or of the along-track position) at each warning
    time; the horizon is the last ``warning_days`` where it stays under the Earth's radius -- the age after which a
    "we will miss" cannot be guaranteed, whatever the model.

    When the spread never reaches the Earth over the range given, the answer is the **last date in the range**, i.e.
    "at least this far": the caller asked a question the range cannot answer, and the number says so instead of
    inventing a horizon past the data.
    """
    w = np.asarray(list(warning_days), float)
    s = np.asarray(list(spread_km), float)
    if w.shape != s.shape or w.size == 0:
        raise ValueError("warning_days and spread_km need one value each, at least one")
    order = np.argsort(w)
    w, s = w[order], s[order]
    inside = s < earth_radius_km
    if not inside.any():
        return 0.0
    last = int(np.nonzero(inside)[0].max())
    if last == w.size - 1:
        return float(w[last])
    # linear interpolation between the last inside point and the first outside one
    w0, s0, w1, s1 = w[last], s[last], w[last + 1], s[last + 1]
    if s1 == s0:
        return float(w1)
    return float(w0 + (earth_radius_km - s0) * (w1 - w0) / (s1 - s0))


def impact_probability(miss_km: float, sigma_km: float, earth_radius_km: float = 6371.0,
                       capture_km: float = 0.0) -> float:
    """Probability that the true miss is inside the capture radius, for a nominal ``miss_km`` and a 1-sigma.

    ``P = P(|miss + error| < R)`` with a Gaussian error: exactly ``erfc((miss - R)/sigma sqrt 2) / 2`` minus the far
    side term, which is the 1-D reading of the b-plane. It is the right reading whenever ``sigma`` is much smaller than
    the capture radius -- the planet looks like a wall from the cloud's point of view -- which is the planetary-defence
    regime (JPL's own 1-sigma on the 2029 miss is 232 km against a 6 371 km radius). For a cloud wider than the planet
    the 2-D geometry of the capture cross-section is what matters and this is not the formula.
    """
    if sigma_km < 0:
        raise ValueError("the sigma cannot be negative")
    r = earth_radius_km + capture_km
    if sigma_km == 0.0:
        return 1.0 if abs(miss_km) < r else 0.0
    k = sigma_km * math.sqrt(2.0)
    return 0.5 * (math.erfc((miss_km - r) / k) - math.erfc((miss_km + r) / k))


def jpl_miss_sigma_km(body: SmallBody | None = None, year: int = 2029, target: str = "Earth") -> dict:
    """The 1-sigma uncertainty of the encounter distance, from the bundled JPL close-approach record.

    JPL publishes the close-approach distance with a 3-sigma range (``dist_min``/``dist_max``) and the b-plane error
    ellipse; the half-range over three is the 1-sigma in the distance direction, which is the number an impact
    probability needs. For the 2029-04-13 Apophis pass the record (solution 199) gives 0.000252172816 au with
    0.000247515-0.000256837 au, i.e. **37 724 km +/- 232 km (1 sigma)**, and a b-plane ellipse of 243.97 x 114.74 km at
    11.16 degrees -- the orbit is known far better than the deflection is worth, and both are in the same table.
    """
    from .smallbodies import cad_reference, load_body

    ref = cad_reference(body if body is not None else load_body("apophis"), year)
    if ref is None:
        raise ValueError(f"the bundled record carries no {target} close approach in {year}")
    sigma = (ref["dist_max_au"] - ref["dist_min_au"]) / 6.0 * AU_KM
    return {"target": target, "cd": ref["cd"], "dist_au": ref["dist_au"], "dist_km": ref["dist_au"] * AU_KM,
            "dist_min_km": ref["dist_min_au"] * AU_KM, "dist_max_km": ref["dist_max_au"] * AU_KM,
            "sigma_km": sigma, "sigma_3_km": 3.0 * sigma, "v_rel_km_s": ref["v_rel_km_s"],
            "sigma_t_s": ref["sigma_t_min"]}


def perturbation_budget(body: SmallBody, jd_encounter: float, *, a2_au_d2: float = 0.0,
                        area_to_mass_m2_kg: float = 0.0, reflectivity: float = 0.0,
                        planets: Sequence[str] = ("mercury", "venus", "earth", "moon", "mars", "jupiter",
                                                  "saturn", "uranus", "neptune"),
                        srp_scales: Sequence[float] = (1.0, 10.0), window_days: float = 30.0, step_days: float = 0.5,
                        tolerance_km: float = 1.0, floor_rtol: float = 1e-12, eph=None) -> list[dict]:
    """What each term in the field is worth at ``jd_encounter``: the miss distance with it taken away.

    One long integration per row -- an asteroid's arc is not reusable, which is why this is the expensive function in
    the module and why it reports the time it took. Every row carries two numbers: ``delta_km``, the change in the miss
    distance, and ``dr_km``, the distance the body's *position* at the encounter moved. The second is the sensitivity
    that does not depend on the encounter geometry; the first is the one the decision is made on.

    ``matters`` is ``|delta_km| > tolerance_km``, and ``tolerance_km`` should be read against the last row: the same
    computation at a tighter integrator tolerance, which is the numerical floor of the whole table. A term smaller than
    that floor does not matter *in this arc* and saying so with the number is the point of the function.
    """
    from .ephemeris import default_ephemeris

    eph = eph or default_ephemeris()
    jd0 = float(jd_encounter) - window_days
    jd1 = float(jd_encounter) + window_days
    base_terms = Terms()
    base_tr = _trajectory(body, jd0, jd1, base_terms, eph)
    base = miss_distance(body, jd0, jd1, step_days=step_days, eph=eph, traj=base_tr)
    r_base, _ = base_tr.heliocentric(np.array([jd_encounter]))
    am = area_to_mass_m2_kg
    variants: list[tuple[str, Terms, dict]] = [
        ("baseline: the field the module ships (Sun, 9 planets, Moon, 1PN)", base_terms, {}),
    ]
    for p in planets:
        variants.append((f"without {p}: its pull taken out of the field", Terms(dropped=(p,)), {}))
    for k in srp_scales:
        variants.append((
            f"solar radiation pressure at {k:g}x (A/M = {am * k:.3g} m^2/kg, "
            f"{solar_radiation_pressure(am * k, reflectivity) * 1e6:.3g} um/s^2 at 1 au)",
            Terms(area_to_mass_m2_kg=am * k, reflectivity=reflectivity), {}))
    if a2_au_d2:
        variants.append((f"Yarkovsky, A2 = {a2_au_d2:.3g} au/day^2 at 1 au", Terms(a2_au_d2=a2_au_d2), {}))
    variants.append((f"the numerical floor: the same run at rtol {floor_rtol:g} instead of {Terms().rtol:g}",
                     Terms(rtol=floor_rtol), {}))

    rows = []
    for label, terms, kw in variants:
        t0 = time.time()
        if terms == base_terms and not kw:
            tr, ca = base_tr, base
        else:
            tr = _trajectory(body, jd0, jd1, terms, eph)
            ca = miss_distance(body, jd0, jd1, step_days=step_days, eph=eph, traj=tr)
        r, _ = tr.heliocentric(np.array([jd_encounter]))
        delta = ca["dist_km"] - base["dist_km"]
        rows.append({"term": label, "dist_km": ca["dist_km"], "delta_km": delta,
                     "dr_km": float(np.linalg.norm(r[0] - r_base[0])), "found": ca["found"],
                     "matters": bool(abs(delta) > tolerance_km), "seconds": time.time() - t0})
    return rows
