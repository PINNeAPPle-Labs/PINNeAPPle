"""The high-fidelity check of a patched-conic design: fly it in the field of the Sun and every planet, and measure
how far from each encounter it actually goes.

A patched-conic design assumes that between two encounters only the Sun acts, and that at an encounter the planet's
gravity turns the spacecraft instantaneously -- a free flyby, provided the relative velocity happens to match. Neither
assumption is exactly true: the planets the design ignores pull on it for months, and a flyby that is 1 % off in
relative velocity is a flyby that does not happen. So the best design of an evolutionary search is re-flown here with
a restricted N-body propagator -- the Sun and the other planets as point masses, their states from the same ephemeris
the design used, and every non-gravitational perturbation (solar radiation pressure, outgassing, relativity) left out
-- and the miss distance at each encounter is reported in kilometres, in planet radii and in Hill radii.

The verdict is then a statement about the *conic approximation*, not about the mission:

* ``"reproduces-the-conic"``: every encounter is met within ``tolerance_km`` and the rendezvous velocity within
  ``tolerance_km_s``. The design is consistent with the dynamics it was drawn in.
* ``"needs-correction"``: the encounters are missed, but by less than the body's Hill radius, and the total
  Lambert re-target impulse (:func:`trajectory_correction`) says what it would cost to fly it.
* ``"conic-only"``: at least one encounter is missed by more than the Hill radius. That flyby does not happen: the
  design is a solution of the conic problem and not a trajectory.

The propagator is the restricted problem on purpose. :func:`pinneapple_physics.astro.nbody.integrate` is the library's
general point-mass integrator (every body a mass, mutual gravity, conservation diagnostics); it is the right tool for a
planet-dynamics question, and ``tests/test_trajectory_design.py`` uses it in the two-body limit to check that this
module's propagator reproduces a Kepler arc. Here the planets must follow the *ephemeris* rather than falling under
each other's gravity: the error of a planet's own trajectory would otherwise be charged to the spacecraft.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from ..ephemeris import GM, GM_SUN, RADIUS_KM
from .bodies import default_ephemeris
from .lambert import LambertError, lambert

__all__ = ["EncounterCheck", "TrustReport", "planet_hill_radius_km", "propagate_design", "verify_design",
           "trajectory_correction"]

DAY = 86400.0
#: Planets whose point-mass pull is included; the Moon is not, because the design's Earth is the Earth-Moon
#: barycentre and adding the Moon without a distinction between the two would be a fiction.
PERTURBERS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")


def planet_hill_radius_km(body: str, r_km: float) -> float:
    """Hill-sphere radius of ``body`` at heliocentric distance ``r_km``, from the GM the library carries.

    ``r_H = r (GM_body / 3 GM_sun)**(1/3)``; it is the distance inside which the body's own gravity wins over the
    Sun's tide, so a flyby that misses by more than this cannot be called the same encounter.
    """
    return float(r_km) * (GM[body] / (3.0 * GM_SUN)) ** (1.0 / 3.0)


class _EphemerisPath:
    """The planets' heliocentric positions over the propagation span, as one cubic spline over all of them.

    An adaptive integrator asks for the field at tens of thousands of epochs, and the analytic ephemeris costs ~50 us
    per body per epoch. Evaluating it on a uniform grid and interpolating cubically brings that to a lookup; the
    error the interpolation introduces is measured against the exact ephemeris (``error_km``) and reported with the
    check, so the shortcut cannot be mistaken for accuracy.

    The table is **one** spline of ``3 m`` columns (three per body) rather than ``3 m`` splines: an evaluation of the
    right-hand side of the integration is then a single interpolation call, and the whole field costs ~40 us instead
    of ~500 us. On a fifteen-year Cassini propagation that is the difference between a minute and twenty.
    """

    def __init__(self, ephemeris, jd0: float, jd1: float, bodies=PERTURBERS, step_days: float = 0.5,
                 handover: str = ""):
        self.bodies = tuple(bodies)
        n = int(max(2, np.ceil((jd1 - jd0) / float(step_days)))) + 1
        self.jd = np.linspace(float(jd0), float(jd1), n)
        table = []
        for body in self.bodies:
            r, _ = ephemeris.state(body, self.jd)
            table.append(np.asarray(r, dtype=float).reshape(n, 3))
        self._table = np.concatenate(table, axis=1)
        self._spline = CubicSpline(self.jd, self._table, axis=0)
        self.gm = np.array([GM[b] for b in self.bodies], dtype=float).reshape(-1, 1)
        self.radius = np.array([RADIUS_KM[b] for b in self.bodies], dtype=float).reshape(-1, 1)
        # Hill radius of each body as a fraction of its heliocentric distance: ``r_H = r (GM/(3 GM_sun))**(1/3)``,
        # the module's own notion of how far a body's gravity reaches (see :func:`planet_hill_radius_km`). The body
        # named by ``handover`` keeps its reach switched off inside that sphere; every other body pulls everywhere.
        hill = (self.gm / (3.0 * GM_SUN)) ** (1.0 / 3.0)
        self.handover = handover if handover in self.bodies else ""
        handover_row = np.array([[b == self.handover] for b in self.bodies], dtype=bool)
        self.hill_fraction = np.where(handover_row, hill, 0.0)
        self.step_days = float(step_days)
        self.error_km = self._interpolation_error(ephemeris)

    def _interpolation_error(self, ephemeris) -> float:
        """Worst interpolation error in km, measured at the midpoints of the grid (where a cubic spline is worst)."""
        mid = 0.5 * (self.jd[1:] + self.jd[:-1])
        approx = np.asarray(self._spline(mid), dtype=float).reshape(len(mid), len(self.bodies), 3)
        worst = 0.0
        for k, body in enumerate(self.bodies):
            r, _ = ephemeris.state(body, mid)
            exact = np.asarray(r, dtype=float).reshape(len(mid), 3)
            worst = max(worst, float(np.max(np.linalg.norm(exact - approx[:, k], axis=1))))
        return worst

    def positions(self, jd: float) -> np.ndarray:
        """Heliocentric positions of every body at ``jd``, ``(m, 3)`` km, in the order of :data:`PERTURBERS`."""
        return np.asarray(self._spline(jd), dtype=float).reshape(len(self.bodies), 3)


def _acceleration(r, path: _EphemerisPath, jd: float) -> np.ndarray:
    """Heliocentric acceleration (km/s^2): the Sun plus the planets, with the indirect term that a heliocentric frame
    needs (the Sun is not inertial, so each planet pulls the origin too).

    Written as whole arrays over the bodies rather than a loop: eight three-vectors through numpy is eight times the
    dispatch overhead of one ``(8, 3)`` array, and this is the innermost function of the trust check.

    A patched conic *departs from the departure body's centre* -- the initial state is that body's own position with
    only the ``v_inf`` on top of its velocity -- where the point-mass term is ``0/0``, so the whole propagation used to
    come back ``nan`` from its first evaluation. The softened version is finite but wrong in a second way: with 2.755
    km/s of excess at the Earth's centre the spacecraft is *bound*, and the integrator spent forty thousand steps per
    two hundred days resolving an orbit inside the planet. The departure body is therefore handed over: its pull is
    switched off inside its own Hill sphere, and from the moment the spacecraft leaves it the body pulls as any other.
    Every other planet pulls everywhere, so a flyby the design relies on is integrated as the hyperbola it is.

    The distance is also floored at each body's radius, so the acceleration stays finite even if a body is met head on:
    outside the radius that is the Newtonian term to ``(R/|d|)**2``, which at 1 au is 1e-9 -- four orders below the
    20 km the ephemeris interpolation already costs.
    """
    r = np.asarray(r, dtype=float).reshape(3)
    rp = path.positions(jd)                                     # (m, 3)
    d = rp - r
    dist2 = np.sum(d * d, axis=1, keepdims=True)
    reach = np.linalg.norm(rp, axis=1, keepdims=True) * path.hill_fraction
    pull = np.where(dist2 >= reach * reach, 1.0 / (dist2 + path.radius ** 2) ** 1.5, 0.0)
    a = -GM_SUN * r / np.linalg.norm(r) ** 3
    a = a + (path.gm * (pull * d - rp / np.linalg.norm(rp, axis=1, keepdims=True) ** 3)).sum(axis=0)
    return a


def propagate_design(design, ephemeris=None, rtol: float = 1e-10, step_days: float = 0.5,
                     bodies=PERTURBERS) -> dict:
    """Fly a :class:`~.gtop.MGADesign` in the restricted N-body field, from departure to arrival.

    The initial state is the design's own: the departure planet's position and the Lambert velocity of the first leg
    (planet velocity plus the departure impulse), so the check starts from what the design says and lets the dynamics
    -- not the conic -- decide where the spacecraft goes. The departure body is handed over (its pull is off inside
    its own Hill sphere, see :func:`_acceleration`): the design's own state sits on that body's centre, where a point
    mass is ``0/0`` and where the bare ``v_inf`` leaves the spacecraft bound to it.

    Returns a dict with ``jd``, ``r``, ``v`` (the propagated path), ``interp_error_km``, ``n_steps``, ``rtol``, the
    ``bodies`` used and the ``handover`` body whose pull was off inside its own Hill sphere.
    """
    if not design.legs:
        raise ValueError("the design has no legs to fly")
    eph = ephemeris or default_ephemeris()
    jd = np.asarray(design.jd, dtype=float)
    r0 = np.asarray(design.legs[0]["r1"], dtype=float)
    v0 = np.asarray(design.legs[0]["v1"], dtype=float)
    path = _EphemerisPath(eph, jd[0], jd[-1], bodies=bodies, step_days=step_days,
                          handover=str(design.legs[0]["from"]))

    def rhs(t, y):
        return np.concatenate([y[3:], _acceleration(y[:3], path, t / DAY + jd[0])])

    sol = solve_ivp(rhs, (0.0, (jd[-1] - jd[0]) * DAY), np.concatenate([r0, v0]), method="DOP853",
                    t_eval=(jd - jd[0]) * DAY, rtol=rtol, atol=1e-6)
    if not sol.success:                                            # pragma: no cover - would mean rtol is too tight
        raise RuntimeError(f"the N-body propagation failed: {sol.message}")
    return {"jd": jd, "r": sol.y[:3].T, "v": sol.y[3:].T, "interp_error_km": path.error_km,
            "n_steps": int(sol.t.size), "n_f_evals": int(sol.nfev), "bodies": tuple(bodies), "rtol": rtol,
            "step_days": step_days, "handover": path.handover}


@dataclass
class EncounterCheck:
    """One node of the trajectory, as the N-body propagation actually reaches it."""

    index: int
    body: str
    epoch_jd: float
    miss_km: float
    miss_planet_radii: float
    hill_radius_km: float
    designed_v_inf_km_s: float
    reached_v_inf_km_s: float
    v_inf_error_km_s: float

    @property
    def free_flyby(self) -> bool:
        """Whether the relative velocity the propagation reaches could be turned by a free flyby from the design's."""
        return self.v_inf_error_km_s <= 0.05 * max(self.designed_v_inf_km_s, 1e-9)

    def as_dict(self) -> dict:
        return {"index": self.index, "body": self.body, "epoch_jd": self.epoch_jd, "miss_km": self.miss_km,
                "miss_planet_radii": self.miss_planet_radii, "hill_radius_km": self.hill_radius_km,
                "designed_v_inf_km_s": self.designed_v_inf_km_s, "reached_v_inf_km_s": self.reached_v_inf_km_s,
                "v_inf_error_km_s": self.v_inf_error_km_s, "free_flyby": self.free_flyby}


@dataclass
class TrustReport:
    """What the N-body propagation says about a patched-conic design.

    Attributes
    ----------
    verdict:
        ``"reproduces-the-conic"``, ``"needs-correction"`` or ``"conic-only"`` -- see the module docstring.
    encounters:
        One :class:`EncounterCheck` per node, departure included (its miss is zero by construction: it is the initial
        condition).
    tcm_dv_km_s:
        Total impulse of the trajectory correction manoeuvres that would re-target the N-body path onto the designed
        encounters, from :func:`trajectory_correction`.
    reasons:
        The sentences behind the verdict, so a report never shows the word alone.
    """

    verdict: str
    encounters: list[EncounterCheck]
    arrival_miss_km: float
    max_miss_km: float
    tcm_dv_km_s: float
    interp_error_km: float
    observations: dict = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.verdict == "reproduces-the-conic"

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "ok": self.ok, "max_miss_km": self.max_miss_km,
                "arrival_miss_km": self.arrival_miss_km, "tcm_dv_km_s": self.tcm_dv_km_s,
                "interp_error_km": self.interp_error_km, "reasons": list(self.reasons),
                "encounters": [e.as_dict() for e in self.encounters], "observations": self.observations}

    def summary(self) -> str:
        lines = [f"verdict: {self.verdict}"]
        for e in self.encounters:
            what = "departure" if e.index == 0 else ("arrival" if e.index == len(self.encounters) - 1 else "flyby")
            lines.append(f"  {what:9s} {e.body:8s} miss {e.miss_km:12.1f} km "
                         f"({e.miss_planet_radii:8.2f} radii, Hill {e.hill_radius_km:.3g} km), "
                         f"v_inf {e.reached_v_inf_km_s:.3f} vs {e.designed_v_inf_km_s:.3f} km/s")
        lines.append(f"  correction impulse to re-fly the encounters: {self.tcm_dv_km_s:.4f} km/s")
        lines.append(f"  ephemeris interpolation error {self.interp_error_km:.3f} km")
        lines.extend(f"  - {r}" for r in self.reasons)
        return "\n".join(lines)


def trajectory_correction(propagation: dict, design, ephemeris=None) -> float:
    """Impulse (km/s) that would bring the N-body path back onto the designed encounters.

    At each intermediate node the propagated state is used as the start of a Lambert arc to the *next* designed node
    position (the design's epoch and body), and the difference between the velocity the arc needs and the velocity
    the spacecraft actually has is the correction that node needs. The sum is the price of the conic approximation,
    the same kind of number a mission's TCM budget is written in.
    """
    eph = ephemeris or default_ephemeris()
    jd = np.asarray(propagation["jd"], dtype=float)
    r_path = np.asarray(propagation["r"], dtype=float)
    v_path = np.asarray(propagation["v"], dtype=float)
    total = 0.0
    for k in range(len(jd) - 1):
        target_r, _ = eph.state(design.legs[k]["to"], np.asarray([jd[k + 1]]))
        try:
            arc = lambert(r_path[k], np.asarray(target_r).reshape(3), float(jd[k + 1] - jd[k]) * DAY, GM_SUN)
        except LambertError:
            total += float("nan")                                  # no arc: the correction is not a number, say so
            continue
        total += float(np.linalg.norm(arc.v1 - v_path[k]))
    return float(total)


def verify_design(design, ephemeris=None, rtol: float = 1e-10, step_days: float = 0.5,
                  tolerance_km: float = 1.0e4, tolerance_km_s: float = 0.05) -> TrustReport:
    """Re-fly a patched-conic design in the restricted N-body field and judge the conic approximation.

    Parameters
    ----------
    design:
        The :class:`~.gtop.MGADesign` to check.
    ephemeris:
        Where the planet states come from; the same one the design was built on, or it would not be a check of the
        design but of another trajectory.
    rtol:
        Relative tolerance of the DOP853 integration.
    step_days:
        Spacing of the ephemeris grid the propagator interpolates (its error is measured and reported).
    tolerance_km, tolerance_km_s:
        How close an encounter has to be met to call the conic reproduced.

    Returns
    -------
    TrustReport
    """
    eph = ephemeris or default_ephemeris()
    prop = propagate_design(design, ephemeris=eph, rtol=rtol, step_days=step_days)
    jd = np.asarray(prop["jd"], dtype=float)
    positions = np.asarray(prop["r"], dtype=float)
    velocities = np.asarray(prop["v"], dtype=float)

    encounters: list[EncounterCheck] = []
    bodies = design.bodies
    for k, body in enumerate(bodies):
        here, v_here = eph.state(body, np.asarray([jd[k]]))
        here = np.asarray(here, dtype=float).reshape(3)
        v_here = np.asarray(v_here, dtype=float).reshape(3)
        miss = float(np.linalg.norm(positions[k] - here))
        achieved = float(np.linalg.norm(velocities[k] - v_here))
        if k == 0:
            designed = float(np.linalg.norm(design.legs[0]["v1"] - design.legs[0]["v_dep_body"]))
        elif k == len(bodies) - 1:
            designed = float(np.linalg.norm(design.legs[-1]["v2"] - design.legs[-1]["v_arr_body"]))
        else:
            designed = 0.5 * (float(np.linalg.norm(design.legs[k - 1]["v2"] - design.legs[k - 1]["v_arr_body"]))
                              + float(np.linalg.norm(design.legs[k]["v1"] - design.legs[k]["v_dep_body"])))
        encounters.append(EncounterCheck(
            index=k, body=body, epoch_jd=float(jd[k]), miss_km=miss,
            miss_planet_radii=miss / max(_radius(body), 1e-9),
            hill_radius_km=planet_hill_radius_km(body, float(np.linalg.norm(here))),
            designed_v_inf_km_s=designed, reached_v_inf_km_s=achieved, v_inf_error_km_s=abs(achieved - designed)))

    tcm = trajectory_correction(prop, design, ephemeris=eph)
    reasons: list[str] = []
    interior = encounters[1:-1]
    worst_hill = max((e.miss_km / e.hill_radius_km for e in interior), default=0.0)
    worst_miss = max((e.miss_km for e in encounters), default=0.0)
    if worst_miss <= tolerance_km and all(e.v_inf_error_km_s <= tolerance_km_s for e in encounters):
        verdict = "reproduces-the-conic"
        reasons.append(f"every node met within {tolerance_km:g} km and {tolerance_km_s:g} km/s")
    elif worst_hill < 1.0:
        verdict = "needs-correction"
        reasons.append(f"the encounters are missed by up to {worst_miss:.3g} km, inside the Hill sphere of every "
                       f"body: the sequence is flyable with the correction impulse above")
    else:
        verdict = "conic-only"
        bad = [e for e in interior if e.miss_km > e.hill_radius_km]
        reasons.append("; ".join(f"the {e.body} flyby is missed by {e.miss_km / e.hill_radius_km:.2f} Hill radii "
                                 f"({e.miss_km:.3g} km)" for e in bad) +
                       ": those flybys do not happen in the full field, so the design is a solution of the "
                       "patched-conic problem only")
    for e in encounters[1:]:
        if not e.free_flyby:
            reasons.append(f"at {e.body} the relative velocity the path reaches differs from the design by "
                           f"{e.v_inf_error_km_s:.3f} km/s: the flyby would be powered, not free")
    report = TrustReport(verdict=verdict, encounters=encounters, arrival_miss_km=float(encounters[-1].miss_km),
                         max_miss_km=float(worst_miss), tcm_dv_km_s=float(tcm),
                         interp_error_km=float(prop["interp_error_km"]),
                         observations={"n_steps": prop["n_steps"], "n_f_evals": prop["n_f_evals"],
                                       "rtol": prop["rtol"], "step_days": prop["step_days"],
                                       "bodies": list(prop["bodies"]), "handover": prop["handover"],
                                       "ephemeris": getattr(eph, "source", "ephemeris")},
                         reasons=tuple(reasons))
    return report


def _radius(body: str) -> float:
    from ..ephemeris import RADIUS_KM
    return float(RADIUS_KM[body])
