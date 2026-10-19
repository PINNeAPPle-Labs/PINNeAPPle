"""RESEARCH ONLY -- set-based encounter feasibility for relative motion (collision risk, rendezvous).

Inspired by the set-based view of PRISM (A. Enayati, "PRISM: Predictive Reachability-Intercept
Set Model for Relative Motion and Intercept Kinematics", Zenodo 22979350, CC BY 4.0): instead of
propagating two point states and checking a miss distance, each body is a *region* of possible
positions that grows with state uncertainty, velocity uncertainty, information latency and
bounded acceleration; two bodies can meet at time t only if their regions overlap.

Scope of this module (deliberately): the generic, dual-use kinematics for civilian questions --
"could these two objects come into contact within T?" (collision screening for satellites,
drones, vehicles) and "can a chaser reach a target region within T?" (rendezvous / docking
feasibility). It contains no targeting, no guidance law and no engagement planning, and none
will be added: those would turn it into weapon-system tooling. It is marked research only:
a ResearchOnlyWarning is emitted on use and PINNEAPPLE_COMMERCIAL_MODE=1 refuses it.

Model: a body with nominal position p0, velocity v0, position uncertainty radius r_p, velocity
uncertainty radius r_v, control latency tau and acceleration bound a_max (free space, no
gravity -- use relative coordinates, e.g. Clohessy-Wiltshire, when that matters) can be anywhere
in the ball
    B(t) = { p0 + v0 t + e :  |e| <= r_p + r_v t + 0.5 a_max max(t - tau, 0)^2 },
which is exact for the double integrator with |a| <= a_max applied after tau (the reachable set
of a bounded-acceleration point mass is a ball of radius a_max t^2 / 2 around the ballistic path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from pinneapple_neural._licencas import require_research_use

NOTICE = ("RESEARCH ONLY: set-based encounter feasibility (collision screening / rendezvous studies). "
          "Not validated for operational use; contains no guidance or targeting and must not be used to "
          "build weapon systems.")


@dataclass(frozen=True)
class BodyState:
    position: Sequence[float]
    velocity: Sequence[float]
    pos_uncertainty: float = 0.0
    vel_uncertainty: float = 0.0
    max_accel: float = 0.0
    latency: float = 0.0

    def center(self, t):
        return np.asarray(self.position, float)[None, :] + np.asarray(t, float)[:, None] * np.asarray(self.velocity, float)[None, :]

    def radius(self, t):
        t = np.asarray(t, float)
        return self.pos_uncertainty + self.vel_uncertainty * t + 0.5 * self.max_accel * np.maximum(t - self.latency, 0.0) ** 2


def encounter_feasibility(a: BodyState, b: BodyState, horizon: float, *, contact_distance: float = 0.0,
                          n: int = 2001, refine: bool = True) -> Dict[str, object]:
    """Margin(t) = |c_a - c_b| - R_a - R_b - contact_distance on [0, horizon].

    ``possible`` is True when the margin reaches 0: the two regions can overlap, i.e. contact
    cannot be ruled out (collision screening) / the encounter is reachable (rendezvous).
    """
    require_research_use("pinneapple_analysis.uncertainty.reachability", NOTICE)
    if not horizon > 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    for body in (a, b):
        if min(body.pos_uncertainty, body.vel_uncertainty, body.max_accel, body.latency) < 0:
            raise ValueError(f"uncertainties, max_accel and latency must be non-negative, got {body}")

    def margin_at(s):
        s = np.atleast_1d(np.asarray(s, float))
        return np.linalg.norm(a.center(s) - b.center(s), axis=1) - a.radius(s) - b.radius(s) - contact_distance

    # The centres move on straight lines, so their distance has one minimum, at the closest approach t_ca. A fast
    # flyby can overlap only between two grid points; sampling t_ca, and searching every interval whose lower bound
    # (closest distance in the interval minus the radii at its end, the radii never shrink) reaches 0, keeps that
    # contact from being missed.
    dp = np.asarray(a.position, float) - np.asarray(b.position, float)
    dv = np.asarray(a.velocity, float) - np.asarray(b.velocity, float)
    vv = float(dv @ dv)
    t_ca = float(np.clip(-(dp @ dv) / vv, 0.0, horizon)) if vv > 0 else 0.0
    t = np.union1d(np.linspace(0.0, horizon, n), [t_ca])
    margin = margin_at(t)
    lo_t, hi_t = t[:-1], t[1:]
    closest = np.linalg.norm(dp[None, :] + np.clip(t_ca, lo_t, hi_t)[:, None] * dv[None, :], axis=1)
    bound = closest - a.radius(hi_t) - b.radius(hi_t) - contact_distance
    extra = []
    for i in np.flatnonzero((bound <= 0) & (margin[:-1] > 0) & (margin[1:] > 0)):
        r = minimize_scalar(lambda s: float(margin_at(s)[0]), bounds=(lo_t[i], hi_t[i]), method="bounded",
                            options={"xatol": 1e-12 * max(horizon, 1.0)})
        if r.fun <= 0:
            extra.append(float(r.x))
    if extra:
        t = np.union1d(t, extra)
        margin = margin_at(t)
    k = int(np.argmax(margin <= 0)) if np.any(margin <= 0) else None
    t_first = None
    if k is not None:
        t_first = float(t[k])
        if refine and k > 0:
            lo, hi = t[k - 1], t[k]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if margin_at(mid)[0] <= 0 else (mid, hi)
            t_first = float(hi)
    return {"possible": k is not None, "t_first": t_first, "min_margin": float(margin.min()),
            "t_min_margin": float(t[int(np.argmin(margin))]), "times": t, "margin": margin}


def time_to_reach_region(chaser: BodyState, target_center: Sequence[float], target_radius: float = 0.0,
                         horizon: float = 1e6) -> Optional[float]:
    """Earliest time at which a (static) target region is inside the chaser's reachable ball."""
    tgt = BodyState(target_center, np.zeros(len(target_center)), pos_uncertainty=target_radius)
    return encounter_feasibility(chaser, tgt, horizon)["t_first"]
