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
    t = np.linspace(0.0, horizon, n)
    margin = np.linalg.norm(a.center(t) - b.center(t), axis=1) - a.radius(t) - b.radius(t) - contact_distance
    k = int(np.argmax(margin <= 0)) if np.any(margin <= 0) else None
    t_first = None
    if k is not None:
        t_first = float(t[k])
        if refine and k > 0:
            lo, hi = t[k - 1], t[k]
            f = lambda s: float(np.linalg.norm(a.center([s]) - b.center([s])) - a.radius([s])[0] - b.radius([s])[0]
                                - contact_distance)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if f(mid) <= 0 else (mid, hi)
            t_first = float(hi)
    return {"possible": k is not None, "t_first": t_first, "min_margin": float(margin.min()),
            "t_min_margin": float(t[int(np.argmin(margin))]), "times": t, "margin": margin}


def time_to_reach_region(chaser: BodyState, target_center: Sequence[float], target_radius: float = 0.0,
                         horizon: float = 1e6) -> Optional[float]:
    """Earliest time at which a (static) target region is inside the chaser's reachable ball."""
    tgt = BodyState(target_center, np.zeros(len(target_center)), pos_uncertainty=target_radius)
    return encounter_feasibility(chaser, tgt, horizon)["t_first"]
