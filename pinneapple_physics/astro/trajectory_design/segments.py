"""Sims-Flanagan low-thrust legs: a leg between two bodies as a sequence of bounded impulses, and the solver that
designs them.

The continuous thrust of an ion engine over an interplanetary leg is represented the way Sims & Flanagan (1997)
proposed and every low-thrust catalogue since uses: the leg of duration ``T`` is cut into ``N`` equal segments, and
inside each one the engine delivers one *impulsive* velocity change applied at the segment midpoint, of magnitude at
most what the engine can produce over the segment, ``(T_max / m_i) T/N``. Between impulses the spacecraft coasts on a
Keplerian arc, so the only integration is :func:`~.lambert.propagate_universal`, and the mass falls by the flow the
engine would use, ``T_max / (Isp g0)`` per second of thrusting. As ``N`` grows the impulses crowd towards the ends of
the leg and the propellant the model needs approaches the impulsive optimum; that convergence is measured in
``docs/core_concepts/trajectory_design.md`` and asserted in ``tests/test_trajectory_design.py``.

    thruster = Thruster.ion(thrust_n=0.5, mass0_kg=2000.0)          # 0.5 N, Isp 3100 s
    leg = SimsFlanaganLeg(r1, v1_body, r2, v2_body, tof_s=90 * DAY, n_segments=12, thruster=thruster)
    design = design_leg(leg)
    design.dv_km_s, design.propellant_kg, design.thrust_fraction

The controls are the **physical** impulses of the segments (km/s, ``(n_segments, 3)``), not dimensionless fractions:
an impulse of 1 km/s has the same meaning whatever the engine, and a solver that works in that space is conditioned
the same way for a 0.5 N ion thruster and for a 5 kN engine -- the fraction form is badly scaled, and a
Gauss-Newton step taken in it overshoots by the stamp of the thrust cap.

Two limits of the model are stated where they are made: the impulses are mid-segment, so thrust can never be
concentrated exactly at the ends, and gravity is Keplerian between impulses, so a leg that grazes a planet is outside
the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..ephemeris import GM_SUN
from .lambert import LambertError, lambert, propagate_universal

__all__ = ["G0", "Thruster", "SimsFlanaganLeg", "SimsFlanaganDesign", "design_leg", "evaluate_leg", "max_impulse"]

#: Standard gravity, m/s^2: the bridge between specific impulse (s) and exhaust velocity (m/s).
G0 = 9.80665


@dataclass(frozen=True)
class Thruster:
    """An engine, as the leg model needs it: maximum thrust, specific impulse and the mass it starts from.

    The acceleration is recomputed at every segment from the *current* mass, so a long leg really does thrust less at
    the end than at the beginning -- the reason propellant is not simply proportional to Δv.
    """

    thrust_n: float
    isp_s: float
    mass0_kg: float
    name: str = ""

    @classmethod
    def ion(cls, thrust_n: float = 0.5, mass0_kg: float = 2000.0, isp_s: float = 3100.0,
            name: str = "ion") -> Thruster:
        """A gridded ion engine: the order of magnitude of a Dawn/NEXT-class thruster."""
        return cls(thrust_n=thrust_n, isp_s=isp_s, mass0_kg=mass0_kg, name=name)

    @classmethod
    def chemical(cls, thrust_n: float = 5000.0, mass0_kg: float = 2000.0, isp_s: float = 320.0,
                 name: str = "high-thrust") -> Thruster:
        """A high-thrust engine: used to drive the model to its impulsive limit, where the answer is Lambert's."""
        return cls(thrust_n=thrust_n, isp_s=isp_s, mass0_kg=mass0_kg, name=name)

    @property
    def mass_flow_kg_s(self) -> float:
        """Propellant flow at full thrust, kg/s."""
        return self.thrust_n / (self.isp_s * G0)

    def acceleration_km_s2(self, mass_kg: float) -> float:
        """Full-thrust acceleration at a given mass, km/s^2 (the model works in km and s)."""
        return self.thrust_n / float(mass_kg) / 1000.0

    def max_dv_km_s(self, segment_s: float, mass_kg: float) -> float:
        """Largest velocity change one segment of ``segment_s`` seconds can deliver at that mass."""
        return self.acceleration_km_s2(mass_kg) * float(segment_s)

    def propellant_for(self, dv_km_s: float) -> float:
        """Propellant (kg) for a given total impulse, from the rocket equation (km/s in, kg out)."""
        exhaust_km_s = self.isp_s * G0 / 1000.0
        return float(self.mass0_kg) * (1.0 - float(np.exp(-float(dv_km_s) / exhaust_km_s)))


@dataclass(frozen=True)
class SimsFlanaganLeg:
    """One low-thrust leg between two bodies.

    Attributes
    ----------
    r_dep, v_dep:
        State of the departure body at departure, km and km/s (heliocentric).
    r_arr, v_arr:
        State of the arrival body at arrival; the leg is closed when the spacecraft reaches both.
    tof_s:
        Leg duration, s.
    n_segments:
        Number of impulsive divisions.
    thruster:
        The engine.
    departure:
        ``"free"`` (default) lets the solver choose the velocity the spacecraft leaves the departure body with, and
        charges the difference as a departure impulse (a launch). ``"fixed"`` leaves with the body's own velocity, so
        the engine supplies everything -- the low-thrust spiral, which is the interesting case when the engine is the
        binding constraint.
    arrival:
        ``"position"`` (default) constrains only the arrival *position*: the encounter is a flyby and the arrival
        velocity is free -- the constraint set of the MGA design this package does, and the one Sims & Flanagan use.
        ``"rendezvous"`` adds the velocity, which is what a capture needs and which costs much more: an impulse cannot
        be applied at the arrival epoch, so the last segment must both place the spacecraft and set its velocity.
    mu:
        Central gravity; the Sun.
    """

    r_dep: np.ndarray
    v_dep: np.ndarray
    r_arr: np.ndarray
    v_arr: np.ndarray
    tof_s: float
    n_segments: int = 10
    thruster: Thruster = None
    departure: str = "free"
    arrival: str = "position"
    mu: float = GM_SUN

    def __post_init__(self):
        if self.thruster is None:
            object.__setattr__(self, "thruster", Thruster.ion())
        if self.n_segments < 1:
            raise ValueError("a Sims-Flanagan leg needs at least one segment")
        if self.arrival not in ("position", "rendezvous"):
            raise ValueError("arrival must be 'position' or 'rendezvous'")
        if self.departure not in ("free", "fixed"):
            raise ValueError("departure must be 'free' or 'fixed'")
        if not self.tof_s > 0:
            raise ValueError("the leg duration must be positive")
        for name in ("r_dep", "v_dep", "r_arr", "v_arr"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3))

    @property
    def segment_s(self) -> float:
        return self.tof_s / self.n_segments

    @property
    def n_controls(self) -> int:
        return 3 * self.n_segments

    @property
    def max_impulse_km_s(self) -> float:
        """Largest impulse one segment can deliver at the initial mass, km/s."""
        return self.thruster.max_dv_km_s(self.segment_s, self.thruster.mass0_kg)

    def full_thrust_dv(self) -> float:
        """Impulse the engine could deliver over the whole leg at full thrust, km/s (the hard ceiling)."""
        return self.thruster.acceleration_km_s2(self.thruster.mass0_kg) * self.tof_s


def max_impulse(leg: SimsFlanaganLeg) -> float:
    """Per-segment impulse ceiling at the initial mass, km/s (the bound the solver works with)."""
    return leg.max_impulse_km_s


def evaluate_leg(leg: SimsFlanaganLeg, impulses, v0=None):
    """Fly one leg with the given impulses and return the end state and what it cost.

    ``impulses`` is ``(n_segments, 3)`` in km/s, the velocity change of each segment; a segment whose request exceeds
    what the engine can deliver in that segment's time is clipped to the limit (and the clipping is visible in the
    returned ``saturated`` mask). ``v0`` is the velocity the spacecraft leaves the departure body with; by default
    the body's own velocity, which is only useful for a check.

    Returns a dict with ``r``, ``v``, ``mass``, ``dv_thrust_km_s``, ``dv_departure_km_s``, ``dv_total_km_s``,
    ``propellant_kg``, ``impulses`` (after clipping), ``caps_km_s`` and ``saturated``.
    """
    dv_in = np.asarray(impulses, dtype=float).reshape(leg.n_segments, 3)
    v0 = np.asarray(leg.v_dep if v0 is None else v0, dtype=float).reshape(3)
    thr = leg.thruster
    mass = float(thr.mass0_kg)
    r = leg.r_dep.copy()
    v = v0.copy()
    dv_thrust = 0.0
    propellant = 0.0
    applied = np.zeros_like(dv_in)
    caps = np.zeros(leg.n_segments)
    saturated = np.zeros(leg.n_segments, dtype=bool)
    half = 0.5 * leg.segment_s
    for k in range(leg.n_segments):
        r, v = propagate_universal(r, v, half, leg.mu)
        cap = thr.max_dv_km_s(leg.segment_s, mass)
        caps[k] = cap
        dv_k = dv_in[k]
        dvn = float(np.linalg.norm(dv_k))
        if dvn > cap:
            dv_k = dv_k * (cap / dvn)
            saturated[k] = True
            dvn = cap
        applied[k] = dv_k
        v = v + dv_k
        burn_s = min(dvn / thr.acceleration_km_s2(mass), leg.segment_s)
        m_used = thr.mass_flow_kg_s * burn_s
        mass = max(mass - m_used, 1e-9)
        propellant += m_used
        dv_thrust += dvn
        r, v = propagate_universal(r, v, half, leg.mu)
    dv_dep = float(np.linalg.norm(v0 - leg.v_dep))
    return {"r": r, "v": v, "mass": mass, "dv_thrust_km_s": dv_thrust, "dv_departure_km_s": dv_dep,
            "dv_total_km_s": dv_dep + dv_thrust, "propellant_kg": propellant, "impulses": applied,
            "caps_km_s": caps, "saturated": saturated}


#: The model functions above were named ``evaluate_controls`` in the first draft; the name is kept as an alias so a
#: caller written against the fraction form still imports.
evaluate_controls = evaluate_leg


@dataclass
class SimsFlanaganDesign:
    """A designed leg: the impulses, what they cost, and how well the leg closes.

    ``dv_km_s`` is the total impulse, including the departure impulse that takes the spacecraft off the departure
    body (``|v0 - dv_dep`` direction in ``v0``); ``propellant_kg`` is what the model burned, segment by segment, and
    ``propellant_ideal_kg`` is the rocket equation applied to the same total impulse. ``closure`` is what is left at
    the arrival body, km and km/s: a design is reported as closed only when both are at the level the solver reached.
    """

    impulses: np.ndarray
    v0: np.ndarray
    thrust_fraction: np.ndarray
    dv_departure_km_s: float
    dv_thrust_km_s: float
    dv_km_s: float
    propellant_kg: float
    propellant_ideal_kg: float
    mass_final_kg: float
    closure_km: float
    closure_km_s: float
    n_segments: int
    n_evaluations: int
    converged: bool

    def as_dict(self) -> dict:
        return {"dv_km_s": self.dv_km_s, "dv_departure_km_s": self.dv_departure_km_s,
                "dv_thrust_km_s": self.dv_thrust_km_s, "propellant_kg": self.propellant_kg,
                "propellant_ideal_kg": self.propellant_ideal_kg, "mass_final_kg": self.mass_final_kg,
                "closure_km": self.closure_km, "closure_km_s": self.closure_km_s, "n_segments": self.n_segments,
                "converged": self.converged}


def design_leg(leg: SimsFlanaganLeg, objective: str = "delta_v", max_iter: int = 200,
               seed: int = 0) -> SimsFlanaganDesign:
    """Design the leg: the impulses (and departure velocity) that reach the arrival body for the least impulse.

    The unknowns are the departure velocity (3) and one impulse vector per segment (3 N). Two stages, deterministic:

    1. **closure** by Gauss-Newton on the six end-state residuals, with a minimum-norm step and a backtracking line
       search -- the map from an impulse to the end state is nearly linear in the impulse, so this converges in a
       handful of iterations and lands on a closed leg with the smallest impulses that close it;
    2. **cost reduction** by L-BFGS-B on "total impulse + a growing penalty on the closure residual", started from the
       closed design, with the penalty raised each round. A refined design is kept only when it closes at least as well
       as the closure stage did, so a cheaper number can never be bought with a wrong trajectory.

    Parameters
    ----------
    leg:
        The leg to close.
    objective:
        ``"delta_v"`` (sum of the impulse magnitudes, km/s) or ``"propellant"`` (kg burned). Both are reported
        either way; the flag only chooses which one the refinement pushes down.
    max_iter:
        Iteration cap of the L-BFGS-B rounds.
    seed:
        Unused except as the start when the leg has no single-revolution arc to start from; kept so a run is
        reproducible by construction.

    Returns
    -------
    SimsFlanaganDesign
    """
    if objective not in ("delta_v", "propellant"):
        raise ValueError("objective must be 'delta_v' or 'propellant'")
    rng = np.random.default_rng(seed)
    n = leg.n_segments
    if leg.full_thrust_dv() <= 0:
        raise ValueError("the leg duration or the mass is zero: no impulse can be delivered")
    r_scale, v_scale = 1e6, 1.0                     # a million km and 1 km/s make the two residuals comparable
    n_res = 3 if leg.arrival == "position" else 6
    dv_cap = max(20.0, 3.0 * float(np.linalg.norm(leg.v_dep)))     # no sane departure impulse exceeds this
    seg_cap = leg.max_impulse_km_s

    free_v0 = leg.departure == "free"

    def unpack(x):
        if not free_v0:
            return leg.v_dep, x.reshape(n, 3)
        return x[:3], x[3:].reshape(n, 3)

    def fly(x):
        v0, dv = unpack(x)
        return evaluate_leg(leg, dv, v0)

    def closure(x):
        out = fly(x)
        pos = (out["r"] - leg.r_arr) / r_scale
        if n_res == 3:
            return pos
        return np.concatenate([pos, (out["v"] - leg.v_arr) / v_scale])

    if free_v0:
        lo = np.concatenate([np.full(3, -dv_cap), np.full(3 * n, -seg_cap)])
        hi = np.concatenate([np.full(3, dv_cap), np.full(3 * n, seg_cap)])
    else:
        lo, hi = np.full(3 * n, -seg_cap), np.full(3 * n, seg_cap)

    # stage 1: the impulsive Lambert departure velocity is a good start when the engine is powerful; with the
    # departure fixed, one uniform tangential impulse over the leg is the natural low-thrust guess
    try:
        v_start = lambert(leg.r_dep, leg.r_arr, leg.tof_s, leg.mu).v1
    except LambertError:
        v_start = leg.v_dep + rng.normal(0.0, 1e-3, 3)
    if free_v0:
        x = np.clip(np.concatenate([v_start, np.zeros(3 * n)]), lo, hi)
    else:
        x = np.clip(np.repeat((v_start - leg.v_dep) / n, n), lo, hi)
    evaluations = 0

    for _ in range(15):
        res = closure(x)
        evaluations += 1
        if np.linalg.norm(res) < 1e-12:
            break
        J = np.empty((n_res, len(x)))
        for k in range(len(x)):
            dx = 1e-7 * max(1.0, abs(x[k]))
            xp = x.copy()
            xp[k] += dx
            J[:, k] = (closure(xp) - res) / dx
            evaluations += 1
        step, *_ = np.linalg.lstsq(J, -res, rcond=1e-12)
        alpha, f0 = 1.0, float(np.linalg.norm(res))
        for _ in range(25):
            trial = np.clip(x + alpha * step, lo, hi)
            f_trial = float(np.linalg.norm(closure(trial)))
            evaluations += 1
            if f_trial < f0:
                x, alpha = trial, -alpha
                break
            alpha *= 0.5
        else:
            break

    best_x = x.copy()
    best_res = float(np.linalg.norm(closure(best_x)))
    best_cost = _cost(fly(best_x), objective)

    # stage 2: push the impulse down while holding the closure through a penalty that grows each round
    for lam in (1e2, 1e4, 1e6, 1e8):
        def penalised(xv, lam=lam):
            res = closure(xv)
            return _cost(fly(xv), objective) + lam * float(np.dot(res, res))

        sol = minimize(penalised, best_x, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)),
                       options={"maxiter": max_iter, "maxfun": 4 * max_iter, "ftol": 1e-14, "gtol": 1e-12})
        evaluations += int(sol.nfev)
        if not np.all(np.isfinite(sol.x)):
            continue
        x_new = np.clip(sol.x, lo, hi)
        res_new = float(np.linalg.norm(closure(x_new)))
        cost_new = _cost(fly(x_new), objective)
        if res_new <= max(1e-10, 2.0 * best_res) and cost_new < best_cost:
            best_x, best_res, best_cost = x_new, res_new, cost_new

    v0, dv = unpack(best_x)
    out = evaluate_leg(leg, dv, v0)
    closure_km = float(np.linalg.norm(out["r"] - leg.r_arr))
    closure_km_s = float(np.linalg.norm(out["v"] - leg.v_arr))
    dv_total = float(out["dv_total_km_s"])
    fraction = np.divide(np.linalg.norm(out["impulses"], axis=1), np.maximum(out["caps_km_s"], 1e-300))
    return SimsFlanaganDesign(
        impulses=out["impulses"], v0=v0, thrust_fraction=fraction,
        dv_departure_km_s=float(out["dv_departure_km_s"]), dv_thrust_km_s=float(out["dv_thrust_km_s"]),
        dv_km_s=dv_total, propellant_kg=float(out["propellant_kg"]),
        propellant_ideal_kg=float(leg.thruster.propellant_for(dv_total)), mass_final_kg=float(out["mass"]),
        closure_km=closure_km, closure_km_s=closure_km_s, n_segments=n, n_evaluations=int(evaluations),
        converged=bool(closure_km < 1e3 and (leg.arrival == "position" or closure_km_s < 1e-3)))


def _cost(out: dict, objective: str) -> float:
    """The number the refinement minimises: total impulse (km/s) or propellant (kg)."""
    if objective == "propellant":
        return float(out["propellant_kg"])
    return float(out["dv_total_km_s"])
