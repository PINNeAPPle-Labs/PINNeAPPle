"""GTOP-style interplanetary problems: a sequence of planets, Lambert arcs between them, and the total impulse the
mission costs.

The encoding is the one the ESA GTOP benchmark set uses for its *multiple gravity assist* (MGA) problems, and the
one Cassini1 is stated in: the decision vector is the departure epoch and one time of flight per leg,

    x = [t0, T1, ..., Tn]        t0 in MJD2000, Ti in days

and the objective is the **total impulse budget** of the mission, in the terms the GTOP MGA model charges it
(``mga.cpp`` in the ESA Advanced Concepts Team toolbox, and the Cassini1 page of the GTOP database):

    J = |v_sc(t0) - v_planet(t0)|                     departure (the launcher's job)
      + sum_k dv_powered(vinf_in, vinf_out, alpha_k)  powered flyby at each intermediate planet
      + dv_capture(vinf_arr)                          orbit insertion at the arrival body
      + penalties                                     a flyby below its minimum pericentre

The three terms that decide the number, and that a patched-conic budget is easy to get wrong:

* **The flyby is not a velocity mismatch.** A gravity assist turns ``v_inf`` for free; only a *change of its
  magnitude* costs impulse, and it is bought at the pericentre of the flyby hyperbola where the spacecraft is
  fastest, so the charge is ``|sqrt(vout**2 + 2 mu/rp) - sqrt(vin**2 + 2 mu/rp)|`` with ``rp`` the pericentre that
  turns ``v_inf`` through the angle the two Lambert legs actually ask for (:func:`powered_flyby`). Charging the
  vector difference ``|v_out - v_in|`` instead -- the naive reading, and the one this module used to implement --
  charges the free turn as well and inflates the Cassini1 optimum by a factor of six.
* **The arrival is an orbit insertion, not the arrival speed.** Cassini1 arrives at Saturn and is captured into an
  ellipse of pericentre 108 950 km and eccentricity 0.98, so the last term is the periapsis burn that takes the
  incoming hyperbola down to that ellipse (:func:`capture_dv`), not ``|v_inf|``.
* **The flyby pericentre is a constraint.** The GTOP problems state a minimum pericentre per body and charge a
  penalty per kilometre below it; a design that turns too hard at too low an altitude is not free, it is infeasible.

The two other readings of the same problem -- the flyby charged as the change of ``|v_inf|`` alone, or left free
with the mismatch constrained -- are kept as ``flyby_cost="difference"`` and ``flyby_cost="free"``, because published
numbers do not always say which one they mean; ``flyby_cost="powered"`` is the GTOP one and the default.

Every leg is a single-revolution Lambert arc (:func:`~.lambert.lambert`); a request that no single revolution can
fly is scored as infeasible rather than approximated.

Planets come from :func:`~.bodies.default_ephemeris` -- the local JPL DE kernel when one is installed, the JPL
approximate elements otherwise. The Cassini1 numbers in :data:`PUBLISHED_BEST` are stated in the ESA ACT's own
analytic elements, which differ from the modern JPL ones by ~1e6 km at these epochs; pass
:class:`~.bodies.GTOPEphemeris` to reproduce them, and read the guide for what the difference costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pinneapple_design.design_optimizer.study import VALID, Constraint, Evaluator

from ..ephemeris import GM, GM_SUN, RADIUS_KM
from .bodies import DESIGN_BODIES, default_ephemeris
from .lambert import LambertError, lambert

__all__ = ["DAY", "MJD2000_JD", "MGASequence", "MGADesign", "MGAProblem", "MGAEvaluator", "CASSINI1", "CASSINI1_BEST_X",
           "cassini1", "PUBLISHED_BEST", "powered_flyby", "capture_dv"]

DAY = 86400.0
#: Julian date of MJD 51544.5 (2000-01-01 12:00 TDB), the origin of MJD2000: ``jd = MJD2000_JD + t``.
MJD2000_JD = 2451544.5

#: The best known solution of GTOP Cassini1 and its objective, quoted from the ESA GTOP database
#: (``ACT/INF/Code/globopt`` toolbox, ``cassini1.m`` and ``mga.cpp``): 4.9307 km/s at
#: ``[-789.8117, 158.302027105278, 449.385873819743, 54.7489684339665, 1024.36205846918, 4552.30796805542]``, found by
#: a particle swarm (M. Stickel, Max-Planck-Institut fuer Astronomie, April 2006). The objective is the total impulse
#: of the mission in the model :mod:`.gtop` implements, and it is reproduced by
#: ``cassini1(GTOPEphemeris())(CASSINI1_BEST_X)`` -- see ``docs/core_concepts/trajectory_design.md``.
CASSINI1_BEST_X = (-789.8117, 158.302027105278, 449.385873819743, 54.7489684339665, 1024.36205846918,
                   4552.30796805542)

PUBLISHED_BEST = {
    "cassini1": {"dv_km_s": 4.9307,
                 "source": "ESA GTOP database, Cassini1 (MGA, powered flybys, Saturn orbit insertion), best known "
                           "solution 4.9307 km/s",
                 "x": CASSINI1_BEST_X,
                 "note": "reproduced to 4.9307 by this module with the ESA ACT ephemeris the benchmark is stated in "
                         "(bodies.GTOPEphemeris); with the modern JPL elements the same decision vector costs more"},
}


def powered_flyby(vin: float, vout: float, alpha: float, mu: float) -> tuple[float, float]:
    """The impulse a *powered* flyby costs: change ``|v_inf|`` from ``vin`` to ``vout`` while turning it ``alpha``.

    The gravity assist turns ``v_inf`` for free -- at any angle up to the one the pericentre radius allows -- and only
    the change of its magnitude has to be bought, at the pericentre of the hyperbola, where the spacecraft moves
    fastest. With the turn angle split between the incoming and outgoing branches,

        sin(delta_in) = 1/(1 + vin**2 rp/mu),   sin(delta_out) = 1/(1 + vout**2 rp/mu),   delta_in + delta_out = alpha

    whose root ``rp`` is unique (the left side falls monotonically from ``pi`` at ``rp = 0`` to ``0`` at infinity), and
    the impulse is ``dv = |sqrt(vout**2 + 2 mu/rp) - sqrt(vin**2 + 2 mu/rp)|``. This is ``PowSwingByInv`` of the GTOP
    toolbox, term for term.

    Parameters
    ----------
    vin, vout:
        Magnitude of the relative velocity before and after the flyby, km/s.
    alpha:
        Angle between the two relative-velocity vectors, radians (``0 <= alpha <= pi``).
    mu:
        Gravitational parameter of the flyby body, km^3/s^2.

    Returns
    -------
    (dv_km_s, rp_km):
        The impulse and the pericentre radius: ``rp`` is what a minimum-pericentre constraint is stated on.
    """
    if not vin > 0.0 or not vout > 0.0:                              # pragma: no cover - a planet state is never zero
        raise ValueError("the relative velocities must be positive")
    # solved in units of mu: with a = 1/v**2 the equation reads asin(a/(a+r)) + ... , and r comes back in km as r*mu
    a_in = 1.0 / (vin * vin)
    a_out = 1.0 / (vout * vout)
    r = 1.0
    for _ in range(30):
        f = (math.asin(a_in / (a_in + r)) + math.asin(a_out / (a_out + r)) - alpha)
        df = (-a_in / (math.sqrt((r + 2.0 * a_in) * r) * (a_in + r))
              - a_out / (math.sqrt((r + 2.0 * a_out) * r) * (a_out + r)))
        r_new = r - f / df if df != 0.0 else 2.0 * r
        if r_new > 0.0:
            converged = abs(r_new - r) <= 1e-8           # the ACT toolbox's own tolerance on r, in km/mu
            r = r_new
            if converged:
                break
        else:
            r *= 0.5
    rp = r * mu
    return abs(math.sqrt(vout * vout + 2.0 / r) - math.sqrt(vin * vin + 2.0 / r)), float(rp)


def capture_dv(vinf_km_s: float, mu: float, rp_km: float, e: float) -> float:
    """The impulse that captures an incoming ``vinf`` into an ellipse of pericentre ``rp_km`` and eccentricity ``e``.

    Both speeds are taken at the pericentre: the hyperbola's, ``sqrt(vinf**2 + 2 mu/rp)``, and the ellipse's,
    ``sqrt(2 mu/rp - mu/rp (1-e))``. The difference is the arrival term of the GTOP ``orbit insertion`` objective
    (``mga.cpp``); it is *not* ``|vinf|``, which would be the cost of stopping dead at Saturn.
    """
    mu_rp = mu / rp_km
    return float(abs(math.sqrt(vinf_km_s * vinf_km_s + 2.0 * mu_rp) - math.sqrt(2.0 * mu_rp - mu_rp * (1.0 - e))))


def _norm3(v) -> float:
    """``|v|`` of a three-vector as a Python float; ``np.linalg.norm`` on a 3-element array costs ~4 us of dispatch."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@dataclass(frozen=True)
class MGASequence:
    """A planet sequence and the box its decision vector lives in.

    Parameters
    ----------
    bodies:
        The bodies flown by, in order: ``n+1`` names for ``n`` legs. The first and last are the departure and
        arrival bodies; the intermediate ones are flybys.
    t0_bounds:
        Range of the departure epoch, MJD2000 (days from 2000-01-01 12:00 TDB).
    tof_bounds:
        Range of each time of flight, days, one pair per leg.
    name:
        Name of the problem.
    flyby_rp_min_km:
        Minimum pericentre radius at each flyby, km, one per intermediate body. ``None`` takes the body's mean
        radius: a trajectory that flies through a planet is not a trajectory.
    flyby_rp_penalty:
        Impulse charged per kilometre of violation of ``flyby_rp_min_km``, km/s/km -- 0.01 for the terrestrial
        planets and 0.001 for Jupiter in the GTOP problems, and 0.01 here. It is what makes the minimum a
        constraint in a penalty-free optimiser instead of a suggestion.
    arrival_orbit:
        ``(perihelion_km, eccentricity)`` of the orbit the arrival body captures the spacecraft into; ``None``
        means the mission ends at the arrival flyby and only the arrival speed is reported, never charged.
    """

    bodies: tuple[str, ...]
    t0_bounds: tuple[float, float]
    tof_bounds: tuple[tuple[float, float], ...]
    name: str = ""
    flyby_rp_min_km: tuple[float, ...] | None = None
    flyby_rp_penalty: tuple[float, ...] | float | None = 0.01
    arrival_orbit: tuple[float, float] | None = None

    def __post_init__(self):
        if len(self.bodies) < 2:
            raise ValueError("a sequence needs at least a departure body and an arrival body")
        if len(self.tof_bounds) != len(self.bodies) - 1:
            raise ValueError(f"{len(self.bodies)} bodies need {len(self.bodies) - 1} times of flight, "
                             f"got {len(self.tof_bounds)}")
        missing = [b for b in self.bodies if b not in DESIGN_BODIES]
        if missing:
            raise ValueError(f"no ephemeris for {sorted(set(missing))}; known: {DESIGN_BODIES}")
        if not self.t0_bounds[1] > self.t0_bounds[0]:
            raise ValueError("t0_bounds must be increasing")
        for lo, hi in self.tof_bounds:
            if not hi > lo:
                raise ValueError("every tof bound must be increasing")
        n_flybys = len(self.bodies) - 2
        for name, value in (("flyby_rp_min_km", self.flyby_rp_min_km), ("flyby_rp_penalty", self.flyby_rp_penalty)):
            if isinstance(value, tuple) and len(value) not in (n_flybys, 1):
                raise ValueError(f"{name} needs {n_flybys} entries for {n_flybys} flybys, got {len(value)}")
        if self.arrival_orbit is not None:
            if len(self.arrival_orbit) != 2:
                raise ValueError("arrival_orbit is (pericentre_km, eccentricity)")
            if not 0.0 <= self.arrival_orbit[1] < 1.0:
                raise ValueError("the capture orbit's eccentricity must be in [0, 1)")

    @property
    def n_legs(self) -> int:
        return len(self.bodies) - 1

    @property
    def n_flybys(self) -> int:
        return max(0, len(self.bodies) - 2)

    @property
    def names(self) -> list[str]:
        return ["t0"] + [f"tof_{k + 1}" for k in range(self.n_legs)]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.t0_bounds[0]] + [b[0] for b in self.tof_bounds], dtype=float)
        hi = np.array([self.t0_bounds[1]] + [b[1] for b in self.tof_bounds], dtype=float)
        return lo, hi

    def rp_min(self, flyby: int) -> float:
        """Minimum pericentre radius of flyby ``flyby`` (0-based, over the intermediate bodies), km."""
        if self.flyby_rp_min_km is None:
            return float(RADIUS_KM[self.bodies[flyby + 1]])
        return float(self.flyby_rp_min_km[flyby] if len(self.flyby_rp_min_km) > 1 else self.flyby_rp_min_km[0])

    def rp_penalty(self, flyby: int) -> float:
        """Impulse charged per kilometre of pericentre violation at flyby ``flyby``, km/s/km."""
        if self.flyby_rp_penalty is None:
            return 0.0
        if isinstance(self.flyby_rp_penalty, tuple):
            return float(self.flyby_rp_penalty[flyby] if len(self.flyby_rp_penalty) > 1 else self.flyby_rp_penalty[0])
        return float(self.flyby_rp_penalty)


#: Cassini1 as the GTOP set states it: Earth, Venus twice, Earth, Jupiter, Saturn; four gravity assists, and the
#: spacecraft captured at Saturn into a 108 950 km x 0.98 ellipse. Bounds, pericentre limits and penalty
#: coefficients from ``cassini1.m`` / ``mga.cpp`` and the problem page of the GTOP database.
CASSINI1 = MGASequence(
    bodies=("earth", "venus", "venus", "earth", "jupiter", "saturn"),
    t0_bounds=(-1000.0, 0.0),
    tof_bounds=((30.0, 400.0), (100.0, 470.0), (30.0, 400.0), (400.0, 2000.0), (1000.0, 6000.0)),
    name="cassini1",
    flyby_rp_min_km=(6351.8, 6351.8, 6778.1, 600000.0),        # Venus, Venus, Earth, Jupiter
    flyby_rp_penalty=(0.01, 0.01, 0.01, 0.001),
    arrival_orbit=(108950.0, 0.98),
)


@dataclass
class MGADesign:
    """One decoded decision vector: the epochs, the arcs, and where the impulse budget goes.

    ``legs`` holds one dict per leg with ``r1``, ``r2`` (km), ``v1``, ``v2`` (the Lambert velocities, km/s),
    ``v_dep_body``, ``v_arr_body``, ``tof_days``, ``transfer_angle_deg`` and ``z`` (the universal variable, which says
    whether the arc is elliptic or hyperbolic). ``flybys`` holds one dict per intermediate body with the two relative
    velocities, the turn angle the legs ask for, the pericentre that would turn them and the impulse it costs.
    """

    x: np.ndarray
    epochs_mjd2000: np.ndarray
    jd: np.ndarray
    legs: list = field(default_factory=list)
    flybys: list = field(default_factory=list)
    dv_departure_km_s: float = float("nan")
    dv_flyby_km_s: tuple = ()
    dv_arrival_km_s: float = float("nan")
    dv_penalty_km_s: float = 0.0
    dv_total_km_s: float = float("inf")
    flyby_mismatch_km_s: tuple = ()
    tof_days: float = float("nan")
    arrival_v_inf_km_s: float = float("nan")
    feasible: bool = False
    reason: str = ""

    @property
    def bodies(self) -> list[str]:
        """The bodies flown by, in order: ``n + 1`` names for ``n`` legs."""
        return [leg["from"] for leg in self.legs] + [self.legs[-1]["to"]] if self.legs else []

    @property
    def flyby_rp_km(self) -> tuple:
        """The pericentre radius each flyby would need, km."""
        return tuple(f["rp_km"] for f in self.flybys)

    @property
    def rp_violation_km(self) -> tuple:
        """How far each flyby pericentre falls below its minimum, km (zero when every flyby is legal)."""
        return tuple(f["rp_violation_km"] for f in self.flybys)

    @property
    def dv_components(self) -> dict:
        return {"departure": self.dv_departure_km_s, "flybys": list(self.dv_flyby_km_s),
                "arrival": self.dv_arrival_km_s, "penalty": self.dv_penalty_km_s, "total": self.dv_total_km_s}

    def as_dict(self) -> dict:
        return {"dv_km_s": self.dv_total_km_s, "dv_departure_km_s": self.dv_departure_km_s,
                "dv_flybys_km_s": list(self.dv_flyby_km_s), "dv_arrival_km_s": self.dv_arrival_km_s,
                "dv_penalty_km_s": self.dv_penalty_km_s,
                "tof_days": self.tof_days, "arrival_v_inf_km_s": self.arrival_v_inf_km_s,
                "flyby_rp_km": list(self.flyby_rp_km), "flyby_rp_violation_km": list(self.rp_violation_km),
                "epochs_mjd2000": list(self.epochs_mjd2000), "feasible": self.feasible, "reason": self.reason}


class MGAProblem:
    """A patched-conic MGA problem: decode a decision vector into arcs and an impulse budget, and score it.

    Parameters
    ----------
    sequence:
        The :class:`MGASequence`.
    ephemeris:
        Where the planet states come from; :func:`~.bodies.default_ephemeris` by default, and ``.source`` is kept.
        Pass :class:`~.bodies.GTOPEphemeris` to score a decision vector in the ephemeris the GTOP benchmarks are
        stated in.
    flyby_cost:
        How a gravity assist is charged. ``"powered"`` (the GTOP definition, and the default) charges the change of
        ``|v_inf|`` bought at the flyby pericentre that turns the trajectory where the two Lambert legs ask it to
        (:func:`powered_flyby`). ``"difference"`` charges ``||v_out| - |v_in||`` -- the same impulse without the
        1/rp boost of the pericentre burn. ``"free"`` charges nothing and reports the mismatch, which then has to be
        constrained (:func:`free_flyby_constraint`).
    arrival_cost:
        ``"auto"`` (the default) charges the orbit insertion :func:`capture_dv` when the sequence states one and
        ``|v_inf|`` -- arriving dead, with no capture -- when it does not; ``"insertion"``, ``"v_inf"`` and
        ``"free"`` force one of the three.
    penalty:
        The objective value given to a decision vector that cannot be flown at all (a Lambert arc that no single
        revolution makes), so an evolutionary search ranks it last instead of crashing on it.
    cache_size:
        How many decoded vectors to keep. A differential-evolution run re-evaluates the same vector a few times
        (``polish``, the mutation of an unchanged parent) and the cache is worth its bookkeeping; keeping every vector
        of a 360 000-evaluation run is not -- it is gigabytes of designs that are never read again, and the allocator
        spends the run collecting them.
    """

    _FLYBY_COSTS = ("powered", "difference", "free")
    _ARRIVAL_COSTS = ("auto", "insertion", "v_inf", "free")

    def __init__(self, sequence: MGASequence, ephemeris=None, flyby_cost: str = "powered", arrival_cost: str = "auto",
                 penalty: float = 1e3, cache_size: int = 4096):
        if flyby_cost not in self._FLYBY_COSTS:
            raise ValueError(f"flyby_cost must be one of {self._FLYBY_COSTS}")
        if arrival_cost not in self._ARRIVAL_COSTS:
            raise ValueError(f"arrival_cost must be one of {self._ARRIVAL_COSTS}")
        self.sequence = sequence
        self.ephemeris = ephemeris or default_ephemeris()
        self.flyby_cost = flyby_cost
        self.arrival_cost = ("insertion" if sequence.arrival_orbit is not None else "v_inf") \
            if arrival_cost == "auto" else arrival_cost
        self.penalty = float(penalty)
        self._cache: dict[tuple, MGADesign] = {}
        self._cache_size = int(cache_size)
        self._flyby_mu = [float(GM[b]) for b in sequence.bodies[1:-1]]

    # -- description -------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.sequence.name

    @property
    def source(self) -> str:
        return getattr(self.ephemeris, "source", "ephemeris")

    @property
    def lo(self) -> np.ndarray:
        return self.sequence.bounds[0]

    @property
    def hi(self) -> np.ndarray:
        return self.sequence.bounds[1]

    def clip(self, x) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float).reshape(-1), self.lo, self.hi)

    # -- decode ------------------------------------------------------------------------------------------------
    def decode(self, x) -> MGADesign:
        """Fly the sequence: planet states, one Lambert arc per leg, and the impulse budget of the GTOP model."""
        x = np.asarray(x, dtype=float).reshape(-1)
        key = tuple(x.tolist())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        x = np.clip(x, self.lo, self.hi)
        tofs = x[1:]
        epochs = np.concatenate([[x[0]], x[0] + np.cumsum(tofs)])
        jd = MJD2000_JD + epochs
        bodies = self.sequence.bodies
        rs, vs = [], []
        for k, body in enumerate(bodies):
            r, v = self.ephemeris.state(body, jd[k])
            rs.append(np.asarray(r, dtype=float).reshape(3))
            vs.append(np.asarray(v, dtype=float).reshape(3))
        design = MGADesign(x=x, epochs_mjd2000=epochs, jd=jd)
        try:
            for k in range(self.sequence.n_legs):
                arc = lambert(rs[k], rs[k + 1], tofs[k] * DAY, GM_SUN)
                design.legs.append({"r1": rs[k], "r2": rs[k + 1], "v1": arc.v1, "v2": arc.v2,
                                    "v_dep_body": vs[k], "v_arr_body": vs[k + 1], "tof_days": float(tofs[k]),
                                    "transfer_angle_deg": arc.transfer_angle_deg, "z": arc.z,
                                    "from": bodies[k], "to": bodies[k + 1]})
        except LambertError as exc:
            design.reason = str(exc)
            design.dv_total_km_s = self.penalty
            self._remember(key, design)
            return design

        dv_dep = _norm3(design.legs[0]["v1"] - vs[0])
        mismatches, dv_fly, penalty = [], [], 0.0
        for k in range(1, self.sequence.n_legs):
            v_in = design.legs[k - 1]["v2"] - vs[k]
            v_out = design.legs[k]["v1"] - vs[k]
            vin, vout = _norm3(v_in), _norm3(v_out)
            cosine = float(v_in[0] * v_out[0] + v_in[1] * v_out[1] + v_in[2] * v_out[2]) / (vin * vout) \
                if vin > 0.0 and vout > 0.0 else 1.0
            alpha = math.acos(min(1.0, max(-1.0, cosine)))
            mu = self._flyby_mu[k - 1]
            if self.flyby_cost == "free":
                dv, rp = 0.0, float("inf")
            else:
                dv, rp = powered_flyby(vin, vout, alpha, mu)
                if self.flyby_cost == "difference":
                    dv = abs(vout - vin)
            rp_min = self.sequence.rp_min(k - 1)
            violation = max(0.0, rp_min - rp)
            penalty += self.sequence.rp_penalty(k - 1) * violation
            mismatches.append(_norm3(v_out - v_in))
            dv_fly.append(dv)
            design.flybys.append({"at": bodies[k], "vinf_in_km_s": vin, "vinf_out_km_s": vout,
                                  "turn_angle_deg": math.degrees(alpha), "rp_km": rp, "rp_min_km": rp_min,
                                  "rp_violation_km": violation, "dv_km_s": dv})

        v_inf_arr = _norm3(design.legs[-1]["v2"] - vs[-1])
        if self.arrival_cost == "insertion":
            rp_t, e_t = self.sequence.arrival_orbit
            dv_arr = capture_dv(v_inf_arr, float(GM[bodies[-1]]), float(rp_t), float(e_t))
        elif self.arrival_cost == "v_inf":
            dv_arr = v_inf_arr
        else:
            dv_arr = 0.0
        design.dv_departure_km_s = dv_dep
        design.dv_arrival_km_s = dv_arr
        design.dv_flyby_km_s = tuple(dv_fly)
        design.dv_penalty_km_s = float(penalty)
        design.flyby_mismatch_km_s = tuple(mismatches)
        design.dv_total_km_s = float(dv_dep + sum(dv_fly) + dv_arr + penalty)
        design.tof_days = float(np.sum(tofs))
        design.arrival_v_inf_km_s = v_inf_arr
        design.feasible = True
        self._remember(key, design)
        return design

    def _remember(self, key, design: MGADesign) -> None:
        """Store a decoded design, dropping the whole cache when it is full: a bounded cache, not an evergrowing one."""
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = design

    # -- score -------------------------------------------------------------------------------------------------
    def objectives(self, x) -> np.ndarray:
        """``[total Δv km/s, total flight time days]`` -- the two the Pareto front is drawn over."""
        d = self.decode(x)
        return np.array([d.dv_total_km_s, d.tof_days])

    def mismatch(self, x) -> float:
        """Sum of the flyby velocity mismatches, km/s (0 when every flyby is free without an impulse)."""
        return float(sum(self.decode(x).flyby_mismatch_km_s))

    def __call__(self, x) -> float:
        return float(self.decode(x).dv_total_km_s)

    def describe_best(self, x) -> dict:
        """The decoded best design, ready for a table or a figure."""
        return self.decode(x).as_dict()


def cassini1(ephemeris=None, **options) -> MGAProblem:
    """The GTOP Cassini1 problem (E-V-V-E-J-S, four gravity assists, Saturn orbit insertion).

    ``options`` go to :class:`MGAProblem`; pass ``ephemeris=GTOPEphemeris()`` to score it in the ephemeris the
    benchmark's published 4.9307 km/s is stated in.
    """
    return MGAProblem(CASSINI1, ephemeris=ephemeris, **options)


class MGAEvaluator(Evaluator):
    """An :class:`MGAProblem` as a ``pinneapple_design`` evaluator, so the library's optimisers and their budget
    accounting apply unchanged.

    Outputs: ``dv`` (total impulse, km/s), ``tof`` (days), ``arrival_v_inf`` (km/s), ``flyby_mismatch`` (km/s) and
    ``"_valid"`` (False when no single-revolution arc exists, which makes the design infeasible instead of merely
    expensive).
    """

    name = "mga-patched-conic"
    fidelity = "model"
    outputs = ("dv", "tof", "arrival_v_inf", "flyby_mismatch")

    def __init__(self, problem: MGAProblem, include_time: bool = False):
        self.problem = problem
        self.include_time = bool(include_time)
        self.description = (f"{problem.name}: {len(problem.sequence.bodies)} bodies, powered flybys "
                            f"({problem.flyby_cost}), {problem.source}")

    def evaluate(self, X: np.ndarray) -> dict[str, np.ndarray]:                       # noqa: N803 (library name)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = len(X)
        dv = np.empty(n)
        tof = np.empty(n)
        vinf = np.empty(n)
        mism = np.empty(n)
        valid = np.ones(n, dtype=bool)
        for i, row in enumerate(X):
            d = self.problem.decode(row)
            dv[i] = d.dv_total_km_s
            tof[i] = d.tof_days
            vinf[i] = d.arrival_v_inf_km_s
            mism[i] = float(sum(d.flyby_mismatch_km_s))
            valid[i] = d.feasible
        return {"dv": dv, "tof": tof, "arrival_v_inf": vinf, "flyby_mismatch": mism, VALID: valid}


def free_flyby_constraint(tolerance_km_s: float) -> Constraint:
    """The constraint the free-flyby reading needs: every flyby mismatch below ``tolerance_km_s``."""
    return Constraint("flyby_mismatch", upper=float(tolerance_km_s), unit="km/s",
                      label=f"a free flyby needs the relative velocity matched to {tolerance_km_s:g} km/s")
