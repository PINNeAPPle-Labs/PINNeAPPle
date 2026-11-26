"""The loop: the lightest arm whose structure, kinematics and compliance all pass.

Three models constrain each other, so the design cannot be searched one variable at a time: a thinner tube saves mass
and fails the stress check; a shorter upper arm gains camber and loads the bushes harder, which steers the wheel; a
stiffer bush takes the steer out and sends the shock into the chassis. The loop below evaluates the whole corner for
one candidate design — the CAD dimension of the arm, the bushes' rate, and two hardpoint positions — and searches for
the lightest that passes:

    design variables      tube OD and wall, the bushings' radial rate, the upper arm's inner-pivot height, the tie
                          rod's inner point
    objective             mass of the corner's two A-arms [kg]
    constraints           stress <= yield / safety factor; Euler factor >= 1.5; camber gain at least -0.25 deg per
                          50 mm of bump; bump steer within 0.10 deg per 50 mm; toe compliance <= 0.35 deg/kN; wheel
                          rate at least 18 N/mm

The search is a bounded pattern search (compass search with a shrinking step) from the nominal design: deterministic,
reproducible, and it shows the trade-off — every visit is recorded, so the report can plot mass against the worst
violation and name what stopped the thinner candidates. The finite-element check is **not** in the loop (it costs
seconds, the closed form milliseconds): the optimum is re-checked with it afterwards, which is what
:mod:`pinneapple_design.suspension.study` does and what makes the answer a verified one.

What the loop does not do: fatigue (a static stress margin is not a life), welds and heat-affected zones, the damper's
reaction on the arm in the cases (conservative), the other corner and the axle's roll stiffness, and any aero or
thermal load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .compliance import wheel_compliance
from .hardpoints import SuspensionSpec, front_corner
from .kinematics import kinematic_sweep, pose
from .loads import worst_case
from .structure import TubeSection, arm_mass_kg

__all__ = ["Design", "DesignResult", "evaluate", "suspension_design", "CONSTRAINTS", "BASELINE_DESIGN", "BOUNDS"]

#: objective and constraint targets; ``sense`` tells the loop which way each one has to hold
CONSTRAINTS: dict[str, dict] = {
    "static_margin": {"limit": 1.5, "sense": "min", "what": "yield / outer-fibre stress (the corner's safety factor)"},
    "buckling_factor": {"limit": 1.5, "sense": "min", "what": "Euler load / axial force on the compressed member"},
    "camber_gain_deg_per_50mm": {"limit": (-0.33, -0.27), "sense": "band",
                                 "what": "camber gained in 50 mm of bump: the target band of the axle"},
    "bump_steer_deg_per_50mm": {"limit": 0.10, "sense": "abs", "what": "toe change in 50 mm of bump"},
    "toe_compliance_deg_per_kn": {"limit": 0.45, "sense": "max", "what": "steer under a 1 kN lateral force"},
    "wheel_rate_n_per_mm": {"limit": 18.0, "sense": "min", "what": "wheel rate at the design height"},
}


@dataclass(frozen=True)
class Design:
    """One candidate: the two tube dimensions, the bushings' radial rate, and two hardpoint offsets."""
    od_mm: float = 32.0
    wall_mm: float = 3.5
    bushing_n_per_mm: float = 900.0
    uca_z_mm: float = 0.0                # the upper arm's inner pivots, moved up (+) or down (-)
    tierod_z_mm: float = 0.0             # the tie rod's inner point, moved up (+) or down (-)

    def apply(self, spec: SuspensionSpec) -> SuspensionSpec:
        """The corner this design describes: the arms of the tube, the bushes of the rate, the two hardpoints moved."""
        tube = TubeSection(self.od_mm * 1e-3, self.wall_mm * 1e-3)
        moved = replace(spec, tube_od_m=tube.od_m, tube_wall_m=tube.wall_m)
        moved = moved.with_bushings(self.bushing_n_per_mm * 1e3)
        dz = np.array([0.0, 0.0, self.uca_z_mm * 1e-3])
        dy = np.array([0.0, 0.0, self.tierod_z_mm * 1e-3])
        return replace(moved, uca_front=moved.uca_front + dz, uca_rear=moved.uca_rear + dz,
                       uca_outer=moved.uca_outer + dz, tierod_inner=moved.tierod_inner + dy)

    def as_dict(self) -> dict[str, float]:
        return {"od_mm": self.od_mm, "wall_mm": self.wall_mm, "bushing_n_per_mm": self.bushing_n_per_mm,
                "uca_z_mm": self.uca_z_mm, "tierod_z_mm": self.tierod_z_mm}


BASELINE_DESIGN = Design()
BOUNDS: dict[str, tuple[float, float]] = {
    "od_mm": (24.0, 56.0), "wall_mm": (2.0, 8.0), "bushing_n_per_mm": (400.0, 1600.0),
    "uca_z_mm": (-40.0, 40.0), "tierod_z_mm": (-30.0, 30.0),
}


def evaluate(spec: SuspensionSpec, design: Design, *, lateral_n: float = 3000.0) -> dict:
    """Run the three models for one candidate and grade it against the constraints.

    Returns
    -------
    out : dict
        The objective (``mass_kg``), the measured quantities (kinematics, stress, compliance), ``violations`` (each
        constraint's amount outside its limit, normalized) and ``feasible``.
    """
    s = design.apply(spec)
    p = pose(s, 0.0)
    sweep = kinematic_sweep(s, (-20.0, 20.0, 10.0))     # three stations: the loop needs slopes, not curves
    summary = sweep.summary()
    sizing = worst_case(s, p)
    comp = wheel_compliance(s, lateral_n=lateral_n)
    measured = {
        "static_margin": float(sizing["static_margin"]),
        "buckling_factor": float(sizing["buckling_factor"]),
        "camber_gain_deg_per_50mm": float(summary["camber_gain_deg_per_50mm"]),
        "bump_steer_deg_per_50mm": float(summary["bump_steer_deg_per_50mm"]),
        "toe_compliance_deg_per_kn": float(comp["toe_deg_per_kn"]),
        "wheel_rate_n_per_mm": float(summary["wheel_rate_n_per_mm"]),
    }
    violations = {}
    for name, rule in CONSTRAINTS.items():
        v, limit, sense = measured[name], rule["limit"], rule["sense"]
        if sense == "min":
            excess = (float(limit) - v) / max(abs(float(limit)), 1e-9)
        elif sense == "max":
            excess = (v - float(limit)) / max(abs(float(limit)), 1e-9)
        elif sense == "band":                    # a target band: outside either edge is a violation
            lo, hi = (float(x) for x in limit)
            excess = max(lo - v, v - hi) / max(hi - lo, 1e-9)
        else:                                    # abs: a band around zero
            excess = (abs(v) - float(limit)) / max(abs(float(limit)), 1e-9)
        violations[name] = float(max(excess, 0.0))
    return {"design": design, "spec": s, "pose": p, "sweep": sweep, "sizing": sizing, "compliance": comp,
            "measured": measured, "violations": violations, "worst_violation": float(max(violations.values())),
            "feasible": all(v <= 0.0 for v in violations.values()),
            "arm_mass_kg": arm_mass_kg(s), "bushing_mass_kg": bushing_mass_kg(design.bushing_n_per_mm),
            "mass_kg": arm_mass_kg(s) + bushing_mass_kg(design.bushing_n_per_mm), "sigma_mpa": sizing["sigma_vm_pa"] / 1e6,
            "sizing_case": sizing["case"], "summary": summary}


#: kg of bushing mass per corner at the nominal rate, and how it grows with stiffness (an estimate, not a part)
BUSHING_MASS_KG = 0.25
BUSHING_MASS_EXPONENT = 0.35


def bushing_mass_kg(rate_n_per_mm: float) -> float:
    """The four bush's mass as a function of their rate [kg]: heavier rubber and sleeves are stiffer ones.

    Returns
    -------
    mass : float [kg]
    """
    return 4.0 * BUSHING_MASS_KG * (rate_n_per_mm / 900.0) ** BUSHING_MASS_EXPONENT


def _score(r: dict) -> float:
    """What the search minimises: the mass plus a heavy penalty per unit of constraint violation, so a feasible design
    always beats an infeasible one that is lighter by less than the penalty."""
    return r["mass_kg"] + 5.0 * sum(r["violations"].values())


def _clip(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    return float(min(max(value, lo), hi))


@dataclass
class DesignResult:
    """The optimum, the baseline it improved on, and every candidate the loop visited."""
    design: Design
    best: dict
    baseline: dict
    history: list[dict] = field(default_factory=list)
    evaluations: int = 0
    frontier_kg: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        b, o = self.baseline, self.best
        return {"mass_kg": o["mass_kg"], "baseline_mass_kg": b["mass_kg"],
                "mass_reduction_pct": 100.0 * (b["mass_kg"] - o["mass_kg"]) / b["mass_kg"],
                "static_margin": o["measured"]["static_margin"], "sigma_mpa": o["sigma_mpa"],
                "buckling_factor": o["measured"]["buckling_factor"],
                "camber_gain_deg_per_50mm": o["measured"]["camber_gain_deg_per_50mm"],
                "bump_steer_deg_per_50mm": o["measured"]["bump_steer_deg_per_50mm"],
                "toe_compliance_deg_per_kn": o["measured"]["toe_compliance_deg_per_kn"],
                "wheel_rate_n_per_mm": o["measured"]["wheel_rate_n_per_mm"],
                "sizing_case": o["sizing_case"], "evaluations": self.evaluations,
                "feasible": o["feasible"], **{f"design_{k}": v for k, v in self.design.as_dict().items()},
                "baseline_feasible": b["feasible"], "baseline_violations": {k: v for k, v in b["violations"].items()
                                                                          if v > 0}}


def suspension_design(spec: SuspensionSpec | None = None, start: Design | None = None, steps: int = 40,
                      step_fraction: float = 0.14, tolerance: float = 1e-3) -> DesignResult:
    """Search the design space for the lightest corner that satisfies every constraint.

    A bounded compass search: at each round every variable is tried up and down by the current step, the best move is
    taken, and the step shrinks when no move helps — so it stalls only where no coordinate direction improves the score,
    which for this problem is the boundary of the stress, camber or compliance constraint. Deterministic: the same start
    and the same number of rounds give the same design.

    The loop keeps two records: the candidate that drives the search (lowest penalised score, which may be infeasible
    while it is still finding its way) and the lightest **feasible** candidate seen anywhere, which is what it returns
    when it has one — a search that walks through an infeasible pocket does not lose the answer it already had.

    Returns
    -------
    result : DesignResult
        ``design`` (the optimum), ``best`` and ``baseline`` (:func:`evaluate` records), ``history`` (every visited
        candidate, with its score) and ``frontier_kg`` (the lightest feasible design found per round).
    """
    spec = spec or front_corner()
    x = dict((start or BASELINE_DESIGN).as_dict())
    baseline = evaluate(spec, Design(**x))
    best = baseline
    lightest = baseline if baseline["feasible"] else None
    history = [{"round": 0, "candidate": x.copy(), "mass_kg": baseline["mass_kg"], "score": _score(baseline),
                "worst_violation": baseline["worst_violation"], "feasible": baseline["feasible"]}]
    frontier = []
    step_of = {k: (hi - lo) * step_fraction for k, (lo, hi) in BOUNDS.items()}
    evaluations = 1
    for rnd in range(1, steps + 1):
        moved = False
        for name in BOUNDS:
            for sign in (+1.0, -1.0):
                trial = dict(x)
                trial[name] = _clip(name, x[name] + sign * step_of[name])
                if abs(trial[name] - x[name]) < tolerance * max(1.0, abs(x[name])):
                    continue
                if name == "wall_mm" and trial["wall_mm"] > 0.45 * trial["od_mm"]:
                    continue
                rec = evaluate(spec, Design(**trial))
                evaluations += 1
                history.append({"round": rnd, "candidate": trial.copy(), "mass_kg": rec["mass_kg"],
                                "score": _score(rec), "worst_violation": rec["worst_violation"],
                                "feasible": rec["feasible"]})
                if _score(rec) < _score(best) - 1e-12:
                    best, x, moved = rec, trial, True
                if rec["feasible"] and (lightest is None or rec["mass_kg"] < lightest["mass_kg"]):
                    lightest = rec
        frontier.append({"round": rnd, "mass_kg": (lightest or best)["mass_kg"],
                         "feasible": bool(lightest), "worst_violation": (lightest or best)["worst_violation"]})
        if not moved:
            for k in step_of:
                step_of[k] *= 0.5
            if max(step_of.values()) < 0.5 * tolerance * 100.0:
                break
    answer = lightest or best
    return DesignResult(design=answer["design"], best=answer, baseline=baseline, history=history,
                        evaluations=evaluations, frontier_kg=frontier)
