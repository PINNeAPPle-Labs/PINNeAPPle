"""The suspension corner as one simulator: a design vector in, the measured numbers out.

This is the object the rest of the benchmark calls. One run is one corner: rigid kinematics (:mod:`.kinematics`), the
two wishbones through CalculiX (:mod:`.structure`), and the elastokinematic assembly of their compliances with the
bushes (:mod:`.elastokinematics`). Nothing here is a surrogate and nothing is estimated: every number in
:data:`OUTPUTS` comes from either a solved constraint system or a solved finite-element job, and a run that cannot
solve comes back ``ok=False`` with the reason instead of a made-up value.

The cost is what makes the benchmark interesting: a kinematics evaluation is ~10 ms and a full run is ~10 s, so a
design study that would take a week of solver time is the target the surrogates are asked to reproduce.
"""
from __future__ import annotations

import os
import time

import numpy as np

from pinneapple_autolab.simulator import SimOutcome, Simulator
from . import structure
from .elastokinematics import BUSH, elastokinematics
from .hardpoints import PARAMETERS, YIELD_STRENGTH, corner, invalid
from .kinematics import metrics

__all__ = ["OUTPUTS", "SuspensionSimulator", "evaluate"]

#: What a run measures. The first six are design quantities a suspension engineer signs off; the last four are the
#: structural and elastokinematic consequences of the same design.
OUTPUTS: tuple[str, ...] = (
    "mass_kg",
    "stress_utilization",
    "camber_gain_deg_per_25mm",
    "bump_steer_deg_per_25mm",
    "roll_centre_mm",
    "scrub_radius_mm",
    "wheel_rate_n_per_mm",
    "motion_ratio",
    "compliance_steer_deg_per_kN",
    "lateral_compliance_mm_per_kN",
)

_ARM_KEYS = ("mass_kg", "vm_max_pa", "compliance_m_per_n", "elements", "dofs", "u_boss_mm")


def evaluate(design: dict[str, float], workdir: str, *, size: float = structure.PRODUCTION_SIZE,
             order: int = structure.PRODUCTION_ORDER, keep: bool = False) -> dict[str, float]:
    """All of :data:`OUTPUTS` for one design, by running everything. Raises on a solver failure."""
    c = corner(design)
    kin = metrics(c)
    arms = {w: structure.response(structure.arm(c, w), os.path.join(workdir, w), size=size, order=order, keep=keep)
            for w in ("lower", "upper")}
    mass = arms["lower"]["mass_kg"] + arms["upper"]["mass_kg"]
    vm = max(arms["lower"]["vm_max_pa"], arms["upper"]["vm_max_pa"])
    ek = elastokinematics(c, {w: arms[w]["compliance_m_per_n"] for w in arms})
    return {
        "mass_kg": float(mass),
        "stress_utilization": float(vm) / YIELD_STRENGTH,
        "camber_gain_deg_per_25mm": kin["camber_gain_deg_per_25mm"],
        "bump_steer_deg_per_25mm": kin["bump_steer_deg_per_25mm"],
        "roll_centre_mm": kin["roll_centre_mm"],
        "scrub_radius_mm": kin["scrub_radius_mm"],
        "wheel_rate_n_per_mm": kin["wheel_rate_n_per_mm"],
        "motion_ratio": kin["motion_ratio"],
        "compliance_steer_deg_per_kN": ek["compliance_steer_deg_per_kN_lateral"],
        "lateral_compliance_mm_per_kN": ek["lateral_compliance_mm_per_kN"],
    }


class SuspensionSimulator(Simulator):
    """A double-wishbone front corner, solved: kinematics, CalculiX on both wishbones, elastokinematics."""
    name = "double_wishbone_corner"
    fidelity = "reference"
    outputs = OUTPUTS

    def __init__(self, size: float = structure.PRODUCTION_SIZE, order: int = structure.PRODUCTION_ORDER,
                 keep: bool = False):
        self.size, self.order, self.keep = size, order, keep

    def check(self, params) -> list[str]:
        """The design cannot be simulated as asked: out of range, or a mechanism that cannot be built."""
        return invalid(params)

    def run(self, params, workdir: str) -> SimOutcome:
        t0 = time.perf_counter()
        try:
            out = evaluate({k: float(params[k]) for k in (p.name for p in PARAMETERS) if k in params},
                           workdir, size=self.size, order=self.order, keep=self.keep)
        except Exception as exc:                                       # noqa: BLE001 - the run did not happen
            return SimOutcome({}, ok=False, message=f"{type(exc).__name__}: {exc}", cost_s=time.perf_counter() - t0)
        return SimOutcome(out, ok=True, cost_s=time.perf_counter() - t0,
                          meta={"mesh_size_m": self.size, "element_order": self.order,
                                "solver": "CalculiX", "bush_n_per_m": BUSH["wishbone_n_per_m"]})


def design_from_row(row, params=PARAMETERS) -> dict[str, float]:
    """A design vector out of a table row / dict / array in :data:`PARAMETERS` order."""
    if isinstance(row, dict):
        return {p.name: float(row[p.name]) for p in params}
    v = np.asarray(row, float)
    return {p.name: float(v[i]) for i, p in enumerate(params)}
