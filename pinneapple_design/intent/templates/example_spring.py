"""Template-authoring example: a helical compression spring (the worked example of the guide
``docs/guides/design_from_intent.md``). It is not registered by default; ``register(CoilSpringTemplate())`` adds it.

Physics (Shigley's Mechanical Engineering Design, 10th ed., ch. 10): shear stress ``tau = K 8 F D / (pi d^3)`` with
``K = 1`` (closed form) or Wahl's curvature factor ``K_W = (4C - 1)/(4C - 4) + 0.615/C`` (reduced), spring rate
``k = G d^4 / (8 D^3 N_a)``; music wire ``S_ut = A / d^m`` with ``A = 2211 MPa mm^m``, ``m = 0.145`` (Table 10-4),
``G = 81.7 GPa`` (Table 10-5), static torsional yield ``S_sy = 0.45 S_ut`` (Table 10-6, set not removed).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import materials as M
from ..spec import ObjectiveChoice, RequirementSpec, Slot
from ..template import (
    CLOSED_FORM,
    REDUCED,
    DesignVariable,
    FailureMode,
    Fidelity,
    ProblemTemplate,
    Uncertain,
)

SHIGLEY = "Shigley's Mechanical Engineering Design, 10th ed., Tables 10-4 and 10-5 (music wire, ASTM A228)"
G_MUSIC = 81.7e9
A_MUSIC, M_MUSIC = 2211.0, 0.145          # S_ut [MPa] = A / d[mm]^m


class CoilSpringTemplate(ProblemTemplate):
    # 1. what selects it and what it can read from a request
    name = "coil_spring"
    title = "Helical compression spring"
    domain = "structures (torsion of a helical wire)"
    keywords = {"spring": 2.5, "compression spring": 1.0, "coil": 0.5}
    slots = (
        Slot("force", "force at full compression", "N", "N", ("takes", "force", "load", "for", "carry"),
             bounds=(0.1, 5e3), critical=True, role="load", question="What force must the spring carry?"),
        Slot("travel", "travel at that force", "m", "mm", ("travel", "deflection", "stroke", "compress"),
             bounds=(1e-3, 0.2), critical=True, role="limit", question="How far must it compress at that force?"),
        Slot("safety_factor", "safety factor on torsional yield", "", "", ("safety factor",), default=1.5,
             default_source="template default (state yours)", bounds=(1.0, 5.0)),
    )
    # 2. the physics, the assumptions, the objectives and the failure modes (each a margin >= 0)
    physics = ("torsion of the helical wire (shear stress with Wahl's curvature correction)",
               "linear spring rate from the wire's shear modulus")
    assumptions = ("music wire (ASTM A228), static load, ends squared and ground (two inactive coils)",)
    objectives = (ObjectiveChoice("mass_g", "min", "mass", ("light", "small")),)
    failure_modes = (
        FailureMode("margin_shear", "wire stays below torsional yield", "S_sy / (n tau) - 1"),
        FailureMode("margin_rate", "travel reached at the force", "1 - |F / k - travel| / travel - 0.95: rate "
                                                                  "within 5 % of the target"),
        FailureMode("margin_index", "spring index 4 <= C <= 12", "manufacturable winding (Shigley sec. 10-5)"),
    )
    fidelities = (Fidelity(CLOSED_FORM, "straight-bar torsion", "tau = 8 F D / (pi d^3)"),
                  Fidelity(REDUCED, "Wahl-corrected torsion", "tau = K_W 8 F D / (pi d^3)"))
    output_units = {"mass_g": "g", "tau_mpa": "MPa", "rate_n_mm": "N/mm"}

    # 3. the design variables
    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        return [DesignVariable("wire_mm", 0.3, 6.0, "mm", "wire diameter d"),
                DesignVariable("coil_mm", 3.0, 60.0, "mm", "mean coil diameter D"),
                DesignVariable("active", 2, 30, "", "active coils N_a", integer=True)]

    # 4. the models: named outputs for a batch of designs, every failure-mode margin and every objective
    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, *, force_scale=1.0, **_: Any):
        X = np.atleast_2d(X)
        d, D, Na = X[:, 0] * 1e-3, X[:, 1] * 1e-3, np.round(X[:, 2])
        F = spec["force"] * np.asarray(force_scale, float)
        C = D / d
        K = (4 * C - 1) / (4 * C - 4) + 0.615 / C if level == REDUCED else np.ones_like(C)
        tau = K * 8 * F * D / (math.pi * d ** 3)
        ssy = 0.45 * A_MUSIC / (d * 1e3) ** M_MUSIC * 1e6
        k = G_MUSIC * d ** 4 / (8 * D ** 3 * Na)
        travel = spec["travel"]
        rho = M.get("steel_1018")["density"]   # carbon-steel density (cited) for the mass objective
        mass = rho * (math.pi * d ** 2 / 4) * math.pi * D * (Na + 2)
        return {"margin_shear": ssy / (spec["safety_factor"] * tau) - 1, "tau_mpa": tau / 1e6,
                "margin_rate": 0.05 - np.abs(spec["force"] / k - travel) / travel,
                "margin_index": np.minimum(C / 4 - 1, 12 / C - 1), "mass_g": mass * 1e3, "rate_n_mm": k / 1e3}

    # 5. what is uncertain, for the robustness study
    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        return [Uncertain("force_scale", 1.0, 0.05, "uniform", "", "force +-5 %")]

    def limits(self, spec: RequirementSpec) -> list[str]:
        return ["Static load: fatigue (Goodman / Zimmerli) and buckling of long springs are not checked."]
