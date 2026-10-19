"""Problem templates: how a kind of request maps to physics, design variables, models and failure modes.

A template is the unit a new domain plugs in with (see ``docs/guides/design_from_intent.md``, "Writing a template").
It declares

* the words that select it (``keywords``) and the values it can read from a request (``slots``);
* the governing physics and the assumptions behind it (``physics``, ``assumptions``);
* the design variables (``variables``), with discrete choices such as materials encoded as integer indices;
* the models at each fidelity (``fidelities``: closed form, reduced / surrogate, full solver) through
  ``evaluate(level, X, spec)``, which returns named outputs for a batch of designs;
* the failure modes, each a *margin* output that must be >= 0 (``failure_modes``);
* the uncertain inputs for the robustness study (``uncertainties``);
* the outputs of a design package: geometry, bill of materials, operating window, build instructions, limits.

Every number a template returns must come from a model, a cited datasheet or the user. Assumptions that are not
measured are listed in ``assumptions`` or ``limits`` so that the report shows them.
"""
from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spec import ObjectiveChoice, RequirementSpec, Slot

CLOSED_FORM, REDUCED, FULL = "closed_form", "reduced", "full"
LEVELS = (CLOSED_FORM, REDUCED, FULL)


@dataclass(frozen=True)
class Fidelity:
    level: str                         # closed_form | reduced | full
    label: str                         # "energy balance", "1-D along-track ODE", "Newton rigid-body contact"
    method: str = ""                   # one line: what is solved and how
    modules: tuple[str, ...] = ()      # Python modules it needs
    executables: tuple[str, ...] = ()  # programs it needs (any one of each tuple entry separated by "|")
    check: Callable[[], tuple[bool, str]] | None = None   # any other requirement

    def available(self) -> tuple[bool, str]:
        for m in self.modules:
            if importlib.util.find_spec(m) is None:
                return False, f"Python module '{m}' is not installed"
        for exe in self.executables:
            if not any(shutil.which(e) for e in exe.split("|")):
                return False, f"program '{exe}' is not on PATH"
        if self.check is not None:
            return self.check()
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        ok, why = self.available()
        return {"level": self.level, "label": self.label, "method": self.method, "available": ok, "reason": why}


@dataclass(frozen=True)
class FailureMode:
    name: str                          # output key holding the margin (>= 0 means the mode does not occur)
    label: str
    description: str
    critical: bool = True


@dataclass(frozen=True)
class DesignVariable:
    name: str
    lower: float
    upper: float
    unit: str = ""
    label: str = ""
    integer: bool = False
    choices: tuple[str, ...] = ()      # for a discrete choice: the value at index i

    def decode(self, x: float) -> Any:
        if self.choices:
            return self.choices[int(np.clip(round(x), 0, len(self.choices) - 1))]
        return int(round(x)) if self.integer else float(x)


@dataclass(frozen=True)
class Uncertain:
    """An uncertain input of the robustness study: ``nominal`` and a symmetric ``half_width`` (uniform) or a
    standard deviation (normal), with the reason for the range."""
    name: str
    nominal: float
    spread: float
    dist: str = "uniform"              # uniform: nominal +- spread; normal: sigma = spread
    unit: str = ""
    source: str = ""


@dataclass
class FullResult:
    """What the high-fidelity solver measured for one design."""
    outputs: dict[str, float]
    checks: list[dict[str, Any]] = field(default_factory=list)   # self-checks: name, passed, value, limit, detail
    level: str = FULL
    label: str = ""
    wall_s: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)       # trajectories, fields, meshes for the package
    notes: list[str] = field(default_factory=list)
    compare: dict[str, str] = field(default_factory=dict)         # full output -> reduced output it verifies
    compare_options: dict[str, Any] = field(default_factory=dict)  # reduced-model options matching the full run
    # how each compared output is judged: "agree" (relative error within the tolerance, the default), "at_least" /
    # "at_most" (the full value must not be below / above the reduced one, e.g. a solver with known extra losses
    # checked against the worst-case envelope)
    compare_kind: dict[str, str] = field(default_factory=dict)


class ProblemTemplate:
    """Base class. Subclasses set the class attributes and implement :meth:`evaluate` and the package hooks."""

    name: str = ""
    title: str = ""
    domain: str = ""
    keywords: dict[str, float] = {}
    physics: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    slots: tuple[Slot, ...] = ()
    objectives: tuple[ObjectiveChoice, ...] = ()
    failure_modes: tuple[FailureMode, ...] = ()
    fidelities: tuple[Fidelity, ...] = ()
    robust_target: float = 0.95        # required probability that every failure mode stays away
    reduced_is_expensive: bool = False  # True: optimise a GP surrogate of the reduced model (seconds per design)
    # model of the Monte-Carlo robustness study; a cheaper one than ``REDUCED`` is calibrated per failure mode to the
    # best model that ran on the design (the full solver, else the reduced model)
    robust_level: str = REDUCED

    # ------------------------------------------------------------------ declaration helpers
    def slot_map(self) -> dict[str, Slot]:
        return {s.name: s for s in self.slots}

    def fidelity(self, level: str) -> Fidelity | None:
        return next((f for f in self.fidelities if f.level == level), None)

    def default_objectives(self) -> list[ObjectiveChoice]:
        return [self.objectives[0]] if self.objectives else []

    def check_scope(self, spec: RequirementSpec) -> list[str]:
        """Reasons the request is outside what this template can design (empty = in scope)."""
        return []

    def extra_questions(self, spec: RequirementSpec) -> list:
        """Questions beyond the critical slots (e.g. "a force or a mass": one of two slots is needed)."""
        return []

    def full_design(self, design: dict[str, Any], spec: RequirementSpec) -> dict[str, Any]:
        """The decoded design variables plus the values the spec fixes (for the package and the full solver)."""
        return dict(design)

    def design_options(self, spec: RequirementSpec) -> dict[str, Any]:
        """Options for :meth:`evaluate` during the optimisation: the worst-case corner of the uncertain inputs, so
        that the optimum is robust and not on the nominal edge. Empty: optimise at nominal values."""
        return {}

    def calibrate(self, full: FullResult) -> dict[str, Any]:
        """Options for :meth:`evaluate` measured by the full solver (e.g. a stress-concentration factor); the
        pipeline re-optimises with them when they change the verdict. Empty: nothing to calibrate."""
        return {}

    # ------------------------------------------------------------------ design problem
    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        raise NotImplementedError

    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, **options: Any) -> dict[str, np.ndarray]:
        """Outputs for a batch of designs ``X`` (rows in :meth:`variables` order) at ``level``. Must include every
        failure-mode margin and every objective."""
        raise NotImplementedError

    def verify(self, design: dict[str, Any], spec: RequirementSpec, workdir: str | None = None) -> FullResult:
        """Run the full-fidelity solver on one decoded design."""
        raise NotImplementedError

    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        return []

    def robust_inputs(self, X: np.ndarray, names: Sequence[str], samples: np.ndarray, spec: RequirementSpec
                      ) -> dict[str, np.ndarray]:
        """Options passed to :meth:`evaluate` for the Monte-Carlo samples (``samples[:, j]`` is uncertain input
        ``names[j]``). The default passes them as keyword arrays."""
        return {n: samples[:, j] for j, n in enumerate(names)}

    # ------------------------------------------------------------------ package
    def geometry(self, spec: RequirementSpec, design: dict[str, Any]):
        """A list of :class:`pinneapple_design.geometry_library.core.Body` (metres, z up)."""
        return []

    def bill_of_materials(self, spec: RequirementSpec, design: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def operating_window(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]
                         ) -> list[dict[str, Any]]:
        return []

    def build_instructions(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
        return []

    def limits(self, spec: RequirementSpec) -> list[str]:
        return []

    def figures(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any], out_dir: str
                ) -> dict[str, str]:
        """Extra figures (name -> file path) for the report."""
        return {}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "domain": self.domain, "physics": list(self.physics),
                "assumptions": list(self.assumptions),
                "slots": [{"name": s.name, "label": s.label, "unit": s.shown_unit(), "critical": s.critical,
                           "default": s.default, "default_source": s.default_source} for s in self.slots],
                "objectives": [{"name": o.name, "sense": o.sense, "label": o.label} for o in self.objectives],
                "failure_modes": [{"name": f.name, "label": f.label, "description": f.description}
                                  for f in self.failure_modes],
                "fidelities": [f.to_dict() for f in self.fidelities]}
