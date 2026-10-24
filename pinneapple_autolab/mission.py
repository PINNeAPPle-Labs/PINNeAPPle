"""A mission as numbers: what to improve, what must hold, what may change, how much simulation it may spend.

    from pinneapple_autolab import Mission, Budget
    from pinneapple_design.design_optimizer.study import Variable, Objective

    m = Mission("Lower drag", statement="Find a lower-drag design at 70 m/s ...",
                variables=[Variable("rear_flap_aoa", 15, 40)], objectives=[Objective("drag_N", "min", "N")],
                constraints=[Limit("downforce_N", lower_frac_of_baseline=1.0, unit="N")],
                baseline={"rear_flap_aoa": 32.0}, budget=Budget(max_runs=30))
    m.resolve({"downforce_N": 4369.0})          # relative limits become absolute once the baseline is measured

A limit relative to the baseline (``lower_frac_of_baseline=1.0``: "preserve the downforce") is only a number after
the baseline has been simulated; until then :meth:`Mission.constraints` refuses to return it, so nothing can be
called feasible against a limit nobody measured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pinneapple_design.design_optimizer.study import Constraint, Objective, Variable

__all__ = ["Budget", "Limit", "Mission", "UnresolvedLimitError"]


class UnresolvedLimitError(RuntimeError):
    """A limit relative to the baseline was used before the baseline was simulated."""


@dataclass
class Budget:
    """Simulation budget. ``max_runs`` reference simulations in total, of which ``reserve_verify`` are kept for the
    final verification of the chosen designs; ``max_wall_s`` (optional) caps the summed solver wall time."""
    max_runs: int = 30
    reserve_verify: int = 3
    max_wall_s: float | None = None

    def __post_init__(self):
        if self.max_runs < 2:
            raise ValueError("a budget needs at least 2 simulations (baseline and one verification)")
        if not 0 <= self.reserve_verify < self.max_runs:
            raise ValueError("reserve_verify must be between 0 and max_runs - 1")


@dataclass
class Limit:
    """``lower <= output <= upper``; a bound can instead be a fraction of the baseline's value
    (``lower_frac_of_baseline=1.0``: at least the baseline's). ``kind``: "physical", "geometry", "validity"."""
    name: str
    lower: float | None = None
    upper: float | None = None
    lower_frac_of_baseline: float | None = None
    upper_frac_of_baseline: float | None = None
    unit: str = ""
    kind: str = "physical"
    label: str = ""

    def __post_init__(self):
        if all(v is None for v in (self.lower, self.upper, self.lower_frac_of_baseline, self.upper_frac_of_baseline)):
            raise ValueError(f"limit {self.name!r} needs a bound")

    @property
    def relative(self) -> bool:
        return self.lower_frac_of_baseline is not None or self.upper_frac_of_baseline is not None

    def resolved(self, baseline: dict[str, float] | None) -> Constraint:
        lo, hi = self.lower, self.upper
        if self.relative:
            if not baseline or self.name not in baseline:
                raise UnresolvedLimitError(f"limit on {self.name!r} is relative to the baseline, which has not been "
                                           "simulated yet")
            b = float(baseline[self.name])
            if self.lower_frac_of_baseline is not None:
                lo = self.lower_frac_of_baseline * b if lo is None else max(lo, self.lower_frac_of_baseline * b)
            if self.upper_frac_of_baseline is not None:
                hi = self.upper_frac_of_baseline * b if hi is None else min(hi, self.upper_frac_of_baseline * b)
        return Constraint(self.name, upper=hi, lower=lo, unit=self.unit, kind=self.kind, label=self.label or self.name)

    def describe(self) -> str:
        parts = []
        if self.lower_frac_of_baseline is not None:
            parts.append(f">= {self.lower_frac_of_baseline:g} x baseline")
        if self.upper_frac_of_baseline is not None:
            parts.append(f"<= {self.upper_frac_of_baseline:g} x baseline")
        if self.lower is not None:
            parts.append(f">= {self.lower:g} {self.unit}".strip())
        if self.upper is not None:
            parts.append(f"<= {self.upper:g} {self.unit}".strip())
        return f"{self.label or self.name} " + " and ".join(parts)


@dataclass
class Mission:
    """What the lab is asked to do. ``acceptance``: relative tolerance per output for a prediction to count as
    confirmed by the reference solver (|prediction - reference| / |reference| <= tol). ``conditions``: the fixed
    operating and boundary conditions (speed, ground, ...); the lab never changes them."""
    title: str
    statement: str
    variables: list[Variable]
    objectives: list[Objective]
    constraints: list[Limit] = field(default_factory=list)
    baseline: dict[str, float] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    acceptance: dict[str, float] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)
    domain: str = ""
    _baseline_outputs: dict[str, float] | None = field(default=None, repr=False)

    def __post_init__(self):
        names = [v.name for v in self.variables]
        missing = [k for k in names if k not in self.baseline]
        if self.baseline and missing:
            raise ValueError(f"baseline lacks variables {missing}")
        for k, v in self.baseline.items():
            var = next((x for x in self.variables if x.name == k), None)
            if var is not None and not var.lower <= v <= var.upper:
                raise ValueError(f"baseline {k}={v} is outside its bounds [{var.lower}, {var.upper}]")

    def resolve(self, baseline_outputs: dict[str, float]) -> list[Constraint]:
        """Fix the relative limits with the measured baseline outputs; returns the absolute constraints."""
        self._baseline_outputs = {k: float(v) for k, v in baseline_outputs.items()}
        return self.constraints_absolute()

    @property
    def resolved(self) -> bool:
        return self._baseline_outputs is not None or not any(c.relative for c in self.constraints)

    def constraints_absolute(self) -> list[Constraint]:
        return [c.resolved(self._baseline_outputs) for c in self.constraints]

    @property
    def baseline_outputs(self) -> dict[str, float] | None:
        return None if self._baseline_outputs is None else dict(self._baseline_outputs)

    def outputs(self) -> list[str]:
        """Outputs the lab must predict: objectives, then constrained outputs."""
        return list(dict.fromkeys([o.name for o in self.objectives] + [c.name for c in self.constraints]))

    def to_dict(self) -> dict[str, Any]:
        d = {"title": self.title, "statement": self.statement, "domain": self.domain,
             "variables": [asdict(v) for v in self.variables], "objectives": [asdict(o) for o in self.objectives],
             "constraints": [asdict(c) | {"text": c.describe()} for c in self.constraints],
             "baseline": dict(self.baseline), "budget": asdict(self.budget), "acceptance": dict(self.acceptance),
             "conditions": dict(self.conditions), "hypotheses": list(self.hypotheses)}
        if self._baseline_outputs is not None:
            d["baseline_outputs"] = dict(self._baseline_outputs)
            d["constraints_absolute"] = [asdict(c) for c in self.constraints_absolute()]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mission:
        m = cls(d["title"], d.get("statement", ""), [Variable(**v) for v in d["variables"]],
                [Objective(**o) for o in d["objectives"]],
                [Limit(**{k: v for k, v in c.items() if k != "text"}) for c in d.get("constraints", [])],
                dict(d.get("baseline", {})), Budget(**d.get("budget", {})), dict(d.get("acceptance", {})),
                dict(d.get("conditions", {})), list(d.get("hypotheses", [])), d.get("domain", ""))
        if "baseline_outputs" in d:
            m.resolve(d["baseline_outputs"])
        return m
