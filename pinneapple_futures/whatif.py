"""The What If Engine: a speculative question turned into a structured experiment, or into an honest refusal.

The engine does not draw a picture and it does not ask a language model. It reads the question, writes down what
changed in the world, and then does the only thing that makes the answer scientific: it says **which of the
equations the Lab can solve actually contain the parameter that changed**. That is what separates "gravity is
2.5 times stronger" from an answer -- the equilibrium temperature of a planet does not contain ``g`` at all, so no
amount of simulation of it will respond to gravity, while the hydrostatic scale height contains ``g`` in its
denominator and responds to nothing else.

    import pinneapple_futures as F
    brief = F.ask("What if Earth's gravity were 2.5 times stronger?")
    print(brief)                    # what changed, which equations contain it, which experiments can run it
    result = brief.run(quick=True)  # the experiment, with the scenario attached, checked as always

The eight steps of the plan (:meth:`Brief.stages`) are the ones the proposal goes through: the scenario, the
phenomena, the models, the parameters and boundary conditions, the construction, the run, the comparison with a
reference, and the report of the results and their limits. When the Lab has no experiment for a question,
:attr:`Brief.understood` is ``False``, the plan names what is missing, and :meth:`Brief.run` raises rather than
inventing a number.

The parameter conversions the interface shows (:meth:`Brief.derived`) are conversions: 2.5 g is 24.52 m/s^2, and
that is arithmetic, not a planetary model.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .core import matching
from .schema import EARTH_GRAVITY, Claim, Derived, Hypothesis, Proposal, Scenario

__all__ = ["Brief", "ask", "RULE_FAMILIES", "DEPENDENCIES", "Changeword"]


@dataclass(frozen=True)
class Changeword:
    """One way of saying "this parameter changed", with the regular expression that reads it."""

    parameter: str
    pattern: str
    convert: Callable[[float], float]
    unit: str
    phrase: str


_NUMBER = r"(-?\d+(?:[.,]\d+)?)"


def _num(text: str) -> float:
    return float(text.replace(",", "."))


#: which equations contain each parameter, and what happens to them. This table is the engine's actual answer:
#: a parameter that appears in no equation cannot be investigated by changing it.
DEPENDENCIES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "gravity": (
        ("hydrostatic column", "H = R T / (M g),  p(z) = p0 exp(-z/H)", "H is inversely proportional to g"),
        ("weight and structural load", "W = m g", "every mass weighs g/g_earth times as much"),
        ("buoyancy and natural convection", "Ra = g beta dT L^3 / (nu alpha)", "the Rayleigh number is linear in g"),
        ("radiative equilibrium", "T_eq = [S (1-a)/(4 sigma)]^(1/4)", "g does not appear: no effect at all"),
    ),
    "surface_temperature": (
        ("radiative equilibrium", "T_eq = [S (1-a)/(4 sigma)]^(1/4)", "the equilibrium is a fourth root of the flux, "
                                                                       "not of the temperature"),
        ("scale height", "H = R T / (M g)", "H is proportional to T"),
        ("conduction", "q = k dT/dx", "the change of a boundary temperature changes the whole field, linearly"),
    ),
    "pressure": (
        ("scale height", "H = R T / (M g)", "pressure does not enter H, only the column mass does"),
        ("column mass", "m_col = p0 / g", "the mass of atmosphere above a point is proportional to p0/g"),
        ("convection in a thin atmosphere", "h ~ k/L", "at low pressure the films stop carrying heat altogether"),
    ),
    "insolation": (
        ("radiative equilibrium", "T_eq = [S (1-a)/(4 sigma)]^(1/4)", "the equilibrium temperature goes as S^(1/4)"),
        ("latitudinal transport", "D d/dx[(1-x^2) dT/dx] - (A + B T) + S(x)(1-a) = 0", "S scales the forcing"),
    ),
    "albedo": (
        ("radiative equilibrium", "T_eq = [S (1-a)/(4 sigma)]^(1/4)", "the absorbed fraction is (1 - a)"),
    ),
}

#: the question families the engine reads, in the order it tries them
RULE_FAMILIES: tuple[Changeword, ...] = (
    Changeword("gravity", rf"gravit(?:y|ies)\s*(?:were|was|is|of|were)?\s*{_NUMBER}\s*(?:times|time|×|x)\s*"
                           rf"(?:stronger|higher|weaker|lower|earth|terrestrial|as (?:strong|large))?",
               lambda v: v * EARTH_GRAVITY, "m/s^2", "times Earth's gravity"),
    Changeword("gravity", rf"gravit(?:y|ies)\s*(?:were|was|is|of)?\s*{_NUMBER}\s*m\s*/\s*s",
               float, "m/s^2", "in m/s^2"),
    Changeword("gravity", rf"gravit(?:y|ies)\s*(?:were|was|is|of)?\s*{_NUMBER}\s*g\b",
               lambda v: v * EARTH_GRAVITY, "m/s^2", "in g"),
    Changeword("gravity", rf"gravidade\s*(?:fosse|for|é|de)?\s*{_NUMBER}\s*(?:vezes|×|x)\s*"
                          rf"(?:maior|menor|mais forte|mais fraca)?",
               lambda v: v * EARTH_GRAVITY, "m/s^2", "vezes a gravidade da Terra"),
    Changeword("insolation", rf"(?:insolation|solar flux|irradiance|fluxo solar)\s*(?:were|was|is|of)?\s*"
                              rf"{_NUMBER}\s*(?:w\s*/\s*m\s*\^?2|times)?",
               float, "W/m^2", "in W/m^2"),
    Changeword("albedo", rf"albedo\s*(?:were|was|is|of)?\s*{_NUMBER}",
               float, "-", "as a fraction"),
    Changeword("surface_temperature", rf"(?:temperature|temperatura)\s*(?:were|was|is|of|fosse)?\s*{_NUMBER}\s*"
                                      rf"(?:k|kelvin|degrees? kelvin)",
               float, "K", "in kelvin"),
    Changeword("pressure", rf"(?:pressure|press[ãa]o)\s*(?:were|was|is|of|fosse)?\s*{_NUMBER}\s*kpa",
               lambda v: v * 1000.0, "Pa", "in kPa"),
    Changeword("pressure", rf"(?:pressure|press[ãa]o)\s*(?:were|was|is|of|fosse)?\s*{_NUMBER}\s*pa",
               float, "Pa", "in Pa"),
)


@dataclass
class Brief:
    """A question read into a scenario, a plan and (when the Lab can answer it) an experiment."""

    question: str
    scenario: Scenario
    parameter: str = ""
    changes: dict[str, Derived] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    phenomena: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    experiments: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------ reading
    @property
    def understood(self) -> bool:
        """Whether the engine read the change at all (a question it cannot parse is not answered)."""
        return bool(self.changes)

    @property
    def answerable(self) -> bool:
        """Whether a registered experiment's equations contain the parameter that changed."""
        return bool(self.experiments)

    def dependencies(self) -> tuple[tuple[str, str, str], ...]:
        """Which equations contain the parameter, and what happens to each."""
        return DEPENDENCIES.get(self.parameter, ())

    # ------------------------------------------------------------------ the plan
    def stages(self) -> dict[str, str]:
        """The eight steps of the experiment this question would create."""
        params = ", ".join(f"{d.name} = {d.value:.6g} {d.unit}" for d in self.changes.values())
        return {
            "1 scenario": self.scenario.title or self.scenario.name,
            "2 phenomena": ", ".join(self.phenomena) or "not identified",
            "3 models": "; ".join(self.models) or "no equation of the Lab contains this parameter",
            "4 parameters": params or "nothing changed",
            "5 experiment": ", ".join(self.experiments) or "none registered",
            "6 run": (f"{self.experiments[0]} with the scenario attached" if self.answerable
                      else "nothing to run: no registered experiment uses this parameter"),
            "7 comparison": ("against the closed forms the experiment carries" if self.answerable
                             else "nothing to compare"),
            "8 assessment": "classify the physics and score the evidence, then state the limits",
        }

    def proposal(self) -> Proposal:
        """The proposal behind the question, ready to be run."""
        exp = self.experiments[0] if self.experiments else ""
        hypotheses = tuple(Hypothesis(
            f"{phenomenon}: {effect}", phenomenon=phenomenon,
            falsified_by=f"a run where {law} does not respond as stated", plausibility="established")
            for phenomenon, law, effect in self.dependencies())
        return Proposal(
            title=self.question.rstrip("?"),
            question=self.question,
            speculation=self.scenario.description or self.scenario.title or self.scenario.name,
            scenario=self.scenario, hypotheses=tuple(hypotheses),
            equations=tuple(law for _, law, _ in self.dependencies()),
            phenomena=self.phenomena, experiment=exp,
            notes=self.notes + self.warnings)

    def run(self, *, quick: bool = False, **params) -> Any:
        """Run the first experiment that can answer the question, in the question's scenario."""
        if not self.answerable:
            raise ValueError(
                f"no registered Futures experiment uses {self.parameter or 'this parameter'!r}; the Lab can "
                f"investigate: {', '.join(sorted(DEPENDENCIES))} (see the atlas for what is planned)")
        return self.proposal().run(quick=quick, **{**self.params, **params})

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "understood": self.understood, "answerable": self.answerable,
                "parameter": self.parameter, "scenario": self.scenario.to_dict(),
                "changes": {k: {"name": v.name, "value": v.value, "unit": v.unit, "how": v.how}
                            for k, v in self.changes.items()},
                "params": dict(self.params),
                "phenomena": list(self.phenomena), "models": list(self.models),
                "experiments": list(self.experiments), "stages": self.stages(),
                "warnings": list(self.warnings), "notes": list(self.notes)}

    def __str__(self) -> str:
        lines = [f"{self.question}", ""]
        if not self.understood:
            lines.append("  the change was not understood; the Lab reads gravity, temperature, pressure, "
                         "insolation and albedo")
        for d in self.changes.values():
            lines.append(f"  {d.name:26s} {d.value:>10.6g} {d.unit:6s} {d.how}")
        deps = self.dependencies()
        if deps:
            lines.append("")
            lines.append("  which equations contain it:")
            for phenomenon, law, effect in deps:
                lines.append(f"    {phenomenon:32s} {law}")
                lines.append(f"    {'':32s} -> {effect}")
        lines.append("")
        lines.append(f"  experiments that can run it: {', '.join(self.experiments) or 'none'}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the engine
#: parameters that live in the model rather than in the scenario: they are handed to the experiment directly
_MODEL_PARAMETERS = ("insolation", "albedo")


def ask(question: str) -> Brief:
    """Read a speculative question into a scenario and a plan.

    It reads the change, converts it, and looks for the equations and the experiments that contain the parameter.
    It never guesses: a question it cannot parse comes back with :attr:`Brief.understood` ``False`` and the reason.
    """
    if not question or not question.strip():
        raise ValueError("a question is needed")
    text = question.lower().strip()
    for rule in RULE_FAMILIES:
        m = re.search(rule.pattern, text)
        if not m:
            continue
        raw = _num(m.group(1))
        value = rule.convert(raw)
        exps = tuple(matching(rule.parameter))
        base = _world_of(exps[0]) if exps else Scenario("question", "A world the Lab has no code for")
        scenario, derived = _scenario_for(rule.parameter, value, question, base)
        deps = DEPENDENCIES.get(rule.parameter, ())
        params = {rule.parameter: value} if rule.parameter in _MODEL_PARAMETERS else {}
        warnings: list[str] = []
        if not exps:
            warnings.append(f"the change to {rule.parameter} is understood, but no registered experiment has "
                            f"an equation that contains it")
        if rule.parameter == "gravity":
            warnings.append("gravity does not enter the radiative equilibrium of a planet: a world can be heavier "
                            "without being hotter")
        if params:
            notes = (f"read as {raw:g} {rule.phrase}, converted to {value:g} {rule.unit}, and handed to the "
                     f"experiment as the model parameter {rule.parameter!r}: the scenario itself is unchanged",)
        else:
            notes = (f"read as {raw:g} {rule.phrase}, converted to {value:g} {rule.unit} and written into the "
                     f"scenario",)
        return Brief(question=question.strip(), scenario=scenario, parameter=rule.parameter,
                     changes={rule.parameter: derived}, params=params,
                     phenomena=tuple(p for p, _, _ in deps), models=tuple(law for _, law, _ in deps),
                     experiments=exps, warnings=tuple(warnings), notes=notes)
    return Brief(question=question.strip(), scenario=Scenario("unread", "A question the Lab could not read"),
                 warnings=("the change was not understood",),
                 notes=(f"the Lab reads: {', '.join(sorted(DEPENDENCIES))}",))


def _world_of(experiment: str) -> Scenario:
    """The world of the experiment that will run the question, so that only the asked parameter changes."""
    from .core import get

    world = get(experiment).world()
    return world if world is not None else Scenario("question", "The world the question describes")


def _scenario_for(parameter: str, value: float, question: str, base: Scenario) -> tuple[Scenario, Derived]:
    base = Scenario(base.name, base.title, question, base.gravity, base.ambient_temperature,
                    base.ambient_pressure, base.inputs,
                    base.claims + (Claim(f"the question changes {parameter}", parameter, "extrapolated",
                                         "a speculative change, by construction"),))
    if parameter == "gravity":
        return base.with_gravity(value, source="the question"), \
            Derived("surface gravity", value, "m/s^2", f"{value / EARTH_GRAVITY:.4g} g")
    if parameter == "surface_temperature":
        return base.with_temperature(value, source="the question"), \
            Derived("reference temperature", value, "K", f"{value - 273.15:.4g} degC")
    if parameter == "pressure":
        sc = Scenario(base.name, base.title, base.description, base.gravity, base.ambient_temperature, value,
                      base.inputs, base.claims)
        return sc, Derived("ambient pressure", value, "Pa", f"{value / 1000.0:.4g} kPa")
    if parameter == "insolation":
        return base, Derived("insolation", value, "W/m^2", "a change of the stellar flux")
    return base, Derived("albedo", value, "-", "a change of the reflected fraction")
