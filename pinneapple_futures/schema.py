"""The schema of a speculative system: what a proposal says, what it rests on, and how far to trust it.

A :class:`Proposal` is the five stages of the Lab written down as data -- speculation, formalization, modeling,
experimentation, assessment -- so that an imagined system becomes a set of checkable statements instead of a
paragraph. Nothing in it is free text alone:

* every number is an :class:`Input` with its provenance (a constant, a measured value, a design choice, a derived
  value or a hypothesis) and its source, because a number without provenance cannot be told apart from an
  invention;
* every law it leans on is a :class:`Claim` with a plausibility class, because "the equations of this story" and
  "the equations of the world" are different things;
* every hypothesis carries what would falsify it, because a statement that cannot be contradicted cannot be
  investigated.

The plausibility classes, weakest link first:

===============  =================================================================================
``established``  compatible with laws and phenomena that are well established
``extrapolated`` a known model extended to conditions that are still poorly studied
``hypothetical`` depends on physics that has not been demonstrated
``fictional``    introduces laws or mechanisms with no known scientific grounding
===============  =================================================================================

A proposal is only as good as its weakest claim: :meth:`Proposal.classification` returns that one, and never a
better one.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

PLAUSIBILITY_CLASSES = ("established", "extrapolated", "hypothetical", "fictional")
_RANK = {c: i for i, c in enumerate(PLAUSIBILITY_CLASSES)}

PROVENANCE_KINDS = ("constant", "measured", "design_input", "derived", "hypothesis")

STAGES = ("speculation", "formalization", "modeling", "experimentation", "assessment")

EARTH_GRAVITY = 9.80665
STANDARD_PRESSURE = 101325.0
STANDARD_TEMPERATURE = 288.15

__all__ = ["PLAUSIBILITY_CLASSES", "PROVENANCE_KINDS", "STAGES", "EARTH_GRAVITY", "STANDARD_PRESSURE",
           "STANDARD_TEMPERATURE", "Input", "Claim", "Derived", "Scenario", "Hypothesis", "Proposal",
           "classification", "weakest"]


def _check_plausibility(name: str, cls: str) -> str:
    if cls not in PLAUSIBILITY_CLASSES:
        raise ValueError(f"{name}: plausibility {cls!r} not in {PLAUSIBILITY_CLASSES}")
    return cls


def weakest(classes: Iterable[str]) -> str:
    """The weakest of the classes given (``fictional`` when the iterable is empty: nothing supports it)."""
    got = [c for c in classes]
    if not got:
        return "fictional"
    for c in got:
        _check_plausibility("weakest", c)
    return max(got, key=lambda c: _RANK[c])


@dataclass(frozen=True)
class Input:
    """A number a proposal uses, with where it comes from.

    ``provenance`` says what kind of number it is, and the five kinds are not interchangeable:

    * ``constant`` -- a fundamental constant, cited (``"CODATA 2022"``);
    * ``measured`` -- a value measured on something real, cited;
    * ``design_input`` -- a choice of the designer (a thickness, a target temperature); it needs a reason but no
      measurement;
    * ``derived`` -- computed by the proposal from other inputs, with the formula in ``source``;
    * ``hypothesis`` -- a number that stands for the speculative part (the conductivity of a material that does not
      exist yet). A ``hypothesis`` input makes every result built on it hypothetical at best.
    """

    name: str
    value: float
    unit: str
    provenance: str
    source: str

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_KINDS:
            raise ValueError(f"input {self.name!r}: provenance {self.provenance!r} not in {PROVENANCE_KINDS}")
        if not self.source:
            raise ValueError(f"input {self.name!r}: a number needs its source")
        if not self.unit:
            raise ValueError(f"input {self.name!r}: a number without a unit is not a physical quantity")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": float(self.value), "unit": self.unit,
                "provenance": self.provenance, "source": self.source}


@dataclass(frozen=True)
class Claim:
    """A law or a mechanism the proposal leans on, with its epistemic class and its source."""

    statement: str
    law: str = ""
    plausibility: str = "established"
    source: str = ""

    def __post_init__(self) -> None:
        _check_plausibility("claim", self.plausibility)
        if self.plausibility != "established" and not self.source:
            raise ValueError(f"claim {self.statement!r}: an unestablished claim needs the source it leans on")

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "law": self.law, "plausibility": self.plausibility,
                "source": self.source}


@dataclass(frozen=True)
class Derived:
    """A scenario parameter converted to a working unit, with how it was converted."""

    name: str
    value: float
    unit: str
    how: str

    def __str__(self) -> str:
        return f"{self.name} = {self.value:.6g} {self.unit} ({self.how})"


@dataclass
class Scenario:
    """The physical conditions of a possible world -- the boundary conditions of everything modelled in it.

    Only the quantities the Lab actually uses are named fields; anything else lives in ``inputs``. The conversions
    the interface shows are :meth:`derived`. They are conversions, not a planetary model: changing ``gravity``
    changes the weight of a thing of a given mass, and nothing else is claimed by the conversion itself.
    """

    name: str
    title: str = ""
    description: str = ""
    gravity: float = EARTH_GRAVITY                 # m/s^2
    ambient_temperature: float = STANDARD_TEMPERATURE   # K
    ambient_pressure: float = STANDARD_PRESSURE    # Pa
    inputs: tuple[Input, ...] = ()
    claims: tuple[Claim, ...] = ()

    def __post_init__(self) -> None:
        if self.gravity <= 0:
            raise ValueError("a scenario needs a positive surface gravity")
        if self.ambient_temperature <= 0:
            raise ValueError("an ambient temperature is absolute (K) and must be positive")
        if self.ambient_pressure < 0:
            raise ValueError("an ambient pressure cannot be negative")

    # ------------------------------------------------------------------ conversions
    def derived(self) -> dict[str, Derived]:
        """The parameter conversions shown next to the scenario (``relative_weight``, ``gravity_si``, ...)."""
        g_ratio = self.gravity / EARTH_GRAVITY
        return {
            "gravity_si": Derived("surface gravity", self.gravity, "m/s^2", "as declared"),
            "relative_weight": Derived("relative weight", g_ratio, "x Earth", "g / 9.80665 m/s^2"),
            "temperature_c": Derived("reference temperature", self.ambient_temperature - 273.15, "degC",
                                     "T[K] - 273.15"),
            "pressure_kpa": Derived("ambient pressure", self.ambient_pressure / 1e3, "kPa", "p[Pa] / 1000"),
        }

    def input(self, name: str) -> Input:
        for i in self.inputs:
            if i.name == name:
                return i
        raise KeyError(f"scenario {self.name!r} has no input {name!r}")

    def with_gravity(self, gravity: float, *, source: str = "") -> Scenario:
        """The same world at another surface gravity (a copy; the original is not touched)."""
        claim = Claim(f"the scenario is evaluated at g = {gravity:.6g} m/s^2", "g",
                      "extrapolated", source or "speculative change of the scenario")
        return Scenario(self.name, self.title, self.description, float(gravity), self.ambient_temperature,
                        self.ambient_pressure, self.inputs, self.claims + (claim,))

    def with_temperature(self, kelvin: float, *, source: str = "") -> Scenario:
        claim = Claim(f"the scenario is evaluated at T = {float(kelvin):.6g} K", "T",
                      "extrapolated", source or "speculative change of the scenario")
        return Scenario(self.name, self.title, self.description, self.gravity, float(kelvin),
                        self.ambient_pressure, self.inputs, self.claims + (claim,))

    # ------------------------------------------------------------------ epistemics
    def classification(self) -> str:
        """The weakest class among the scenario's claims (``established`` when it makes none)."""
        return weakest([c.plausibility for c in self.claims]) if self.claims else "established"

    def hypothetical_inputs(self) -> list[Input]:
        return [i for i in self.inputs if i.provenance == "hypothesis"]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description,
                "gravity": self.gravity, "ambient_temperature": self.ambient_temperature,
                "ambient_pressure": self.ambient_pressure,
                "inputs": [i.to_dict() for i in self.inputs], "claims": [c.to_dict() for c in self.claims],
                "classification": self.classification(),
                "derived": {k: {"value": v.value, "unit": v.unit, "how": v.how}
                            for k, v in self.derived().items()}}


#: A scenario used when the caller only wants to demonstrate the machinery.
EARTH = Scenario("earth", "Earth, present day", "Sea level, standard atmosphere.",
                 inputs=(Input("g_earth", EARTH_GRAVITY, "m/s^2", "constant", "standard gravity (CGPM, 1901)"),
                         Input("T_earth", STANDARD_TEMPERATURE, "K", "constant",
                               "standard atmosphere 15 degC, ISO 2533"),
                         Input("p_earth", STANDARD_PRESSURE, "Pa", "constant", "standard atmosphere, ISO 2533")),
                 claims=(Claim("Newtonian gravity at the surface", "F = m g", "established",
                               "classical mechanics"),))


@dataclass(frozen=True)
class Hypothesis:
    """A statement the proposal makes about the world, and the observation that would prove it wrong."""

    statement: str
    phenomenon: str = ""
    falsified_by: str = ""
    plausibility: str = "extrapolated"

    def __post_init__(self) -> None:
        _check_plausibility("hypothesis", self.plausibility)
        if not self.falsified_by:
            raise ValueError(f"hypothesis {self.statement!r}: say what would falsify it")

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "phenomenon": self.phenomenon,
                "falsified_by": self.falsified_by, "plausibility": self.plausibility}


@dataclass
class Proposal:
    """A speculative system written down as the five stages of the Lab, and the experiment that investigates it.

    ``experiment`` names a registered :class:`pinneapple_futures.core.SpeculativeExperiment`; :meth:`missing`
    lists the stages that are not filled in yet, and :meth:`run` refuses to run while anything is missing --
    an incomplete proposal is a question, not an experiment.
    """

    title: str
    speculation: str
    scenario: Scenario
    hypotheses: tuple[Hypothesis, ...] = ()
    equations: tuple[str, ...] = ()
    phenomena: tuple[str, ...] = ()
    experiment: str = ""
    question: str = ""
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ stages
    def stages(self) -> dict[str, str]:
        """One line per stage: what the proposal has for it, or what it still needs."""
        return {
            "speculation": self.speculation,
            "formalization": "; ".join(self.equations) or "MISSING: the equations",
            "modeling": f"experiment {self.experiment}: " + ", ".join(self.phenomena) if self.experiment
                        else "MISSING: the experiment that models it",
            "experimentation": "run the experiment and compare with its references" if self.experiment
                               else "MISSING: nothing to run",
            "assessment": "classify the physics and score the evidence" if not self.missing()
                          else "MISSING: " + ", ".join(self.missing()),
        }

    def missing(self) -> list[str]:
        """The stages that cannot be filled in yet, in order."""
        out = []
        if not self.equations:
            out.append("formalization (no equation)")
        if not self.hypotheses:
            out.append("formalization (no falsifiable hypothesis)")
        if not self.experiment:
            out.append("modeling (no registered experiment)")
        return out

    @property
    def complete(self) -> bool:
        return not self.missing()

    # ------------------------------------------------------------------ epistemics
    def classification(self) -> str:
        """The weakest link: the scenario's claims and the hypotheses together."""
        return weakest([self.scenario.classification()] + [h.plausibility for h in self.hypotheses])

    def inputs(self) -> list[Input]:
        return list(self.scenario.inputs)

    def run(self, *, quick: bool = False, **params):
        """Run the experiment the proposal names, with the proposal's scenario and hypotheses attached."""
        if not self.experiment:
            raise ValueError(f"proposal {self.title!r} has nothing to run: {', '.join(self.missing())}")
        from .core import get

        return get(self.experiment).run_with(self, quick=quick, **params)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "question": self.question, "speculation": self.speculation,
                "scenario": self.scenario.to_dict(), "hypotheses": [h.to_dict() for h in self.hypotheses],
                "equations": list(self.equations), "phenomena": list(self.phenomena),
                "experiment": self.experiment, "stages": self.stages(), "missing": self.missing(),
                "classification": self.classification(), "notes": list(self.notes), "extra": self.extra}

    def __str__(self) -> str:
        lines = [f"{self.title}  [{self.classification()}]"]
        for name, text in self.stages().items():
            lines.append(f"  {name:14s} {text}")
        return "\n".join(lines)


def inputs_of(values: Sequence[Input]) -> dict[str, float]:
    """``{name: value}`` from a sequence of inputs (the numbers a run can use)."""
    return {i.name: float(i.value) for i in values}


def classification(*items: Any) -> str:
    """The weakest epistemic class of anything with a ``classification()`` method, a ``plausibility`` attribute or
    a class name in :data:`PLAUSIBILITY_CLASSES` -- the one number that cannot be averaged."""
    found = []
    for it in items:
        if isinstance(it, str) and it in PLAUSIBILITY_CLASSES:
            found.append(it)
            continue
        fn = getattr(it, "classification", None)
        if callable(fn):
            found.append(fn())
            continue
        if hasattr(it, "plausibility"):
            found.append(getattr(it, "plausibility"))
            continue
        raise TypeError(f"{it!r} has no epistemic class")
    return weakest(found)
