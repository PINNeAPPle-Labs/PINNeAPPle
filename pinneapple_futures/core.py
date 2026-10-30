"""What a speculative experiment is, and what its run returns.

The Futures Lab is an application layer of PINNeAPPle, not a fork of it: an experiment here is a
:class:`pinneapple_science.core.Experiment` (same declaration of area, level, kind and dimensions; same
:class:`~pinneapple_science.core.Result` with its comparisons, physics checks and trust report), and a result is
that same result plus the two things the Lab adds -- the epistemic class of the proposal behind it and the
:class:`~pinneapple_futures.plausibility.Assessment` of its evidence.

    import pinneapple_futures as F
    exp = F.get("futures.exoplanet_laboratory")
    scenario = F.Scenario("super_earth", gravity=2.5 * F.schema.EARTH_GRAVITY)
    res = exp.run_with(scenario, quick=True)
    print(res)                     # comparisons, physics checks, trust decision and the assessment panel
    print(res.assessment.headline())   # "physics SUPPORTED | engineering NOT ASSESSED | class extrapolated"

An experiment that declares ``scenario_params`` accepts those fields of a scenario as parameters; anything else in
the scenario is reported as not used rather than silently ignored -- a parameter the equations do not contain
cannot be tested by changing it.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from pinneapple_science.core import Experiment, Result, jsonable

from .plausibility import Assessment, assess
from .schema import Proposal, Scenario

#: modules that define experiments (each class with a ``name`` registers itself through :func:`register`)
MODULES = ("pinneapple_futures.labs.mars_habitat", "pinneapple_futures.labs.programmable_matter",
           "pinneapple_futures.labs.exoplanet")

_REGISTRY: dict[str, type[SpeculativeExperiment]] = {}
_loaded = False

__all__ = ["SpeculativeResult", "SpeculativeExperiment", "register", "get", "names", "experiments", "run",
           "MODULES"]


class SpeculativeResult(Result):
    """A science result plus the epistemic class and the evidence assessment.

    The assessment is built when the trust report is built, so it can never describe a result that was not
    checked; ``proposal`` is the proposal behind the run, when there was one.
    """

    classification: str = ""
    proposal: dict[str, Any] | None = None
    assessment: Assessment | None = None

    # ------------------------------------------------------------------ evidence
    def build_trust(self) -> Any:
        trust = super().build_trust()
        self.assessment = assess(self, epistemic=self.classification)
        return trust

    def engineering(self, name: str, passed: bool | None, *, value: float | None = None,
                    limit: float | None = None, detail: str = "") -> bool | None:
        """Record a technological-feasibility check (the axis a simulation otherwise cannot fill)."""
        return self.check(name, passed, value=value, limit=limit, detail=detail, source="engineering")

    def bench(self, name: str, passed: bool | None, *, value: float | None = None,
              limit: float | None = None, detail: str = "") -> bool | None:
        """Record a check against a measurement (the axis only a measurement can fill)."""
        return self.check(name, passed, value=value, limit=limit, detail=detail, source="bench")

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["schema"] = "pinneapple.futures.result/1"
        d["classification"] = self.classification
        d["proposal"] = self.proposal
        d["assessment"] = None if self.assessment is None else self.assessment.to_dict()
        return d

    def __repr__(self) -> str:
        base = super().__repr__()
        if self.assessment is None:
            return base
        return base + "\n" + str(self.assessment)


class SpeculativeExperiment(Experiment):
    """An experiment over a hypothetical system: a science experiment that also knows its scenario."""

    result_cls = SpeculativeResult
    #: scenario fields this experiment's equations actually contain (``("gravity", "ambient_temperature", ...)``)
    scenario_params: tuple[str, ...] = ()
    #: the questions this experiment can turn into a run (used by the What If Engine)
    answers: tuple[str, ...] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        bad = [p for p in cls.scenario_params
               if p not in ("gravity", "ambient_temperature", "ambient_pressure")]
        if bad:
            raise ValueError(f"{cls.name}: unknown scenario parameters {bad}")

    # ------------------------------------------------------------------ running a proposal
    def run(self, dim: str | None = None, *, quick: bool = False, **params) -> SpeculativeResult:
        """Run in the experiment's own world, and classify the result by what that world rests on."""
        res = super().run(dim, quick=quick, **params)
        if not res.classification:
            world = self.world()
            res.classification = world.classification() if world is not None else "fictional"
            res.build_trust()
        return res

    def run_with(self, proposal: Proposal, *, quick: bool = False, **params) -> SpeculativeResult:
        """Run in the proposal's scenario, carrying its hypotheses and classification into the result."""
        sc = proposal.scenario
        supplied = {p: getattr(sc, p) for p in self.scenario_params}
        clash = sorted(set(supplied) & set(params))
        if clash:
            raise TypeError(f"{self.name}: {clash} came from both the proposal's scenario and the call")
        res = super().run(quick=quick, **{**supplied, **params})
        res.classification = proposal.classification()
        res.proposal = proposal.to_dict()
        res.build_trust()
        return res

    def accepts(self, scenario: Scenario) -> dict[str, float]:
        """The parameters this experiment takes from ``scenario`` (and nothing else)."""
        return {p: float(getattr(scenario, p)) for p in self.scenario_params}

    def unused(self, scenario: Scenario) -> list[str]:
        """The scenario fields that differ from Earth's and that this experiment's equations do not contain."""
        out = []
        if "gravity" not in self.scenario_params and abs(scenario.gravity - 9.80665) > 1e-9:
            out.append("gravity")
        if "ambient_temperature" not in self.scenario_params and abs(scenario.ambient_temperature - 288.15) > 1e-9:
            out.append("ambient_temperature")
        if "ambient_pressure" not in self.scenario_params and abs(scenario.ambient_pressure - 101325.0) > 1e-9:
            out.append("ambient_pressure")
        return out

    def card(self) -> dict[str, Any]:
        d = super().card()
        d["scenario_params"] = list(self.scenario_params)
        d["answers"] = list(self.answers)
        world = self.world()
        d["scenario"] = None if world is None else world.to_dict()
        return jsonable(d)

    def world(self) -> Scenario | None:
        """The default world this experiment runs in, with the provenance of every number it uses."""
        return None


# ---------------------------------------------------------------------------------------------- registry
def register(cls: type[SpeculativeExperiment]) -> type[SpeculativeExperiment]:
    """Class decorator: add an experiment to the Futures catalogue."""
    if not cls.name:
        raise ValueError("a futures experiment needs a name")
    if not cls.name.startswith("futures."):
        raise ValueError(f"{cls.name!r}: a futures experiment is named 'futures.<something>'")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"experiment {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def _load() -> None:
    global _loaded
    if not _loaded:
        for m in MODULES:
            importlib.import_module(m)
        _loaded = True


def get(name: str) -> SpeculativeExperiment:
    _load()
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown futures experiment {name!r}; known: {', '.join(sorted(_REGISTRY))}") from None


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def experiments(area: str | None = None) -> list[dict[str, Any]]:
    """Cards of the registered experiments, optionally only those of one atlas area."""
    _load()
    out = []
    for n in sorted(_REGISTRY):
        e = _REGISTRY[n]
        if area and not (e.area == area or e.area.startswith(area + ".") or e.area.split(".")[-1] == area):
            continue
        out.append(e().card())
    return out


def run(name: str, *, quick: bool = False, **params) -> SpeculativeResult:
    """Run one experiment in its first dimension, with parameters over its defaults."""
    return get(name).run(quick=quick, **params)


def matching(parameter: str) -> list[str]:
    """The experiments that can investigate ``parameter``.

    An experiment qualifies when it names the parameter in ``answers`` or when its equations contain the scenario
    field the parameter maps to -- the difference between a claim that something is investigated and a model that
    actually contains it.
    """
    _load()
    field = {"surface_temperature": "ambient_temperature", "pressure": "ambient_pressure"}.get(parameter, parameter)
    return sorted(n for n, e in _REGISTRY.items()
                  if parameter in e.answers or field in e.scenario_params)


def finite(x: Any) -> float:
    """The float, refusing anything that is not a finite number (a result is never NaN)."""
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"expected a finite number, got {x!r}")
    return v
