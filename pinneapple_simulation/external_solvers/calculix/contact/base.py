"""The three things every contact case is made of — a parameter, a built deck and a case — plus the two readers the
checks share.

They live here, and not in :mod:`.cases`, because the library is now more than one file: :mod:`.cases` holds the
friction cases, :mod:`.roark_cases` holds the closed-form problems of Roark's Table 14.1, and both build the same
kind of object. ``cases`` re-exports these names, so ``from .cases import Case`` keeps working.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import deck as D
from . import verify as V
from .mesh import Grid


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    default: float
    unit: str
    low: float
    high: float
    help: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "default": self.default, "unit": self.unit,
                "low": self.low, "high": self.high, "help": self.help}


@dataclass
class Built:
    """A case built at one set of parameters: the deck, the closed-form reference and what the checks need to find
    their way back into the result files."""
    deck: str
    analytic: dict[str, float]
    coords: dict[int, tuple[float, float, float]]
    interface: tuple[int, ...]           # slave nodes: the ones that carry contact results
    totals: tuple[str, ...]              # node sets whose reaction totals are the measured forces
    frame: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: step times whose contact state the checks read (name -> time), for a case with more than one step
    times: dict[str, float] = field(default_factory=dict)
    #: anything else a case needs to find its way back (a preload, an interference)
    extra: dict[str, float] = field(default_factory=dict)
    #: node sets whose displacement the checks read (the loaded face, a free surface), name -> what it is
    surfaces: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    summary: str
    physics: str
    assumptions: tuple[str, ...]
    references: tuple[tuple[str, str], ...]      # (what it supports, url)
    params: tuple[Param, ...]
    build: Callable[[dict[str, float]], Built]
    checks: Callable[[dict[str, float], Built, dict[str, Any]], list[V.Check]]
    #: the test that runs this case against the solver, relative to the repository root
    test: str = "tests/test_prepomax.py"

    def param(self, name: str) -> Param:
        return next(p for p in self.params if p.name == name)

    def resolve(self, raw: dict[str, Any] | None) -> dict[str, float]:
        """The parameters of one run: the defaults, overridden by what was given, each inside its declared range."""
        given = {k: v for k, v in (raw or {}).items()}
        unknown = sorted(set(given) - {p.name for p in self.params})
        if unknown:
            raise ValueError(f"{self.id} has no parameter(s) {', '.join(unknown)}; "
                             f"its parameters are {', '.join(p.name for p in self.params)}")
        out: dict[str, float] = {}
        for p in self.params:
            v = given.get(p.name, p.default)
            try:
                v = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{p.name} must be a number, got {v!r}") from e
            if not math.isfinite(v) or not p.low <= v <= p.high:
                raise ValueError(f"{p.name} = {v:g} {p.unit} is outside the range this case is built for "
                                 f"({p.low:g} to {p.high:g} {p.unit})")
            out[p.name] = v
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "summary": self.summary, "physics": self.physics,
                "assumptions": list(self.assumptions),
                "references": [{"supports": s, "url": u} for s, u in self.references],
                "params": [p.as_dict() for p in self.params], "units": D.UNITS}


# ----------------------------------------------------------------------------------------------------- shared bits
def mesh_size(n: float) -> int:
    return max(2, int(round(n)))


def stack(deck: D.Deck, blocks: list[tuple[str, Grid]], material_name: str, E: float, nu: float) -> None:
    D.geometry(deck, blocks)
    D.material(deck, material_name, E, nu)
    for name, _ in blocks:
        D.section(deck, name, material_name)


def reaction(parsed: dict[str, Any], name: str) -> tuple[float, float, float] | None:
    totals = parsed.get("totals") or {}
    t = totals.get(name)
    return tuple(t.values) if t else None  # type: ignore[return-value]


def contact_of(parsed: dict[str, Any]) -> dict[int, dict[str, float]]:
    return parsed.get("contact") or {}
