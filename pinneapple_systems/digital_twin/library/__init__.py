"""The Digital Twin Library: reference models of engineering systems, one per domain.

What this package is
--------------------
Each module here is a *reference model* of a real system written to be read, run and checked
in seconds: FPSO and subsea production, a blast furnace, a chemical plant, a reusable
booster, a lunar base, a gas pipeline network, a compression station, a green hydrogen plant and a
desalination train. They share :class:`~.base.ReferenceModel`, so every one of them takes
SI parameters with declared ranges, returns a machine-readable :class:`~.base.TwinRun` with
its conservation balances and its invariants, and states the limits of its own physics.

Where a twin goes after this
----------------------------
The fidelity ladder of the Digital Twin Library issues is 0D -> 1D -> 2D/3D -> trained
surrogate -> commissioning. This package is the bottom rung, and it is the rung the others
need: a model cheap enough to run in CI is also the model that generates the solver data a
surrogate is trained on, and the one a
:class:`~pinneapple_systems.digital_twin.product.spec.TwinSpec` can wrap to be commissioned
against measured history.

Use
---
>>> from pinneapple_systems.digital_twin.library import catalogue, run, describe
>>> [entry.slug for entry in catalogue()]
['blast_furnace', 'chemical_plant', 'compression_station', 'desalination_plant', 'fpso_subsea',
 'green_hydrogen_plant', 'lunar_base', 'pipeline_network', 'reusable_booster']
>>> result = run("fpso_subsea")               # the declared default scenario
>>> result.ok
True
>>> print(result.report())
>>> print(describe("fpso_subsea"))            # parameters, assumptions, limits, sources
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import (
    Balance,
    BalanceFailed,
    Check,
    LibraryError,
    Parameter,
    ParameterError,
    Quantity,
    ReferenceModel,
    TwinRun,
    close,
    relative_error,
)
from .blast_furnace import BlastFurnace
from .chemical_plant import ChemicalPlant
from .compression_station import CompressionStation
from .desalination import DesalinationPlant
from .fpso import FPSOSubsea
from .green_hydrogen import GreenHydrogenPlant
from .lunar_base import LunarBase
from .pipeline_network import PipelineNetwork
from .reusable_booster import ReusableBooster

__all__ = [
    # contract
    "ReferenceModel", "TwinRun", "Parameter", "Quantity", "Balance", "Check",
    "LibraryError", "ParameterError", "BalanceFailed", "close", "relative_error",
    # catalogue
    "TWINS", "TwinEntry", "catalogue", "build", "run", "describe", "compare",
    # systems
    "FPSOSubsea", "BlastFurnace", "ChemicalPlant", "ReusableBooster", "LunarBase",
    "PipelineNetwork", "CompressionStation", "GreenHydrogenPlant", "DesalinationPlant",
]

#: Every system in the library, by slug.
TWINS: dict[str, type[ReferenceModel]] = {
    FPSOSubsea.slug: FPSOSubsea,
    BlastFurnace.slug: BlastFurnace,
    ChemicalPlant.slug: ChemicalPlant,
    ReusableBooster.slug: ReusableBooster,
    LunarBase.slug: LunarBase,
    PipelineNetwork.slug: PipelineNetwork,
    CompressionStation.slug: CompressionStation,
    GreenHydrogenPlant.slug: GreenHydrogenPlant,
    DesalinationPlant.slug: DesalinationPlant,
}


class TwinEntry:
    """One row of the catalogue: what the system is, what it solves, and what it does not."""

    __slots__ = ("slug", "title", "physics", "fidelity", "status", "model")

    def __init__(self, model: type[ReferenceModel], *, fidelity: str = "0D/1D",
                 status: str = "reference model") -> None:
        self.slug = model.slug
        self.title = model.title
        self.physics = model.physics
        self.fidelity = fidelity
        self.status = status
        self.model = model

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug, "title": self.title, "physics": self.physics,
            "fidelity": self.fidelity, "status": self.status,
            "n_parameters": len(self.model.parameters),
            "limits": list(self.model.limits),
        }

    def __repr__(self) -> str:
        return f"TwinEntry({self.slug!r}, {self.fidelity}, {self.status!r})"


#: Fidelity and status per system. Kept beside the registry rather than in each module so the
#: ladder is visible in one place: what is a 0D reference model today, what is planned next.
_LADDER = {
    "fpso_subsea": ("0D/1D", "reference model; 1D transient and multiphase simulator comparison planned"),
    "blast_furnace": ("0D", "reference model; shaft profile and wustite equilibrium are the next rung"),
    "chemical_plant": ("0D", "reference model; kinetics and a rate-based reactor are the next rung"),
    "reusable_booster": ("3-DOF", "reference model; six-degree-of-freedom entry is the next rung"),
    "lunar_base": ("0D", "reference model; a transient thermal and power simulation is the next rung"),
    "pipeline_network": ("1D steady", "reference model; transient linepack and a multiphase mixture are the next rung"),
    "compression_station": ("0D", "reference model; a vendor map with speed lines and surge margin is the next rung"),
    "green_hydrogen_plant": ("0D", "reference model; a polarisation curve and a stack thermal model are the next rung"),
    "desalination_plant": ("0D", "reference model; a membrane transport model with fouling is the next rung"),
}


def catalogue() -> list[TwinEntry]:
    """Every system in the library, in slug order, with its fidelity and status."""
    entries = []
    for slug, model in sorted(TWINS.items()):
        fidelity, status = _LADDER.get(slug, ("0D", "reference model"))
        entries.append(TwinEntry(model, fidelity=fidelity, status=status))
    return entries


def build(slug: str) -> ReferenceModel:
    """Instantiate one system by slug (the model is stateless; this is for ``describe``)."""
    try:
        model = TWINS[slug]
    except KeyError as exc:
        raise LibraryError(f"no twin {slug!r} in the library; it has {sorted(TWINS)}") from exc
    return model()


def run(slug: str, values: Mapping[str, Any] | None = None, *, strict: bool = True,
        **kwargs: Any) -> TwinRun:
    """Run one system: ``run("fpso_subsea", production_days=30)``."""
    return build(slug).simulate(values, strict=strict, **kwargs)


def describe(slug: str) -> str:
    """The parameter table, assumptions, limits and sources of one system, as text."""
    return build(slug).describe()


def compare(slugs: list[str] | None = None, *, values: Mapping[str, Any] | None = None) -> str:
    """Run several systems on their declared defaults and tabulate the verdict of each."""
    rows = []
    for slug in (slugs or sorted(TWINS)):
        result = run(slug, values)
        rows.append((slug, "PASS" if result.ok else "FAIL",
                     f"{len(result.balances)} balances", f"{len(result.checks)} invariants"))
    width = max(len(r[0]) for r in rows) if rows else 0
    return "\n".join(f"{slug:<{width}s}  {verdict:<4s}  {b}, {c}" for slug, verdict, b, c in rows)
