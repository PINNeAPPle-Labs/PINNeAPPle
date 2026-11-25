"""The registry: every twin in the library, by slug, with its fidelity and what it answers.

This is the discoverability half of the shared-twin-specification issue (#1802): a catalogue a
reader can print, a ``build``/``run`` pair that is the whole surface, and a ladder that records what
each system is today and what the next rung above it is meant to be.

It lives in its own module so that :mod:`.scenarios` can name twins without importing the package
that imports it — a registry that can only be reached through the package it registers is a
circular import waiting for the first module that needs to enumerate it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import LibraryError, ReferenceModel, TwinRun
from .blast_furnace import BlastFurnace
from .carbon_capture import CarbonCapturePlant
from .chemical_plant import ChemicalPlant
from .compression_station import CompressionStation
from .data_center import DataCentre
from .desalination import DesalinationPlant
from .fpso import FPSOSubsea
from .green_hydrogen import GreenHydrogenPlant
from .lunar_base import LunarBase
from .pipeline_network import PipelineNetwork
from .refrigeration import RefrigerationPlant
from .reusable_booster import ReusableBooster

__all__ = ["TWINS", "TwinEntry", "catalogue", "build", "run", "describe", "compare"]

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
    RefrigerationPlant.slug: RefrigerationPlant,
    DataCentre.slug: DataCentre,
    CarbonCapturePlant.slug: CarbonCapturePlant,
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
    "industrial_refrigeration": ("0D", "reference model; a real property database and a compressor map are the next rung"),
    "data_center_cooling": ("0D", "reference model; the hall's air distribution needs CFD, which is the next rung"),
    "carbon_capture_plant": ("0D", "reference model; absorber kinetics and a column model are the next rung"),
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
