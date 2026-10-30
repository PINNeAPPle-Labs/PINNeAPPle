"""The atlas of the Futures Lab: the directions it works in, and the code behind each one.

Eight directions, each with subareas, a status per subarea (``available``, ``partial``, ``planned``) and pointers
to the code that covers it -- a registered experiment of this package or a module of the rest of PINNeAPPle. The
atlas points to existing code and never copies it: :func:`validate` checks every pointer against the repository, so
the page cannot claim a capability the code does not have.

    import pinneapple_futures as F
    at = F.atlas()
    print(at)                      # one line per direction
    at.to_markdown()               # the table of docs/guides/futures_lab.md
    assert F.validate() == []      # every pointer names code that exists

A direction may be ``planned`` with no code at all: the Lab is honest about what it has not built yet, and the
number of planned subareas is a research agenda, not a promise.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

STATUSES = ("available", "partial", "planned")
POINTER_KINDS = ("experiment", "module")

__all__ = ["Area", "Atlas", "load", "validate", "STATUSES", "POINTER_KINDS"]


@dataclass
class Area:
    """One direction of the Lab and its subareas."""

    id: str
    title: str
    description: str
    subareas: list[dict[str, Any]] = field(default_factory=list)

    def status(self) -> str:
        """The best status among the subareas (``planned`` when there is none)."""
        got = [s.get("status") for s in self.subareas]
        for st in ("available", "partial"):
            if st in got:
                return st
        return "planned"

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for s in self.subareas:
            out[s.get("status", "planned")] += 1
        return out

    def __str__(self) -> str:
        c = self.counts()
        return (f"{self.id:32s} {self.status():9s} "
                f"{c['available']} available, {c['partial']} partial, {c['planned']} planned")


@dataclass
class Atlas:
    areas: list[Area]

    def area(self, ident: str) -> Area:
        for a in self.areas:
            if a.id == ident or a.id.split(".")[-1] == ident:
                return a
        raise KeyError(f"no area {ident!r}; known: {', '.join(a.id for a in self.areas)}")

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for a in self.areas:
            for k, v in a.counts().items():
                out[k] += v
        return out

    def rows(self) -> list[dict[str, Any]]:
        return [{"area": a.id, "title": a.title, **s} for a in self.areas for s in a.subareas]

    def to_markdown(self) -> str:
        lines = ["| Direction | Subarea | Status | Covered by |", "|---|---|---|---|"]
        for r in self.rows():
            ptr = ", ".join(f"`{p['name']}`" if p["kind"] == "experiment" else f"`{p['module']}`"
                            for p in r.get("covered_by", []))
            lines.append(f"| {r['title']} | {r['name']} | {r['status']} | {ptr or '--'} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        head = f"Futures Lab atlas: {len(self.areas)} directions, {self.counts()}"
        return "\n".join([head] + [str(a) for a in self.areas])


def _exp(name: str) -> dict[str, str]:
    return {"kind": "experiment", "name": name}


def _mod(module: str) -> dict[str, str]:
    return {"kind": "module", "module": module}


#: the directions of the Lab, as data
_AREAS: list[dict[str, Any]] = [
    {"id": "speculative_physics", "title": "Speculative Physics",
     "description": "Physics formalised for conditions that do not exist here, checked against its own limits.",
     "subareas": [
         {"name": "equations for hypothetical systems", "status": "available",
          "covered_by": [_exp("futures.exoplanet_laboratory")],
          "note": "a world's parameters changed, with the analytic references that make the change checkable"},
         {"name": "plausibility classification", "status": "available",
          "covered_by": [_mod("pinneapple_futures.plausibility")],
          "note": "four epistemic classes and six evidence axes that are never averaged into one number"},
         {"name": "physics beyond the Standard Model", "status": "planned", "covered_by": []},
     ]},
    {"id": "future_engineering", "title": "Future Engineering",
     "description": "Technologies that do not exist yet, sized with the physics that does.",
     "subareas": [
         {"name": "habitats in hostile environments", "status": "available",
          "covered_by": [_exp("futures.mars_habitat")],
          "note": "envelope, heater budget and pressure hull, each against a closed form"},
         {"name": "propulsion thermodynamics", "status": "partial",
          "covered_by": [_mod("pinneapple_physics.closed_form.rocket_nozzle")],
          "note": "isothermal and equilibrium nozzle performance exist; nothing above the chemical limit does"},
         {"name": "space elevators and megastructures", "status": "planned", "covered_by": []},
         {"name": "submarine cities", "status": "planned", "covered_by": []},
     ]},
    {"id": "artificial_life", "title": "Artificial Life",
     "description": "Organisms, ecologies and pattern-forming systems that are designed rather than found.",
     "subareas": [
         {"name": "reaction-diffusion and morphogenesis", "status": "partial",
          "covered_by": [_mod("pinneapple_physics.pde_environment")],
          "note": "the PDE machinery is there; a worked futures experiment is not"},
         {"name": "population dynamics", "status": "partial",
          "covered_by": [_mod("pinneapple_simulation.numerical_solvers")],
          "note": "ODE solvers and SIR-style presets exist in the library"},
         {"name": "evolution in alternative environments", "status": "planned", "covered_by": []},
     ]},
    {"id": "planetary_systems", "title": "Planetary Systems",
     "description": "Worlds with other gravities, atmospheres and rotations.",
     "subareas": [
         {"name": "radiative equilibrium and climate balance", "status": "available",
          "covered_by": [_exp("futures.exoplanet_laboratory")],
          "note": "the Legendre solution of the diffusive energy-balance model, reproduced by the run"},
         {"name": "atmospheric columns and scale heights", "status": "available",
          "covered_by": [_exp("futures.exoplanet_laboratory")],
          "note": "where gravity enters, measured against ISO 2533 on Earth"},
         {"name": "orbital dynamics of hypothetical systems", "status": "partial",
          "covered_by": [_mod("pinneapple_physics.astro")],
          "note": "two-body with J2, conjunction screening and CW relative motion are in the library"},
     ]},
    {"id": "advanced_materials", "title": "Advanced Materials",
     "description": "Materials whose properties are functions, not constants.",
     "subareas": [
         {"name": "conductivity that depends on the field", "status": "available",
          "covered_by": [_exp("futures.programmable_matter")],
          "note": "k(T) solved against its implicit closed form, with the point where it stops conducting"},
         {"name": "graded metamaterials", "status": "available",
          "covered_by": [_exp("futures.programmable_matter")],
          "note": "k(x) against the series-resistance integral"},
         {"name": "self-healing and adaptive structures", "status": "planned", "covered_by": []},
     ]},
    {"id": "post_biological_systems", "title": "Post-Biological Systems",
     "description": "Systems whose substrate is not biology but whose organisation is.",
     "subareas": [
         {"name": "substrate-independent information processing", "status": "planned", "covered_by": []},
         {"name": "artificial metabolism and closed loops", "status": "planned", "covered_by": []},
     ]},
    {"id": "computational_worldbuilding", "title": "Computational Worldbuilding",
     "description": "A world built from its laws: constants, matter and conditions, and the phenomena that follow.",
     "subareas": [
         {"name": "scenario composition and parameter dependence", "status": "available",
          "covered_by": [_mod("pinneapple_futures.whatif")],
          "note": "the What If Engine says which equations contain the parameter, instead of pretending all do"},
         {"name": "world-scale coupling of physics models", "status": "partial",
          "covered_by": [_mod("pinneapple_systems.cosimulation")],
          "note": "the co-simulation graph exists; no world has been assembled from it here"},
     ]},
    {"id": "civilization_scale_engineering", "title": "Civilization-Scale Engineering",
     "description": "Energy, water, transport and industry modelled together, with their limits.",
     "subareas": [
         {"name": "plant-scale energy and heat integration", "status": "partial",
          "covered_by": [_mod("pinneapple_systems.operations")],
          "note": "heat exchangers, plants and control are in the library"},
         {"name": "infrastructure interdependence", "status": "planned", "covered_by": []},
     ]},
]


def load() -> Atlas:
    """The atlas, built from the data above."""
    return Atlas([Area(**copy.deepcopy({k: v for k, v in d.items()})) for d in _AREAS])


def _module_exists(module: str) -> bool:
    """Whether the module or package is in the repository (checked by path, so nothing is imported)."""
    parts = module.split(".")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, *parts)
    return os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))


def validate(atlas: Atlas | None = None) -> list[str]:
    """Every pointer against the code: returns the problems found (an empty list is a good atlas)."""
    at = atlas or load()
    from .core import names

    registered = set(names())
    problems: list[str] = []
    seen: set[str] = set()
    for area in at.areas:
        if area.id in seen:
            problems.append(f"{area.id}: duplicated direction")
        seen.add(area.id)
        for sub in area.subareas:
            if sub.get("status") not in STATUSES:
                problems.append(f"{area.id}/{sub['name']}: status {sub.get('status')!r} not in {STATUSES}")
            pointers = sub.get("covered_by", [])
            if sub.get("status") != "planned" and not pointers:
                problems.append(f"{area.id}/{sub['name']}: {sub.get('status')} with no code behind it")
            for ptr in pointers:
                kind = ptr.get("kind")
                if kind not in POINTER_KINDS:
                    problems.append(f"{area.id}/{sub['name']}: pointer kind {kind!r} not in {POINTER_KINDS}")
                elif kind == "experiment" and ptr.get("name") not in registered:
                    problems.append(f"{area.id}/{sub['name']}: no experiment {ptr.get('name')!r}")
                elif kind == "module" and not _module_exists(ptr["module"]):
                    problems.append(f"{area.id}/{sub['name']}: no module {ptr['module']!r}")
    # the other direction: an experiment nobody points at is a capability the page would hide
    pointed = {p["name"] for a in at.areas for s in a.subareas for p in s.get("covered_by", [])
               if p.get("kind") == "experiment"}
    for name in sorted(registered - pointed):
        problems.append(f"experiment {name!r} is registered but no direction of the atlas points at it")
    return problems
