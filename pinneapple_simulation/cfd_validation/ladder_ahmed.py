"""The second family of the ladder (#294): variants of the Ahmed body, and what each mesh rung costs and gets wrong.

:mod:`~pinneapple_simulation.cfd_validation.ladder` measures the ladder on a NACA 0012 section. That family answers
"is the cheap rung a stand-in for the fine one" for an *airfoil*; this module asks the same question of the family the
repository actually runs a campaign on — the Ahmed body, whose slant angle is the design variable — because a rung
that ranks sections right need not rank bluff bodies right.

Why this family and not the wings
---------------------------------
The issue names two candidates: the airliner wings of ``apps/aero_optimizer`` and the Ahmed body's variants (#241). The
wings family's fine rung is a 3-D RANS of a whole aircraft (about 0.4 M cells, roughly 12 min a design on four cores),
and the repository holds committed 3-D solutions for **two** aircraft — two points cannot be rank-correlated, and
re-running enough designs to rank was not possible on the machine this was measured on (see :data:`NOT_MEASURED`).
The Ahmed body's variants are already solved, on one machine, with two meshes for the same six designs, with the wall
``Cp`` of every run committed: that is a cost-accuracy table that can be *read* rather than promised.

The rungs
---------
The rungs are levels of the same case at fewer cells, as in the first family: the campaign mesh (``L1``, 212 k cells,
the mesh the design-of-experiments runs on) and the fine mesh (``L2``, 534 k cells, the reference). The grid-convergence
family has four members (``L0`` to ``L3``), but ``L0`` and ``L3`` were solved at the baseline design only, and one
design cannot be ranked: a rung of a *ranking* needs every design, so they appear in the cost-accuracy curve at a
single design and not in the table (:attr:`MESH_STUDY`).

Error against the fine rung, and against the experiment
------------------------------------------------------
``CD_error_rel`` and ``CL_error_abs`` are against the fine rung, as in the first family: the question is which rung
can replace the expensive one, not whether the expensive one is right. The experiment is in the table too, as a
separate column (``CD_error_rel_vs_experiment``): Ahmed, Ramm & Faltin (1984) measured the drag of these same six slant
angles, so the distance from the measurement is available for every rung — and it is the column that says whether the
mesh refinement alone can close the gap. The published values are a transcription whose provenance the study records
value by value in :data:`pinneapple_autolab.domains.ahmed.MEASUREMENTS`; three of the six are quoted in secondary
sources, three are the transcription only.

The ``Cp`` column
-----------------
Every run stores the skin pressure on the same 1751 template vertices, so the rungs are compared vertex by vertex with
no interpolation: ``Cp_error_rms`` = ``sqrt(mean(dCp^2))`` and ``Cp_error_max_abs`` = ``max(|dCp|)``, both in units of
the free-stream dynamic pressure, plus the region the largest difference falls in — a coarse mesh misses the slant's
suction peak long before it misses the drag, and naming the region is what makes the column actionable.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ladder import _stat, calibrate, spearman

__all__ = ["AhmedDesign", "SLANTS", "LEVELS", "MESH_STUDY", "NOT_MEASURED", "MULTIFIDELITY", "default_records",
           "read_rung", "ladder", "cp_error", "regions_of"]

#: The slant angles Ahmed, Ramm & Faltin (1984) measured at 60 m/s and 50 mm clearance: the family's designs.
SLANTS = (0.0, 12.5, 25.0, 30.0, 35.0, 40.0)

CLEARANCE, SPEED = 0.05, 60.0
DESIGN_ITERATIONS = 800

#: The rungs of the table, cheapest first. ``prefix`` is the id the campaign's records gave that mesh's runs; the cell
#: count is read from each record rather than written here, so the table cannot disagree with the run it is built from.
LEVELS = (
    {"name": "ahmed-L1", "mesh": "campaign", "prefix": "val"},
    {"name": "ahmed-L2", "mesh": "L2", "prefix": "valL2"},
)

#: The whole grid-convergence family at the baseline design (25 deg, 50 mm, 60 m/s, 1000 iterations), from the
#: campaign's probe runs: the cost-accuracy curve where only one design is needed, and the reason ``L0``/``L3`` are not
#: rungs of the table (they exist for the baseline alone).
MESH_STUDY = (
    {"rung": "ahmed-L0", "cells": 84223, "seconds": 132.1, "CD": 0.28899965610083705},
    {"rung": "ahmed-L1", "cells": 207080, "seconds": 448.4, "CD": 0.29656455281749244},
    {"rung": "ahmed-L2", "cells": 534472, "seconds": 1157.1, "CD": 0.2707438409326792},
    {"rung": "ahmed-L3", "cells": 1398073, "seconds": 6410.1, "CD": 0.2639794841992436},
)

#: What could not be measured here, and the number that says why. The native runs of the coarse mesh were started on
#: this machine and abandoned: OpenFOAM v2606 on 10 cores against the campaign's 4-core Linux box ran the L0 case at
#: 2.3 s an iteration under a load average of 29-39 (four other solver jobs from other sessions on the same machine)
#: against 0.13 s an iteration there, so six designs would take about three hours and the cost column would report the
#: machine's contention instead of the mesh. Mixing that column with the campaign's would invert the ordering.
NOT_MEASURED = {
    "ahmed-L0": "solved at the baseline design only; a native ladder on this machine ran at 2.3 s/iteration under "
                "load average 29-39 (campaign: 0.13 s/iteration), about 3 h for six designs, and its cost column "
                "would measure contention rather than the mesh",
    "ahmed-L3": "solved at the baseline design only, 6410 s for that one design",
    "vlm": "a lifting-surface method: the Ahmed body has no wing, and the wings family it belongs to "
           "(apps/aero_optimizer) holds committed 3-D solutions for two aircraft — too few to rank",
    "lbm": "a low-Reynolds method; this family is at Re = 4.2e6 on the length",
    "surrogate": "a fit to the campaign's 64 runs, not a solver rung: it does not solve the same case at fewer cells",
}

MULTIFIDELITY = {"quantity": "CD",
                 "levels": [{"name": "ahmed-L1", "cost": 274}, {"name": "ahmed-L2", "cost": 1157}],
                 "note": "wall seconds per design on the campaign's machine (4-core Xeon, OpenFOAM v1912), from the "
                         "measured table in examples/end_to_end/ahmed_body/results"}


@dataclass(frozen=True)
class AhmedDesign:
    """One point of the family: a slant angle, at the wind-tunnel clearance and speed of Ahmed et al. (1984)."""

    slant_deg: float
    clearance: float = CLEARANCE
    speed: float = SPEED

    def params(self) -> dict[str, float]:
        return {"slant_deg": float(self.slant_deg), "clearance": float(self.clearance), "speed": float(self.speed)}

    def to_dict(self) -> dict[str, Any]:
        return self.params()


def default_records() -> str:
    """The campaign's committed records (``results/cfd_runs``): the only place these runs were measured."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "examples", "end_to_end", "ahmed_body", "results",
                                        "cfd_runs"))


def _environment(records: str) -> dict[str, Any]:
    """The machine and solver the runs were made on, as the campaign's own ``environment.json`` records it: every cost
    in the table belongs to that machine, and saying so is the difference between a cost column and a guess."""
    path = os.path.join(records, "environment.json")
    env = json.load(open(path)) if os.path.exists(path) else {}
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    try:                                    # a relative path in the JSON: an absolute one says where this was run
        where = os.path.relpath(os.path.abspath(records), root)
    except ValueError:
        where = records
    return {"records": records if where.startswith("..") else where,
            "cpu": env.get("cpu"), "cpu_count": env.get("cpu_count"),
            "platform": env.get("platform"), "python": env.get("python"),
            "openfoam": "v1912 (the campaign's image; see examples/end_to_end/ahmed_body/README.md)",
            "git_commit": env.get("git_commit")}


def regions_of(design: AhmedDesign) -> np.ndarray:
    """Which body region each template vertex belongs to (front, roof, slant, base, side, underbody)."""
    from pinneapple_autolab.domains.ahmed import field_vertices, region_of
    return region_of(field_vertices(design.params()), design.params())


def read_rung(records: str, level: Mapping[str, Any], design: AhmedDesign) -> dict[str, Any]:
    """One rung of one design, read back from the campaign's record and its committed field.

    The record is evidence: a value not in it was not measured. A missing record (or a failed one) returns the row with
    ``CD`` and ``Cp`` ``None`` and the reason, so the table can say that a rung was not measured instead of skipping it.
    """
    rid = f"{level['prefix']}_{design.slant_deg:04.1f}"
    path = os.path.join(records, "results", f"{rid}.json")
    if not os.path.exists(path):
        return {"rung": level["name"], "design": design.to_dict(), "CD": None, "CL": None, "Cp": None,
                "cells": 0, "seconds": 0.0, "status": "missing record", "error": f"no {rid}.json"}
    rec = json.load(open(path))
    out: dict[str, Any] = {"rung": level["name"], "design": design.to_dict(), "id": rid,
                           "CD": rec["outputs"].get("CD"), "CL": rec["outputs"].get("CL"),
                           "CD_pressure": rec["outputs"].get("CD_pressure"),
                           "CD_friction": rec["outputs"].get("CD_friction"),
                           "cells": int(rec["meta"].get("cells") or 0), "seconds": float(rec["cost_s"]),
                           "mesh": rec["meta"].get("mesh"),
                           # the rung is only comparable if it solved the same problem: the campaign's
                           # validation runs are all 800 iterations, and a record that is not says so
                           "iterations": rec["meta"].get("iterations"),
                           "same_iterations": rec["meta"].get("iterations") == DESIGN_ITERATIONS,
                           "CD_iteration_std": (rec["meta"].get("iteration_std") or {}).get("CD"),
                           "status": "measured" if rec["ok"] else "failed", "error": rec["message"]
                           if not rec["ok"] else None}
    field = os.path.join(records, "fields", f"{rid}.npz")
    out["Cp"] = np.load(field)["cp"] if os.path.exists(field) else None
    return out


def cp_error(reference: Mapping[str, Any], other: Mapping[str, Any], regions: np.ndarray | None = None) -> dict:
    """How far one rung's skin ``Cp`` is from the reference's, vertex by vertex on the shared template.

    Both rungs are sampled on the same template vertices, so no interpolation is needed and every vertex counts. The
    norm is the dimensionless one: ``rms`` = ``sqrt(mean(dCp^2))`` and ``max_abs`` = ``max(|dCp|)`` in units of the
    free-stream dynamic pressure; ``region`` names where the largest difference sits, and ``rms_by_region`` gives the
    same rms per body region, since a coarse mesh gets the flat faces right and the separated wake wrong.
    """
    a, b = reference.get("Cp"), other.get("Cp")
    if a is None or b is None or len(a) != len(b):
        return {"rms": None, "max_abs": None, "region": None, "rms_by_region": {}, "vertices": 0}
    d = np.asarray(b, float) - np.asarray(a, float)
    out: dict[str, Any] = {"rms": float(np.sqrt(np.mean(d ** 2))), "max_abs": float(np.max(np.abs(d))),
                           "region": None, "rms_by_region": {}, "vertices": int(d.size)}
    if regions is not None and len(regions) == d.size:
        out["region"] = str(regions[int(np.argmax(np.abs(d)))])
        out["rms_by_region"] = {str(r): float(np.sqrt(np.mean(d[regions == r] ** 2)))
                                for r in np.unique(regions) if int((regions == r).sum())}
    return out


def ladder(records: str | None = None, *, designs: Sequence[AhmedDesign] | None = None,
           levels: Sequence[Mapping[str, Any]] = LEVELS, log=print) -> dict[str, Any]:
    """The cost-accuracy table of the Ahmed family: every design on every rung, error against the finest, and rankings.

    Reads the campaign's committed records (this family is measured there, not here — see :data:`NOT_MEASURED`), builds
    the same table :func:`~pinneapple_simulation.cfd_validation.ladder.ladder` builds for the airfoil, and adds the
    experimental drag as a second reference: the column that tells whether the coarse rung's error is the refinement's
    or the model's.
    """
    from pinneapple_autolab.domains.ahmed import AHMED_1984_CD

    records = records or default_records()
    designs = list(designs or [AhmedDesign(s) for s in SLANTS])
    rows: list[dict[str, Any]] = []
    for d in designs:
        regions = regions_of(d)
        for level in levels:
            log(f"  {level['name']} slant {d.slant_deg:g} deg")
            row = read_rung(records, level, d)
            row["_regions"] = regions
            rows.append(row)
    finest = levels[-1]["name"]
    ref = {(r["design"]["slant_deg"], r["rung"]): r for r in rows if r["rung"] == finest}
    for r in rows:
        base = ref.get((r["design"]["slant_deg"], finest))
        regions = r.pop("_regions")                 # borrowed, not part of the row: a numpy array does not JSON
        cd, cd_ref = r.get("CD"), (base or {}).get("CD")
        r["CD_error_rel"] = None if cd is None or cd_ref in (None, 0) else float(abs(cd - cd_ref) / abs(cd_ref))
        r["CD_error_abs"] = None if cd is None or cd_ref is None else float(abs(cd - cd_ref))
        cl, cl_ref = r.get("CL"), (base or {}).get("CL")
        r["CL_error_abs"] = None if cl is None or cl_ref is None else float(abs(cl - cl_ref))
        r["Cp_error"] = cp_error(base, r, regions) if base else \
            {"rms": None, "max_abs": None, "region": None, "rms_by_region": {}, "vertices": 0}
        measured = AHMED_1984_CD.get(round(float(r["design"]["slant_deg"]), 1))
        r["CD_error_rel_vs_experiment"] = (None if cd is None or not measured
                                           else float(abs(cd - measured) / measured))
    table = []
    for level in levels:
        name = level["name"]
        mine = [r for r in rows if r["rung"] == name]
        ok = [r for r in mine if r.get("CD") is not None]
        entry: dict[str, Any] = {
            "rung": name, "mesh": level["mesh"], "designs": len(mine), "measured": len(ok),
            "cells": int(mine[0]["cells"]) if mine else 0,
            "iterations": mine[0].get("iterations") if mine else None,
            "same_definition": all(r.get("same_iterations") for r in ok),
            "seconds_per_design": round(float(np.mean([r["seconds"] for r in ok])), 1) if ok else None,
            "CD_error_rel": _stat([r.get("CD_error_rel") for r in ok]) if name != finest else 0.0,
            "CD_error_abs": _stat([r.get("CD_error_abs") for r in ok]) if name != finest else 0.0,
            "CL_error_abs": _stat([r.get("CL_error_abs") for r in ok]) if name != finest else 0.0,
            "Cp_error_rms": _stat([r["Cp_error"]["rms"] for r in ok]),
            "Cp_error_max_abs": max((r["Cp_error"]["max_abs"] for r in ok
                                     if r["Cp_error"]["max_abs"] is not None), default=None),
            "CD_iteration_std": _stat([r.get("CD_iteration_std") for r in ok]),
            "CD_error_rel_vs_experiment": _stat([r.get("CD_error_rel_vs_experiment") for r in ok]),
        }
        for q in ("CD", "CL"):
            pairs = [(r.get(q), ref[(r["design"]["slant_deg"], finest)].get(q)) for r in ok]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            entry[f"spearman_{q}"] = (spearman([a for a, _ in pairs], [b for _, b in pairs])
                                      if len(pairs) == len(ref) else float("nan"))
        table.append(entry)
    calib = {}
    for quantity in ("CD", "CL"):
        try:
            calib[quantity] = {
                "two designs (offset)": calibrate(rows, "ahmed-L1", quantity=quantity, designs=[25.0, 35.0],
                                                  reference=finest, key=lambda r: r["design"]["slant_deg"]),
                "all six (offset and slope)": calibrate(rows, "ahmed-L1", quantity=quantity, designs=list(SLANTS),
                                                        reference=finest, key=lambda r: r["design"]["slant_deg"]),
            }
        except (ValueError, KeyError) as exc:                   # a rung that was not measured cannot be calibrated
            calib[quantity] = {"error": f"{type(exc).__name__}: {exc}"}
    regions = {f"{r['rung']}@{r['design']['slant_deg']:g}": r["Cp_error"] for r in rows
               if r["rung"] != finest and r["Cp_error"]["rms"] is not None}
    return {"family": f"Ahmed body, Re = {SPEED * 1.044 / 1.5e-5:.2e} on the length, slant angle",
            "designs": [d.to_dict() for d in designs], "reference_rung": finest, "table": table,
            "rows": [{k: v for k, v in r.items() if k != "Cp"} for r in rows],
            "mesh_study": list(MESH_STUDY), "multifidelity": MULTIFIDELITY, "not_in_the_table": NOT_MEASURED,
            "calibration": calib, "cp_by_design": regions,
            "measured_on": _environment(records)}
