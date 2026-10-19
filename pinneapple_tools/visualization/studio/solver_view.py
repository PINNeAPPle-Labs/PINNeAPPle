"""The result of a solver run as a 3-D view, to check it before its numbers are used: the mesh, its fields, the
deformed shape (CalculiX displacement, with a scale factor) and the regions where the case defines its boundary
conditions and loads.

    from pinneapple_tools.visualization.studio.solver_view import solver_result_view
    fv, info = solver_result_view("runs/r1/c01.solve")     # a folder with <job>.frd (+ <job>.inp), or an OpenFOAM case
    fv.glb(), fv.meta                                       # for /shared/fieldview3d.js
    info["conditions"]                                      # what was marked, from the deck or the case files

* **CalculiX:** the ``.frd`` (nodes, elements, last increment of each result: DISP, STRESS as von Mises, ...) and,
  from the ``.inp`` next to it, ``*BOUNDARY`` (supports) and ``*CLOAD`` / ``*DLOAD`` (loads) as marked node sets.
* **OpenFOAM:** the polyMesh, the volume fields of the latest time (vectors as magnitude) and every non-empty patch
  with its conditions from the first time directory, all as boundary conditions. A patch with a non-zero fixed vector
  value (a moving wall, an inlet) is the one that drives the flow: its role says so, and a note gives its value next
  to the cell values the colours show (a wall value is not a cell value).

Above ``max_faces`` triangles the surface is decimated (``VIEW_MAX_FACES``, from the size guide in
docs/core_concepts/studio.md); the description says so.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .fieldview import VIEW_MAX_FACES, FieldView, displacement_factor, field_view

__all__ = ["solver_result_view", "find_result"]


def find_result(folder: str | Path) -> tuple[str, Path] | None:
    """``("calculix", <job>.frd)`` or ``("openfoam", <case dir>)`` for a step folder (or the case itself), else None."""
    d = Path(folder)
    for case in (d, d / "case"):
        if (case / "system").is_dir() and (case / "constant" / "polyMesh").is_dir():
            return "openfoam", case
    frd = sorted(d.glob("*.frd"))
    if frd:
        return "calculix", frd[0]
    return None


def _files(case: Path) -> dict[str, bytes]:
    skip = re.compile(r"(^|/)(processor\d+|postProcessing|VTK|dynamicCode)(/|$)")
    return {p.relative_to(case).as_posix(): p.read_bytes() for p in sorted(case.rglob("*"))
            if p.is_file() and not skip.search(p.relative_to(case).as_posix()) and not p.name.startswith("log.")}


def _calculix(frd: Path) -> tuple[Any, list, dict, list, dict]:
    from pinneapple_data.cae.frd import read_frd
    from pinneapple_data.cae.inp import deck_conditions, read_deck
    m = read_frd(frd.read_bytes())
    fields, deform = [], {}
    for name, a in m.point_data.items():
        a = np.asarray(a, float)
        if a.ndim == 2 and a.shape[1] == 4:                  # x, y, z and a magnitude slot (often empty)
            a = a[:, :3]
        comps = 1 if a.ndim == 1 else a.shape[1]
        if not np.isfinite(a).any():
            continue
        lab = {"DISP": "displacement", "STRESS": "von Mises stress"}.get(name, name)
        if comps == 6 and name != "STRESS":
            lab += " (von Mises equivalent)"
        if name == "DISP" or displacement_factor(name, "", comps):
            deform["displacement"] = a[:, :3]
            lab += " magnitude"
        fields.append({"name": name, "values": a, "label": lab, "deform": "displacement"})
    for f in fields:
        if not deform:
            f["deform"] = None
    conditions, markers = [], []
    inp = frd.with_suffix(".inp")
    if inp.is_file():
        ids = np.asarray(m.source.get("node_ids", np.arange(1, m.n_points + 1)), np.int64)
        lut = {int(n): i for i, n in enumerate(ids)}
        for c in deck_conditions(read_deck(inp.read_text(errors="replace"), inp.name)):
            idx = [lut[n] for n in c["node_ids"] if n in lut]
            conditions.append({k: v for k, v in c.items() if k != "node_ids"} | {"nodes": len(idx)})
            if idx:
                markers.append({"name": c["name"], "kind": c["kind"], "points": m.points[idx],
                                "detail": f"{c['keyword']} (line {c['line']}): {c['detail']}"})
    info = {"solver": "CalculiX", "file": frd.name, "deck": inp.name if inp.is_file() else None,
            "time": m.source.get("time"), "length_unit": "model units"}
    return m, fields, deform, markers, {"conditions": conditions, **info}


def _openfoam(case: Path) -> tuple[Any, list, dict, list, dict]:
    from pinneapple_data.cae import foam
    from pinneapple_data.cae.io import read_any
    fs = _files(case)
    m = read_any(fs, fields=True)
    fields = []
    for name, a in m.cell_data.items():
        a = np.asarray(a, float)
        fields.append({"name": name, "values": a, "where": "cell", "unit": m.units.get(name, ""),
                       "label": name + (" magnitude" if a.ndim == 2 else "")})
    bcs = foam.boundary_conditions(fs, m)
    p = m.poly
    conditions, markers, notes = [], [], []
    for pa in p.patches:
        if pa["name"] not in bcs:
            continue
        cond = bcs[pa["name"]]
        driving = {}                                             # field -> magnitude of a non-zero fixed vector
        for k, v in cond.items():
            if re.match(r"fixedValue uniform \(", v):
                vec = np.array([float(x) for x in re.findall(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?", v.split("(", 1)[1])])
                if np.any(vec):
                    driving[k] = float(np.linalg.norm(vec))
        kind, role = "bc", ("driving " + ("wall" if pa["type"] == "wall" else "boundary")) if driving else ""
        for k, mag in driving.items():
            top = float(np.nanmax(np.linalg.norm(np.atleast_2d(m.cell_data[k]).reshape(m.n_cells, -1), axis=1))) \
                if k in m.cell_data else None
            notes.append(f"Colours are cell-centre values: the {pa['name']} {role} has |{k}| = {mag:.4g}"
                         f"{' ' + m.units[k] if m.units.get(k) else ''} (its fixed value), above the largest cell value"
                         f"{f' {top:.4g}' if top is not None else ''}." if top is not None and top < mag else
                         f"{pa['name']} ({role}): |{k}| = {mag:.4g} (its fixed value).")
        f = np.arange(pa["startFace"], pa["startFace"] + pa["nFaces"])
        P = np.array([m.points[p.face_nodes[p.face_offsets[i]:p.face_offsets[i + 1]]].mean(0) for i in f]) \
            if len(f) else np.zeros((0, 3))
        detail = "; ".join(f"{k}: {v}" for k, v in cond.items())
        conditions.append({"name": pa["name"], "kind": kind, "role": role, "type": pa["type"],
                           "faces": int(pa["nFaces"]), "detail": detail})
        if len(P):
            markers.append({"name": f"{pa['name']} ({role})" if role else pa["name"], "kind": kind, "points": P,
                            "detail": f"{pa['type']} patch: {detail}"})
    empty = [pa["name"] for pa in p.patches if pa["type"] == "empty"]
    info = {"solver": m.source.get("application") or "OpenFOAM", "file": case.name, "time": m.source.get("time"),
            "length_unit": "m", "conditions": conditions,
            "notes": ([f"2-D case (empty patch {', '.join(empty)}): one of its two sides is shown"] if empty else [])
            + notes}
    return m, fields, {}, markers, info


def solver_result_view(folder: str | Path, *, max_faces: int | None = VIEW_MAX_FACES
                       ) -> tuple[FieldView, dict[str, Any]]:
    """View of the CalculiX or OpenFOAM result in ``folder`` (see the module docstring). Raises ``ValueError`` when
    there is no result to show."""
    found = find_result(folder)
    if found is None:
        raise ValueError("no CalculiX .frd or OpenFOAM case in this folder")
    kind, path = found
    mesh, fields, deform, markers, info = (_calculix if kind == "calculix" else _openfoam)(path)
    if not fields:
        raise ValueError(f"the {info['solver']} result has no field to show")
    fv = field_view(mesh, fields, deform=deform, markers=markers, max_faces=max_faces, title=info["file"],
                    length_unit="" if info["length_unit"] == "model units" else info["length_unit"],
                    notes=info.get("notes", []))
    info.update(kind=kind, points=int(mesh.n_points), cells=int(mesh.n_cells), fields=[f["name"] for f in fields])
    return fv, info
