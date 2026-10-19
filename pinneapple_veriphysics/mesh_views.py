"""3-D views of the reference comparisons of a Validation Lab run: for every result with a reference whose data is 3-D
and lives on a mesh (OpenFOAM case, VTU/VTK with cells), the prediction, the reference and the pointwise error on the
mesh, the regions with the largest error and the declared boundary conditions.

    from pinneapple_veriphysics.mesh_views import mesh_views
    for v in mesh_views(workdir, report=lab_dict):        # after run_validation(workdir)
        v["view"].glb(), v["view"].meta                    # pp.viz.field_view: for /shared/fieldview3d.js
        v["errors"]["U"]["hotspots"]                       # largest-error regions: centre, extent, max, share

The values are paired exactly as ``pinneapple check`` pairs them for PC701 (same files, same time, the reference
mapped to the result's points by nearest neighbour when the two differ), so the relative L2 shown is the PC701 number.
2-D results, point clouds without cells, tables and models get no 3-D view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["mesh_views"]


def _box_tags(lo, hi, dmin, dmax, tol: float = 0.01) -> list[str]:
    """Faces of the domain's bounding box that a region touches ("x=min", "z=max", ...)."""
    size = np.maximum(np.asarray(dmax) - np.asarray(dmin), 1e-300)
    out = []
    for k, a in enumerate("xyz"):
        if (lo[k] - dmin[k]) <= tol * size[k]:
            out.append(f"{a}=min")
        if (dmax[k] - hi[k]) <= tol * size[k]:
            out.append(f"{a}=max")
    return out


def _geometry(fd, path: Path):
    """(cae mesh, where, locations of the mesh values) for a result, or None when it has no cells."""
    if fd.kind == "fv":
        return fd.mesh, "cell", np.asarray(fd.geom["cell_centres"], float)
    from pinneapple_data.cae.io import read_meshio
    m = read_meshio(path.name, path.read_bytes())
    if not m.blocks:
        return None
    if m.point_data:
        return m, "point", np.asarray(m.points, float)
    cent = np.array([m.points[c].mean(0) for b in m.blocks for c in b.conn])
    return m, "cell", cent


def _bc_markers(t, fd, mesh, locs) -> list[dict[str, Any]]:
    """Declared boundary conditions: a patch of an OpenFOAM case (its face centres) or a plane "x=min" of the data."""
    out = []
    for key, vals in (t.bc or {}).items():
        detail = ", ".join(f"{k} = {v}" for k, v in vals.items()) if isinstance(vals, dict) else str(vals)
        P = None
        if fd.kind == "fv" and mesh.poly is not None:
            pa = next((p for p in mesh.poly.patches if p["name"] == key), None)
            if pa is not None:
                f = np.arange(pa["startFace"], pa["startFace"] + pa["nFaces"])
                offs, flat = mesh.poly.face_offsets, mesh.poly.face_nodes
                P = np.array([mesh.points[flat[offs[i]:offs[i + 1]]].mean(0) for i in f]) if len(f) else None
        elif "=" in key and key.split("=")[0].strip() in ("x", "y", "z"):
            a, v = key.replace(" ", "").split("=", 1)
            c = locs[:, "xyz".index(a)]
            target = c.min() if v == "min" else c.max() if v == "max" else float(v)
            sel = np.abs(c - target) <= 1e-6 * max(float(np.ptp(c)), 1e-300)
            P = locs[sel] if sel.any() else None
        if P is not None and len(P):
            out.append({"name": key, "kind": "bc", "points": P, "detail": f"declared: {detail}"})
    return out


def mesh_views(workdir: str | Path, config: str | Path | None = None, *, report: dict[str, Any] | None = None,
               max_faces: int | None = None, n_regions: int = 8, mesh_unit: str = "m") -> list[dict[str, Any]]:
    """One view per 3-D result target with a reference: ``{"target", "view" (FieldView), "errors" {field: error map
    dict}, "fields", "interpolated", "notes"}``. ``report``: the lab report, whose PC701 values are quoted next to the
    pointwise numbers. ``mesh_unit``: the length unit of the coordinates (displacements are drawn in it)."""
    from scipy.spatial import cKDTree

    from pinneapple_analysis.validation.error_map import error_map, pointwise_error
    from pinneapple_check import load_config
    from pinneapple_check.checks.results import _nearest
    from pinneapple_check.config import ResultTarget
    from pinneapple_check.fields import LoadError, load_result
    from pinneapple_tools.visualization.studio.fieldview import (
        VIEW_MAX_FACES,
        displacement_factor,
        field_view,
    )

    from .validation_lab import CONFIG_NAME

    root = Path(workdir).resolve()
    cfg = load_config(Path(config) if config else root / CONFIG_NAME, root=root)
    pc701 = {(o["target"], o["subject"]): o.get("value") for o in (report or {}).get("outcomes", [])
             if o.get("rule") == "PC701"}
    out = []
    for t in cfg.targets:
        if not isinstance(t, ResultTarget) or t.reference is None or not Path(t.path).exists():
            continue
        try:
            fd = load_result(Path(t.path), t.kind, t.time, None)
            rd = load_result(Path(t.reference.path), None, t.time, None)
        except (LoadError, OSError, ValueError):
            continue                                          # PC001 / PC703 already report it
        if fd.kind not in ("fv", "points", "grid") or fd.dim != 3 or "t" in fd.axes:
            continue
        geo = _geometry(fd, Path(t.path))
        if geo is None:
            continue
        mesh, where, locs = geo
        X = fd.coords()
        dist, order = cKDTree(X).query(locs)                  # mesh value i <- data value order[i]
        if len(X) != len(locs) or float(dist.max()) > 1e-9 * max(float(np.ptp(locs, axis=0).max()), 1e-300):
            continue
        names = [n for n in (t.reference.fields or [n for n in fd.fields if n in rd.fields])
                 if n in fd.fields and n in rd.fields]
        fields, errors, markers, deform, interp, pairs = [], {}, [], {}, [], {}
        dmin, dmax = locs.min(0), locs.max(0)
        units = {**fd.units, **(t.units or {})}
        for n in names:                                       # the compared values, in the mesh's order
            a, b = fd.flat(n), rd.flat(n)
            if a.shape != b.shape:
                b = _nearest(rd, fd, n)
                interp.append(n)
            pairs[n] = (a[order], b[order])
            comps = a.shape[1] if a.ndim == 2 else 1
            fac = displacement_factor(n, str(units.get(n) or ""), comps, mesh_unit) if where == "point" else None
            if fac is not None and not deform:                # the first displacement deforms every field
                deform = {"prediction": pairs[n][0] * fac, "reference": pairs[n][1] * fac}
        dk = ("prediction", "reference") if deform else (None, None)
        for n, (a, b) in pairs.items():
            unit = str(units.get(n) or "")
            em = error_map(locs, a, b, n_hotspots=n_regions, unit=unit)
            v701 = pc701.get((t.name, n))
            d = em.to_dict({"PC701 in the report": v701} if v701 is not None else None)
            for k, h in enumerate(d["hotspots"]):
                h["name"] = f"R{k + 1}"
                h["touches"] = _box_tags(h["bbox_min"], h["bbox_max"], dmin, dmax)
            errors[n] = d
            mag = " magnitude" if a.ndim == 2 else ""
            keys = [f"{n}:pred", f"{n}:ref", f"{n}:error"]
            fields += [{"name": keys[0], "values": a, "where": where, "label": f"Prediction {n}{mag}", "unit": unit,
                        "group": n, "deform": dk[0]},
                       {"name": keys[1], "values": b, "where": where, "label": f"Reference {n}{mag}", "unit": unit,
                        "group": n, "deform": dk[1]},
                       {"name": keys[2], "values": pointwise_error(a, b), "where": where,
                        "label": f"|prediction − reference| {n}", "unit": unit, "kind": "error", "deform": dk[0],
                        "note": f"relative L2 {100 * em.rel_l2:.2f} %"}]
            for k, h in enumerate(em.hotspots):
                markers.append({"name": f"R{k + 1}", "kind": "hotspot", "points": locs[h["indices"]], "centre": h["at"],
                                "fields": keys, "detail": f"|error| up to {h['max']:.3g} {unit}",
                                "label": k < 3})              # the first three are labelled; all are in the table
        if not fields:
            continue
        markers = _bc_markers(t, fd, mesh, locs) + markers
        notes = [f"{n}: reference mapped to the result's points by nearest neighbour (as PC701)" for n in interp]
        fv = field_view(mesh, fields, deform=deform, markers=markers, title=t.name, length_unit=mesh_unit,
                        max_faces=VIEW_MAX_FACES if max_faces is None else max_faces, notes=notes)
        out.append({"target": t.name, "view": fv, "errors": errors, "fields": [f["name"] for f in fields],
                    "interpolated": interp, "notes": notes,
                    "reference": str(Path(t.reference.path).relative_to(root)) if Path(t.reference.path).is_relative_to(root) else str(t.reference.path)})
    return out
