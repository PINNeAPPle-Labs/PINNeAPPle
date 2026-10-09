"""Mesh health report: checks with a verdict, statistics and histograms, problem regions (clusters of bad cells, with
where they are) and what to do about each issue. Also the payload for a 3D view (boundary surface coloured by a
metric, plus the bad cells)."""
from __future__ import annotations

import string
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import geometry as G
from .model import Mesh
from .quality import METRICS, _hist, _stats, element_metrics, fv_metrics, surface_checks

ACTIONS = {
    "non_orthogonality": ("Add non-orthogonal correctors (nNonOrthogonalCorrectors 2-3) and a limited snGrad "
                          "(limited corrected 0.33); better, remesh the region: smoother size transitions, more "
                          "snapping iterations, or lower maxNonOrtho in meshQualityDict."),
    "skewness": ("Smooth the region (fewer abrupt size changes, better feature snapping); for snappyHexMesh raise "
                 "nSmoothPatch / nSolveIter and tighten maxBoundarySkewness / maxInternalSkewness."),
    "aspect_ratio": ("Check the stretching: very long cells are fine aligned with the flow in boundary layers, "
                     "not across gradients. Reduce the expansion ratio or refine in the long direction."),
    "volume": "The mesh is invalid: regenerate it. Look for inverted or collapsed cells in the marked region.",
    "scaled_jacobian": ("Optimise the mesh (Gmsh: Optimize / OptimizeNetgen; HighOrderOptimize for quadratic "
                        "elements), refine at sharp features and thin walls, avoid wedge-shaped slivers."),
    "element_skewness": ("Remesh or optimise the region; for tetrahedra, sliver removal (Gmsh Optimize, TetGen -q) "
                         "fixes most cases."),
    "edge_ratio": "Refine the long edges or use a structured/swept mesh in thin regions.",
    "regions": ("The mesh falls apart in disconnected pieces: stray bodies, nodes that were not merged, or parts that "
                "should be tied/contacted. Merge coincident nodes or define the interface (TIE, contact), or remove "
                "the stray parts."),
    "hinge": ("Elements touch only through an edge or a node: no load path through the face. Usually two parts "
              "meshed separately; merge them through a shared face."),
    "open_edges": ("The surface has holes: close them before volume meshing (snappyHexMesh needs a closed surface "
                   "to tell inside from outside)."),
    "nonmanifold_edges": "Edges shared by more than two faces: separate the bodies or remove internal faces.",
    "orientation": "Normals are inconsistent: re-orient the surface (e.g. surfaceOrient, MeshLab, Gmsh Reverse).",
    "degenerate_faces": "Remove zero-area triangles (surfaceClean, MeshLab 'remove zero-area faces').",
    "duplicate_faces": "Remove duplicated triangles.",
    "openness": "Cells are not closed: the face list is corrupt. Regenerate the mesh.",
    "face_pyramids": "Face pyramids with negative volume: badly warped or concave cells. Remesh the region.",
    "nonmanifold_faces": "Faces shared by more than two elements: overlapping elements. Remesh.",
}

FV_KEYS = ("non_orthogonality", "skewness", "aspect_ratio", "volume")
FE_KEYS = ("scaled_jacobian", "element_skewness", "edge_ratio")


def _status(v: float, spec: Dict[str, Any]) -> str:
    if spec["better"] == "low":
        if spec["fail"] is not None and v > spec["fail"]:
            return "fail"
        if spec["warn"] is not None and v > spec["warn"]:
            return "warn"
    else:
        if spec["fail"] is not None and v <= spec["fail"]:
            return "fail"
        if spec["warn"] is not None and v < spec["warn"]:
            return "warn"
    return "pass"


def _bad_mask(v: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    t = spec["warn"] if spec["warn"] is not None else spec["fail"]
    if t is None:
        return np.zeros(len(v), bool)
    return (v > t) if spec["better"] == "low" else (v < t) if spec["warn"] is not None else (v <= t)


def build(mesh: Mesh, target: str = "auto", max_regions: int = 12) -> Dict[str, Any]:
    """target: 'cfd' (finite-volume thresholds lead), 'fea' (element-shape thresholds lead) or 'auto'."""
    if target == "auto":
        target = "cfd" if mesh.poly is not None or mesh.source.get("kind") in ("openfoam", "openfoam_polymesh") else \
            "surface" if mesh.is_surface else "fea"
    out: Dict[str, Any] = {"summary": {"kind": mesh.kind, "format": mesh.source.get("format"), "dim": mesh.dim,
                                       "cells": mesh.n_cells, "points": mesh.n_points,
                                       "element_types": mesh.element_counts(), "bounding_box": mesh.bbox(),
                                       "target": target},
                           "checks": [], "metrics": {}, "regions": [], "notes": []}
    checks = out["checks"]
    cell_vals: Dict[str, np.ndarray] = {}
    adj = None
    centres = None
    if mesh.dim == 3:
        fv = fv_metrics(mesh)
        fm, cg = fv["fm"], fv["cg"]
        centres = cg["centres"]
        adj = G.cell_adjacency(fm)
        for k in FV_KEYS:
            cell_vals[k] = fv["cell"][k if k != "skewness" else "skewness"]
        stats_src = {"non_orthogonality": fv["face"]["non_orthogonality"],
                     "skewness": np.concatenate([fv["face"]["skewness_internal"], fv["face"]["skewness_boundary"]]),
                     "aspect_ratio": fv["cell"]["aspect_ratio"], "volume": fv["cell"]["volume"]}
        out["summary"].update(faces=int(fm.n_faces), internal_faces=int(fm.n_internal),
                              total_volume=float(cg["volumes"].sum()),
                              solution_directions=int(fv["dirs"].sum()),
                              patches=[{"name": n, "type": t, "faces": int(np.sum(fm.patch_of_face == i))}
                                       for i, (n, t) in enumerate(zip(fm.patch_names, fm.patch_types))])
        for k, v in stats_src.items():
            st = _stats(v, METRICS[k]["better"])
            if k == "non_orthogonality":
                st["mean"] = fv["non_orthogonality_average"]
            spec = METRICS[k]
            n_bad = int(np.sum(_bad_mask(v, spec)))
            out["metrics"][k] = {**st, "hist": _hist(v, k), "bad": n_bad, "of": int(len(v)),
                                 "per": "face" if k in ("non_orthogonality", "skewness") else "cell",
                                 "status": _status(st["worst"], spec) if st else "pass", "family": "fv"}
        fe_origin = mesh.poly is None                                 # FE meshes: these follow from inverted elements
        if fv["face_pyramids_bad"] and not fe_origin:
            checks.append(_check("face_pyramids", "fail", f"{fv['face_pyramids_bad']} face pyramids with negative "
                                 "volume", "Concave or warped cells.", ACTIONS["face_pyramids"]))
        nopen = int(np.sum(fv["cell"]["openness"] > 1e-6))
        if nopen and not fe_origin:
            checks.append(_check("openness", "fail", f"{nopen} open cells (faces do not close)", "",
                                 ACTIONS["openness"]))
        if fm.nonmanifold_faces:
            checks.append(_check("nonmanifold_faces", "fail", f"{fm.nonmanifold_faces} faces shared by more than "
                                 "two elements", "", ACTIONS["nonmanifold_faces"]))
        if fm.flipped_blocks:
            out["notes"].append(f"Node order of {', '.join(fm.flipped_blocks)} follows the opposite convention; "
                                "it was re-oriented before checking (not an error).")
    if mesh.poly is None and mesh.dim >= 2:
        em = element_metrics(mesh)
        for k in FE_KEYS:
            v = em[k]
            if not np.isfinite(v).any():
                continue
            cell_vals[k] = v
            st = _stats(v, METRICS[k]["better"])
            out["metrics"][k] = {**st, "hist": _hist(v, k), "bad": int(np.sum(_bad_mask(np.nan_to_num(v, nan=1.0 if k == "scaled_jacobian" else 0.0), METRICS[k]))),
                                 "of": int(np.isfinite(v).sum()), "per": "element",
                                 "status": _status(st["worst"], METRICS[k]), "family": "fe"}
    lead = FV_KEYS if target == "cfd" else ("element_skewness", "edge_ratio") if target == "surface" else FE_KEYS
    if target == "cfd" and mesh.poly is None and mesh.dim == 3:
        lead = FV_KEYS + ("scaled_jacobian",)
    for k, mt in out["metrics"].items():
        spec = METRICS[k]
        graded = k in lead
        mt["graded"] = graded
        if not graded:
            continue
        st = mt["status"]
        if k == "volume":
            nneg = int(np.sum(cell_vals["volume"] <= 0))
            st = "fail" if nneg else "pass"
            msg = f"{nneg} cells with zero or negative volume" if nneg else "All cell volumes positive"
        else:
            msg = (f"{spec['label']}: worst {_fmt(mt['worst'])}{(' ' + spec['unit']) if spec['unit'] and spec['unit'] != 'm3 (model units)' else ''}"
                   f", {mt['bad']} of {mt['of']} {mt['per']}s beyond {_fmt(spec['warn'] if spec['warn'] is not None else spec['fail'])}")
        checks.append(_check(k, st, msg, spec["why"], ACTIONS[k] if st != "pass" else "", metric=k))
    # connectivity
    if mesh.dim == 3 and adj is not None:
        lab = G.regions(adj)
        nreg = int(lab.max() + 1) if len(lab) else 0
        sizes = np.bincount(lab)
        out["summary"]["regions"] = nreg
        if nreg > 1:
            parts = []
            for r in np.argsort(-sizes)[:8]:
                c = centres[lab == r]
                parts.append({"cells": int(sizes[r]), "centre": c.mean(0).tolist()})
            checks.append(_check("regions", "warn", f"{nreg} disconnected regions (largest {sizes.max()} cells, "
                                 f"smallest {sizes.min()})", "Each region is solved independently; a stray region "
                                 "has no boundary conditions and no load path.", ACTIONS["regions"], parts=parts))
        else:
            checks.append(_check("regions", "pass", "One connected region", ""))
        if mesh.poly is None:
            nod = connected_components(G.element_adjacency_by_nodes(mesh), directed=False)[0]
            if nod < nreg:
                checks.append(_check("hinge", "warn", f"{nreg - nod} region(s) touch the rest only through nodes or "
                                     "edges", "", ACTIONS["hinge"]))
    elif mesh.dim == 2 and mesh.poly is None:
        a = G.element_adjacency_by_nodes(mesh)
        nreg, lab = connected_components(a, directed=False)
        out["summary"]["regions"] = int(nreg)
        if not mesh.is_surface:
            checks.append(_check("regions", "warn" if nreg > 1 else "pass",
                                 f"{nreg} disconnected regions" if nreg > 1 else "One connected region", "",
                                 ACTIONS["regions"] if nreg > 1 else ""))
        adj = a
        centres = _fe_centres(mesh)
    if mesh.is_surface or out["summary"]["target"] == "surface":
        sc = surface_checks(mesh)
        shell = sc.pop("shell_of_face", None)
        out["surface"] = sc
        for key, label, sev in (("open_edges", "open edges (holes)", "fail"),
                                ("nonmanifold_edges", "non-manifold edges", "fail"),
                                ("inconsistent_orientation_edges", "edges between faces with opposite normals", "warn"),
                                ("degenerate_faces", "zero-area faces", "warn"),
                                ("duplicate_faces", "duplicated faces", "warn")):
            n = sc.get(key, 0)
            act = ACTIONS.get(key.replace("inconsistent_orientation_edges", "orientation"), "")
            checks.append(_check(key, sev if n else "pass", f"{n} {label}" if n else f"No {label}", "", act if n else ""))
        checks.append(_check("shells", "pass" if sc["shells"] == 1 else "warn", f"{sc['shells']} separate surface "
                             "shell(s)", "", ACTIONS["regions"] if sc["shells"] > 1 else ""))
        if sc.get("enclosed_volume") is not None and sc["enclosed_volume"] < 0:
            checks.append(_check("orientation", "warn", "Normals point inwards (negative enclosed volume)", "",
                                 ACTIONS["orientation"]))
    # problem regions: clusters of cells beyond a threshold of a graded metric
    if centres is None and mesh.poly is None and mesh.blocks:
        centres = _fe_centres(mesh)
    if centres is not None and adj is not None:
        out["regions"] = _problem_regions(mesh, cell_vals, lead, adj, centres, max_regions)
    sev = [c["status"] for c in checks]
    out["status"] = "FAIL" if "fail" in sev else "WARNING" if "warn" in sev else "PASS"
    out["usage"] = {"elements": int(mesh.n_cells)}
    out["_cell_values"] = cell_vals
    out["_centres"] = centres
    return out


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    a = abs(v)
    return f"{v:.3g}" if (a >= 1e-3 and a < 1e5) or a == 0 else f"{v:.2e}"


def _check(key: str, status: str, message: str, why: str, action: str = "", **extra) -> Dict[str, Any]:
    return {"key": key, "status": status, "message": message, "why": why, "action": action, **extra}


def _fe_centres(mesh: Mesh) -> np.ndarray:
    return np.concatenate([mesh.points[b.corners].mean(1) for b in mesh.blocks])


def _problem_regions(mesh: Mesh, vals: Dict[str, np.ndarray], lead, adj, centres, max_regions) -> List[Dict[str, Any]]:
    bad = np.zeros(mesh.n_cells, bool)
    severity = np.zeros(mesh.n_cells)
    why = {}
    for k in lead:
        if k not in vals:
            continue
        v = np.nan_to_num(vals[k], nan=1.0 if METRICS[k]["better"] == "high" else 0.0)
        spec = METRICS[k]
        if k == "volume":
            m = v <= 0
            s = m * 3.0
        else:
            m = _bad_mask(v, spec)
            t = spec["warn"] if spec["warn"] is not None else spec["fail"]
            s = np.where(m, (v / t) if spec["better"] == "low" else (1 + (t - v) / max(abs(t), 0.05)), 0.0)
            if spec["fail"] is not None:
                s = np.where((v > spec["fail"]) if spec["better"] == "low" else (v <= spec["fail"]), s + 2, s)
        bad |= m
        severity = np.maximum(severity, s)
        why[k] = m
    idx = np.nonzero(bad)[0]
    if not len(idx):
        return []
    sub = adj[idx][:, idx]
    n, lab = connected_components(sub, directed=False)
    regs = []
    for r in range(n):
        cells = idx[lab == r]
        c = centres[cells]
        worst = {}
        for k, m in why.items():
            if m[cells].any():
                v = vals[k][cells]
                w = np.nanmax(v) if METRICS[k]["better"] == "low" else np.nanmin(v)
                worst[k] = {"worst": float(w), "cells": int(m[cells].sum()),
                            "at": mesh.cell_label(int(cells[np.nanargmax(v) if METRICS[k]["better"] == "low" else np.nanargmin(v)]))}
        regs.append({"cells": int(len(cells)), "centre": c.mean(0).tolist(), "bbox_min": c.min(0).tolist(),
                     "bbox_max": c.max(0).tolist(), "severity": float(severity[cells].max()), "worst": worst,
                     "sample_cells": cells[np.argsort(-severity[cells])[:200]].tolist(),
                     "status": "fail" if severity[cells].max() >= 2 else "warn"})
    regs.sort(key=lambda r: (-r["severity"], -r["cells"]))
    labels = list(string.ascii_uppercase) + [f"A{i}" for i in range(1, 100)]
    for i, r in enumerate(regs[:max_regions]):
        r["name"] = f"Region {labels[i]}"
        r["nearest_patch"] = _nearest_patch(mesh, r["centre"])
    extra = len(regs) - max_regions
    out = regs[:max_regions]
    if extra > 0:
        out.append({"name": f"+{extra} smaller regions", "cells": int(sum(r["cells"] for r in regs[max_regions:])),
                    "severity": 0, "worst": {}, "status": "warn", "centre": None, "sample_cells": []})
    return out


def _nearest_patch(mesh: Mesh, centre) -> Optional[str]:
    try:
        fm = G.face_mesh(mesh) if mesh.dim == 3 else None
    except Exception:
        fm = None
    if fm is None or not fm.patch_names:
        return None
    fg = mesh._cache.get("fg_for_patch")
    if fg is None:
        fg = G.face_geometry(fm)
        mesh._cache["fg_for_patch"] = fg
    bc = fg["centres"][fm.n_internal:]
    ok = fm.patch_of_face >= 0
    if not ok.any():
        return None
    d = np.linalg.norm(bc[ok] - np.asarray(centre), axis=1)
    return fm.patch_names[int(fm.patch_of_face[ok][np.argmin(d)])]


# ------------------------------------------------------------------------------------------ 3D view payload
def view_payload(mesh: Mesh, rep: Dict[str, Any], max_tris: int = 250_000) -> Dict[str, Any]:
    """Boundary surface (triangles) with each face's owner-cell metric values, and the problem regions' cells."""
    vals = rep["_cell_values"]
    if mesh.dim == 3:
        fm = G.face_mesh(mesh)
        ni = fm.n_internal
        faces = range(ni, fm.n_faces)
        offs, flat = fm.face_offsets, fm.face_nodes
        sizes = np.diff(offs)[ni:]
        tri_rows, owner = [], []
        for k in np.unique(sizes):
            fidx = np.nonzero(sizes == k)[0] + ni
            nodes = flat[offs[fidx][:, None] + np.arange(k)]
            for j in range(1, k - 1):
                tri_rows.append(np.stack([nodes[:, 0], nodes[:, j], nodes[:, j + 1]], 1))
                owner.append(fm.owner[fidx])
        tris = np.concatenate(tri_rows)
        own = np.concatenate(owner)
    else:
        rows, own_l, start = [], [], 0
        for b in mesh.blocks:
            c = b.corners
            for j in range(1, c.shape[1] - 1):
                rows.append(np.stack([c[:, 0], c[:, j], c[:, j + 1]], 1))
                own_l.append(np.arange(start, start + len(c)))
            start += len(c)
        tris, own = np.concatenate(rows), np.concatenate(own_l)
    decimated = False
    if len(tris) > max_tris:
        keep = np.random.default_rng(0).choice(len(tris), max_tris, replace=False)
        tris, own = tris[keep], own[keep]
        decimated = True
    used, inv = np.unique(tris.ravel(), return_inverse=True)
    pts = mesh.points[used]
    out = {"points": np.round(pts, 9).ravel().tolist(), "tris": inv.reshape(-1, 3).ravel().tolist(),
           "decimated": decimated, "fields": {}}
    for k, v in vals.items():
        a = v[own]
        out["fields"][k] = [None if not np.isfinite(x) else float(f"{x:.5g}") for x in a]
    cents = rep.get("_centres")
    if cents is not None:
        out["regions"] = [{"name": r.get("name"), "status": r.get("status"),
                           "points": np.round(cents[r["sample_cells"]], 9).ravel().tolist()} for r in rep["regions"]
                          if r.get("sample_cells")]
    return out
