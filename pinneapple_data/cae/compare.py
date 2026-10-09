"""Compare two results field by field: simulation vs simulation (different meshes or models), vs experiment
(sensor tables), vs AI predictions (point tables or meshes), vs analytical or benchmark data.

The candidate is interpolated onto the reference locations (cell centres for cell fields, nodes for point fields):
directly when both use the same locations, otherwise by linear (Delaunay) interpolation inside the candidate's
domain and nearest-neighbour outside it (counted and reported). Metrics per field and per component: MAE, RMSE,
maximum absolute error and where, bias, relative L2 error, NRMSE (RMSE / reference range), R². Cell fields are
volume-weighted when cell volumes are known.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .model import Mesh

ALIASES = [
    {"u", "velocity", "vel", "u_mean", "umean", "velocity_vector", "v"},
    {"p", "pressure", "p_static", "static_pressure", "pmean", "p_mean", "kinematic_pressure"},
    {"t", "temperature", "temp", "ndtemp", "nt"},
    {"disp", "displacement", "u_fea", "deflection"},
    {"k", "tke", "turbulent_kinetic_energy"},
    {"epsilon", "eps", "dissipation"}, {"omega", "specific_dissipation"}, {"nut", "nu_t", "turbulent_viscosity"},
    {"stress", "sigma", "s"},
]
VECTOR_SUFFIX = ["x", "y", "z"]


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _alias_group(name: str) -> Optional[int]:
    n = _norm(name)
    for i, g in enumerate(ALIASES):
        if n in g:
            return i
    return None


def _split(field: str) -> Tuple[str, Optional[int]]:
    m = re.fullmatch(r"(.+)\[(x|y|z|0|1|2)\]", field)
    if not m:
        return field, None
    return m.group(1), {"x": 0, "y": 1, "z": 2}.get(m.group(2), int(m.group(2)) if m.group(2).isdigit() else 0)


def locations(m: Mesh, field: str) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    """(points where the field lives, 'cell'|'point', weights)."""
    field = _split(field)[0]
    if field in m.point_data:
        return m.points, "point", None
    if field in m.cell_data:
        cc, w = cell_centres_volumes(m)
        return cc, "cell", w
    raise KeyError(field)


def cell_centres_volumes(m: Mesh) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if "cc" in m._cache:
        return m._cache["cc"], m._cache["cv"]
    if m.poly is not None or m.dim == 3:
        from . import geometry as G
        fm = G.face_mesh(m)
        cg = G.cell_geometry(fm, G.face_geometry(fm))
        cc, cv = cg["centres"], cg["volumes"]
    else:
        cc = np.concatenate([m.points[b.corners].mean(1) for b in m.blocks]) if m.blocks else m.points
        cv = None
    m._cache["cc"], m._cache["cv"] = cc, cv
    return cc, cv


def field_value(m: Mesh, f: str) -> np.ndarray:
    base, comp = _split(f)
    v = np.asarray(m.point_data.get(base) if base in m.point_data else m.cell_data[base], float)
    return v[:, comp] if comp is not None and v.ndim == 2 else v


def fields_of(m: Mesh) -> Dict[str, Dict[str, Any]]:
    out = {}
    for f in list(m.point_data) + list(m.cell_data):
        v = field_value(m, f)
        out[f] = {"where": "point" if f in m.point_data else "cell", "components": 1 if v.ndim == 1 else v.shape[1],
                  "unit": m.units.get(f), "n": int(len(v))}
    return out


def match_fields(ref: Mesh, cand: Mesh, mapping: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """Pairs (reference field, candidate field): explicit mapping first, then same name (case-insensitive), then
    aliases (U ~ velocity, p ~ pressure, T ~ temperature, DISP ~ displacement, ...)."""
    rf, cf = fields_of(ref), fields_of(cand)
    pairs: List[Tuple[str, str]] = []
    used = set()
    for r, c in (mapping or {}).items():
        if r in rf and c in cf:
            pairs.append((r, c))
            used.add(c)
    taken = {p[0] for p in pairs}
    for r in rf:
        if r in taken:
            continue
        hit = next((c for c in cf if c not in used and _norm(c) == _norm(r)), None)
        if hit is None:
            g = _alias_group(r)
            hit = next((c for c in cf if c not in used and g is not None and _alias_group(c) == g), None)
        if hit is not None and (rf[r]["components"] == cf[hit]["components"] or
                                min(rf[r]["components"], cf[hit]["components"]) == 1):
            pairs.append((r, hit))
            used.add(hit)
            continue
        # a scalar reference that is one component of a candidate vector: Ux / U_x / DISP_z / uz -> U[x]
        mm = re.fullmatch(r"(.+?)[_:.]?([xyzXYZ012])", r)
        if rf[r]["components"] == 1 and mm:
            base, comp = mm.group(1), mm.group(2).lower()
            g = _alias_group(base)
            vec_hit = next((c for c in cf if cf[c]["components"] >= 2 and (_norm(c) == _norm(base) or
                            (g is not None and _alias_group(c) == g))), None)
            if vec_hit is not None:
                pairs.append((r, f"{vec_hit}[{comp}]"))
    return pairs


def _active_dims(*pts: np.ndarray) -> np.ndarray:
    allp = np.vstack(pts)
    ext = allp.max(0) - allp.min(0)
    return ext > 1e-6 * max(ext.max(), 1e-300)


def _boundary_centres(m: Mesh, bvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centres of the boundary faces that carry a value (empty patches excluded)."""
    if "bfc" not in m._cache:
        from . import geometry as G
        fm = G.face_mesh(m)
        m._cache["bfc"] = G.face_geometry(fm)["centres"][fm.n_internal:]
    ok = np.all(np.isfinite(bvals.reshape(len(bvals), -1)), axis=1)
    return m._cache["bfc"][ok], ok


class Interpolator:
    """Values of a candidate field at query points: identical locations, linear (Delaunay), nearest outside."""

    def __init__(self, src_pts: np.ndarray, dst_pts: np.ndarray, method: str = "auto"):
        self.dims = _active_dims(src_pts, dst_pts)
        s, d = src_pts[:, self.dims], dst_pts[:, self.dims]
        scale = float(np.max(np.ptp(np.vstack([s, d]), axis=0))) or 1.0
        self.identity = None
        self.tree = cKDTree(s)
        dist, idx = self.tree.query(d)
        self.nn_idx, self.nn_dist = idx, dist / scale
        if np.all(dist < 1e-7 * scale):
            self.identity = idx                                      # same locations: no interpolation
            self.method = "same locations"
            self.outside = 0
            return
        if method == "auto":
            method = "linear" if len(s) <= 400_000 and s.shape[1] >= 1 else "nearest"
        self.method = method
        self.weights = None
        self.outside = 0
        if method == "linear" and s.shape[1] >= 2:
            from scipy.spatial import Delaunay
            try:
                tri = Delaunay(s)
                simp = tri.find_simplex(d)
                ok = simp >= 0
                T = tri.transform[simp[ok]]
                b = np.einsum("ijk,ik->ij", T[:, :s.shape[1]], d[ok] - T[:, s.shape[1]])
                bary = np.column_stack([b, 1 - b.sum(1)])
                self.verts = tri.simplices[simp[ok]]
                self.bary = bary
                self.ok = ok
                self.outside = int((~ok).sum())
                return
            except Exception:                                         # degenerate geometry: fall back
                self.method = "nearest"
        elif method == "linear" and s.shape[1] == 1:
            self.method = "linear-1d"
            return
        self.method = "nearest"

    def __call__(self, values: np.ndarray, src_pts: Optional[np.ndarray] = None, dst_pts: Optional[np.ndarray] = None) -> np.ndarray:
        if self.identity is not None:
            return values[self.identity]
        if self.method == "linear":
            out = values[self.nn_idx].astype(float).copy()
            v = values[self.verts]                                    # (m, k+1, [c])
            out[self.ok] = np.einsum("ij,ij...->i...", self.bary, v)
            return out
        if self.method == "linear-1d":
            x = src_pts[:, self.dims][:, 0]
            o = np.argsort(x)
            q = dst_pts[:, self.dims][:, 0]
            if values.ndim == 1:
                return np.interp(q, x[o], values[o])
            return np.column_stack([np.interp(q, x[o], values[o, j]) for j in range(values.shape[1])])
        return values[self.nn_idx]


def _metrics(r: np.ndarray, c: np.ndarray, w: Optional[np.ndarray]) -> Dict[str, float]:
    ok = np.isfinite(r) & np.isfinite(c)
    r, c = r[ok], c[ok]
    ww = (w[ok] / w[ok].sum()) if w is not None and len(w) == len(ok) else np.full(len(r), 1.0 / max(len(r), 1))
    e = c - r
    mae = float(np.sum(ww * np.abs(e)))
    rmse = float(np.sqrt(np.sum(ww * e * e)))
    i = int(np.argmax(np.abs(e))) if len(e) else 0
    ref_l2 = float(np.sqrt(np.sum(ww * r * r)))
    rng = float(r.max() - r.min()) if len(r) else 0.0
    var = float(np.sum(ww * (r - np.sum(ww * r)) ** 2))
    return {"mae": mae, "rmse": rmse, "max_abs": float(np.abs(e[i])) if len(e) else 0.0, "max_index_valid": i,
            "p99_abs": float(np.percentile(np.abs(e), 99)) if len(e) else 0.0,
            "bias": float(np.sum(ww * e)), "rel_l2": rmse / ref_l2 if ref_l2 > 0 else None,
            "nrmse": rmse / rng if rng > 0 else None, "r2": 1 - rmse ** 2 / var if var > 0 else None,
            "ref_min": float(r.min()) if len(r) else None, "ref_max": float(r.max()) if len(r) else None, "n": int(len(r))}


def compare(ref: Mesh, cand: Mesh, mapping: Optional[Dict[str, str]] = None, method: str = "auto",
            fields: Optional[List[str]] = None) -> Dict[str, Any]:
    pairs = match_fields(ref, cand, mapping)
    if fields:
        pairs = [p for p in pairs if p[0] in fields]
    if not pairs:
        raise ValueError("no common field: reference has " + ", ".join(fields_of(ref)) + "; candidate has "
                         + ", ".join(fields_of(cand)) + ". Map them explicitly (e.g. {\"T\": \"temperature\"}).")
    results, maps = [], {}
    interp_cache: Dict[Tuple[str, str], Interpolator] = {}
    for rname, cname in pairs:
        rp, rwhere, w = locations(ref, rname)
        cp, cwhere, _ = locations(cand, cname)
        cbase = _split(cname)[0]
        with_bnd = cwhere == "cell" and cbase in cand.boundary_values and cand.poly is not None
        if with_bnd:
            bfc, bmask = _boundary_centres(cand, cand.boundary_values[cbase])
            cp = np.vstack([cp, bfc])
            cwhere = "cell+boundary"
        key = (rwhere, cwhere)
        it = interp_cache.get(key)
        if it is None:
            it = Interpolator(cp, rp, method)
            interp_cache[key] = it
        rv = field_value(ref, rname)
        cvals = field_value(cand, cname)
        if with_bnd:
            bv = cand.boundary_values[cbase][bmask]
            comp = _split(cname)[1]
            if comp is not None and bv.ndim == 2:
                bv = bv[:, comp]
            cvals = np.concatenate([cvals, bv])
        cv = it(cvals, cp, rp)
        if rv.ndim != cv.ndim:                                       # vector vs magnitude
            rv = np.linalg.norm(rv, axis=1) if rv.ndim == 2 else rv
            cv = np.linalg.norm(cv, axis=1) if cv.ndim == 2 else cv
        comps: Dict[str, Dict[str, float]] = {}
        if rv.ndim == 2:
            mag_r, mag_c = np.linalg.norm(rv, axis=1), np.linalg.norm(cv, axis=1)
            comps["magnitude"] = _metrics(mag_r, mag_c, w)
            for j in range(rv.shape[1]):
                if np.ptp(rv[:, j]) == 0 and np.ptp(cv[:, j]) == 0:
                    continue                                          # e.g. Uz in a 2D case
                comps[VECTOR_SUFFIX[j] if j < 3 else str(j)] = _metrics(rv[:, j], cv[:, j], w)
            err = np.linalg.norm(cv - rv, axis=1)
            refmag = mag_r
            main = "vector"
            vecm = _metrics(np.zeros(len(rv)), err, w)
            ref_l2 = float(np.sqrt(np.sum((w / w.sum() if w is not None else 1.0 / len(rv)) * (mag_r ** 2))))
            summary = {"mae": vecm["mae"], "rmse": vecm["rmse"], "max_abs": vecm["max_abs"], "p99_abs": vecm["p99_abs"],
                       "rel_l2": vecm["rmse"] / ref_l2 if ref_l2 > 0 else None,
                       "nrmse": vecm["rmse"] / float(np.ptp(mag_r)) if np.ptp(mag_r) > 0 else None,
                       "r2": comps["magnitude"]["r2"], "bias": comps["magnitude"]["bias"]}
        else:
            comps["value"] = _metrics(rv, cv, w)
            err = np.abs(cv - rv)
            refmag = rv
            main = "scalar"
            summary = {k: comps["value"][k] for k in ("mae", "rmse", "max_abs", "p99_abs", "rel_l2", "nrmse", "r2", "bias")}
        imax = int(np.nanargmax(err)) if len(err) else 0
        unit = ref.units.get(rname) or cand.units.get(_split(cname)[0])
        cu = cand.units.get(_split(cname)[0])
        res = {"reference": rname, "candidate": cname, "type": main, "where": rwhere, "unit": unit,
               "unit_candidate": cu, "unit_mismatch": bool(unit and cu and _norm(unit) != _norm(cu)),
               "summary": summary, "components": comps, "max_at": rp[imax].tolist(),
               "interpolation": it.method, "outside_candidate": it.outside, "n": int(len(err)),
               "weighted": w is not None}
        results.append(res)
        maps[rname] = {"error": err, "ref": refmag, "cand": (np.linalg.norm(cv, axis=1) if cv.ndim == 2 else cv),
                       "points": rp, "where": rwhere}
    # global: mean over fields of the range-normalised errors (dimensionless, comparable across fields)
    nr = [r["summary"]["nrmse"] for r in results if r["summary"]["nrmse"] is not None]
    nm = []
    nmax = []
    for r in results:
        rng = None
        comp = r["components"].get("magnitude") or r["components"].get("value")
        if comp and comp["ref_max"] is not None:
            rng = comp["ref_max"] - comp["ref_min"]
        if rng:
            nm.append(r["summary"]["mae"] / rng)
            nmax.append(r["summary"]["max_abs"] / rng)
    glob = {"mae": float(np.mean(nm)) if nm else None, "rmse": float(np.mean(nr)) if nr else None,
            "max": float(np.max(nmax)) if nmax else None, "note": "mean over fields of MAE and RMSE divided by the "
            "reference range of each field; max = largest error relative to its field's range"}
    return {"fields": results, "global": glob, "pairs": pairs, "_maps": maps,
            "reference_fields": fields_of(ref), "candidate_fields": fields_of(cand)}


# --------------------------------------------------------------------------------------------- views for the UI
def _round(a: np.ndarray, sig: int = 5) -> List[Optional[float]]:
    return [None if not np.isfinite(x) else float(f"{x:.{sig}g}") for x in np.asarray(a, float).ravel()]


def surface(m: Mesh, max_tris: int = 300_000) -> Optional[Dict[str, Any]]:
    """Boundary surface for display: triangles with the cell each one belongs to. For 2D OpenFOAM cases (empty
    patches) only one of the two empty sides is kept."""
    if m.poly is None and not m.blocks:
        return None
    if m.dim == 3:
        from . import geometry as G
        fm = G.face_mesh(m)
        ni = fm.n_internal
        sel = np.arange(ni, fm.n_faces)
        if "empty" in fm.patch_types:
            fg = G.face_geometry(fm)
            emp = np.isin(fm.patch_of_face, [i for i, t in enumerate(fm.patch_types) if t == "empty"])
            n = fg["areas"][ni:]
            ax = int(np.argmax(np.abs(n[emp]).mean(0))) if emp.any() else 2
            sel = sel[emp & (n[:, ax] > 0)]
        offs, flat = fm.face_offsets, fm.face_nodes
        sizes = np.diff(offs)[sel]
        rows, own = [], []
        for k in np.unique(sizes):
            f = sel[sizes == k]
            nodes = flat[offs[f][:, None] + np.arange(k)]
            for j in range(1, k - 1):
                rows.append(np.stack([nodes[:, 0], nodes[:, j], nodes[:, j + 1]], 1))
                own.append(fm.owner[f])
        tris, owner = np.concatenate(rows), np.concatenate(own)
    else:
        rows, own, start = [], [], 0
        for b in m.blocks:
            c = b.corners
            for j in range(1, c.shape[1] - 1):
                rows.append(np.stack([c[:, 0], c[:, j], c[:, j + 1]], 1))
                own.append(np.arange(start, start + len(c)))
            start += len(c)
        tris, owner = np.concatenate(rows), np.concatenate(own)
    if len(tris) > max_tris:
        keep = np.random.default_rng(0).choice(len(tris), max_tris, replace=False)
        tris, owner = tris[keep], owner[keep]
    used, inv = np.unique(tris.ravel(), return_inverse=True)
    cells, cinv = np.unique(owner, return_inverse=True)
    return {"points": np.round(m.points[used], 9).ravel().tolist(), "tris": inv.reshape(-1, 3).ravel().tolist(),
            "tri_cell": cinv.tolist(), "cells": cells, "vertex_index": used}


def views(ref: Mesh, res: Dict[str, Any], max_points: int = 60000) -> Dict[str, Any]:
    """Per-field display payload: a surface coloured per cell (or per node), a line plot, or a 2D/3D point map."""
    out: Dict[str, Any] = {"fields": {}}
    surf = None
    for name, mp in res["_maps"].items():
        pts, where = mp["points"], mp["where"]
        has_cells = ref.poly is not None or bool(ref.blocks)
        if has_cells and (where == "cell" or where == "point"):
            if surf is None:
                surf = surface(ref)
                out["surface"] = {k: v for k, v in surf.items() if k in ("points", "tris", "tri_cell")}
            if where == "cell":
                sel = surf["cells"]
                out["fields"][name] = {"kind": "surface_cell", "error": _round(mp["error"][sel]),
                                       "ref": _round(mp["ref"][sel]), "cand": _round(mp["cand"][sel])}
            else:
                sel = surf["vertex_index"]
                out["fields"][name] = {"kind": "surface_point", "error": _round(mp["error"][sel]),
                                       "ref": _round(mp["ref"][sel]), "cand": _round(mp["cand"][sel])}
            continue
        act = _active_dims(pts)
        if act.sum() <= 1:
            ax = int(np.argmax(np.ptp(pts, axis=0)))
            o = np.argsort(pts[:, ax])
            out["fields"][name] = {"kind": "line", "axis": "xyz"[ax], "s": _round(pts[o, ax], 7),
                                   "ref": _round(mp["ref"][o]), "cand": _round(mp["cand"][o]),
                                   "error": _round(mp["error"][o])}
            continue
        idx = np.arange(len(pts))
        if len(idx) > max_points:
            idx = np.random.default_rng(0).choice(len(idx), max_points, replace=False)
        dims = np.nonzero(act)[0]
        out["fields"][name] = {"kind": "scatter2d" if len(dims) == 2 else "points3d", "axes": ["xyz"[d] for d in dims],
                               "xy": np.round(pts[idx][:, dims], 9).ravel().tolist(), "error": _round(mp["error"][idx]),
                               "ref": _round(mp["ref"][idx]), "cand": _round(mp["cand"][idx])}
    for name, mp in res["_maps"].items():
        e, r, c = mp["error"], mp["ref"], mp["cand"]
        ok = np.isfinite(e) & np.isfinite(r) & np.isfinite(c)
        idx = np.nonzero(ok)[0]
        if len(idx) > 3000:
            idx = np.random.default_rng(1).choice(idx, 3000, replace=False)
        hist, edges = np.histogram(e[ok], bins=30) if ok.any() else (np.zeros(0), np.zeros(1))
        worst = np.argsort(-np.nan_to_num(e, nan=-np.inf))[:12]
        out["fields"][name].update(
            parity={"ref": _round(r[idx]), "cand": _round(c[idx])},
            hist={"edges": _round(edges), "counts": hist.tolist()},
            worst=[{"at": mp["points"][i].tolist(), "ref": float(r[i]), "cand": float(c[i]), "error": float(e[i])}
                   for i in worst if np.isfinite(e[i])])
    return out
