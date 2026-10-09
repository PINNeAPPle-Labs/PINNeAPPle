"""Mesh quality: finite-volume metrics as OpenFOAM's checkMesh computes them, element shape metrics for finite-element
meshes, connectivity, and surface (STL) checks; problem regions are clusters of bad cells.

Every metric is defined in METRICS (what it measures, the threshold and the source of the threshold), so the report
can explain itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import geometry as G
from .model import CELL_EDGES, Mesh

METRICS = {
    "non_orthogonality": {
        "label": "Non-orthogonality", "unit": "deg", "better": "low", "warn": 70.0, "fail": 85.0,
        "what": "Angle between the face normal and the line joining the two cell centres.",
        "why": "The face gradient (diffusion, pressure) is evaluated along the centre-to-centre line; above 70° the "
               "non-orthogonal correction dominates and the solver can oscillate or diverge.",
        "source": "OpenFOAM checkMesh: faces above 70° are 'severely non-orthogonal'"},
    "skewness": {
        "label": "Skewness (face)", "unit": "", "better": "low", "warn": 4.0, "fail": 20.0,
        "what": "Distance between the face centre and the point where the centre-to-centre line crosses the face, "
                "relative to the face size.",
        "why": "Face values are interpolated at the wrong point: accuracy drops and convective schemes lose boundedness.",
        "source": "OpenFOAM checkMesh: above 4 the faces are 'highly skew' (snappyHexMesh allows 20 on boundary faces)"},
    "aspect_ratio": {
        "label": "Aspect ratio (cell)", "unit": "", "better": "low", "warn": 1000.0, "fail": None,
        "what": "Largest to smallest cell extent (OpenFOAM's definition from the face areas).",
        "why": "Very stretched cells are fine in boundary layers along the flow but slow convergence elsewhere.",
        "source": "OpenFOAM checkMesh threshold 1000"},
    "volume": {
        "label": "Cell volume", "unit": "m3 (model units)", "better": "high", "warn": None, "fail": 0.0,
        "what": "Cell volume from the pyramid decomposition.",
        "why": "Zero or negative volumes are inverted or collapsed cells: no solver can run on them.",
        "source": "Any solver; OpenFOAM checkMesh 'Cell volumes'"},
    "scaled_jacobian": {
        "label": "Scaled Jacobian (min)", "unit": "", "better": "high", "warn": 0.2, "fail": 0.0,
        "what": "Smallest corner Jacobian normalised by the edge lengths: 1 for a perfect element, 0 for a flat one, "
                "negative for an inverted one.",
        "why": "Finite-element integration needs a positive Jacobian; below about 0.2 stiffness and stresses degrade.",
        "source": "Verdict library definition; FEA pre-processors commonly warn below 0.2"},
    "element_skewness": {
        "label": "Element skewness", "unit": "", "better": "low", "warn": 0.9, "fail": 0.98,
        "what": "0 for an equilateral/equiangular element, 1 for a degenerate one (volume-based for triangles and "
                "tetrahedra, equiangle for the others).",
        "why": "Highly skewed elements give poor accuracy in FEA and poor convergence in CFD.",
        "source": "ANSYS meshing guideline: 0.9-0.98 bad, above 0.98 degenerate"},
    "edge_ratio": {
        "label": "Edge ratio", "unit": "", "better": "low", "warn": 20.0, "fail": None,
        "what": "Longest to shortest edge of the element.",
        "why": "Elongated elements are acceptable along smooth gradients, harmful across them.",
        "source": "Common FEA guideline: warn above 20"},
}


def _stats(v: np.ndarray, better: str) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if not len(v):
        return {}
    worst = float(v.max() if better == "low" else v.min())
    return {"worst": worst, "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p99": float(np.percentile(v, 99 if better == "low" else 1)), "min": float(v.min()), "max": float(v.max())}


def _hist(v: np.ndarray, metric: str, bins: int = 30) -> Dict[str, Any]:
    v = v[np.isfinite(v)]
    if not len(v):
        return {"edges": [], "counts": []}
    lo, hi = float(v.min()), float(v.max())
    spec = METRICS[metric]
    for t in (spec["warn"], spec["fail"]):
        if t is not None:
            lo, hi = min(lo, t), max(hi, t) if metric != "aspect_ratio" else hi
    if metric == "aspect_ratio" and hi / max(lo, 1e-12) > 50:
        edges = np.logspace(np.log10(max(lo, 1.0)), np.log10(hi * 1.0001), bins + 1)
    else:
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), edges)
    return {"edges": edges.tolist(), "counts": counts.tolist(), "log": bool(metric == "aspect_ratio" and edges[-1] / edges[0] > 50)}


# ------------------------------------------------------------------------------------------ finite-volume metrics
def fv_metrics(mesh: Mesh) -> Dict[str, Any]:
    fm = G.face_mesh(mesh)
    fg = G.face_geometry(fm)
    cg = G.cell_geometry(fm, fg)
    ni, nc = fm.n_internal, fm.n_cells
    own, nei = fm.owner, fm.neighbour
    fc, sf, cc = fg["centres"], fg["areas"], cg["centres"]
    smag = np.linalg.norm(sf, axis=1)
    # non-orthogonality (internal faces)
    d = cc[nei] - cc[own[:ni]]
    cosang = np.einsum("ij,ij->i", d, sf[:ni]) / np.maximum(np.linalg.norm(d, axis=1) * smag[:ni], G.TINY)
    nonorth = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    # skewness (internal and boundary faces), OpenFOAM primitiveMeshTools::faceSkewness / boundaryFaceSkewness
    cpf = fc - cc[own]
    dd = np.empty_like(cpf)
    dd[:ni] = d
    nhat = sf[ni:] / np.maximum(smag[ni:], G.TINY)[:, None]
    dd[ni:] = nhat * np.einsum("ij,ij->i", nhat, cpf[ni:])[:, None]
    sv = cpf - (np.einsum("ij,ij->i", sf, cpf) / (np.einsum("ij,ij->i", sf, dd) + 1e-300))[:, None] * dd
    svmag = np.linalg.norm(sv, axis=1)
    svhat = sv / (svmag + 1e-300)[:, None]
    fd = np.linalg.norm(dd, axis=1)
    fd[:ni] *= 0.2                                                   # internal faces
    fd[ni:] *= 0.4                                                   # boundary faces (boundaryFaceSkewness)
    fd += 1e-300
    sizes = np.diff(fm.face_offsets)
    for k in np.unique(sizes):
        idx = np.nonzero(sizes == k)[0]
        nodes = fm.face_nodes[fm.face_offsets[idx][:, None] + np.arange(k)]
        proj = np.abs(np.einsum("ij,ikj->ik", svhat[idx], fm.points[nodes] - fc[idx][:, None, :])).max(1)
        fd[idx] = np.maximum(fd[idx], proj)
    skew = svmag / fd
    # aspect ratio (cellClosedness)
    sumclosed = np.zeros((nc, 3))
    sumvec = np.zeros((nc, 3))
    for c in range(3):
        sumclosed[:, c] = np.bincount(own, np.abs(sf[:, c]), nc) + np.bincount(nei, np.abs(sf[:ni, c]), nc)
        sumvec[:, c] = np.bincount(own, sf[:, c], nc) - np.bincount(nei, sf[:ni, c], nc)
    dirs = G.solution_dirs(fm, fg)
    sc = sumclosed[:, dirs]
    ar = sc.max(1) / (sc.min(1) + 1e-300)
    vol = cg["volumes"]
    if dirs.sum() == 3:
        ar = np.maximum(ar, (1.0 / 6.0) * sumclosed.sum(1) / np.maximum(vol, 1e-300) ** (2.0 / 3.0))
    openness = np.linalg.norm(sumvec, axis=1) / np.maximum(sumclosed.max(1), 1e-300)
    # per-cell worst of the face metrics (for maps and problem regions)
    cell_nonorth = np.zeros(nc)
    np.maximum.at(cell_nonorth, own[:ni], nonorth)
    np.maximum.at(cell_nonorth, nei, nonorth)
    cell_skew = np.zeros(nc)
    np.maximum.at(cell_skew, own, skew)
    np.maximum.at(cell_skew, nei, skew[:ni])
    bad_pyr = int(np.sum(cg["pyramid_own"] <= 0) + np.sum(cg["pyramid_nei"] <= 0))
    return {"fm": fm, "fg": fg, "cg": cg, "dirs": dirs,
            "face": {"non_orthogonality": nonorth, "skewness_internal": skew[:ni], "skewness_boundary": skew[ni:],
                     "flatness": fg["flatness"]},
            "cell": {"non_orthogonality": cell_nonorth, "skewness": cell_skew, "aspect_ratio": ar, "volume": vol,
                     "openness": openness},
            "face_pyramids_bad": bad_pyr, "face_areas": smag,
            "non_orthogonality_average": float(np.degrees(np.arccos(min(1.0, cosang.mean())))) if ni else 0.0}


# ------------------------------------------------------------------------------------------ element metrics
_CORNER_TRIPLES = {
    "tetra": [(0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3), (3, 2, 1, 0)],
    "hexahedron": [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7), (4, 7, 5, 0), (5, 4, 6, 1),
                   (6, 5, 7, 2), (7, 6, 4, 3)],
    "wedge": [(0, 1, 2, 3), (1, 2, 0, 4), (2, 0, 1, 5), (3, 5, 4, 0), (4, 3, 5, 1), (5, 4, 3, 2)],
    "pyramid": [(0, 1, 3, 4), (1, 2, 0, 4), (2, 3, 1, 4), (3, 0, 2, 4)],
}
_FACE_ANGLES = {"tetra": [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
                "hexahedron": [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
                "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
                "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
                "triangle": [(0, 1, 2)], "quad": [(0, 1, 2, 3)]}


# ideal elements in VTK node order (tetra: (0,1,2) normal towards 3; wedge: (0,1,2) normal away from (3,4,5))
_IDEAL = {"tetra": [[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]],
          "hexahedron": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
          "wedge": [[0, 0, 0], [0.5, 0.75 ** 0.5, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.75 ** 0.5, 1], [1, 0, 1]],
          "pyramid": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5 ** 0.5]]}


def _ref_norm(t: str) -> np.ndarray:
    """Per-corner signed normalisation so that the ideal element (regular tetrahedron, cube, equilateral wedge,
    pyramid with equal edges) gives exactly 1 at every corner (Verdict's convention)."""
    p = np.array(_IDEAL[t], float)
    out = []
    for c, a, b, e in _CORNER_TRIPLES[t]:
        u, v, w = p[a] - p[c], p[b] - p[c], p[e] - p[c]
        out.append(np.linalg.det(np.stack([u, v, w])) / (np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(w)))
    return np.array(out)


def _angles(p: np.ndarray, face) -> np.ndarray:
    k = len(face)
    out = []
    for i in range(k):
        a, b, c = p[:, face[i - 1]], p[:, face[i]], p[:, face[(i + 1) % k]]
        u, v = a - b, c - b
        cs = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), G.TINY)
        out.append(np.degrees(np.arccos(np.clip(cs, -1, 1))))
    return np.stack(out, 1)


def element_metrics(mesh: Mesh) -> Dict[str, np.ndarray]:
    """Scaled Jacobian, skewness and edge ratio for every cell of the mesh's top dimension (global cell order)."""
    nc = mesh.n_cells
    sj = np.full(nc, np.nan)
    skew = np.full(nc, np.nan)
    er = np.full(nc, np.nan)
    if mesh.dim == 3:
        G.oriented_blocks(mesh)
    start = 0
    planar_n = None
    if mesh.dim == 2 and not mesh.is_surface:
        q = mesh.points - mesh.points.mean(0)
        planar_n = np.linalg.svd(q, full_matrices=False)[2][-1]
    for b in mesh.blocks:
        t = b.linear
        n = len(b.conn)
        sl = slice(start, start + n)
        start += n
        if b.dim != mesh.dim or t not in CELL_EDGES:
            continue
        p = mesh.points[b.corners]
        e = np.stack([np.linalg.norm(p[:, j] - p[:, i], axis=2 - 1) for i, j in CELL_EDGES[t]], 1)
        er[sl] = e.max(1) / np.maximum(e.min(1), G.TINY)
        if t in _CORNER_TRIPLES:
            nrm = _ref_norm(t)
            vals = []
            for (c, a, bb, d), s in zip(_CORNER_TRIPLES[t], 1.0 / nrm):
                u, v, w = p[:, a] - p[:, c], p[:, bb] - p[:, c], p[:, d] - p[:, c]
                det = s * np.einsum("ij,ij->i", np.cross(u, v), w)
                vals.append(det / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
                                             * np.linalg.norm(w, axis=1), G.TINY))
            sj[sl] = np.min(vals, 0)
        elif t in ("triangle", "quad"):
            k = 3 if t == "triangle" else 4
            vals = []
            nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            ref = planar_n if planar_n is not None else nrm / np.maximum(np.linalg.norm(nrm, axis=1), G.TINY)[:, None]
            for i in range(k):
                u, v = p[:, (i + 1) % k] - p[:, i], p[:, (i - 1) % k] - p[:, i]
                c = np.cross(u, v)
                dot = c @ ref if ref.ndim == 1 else np.einsum("ij,ij->i", c, ref)
                vals.append(dot / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), G.TINY))
            sj[sl] = np.min(vals, 0) * (2 / 3 ** 0.5 if t == "triangle" else 1.0)
        if t == "tetra":                                              # volume-based skewness (circumsphere)
            a, bb, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
            vol = np.abs(np.einsum("ij,ij->i", np.cross(a, bb), c)) / 6
            num = (np.einsum("ij,ij->i", a, a)[:, None] * np.cross(bb, c) + np.einsum("ij,ij->i", bb, bb)[:, None]
                   * np.cross(c, a) + np.einsum("ij,ij->i", c, c)[:, None] * np.cross(a, bb))
            r = np.linalg.norm(num, axis=1) / np.maximum(12 * vol, G.TINY)
            skew[sl] = np.clip(1 - vol / np.maximum(8 * np.sqrt(3) / 27 * r ** 3, G.TINY), 0, 1)
        elif t == "triangle":
            la, lb, lc = (np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in ((1, 2), (2, 0), (0, 1)))
            area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
            r = la * lb * lc / np.maximum(4 * area, G.TINY)
            skew[sl] = np.clip(1 - area / np.maximum(3 * np.sqrt(3) / 4 * r ** 2, G.TINY), 0, 1)
        else:
            worst = np.zeros(n)
            for f in _FACE_ANGLES[t]:
                ang = _angles(p, f)
                te = 60.0 if len(f) == 3 else 90.0
                worst = np.maximum(worst, np.maximum((ang.max(1) - te) / (180 - te), (te - ang.min(1)) / te))
            skew[sl] = np.clip(worst, 0, 1)
    return {"scaled_jacobian": sj, "element_skewness": skew, "edge_ratio": er}


# ------------------------------------------------------------------------------------------ surface checks
def surface_checks(mesh: Mesh) -> Dict[str, Any]:
    """Triangle/quad surface (STL, shell mesh): open and non-manifold edges, orientation, degenerate faces, shells."""
    tris = [b.corners for b in mesh.blocks if b.dim == 2]
    if not tris:
        return {}
    faces = [f for f in tris]
    edges, owner_face, direction = [], [], []
    fid0 = 0
    for f in faces:
        k = f.shape[1]
        for i in range(k):
            a, b = f[:, i], f[:, (i + 1) % k]
            edges.append(np.stack([np.minimum(a, b), np.maximum(a, b)], 1))
            direction.append(a < b)
            owner_face.append(np.arange(fid0, fid0 + len(f)))
        fid0 += len(f)
    e = np.concatenate(edges)
    dirn = np.concatenate(direction)
    of = np.concatenate(owner_face)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    open_edges = int(np.sum(cnt == 1))
    nonmanifold = int(np.sum(cnt > 2))
    # orientation: an edge shared by two faces should be traversed in opposite directions
    two = np.nonzero(cnt == 2)[0]
    order = np.argsort(inv, kind="stable")
    pos = np.searchsorted(inv[order], two)
    i1, i2 = order[pos], order[pos + 1]
    flipped_pairs = int(np.sum(dirn[i1] == dirn[i2]))
    nf = fid0
    adj = sp.coo_matrix((np.ones(len(i1)), (of[i1], of[i2])), shape=(nf, nf))
    n_shells, shell = connected_components(adj + adj.T, directed=False)
    allp = np.concatenate([mesh.points[f[:, :3]] for f in faces])
    area = 0.5 * np.linalg.norm(np.cross(allp[:, 1] - allp[:, 0], allp[:, 2] - allp[:, 0]), axis=1)
    scale = np.linalg.norm(mesh.points.max(0) - mesh.points.min(0))
    degenerate = int(np.sum(area < 1e-12 * scale ** 2))
    keys = np.sort(np.concatenate([f[:, :3] for f in faces]), 1)
    dup = int(len(keys) - len(np.unique(keys, axis=0)))
    vol = None
    if open_edges == 0 and nonmanifold == 0:
        p = allp
        vol = float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)
    return {"faces": int(nf), "open_edges": open_edges, "nonmanifold_edges": nonmanifold,
            "inconsistent_orientation_edges": flipped_pairs, "shells": int(n_shells), "degenerate_faces": degenerate,
            "duplicate_faces": dup, "watertight": bool(open_edges == 0 and nonmanifold == 0),
            "enclosed_volume": vol, "area": float(area.sum()), "shell_of_face": shell}
