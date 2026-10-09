"""Finite-volume geometry of any mesh, computed the way OpenFOAM does (primitiveMesh): face centres and area vectors
by triangle decomposition about the face average point, cell centres and volumes by pyramid decomposition about the
average of the face centres. FE volume meshes are turned into the same face/owner/neighbour form first, so the
finite-volume metrics (non-orthogonality, skewness, aspect ratio) apply to Gmsh, VTK and Abaqus/CalculiX meshes too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .model import CELL_FACES, Mesh

TINY = 1e-300
_FLIP = {"tetra": (0, 2, 1, 3), "hexahedron": (0, 3, 2, 1, 4, 7, 6, 5), "wedge": (0, 2, 1, 3, 5, 4),
         "pyramid": (0, 3, 2, 1, 4)}


@dataclass
class FaceMesh:
    points: np.ndarray
    face_nodes: np.ndarray          # flat
    face_offsets: np.ndarray        # (nf+1,)
    owner: np.ndarray               # (nf,)
    neighbour: np.ndarray           # (n_internal,), faces 0..n_internal-1 are internal
    n_cells: int
    patch_of_face: np.ndarray       # (nf - n_internal,) patch index of each boundary face, -1 if none
    patch_names: List[str]
    patch_types: List[str]
    nonmanifold_faces: int = 0      # FE faces shared by more than two cells
    flipped_blocks: List[str] = None

    @property
    def n_faces(self) -> int:
        return len(self.owner)

    @property
    def n_internal(self) -> int:
        return len(self.neighbour)


def signed_volume_sign(btype: str, pts: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Sign of a simple volume estimate of each element (tetra decomposition about its first corner)."""
    p = pts[conn]
    if btype == "tetra":
        return np.sign(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]))
    c = p.mean(1)
    vol = np.zeros(len(conn))
    for f in CELL_FACES[btype]:                                       # outward faces: sum of pyramid volumes
        fc = p[:, list(f)].mean(1)
        for a, b in zip(f, f[1:] + f[:1]):
            vol += np.einsum("ij,ij->i", np.cross(p[:, a] - fc, p[:, b] - fc), fc - c)
    return np.sign(vol)


def oriented_blocks(mesh: Mesh) -> List[str]:
    """Element node-order conventions differ (VTK, Gmsh, Abaqus wedges...). If most elements of a block come out with
    negative volume, the whole block uses the opposite convention: flip it, so only truly inverted elements stay
    negative. Returns the names of the flipped blocks."""
    flipped = []
    for b in mesh.blocks:
        t = b.linear
        if t not in _FLIP or b.dim != 3:
            continue
        s = signed_volume_sign(t, mesh.points, b.corners)
        if np.mean(s < 0) > 0.5:
            perm = list(_FLIP[t]) + list(range(len(_FLIP[t]), b.conn.shape[1]))
            b.conn = b.conn[:, perm] if b.conn.shape[1] == len(_FLIP[t]) else np.concatenate(
                [b.corners[:, list(_FLIP[t])], b.conn[:, len(_FLIP[t]):]], 1)
            flipped.append(b.type)
    return flipped


def face_mesh(mesh: Mesh) -> FaceMesh:
    """OpenFOAM polyMesh as is; FE volume meshes converted (internal faces first, then boundary faces)."""
    if "facemesh" in mesh._cache:
        return mesh._cache["facemesh"]
    if mesh.poly is not None:
        p = mesh.poly
        pof = np.full(p.n_faces - p.n_internal, -1, np.int64)
        for k, pt in enumerate(p.patches):
            pof[pt["startFace"] - p.n_internal: pt["startFace"] - p.n_internal + pt["nFaces"]] = k
        fm = FaceMesh(mesh.points, p.face_nodes, p.face_offsets, p.owner, p.neighbour, p.n_cells, pof,
                      [x["name"] for x in p.patches], [x["type"] for x in p.patches], 0, [])
        mesh._cache["facemesh"] = fm
        return fm
    if mesh.dim != 3:
        raise ValueError("finite-volume geometry needs a 3D volume mesh")
    flipped = oriented_blocks(mesh)
    rows, cell_of = [], []
    start = 0
    for b in mesh.blocks:
        if b.dim != 3:
            start += len(b.conn)
            continue
        cor = b.corners
        cid = np.arange(start, start + len(cor))
        for f in CELL_FACES[b.linear]:
            q = np.full((len(cor), 4), -1, np.int64)
            q[:, : len(f)] = cor[:, list(f)]
            rows.append(q)
            cell_of.append(cid)
        start += len(b.conn)
    allf = np.concatenate(rows)
    cells = np.concatenate(cell_of)
    key = np.sort(np.where(allf < 0, np.iinfo(np.int64).max, allf), 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    first = np.ones(len(order), bool)
    first[1:] = inv[order][1:] != inv[order][:-1]
    fi = order[first]                                                # first occurrence of each unique face
    c_face = cnt[inv[fi]]
    second = np.full(len(cnt), -1, np.int64)
    rest = order[~first]
    keep_rest = np.ones(len(rest), bool)
    keep_rest[1:] = inv[rest][1:] != inv[rest][:-1]                  # second occurrence only
    second[inv[rest[keep_rest]]] = rest[keep_rest]
    internal = c_face >= 2
    int_f, bnd_f = fi[internal], fi[~internal]
    own = np.concatenate([cells[int_f], cells[bnd_f]])
    nei = cells[second[inv[int_f]]]
    faces = np.concatenate([allf[int_f], allf[bnd_f]])
    sizes = (faces >= 0).sum(1)
    flat = faces[faces >= 0]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    # boundary patches from named 2D groups (Gmsh physical surfaces, element sets): match by node set
    pof = np.full(len(bnd_f), -1, np.int64)
    names, types = [], []
    if mesh.boundary_blocks:
        bkey = np.sort(np.where(allf[bnd_f] < 0, np.iinfo(np.int64).max, allf[bnd_f]), 1)
        lookup = {tuple(r): i for i, r in enumerate(bkey)}
        for name, blk in mesh.boundary_blocks:
            if blk.dim != 2:
                continue
            k = len(names)
            names.append(name)
            types.append("patch")
            q = np.full((len(blk.corners), 4), -1, np.int64)
            q[:, : blk.corners.shape[1]] = blk.corners
            for r in np.sort(np.where(q < 0, np.iinfo(np.int64).max, q), 1):
                i = lookup.get(tuple(r))
                if i is not None:
                    pof[i] = k
    fm = FaceMesh(mesh.points, flat, offs, own, nei, mesh.n_cells, pof, names, types,
                  int(np.sum(c_face > 2)), flipped)
    mesh._cache["facemesh"] = fm
    return fm


def face_geometry(fm: FaceMesh) -> Dict[str, np.ndarray]:
    """Face centres (nf,3), area vectors (nf,3), and the per-face corner groups used later."""
    nf = fm.n_faces
    sizes = np.diff(fm.face_offsets)
    ctr = np.zeros((nf, 3))
    area = np.zeros((nf, 3))
    flat_area_ratio = np.ones(nf)
    for k in np.unique(sizes):
        idx = np.nonzero(sizes == k)[0]
        nodes = fm.face_nodes[fm.face_offsets[idx][:, None] + np.arange(k)]
        p = fm.points[nodes]                                          # (m, k, 3)
        if k == 3:
            area[idx] = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            ctr[idx] = p.mean(1)
            continue
        fc = p.mean(1)
        sum_n = np.zeros((len(idx), 3))
        sum_a = np.zeros(len(idx))
        sum_ac = np.zeros((len(idx), 3))
        for i in range(k):
            a, b = p[:, i], p[:, (i + 1) % k]
            c = a + b + fc
            n = np.cross(b - a, fc - a)
            mag = np.linalg.norm(n, axis=1)
            sum_n += n
            sum_a += mag
            sum_ac += mag[:, None] * c
        ctr[idx] = (1.0 / 3.0) * sum_ac / np.maximum(sum_a, TINY)[:, None]
        area[idx] = 0.5 * sum_n
        flat_area_ratio[idx] = np.linalg.norm(sum_n, axis=1) / np.maximum(sum_a, TINY)
    return {"centres": ctr, "areas": area, "flatness": flat_area_ratio}


def cell_geometry(fm: FaceMesh, fg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    nc, ni = fm.n_cells, fm.n_internal
    own, nei = fm.owner, fm.neighbour
    fc, sf = fg["centres"], fg["areas"]
    nfaces = np.bincount(own, minlength=nc) + np.bincount(nei, minlength=nc)
    c_est = np.zeros((nc, 3))
    for d in range(3):
        c_est[:, d] = (np.bincount(own, fc[:, d], nc) + np.bincount(nei, fc[: ni, d], nc)) / np.maximum(nfaces, 1)
    pv_own = np.einsum("ij,ij->i", sf, fc - c_est[own])
    pv_nei = np.einsum("ij,ij->i", sf[:ni], c_est[nei] - fc[:ni])
    pv_own_c, pv_nei_c = pv_own, pv_nei                              # OpenFOAM v1912: no clipping
    pc_own = 0.75 * fc + 0.25 * c_est[own]
    pc_nei = 0.75 * fc[:ni] + 0.25 * c_est[nei]
    vol = np.bincount(own, pv_own_c, nc) + np.bincount(nei, pv_nei_c, nc)
    ctr = np.zeros((nc, 3))
    for d in range(3):
        ctr[:, d] = (np.bincount(own, pv_own_c * pc_own[:, d], nc) + np.bincount(nei, pv_nei_c * pc_nei[:, d], nc))
    ok = np.abs(vol) > 1e-300
    ctr[ok] /= vol[ok][:, None]
    ctr[~ok] = c_est[~ok]
    signed = (np.bincount(own, pv_own, nc) + np.bincount(nei, pv_nei, nc)) / 3.0
    return {"centres": ctr, "volumes": vol / 3.0, "signed_volumes": signed, "n_faces": nfaces,
            "pyramid_own": pv_own, "pyramid_nei": pv_nei}


def cell_adjacency(fm: FaceMesh) -> sp.csr_matrix:
    ni = fm.n_internal
    a = sp.coo_matrix((np.ones(ni), (fm.owner[:ni], fm.neighbour)), shape=(fm.n_cells, fm.n_cells))
    return (a + a.T).tocsr()


def regions(adj: sp.csr_matrix) -> np.ndarray:
    return connected_components(adj, directed=False)[1]


def element_adjacency_by_nodes(mesh: Mesh) -> sp.csr_matrix:
    """Cells sharing at least one node (for surface/planar meshes and the node-connectivity check)."""
    rows, cols = [], []
    start = 0
    for b in mesh.blocks:
        if b.dim != mesh.dim:
            start += len(b.conn)
            continue
        cor = b.corners
        cid = np.repeat(np.arange(start, start + len(cor)), cor.shape[1])
        rows.append(cid)
        cols.append(cor.ravel())
        start += len(b.conn)
    r, c = np.concatenate(rows), np.concatenate(cols)
    inc = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(mesh.n_cells, mesh.n_points))
    return (inc @ inc.T).tocsr()


def solution_dirs(fm: FaceMesh, fg: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Directions with no 'empty' patch (OpenFOAM 2D/1D cases): boolean (3,)."""
    if "empty" not in fm.patch_types:
        return np.ones(3, bool)
    ni = fm.n_internal
    mask = np.isin(fm.patch_of_face, [i for i, t in enumerate(fm.patch_types) if t == "empty"])
    n = fg["areas"][ni:][mask]
    n = np.abs(n) / np.maximum(np.linalg.norm(n, axis=1), TINY)[:, None]
    return ~(n.mean(0) > 0.5)
