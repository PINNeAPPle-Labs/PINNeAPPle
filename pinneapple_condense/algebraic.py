"""Algebraic condensation: the method of :mod:`.substructure` for *any* assembled sparse operator.

Given a sparse matrix ``A`` (symmetric or not, saddle-point or not), the positions of its unknowns and a block
size, the unknowns are split into block interiors and a **separator**:

* every unknown gets the block that contains its position;
* of every coupled pair of unknowns (pattern of ``A + A^T``) in two different blocks, the one in the lower-numbered
  block is a separator -- so no interior unknown of one block couples to an interior unknown of another, whatever
  the stencil (finite volumes, staggered velocity-pressure, RBF-FD);
* ``primal`` (optional, per unknown) marks extra separators, e.g. one pressure per block of a saddle-point system,
  whose interior would otherwise be singular (pressure defined up to a constant).

Each block's interior is eliminated exactly: ``S = A_SS - sum_b A_SI^b (A_II^b)^-1 A_IS^b``. Blocks are compared
by **content** -- their interior and coupling submatrices in a canonical local order (positions relative to the
block, then field) -- and the factorisation and the local Schur complement of identical blocks are computed once.
Nothing is assumed about repetition: a block that differs in one entry is a new block.

**Reduced interfaces.** Separator unknowns shared by exactly two blocks form faces (keyed by the block pair and
field); unknowns shared by three or more blocks (corners) and primal unknowns stay unreduced. Each face is
projected on ``m`` modes of its face type (orientation and field), with the constant mode always included; the modes
come from :func:`face_modes` (POD of separator snapshots of a *different*, smaller problem). The reduced problem is
the Galerkin projection ``Phi^T S Phi``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Partition:
    """Blocks of the unknowns: ``block`` per unknown (flat id), ``sep`` mask, and per block its interior and adjacent
    separator unknowns in canonical local order."""
    block: np.ndarray
    sep: np.ndarray
    interiors: list[np.ndarray]
    adjacent: list[np.ndarray]
    shape: tuple[int, int]
    size: tuple[float, float]
    origin: tuple[float, float]
    coords: np.ndarray
    fieldid: np.ndarray


def _canonical(idx, coords, fieldid, origin, h):
    """Sort ``idx`` by position relative to ``origin`` (rounded to ``h / 1000``) then field."""
    rel = np.round((coords[idx] - origin) / (h * 1e-3)).astype(np.int64)
    order = np.lexsort((fieldid[idx], rel[:, 1], rel[:, 0]))
    return idx[order], rel[order]


def partition(A: sp.spmatrix, coords: np.ndarray, block_size: tuple[float, float], origin=(0.0, 0.0),
              fieldid: np.ndarray | None = None, primal: np.ndarray | None = None, h: float | None = None) -> Partition:
    """Blocks of side ``block_size`` from ``origin``; separators from the coupling graph (see the module docstring).
    ``h``: a length below the smallest spacing (for the canonical rounding; default block size / 1e4)."""
    n = A.shape[0]
    coords = np.asarray(coords, float)
    fieldid = np.zeros(n, np.int64) if fieldid is None else np.asarray(fieldid, np.int64)
    bx, by = block_size
    eps = 1e-9 * min(bx, by)
    ix = np.floor((coords[:, 0] - origin[0] + eps) / bx).astype(np.int64)
    iy = np.floor((coords[:, 1] - origin[1] + eps) / by).astype(np.int64)
    # blocks from the extent: an unknown exactly on the far boundary belongs to the last block, not a new one
    nbx = max(1, int(np.ceil((coords[:, 0].max() - origin[0]) / bx - 1e-9)))
    nby = max(1, int(np.ceil((coords[:, 1].max() - origin[1]) / by - 1e-9)))
    ix = np.clip(ix, 0, nbx - 1)
    iy = np.clip(iy, 0, nby - 1)
    block = ix * nby + iy
    G = (abs(A) + abs(A).T).tocoo()
    cross = block[G.row] != block[G.col]
    r, c = G.row[cross], G.col[cross]
    low = np.where(block[r] < block[c], r, c)
    sep = np.zeros(n, bool)
    sep[low] = True
    if primal is not None:
        sep |= np.asarray(primal, bool)
    h = h or 1e-4 * min(bx, by)
    Gc = G.tocsr()
    interiors, adjacent = [], []
    for b in range(nbx * nby):
        bi = np.nonzero((block == b) & ~sep)[0]
        org = (origin[0] + (b // nby) * bx, origin[1] + (b % nby) * by)
        if len(bi):
            nb = np.unique(Gc[bi].indices)
            S = nb[sep[nb]]
        else:
            S = np.zeros(0, np.int64)
        bi, _ = _canonical(bi, coords, fieldid, org, h)
        S, _ = _canonical(S, coords, fieldid, org, h)
        interiors.append(bi)
        adjacent.append(S)
    return Partition(block, sep, interiors, adjacent, (nbx, nby), (bx, by), tuple(origin), coords, fieldid)


def _key(*mats) -> str:
    hsh = hashlib.blake2b(digest_size=16)
    for M in mats:
        M = M.tocsr()
        M.sort_indices()
        for a in (np.asarray(M.shape), M.indptr, M.indices, M.data):
            hsh.update(np.ascontiguousarray(a).tobytes())
    return hsh.hexdigest()


@dataclass
class _Local:
    lu: object
    X: np.ndarray            # A_II^-1 A_IS (dense)
    Sb: np.ndarray           # A_SI A_II^-1 A_IS (dense)
    A_SI: sp.csr_matrix


@dataclass
class Condensation:
    """Exact algebraic condensation of ``A x = b`` over a :class:`Partition`."""
    A: sp.csr_matrix
    part: Partition
    timings: dict = field(default_factory=dict)

    def prepare(self) -> Condensation:
        t0 = time.perf_counter()
        A = self.A.tocsr()
        p = self.part
        self.sepidx = np.nonzero(p.sep)[0]
        self.slot = -np.ones(A.shape[0], np.int64)
        self.slot[self.sepidx] = np.arange(len(self.sepidx))
        self.local: dict[str, _Local] = {}
        self.keys: list[str | None] = []
        rows, cols, vals = [], [], []
        t_fac = 0.0
        for bi, S in zip(p.interiors, p.adjacent, strict=True):
            if not len(bi):
                self.keys.append(None)
                continue
            A_II, A_IS, A_SI = A[bi][:, bi], A[bi][:, S], A[S][:, bi]
            k = _key(A_II, A_IS, A_SI)
            self.keys.append(k)
            if k not in self.local:
                t1 = time.perf_counter()
                lu = spla.splu(A_II.tocsc())
                X = lu.solve(A_IS.toarray()) if len(S) else np.zeros((len(bi), 0))
                Sb = A_SI @ X if len(S) else np.zeros((0, 0))
                self.local[k] = _Local(lu, X, np.asarray(Sb), A_SI.tocsr())
                t_fac += time.perf_counter() - t1
            loc = self.local[k]
            if len(S):
                s = self.slot[S]
                rr, cc = np.meshgrid(s, s, indexing="ij")
                rows.append(rr.ravel())
                cols.append(cc.ravel())
                vals.append(-loc.Sb.ravel())
        A_SS = A[self.sepidx][:, self.sepidx].tocoo()
        rows.append(A_SS.row)
        cols.append(A_SS.col)
        vals.append(A_SS.data)
        ns = len(self.sepidx)
        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ns, ns))
        self.timings.update({"local": t_fac, "distinct_blocks": len(self.local),
                             "blocks": sum(k is not None for k in self.keys), "separator": ns,
                             "interior": int(A.shape[0] - ns), "prepare": time.perf_counter() - t0})
        return self

    def rhs(self, b: np.ndarray) -> np.ndarray:
        g = b[self.sepidx].astype(float).copy()
        for bi, S, k in zip(self.part.interiors, self.part.adjacent, self.keys, strict=True):
            if k is None or not len(S):
                continue
            loc = self.local[k]
            g[self.slot[S]] -= loc.A_SI @ loc.lu.solve(b[bi])
        return g

    def recover(self, xs: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = np.zeros(self.A.shape[0])
        x[self.sepidx] = xs
        for bi, S, k in zip(self.part.interiors, self.part.adjacent, self.keys, strict=True):
            if k is None:
                continue
            loc = self.local[k]
            x[bi] = loc.lu.solve(b[bi]) - (loc.X @ xs[self.slot[S]] if len(S) else 0.0)
        return x

    def solve(self, b: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        g = self.rhs(b)
        t1 = time.perf_counter()
        xs = spla.spsolve(self.S.tocsc(), g)
        t2 = time.perf_counter()
        x = self.recover(xs, b)
        return x, {"rhs_s": t1 - t0, "separator_solve_s": t2 - t1, "recover_s": time.perf_counter() - t2,
                   "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------ reduced faces
def faces(part: Partition, A: sp.spmatrix) -> dict:
    """Separator unknowns grouped by the set of blocks they couple to. Two blocks: a face, keyed by
    ``(orientation, field, length)`` (orientation 0: the two blocks are x-neighbours) with canonical positions along it;
    three or more blocks, or primal unknowns: kept full."""
    G = (abs(A) + abs(A).T).tocsr()
    sepidx = np.nonzero(part.sep)[0]
    nby = part.shape[1]
    groups: dict[tuple, list[int]] = {}
    for s in sepidx:
        nb = G.indices[G.indptr[s]:G.indptr[s + 1]]
        bl = tuple(sorted(set(part.block[nb].tolist()) | {int(part.block[s])}))
        groups.setdefault(bl, []).append(int(s))
    out = {"faces": [], "full": []}
    for bl, idx in groups.items():
        idx = np.asarray(idx)
        if len(bl) != 2:
            out["full"].extend(idx.tolist())
            continue
        a, b = bl
        orient = 0 if (b - a) == nby else 1 if (b - a) == 1 else -1
        if orient < 0:
            out["full"].extend(idx.tolist())
            continue
        along = 1 if orient == 0 else 0                  # an x-face runs along y, and the other way round
        for f in np.unique(part.fieldid[idx]):
            j = idx[part.fieldid[idx] == f]
            pos = part.coords[j, along]
            side = part.coords[j, 1 - along]
            o = np.lexsort((np.round(side / part.size[1 - along] * 1e6), np.round(pos / part.size[along] * 1e6)))
            out["faces"].append({"blocks": bl, "type": (orient, int(f), len(j)), "idx": j[o]})
    out["full"] = np.asarray(sorted(out["full"]), np.int64)
    return out


def face_modes(snapshots: list[np.ndarray], fc: dict, m: int) -> dict:
    """POD modes per face type from separator snapshots (full-length solution vectors of the snapshot problem with
    its face description ``fc``): the constant plus the ``m - 1`` leading left singular vectors of the
    constant-free face traces."""
    by_type: dict[tuple, list[np.ndarray]] = {}
    for f in fc["faces"]:
        for x in snapshots:
            by_type.setdefault(f["type"], []).append(x[f["idx"]])
    modes = {}
    for t, cols in by_type.items():
        lens = {len(c) for c in cols}
        if len(lens) != 1:
            raise ValueError(f"faces of type {t} differ in length {sorted(lens)} in the snapshot problem")
        X = np.stack(cols, 1)
        n = X.shape[0]
        one = np.ones((n, 1)) / np.sqrt(n)
        R = X - one @ (one.T @ X)
        U, sv, _ = np.linalg.svd(R, full_matrices=False)
        k = max(0, min(m - 1, int((sv > 1e-12 * max(sv[0], 1e-300)).sum())))
        modes[t] = np.hstack([one, U[:, :k]])
    return modes


@dataclass
class Reduced:
    """Galerkin-reduced separator problem of a :class:`Condensation` (exact when every face keeps all its modes)."""
    cond: Condensation
    fc: dict
    modes: dict

    def prepare(self) -> Reduced:
        t0 = time.perf_counter()
        c = self.cond
        rows, cols, vals, col = [], [], [], 0
        unseen = []
        for f in self.fc["faces"]:
            Phi = self.modes.get(f["type"])
            if Phi is None:                              # a face type the snapshot problem did not have: kept full
                unseen.append(f["idx"])
                continue
            s = c.slot[f["idx"]]
            rr, cc = np.meshgrid(s, np.arange(col, col + Phi.shape[1]), indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(Phi.ravel())
            col += Phi.shape[1]
        full = c.slot[np.concatenate([self.fc["full"]] + unseen).astype(np.int64)]
        rows.append(full)
        cols.append(np.arange(col, col + len(full)))
        vals.append(np.ones(len(full)))
        col += len(full)
        self.Phi = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(c.S.shape[0], col))
        self.Sr = (self.Phi.T @ c.S @ self.Phi).tocsc()
        self.lu = spla.splu(self.Sr)
        self.timings = {"reduced_dofs": col, "separator": c.S.shape[0], "unseen_faces": len(unseen),
                        "prepare": time.perf_counter() - t0}
        return self

    def solve(self, b: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        c = self.cond
        g = c.rhs(b)
        xs = self.Phi @ self.lu.solve(self.Phi.T @ g)
        x = c.recover(xs, b)
        return x, {"seconds": time.perf_counter() - t0}
