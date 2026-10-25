"""Hierarchical (nested) condensation with reuse of identical subtrees.

The block array is split recursively in halves (the longer side first) down to single blocks. Each node of the tree
is a rectangle of blocks whose *ports* are the reduced interface dofs of its boundary faces (west, east, south,
north; ``m`` modes per face, :func:`.substructure.interface_modes`). Merging two children eliminates their shared
faces (a dense Schur complement), which is nested dissection on the reduced interfaces.

Two nodes with the same shape and the same children (recursively, the same arrangement of block variants) have the
same operator, so operators are memoised by that key: a uniform ``2^L x 2^L`` array needs ``2L`` merges whatever its
``4^L`` blocks, and an array with a few variants needs a few per level. The cost to build therefore follows the
number of *distinct* subtrees (the physical complexity of the layout), plus the dense algebra on the top separator,
whose size grows with the side of the array (sqrt of the block count).

A power map is pushed up the tree (each node condenses its children's loads with its stored elimination), the root
system is solved, and port values are pushed down; the downward pass can be limited to the path to one block
(:meth:`Hierarchy.block_ports`), which is what a point query needs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .package import BASE, Package
from .substructure import FLUXES, BlockModel, block_model, interface_modes


@dataclass
class _Op:
    """A memoised node operator."""
    dims: tuple[int, int]
    split: str | None = None
    kids: tuple[int, int] | None = None
    S: np.ndarray | None = None
    X: np.ndarray | None = None
    cho: tuple | None = None
    n_o: int = 0
    mapA: np.ndarray | None = None
    mapB: np.ndarray | None = None


def _port_ranges(nx: int, ny: int, m: int) -> dict[str, np.ndarray]:
    o = np.cumsum([0, ny, ny, nx, nx])
    return {s: np.arange(o[i] * m, o[i + 1] * m) for i, s in enumerate("WESN")}


def _maps(split: str, a: tuple[int, int], b: tuple[int, int], m: int):
    """Positions of the children's ports in the parent's assembled vector [outer ports; shared ports]."""
    pa, pb = _port_ranges(*a, m), _port_ranges(*b, m)
    if split == "x":
        nx, ny = a[0] + b[0], a[1]
    else:
        nx, ny = a[0], a[1] + b[1]
    pp = _port_ranges(nx, ny, m)
    n_o = 2 * (nx + ny) * m
    mA = np.empty(sum(len(v) for v in pa.values()), int)
    mB = np.empty(sum(len(v) for v in pb.values()), int)
    if split == "x":
        sh = n_o + np.arange(ny * m)
        mA[pa["W"]], mA[pa["E"]] = pp["W"], sh
        mA[pa["S"]], mA[pa["N"]] = pp["S"][: a[0] * m], pp["N"][: a[0] * m]
        mB[pb["W"]], mB[pb["E"]] = sh, pp["E"]
        mB[pb["S"]], mB[pb["N"]] = pp["S"][a[0] * m:], pp["N"][a[0] * m:]
        n_i = ny * m
    else:
        sh = n_o + np.arange(nx * m)
        mA[pa["S"]], mA[pa["N"]] = pp["S"], sh
        mA[pa["W"]], mA[pa["E"]] = pp["W"][: a[1] * m], pp["E"][: a[1] * m]
        mB[pb["S"]], mB[pb["N"]] = sh, pp["N"]
        mB[pb["W"]], mB[pb["E"]] = pp["W"][a[1] * m:], pp["E"][a[1] * m:]
        n_i = nx * m
    return (nx, ny), n_o, n_i, mA, mB


class Hierarchy:
    """Nested condensation of a package (uniform ``m`` interface modes on every face).

    ``prepare()``: block condensation (per distinct variant), interface modes, and one merge per distinct subtree.
    ``solve(power)``: upward load pass, root solve, downward pass; ``observables()`` as in
    :class:`.substructure.Condensed`."""

    def __init__(self, pkg: Package, m: int = 6, *, modes: str = "pod", keep_root: bool = True):
        self.pkg, self.m, self.modes, self.keep_root = pkg, m, modes, keep_root
        self.timings: dict[str, float] = {}

    # -- tree ----------------------------------------------------------------------------------------------------
    def _build_tree(self):
        nbx, nby = self.pkg.shape
        levels = [np.array([[0, nbx, 0, nby]])]
        parents = []
        while True:
            R = levels[-1]
            nx, ny = R[:, 1] - R[:, 0], R[:, 3] - R[:, 2]
            inner = np.nonzero((nx > 1) | (ny > 1))[0]
            if not len(inner):
                break
            r = R[inner]
            sx = (nx[inner] >= ny[inner])
            mid_x = r[:, 0] + nx[inner] // 2
            mid_y = r[:, 2] + ny[inner] // 2
            A = r.copy()
            B = r.copy()
            A[sx, 1], B[sx, 0] = mid_x[sx], mid_x[sx]
            A[~sx, 3], B[~sx, 2] = mid_y[~sx], mid_y[~sx]
            kids = np.concatenate([A, B])
            k = len(inner)
            parents.append({"inner": inner, "split_x": sx, "A": np.arange(k), "B": np.arange(k, 2 * k)})
            levels.append(kids)
        self.levels, self.parents = levels, parents

    def prepare(self) -> Hierarchy:
        t0 = time.perf_counter()
        pkg, st, m = self.pkg, self.pkg.stack, self.m
        self.models: list[BlockModel] = [block_model(st, v) for v in pkg.variants]
        self.timings["blocks"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        base = block_model(st, BASE)
        basis = {o: interface_modes(base, o, m, self.modes) for o in ("x", "y")}
        Phi = sla.block_diag(basis["x"], basis["x"], basis["y"], basis["y"])
        self.Phi = Phi
        if Phi.shape[1] != 4 * m:
            raise ValueError("m must be smaller than the port nodes of a face")
        self.leaf = [{"Sr": Phi.T @ bm.S @ Phi, "Wr": Phi.T @ bm.W, "Otr": bm.O_t @ Phi, "Op": bm.O_p}
                     for bm in self.models]
        self.timings["reduce"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        self._build_tree()
        # keys bottom-up: leaf key = variant index; internal key = id of (split, keyA, keyB)
        self.ops: list[_Op] = [_Op((1, 1), S=lf["Sr"]) for lf in self.leaf]
        memo: dict[tuple, int] = {}
        keys = [None] * len(self.levels)
        for lev in range(len(self.levels) - 1, -1, -1):
            R = self.levels[lev]
            leafy = ((R[:, 1] - R[:, 0]) == 1) & ((R[:, 3] - R[:, 2]) == 1)
            kk = np.empty(len(R), int)
            kk[leafy] = pkg.layout[R[leafy, 0], R[leafy, 2]]
            if lev < len(self.parents):
                par = self.parents[lev]
                ka, kb = keys[lev + 1][par["A"]], keys[lev + 1][par["B"]]
                trip = np.stack([par["split_x"].astype(int), ka, kb], 1)
                uniq, inv = np.unique(trip, axis=0, return_inverse=True)
                ids = np.empty(len(uniq), int)
                for u, (sx, a, b) in enumerate(uniq):
                    t = (int(sx), int(a), int(b))
                    if t not in memo:
                        memo[t] = self._merge("x" if sx else "y", int(a), int(b))
                    ids[u] = memo[t]
                kk[par["inner"]] = ids[inv.ravel()]
            keys[lev] = kk
        self.keys = keys
        self.timings["merges"] = time.perf_counter() - t2
        self.timings["distinct_nodes"] = len(self.ops)
        self.timings["distinct_merges"] = len(memo)
        t3 = time.perf_counter()
        root = self.ops[int(keys[0][0])]
        self.root_cho = sla.cho_factor(root.S)
        if not self.keep_root:
            root.S = None
        self.timings["root"] = time.perf_counter() - t3
        for op in self.ops:                       # only the eliminations are needed from here on
            if op is not root:
                op.S = None
        self.timings["prepare"] = time.perf_counter() - t0
        self.timings["root_ports"] = int(root.S.shape[0]) if root.S is not None else int(len(self.root_cho[0]))
        return self

    def _merge(self, split: str, a: int, b: int) -> int:
        A, B = self.ops[a], self.ops[b]
        dims, n_o, n_i, mA, mB = _maps(split, A.dims, B.dims, self.m)
        N = n_o + n_i
        S = np.zeros((N, N))
        S[np.ix_(mA, mA)] += A.S
        S[np.ix_(mB, mB)] += B.S
        Soo, Soi, Sii = S[:n_o, :n_o], S[:n_o, n_o:], S[n_o:, n_o:]
        cho = sla.cho_factor(Sii)
        X = sla.cho_solve(cho, Soi.T).T                     # S_oi S_ii^-1
        Sp = Soo - X @ Soi.T
        self.ops.append(_Op(dims, split, (a, b), 0.5 * (Sp + Sp.T), X, cho, n_o, mA, mB))
        return len(self.ops) - 1

    # -- solve ---------------------------------------------------------------------------------------------------
    def _groups(self, lev: int) -> dict[int, np.ndarray]:
        """Nodes of a level grouped by operator key (cached), and each node's row in its group."""
        if not hasattr(self, "_grp"):
            self._grp, self._row = {}, {}
        if lev not in self._grp:
            kk = self.keys[lev]
            g = {int(k): np.nonzero(kk == k)[0] for k in np.unique(kk)}
            row = np.empty(len(kk), int)
            for nodes in g.values():
                row[nodes] = np.arange(len(nodes))
            self._grp[lev], self._row[lev] = g, row
        return self._grp[lev]

    def upward(self, pb: np.ndarray):
        """Condensed loads of every node, per level ``{key: (n_nodes, n_ports)}``, and the shared-port loads of the
        merges, per level ``{key: (n_nodes, n_shared)}``."""
        L = len(self.levels)
        loads: list[dict] = [None] * L
        inner: list[dict] = [None] * L
        for lev in range(L - 1, -1, -1):
            R = self.levels[lev]
            groups = self._groups(lev)
            out, gin = {}, {}
            is_inner = np.zeros(len(R), bool)
            pos = np.full(len(R), -1)
            if lev < len(self.parents):
                par = self.parents[lev]
                is_inner[par["inner"]] = True
                pos[par["inner"]] = np.arange(len(par["inner"]))
                crow = self._row[lev + 1]
            for key, nodes in groups.items():
                if not is_inner[nodes[0]]:
                    bi, bj = R[nodes, 0], R[nodes, 2]
                    out[key] = pb[bi, bj] @ self.leaf[key]["Wr"].T
                    continue
                op = self.ops[key]
                p = pos[nodes]
                ka, kb = op.kids
                gA = loads[lev + 1][ka][crow[par["A"][p]]]
                gB = loads[lev + 1][kb][crow[par["B"][p]]]
                asm = np.zeros((len(nodes), op.n_o + op.X.shape[1]))
                asm[:, op.mapA] += gA
                asm[:, op.mapB] += gB
                gi = asm[:, op.n_o:]
                out[key] = asm[:, :op.n_o] - gi @ op.X.T
                gin[key] = gi
            loads[lev], inner[lev] = out, gin
        return loads, inner

    def solve(self, power: np.ndarray | None = None) -> dict:
        t0 = time.perf_counter()
        pb = self.pkg.block_power(power)
        loads, inner = self.upward(pb)
        t_up = time.perf_counter() - t0
        rk = int(self.keys[0][0])
        t_root = sla.cho_solve(self.root_cho, loads[0][rk][0])
        nbx, nby = self.pkg.shape
        tb = np.empty((nbx, nby, 4 * self.m))
        ports = {rk: t_root[None, :]}
        for lev in range(len(self.levels)):
            R = self.levels[lev]
            groups = self._groups(lev)
            if lev == len(self.parents):
                nxt = None
            else:
                par = self.parents[lev]
                pos = np.full(len(R), -1)
                pos[par["inner"]] = np.arange(len(par["inner"]))
                nxt_groups = self._groups(lev + 1)
                crow = self._row[lev + 1]
                nxt = {k: np.empty((len(v), self._n_ports(k))) for k, v in nxt_groups.items()}
            for key, nodes in groups.items():
                T = ports[key]
                op = self.ops[key]
                if op.kids is None:
                    tb[R[nodes, 0], R[nodes, 2]] = T
                    continue
                ti = sla.cho_solve(op.cho, inner[lev][key].T).T - T @ op.X
                ta = np.concatenate([T, ti], 1)
                p = pos[nodes]
                ka, kb = op.kids
                nxt[ka][crow[par["A"][p]]] = ta[:, op.mapA]
                nxt[kb][crow[par["B"][p]]] = ta[:, op.mapB]
            ports = nxt
        return {"tb": tb, "pb": pb, "seconds": time.perf_counter() - t0, "seconds_upward": t_up,
                "inner": inner, "t_root": t_root}

    def _n_ports(self, key: int) -> int:
        nx, ny = self.ops[key].dims
        return 2 * (nx + ny) * self.m

    def observables(self, sol: dict) -> dict[str, np.ndarray]:
        n2 = self.pkg.stack.n ** 2
        tb, pb = sol["tb"], sol["pb"]
        out = np.empty(tb.shape[:2] + (n2 + len(FLUXES),))
        for v, lf in enumerate(self.leaf):
            s = self.pkg.layout == v
            out[s] = tb[s] @ lf["Otr"].T + pb[s] @ lf["Op"].T
        res = {"Tj": out[..., :n2]}
        for i, name in enumerate(FLUXES):
            res[name] = out[..., n2 + i]
        return res

    def block_ports(self, sol: dict, bi: int, bj: int) -> np.ndarray:
        """Reduced port values of block ``(I, J)`` by the downward pass along its root-to-leaf path only (needs the
        upward pass of ``sol``; one small dense product per level instead of the whole tree)."""
        to, node = sol["t_root"], 0
        for lev, par in enumerate(self.parents):
            pos = np.nonzero(par["inner"] == node)[0]
            if not len(pos):
                break
            i = pos[0]
            key = int(self.keys[lev][node])
            op = self.ops[key]
            gi = sol["inner"][lev][key][self._row[lev][node]]
            ti = sla.cho_solve(op.cho, gi) - op.X.T @ to
            ta = np.concatenate([to, ti])
            a, b = par["A"][i], par["B"][i]
            ra = self.levels[lev + 1][a]
            if ra[0] <= bi < ra[1] and ra[2] <= bj < ra[3]:
                to, node = ta[op.mapA], a
            else:
                to, node = ta[op.mapB], b
        return to

    def recover(self, sol: dict, bi: int, bj: int, tb: np.ndarray | None = None) -> np.ndarray:
        """Cell temperatures of block ``(I, J)`` (path-only downward pass unless ``tb`` is given)."""
        t = self.block_ports(sol, bi, bj) if tb is None else tb
        return self.models[self.pkg.layout[bi, bj]].recover(self.Phi @ t, sol["pb"][bi, bj])


def _face_layout(shape: tuple[int, int]):
    nbx, nby = shape
    bi, bj = np.meshgrid(np.arange(nbx), np.arange(nby), indexing="ij")
    nxf = (nbx + 1) * nby
    faces = np.stack([bi * nby + bj, (bi + 1) * nby + bj, nxf + bi * (nby + 1) + bj, nxf + bi * (nby + 1) + bj + 1], -1)
    n = nxf + nbx * (nby + 1)
    sides = np.full(n, 2)
    fx = np.arange(nxf).reshape(nbx + 1, nby)
    fy = nxf + np.arange(nbx * (nby + 1)).reshape(nbx, nby + 1)
    sides[np.r_[fx[0], fx[-1], fy[:, 0], fy[:, -1]]] = 1
    return faces, nxf, sides


def estimate(h: Hierarchy, sol: dict, chunk: int = 4096) -> np.ndarray:
    """Per-face error indicator of a hierarchical solution, as :meth:`.substructure.Condensed.estimate`: the fine
    residual on the port nodes (interiors are exact) measured in the inverse two-sided face Schur complement."""
    pkg = h.pkg
    base = block_model(pkg.stack, BASE)
    nf = base.ports["x-"].stop - base.ports["x-"].start
    faces, nxf, sides = _face_layout(pkg.shape)
    r = np.zeros((len(sides), nf))
    tb, pb = sol["tb"], sol["pb"]
    for v, bm in enumerate(h.models):
        sel = np.argwhere(pkg.layout == v)
        for c0 in range(0, len(sel), chunk):
            s = sel[c0:c0 + chunk]
            rb = (tb[s[:, 0], s[:, 1]] @ h.Phi.T) @ bm.S.T - pb[s[:, 0], s[:, 1]] @ bm.W.T
            f = faces[s[:, 0], s[:, 1]]
            for i, name in enumerate(("x-", "x+", "y-", "y+")):
                np.add.at(r, f[:, i], rb[:, bm.ports[name]])
    eta = np.empty(len(sides))
    for o, sl in (("x", slice(0, nxf)), ("y", slice(nxf, len(sides)))):
        lo, hi = base.ports[o + "-"], base.ports[o + "+"]
        for sd, M in ((2, base.S[lo, lo] + base.S[hi, hi]), (1, base.S[lo, lo])):
            idx = np.nonzero(sides[sl] == sd)[0] + sl.start
            if len(idx):
                cf = sla.cho_factor(M)
                for c0 in range(0, len(idx), chunk * 4):
                    ii = idx[c0:c0 + chunk * 4]
                    eta[ii] = np.sqrt(np.maximum(0.0, np.einsum("ij,ji->i", r[ii], sla.cho_solve(cf, r[ii].T))))
    return eta
