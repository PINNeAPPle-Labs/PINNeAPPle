"""Static condensation of repeated blocks, reduced interfaces, on-demand recovery and an error-driven refinement.

Every block of a :class:`.package.Package` is one voxel box with *port* nodes on its four lateral faces
(:func:`.fv.assemble`). Eliminating its cells gives the block's Schur complement on its ports,

    S = K_pp - K_pc K_cc^-1 K_cp,      g = W p,   W = -K_pc K_cc^-1 E,

with ``p`` the power of its active cells. Blocks with the same :class:`.package.Variant` have the same ``S``, ``W``
and the same factor of ``K_cc``: :func:`block_model` computes them once per distinct block (a cache keyed by stack
and variant), whatever the number of blocks.

The port field of each shared face is then restricted to a few *interface modes* ``Phi`` (the constant plus the
softest modes of the face's two-sided Schur complement, area weighted: characteristic constraint modes after
Castanier, Tan and Pierre, AIAA J. 39 (2001) 1182). The reduced global problem is a Galerkin restriction of the
exact one (an energy-norm best approximation over the chosen interface space; with the constant in every face basis
the global energy balance holds exactly). With ``Phi = I`` on every face it is the exact fine-grid solution.

Interior fields are never formed unless asked for: :meth:`Condensed.recover` rebuilds one block from its ports
(``T_c = Z p - Y t``), and linear observables (junction temperatures, the heat through each bump, TIM, lid) come
from precomputed per-block maps.

Error estimate: the interiors are exact, so the fine residual lives only on the port nodes; per face
``eta_F^2 = r_F^T S_FF^-1 r_F`` (a face-block Jacobi approximation of ``r^T S^-1 r``, the squared energy norm of the
error). :meth:`Condensed.refine` restores the full port field (``Phi = I``) on the faces holding the largest share
of the estimate (Doerfler marking) and solves again.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import fv
from .package import BASE, Package, Stack, Variant

#: names of the per-block linear observables (``Tj`` is a vector: the n x n junction cells)
FLUXES = ("q_tim", "q_ubump", "q_tsv", "q_c4", "q_top", "q_bottom")


@dataclass
class BlockModel:
    """The condensed form of one distinct block (see the module docstring for the symbols)."""
    stack: Stack
    variant: Variant
    S: np.ndarray
    W: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    ports: dict[str, slice]
    Om: sp.csr_matrix
    obs_names: list[str]
    seconds: float
    O_p: np.ndarray = field(default=None, repr=False)
    O_t: np.ndarray = field(default=None, repr=False)

    @property
    def n_ports(self) -> int:
        return self.S.shape[0]

    def recover(self, t_ports: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Cell temperatures of the block, ``(n, n, nz)``, from its fine port temperatures and active powers."""
        n, nz = self.stack.n, self.stack.nz
        return (self.Z @ p - self.Y @ t_ports).reshape(n, n, nz)


def block_functionals(stack: Stack, variant: Variant, sysb: fv.System) -> tuple[sp.csr_matrix, list[str]]:
    """Rows of the linear observables of one block over its cells: the ``n * n`` junction temperatures, then the
    vertical heat (W, positive upwards) through the die/TIM interface, the microbump, the TSV and the C4 bump (each
    at the middle face of its layer, over the inclusion cells), and the Robin outflow (positive out) of the top
    and bottom."""
    n, nz = stack.n, stack.nz
    nc = n * n * nz
    idx = np.arange(nc).reshape(n, n, nz)
    sl = stack.layer_slices()
    props = stack.properties(variant)
    rows = [sp.csr_matrix((np.ones(n * n), (np.arange(n * n), idx[:, :, stack.active_k].ravel())), shape=(n * n, nc))]
    names = [f"Tj[{i}]" for i in range(n * n)]

    def face_flux(f, mask):
        g = (sysb.Gz[:, :, f] * mask).ravel()
        lo, hi = idx[:, :, f].ravel(), idx[:, :, f + 1].ravel()
        r = np.zeros(nc)
        np.add.at(r, lo, g)
        np.add.at(r, hi, -g)
        return r
    full = np.ones((n, n))
    vec = [face_flux(sl["die"].stop - 1, full)]
    for layer in ("ubump", "interposer", "c4"):
        s = sl[layer]
        mask = props["masks"][layer] if props["masks"][layer].any() else full
        vec.append(face_flux(s.start + (s.stop - s.start) // 2 - 1, mask))
    r = np.zeros(nc)
    r[idx[:, :, -1].ravel()] = sysb.g_top.ravel()
    vec.append(r)
    r = np.zeros(nc)
    r[idx[:, :, 0].ravel()] = sysb.g_bottom.ravel()
    vec.append(r)
    rows.append(sp.csr_matrix(np.array(vec)))
    return sp.vstack(rows).tocsr(), names + list(FLUXES)


_CACHE: dict[tuple, BlockModel] = {}


def block_model(stack: Stack, variant: Variant = BASE, *, cache: bool = True, shift: float = 0.0) -> BlockModel:
    """Condense one block (cached per ``(stack, variant, shift)``). ``shift`` ``s`` condenses ``K + s C`` instead
    of ``K`` (the Laplace-domain operator at rate ``s`` [1/s]; one implicit-Euler step for ``s = 1/dt``)."""
    key = (stack, variant.key, float(shift))
    if cache and key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    props = stack.properties(variant)
    sysb = fv.assemble((props["k_lat"], props["k_lat"], props["k"]), props["dz"], props["R"], stack.dx, stack.h_top, stack.h_bottom, ports=True,
                       rho_c=props["rho_c"] if shift else None)
    nc = sysb.n_cells
    K = sysb.K.tocsr()
    if shift:
        K = (K + sp.diags(np.concatenate([shift * sysb.C, np.zeros(K.shape[0] - nc)]))).tocsr()
    Kcc, Kcp, Kpp = K[:nc, :nc], K[:nc, nc:], K[nc:, nc:]
    lu = fv.factor(Kcc)
    Y = lu.solve(Kcp.toarray())
    S = Kpp.toarray() - Kcp.T @ Y
    S = 0.5 * (S + S.T)
    n = stack.n
    idx = np.arange(nc).reshape(n, n, stack.nz)
    E = np.zeros((nc, n * n))
    E[idx[:, :, stack.active_k].ravel(), np.arange(n * n)] = 1.0
    Z = lu.solve(E)
    W = -(Kcp.T @ Z)
    ports = {k: slice(v.start - nc, v.stop - nc) for k, v in sysb.ports.items()}
    Om, names = block_functionals(stack, variant, sysb)
    bm = BlockModel(stack, variant, S, np.asarray(W), Y, Z, ports, Om, names, time.perf_counter() - t0)
    bm.O_p = np.asarray(Om @ Z)
    bm.O_t = -np.asarray(Om @ Y)
    if cache:
        _CACHE[key] = bm
    return bm


def clear_cache() -> None:
    _CACHE.clear()


def face_weights(stack: Stack) -> np.ndarray:
    """Area (m^2) of each port node of a face, order ``(lateral, k)``."""
    return (stack.dx * stack.dz()[None, :] * np.ones((stack.n, 1))).ravel()


_SNAPSHOTS: dict[tuple, dict[str, np.ndarray]] = {}


def trace_snapshots(stack: Stack, patch: int = 4, shifts: tuple[float, ...] = (0.0,), *, global_size: int = 16,
                    global_maps: int = 24) -> dict[str, np.ndarray]:
    """Interface traces the face modes are built from (cached per stack), two groups of equal weight:

    * *local*: the interior faces of a ``patch x patch`` array of base blocks under every unit load (one active cell
      at a time), solved exactly -- every near-field trace a power map can produce; with ``shifts``, also for
      ``K + s C`` at each rate ``s`` (transient traces, Laplace domain), each shift with equal weight;
    * *global*: the interior faces of a ``global_size x global_size`` package under ``global_maps`` random hotspot
      maps (background, number and width of the hotspots drawn at random), solved exactly by the unreduced
      hierarchy -- the far-field traces where the lid, die, interposer and substrate spread heat laterally at
      different rates. Without them the modes fit the near field only and the error grows with the package
      (measured: 8.8 K on Tmax at 64 x 64 blocks with 6 modes, see the report).

    Returns ``{"x": (n_face, n_snap), "y": ...}``."""
    key = (stack, patch, tuple(float(x) for x in shifts), global_size, global_maps)
    if key in _SNAPSHOTS:
        return _SNAPSHOTS[key]
    path = _disk_cache(key)
    if path is not None and path.exists():
        with np.load(path) as z:
            _SNAPSHOTS[key] = {"x": z["x"], "y": z["y"]}
        return _SNAPSHOTS[key]
    n2 = stack.n ** 2
    pkg = Package(stack, np.zeros((patch, patch), int), [BASE], np.zeros((patch * stack.n,) * 2))
    pbs = np.zeros((patch * patch * n2, patch, patch, n2))
    for b in range(patch * patch):
        bi, bj = divmod(b, patch)
        pbs[b * n2:(b + 1) * n2, bi, bj, :] = np.eye(n2)
    nxf = (patch + 1) * patch
    fx = np.arange(nxf).reshape(patch + 1, patch)[1:-1].ravel()
    fy = nxf + np.arange(patch * (patch + 1)).reshape(patch, patch + 1)[:, 1:-1].ravel()
    groups = {"x": [], "y": []}
    for s in shifts:
        c = Condensed(pkg, m=10 ** 9, shift=s).prepare()
        X = c.lu.solve(np.stack([c.rhs(pb) for pb in pbs], 1))
        for o, faces in (("x", fx), ("y", fy)):
            Xo = np.concatenate([X[c.face_off[f]:c.face_off[f + 1]] for f in faces], 1)
            groups[o].append(Xo / np.linalg.norm(Xo) / np.sqrt(len(shifts)))
    if global_maps:
        from .hierarchy import Hierarchy
        from .package import hotspot_power
        G = global_size
        gp = Package(stack, np.zeros((G, G), int), [BASE], np.zeros((G * stack.n,) * 2))
        h = Hierarchy(gp, stack.n * stack.nz).prepare()
        bm = block_model(stack, BASE)
        tx, ty = [], []
        for i in range(global_maps):
            rng = np.random.default_rng(1000 + i)
            pw = hotspot_power(stack, (G, G), 1.0, seed=1000 + i, hotspots=int(rng.integers(1, 5)),
                               width=float(rng.uniform(0.03, 0.3)), background=float(rng.uniform(0.0, 0.6)))
            tb = h.solve(pw)["tb"]
            tx.append(tb[1:, :, bm.ports["x-"]].reshape(-1, bm.ports["x-"].stop - bm.ports["x-"].start))
            ty.append(tb[:, 1:, bm.ports["y-"]].reshape(-1, bm.ports["y-"].stop - bm.ports["y-"].start))
        for o, t in (("x", tx), ("y", ty)):
            Xo = np.concatenate(t).T
            groups[o].append(Xo / np.linalg.norm(Xo))
    out = {o: np.concatenate(v, 1) for o, v in groups.items()}
    _SNAPSHOTS[key] = out
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return out


def _disk_cache(key):
    """File of a snapshot set: ``$PINNEAPPLE_CONDENSE_CACHE`` (default ``~/.cache/pinneapple_condense``; empty
    disables it), named by a hash of the stack and the snapshot parameters."""
    import hashlib
    import os
    from pathlib import Path
    root = os.environ.get("PINNEAPPLE_CONDENSE_CACHE", str(Path.home() / ".cache" / "pinneapple_condense"))
    if not root:
        return None
    h = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
    return Path(root) / f"traces_{h}.npz"


def _left_modes(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left singular vectors and values of a wide matrix, from the eigenvectors of ``X X^T`` (rows << columns)."""
    lam, U = np.linalg.eigh(X @ X.T)
    order = np.argsort(-lam)
    return U[:, order], np.sqrt(np.maximum(lam[order], 0.0))


def layered_modes(stack: Stack, orient: str, m: int, shifts: tuple[float, ...] = (0.0,), *,
                  groups: str = "layer") -> np.ndarray:
    """``m`` face modes that never tie two layers together: each layer gets its own constant, and the remaining
    ``m - n_layers`` modes are the leading per-layer POD modes of the traces (:func:`trace_snapshots`), allotted
    greedily by singular value across layers. A mode spanning several layers would fix the ratio of, say, lid and die
    temperatures at every face, a thermal short across the low-conductivity TIM and underfill that grows with the heat
    flux crossing them (measured, see the report). Columns are orthonormal in the area-weighted inner product."""
    n, nz = stack.n, stack.nz
    nf = n * nz
    rows = np.arange(nf).reshape(n, nz)                      # node (lateral, k)
    slices = list(stack.layer_slices().values()) if groups == "layer" else [slice(k, k + 1) for k in range(nz)]
    L = len(slices)
    if m >= nf:
        return np.eye(nf)
    if m < L:
        raise ValueError(f"layered modes need m >= {L} (one constant per layer)")
    w = face_weights(stack)
    s = np.sqrt(w)
    X = trace_snapshots(stack, shifts=shifts)[orient] * s[:, None]
    cand, base = [], []
    for sl in slices:
        r = rows[:, sl].ravel()
        c = np.zeros(nf)
        c[r] = s[r]
        c /= np.linalg.norm(c)
        base.append(c)
        Xr = X[r] - np.outer(c[r], c[r] @ X[r])
        U, sv = _left_modes(Xr)
        for j in range(min(len(sv), len(r) - 1)):
            v = np.zeros(nf)
            v[r] = U[:, j]
            cand.append((sv[j], v))
    cand.sort(key=lambda t: -t[0])
    cols = base + [v for _, v in cand[: m - L]]
    return np.column_stack(cols) / s[:, None]


def interface_modes(bm: BlockModel, orient: str, m: int, method: str = "pod",
                    shifts: tuple[float, ...] = (0.0,)) -> np.ndarray:
    """``m`` modes of a shared face (``orient`` ``"x"`` or ``"y"``), columns orthonormal in the area-weighted inner
    product, the first one constant. ``method="pod"`` (default): the rest are the leading POD modes of the traces
    of :func:`trace_snapshots` (load-informed Ritz-type modes); ``"cc"``: the softest eigenvectors of the two-sided
    face Schur complement (characteristic constraint modes; measured to localise in the low-conductivity layers
    and to miss the vertical temperature profile, see the report). ``shifts``: POD over steady and Laplace-shifted
    traces (:func:`trace_snapshots`; for transient models). ``m >= n_face`` returns the identity."""
    lo, hi = bm.ports[orient + "-"], bm.ports[orient + "+"]
    nf = lo.stop - lo.start
    if m >= nf:
        return np.eye(nf)
    w = face_weights(bm.stack)
    s = np.sqrt(w)
    c = s / np.linalg.norm(s)
    P = np.eye(nf) - np.outer(c, c)
    if method == "cc":
        S2 = bm.S[hi, hi] + bm.S[lo, lo]
        lam, V = np.linalg.eigh(P @ (S2 / s[:, None] / s[None, :]) @ P)
        V = V[:, np.abs(V.T @ c) < 0.5][:, : m - 1]
    elif method == "pod":
        V = _left_modes(P @ (trace_snapshots(bm.stack, shifts=shifts)[orient] * s[:, None]))[0][:, : m - 1]
    elif method in ("layered", "rows"):
        return layered_modes(bm.stack, orient, m, shifts, groups="layer" if method == "layered" else "row")
    else:
        raise ValueError(f"unknown interface-mode method {method!r}")
    return np.column_stack([c, V]) / s[:, None]


@dataclass
class CondensedState:
    """A solved condensed configuration: the interface dofs and the block powers it was solved for."""
    t: np.ndarray
    pb: np.ndarray
    seconds: float


class Condensed:
    """The condensed package: block operators reused across identical blocks, ``m`` interface modes per face
    (``refined_x`` / ``refined_y``: faces kept exact; ``face_m``: modes of every face, x faces first), a global
    sparse interface system factorised once.

    ``prepare()`` is the cost to build (once per geometry, materials and BCs); ``solve(power)`` is the cost per new
    power map (one right-hand side and two triangular solves)."""

    def __init__(self, pkg: Package, m: int = 6, *, refined_x: np.ndarray | None = None,
                 refined_y: np.ndarray | None = None, modes: str = "pod", shift: float = 0.0,
                 face_m: np.ndarray | None = None):
        self.pkg, self.m, self.modes, self.shift = pkg, m, modes, shift
        nbx, nby = pkg.shape
        self.refined_x = np.zeros((nbx + 1, nby), bool) if refined_x is None else np.asarray(refined_x, bool)
        self.refined_y = np.zeros((nbx, nby + 1), bool) if refined_y is None else np.asarray(refined_y, bool)
        self._face_m = None if face_m is None else np.asarray(face_m, int)
        self.timings: dict[str, float] = {}
        self._prepared = False

    # -- topology ------------------------------------------------------------------------------------------------
    def _face_ids(self):
        nbx, nby = self.pkg.shape
        bi, bj = np.meshgrid(np.arange(nbx), np.arange(nby), indexing="ij")
        nxf = (nbx + 1) * nby
        return np.stack([bi * nby + bj, (bi + 1) * nby + bj, nxf + bi * (nby + 1) + bj, nxf + bi * (nby + 1) + bj + 1], -1)

    def _basis(self, orient: str, m: int) -> np.ndarray:
        key = (orient, int(m))
        if key not in self._bases:
            self._bases[key] = interface_modes(self._base, orient, int(m), self.modes)
        return self._bases[key]

    def prepare(self) -> Condensed:
        t0 = time.perf_counter()
        pkg, st = self.pkg, self.pkg.stack
        self.models = [block_model(st, v, shift=self.shift) for v in pkg.variants]
        self.timings["blocks"] = time.perf_counter() - t0
        self.timings["distinct_blocks"] = len(self.models)
        base = block_model(st, BASE, shift=self.shift)
        self._base, self._bases = base, {}
        t1 = time.perf_counter()
        self.basis = {o: self._basis(o, self.m) for o in ("x", "y")}
        nf = base.ports["x-"].stop - base.ports["x-"].start
        self.n_face = nf
        nxf = self.refined_x.size
        if self._face_m is None:
            flags = np.concatenate([self.refined_x.ravel(), self.refined_y.ravel()])
            fm = np.where(flags, nf, self.m)
        else:
            fm = np.minimum(self._face_m, nf)
        self.face_m = fm
        self.face_flags = fm >= nf
        sizes = np.array([self._basis("x" if f < nxf else "y", k).shape[1] for f, k in enumerate(fm)]) \
            if (fm != self.m).any() else np.full(len(fm), self.basis["x"].shape[1])
        self.face_off = np.concatenate([[0], np.cumsum(sizes)])
        self.n_dofs = int(self.face_off[-1])
        self.block_faces = self._face_ids()                        # (nbx, nby, 4)
        bm_ = fm[self.block_faces]                                  # (nbx, nby, 4) modes of each block face
        rows_key = np.concatenate([pkg.layout[..., None], bm_], -1).reshape(-1, 5)
        uniq, inv = np.unique(rows_key, axis=0, return_inverse=True)
        self.group_key = inv.reshape(pkg.shape)
        self.groups: dict[int, dict] = {}
        for key, (v, *fms) in enumerate(uniq):
            bm = self.models[v]
            Phis = [self._basis("x" if i < 2 else "y", fms[i]) for i in range(4)]
            Phi = sla.block_diag(*Phis)
            sel = np.argwhere(self.group_key == key)
            fo = self.face_off[self.block_faces[sel[:, 0], sel[:, 1]]]          # (nb, 4)
            local = np.concatenate([np.arange(Ph.shape[1]) for Ph in Phis])
            which = np.repeat(np.arange(4), [Ph.shape[1] for Ph in Phis])
            D = fo[:, which] + local[None, :]
            self.groups[int(key)] = {"model": bm, "Phi": Phi, "blocks": sel, "dofs": D,
                                     "Sr": Phi.T @ bm.S @ Phi, "Wr": Phi.T @ bm.W, "Otr": bm.O_t @ Phi}
        self.timings["reduce"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        rows, cols, vals = [], [], []
        for g in self.groups.values():
            D = g["dofs"]
            d = D.shape[1]
            rows.append(np.repeat(D, d, axis=1).ravel())
            cols.append(np.tile(D, (1, d)).ravel())
            vals.append(np.broadcast_to(g["Sr"].ravel(), (len(D), d * d)).ravel())
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_dofs, self.n_dofs))
        self.A = A
        self.timings["assemble"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        self.lu = fv.factor(A)
        self.timings["factor"] = time.perf_counter() - t3
        self.timings["prepare"] = time.perf_counter() - t0
        self._prepared = True
        return self

    # -- solve -----------------------------------------------------------------------------------------------------
    def rhs(self, pb: np.ndarray) -> np.ndarray:
        g = np.zeros(self.n_dofs)
        for gr in self.groups.values():
            P = pb[gr["blocks"][:, 0], gr["blocks"][:, 1]]
            np.add.at(g, gr["dofs"].ravel(), (P @ gr["Wr"].T).ravel())
        return g

    def solve(self, power: np.ndarray | None = None) -> CondensedState:
        if not self._prepared:
            self.prepare()
        t0 = time.perf_counter()
        pb = self.pkg.block_power(power)
        t = self.lu.solve(self.rhs(pb))
        return CondensedState(t, pb, time.perf_counter() - t0)

    # -- outputs ---------------------------------------------------------------------------------------------------
    def observables(self, state: CondensedState) -> dict[str, np.ndarray]:
        """``Tj`` (nbx, nby, n*n) and each flux of :data:`FLUXES` per block (nbx, nby)."""
        n2 = self.pkg.stack.n ** 2
        nbx, nby = self.pkg.shape
        out = np.empty((nbx, nby, n2 + len(FLUXES)))
        for gr in self.groups.values():
            b = gr["blocks"]
            vals = state.t[gr["dofs"]] @ gr["Otr"].T + state.pb[b[:, 0], b[:, 1]] @ gr["model"].O_p.T
            out[b[:, 0], b[:, 1]] = vals
        res = {"Tj": out[..., :n2]}
        for i, name in enumerate(FLUXES):
            res[name] = out[..., n2 + i]
        return res

    def _block_group(self, bi: int, bj: int):
        gr = self.groups[int(self.group_key[bi, bj])]
        row = np.nonzero((gr["blocks"][:, 0] == bi) & (gr["blocks"][:, 1] == bj))[0][0]
        return gr, row

    def port_values(self, state: CondensedState, bi: int, bj: int) -> np.ndarray:
        """Fine port temperatures of block ``(I, J)`` (the interface modes expanded)."""
        gr, row = self._block_group(bi, bj)
        return gr["Phi"] @ state.t[gr["dofs"][row]]

    def recover(self, state: CondensedState, bi: int, bj: int) -> np.ndarray:
        """Cell temperatures of block ``(I, J)``, ``(n, n, nz)``, on demand."""
        gr, _ = self._block_group(bi, bj)
        return gr["model"].recover(self.port_values(state, bi, bj), state.pb[bi, bj])

    def field(self, state: CondensedState) -> np.ndarray:
        """Every cell (for checks on small packages): ``(nbx * n, nby * n, nz)``."""
        st = self.pkg.stack
        nbx, nby = self.pkg.shape
        T = np.empty((nbx, nby, st.n, st.n, st.nz))
        for gr in self.groups.values():
            bm = gr["model"]
            b = gr["blocks"]
            tp = state.t[gr["dofs"]] @ gr["Phi"].T
            T[b[:, 0], b[:, 1]] = (state.pb[b[:, 0], b[:, 1]] @ bm.Z.T - tp @ bm.Y.T).reshape(len(b), st.n, st.n,
                                                                                              st.nz)
        return T.transpose(0, 2, 1, 3, 4).reshape(nbx * st.n, nby * st.n, st.nz)

    # -- error estimate and refinement ----------------------------------------------------------------------------
    def residual(self, state: CondensedState) -> np.ndarray:
        """Fine residual (W) on every port node, ``(n_faces, n_face)``: the heat-flow mismatch of each face."""
        nfaces = len(self.face_off) - 1
        r = np.zeros((nfaces, self.n_face))
        for gr in self.groups.values():
            bm = gr["model"]
            b = gr["blocks"]
            tp = state.t[gr["dofs"]] @ gr["Phi"].T
            rb = tp @ bm.S.T - state.pb[b[:, 0], b[:, 1]] @ bm.W.T          # (nb, n_ports)
            faces = self.block_faces[b[:, 0], b[:, 1]]
            for i, name in enumerate(("x-", "x+", "y-", "y+")):
                np.add.at(r, faces[:, i], rb[:, bm.ports[name]])
        return r

    def _sides(self) -> np.ndarray:
        nbx, nby = self.pkg.shape
        nxf = (nbx + 1) * nby
        sides = np.full(len(self.face_off) - 1, 2)
        fx = np.arange(nxf).reshape(nbx + 1, nby)
        fy = nxf + np.arange(nbx * (nby + 1)).reshape(nbx, nby + 1)
        sides[np.r_[fx[0], fx[-1], fy[:, 0], fy[:, -1]]] = 1
        return sides

    def _face_solve(self, r: np.ndarray) -> np.ndarray:
        """``S_FF^-1 r_F`` on every face (two-sided face Schur complement of the base block; one side on the
        package edge): the local solve of the estimate and of the preconditioner of :meth:`correct`."""
        if not hasattr(self, "_face_inv"):
            base = block_model(self.pkg.stack, BASE)
            self._face_inv = {}
            for o in ("x", "y"):
                lo, hi = base.ports[o + "-"], base.ports[o + "+"]
                self._face_inv[o, 2] = sla.cho_factor(base.S[lo, lo] + base.S[hi, hi])
                self._face_inv[o, 1] = sla.cho_factor(base.S[lo, lo])
        nxf = self.refined_x.size
        sides = self._sides()
        z = np.empty_like(r)
        for o, sl in (("x", slice(0, nxf)), ("y", slice(nxf, len(r)))):
            for sd in (1, 2):
                sel = np.nonzero(sides[sl] == sd)[0] + sl.start
                if len(sel):
                    z[sel] = sla.cho_solve(self._face_inv[o, sd], r[sel].T).T
        return z

    def estimate(self, state: CondensedState) -> np.ndarray:
        """Per-face error indicator ``eta_F`` (sqrt(W K)); ``sqrt(sum eta^2)`` estimates the energy-norm error."""
        r = self.residual(state)
        return np.sqrt(np.maximum(0.0, np.einsum("ij,ij->i", r, self._face_solve(r))))

    # -- exact correction -----------------------------------------------------------------------------------------
    def _face_bases(self) -> list[np.ndarray]:
        nxf = self.refined_x.size
        return [self._basis("x" if f < nxf else "y", k) for f, k in enumerate(self.face_m)]

    def expand(self, t: np.ndarray) -> np.ndarray:
        """Reduced interface dofs -> fine port values ``(n_faces, n_face)``."""
        B = self._face_bases()
        return np.stack([B[f] @ t[self.face_off[f]:self.face_off[f + 1]] for f in range(len(B))])

    def _restrict(self, R: np.ndarray) -> np.ndarray:
        B = self._face_bases()
        return np.concatenate([B[f].T @ R[f] for f in range(len(B))])

    def _apply_exact(self, U: np.ndarray) -> np.ndarray:
        """The exact (unreduced) interface operator on fine port values: ``sum_b S_b U_b``."""
        out = np.zeros_like(U)
        nf = self.n_face
        for v, bm in enumerate(self.models):
            sel = np.argwhere(self.pkg.layout == v)
            faces = self.block_faces[sel[:, 0], sel[:, 1]]                       # (nb, 4)
            Y = U[faces].reshape(len(sel), 4 * nf) @ bm.S.T
            np.add.at(out, faces.ravel(), Y.reshape(-1, nf))
        return out

    def _exact_rhs(self, pb: np.ndarray) -> np.ndarray:
        out = np.zeros((len(self.face_off) - 1, self.n_face))
        for v, bm in enumerate(self.models):
            sel = np.argwhere(self.pkg.layout == v)
            faces = self.block_faces[sel[:, 0], sel[:, 1]]
            np.add.at(out, faces.ravel(), (pb[sel[:, 0], sel[:, 1]] @ bm.W.T).reshape(-1, self.n_face))
        return out

    def correct(self, state: CondensedState, rtol: float = 1e-8, maxiter: int = 300) -> dict:
        """Solve the *exact* interface problem by conjugate gradients, preconditioned by this reduced model (coarse
        space: the interface modes, solved with the factorised reduced matrix) plus face-block Jacobi, starting
        from the reduced solution. The answer converges to the fine solution whatever ``m``; the reduced model sets
        how many iterations it takes. Returns the fine port values ``U``, the relative residual per iteration and
        the time."""
        t0 = time.perf_counter()
        g = self._exact_rhs(state.pb)
        U = self.expand(state.t)
        r = g - self._apply_exact(U)

        def prec(R):
            return self.expand(self.lu.solve(self._restrict(R))) + self._face_solve(R)
        z = prec(r)
        d = z.copy()
        rz = float((r * z).sum())
        gn = float(np.linalg.norm(g))
        hist = [float(np.linalg.norm(r)) / gn]
        for _ in range(maxiter):
            if hist[-1] <= rtol:
                break
            Ad = self._apply_exact(d)
            alpha = rz / float((d * Ad).sum())
            U += alpha * d
            r -= alpha * Ad
            hist.append(float(np.linalg.norm(r)) / gn)
            z = prec(r)
            rz_new = float((r * z).sum())
            d = z + (rz_new / rz) * d
            rz = rz_new
        return {"U": U, "history": hist, "iterations": len(hist) - 1, "seconds": time.perf_counter() - t0,
                "pb": state.pb}

    def observables_fine(self, U: np.ndarray, pb: np.ndarray) -> dict[str, np.ndarray]:
        """Observables from fine port values (after :meth:`correct`)."""
        n2 = self.pkg.stack.n ** 2
        nbx, nby = self.pkg.shape
        out = np.empty((nbx, nby, n2 + len(FLUXES)))
        for v, bm in enumerate(self.models):
            sel = np.argwhere(self.pkg.layout == v)
            faces = self.block_faces[sel[:, 0], sel[:, 1]]
            Ub = U[faces].reshape(len(sel), -1)
            out[sel[:, 0], sel[:, 1]] = Ub @ bm.O_t.T + pb[sel[:, 0], sel[:, 1]] @ bm.O_p.T
        res = {"Tj": out[..., :n2]}
        for i, name in enumerate(FLUXES):
            res[name] = out[..., n2 + i]
        return res

    def refine(self, eta: np.ndarray, fraction: float = 0.5, mode: str = "p") -> Condensed:
        """A new model refined on the faces holding ``fraction`` of ``sum eta^2`` (Doerfler marking): ``mode="p"``
        doubles their interface modes, ``"full"`` restores their full port field (no reduction)."""
        order = np.argsort(-eta ** 2)
        cum = np.cumsum(eta[order] ** 2)
        k = int(np.searchsorted(cum, fraction * cum[-1]) + 1)
        fm = self.face_m.copy()
        fm[order[:k]] = self.n_face if mode == "full" else np.minimum(2 * fm[order[:k]], self.n_face)
        return Condensed(self.pkg, self.m, modes=self.modes, shift=self.shift, face_m=fm)


def reference_observables(pkg: Package, T: np.ndarray) -> dict[str, np.ndarray]:
    """The observables of :meth:`Condensed.observables` from a full cell field ``(nx, ny, nz)`` (same rows)."""
    st = pkg.stack
    nbx, nby = pkg.shape
    n, nz = st.n, st.nz
    Tb = T.reshape(nbx, n, nby, n, nz).transpose(0, 2, 1, 3, 4).reshape(nbx, nby, -1)
    out = np.empty((nbx, nby, n * n + len(FLUXES)))
    for t, v in enumerate(pkg.variants):
        bm = block_model(st, v)
        sel = pkg.layout == t
        out[sel] = Tb[sel] @ bm.Om.T
    # the Robin rows carry no source term; the junction and flux rows are linear in cell temperatures only
    res = {"Tj": out[..., :n * n]}
    for i, name in enumerate(FLUXES):
        res[name] = out[..., n * n + i]
    return res


def energy_norm(K: sp.spmatrix, e: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, e @ (K @ e))))


def errors(obs: dict[str, np.ndarray], ref: dict[str, np.ndarray], total_power: float) -> dict[str, float]:
    """Error of observables against a reference: Tmax and junction temperatures (K), interface heats relative to
    the reference total of that interface, and the energy balance (outflow - power, relative)."""
    out = {"Tmax_ref": float(ref["Tj"].max()), "Tmax": float(obs["Tj"].max())}
    out["err_Tmax"] = abs(out["Tmax"] - out["Tmax_ref"])
    out["err_Tj_max"] = float(np.abs(obs["Tj"] - ref["Tj"]).max())
    out["err_Tj_rel_l2"] = float(np.linalg.norm(obs["Tj"] - ref["Tj"]) / np.linalg.norm(ref["Tj"]))
    for name in FLUXES:
        tot = np.abs(ref[name]).sum()
        out[f"err_{name}"] = float(np.abs(obs[name] - ref[name]).sum() / max(tot, 1e-300))
    out["energy_balance"] = float((obs["q_top"].sum() + obs["q_bottom"].sum() - total_power) / total_power)
    return out


def tmax_cells(T: np.ndarray, active_k: int) -> tuple[float, float]:
    """(max over every cell, max over the active row): equal when all heat is generated on the active row with
    non-negative power (discrete maximum principle: every other cell is a weighted mean of its neighbours)."""
    return float(T.max()), float(T[:, :, active_k].max())


def iterate_refinement(pkg: Package, m: int, tol: float, *, fraction: float = 0.5, max_steps: int = 6,
                       power: np.ndarray | None = None) -> Iterable[tuple[Condensed, CondensedState, np.ndarray]]:
    """Solve, estimate, refine until ``sqrt(sum eta^2) <= tol * sqrt(t^T A t)`` (relative energy estimate)."""
    model = Condensed(pkg, m).prepare()
    for _ in range(max_steps):
        state = model.solve(power)
        eta = model.estimate(state)
        yield model, state, eta
        energy = np.sqrt(max(state.t @ (model.A @ state.t), 1e-300))
        if np.sqrt((eta ** 2).sum()) <= tol * energy or model.face_flags.all():
            return
        model = model.refine(eta, fraction).prepare()
