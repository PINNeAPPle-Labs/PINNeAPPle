"""Meshless solution of the chip-package benchmark (track 1b): points straight from the implicit geometry, no volume
mesh.

The package of :mod:`.package` is described implicitly: layers are z intervals, and each block's inclusion (bump,
TSV) is a square prism given by its centre and side. :func:`point_cloud` places nodes from that description only:

* a jittered lattice inside each material region, spacing ``h`` per layer (at least ``per_layer`` nodes across each
  layer), optionally ``refine`` times finer within ``refine_band`` of the inclusion walls; nodes closer than
  ``gap * h`` to a surface are dropped;
* lattices on every surface: each interface (layer planes, inclusion walls) gets a *pair* of nodes, one per side,
  and each outer boundary (top, bottom, sides) one node; surface nodes at edges are skipped.

Equations (``method``: stencils from :mod:`pinneapple_simulation.meshless.stencils`, ``rbf_fd`` or ``gfdm``):

* inside a region, ``k lap T + q = 0`` with a stencil of the nearest nodes *of the same region only*;
* top and bottom, Robin ``-k dT/dn = h T``; sides, adiabatic ``dT/dn = 0``;
* each interface pair ``(a, b)`` (normal ``n`` from ``a`` to ``b``), two equations from one-sided stencils:
  flux continuity ``k_a dT_a/dn = k_b dT_b/dn`` and the contact law ``T_a - T_b = R (-k_a dT_a/dn)``.

``interface="naive"`` drops the pairs: one node per interface point, stencils that cross materials, and the
equation of each node's own material -- the usual collocation shortcut, kept to measure how it fails.

The heat source is the power map of the finite-volume reference spread over the same active row (bottom
``1/nz`` of the die), so both models discretise the same continuous problem.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .package import MATERIALS, Package

AXES = "xyz"


@dataclass
class Cloud:
    X: np.ndarray                    # (N, 3) node coordinates (m)
    region: np.ndarray               # (N,) region id
    kind: np.ndarray                 # (N,) 0 interior, 1 robin top, 2 robin bottom, 3 adiabatic side, 4 interface
    normal: np.ndarray               # (N, 3) outward normal (boundary) or a->b normal (interface)
    partner: np.ndarray              # (N,) the other node of an interface pair, -1 otherwise
    area: np.ndarray                 # (N,) surface weight of a surface node (m^2), 0 inside
    k: np.ndarray                    # (N,) conductivity of the node's material
    R: np.ndarray                    # (N,) contact resistance of the node's interface (m^2 K / W)
    seconds: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.X)


class Geometry:
    """Implicit description of a package: what material is at any point, and its surfaces."""

    def __init__(self, pkg: Package):
        self.pkg = pkg
        st = pkg.stack
        self.st = st
        self.p = st.pitch
        self.W = (pkg.shape[0] * st.pitch, pkg.shape[1] * st.pitch)
        self.z = np.concatenate([[0.0], np.cumsum([L.thickness for L in st.layers])])
        self.layers = st.layers
        self.props = [st.properties(v) for v in pkg.variants]
        contact = {(a, b): r for a, b, r in st.contact}
        self.Rz = np.zeros(len(self.z))                 # contact resistance on the plane z[l] (between l-1 and l)
        for l in range(1, len(self.layers)):
            self.Rz[l] = contact.get((self.layers[l - 1].name, self.layers[l].name), 0.0)
        self.variant_scale = []                          # per variant: per layer inclusion k factor, present?
        for v in pkg.variants:
            f = dict(v.inclusion_k)
            self.variant_scale.append({L.name: (f.get(L.name, 1.0), L.inclusion is not None and L.name not in
                                                v.removed) for L in self.layers})
        cscale = [{(a, b): fct for a, b, fct in v.contact} for v in pkg.variants]
        self.contact_scale = cscale

    def layer(self, z):
        return np.clip(np.searchsorted(self.z, z, side="right") - 1, 0, len(self.layers) - 1)

    def block(self, x, y):
        nbx, nby = self.pkg.shape
        return (np.clip((x / self.p).astype(int), 0, nbx - 1), np.clip((y / self.p).astype(int), 0, nby - 1))

    def in_inclusion(self, X, lay=None):
        lay = self.layer(X[:, 2]) if lay is None else lay
        I, J = self.block(X[:, 0], X[:, 1])
        var = self.pkg.layout[I, J]
        lx = X[:, 0] - (I + 0.5) * self.p
        ly = X[:, 1] - (J + 0.5) * self.p
        out = np.zeros(len(X), bool)
        for l, L in enumerate(self.layers):
            if not L.inclusion:
                continue
            sel = lay == l
            if not sel.any():
                continue
            present = np.array([self.variant_scale[v][L.name][1] for v in range(len(self.pkg.variants))])[var[sel]]
            half = L.side * self.p / 2
            out[sel] = present & (np.abs(lx[sel]) < half) & (np.abs(ly[sel]) < half)
        return out

    def region(self, X):
        """Region id: layer index for the matrix, or ``n_layers + (l * nbx + I) * nby + J`` for an inclusion."""
        lay = self.layer(X[:, 2])
        inc = self.in_inclusion(X, lay)
        I, J = self.block(X[:, 0], X[:, 1])
        nbx, nby = self.pkg.shape
        return np.where(inc, len(self.layers) + (lay * nbx + I) * nby + J, lay)

    def conductivity(self, X, region=None):
        lay = self.layer(X[:, 2])
        inc = self.in_inclusion(X, lay) if region is None else region >= len(self.layers)
        I, J = self.block(X[:, 0], X[:, 1])
        var = self.pkg.layout[I, J]
        k = np.empty(len(X))
        for l, L in enumerate(self.layers):
            sel = lay == l
            k[sel] = MATERIALS[L.matrix][0]
            if L.inclusion:
                s = sel & inc
                fac = np.array([self.variant_scale[v][L.name][0] for v in range(len(self.pkg.variants))])
                k[s] = MATERIALS[L.inclusion][0] * fac[var[s]]
        return k

    def source(self, X):
        """Volumetric heat (W/m^3) of the finite-volume power map on the active row of the die."""
        st = self.st
        ka = st.active_k
        dz = st.dz()
        z0 = dz[:ka].sum()
        z1 = z0 + dz[ka]
        dx = st.dx
        nx, ny = self.pkg.shape[0] * st.n, self.pkg.shape[1] * st.n
        i = np.clip((X[:, 0] / dx).astype(int), 0, nx - 1)
        j = np.clip((X[:, 1] / dx).astype(int), 0, ny - 1)
        inside = (X[:, 2] >= z0) & (X[:, 2] < z1)
        return np.where(inside, self.pkg.power[i, j] / (dx * dx * dz[ka]), 0.0)


def _lattice2(a0, a1, b0, b1, h, rng, jitter=0.0, margin=0.5):
    """Cell-centred lattice on the rectangle, about ``h`` apart, ``margin * h`` clear of its edges."""
    na, nb = max(1, int(round((a1 - a0) / h))), max(1, int(round((b1 - b0) / h)))
    ha, hb = (a1 - a0) / na, (b1 - b0) / nb
    A, B = np.meshgrid(a0 + (np.arange(na) + 0.5) * ha, b0 + (np.arange(nb) + 0.5) * hb, indexing="ij")
    A, B = A.ravel(), B.ravel()
    if jitter:
        A = A + jitter * ha * rng.uniform(-0.5, 0.5, A.size)
        B = B + jitter * hb * rng.uniform(-0.5, 0.5, B.size)
    keep = (A - a0 > margin * ha * 0.99) & (a1 - A > margin * ha * 0.99) & (B - b0 > margin * hb * 0.99) & \
        (b1 - B > margin * hb * 0.99)
    return A[keep], B[keep], ha * hb


def point_cloud(pkg: Package, *, h_xy: float | None = None, per_layer: int = 3, h_max: float = 25e-6,
                per_feature: int = 4, band_h: float = 2.0, gap: float = 0.3, jitter: float = 0.3,
                interface: str = "pairs", seed: int = 0, density=None, ghosts: bool = False,
                ghost_h: float = 1.0, sides: str = "mirror") -> Cloud:
    """Nodes from the implicit geometry (see the module docstring); ``h_xy``: spacing cap (default: the pitch /
    6); ``per_layer``: nodes across each layer; ``per_feature``: nodes across each inclusion. ``density``: an
    optional callable ``X -> factor in (0, 1]`` that thins the interior lattice (adaptive refinement from an error
    indicator is done by passing a finer ``h_xy`` and a density that keeps the fine nodes only where needed)."""
    t0 = time.perf_counter()
    g = Geometry(pkg)
    rng = np.random.default_rng(seed)
    W0, W1 = g.W
    h_xy = h_xy or pkg.stack.pitch / 6          # also the cap of every layer's spacing
    pts, kind, nrm, area, Rr, pair_a = [], [], [], [], [], []

    def add(P, kd, n_, ar, R=0.0):
        pts.append(P)
        kind.append(np.full(len(P), kd))
        nrm.append(np.broadcast_to(np.asarray(n_, float), P.shape).copy())
        area.append(np.full(len(P), ar))
        Rr.append(np.full(len(P), R))

    hl = np.array([min(h_max, h_xy, L.thickness / per_layer) for L in g.layers])
    # spacing at small features: at least ``per_feature`` nodes across each inclusion, in a band of ``band_h``
    # spacings around it
    hi = np.array([min(hl[l], L.side * g.p / per_feature) if L.inclusion else hl[l] for l, L in enumerate(g.layers)])
    band = band_h * hi
    # interior lattices, per layer
    st = pkg.stack
    z_src = st.dz()[: st.active_k].sum() + st.dz()[st.active_k]          # top of the heated row
    for l, L in enumerate(g.layers):
        h = hl[l]
        slabs = [(g.z[l], g.z[l + 1])]
        if g.z[l] < z_src < g.z[l + 1]:                                    # rows aligned with the source step
            slabs = [(g.z[l], z_src), (z_src, g.z[l + 1])]
        zs, hzs = [], []
        for za, zb in slabs:
            nz = max(1, int(round((zb - za) / h)))
            zs.append(za + (np.arange(nz) + 0.5) * (zb - za) / nz)
            hzs.append(np.full(nz, (zb - za) / nz))
        zs, hzs = np.concatenate(zs), np.concatenate(hzs)
        nz = len(zs)
        A, B, _ = _lattice2(0, W0, 0, W1, h, rng, jitter, margin=0.25)
        Z = np.repeat(zs, len(A))
        P = np.column_stack([np.tile(A, nz), np.tile(B, nz), Z])
        P[:, 2] += jitter * np.repeat(hzs, len(A)) * rng.uniform(-0.5, 0.5, len(P))
        if L.inclusion and hi[l] < h:
            # small features: spacing hi inside the inclusions and within a band around their walls
            fine = _lattice3(g, l, hi[l], rng, jitter, z_src)
            dwf = _wall_distance(g, fine, np.full(len(fine), l))
            infine = g.in_inclusion(fine, np.full(len(fine), l)) | (dwf < band[l])
            dwc = _wall_distance(g, P, np.full(len(P), l))
            incoarse = g.in_inclusion(P, np.full(len(P), l)) | (dwc < band[l])
            P = np.vstack([P[~incoarse], fine[infine]])
        add(P, 0, 0.0, 0.0)
    X = np.vstack(pts)
    # drop interior nodes too close to a surface
    lay = g.layer(X[:, 2])
    hloc = hl[lay]
    dz = np.minimum(X[:, 2] - g.z[lay], g.z[lay + 1] - X[:, 2])
    dxy = np.minimum.reduce([X[:, 0], W0 - X[:, 0], X[:, 1], W1 - X[:, 1]])
    dw = _wall_distance(g, X, lay)
    hloc_eff = np.where((dw < band[lay]) | g.in_inclusion(X, lay), hi[lay], hloc)
    keep = (dz > gap * hloc_eff) & (dxy > gap * hloc_eff) & (dw > gap * hloc_eff)
    X = X[keep]
    pts = [X]
    kind = [np.zeros(len(X), int)]
    nrm = [np.zeros((len(X), 3))]
    area = [np.zeros(len(X))]
    Rr = [np.zeros(len(X))]
    # outer boundaries: top and bottom (Robin), sides (adiabatic)
    for zb, kd, n_, l in ((g.z[-1], 1, (0, 0, 1), len(g.layers) - 1), (0.0, 2, (0, 0, -1), 0)):
        A, B, ar = _lattice2(0, W0, 0, W1, hl[l], rng, 0.0, margin=0.5)
        add(np.column_stack([A, B, np.full(len(A), zb)]), kd, n_, ar)
    for l, L in enumerate(g.layers if sides == "nodes" else ()):
        for axis, val, n_ in ((0, 0.0, (-1, 0, 0)), (0, W0, (1, 0, 0)), (1, 0.0, (0, -1, 0)), (1, W1, (0, 1, 0))):
            other = W1 if axis == 0 else W0
            A, B, ar = _lattice2(0, other, g.z[l], g.z[l + 1], hl[l], rng, 0.0, margin=0.5)
            P = np.zeros((len(A), 3))
            P[:, axis] = val
            P[:, 1 - axis] = A
            P[:, 2] = B
            add(P, 3, n_, ar)
    # interfaces: layer planes and inclusion walls, as node pairs (a below / inside, b above / outside)
    pairs = []
    for l in range(1, len(g.layers)):
        # the plane's spacing follows the layers on both sides: fine over the inclusion footprints (and their band)
        # of either layer, coarse elsewhere -- a plane finer than its neighbourhood gives coplanar stencils
        h = min(hl[l - 1], hl[l])
        hf = min(hi[l - 1], hi[l])
        A, B, ar = _lattice2(0, W0, 0, W1, h, rng, 0.0, margin=0.5)
        P = np.column_stack([A, B, np.full(len(A), g.z[l])])

        def near_feature(Q, lim):
            out = np.zeros(len(Q), bool)
            for ll in (l - 1, l):
                if hi[ll] < hl[ll]:
                    lay_ = np.full(len(Q), ll)
                    out |= g.in_inclusion(Q, lay_) | (_wall_distance(g, Q, lay_) < lim[ll])
            return out
        if hf < h:
            P = P[~near_feature(P, band)]
            A2, B2, ar2 = _lattice2(0, W0, 0, W1, hf, rng, 0.0, margin=0.5)
            P2 = np.column_stack([A2, B2, np.full(len(A2), g.z[l])])
            P2 = P2[near_feature(P2, band)]
        else:
            P2, ar2 = np.zeros((0, 3)), 0.0
        # no plane node on (or next to) the edge where an inclusion wall meets the plane
        for Q, hh, a_ in ((P, h, ar), (P2, hf, ar2)):
            dw = np.minimum(_wall_distance(g, Q, np.full(len(Q), l - 1)), _wall_distance(g, Q, np.full(len(Q), l)))
            Q = Q[dw > gap * hh]
            if len(Q):
                pairs.append((Q, (0, 0, 1), a_, g.Rz[l], -1e-12, 1e-12))
    for l, L in enumerate(g.layers):
        if not L.inclusion:
            continue
        half = L.side * g.p / 2
        hw = hi[l]
        for I in range(pkg.shape[0]):
            for J in range(pkg.shape[1]):
                v = pkg.layout[I, J]
                if not g.variant_scale[v][L.name][1]:
                    continue
                cx, cy = (I + 0.5) * g.p, (J + 0.5) * g.p
                for axis, sgn in ((0, -1), (0, 1), (1, -1), (1, 1)):
                    c_other = cy if axis == 0 else cx
                    A, B, ar = _lattice2(c_other - half, c_other + half, g.z[l], g.z[l + 1], hw, rng, 0.0, 0.5)
                    P = np.zeros((len(A), 3))
                    P[:, axis] = (cx if axis == 0 else cy) + sgn * half
                    P[:, 1 - axis] = A
                    P[:, 2] = B
                    n_ = np.zeros(3)
                    n_[axis] = sgn
                    pairs.append((P, n_, ar, 0.0, None, None))
    for P, n_, ar, R, ea, eb in pairs:
        n_ = np.asarray(n_, float)
        eps = 1e-10
        if interface == "pairs":
            na = len(np.concatenate(kind))
            add(P, 4, n_, ar, R)
            add(P.copy(), 4, n_, ar, R)
            pair_a.append((na, len(P), n_, eps))
        else:
            add(P, 0, 0.0, 0.0)
    X = np.vstack(pts)
    kind = np.concatenate(kind)
    nrm = np.vstack(nrm)
    area = np.concatenate(area)
    Rr = np.concatenate(Rr)
    partner = np.full(len(X), -1)
    # regions: side a of a pair is evaluated a hair before the surface along -n, side b along +n
    probe = X.copy()
    for na, m, n_, eps in pair_a:
        a, b = np.arange(na, na + m), np.arange(na + m, na + 2 * m)
        probe[a] -= 1e-9 * n_
        probe[b] += 1e-9 * n_
        partner[a], partner[b] = b, a
    # boundary nodes: probe slightly inside
    bnd = (kind >= 1) & (kind <= 3)
    probe[bnd] -= 1e-9 * nrm[bnd]
    region = g.region(probe)
    k = g.conductivity(probe, region)
    ghost = np.full(len(X), -1)
    if ghosts:
        # one ghost node per surface node, a local spacing outside its own region: the surface node then carries
        # both its boundary or interface condition and the PDE (the ghost's row), which removes the instability of
        # collocating Neumann-type conditions alone (measured: the error grew with refinement without them)
        own = np.nonzero(kind >= 1)[0]
        lay = g.layer(probe[own, 2])
        hloc = np.where(g.in_inclusion(probe[own], lay) | (_wall_distance(g, probe[own], lay) < band[lay]),
                        hi[lay], hl[lay])
        out_dir = nrm[own].copy()
        side_b = (kind[own] == 4) & (partner[own] < own)
        out_dir[side_b] *= -1
        G = X[own] + (ghost_h * hloc)[:, None] * out_dir
        ng = len(own)
        ghost[own] = len(X) + np.arange(ng)
        X = np.vstack([X, G])
        region = np.concatenate([region, region[own]])
        kind = np.concatenate([kind, np.full(ng, 5)])
        nrm = np.vstack([nrm, nrm[own]])
        partner = np.concatenate([partner, np.full(ng, -1)])
        ghost = np.concatenate([ghost, own])                     # a ghost points back to its owner
        area = np.concatenate([area, np.zeros(ng)])
        Rr = np.concatenate([Rr, np.zeros(ng)])
        k = np.concatenate([k, k[own]])
        probe = np.vstack([probe, probe[own]])
    cl = Cloud(X, region, kind, nrm, partner, area, k, Rr, time.perf_counter() - t0,
               {"h_layers": hl.tolist(), "n_interface_pairs": int((kind == 4).sum() // 2),
                "ghosts": bool(ghosts), "sides": sides})
    cl.info["probe"] = probe
    cl.info["ghost"] = ghost
    return cl


def _lattice3(g: Geometry, l: int, h: float, rng, jitter: float, z_src: float) -> np.ndarray:
    """Jittered lattice filling layer ``l`` at spacing ``h`` (rows aligned with the source step)."""
    za, zb = g.z[l], g.z[l + 1]
    slabs = [(za, z_src), (z_src, zb)] if za < z_src < zb else [(za, zb)]
    zs, hz = [], []
    for a, b in slabs:
        nz = max(1, int(round((b - a) / h)))
        zs.append(a + (np.arange(nz) + 0.5) * (b - a) / nz)
        hz.append(np.full(nz, (b - a) / nz))
    zs, hz = np.concatenate(zs), np.concatenate(hz)
    A, B, _ = _lattice2(0, g.W[0], 0, g.W[1], h, rng, jitter, margin=0.25)
    P = np.column_stack([np.tile(A, len(zs)), np.tile(B, len(zs)), np.repeat(zs, len(A))])
    P[:, 2] += jitter * np.repeat(hz, len(A)) * rng.uniform(-0.5, 0.5, len(P))
    return P


def _wall_distance(g: Geometry, X, lay):
    """Distance to the nearest inclusion wall of the node's layer and block (inf where the layer has none)."""
    d = np.full(len(X), np.inf)
    I, J = g.block(X[:, 0], X[:, 1])
    lx = np.abs(X[:, 0] - (I + 0.5) * g.p)
    ly = np.abs(X[:, 1] - (J + 0.5) * g.p)
    for l, L in enumerate(g.layers):
        if not L.inclusion:
            continue
        sel = lay == l
        half = L.side * g.p / 2
        dd = np.minimum(np.where(ly[sel] < half, np.abs(lx[sel] - half), np.hypot(lx[sel] - half, ly[sel] - half)),
                        np.where(lx[sel] < half, np.abs(ly[sel] - half), np.hypot(lx[sel] - half, ly[sel] - half)))
        d[sel] = dd
    return d


def _near_walls(g, P, l, band):
    return _wall_distance(g, P, np.full(len(P), l)) < band


@dataclass
class MeshlessSolution:
    cloud: Cloud
    T: np.ndarray
    A: sp.csr_matrix
    b: np.ndarray
    timings: dict
    stencils: dict


def region_stencils(X: np.ndarray, region: np.ndarray, Q: np.ndarray, q_region: np.ndarray, n: int,
                    box: tuple[float, float] | None, reach: float, normal: np.ndarray | None = None):
    """Stencils of the ``n`` nearest nodes of the same region for every query point. With ``box = (W0, W1)`` the
    package sides are mirror planes (adiabatic): nodes within ``reach`` of a side (or two) are reflected and a
    reflected copy stands for its original node. Returns ``(Xe, idx_e, col)``: the extended point array, the stencil
    indices into it, and the original node of each extended point."""
    ext, col = [X], [np.arange(len(X))]
    if box is not None:
        W = (0.0, box[0]), (0.0, box[1])
        for ax in (0, 1):
            for w in W[ax]:
                near = np.abs(X[:, ax] - w) < reach
                M = X[near].copy()
                M[:, ax] = 2 * w - M[:, ax]
                ext.append(M)
                col.append(np.nonzero(near)[0])
        for wx in W[0]:
            for wy in W[1]:
                near = (np.abs(X[:, 0] - wx) < reach) & (np.abs(X[:, 1] - wy) < reach)
                M = X[near].copy()
                M[:, 0], M[:, 1] = 2 * wx - M[:, 0], 2 * wy - M[:, 1]
                ext.append(M)
                col.append(np.nonzero(near)[0])
    Xe, col = np.vstack(ext), np.concatenate(col)
    re = region[col]
    idx = np.empty((len(Q), n), int)
    for r in np.unique(q_region):
        sel = np.nonzero(re == r)[0]
        qs = np.nonzero(q_region == r)[0]
        if len(sel) < n:
            raise ValueError(f"region {r} has {len(sel)} nodes, fewer than the stencil size {n}")
        tree = cKDTree(Xe[sel])
        if normal is None:
            _, nb = tree.query(Q[qs], k=n)
            idx[qs] = sel[nb]
            continue
        nrm = normal[qs]
        surf = np.abs(nrm).sum(1) > 0
        if (~surf).any():
            _, nb = tree.query(Q[qs[~surf]], k=n)
            idx[qs[~surf]] = sel[nb]
        if surf.any():
            # surface nodes: at most n // 3 stencil nodes in the node's own plane, so the normal direction is
            # sampled (a stencil of coplanar nodes leaves the normal derivative undetermined)
            qq = qs[surf]
            m = min(len(sel), 5 * n)
            dist, nb = tree.query(Q[qq], k=m)
            off = np.abs(((Xe[sel][nb] - Q[qq][:, None, :]) * nrm[surf][:, None, :]).sum(-1))
            coplanar = off < 0.2 * np.maximum(dist[:, 1:2], 1e-30)
            for r_, (c, o) in enumerate(zip(coplanar, nb)):
                keep_c = np.nonzero(c)[0][: n // 3]
                keep_o = np.nonzero(~c)[0][: n - len(keep_c)]
                pick = np.sort(np.concatenate([keep_c, keep_o]))
                if len(pick) < n:
                    pick = np.arange(n)
                idx[qq[r_]] = sel[o[pick]]
    return Xe, idx, col


def assemble(cloud: Cloud, pkg: Package, *, method: str = "rbf_fd", degree: int = 2, n: int | None = None,
             interface: str = "pairs", oversample: bool = False):
    """The collocation system of the module docstring. Returns ``(A, b, stencils)``."""
    from pinneapple_simulation.meshless.stencils import monomials, stencil_weights
    g = Geometry(pkg)
    X = cloud.X
    N = cloud.n
    n = n or int(2.5 * len(monomials(3, degree)))
    st = pkg.stack
    hT, hB = st.h_top, st.h_bottom
    region = cloud.region
    t0 = time.perf_counter()
    box = g.W if cloud.info.get("sides") == "mirror" else None
    reach = 6.0 * max(cloud.info["h_layers"])
    if interface == "pairs":
        Xe, idx_e, col = region_stencils(X, region, X, region, n, box, reach, cloud.normal)
    else:
        Xe, idx_e, col = region_stencils(X, np.zeros(N, int), X, np.zeros(N, int), n, box, reach)
    idx = col[idx_e]                                                    # columns: the original nodes
    t_knn = time.perf_counter() - t0
    W = stencil_weights(Xe, X, idx_e, ("d0", "d1", "d2", "lap"), method=method, degree=degree)
    t_w = time.perf_counter() - t0 - t_knn
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    kind, nrm, k = cloud.kind, cloud.normal, cloud.k

    def put(r_idx, w, c_idx=None):
        c_idx = idx[r_idx] if c_idx is None else c_idx
        rows.append(np.repeat(r_idx, c_idx.shape[1]))
        cols.append(c_idx.ravel())
        vals.append(w.ravel())
    dn = sum(W[f"d{a}"] * nrm[:, a:a + 1] for a in range(3))           # d/dn weights at every node
    inner = np.nonzero(kind == 0)[0]
    put(inner, W["lap"][inner] * k[inner, None])
    b[inner] = -g.source(X[inner])
    for kd, hh in ((1, hT), (2, hB)):
        s = np.nonzero(kind == kd)[0]
        put(s, k[s, None] * dn[s])
        put(s, np.full((len(s), 1), hh), s[:, None])
    s = np.nonzero(kind == 3)[0]
    put(s, dn[s])
    a = np.nonzero((kind == 4) & (cloud.partner > np.arange(N)))[0]      # side a of each pair
    bb = cloud.partner[a]
    # flux continuity on row a: k_a dT_a/dn - k_b dT_b/dn = 0
    put(a, k[a, None] * dn[a])
    put(a, -k[bb, None] * dn[bb], idx[bb])
    # contact law on row b: T_a - T_b + R k_a dT_a/dn = 0
    put(bb, np.ones((len(a), 1)), a[:, None])
    put(bb, -np.ones((len(a), 1)), bb[:, None])
    Rk = cloud.R[a] * k[a]
    put(bb, Rk[:, None] * dn[a], idx[a])
    if oversample:
        # least-squares collocation: the PDE also at extra points (one per interior node, a random offset of up to
        # 0.35 of the local spacing, kept in the node's region and away from surfaces)
        rng = np.random.default_rng(1)
        inner_all = np.nonzero(kind == 0)[0]
        hloc = np.array(cloud.info["h_layers"])[g.layer(X[inner_all, 2])]
        Y = X[inner_all] + 0.35 * hloc[:, None] * rng.uniform(-1, 1, (len(inner_all), 3))
        ok = g.region(Y) == region[inner_all]
        ok &= (Y[:, 0] > 0) & (Y[:, 0] < g.W[0]) & (Y[:, 1] > 0) & (Y[:, 1] < g.W[1])
        Y, ry = Y[ok], region[inner_all][ok]
        box = g.W if cloud.info.get("sides") == "mirror" else None
        Ye, iy, cy = region_stencils(X, region if interface == "pairs" else np.zeros(N, int), Y,
                                     ry if interface == "pairs" else np.zeros(len(Y), int), n, box,
                                     6.0 * max(cloud.info["h_layers"]))
        wy = stencil_weights(Ye, Y, iy, ("lap",), method=method, degree=degree)["lap"]
        ky = g.conductivity(Y)
        extra_rows = N + np.arange(len(Y))
        put(extra_rows, wy * ky[:, None], cy[iy])
        b = np.concatenate([b, -g.source(Y)])
    gh = cloud.info.get("ghost")
    if gh is not None and (kind == 5).any():
        gs = np.nonzero(kind == 5)[0]
        own = gh[gs]
        put(gs, W["lap"][own] * k[own, None], idx[own])                # the PDE at the owner, on the ghost's row
        b[gs] = -g.source(cloud.info["probe"][own])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(b), N))
    return A, b, {"n": n, "degree": degree, "method": method, "knn_s": t_knn, "weights_s": t_w, "idx": idx, "W": W}


def solve(pkg: Package, cloud: Cloud | None = None, *, method: str = "rbf_fd", degree: int = 2, n: int | None = None,
          interface: str = "pairs", solver: str = "auto", condition: bool = False, oversample: bool = False,
          **cloud_kw) -> MeshlessSolution:
    """Point cloud (unless given), assembly, solve. ``solver``: sparse LU, or GMRES with an ILU preconditioner above
    2e5 nodes. ``condition``: also a 1-norm condition-number estimate (an extra factorisation)."""
    timings = {}
    if cloud is None:
        cloud = point_cloud(pkg, interface=interface, **cloud_kw)
    timings["points_s"] = cloud.seconds
    t0 = time.perf_counter()
    A, b, stn = assemble(cloud, pkg, method=method, degree=degree, n=n, interface=interface, oversample=oversample)
    timings["assemble_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    Ac = A.tocsc()
    if A.shape[0] > A.shape[1]:
        # weighted least squares through the augmented (saddle-point) system [[I, A], [A^T, 0]], rows scaled to unit
        # max: the condition number of A, not its square
        d = 1.0 / np.maximum(abs(A).max(1).toarray().ravel(), 1e-300)
        As = sp.diags(d) @ A
        m_, n_ = As.shape
        K = sp.bmat([[sp.eye(m_), As], [As.T, None]]).tocsc()
        lu = spla.splu(K, permc_spec="COLAMD")
        T = lu.solve(np.concatenate([d * b, np.zeros(n_)]))[m_:]
        timings["solve_s"] = time.perf_counter() - t1
        timings["residual"] = float(np.linalg.norm(d * (A @ T - b)) / max(np.linalg.norm(d * b), 1e-300))
        return MeshlessSolution(cloud, T, A, b, timings, stn)
    if solver == "auto":
        solver = "direct" if cloud.n <= 200_000 else "gmres"
    # rows equilibrated to unit max: PDE rows (~k/h^2), boundary rows (~k/h) and contact rows (~1) differ by 12
    # orders of magnitude; unscaled, the factorisation loses the small rows (measured: 3 % error became garbage)
    d = 1.0 / np.maximum(abs(A).max(1).toarray().ravel(), 1e-300)
    Ac = (sp.diags(d) @ A).tocsc()
    bs = d * b
    if solver == "direct":
        lu = spla.splu(Ac, permc_spec="COLAMD")
        T = lu.solve(bs)
    else:
        ilu = spla.spilu(Ac, drop_tol=1e-5, fill_factor=20)
        M = spla.LinearOperator(A.shape, ilu.solve)
        T, info = spla.gmres(Ac, bs, M=M, rtol=1e-10, restart=200, maxiter=50)
        lu = None
    timings["solve_s"] = time.perf_counter() - t1
    timings["residual"] = float(np.abs(Ac @ T - bs).max())
    if condition and lu is not None:
        Ainv = spla.LinearOperator(A.shape, matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans="T"))
        timings["cond1_est"] = float(spla.onenormest(Ac) * spla.onenormest(Ainv))
    return MeshlessSolution(cloud, T, A, b, timings, stn)


def evaluate(sol: MeshlessSolution, pkg: Package, P: np.ndarray, *, degree: int | None = None) -> np.ndarray:
    """Temperature at points ``P`` (interpolated with stencils of the region each point lies in)."""
    from pinneapple_simulation.meshless.stencils import stencil_weights
    g = Geometry(pkg)
    cl = sol.cloud
    deg = degree or sol.stencils["degree"]
    n = sol.stencils["n"]
    reg = g.region(P)
    box = g.W if cl.info.get("sides") == "mirror" else None
    Xe, idx_e, col = region_stencils(cl.X, cl.region, P, reg, n, box, 6.0 * max(cl.info["h_layers"]))
    w = stencil_weights(Xe, P, idx_e, ("eval",), method="rbf_fd", degree=deg)["eval"]
    return (w * sol.T[col[idx_e]]).sum(1)


def observables(sol: MeshlessSolution, pkg: Package) -> dict:
    """Junction temperatures at the centres of the reference's active cells (``Tj`` as in
    :func:`.substructure.reference_observables`), the die/TIM heat, the outflows and the energy balance."""
    st = pkg.stack
    n = st.n
    nbx, nby = pkg.shape
    dz = st.dz()
    ka = st.active_k
    zc = dz[:ka].sum() + 0.5 * dz[ka]
    xs = (np.arange(nbx * n) + 0.5) * st.dx
    ys = (np.arange(nby * n) + 0.5) * st.dx
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, zc)])
    tj = evaluate(sol, pkg, P).reshape(nbx * n, nby * n)
    cl = sol.cloud
    W = sol.stencils["W"]
    idx = sol.stencils["idx"]
    dT = {a: (W[f"d{a}"] * sol.T[idx]).sum(1) for a in range(3)}
    dn = sum(dT[a] * cl.normal[:, a] for a in range(3))
    top = cl.kind == 1
    bot = cl.kind == 2
    q_top = float((st.h_top * sol.T[top] * cl.area[top]).sum())
    q_bot = float((st.h_bottom * sol.T[bot] * cl.area[bot]).sum())
    names = [L.name for L in st.layers]
    q_tim = float("nan")
    if "tim" in names:
        z_tim = Geometry(pkg).z[names.index("tim")]
        a = (cl.kind == 4) & (cl.partner > np.arange(cl.n)) & (np.abs(cl.X[:, 2] - z_tim) < 1e-9) & \
            (np.abs(cl.normal[:, 2]) > 0.5)
        q_tim = float((-cl.k[a] * dn[a] * cl.area[a]).sum())
    P_tot = float(pkg.power.sum())
    return {"Tj": tj, "Tmax": float(tj.max()), "q_tim": q_tim, "q_top": q_top, "q_bottom": q_bot,
            "energy_balance": (q_top + q_bot) / P_tot - 1}
