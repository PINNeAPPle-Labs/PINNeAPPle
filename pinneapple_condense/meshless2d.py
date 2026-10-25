"""2-D meshless (RBF-FD / GFDM) discretisations of the CFD falsification cases (``docs/research/
condensation_falsification.md``, section C), on point clouds built from the implicit geometry of a square-pin array.

* :func:`cloud` -- jittered lattice points of one unit cell (pin and fluid), replicated over the array; coordinates
  are rounded to a dyadic grid so that a block shifted by an integer has bit-identical stencil offsets (identical
  blocks then give identical matrices, which :mod:`.algebraic` detects by content);
* :func:`advection_diffusion` -- ``u . grad T - alpha lap T = q`` collocated with the stencils of
  :mod:`pinneapple_simulation.meshless.stencils`; inlet ``T = 0``, zero normal gradient elsewhere;
* :func:`stokes` -- collocated velocity-pressure Stokes on the fluid points (no-slip on the pin walls, uniform inlet,
  free-slip sides, ``p = 0`` and zero velocity gradient at the outlet);
* :func:`hybrid_conduction` -- multi-material conduction with a contact resistance on the pin walls: finite volumes
  on the cells of a band around every interface, RBF-FD on scattered points elsewhere, coupled by RBF interpolation of
  the FV cells' missing neighbours.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from pinneapple_simulation.meshless.stencils import stencil_weights

_Q = 2.0 ** -20                                     # dyadic coordinate grid


def _round(X):
    return np.round(X / _Q) * _Q


def pin_sdf(X, pins, pin_frac, offset=(1.0, 0.0)):
    """Signed distance to the nearest square pin (negative inside); unit cells of side 1 from ``offset``."""
    x = X[:, 0] - offset[0]
    y = X[:, 1] - offset[1]
    cx = np.clip(np.floor(x), 0, pins[0] - 1) + 0.5
    cy = np.clip(np.floor(y), 0, pins[1] - 1) + 0.5
    a = pin_frac / 2
    dx, dy = np.abs(x - cx) - a, np.abs(y - cy) - a
    out = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)
    inside_array = (x >= 0) & (x <= pins[0])
    return np.where(inside_array, out, np.inf)


def cloud(pins, h: float, pin_frac: float = 0.5, jitter: float = 0.25, inlet_blocks: int = 1, outlet_blocks: int = 1,
          seed: int = 0, fluid_only: bool = False, wall_points: bool = False):
    """Points over ``(inlet + pins_x + outlet) x pins_y`` unit cells: one jittered lattice of spacing ``h`` per unit
    cell, the same in every cell (shifted by integers). ``fluid_only`` drops the points inside (and within 0.3 h of)
    the pins; ``wall_points`` adds points on the pin walls (spacing ``h``). Returns the points, a boolean "on a pin
    wall" mask and the box."""
    rng = np.random.default_rng(seed)
    m = int(round(1.0 / h))
    g = (np.arange(m) + 0.5) / m
    U0 = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    U0 = np.clip(U0 + rng.uniform(-jitter, jitter, U0.shape) * h, 0.25 * h, 1 - 0.25 * h)
    a = pin_frac / 2
    Up = U0
    if fluid_only:
        Up = U0[(np.abs(U0[:, 0] - 0.5) > a + 0.3 * h) | (np.abs(U0[:, 1] - 0.5) > a + 0.3 * h)]
    W = np.zeros((0, 2))
    if wall_points:
        k = max(1, int(round(pin_frac / h)))
        s = np.linspace(-a, a, k + 1)
        W = np.concatenate([np.column_stack([s, np.full_like(s, -a)]), np.column_stack([s, np.full_like(s, a)]),
                            np.column_stack([np.full(k - 1, -a), s[1:-1]]),
                            np.column_stack([np.full(k - 1, a), s[1:-1]])]) + 0.5
    nx = inlet_blocks + pins[0] + outlet_blocks
    P, wall = [], []
    for i in range(nx):
        in_array = inlet_blocks <= i < inlet_blocks + pins[0]
        for j in range(pins[1]):
            base = Up if in_array else U0
            P.append(base + [i, j])
            wall.append(np.zeros(len(base), bool))
            if wall_points and in_array:
                P.append(W + [i, j])
                wall.append(np.ones(len(W), bool))
    X = _round(np.concatenate(P))
    return X, np.concatenate(wall), (0.0, float(nx), 0.0, float(pins[1]))


def bilinear_mac(g, u, v, X):
    """Velocity of a MAC solution (grid ``g`` of :mod:`.flow`, lengths in unit cells) at points ``X``."""
    from scipy.interpolate import RegularGridInterpolator
    h = g.h
    xu, yu = np.arange(g.nx + 1) * h, (np.arange(g.ny) + 0.5) * h
    xv, yv = (np.arange(g.nx) + 0.5) * h, np.arange(g.ny + 1) * h
    fu = RegularGridInterpolator((xu, yu), u, bounds_error=False, fill_value=None)
    fv_ = RegularGridInterpolator((xv, yv), v, bounds_error=False, fill_value=None)
    return fu(X), fv_(X)


def snap_to_box(X, box, h):
    """Points within h / 2 of the box moved onto it (boundary nodes: the conditions are collocated where they hold)."""
    x0, x1, y0, y1 = box
    X = X.copy()
    for c, lo, hi in ((0, x0, x1), (1, y0, y1)):
        X[X[:, c] < lo + 0.5 * h, c] = lo
        X[X[:, c] > hi - 0.5 * h, c] = hi
    return _round(X)


def knn(X, Q, n, tree=None):
    """The ``n`` nearest nodes of each query point, ties broken by the offset (not by the k-d tree's traversal
    order): a block replicated by an integer shift then gets the same stencils, so the same matrix."""
    tree = tree or cKDTree(X)
    k = min(len(X), n + 8)
    d, idx = tree.query(Q, k=k)
    off = np.round((X[idx] - Q[:, None, :]) / _Q).astype(np.int64)
    dq = np.round(d / (1e-9 * max(np.abs(X).max(), 1.0))).astype(np.int64)
    out = np.empty((len(Q), n), np.int64)
    for i in range(len(Q)):
        o = np.lexsort((off[i, :, 1], off[i, :, 0], dq[i]))
        out[i] = idx[i, o[:n]]
    return out


def _neighbours(X, n):
    return knn(X, X, n)


def _boundary_masks(X, box, h):
    """Boundary nodes (on the box after :func:`snap_to_box`); a corner belongs to the inlet / outlet first."""
    x0, x1, y0, y1 = box
    tol = 1e-9
    inl, out = X[:, 0] < x0 + tol, X[:, 0] > x1 - tol
    return {"in": inl, "out": out & ~inl, "bot": (X[:, 1] < y0 + tol) & ~inl & ~out,
            "top": (X[:, 1] > y1 - tol) & ~inl & ~out}


def advection_diffusion(X, box, h, vel, alpha, q, *, method="rbf_fd", n=15, degree=2):
    """Collocated ``vel . grad T - alpha lap T = q``. Points within 3 h / 8 of the box carry the boundary condition:
    inlet ``T = 0``; outlet, top and bottom zero normal gradient."""
    N = len(X)
    idx = _neighbours(X, n)
    w = stencil_weights(X, X, idx, ("d0", "d1", "lap"), method=method, degree=degree)
    ux, uy = vel
    rows = np.repeat(np.arange(N), n)
    vals = (ux[:, None] * w["d0"] + uy[:, None] * w["d1"] - alpha * w["lap"]).ravel()
    b = np.asarray(q, float).copy()
    bm = _boundary_masks(X, box, h)
    vals = vals.reshape(N, n)
    for name, op in (("out", "d0"), ("bot", "d1"), ("top", "d1")):
        s = bm[name]
        vals[s] = w[op][s]
        b[s] = 0.0
    s = bm["in"]
    vals[s] = 0.0
    vals[s, 0] = 1.0                                  # the point itself is idx[:, 0]
    b[s] = 0.0
    A = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=(N, N))
    return A, b


def stokes(X, wall, box, h, nu=1.0, *, method="rbf_fd", n=20, degree=2):
    """Collocated Stokes (u, v, p at every fluid point). Rows: momentum and continuity at interior points;
    no-slip on pin-wall points (continuity replaced by the normal momentum balance there is not used: pressure at
    wall points is left to the continuity equation); inlet u = 1, v = 0; outlet p = 0, du/dx = dv/dx = 0;
    free-slip sides v = 0, du/dy = 0. Unknown order: u (N), v (N), p (N)."""
    N = len(X)
    idx = _neighbours(X, n)
    w = stencil_weights(X, X, idx, ("d0", "d1", "lap"), method=method, degree=degree)
    bm = _boundary_masks(X, box, h)
    R, C, V = [], [], []
    b = np.zeros(3 * N)

    def put(r, cols, vals):
        R.append(np.full(len(cols), r))
        C.append(cols)
        V.append(vals)
    for i in range(N):
        nb = idx[i]
        if bm["in"][i]:
            put(i, [i], [1.0])
            b[i] = 1.0
            put(N + i, [N + i], [1.0])
        elif wall[i]:
            put(i, [i], [1.0])
            put(N + i, [N + i], [1.0])
        elif bm["out"][i]:
            put(i, nb, w["d0"][i])
            put(N + i, N + nb, w["d0"][i])
        elif bm["bot"][i] or bm["top"][i]:
            put(i, nb, w["d1"][i])
            put(N + i, [N + i], [1.0])
        else:
            put(i, np.concatenate([nb, 2 * N + nb]), np.concatenate([-nu * w["lap"][i], w["d0"][i]]))
            put(N + i, np.concatenate([N + nb, 2 * N + nb]), np.concatenate([-nu * w["lap"][i], w["d1"][i]]))
        if bm["out"][i]:
            put(2 * N + i, [2 * N + i], [1.0])        # p = 0
        else:
            put(2 * N + i, np.concatenate([nb, N + nb]), np.concatenate([w["d0"][i], w["d1"][i]]))
    A = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(3 * N, 3 * N))
    return A, b


# -------------------------------------------------------------------------------------------- hybrid conduction
def hybrid_conduction(pins, cells: int, pin_frac: float, k_pin: float, k_mat: float, R_c: float, *, band: int = 2,
                      point_h: float | None = None, jitter: float = 0.25, n: int = 12, degree: int = 2,
                      seed: int = 0, q_pin: float = 1.0, h_top: float = 1.0):
    """Conduction ``-div(k grad T) = q`` (source ``q_pin`` in the pins) in a pins_x x pins_y array of unit cells,
    adiabatic sides, cooled on top (Robin ``h_top``, T_inf = 0), with a contact resistance ``R_c`` on every pin
    wall. FV cells (``cells`` per unit cell side) within ``band`` cells of a pin wall; RBF-FD points (spacing
    ``point_h``, default 1 / cells) elsewhere. Returns the matrix, right-hand side, unknown positions, the kind of
    each unknown (0 FV cell, 1 point) and the conductivity at each unknown."""
    h = 1.0 / cells
    ph = point_h or h
    a = pin_frac / 2
    nx, ny = pins[0] * cells, pins[1] * cells
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    xc, yc = (i + 0.5) * h, (j + 0.5) * h
    lx, ly = xc % 1.0 - 0.5, yc % 1.0 - 0.5
    pin_cell = (np.abs(lx) < a) & (np.abs(ly) < a)
    # cells within ``band`` cells of a pin wall (on either side)
    near_x = np.abs(ly) < a + band * h
    near_y = np.abs(lx) < a + band * h
    in_band = ((np.abs(np.abs(lx) - a) < band * h) & near_x) | ((np.abs(np.abs(ly) - a) < band * h) & near_y)
    fvc = np.argwhere(in_band)
    Xf = np.column_stack([xc[in_band], yc[in_band]])
    kf = np.where(pin_cell[in_band], k_pin, k_mat)
    # scattered points outside the band
    rng = np.random.default_rng(seed)
    m = int(round(1.0 / ph))
    g = (np.arange(m) + 0.5) / m
    U = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + rng.uniform(-jitter, jitter, (m * m, 2)) * ph
    U = np.clip(U, 0.25 * ph, 1 - 0.25 * ph)
    lu = U - 0.5
    ok = ~((np.abs(np.abs(lu[:, 0]) - a) < (band + 0.5) * h) & (np.abs(lu[:, 1]) < a + (band + 0.5) * h)) & \
        ~((np.abs(np.abs(lu[:, 1]) - a) < (band + 0.5) * h) & (np.abs(lu[:, 0]) < a + (band + 0.5) * h))
    U = U[ok]
    Xp = snap_to_box(np.concatenate([U + [bi, bj] for bi in range(pins[0]) for bj in range(pins[1])]),
                     (0.0, float(pins[0]), 0.0, float(pins[1])), ph)
    lp = Xp % 1.0 - 0.5
    kp = np.where((np.abs(lp[:, 0]) < a) & (np.abs(lp[:, 1]) < a), k_pin, k_mat)
    X = np.concatenate([Xf, Xp])
    kind = np.r_[np.zeros(len(Xf), int), np.ones(len(Xp), int)]
    kk = np.r_[kf, kp]
    nf, N = len(Xf), len(X)
    cell_id = -np.ones((nx, ny), int)
    cell_id[in_band] = np.arange(nf)
    tree = cKDTree(X)
    R, C, V = [], [], []
    b = np.zeros(N)
    qf = np.where(pin_cell[in_band], q_pin, 0.0)
    b[:nf] = qf * h * h
    # FV rows: neighbour cell in the band -> its unknown; outside -> RBF interpolation at its centre
    ghosts, ghost_rows, ghost_g = [], [], []
    diag = np.zeros(N)
    for r, (ci, cj) in enumerate(fvc):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= nx or (nj < 0):
                continue                               # adiabatic sides and bottom
            if nj >= ny:                               # top: Robin
                gt = h / (h / (2 * kf[r]) + 1.0 / h_top)
                diag[r] += gt
                continue
            k2 = k_pin if pin_cell[ni, nj] else k_mat
            wall = pin_cell[ci, cj] != pin_cell[ni, nj]
            gface = h / (h / (2 * kf[r]) + (R_c if wall else 0.0) + h / (2 * k2))   # face length h
            diag[r] += gface
            c2 = cell_id[ni, nj]
            if c2 >= 0:
                R.append(r)
                C.append(c2)
                V.append(-gface)
            else:
                ghosts.append([(ni + 0.5) * h, (nj + 0.5) * h])
                ghost_rows.append(r)
                ghost_g.append(gface)
    if ghosts:
        G = np.array(ghosts)
        gi = knn(X, G, n, tree)
        we = stencil_weights(X, G, gi, ("eval",), method="rbf_fd", degree=degree)["eval"]
        for r, g_, cols, wts in zip(ghost_rows, ghost_g, gi, we, strict=True):
            R.extend([r] * len(cols))
            C.extend(cols.tolist())
            V.extend((-g_ * wts).tolist())
    # point rows: k lap T = -q with k constant around the point (no interface inside the stencil, by the band)
    pi = knn(X, Xp, n, tree)
    wl = stencil_weights(X, Xp, pi, ("lap", "d0", "d1"), method="rbf_fd", degree=degree)
    top = Xp[:, 1] >= pins[1] - 1e-9
    side = ((Xp[:, 0] <= 1e-9) | (Xp[:, 0] >= pins[0] - 1e-9)) & ~top
    bottom = (Xp[:, 1] <= 1e-9) & ~side
    for t, (cols, wts) in enumerate(zip(pi, wl["lap"], strict=True)):
        r = nf + t
        if top[t]:                                     # Robin on top: -k dT/dy = h_top T
            R.extend([r] * len(cols))
            C.extend(cols.tolist())
            V.extend((kp[t] * wl["d1"][t]).tolist())
            diag[r] += h_top
            continue
        if side[t] or bottom[t]:                       # adiabatic
            R.extend([r] * len(cols))
            C.extend(cols.tolist())
            V.extend((wl["d0"][t] if side[t] else wl["d1"][t]).tolist())
            continue
        R.extend([r] * len(cols))
        C.extend(cols.tolist())
        V.extend((-kp[t] * wts).tolist())
        qq = q_pin if kp[t] == k_pin else 0.0
        b[r] = qq
    R.extend(range(N))
    C.extend(range(N))
    V.extend(diag.tolist())
    A = sp.csr_matrix((np.array(V), (np.array(R), np.array(C))), shape=(N, N))
    return A, b, X, kind, kk


def fv_conduction(pins, cells, pin_frac, k_pin, k_mat, R_c, q_pin=1.0, h_top=1.0):
    """The same conduction problem on finite volumes everywhere (the reference and the equal-unknowns baseline)."""
    h = 1.0 / cells
    nx, ny = pins[0] * cells, pins[1] * cells
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    lx, ly = ((i + 0.5) * h) % 1.0 - 0.5, ((j + 0.5) * h) % 1.0 - 0.5
    a = pin_frac / 2
    pin_cell = (np.abs(lx) < a) & (np.abs(ly) < a)
    k = np.where(pin_cell, k_pin, k_mat)
    idx = np.arange(nx * ny).reshape(nx, ny)
    R, C, V = [], [], []
    diag = np.zeros(nx * ny)
    for (s0, s1) in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        k1, k2 = k[s0], k[s1]
        wall = pin_cell[s0] != pin_cell[s1]
        gf = h / (h / (2 * k1) + h / (2 * k2) + np.where(wall, R_c, 0.0))      # face length h
        a1, a2 = idx[s0].ravel(), idx[s1].ravel()
        R += [a1, a2]
        C += [a2, a1]
        V += [-gf.ravel(), -gf.ravel()]
        np.add.at(diag, a1, gf.ravel())
        np.add.at(diag, a2, gf.ravel())
    gt = h / (h / (2 * k[:, -1]) + 1.0 / h_top)
    np.add.at(diag, idx[:, -1], gt)
    R.append(np.arange(nx * ny))
    C.append(np.arange(nx * ny))
    V.append(diag)
    A = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(nx * ny, nx * ny))
    b = (np.where(pin_cell, q_pin, 0.0) * h * h).ravel()
    X = np.column_stack([((i + 0.5) * h).ravel(), ((j + 0.5) * h).ravel()])
    return A, b, X, k.ravel()
