"""Steady 2-D incompressible flow and heat on a staggered (MAC) finite-volume grid, assembled as one sparse system:
the reference and the operator of the CFD falsification tests (``docs/research/condensation_falsification.md``).

Grid: ``nx x ny`` square cells of side ``h``; ``solid`` marks blocked cells (pins, fins). Unknowns: ``u`` on vertical
faces ``(nx + 1) x ny``, ``v`` on horizontal faces ``nx x (ny + 1)``, ``p`` in fluid cells. Velocities on faces of
solid cells, on the inlet and on the top/bottom walls are Dirichlet (identity rows), so every face keeps its unknown
and the continuity rows keep their pattern.

**Momentum (Oseen).** ``(a . grad) u - nu lap u + grad p = f`` with a frozen advecting velocity ``a`` (zero: Stokes;
the previous iterate: Picard for Navier-Stokes). Convection uses the hybrid scheme (central where the face Peclet
number is below 2, upwind above; Patankar, *Numerical Heat Transfer and Fluid Flow*, 1980, ch. 5).

* inlet ``x = 0``: given ``u`` profile, ``v = 0``;
* outlet ``x = L``: zero normal gradient of velocity and ``p = 0`` in the ghost cell (sets the pressure level);
* top and bottom: ``v = 0`` and either no-slip (``u`` ghost ``-u``, or ``2 U - u`` for a moving lid) or free slip
  (zero gradient), so a strip of a periodic pin array can be represented with free-slip sides;
* ``closed=True`` (cavity): walls on all four sides and one pressure fixed to zero.

**Energy / scalar.** ``rho c (u . grad T) - div(k grad T) = q`` on all cells (fluid and solid) with the face
velocities of the flow solution, harmonic-mean face conductivities, hybrid convection, ``T = T_in`` at the inlet,
zero diffusive flux at the outlet and on the walls.

:func:`partition_kw` gives the positions and fields of the unknowns for :mod:`.algebraic`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _A(pe):
    """Hybrid-scheme weight of the diffusive conductance."""
    return np.maximum(0.0, 1.0 - 0.5 * np.abs(pe))


@dataclass
class Grid:
    nx: int
    ny: int
    h: float
    solid: np.ndarray | None = None                  # (nx, ny) bool
    walls: str = "noslip"                            # top/bottom: "noslip" | "slip"
    closed: bool = False                             # cavity: walls on all sides
    lid: float = 0.0                                 # top-wall speed (closed cavity)
    index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.solid is None:
            self.solid = np.zeros((self.nx, self.ny), bool)
        nx, ny = self.nx, self.ny
        nu_, nv_ = (nx + 1) * ny, nx * (ny + 1)
        self.iu = np.arange(nu_).reshape(nx + 1, ny)
        self.iv = nu_ + np.arange(nv_).reshape(nx, ny + 1)
        fluid = ~self.solid
        self.ip = -np.ones((nx, ny), np.int64)
        self.ip[fluid] = nu_ + nv_ + np.arange(int(fluid.sum()))
        self.n = nu_ + nv_ + int(fluid.sum())

    # -- which faces are Dirichlet ----------------------------------------------------------------------------------
    def u_fixed(self) -> np.ndarray:
        s = self.solid
        f = np.zeros((self.nx + 1, self.ny), bool)
        f[1:-1] = s[:-1] | s[1:]
        f[0] = True                                  # inlet (or the left wall of a cavity)
        f[-1] = s[-1] if not self.closed else True
        return f

    def v_fixed(self) -> np.ndarray:
        s = self.solid
        f = np.zeros((self.nx, self.ny + 1), bool)
        f[:, 1:-1] = s[:, :-1] | s[:, 1:]
        f[:, 0] = f[:, -1] = True
        return f

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and field ids (0 u, 1 v, 2 p) of the unknowns."""
        h, nx, ny = self.h, self.nx, self.ny
        X = np.zeros((self.n, 2))
        F = np.zeros(self.n, np.int64)
        i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")
        X[self.iu.ravel()] = np.column_stack([i.ravel() * h, (j.ravel() + 0.5) * h])
        i, j = np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij")
        X[self.iv.ravel()] = np.column_stack([(i.ravel() + 0.5) * h, j.ravel() * h])
        F[self.iv.ravel()] = 1
        i, j = np.nonzero(self.ip >= 0)
        X[self.ip[i, j]] = np.column_stack([(i + 0.5) * h, (j + 0.5) * h])
        F[self.ip[i, j]] = 2
        return X, F


def assemble_oseen(g: Grid, nu: float, inlet: np.ndarray | None = None, adv: tuple | None = None,
                   force: tuple[float, float] = (0.0, 0.0)) -> tuple[sp.csr_matrix, np.ndarray]:
    """The Oseen (Stokes when ``adv`` is None) system ``A x = b``; ``inlet``: u on the inlet faces (ny,);
    ``adv``: advecting face velocities ``(au (nx+1, ny), av (nx, ny+1))``; ``force``: body force per unit mass."""
    nx, ny, h = g.nx, g.ny, g.h
    au = np.zeros((nx + 1, ny)) if adv is None else adv[0]
    av = np.zeros((nx, ny + 1)) if adv is None else adv[1]
    uf, vf = g.u_fixed(), g.v_fixed()
    R, C, V = [], [], []
    b = np.zeros(g.n)

    def add(r, c, v):
        R.append(r)
        C.append(c)
        V.append(v)

    D = nu                                           # diffusive conductance of a face (square cells)
    # ---- u-momentum
    for i in range(nx + 1):
        for j in range(ny):
            r = g.iu[i, j]
            if uf[i, j]:
                add(r, r, 1.0)
                if i == 0 and not g.closed and inlet is not None:
                    b[r] = inlet[j]
                continue
            # face fluxes of the u control volume (per unit depth)
            Fe = 0.5 * (au[i, j] + (au[i + 1, j] if i < nx else au[i, j])) * h
            Fw = 0.5 * (au[i, j] + au[i - 1, j]) * h
            Fn = 0.5 * (av[i - 1, j + 1] + (av[i, j + 1] if i < nx else av[i - 1, j + 1])) * h
            Fs = 0.5 * (av[i - 1, j] + (av[i, j] if i < nx else av[i - 1, j])) * h
            aP = 0.0
            for F, nb, sign in ((Fe, (i + 1, j), -1), (Fw, (i - 1, j), +1)):
                a_nb = D * _A(F / D) + max(sign * F, 0.0)
                ii, jj = nb
                if ii > nx:                          # outlet: zero-gradient ghost = the face itself
                    continue
                aP += a_nb
                if uf[ii, jj]:                       # Dirichlet neighbour: the inlet profile, a wall or a solid face
                    if ii == 0 and inlet is not None and not g.closed:
                        b[r] += a_nb * inlet[jj]
                else:
                    add(r, g.iu[ii, jj], -a_nb)
            for F, jj, sign in ((Fn, j + 1, -1), (Fs, j - 1, +1)):
                a_nb = D * _A(F / D) + max(sign * F, 0.0)
                if 0 <= jj < ny:
                    aP += a_nb
                    if not uf[i, jj]:                # a Dirichlet neighbour here is a solid face: zero
                        add(r, g.iu[i, jj], -a_nb)
                else:                                # top / bottom wall
                    if g.walls == "noslip" or g.closed:
                        wall_u = g.lid if (jj == ny and g.closed) else 0.0
                        aP += 2 * D                  # ghost -u (+ 2 U_wall): wall at h / 2
                        b[r] += 2 * D * wall_u
            aP += (Fe - Fw) + (Fn - Fs)              # zero for a divergence-free advecting field
            add(r, r, aP)
            pl = g.ip[i - 1, j]
            pr = g.ip[i, j] if i < nx else -2
            if pr >= 0:
                add(r, pr, h)
            if pl >= 0:
                add(r, pl, -h)
            b[r] += force[0] * h * h
    # ---- v-momentum
    for i in range(nx):
        for j in range(ny + 1):
            r = g.iv[i, j]
            if vf[i, j]:
                add(r, r, 1.0)
                continue
            Fn = 0.5 * (av[i, j] + av[i, j + 1]) * h
            Fs = 0.5 * (av[i, j] + av[i, j - 1]) * h
            Fe = 0.5 * (au[i + 1, j - 1] + au[i + 1, j]) * h
            Fw = 0.5 * (au[i, j - 1] + au[i, j]) * h
            aP = 0.0
            for F, jj, sign in ((Fn, j + 1, -1), (Fs, j - 1, +1)):
                a_nb = D * _A(F / D) + max(sign * F, 0.0)
                aP += a_nb
                if not vf[i, jj]:
                    add(r, g.iv[i, jj], -a_nb)
            for F, ii, sign in ((Fe, i + 1, -1), (Fw, i - 1, +1)):
                a_nb = D * _A(F / D) + max(sign * F, 0.0)
                if 0 <= ii < nx:
                    aP += a_nb
                    if not vf[ii, j]:
                        add(r, g.iv[ii, j], -a_nb)
                elif ii < 0:                         # inlet (v = 0) or the left cavity wall: ghost -v
                    aP += 2 * D
                else:                                # outlet: zero gradient (ghost = v), or the right cavity wall
                    if g.closed:
                        aP += 2 * D
            aP += (Fe - Fw) + (Fn - Fs)
            add(r, r, aP)
            add(r, g.ip[i, j], h)
            add(r, g.ip[i, j - 1], -h)
            b[r] += force[1] * h * h
    # ---- continuity
    first = True
    for i in range(nx):
        for j in range(ny):
            r = g.ip[i, j]
            if r < 0:
                continue
            if g.closed and first:                   # closed domain: one pressure fixed
                add(r, r, 1.0)
                first = False
                continue
            add(r, g.iu[i + 1, j], h)
            add(r, g.iu[i, j], -h)
            add(r, g.iv[i, j + 1], h)
            add(r, g.iv[i, j], -h)
    A = sp.csr_matrix((np.array(V, float), (np.array(R), np.array(C))), shape=(g.n, g.n))
    return A, b


def split(g: Grid, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = x[g.iu]
    v = x[g.iv]
    p = np.full((g.nx, g.ny), np.nan)
    m = g.ip >= 0
    p[m] = x[g.ip[m]]
    return u, v, p


def solve(A, b) -> np.ndarray:
    return spla.spsolve(A.tocsc(), b)


def picard(g: Grid, nu: float, inlet=None, tol: float = 1e-8, maxiter: int = 200, relax: float = 1.0,
           linear_solver=None, force=(0.0, 0.0)) -> dict:
    """Navier-Stokes by Picard (Oseen) iteration from Stokes: ``linear_solver(A, b) -> x`` (default: direct).
    Returns the solution, the iteration history of the relative velocity change and the per-iteration systems'
    solve times."""
    import time
    linear_solver = linear_solver or solve
    A, b = assemble_oseen(g, nu, inlet, None, force)
    t0 = time.perf_counter()
    x = linear_solver(A, b)
    times = [time.perf_counter() - t0]
    hist = []
    for _ in range(maxiter):
        u, v, _ = split(g, x)
        A, b = assemble_oseen(g, nu, inlet, (u, v), force)
        t0 = time.perf_counter()
        xn = linear_solver(A, b)
        times.append(time.perf_counter() - t0)
        nvel = len(g.iu.ravel()) + len(g.iv.ravel())
        dx = np.linalg.norm(xn[:nvel] - x[:nvel]) / max(np.linalg.norm(xn[:nvel]), 1e-300)
        x = relax * xn + (1 - relax) * x
        hist.append(float(dx))
        if dx < tol:
            break
    return {"x": x, "history": hist, "iterations": len(hist), "solve_s": times, "A": A, "b": b}


# ------------------------------------------------------------------------------------------------- energy / scalar
@dataclass
class ScalarIndex:
    nx: int
    ny: int

    @property
    def n(self):
        return self.nx * self.ny


def assemble_scalar(g: Grid, u: np.ndarray, v: np.ndarray, k: np.ndarray, rhoc: np.ndarray, q: np.ndarray,
                    T_in: float = 0.0, scheme: str = "hybrid") -> tuple[sp.csr_matrix, np.ndarray]:
    """``rho c (u . grad T) - div(k grad T) = q`` on every cell; ``k``, ``rhoc``, ``q`` are (nx, ny) cell fields
    (``rhoc`` zero in solids: no advection there; face velocities are zero on solid faces anyway). ``scheme``:
    "hybrid" or "upwind"."""
    nx, ny, h = g.nx, g.ny, g.h
    idx = np.arange(nx * ny).reshape(nx, ny)
    R, C, V = [], [], []
    b = (q * h * h).ravel().astype(float).copy()
    diag = np.zeros(nx * ny)

    def wts(F, Dc):
        if scheme == "upwind":
            return Dc + max(-F, 0.0), Dc + max(F, 0.0)
        pe = F / Dc if Dc > 0 else np.inf
        aE = Dc * _A(pe) + max(-F, 0.0)
        aW = Dc * _A(pe) + max(F, 0.0)
        return aE, aW

    # x-faces between cells (i-1, j) and (i, j)
    for i in range(1, nx):
        for j in range(ny):
            kf = 2 * k[i - 1, j] * k[i, j] / (k[i - 1, j] + k[i, j])
            rc = 0.5 * (rhoc[i - 1, j] + rhoc[i, j])
            F = rc * u[i, j] * h
            a_to_right, a_to_left = wts(F, kf)      # coefficient of T_right in the left cell, and the other way
            L, Rr = idx[i - 1, j], idx[i, j]
            R += [L, L, Rr, Rr]
            C += [L, Rr, Rr, L]
            V += [a_to_right + F, -a_to_right, a_to_left - F, -a_to_left]
    for i in range(nx):
        for j in range(1, ny):
            kf = 2 * k[i, j - 1] * k[i, j] / (k[i, j - 1] + k[i, j])
            rc = 0.5 * (rhoc[i, j - 1] + rhoc[i, j])
            F = rc * v[i, j] * h
            a_up, a_down = wts(F, kf)
            L, Rr = idx[i, j - 1], idx[i, j]
            R += [L, L, Rr, Rr]
            C += [L, Rr, Rr, L]
            V += [a_up + F, -a_up, a_down - F, -a_down]
    if not g.closed:
        for j in range(ny):                          # inlet: T = T_in at the face (half-cell conductance), inflow
            F = rhoc[0, j] * u[0, j] * h
            Dc = 2 * k[0, j]
            diag[idx[0, j]] += Dc + max(F, 0.0)
            b[idx[0, j]] += (Dc + max(F, 0.0)) * T_in
            F = rhoc[-1, j] * u[-1, j] * h          # outlet: upwind outflow, no diffusion
            diag[idx[-1, j]] += max(F, 0.0)
    R += list(idx.ravel())
    C += list(idx.ravel())
    V += list(diag)
    A = sp.csr_matrix((np.array(V, float), (np.array(R), np.array(C))), shape=(nx * ny, nx * ny))
    return A, b


def scalar_coords(g: Grid) -> np.ndarray:
    i, j = np.meshgrid(np.arange(g.nx), np.arange(g.ny), indexing="ij")
    return np.column_stack([(i.ravel() + 0.5) * g.h, (j.ravel() + 0.5) * g.h])


def pin_array(n_pins: tuple[int, int], cells: int, pin: int, h: float, inlet_cells: int = 0,
              outlet_cells: int = 0) -> np.ndarray:
    """Solid mask of an array of square pins: ``cells`` per unit cell side, a ``pin x pin`` pin centred in each,
    with ``inlet_cells`` / ``outlet_cells`` empty columns before and after."""
    px, py = n_pins
    unit = np.zeros((cells, cells), bool)
    a = (cells - pin) // 2
    unit[a:a + pin, a:a + pin] = True
    s = np.tile(unit, (px, py))
    return np.vstack([np.zeros((inlet_cells, py * cells), bool), s, np.zeros((outlet_cells, py * cells), bool)])
