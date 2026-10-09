"""Vortex lattice method for the wing + horizontal tail of an Airframe.

Horseshoe vortices (bound leg on the panel quarter chord, trailing legs to downstream infinity along x), control points
at the panel three-quarter chord with the camber-line slope, twist and dihedral in the normals. The solution is linear
in the angle of attack and the tail incidence, so three solves give everything: lift, moment and the spanwise loading
as linear functions, induced drag (Trefftz plane) as a quadratic one. Trim, neutral point, stall station and drag at
any flight condition are then cheap.

Axes as the airframe: x aft, y right, z up. Free stream at angle alpha: V = (cos a, 0, sin a) * (-1) relative to the
aircraft moving forward, i.e. the air comes from -x... we use the usual aero convention with the air flowing towards
+x: V_inf = (cos a, 0, sin a).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .geometry import surfaces


@dataclass
class Lattice:
    P1: np.ndarray        # (N, 3) bound vortex start
    P2: np.ndarray        # (N, 3) bound vortex end
    C: np.ndarray         # (N, 3) control points
    nrm0: np.ndarray      # (N, 3) normals at zero incidence (camber + geometric twist/incidence included)
    surf: np.ndarray      # (N,) 0 = wing, 1 = htail
    strip: np.ndarray     # (N,) strip index (spanwise) within the whole lattice
    chord: np.ndarray     # (Nstrips,) local chord per strip
    yc: np.ndarray        # (Nstrips,) strip centre y
    dy: np.ndarray        # (Nstrips,) strip width (projected on y)
    strip_surf: np.ndarray
    tail_rot: np.ndarray  # (N, 3) d(normal)/d(i_t) for tail panels (zero for the wing)


def _camber_slope(shape, xc):
    yu, yl = surfaces(shape, xc)
    camber = (yu + yl) / 2
    return np.gradient(camber, xc)


def build_lattice(af, nc: int = 5, ns_wing: int = 14, ns_tail: int = 6, tail: bool = True) -> Lattice:
    P1, P2, C, N, S, ST, TR = [], [], [], [], [], [], []
    chords, ycs, dys, ssurf = [], [], [], []
    strip_id = 0

    def surface(kind, xle_root, z_root, b2, cr, lam, sweep, dihed, inc_root, twist, shape, ns):
        nonlocal strip_id
        # spanwise edges, cosine spacing over the full span (-b2..b2)
        th = np.linspace(0, math.pi, 2 * ns + 1)
        ye = -b2 * np.cos(th)
        xcs = np.linspace(0, 1, 401)
        slope = _camber_slope(shape, xcs) if shape is not None else np.zeros_like(xcs)
        for k in range(len(ye) - 1):
            ya, yb = ye[k], ye[k + 1]
            ea, eb = abs(ya) / b2, abs(yb) / b2
            def lead(y, e):
                return np.array([xle_root + e * b2 * math.tan(math.radians(sweep)), y,
                                 z_root + e * b2 * math.tan(math.radians(dihed))])
            ca, cb = cr * (1 - (1 - lam) * ea), cr * (1 - (1 - lam) * eb)
            La, Lb = lead(ya, ea), lead(yb, eb)
            ym = -b2 * math.cos(0.5 * (th[k] + th[k + 1]))         # theta midpoint: exact Trefftz drag, better loads
            f = (ym - ya) / (yb - ya)
            em = abs(ym) / b2
            cm = cr * (1 - (1 - lam) * em)
            twm = inc_root + twist * em
            for i in range(nc):
                x0, x1 = i / nc, (i + 1) / nc
                xq, x3 = x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0)
                P1.append(La + [xq * ca, 0, 0]); P2.append(Lb + [xq * cb, 0, 0])
                c3 = La + f * (Lb - La) + [x3 * cm, 0, 0]
                C.append(c3)
                # normal: panel spanwise tangent x chord direction rotated by local incidence + camber slope
                dz = float(np.interp(x3, xcs, slope))
                ang = math.radians(twm) - math.atan(dz)               # positive incidence = nose up
                t_chord = np.array([math.cos(ang), 0, -math.sin(ang)])
                t_span = (Lb - La) / np.linalg.norm(Lb - La)
                n = np.cross(t_chord, t_span)
                n = n if n[2] > 0 else -n
                N.append(n / np.linalg.norm(n))
                # d n / d(incidence) (tail incidence derivative): rotate chord direction
                t_d = np.array([-math.sin(ang), 0, -math.cos(ang)])
                nd = np.cross(t_d, t_span)
                nd = nd * (1 if np.dot(np.cross(t_chord, t_span), n) > 0 else -1)
                TR.append(nd if kind == 1 else np.zeros(3))
                S.append(kind); ST.append(strip_id)
            chords.append(cm); ycs.append(ym); dys.append(yb - ya); ssurf.append(kind)
            strip_id += 1

    z0 = af.wing_z_root()
    surface(0, af.wing_x, z0, af.span / 2, af.root_chord, af.taper, af.sweep_le, af.dihedral, af.incidence,
            af.twist, af.section, ns_wing)
    if not tail:
        return Lattice(np.array(P1), np.array(P2), np.array(C), np.array(N), np.array(S), np.array(ST),
                       np.array(chords), np.array(ycs), np.array(dys), np.array(ssurf), np.array(TR))
    ht = af.htail()
    surface(1, ht["x_le"], ht["z"], ht["span"] / 2, ht["root_chord"], ht["taper"], ht.get("sweep_le", 8.0),
            ht.get("dihedral", 0.0), 0.0, 0.0, None, ns_tail)
    return Lattice(np.array(P1), np.array(P2), np.array(C), np.array(N), np.array(S), np.array(ST),
                   np.array(chords), np.array(ycs), np.array(dys), np.array(ssurf), np.array(TR))


def _seg_induced(P, A, B, eps=1e-9):
    """Velocity at points P (M,3) induced by unit-strength segments A->B (N,3): (M, N, 3)."""
    r1 = P[:, None, :] - A[None]
    r2 = P[:, None, :] - B[None]
    r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = (cr ** 2).sum(-1)
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    k = (np.einsum("nk,mnk->mn", r0, r1 / (n1[..., None] + eps) - r2 / (n2[..., None] + eps))) / (4 * math.pi * (cr2 + eps))
    k = np.where(cr2 < 1e-10, 0.0, k)
    return cr * k[..., None]


def _semi_induced(P, A, d, eps=1e-9):
    """Semi-infinite vortex from A along unit direction d to infinity: (M, N, 3)."""
    r = P[:, None, :] - A[None]
    nr = np.linalg.norm(r, axis=-1)
    cr = np.cross(d[None, None, :] * np.ones_like(r), r)
    cr2 = (cr ** 2).sum(-1)
    k = (1 + np.einsum("k,mnk->mn", d, r) / (nr + eps)) / (4 * math.pi * (cr2 + eps))
    k = np.where(cr2 < 1e-10, 0.0, k)
    return cr * k[..., None]


def horseshoe_induced(P, L: Lattice, legs=True):
    d = np.array([1.0, 0, 0])
    v = _seg_induced(P, L.P1, L.P2)
    if legs:
        v = v - _semi_induced(P, L.P1, d) + _semi_induced(P, L.P2, d)
    return v


@dataclass
class VLMSolution:
    L: Lattice
    G: np.ndarray          # (3, N) circulation for unit [1, alpha (rad), i_t (rad)] at V_inf = 1
    s_ref: float
    c_ref: float
    b_ref: float
    x_ref: float
    CL: np.ndarray         # (3,) linear coefficients
    Cm: np.ndarray         # (3,)
    CLw: np.ndarray        # (3,) wing-only lift
    cl_strip: np.ndarray   # (3, Nstrips) local cl
    Q: np.ndarray          # (3, 3) induced drag quadratic form: CDi = u^T Q u, u = [1, a, i_t]

    def coefficients(self, alpha: float, it: float) -> Dict[str, float]:
        u = np.array([1.0, alpha, it])
        return {"CL": float(self.CL @ u), "Cm": float(self.Cm @ u), "CDi": float(u @ self.Q @ u),
                "CLw": float(self.CLw @ u)}

    def trim(self, CL: float) -> Tuple[float, float]:
        """(alpha, i_t) in radians giving this CL with zero pitching moment."""
        A = np.array([[self.CL[1], self.CL[2]], [self.Cm[1], self.Cm[2]]])
        b = np.array([CL - self.CL[0], -self.Cm[0]])
        a, it = np.linalg.solve(A, b)
        return float(a), float(it)

    def strip_cl(self, alpha, it):
        return self.cl_strip.T @ np.array([1.0, alpha, it])

    @property
    def neutral_point(self) -> float:
        """x of the neutral point (aircraft, fixed tail incidence)."""
        return self.x_ref - self.Cm[1] / self.CL[1] * self.c_ref


def solve(af, x_ref: float, nc: int = 5, ns_wing: int = 14, ns_tail: int = 6, tail: bool = True) -> VLMSolution:
    L = build_lattice(af, nc, ns_wing, ns_tail, tail)
    N = len(L.C)
    Aind = horseshoe_induced(L.C, L)                      # (N, N, 3)
    A = np.einsum("mnk,mk->mn", Aind, L.nrm0)
    # right-hand sides: V = (cos a, 0, sin a) ~ (1, 0, a) linearised; tail incidence rotates tail normals
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    rhs0 = -(L.nrm0 @ ex)
    rhsa = -(L.nrm0 @ ez)
    rhsi = -(L.tail_rot @ ex)
    G = np.linalg.solve(A, np.stack([rhs0, rhsa, rhsi], 1)).T          # (3, N)
    S, c, b = af.wing_area, af.mac, af.span
    # forces by Kutta-Joukowski on bound legs with the free stream only (linear lift), q = 0.5
    dl = L.P2 - L.P1
    mid = 0.5 * (L.P1 + L.P2)
    F_unit = np.cross(ex, dl)                                          # (N, 3) force per unit Gamma (rho V = 1)
    # lift ~ z component; moment about y through x_ref: M = (r x F)_y ; nose-up positive -> -My? (x aft, z up)
    Fz = F_unit[:, 2]
    r = mid - [x_ref, 0, 0]
    My = r[:, 2] * F_unit[:, 0] - r[:, 0] * F_unit[:, 2]               # about +y (right wing): nose-up positive
    CL = G @ Fz / (0.5 * S)
    Cm = G @ My / (0.5 * S * c)
    CLw = (G * (L.surf == 0)) @ Fz / (0.5 * S)
    # strip loading: sum Gamma over chordwise panels; local cl = 2 Gamma_strip / c_local
    ns = len(L.chord)
    Mstrip = np.zeros((ns, N))
    Mstrip[L.strip, np.arange(N)] = 1.0
    gam_strip = G @ Mstrip.T                                            # (3, ns)
    cl_strip = 2 * gam_strip / L.chord[None]
    # Trefftz plane: trailing vortices of strength (Gamma_strip difference) at strip edges in the y-z plane
    # induced downwash at strip centres from 2D point vortices at each strip edge
    edges_y, edges_z, wst = [], [], []
    for s in range(ns):
        sel = L.strip == s
        k = np.where(sel)[0][0]
        edges_y.append((L.P1[k, 1], L.P2[k, 1])); edges_z.append((L.P1[k, 2], L.P2[k, 2]))
    edges_y, edges_z = np.array(edges_y), np.array(edges_z)
    fy = (L.yc - edges_y[:, 0]) / (edges_y[:, 1] - edges_y[:, 0])
    yc, zc = L.yc, edges_z[:, 0] + fy * (edges_z[:, 1] - edges_z[:, 0])
    # vortex filaments: +Gamma at P1 edge (y1), -Gamma at P2 edge (y2) -> 2D velocity at (yc, zc)
    def w2d(yv, zv, y, z):
        dy_, dz_ = y[:, None] - yv[None], z[:, None] - zv[None]
        r2 = dy_ ** 2 + dz_ ** 2 + 1e-9
        # vortex along +x of unit strength: v = (-dz, dy)/(2 pi r^2) in (y, z)
        return -dz_ / (2 * math.pi * r2), dy_ / (2 * math.pi * r2)
    vy1, vz1 = w2d(edges_y[:, 0], edges_z[:, 0], yc, zc)
    vy2, vz2 = w2d(edges_y[:, 1], edges_z[:, 1], yc, zc)
    # normal of each strip in the Trefftz plane
    ty, tz = edges_y[:, 1] - edges_y[:, 0], edges_z[:, 1] - edges_z[:, 0]
    ln = np.hypot(ty, tz)
    ny_, nz_ = -tz / ln, ty / ln
    Wn = (-(vy1 - vy2)) * ny_[:, None] + (-(vz1 - vz2)) * nz_[:, None]   # (ns, ns): normal wash per unit strip Gamma
    # Di = -rho/2 sum Gamma_i w_i ds_i  (w negative for downwash) -> CDi = sum Gamma_i (-w_i) ln_i / (V^2 S)
    Q = np.einsum("ai,ij,bj,i->ab", gam_strip, -Wn, gam_strip, ln) / (S) * 1.0
    Q = 0.5 * (Q + Q.T)
    return VLMSolution(L, G, S, c, b, x_ref, CL, Cm, CLw, cl_strip, Q)


def streamlines(sol: VLMSolution, alpha: float, it: float, seeds: np.ndarray, steps: int = 220, h: float = 0.08,
                core: float = 0.08) -> np.ndarray:
    """Potential-flow streamlines (free stream + horseshoe vortices, desingularised with a core radius).
    The wake is the flat horseshoe sheet: it shows the downwash and the swirl around the tips, not the roll-up.
    Returns (n_seeds, steps+1, 3)."""
    L = sol.L
    G = sol.G.T @ np.array([1.0, alpha, it])                  # (N,)
    Vinf = np.array([math.cos(alpha), 0.0, math.sin(alpha)])

    def vel(P):
        v = horseshoe_induced(P, L)                          # (M, N, 3)
        # Rankine-like core: damp the induced velocity near the filaments
        r = np.linalg.norm(v, axis=-1, keepdims=True)
        lim = 1.0 / (2 * math.pi * core)
        v = np.where(r > lim, v * lim / (r + 1e-12), v)
        return Vinf + np.einsum("mnk,n->mk", v, G)

    X = np.asarray(seeds, float).copy()
    out = [X.copy()]
    for _ in range(steps):
        v1 = vel(X)
        Xm = X + 0.5 * h * v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = vel(Xm)
        X = X + h * v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        out.append(X.copy())
    return np.stack(out, 1)
