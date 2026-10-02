"""Compressible Navier-Stokes / Euler finite-volume solver (2-D and 3-D Cartesian), PyTorch.

Scheme
------
* Conservative variables U = (rho, rho*u_d..., E) on a uniform Cartesian grid, two ghost layers.
* MUSCL reconstruction of primitive variables (rho, u_d..., p) with a minmod or van Leer limiter.
* HLLC approximate Riemann solver (Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, ch. 10)
  with Davis wave-speed estimates.
* Viscous fluxes: Newtonian stress with Stokes' hypothesis, Fourier conduction (constant Prandtl number),
  power-law viscosity mu = mu_ref * T^omega; normal derivatives compact at faces, tangential derivatives
  averaged from the two adjacent cells.
* SSP-RK2 (Heun) time integration, CFL + viscous time-step limit.

Non-dimensionalisation: reference density, reference sound speed and a length H. With p = rho*T/gamma the
reference state has rho = 1, T = 1, p = 1/gamma, c = 1; mu_ref = (rho u H / Re) for a chosen Mach.

Boundary conditions (per face): "supersonic_inflow" (Dirichlet on every primitive), "pressure_outlet"
(extrapolate rho and velocity, impose static pressure -- the fixed-pressure outlet), "extrapolate",
"wall" (no-slip, adiabatic), "slip" (inviscid wall / symmetry), "periodic".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["CompressibleConfig", "CompressibleFV", "sod_exact"]


@dataclass
class CompressibleConfig:
    shape: Tuple[int, ...]                  # interior cells per dimension, e.g. (nx, ny) or (nx, ny, nz)
    lengths: Tuple[float, ...]              # domain lengths
    gamma: float = 1.4
    Pr: float = 0.72
    mu_ref: float = 0.0                      # 0 -> Euler
    omega: float = 0.76                      # mu = mu_ref * T**omega
    cfl: float = 0.4
    limiter: str = "minmod"                  # "minmod" | "vanleer"
    bc: Dict[str, str] = field(default_factory=dict)   # keys "x-", "x+", "y-", "y+", ("z-", "z+")
    inflow: Optional[Sequence[float]] = None            # primitive state (rho, u..., p) for supersonic inflow
    p_out: float = 1.0 / 1.4                            # static pressure for "pressure_outlet"
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    rho_floor: float = 1e-6
    p_floor: float = 1e-6


def _minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))


def _vanleer(a, b):
    return torch.where(a * b > 0, 2 * a * b / (a + b + 1e-30), torch.zeros_like(a))


class CompressibleFV:
    def __init__(self, cfg: CompressibleConfig):
        self.cfg = c = cfg
        self.nd = len(c.shape)
        self.nv = self.nd + 2
        self.dx = [L / n for L, n in zip(c.lengths, c.shape)]
        self.g = c.gamma
        self.dev = torch.device(c.device)
        self.lim = _minmod if c.limiter == "minmod" else _vanleer
        self.W = torch.zeros((self.nv,) + tuple(n + 4 for n in c.shape), dtype=c.dtype, device=self.dev)
        self.t = 0.0
        if c.inflow is not None:
            self.inflow = torch.tensor(c.inflow, dtype=c.dtype, device=self.dev).view((self.nv,) + (1,) * self.nd)
        self.cp = 1.0 / (c.gamma - 1.0)              # with p = rho T / gamma  -> R = 1/gamma

    # ------------------------------------------------------------------ state conversion
    def interior(self):
        return (slice(None),) + tuple(slice(2, -2) for _ in range(self.nd))

    def prim_to_cons(self, W):
        rho, vel, p = W[0], W[1:1 + self.nd], W[-1]
        E = p / (self.g - 1) + 0.5 * rho * (vel ** 2).sum(0)
        return torch.cat([rho[None], rho[None] * vel, E[None]], 0)

    def cons_to_prim(self, U):
        rho = U[0].clamp(min=self.cfg.rho_floor)
        vel = U[1:1 + self.nd] / rho
        p = ((self.g - 1) * (U[-1] - 0.5 * rho * (vel ** 2).sum(0))).clamp(min=self.cfg.p_floor)
        return torch.cat([rho[None], vel, p[None]], 0)

    def set_state(self, W_interior):
        self.W[self.interior()] = torch.as_tensor(W_interior, dtype=self.cfg.dtype, device=self.dev)
        self.apply_bc(self.W)

    # ------------------------------------------------------------------ boundary conditions
    def _face(self, W, d, side, ghost):
        """Slice of ghost layer `ghost` (0 = adjacent, 1 = outer) on face (d, side)."""
        idx = [slice(None)] * (1 + self.nd)
        if side == "-":
            idx[1 + d] = 1 - ghost
        else:
            idx[1 + d] = -2 + ghost
        return tuple(idx)

    def _mirror_src(self, d, side, ghost):
        idx = [slice(None)] * (1 + self.nd)
        idx[1 + d] = (2 + ghost) if side == "-" else (-3 - ghost)
        return tuple(idx)

    def apply_bc(self, W):
        names = "xyz"
        for d in range(self.nd):
            for side in "-+":
                kind = self.cfg.bc.get(f"{names[d]}{side}", "extrapolate")
                for g in (0, 1):
                    dst, src = self._face(W, d, side, g), self._mirror_src(d, side, g)
                    if kind == "periodic":
                        idx = [slice(None)] * (1 + self.nd)
                        n = self.cfg.shape[d]
                        idx[1 + d] = (n + 1 - g) if side == "-" else (2 + g)   # wrap from the opposite interior
                        W[dst] = W[tuple(idx)]
                    elif kind == "supersonic_inflow":
                        W[dst] = self.inflow.reshape((self.nv,) + (1,) * (self.nd - 1)).expand_as(W[dst])
                    elif kind in ("wall", "slip"):
                        W[dst] = W[src]
                        if kind == "wall":
                            W[dst][1:1 + self.nd] = -W[src][1:1 + self.nd]
                        else:
                            W[dst][1 + d] = -W[src][1 + d]
                    elif kind == "pressure_outlet":
                        adj = self._mirror_src(d, side, 0)
                        W[dst] = W[adj]
                        # impose the static pressure on the boundary face (ghost = 2 p_out - interior), subsonic only
                        vel_n = W[adj][1 + d].abs()
                        c_ = torch.sqrt(self.g * W[adj][-1] / W[adj][0])
                        sub = vel_n < c_
                        W[dst][-1] = torch.where(sub, (2 * self.cfg.p_out - W[adj][-1]).clamp(min=self.cfg.p_floor), W[adj][-1])
                    else:  # extrapolate
                        W[dst] = W[self._mirror_src(d, side, 0)]
        return W

    # ------------------------------------------------------------------ fluxes
    def _phys_flux(self, W, d):
        rho, p = W[0], W[-1]
        vel = W[1:1 + self.nd]
        un = vel[d]
        E = p / (self.g - 1) + 0.5 * rho * (vel ** 2).sum(0)
        F = [rho * un] + [rho * un * vel[k] + (p if k == d else 0) for k in range(self.nd)] + [(E + p) * un]
        return torch.stack(F, 0), E

    def _hllc(self, WL, WR, d):
        g = self.g
        rL, rR, pL, pR = WL[0], WR[0], WL[-1], WR[-1]
        uL, uR = WL[1 + d], WR[1 + d]
        cL, cR = torch.sqrt(g * pL / rL), torch.sqrt(g * pR / rR)
        SL = torch.minimum(uL - cL, uR - cR)
        SR = torch.maximum(uL + cL, uR + cR)
        num = pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)
        den = rL * (SL - uL) - rR * (SR - uR)
        Ss = num / torch.where(den.abs() < 1e-14, torch.full_like(den, 1e-14), den)
        FL, EL = self._phys_flux(WL, d)
        FR, ER = self._phys_flux(WR, d)
        UL = torch.cat([rL[None], rL[None] * WL[1:1 + self.nd], EL[None]], 0)
        UR = torch.cat([rR[None], rR[None] * WR[1:1 + self.nd], ER[None]], 0)

        def ustar(r, u, W, S, E, p):
            fac = r * (S - u) / (S - Ss)
            comps = [fac]
            for k in range(self.nd):
                comps.append(fac * (Ss if k == d else W[1 + k]))
            comps.append(fac * (E / r + (Ss - u) * (Ss + p / (r * (S - u)))))
            return torch.stack(comps, 0)

        UsL, UsR = ustar(rL, uL, WL, SL, EL, pL), ustar(rR, uR, WR, SR, ER, pR)
        F = torch.where(SL >= 0, FL,
            torch.where(Ss >= 0, FL + SL * (UsL - UL),
            torch.where(SR > 0, FR + SR * (UsR - UR), FR)))
        return F

    def _slices_other(self, d, full_d=True):
        """Index that keeps interior cells in all dims except d (kept full with ghosts)."""
        idx = [slice(None)]
        for e in range(self.nd):
            idx.append(slice(None) if e == d else slice(2, -2))
        return tuple(idx)

    def _shift(self, A, d, a, b):
        idx = [slice(None)] * A.dim()
        idx[1 + d] = slice(a, b if b != 0 else None)
        return A[tuple(idx)]

    def rhs(self, W):
        """dU/dt on interior cells from a ghost-filled primitive array W."""
        c = self.cfg
        out = torch.zeros((self.nv,) + tuple(c.shape), dtype=c.dtype, device=self.dev)
        grads = self._cell_gradients(W) if c.mu_ref > 0 else None
        for d in range(self.nd):
            Q = W[self._slices_other(d)]
            dm = self._shift(Q, d, 1, -1) - self._shift(Q, d, 0, -2)
            dp = self._shift(Q, d, 2, 0) - self._shift(Q, d, 1, -1)
            s = self.lim(dm, dp)                                     # slopes for padded 1..n+2
            WL = self._shift(Q, d, 1, -2) + 0.5 * self._shift(s, d, 0, -1)
            WR = self._shift(Q, d, 2, -1) - 0.5 * self._shift(s, d, 1, 0)
            WL[0] = WL[0].clamp(min=c.rho_floor); WR[0] = WR[0].clamp(min=c.rho_floor)
            WL[-1] = WL[-1].clamp(min=c.p_floor); WR[-1] = WR[-1].clamp(min=c.p_floor)
            F = self._hllc(WL, WR, d)
            if c.mu_ref > 0:
                F = F - self._viscous_flux(W, grads, d)
            out -= (self._shift(F, d, 1, 0) - self._shift(F, d, 0, -1)) / self.dx[d]
        return out

    def _cell_gradients(self, W):
        """Central-difference gradients of (velocity, T) at every padded cell except the outermost layer."""
        vel = W[1:1 + self.nd]
        T = self.g * W[-1] / W[0]
        q = torch.cat([vel, T[None]], 0)                             # (nd+1, ...)
        grads = []
        for e in range(self.nd):
            gq = torch.zeros_like(q)
            idx_c = [slice(None)] * q.dim(); idx_p = list(idx_c); idx_m = list(idx_c)
            idx_c[1 + e] = slice(1, -1); idx_p[1 + e] = slice(2, None); idx_m[1 + e] = slice(0, -2)
            gq[tuple(idx_c)] = (q[tuple(idx_p)] - q[tuple(idx_m)]) / (2 * self.dx[e])
            grads.append(gq)
        return q, grads

    def _viscous_flux(self, W, qg, d):
        c = self.cfg
        q, grads = qg
        # face values between padded cells j and j+1 for j = 1..n+1 along d, interior in other dims
        sl = self._slices_other(d)
        qd = q[sl]
        qL, qR = self._shift(qd, d, 1, -2), self._shift(qd, d, 2, -1)
        qf = 0.5 * (qL + qR)
        dq = {}
        for e in range(self.nd):
            if e == d:
                dq[e] = (qR - qL) / self.dx[d]
            else:
                ge = grads[e][sl]
                dq[e] = 0.5 * (self._shift(ge, d, 1, -2) + self._shift(ge, d, 2, -1))
        T = qf[-1].clamp(min=1e-6)
        mu = c.mu_ref * T ** c.omega
        k = mu * self.cp / c.Pr
        div = sum(dq[e][e] for e in range(self.nd))
        tau = [mu * (dq[d][j] + dq[j][d]) - (2.0 / 3.0 if j == d else 0.0) * mu * div for j in range(self.nd)]
        energy = sum(qf[j] * tau[j] for j in range(self.nd)) + k * dq[d][-1]
        return torch.stack([torch.zeros_like(T)] + tau + [energy], 0)

    # ------------------------------------------------------------------ time stepping
    def dt(self):
        c = self.cfg
        W = self.W[self.interior()]
        cs = torch.sqrt(self.g * W[-1] / W[0])
        inv = sum((W[1 + d].abs() + cs) / self.dx[d] for d in range(self.nd))
        dt = c.cfl / float(inv.max())
        if c.mu_ref > 0:
            T = self.g * W[-1] / W[0]
            nu = float((c.mu_ref * T ** c.omega / W[0]).max()) * max(1.0, self.g / c.Pr)
            dt = min(dt, 0.25 / (nu * sum(1 / h ** 2 for h in self.dx)))
        return dt

    def step(self, dt=None):
        dt = dt or self.dt()
        I = self.interior()
        U0 = self.prim_to_cons(self.W[I])
        U1 = U0 + dt * self.rhs(self.W)
        W1 = self.W.clone()
        W1[I] = self.cons_to_prim(U1)
        self.apply_bc(W1)
        U2 = 0.5 * (U0 + U1 + dt * self.rhs(W1))
        self.W[I] = self.cons_to_prim(U2)
        self.apply_bc(self.W)
        self.t += dt
        return dt

    def mach(self):
        W = self.W[self.interior()]
        return torch.sqrt((W[1:1 + self.nd] ** 2).sum(0)) / torch.sqrt(self.g * W[-1] / W[0])


# ------------------------------------------------------------------ exact Riemann solution (Sod verification)
def sod_exact(x, t, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), gamma=1.4, x0=0.5):
    """Exact solution of the 1-D Riemann problem (Toro ch. 4); returns rho, u, p on x."""
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    g1, g2 = (gamma - 1) / (2 * gamma), (gamma + 1) / (2 * gamma)

    def f(p, r, pk, ck):
        if p > pk:
            A, B = 2 / ((gamma + 1) * r), (gamma - 1) / (gamma + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ck / (gamma - 1) * ((p / pk) ** g1 - 1)

    from scipy.optimize import brentq
    ps = brentq(lambda p: f(p, rL, pL, cL) + f(p, rR, pR, cR) + uR - uL, 1e-8, 10 * max(pL, pR))
    us = 0.5 * (uL + uR) + 0.5 * (f(ps, rR, pR, cR) - f(ps, rL, pL, cL))
    rsL = rL * (ps / pL) ** (1 / gamma)                                   # left rarefaction
    rsR = rR * ((ps / pR) + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * ps / pR + 1)   # right shock
    SR = uR + cR * np.sqrt(g2 * ps / pR + g1)
    csL = cL * (ps / pL) ** g1
    rho, u, p = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    for i, xi in enumerate(x):
        s = (xi - x0) / max(t, 1e-12)
        if s < uL - cL:
            rho[i], u[i], p[i] = rL, uL, pL
        elif s < us - csL:
            uu = 2 / (gamma + 1) * (cL + (gamma - 1) / 2 * uL + s)
            cc = 2 / (gamma + 1) * (cL + (gamma - 1) / 2 * (uL - s))
            rho[i], u[i], p[i] = rL * (cc / cL) ** (2 / (gamma - 1)), uu, pL * (cc / cL) ** (2 * gamma / (gamma - 1))
        elif s < us:
            rho[i], u[i], p[i] = rsL, us, ps
        elif s < SR:
            rho[i], u[i], p[i] = rsR, us, ps
        else:
            rho[i], u[i], p[i] = rR, uR, pR
    return rho, u, p
