"""Transient 2-D heated-channel CFD: incompressible Navier-Stokes + energy, with time-varying
actuators (inlet velocity U_in(t), heater power Q_h(t)).

Non-dimensional problem (length H, velocity U_ref, temperature q_ref*H/k)::

    du/dt + (u.grad)u = -grad p + (1/Re) lap u,     div u = 0
    dT/dt + u.grad T  = (1/Pe) lap T,                Pe = Re*Pr

Domain [0, L] x [0, 1]. Inlet (x=0): parabolic u = 6 U_in(t) y(1-y), v = 0, T = 0.
Walls (y=0,1): no slip; adiabatic except the heater segment x in [x_h0, x_h1] on the bottom
wall, where -dT/dy = Q_h(t). Outlet (x=L): zero-gradient u, v, T with a global mass
correction, p = 0.

Discretisation: staggered MAC grid, explicit time stepping with the donor-cell/central blend
(parameter gamma) for convection and a Chorin projection for pressure (Griebel, Dornseifer &
Neunhoeffer, "Numerical Simulation in Fluid Dynamics", SIAM 1998, ch. 3 and 9). The
pressure-Poisson matrix is factorised once (sparse LU) and reused every step.

Validated in the accompanying experiment against plane Poiseuille flow (velocity profile,
dp/dx = -12/Re) and the fully developed Nusselt number for parallel plates with one wall at
uniform flux and the other adiabatic (Nu_Dh = 5.385, Shah & London 1978), plus a
three-grid Richardson/GCI study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["ChannelConfig", "ThermalChannel2D"]


@dataclass
class ChannelConfig:
    nx: int = 128
    ny: int = 32
    L: float = 10.0
    Re: float = 100.0
    Pr: float = 0.71
    heater: tuple = (2.0, 4.0)
    dt: float = 0.01
    gamma_safety: float = 1.2
    record_every: float = 0.2


class ThermalChannel2D:
    def __init__(self, cfg: Optional[ChannelConfig] = None):
        self.cfg = c = cfg or ChannelConfig()
        self.dx, self.dy = c.L / c.nx, 1.0 / c.ny
        self.xc = (np.arange(c.nx) + 0.5) * self.dx
        self.yc = (np.arange(c.ny) + 0.5) * self.dy
        self.heater_mask = (self.xc >= c.heater[0]) & (self.xc <= c.heater[1])
        self.Pe = c.Re * c.Pr
        self._factor_poisson()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _factor_poisson(self):
        c, dx, dy = self.cfg, self.dx, self.dy
        nx, ny = c.nx, c.ny
        N = nx * ny
        idx = lambda i, j: i * ny + j
        rows, cols, vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                d = 0.0
                k = idx(i, j)
                if i > 0:
                    rows.append(k); cols.append(idx(i - 1, j)); vals.append(1 / dx ** 2); d -= 1 / dx ** 2
                # inlet: Neumann (ghost = interior) -> no contribution
                if i < nx - 1:
                    rows.append(k); cols.append(idx(i + 1, j)); vals.append(1 / dx ** 2); d -= 1 / dx ** 2
                else:
                    d -= 2 / dx ** 2  # outlet Dirichlet p=0 on the face: ghost = -p
                if j > 0:
                    rows.append(k); cols.append(idx(i, j - 1)); vals.append(1 / dy ** 2); d -= 1 / dy ** 2
                if j < ny - 1:
                    rows.append(k); cols.append(idx(i, j + 1)); vals.append(1 / dy ** 2); d -= 1 / dy ** 2
                rows.append(k); cols.append(k); vals.append(d)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        self._lu = spla.splu(A)

    def inlet_profile(self, U):
        y = self.yc
        return 6.0 * U * y * (1 - y)

    def reset(self, U0: float = 1.0, Q0: float = 0.0, fully_developed: bool = True):
        c = self.cfg
        nx, ny = c.nx, c.ny
        self.u = np.zeros((nx + 1, ny + 2))   # includes ghost rows j=0, ny+1
        self.v = np.zeros((nx + 2, ny + 1))   # includes ghost cols i=0, nx+1
        self.p = np.zeros((nx + 2, ny + 2))
        self.T = np.zeros((nx + 2, ny + 2))
        if fully_developed:
            self.u[:, 1:-1] = self.inlet_profile(U0)[None, :]
            self.p[1:-1, 1:-1] = (12.0 / c.Re) * U0 * (c.L - self.xc)[:, None]
        self.t = 0.0
        self.U, self.Q = U0, Q0
        self._apply_bc()

    # ------------------------------------------------------------------ boundary conditions
    def _apply_bc(self):
        u, v, T = self.u, self.v, self.T
        dy = self.dy
        u[0, 1:-1] = self.inlet_profile(self.U)
        u[-1, 1:-1] = u[-2, 1:-1]
        qin = u[0, 1:-1].sum()
        qout = u[-1, 1:-1].sum()
        if abs(qout) > 1e-12:
            u[-1, 1:-1] *= qin / qout           # global mass conservation at the outlet
        u[:, 0] = -u[:, 1]                       # no-slip bottom
        u[:, -1] = -u[:, -2]                     # no-slip top
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[0, :] = -v[1, :]                       # v = 0 on the inlet face
        v[-1, :] = v[-2, :]                      # zero gradient at outlet
        T[0, :] = -T[1, :]                       # T = 0 at inlet
        T[-1, :] = T[-2, :]
        T[:, -1] = T[:, -2]                      # adiabatic top
        T[:, 0] = T[:, 1]                        # adiabatic bottom ...
        T[1:-1, 0][self.heater_mask] = T[1:-1, 1][self.heater_mask] + self.Q * dy  # ... except heater: -dT/dy = Q

    # ------------------------------------------------------------------ one step
    def step(self, U: float, Q: float):
        c, dx, dy, dt = self.cfg, self.dx, self.dy, self.cfg.dt
        self.U, self.Q = float(U), float(Q)
        self._apply_bc()
        u, v, T = self.u, self.v, self.T
        umax = max(np.abs(u).max(), 1e-12)
        vmax = max(np.abs(v).max(), 1e-12)
        g = min(1.0, c.gamma_safety * max(umax * dt / dx, vmax * dt / dy))
        Re = c.Re

        # ---- F (x-momentum) on interior u faces i=1..nx-1, j=1..ny
        uc = u[1:-1, 1:-1]
        ue, uw = u[2:, 1:-1], u[:-2, 1:-1]
        un, us = u[1:-1, 2:], u[1:-1, :-2]
        lap_u = (ue - 2 * uc + uw) / dx ** 2 + (un - 2 * uc + us) / dy ** 2
        du2dx = (((uc + ue) / 2) ** 2 - ((uw + uc) / 2) ** 2) / dx + g / dx * (
            np.abs(uc + ue) / 2 * (uc - ue) / 2 - np.abs(uw + uc) / 2 * (uw - uc) / 2)
        vn = (v[1:-2, 1:] + v[2:-1, 1:]) / 2          # v at (i+1/2, j) north face of u-cell
        vs = (v[1:-2, :-1] + v[2:-1, :-1]) / 2
        duvdy = (vn * (uc + un) / 2 - vs * (us + uc) / 2) / dy + g / dy * (
            np.abs(vn) * (uc - un) / 2 - np.abs(vs) * (us - uc) / 2)
        F = u.copy()
        F[1:-1, 1:-1] = uc + dt * (lap_u / Re - du2dx - duvdy)

        # ---- G (y-momentum) on interior v faces i=1..nx, j=1..ny-1
        vc = v[1:-1, 1:-1]
        ve, vw = v[2:, 1:-1], v[:-2, 1:-1]
        vN, vS = v[1:-1, 2:], v[1:-1, :-2]
        lap_v = (ve - 2 * vc + vw) / dx ** 2 + (vN - 2 * vc + vS) / dy ** 2
        dv2dy = (((vc + vN) / 2) ** 2 - ((vS + vc) / 2) ** 2) / dy + g / dy * (
            np.abs(vc + vN) / 2 * (vc - vN) / 2 - np.abs(vS + vc) / 2 * (vS - vc) / 2)
        ue_ = (u[1:, 1:-2] + u[1:, 2:-1]) / 2          # u at east face of v-cell
        uw_ = (u[:-1, 1:-2] + u[:-1, 2:-1]) / 2
        duvdx = (ue_ * (vc + ve) / 2 - uw_ * (vw + vc) / 2) / dx + g / dx * (
            np.abs(ue_) * (vc - ve) / 2 - np.abs(uw_) * (vw - vc) / 2)
        G = v.copy()
        G[1:-1, 1:-1] = vc + dt * (lap_v / Re - dv2dy - duvdx)

        # ---- temperature (explicit, same donor-cell blend), uses old velocities
        Tc = T[1:-1, 1:-1]
        TE, TW, TN, TS = T[2:, 1:-1], T[:-2, 1:-1], T[1:-1, 2:], T[1:-1, :-2]
        uR, uL = u[1:, 1:-1], u[:-1, 1:-1]
        vT, vB = v[1:-1, 1:], v[1:-1, :-1]
        duTdx = (uR * (Tc + TE) / 2 - uL * (TW + Tc) / 2) / dx + g / dx * (
            np.abs(uR) * (Tc - TE) / 2 - np.abs(uL) * (TW - Tc) / 2)
        dvTdy = (vT * (Tc + TN) / 2 - vB * (TS + Tc) / 2) / dy + g / dy * (
            np.abs(vT) * (Tc - TN) / 2 - np.abs(vB) * (TS - Tc) / 2)
        lapT = (TE - 2 * Tc + TW) / dx ** 2 + (TN - 2 * Tc + TS) / dy ** 2
        T[1:-1, 1:-1] = Tc + dt * (lapT / self.Pe - duTdx - dvTdy)

        # ---- pressure Poisson and projection
        rhs = ((F[1:, 1:-1] - F[:-1, 1:-1]) / dx + (G[1:-1, 1:] - G[1:-1, :-1]) / dy) / dt
        p = self._lu.solve(rhs.ravel()).reshape(c.nx, c.ny)
        P = self.p
        P[1:-1, 1:-1] = p
        P[0, :] = P[1, :]
        P[-1, :] = -P[-2, :]
        P[:, 0] = P[:, 1]
        P[:, -1] = P[:, -2]
        u[1:-1, 1:-1] = F[1:-1, 1:-1] - dt / dx * (P[2:-1, 1:-1] - P[1:-2, 1:-1])
        v[1:-1, 1:-1] = G[1:-1, 1:-1] - dt / dy * (P[1:-1, 2:-1] - P[1:-1, 1:-2])
        self.t += dt
        self._apply_bc()

    # ------------------------------------------------------------------ observables
    def fields(self) -> Dict[str, np.ndarray]:
        """Cell-centred u, v, p, T, each (nx, ny)."""
        return {"u": 0.5 * (self.u[1:, 1:-1] + self.u[:-1, 1:-1]),
                "v": 0.5 * (self.v[1:-1, 1:] + self.v[1:-1, :-1]),
                "p": self.p[1:-1, 1:-1].copy(), "T": self.T[1:-1, 1:-1].copy()}

    def qoi(self) -> Dict[str, float]:
        uo, To = self.u[-1, 1:-1], 0.5 * (self.T[-1, 1:-1] + self.T[-2, 1:-1])
        Tw = self.T[1:-1, 1][self.heater_mask] + 0.5 * self.Q * self.dy   # heater wall temperature
        return {"T_out": float((uo * To).sum() / max(uo.sum(), 1e-12)),
                "T_max": float(max(self.T[1:-1, 1:-1].max(), Tw.max() if Tw.size else -np.inf)),
                "dp": float(self.p[1, 1:-1].mean() - 0.5 * (self.p[-2, 1:-1] + self.p[-1, 1:-1]).mean()),
                "heat_in": float(self.Q * self.heater_mask.sum() * self.dx / self.Pe),
                "heat_out": float((uo * To).sum() * self.dy),
                "energy": float(self.T[1:-1, 1:-1].sum() * self.dx * self.dy)}

    def divergence(self) -> float:
        d = (self.u[1:, 1:-1] - self.u[:-1, 1:-1]) / self.dx + (self.v[1:-1, 1:] - self.v[1:-1, :-1]) / self.dy
        return float(np.abs(d).max())

    def run(self, U_fn: Callable[[float], float], Q_fn: Callable[[float], float], t_end: float,
            record: bool = True):
        """Integrate to t_end; returns dict with t, U, Q, fields (T_rec, 4, nx, ny) and QoIs."""
        c = self.cfg
        n_steps = int(round(t_end / c.dt))
        every = max(1, int(round(c.record_every / c.dt)))
        rec = {"t": [], "U": [], "Q": [], "fields": [], "qoi": []}
        for k in range(n_steps + 1):
            if record and k % every == 0:
                f = self.fields()
                rec["t"].append(self.t); rec["U"].append(self.U); rec["Q"].append(self.Q)
                rec["fields"].append(np.stack([f["u"], f["v"], f["p"], f["T"]]).astype(np.float32))
                rec["qoi"].append(self.qoi())
            if k < n_steps:
                self.step(U_fn(self.t), Q_fn(self.t))
                if not np.isfinite(self.u).all():
                    raise FloatingPointError(f"solution blew up at t={self.t:.3f}")
        rec["fields"] = np.stack(rec["fields"]) if rec["fields"] else None
        return rec
