"""Numba kernels for :class:`~pinneapple_physics.blackhole.hydro.AccretionFlow` (``RIAFConfig(backend="numba")``).

The same scheme as the PyTorch reference in ``hydro.py`` (MUSCL-minmod reconstruction, HLL fluxes, azimuthal viscous
stresses, exact geometric sources, SSP-RK2, caps and boundaries), written as fused loops. On the 128 x 64 grid the
PyTorch version is dominated by kernel-launch overhead (~10 ms per step on one core); this one runs the step in a
fraction of that, which is what makes simulations as long as those of Duarte et al. (2022) -- 5 x 10^5 GM/c^3 --
possible on a laptop. ``tests/test_blackhole_weather.py`` checks that both backends agree to round-off.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

NG = 2


@njit(cache=True, inline="always")
def _minmod(a, b):
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


@njit(cache=True, inline="always")
def _nu(rho, p, r, gamma, alpha, visc):
    # visc: 0 none, 1 SS, 2 ST
    if visc == 1:
        return alpha * gamma * p / rho * (r - 2.0) * math.sqrt(r)
    if visc == 2:
        return alpha * math.sqrt(r)
    return 0.0


@njit(cache=True, fastmath=True, inline="always")
def _hll(rl, vrl, vtl, vpl, pl, rr_, vrr, vtr, vpr, pr, d, Rf, gamma, out):
    """HLL flux of (rho, rho vr, rho vth, rho l, E) through a face normal to d (0 r, 1 theta)."""
    vnl = vrl if d == 0 else vtl
    vnr = vrr if d == 0 else vtr
    ll, lr = vpl * Rf, vpr * Rf
    El = pl / (gamma - 1.0) + 0.5 * rl * (vrl * vrl + vtl * vtl + vpl * vpl)
    Er = pr / (gamma - 1.0) + 0.5 * rr_ * (vrr * vrr + vtr * vtr + vpr * vpr)
    cl = math.sqrt(gamma * pl / rl)
    cr = math.sqrt(gamma * pr / rr_)
    SL = min(vnl - cl, vnr - cr)
    SR = max(vnl + cl, vnr + cr)
    if SL > 0.0:
        SL = 0.0
    if SR < 0.0:
        SR = 0.0
    # left and right conserved states and fluxes
    UL0, UL1, UL2, UL3, UL4 = rl, rl * vrl, rl * vtl, rl * ll, El
    UR0, UR1, UR2, UR3, UR4 = rr_, rr_ * vrr, rr_ * vtr, rr_ * lr, Er
    FL0 = rl * vnl
    FL1 = rl * vrl * vnl + (pl if d == 0 else 0.0)
    FL2 = rl * vtl * vnl + (pl if d == 1 else 0.0)
    FL3 = rl * ll * vnl
    FL4 = (El + pl) * vnl
    FR0 = rr_ * vnr
    FR1 = rr_ * vrr * vnr + (pr if d == 0 else 0.0)
    FR2 = rr_ * vtr * vnr + (pr if d == 1 else 0.0)
    FR3 = rr_ * lr * vnr
    FR4 = (Er + pr) * vnr
    den = SR - SL + 1e-300
    out[0] = (SR * FL0 - SL * FR0 + SL * SR * (UR0 - UL0)) / den
    out[1] = (SR * FL1 - SL * FR1 + SL * SR * (UR1 - UL1)) / den
    out[2] = (SR * FL2 - SL * FR2 + SL * SR * (UR2 - UL2)) / den
    out[3] = (SR * FL3 - SL * FR3 + SL * SR * (UR3 - UL3)) / den
    out[4] = (SR * FL4 - SL * FR4 + SL * SR * (UR4 - UL4)) / den


@njit(cache=True, fastmath=True)
def rhs(W, r, th, rf, thf, area_r, area_t, vol, inv_r, geo_t, Rc, gamma, alpha, visc, dU, fluxes):
    """dU on interior cells (5, nr, nth) and boundary fluxes [mass_in, mass_out, angmom_in, angmom_out]."""
    N, M = W.shape[1], W.shape[2]
    nr, nt = N - 2 * NG, M - 2 * NG
    Fr = np.zeros((5, nr + 1, nt))
    Ft = np.zeros((5, nr, nt + 1))
    f = np.zeros(5)
    WL = np.zeros(5)
    WRs = np.zeros(5)
    visc_on = visc != 0 and alpha > 0.0
    RN = np.zeros((N, M))                       # rho * nu per cell, once
    if visc_on:
        for i in range(N):
            for j in range(M):
                RN[i, j] = W[0, i, j] * _nu(W[0, i, j], W[4, i, j], r[i], gamma, alpha, visc)
    # ---- r faces: face index a (0..nr) is the left face of interior cell i = a + NG
    for a in range(nr + 1):
        i = a + NG
        for jj in range(nt):
            j = jj + NG
            for q in range(5):
                WL[q] = W[q, i - 1, j] + 0.5 * _minmod(W[q, i - 1, j] - W[q, i - 2, j], W[q, i, j] - W[q, i - 1, j])
                WRs[q] = W[q, i, j] - 0.5 * _minmod(W[q, i, j] - W[q, i - 1, j], W[q, i + 1, j] - W[q, i, j])
            s = math.sin(th[j])
            Rf = rf[i] * s
            _hll(WL[0], WL[1], WL[2], WL[3], WL[4], WRs[0], WRs[1], WRs[2], WRs[3], WRs[4], 0, Rf, gamma, f)
            for q in range(5):
                Fr[q, a, jj] = f[q]
            if visc_on:
                Rm, Rp = abs(r[i - 1] * s), abs(r[i] * s)
                Om_m = W[3, i - 1, j] / max(Rm, 1e-12)
                Om_p = W[3, i, j] / max(Rp, 1e-12)
                rn = 0.5 * (RN[i - 1, j] + RN[i, j])
                T = rn * rf[i] * s * (Om_p - Om_m) / (r[i] - r[i - 1])
                Fr[3, a, jj] -= rf[i] * s * T
                Fr[4, a, jj] -= 0.5 * (W[3, i - 1, j] + W[3, i, j]) * T
    # ---- theta faces: face index b (0..nt) is the lower face of interior cell j = b + NG
    for ii in range(nr):
        i = ii + NG
        for b in range(nt + 1):
            j = b + NG
            for q in range(5):
                WL[q] = W[q, i, j - 1] + 0.5 * _minmod(W[q, i, j - 1] - W[q, i, j - 2], W[q, i, j] - W[q, i, j - 1])
                WRs[q] = W[q, i, j] - 0.5 * _minmod(W[q, i, j] - W[q, i, j - 1], W[q, i, j + 1] - W[q, i, j])
            sf = math.sin(thf[j])
            Rf = r[i] * sf
            _hll(WL[0], WL[1], WL[2], WL[3], WL[4], WRs[0], WRs[1], WRs[2], WRs[3], WRs[4], 1, Rf, gamma, f)
            for q in range(5):
                Ft[q, ii, b] = f[q]
            if visc_on:
                Rm, Rp = abs(Rc[i, j - 1]), abs(Rc[i, j])
                Om_m = W[3, i, j - 1] / max(Rm, 1e-12)
                Om_p = W[3, i, j] / max(Rp, 1e-12)
                rn = 0.5 * (RN[i, j - 1] + RN[i, j])
                T = rn * sf * (Om_p - Om_m) / (th[j] - th[j - 1])
                Ft[3, ii, b] -= r[i] * sf * T
                Ft[4, ii, b] -= 0.5 * (W[3, i, j - 1] + W[3, i, j]) * T
    # ---- divergence and sources
    m_in = m_out = l_in = l_out = 0.0
    for ii in range(nr):
        i = ii + NG
        g_r = 1.0 / (r[i] - 2.0) ** 2
        for jj in range(nt):
            j = jj + NG
            V = vol[i, j]
            Ar0, Ar1 = area_r[i, j], area_r[i + 1, j]
            At0, At1 = area_t[i, j], area_t[i, j + 1]
            rho, vr, vth, vph, p = W[0, i, j], W[1, i, j], W[2, i, j], W[3, i, j], W[4, i, j]
            for q in range(5):
                dU[q, ii, jj] = -((Ar1 * Fr[q, ii + 1, jj] - Ar0 * Fr[q, ii, jj]) +
                                  (At1 * Ft[q, ii, jj + 1] - At0 * Ft[q, ii, jj])) / V
            cot = math.cos(th[j]) / math.sin(th[j])
            ir = inv_r[i]
            dU[1, ii, jj] += ir * (rho * (vth * vth + vph * vph) + 2.0 * p) - rho * g_r
            dU[2, ii, jj] += ir * (p * geo_t[j] + rho * vph * vph * cot - rho * vr * vth)
            dU[4, ii, jj] += -rho * vr * g_r
    for jj in range(nt):
        j = jj + NG
        m_in -= area_r[NG, j] * Fr[0, 0, jj]
        m_out += area_r[NG + nr, j] * Fr[0, nr, jj]
        l_in -= area_r[NG, j] * Fr[3, 0, jj]
        l_out += area_r[NG + nr, j] * Fr[3, nr, jj]
    fluxes[0], fluxes[1], fluxes[2], fluxes[3] = m_in, m_out, l_in, l_out


@njit(cache=True)
def prim_to_cons(W, Rc, gamma, U):
    """Interior primitives of W (with ghosts) -> conserved U (5, nr, nth)."""
    nr, nt = U.shape[1], U.shape[2]
    for ii in range(nr):
        i = ii + NG
        for jj in range(nt):
            j = jj + NG
            rho, vr, vth, vph, p = W[0, i, j], W[1, i, j], W[2, i, j], W[3, i, j], W[4, i, j]
            U[0, ii, jj] = rho
            U[1, ii, jj] = rho * vr
            U[2, ii, jj] = rho * vth
            U[3, ii, jj] = rho * vph * Rc[i, j]
            U[4, ii, jj] = p / (gamma - 1.0) + 0.5 * rho * (vr * vr + vth * vth + vph * vph)


@njit(cache=True)
def cons_to_prim(U, W, Rc, r, gamma, rho_floor, p_floor, v_max, t_cap):
    """Conserved U -> interior primitives of W, with floors and caps; returns the number of capped cells."""
    nr, nt = U.shape[1], U.shape[2]
    capped = 0
    for ii in range(nr):
        i = ii + NG
        for jj in range(nt):
            j = jj + NG
            rho = max(U[0, ii, jj], rho_floor)
            vr = U[1, ii, jj] / rho
            vth = U[2, ii, jj] / rho
            vph = U[3, ii, jj] / (rho * Rc[i, j])
            vm = math.sqrt(vr * vr + vth * vth + vph * vph)
            if vm > v_max:
                sc = v_max / vm
                vr *= sc
                vth *= sc
                vph *= sc
                capped += 1
            ek = 0.5 * rho * (vr * vr + vth * vth + vph * vph)
            p = max((gamma - 1.0) * (U[4, ii, jj] - ek), p_floor)
            if t_cap > 0.0:
                pmax = t_cap * rho / (gamma * (r[i] - 2.0))
                if p > pmax:
                    p = pmax
                    capped += 1
            W[0, i, j], W[1, i, j], W[2, i, j], W[3, i, j], W[4, i, j] = rho, vr, vth, vph, p
    return capped


@njit(cache=True)
def boundaries(W, outer, has_outer):
    N, M = W.shape[1], W.shape[2]
    for k in range(NG):
        for q in range(5):
            for j in range(M):
                W[q, k, j] = W[q, NG, j]
                W[q, N - 1 - k, j] = W[q, N - 1 - NG, j]
    for k in range(NG):
        for j in range(M):
            if W[1, k, j] > 0.0:
                W[1, k, j] = 0.0
    if has_outer:
        for q in range(5):
            for k in range(NG):
                for j in range(M):
                    W[q, N - NG + k, j] = outer[q, k, j]
    else:
        for k in range(NG):
            for j in range(M):
                if W[1, N - NG + k, j] < 0.0:
                    W[1, N - NG + k, j] = 0.0
    for k in range(NG):
        for q in range(5):
            for i in range(N):
                W[q, i, NG - 1 - k] = W[q, i, NG + k]
                W[q, i, M - NG + k] = W[q, i, M - NG - 1 - k]
    for i in range(N):
        for k in range(NG):
            W[2, i, k] = -W[2, i, k]
            W[2, i, M - NG + k] = -W[2, i, M - NG + k]


@njit(cache=True)
def time_step(W, r, dr, dth, gamma, alpha, visc, cfl):
    N, M = W.shape[1], W.shape[2]
    dt_h = 1e300
    dt_v = 1e300
    for i in range(NG, N - NG):
        rdt = r[i] * dth
        for j in range(NG, M - NG):
            cs = math.sqrt(gamma * W[4, i, j] / W[0, i, j])
            a = dr[i] / (abs(W[1, i, j]) + cs)
            b = rdt / (abs(W[2, i, j]) + cs)
            if a < dt_h:
                dt_h = a
            if b < dt_h:
                dt_h = b
            if visc != 0 and alpha > 0.0:
                nu = _nu(W[0, i, j], W[4, i, j], r[i], gamma, alpha, visc)
                if nu > 0.0:
                    dm = min(dr[i], rdt)
                    v = 0.25 * dm * dm / nu
                    if v < dt_v:
                        dt_v = v
    return min(cfl * dt_h, dt_v)


class NumbaStepper:
    """Holds the numpy geometry of an AccretionFlow and advances its state with the numba kernels."""

    def __init__(self, flow):
        c = flow.cfg
        g = NG
        np_ = lambda t: np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)  # noqa: E731
        self.flow = flow
        self.r = np_(flow.r[:, 0])
        self.th = np_(flow.th[0])
        self.rf = np.asarray(flow.rf_np, np.float64)
        self.thf = np.asarray(flow.thf_np, np.float64)
        self.area_r = np_(flow.area_r)
        self.area_t = np_(flow.area_t)
        self.vol = np_(flow.vol)
        self.inv_r = np_(flow.inv_r[:, 0])
        self.geo_t = np_(flow.geo_t[0])
        self.Rc = np_(flow.R)
        self.dr = np_(flow.dr_c[:, 0])
        self.dth = math.pi / c.ntheta
        self.visc = {"none": 0, "SS": 1, "ST": 2}[c.viscosity]
        self.U0 = np.zeros((5, c.nr, c.ntheta))
        self.U1 = np.zeros_like(self.U0)
        self.dU = np.zeros_like(self.U0)
        self.fl1 = np.zeros(4)
        self.fl2 = np.zeros(4)
        self.g = g

    def W(self):
        return self.flow.W.numpy()          # shares memory with the torch tensor (CPU, float64)

    def dt(self):
        c = self.flow.cfg
        return time_step(self.W(), self.r, self.dr, self.dth, c.gamma, c.alpha, self.visc, c.cfl)

    def _rhs(self, W, fl):
        c = self.flow.cfg
        rhs(W, self.r, self.th, self.rf, self.thf, self.area_r, self.area_t, self.vol, self.inv_r, self.geo_t,
            self.Rc, c.gamma, c.alpha, self.visc, self.dU, fl)

    def step(self, dt):
        c = self.flow.cfg
        W = self.W()
        outer = self.flow._outer
        has_outer = outer is not None
        outer_np = outer.numpy() if has_outer else np.zeros((5, NG, W.shape[2]))
        tcap = -1.0 if c.t_cap is None else float(c.t_cap)
        prim_to_cons(W, self.Rc, c.gamma, self.U0)
        self._rhs(W, self.fl1)
        self.U1[:] = self.U0 + dt * self.dU
        cons_to_prim(self.U1, W, self.Rc, self.r, c.gamma, c.rho_floor, c.p_floor, c.v_max, tcap)
        boundaries(W, outer_np, has_outer)
        self._rhs(W, self.fl2)
        self.U1[:] = 0.5 * (self.U0 + self.U1 + dt * self.dU)
        capped = cons_to_prim(self.U1, W, self.Rc, self.r, c.gamma, c.rho_floor, c.p_floor, c.v_max, tcap)
        boundaries(W, outer_np, has_outer)
        return self.fl1, self.fl2, capped
