"""Viscous hydrodynamics of accretion around a Schwarzschild black hole, axisymmetric (r, theta), PyTorch.

The set-up of the simulations behind Duarte, Nemmen & Navarro (2022, MNRAS 512, 5848), "Black hole weather
forecasting with deep learning", which were run with PLUTO (Almeida & Nemmen 2020): a radiatively inefficient
accretion flow (RIAF) that grows from a hot torus of gas around the hole.

Physics
-------
* Units G = M = c = 1: lengths in GM/c^2, times in GM/c^3 ("M"). Schwarzschild radius r_s = 2.
* Paczynski-Wiita potential Phi = -1 / (r - 2): Newtonian hydro that reproduces the innermost stable orbit
  (r = 6) and the marginally bound orbit (r = 4) of the Schwarzschild metric.
* Adiabatic gas (gamma = 5/3), no cooling (radiatively inefficient), total energy evolved, so viscous heating
  stays in the gas.
* Shear viscosity with only the azimuthal stress components T_rphi and T_thetaphi, the usual choice for
  alpha-viscosity RIAF simulations (Stone, Pringle & Begelman 1999), with two prescriptions:
  ``"SS"`` (Shakura-Sunyaev) nu = alpha c_s^2 / Omega_K and ``"ST"`` (Stone et al. K-model) nu = alpha r^(1/2).
* Initial condition: an equilibrium torus with a power-law specific angular momentum l(R) = l_c (R/R_c)^a,
  in a cold, tenuous atmosphere.

Numerics
--------
Finite volume on a log-spaced r grid and a uniform theta grid in [0, pi]; MUSCL (minmod) reconstruction of
primitive variables, HLL fluxes, SSP-RK2. The conserved variables are (rho, rho v_r, rho v_theta, rho l, E)
with l = r sin(theta) v_phi, so angular momentum is conserved to round-off: its only sources are the fluxes
through the radial boundaries. Exact geometric source terms (cell-averaged 1/r and the theta-face area
difference) keep a uniform pressure in balance to round-off.

Boundaries: outflow at r_in and r_out (no inflow: v_r <= 0 at r_in, v_r >= 0 at r_out), reflecting at the
poles. The mass and angular momentum that leave through each radial boundary are accumulated, so the global
budgets close (``budget()``), which the tests check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = ["RIAFConfig", "AccretionFlow", "torus_state", "bondi_pw"]


@dataclass
class RIAFConfig:
    nr: int = 128
    ntheta: int = 64
    r_in: float = 4.0                 # 2 Schwarzschild radii
    r_out: float = 400.0              # 200 Schwarzschild radii
    gamma: float = 5.0 / 3.0
    alpha: float = 0.1
    viscosity: str = "SS"             # "SS" | "ST" | "none"
    cfl: float = 0.3
    rho_floor: float = 1e-7
    p_floor: float = 1e-11
    v_max: float = 0.9                # speed cap (units of c) for near-empty cells
    t_cap: Optional[float] = 2.0      # temperature ceiling c_s^2 <= t_cap |Phi| (None: off)
    outer_bc: str = "outflow"         # "outflow" (no inflow) | "fixed" (ghosts keep their initial state, e.g. Bondi)
    # torus (pressure maximum at R_c on the equator, inner edge at R_edge, l = l_c (R / R_c)**a)
    torus_rc: float = 20.0
    torus_edge: float = 12.0                # outer edge ~58 for a = 0
    torus_a: float = 0.0
    rho_atm: float = 1e-4             # atmosphere density at r_in (falls as r**-1.5), relative to torus max
    perturbation: float = 0.01        # relative random density noise in the torus (breaks symmetry)
    seed: int = 0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def name(self) -> str:
        return f"PL{self.torus_a:g}{self.viscosity}{self.alpha:g}"


def _phi(r):
    return -1.0 / (r - 2.0)


def _dphi(r):
    return 1.0 / (r - 2.0) ** 2


def _omega_k(r):
    return 1.0 / ((r - 2.0) * torch.sqrt(r) if torch.is_tensor(r) else (r - 2.0) * math.sqrt(r))


def _minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))


class AccretionFlow:
    """2.5-D (axisymmetric, with v_phi) viscous accretion flow; ``run`` saves snapshots of the primitives."""

    NG = 2

    def __init__(self, cfg: RIAFConfig):
        self.cfg = c = cfg
        dt_, dev = c.dtype, torch.device(c.device)
        self.kw = dict(dtype=dt_, device=dev)
        g = self.NG
        # faces including ghosts: r log-spaced, theta uniform (ghost faces continue the spacing)
        dlr = math.log(c.r_out / c.r_in) / c.nr
        rf = c.r_in * np.exp(dlr * np.arange(-g, c.nr + g + 1))
        dth = math.pi / c.ntheta
        thf = dth * np.arange(-g, c.ntheta + g + 1)
        self.rf_np, self.thf_np = rf, thf
        rf_t, thf_t = torch.tensor(rf, **self.kw), torch.tensor(thf, **self.kw)
        r0, r1 = rf_t[:-1], rf_t[1:]
        t0, t1 = thf_t[:-1], thf_t[1:]
        # cell centres: volume-weighted r, mid theta
        self.r = (0.75 * (r1 ** 4 - r0 ** 4) / (r1 ** 3 - r0 ** 3))[:, None]
        self.th = (0.5 * (t0 + t1))[None, :]
        self.inv_r = (1.5 * (r1 ** 2 - r0 ** 2) / (r1 ** 3 - r0 ** 3))[:, None]          # <1/r> over the cell
        self.dcos = (torch.cos(t0) - torch.cos(t1))[None, :]                            # > 0 inside [0, pi]
        self.vol = ((r1 ** 3 - r0 ** 3) / 3.0)[:, None] * self.dcos                    # / (2 pi)
        self.area_r = (rf_t ** 2)[:, None] * self.dcos                                  # r faces (nr+2g+1, nth+2g)
        self.area_t = (0.5 * (r1 ** 2 - r0 ** 2))[:, None] * torch.sin(thf_t)[None, :].abs()  # theta faces
        self.geo_t = (torch.sin(t1) - torch.sin(t0))[None, :] / self.dcos                # pressure source in theta
        self.sin = torch.sin(self.th)
        self.R = self.r * self.sin                                                     # cylindrical radius
        self.rf_c = rf_t[:, None]                                                      # r at r-faces
        self.thf_c = thf_t[None, :]
        self.dr_c = (r1 - r0)[:, None]
        self.W = torch.zeros((5, c.nr + 2 * g, c.ntheta + 2 * g), **self.kw)          # rho, vr, vth, vphi, p
        self.t = 0.0
        self.boundary = {"mass_in": 0.0, "mass_out": 0.0, "angmom_in": 0.0, "angmom_out": 0.0}
        self.mdot_in = 0.0                                                             # latest accretion rate
        self.capped = 0
        self.ii = (slice(None), slice(g, -g), slice(g, -g))

    # ------------------------------------------------------------------ state
    def interior(self, A: torch.Tensor) -> torch.Tensor:
        return A[..., self.NG:-self.NG, self.NG:-self.NG]

    def set_primitives(self, W_int) -> None:
        self.W[self.ii] = torch.as_tensor(W_int, **self.kw)
        self._outer = None
        if self.cfg.outer_bc == "fixed":                # hold the outer ghosts at the initial outermost state
            g = self.NG
            self.W[:, -g:, :] = self.W[:, -g - 1:-g, :]
            self._outer = self.W[:, -g:, :].clone()
        self._bc(self.W)
        if self._outer is not None:
            self._outer = self.W[:, -self.NG:, :].clone()   # with the pole ghosts filled in

    def prim_to_cons(self, W, R=None):
        rho, vr, vth, vph, p = W
        R = self.R if R is None else R
        return torch.stack([rho, rho * vr, rho * vth, rho * vph * R,
                            p / (self.cfg.gamma - 1) + 0.5 * rho * (vr ** 2 + vth ** 2 + vph ** 2)])

    def cons_to_prim(self, U):
        c = self.cfg
        rho = U[0].clamp(min=c.rho_floor)
        vr, vth = U[1] / rho, U[2] / rho
        vph = U[3] / (rho * self.R)
        ek = 0.5 * rho * (vr ** 2 + vth ** 2 + vph ** 2)
        p = ((c.gamma - 1) * (U[4] - ek)).clamp(min=c.p_floor)
        return torch.stack([rho, vr, vth, vph, p])

    # ------------------------------------------------------------------ boundaries
    def _bc(self, W):
        g = self.NG
        for k in range(g):
            W[:, k, :] = W[:, g, :]                      # inner r: zero gradient, no inflow
            W[:, -1 - k, :] = W[:, -1 - g, :]            # outer r
        W[1, :g, :] = W[1, :g, :].clamp(max=0.0)
        if getattr(self, "_outer", None) is not None:
            W[:, -g:, :] = self._outer
        else:
            W[1, -g:, :] = W[1, -g:, :].clamp(min=0.0)
        for k in range(g):                               # poles: reflect (v_theta flips sign)
            W[:, :, g - 1 - k] = W[:, :, g + k]
            W[:, :, -g + k] = W[:, :, -g - 1 - k]
        W[2, :, :g] = -W[2, :, :g]
        W[2, :, -g:] = -W[2, :, -g:]
        return W

    # ------------------------------------------------------------------ physics pieces
    def _nu(self, W):
        c = self.cfg
        if c.viscosity == "none" or c.alpha == 0:
            return torch.zeros_like(W[0])
        if c.viscosity == "SS":
            cs2 = c.gamma * W[4] / W[0]
            return c.alpha * cs2 / _omega_k(self.r)
        if c.viscosity == "ST":
            return (c.alpha * torch.sqrt(self.r)).expand_as(W[0])
        raise ValueError(f"unknown viscosity '{c.viscosity}'")

    def _flux(self, WL, WR, d, Rf):
        """HLL flux through faces normal to direction d (0 = r, 1 = theta); Rf = cylindrical radius at the faces."""
        gam = self.cfg.gamma

        def phys(W):
            rho, vr, vth, vph, p = W
            vn = vr if d == 0 else vth
            l = vph * Rf
            E = p / (gam - 1) + 0.5 * rho * (vr ** 2 + vth ** 2 + vph ** 2)
            U = torch.stack([rho, rho * vr, rho * vth, rho * l, E])
            F = torch.stack([rho * vn, rho * vr * vn + (p if d == 0 else 0 * p),
                             rho * vth * vn + (p if d == 1 else 0 * p), rho * l * vn, (E + p) * vn])
            return U, F, vn, torch.sqrt(gam * p / rho)

        UL, FL, vL, cL = phys(WL)
        UR, FR, vR, cR = phys(WR)
        SL = torch.minimum(vL - cL, vR - cR).clamp(max=0.0)
        SR = torch.maximum(vL + cL, vR + cR).clamp(min=0.0)
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL + 1e-300)

    def _recon(self, W, d):
        """Left/right states at the interior faces along d (faces between cells 1..n+2g-2)."""
        if d == 0:
            dm, dp = W[:, 1:-1] - W[:, :-2], W[:, 2:] - W[:, 1:-1]
            s = _minmod(dm, dp)
            Wc = W[:, 1:-1]
            return (Wc + 0.5 * s)[:, :-1], (Wc - 0.5 * s)[:, 1:]
        dm, dp = W[:, :, 1:-1] - W[:, :, :-2], W[:, :, 2:] - W[:, :, 1:-1]
        s = _minmod(dm, dp)
        Wc = W[:, :, 1:-1]
        return (Wc + 0.5 * s)[:, :, :-1], (Wc - 0.5 * s)[:, :, 1:]

    def rhs(self, W):
        """dU/dt on the interior cells, plus the mass / angular-momentum fluxes through r_in and r_out."""
        c, g = self.cfg, self.NG
        gam = c.gamma
        rho, vr, vth, vph, p = W
        # ---- r direction: faces g .. n+g (the n+1 faces bounding the interior), theta interior
        nr, nt = c.nr, c.ntheta
        WL, WR = self._recon(W, 0)                       # output face m lies between cells m+1 and m+2
        WL, WR = WL[:, g - 2:g - 1 + nr, g:-g], WR[:, g - 2:g - 1 + nr, g:-g]
        rfc = self.rf_c[g:g + nr + 1]                     # (nr+1, 1)
        sin_c = self.sin[:, g:-g]
        Fr = self._flux(WL, WR, 0, rfc * sin_c)
        # ---- theta direction
        WL, WR = self._recon(W, 1)
        WL, WR = WL[:, g:-g, g - 2:g - 1 + nt], WR[:, g:-g, g - 2:g - 1 + nt]
        thf = self.thf_c[:, g:g + nt + 1]
        Rft = self.r[g:-g] * torch.sin(thf)
        Ft = self._flux(WL, WR, 1, Rft)
        # ---- viscous stresses (only T_rphi, T_thetaphi)
        nu = self._nu(W)
        if c.viscosity != "none" and c.alpha > 0:
            Om = vph / self.R.abs().clamp(min=1e-12)
            rn = rho * nu
            # r faces
            rn_f = 0.5 * (rn[g - 1:-g, g:-g] + rn[g:-(g - 1) or None, g:-g])
            dOm = (Om[g:-(g - 1) or None, g:-g] - Om[g - 1:-g, g:-g]) / (self.r[g:-(g - 1) or None] - self.r[g - 1:-g])
            T_rp = rn_f * rfc * sin_c * dOm
            vph_f = 0.5 * (vph[g - 1:-g, g:-g] + vph[g:-(g - 1) or None, g:-g])
            Fr = Fr.clone()
            Fr[3] = Fr[3] - rfc * sin_c * T_rp
            Fr[4] = Fr[4] - vph_f * T_rp
            # theta faces
            rn_f = 0.5 * (rn[g:-g, g - 1:-g] + rn[g:-g, g:-(g - 1) or None])
            dth = self.th[:, g:-(g - 1) or None] - self.th[:, g - 1:-g]
            dOm = (Om[g:-g, g:-(g - 1) or None] - Om[g:-g, g - 1:-g]) / dth
            sf = torch.sin(thf)
            T_tp = rn_f * sf * dOm
            vph_f = 0.5 * (vph[g:-g, g - 1:-g] + vph[g:-g, g:-(g - 1) or None])
            Ft = Ft.clone()
            Ft[3] = Ft[3] - self.r[g:-g] * sf * T_tp
            Ft[4] = Ft[4] - vph_f * T_tp
        Ar = self.area_r[g:g + nr + 1, g:-g]              # (nr+1, nth)
        At = self.area_t[g:-g, g:g + nt + 1]              # (nr, nth+1)
        vol = self.vol[g:-g, g:-g]
        aFr, aFt = Ar * Fr, At * Ft
        dU = -((aFr[:, 1:] - aFr[:, :-1]) + (aFt[:, :, 1:] - aFt[:, :, :-1])) / vol
        # ---- sources
        Wi = W[self.ii]
        rho, vr, vth, vph, p = Wi
        ir = self.inv_r[g:-g]
        r = self.r[g:-g]
        cot_geo = self.geo_t[:, g:-g]
        cot = torch.cos(self.th[:, g:-g]) / self.sin[:, g:-g]
        S = torch.zeros_like(dU)
        S[1] = ir * (rho * (vth ** 2 + vph ** 2) + 2 * p) - rho * _dphi(r)
        S[2] = ir * (p * cot_geo + rho * vph ** 2 * cot - rho * vr * vth)
        S[4] = -rho * vr * _dphi(r)
        fluxes = {"mass_in": -aFr[0, 0].sum(), "mass_out": aFr[0, -1].sum(),
                  "angmom_in": -aFr[3, 0].sum(), "angmom_out": aFr[3, -1].sum()}
        return dU + S, fluxes, nu

    def dt(self, nu=None) -> float:
        c, g = self.cfg, self.NG
        Wi = self.W[self.ii]
        cs = torch.sqrt(c.gamma * Wi[4] / Wi[0])
        dr = self.dr_c[g:-g]
        rdt = self.r[g:-g] * (math.pi / c.ntheta)
        dt_h = torch.minimum(dr / (Wi[1].abs() + cs), rdt / (Wi[2].abs() + cs)).min()
        if nu is None:
            nu = self._nu(self.W)
        nu_i = self.interior(nu)
        if float(nu_i.max()) > 0:
            dmin = torch.minimum(dr, rdt) ** 2
            dt_v = (0.25 * dmin / nu_i.clamp(min=1e-300)).min()      # explicit diffusion limit
            return float(torch.minimum(c.cfl * dt_h, dt_v))
        return float(c.cfl * dt_h)

    def step(self, dt: Optional[float] = None) -> float:
        dt = self.dt() if dt is None else dt
        g = self.NG
        U0 = self.prim_to_cons(self.W[self.ii], self.R[g:-g, g:-g])
        k1, f1, _ = self.rhs(self.W)
        U1 = U0 + dt * k1
        self.W[self.ii] = self.cons_to_prim_i(U1)
        self._bc(self.W)
        k2, f2, _ = self.rhs(self.W)
        U2 = 0.5 * (U0 + U1 + dt * k2)
        self.W[self.ii] = self.cons_to_prim_i(U2)
        self._bc(self.W)
        two_pi = 2 * math.pi
        for k in self.boundary:
            self.boundary[k] += float(0.5 * dt * (f1[k] + f2[k])) * two_pi
        self.mdot_in = float(0.5 * (f1["mass_in"] + f2["mass_in"])) * two_pi
        self.t += dt
        return dt

    def cons_to_prim_i(self, U):
        """cons_to_prim on interior cells (uses the interior cylindrical radius), with the usual safety caps of
        black-hole hydro codes in the tenuous atmosphere: speeds below ``v_max`` (in units of c) and a
        temperature ceiling c_s^2 <= t_cap |Phi|. Both act only on near-empty cells; ``self.capped`` counts them."""
        c, g = self.cfg, self.NG
        R = self.R[g:-g, g:-g]
        r = self.r[g:-g]
        rho = U[0].clamp(min=c.rho_floor)
        vr, vth = U[1] / rho, U[2] / rho
        vph = U[3] / (rho * R)
        vmag = torch.sqrt(vr ** 2 + vth ** 2 + vph ** 2)
        scale = (c.v_max / vmag.clamp(min=1e-300)).clamp(max=1.0)
        vr, vth, vph = vr * scale, vth * scale, vph * scale
        ek = 0.5 * rho * (vr ** 2 + vth ** 2 + vph ** 2)
        p = ((c.gamma - 1) * (U[4] - ek)).clamp(min=c.p_floor)
        self.capped = int((scale < 1).sum())
        if c.t_cap is not None:
            p_max = c.t_cap * rho / (c.gamma * (r - 2.0))
            self.capped += int((p > p_max).sum())
            p = torch.minimum(p, p_max)
        return torch.stack([rho, vr, vth, vph, p])

    # ------------------------------------------------------------------ diagnostics
    def totals(self) -> Dict[str, float]:
        g = self.NG
        U = self.prim_to_cons(self.W)[:, g:-g, g:-g]
        vol = self.vol[g:-g, g:-g] * 2 * math.pi
        return {"mass": float((U[0] * vol).sum()), "angmom": float((U[3] * vol).sum())}

    def budget(self) -> Dict[str, float]:
        """Totals plus everything that crossed the radial boundaries (in - out), for conservation checks."""
        tot = self.totals()
        b = self.boundary
        return {"mass": tot["mass"] + b["mass_in"] + b["mass_out"],
                "angmom": tot["angmom"] + b["angmom_in"] + b["angmom_out"]}

    def primitives(self) -> np.ndarray:
        """Interior primitive fields (5, nr, ntheta): rho, v_r, v_theta, v_phi, p."""
        return self.interior(self.W).cpu().numpy()

    def grid(self) -> Dict[str, np.ndarray]:
        g = self.NG
        return {"r": self.r[g:-g, 0].cpu().numpy(), "theta": self.th[0, g:-g].cpu().numpy(),
                "r_faces": self.rf_np[g:-g], "theta_faces": self.thf_np[g:-g]}

    def run(self, t_end: float, every: float, *, callback=None, max_steps: int = 10 ** 9) -> Dict[str, np.ndarray]:
        """Advance to ``t_end``, saving the primitives every ``every`` (times are hit exactly).

        Returns {"t": (T,), "frames": (T, 5, nr, ntheta) float32, "mdot": (T,)}, frame 0 = current state.
        """
        times, frames, mdot = [self.t], [self.primitives().astype(np.float32)], [self.mdot_in]
        t_next, n = self.t + every, 0
        while self.t < t_end - 1e-9 and n < max_steps:
            dt = min(self.dt(), t_next - self.t)
            self.step(dt)
            n += 1
            if not np.isfinite(self.mdot_in):
                raise FloatingPointError(f"non-finite state at t = {self.t:.3f}")
            if self.t >= t_next - 1e-9:
                times.append(self.t)
                frames.append(self.primitives().astype(np.float32))
                mdot.append(self.mdot_in)
                if callback is not None:
                    callback(self, len(frames) - 1)
                t_next += every
        return {"t": np.asarray(times), "frames": np.stack(frames), "mdot": np.asarray(mdot)}


# ---------------------------------------------------------------------- initial conditions
def torus_state(flow: AccretionFlow) -> np.ndarray:
    """Equilibrium torus with l(R) = l_c (R/R_c)^a (pressure maximum at R_c, inner edge at R_edge) in an atmosphere.

    Enthalpy h = gamma/(gamma-1) p/rho = C - Phi(r) + int l^2/R^3 dR; with the polytrope p = K rho^gamma and
    the density maximum normalised to 1.
    """
    c, g = flow.cfg, AccretionFlow.NG
    r = flow.r[g:-g].cpu().numpy()
    th = flow.th[:, g:-g].cpu().numpy()
    R = r * np.sin(th)
    a, Rc, Re, gam = c.torus_a, c.torus_rc, c.torus_edge, c.gamma
    lc = Rc ** 1.5 / (Rc - 2.0)                         # Keplerian (Paczynski-Wiita) at the pressure maximum

    def psi(Rr):                                        # int l^2 / R^3 dR
        return -lc ** 2 * (Rr / Rc) ** (2 * a) / ((2 - 2 * a) * Rr ** 2)

    C = _phi(Re) - psi(Re)
    h = C - _phi(r) + psi(np.maximum(R, 1e-6))
    hmax = C - _phi(Rc) + psi(Rc)
    if hmax <= 0:
        raise ValueError("torus parameters give no bound torus (h_max <= 0)")
    K = (gam - 1) / gam * hmax                          # so that rho = 1 where h = hmax
    inside = h > 0
    rho_t = np.where(inside, (np.clip(h, 0, None) * (gam - 1) / (gam * K)) ** (1 / (gam - 1)), 0.0)
    p_t = K * rho_t ** gam
    rho_atm = c.rho_atm * (r / c.r_in) ** -1.5 * np.ones_like(th)
    p_atm = 1e-2 * rho_atm / r                          # cold: c_s^2 ~ 1e-2 v_ff^2
    torus = inside & (rho_t > rho_atm)
    rng = np.random.default_rng(c.seed)
    rho = np.where(torus, rho_t * (1 + c.perturbation * rng.uniform(-1, 1, rho_t.shape)), rho_atm)
    p = np.where(torus, p_t, p_atm)
    l = lc * (np.maximum(R, 1e-6) / Rc) ** a
    vph = np.where(torus, l / np.maximum(R, 1e-6), 0.0)
    zero = np.zeros_like(rho)
    return np.stack([rho, zero, zero, vph, p])


def bondi_pw(r: np.ndarray, *, cs_inf: float, gamma: float = 1.4, rho_inf: float = 1.0):
    """Transonic Bondi accretion in the Paczynski-Wiita potential (exact, by root finding).

    Bernoulli v^2/2 + c^2/(gamma-1) + Phi(r) = c_inf^2/(gamma-1) and r^2 rho v = const through the sonic point
    r_c, where v_c^2 = c_c^2 = r_c / (2 (r_c - 2)^2). Returns (rho, v_r < 0, p, mdot, r_c).
    """
    from scipy.optimize import brentq

    n = 1.0 / (gamma - 1)
    B = n * cs_inf ** 2

    def f(rc):
        cc2 = rc / (2 * (rc - 2) ** 2)
        return cc2 * (0.5 + n) - 1 / (rc - 2) - B

    grid = 2.0 + np.logspace(-4, 7, 4000)
    vals = np.array([f(x) for x in grid])
    flips = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if not len(flips):
        raise ValueError("no sonic point for these parameters")
    k = flips[-1]                                       # outer (X-type) sonic point, the classical Bondi branch
    rc = brentq(f, grid[k], grid[k + 1])
    cc2 = rc / (2 * (rc - 2) ** 2)
    rho_c = rho_inf * (cc2 / cs_inf ** 2) ** n
    lam = rho_c * math.sqrt(cc2) * rc ** 2               # r^2 rho |v|
    rho_out, v_out = np.empty_like(r), np.empty_like(r)
    for i, ri in enumerate(r):
        def g(ln_rho):
            rho = math.exp(ln_rho)
            c2 = cs_inf ** 2 * (rho / rho_inf) ** (gamma - 1)
            v = lam / (rho * ri ** 2)
            return 0.5 * v * v + n * c2 - 1 / (ri - 2) - B
        # the density where v = c splits the two roots: outside r_c the flow is subsonic (denser root),
        # inside it is supersonic
        rho_s = (lam / (ri ** 2 * cs_inf) * rho_inf ** ((gamma - 1) / 2)) ** (2 / (gamma + 1))
        lo, hi = (math.log(rho_s), math.log(rho_s) + 30) if ri > rc else (math.log(rho_s) - 30, math.log(rho_s))
        ln_rho = math.log(rho_s) if g(math.log(rho_s)) >= 0 else brentq(g, lo, hi)
        rho_out[i] = math.exp(ln_rho)
        v_out[i] = -lam / (rho_out[i] * ri ** 2)
    p = rho_inf * cs_inf ** 2 / gamma * (rho_out / rho_inf) ** gamma
    return rho_out, v_out, p, 4 * math.pi * lam, rc
