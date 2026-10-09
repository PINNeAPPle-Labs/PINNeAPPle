"""Unsaturated flow in soil: the Richards equation in 1-D with the classical water-retention / conductivity models.

Retention and conductivity (h = pressure head [L], negative when unsaturated; Se = effective saturation):

* :class:`VanGenuchten` (van Genuchten 1980, SSSAJ 44:892) with Mualem's conductivity (Mualem 1976, WRR 12:513):
  Se = [1 + |alpha h|^n]^-m, m = 1 - 1/n;  K = Ks Se^l [1 - (1 - Se^(1/m))^m]^2, l = 0.5.
* :class:`BrooksCorey` (Brooks & Corey 1964, Hydrology Paper 3): Se = (h_b/h)^lambda for h < h_b;
  K = Ks Se^((2 + 3 lambda)/lambda).
* :class:`ClappHornberger` (Clapp & Hornberger 1978, WRR 14:601; the same power laws as Campbell 1974, Soil Sci.
  117:311): theta/theta_s = (psi_s/h)^(1/b); K = Ks (theta/theta_s)^(2b + 3).
* :class:`Gardner` (Gardner 1958, Soil Sci. 85:228): K = Ks exp(alpha h), Se = exp(alpha h); the model with an exact
  steady infiltration profile (:func:`gardner_steady_infiltration`), used to verify the solver.

Solver (:func:`solve_richards`): mixed-form (theta-h) equation, z positive upward,

    d theta/dt = d/dz [ K(h) (dh/dz + 1) ] - S(z, t),

cell-centred finite volumes, backward Euler in time and the modified Picard iteration of Celia, Bouloutas & Zarba
(1990, WRR 26:1483). The mixed form makes the scheme mass-conservative: the mass-balance ratio reported with each
result is 1 to the Picard tolerance, unlike the h-form. Boundary conditions: prescribed head, prescribed flux
(positive upward), or free drainage (unit gradient) at the bottom. Time step adapts to the Picard iteration count.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["VanGenuchten", "BrooksCorey", "ClappHornberger", "Campbell", "Gardner", "Boundary", "RichardsResult",
           "solve_richards", "gardner_steady_infiltration", "CELIA_1990_SOIL"]


@dataclass(frozen=True)
class VanGenuchten:
    theta_r: float
    theta_s: float
    alpha: float            # [1/L]
    n: float
    Ks: float               # [L/T]
    l: float = 0.5          # noqa: E741  (Mualem pore-connectivity parameter, named l in the literature)

    @property
    def m(self) -> float:
        return 1 - 1 / self.n

    def Se(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < 0, (1 + np.abs(self.alpha * h) ** self.n) ** (-self.m), 1.0)

    def theta(self, h):
        return self.theta_r + (self.theta_s - self.theta_r) * self.Se(h)

    def K(self, h):
        se = np.clip(self.Se(h), 1e-300, 1.0)
        return self.Ks * se ** self.l * (1 - (1 - se ** (1 / self.m)) ** self.m) ** 2

    def C(self, h):
        """Specific moisture capacity d theta / dh."""
        h = np.asarray(h, dtype=float)
        ah = np.abs(self.alpha * h)
        c = (self.theta_s - self.theta_r) * self.m * self.n * self.alpha * ah ** (self.n - 1) * (1 + ah ** self.n) ** (-self.m - 1)
        return np.where(h < 0, c, 0.0)


@dataclass(frozen=True)
class BrooksCorey:
    theta_r: float
    theta_s: float
    h_b: float              # air-entry (bubbling) head, negative [L]
    lam: float              # pore-size distribution index
    Ks: float

    def Se(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < self.h_b, (self.h_b / np.minimum(h, self.h_b)) ** self.lam, 1.0)

    def theta(self, h):
        return self.theta_r + (self.theta_s - self.theta_r) * self.Se(h)

    def K(self, h):
        return self.Ks * self.Se(h) ** ((2 + 3 * self.lam) / self.lam)

    def C(self, h):
        h = np.asarray(h, dtype=float)
        hh = np.minimum(h, self.h_b)
        c = (self.theta_s - self.theta_r) * self.lam * (self.h_b / hh) ** self.lam / np.abs(hh)
        return np.where(h < self.h_b, c, 0.0)


@dataclass(frozen=True)
class ClappHornberger:
    theta_s: float
    psi_s: float            # saturated (air-entry) head, negative [L]
    b: float
    Ks: float
    theta_r: float = 0.0    # the original model has none; kept for a common interface

    def Se(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < self.psi_s, (self.psi_s / np.minimum(h, self.psi_s)) ** (1 / self.b), 1.0)

    def theta(self, h):
        return self.theta_r + (self.theta_s - self.theta_r) * self.Se(h)

    def K(self, h):
        return self.Ks * self.Se(h) ** (2 * self.b + 3)

    def C(self, h):
        h = np.asarray(h, dtype=float)
        hh = np.minimum(h, self.psi_s)
        c = (self.theta_s - self.theta_r) / self.b * (self.psi_s / hh) ** (1 / self.b) / np.abs(hh)
        return np.where(h < self.psi_s, c, 0.0)


Campbell = ClappHornberger


@dataclass(frozen=True)
class Gardner:
    theta_r: float
    theta_s: float
    alpha: float
    Ks: float

    def Se(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < 0, np.exp(self.alpha * np.minimum(h, 0.0)), 1.0)

    def theta(self, h):
        return self.theta_r + (self.theta_s - self.theta_r) * self.Se(h)

    def K(self, h):
        return self.Ks * self.Se(h)

    def C(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < 0, (self.theta_s - self.theta_r) * self.alpha * np.exp(self.alpha * np.minimum(h, 0.0)), 0.0)


# Celia et al. (1990) test soil (cm, s): van Genuchten parameters of their infiltration problem
CELIA_1990_SOIL = VanGenuchten(theta_r=0.102, theta_s=0.368, alpha=0.0335, n=2.0, Ks=0.00922)


@dataclass(frozen=True)
class Boundary:
    """``kind``: "head" (value = h), "flux" (value = Darcy flux, positive upward, i.e. infiltration < 0) or
    "free_drainage" (bottom only). ``value`` may be a function of time."""
    kind: str
    value: float | Callable[[float], float] = 0.0

    def at(self, t: float) -> float:
        return float(self.value(t)) if callable(self.value) else float(self.value)


@dataclass
class RichardsResult:
    z: np.ndarray
    t: np.ndarray
    h: np.ndarray           # (n_out, n_cells)
    theta: np.ndarray
    flux_top: np.ndarray    # Darcy flux through the top face at each output time (positive upward)
    flux_bottom: np.ndarray
    mass_balance_ratio: float
    n_steps: int
    n_picard: int

    def to_dict(self) -> dict[str, object]:
        return {"z": self.z.tolist(), "t": self.t.tolist(), "h": self.h.tolist(), "theta": self.theta.tolist(),
                "mass_balance_ratio": self.mass_balance_ratio, "n_steps": self.n_steps}


def _face_K(Kc: np.ndarray) -> np.ndarray:
    return 0.5 * (Kc[1:] + Kc[:-1])                                  # arithmetic mean (Celia et al. 1990)


def solve_richards(soil, depth: float, n_cells: int, h0: float | np.ndarray, t_end: float,
                   top: Boundary, bottom: Boundary, dt0: float = 1.0, dt_min: float = 1e-6,
                   dt_max: float | None = None, t_out: np.ndarray | None = None, tol: float = 1e-6,
                   max_picard: int = 30, sink: Callable[[np.ndarray, float], np.ndarray] | None = None) -> RichardsResult:
    """Column from z = -depth (bottom) to z = 0 (surface). ``h0``: initial head (scalar or per cell, bottom first).
    ``sink(z, t)``: volumetric sink [1/T] (root water uptake). Returns profiles at ``t_out`` (default: 11 times)."""
    dz = depth / n_cells
    z = -depth + dz * (np.arange(n_cells) + 0.5)
    h = np.full(n_cells, float(h0)) if np.ndim(h0) == 0 else np.array(h0, dtype=float)
    t_out = np.linspace(0, t_end, 11) if t_out is None else np.asarray(t_out, dtype=float)
    dt_max = dt_max or t_end / 20
    out_h, out_th, out_qt, out_qb = [], [], [], []
    t, dt, k_out, steps, picard_total = 0.0, dt0, 0, 0, 0
    water0 = float(np.sum(soil.theta(h)) * dz)
    net_in = 0.0                                                     # cumulative boundary inflow minus sink

    def fluxes(hh, tt):
        Kc = soil.K(hh)
        Kf = _face_K(Kc)
        q_int = -Kf * ((hh[1:] - hh[:-1]) / dz + 1.0)                # interior faces, positive upward
        if top.kind == "head":
            Kt = 0.5 * (Kc[-1] + soil.K(top.at(tt)))
            qt = -Kt * ((top.at(tt) - hh[-1]) / (dz / 2) + 1.0)
        else:
            qt = top.at(tt)
        if bottom.kind == "head":
            Kb = 0.5 * (Kc[0] + soil.K(bottom.at(tt)))
            qb = -Kb * ((hh[0] - bottom.at(tt)) / (dz / 2) + 1.0)
        elif bottom.kind == "free_drainage":
            qb = -Kc[0]
        else:
            qb = bottom.at(tt)
        return q_int, qt, qb

    def record(hh, tt):
        _, qt, qb = fluxes(hh, tt)
        out_h.append(hh.copy())
        out_th.append(soil.theta(hh))
        out_qt.append(qt)
        out_qb.append(qb)

    while k_out < len(t_out) and t_out[k_out] <= 1e-12:
        record(h, 0.0)
        k_out += 1
    while t < t_end - 1e-12:
        dt = min(dt, dt_max, t_end - t)
        if k_out < len(t_out):
            dt = min(dt, t_out[k_out] - t)
        tn = t + dt
        theta_n = soil.theta(h)
        hm = h.copy()
        converged = False
        for it in range(1, max_picard + 1):  # noqa: B007  (the count is used after the loop)
            Kc = soil.K(hm)
            Kf = _face_K(Kc)
            Cm = soil.C(hm)
            N = n_cells
            lower = np.zeros(N)
            diag = Cm / dt
            upper = np.zeros(N)
            rhs = Cm / dt * hm - (soil.theta(hm) - theta_n) / dt
            a = Kf / dz ** 2                                         # face conductances
            diag[:-1] += a
            diag[1:] += a
            upper[:-1] = -a
            lower[1:] = -a
            # gravity part of -dq/dz with q = -K (dh/dz + 1): +K_face/dz below each face, -K_face/dz above it
            rhs[:-1] += Kf / dz
            rhs[1:] -= Kf / dz
            if top.kind == "head":
                ht = top.at(tn)
                Kt = 0.5 * (Kc[-1] + soil.K(ht))
                at = Kt / (dz * dz / 2)
                diag[-1] += at
                rhs[-1] += at * ht + Kt / dz
            else:
                rhs[-1] += -top.at(tn) / dz                         # q_top positive upward leaves the column
            if bottom.kind == "head":
                hb = bottom.at(tn)
                Kb = 0.5 * (Kc[0] + soil.K(hb))
                ab = Kb / (dz * dz / 2)
                diag[0] += ab
                rhs[0] += ab * hb - Kb / dz
            elif bottom.kind == "free_drainage":
                rhs[0] -= Kc[0] / dz
            else:
                rhs[0] += bottom.at(tn) / dz
            if sink is not None:
                rhs -= sink(z, tn)
            h_new = _thomas(lower, diag, upper, rhs)
            if not np.all(np.isfinite(h_new)):
                break
            change = np.max(np.abs(h_new - hm))
            hm = h_new
            if change < tol * max(1.0, np.max(np.abs(hm))):
                converged = True
                break
        picard_total += it
        if not converged:
            if dt <= dt_min:
                raise RuntimeError(f"Richards solver did not converge at t={t:g} with dt={dt:g}")
            dt = max(dt / 3, dt_min)
            continue
        # cumulative inflow over the step (fluxes at the new level, as in the implicit scheme)
        _, qt, qb = fluxes(hm, tn)
        net_in += dt * (qb - qt)
        if sink is not None:
            net_in -= dt * float(np.sum(sink(z, tn)) * dz)
        h, t, steps = hm, tn, steps + 1
        while k_out < len(t_out) and t >= t_out[k_out] - 1e-9:
            record(h, t)
            k_out += 1
        dt = dt * 1.3 if it <= 4 else (dt * 0.7 if it > 10 else dt)
    water = float(np.sum(soil.theta(h)) * dz)
    ratio = (water - water0) / net_in if abs(net_in) > 1e-300 else 1.0
    return RichardsResult(z, t_out[: len(out_h)], np.array(out_h), np.array(out_th), np.array(out_qt),
                          np.array(out_qb), float(ratio), steps, picard_total)


def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(b)
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def gardner_steady_infiltration(soil: Gardner, z_above_table: np.ndarray, q0: float) -> np.ndarray:
    """Exact steady head profile above a water table (h = 0 at z = 0) under a constant infiltration rate q0 > 0
    (downward) for the Gardner model: K(z) = q0 + (Ks - q0) exp(-alpha z), h = ln(K/Ks)/alpha."""
    K = q0 + (soil.Ks - q0) * np.exp(-soil.alpha * np.asarray(z_above_table, dtype=float))
    return np.log(K / soil.Ks) / soil.alpha
