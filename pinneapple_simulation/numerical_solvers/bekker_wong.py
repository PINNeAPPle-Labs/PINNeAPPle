"""Bekker-Wong numerical solver for rigid-wheel / deformable-soil interaction.

Model (Wong & Reece 1967; Wong 1978)
-----------------------------------
Contact arc: entry angle theta_f = arccos(1 - z/R), rear angle theta_r = -rear_angle_ratio*theta_f,
maximum-stress angle theta_m = (a0 + a1*s)*theta_f.

    sigma(theta) = (k_c/b + k_phi) * (R (cos theta* - cos theta_f))^n
        theta* = theta                                                    (front region, theta >= theta_m)
        theta* = theta_f - (theta - theta_r)/(theta_m - theta_r)*(theta_f - theta_m)   (rear region)
    j(theta)     = R [(theta_f - theta) - (1 - s)(sin theta_f - sin theta)]
    tau(theta)   = sign(j) (c + sigma tan phi)(1 - exp(-|j|/K))               (Janosi-Hanamoto)

    F_x = R b int (tau cos - sigma sin),  F_z = R b int (sigma cos + tau sin),  M_y = R^2 b int tau

Robustness features (vs. the first version of this module)
-----------------------------------------------------------
* Input validation (0 < z < R, -1 <= s < 1, positive soil/wheel parameters) -- raises instead
  of silently returning garbage.
* ``forces`` passes the integrand's kinks (theta_m, 0) as ``quad`` break points and exposes the
  quadrature error estimate.
* ``forces_batch`` / :func:`bekker_wong_forces_torch`: vectorised piecewise Gauss-Legendre
  quadrature (numpy or differentiable torch, per-sample soil parameters allowed), verified
  against adaptive ``quad``; ~10^4x faster for batches.
* ``sinkage_from_load`` brackets and solves F_z(z) = W with Brent's method (F_z is monotone in z
  on the checked domain) instead of an unguarded secant iteration.
* ``generate_dataset`` never writes fabricated zeros for failed evaluations; it raises (strict)
  or drops the point and reports how many were dropped.
* Shear stress is sign-aware, so braking (s < 0) no longer produces an exponentially growing
  tau for negative shear displacement.

References
----------
- Bekker, M.G. (1969). Introduction to Terrain-Vehicle Systems. U Michigan Press.
- Wong, J.Y. (1978). Theory of Ground Vehicles. Wiley.
- Wong, J.Y., Reece, A.R. (1967). Prediction of rigid wheel performance based on the analysis of
  soil-wheel stresses. J. Terramechanics 4(1).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry
from ..particle_dynamics.terramechanics import SoilParams, WheelParams

try:
    from scipy.integrate import quad as _quad
    from scipy.optimize import brentq as _brentq
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _validate(soil: SoilParams, wheel: WheelParams) -> None:
    for name in ("K", "k_phi", "n"):
        if not getattr(soil, name) > 0:
            raise ValueError(f"SoilParams.{name} must be > 0, got {getattr(soil, name)}")
    if soil.c < 0 or soil.k_c < 0:
        raise ValueError("SoilParams.c and k_c must be >= 0")
    if not 0 < soil.phi_deg < 90:
        raise ValueError(f"SoilParams.phi_deg must be in (0, 90), got {soil.phi_deg}")
    if not (wheel.R > 0 and wheel.b > 0):
        raise ValueError("WheelParams.R and b must be > 0")


@SolverRegistry.register(
    name="bekker_wong",
    family="terramechanics",
    description="Bekker-Wong rigid-wheel / deformable-soil force model",
    tags=["rover", "lunar", "soil", "terramechanics"],
)
class BekkerWongSolver(SolverBase):
    """Numerical Bekker-Wong solver.

    Computes drawbar pull (F_x), normal load (F_z), and driving torque (M_y)
    for a rigid wheel rolling on deformable soil.

    Parameters
    ----------
    soil : SoilParams
    wheel : WheelParams
    rear_angle_ratio : theta_r = -rear_angle_ratio * theta_f (1/3 reproduces the original module;
        0 gives the classical "no rear contact" assumption).
    """

    def __init__(
        self,
        soil: Optional[SoilParams] = None,
        wheel: Optional[WheelParams] = None,
        rear_angle_ratio: float = 1.0 / 3.0,
    ):
        super().__init__()
        self.soil = soil or SoilParams()
        self.wheel = wheel or WheelParams()
        if not 0.0 <= rear_angle_ratio < 1.0:
            raise ValueError("rear_angle_ratio must be in [0, 1)")
        self.rear_angle_ratio = float(rear_angle_ratio)
        _validate(self.soil, self.wheel)

    # ------------------------------------------------------------------
    # Contact angle geometry
    # ------------------------------------------------------------------

    def _check_inputs(self, slip: float, z: float) -> None:
        if not (0.0 < z < self.wheel.R):
            raise ValueError(f"sinkage z must be in (0, R={self.wheel.R}), got {z}")
        if not (-1.0 <= slip < 1.0):
            raise ValueError(f"slip must be in [-1, 1), got {slip}")

    def _contact_angles(self, z: float) -> Tuple[float, float, float]:
        """Entry, mid (at s=0) and rear contact angles for sinkage z [m]."""
        return self._contact_angles_slip(z, 0.0)

    def _contact_angles_slip(self, z: float, slip: float) -> Tuple[float, float, float]:
        """Contact angles accounting for slip ratio."""
        R = self.wheel.R
        s = self.soil
        theta_f = math.acos(max(-1.0, min(1.0, 1.0 - z / R)))
        theta_m = (s.a0 + s.a1 * slip) * theta_f
        theta_r = -self.rear_angle_ratio * theta_f
        return theta_f, theta_m, theta_r

    # ------------------------------------------------------------------
    # Stress distributions
    # ------------------------------------------------------------------

    def sigma(self, theta: float, z: float, slip: float) -> float:
        """Normal stress at contact angle theta [Pa] using Bekker pressure-sinkage."""
        s = self.soil
        R = self.wheel.R
        theta_f, theta_m, theta_r = self._contact_angles_slip(z, slip)
        ksn = s.k_c / self.wheel.b + s.k_phi

        if theta >= theta_m:
            h = R * max(math.cos(theta) - math.cos(theta_f), 0.0)
        else:
            denom = max(theta_m - theta_r, 1e-9)
            ratio = (theta_f - theta_m) * (theta - theta_r) / denom
            h = R * max(math.cos(theta_f - ratio) - math.cos(theta_f), 0.0)

        return ksn * (h ** s.n)

    def tau(self, theta: float, z: float, slip: float) -> float:
        """Shear stress at contact angle theta [Pa] -- Mohr-Coulomb + Janosi-Hanamoto (sign-aware)."""
        s = self.soil
        R = self.wheel.R
        theta_f, _, _ = self._contact_angles_slip(z, slip)
        sig = self.sigma(theta, z, slip)
        j = R * ((theta_f - theta) - (1.0 - slip) * (math.sin(theta_f) - math.sin(theta)))
        return math.copysign(1.0, j) * (s.c + sig * s.tan_phi) * (1.0 - math.exp(-abs(j) / s.K))

    # ------------------------------------------------------------------
    # Force integration
    # ------------------------------------------------------------------

    def forces(
        self, slip: float, z: float, *, return_error: bool = False
    ):
        """Compute (F_x, F_z, M_y) via adaptive quadrature over the contact patch.

        Parameters
        ----------
        slip : slip ratio [-], 0 = free rolling, 1 = locked wheel spinning in place
        z : sinkage [m], 0 < z < R
        return_error : also return the absolute quadrature error estimates.
        """
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required for BekkerWongSolver.forces()")
        self._check_inputs(slip, z)

        R = self.wheel.R
        b = self.wheel.b
        theta_f, theta_m, theta_r = self._contact_angles_slip(z, slip)
        pts = [p for p in (theta_m, 0.0) if theta_r < p < theta_f]

        def integrand_fx(theta: float) -> float:
            return self.tau(theta, z, slip) * math.cos(theta) - self.sigma(theta, z, slip) * math.sin(theta)

        def integrand_fz(theta: float) -> float:
            return self.sigma(theta, z, slip) * math.cos(theta) + self.tau(theta, z, slip) * math.sin(theta)

        def integrand_my(theta: float) -> float:
            return self.tau(theta, z, slip)

        kw = dict(limit=200, points=pts or None, epsabs=1e-10, epsrel=1e-10)
        Fx, eFx = _quad(integrand_fx, theta_r, theta_f, **kw)
        Fz, eFz = _quad(integrand_fz, theta_r, theta_f, **kw)
        My, eMy = _quad(integrand_my, theta_r, theta_f, **kw)

        out = (R * b * Fx, R * b * Fz, R * R * b * My)
        if return_error:
            return out, (R * b * eFx, R * b * eFz, R * R * b * eMy)
        return out

    def forces_batch(self, slip, z, n_gl: int = 48) -> np.ndarray:
        """Vectorised Gauss-Legendre evaluation for arrays of (slip, z). Returns (N, 3)."""
        slip = torch.as_tensor(np.atleast_1d(slip), dtype=torch.float64)
        z = torch.as_tensor(np.atleast_1d(z), dtype=torch.float64)
        if (z <= 0).any() or (z >= self.wheel.R).any():
            raise ValueError("all sinkages must be in (0, R)")
        if (slip < -1).any() or (slip >= 1).any():
            raise ValueError("all slips must be in [-1, 1)")
        s = self.soil
        out = bekker_wong_forces_torch(
            slip, z, R=self.wheel.R, b=self.wheel.b, c=s.c, tan_phi=s.tan_phi, K=s.K, k_c=s.k_c,
            k_phi=s.k_phi, n=s.n, a0=s.a0, a1=s.a1, rear_angle_ratio=self.rear_angle_ratio, n_gl=n_gl)
        return out.numpy()

    def sinkage_from_load(
        self,
        W: float,
        slip: float,
        z_lo: float = 1e-6,
        z_hi: Optional[float] = None,
        xtol: float = 1e-9,
        **_legacy,
    ) -> float:
        """Sinkage z with F_z(slip, z) = W, by Brent's method on a bracket.

        Raises ``ValueError`` if W is not attainable inside (z_lo, z_hi) instead of
        returning an unconverged iterate.
        """
        if not W > 0:
            raise ValueError("load W must be > 0")
        z_hi = z_hi if z_hi is not None else 0.9 * self.wheel.R
        f = lambda z: self.forces(slip, z)[1] - W
        f_lo, f_hi = f(z_lo), f(z_hi)
        if f_lo > 0 or f_hi < 0:
            raise ValueError(f"load W={W} N not bracketed by F_z(z_lo)={f_lo + W:.4g}, F_z(z_hi)={f_hi + W:.4g}")
        return float(_brentq(f, z_lo, z_hi, xtol=xtol))

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_slip: int = 40,
        n_sink: int = 30,
        slip_range: Tuple[float, float] = (0.0, 0.75),
        sink_range: Tuple[float, float] = (0.002, 0.058),
        n_lhs: int = 500,
        seed: int = 42,
        strict: bool = True,
        report: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate (slip, sinkage) -> (F_x, F_z, M_y) dataset (grid + Latin hypercube).

        A failed evaluation raises when ``strict`` (default); otherwise the point is dropped and
        counted in ``report["n_failed"]`` -- it is never replaced by a fabricated value.
        """
        from scipy.stats import qmc

        s_grid = np.linspace(*slip_range, n_slip)
        z_grid = np.linspace(*sink_range, n_sink)
        ss, zz = np.meshgrid(s_grid, z_grid)
        grid_pts = np.stack([ss.ravel(), zz.ravel()], axis=1)
        pts = grid_pts
        if n_lhs > 0:
            lhs = qmc.scale(qmc.LatinHypercube(d=2, seed=seed).random(n_lhs),
                            [slip_range[0], sink_range[0]], [slip_range[1], sink_range[1]])
            pts = np.vstack([grid_pts, lhs])

        keep, Y_list, n_failed = [], [], 0
        for i, (slip, z) in enumerate(pts):
            try:
                Y_list.append(self.forces(float(slip), float(z)))
                keep.append(i)
            except Exception:
                if strict:
                    raise
                n_failed += 1
        if report is not None:
            report["n_failed"] = n_failed
            report["n_requested"] = len(pts)
        return pts[keep].astype(np.float32), np.array(Y_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # SolverBase interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> SolverOutput:
        """Evaluate forces for a batch of (slip, sinkage) inputs, shape (N, 2) -> (N, 3)."""
        res = self.forces_batch(x[:, 0].detach().cpu().numpy(), x[:, 1].detach().cpu().numpy())
        result = torch.tensor(res, dtype=x.dtype, device=x.device)
        return SolverOutput(result=result, losses={}, extras={})


# ----------------------------------------------------------------------------
# Vectorised / differentiable quadrature
# ----------------------------------------------------------------------------

_GL_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _gl(n: int):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def bekker_wong_forces_torch(slip, z, *, R, b, c, tan_phi, K, k_c, k_phi, n, a0=0.40, a1=0.15,
                             rear_angle_ratio=1.0 / 3.0, n_gl: int = 48) -> torch.Tensor:
    """Bekker-Wong (F_x, F_z, M_y) for batches, piecewise Gauss-Legendre on the three smooth
    sub-intervals [theta_r, min(0,theta_m)], [.., theta_m], [theta_m, theta_f].

    Every soil/wheel argument may be a scalar or a tensor broadcastable to ``slip`` (per-sample
    soils). Differentiable w.r.t. every tensor argument. Returns (N, 3).
    """
    dt = slip.dtype if torch.is_tensor(slip) else torch.float64
    T = lambda v: torch.as_tensor(v, dtype=dt)
    slip, z = T(slip).reshape(-1), T(z).reshape(-1)
    R, b, c, tan_phi, K, k_c, k_phi, n = (T(v) for v in (R, b, c, tan_phi, K, k_c, k_phi, n))
    col = lambda v: v.reshape(-1, 1) if v.ndim == 1 else v   # per-sample soil -> broadcast over GL nodes
    c, tan_phi, K, k_c, k_phi, n = (col(v) for v in (c, tan_phi, K, k_c, k_phi, n))
    theta_f = torch.acos(torch.clamp(1 - z / R, -1.0, 1.0))
    theta_m = (a0 + a1 * slip) * theta_f
    theta_r = -rear_angle_ratio * theta_f
    x, w = (torch.as_tensor(v, dtype=dt) for v in _gl(n_gl))
    ksn = k_c / b + k_phi

    def integrate(lo, hi):
        half = 0.5 * (hi - lo)[:, None]
        th = 0.5 * (hi + lo)[:, None] + half * x[None]
        rear = th < theta_m[:, None]
        ratio = (theta_f - theta_m)[:, None] * (th - theta_r[:, None]) / (theta_m - theta_r).clamp(min=1e-12)[:, None]
        th_eq = torch.where(rear, theta_f[:, None] - ratio, th)
        h = (R * (torch.cos(th_eq) - torch.cos(theta_f)[:, None])).clamp(min=0.0)
        sig = ksn * torch.where(h > 0, h.clamp(min=1e-30) ** n, torch.zeros_like(h))
        j = R * ((theta_f[:, None] - th) - (1 - slip)[:, None] * (torch.sin(theta_f)[:, None] - torch.sin(th)))
        tau = torch.sign(j) * (c + sig * tan_phi) * (1 - torch.exp(-j.abs() / K))
        fx = (tau * torch.cos(th) - sig * torch.sin(th))
        fz = (sig * torch.cos(th) + tau * torch.sin(th))
        return torch.stack([(fx * w).sum(1) * half[:, 0], (fz * w).sum(1) * half[:, 0], (tau * w).sum(1) * half[:, 0]], 1)

    mid0 = torch.minimum(torch.zeros_like(theta_m), theta_m).clamp(min=theta_r)
    I = integrate(theta_r, mid0) + integrate(mid0, theta_m) + integrate(theta_m, theta_f)
    return torch.stack([R * b * I[:, 0], R * b * I[:, 1], R * R * b * I[:, 2]], 1)


__all__ = ["BekkerWongSolver", "bekker_wong_forces_torch"]
