"""RANS/LES turbulence closure presets for PINN training.

Implements k-omega SST (Menter 1994) and Spalart-Allmaras (SA, 1992) RANS
models (2D, steady), and three algebraic subgrid-scale closures for
large-eddy simulation of filtered incompressible Navier-Stokes in 3D:
Smagorinsky (1963), dynamic Smagorinsky (Germano 1991, Lilly 1992) and
WALE (Nicoud & Ducros, 1999), as PDE residual functions compatible with
PINNeAPPle's PINNFactory.

The three LES closures share one residual assembly (:class:`_LESResiduals3D`)
and differ only in how the subgrid viscosity is obtained -- the constant
model, the Germano-identity model that computes its own coefficient from
the resolved field, and the wall-adapting model that needs neither a
constant nor a damping function.

Each residual class is callable:
    residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
    r_dict = residuals(model, x_col)   # returns dict of residual tensors

The dict keys match the equation names and can be directly summed into a
PINN loss::

    loss = sum(r.pow(2).mean() for k, r in r_dict.items() if k != "nu_sgs")

(``nu_sgs`` is the LES closures' diagnostic of the eddy viscosity they
produced on those points -- not a residual, so it is not part of the loss.)
"""
from __future__ import annotations

import torch
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# k-omega SST constants  (Menter 1994)
# ---------------------------------------------------------------------------

SST_CONSTS: Dict[str, float] = {
    "sigma_k1":  0.85,
    "sigma_k2":  1.0,
    "sigma_w1":  0.5,
    "sigma_w2":  0.856,
    "beta1":     0.075,
    "beta2":     0.0828,
    "beta_star": 0.09,
    "kappa":     0.41,
    "a1":        0.31,
    "gamma1":    5.0 / 9.0,
    "gamma2":    0.44,
}


# ---------------------------------------------------------------------------
# Helper: first / second-order autograd
# ---------------------------------------------------------------------------

def _grad1(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of scalar/vector field f w.r.t. x (same-shape as x)."""
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


def _laplacian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Scalar Laplacian of f w.r.t. x: sum_i d^2f/dx_i^2. f must be (N,1)."""
    gf = _grad1(f, x)  # (N, D)
    lap = torch.zeros(f.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        g_i = gf[:, i : i + 1]
        lap = lap + _grad1(g_i, x)[:, i : i + 1]
    return lap


def _hessian_components_2d(f: torch.Tensor, x: torch.Tensor, grad_f: torch.Tensor):
    """Given grad_f = _grad1(f, x) for 2D x, return (f_xx, f_xy, f_yy)."""
    gx = _grad1(grad_f[:, 0:1], x)
    gy = _grad1(grad_f[:, 1:2], x)
    f_xx = gx[:, 0:1]
    f_xy = gx[:, 1:2]
    f_yy = gy[:, 1:2]
    return f_xx, f_xy, f_yy


def _viscous_stress_divergence_2d(
    mu_eff: torch.Tensor,
    mu_eff_grad: torch.Tensor,
    x: torch.Tensor,
    u_grad: torch.Tensor,
    v_grad: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
):
    """div[mu_eff (grad u + grad u^T)] for 2D velocity (u, v).

    Computed exactly from the full Hessians of u, v (no incompressibility
    assumption) — required whenever mu_eff varies spatially, since
    div[mu_eff grad u] alone drops the mu_eff_x*u_x / mu_eff_y*v_x terms
    that dominate across a turbulent boundary layer.
    """
    u_x = u_grad[:, 0:1]; u_y = u_grad[:, 1:2]
    v_x = v_grad[:, 0:1]; v_y = v_grad[:, 1:2]
    u_xx, u_xy, u_yy = _hessian_components_2d(u, x, u_grad)
    v_xx, v_xy, v_yy = _hessian_components_2d(v, x, v_grad)
    mu_x = mu_eff_grad[:, 0:1]; mu_y = mu_eff_grad[:, 1:2]

    visc_x = 2.0 * (mu_x * u_x + mu_eff * u_xx) + (mu_y * (u_y + v_x) + mu_eff * (u_yy + v_xy))
    visc_y = (mu_x * (u_y + v_x) + mu_eff * (u_xy + v_xx)) + 2.0 * (mu_y * v_y + mu_eff * v_yy)
    return visc_x, visc_y


def _hessian_components_3d(f: torch.Tensor, x: torch.Tensor, grad_f: torch.Tensor):
    """Given grad_f = _grad1(f, x) for 3D x, return the 6 independent
    second partials (f_xx, f_yy, f_zz, f_xy, f_xz, f_yz)."""
    gx = _grad1(grad_f[:, 0:1], x)
    gy = _grad1(grad_f[:, 1:2], x)
    gz = _grad1(grad_f[:, 2:3], x)
    f_xx = gx[:, 0:1]
    f_yy = gy[:, 1:2]
    f_zz = gz[:, 2:3]
    f_xy = gx[:, 1:2]
    f_xz = gx[:, 2:3]
    f_yz = gy[:, 2:3]
    return f_xx, f_yy, f_zz, f_xy, f_xz, f_yz


def _viscous_stress_divergence_3d(
    mu_eff: torch.Tensor,
    mu_eff_grad: torch.Tensor,
    x: torch.Tensor,
    u_grad: torch.Tensor,
    v_grad: torch.Tensor,
    w_grad: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
):
    """div[mu_eff (grad u_i + grad u^T e_i)] for 3D velocity (u, v, w).

    Same construction as :func:`_viscous_stress_divergence_2d` (full
    Hessians of u, v, w -- no incompressibility assumption), extended
    with the third spatial direction: e.g. for the x-momentum component,
    ``sum_j d/dx_j[mu_eff*(du/dx_j + du_j/dx)]`` over j in {x, y, z}
    expands to the ``visc_x`` expression below (and cyclically for
    ``visc_y``, ``visc_z``); required whenever mu_eff varies spatially,
    exactly as in the 2D case.
    """
    u_x = u_grad[:, 0:1]; u_y = u_grad[:, 1:2]; u_z = u_grad[:, 2:3]
    v_x = v_grad[:, 0:1]; v_y = v_grad[:, 1:2]; v_z = v_grad[:, 2:3]
    w_x = w_grad[:, 0:1]; w_y = w_grad[:, 1:2]; w_z = w_grad[:, 2:3]

    u_xx, u_yy, u_zz, u_xy, u_xz, u_yz = _hessian_components_3d(u, x, u_grad)
    v_xx, v_yy, v_zz, v_xy, v_xz, v_yz = _hessian_components_3d(v, x, v_grad)
    w_xx, w_yy, w_zz, w_xy, w_xz, w_yz = _hessian_components_3d(w, x, w_grad)

    mu_x = mu_eff_grad[:, 0:1]; mu_y = mu_eff_grad[:, 1:2]; mu_z = mu_eff_grad[:, 2:3]

    visc_x = (
        2.0 * (mu_x * u_x + mu_eff * u_xx)
        + (mu_y * (u_y + v_x) + mu_eff * (u_yy + v_xy))
        + (mu_z * (u_z + w_x) + mu_eff * (u_zz + w_xz))
    )
    visc_y = (
        (mu_x * (v_x + u_y) + mu_eff * (v_xx + u_xy))
        + 2.0 * (mu_y * v_y + mu_eff * v_yy)
        + (mu_z * (v_z + w_y) + mu_eff * (v_zz + w_yz))
    )
    visc_z = (
        (mu_x * (w_x + u_z) + mu_eff * (w_xx + u_xz))
        + (mu_y * (w_y + v_z) + mu_eff * (w_yy + v_yz))
        + 2.0 * (mu_z * w_z + mu_eff * w_zz)
    )
    return visc_x, visc_y, visc_z


# ---------------------------------------------------------------------------
# Feature 10a: k-omega SST
# ---------------------------------------------------------------------------

class KOmegaSSTResiduals:
    """k-omega SST two-equation turbulence model residuals for PINN.

    Equations solved (steady RANS, incompressible mean flow):

    2D form -- ``model: (N,2) -> (N,5)``, ``(x,y) -> (u,v,p,k,omega)``::

        Momentum-x:  rho*(u u_x + v u_y) + p_x - d/dx[(mu+mu_t) 2 u_x]
                     - d/dy[(mu+mu_t)(u_y + v_x)] = 0
        Momentum-y:  rho*(u v_x + v v_y) + p_y - d/dx[(mu+mu_t)(u_y + v_x)]
                     - d/dy[(mu+mu_t) 2 v_y] = 0
        Continuity:  u_x + v_y = 0
        k-equation:  u k_x + v k_y - d/dx[(nu+sigma_k nu_t) k_x]
                     - d/dy[(nu+sigma_k nu_t) k_y] - P_k + beta* k omega = 0
        omega-eq:    u w_x + v w_y - d/dx[(nu+sigma_w nu_t) w_x]
                     - d/dy[(nu+sigma_w nu_t) w_y]
                     - gamma P_k/nu_t + beta omega^2 - CD_kw = 0

    3D form -- ``model: (N,3) -> (N,6)``, ``(x,y,z) -> (u,v,w,p,k,omega)``,
    the genuine 3D generalization of the above (mirroring how
    ``WALEResiduals`` below went from a 2D-style helper to a real 3D LES
    closure): every equation gains the third velocity component and its
    z-derivatives, and the viscous/diffusion terms gain the full set of
    z-direction cross terms (no incompressibility-based term-dropping --
    mu_eff/nu_eff still vary spatially, exactly as in the 2D case)::

        Momentum-i (i=x,y,z):  rho*(u.grad)u_i + p_i
                     - div[(mu+mu_t)*(grad u_i + grad u^T e_i)] = 0
        Continuity:  u_x + v_y + w_z = 0
        k-equation:  u k_x + v k_y + w k_z - div[(nu+sigma_k nu_t) grad k]
                     - P_k + beta_star*k*omega = 0
        omega-eq:    u w_x + v w_y + w w_z - div[(nu+sigma_w nu_t) grad omega]
                     - gamma*S_mag^2 + beta*omega^2 - CD_kw = 0

    where ``S_mag = sqrt(2 S_ij S_ij)`` now uses the full 3D symmetric
    strain-rate tensor (6 independent components: S_11=u_x, S_22=v_y,
    S_33=w_z, S_12=0.5(u_y+v_x), S_13=0.5(u_z+w_x), S_23=0.5(v_z+w_y))
    and ``CD_kw`` uses the 3D cross-diffusion ``k_x w_x + k_y w_y + k_z w_z``.

    ``__call__`` dispatches on ``x_col.shape[1]`` (2 -> :meth:`_call_2d`,
    the original, numerically-unchanged 2D path; 3 -> :meth:`_call_3d`,
    the new 3D path below) so existing 2D callers keep their exact
    residual values. ``self.eddy_viscosity``, ``self._blend_const`` and
    ``SST_CONSTS`` are reused unchanged by both paths -- the constants and
    blending-function machinery are dimension-independent. The unrelated
    1D :meth:`channel_residuals` reduction (used by
    ``templates/33_rans_turbulence.py``) is untouched by any of this.

    Usage::

        residuals = KOmegaSSTResiduals(nu=1e-5, rho=1.0)
        r = residuals(model_uvpkw, x_col)      # x_col: (N,2) -> 2D path
        r3 = residuals(model_uvwpkw, x_col3d)  # x_col3d: (N,3) -> 3D path
        # r is a dict with keys: momentum_x, momentum_y, [momentum_z,]
        #                        continuity, k_eq, omega_eq
        loss = sum(v.pow(2).mean() for v in r.values())
    """

    def __init__(
        self,
        nu: float = 1e-5,
        rho: float = 1.0,
        consts: Optional[Dict[str, float]] = None,
    ) -> None:
        self.nu = nu          # kinematic viscosity
        self.rho = rho        # density
        self.mu = nu * rho    # dynamic viscosity
        self.c = {**SST_CONSTS, **(consts or {})}

    # ------------------------------------------------------------------
    # Blending function helpers (simplified: use inner-layer constants)
    # A full implementation requires wall-distance; here we default F1=0
    # (outer layer), giving the k-omega SST outer constants.
    # Pass consts={'F1': 1.0} to force inner-layer constants.
    # ------------------------------------------------------------------

    def _blend_const(self, name1: str, name2: str, F1: float = 0.0) -> float:
        """Blend constant: phi = F1*phi1 + (1-F1)*phi2."""
        return F1 * self.c[name1] + (1.0 - F1) * self.c[name2]

    def eddy_viscosity(
        self,
        k: torch.Tensor,
        omega: torch.Tensor,
        S_mag: torch.Tensor,
        F2: float = 1.0,
    ) -> torch.Tensor:
        """Compute turbulent viscosity mu_t (Bradshaw limiter included).

        Parameters
        ----------
        k, omega : (N,1) turbulence fields
        S_mag    : (N,1) magnitude of strain-rate tensor S = |S_ij|
        F2       : blending function (0=outer, 1=inner wall region)
        """
        a1 = self.c["a1"]
        denom = torch.clamp(
            torch.maximum(a1 * omega, S_mag * F2),
            min=1e-10,
        )
        return self.rho * a1 * k / denom

    def channel_residuals(
        self,
        y: torch.Tensor,
        U: torch.Tensor,
        k: torch.Tensor,
        omega: torch.Tensor,
        u_tau: float = 1.0,
        delta: float = 1.0,
        F2: float = 1.0,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """1D (wall-normal-only) k-omega SST reduction for fully-developed
        plane channel flow -- every streamwise derivative is zero by the
        fully-developed assumption, and the mean-momentum equation reduces
        to a total-shear-stress balance driven by the constant wall-unit
        pressure gradient ``-u_tau**2/delta`` (see e.g. Pope, *Turbulent
        Flows*, Ch. 7, for the derivation). This is the convenience form a
        1D channel-flow PINN wants; :meth:`__call__` is the general 2D
        form.

        Parameters
        ----------
        y, U, k, omega : (N, 1) tensors; ``y`` must have
            ``requires_grad=True``.
        u_tau, delta : friction velocity and half-channel height (both 1
            in the usual wall-unit nondimensionalisation, u_tau = 1 by
            construction).
        F2 : SST blending function for the eddy-viscosity limiter (see
            :meth:`eddy_viscosity`); 1.0 (inner/wall-region limiter) is the
            right default for a wall-bounded channel.

        Returns
        -------
        ``(res_momentum, res_k, res_omega)``, each (N, 1).
        """
        c = self.c
        U_y = _grad1(U, y)
        S_mag = U_y.abs()
        nu_t = self.eddy_viscosity(k, omega, S_mag, F2=F2)
        mu_eff = self.nu + nu_t

        res_momentum = _grad1(mu_eff * U_y, y) + (u_tau ** 2) / delta

        k_y = _grad1(k, y)
        omega_y = _grad1(omega, y)
        Pk = nu_t * U_y ** 2
        diff_k = _grad1((self.nu + c["sigma_k1"] * nu_t) * k_y, y)
        res_k = diff_k + Pk - c["beta_star"] * k * omega

        diff_w = _grad1((self.nu + c["sigma_w1"] * nu_t) * omega_y, y)
        gamma = self._blend_const("gamma1", "gamma2", F1=1.0)
        res_omega = diff_w + (gamma / nu_t.clamp_min(1e-8)) * Pk - c["beta1"] * omega ** 2

        return res_momentum, res_k, res_omega

    def __call__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        F1: float = 0.0,
        F2: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all RANS + k-omega SST residuals via autograd.

        Dispatches on ``x_col.shape[1]``: 2 -> the original 2D closure
        (:meth:`_call_2d`, ``model: (N,2)->(N,5)``, unchanged from before
        this 3D generalization); 3 -> the new 3D closure (:meth:`_call_3d`,
        ``model: (N,3)->(N,6)``). See the class docstring for both
        equation sets.

        Parameters
        ----------
        model  : callable (N,2) -> (N,5) [2D: (u,v,p,k,omega)] or
                 (N,3) -> (N,6) [3D: (u,v,w,p,k,omega)]
        x_col  : (N,2) or (N,3) collocation points, will get
                 requires_grad=True
        F1, F2 : SST blending functions (scalars; provide per-point tensors
                 if desired by monkey-patching or subclassing)

        Returns
        -------
        dict with keys: 'momentum_x', 'momentum_y', ['momentum_z',]
                        'continuity', 'k_eq', 'omega_eq'
        """
        n_coords = x_col.shape[1]
        if n_coords == 2:
            return self._call_2d(model, x_col, F1=F1, F2=F2)
        elif n_coords == 3:
            return self._call_3d(model, x_col, F1=F1, F2=F2)
        raise ValueError(
            "KOmegaSSTResiduals expects x_col with 2 columns (x,y, 2D) or "
            f"3 columns (x,y,z, 3D); got shape {tuple(x_col.shape)}"
        )

    def _call_2d(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        F1: float = 0.0,
        F2: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Original 2D steady RANS + k-omega SST closure. See
        :meth:`__call__`/the class docstring for the equations;
        numerically unchanged by the 3D generalization."""
        x = x_col.clone().requires_grad_(True)
        out = model(x)                          # (N, 5)
        u     = out[:, 0:1]
        v     = out[:, 1:2]
        p     = out[:, 2:3]
        k     = torch.clamp(out[:, 3:4], min=0.0)    # k >= 0
        omega = torch.clamp(out[:, 4:5], min=1e-6)   # omega > 0

        # ---- first derivatives ----
        u_grad = _grad1(u, x)              # (N, 2)
        v_grad = _grad1(v, x)
        p_grad = _grad1(p, x)
        k_grad = _grad1(k, x)
        w_grad = _grad1(omega, x)

        u_x = u_grad[:, 0:1];  u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1];  v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1];  p_y = p_grad[:, 1:2]
        k_x = k_grad[:, 0:1];  k_y = k_grad[:, 1:2]
        w_x = w_grad[:, 0:1];  w_y = w_grad[:, 1:2]

        # ---- blended constants ----
        sigma_k = self._blend_const("sigma_k1", "sigma_k2", F1)
        sigma_w = self._blend_const("sigma_w1", "sigma_w2", F1)
        beta    = self._blend_const("beta1",    "beta2",    F1)
        gamma   = self._blend_const("gamma1",   "gamma2",   F1)
        beta_s  = self.c["beta_star"]

        # ---- strain-rate magnitude |S| = sqrt(2 S_ij S_ij) ----
        # 2D: S_11=u_x, S_22=v_y, S_12=S_21=0.5(u_y+v_x)
        S_mag = torch.sqrt(
            2.0 * (u_x**2 + v_y**2 + 0.5 * (u_y + v_x)**2) + 1e-12
        )

        # ---- eddy viscosity ----
        nu_t = self.eddy_viscosity(k, omega, S_mag, F2) / self.rho  # kinematic
        mu_t = nu_t * self.rho                                        # dynamic
        mu_eff = self.mu + mu_t

        # ---- production of k: P_k = mu_t * S^2 (with realizability clip) ----
        P_k = torch.clamp(mu_t * S_mag**2, max=10.0 * beta_s * self.rho * k * omega + 1e-12)

        # ---- continuity ----
        cont = u_x + v_y

        # ---- momentum-x/y: rho(u.grad u) + grad p - div[mu_eff(grad u + grad u^T)] = 0 ----
        mu_eff_grad = _grad1(mu_eff, x)         # (N, 2)
        visc_x, visc_y = _viscous_stress_divergence_2d(
            mu_eff, mu_eff_grad, x, u_grad, v_grad, u, v
        )

        mom_x = (
            self.rho * (u * u_x + v * u_y)
            + p_x
            - visc_x
        )

        mom_y = (
            self.rho * (u * v_x + v * v_y)
            + p_y
            - visc_y
        )

        # ---- k-equation ----
        nu_k = self.nu + sigma_k * nu_t
        lap_k = _laplacian(k, x)
        nu_k_grad = _grad1(
            (self.nu + sigma_k * nu_t) * torch.ones_like(k), x
        )
        diff_k = (self.nu + sigma_k * nu_t) * lap_k + nu_k_grad[:, 0:1] * k_x + nu_k_grad[:, 1:2] * k_y
        k_eq = (
            u * k_x + v * k_y
            - diff_k
            - P_k / self.rho
            + beta_s * k * omega
        )

        # ---- omega-equation ----
        # cross-diffusion term CD_kw = 2 sigma_w2 / omega * (k_x w_x + k_y w_y)
        CD_kw = torch.clamp(
            2.0 * self.c["sigma_w2"] / omega * (k_x * w_x + k_y * w_y),
            min=0.0,
        )
        lap_w = _laplacian(omega, x)
        nu_w_grad = _grad1(
            (self.nu + sigma_w * nu_t) * torch.ones_like(omega), x
        )
        diff_w = (self.nu + sigma_w * nu_t) * lap_w + nu_w_grad[:, 0:1] * w_x + nu_w_grad[:, 1:2] * w_y
        omega_eq = (
            u * w_x + v * w_y
            - diff_w
            - gamma * self.rho * S_mag**2
            + beta * omega**2
            - (1.0 - F1) * CD_kw
        )

        return {
            "momentum_x": mom_x,
            "momentum_y": mom_y,
            "continuity":  cont,
            "k_eq":        k_eq,
            "omega_eq":    omega_eq,
        }

    def _call_3d(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        F1: float = 0.0,
        F2: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """3D steady RANS + k-omega SST closure. See the class docstring
        for the equations; a genuine 3D generalization of :meth:`_call_2d`
        (third velocity component, z-derivatives, full 3D strain-rate
        tensor, 3D cross-diffusion) -- not a 2D-plus-padding shortcut.

        Parameters
        ----------
        model  : callable (N,3) -> (N,6): outputs (u, v, w, p, k, omega)
        x_col  : (N,3) collocation points, will get requires_grad=True
        F1, F2 : SST blending functions (scalars)

        Returns
        -------
        dict with keys: 'momentum_x', 'momentum_y', 'momentum_z',
                        'continuity', 'k_eq', 'omega_eq'
        """
        x = x_col.clone().requires_grad_(True)
        out = model(x)                          # (N, 6)
        u     = out[:, 0:1]
        v     = out[:, 1:2]
        w     = out[:, 2:3]
        p     = out[:, 3:4]
        k     = torch.clamp(out[:, 4:5], min=0.0)    # k >= 0
        omega = torch.clamp(out[:, 5:6], min=1e-6)   # omega > 0

        # ---- first derivatives ----
        u_grad  = _grad1(u, x)              # (N, 3)
        v_grad  = _grad1(v, x)
        w_grad  = _grad1(w, x)
        p_grad  = _grad1(p, x)
        k_grad  = _grad1(k, x)
        om_grad = _grad1(omega, x)

        u_x = u_grad[:, 0:1];  u_y = u_grad[:, 1:2];  u_z = u_grad[:, 2:3]
        v_x = v_grad[:, 0:1];  v_y = v_grad[:, 1:2];  v_z = v_grad[:, 2:3]
        w_x = w_grad[:, 0:1];  w_y = w_grad[:, 1:2];  w_z = w_grad[:, 2:3]
        p_x = p_grad[:, 0:1];  p_y = p_grad[:, 1:2];  p_z = p_grad[:, 2:3]
        k_x = k_grad[:, 0:1];  k_y = k_grad[:, 1:2];  k_z = k_grad[:, 2:3]
        om_x = om_grad[:, 0:1]; om_y = om_grad[:, 1:2]; om_z = om_grad[:, 2:3]

        # ---- blended constants ----
        sigma_k = self._blend_const("sigma_k1", "sigma_k2", F1)
        sigma_w = self._blend_const("sigma_w1", "sigma_w2", F1)
        beta    = self._blend_const("beta1",    "beta2",    F1)
        gamma   = self._blend_const("gamma1",   "gamma2",   F1)
        beta_s  = self.c["beta_star"]

        # ---- strain-rate magnitude |S| = sqrt(2 S_ij S_ij), full 3D ----
        # S_11=u_x, S_22=v_y, S_33=w_z,
        # S_12=S_21=0.5(u_y+v_x), S_13=S_31=0.5(u_z+w_x), S_23=S_32=0.5(v_z+w_y)
        S_mag = torch.sqrt(
            2.0 * (u_x**2 + v_y**2 + w_z**2)
            + (u_y + v_x)**2 + (u_z + w_x)**2 + (v_z + w_y)**2
            + 1e-12
        )

        # ---- eddy viscosity ----
        nu_t = self.eddy_viscosity(k, omega, S_mag, F2) / self.rho  # kinematic
        mu_t = nu_t * self.rho                                        # dynamic
        mu_eff = self.mu + mu_t

        # ---- production of k: P_k = mu_t * S^2 (with realizability clip) ----
        P_k = torch.clamp(mu_t * S_mag**2, max=10.0 * beta_s * self.rho * k * omega + 1e-12)

        # ---- continuity ----
        cont = u_x + v_y + w_z

        # ---- momentum-x/y/z: rho(u.grad u) + grad p - div[mu_eff(grad u + grad u^T)] = 0 ----
        mu_eff_grad = _grad1(mu_eff, x)         # (N, 3)
        visc_x, visc_y, visc_z = _viscous_stress_divergence_3d(
            mu_eff, mu_eff_grad, x, u_grad, v_grad, w_grad, u, v, w
        )

        mom_x = (
            self.rho * (u * u_x + v * u_y + w * u_z)
            + p_x
            - visc_x
        )

        mom_y = (
            self.rho * (u * v_x + v * v_y + w * v_z)
            + p_y
            - visc_y
        )

        mom_z = (
            self.rho * (u * w_x + v * w_y + w * w_z)
            + p_z
            - visc_z
        )

        # ---- k-equation ----
        lap_k = _laplacian(k, x)
        nu_k_grad = _grad1(
            (self.nu + sigma_k * nu_t) * torch.ones_like(k), x
        )
        diff_k = (
            (self.nu + sigma_k * nu_t) * lap_k
            + nu_k_grad[:, 0:1] * k_x + nu_k_grad[:, 1:2] * k_y + nu_k_grad[:, 2:3] * k_z
        )
        k_eq = (
            u * k_x + v * k_y + w * k_z
            - diff_k
            - P_k / self.rho
            + beta_s * k * omega
        )

        # ---- omega-equation ----
        # cross-diffusion term CD_kw = 2 sigma_w2 / omega * (k_x w_x + k_y w_y + k_z w_z)
        CD_kw = torch.clamp(
            2.0 * self.c["sigma_w2"] / omega * (k_x * om_x + k_y * om_y + k_z * om_z),
            min=0.0,
        )
        lap_w = _laplacian(omega, x)
        nu_w_grad = _grad1(
            (self.nu + sigma_w * nu_t) * torch.ones_like(omega), x
        )
        diff_w = (
            (self.nu + sigma_w * nu_t) * lap_w
            + nu_w_grad[:, 0:1] * om_x + nu_w_grad[:, 1:2] * om_y + nu_w_grad[:, 2:3] * om_z
        )
        omega_eq = (
            u * om_x + v * om_y + w * om_z
            - diff_w
            - gamma * self.rho * S_mag**2
            + beta * omega**2
            - (1.0 - F1) * CD_kw
        )

        return {
            "momentum_x": mom_x,
            "momentum_y": mom_y,
            "momentum_z": mom_z,
            "continuity":  cont,
            "k_eq":        k_eq,
            "omega_eq":    omega_eq,
        }


# ---------------------------------------------------------------------------
# Feature 10b: Spalart-Allmaras (SA-1992)
# ---------------------------------------------------------------------------

class SpalartAllmarasResiduals:
    """Spalart-Allmaras one-equation turbulence model residuals.

    Transport equation for modified eddy viscosity nu_tilde::

        D nu_tilde / Dt = cb1 * S_tilde * nu_tilde
                        - cw1 * fw * (nu_tilde / d)^2
                        + (1/sigma) * div[(nu + nu_tilde) grad nu_tilde]
                        + (cb2/sigma) |grad nu_tilde|^2

    Usage::

        sa = SpalartAllmarasResiduals(nu=1e-5)
        r = sa(model_uvpn, x_col, wall_distance_fn=lambda x: x[:,1:2])
        # r: dict with 'momentum_x', 'momentum_y', 'continuity', 'sa_eq'
    """

    SA_CONSTS: Dict[str, float] = {
        "cb1":   0.1355,
        "cb2":   0.622,
        "sigma": 2.0 / 3.0,
        "kappa": 0.41,
        "cw2":   0.3,
        "cw3":   2.0,
        "cv1":   7.1,
        "ct3":   1.2,
        "ct4":   0.5,
        # cw1 computed in __init__
    }

    def __init__(self, nu: float = 1e-5) -> None:
        self.nu = nu
        self.c = dict(self.SA_CONSTS)
        cb1   = self.c["cb1"]
        cb2   = self.c["cb2"]
        sigma = self.c["sigma"]
        kappa = self.c["kappa"]
        self.c["cw1"] = cb1 / kappa**2 + (1.0 + cb2) / sigma

    # ------------------------------------------------------------------
    # SA auxiliary functions
    # ------------------------------------------------------------------

    def _fv1(self, chi: torch.Tensor) -> torch.Tensor:
        cv1 = self.c["cv1"]
        return chi**3 / (chi**3 + cv1**3)

    def _fv2(self, chi: torch.Tensor, fv1: torch.Tensor) -> torch.Tensor:
        return 1.0 - chi / (1.0 + chi * fv1)

    def _fw(self, r: torch.Tensor) -> torch.Tensor:
        cw2 = self.c["cw2"]
        cw3 = self.c["cw3"]
        g = r + cw2 * (r**6 - r)
        return g * ((1.0 + cw3**6) / (g**6 + cw3**6)) ** (1.0 / 6.0)

    def __call__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        wall_distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute SA residuals.

        Parameters
        ----------
        model            : callable (N,2) -> (N,4): (u, v, p, nu_tilde)
        x_col            : (N,2) collocation points
        wall_distance_fn : callable (N,2) -> (N,1) returning wall distance d.
                           If None, uses x[:,1:2] (distance from lower wall).

        Returns
        -------
        dict with keys: 'momentum_x', 'momentum_y', 'continuity', 'sa_eq'
        """
        x = x_col.clone().requires_grad_(True)
        out = model(x)                          # (N, 4)
        u      = out[:, 0:1]
        v      = out[:, 1:2]
        p      = out[:, 2:3]
        nu_t   = torch.clamp(out[:, 3:4], min=0.0)   # nu_tilde >= 0

        # wall distance
        if wall_distance_fn is not None:
            d = torch.clamp(wall_distance_fn(x), min=1e-6)
        else:
            d = torch.clamp(x[:, 1:2], min=1e-6)

        # ---- first derivatives ----
        u_grad  = _grad1(u,    x)
        v_grad  = _grad1(v,    x)
        p_grad  = _grad1(p,    x)
        nt_grad = _grad1(nu_t, x)

        u_x = u_grad[:, 0:1];  u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1];  v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1];  p_y = p_grad[:, 1:2]
        nt_x = nt_grad[:, 0:1]; nt_y = nt_grad[:, 1:2]

        # ---- SA auxiliary ----
        chi  = nu_t / self.nu
        fv1  = self._fv1(chi)
        fv2  = self._fv2(chi, fv1)
        kappa = self.c["kappa"]

        # vorticity magnitude omega = |dv/dx - du/dy|
        vort = torch.abs(v_x - u_y) + 1e-12

        # modified strain rate S_tilde = omega + nu_tilde / (kappa^2 d^2) * fv2
        S_tilde = torch.clamp(
            vort + nu_t / (kappa**2 * d**2) * fv2,
            min=1e-10,
        )

        r = torch.clamp(nu_t / (S_tilde * kappa**2 * d**2 + 1e-12), max=10.0)
        fw = self._fw(r)

        # production / destruction
        cb1   = self.c["cb1"]
        cw1   = self.c["cw1"]
        cb2   = self.c["cb2"]
        sigma = self.c["sigma"]

        production   = cb1 * S_tilde * nu_t
        destruction  = cw1 * fw * (nu_t / d)**2

        # diffusion term: (1/sigma) div[(nu + nu_tilde) grad nu_tilde]
        nu_eff = self.nu + nu_t                   # (N,1)
        lap_nt = _laplacian(nu_t, x)
        nu_eff_grad = _grad1(nu_eff, x)           # (N, 2)
        diffusion = (1.0 / sigma) * (
            nu_eff * lap_nt
            + nu_eff_grad[:, 0:1] * nt_x
            + nu_eff_grad[:, 1:2] * nt_y
        )

        # cb2/sigma |grad nu_tilde|^2
        cross = (cb2 / sigma) * (nt_x**2 + nt_y**2)

        sa_eq = (
            u * nt_x + v * nt_y
            - production
            + destruction
            - diffusion
            - cross
        )

        # ---- RANS momentum with SA eddy viscosity ----
        nu_turb = nu_t * fv1                      # actual eddy viscosity
        nu_total = self.nu + nu_turb
        nu_tot_grad = _grad1(nu_total, x)
        visc_x, visc_y = _viscous_stress_divergence_2d(
            nu_total, nu_tot_grad, x, u_grad, v_grad, u, v
        )

        mom_x = (
            u * u_x + v * u_y
            + p_x
            - visc_x
        )
        mom_y = (
            u * v_x + v * v_y
            + p_y
            - visc_y
        )
        cont = u_x + v_y

        return {
            "momentum_x": mom_x,
            "momentum_y": mom_y,
            "continuity":  cont,
            "sa_eq":       sa_eq,
        }


def _laplacian_indices(f: torch.Tensor, x: torch.Tensor, indices) -> torch.Tensor:
    """Like ``_laplacian``, but summed only over the given column indices
    of ``x`` -- needed once ``x`` carries a non-spatial column (time) that
    must NOT be included in a *spatial* Laplacian. ``_laplacian`` itself is
    left untouched (it sums every column of ``x``, which is correct for
    the RANS closures above -- their ``x`` is purely spatial, no time
    column -- so changing its default would be a silent behaviour change
    for existing callers).
    """
    gf = _grad1(f, x)
    lap = torch.zeros(f.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in indices:
        g_i = gf[:, i : i + 1]
        # A field that is exactly linear in x has a *constant* first
        # derivative, which carries no autograd graph: its second derivative
        # is zero, but asking autograd for it raises instead of saying so.
        # Skipping the term is the correct contribution, and keeps the
        # residual evaluable on an analytic (non-network) field -- which is
        # how these closures are checked against their closed forms.
        if g_i.requires_grad:
            lap = lap + _grad1(g_i, x)[:, i : i + 1]
    return lap


# ---------------------------------------------------------------------------
# LES: one filtered-Navier-Stokes assembly, three algebraic SGS closures
# ---------------------------------------------------------------------------

def _dev3(A: torch.Tensor) -> torch.Tensor:
    """Deviatoric (traceless) part of a batch of 3x3 tensors ``(N, 3, 3)``."""
    tr = (A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]) / 3.0
    eye = torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0)
    return A - tr.view(-1, 1, 1) * eye


def filter_stencil(
    x: torch.Tensor,
    neighbors: int = 8,
    kernel: str = "gaussian",
    bandwidth: Optional[float] = None,
    spatial_indices=(0, 1, 2),
):
    """Neighbour stencil for a mesh-free (scattered-point) spatial filter.

    A PINN is evaluated on scattered collocation points, so the box/Gaussian
    test filter the dynamic procedure needs cannot be a grid stencil. This
    builds the equivalent from the point cloud itself: the ``k`` nearest
    neighbours of every point (in the spatial coordinates only -- the time
    column, if present, is not part of the distance), weighted by ``kernel``
    and normalised to sum to one, so a constant field filters to itself on
    every point.

    Parameters
    ----------
    x : torch.Tensor
        ``(N, D)`` sample points.
    neighbors : int
        Stencil size ``k`` (including the point itself). ``k = 1`` is the
        identity filter, which carries no scale information at all -- the
        dynamic procedure's ``L_ij`` is then identically zero by
        construction -- so ``k < 2`` is rejected outright rather than
        silently returning a zero eddy viscosity.
    kernel : {"gaussian", "box"}
        ``"box"`` gives every neighbour the same weight; ``"gaussian"``
        weights it by ``exp(-r^2 / 2 bandwidth^2)``.
    bandwidth : float or array-like, optional
        Gaussian kernel width, in the units of ``x``: one scalar (same
        width everywhere) or one value per point ``(N,)``. Defaults to the
        median stencil radius (one filter width); ignored by ``"box"``.
    spatial_indices : sequence of int
        Columns of ``x`` that are spatial and therefore enter the distance.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(idx, w)``, both ``(N, k)``: the neighbour indices and their
        normalised weights. Deliberately returned as numpy (the stencil is a
        property of the point cloud, not of the network's parameters) and
        used only to gather/average *values*, so the filter itself stays
        differentiable through the gathered tensors.
    """
    import numpy as np

    pts = np.asarray(x.detach().cpu(), dtype=np.float64)[:, list(spatial_indices)]
    n = pts.shape[0]
    k = int(min(max(int(neighbors), 1), n))
    if k < 2:
        raise ValueError(
            "a filter stencil needs at least 2 points (the point itself plus one neighbour); "
            f"got neighbors={neighbors} on a batch of {n} points"
        )

    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(pts).query(pts, k=k)
    except ImportError:  # pragma: no cover - scipy is a core dependency
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        idx = np.argsort(d, axis=1)[:, :k]
        dist = np.take_along_axis(d, idx, axis=1)
    dist = np.asarray(dist, dtype=np.float64).reshape(n, k)
    idx = np.asarray(idx).reshape(n, k)

    if kernel == "box":
        w = np.ones_like(dist)
    elif kernel == "gaussian":
        if bandwidth is None:
            sigma = np.full(n, float(np.median(dist[:, -1])))
        else:
            bw = bandwidth.detach().cpu().numpy() if hasattr(bandwidth, "detach") else np.asarray(bandwidth)
            sigma = np.broadcast_to(np.asarray(bw, dtype=np.float64).reshape(-1), (n,))
        sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
        w = np.exp(-0.5 * (dist / sigma[:, None]) ** 2)
    else:
        raise ValueError(f"kernel must be 'box' or 'gaussian'; got {kernel!r}")

    w = w / w.sum(axis=1, keepdims=True)
    return idx, w


def apply_stencil(F: torch.Tensor, idx, w) -> torch.Tensor:
    """Filter point values ``F`` (``(N, ...)``) with a ``(N, k)`` stencil.

    Differentiable in ``F``: this is a weighted gather over the *values*
    already evaluated at every point of the batch, so no extra forward pass
    through the network is needed and gradients reach the model parameters
    through the filter.
    """
    idx_t = torch.as_tensor(idx, device=F.device)
    w_t = torch.as_tensor(w, device=F.device, dtype=F.dtype)
    gathered = F[idx_t]                                        # (N, k, ...)
    w_t = w_t.reshape(w_t.shape + (1,) * (F.dim() - 1))
    return (gathered * w_t).sum(dim=1)


class _LESResiduals3D:
    """Shared residual assembly for algebraic 3D large-eddy closures.

    Every algebraic SGS model differs from the others in exactly one place:
    how the subgrid viscosity ``nu_sgs`` is obtained from the resolved
    fields. Everything else -- the momentum convection, the pressure
    gradient, the diffusion by the *effective* viscosity ``nu + nu_sgs``
    (including the ``grad nu_eff`` terms a spatially varying viscosity
    brings, which dominate near a wall) and continuity -- is identical.
    This base class holds that common part once, so :class:`Smagorinsky
    Residuals`, :class:`DynamicSmagorinskyResiduals` and :class:`WALE
    Residuals` below stay comparable term by term: same equation, same
    discretisation (autograd), differing only in the closure.

    Subclasses implement :meth:`sgs_viscosity` and nothing else. The
    residual returned is that of the *filtered* incompressible
    Navier-Stokes equations, with the SGS stress model folded into an
    effective viscosity::

        du_i/dt + u_j du_i/dx_j = -dp/dx_i + d/dx_j[(nu + nu_sgs)
                                    (du_i/dx_j + du_j/dx_i)] + f_i
        du_i/dx_i = 0

    with ``p`` understood as pressure over density. That is the standard
    eddy-viscosity form of the filtered equations, and it is a modelling
    choice shared by all three closures here -- not the general SGS stress
    divergence ``d tau_ij/dx_j``, which an eddy-viscosity model happens to
    write that way.
    """

    def __init__(
        self,
        nu: float = 1e-5,
        delta: float = 1.0,
        spatial_indices=(0, 1, 2),
        t_index: int = 3,
    ) -> None:
        self.nu = float(nu)
        self.delta = delta
        self.spatial_indices = list(spatial_indices)
        self.t_index = int(t_index)

    def _delta_at(self, x: torch.Tensor) -> torch.Tensor:
        if callable(self.delta):
            return self.delta(x)
        return torch.full((x.shape[0], 1), float(self.delta), device=x.device, dtype=x.dtype)

    def sgs_viscosity(
        self,
        g: torch.Tensor,
        S: torch.Tensor,
        U: torch.Tensor,
        delta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Subgrid viscosity ``(N, 1)`` from the resolved velocity gradient.

        Parameters
        ----------
        g : torch.Tensor
            ``(N, 3, 3)`` resolved velocity gradient, ``g[:, i, j] = du_i/dx_j``.
        S : torch.Tensor
            ``(N, 3, 3)`` symmetric strain-rate tensor, ``S = (g + g^T)/2``.
        U : torch.Tensor
            ``(N, 3)`` resolved velocity components.
        delta : torch.Tensor
            ``(N, 1)`` filter width at each point (constant or mesh-derived).
        x : torch.Tensor
            ``(N, D)`` sample points, for closures that filter (depend on
            neighbouring points) rather than being purely local.
        """
        raise NotImplementedError

    def __call__(self, model, x: torch.Tensor, forcing: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        sp = self.spatial_indices
        y = model(x)
        u, v, w, p = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
        U = torch.cat([u, v, w], dim=1)

        ut = _grad1(u, x)[:, self.t_index : self.t_index + 1]
        vt = _grad1(v, x)[:, self.t_index : self.t_index + 1]
        wt = _grad1(w, x)[:, self.t_index : self.t_index + 1]

        g = torch.stack([_grad1(U[:, i : i + 1], x)[:, sp] for i in range(3)], dim=1)  # g[:,i,j]=dU_i/dx_j

        conv = torch.zeros_like(U)
        for i in range(3):
            for j in range(3):
                conv[:, i] = conv[:, i] + U[:, j] * g[:, i, j]

        gp = _grad1(p, x)
        px, py, pz = gp[:, sp[0] : sp[0] + 1], gp[:, sp[1] : sp[1] + 1], gp[:, sp[2] : sp[2] + 1]

        delta = self._delta_at(x)
        S = 0.5 * (g + g.transpose(1, 2))
        nu_sgs = self.sgs_viscosity(g, S, U, delta, x)
        nu_eff = self.nu + nu_sgs

        # A closure can return a *constant* nu_sgs (e.g. a linear velocity
        # field, where the strain -- and therefore every algebraic SGS
        # viscosity -- is spatially uniform). That constant carries no
        # autograd graph, so differentiating it w.r.t. x raises instead of
        # returning the zero gradient that is the correct answer; keep the
        # zero-gradient case explicit rather than turning it into an error
        # in the middle of a physically valid evaluation.
        if nu_eff.requires_grad:
            gnu = _grad1(nu_eff, x)
        else:
            gnu = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        nu_x, nu_y, nu_z = gnu[:, sp[0] : sp[0] + 1], gnu[:, sp[1] : sp[1] + 1], gnu[:, sp[2] : sp[2] + 1]
        gdotS_x = nu_x * S[:, 0, 0:1] + nu_y * S[:, 0, 1:2] + nu_z * S[:, 0, 2:3]
        gdotS_y = nu_x * S[:, 1, 0:1] + nu_y * S[:, 1, 1:2] + nu_z * S[:, 1, 2:3]
        gdotS_z = nu_x * S[:, 2, 0:1] + nu_y * S[:, 2, 1:2] + nu_z * S[:, 2, 2:3]

        lap_u = _laplacian_indices(u, x, sp)
        lap_v = _laplacian_indices(v, x, sp)
        lap_w = _laplacian_indices(w, x, sp)

        diff_u = nu_eff * lap_u + 2.0 * gdotS_x
        diff_v = nu_eff * lap_v + 2.0 * gdotS_y
        diff_w = nu_eff * lap_w + 2.0 * gdotS_z

        f = forcing if forcing is not None else torch.zeros(3, device=x.device, dtype=x.dtype)
        r_mom_u = ut + conv[:, 0:1] + px - diff_u - f[0]
        r_mom_v = vt + conv[:, 1:2] + py - diff_v - f[1]
        r_mom_w = wt + conv[:, 2:3] + pz - diff_w - f[2]
        r_cont = g[:, 0, 0:1] + g[:, 1, 1:2] + g[:, 2, 2:3]

        return {
            "momentum_x": r_mom_u,
            "momentum_y": r_mom_v,
            "momentum_z": r_mom_w,
            "continuity": r_cont,
            "nu_sgs": nu_sgs.detach(),
        }


def van_driest(y_plus: torch.Tensor, a_plus: float = 25.0) -> torch.Tensor:
    """Van Driest wall damping ``D = (1 - exp(-y+/A+))^2``.

    Offered as a composable factor rather than a hidden default: the
    damping needs the wall distance *and* the friction velocity, which are
    problem data, not closure data. Pass it as
    ``damping=lambda x: van_driest(y_plus_fn(x))`` and the closure applies
    ``D`` to its ``nu_sgs``. ``A+ = 25`` is the classical Van Driest value.
    """
    return (1.0 - torch.exp(-y_plus.clamp_min(0.0) / float(a_plus))) ** 2


class SmagorinskyResiduals(_LESResiduals3D):
    """Smagorinsky (1963) subgrid closure for filtered incompressible
    Navier-Stokes, as a 3D LES residual in the same shape as
    :class:`WALEResiduals` -- the constant-coefficient model the dynamic
    procedure below generalises, and the one OpenFOAM spells
    ``Smagorinsky`` with ``Cs`` in ``constant/turbulenceProperties``.

    ::

        nu_sgs = (Cs*Delta)^2 * |S|,      |S| = sqrt(2 S_ij S_ij)

    ``Cs = 0.17`` is the classical Lilly (1966) value for homogeneous
    turbulence, and also the default used by this repo's LBM solvers
    (``LBMSolver``'s ``Cs``, Hou 1994) -- the same number, so a PINN-LES
    and an LBM-LES of the same flow start from the same closure.

    Two arguments exist because the plain model fails in two well-known
    places, and both failures are cheap to state explicitly rather than
    hide:

    * ``damping`` -- the constant model does not vanish at a wall as it
      must (``nu_sgs ~ y^3``). Pass ``damping=lambda x: van_driest(y_plus(x))``
      to recover the Van Driest-damped model. Off by default: it needs
      ``y+``, which only the problem knows.
    * ``delta`` -- a constant filter width is right only for a uniform
      mesh. Pass a callable (``delta_fn(x) -> (N, 1)``) for a real mesh,
      e.g. from ``cubeRootVol`` per-cell sizes read by
      ``pinneapple_simulation.external_solvers.openfoam.mesh_reader``.

    Usage::

        residuals = SmagorinskyResiduals(nu=1e-5, cs=0.17, delta=0.01)
        r = residuals(model_uvwp, x_col)          # x_col: (N, 4) = (x, y, z, t)
        loss = sum(v.pow(2).mean() for k, v in r.items() if k != "nu_sgs")

    ``forcing`` (a length-3 tensor, e.g. a learned
    ``torch.nn.Parameter``) plays the same role as in
    :class:`WALEResiduals`: a uniform driving source for a periodic
    channel/duct held at a target bulk velocity.
    """

    def __init__(
        self,
        nu: float = 1e-5,
        cs: float = 0.17,
        delta: float = 1.0,
        damping: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        spatial_indices=(0, 1, 2),
        t_index: int = 3,
    ) -> None:
        super().__init__(nu=nu, delta=delta, spatial_indices=spatial_indices, t_index=t_index)
        self.cs = float(cs)
        self.damping = damping

    def sgs_viscosity(self, g, S, U, delta, x):
        SS = (S * S).sum(dim=(1, 2))
        magS = torch.sqrt(2.0 * SS.clamp_min(0.0)).reshape(-1, 1)   # |S| = sqrt(2 S:S)
        nu_sgs = (self.cs * delta.reshape(-1, 1)) ** 2 * magS
        if self.damping is not None:
            nu_sgs = nu_sgs * self.damping(x).reshape(-1, 1)
        return nu_sgs


class DynamicSmagorinskyResiduals(_LESResiduals3D):
    """Germano-identity (dynamic) Smagorinsky closure -- Germano et al.
    (1991), least-squares coefficient by Lilly (1992) -- as a 3D LES
    residual, with the coefficient computed at run time from the resolved
    field instead of fixed a priori.

    The idea the material calls out for ``Dynamic Smagorinsky``: the same
    Smagorinsky ansatz at two filter widths, ``Delta`` and the test width
    ``DeltaHat = test_ratio * Delta``, gives a relation between the
    test-filtered resolved stresses and the two model stresses that can be
    solved for ``C = Cs^2``::

        L_ij  = u_i u_j (hat) - u_hat_i u_hat_j
        M_ij  = (DeltaHat/Delta)^2 |S_hat| S_hat_ij - (|S| S_ij) (hat)
        C     = -<L_ij M_ij> / (2 Delta^2 <M_ij M_ij>)
        nu_sgs = C * Delta^2 * |S|

    so the model adapts: it switches itself off where the flow is
    resolved or laminar (``C -> 0``), and grows in the strained,
    under-resolved regions, without the constant ``Cs`` ever being fitted
    to a particular case. This is the property that makes it the default
    LES closure in production codes.

    **The filter on scattered points.** The test filter is the one part of
    the dynamic procedure that a point-sampled PINN does not get for free:
    there is no grid to apply a box filter on. It is built here from the
    collocation point cloud itself (``filter_stencil``: k nearest
    neighbours, Gaussian or box weights, normalised), which is exactly the
    mesh-free analogue and stays differentiable -- filtering happens over
    *values already evaluated* at every point of the batch, so one forward
    pass per step is still enough. Its width follows the filter width by
    default (``bandwidth = test_ratio * Delta``), so ``Delta`` really is
    the scale being split at, rather than a constant that cancels out of
    the Germano identity.

    Two approximations are made and are worth stating plainly, because
    they are the model's, not an implementation shortcut:

    * the test-level strain is taken from the filtered velocity gradient,
      ``S_hat ~= strain(filter(g))`` -- the usual commutation assumption.
      It is exact for a uniform filter in a homogeneous direction.
    * ``<.>`` is a spatial average. ``averaging="batch"`` (default)
      averages over the whole point batch, i.e. assumes the batch spans
      homogeneous directions, which is what the dynamic procedure requires
      and what a plane-averaged channel/duct LES does. ``averaging="none"``
      computes a pointwise coefficient instead -- admissible, but noisier
      and prone to the negative-viscosity excursions the averaging exists
      to suppress.

    ``clip=True`` (Lilly) floors ``C`` at zero: a negative ``nu_sgs`` is
    anti-diffusive and, in a transient PINN solve, destabilising. Set it
    to ``False`` to see the unclipped coefficient (useful as a diagnostic
    of where the model would have backscattered).

    ``stencil=(idx, w)`` (from :func:`filter_stencil`) can be passed once
    and reused: for a fixed collocation set -- the usual case when the
    training points are sampled once and held -- this skips rebuilding the
    neighbour search on every residual evaluation. Leave it ``None`` to
    build it from ``x`` on each call.

    Usage::

        residuals = DynamicSmagorinskyResiduals(nu=1e-5, delta=0.01, test_ratio=2.0)
        r = residuals(model_uvwp, x_col)
        r["nu_sgs"]                      # local eddy viscosity, (N, 1)
        residuals.last_germano["C"]      # the coefficient actually used

    ``self.last_germano`` keeps the ``L``, ``M`` and ``C`` of the most
    recent call (detached) for inspection: ``C`` is where you check that
    the model is not silently clipping everything to zero, and it is what
    the least-squares optimality test in the test suite checks.
    """

    def __init__(
        self,
        nu: float = 1e-5,
        delta: float = 1.0,
        test_ratio: float = 2.0,
        neighbors: int = 8,
        kernel: str = "gaussian",
        bandwidth: Optional[float] = None,
        averaging: str = "batch",
        clip: bool = True,
        cs_max: Optional[float] = None,
        deviatoric: bool = True,
        stencil: Optional[tuple] = None,
        spatial_indices=(0, 1, 2),
        t_index: int = 3,
    ) -> None:
        super().__init__(nu=nu, delta=delta, spatial_indices=spatial_indices, t_index=t_index)
        if float(test_ratio) <= 1.0:
            raise ValueError(
                f"test_ratio must be > 1 (the test filter has to be wider than the grid filter); got {test_ratio}"
            )
        if averaging not in ("batch", "none"):
            raise ValueError(f"averaging must be 'batch' or 'none'; got {averaging!r}")
        self.test_ratio = float(test_ratio)
        self.neighbors = int(neighbors)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.averaging = averaging
        self.clip = bool(clip)
        self.cs_max = None if cs_max is None else float(cs_max)
        self.deviatoric = bool(deviatoric)
        if stencil is None:
            self.stencil = None
        else:
            import numpy as np

            self.stencil = (np.asarray(stencil[0]), np.asarray(stencil[1]))
        self.last_germano: Dict[str, Optional[torch.Tensor]] = {"L": None, "M": None, "C": None}

    def sgs_viscosity(self, g, S, U, delta, x):
        sp = self.spatial_indices
        d = delta.reshape(-1, 1)
        # The test filter has to be *wider than the grid filter*, so its
        # width follows Delta by default (DeltaHat = test_ratio * Delta); an
        # explicit `bandwidth` overrides it.
        if self.stencil is not None:
            idx, w = self.stencil
        else:
            bandwidth = self.bandwidth if self.bandwidth is not None else (self.test_ratio * d).reshape(-1)
            idx, w = filter_stencil(
                x,
                neighbors=self.neighbors,
                kernel=self.kernel,
                bandwidth=bandwidth,
                spatial_indices=sp,
            )

        ratio2 = self.test_ratio ** 2

        # Test-filtered quantities: values only (commutation assumption for S_hat).
        U_hat = apply_stencil(U, idx, w)                              # (N, 3)
        uu_hat = apply_stencil(U.unsqueeze(-1) * U.unsqueeze(-2), idx, w)   # (N, 3, 3)
        g_hat = apply_stencil(g, idx, w)                              # (N, 3, 3)
        S_hat = 0.5 * (g_hat + g_hat.transpose(1, 2))
        magS_hat = torch.sqrt(2.0 * (S_hat * S_hat).sum(dim=(1, 2)).clamp_min(0.0)).reshape(-1, 1)

        magS = torch.sqrt(2.0 * (S * S).sum(dim=(1, 2)).clamp_min(0.0)).reshape(-1, 1)

        # L_ij = filter(u_i u_j) - filter(u_i) filter(u_j)
        L = uu_hat - U_hat.unsqueeze(-1) * U_hat.unsqueeze(-2)
        # M_ij = (DeltaHat/Delta)^2 |S_hat| S_hat_ij - filter(|S| S_ij)
        M = ratio2 * magS_hat.unsqueeze(-1) * S_hat - apply_stencil(magS.unsqueeze(-1) * S, idx, w)

        if self.deviatoric:
            L = _dev3(L)
            M = _dev3(M)

        LM = (L * M).sum(dim=(1, 2))
        MM = (M * M).sum(dim=(1, 2))
        d2 = d.reshape(-1) ** 2

        # Least squares on L_ij = -2 C Delta^2 M_ij  ->  C = -<L:M> / (2 Delta^2 <M:M>)
        if self.averaging == "batch":
            denom = d2.mean() * MM.mean()
            C = (-0.5 * LM.mean() / denom) if float(denom.detach()) != 0.0 else LM.new_zeros(())
        else:
            C = torch.where(MM > 0.0, -0.5 * LM / (d2 * MM.clamp_min(1e-300)), torch.zeros_like(LM))

        if self.clip:
            C = C.clamp_min(0.0)
        if self.cs_max is not None:
            C = C.clamp_max(self.cs_max ** 2)

        nu_sgs = (C * d2).reshape(-1, 1) * magS
        self.last_germano = {"L": L.detach(), "M": M.detach(), "C": C.detach()}
        return nu_sgs


# ---------------------------------------------------------------------------
# Feature 10c: WALE (Wall-Adapting Local Eddy-viscosity) LES closure, 3D
# ---------------------------------------------------------------------------

class WALEResiduals(_LESResiduals3D):
    """WALE (Wall-Adapting Local Eddy-viscosity) subgrid closure for
    filtered incompressible Navier-Stokes -- Nicoud & Ducros (1999) -- as
    an algebraic (no transport equation) 3D LES residual.

    Unlike the RANS closures above, WALE needs no extra transported
    unknowns: the eddy viscosity is a pointwise algebraic function of the
    resolved velocity-gradient tensor, so the model only maps
    ``(x, y, z, t) -> (u, v, w, p)`` -- and, unlike the 2D-only RANS
    closures in this module, the residual itself is a genuine 3D one.
    It is one of three 3D LES closures here, sharing one assembly
    (:class:`_LESResiduals3D`) with :class:`SmagorinskyResiduals` and
    :class:`DynamicSmagorinskyResiduals`, so the three can be compared on
    the same flow term by term; WALE is the one that needs no ``Cs`` and
    no damping to give the right ``y^3`` wall behaviour.

    ::

        nu_sgs = (Cw*Delta)^2 * (Sd:Sd)^{3/2} / ((S:S)^{5/2} + (Sd:Sd)^{5/4})

    where ``S`` is the resolved strain-rate tensor and ``Sd`` is the
    traceless symmetric part of the squared velocity-gradient tensor.
    ``Delta`` is the LES filter width; pass either a constant (uniform
    mesh) or a callable ``delta_fn(x) -> (N, 1)`` (e.g. a mesh-derived
    ``cubeRootVol`` lookup -- see ``pinneapple_simulation.external_solvers
    .openfoam.mesh_reader`` for the per-cell sizes such a lookup would be
    built from).

    Usage::

        residuals = WALEResiduals(nu=1e-5, cw=0.325, delta=0.01)
        r = residuals(model_uvwp, x_col)   # x_col: (N, 4) = (x, y, z, t)
        loss = sum(v.pow(2).mean() for k, v in r.items() if k != "nu_sgs")

    A spatially-uniform, time-independent driving source (matching
    OpenFOAM's ``meanVelocityForce`` fvOption for a periodic channel/duct
    held to a target bulk velocity) can be folded in via the optional
    ``forcing`` argument to ``__call__`` -- e.g. a
    ``torch.nn.Parameter(torch.zeros(3))`` learned jointly with the
    network and pinned down by an auxiliary bulk-velocity penalty in the
    training loop (that penalty is problem data, not part of this
    closure, so it is not computed here).
    """

    def __init__(
        self,
        nu: float = 1e-5,
        cw: float = 0.325,
        delta=1.0,
        spatial_indices=(0, 1, 2),
        t_index: int = 3,
    ) -> None:
        super().__init__(nu=nu, delta=delta, spatial_indices=spatial_indices, t_index=t_index)
        self.cw = cw

    def sgs_viscosity(self, g, S, U, delta, x):
        g2 = torch.einsum("nij,njk->nik", g, g)
        trace_g2 = (g2[:, 0, 0] + g2[:, 1, 1] + g2[:, 2, 2]) / 3.0
        eye = torch.eye(3, device=g.device, dtype=g.dtype).unsqueeze(0)
        Sd = 0.5 * (g2 + g2.transpose(1, 2)) - trace_g2.view(-1, 1, 1) * eye
        SS = (S * S).sum(dim=(1, 2))
        SdSd = (Sd * Sd).sum(dim=(1, 2))
        num = (self.cw * delta.reshape(-1)) ** 2 * SdSd.clamp_min(0.0).pow(1.5)
        den = SS.clamp_min(0.0).pow(2.5) + SdSd.clamp_min(0.0).pow(1.25) + 1e-12
        return (num / den).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_rans_preset(model_name: str = "k-omega-sst", **kwargs):
    """Get a RANS/LES turbulence closure residual object by name.

    Parameters
    ----------
    model_name : one of 'k-omega-sst', 'spalart-allmaras', 'sa',
        'smagorinsky', 'dynamic-smagorinsky' (alias 'dsmag'), 'wale'
    **kwargs   : forwarded to the residual class constructor

    Returns
    -------
    Callable residual object (KOmegaSSTResiduals, SpalartAllmarasResiduals,
    SmagorinskyResiduals, DynamicSmagorinskyResiduals or WALEResiduals)

    Examples
    --------
    >>> res = get_rans_preset("k-omega-sst", nu=1e-5, rho=1.0)
    >>> res = get_rans_preset("spalart-allmaras", nu=1.5e-5)
    >>> res = get_rans_preset("smagorinsky", nu=1e-5, cs=0.17, delta=0.01)
    >>> res = get_rans_preset("dynamic-smagorinsky", nu=1e-5, delta=0.01)
    >>> res = get_rans_preset("wale", nu=1e-5, cw=0.325, delta=0.01)
    """
    _models = {
        "k-omega-sst":          KOmegaSSTResiduals,
        "spalart-allmaras":     SpalartAllmarasResiduals,
        "sa":                   SpalartAllmarasResiduals,
        "smagorinsky":          SmagorinskyResiduals,
        "dynamic-smagorinsky":  DynamicSmagorinskyResiduals,
        "dsmag":                DynamicSmagorinskyResiduals,
        "wale":                 WALEResiduals,
    }
    if model_name not in _models:
        raise ValueError(
            f"Unknown RANS/LES model: '{model_name}'. "
            f"Available: {sorted(_models.keys())}"
        )
    return _models[model_name](**kwargs)


__all__ = [
    "SST_CONSTS",
    "KOmegaSSTResiduals",
    "SpalartAllmarasResiduals",
    "SmagorinskyResiduals",
    "DynamicSmagorinskyResiduals",
    "WALEResiduals",
    "van_driest",
    "filter_stencil",
    "apply_stencil",
    "get_rans_preset",
]
