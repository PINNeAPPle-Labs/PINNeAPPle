"""
PDE-residual losses of the arena problems, ``(model, {"X": collocation points}) -> (loss, info)``.

Loss lookup order for each problem:
  1. Native PyTorch autograd residual of this module, built with the experiment's parameters (nu, alpha, c, ...)
  2. The PINNeAPPle preset's compiled PDE residual (``pinneapple_physics.pinn_solver.compile_problem``) via the
     explicit preset_key, then via problem_id; it uses the preset's own parameters, and only its ``pde`` term is
     used (boundary and initial conditions come in through the data). The preset's coordinate order must match
     the arena data (e.g. ``(x, t)`` for Burgers).
  3. None  (supervised-only training)

The source arena tried the preset first, but called it with an outdated signature, so every preset loss raised
and was skipped silently; the wrapper below adapts it, and the native residual (parameter-aware) comes first.
"""
# ruff: noqa: E702  (residuals keep one derivative per statement on one line)
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pinneapple_physics.pinn_solver import LossWeights, compile_problem
    HAS_PINNEAPPLE = True
except ImportError:
    HAS_PINNEAPPLE = False
    compile_problem = None
    LossWeights = None

try:
    import pinneapple as pp
    HAS_PP = True
except ImportError:
    HAS_PP = False
    pp = None


# ---------------------------------------------------------------------------
# Native PyTorch autograd physics loss implementations
# ---------------------------------------------------------------------------

def _forward(model: Any, X: Any) -> Any:
    """Unified model forward pass — handles tuple/namedtuple/attribute returns."""
    try:
        out = model(X)
    except TypeError:
        inputs = [X[:, i : i + 1] for i in range(X.shape[1])]
        out = model(*inputs)
    if hasattr(out, "y"):
        return out.y
    if isinstance(out, tuple):
        out = out[0]
    if HAS_TORCH:
        if out.dim() == 1:
            out = out.unsqueeze(-1)
    return out


def _grad(output: Any, input_: Any) -> Any:
    """∂output/∂input_ via autograd with create_graph=True for higher-order derivatives."""
    if not HAS_TORCH:
        return None
    import torch
    g = torch.autograd.grad(
        output, input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(input_) if g is None else g


# Each factory takes (params, w_pde, w_bc, w_ic) and returns loss_fn(model, batch)
# loss_fn returns (loss_tensor, info_dict)

def _make_burgers_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_t + u·u_x − ν·u_xx = 0"""
    nu = float(params.get("nu", 0.01 / math.pi))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        r = u_t + u * u_x - nu * u_xx
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_burgers": float(loss.item())}

    return _loss


def _make_heat_1d_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_t − α·u_xx = 0"""
    alpha = float(params.get("alpha", 0.1))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        r = u_t - alpha * u_xx
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_heat_1d": float(loss.item())}

    return _loss


def _make_heat_2d_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_t − α·(u_xx + u_yy) = 0"""
    alpha = float(params.get("alpha", 0.1))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 3): [x, y, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_y = du[:, 1:2];  u_t = du[:, 2:3]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_yy = _grad(u_y, X)[:, 1:2]
        r = u_t - alpha * (u_xx + u_yy)
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_heat_2d": float(loss.item())}

    return _loss


def _make_wave_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_tt − c²·u_xx = 0"""
    c = float(params.get("c", 1.0))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_tt = _grad(u_t, X)[:, 1:2]
        r = u_tt - c ** 2 * u_xx
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_wave": float(loss.item())}

    return _loss


def _make_ns_2d_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """Steady 2D incompressible NS: x-momentum + y-momentum + continuity"""
    nu = 1.0 / float(params.get("Re", 40.0))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, y]
        out = _forward(model, X)
        if out.shape[1] < 3:
            return torch.tensor(0.0, device=X.device, requires_grad=True), {}
        u = out[:, 0:1];  v = out[:, 1:2];  p = out[:, 2:3]
        du = _grad(u, X);  dv = _grad(v, X);  dp = _grad(p, X)
        u_x = du[:, 0:1];  u_y = du[:, 1:2]
        v_x = dv[:, 0:1];  v_y = dv[:, 1:2]
        p_x = dp[:, 0:1];  p_y = dp[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1];  u_yy = _grad(u_y, X)[:, 1:2]
        v_xx = _grad(v_x, X)[:, 0:1];  v_yy = _grad(v_y, X)[:, 1:2]
        r_u   = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        r_v   = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        r_div = u_x + v_y
        loss = w_pde * torch.mean(r_u ** 2 + r_v ** 2 + r_div ** 2)
        return loss, {"pde_ns_2d": float(loss.item())}

    return _loss


def _make_poisson_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """−(u_xx + u_yy) = f  with f = 2π²·sin(πx)·sin(πy)"""
    pi = math.pi

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, y]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_y = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_yy = _grad(u_y, X)[:, 1:2]
        f = 2.0 * pi ** 2 * torch.sin(pi * X[:, 0:1]) * torch.sin(pi * X[:, 1:2])
        r = -(u_xx + u_yy) - f
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_poisson": float(loss.item())}

    return _loss


def _make_laplace_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_xx + u_yy = 0"""
    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_y = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_yy = _grad(u_y, X)[:, 1:2]
        r = u_xx + u_yy
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_laplace": float(loss.item())}

    return _loss


def _make_helmholtz_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_xx + u_yy + k²·u = 0"""
    m = int(params.get("m", 1))
    n = int(params.get("n_mode", 1))
    k2 = float((m ** 2 + n ** 2) * math.pi ** 2)

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_y = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_yy = _grad(u_y, X)[:, 1:2]
        r = u_xx + u_yy + k2 * u
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_helmholtz": float(loss.item())}

    return _loss


def _make_kdv_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_t + 6·u·u_x + u_xxx = 0"""
    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx  = _grad(u_x,  X)[:, 0:1]
        u_xxx = _grad(u_xx, X)[:, 0:1]
        r = u_t + 6.0 * u * u_x + u_xxx
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_kdv": float(loss.item())}

    return _loss


def _make_klein_gordon_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """u_tt − c²·u_xx + m²·u = 0"""
    c = float(params.get("c", 1.0))
    m = float(params.get("m", 1.0))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 2): [x, t]
        u = _forward(model, X)[:, 0:1]
        du = _grad(u, X)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx = _grad(u_x, X)[:, 0:1]
        u_tt = _grad(u_t, X)[:, 1:2]
        r = u_tt - c ** 2 * u_xx + m ** 2 * u
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_klein_gordon": float(loss.item())}

    return _loss


def _make_channel_poiseuille_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """μ·d²u/dy² = dp/dx  (known pressure gradient)"""
    mu   = float(params.get("mu",   1.0))
    dpdx = float(params.get("dpdx", -2.0))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 1): [y]
        u = _forward(model, X)[:, 0:1]
        u_y  = _grad(u, X)[:, 0:1]
        u_yy = _grad(u_y, X)[:, 0:1]
        r = mu * u_yy - dpdx
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_poiseuille": float(loss.item())}

    return _loss


def _make_beam_bending_loss(params: dict, w_pde: float = 1.0, **_kw: Any):
    """EI·d⁴w/dx⁴ = q"""
    EI = float(params.get("E", 1.0)) * float(params.get("I", 1.0))
    q  = float(params.get("q", 1.0))

    def _loss(model: Any, batch: dict):
        import torch
        X = batch["X"].clone().requires_grad_(True)          # (N, 1): [x]
        w   = _forward(model, X)[:, 0:1]
        w1  = _grad(w,  X)[:, 0:1]
        w2  = _grad(w1, X)[:, 0:1]
        w3  = _grad(w2, X)[:, 0:1]
        w4  = _grad(w3, X)[:, 0:1]
        r   = EI * w4 - q
        loss = w_pde * torch.mean(r ** 2)
        return loss, {"pde_beam": float(loss.item())}

    return _loss


# Maps problem_id → factory function (native autograd implementations)
_NATIVE_LOSS_FACTORIES: dict[str, Any] = {
    "burgers_1d":             _make_burgers_loss,
    "heat_equation_1d":       _make_heat_1d_loss,
    "heat_equation_2d":       _make_heat_2d_loss,
    "wave_equation":          _make_wave_loss,
    "navier_stokes_2d":       _make_ns_2d_loss,
    "lid_driven_cavity":      _make_ns_2d_loss,
    "poisson_electrostatics": _make_poisson_loss,
    "magnetostatics_2d":      _make_poisson_loss,
    "laplace_potential":      _make_laplace_loss,
    "helmholtz_wave":         _make_helmholtz_loss,
    "kdv_equation":           _make_kdv_loss,
    "klein_gordon":           _make_klein_gordon_loss,
    "channel_poiseuille":     _make_channel_poiseuille_loss,
    "beam_bending":           _make_beam_bending_loss,
}


# ---------------------------------------------------------------------------
# Problem metadata: which problems support physics loss and what PDEs they use
# ---------------------------------------------------------------------------

_PROBLEM_PHYSICS_INFO: dict[str, dict[str, Any]] = {
    "burgers_1d": {
        "supports_physics_loss": True,
        "preset_key": "burgers_1d",
        "equations": ["u_t + u*u_x = nu*u_xx"],
        "pde_kind": "burgers",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u",),
        "description": "1D viscous Burgers equation with Cole-Hopf IC",
    },
    "navier_stokes_2d": {
        "supports_physics_loss": True,
        "preset_key": "ns_incompressible_2d_default",
        "equations": ["rho*(u_t + u*u_x + v*u_y) = -p_x + mu*lap(u)", "div(u)=0"],
        "pde_kind": "navier_stokes_incompressible",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("u", "v", "p"),
        "description": "2D incompressible Navier-Stokes (Kovasznay flow benchmark)",
    },
    "lid_driven_cavity": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["rho*(u*u_x + v*u_y) = -p_x + mu*lap(u)", "div(u)=0"],
        "pde_kind": "navier_stokes_incompressible",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("u", "v", "p"),
        "description": "Steady 2D lid-driven cavity flow",
    },
    "bernoulli_flow": {
        "supports_physics_loss": False,
        "preset_key": None,
        "equations": ["p + 0.5*rho*u^2 = const", "A*u = const"],
        "pde_kind": None,
        "dim": 1,
        "coords": ("x",),
        "fields": ("u", "p"),
        "description": "1D inviscid Bernoulli flow in converging duct",
    },
    "stokes_flow": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["mu*lap(u) = grad(p)", "div(u)=0"],
        "pde_kind": "stokes",
        "dim": 1,
        "coords": ("y",),
        "fields": ("u",),
        "description": "Stokes (creeping) flow between parallel plates",
    },
    "darcy_flow": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u = -(K/mu)*grad(p)", "div(u)=0"],
        "pde_kind": "darcy",
        "dim": 1,
        "coords": ("x",),
        "fields": ("u", "p"),
        "description": "Darcy porous media flow",
    },
    "channel_poiseuille": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["mu*d^2u/dy^2 = dp/dx"],
        "pde_kind": "poisson",
        "dim": 1,
        "coords": ("y",),
        "fields": ("u",),
        "description": "Poiseuille channel flow with parabolic profile",
    },
    "linear_elasticity_2d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["div(sigma) + f = 0", "sigma = C:eps"],
        "pde_kind": "linear_elasticity",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("ux", "uy"),
        "description": "2D plane-stress linear elasticity",
    },
    "beam_bending": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["EI*d^4w/dx^4 = q(x)"],
        "pde_kind": "biharmonic",
        "dim": 1,
        "coords": ("x",),
        "fields": ("w",),
        "description": "Euler-Bernoulli beam bending under uniform load",
    },
    "poisson_electrostatics": {
        "supports_physics_loss": True,
        "preset_key": "poisson_2d",
        "equations": ["-lap(phi) = rho/epsilon0"],
        "pde_kind": "poisson",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("u",),
        "description": "Poisson equation for electrostatic potential",
    },
    "helmholtz_wave": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["lap(u) + k^2*u = f"],
        "pde_kind": "helmholtz",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("u",),
        "description": "Helmholtz wave equation with Dirichlet BCs",
    },
    "laplace_potential": {
        "supports_physics_loss": True,
        "preset_key": "laplace_2d",
        "equations": ["lap(u) = 0"],
        "pde_kind": "laplace",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("u",),
        "description": "Laplace equation for potential field",
    },
    "schrodinger_stationary": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["-hbar^2/(2m)*d^2psi/dx^2 + V*psi = E*psi"],
        "pde_kind": "schrodinger_stationary",
        "dim": 1,
        "coords": ("x",),
        "fields": ("psi",),
        "description": "Stationary Schrodinger equation (particle in a box)",
    },
    "schrodinger_timedep": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["i*hbar*dpsi/dt = -hbar^2/(2m)*lap(psi) + V*psi"],
        "pde_kind": "schrodinger",
        "dim": 2,
        "coords": ("x", "t"),
        "fields": ("psi_r", "psi_i"),
        "description": "Time-dependent Schrodinger equation with Gaussian IC",
    },
    "heat_equation_1d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u_t = alpha*u_xx"],
        "pde_kind": "heat",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u",),
        "description": "1D heat/diffusion equation with Fourier series solution",
    },
    "heat_equation_2d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u_t = alpha*(u_xx + u_yy)"],
        "pde_kind": "heat",
        "dim": 2,
        "coords": ("x", "y", "t"),
        "fields": ("u",),
        "description": "2D heat equation (transient)",
    },
    "wave_equation": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u_tt = c^2*u_xx"],
        "pde_kind": "wave",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u",),
        "description": "1D wave equation (d'Alembert solution)",
    },
    "kdv_equation": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u_t + 6*u*u_x + u_xxx = 0"],
        "pde_kind": "kdv",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u",),
        "description": "Korteweg-de Vries equation (soliton solution)",
    },
    "klein_gordon": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["u_tt - c^2*u_xx + m^2*u = 0"],
        "pde_kind": "klein_gordon",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u",),
        "description": "Klein-Gordon relativistic wave equation",
    },
    "thermoelasticity_2d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["div(sigma) = 0", "sigma = C:(eps - alpha_T*DeltaT*I)", "lap(T) = 0"],
        "pde_kind": "thermoelasticity",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("ux", "uy", "T"),
        "description": "2D coupled thermoelasticity",
    },
    "navier_stokes_3d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["rho*(u*grad)u = -grad(p) + mu*lap(u)", "div(u)=0"],
        "pde_kind": "navier_stokes_incompressible",
        "dim": 3,
        "coords": ("x", "y", "z"),
        "fields": ("u", "v", "w", "p"),
        "description": "3D Navier-Stokes (Hagen-Poiseuille pipe flow)",
    },
    "crash_analysis": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["rho*u_tt = E*u_xx"],
        "pde_kind": "wave",
        "dim": 1,
        "coords": ("x", "t"),
        "fields": ("u", "sigma"),
        "description": "1D elastic wave propagation for crash impact analysis",
    },
    "gross_pitaevskii": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["(-hbar^2/(2m)*lap + V + g*|psi|^2)*psi = mu*psi"],
        "pde_kind": "gross_pitaevskii",
        "dim": 1,
        "coords": ("x",),
        "fields": ("psi",),
        "description": "Gross-Pitaevskii equation for BEC ground state",
    },
    "magnetostatics_2d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["lap(A_z) = -mu0*J_z"],
        "pde_kind": "poisson",
        "dim": 2,
        "coords": ("x", "y"),
        "fields": ("A_z",),
        "description": "2D magnetostatics via vector potential",
    },
    "linear_elasticity_3d": {
        "supports_physics_loss": True,
        "preset_key": None,
        "equations": ["div(sigma) + f = 0", "sigma = lam*div(u)*I + 2*mu*eps"],
        "pde_kind": "linear_elasticity",
        "dim": 3,
        "coords": ("x", "y", "z"),
        "fields": ("ux", "uy", "uz"),
        "description": "3D linear elasticity (Lame equations)",
    },
}


class PhysicsLossGenerator:
    """
    Compiles physics (PDE residual) losses for Arena experiments.

    Lookup order
    ------------
    1. PINNeAPPle via ``preset_key`` in _PROBLEM_PHYSICS_INFO  (most accurate)
    2. PINNeAPPle via ``problem_id`` used directly as a preset key  (broader coverage)
    3. Native PyTorch autograd implementation from ``_NATIVE_LOSS_FACTORIES``
    4. ``None``  — experiment continues with supervised loss only
    """

    def __init__(self, problem_id: str, parameters: dict):
        self.problem_id = problem_id
        self.parameters = parameters
        self._info = _PROBLEM_PHYSICS_INFO.get(problem_id, {})
        self._compiled_loss: Any | None = None

    def needs_physics_loss(self) -> bool:
        """True when the problem has a PDE that can be used as a residual constraint."""
        return bool(self._info.get("supports_physics_loss", False)) or (
            self.problem_id in _NATIVE_LOSS_FACTORIES
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _try_pinneapple(
        self, key: str, w_pde: float, w_bc: float, w_ic: float, w_data: float
    ) -> Any | None:
        """Attempt to compile a loss via pp.get_preset(key). Returns None on failure."""
        if not (HAS_PINNEAPPLE and HAS_PP):
            return None
        try:
            weights = LossWeights(w_pde=w_pde, w_bc=w_bc, w_ic=w_ic, w_data=w_data)
            spec = pp.get_preset(key)
            raw = compile_problem(spec, weights=weights)

            def compiled(model: Any, batch: dict, _raw=raw):
                out = _raw(model, None, {"x_col": batch["X"]})
                loss = out.get("pde", out["total"])
                return loss, {"pde_preset": float(loss.detach().item())}

            self._compiled_loss = compiled
            logger.info(
                "Physics loss compiled via PINNeAPPle preset '%s' for '%s'",
                key, self.problem_id,
            )
            return compiled
        except Exception as exc:
            logger.debug("PINNeAPPle preset '%s' failed: %s", key, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_losses(self, loss_weights: dict[str, float] | None = None) -> Any | None:
        """
        Return a compiled loss callable ``(model, batch) → (loss_tensor, info_dict)``,
        or ``None`` if no physics loss is available for this problem.
        """
        if not HAS_TORCH:
            return None

        if self._compiled_loss is not None:
            return self._compiled_loss

        lw = loss_weights or {}
        w_pde  = float(lw.get("pde",  1.0))
        w_bc   = float(lw.get("bc",  10.0))
        w_ic   = float(lw.get("ic",  10.0))
        w_data = float(lw.get("data", 1.0))

        # ── 1. Native PyTorch autograd residual (uses the experiment's parameters)
        factory = _NATIVE_LOSS_FACTORIES.get(self.problem_id)
        if factory is not None:
            try:
                compiled = factory(self.parameters, w_pde=w_pde, w_bc=w_bc, w_ic=w_ic)
                self._compiled_loss = compiled
                logger.info(
                    "Physics loss compiled via native autograd for '%s'", self.problem_id
                )
                return compiled
            except Exception as exc:
                logger.warning(
                    "Native physics loss factory failed for '%s': %s", self.problem_id, exc
                )

        # ── 2. PINNeAPPle preset: explicit preset_key, then problem_id as the key
        preset_key = self._info.get("preset_key")
        for key in [k for k in (preset_key, self.problem_id) if k]:
            compiled = self._try_pinneapple(key, w_pde, w_bc, w_ic, w_data)
            if compiled is not None:
                return compiled

        logger.debug("No physics loss available for '%s'", self.problem_id)
        return None

    def get_loss_description(self) -> dict:
        """Human-readable description of available physics losses for this problem."""
        info = self._info
        preset_key = info.get("preset_key")
        native_available = self.problem_id in _NATIVE_LOSS_FACTORIES
        return {
            "problem_id":            self.problem_id,
            "supports_physics_loss": self.needs_physics_loss(),
            "equations":             info.get("equations", []),
            "pde_kind":              info.get("pde_kind"),
            "dim":                   info.get("dim", 1),
            "coords":                list(info.get("coords", [])),
            "fields":                list(info.get("fields", [])),
            "description":           info.get("description", ""),
            "has_pinneapple":        HAS_PINNEAPPLE,
            "preset_available":      HAS_PP and preset_key is not None,
            "preset_key":            preset_key,
            "native_autograd":       native_available,
        }

    @staticmethod
    def list_supported_problems() -> list[str]:
        """Problem IDs that have at least one physics loss source available."""
        from_info = {
            pid for pid, info in _PROBLEM_PHYSICS_INFO.items()
            if info.get("supports_physics_loss")
        }
        return sorted(from_info | set(_NATIVE_LOSS_FACTORIES.keys()))

    @staticmethod
    def get_all_descriptions() -> dict[str, dict]:
        """Descriptions for all problems that have physics loss metadata."""
        result = {}
        all_ids = set(_PROBLEM_PHYSICS_INFO.keys()) | set(_NATIVE_LOSS_FACTORIES.keys())
        for pid in all_ids:
            result[pid] = PhysicsLossGenerator(pid, {}).get_loss_description()
        return result
