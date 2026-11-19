"""Finite Element Method — problem-agnostic P1/Q1 solver."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _make_structured_mesh(nx: int, ny: int, bounds: Dict[str, Tuple[float, float]]):
    """Build a regular (nx x ny) quad mesh; returns nodes (N,2) and elem (E,4)."""
    x0, x1 = bounds.get("x", (0.0, 1.0))
    y0, y1 = bounds.get("y", (0.0, 1.0))
    xs = torch.linspace(x0, x1, nx + 1)
    ys = torch.linspace(y0, y1, ny + 1)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    nodes = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N,2)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elems.append([n0, n1, n2, n3])
    elems = torch.tensor(elems, dtype=torch.long)  # (E,4)
    return nodes, elems


# ---------------------------------------------------------------------------
# Bilinear Q1 shape functions on [-1,1]^2
# ---------------------------------------------------------------------------

_GQ = 1.0 / 3.0 ** 0.5  # Gauss point coord
_QP = torch.tensor([[-_GQ, -_GQ], [_GQ, -_GQ], [_GQ, _GQ], [-_GQ, _GQ]])  # (4,2)
_QW = torch.ones(4)


def _q1_shape(xi: float, eta: float) -> torch.Tensor:
    """Q1 shape functions at (xi, eta) in [-1,1]^2."""
    return 0.25 * torch.tensor([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


def _q1_grad(xi: float, eta: float) -> torch.Tensor:
    """Returns (2,4) matrix of dN/d(xi,eta)."""
    dxi = 0.25 * torch.tensor([
        -(1 - eta), (1 - eta), (1 + eta), -(1 + eta),
    ])
    deta = 0.25 * torch.tensor([
        -(1 - xi), -(1 + xi), (1 + xi), (1 - xi),
    ])
    return torch.stack([dxi, deta], dim=0)  # (2,4)


def _elem_stiffness_scalar(xy: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """
    Q1 element stiffness K_e (4,4) for -coeff * Δu = f.
    xy: (4,2) node coordinates.
    """
    Ke = torch.zeros(4, 4, device=xy.device)
    qp = _QP.to(xy.device)
    for q in range(4):
        xi, eta = qp[q, 0].item(), qp[q, 1].item()
        dN = _q1_grad(xi, eta).to(xy.device)   # (2,4)
        J = dN @ xy                              # (2,2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        Jinv = torch.zeros(2, 2, device=xy.device)
        Jinv[0, 0] = J[1, 1]; Jinv[0, 1] = -J[0, 1]
        Jinv[1, 0] = -J[1, 0]; Jinv[1, 1] = J[0, 0]
        Jinv = Jinv / detJ
        B = Jinv @ dN  # (2,4)  physical gradients
        Ke = Ke + coeff * detJ * (B.T @ B)
    return Ke


def _elem_mass(xy: torch.Tensor) -> torch.Tensor:
    """Q1 element mass M_e (4,4)."""
    Me = torch.zeros(4, 4, device=xy.device)
    qp = _QP.to(xy.device)
    for q in range(4):
        xi, eta = qp[q, 0].item(), qp[q, 1].item()
        N = _q1_shape(xi, eta).to(xy.device)   # (4,)
        dN = _q1_grad(xi, eta).to(xy.device)
        J = dN @ xy
        detJ = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]).abs()
        Me = Me + detJ * torch.outer(N, N)
    return Me


# ---------------------------------------------------------------------------
# Assembler (Poisson / Laplace / Helmholtz)
# ---------------------------------------------------------------------------

def _assemble_poisson(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    coeff: float,
    k2: float,
    source: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble K (N,N) and f (N,) for -(coeff*Δ + k²)u = source."""
    N = nodes.shape[0]
    K = torch.zeros(N, N, device=nodes.device)
    f = torch.zeros(N, device=nodes.device)

    for e in range(elems.shape[0]):
        idx = elems[e]           # (4,)
        xy = nodes[idx]          # (4,2)
        Ke = _elem_stiffness_scalar(xy, coeff)
        if k2 != 0.0:
            Me = _elem_mass(xy)
            Ke = Ke - k2 * Me
        # element load: f_e = source * M_e @ [1,1,1,1]
        fe = source * _elem_mass(xy) @ torch.ones(4, device=nodes.device)
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += Ke[i, j]
            f[idx[i]] += fe[i]
    return K, f


# ---------------------------------------------------------------------------
# Axisymmetric elasticity assembler  (ur, uz unknowns, r-weighted)
# ---------------------------------------------------------------------------

def _assemble_axisym_elasticity(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    E: float,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plane-strain axisymmetric elasticity stiffness (interleaved DOF ordering).
    DOF: [ur0, uz0, ur1, uz1, ...] — 2N total.
    Returns K (2N,2N), f zeros (2N,).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    # Voigt 4-component strain: [err, ezz, etheta, erz]
    # C matrix (4x4 plane-strain axisymmetric)
    C = torch.tensor([
        [lam + 2 * mu, lam, lam, 0.0],
        [lam, lam + 2 * mu, lam, 0.0],
        [lam, lam, lam + 2 * mu, 0.0],
        [0.0, 0.0, 0.0, mu],
    ])
    N_nodes = nodes.shape[0]
    ndof = 2 * N_nodes
    K = torch.zeros(ndof, ndof, device=nodes.device)

    qp = _QP.to(nodes.device)
    for e in range(elems.shape[0]):
        idx = elems[e]           # (4,)
        xy = nodes[idx]          # (4,2)  col0=r, col1=z
        Ke = torch.zeros(8, 8, device=nodes.device)
        for q in range(4):
            xi, eta = qp[q, 0].item(), qp[q, 1].item()
            N = _q1_shape(xi, eta).to(nodes.device)   # (4,)
            dN = _q1_grad(xi, eta).to(nodes.device)   # (2,4)
            J = dN @ xy  # (2,2)
            detJ = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]).abs()
            Jinv = torch.zeros(2, 2, device=nodes.device)
            Jinv[0, 0] = J[1, 1]; Jinv[0, 1] = -J[0, 1]
            Jinv[1, 0] = -J[1, 0]; Jinv[1, 1] = J[0, 0]
            Jinv = Jinv / (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            dNxy = Jinv @ dN  # (2,4): dN/dr, dN/dz
            r = (N * xy[:, 0]).sum().clamp_min(1e-12)
            # B matrix (4,8): strains from [ur,uz] DOFs
            B = torch.zeros(4, 8, device=nodes.device)
            for a in range(4):
                col_r = 2 * a
                col_z = 2 * a + 1
                B[0, col_r] = dNxy[0, a]          # err = dur/dr
                B[1, col_z] = dNxy[1, a]          # ezz = duz/dz
                B[2, col_r] = N[a] / r             # etheta = ur/r
                B[3, col_r] = dNxy[1, a]           # erz (r-part)
                B[3, col_z] = dNxy[0, a]           # erz (z-part)
            Ke = Ke + r * detJ * (B.T @ C.to(nodes.device) @ B)
        # scatter into global
        for i in range(4):
            for j in range(4):
                for di in range(2):
                    for dj in range(2):
                        K[2 * idx[i] + di, 2 * idx[j] + dj] += Ke[2 * i + di, 2 * j + dj]

    f = torch.zeros(ndof, device=nodes.device)
    return K, f


# ---------------------------------------------------------------------------
# 2-D plane elasticity assembler  (ux, uy unknowns; plane stress / plane strain)
# ---------------------------------------------------------------------------

#: Constitutive models this assembler builds, and the normalised kind strings it takes.
_PLANE_MODELS: tuple[str, ...] = ("plane_stress", "plane_strain")

_PLANE_KIND_MODEL: dict[str, str | None] = {
    # the kinds named by the elasticity capability (capabilities.py: kind="linear_elasticity")
    "linear_elasticity": None,                        # model from params.model / meta.formulation
    "linear_elasticity_2d": None,
    "elasticity": None,
    # the kind names the presets declare (presets/structural.py)
    "linear_elasticity_plane_stress": "plane_stress",
    "linear_elasticity_plane_strain": "plane_strain",
    "linear_elasticity_2d_plane_stress": "plane_stress",
    "linear_elasticity_2d_plane_strain": "plane_strain",
    "plane_stress": "plane_stress",
    "plane_strain": "plane_strain",
    "plane_stress_2d": "plane_stress",
    "plane_strain_2d": "plane_strain",
}


def _elastic_constants(params: dict[str, Any]) -> tuple[float, float]:
    """``(E, nu)`` from ``params``, from Lamé's ``lambda``/``mu`` when those are all that is given.

    ``E = mu*(3*lam+2*mu)/(lam+mu)``, ``nu = lam/(2*(lam+mu))``. Defaults (E=200e9, nu=0.3, the
    elasticity capability's own defaults) apply only when neither pair is given at all.
    """
    E = params.get("E")
    nu = params.get("nu")
    lam = params.get("lambda", params.get("lam"))
    mu = params.get("mu")
    if (E is None or nu is None) and lam is not None and mu is not None:
        lam, mu = float(lam), float(mu)
        if lam + mu != 0.0:
            if E is None:
                E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            if nu is None:
                nu = lam / (2.0 * (lam + mu))
    E = 200e9 if E is None else float(E)
    nu = 0.3 if nu is None else float(nu)
    # -1 < nu < 0.5 keeps both D matrices positive definite; outside it the solve is meaningless
    if not -1.0 < nu < 0.5:
        raise ValueError(f"linear elasticity needs -1 < nu < 0.5, got nu={nu}")
    if not E > 0.0:
        raise ValueError(f"linear elasticity needs a positive Young's modulus, got E={E}")
    return E, nu


def _plane_elasticity_model(kind: str, params: dict[str, Any], spec) -> str | None:
    """Constitutive model (``plane_stress`` / ``plane_strain``) for this spec, or None to fall through.

    Precedence: ``params["model"]`` > the kind's own model > ``pde.meta["formulation"]`` >
    ``"plane_stress"``. Only 2-D specs are taken: the mesh here is an (nx, ny) Q1 grid, so a 3-D
    ``linear_elasticity`` spec keeps falling through to the Poisson assembly exactly as before.
    """
    if int(getattr(spec, "dim", 2) or 2) != 2:
        return None
    if kind not in _PLANE_KIND_MODEL:
        return None
    meta = getattr(spec.pde, "meta", {}) or {}
    model = params.get("model") or _PLANE_KIND_MODEL[kind] or meta.get("formulation") or "plane_stress"
    model = str(model).lower().strip()
    if model not in _PLANE_MODELS:
        raise ValueError(f"unknown 2-D elasticity model {model!r}; use one of {_PLANE_MODELS}")
    return model


def _assemble_plane_elasticity(
    nodes: torch.Tensor,
    elems: torch.Tensor,
    E: float,
    nu: float,
    thickness: float = 1.0,
    model: str = "plane_stress",
    body_force: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q1 plane-stress / plane-strain elasticity stiffness and load (interleaved DOF ordering).

    DOF: ``[ux0, uy0, ux1, uy1, ...]`` -- 2N total, the same interleaving
    ``_assemble_axisym_elasticity`` returns, so a nodal index ``d`` is DOF ``2*d + component``
    (0 = ux, 1 = uy) for BC application and ``_linear_solve`` alike.

    Strain is the 3-component Voigt vector ``[exx, eyy, gxy]`` with the ENGINEERING shear strain
    ``gxy = 2*exy``, so that a unit shear strain (``u_x = y``) gives ``gxy = 1`` and the energy is
    ``0.5 * D[2,2]``:

      plane stress:  D = E/(1-nu^2) [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
      plane strain:  D = E/((1+nu)(1-2nu)) [[1-nu, nu, 0], [nu, 1-nu, 0], [0, 0, (1-2nu)/2]]

    ``body_force`` is a uniform load per unit volume ``[N/m^3]``, integrated as ``N * b * thickness``
    (``f`` is zero for the default zero load, like the axisymmetric assembly).

    The assembly runs in float64 whatever ``nodes`` carries: with E ~ 1e11 Pa the stiffness entries
    lose every digit that matters in float32 against a load of a few newtons.
    """
    if model not in _PLANE_MODELS:
        raise ValueError(f"unknown 2-D elasticity model {model!r}; use one of {_PLANE_MODELS}")
    E, nu, t = float(E), float(nu), float(thickness)
    if model == "plane_stress":
        D = (E / (1.0 - nu * nu)) * torch.tensor([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
    else:
        D = (E / ((1.0 + nu) * (1.0 - 2.0 * nu))) * torch.tensor([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])
    bx, by = float(body_force[0]), float(body_force[1])

    N_nodes = nodes.shape[0]
    ndof = 2 * N_nodes
    K = torch.zeros(ndof, ndof, dtype=torch.float64, device=nodes.device)
    f = torch.zeros(ndof, dtype=torch.float64, device=nodes.device)
    D = D.to(dtype=torch.float64, device=nodes.device)

    qp = _QP.to(nodes.device)
    for e in range(elems.shape[0]):
        idx = elems[e]                                 # (4,)
        xy = nodes[idx].to(torch.float64)              # (4,2)
        Ke = torch.zeros(8, 8, dtype=torch.float64, device=nodes.device)
        fe = torch.zeros(8, dtype=torch.float64, device=nodes.device)
        for q in range(4):
            xi, eta = qp[q, 0].item(), qp[q, 1].item()
            N = _q1_shape(xi, eta).to(torch.float64)   # (4,)
            dN = _q1_grad(xi, eta).to(torch.float64)   # (2,4)
            J = dN @ xy                                # (2,2)
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            Jinv = torch.zeros(2, 2, dtype=torch.float64, device=nodes.device)
            Jinv[0, 0] = J[1, 1]
            Jinv[0, 1] = -J[0, 1]
            Jinv[1, 0] = -J[1, 0]
            Jinv[1, 1] = J[0, 0]
            Jinv = Jinv / detJ
            dNxy = Jinv @ dN                           # (2,4): dN/dx, dN/dy
            # B (3,8): strains [exx, eyy, gxy] from the interleaved [ux, uy] DOFs
            B = torch.zeros(3, 8, dtype=torch.float64, device=nodes.device)
            for a in range(4):
                B[0, 2 * a] = dNxy[0, a]               # exx = dux/dx
                B[1, 2 * a + 1] = dNxy[1, a]           # eyy = duy/dy
                B[2, 2 * a] = dNxy[1, a]               # gxy = dux/dy + duy/dx
                B[2, 2 * a + 1] = dNxy[0, a]
            w = t * detJ.abs()
            Ke = Ke + w * (B.T @ D @ B)
            for a in range(4):
                fe[2 * a] += w * N[a] * bx
                fe[2 * a + 1] += w * N[a] * by
        for i in range(4):
            for j in range(4):
                for di in range(2):
                    for dj in range(2):
                        K[2 * idx[i] + di, 2 * idx[j] + dj] += Ke[2 * i + di, 2 * j + dj]
            f[2 * idx[i]] += fe[2 * i]
            f[2 * idx[i] + 1] += fe[2 * i + 1]
    return K, f


# The two displacement components of a 2-D elasticity spec, by the field names a condition may use.
_UX_FIELDS = ("ux", "u_x", "dx", "ex")
_UY_FIELDS = ("uy", "u_y", "dy", "ey")


def _plane_components(cond) -> tuple[tuple[int, int], ...]:
    """``(component, value_fn_column)`` pairs for one Dirichlet condition on a 2-component field.

    The presets name their displacement fields (``fields=("ux","uy")`` with a ``value_fn``
    returning ``(N, 2)``), so the column follows the name. A condition with no usable name --
    ``fields=("u",)``, the scalar Poisson convention -- applies its single value to BOTH
    components (a held boundary), and two unnamed values are taken as (ux, uy) in order.
    """
    names = tuple(str(f).lower().strip() for f in (getattr(cond, "fields", ()) or ()))
    pairs = []
    for i, nm in enumerate(names):
        if nm in _UY_FIELDS:
            pairs.append((1, i))
        elif nm in _UX_FIELDS:
            pairs.append((0, i))
    if pairs:
        return tuple(pairs[:2])
    if len(names) >= 2:
        return ((0, 0), (1, 1))
    return ((0, 0), (1, 0))


def _apply_plane_dirichlet(K, f, cond, nodes: torch.Tensor, idx: torch.Tensor):
    """Apply one Dirichlet condition to the 2-DOF-per-node system, component by component."""
    from . import _bc

    pts = nodes[idx]
    for comp, column in _plane_components(cond):
        try:
            vals = _bc.values(cond, pts, field=column)
        except IndexError:     # a value_fn that returns one column for a two-field condition
            vals = _bc.values(cond, pts, field=0)
        K, f = _apply_dirichlet(K, f, 2 * idx + comp, vals.to(device=K.device, dtype=K.dtype))
    return K, f


# ---------------------------------------------------------------------------
# BC application
# ---------------------------------------------------------------------------

def _apply_dirichlet(K: torch.Tensor, f: torch.Tensor, dofs: torch.Tensor, vals: torch.Tensor):
    """Row/column elimination for Dirichlet DOFs."""
    for k, d in enumerate(dofs.tolist()):
        d = int(d)
        f -= K[:, d] * vals[k]
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0
        f[d] = vals[k]
    return K, f


def _boundary_dofs_rect(nodes: torch.Tensor, bounds: Dict[str, Tuple[float, float]], tol: float = 1e-9):
    """Return dict of edge → node indices for rectangular domain."""
    x0, x1 = bounds.get("x", (0.0, 1.0))
    y0, y1 = bounds.get("y", (0.0, 1.0))
    x = nodes[:, 0]; y = nodes[:, 1]
    return {
        "left":   torch.where(torch.abs(x - x0) < tol)[0],
        "right":  torch.where(torch.abs(x - x1) < tol)[0],
        "bottom": torch.where(torch.abs(y - y0) < tol)[0],
        "top":    torch.where(torch.abs(y - y1) < tol)[0],
    }


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="fem",
    family="pde",
    description="Finite Element Method — problem-agnostic Q1 elements for Poisson/Helmholtz and axisymmetric elasticity.",
    tags=["fem", "pde", "agnostic", "elliptic", "elasticity"],
)
class FEMSolver(SolverBase):
    """
    Problem-agnostic FEM solver.

    Supported PDE kinds (spec.pde.kind):
      poisson / laplace            — -coeff·Δu = source  (coeff default 1)
      helmholtz                    — -Δu - k²u = source  (k from params.k)
      axisymmetric_linear_elasticity — r-weighted weak form (E, nu from params)
      linear_elasticity / linear_elasticity_2d / elasticity / plane_stress / plane_strain
      (and the presets' linear_elasticity_plane_stress / linear_elasticity_plane_strain),
      at dim=2 only — two-field (ux, uy) plane stress or plane strain, Q1 quadrilaterals;
      a 3-D elasticity spec still falls through to the Poisson assembly

    Params consumed from PDETermSpec.params:
      coeff  — diffusion coefficient (poisson, default 1.0)
      source — uniform RHS scalar     (poisson/helmholtz, default 0.0)
      k      — wavenumber             (helmholtz)
      E, nu  — Young's modulus, Poisson's ratio (elasticity; derived from Lame's
               lambda/mu when only those are given; 200e9 / 0.3 only if neither)
      model  — "plane_stress" (default) or "plane_strain"
      thickness — out-of-plane thickness [m] for the 2-D elasticity stiffness (default 1.0)
      body_force — uniform load per unit volume (bx, by) for the 2-D elasticity solve
                   (default (0, 0))

    Conditions from ProblemSpec.conditions:
      DirichletBC with .selector as tag string ("left","right","bottom","top")
      or callable selector(nodes) -> bool mask

    Linear system solved by CG or direct (torch.linalg.solve).
    """

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        solver: str = "cg",
        tol: float = 1e-8,
        max_iter: int = 4000,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.solver_name = str(solver).lower().strip()
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @classmethod
    def from_problem_spec(cls, spec, nx: int = 32, ny: int = 32, **kwargs) -> "FEMSolver":
        return cls(nx=nx, ny=ny, **kwargs)

    # ------------------------------------------------------------------
    def _linear_solve(self, K: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if self.solver_name == "direct":
            return torch.linalg.solve(K, f)
        # CG
        x = torch.zeros_like(f)
        r = f - K @ x
        p = r.clone()
        rsold = (r * r).sum()
        for _ in range(self.max_iter):
            Kp = K @ p
            alpha = rsold / (p * Kp).sum().clamp_min(1e-14)
            x = x + alpha * p
            r = r - alpha * Kp
            rsnew = (r * r).sum()
            if torch.sqrt(rsnew).item() < self.tol:
                break
            p = r + (rsnew / rsold.clamp_min(1e-14)) * p
            rsold = rsnew
        return x

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec) -> SolverOutput:
        kind = spec.pde.kind.lower().replace("-", "_").replace(" ", "_")
        bounds = dict(spec.domain_bounds)
        params = dict(spec.pde.params)
        conditions = spec.conditions

        nodes, elems = _make_structured_mesh(self.nx, self.ny, bounds)
        edge_map = _boundary_dofs_rect(nodes, bounds)

        plane_model: str | None = None
        E = nu = thickness = 0.0
        if kind in ("axisymmetric_linear_elasticity", "linear_elasticity_axisymmetric"):
            K, f = _assemble_axisym_elasticity(
                nodes, elems,
                E=float(params.get("E", 200e9)),
                nu=float(params.get("nu", 0.3)),
            )
        elif (plane_model := _plane_elasticity_model(kind, params, spec)) is not None:
            E, nu = _elastic_constants(params)
            thickness = float(params.get("thickness", 1.0))
            body_force = tuple(float(v) for v in params.get("body_force", (0.0, 0.0)))
            K, f = _assemble_plane_elasticity(
                nodes, elems, E, nu,
                thickness=thickness,
                model=plane_model,
                body_force=(body_force + (0.0, 0.0))[:2],
            )
        else:
            coeff  = float(params.get("coeff",  1.0))
            source = float(params.get("source", 0.0))
            k2     = float(params.get("k",      0.0)) ** 2
            K, f = _assemble_poisson(nodes, elems, coeff, k2, source)

        # Apply boundary conditions (ConditionSpec kind="dirichlet"; tags left/right/bottom/top/boundary)
        from . import _bc
        for cond in _bc.dirichlet_conditions(spec):
            dofs = _bc.select(cond, nodes, edges=edge_map, boundary=_bc.boundary_rect(nodes, edge_map))
            if plane_model is not None:
                # 2 dofs per node: the condition picks its component from its own field names
                K, f = _apply_plane_dirichlet(K, f, cond, nodes, dofs)
                continue
            vals = _bc.values(cond, nodes[dofs])
            K, f = _apply_dirichlet(K, f, dofs, vals.to(nodes.device))

        u = self._linear_solve(K, f)

        residual = torch.norm(K @ u - f) / torch.norm(f).clamp_min(1e-12)
        if plane_model is not None:
            shape = (self.nx + 1, self.ny + 1)
            ux = u[0::2].reshape(shape)
            uy = u[1::2].reshape(shape)
            return SolverOutput(
                result=ux,
                losses={"residual": residual},
                extras={
                    "nodes": nodes,
                    "elems": elems,
                    "K": K,
                    "f_rhs": f,
                    "kind": kind,
                    "solver": self.solver_name,
                    "method": f"fem_{plane_model}",
                    "model": plane_model,
                    "E": E,
                    "nu": nu,
                    "thickness": thickness,
                    "ux": ux,
                    "uy": uy,
                    "u": u,
                },
            )
        return SolverOutput(
            result=u.reshape(self.nx + 1, self.ny + 1),
            losses={"residual": residual},
            extras={
                "nodes": nodes,
                "elems": elems,
                "K": K,
                "f_rhs": f,
                "kind": kind,
                "solver": self.solver_name,
            },
        )

    # ------------------------------------------------------------------
    def forward(self, *, spec=None, mesh: Any = None, params: Dict[str, Any] = None) -> SolverOutput:
        """
        Two call styles:
          forward(spec=problem_spec)                       # problem-agnostic
          forward(mesh=my_mesh, params={"K": K, "f": f})  # legacy manual assembly
        """
        if spec is not None:
            return self.solve_from_spec(spec)

        # Legacy: user provides pre-assembled K, f inside params
        K = params["K"]
        f = params["f"]
        if params.get("apply_bcs_fn") is not None:
            K, f = params["apply_bcs_fn"](K, f, params)
        u = self._linear_solve(K, f)
        return SolverOutput(
            result=u,
            losses={"total": torch.tensor(0.0, device=u.device)},
            extras={"K": K, "f": f, "solver": self.solver_name},
        )
