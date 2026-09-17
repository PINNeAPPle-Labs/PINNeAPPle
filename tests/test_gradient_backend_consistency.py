"""Cross-backend consistency for SymbolicPDE's pluggable ``grad_method``
(see ``pinneapple_physics/symbolic_pde/gradient_backends.py`` and the
insights note this session imports from PhysicsNeMo's architecture:
separate "what is the PDE" (the SymPy expression) from "how do we compute a
derivative given the data's shape" -- ``autograd`` over scattered points vs.
``finite_difference``/``spectral`` over a structured grid, reusing
``pinneapple_neural``'s already-tested PINO derivative primitives).

What is actually checked here (three real preset PDE *kinds* from
``pinneapple_physics/pde_environment/presets/`` -- confirmed against
``tests/test_manufactured_solutions.py`` and the string-``kind`` dispatch in
``pinneapple_physics/pinn_solver/compiler/compile.py`` before being used,
see each test's docstring for the exact residual formula matched):

1. The SAME symbolic residual, evaluated through all 3 backends on the SAME
   smooth probe field, produces CONSISTENT residual values -- autograd
   (exact, differentiating the closed-form analytic function) is treated as
   ground truth; spectral (FFT-based, exact for a periodic sinusoid on a
   matching-period domain) should agree with it to near machine precision;
   finite-difference (2nd-order central stencils) should agree with it to
   O(dx^2) truncation error, not exactly.
2. ``grad_method="autograd"`` (the default, used by every existing caller of
   SymbolicPDE) is untouched: a direct manufactured-solution check
   (identical in spirit to ``test_manufactured_solutions.py``'s own
   ``laplace_2d`` case) still gives ~0 residual for the true harmonic
   solution.
3. The new grad_method/grid validation raises clear errors for
   mismatched usage (e.g. calling ``to_grid_residual_fn`` on an
   ``autograd``-mode instance, or building a grid-mode instance with no
   ``grid``).

Probe fields are deliberately chosen periodic-compatible on their sampling
domain (matching the implicit periodicity assumption of the FFT-based
spectral backend) -- this is a DERIVATIVE-OPERATOR agreement check, not a
second manufactured-solution check: the probe fields below generally do NOT
solve the PDE (residual is nonzero), which is fine and expected, since the
point is "do the three backends compute the same u_xx / u_xy / etc. for the
same u", not "is this the true solution". float64 is used throughout so
truncation-error comparisons aren't muddied by float32 roundoff (the same
practice already used in this file's neighbors, e.g.
``test_audit_physics_compressible_euler_rotating_3d_matches_independent_closed_form``).
"""
from __future__ import annotations

import math

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pinneapple_physics.symbolic_pde.compiler import SymbolicPDE
from pinneapple_physics.symbolic_pde.gradient_backends import GridAxisSpec


class _ExactFn(nn.Module):
    """Wraps a plain torch-differentiable closed-form function as a fake
    "model" for ``SymbolicPDE.to_residual_fn`` (same helper pattern as
    ``test_manufactured_solutions.py``'s ``_ExactFn``)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


def _grid_coords(shape_lengths):
    """shape_lengths: list of (n, L) pairs -> meshgrid of coordinate tensors,
    each sampled at n points over [0, L) (matches pino.py's periodic-domain
    convention: k = 2*pi*fftfreq(n, d=L/n))."""
    axes = [torch.arange(n, dtype=torch.float64) * (L / n) for n, L in shape_lengths]
    return torch.meshgrid(*axes, indexing="ij")


@pytest.fixture(autouse=True)
def _float64_cpu():
    """float64 (for tight truncation-error comparisons) is not supported by
    the MPS backend -- explicitly pin the default device to CPU for the
    duration of these tests, in addition to the dtype, so they're immune to
    whatever ambient `torch.set_default_device(...)` state an earlier test
    in the full suite may have left behind (observed in practice: some
    earlier test in a full `pytest tests/` run leaves the global default
    device set to "mps", which breaks any later bare `torch.arange(...,
    dtype=torch.float64)` call -- a pre-existing, unrelated test-isolation
    gap this fixture works around rather than relying on suite-wide state)."""
    prev_dtype = torch.get_default_dtype()
    prev_device = torch.get_default_device()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(prev_device)
    torch.set_default_dtype(prev_dtype)


def test_gradient_backends_agree_laplace_2d():
    """laplace_2d preset kind: residual = u_xx + u_yy (see
    ``pde_environment/presets/academics.py::laplace_2d_default`` and
    ``compile.py``'s "laplace" branch). Probe field u = sin(pi x) sin(pi y)
    on [0,2)x[0,2) (period-2 sinusoid over a length-2 domain -> exactly
    periodic, ideal for the FFT-based spectral backend); NOT itself a
    Laplace solution (its Laplacian is -2*pi^2*u, not 0) -- used purely to
    check the three backends compute the SAME second-derivative operator."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2)

    Lx, Ly = 2.0, 2.0
    nx, ny = 64, 64
    grid = GridAxisSpec(coord_names=("x", "y"), dims=(0, 1), L=(Lx, Ly))

    pde_ag = SymbolicPDE(expr, [x, y], [u])
    pde_fd = SymbolicPDE(expr, [x, y], [u], grad_method="finite_difference", grid=grid)
    pde_sp = SymbolicPDE(expr, [x, y], [u], grad_method="spectral", grid=grid)

    def u_fn(X, Y):
        return torch.sin(math.pi * X) * torch.sin(math.pi * Y)

    X, Y = _grid_coords([(nx, Lx), (ny, Ly)])
    u_grid = u_fn(X, Y)

    coords_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).requires_grad_(True)
    model = _ExactFn(lambda c: u_fn(c[:, 0:1], c[:, 1:2]))
    res_ag = pde_ag.to_residual_fn(model)(coords_flat).reshape(nx, ny).detach()
    res_fd = pde_fd.to_grid_residual_fn()({"u": u_grid})
    res_sp = pde_sp.to_grid_residual_fn()({"u": u_grid})

    exact = -2.0 * math.pi ** 2 * u_grid
    assert (res_ag - exact).abs().mean().item() < 1e-10, "autograd should match the analytic Laplacian almost exactly"

    interior = (slice(4, -4), slice(4, -4))
    sp_err = (res_sp - res_ag)[interior].abs()
    fd_err = (res_fd - res_ag)[interior].abs()
    assert sp_err.mean().item() < 1e-8, f"spectral vs autograd mean diff should be ~machine precision, got {sp_err.mean().item():.3e}"
    assert fd_err.mean().item() < 3e-2, f"finite_difference vs autograd mean diff should be small (O(dx^2)), got {fd_err.mean().item():.3e}"
    assert fd_err.max().item() < 5e-2, f"finite_difference vs autograd max diff should be bounded, got {fd_err.max().item():.3e}"


def test_gradient_backends_agree_burgers_1d():
    """burgers_1d preset kind: residual = u_t + u*u_x - nu*u_xx (see
    ``academics.py::burgers_1d_default`` and ``compile.py``'s "burgers"
    branch -- nu default 0.01 matches the preset's own default). Probe
    field 0.1 + 0.05*sin(2*pi*x)*cos(2*pi*t) on [0,1)x[0,1), periodic in
    both x and t."""
    xs, ts = sp.symbols("x t")
    uf = sp.Function("u")
    nu_sym = sp.Symbol("nu")
    expr = uf(xs, ts).diff(ts, 1) + uf(xs, ts) * uf(xs, ts).diff(xs, 1) - nu_sym * uf(xs, ts).diff(xs, 2)

    Lx, Lt = 1.0, 1.0
    nx, nt = 64, 64
    grid = GridAxisSpec(coord_names=("x", "t"), dims=(0, 1), L=(Lx, Lt))
    nu_val = 0.01
    params = {"nu": torch.tensor(nu_val)}

    pde_ag = SymbolicPDE(expr, [xs, ts], [uf], param_syms=[nu_sym])
    pde_fd = SymbolicPDE(expr, [xs, ts], [uf], param_syms=[nu_sym], grad_method="finite_difference", grid=grid)
    pde_sp = SymbolicPDE(expr, [xs, ts], [uf], param_syms=[nu_sym], grad_method="spectral", grid=grid)

    def u_fn(X, T):
        return 0.1 + 0.05 * torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * T)

    X, T = _grid_coords([(nx, Lx), (nt, Lt)])
    u_grid = u_fn(X, T)

    coords_flat = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1).requires_grad_(True)
    model = _ExactFn(lambda c: u_fn(c[:, 0:1], c[:, 1:2]))
    res_ag = pde_ag.to_residual_fn(model, params=params)(coords_flat).reshape(nx, nt).detach()
    res_fd = pde_fd.to_grid_residual_fn(params=params)({"u": u_grid})
    res_sp = pde_sp.to_grid_residual_fn(params=params)({"u": u_grid})

    interior = (slice(4, -4), slice(4, -4))
    sp_err = (res_sp - res_ag)[interior].abs()
    fd_err = (res_fd - res_ag)[interior].abs()
    assert sp_err.mean().item() < 1e-8, f"spectral vs autograd mean diff should be ~machine precision, got {sp_err.mean().item():.3e}"
    assert fd_err.mean().item() < 2e-3, f"finite_difference vs autograd mean diff should be small, got {fd_err.mean().item():.3e}"
    assert fd_err.max().item() < 3e-3, f"finite_difference vs autograd max diff should be bounded, got {fd_err.max().item():.3e}"


def test_gradient_backends_agree_reaction_diffusion_2d():
    """reaction_diffusion_2d preset kind: dC/dt = D*laplacian(C) - lambda*C,
    i.e. residual = C_t - D*(C_xx+C_yy) + lambda*C (D=0.5, lambda=0.3 --
    same values used by ``test_manufactured_solutions.py``'s own
    reaction_diffusion_2d MMS test). Probe field periodic in x, y, and t on
    [0,1)^3."""
    xr, yr, tr = sp.symbols("x y t")
    Cf = sp.Function("C")
    D_sym, lam_sym = sp.symbols("D lambda")
    expr = (
        Cf(xr, yr, tr).diff(tr, 1)
        - D_sym * (Cf(xr, yr, tr).diff(xr, 2) + Cf(xr, yr, tr).diff(yr, 2))
        + lam_sym * Cf(xr, yr, tr)
    )

    L = 1.0
    n = 32
    grid = GridAxisSpec(coord_names=("x", "y", "t"), dims=(0, 1, 2), L=(L, L, L))
    D_val, lam_val = 0.5, 0.3
    params = {"D": torch.tensor(D_val), "lambda": torch.tensor(lam_val)}

    pde_ag = SymbolicPDE(expr, [xr, yr, tr], [Cf], param_syms=[D_sym, lam_sym])
    pde_fd = SymbolicPDE(expr, [xr, yr, tr], [Cf], param_syms=[D_sym, lam_sym], grad_method="finite_difference", grid=grid)
    pde_sp = SymbolicPDE(expr, [xr, yr, tr], [Cf], param_syms=[D_sym, lam_sym], grad_method="spectral", grid=grid)

    def c_fn(X, Y, T):
        return 1.0 + 0.1 * torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y) * torch.cos(2 * math.pi * T)

    X, Y, T = _grid_coords([(n, L), (n, L), (n, L)])
    c_grid = c_fn(X, Y, T)

    coords_flat = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1).requires_grad_(True)
    model = _ExactFn(lambda c: c_fn(c[:, 0:1], c[:, 1:2], c[:, 2:3]))
    res_ag = pde_ag.to_residual_fn(model, params=params)(coords_flat).reshape(n, n, n).detach()
    res_fd = pde_fd.to_grid_residual_fn(params=params)({"C": c_grid})
    res_sp = pde_sp.to_grid_residual_fn(params=params)({"C": c_grid})

    interior = (slice(2, -2), slice(2, -2), slice(2, -2))
    sp_err = (res_sp - res_ag)[interior].abs()
    fd_err = (res_fd - res_ag)[interior].abs()
    assert sp_err.mean().item() < 1e-8, f"spectral vs autograd mean diff should be ~machine precision, got {sp_err.mean().item():.3e}"
    assert fd_err.mean().item() < 2e-2, f"finite_difference vs autograd mean diff should be small, got {fd_err.mean().item():.3e}"
    assert fd_err.max().item() < 3e-2, f"finite_difference vs autograd max diff should be bounded, got {fd_err.max().item():.3e}"


def test_default_grad_method_autograd_unchanged_manufactured_solution():
    """grad_method defaults to "autograd" -- confirms SymbolicPDE's original
    (pre-existing) behavior is preserved exactly: the laplace_2d kind's
    residual (u_xx+u_yy) must be ~0 for the exact harmonic solution
    u=x^2-y^2, same manufactured-solution check style as
    ``test_manufactured_solutions.py``'s own laplace_2d test (that one goes
    through ``compile_problem``'s independent hand-written autograd_ops
    path; this one goes through SymbolicPDE's autograd path -- both should,
    and do, agree the exact harmonic solution gives ~0 residual)."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2)

    pde = SymbolicPDE(expr, [x, y], [u])
    assert pde.grad_method == "autograd"
    assert pde.grid is None

    exact = _ExactFn(lambda c: c[:, 0:1] ** 2 - c[:, 1:2] ** 2)
    coords = torch.rand(256, 2, requires_grad=True)
    res = pde.to_residual_fn(exact)(coords)
    assert float(res.abs().max().item()) < 1e-8, f"expected ~0 residual for the exact harmonic solution, got {float(res.abs().max().item())}"


def test_grad_method_validation_errors():
    """The new grad_method/grid plumbing should fail loudly and clearly on
    misuse, rather than silently doing the wrong thing."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2)

    with pytest.raises(ValueError, match="requires a `grid`"):
        SymbolicPDE(expr, [x, y], [u], grad_method="finite_difference")

    with pytest.raises(ValueError, match="Unknown grad_method"):
        SymbolicPDE(expr, [x, y], [u], grad_method="bogus")  # type: ignore[arg-type]

    grid = GridAxisSpec(coord_names=("x", "y"), dims=(0, 1), L=(1.0, 1.0))
    pde_grid = SymbolicPDE(expr, [x, y], [u], grad_method="spectral", grid=grid)
    with pytest.raises(ValueError, match="to_residual_fn is for grad_method"):
        pde_grid.to_residual_fn(_ExactFn(lambda c: c[:, 0:1]))

    pde_ag = SymbolicPDE(expr, [x, y], [u])
    with pytest.raises(ValueError, match="to_grid_residual_fn is for grad_method"):
        pde_ag.to_grid_residual_fn()
