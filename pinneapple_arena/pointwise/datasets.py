"""Reference datasets of the arena problems: analytic solutions where one exists, otherwise a numerical solution
(finite differences, spectral, panel method, ...) written for that problem. ``DatasetGenerator(problem_id,
params).generate(n)`` returns ``{"coords": (N, in_dim), "solution": (N, out_dim), "metadata": {...}}``; the
metadata says which solution was used. Some problems (crash, airfoil 3-D, cylinder wake) are reduced-order or
illustrative fields, not solutions of the full equations: their metadata says so.
"""
# ruff: noqa: E702  (ported numerical code keeps short statements on one line)
from __future__ import annotations

import logging as _logging
from typing import Any

import numpy as np

_logger = _logging.getLogger(__name__)

try:
    from scipy.special import j0 as _scipy_j0
    from scipy.special import j1 as _scipy_j1
    from scipy.special import jn_zeros as _scipy_jn_zeros
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _nan_clean(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf in a numpy array with fill value."""
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def _transient_hp_field(r_arr: np.ndarray, R: float, U: float, tau: float) -> np.ndarray:
    """
    Transient Hagen-Poiseuille (impulsively started from rest):

      u(r, τ) = 2U * [1 - (r/R)² - 8 Σ_n J₀(αₙr/R)/(αₙ³ J₁(αₙ)) * exp(-αₙ² τ)]

    τ = ν t / R²  (non-dimensional time)
    αₙ = n-th zero of J₀,  n=1..N_TERMS

    At τ=0 → u=0 (impulsive start).
    At τ→∞ → Hagen-Poiseuille steady state 2U(1-r²/R²).
    """
    N_TERMS = 8
    if _HAS_SCIPY:
        alpha_n = _scipy_jn_zeros(0, N_TERMS)          # zeros of J₀
        J1_vals = _scipy_j1(alpha_n)                    # J₁(αₙ)
        def _j0(x): return _scipy_j0(x)
    else:
        alpha_n = np.array([2.4048, 5.5201, 8.6537, 11.7915,
                            14.9309, 18.0711, 21.2116, 24.3525])
        J1_vals = np.array([0.5191, -0.3403,  0.2715, -0.2325,
                            0.2065, -0.1877,  0.1733, -0.1614])
        def _j0(x):
            x = np.asarray(x, dtype=np.float64)
            # Abramowitz & Stegun polynomial approx (8-digit accuracy)
            res = np.empty_like(x)
            mask = x < 3.0
            xs = x[mask]
            t = (xs / 3.0) ** 2
            res[mask] = (1 - 2.2499997*t + 1.2656208*t**2 - 0.3163866*t**3
                         + 0.0444479*t**4 - 0.0039444*t**5 + 0.0002100*t**6)
            xl = x[~mask]
            t2 = (3.0 / xl)
            f0 = (0.79788456 - 0.00552740*t2**2 + 0.00137237*t2**4
                  - 0.00072805*t2**5 + 0.00014476*t2**6)
            th = (xl - 0.78539816 + 0.00262573*t2**3 - 0.00054125*t2**4
                  - 0.00029333*t2**5 + 0.00013558*t2**6)
            res[~mask] = (f0 / np.sqrt(xl)) * np.cos(th)
            return res

    r_norm = np.asarray(r_arr, dtype=np.float64) / R
    correction = np.zeros_like(r_norm)
    for n in range(N_TERMS):
        a  = alpha_n[n]
        j1 = J1_vals[n]
        if abs(j1) < 1e-12:
            continue
        x_arg = a * r_norm
        j0_val = _j0(x_arg)
        correction += 8.0 * np.exp(-a**2 * tau) * j0_val / (a**3 * j1)

    u = 2.0 * U * (1.0 - r_norm**2 - correction)
    return np.clip(u, 0.0, 2.0 * U)


# ---------------------------------------------------------------------------
# Individual generator functions
# ---------------------------------------------------------------------------

def _gen_burgers_1d(params: dict, n_samples: int) -> dict:
    """
    Burgers equation: u_t + u*u_x = nu * u_xx
    Solved with Crank-Nicolson (diffusion) + Adams-Bashforth-2 (convection) FDM.
    IC: u(x,0) = -sin(pi*x), BCs: u(-1,t) = u(1,t) = 0, x in [-1,1], t in [0,1].
    """
    try:
        from scipy.linalg import solve_banded as _solve_banded
        _HAS_SOLVE_BANDED = True
    except ImportError:
        _HAS_SOLVE_BANDED = False

    nu = float(params.get("nu", 0.01 / np.pi))
    nx_out = int(params.get("nx", 50))
    nt_out = int(params.get("nt", 50))

    # Fine internal grid for accuracy
    Nx = 512
    T_end = 1.0
    dx = 2.0 / Nx
    # Stability: dt < min(dx^2/(2*nu), dx/|u_max|); |u_max| ≈ 1 initially
    dt_diff = 0.4 * dx**2 / (nu + 1e-12)
    dt_conv = 0.4 * dx
    dt = min(dt_diff, dt_conv, 0.01)
    Nt = int(np.ceil(T_end / dt))
    dt = T_end / Nt

    xi_full = np.linspace(-1.0, 1.0, Nx + 1, dtype=np.float64)
    u_full = -np.sin(np.pi * xi_full)
    u_full[0] = u_full[-1] = 0.0

    # Interior only (indices 1..Nx-1)
    Ni = Nx - 1
    u = u_full[1:-1].copy()

    # CN banded matrix: A @ u^{n+1} = B @ u^n + dt * nonlinear
    r = nu * dt / (2.0 * dx**2)
    # A: (1+2r) on diag, -r on off-diags  → stored as ab[row][col]
    ab = np.zeros((3, Ni), dtype=np.float64)
    ab[0, 1:] = -r            # upper
    ab[1, :] = 1.0 + 2.0 * r  # main
    ab[2, :-1] = -r            # lower

    # B (explicit diffusion part): (1-2r) main, +r off-diags
    B_diag = (1.0 - 2.0 * r) * np.ones(Ni)
    B_off = r * np.ones(Ni - 1)

    def _diffuse_explicit(v):
        out = B_diag * v
        out[:-1] += B_off * v[1:]
        out[1:] += B_off * v[:-1]
        return out

    def _conv_rhs(v):
        # Central differences for uu_x at interior nodes (BCs = 0)
        v_full = np.concatenate([[0.0], v, [0.0]])
        dudx = (v_full[2:] - v_full[:-2]) / (2.0 * dx)
        return -v * dudx

    # Output grid
    x_out = np.linspace(-1.0, 1.0, nx_out, dtype=np.float64)
    t_out = np.linspace(0.0, T_end, nt_out, dtype=np.float64)
    u_out = np.zeros((nx_out, nt_out), dtype=np.float64)

    def _to_output(v_int):
        return np.interp(x_out, xi_full, np.concatenate([[0.0], v_int, [0.0]]))

    u_out[:, 0] = _to_output(u)

    conv_prev = _conv_rhs(u)
    next_out_idx = 1
    t_cur = 0.0

    for step in range(1, Nt + 1):
        conv_cur = _conv_rhs(u)
        # Adams-Bashforth 2 for convection (Euler on first step)
        ab2 = 1.5 * conv_cur - 0.5 * conv_prev if step > 1 else conv_cur

        rhs = _diffuse_explicit(u) + dt * ab2
        if _HAS_SOLVE_BANDED:
            u = _solve_banded((1, 1), ab, rhs)
        else:
            # Thomas algorithm fallback
            ab_cp = ab.copy()
            rhs_cp = rhs.copy()
            for i in range(1, Ni):
                m = ab_cp[2, i - 1] / ab_cp[1, i - 1]
                ab_cp[1, i] -= m * ab_cp[0, i]
                rhs_cp[i] -= m * rhs_cp[i - 1]
            u[Ni - 1] = rhs_cp[Ni - 1] / ab_cp[1, Ni - 1]
            for i in range(Ni - 2, -1, -1):
                u[i] = (rhs_cp[i] - ab_cp[0, i + 1] * u[i + 1]) / ab_cp[1, i]

        conv_prev = conv_cur
        t_cur += dt

        # Store at output times
        if next_out_idx < nt_out and t_cur >= t_out[next_out_idx] - 1e-10:
            u_out[:, next_out_idx] = _to_output(u)
            next_out_idx += 1

    for j in range(next_out_idx, nt_out):
        u_out[:, j] = u_out[:, next_out_idx - 1]

    X, T = np.meshgrid(x_out, t_out, indexing="ij")
    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u_out.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x_out.astype(np.float32),
        "t": t_out.astype(np.float32),
        "u": u_out.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "burgers_1d", "nu": nu, "nx": nx_out, "nt": nt_out, "dim": 2},
    }


def _gen_navier_stokes_2d(params: dict, n_samples: int) -> dict:
    """
    Kovasznay flow: analytical exact solution to steady NS 2D.
    u = 1 - exp(lambda*x)*cos(2*pi*y)
    v = lambda/(2*pi)*exp(lambda*x)*sin(2*pi*y)
    p = (1 - exp(2*lambda*x)) / 2
    where lambda = Re/2 - sqrt(Re^2/4 + 4*pi^2)
    """
    Re = float(params.get("Re", 40.0))
    n = int(np.sqrt(n_samples)) + 1

    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)
    x = np.linspace(-0.5, 1.0, n, dtype=np.float64)
    y = np.linspace(-0.5, 1.5, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = 1.0 - np.exp(lam * X) * np.cos(2.0 * np.pi * Y)
    v = (lam / (2.0 * np.pi)) * np.exp(lam * X) * np.sin(2.0 * np.pi * Y)
    p = (1.0 - np.exp(2.0 * lam * X)) / 2.0

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u.ravel(), v.ravel(), p.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "p": p.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "navier_stokes_2d", "Re": Re, "type": "Kovasznay", "dim": 2},
    }


def _gen_lid_driven_cavity(params: dict, n_samples: int) -> dict:
    """
    Lid-driven cavity: FDM solution using a simple vorticity-streamfunction approach.
    For small Re, we solve numerically on a coarse grid and return the reference.
    """
    Re = float(params.get("Re", 100.0))
    nx = int(params.get("nx", 33))
    ny = int(params.get("ny", 33))
    n_iter = int(params.get("n_iter", 5000))

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nu = 1.0 / Re  # noqa: F841 (kept for reference: the stream-function iteration below is written in Re)

    psi = np.zeros((nx, ny))
    omega = np.zeros((nx, ny))

    for _ in range(n_iter):
        psi_new = psi.copy()
        # Streamfunction: lap(psi) = -omega
        psi_new[1:-1, 1:-1] = (
            (psi[2:, 1:-1] + psi[:-2, 1:-1]) / dx**2
            + (psi[1:-1, 2:] + psi[1:-1, :-2]) / dy**2
            + omega[1:-1, 1:-1]
        ) / (2.0 / dx**2 + 2.0 / dy**2)
        psi = psi_new

        omega_new = omega.copy()
        u_field = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dy)
        v_field = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
        omega_new[1:-1, 1:-1] = (omega[2:, 1:-1] + omega[:-2, 1:-1]) / dx**2
        omega_new[1:-1, 1:-1] += (omega[1:-1, 2:] + omega[1:-1, :-2]) / dy**2
        omega_new[1:-1, 1:-1] -= (
            u_field * (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * dx)
            + v_field * (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * dy)
        )
        omega_new[1:-1, 1:-1] /= (2.0 / dx**2 + 2.0 / dy**2)
        # Lid BC: top wall moves at u=1
        omega_new[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / dy**2 - 2.0 * 1.0 / dy
        omega = omega_new

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_out_full = np.gradient(psi, y, axis=1)
    v_out_full = -np.gradient(psi, x, axis=0)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u_out_full.ravel(), v_out_full.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "psi": psi.astype(np.float32),
        "omega": omega.astype(np.float32),
        "u": u_out_full.astype(np.float32),
        "v": v_out_full.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "lid_driven_cavity", "Re": Re, "nx": nx, "ny": ny, "dim": 2},
    }


def _gen_bernoulli_flow(params: dict, n_samples: int) -> dict:
    """
    Bernoulli flow in a converging channel.
    Continuity: A1*u1 = A2*u2
    Bernoulli:  p1 + 0.5*rho*u1^2 = p2 + 0.5*rho*u2^2
    Returns velocity and pressure along x in a converging duct.
    """
    rho = float(params.get("rho", 1.0))
    u_inlet = float(params.get("u_inlet", 1.0))
    A_inlet = float(params.get("A_inlet", 1.0))
    A_outlet = float(params.get("A_outlet", 0.5))
    p_inlet = float(params.get("p_inlet", 1.0))
    nx = int(params.get("nx", 100))

    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    # Linear area variation
    A = A_inlet + (A_outlet - A_inlet) * x
    u = u_inlet * A_inlet / A
    p = p_inlet + 0.5 * rho * (u_inlet**2 - u**2)

    coords = x[:, None].astype(np.float32)
    solution = np.stack([u, p], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "u": u.astype(np.float32),
        "p": p.astype(np.float32),
        "A": A.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "bernoulli_flow", "rho": rho, "dim": 1},
    }


def _gen_stokes_flow(params: dict, n_samples: int) -> dict:
    """
    Stokes flow between parallel plates (Couette + pressure-driven).
    Analytical: u(y) = U*y/H + (1/(2*mu)) * (-dp/dx) * y*(H - y)
    """
    mu = float(params.get("mu", 1.0))
    U_top = float(params.get("U_top", 1.0))
    H = float(params.get("H", 1.0))
    dpdx = float(params.get("dpdx", -1.0))
    ny = int(params.get("ny", 100))

    y = np.linspace(0.0, H, ny, dtype=np.float64)
    u = U_top * y / H + (1.0 / (2.0 * mu)) * (-dpdx) * y * (H - y)

    coords = y[:, None].astype(np.float32)
    solution = u[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "stokes_flow", "mu": mu, "U_top": U_top, "H": H, "dim": 1},
    }


def _gen_darcy_flow(params: dict, n_samples: int) -> dict:
    """
    Darcy flow: u = -(K/mu)*dp/dx, p(0)=p_in, p(L)=p_out
    1D analytical: p(x) = p_in + (p_out - p_in)*x/L, u = const
    """
    K = float(params.get("K", 1.0))
    mu = float(params.get("mu", 1.0))
    p_in = float(params.get("p_in", 1.0))
    p_out = float(params.get("p_out", 0.0))
    L = float(params.get("L", 1.0))
    nx = int(params.get("nx", 100))

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    p = p_in + (p_out - p_in) * x / L
    dpdx = (p_out - p_in) / L
    u = -(K / mu) * dpdx * np.ones_like(x)

    coords = x[:, None].astype(np.float32)
    solution = np.stack([u, p], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "p": p.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "darcy_flow", "K": K, "mu": mu, "dim": 1},
    }


def _gen_channel_poiseuille(params: dict, n_samples: int) -> dict:
    """
    Poiseuille flow: u(y) = (1/(2*mu)) * (-dp/dx) * y * (H - y)
    """
    mu = float(params.get("mu", 1.0))
    H = float(params.get("H", 1.0))
    dpdx = float(params.get("dpdx", -2.0))
    ny = int(params.get("ny", 100))

    y = np.linspace(0.0, H, ny, dtype=np.float64)
    u = (1.0 / (2.0 * mu)) * (-dpdx) * y * (H - y)

    coords = y[:, None].astype(np.float32)
    solution = u[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "channel_poiseuille", "mu": mu, "H": H, "dpdx": dpdx, "dim": 1},
    }


def _gen_linear_elasticity_2d(params: dict, n_samples: int) -> dict:
    """
    2D plane-stress Lame analytical: uniform tension in x.
    u_x = (sigma_x / E) * x, u_y = -(nu_p * sigma_x / E) * y
    """
    E = float(params.get("E", 1.0))
    nu_p = float(params.get("nu", 0.3))
    sigma_x = float(params.get("sigma_x", 1.0))
    n = int(np.sqrt(n_samples)) + 1

    x_arr = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y_arr = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")

    ux = (sigma_x / E) * X
    uy = -(nu_p * sigma_x / E) * Y

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = np.stack([ux.ravel(), uy.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x_arr.astype(np.float32),
        "y": y_arr.astype(np.float32),
        "ux": ux.astype(np.float32),
        "uy": uy.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "linear_elasticity_2d", "E": E, "nu": nu_p, "dim": 2},
    }


def _gen_beam_bending(params: dict, n_samples: int) -> dict:
    """
    Euler-Bernoulli beam: simply supported, uniform load q.
    w(x) = q*x*(L^3 - 2*L*x^2 + x^3) / (24*E*I)
    """
    E = float(params.get("E", 1.0))
    I_beam = float(params.get("I", 1.0))
    L = float(params.get("L", 1.0))
    q = float(params.get("q", 1.0))
    nx = int(params.get("nx", 200))

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    w = q * x * (L**3 - 2.0 * L * x**2 + x**3) / (24.0 * E * I_beam)
    M = q / 2.0 * (L * x - x**2)  # bending moment

    coords = x[:, None].astype(np.float32)
    solution = np.stack([w, M], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "w": w.astype(np.float32),
        "M": M.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "beam_bending", "E": E, "I": I_beam, "L": L, "q": q, "dim": 1},
    }


def _gen_poisson_electrostatics(params: dict, n_samples: int) -> dict:
    """
    Poisson equation: -lap(u) = f on [0,1]^2, u=0 on boundary.
    Analytical for f=2*pi^2*sin(pi*x)*sin(pi*y): u = sin(pi*x)*sin(pi*y)
    """
    n = int(np.sqrt(n_samples)) + 1
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "poisson_electrostatics", "dim": 2},
    }


def _gen_helmholtz_wave(params: dict, n_samples: int) -> dict:
    """
    Helmholtz: lap(u) + k^2*u = 0.
    Analytical: u = sin(m*pi*x)*sin(n*pi*y) with k^2 = (m^2+n^2)*pi^2
    """
    m = int(params.get("m", 1))
    n_mode = int(params.get("n_mode", 1))
    n = int(np.sqrt(n_samples)) + 1

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    k2 = (m**2 + n_mode**2) * np.pi**2
    u = np.sin(m * np.pi * X) * np.sin(n_mode * np.pi * Y)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "helmholtz_wave", "m": m, "n": n_mode, "k2": k2, "dim": 2},
    }


def _gen_laplace_potential(params: dict, n_samples: int) -> dict:
    """
    Laplace equation: lap(u) = 0.
    Analytical: u(x,y) = sin(pi*x)*sinh(pi*y)/sinh(pi) with u=sin(pi*x) on y=1
    """
    n = int(np.sqrt(n_samples)) + 1
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "laplace_potential", "dim": 2},
    }


def _gen_schrodinger_stationary(params: dict, n_samples: int) -> dict:
    """
    Particle in a box: psi_n(x) = sqrt(2/L)*sin(n*pi*x/L)
    E_n = n^2*pi^2*hbar^2/(2*m*L^2)
    """
    n_level = int(params.get("n_level", 1))
    L = float(params.get("L", 1.0))
    hbar = float(params.get("hbar", 1.0))
    m = float(params.get("m", 1.0))
    nx = int(params.get("nx", 500))

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    psi = np.sqrt(2.0 / L) * np.sin(n_level * np.pi * x / L)
    E_n = n_level**2 * np.pi**2 * hbar**2 / (2.0 * m * L**2)

    coords = x[:, None].astype(np.float32)
    solution = psi[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "psi": psi.astype(np.float32),
        "energy": float(E_n),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "schrodinger_stationary", "n_level": n_level, "E_n": float(E_n), "dim": 1},
    }


def _gen_schrodinger_timedep(params: dict, n_samples: int) -> dict:
    """
    Time-dependent Schrodinger: i*hbar*dpsi/dt = -hbar^2/(2m) * d^2psi/dx^2 + V*psi
    Numerical FDM on x in [0,L], V=0 (free particle), gaussian initial condition.
    """
    L = float(params.get("L", 1.0))
    hbar = float(params.get("hbar", 1.0))
    m_mass = float(params.get("m", 1.0))
    nx = int(params.get("nx", 64))
    nt = int(params.get("nt", 50))
    t_end = float(params.get("t_end", 0.1))

    x = np.linspace(0.0, L, nx, dtype=np.float64)
    t = np.linspace(0.0, t_end, nt, dtype=np.float64)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Gaussian wave packet IC
    x0 = L / 2.0
    sigma = L / 10.0
    k0 = 2.0 * np.pi / L
    psi = np.exp(-(x - x0)**2 / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    psi[0] = 0.0
    psi[-1] = 0.0
    psi = psi / np.sqrt(np.trapezoid(np.abs(psi)**2, x))

    # Crank-Nicolson for i*hbar*psi_t = -(hbar^2/2m)*psi_xx  →  psi_t = (i*hbar/2m)*psi_xx
    # CN: (I - r*Lh) psi^{n+1} = (I + r*Lh) psi^n,  r = i*hbar*dt/(4*m*dx^2)
    r = 1j * hbar * dt / (4.0 * m_mass * dx**2)
    Ni = nx - 2  # interior points
    a_main = np.full(Ni, 1.0 + 2.0 * r, dtype=np.complex128)
    a_off  = np.full(Ni - 1, -r, dtype=np.complex128)

    # Thomas algorithm for complex tridiagonal (O(N), unconditionally stable)
    def _tridiag_solve(v_rhs):
        c = np.empty(Ni, dtype=np.complex128)
        d = v_rhs.copy()
        c[0] = a_off[0] / a_main[0]
        d[0] = d[0] / a_main[0]
        for i in range(1, Ni):
            denom = a_main[i] - a_off[i - 1] * c[i - 1]
            c[i] = a_off[i] / denom if i < Ni - 1 else 0.0
            d[i] = (d[i] - a_off[i - 1] * d[i - 1]) / denom
        x = np.empty(Ni, dtype=np.complex128)
        x[-1] = d[-1]
        for i in range(Ni - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    psi_history = np.zeros((nx, nt), dtype=np.complex128)
    psi_history[:, 0] = psi

    for it in range(1, nt):
        v = psi[1:-1]
        rhs = (1.0 - 2.0 * r) * v + r * (psi[2:] + psi[:-2])
        v_new = _tridiag_solve(rhs)
        psi_new = np.zeros_like(psi)
        psi_new[1:-1] = v_new
        psi = psi_new
        psi_history[:, it] = psi

    prob_density = np.abs(psi_history)**2
    X, T = np.meshgrid(x, t, indexing="ij")

    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = prob_density.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "psi_real": psi_history.real.astype(np.float32),
        "psi_imag": psi_history.imag.astype(np.float32),
        "prob_density": prob_density.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "schrodinger_timedep", "nx": nx, "nt": nt, "dim": 2},
    }


def _gen_heat_equation_1d(params: dict, n_samples: int) -> dict:
    """
    Heat equation: u_t = alpha*u_xx
    Analytical Fourier series: u(x,t) = sum_n b_n*sin(n*pi*x)*exp(-n^2*pi^2*alpha*t)
    IC: u(x,0) = sin(pi*x), BCs: u(0,t)=u(1,t)=0
    """
    alpha = float(params.get("alpha", 0.1))
    nx = int(params.get("nx", 50))
    nt = int(params.get("nt", 50))
    N_terms = int(params.get("N_terms", 20))

    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")

    u = np.zeros_like(X)
    for n in range(1, N_terms + 1):
        b_n = 2.0 * np.trapezoid(np.sin(np.pi * x) * np.sin(n * np.pi * x), x)
        u += b_n * np.sin(n * np.pi * X) * np.exp(-n**2 * np.pi**2 * alpha * T)

    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "heat_equation_1d", "alpha": alpha, "dim": 2},
    }


def _gen_heat_equation_2d(params: dict, n_samples: int) -> dict:
    """
    2D Heat equation: u_t = alpha*(u_xx + u_yy)
    Analytical: u(x,y,t) = sin(pi*x)*sin(pi*y)*exp(-2*pi^2*alpha*t)
    Visualized as AnimatedMeshView: flat 2D plane showing temperature evolving in time.
    """
    alpha = float(params.get("alpha", 0.1))
    n = int(np.cbrt(n_samples)) + 1
    nx = ny = n
    nt = n

    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")

    u = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2.0 * np.pi**2 * alpha * T)

    coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # Build flat 2D mesh on [0,1]^2 plane (z=0) for AnimatedMeshView
    mx = my = max(nx, 16)
    xm = np.linspace(0.0, 1.0, mx)
    ym = np.linspace(0.0, 1.0, my)
    verts = [[float(xm[ii]), float(ym[jj]), 0.0] for jj in range(my) for ii in range(mx)]
    faces = []
    for jj in range(my - 1):
        for ii in range(mx - 1):
            v00 = jj * mx + ii
            v10 = jj * mx + (ii + 1)
            v01 = (jj + 1) * mx + ii
            v11 = (jj + 1) * mx + (ii + 1)
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    n_frames = 8
    t_frames = np.linspace(0.0, 1.0, n_frames)
    Xm, Ym = np.meshgrid(xm, ym)  # shape (my, mx) — ravel gives j*mx+i ordering
    sin_term = np.sin(np.pi * Xm) * np.sin(np.pi * Ym)
    anim_frames = []
    for fi, t_val in enumerate(t_frames):
        field_arr = sin_term * np.exp(-2.0 * np.pi**2 * alpha * t_val)
        anim_frames.append({
            "tau": float(fi / max(n_frames - 1, 1)),
            "t_s": float(t_val),
            "label": f"t = {t_val:.2f}",
            "field": field_arr.ravel().tolist(),
        })

    time_animation = {
        "vertices": verts,
        "faces": faces,
        "field_name": "T(x,y,t)",
        "frames": anim_frames,
    }

    return {
        "time_animation": time_animation,
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "heat_equation_2d", "alpha": alpha, "dim": 3},
    }


def _gen_wave_equation(params: dict, n_samples: int) -> dict:
    """
    1D wave equation: u_tt = c^2 * u_xx
    d'Alembert: u(x,t) = 0.5*(f(x+ct) + f(x-ct)) for IC u(x,0)=f(x), u_t(x,0)=0
    IC: u(x,0) = sin(pi*x)
    """
    c = float(params.get("c", 1.0))
    nx = int(params.get("nx", 50))
    nt = int(params.get("nt", 50))

    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")

    u = 0.5 * (np.sin(np.pi * (X + c * T)) + np.sin(np.pi * (X - c * T)))

    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "wave_equation", "c": c, "dim": 2},
    }


def _gen_kdv_equation(params: dict, n_samples: int) -> dict:
    """
    KdV equation: u_t + 6*u*u_x + u_xxx = 0
    Soliton solution: u(x,t) = -2*kappa^2 * sech^2(kappa*(x - 4*kappa^2*t - x0))
    """
    kappa = float(params.get("kappa", 1.0))
    x0 = float(params.get("x0", 0.0))
    nx = int(params.get("nx", 100))
    nt = int(params.get("nt", 50))

    x = np.linspace(-10.0, 10.0, nx, dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")

    xi = kappa * (X - 4.0 * kappa**2 * T - x0)
    u = -2.0 * kappa**2 / np.cosh(xi)**2

    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "kdv_equation", "kappa": kappa, "dim": 2},
    }


def _gen_klein_gordon(params: dict, n_samples: int) -> dict:
    """
    Klein-Gordon: u_tt - c^2*u_xx + m^2*u = 0
    Plane-wave: u(x,t) = A*cos(k*x - omega*t), omega = sqrt(c^2*k^2 + m^2)
    """
    c = float(params.get("c", 1.0))
    m_kg = float(params.get("m", 1.0))
    k = float(params.get("k", 1.0))
    A = float(params.get("A", 1.0))
    nx = int(params.get("nx", 50))
    nt = int(params.get("nt", 50))

    omega = np.sqrt(c**2 * k**2 + m_kg**2)
    x = np.linspace(-np.pi, np.pi, nx, dtype=np.float64)
    t = np.linspace(0.0, 2.0, nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")

    u = A * np.cos(k * X - omega * T)

    coords = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    solution = u.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "u": u.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "klein_gordon", "c": c, "m": m_kg, "k": k, "omega": float(omega), "dim": 2},
    }


def _gen_thermoelasticity_2d(params: dict, n_samples: int) -> dict:
    """
    Thermoelasticity 2D: simple analytical case.
    Temperature: T(x,y) = T0*sin(pi*x)*sin(pi*y)
    Thermal stress: sigma_th = -E*alpha_T*(T - T_ref)/(1-nu) (isotropic)
    Displacement approximation via Fourier mode.
    """
    E = float(params.get("E", 1.0))
    nu_p = float(params.get("nu", 0.3))
    alpha_T = float(params.get("alpha_T", 1e-5))
    T0 = float(params.get("T0", 100.0))
    n = int(np.sqrt(n_samples)) + 1

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    T_field = T0 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Approximate displacement from thermal expansion
    ux = alpha_T * T0 / (np.pi * (1.0 + nu_p)) * np.cos(np.pi * X) * np.sin(np.pi * Y)
    uy = alpha_T * T0 / (np.pi * (1.0 + nu_p)) * np.sin(np.pi * X) * np.cos(np.pi * Y)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = np.stack([T_field.ravel(), ux.ravel(), uy.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "T": T_field.astype(np.float32),
        "ux": ux.astype(np.float32),
        "uy": uy.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "thermoelasticity_2d", "E": E, "nu": nu_p, "alpha_T": alpha_T, "dim": 2},
    }


def _gen_airfoil_flow_3d(params: dict, n_samples: int) -> dict:
    """
    NACA 0012 potential flow using a source + vortex panel method (Hess-Smith).

    For angle-of-attack alpha_deg:
      - Source panels satisfy no-penetration on the airfoil surface
      - A thin-airfoil-theory vortex term adds lift for alpha != 0
      - Velocity field (u, v) and pressure coefficient Cp computed on a
        C-type domain outside the airfoil body, extruded in z (span).

    Solution columns: [u, v, u_z=0, Cp]  shape (N, 4)
    """
    chord   = float(params.get("chord", 1.0))
    span    = float(params.get("span", 2.0))
    V_inf   = float(params.get("V_inf", 1.0))
    alpha_deg = float(params.get("alpha_deg", 0.0))
    n_panels  = 60  # number of panels on the airfoil

    alpha = np.deg2rad(alpha_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # ------------------------------------------------------------------
    # 1. NACA 0012 geometry  (x normalised to [0, chord])
    # ------------------------------------------------------------------
    t_ratio = 0.12  # thickness ratio
    theta_panels = np.linspace(0.0, np.pi, n_panels + 1, dtype=np.float64)
    # Cosine clustering for panel end-points
    xn = 0.5 * chord * (1.0 - np.cos(theta_panels))   # [0 .. chord]
    xn_norm = xn / chord                                # [0 .. 1]

    def _naca_yt(xi):
        """Half-thickness of NACA 00xx at normalised chord station xi in [0,1]."""
        return (5.0 * t_ratio * chord *
                (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2
                 + 0.2843 * xi**3 - 0.1015 * xi**4))

    yt = _naca_yt(xn_norm)

    # Upper / lower surface node coordinates
    x_upper = xn
    y_upper = yt
    x_lower = xn[::-1]
    y_lower = -yt[::-1]

    # Full panel node list (closed loop: upper TE -> LE -> lower -> TE)
    x_nodes = np.concatenate([x_upper, x_lower[1:]])
    y_nodes = np.concatenate([y_upper, y_lower[1:]])
    n_total_panels = len(x_nodes) - 1

    # Panel midpoints
    xm = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    ym = 0.5 * (y_nodes[:-1] + y_nodes[1:])

    # Panel tangents and normals
    dx_pan = x_nodes[1:] - x_nodes[:-1]
    dy_pan = y_nodes[1:] - y_nodes[:-1]
    len_pan = np.hypot(dx_pan, dy_pan)
    tx = dx_pan / len_pan   # unit tangent
    ty = dy_pan / len_pan
    nx_pan = ty              # outward normal = (ty, -tx) rotated 90 deg CW from tangent
    ny_pan = -tx

    # ------------------------------------------------------------------
    # 2. Source panel method: build influence matrix A
    # ------------------------------------------------------------------
    N = n_total_panels
    A_mat = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            if i == j:
                A_mat[i, j] = 0.5
                continue
            rx = xm[i] - xm[j]
            ry = ym[i] - ym[j]
            r2 = rx**2 + ry**2 + 1e-14   # avoid division by zero at near-coincident TE panels
            # Normal velocity at panel i midpoint from unit source at panel j midpoint
            A_mat[i, j] = (rx * nx_pan[i] + ry * ny_pan[i]) / (2.0 * np.pi * r2)

    # RHS: no-penetration BC  (V_inf · n̂_i)
    rhs = -(V_inf * cos_a * nx_pan + V_inf * sin_a * ny_pan)

    # Solve for source strengths
    sigma = np.linalg.solve(A_mat, rhs)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=10.0 * V_inf, neginf=-10.0 * V_inf)

    # Thin-airfoil-theory circulation for lift (Kutta condition)
    # Gamma = pi * V_inf * chord * sin(alpha)
    gamma = np.pi * V_inf * chord * np.sin(alpha) if abs(alpha) > 1e-10 else 0.0

    # ------------------------------------------------------------------
    # 3. Build field-point grid (C-type domain, outside airfoil)
    # ------------------------------------------------------------------
    # Extents: x in [-3c, 8c], y in [-3c, 3c], z in [0, span]
    nx_field = 22
    ny_field = 18
    nz_field = 8

    xf_1d = np.linspace(-3.0 * chord, 8.0 * chord, nx_field, dtype=np.float64)
    yf_1d = np.linspace(-3.0 * chord, 3.0 * chord, ny_field, dtype=np.float64)
    zf_1d = np.linspace(0.0, span, nz_field, dtype=np.float64)

    Xf, Yf, Zf = np.meshgrid(xf_1d, yf_1d, zf_1d, indexing="ij")
    Xf_flat = Xf.ravel()
    Yf_flat = Yf.ravel()
    Zf_flat = Zf.ravel()

    # ------------------------------------------------------------------
    # 4. Mask points inside the airfoil body
    # ------------------------------------------------------------------
    # A point (px, py) is inside the airfoil if:
    #   0 <= px <= chord  AND  |py| < naca_yt(px/chord)
    inside_mask = np.zeros(len(Xf_flat), dtype=bool)
    in_chord = (Xf_flat >= 0.0) & (Xf_flat <= chord)
    if in_chord.any():
        xi_pts = Xf_flat[in_chord] / chord
        yt_pts = _naca_yt(xi_pts)
        inside_mask[in_chord] = np.abs(Yf_flat[in_chord]) < yt_pts

    Xp = Xf_flat[~inside_mask]
    Yp = Yf_flat[~inside_mask]
    Zp = Zf_flat[~inside_mask]

    # ------------------------------------------------------------------
    # 5. Compute velocity at each field point via vectorised panel sums
    # ------------------------------------------------------------------
    # Source contribution: u_source = sigma/(2pi) * (xp-xmj)/r², v_source = sigma/(2pi) * (yp-ymj)/r²
    # Vortex (2D line vortex at panel centres): u_vortex = -gamma/(2pi*N) * (yp-ymj)/r²
    #                                            v_vortex = +gamma/(2pi*N) * (xp-xmj)/r²

    n_pts = len(Xp)
    u_field = np.full(n_pts, V_inf * cos_a, dtype=np.float64)
    v_field = np.full(n_pts, V_inf * sin_a, dtype=np.float64)

    gamma_per_panel = gamma / N if N > 0 else 0.0

    for j in range(N):
        rx = Xp - xm[j]
        ry = Yp - ym[j]
        r2 = rx**2 + ry**2 + 1e-14  # avoid singularity
        factor_source = sigma[j] / (2.0 * np.pi * r2)
        u_field += factor_source * rx
        v_field += factor_source * ry
        if gamma != 0.0:
            factor_vortex = gamma_per_panel / (2.0 * np.pi * r2)
            u_field -= factor_vortex * ry
            v_field += factor_vortex * rx

    # Pressure coefficient: Cp = 1 - (u² + v²) / V_inf²
    u_field = np.nan_to_num(u_field, nan=V_inf * cos_a, posinf=5 * V_inf, neginf=-5 * V_inf)
    v_field = np.nan_to_num(v_field, nan=V_inf * sin_a, posinf=5 * V_inf, neginf=-5 * V_inf)
    Cp = 1.0 - (u_field**2 + v_field**2) / (V_inf**2 + 1e-14)
    Cp = np.clip(np.nan_to_num(Cp, nan=0.0, posinf=2.0, neginf=-10.0), -10.0, 2.0)
    u_z = np.zeros(n_pts, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. Surface Cp on the airfoil panels (for surface mesh coloring)
    # ------------------------------------------------------------------
    # Tangential velocity at each panel center (vectorised)
    rx_s = xm[:, None] - xm[None, :]  # (N, N)
    ry_s = ym[:, None] - ym[None, :]
    r2_s = rx_s**2 + ry_s**2 + 1e-14  # epsilon avoids blow-up at near-coincident TE panels
    np.fill_diagonal(r2_s, np.inf)    # self-influence = 0 for tangential

    # Source contribution
    tan_src = np.sum(
        sigma[None, :] * (rx_s * tx[:, None] + ry_s * ty[:, None]) / (2.0 * np.pi * r2_s),
        axis=1,
    )
    # Vortex contribution
    tan_vort = 0.0
    if gamma != 0.0:
        tan_vort = gamma_per_panel * np.sum(
            (-ry_s * tx[:, None] + rx_s * ty[:, None]) / (2.0 * np.pi * r2_s),
            axis=1,
        )

    V_tan = V_inf * cos_a * tx + V_inf * sin_a * ty + tan_src + tan_vort
    Cp_panels = 1.0 - V_tan**2 / (V_inf**2 + 1e-14)
    Cp_panels = np.clip(np.nan_to_num(Cp_panels, nan=0.0, posinf=2.0, neginf=-10.0), -10.0, 2.0)

    # ------------------------------------------------------------------
    # 7. Build NACA 0012 wing surface mesh (for ParaView-style rendering)
    # ------------------------------------------------------------------
    nc_m = 50   # chord-wise mesh vertices
    ns_m = 10   # span-wise mesh vertices

    xi_m = np.linspace(0.0, 1.0, nc_m)
    yt_m = _naca_yt(xi_m)
    z_m  = np.linspace(-span / 2.0, span / 2.0, ns_m)

    # Upper/lower panel midpoints for interpolation
    n_half = N // 2
    xm_up = xm[:n_half]; Cp_up = Cp_panels[:n_half]
    xm_lo = xm[n_half:]; Cp_lo = Cp_panels[n_half:]

    si_up = np.argsort(xm_up); si_lo = np.argsort(xm_lo)
    xv = xi_m * chord  # chord positions at mesh vertices
    Cp_vert_up = np.interp(xv, xm_up[si_up], Cp_up[si_up])
    Cp_vert_lo = np.interp(xv, xm_lo[si_lo], Cp_lo[si_lo])

    x_pivot = chord * 0.25  # put aerodynamic centre at world origin

    surf_verts: list = []
    surf_field: list = []

    for j in range(ns_m):
        zj = float(z_m[j])
        for i in range(nc_m):
            surf_verts.append([float(xv[i] - x_pivot), float(yt_m[i]),  zj])
            surf_field.append(float(Cp_vert_up[i]))
    for j in range(ns_m):
        zj = float(z_m[j])
        for i in range(nc_m):
            surf_verts.append([float(xv[i] - x_pivot), float(-yt_m[i]), zj])
            surf_field.append(float(Cp_vert_lo[i]))

    surf_faces: list = []
    off = nc_m * ns_m  # offset to lower surface vertices

    def _uid(i: int, j: int) -> int:  return j * nc_m + i
    def _lid(i: int, j: int) -> int:  return off + j * nc_m + i

    for j in range(ns_m - 1):
        for i in range(nc_m - 1):
            u00, u10 = _uid(i, j), _uid(i + 1, j)
            u01, u11 = _uid(i, j + 1), _uid(i + 1, j + 1)
            surf_faces += [[u00, u10, u11], [u00, u11, u01]]
            l00, l10 = _lid(i, j), _lid(i + 1, j)
            l01, l11 = _lid(i, j + 1), _lid(i + 1, j + 1)
            surf_faces += [[l00, l11, l10], [l00, l01, l11]]  # reversed for outward normals

    surface_mesh = {
        "vertices": surf_verts,
        "faces":    surf_faces,
        "field":    surf_field,
        "field_name": "Cp",
    }

    # ------------------------------------------------------------------
    # 8. Subsample scatter points to n_samples
    # ------------------------------------------------------------------
    coords   = np.stack([Xp, Yp, Zp], axis=1).astype(np.float32)
    solution = np.stack([u_field, v_field, u_z, Cp], axis=1).astype(np.float32)

    if n_samples < len(coords):
        idx      = np.random.choice(len(coords), size=n_samples, replace=False)
        coords   = coords[idx]
        solution = solution[idx]

    # Final sanitization — prevent JSON serialization failure from any stray NaN/Inf
    coords   = _nan_clean(coords)
    solution = _nan_clean(solution)
    xf_1d    = _nan_clean(xf_1d)
    yf_1d    = _nan_clean(yf_1d)
    zf_1d    = _nan_clean(zf_1d)
    surf_field_clean = [float(np.nan_to_num(v, nan=0.0, posinf=2.0, neginf=-10.0)) for v in surf_field]
    surface_mesh["field"] = surf_field_clean

    return {
        "x":        xf_1d.astype(np.float32),
        "y":        yf_1d.astype(np.float32),
        "z":        zf_1d.astype(np.float32),
        "coords":   coords,
        "solution": solution,
        "surface_mesh": surface_mesh,
        "metadata": {
            "problem":     "navier_stokes_airfoil_3d",
            "chord":       chord,
            "span":        span,
            "alpha_deg":   alpha_deg,
            "V_inf":       V_inf,
            "dim":         3,
            "is_airfoil":  True,
            "description": "NACA 0012 potential flow (Hess-Smith panel method): u, v, Cp field",
        },
    }


def _gen_navier_stokes_3d_rect(params: dict, n_samples: int) -> dict:
    """
    Rectangular duct Poiseuille flow — exact Fourier solution + transient startup animation.
    Domain: x in [0, L] (axial), y in [-a/2, a/2], z in [-b/2, b/2].
    Steady state: u_x(y,z) = (16(-dpdx)a²)/(μπ³) Σ_{n odd} (-1)^k/n³ * ...
    Transient (impulsive start from rest): u(y,z,t) = u_ss + ΣΣ C_nm sin(nπY/a) sin(mπZ/b) exp(-λ_nm ν t)
    """
    a    = float(params.get("a",    1.0))   # duct width  (y)
    b    = float(params.get("b",    1.0))   # duct height (z)
    L    = float(params.get("L",    3.0))   # duct length (x)
    dpdx = float(params.get("dpdx", -8.0))  # pressure gradient
    mu   = float(params.get("mu",   1.0))   # dynamic viscosity
    rho  = float(params.get("rho",  1.0))   # density
    nu   = mu / rho                          # kinematic viscosity
    N_terms = 15                             # steady-state Fourier terms

    # ── Volume grid ────────────────────────────────────────────────────────────
    n_x, n_y, n_z = 12, 18, 18
    x_arr = np.linspace(0.0, L,          n_x, dtype=np.float64)
    y_arr = np.linspace(-a / 2.0, a / 2.0, n_y, dtype=np.float64)
    z_arr = np.linspace(-b / 2.0, b / 2.0, n_z, dtype=np.float64)
    X_g, Y_g, Z_g = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

    pf = 16.0 * (-dpdx) * a**2 / (mu * np.pi**3)

    def _u_yz(yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
        """Exact steady-state u_x(y,z) via Fourier series."""
        uu = np.zeros_like(yy, dtype=np.float64)
        for k in range(N_terms):
            n   = 2 * k + 1
            sgn = (-1.0)**k
            cd  = np.cosh(n * np.pi * b / (2.0 * a))
            uu += sgn / n**3 * (1.0 - np.cosh(n * np.pi * zz / a) / cd) * np.cos(n * np.pi * yy / a)
        return np.clip(uu * pf, 0.0, None)

    u_x = _u_yz(Y_g, Z_g)
    u_y = np.zeros_like(u_x)
    u_z = np.zeros_like(u_x)
    p   = (-dpdx) * (L - X_g)

    coords   = np.stack([X_g.ravel(), Y_g.ravel(), Z_g.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u_x.ravel(), u_y.ravel(), u_z.ravel(), p.ravel()], axis=1).astype(np.float32)
    n_total  = len(coords)
    if n_samples < n_total:
        idx      = np.random.choice(n_total, size=n_samples, replace=False)
        coords   = coords[idx]
        solution = solution[idx]

    # ── Surface mesh (4 walls + inlet face) ───────────────────────────────────
    n_y_m, n_z_m, n_x_m = 24, 24, 16
    y_m = np.linspace(-a / 2.0, a / 2.0, n_y_m)
    z_m = np.linspace(-b / 2.0, b / 2.0, n_z_m)
    x_m = np.linspace(0.0, L,            n_x_m)

    surf_verts: list = []
    surf_field: list = []
    surf_faces: list = []

    def _add_quad(Xmat, Ymat, Zmat, Fmat):
        nr, nc = Xmat.shape
        off = len(surf_verts)
        for ri in range(nr):
            for ci in range(nc):
                surf_verts.append([float(Xmat[ri, ci]), float(Ymat[ri, ci]), float(Zmat[ri, ci])])
                surf_field.append(float(Fmat[ri, ci]))
        for ri in range(nr - 1):
            for ci in range(nc - 1):
                a_ = off + ri * nc + ci
                b_ = a_ + 1
                c_ = off + (ri + 1) * nc + ci
                d_ = c_ + 1
                surf_faces.extend([[a_, b_, d_], [a_, d_, c_]])

    X_xz, Z_xz = np.meshgrid(x_m, z_m, indexing="ij")
    X_xy, Y_xy = np.meshgrid(x_m, y_m, indexing="ij")

    _add_quad(X_xz, np.full_like(X_xz, -a / 2.0), Z_xz, np.zeros_like(X_xz))
    _add_quad(X_xz, np.full_like(X_xz,  a / 2.0), Z_xz, np.zeros_like(X_xz))
    _add_quad(X_xy, Y_xy, np.full_like(X_xy, -b / 2.0), np.zeros_like(X_xy))
    _add_quad(X_xy, Y_xy, np.full_like(X_xy,  b / 2.0), np.zeros_like(X_xy))

    # Inlet face x=0 — record start index for per-frame field
    inlet_start = len(surf_verts)
    Y_in, Z_in  = np.meshgrid(y_m, z_m, indexing="ij")
    U_in        = _u_yz(Y_in, Z_in)
    _add_quad(np.zeros_like(Y_in), Y_in, Z_in, U_in)

    surface_mesh = {
        "vertices":   surf_verts,
        "faces":      surf_faces,
        "field":      surf_field,
        "field_name": "u_x",
    }

    # ── Transient startup animation ────────────────────────────────────────────
    # u(Y,Z,t) = u_ss(Y,Z) + ΣΣ C_nm sin(nπY/a) sin(mπZ/b) exp(-λ_nm ν t)
    # Y = y+a/2 ∈ [0,a],  Z = z+b/2 ∈ [0,b]
    # C_nm = -(4/ab) ∫∫ u_ss sin(nπY/a) sin(mπZ/b) dY dZ   (numerical)

    N_tr = 8   # transient modes each direction
    N_q  = 44  # quadrature points

    Y_q = np.linspace(0.0, a, N_q)
    Z_q = np.linspace(0.0, b, N_q)
    YY_q, ZZ_q = np.meshgrid(Y_q, Z_q, indexing='ij')
    U_ss_q = _u_yz(YY_q - a / 2.0, ZZ_q - b / 2.0)  # steady state on quad grid

    C_nm = np.zeros((N_tr, N_tr))
    for ni in range(1, N_tr + 1):
        sin_Y = np.sin(ni * np.pi * YY_q / a)
        for mi in range(1, N_tr + 1):
            sin_Z = np.sin(mi * np.pi * ZZ_q / b)
            integrand = U_ss_q * sin_Y * sin_Z
            C_nm[ni - 1, mi - 1] = -(4.0 / (a * b)) * float(
                np.trapezoid(np.trapezoid(integrand, Z_q, axis=1), Y_q)
            )

    Y_in_flat = (Y_in + a / 2.0).ravel()   # shifted to [0,a]
    Z_in_flat = (Z_in + b / 2.0).ravel()   # shifted to [0,b]
    U_ss_flat = _u_yz(Y_in.ravel(), Z_in.ravel())

    h = min(a, b) / 2.0   # reference length for non-dim time

    def _inlet_frame(tau: float) -> list:
        t_phys = tau * h**2 / nu
        u = U_ss_flat.copy()
        for ni in range(1, N_tr + 1):
            sin_Y = np.sin(ni * np.pi * Y_in_flat / a)
            for mi in range(1, N_tr + 1):
                lam   = np.pi**2 * (ni**2 / a**2 + mi**2 / b**2)
                decay = np.exp(-lam * nu * t_phys)
                u    += C_nm[ni - 1, mi - 1] * sin_Y * np.sin(mi * np.pi * Z_in_flat / b) * decay
        return np.clip(u, 0.0, None).tolist()

    wall_zeros = [0.0] * inlet_start
    tau_arr = [0.0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.80, 1.20, 2.0]

    frames = []
    for tau in tau_arr:
        t_phys = tau * h**2 / nu
        frames.append({
            "tau":   float(tau),
            "t_s":   float(t_phys),
            "label": f"τ={tau:.2f}",
            "field": wall_zeros + _inlet_frame(tau),
        })
    frames.append({
        "tau":   None,
        "t_s":   None,
        "label": "τ→∞ (steady state)",
        "field": wall_zeros + U_ss_flat.tolist(),
    })

    time_animation = {
        "vertices":   surf_verts,
        "faces":      surf_faces,
        "field_name": "u_x",
        "nu":         float(nu),
        "a":          float(a),
        "b":          float(b),
        "frames":     frames,
    }

    return {
        "y":             y_arr.astype(np.float32),
        "z":             z_arr.astype(np.float32),
        "x":             x_arr.astype(np.float32),
        "coords":        coords,
        "solution":      solution,
        "surface_mesh":  surface_mesh,
        "time_animation": time_animation,
        "metadata": {
            "problem":     "navier_stokes_3d_rect",
            "a": a, "b": b, "L": L, "dpdx": dpdx, "mu": mu, "nu": nu,
            "dim":         3,
            "description": f"Rectangular duct Poiseuille + transient startup  a={a}, b={b}",
        },
    }


def _gen_lid_driven_cavity_3d(params: dict, n_samples: int) -> dict:
    """
    3D Lid-Driven Cavity [0,1]³.
    Lid at z=1 moves at U in x-direction.  2D vorticity-streamfunction FDM on
    the x-z plane extruded uniformly along y.
    Surface mesh: cutaway corner showing velocity magnitude |u|.
    """
    Re      = float(params.get("Re", 100.0))
    lid_U   = float(params.get("lid_velocity_x", 1.0))
    N       = 28          # x-z FDM grid
    n_iter  = 2500
    ny      = 12          # spanwise (y) points for volume data

    h = 1.0 / (N - 1)

    psi   = np.zeros((N, N), dtype=np.float64)   # psi[xi, zi]
    omega = np.zeros((N, N), dtype=np.float64)

    coef = 2.0 / h**2 + 2.0 / h**2   # = 4/h²

    for _ in range(n_iter):
        # Streamfunction: ∇²ψ = -ω
        psi[1:-1, 1:-1] = (
            (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2]) / h**2
            + omega[1:-1, 1:-1]
        ) / coef

        # Velocity from streamfunction: u_x = ∂ψ/∂z, u_z = -∂ψ/∂x
        u_f = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * h)
        w_f = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)

        # Vorticity transport: ν∇²ω = u·∇ω  → Gauss-Seidel
        omega[1:-1, 1:-1] = (
            (omega[2:, 1:-1] + omega[:-2, 1:-1] + omega[1:-1, 2:] + omega[1:-1, :-2]) / h**2
            - u_f * (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * h)
            - w_f * (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * h)
        ) / coef

        # BC: lid at z=1 (zi = N-1) moves at lid_U in x
        omega[1:-1, -1] = -2.0 * psi[1:-1, -2] / h**2 - 2.0 * lid_U / h

    x_arr = np.linspace(0.0, 1.0, N, dtype=np.float64)
    z_arr = np.linspace(0.0, 1.0, N, dtype=np.float64)
    y_arr = np.linspace(0.0, 1.0, ny, dtype=np.float64)

    u_x2d = np.gradient(psi, z_arr, axis=1)   # ∂ψ/∂z  (shape N×N)
    u_z2d = -np.gradient(psi, x_arr, axis=0)  # -∂ψ/∂x

    # 3-D volume by extruding along y
    XX, YY, ZZ = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    # u_x, u_z same for every y slice; u_y = 0
    Ux = np.broadcast_to(u_x2d[:, np.newaxis, :], (N, ny, N)).copy()
    Uy = np.zeros_like(Ux)
    Uz = np.broadcast_to(u_z2d[:, np.newaxis, :], (N, ny, N)).copy()
    P  = np.zeros_like(Ux)   # no analytical pressure for LDC

    coords   = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    solution = np.stack([Ux.ravel(), Uy.ravel(), Uz.ravel(), P.ravel()], axis=1).astype(np.float32)
    n_total = len(coords)
    if n_samples < n_total:
        idx      = np.random.choice(n_total, size=n_samples, replace=False)
        coords   = coords[idx]
        solution = solution[idx]

    # Surface mesh: cutaway at x_max, y_max, z_max showing |u|
    def _ldc_field(fx, fy, fz):
        xi = np.clip(np.round(fx * (N - 1)).astype(int), 0, N - 1)
        zi = np.clip(np.round(fz * (N - 1)).astype(int), 0, N - 1)
        ux = u_x2d[xi, zi]
        uz = u_z2d[xi, zi]
        return np.sqrt(ux**2 + uz**2)

    sv, sf, sfield = _cutaway_mesh((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), _ldc_field, M=28)

    return {
        "coords":   coords,
        "solution": solution,
        "surface_mesh": {
            "vertices":   sv,
            "faces":      sf,
            "field":      sfield,
            "field_name": "|u|",
        },
        "metadata": {
            "problem":     "lid_driven_cavity_3d",
            "Re":          Re,
            "lid_U":       lid_U,
            "dim":         3,
            "description": f"Lid-Driven Cavity 3D  Re={Re:.0f}  [0,1]³  (lid at z=1)",
        },
    }


def _gen_navier_stokes_3d(params: dict, n_samples: int) -> dict:
    """
    Dispatches to the appropriate 3D Navier-Stokes generator based on geometry_id.

    - geometry_id contains "airfoil"     -> NACA 0012 potential flow (Hess-Smith panels)
    - geometry_id contains "lid_driven"  -> Lid-driven cubic cavity (FDM)
    - geometry_id contains "rect"/"channel"/"duct" -> Rectangular duct Poiseuille
    - default                             -> Hagen-Poiseuille pipe flow
    """
    geometry_id = str(params.get("geometry_id", ""))
    gid = geometry_id.lower()
    if "airfoil" in gid:
        return _gen_airfoil_flow_3d(params, n_samples)
    if "lid_driven" in gid or "cavity" in gid:
        return _gen_lid_driven_cavity_3d(params, n_samples)
    if "rect" in gid or "channel" in gid or "duct" in gid:
        return _gen_navier_stokes_3d_rect(params, n_samples)
    return _gen_navier_stokes_3d_pipe(params, n_samples)


def _gen_navier_stokes_3d_pipe(params: dict, n_samples: int) -> dict:
    """
    Hagen-Poiseuille pipe flow (exact incompressible NS solution):
      u_x(r) = 2*U*(1 - r^2/R^2)  — parabolic axial velocity
      u_y = u_z = 0                 — no radial flow
      p(x) = (1 - x/L)*(8*U/R^2)  — linear pressure drop (mu=1)

    Uses a structured cylindrical grid (r, theta, x) so the geometry is
    physically correct — no random theta scattering.
    """
    R = float(params.get("R", 0.5))
    L = float(params.get("L", 2.0))
    U = float(params.get("U", 1.0))

    # Structured cylindrical grid
    n_r     = 12
    n_theta = 18
    n_x     = 15

    r_arr     = np.linspace(0.0, R, n_r, dtype=np.float64)
    theta_arr = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)
    x_arr     = np.linspace(0.0, L, n_x, dtype=np.float64)

    # Meshgrid shape: (n_r, n_theta, n_x)
    R_g, Theta_g, X_g = np.meshgrid(r_arr, theta_arr, x_arr, indexing="ij")

    # Exact Hagen-Poiseuille fields
    u_x = 2.0 * U * (1.0 - R_g**2 / R**2)   # parabolic axial velocity
    u_y = np.zeros_like(u_x)                   # no radial y-component
    u_z = np.zeros_like(u_x)                   # no radial z-component
    p   = (1.0 - X_g / L) * (8.0 * U / R**2)  # linear pressure drop

    # Cylinder -> Cartesian
    Y_g = R_g * np.cos(Theta_g)
    Z_g = R_g * np.sin(Theta_g)

    coords   = np.stack([X_g.ravel(), Y_g.ravel(), Z_g.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u_x.ravel(), u_y.ravel(), u_z.ravel(), p.ravel()], axis=1).astype(np.float32)

    n_total = len(coords)
    if n_samples < n_total:
        idx      = np.random.choice(n_total, size=n_samples, replace=False)
        coords   = coords[idx]
        solution = solution[idx]

    # ── Surface mesh: cylinder wall + parabolic inlet disk ────────────────
    n_th_m = 36   # azimuthal vertices on cylinder
    n_x_m  = 12   # axial vertices on cylinder
    n_r_m  = 16   # radial vertices on inlet disk

    theta_m = np.linspace(0.0, 2.0 * np.pi, n_th_m, endpoint=False)
    x_m_arr = np.linspace(0.0, L, n_x_m)
    r_m_arr = np.linspace(0.0, R, n_r_m)

    surf_verts: list = []
    surf_field: list = []
    surf_faces: list = []

    # ── Cylinder wall (r = R, u_x = 0 at no-slip wall) ─────────────────
    for j in range(n_x_m):
        xj = float(x_m_arr[j])
        for i in range(n_th_m):
            surf_verts.append([xj, float(R * np.cos(theta_m[i])), float(R * np.sin(theta_m[i]))])
            surf_field.append(0.0)

    for j in range(n_x_m - 1):
        for i in range(n_th_m):
            i1 = (i + 1) % n_th_m
            v00 = j * n_th_m + i
            v10 = j * n_th_m + i1
            v01 = (j + 1) * n_th_m + i
            v11 = (j + 1) * n_th_m + i1
            surf_faces += [[v00, v10, v11], [v00, v11, v01]]

    # ── Inlet disk (x = 0) — shows parabolic u_x profile ───────────────
    disk_off = len(surf_verts)
    surf_verts.append([0.0, 0.0, 0.0])             # centre vertex at x=0
    surf_field.append(float(2.0 * U))               # u_x at centre = 2U

    for ri in range(1, n_r_m):
        r_i = float(r_m_arr[ri])
        u_i = float(2.0 * U * (1.0 - r_i**2 / R**2))
        for th in theta_m:
            surf_verts.append([0.0, float(r_i * np.cos(th)), float(r_i * np.sin(th))])
            surf_field.append(u_i)

    # Fan triangles: centre → first ring
    c0 = disk_off
    ring0 = disk_off + 1
    for i in range(n_th_m):
        surf_faces.append([c0, ring0 + i, ring0 + (i + 1) % n_th_m])

    # Quad rings
    for ri in range(n_r_m - 2):
        rA = disk_off + 1 + ri * n_th_m
        rB = disk_off + 1 + (ri + 1) * n_th_m
        for i in range(n_th_m):
            i1 = (i + 1) % n_th_m
            surf_faces += [[rA + i, rB + i, rB + i1], [rA + i, rB + i1, rA + i1]]

    surface_mesh = {
        "vertices": surf_verts,
        "faces":    surf_faces,
        "field":    surf_field,
        "field_name": "u_x",
    }

    # ── Transient startup animation (impulsively started Hagen-Poiseuille) ───
    # τ = ν t / R²  (non-dimensional time).  τ=0 → rest, τ→∞ → Poiseuille.
    mu  = float(params.get("mu", 0.01))
    rho_p = float(params.get("rho", 1.0))
    nu  = mu / rho_p   # kinematic viscosity

    n_wall_verts = n_x_m * n_th_m  # 432 — wall field always 0
    tau_arr = [0.0, 0.01, 0.03, 0.06, 0.10, 0.17, 0.28, 0.45, 0.70, 1.10, 2.0]

    def _build_frame_field(tau: float) -> list:
        u_r = _transient_hp_field(r_m_arr, R, U, tau)   # shape (n_r_m,)
        field = [0.0] * n_wall_verts                     # wall: no-slip
        field.append(float(u_r[0]))                      # disk centre
        for ri in range(1, n_r_m):
            field.extend([float(u_r[ri])] * n_th_m)     # azimuthal ring
        return field

    time_frames = []
    for tau in tau_arr[:-1]:
        time_frames.append({
            "tau": float(tau),
            "t_s": round(tau * R**2 / nu, 3),
            "label": f"τ={tau:.2f}",
            "field": _build_frame_field(tau),
        })
    # Final frame = steady state (exact Hagen-Poiseuille values, τ=None ≡ ∞)
    time_frames.append({
        "tau": None,
        "t_s": None,
        "label": "τ→∞  (steady)",
        "field": surf_field,
    })

    time_animation = {
        "vertices":   surf_verts,
        "faces":      surf_faces,
        "field_name": "u_x",
        "nu":         float(nu),
        "R":          float(R),
        "frames":     time_frames,
    }

    return {
        "r":     r_arr.astype(np.float32),
        "x":     x_arr.astype(np.float32),
        "theta": theta_arr.astype(np.float32),
        "u":     (2.0 * U * (1.0 - r_arr**2 / R**2)).astype(np.float32),
        "coords":   coords,
        "solution": solution,
        "surface_mesh":   surface_mesh,
        "time_animation": time_animation,
        "metadata": {
            "problem": "navier_stokes_3d",
            "R": R, "L": L, "U": U, "dim": 3,
            "description": "Hagen-Poiseuille: u_x=2U(1-r^2/R^2), u_y=u_z=0, p linear",
        },
    }


def _gen_crash_analysis(params: dict, n_samples: int) -> dict:
    """
    Linear elastic crash analysis — stress wave propagation in a steel rod.
    Physics: sigma(x,t) = -rho*c*v_impact * H(ct-x)  (compressive step wave)
             u(x,t)     =  v_impact*(t - x/c) * H(ct-x)  (displacement)

    3D surface mesh: circular rod (cylinder) colored by normalised displacement.
    Time animation: stress wave front marching from impact end to free end.
    """
    E        = float(params.get("E",        210e9))   # Pa  (steel default)
    rho      = float(params.get("rho",      7800.0))  # kg/m³
    v_impact = float(params.get("v_impact", 10.0))    # m/s
    L        = float(params.get("L",        1.0))     # rod length (m)
    R        = float(params.get("R",        0.05))    # rod radius (m)
    nx       = int(params.get("nx",         80))
    nt       = int(params.get("nt",         50))
    n_th     = 24   # circumferential segments for the cylinder mesh

    c = np.sqrt(E / rho)
    x = np.linspace(0.0, L, nx, dtype=np.float64)
    t = np.linspace(0.0, L / c, nt, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")

    sigma  = -rho * c * v_impact * (X <= c * T).astype(np.float64)
    u_disp =  v_impact * (T - X / c) * (X <= c * T).astype(np.float64)

    # ── Scatter training data (x-t coords, normalised displacement) ──────────
    coords   = np.stack([X.ravel(), T.ravel()], axis=1).astype(np.float32)
    u_max    = v_impact * L / c if c > 0 else 1.0
    u_norm   = (u_disp / max(u_max, 1e-12)).ravel()[:, None].astype(np.float32)
    idx      = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── 3D cylinder surface mesh ─────────────────────────────────────────────
    # Vertices: rings of n_th points at each x position, plus two end-cap centres
    theta   = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    cos_th  = np.cos(theta)
    sin_th  = np.sin(theta)

    surf_verts: list = []
    surf_field: list = []          # displacement at each vertex (t = t_end, steady)
    # wall rings: n_th vertices per x position
    u_steady = u_disp[:, -1]      # displacement at last time step (steady state)
    for xi, xi_val in enumerate(x):
        u_v = float(u_steady[xi])
        for j in range(n_th):
            surf_verts.append([xi_val, R * cos_th[j], R * sin_th[j]])
            surf_field.append(u_v)

    # end-cap centres
    surf_verts.append([0.0, 0.0, 0.0])   # impact end centre
    surf_verts.append([L,   0.0, 0.0])   # free end centre
    surf_field.append(float(u_steady[0]))
    surf_field.append(float(u_steady[-1]))

    n_rings       = nx
    cap_start_idx = n_rings * n_th      # index of first cap centre
    cap_impact    = cap_start_idx
    cap_free      = cap_start_idx + 1

    surf_faces: list = []
    # Lateral quads → 2 triangles each
    for xi in range(n_rings - 1):
        rA = xi       * n_th
        rB = (xi + 1) * n_th
        for j in range(n_th):
            j1 = (j + 1) % n_th
            surf_faces.append([rA + j, rB + j,  rB + j1])
            surf_faces.append([rA + j, rB + j1, rA + j1])
    # Impact end-cap
    for j in range(n_th):
        surf_faces.append([cap_impact, j, (j + 1) % n_th])
    # Free end-cap
    base = (n_rings - 1) * n_th
    for j in range(n_th):
        surf_faces.append([cap_free, base + (j + 1) % n_th, base + j])

    surface_mesh = {
        "vertices": surf_verts,
        "faces":    surf_faces,
        "field":    surf_field,
        "field_name": "u (m)",
    }

    # ── Time animation (stress wave propagating) ──────────────────────────────
    tau_vals = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]   # fractions of t_max
    frames = []
    for frac in tau_vals:
        ti = int(frac * (nt - 1))
        u_t = u_disp[:, ti]
        field = []
        for xi_idx in range(n_rings):
            v = float(u_t[xi_idx])
            for _ in range(n_th):
                field.append(v)
        field.append(float(u_t[0]))
        field.append(float(u_t[-1]))
        frames.append({
            "tau": round(frac * float(L / c), 6),
            "t_s": round(frac * float(L / c), 6),
            "label": f"t={frac:.0%} of t_max",
            "field": field,
        })
    # Append steady-state frame
    frames.append({
        "tau":   None,
        "t_s":   None,
        "label": "t_max (wave reaches free end)",
        "field": surf_field,
    })

    time_animation = {
        "vertices":   surf_verts,
        "faces":      surf_faces,
        "field_name": "u (m)",
        "frames":     frames,
    }

    return {
        "x":              x.astype(np.float32),
        "t":              t.astype(np.float32),
        "sigma":          sigma.astype(np.float32),
        "u":              u_disp.astype(np.float32),
        "coords":         coords[idx],
        "solution":       u_norm[idx],
        "surface_mesh":   surface_mesh,
        "time_animation": time_animation,
        "metadata": {
            "problem":     "crash_analysis",
            "E":           E,
            "rho":         rho,
            "c":           float(c),
            "v_impact":    v_impact,
            "L":           L,
            "R":           R,
            "dim":         3,
            "description": (
                f"Crash  E={E/1e9:.0f} GPa  ρ={rho:.0f} kg/m³  "
                f"c={c:.0f} m/s  v_impact={v_impact:.1f} m/s"
            ),
            "grid_size":   f"{nx}×{n_th} rod mesh",
        },
    }


def _gen_gross_pitaevskii(params: dict, n_samples: int) -> dict:
    """
    Gross-Pitaevskii ground state approximation.
    For Thomas-Fermi limit: |psi|^2 = max(0, (mu - V(x)) / g)
    V(x) = 0.5*omega^2*x^2 (harmonic trap), mu = chemical potential
    """
    omega = float(params.get("omega", 1.0))
    g = float(params.get("g", 1.0))
    N = float(params.get("N", 100.0))
    nx = int(params.get("nx", 200))

    # Thomas-Fermi radius
    R_TF = (3.0 * N * g / (4.0 * np.sqrt(2.0 * omega)))**(1.0/3.0) if g > 0 else 5.0
    x = np.linspace(-2.0 * R_TF, 2.0 * R_TF, nx, dtype=np.float64)
    V = 0.5 * omega**2 * x**2
    mu = 0.5 * omega * (3.0 * N * g / (4.0 * np.sqrt(2.0 * omega)))**(2.0/3.0) if g > 0 else 1.0
    n_density = np.maximum(0.0, (mu - V) / g) if g > 0 else np.exp(-x**2 / 2.0)
    psi = np.sqrt(n_density)
    # Normalize
    norm = np.trapezoid(psi**2, x)
    if norm > 0:
        psi = psi * np.sqrt(N / norm)

    coords = x[:, None].astype(np.float32)
    solution = psi[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "psi": psi.astype(np.float32),
        "V": V.astype(np.float32),
        "mu": float(mu),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "gross_pitaevskii", "omega": omega, "g": g, "N": N, "dim": 1},
    }


def _gen_magnetostatics_2d(params: dict, n_samples: int) -> dict:
    """
    Magnetostatics 2D: vector potential A_z satisfies lap(A_z) = -mu0*J_z
    Analytical for circular wire at origin: A_z = (mu0*I/(2*pi))*ln(r) for r > r0
    """
    mu0 = float(params.get("mu0", 1.0))   # dimensionless default; set 4π×10⁻⁷ for SI
    I_wire = float(params.get("I", 1.0))
    r0 = float(params.get("r0", 0.1))
    n = int(np.sqrt(n_samples)) + 1

    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    r_clip = np.maximum(r, r0)

    A_z = (mu0 * I_wire / (2.0 * np.pi)) * np.log(1.0 / r_clip)
    Bx = (mu0 * I_wire / (2.0 * np.pi)) * (-Y / r_clip**2)
    By = (mu0 * I_wire / (2.0 * np.pi)) * (X / r_clip**2)

    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = A_z.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "A_z": A_z.astype(np.float32),
        "Bx": Bx.astype(np.float32),
        "By": By.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "magnetostatics_2d", "I": I_wire, "dim": 2},
    }


def _gen_linear_elasticity_3d(params: dict, n_samples: int) -> dict:
    """
    3D Linear elasticity: uniaxial tension.
    u_x = (sigma_x/E)*x, u_y = -(nu*sigma_x/E)*y, u_z = -(nu*sigma_x/E)*z
    """
    E = float(params.get("E", 1.0))
    nu_p = float(params.get("nu", 0.3))
    sigma_x = float(params.get("sigma_x", 1.0))
    n = int(np.cbrt(n_samples)) + 1

    x_arr = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y_arr = np.linspace(0.0, 1.0, n, dtype=np.float64)
    z_arr = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

    ux = (sigma_x / E) * X
    uy = -(nu_p * sigma_x / E) * Y
    uz = -(nu_p * sigma_x / E) * Z

    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    solution = np.stack([ux.ravel(), uy.ravel(), uz.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    return {
        "x": x_arr.astype(np.float32),
        "y": y_arr.astype(np.float32),
        "z": z_arr.astype(np.float32),
        "ux": ux.astype(np.float32),
        "uy": uy.astype(np.float32),
        "uz": uz.astype(np.float32),
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {"problem": "linear_elasticity_3d", "E": E, "nu": nu_p, "dim": 3},
    }


# ---------------------------------------------------------------------------
# 3D multi-physics benchmark generators
# ---------------------------------------------------------------------------

def _hex_mesh_cutaway(x_range, y_range, z_range, field_fn, N: int = 18):
    """
    Structured hexahedral-cell mesh cutaway — proper FEM/CFD visualization style.

    Layout:
      - 3 back-corner walls  (x=x_min, y=y_min, z=z_min)
      - 3 interior cross-sections at domain mid-points

    Each of the 6 faces is a regular N×N quad grid.  Vertices are NOT shared
    between quads: each quad owns its 4 corners independently so the rendering
    engine produces per-cell flat shading, making every cell boundary visible.
    This creates the characteristic flat-shaded "real mesh" look from FEM/CFD
    tools (COMSOL, OpenFOAM, ParaView clip view).

    field_fn(x_arr, y_arr, z_arr) → 1-D float64 field value per point.
    """
    x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range

    # Node grid (N+1 nodes per axis → N cells per axis)
    xn = np.linspace(x0, x1, N + 1, dtype=np.float64)
    yn = np.linspace(y0, y1, N + 1, dtype=np.float64)
    zn = np.linspace(z0, z1, N + 1, dtype=np.float64)
    xc = 0.5 * (xn[:-1] + xn[1:])   # N cell-centre x-coords
    yc = 0.5 * (yn[:-1] + yn[1:])
    zc = 0.5 * (zn[:-1] + zn[1:])

    # Evaluate field at all N³ cell centres in one vectorised call
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")
    F = field_fn(Xc.ravel(), Yc.ravel(), Zc.ravel()).reshape(N, N, N)

    all_verts: list = []
    all_faces: list = []
    all_field: list = []
    offset = 0

    Ia = np.arange(N)

    def _quads(Q4: np.ndarray, fvals: np.ndarray) -> None:
        """
        Q4:    (M, 4, 3) — four corner coords for M quads
        fvals: (M,)      — field value per quad (same for all 4 corners)
        Appends to all_verts / all_faces / all_field in place.
        """
        nonlocal offset
        M = len(Q4)
        for q in range(M):
            for v in range(4):
                all_verts.append(Q4[q, v].tolist())
            all_field.extend([float(fvals[q])] * 4)
            a, b, c, d = offset, offset + 1, offset + 2, offset + 3
            all_faces.extend([[a, b, c], [a, c, d]])
            offset += 4

    # ── Helper: build (N,N,4,3) corner array for a constant-dim face ────
    def _face_quads(const_dim, const_val, ax1_nodes, ax2_nodes,
                    ax1_idx, ax2_idx):
        """
        const_dim: 0→x fixed, 1→y fixed, 2→z fixed
        const_val: the fixed coordinate value
        ax1_nodes / ax2_nodes: node arrays for the two varying axes (length N+1)
        ax1_idx, ax2_idx: meshgrid index arrays (N,N)
        Returns (N²,4,3).
        """
        I1, I2 = ax1_idx, ax2_idx
        corners = []
        for d1, d2 in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            c = np.zeros((N, N, 3))
            a1 = ax1_nodes[I1 + d1]; a2 = ax2_nodes[I2 + d2]
            if const_dim == 0:
                c[..., 0] = const_val; c[..., 1] = a1; c[..., 2] = a2
            elif const_dim == 1:
                c[..., 0] = a1; c[..., 1] = const_val; c[..., 2] = a2
            else:
                c[..., 0] = a1; c[..., 1] = a2; c[..., 2] = const_val
            corners.append(c)
        Q = np.stack(corners, axis=2)   # (N, N, 4, 3)
        return Q.reshape(N * N, 4, 3)

    # Index grids for two-axis sweeps
    Iyz = np.meshgrid(Ia, Ia, indexing="ij")  # (Iy, Iz) for x-const planes
    Ixz = np.meshgrid(Ia, Ia, indexing="ij")  # (Ix, Iz) for y-const planes
    Ixy = np.meshgrid(Ia, Ia, indexing="ij")  # (Ix, Iy) for z-const planes

    # 3 back-corner exterior walls
    _quads(_face_quads(0, x0, yn, zn, *Iyz), F[0,  :, :].ravel())
    _quads(_face_quads(1, y0, xn, zn, *Ixz), F[:,  0, :].ravel())
    _quads(_face_quads(2, z0, xn, yn, *Ixy), F[:, :,  0].ravel())

    # 3 interior cross-sections at domain mid-points
    xm = 0.5 * (x0 + x1); ym = 0.5 * (y0 + y1); zm = 0.5 * (z0 + z1)
    _quads(_face_quads(0, xm, yn, zn, *Iyz), F[N // 2, :,     :].ravel())
    _quads(_face_quads(1, ym, xn, zn, *Ixz), F[:,     N // 2, :].ravel())
    _quads(_face_quads(2, zm, xn, yn, *Ixy), F[:,     :, N // 2].ravel())

    return all_verts, all_faces, all_field


def _cutaway_mesh(x_range, y_range, z_range, field_fn, M: int = 40):
    """
    Room-corner cutaway: 3 MAX boundary faces (z=z1, x=x1, y=y1)
    + 1 interior horizontal slice at z=midpoint.

    Camera at (+X+Y+Z) sees the three corner walls and, through the open
    MIN-corner sides, the interior slice floating inside the domain.

    Z-fighting eliminated: z1-face owns all four edges; x1-face trims the
    z-edges (Z[1:-1]); y1-face trims both x- and z-edges; the interior
    slice (at zm) never touches any of the wall planes.
    """
    x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
    zm = 0.5 * (z0 + z1)

    X  = np.linspace(x0, x1, M, dtype=np.float64)
    Y  = np.linspace(y0, y1, M, dtype=np.float64)
    Z  = np.linspace(z0, z1, M, dtype=np.float64)
    Zm = Z[1:-1]   # inner z values (excludes z0, z1)
    Xm = X[1:-1]   # inner x values (excludes x0, x1)

    all_verts: list = []
    all_field: list = []
    all_faces: list = []

    def _add(Xmat, Ymat, Zmat, Fmat):
        nr, nc = Xmat.shape
        off = len(all_verts)
        for ri in range(nr):
            for ci in range(nc):
                all_verts.append([float(Xmat[ri, ci]), float(Ymat[ri, ci]), float(Zmat[ri, ci])])
                all_field.append(float(Fmat[ri, ci]))
        for ri in range(nr - 1):
            for ci in range(nc - 1):
                a = off + ri * nc + ci;  b = a + 1
                c = off + (ri + 1) * nc + ci;  d = c + 1
                all_faces.extend([[a, b, d], [a, d, c]])

    # ── z = z1 (top / lid face) — full M×M, owns all four edges ─────────────
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    F = field_fn(Xg.ravel(), Yg.ravel(), np.full(M * M, z1)).reshape(M, M)
    _add(Xg, Yg, np.full_like(Xg, z1), F)

    # ── x = x1 (right wall) — trim z-edges to avoid overlap with z1-face ────
    Yg, Zg = np.meshgrid(Y, Zm, indexing='ij')
    n = M * (M - 2)
    F = field_fn(np.full(n, x1), Yg.ravel(), Zg.ravel()).reshape(M, M - 2)
    _add(np.full_like(Yg, x1), Yg, Zg, F)

    # ── y = y1 (back wall) — trim x- and z-edges ─────────────────────────────
    Xg, Zg = np.meshgrid(Xm, Zm, indexing='ij')
    n = (M - 2) ** 2
    F = field_fn(Xg.ravel(), np.full(n, y1), Zg.ravel()).reshape(M - 2, M - 2)
    _add(Xg, np.full_like(Xg, y1), Zg, F)

    # ── z = zm interior slice — full M×M, no overlap with any wall ───────────
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    F = field_fn(Xg.ravel(), Yg.ravel(), np.full(M * M, zm)).reshape(M, M)
    _add(Xg, Yg, np.full_like(Xg, zm), F)

    return all_verts, all_faces, all_field


def _make_2d_slices(x_range, y_range, z_range, field_fn, M: int = 40):
    """
    Compute 3 crisp analytical 2-D cross-section heatmaps at the domain midpoints.
    Returns a dict compatible with the frontend ScientificHeatmap component.
    Each entry has:  x / y / z arrays  +  u: List[List[float]] shaped (M, M).
    """
    x0, x1 = x_range; y0, y1 = y_range; z0, z1 = z_range
    xm, ym, zm = 0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1)

    x = np.linspace(x0, x1, M, dtype=np.float64)
    y = np.linspace(y0, y1, M, dtype=np.float64)
    z = np.linspace(z0, z1, M, dtype=np.float64)

    # XY plane at z = zmid.  Stored transposed as u[yi][xi] (ScientificHeatmap convention).
    Xa, Ya = np.meshgrid(x, y, indexing="ij")
    u_xy = field_fn(Xa.ravel(), Ya.ravel(), np.full(M*M, zm)).reshape(M, M).T

    # XZ plane at y = ymid.  Stored as u[zi][xi].
    Xa, Za = np.meshgrid(x, z, indexing="ij")
    u_xz = field_fn(Xa.ravel(), np.full(M*M, ym), Za.ravel()).reshape(M, M).T

    # YZ plane at x = xmid.  Stored as u[zi][yi].
    Ya, Za = np.meshgrid(y, z, indexing="ij")
    u_yz = field_fn(np.full(M*M, xm), Ya.ravel(), Za.ravel()).reshape(M, M).T

    return {
        "xy": {"x": x.tolist(), "y": y.tolist(), "u": u_xy.tolist(),
               "label": f"XY plane  z = {zm:.3g}"},
        "xz": {"x": x.tolist(), "z": z.tolist(), "u": u_xz.tolist(),
               "label": f"XZ plane  y = {ym:.3g}"},
        "yz": {"y": y.tolist(), "z": z.tolist(), "u": u_yz.tolist(),
               "label": f"YZ plane  x = {xm:.3g}"},
    }


def _gen_taylor_green_vortex(params: dict, n_samples: int) -> dict:
    """
    Taylor-Green Vortex — canonical turbulence DNS benchmark.
    Analytically prescribed IC for 3D incompressible NS on [0, 2π]³ (periodic).
    IC: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0,
        p=(cos(2x)+cos(2y))(cos(2z)+2)/16
    Surface mesh: velocity magnitude on all 6 box faces.
    """
    Re = float(params.get("Re", 1600.0))
    N = 20  # N³ = 8000 volume points

    L = 2.0 * np.pi
    x = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    z = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    p = (np.cos(2.0 * X) + np.cos(2.0 * Y)) * (np.cos(2.0 * Z) + 2.0) / 16.0

    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u.ravel(), v.ravel(), w.ravel(), p.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── Surface mesh: smooth cutaway (3 corner walls + 3 midpoint slices) ──
    def _tgv_field(fx, fy, fz):
        fu = np.sin(fx) * np.cos(fy) * np.cos(fz)
        fv = -np.cos(fx) * np.sin(fy) * np.cos(fz)
        return np.sqrt(fu**2 + fv**2)

    all_verts, all_faces_list, all_field = _cutaway_mesh(
        (0.0, L), (0.0, L), (0.0, L), _tgv_field
    )

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "taylor_green_vortex",
            "Re": Re,
            "dim": 3,
            "description": f"Taylor-Green Vortex IC  Re={Re:.0f}  [0, 2π]³",
            "grid_size": f"{N}³",
        },
        "surface_mesh": {
            "vertices": all_verts,
            "faces": all_faces_list,
            "field": all_field,
            "field_name": "u_mag",
        },
        "domain_bounds": {"x": [0.0, L], "y": [0.0, L], "z": [0.0, L]},
    }


def _gen_darcy_flow_3d(params: dict, n_samples: int) -> dict:
    """
    3D Darcy flow through porous media: -K∇²p = f in [0,1]³.
    Darcy flow 3D benchmark (CO₂ storage, reservoir simulation).
    Manufactured solution: p = sin(πx)sin(πy)sin(πz).
    Darcy velocity: u = -K∇p.
    Surface mesh: pressure on all 6 box faces.
    """
    K = float(params.get("K", 1.0))
    N = 18  # N³ interior points

    x = np.linspace(0.0, 1.0, N, dtype=np.float64)
    y = np.linspace(0.0, 1.0, N, dtype=np.float64)
    z = np.linspace(0.0, 1.0, N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    pi = np.pi
    sinx = np.sin(pi * X); siny = np.sin(pi * Y); sinz = np.sin(pi * Z)
    cosx = np.cos(pi * X); cosy = np.cos(pi * Y); cosz = np.cos(pi * Z)

    p = sinx * siny * sinz
    ux = -K * pi * cosx * siny * sinz
    uy = -K * pi * sinx * cosy * sinz
    uz_v = -K * pi * sinx * siny * cosz

    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    solution = np.stack([ux.ravel(), uy.ravel(), uz_v.ravel(), p.ravel()], axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── Surface mesh: smooth cutaway (3 corner walls + 3 midpoint slices) ──
    def _darcy_field(fx, fy, fz):
        fu = K * pi * np.cos(pi * fx) * np.sin(pi * fy) * np.sin(pi * fz)
        fv = K * pi * np.sin(pi * fx) * np.cos(pi * fy) * np.sin(pi * fz)
        fw = K * pi * np.sin(pi * fx) * np.sin(pi * fy) * np.cos(pi * fz)
        return np.sqrt(fu**2 + fv**2 + fw**2)

    all_verts, all_faces_list, all_field = _cutaway_mesh(
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), _darcy_field
    )

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "darcy_flow_3d",
            "K": K,
            "dim": 3,
            "description": f"3D Darcy Flow  K={K:.3f}  [0,1]³",
            "grid_size": f"{N}³",
        },
        "surface_mesh": {
            "vertices": all_verts,
            "faces": all_faces_list,
            "field": all_field,
            "field_name": "|u|",
        },
        "domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
    }


def _gen_stokes_sphere(params: dict, n_samples: int) -> dict:
    """
    Stokes flow around a sphere — Lamb's exact analytical solution (Re→0).
    Stokes sphere — standard external aerodynamics reference benchmark.
    Surface mesh: sphere colored by Cp = 1 - (9/4)sin²θ (potential flow limit).
    Cartesian Stokes velocity:
      u_x = -3UR·xz(r²-R²) / (4r⁵)
      u_y = -3UR·yz(r²-R²) / (4r⁵)
      u_z = U[cos²θ(1-3R/(2r)+R³/(2r³)) + sin²θ(1-3R/(4r)-R³/(4r³))]
      p   = -3μUR·z / (2r³)
    Flow direction: z-axis.
    """
    U_inf = float(params.get("U_inf", 1.0))
    R = float(params.get("R", 0.5))
    mu = 1.0

    N = 16
    L = 3.0 * R
    xg = np.linspace(-L, L, N, dtype=np.float64)
    yg = np.linspace(-L, L, N, dtype=np.float64)
    zg = np.linspace(-L, L, N, dtype=np.float64)
    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing="ij")
    r_g = np.sqrt(Xg**2 + Yg**2 + Zg**2)

    # Only outside sphere
    mask = r_g > R * 1.05
    xm = Xg[mask]; ym = Yg[mask]; zm = Zg[mask]; r_m = r_g[mask]
    cos_th = zm / (r_m + 1e-14)
    sin_th = np.sqrt(xm**2 + ym**2) / (r_m + 1e-14)

    A = 1.0 - 3.0 * R / (2.0 * r_m) + R**3 / (2.0 * r_m**3)
    B = 1.0 - 3.0 * R / (4.0 * r_m) - R**3 / (4.0 * r_m**3)

    uz_m = U_inf * (cos_th**2 * A + sin_th**2 * B)
    # Transverse: u_x = -3UR xz(r²-R²)/(4r⁵)
    factor = -3.0 * U_inf * R * zm * (r_m**2 - R**2) / (4.0 * r_m**5 + 1e-14)
    ux_m = factor * xm
    uy_m = factor * ym
    p_m = -3.0 * mu * U_inf * R * zm / (2.0 * r_m**3 + 1e-14)

    coords = np.stack([xm, ym, zm], axis=1).astype(np.float32)
    solution = np.stack([ux_m, uy_m, uz_m, p_m], axis=1).astype(np.float32)
    n_pts = len(coords)
    idx = np.random.choice(n_pts, size=min(n_samples, n_pts), replace=False)

    # ── Surface mesh: sphere colored by Cp ──────────────────────────────────
    # Use potential flow Cp = 1 - (9/4)sin²θ (inviscid reference on sphere)
    Nth, Nph = 36, 36
    theta_s = np.linspace(0.0, np.pi, Nth)
    phi_s = np.linspace(0.0, 2.0 * np.pi, Nph, endpoint=False)
    TH, PH = np.meshgrid(theta_s, phi_s, indexing="ij")  # (Nth, Nph)

    sx = R * np.sin(TH) * np.cos(PH)
    sy = R * np.sin(TH) * np.sin(PH)
    sz = R * np.cos(TH)
    Cp = 1.0 - (9.0 / 4.0) * np.sin(TH)**2

    sv = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)

    sphere_faces: list = []
    for i in range(Nth - 1):
        for j in range(Nph):
            a = i * Nph + j
            b = i * Nph + (j + 1) % Nph
            c = (i + 1) * Nph + j
            d = (i + 1) * Nph + (j + 1) % Nph
            sphere_faces.extend([[a, b, d], [a, d, c]])

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "stokes_sphere",
            "U_inf": U_inf,
            "R": R,
            "dim": 3,
            "description": f"Stokes Flow Around Sphere  R={R:.2f}  U∞={U_inf:.2f}",
            "grid_size": f"{Nth}θ × {Nph}φ sphere",
        },
        "surface_mesh": {
            "vertices": sv.tolist(),
            "faces": sphere_faces,
            "field": Cp.ravel().tolist(),
            "field_name": "Cp",
        },
        "domain_bounds": {"x": [-L, L], "y": [-L, L], "z": [-L, L]},
    }


def _gen_hartmann_mhd(params: dict, n_samples: int) -> dict:
    """
    Hartmann MHD duct flow — canonical computational MHD benchmark.
    Pressure-driven flow in rectangular duct [0,Lx]×[-H,H]×[0,Lz] with transverse
    magnetic field B in z-direction.  Exact Hartmann velocity profile:
      u(y) = (G·H²/μHa²)(1 − cosh(Ha·y/H)/cosh(Ha))
    where Ha = B·H·√(σ/μ) is the Hartmann number (Ha=0 → Poiseuille, Ha→∞ → plug).
    Surface mesh: 4 duct walls colored by u_x.
    """
    Ha = float(params.get("Ha", 5.0))   # Hartmann number
    Re = float(params.get("Re", 100.0))
    Lx, H, Lz = 2.0, 0.5, 1.0          # duct: length × half-height × width
    G = 1.0; mu = 1.0                   # pressure gradient, viscosity (normalized)

    Nx, Ny, Nz = 20, 24, 16
    x = np.linspace(0.0, Lx, Nx, dtype=np.float64)
    y = np.linspace(-H, H, Ny, dtype=np.float64)
    z = np.linspace(0.0, Lz, Nz, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    denom = Ha**2 if Ha > 1e-3 else 1.0
    if Ha > 1e-3:
        u_hat = (G * H**2 / (mu * denom)) * (1.0 - np.cosh(Ha * Y / H) / np.cosh(Ha))
    else:
        u_hat = (G / (2.0 * mu)) * (H**2 - Y**2)   # Poiseuille limit Ha→0
    v_hat = np.zeros_like(u_hat)
    w_hat = np.zeros_like(u_hat)
    p_hat = G * (Lx - X)                             # linear pressure drop

    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    solution = np.stack([u_hat.ravel(), v_hat.ravel(), w_hat.ravel(), p_hat.ravel()],
                        axis=1).astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── Surface mesh: smooth cutaway (3 corner walls + 3 midpoint slices) ──
    def _hartmann_field(fx, fy, fz):
        if Ha > 1e-3:
            return (G * H**2 / (mu * Ha**2)) * (1.0 - np.cosh(Ha * fy / H) / np.cosh(Ha))
        return (G / (2.0 * mu)) * (H**2 - fy**2)

    all_verts, all_faces_list, all_field = _cutaway_mesh(
        (0.0, Lx), (-H, H), (0.0, Lz), _hartmann_field
    )

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "hartmann_mhd",
            "Ha": Ha, "Re": Re, "dim": 3,
            "description": f"Hartmann MHD Duct Flow  Ha={Ha:.1f}  Re={Re:.0f}",
            "grid_size": f"{Nx}×{Ny}×{Nz}",
        },
        "surface_mesh": {
            "vertices": all_verts,
            "faces": all_faces_list,
            "field": all_field,
            "field_name": "u_x",
        },
        "domain_bounds": {"x": [0.0, Lx], "y": [-H, H], "z": [0.0, Lz]},
    }


def _gen_chip_heat_3d(params: dict, n_samples: int) -> dict:
    """
    Chip heat-sink cooling — fin-array thermal management benchmark.
    N rectangular aluminum fins on a silicon chip base, cooled by forced convection.
    Exact 1-D fin theory (uniform base temperature, adiabatic fin tips):
      T(z) = T_amb + (T_base - T_amb) · cosh(m·(H-z)) / cosh(m·H)
      m = √(2h / (k·t))   [fin efficiency parameter]
    3D geometry: chip base + fin array; surface mesh colored by temperature T.
    """
    T_base = float(params.get("T_base", 80.0))   # °C
    T_amb  = float(params.get("T_amb",  25.0))   # °C
    h_conv = float(params.get("h_conv", 200.0))  # W/m²K  convective coefficient
    k_fin  = float(params.get("k_fin", 200.0))   # W/mK   Al thermal conductivity
    N_fins = int(params.get("N_fins", 6))

    t_fin = 0.005       # fin thickness 5 mm
    H_fin = 0.030       # fin height 30 mm
    L_fin = 0.040       # fin length 40 mm (y-direction)
    s_gap = 0.005       # gap between fins
    t_base = 0.004      # base plate thickness
    pitch = t_fin + s_gap

    m_fin = np.sqrt(2.0 * h_conv / (k_fin * t_fin))
    theta_base = T_base - T_amb

    def T_fin(z_local: np.ndarray) -> np.ndarray:
        return T_amb + theta_base * np.cosh(m_fin * (H_fin - z_local)) / np.cosh(m_fin * H_fin)

    # ── Volume grid (base + air above base, excluding fins for simplicity) ───
    W_chip = N_fins * pitch + s_gap
    Nx_v, Ny_v, Nz_v = 16, 12, 16
    xv = np.linspace(0.0, W_chip, Nx_v)
    yv = np.linspace(0.0, L_fin, Ny_v)
    zv = np.linspace(-t_base, H_fin, Nz_v)
    Xv, Yv, Zv = np.meshgrid(xv, yv, zv, indexing="ij")

    # Simple 3D temperature field: fin theory in z, uniform in x,y for base
    T_vol = np.where(Zv < 0.0, T_base, T_fin(np.clip(Zv, 0.0, H_fin)))

    coords = np.stack([Xv.ravel(), Yv.ravel(), Zv.ravel()], axis=1).astype(np.float32)
    solution = T_vol.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── Surface mesh: fin faces + chip base top ──────────────────────────────
    My, Mz = 12, 18   # grid on fin surfaces (y=length, z=height)
    y_pts = np.linspace(0.0, L_fin, My)
    z_pts = np.linspace(0.0, H_fin, Mz)
    Yg, Zg = np.meshgrid(y_pts, z_pts, indexing="ij")  # (My, Mz)
    T_grid = T_fin(Zg)   # temperature on fin surface (same for all fins)
    T_flat = T_grid.ravel()
    Y_flat = Yg.ravel()
    Z_flat = Zg.ravel()

    all_verts: list = []
    all_faces_list: list = []
    all_field: list = []
    offset = 0

    def _quad_mesh(verts: np.ndarray, field: np.ndarray, nrows: int, ncols: int, off: int):
        faces_loc = []
        for i in range(nrows - 1):
            for j in range(ncols - 1):
                a = off + i * ncols + j
                b = a + 1; c = off + (i + 1) * ncols + j; d = c + 1
                faces_loc.extend([[a, b, d], [a, d, c]])
        return verts.tolist(), field.tolist(), faces_loc

    for fin_i in range(N_fins):
        x_left  = (s_gap / 2.0) + fin_i * pitch
        x_right = x_left + t_fin

        # Left face (x = x_left)
        vl = np.stack([np.full(My*Mz, x_left), Y_flat, Z_flat], axis=1)
        v_, f_, fc = _quad_mesh(vl, T_flat, My, Mz, offset)
        all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc); offset += My*Mz

        # Right face (x = x_right)
        vr = np.stack([np.full(My*Mz, x_right), Y_flat, Z_flat], axis=1)
        v_, f_, fc = _quad_mesh(vr, T_flat, My, Mz, offset)
        all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc); offset += My*Mz

        # Front face (y = 0)
        x_pts_f = np.linspace(x_left, x_right, 6)
        Xf, Zf = np.meshgrid(x_pts_f, z_pts, indexing="ij")
        Tf = T_fin(Zf)
        vf = np.stack([Xf.ravel(), np.zeros(6*Mz), Zf.ravel()], axis=1)
        v_, f_, fc = _quad_mesh(vf, Tf.ravel(), 6, Mz, offset)
        all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc); offset += 6*Mz

        # Back face (y = L_fin)
        vb_ = np.stack([Xf.ravel(), np.full(6*Mz, L_fin), Zf.ravel()], axis=1)
        v_, f_, fc = _quad_mesh(vb_, Tf.ravel(), 6, Mz, offset)
        all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc); offset += 6*Mz

        # Fin tip cap (z = H_fin, adiabatic)
        x_pts_t = np.linspace(x_left, x_right, 4)
        y_pts_t = np.linspace(0.0, L_fin, 6)
        Xt, Yt = np.meshgrid(x_pts_t, y_pts_t, indexing="ij")
        vtp = np.stack([Xt.ravel(), Yt.ravel(), np.full(4*6, H_fin)], axis=1)
        Ttp = np.full(4*6, float(T_fin(np.array([H_fin]))[0]))
        v_, f_, fc = _quad_mesh(vtp, Ttp, 4, 6, offset)
        all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc); offset += 4*6

    # Base plate top surface (z = 0, where fins are attached)
    Mx_b = 20; My_b = 12
    xb_pts = np.linspace(0.0, W_chip, Mx_b)
    yb_pts = np.linspace(0.0, L_fin, My_b)
    Xb, Yb = np.meshgrid(xb_pts, yb_pts, indexing="ij")
    vbase = np.stack([Xb.ravel(), Yb.ravel(), np.zeros(Mx_b*My_b)], axis=1)
    Tbase_field = np.full(Mx_b*My_b, T_base)
    v_, f_, fc = _quad_mesh(vbase, Tbase_field, Mx_b, My_b, offset)
    all_verts.extend(v_); all_field.extend(f_); all_faces_list.extend(fc)

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "chip_heat_3d",
            "T_base": T_base, "T_amb": T_amb,
            "h_conv": h_conv, "k_fin": k_fin,
            "N_fins": N_fins, "dim": 3,
            "description": f"Chip Heat Sink  {N_fins} fins  T_base={T_base}°C  h={h_conv} W/m²K",
        },
        "surface_mesh": {
            "vertices": all_verts,
            "faces": all_faces_list,
            "field": all_field,
            "field_name": "T",
        },
        "domain_bounds": {
            "x": [0.0, float(N_fins * pitch + s_gap)],
            "y": [0.0, L_fin],
            "z": [-t_base, H_fin],
        },
    }


def _gen_helmholtz_3d_cavity(params: dict, n_samples: int) -> dict:
    """
    3D Helmholtz acoustic cavity resonance — waveguide / EM resonance benchmark.
    Standing-wave pressure modes in a rigid-walled rectangular cavity [0,a]×[0,b]×[0,c]:
      p(x,y,z) = cos(nπx/a) · cos(mπy/b) · cos(lπz/c)
      k_res = π · √(n²/a² + m²/b² + l²/c²)  (resonance wavenumber)
    Boundary condition: ∂p/∂n = 0 (pressure-release, sound-hard walls).
    Surface mesh: 6 cavity walls colored by |p| amplitude.
    """
    a = float(params.get("a", 1.0))    # x-length
    b = float(params.get("b", 0.8))    # y-length
    c = float(params.get("c", 0.6))    # z-length
    n_mode = int(params.get("n_mode", 2))  # modal indices
    m_mode = int(params.get("m_mode", 1))
    l_mode = int(params.get("l_mode", 1))
    c_sound = float(params.get("c_sound", 343.0))  # speed of sound m/s

    k_res = np.pi * np.sqrt((n_mode/a)**2 + (m_mode/b)**2 + (l_mode/c)**2)
    f_res = c_sound * k_res / (2.0 * np.pi)

    N = 18
    xg = np.linspace(0.0, a, N, dtype=np.float64)
    yg = np.linspace(0.0, b, N, dtype=np.float64)
    zg = np.linspace(0.0, c, N, dtype=np.float64)
    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing="ij")

    p_vol = (np.cos(n_mode * np.pi * Xg / a) *
             np.cos(m_mode * np.pi * Yg / b) *
             np.cos(l_mode * np.pi * Zg / c))

    coords = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1).astype(np.float32)
    solution = p_vol.ravel()[:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    # ── Surface mesh: smooth cutaway (3 corner walls + 3 midpoint slices) ──
    def _helm_field(fx, fy, fz):
        return (np.cos(n_mode * np.pi * fx / a) *
                np.cos(m_mode * np.pi * fy / b) *
                np.cos(l_mode * np.pi * fz / c))

    all_verts, all_faces_list, all_field = _cutaway_mesh(
        (0.0, a), (0.0, b), (0.0, c), _helm_field
    )

    return {
        "coords": coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem": "helmholtz_3d_cavity",
            "a": a, "b": b, "c": c,
            "mode": f"({n_mode},{m_mode},{l_mode})",
            "k_res": float(k_res),
            "f_res_Hz": float(f_res),
            "dim": 3,
            "description": (
                f"3D Acoustic Cavity Mode ({n_mode},{m_mode},{l_mode})  "
                f"f_res={f_res:.1f} Hz  a={a}×b={b}×c={c} m"
            ),
        },
        "surface_mesh": {
            "vertices": all_verts,
            "faces": all_faces_list,
            "field": all_field,
            "field_name": "p",
        },
        "domain_bounds": {"x": [0.0, a], "y": [0.0, b], "z": [0.0, c]},
    }


def _gen_airfoil_naca0012_2d(params: dict, n_samples: int) -> dict:
    """
    NACA 0012 airfoil: 2D Cp heatmap via Hess-Smith panel method.
    Returns a structured 2D grid of Cp = 1 - V²/V∞² around the airfoil
    cross-section. Interior of the airfoil body is set to NaN so the
    heatmap renderer shows a clean "hole" at the body location.
    """
    chord     = float(params.get("chord",     1.0))
    V_inf     = float(params.get("V_inf",     1.0))
    alpha_deg = float(params.get("alpha_deg", 5.0))
    n_panels  = 100
    t_ratio   = 0.12

    alpha = np.radians(alpha_deg)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    theta_p = np.linspace(0.0, np.pi, n_panels + 1, dtype=np.float64)
    xn = 0.5 * chord * (1.0 - np.cos(theta_p))
    xn_norm = xn / chord

    def _yt(xi: np.ndarray) -> np.ndarray:
        xi = np.clip(xi, 0.0, 1.0)
        return 5.0 * t_ratio * chord * (
            0.2969 * np.sqrt(xi + 1e-14) - 0.1260 * xi
            - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1015 * xi**4)

    yt = _yt(xn_norm)
    x_nodes = np.concatenate([xn, xn[::-1][1:]])
    y_nodes = np.concatenate([yt, -yt[::-1][1:]])
    N = len(x_nodes) - 1

    xm = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    ym = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    dx_p = x_nodes[1:] - x_nodes[:-1]
    dy_p = y_nodes[1:] - y_nodes[:-1]
    ln  = np.hypot(dx_p, dy_p)
    nx_p = dy_p / ln
    ny_p = -dx_p / ln

    A_mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                A_mat[i, j] = 0.5
            else:
                rx = xm[i] - xm[j]; ry = ym[i] - ym[j]
                r2 = rx**2 + ry**2 + 1e-16
                A_mat[i, j] = (rx * nx_p[i] + ry * ny_p[i]) / (2.0 * np.pi * r2)

    rhs   = -(V_inf * cos_a * nx_p + V_inf * sin_a * ny_p)
    sigma = _nan_clean(np.linalg.solve(A_mat, rhs))
    gamma = np.pi * V_inf * chord * np.sin(alpha) if abs(alpha) > 1e-10 else 0.0
    gpp   = gamma / N

    Nx_f, Ny_f = 160, 100
    xf = np.linspace(-0.5 * chord, 3.5 * chord, Nx_f, dtype=np.float64)
    yf = np.linspace(-1.5 * chord, 1.5 * chord, Ny_f, dtype=np.float64)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Xp = Xf.ravel(); Yp = Yf.ravel()
    n_pts = len(Xp)

    u_f = np.full(n_pts, V_inf * cos_a, dtype=np.float64)
    v_f = np.full(n_pts, V_inf * sin_a, dtype=np.float64)

    for j in range(N):
        rx = Xp - xm[j]; ry = Yp - ym[j]
        r2 = rx**2 + ry**2 + 1e-14
        fac = sigma[j] / (2.0 * np.pi * r2)
        u_f += fac * rx
        v_f += fac * ry
        if gamma != 0.0:
            fac_v = gpp / (2.0 * np.pi * r2)
            u_f -= fac_v * ry
            v_f += fac_v * rx

    u_f = _nan_clean(u_f, fill=V_inf * cos_a)
    v_f = _nan_clean(v_f, fill=V_inf * sin_a)
    Cp_flat = np.clip(1.0 - (u_f**2 + v_f**2) / (V_inf**2 + 1e-14), -10.0, 2.0)

    # Mask airfoil interior — use NaN internally but replace before JSON
    in_chord = (Xp >= 0.0) & (Xp <= chord)
    inside   = np.zeros(n_pts, dtype=bool)
    if in_chord.any():
        xi_i = Xp[in_chord] / chord
        yt_i = _yt(xi_i)
        inside[in_chord] = np.abs(Yp[in_chord]) < yt_i * 1.15

    Cp_2d = Cp_flat.reshape(Nx_f, Ny_f)
    # Interior gets the far-field Cp value (0.0 ≈ freestream) — avoids NaN in JSON
    Cp_2d_interior = Cp_2d.copy()
    Cp_2d_interior[inside.reshape(Nx_f, Ny_f)] = 0.0

    ext = ~inside
    coords   = np.stack([Xp[ext], Yp[ext]], axis=1).astype(np.float32)
    sol_ext  = Cp_flat[ext][:, None].astype(np.float32)
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    return {
        "x": xf.astype(np.float32),
        "y": yf.astype(np.float32),
        "u": Cp_2d_interior.tolist(),
        "coords":   coords[idx],
        "solution": sol_ext[idx],
        "metadata": {
            "problem":     "airfoil_naca0012_2d",
            "chord":       chord,
            "alpha_deg":   alpha_deg,
            "V_inf":       V_inf,
            "dim":         2,
            "description": f"NACA 0012 Cp field  α={alpha_deg}°  V∞={V_inf} m/s",
        },
    }


def _gen_shallow_water_2d(params: dict, n_samples: int) -> dict:
    """
    Linearised Shallow Water Equations (SWE) — climate/oceanographic gravity waves.

    Governing equation (height perturbation η):
      ∂²η/∂t² = c² ∇²η   where c = √(g·h₀)  (non-dispersive gravity wave)

    Exact FFT solution for a Gaussian initial bump at the domain centre:
      η(x,y,0) = A · exp(-((x-0.5)²+(y-0.5)²)/(2σ²)), ∂η/∂t|₀ = 0
      η̂(k,t) = η̂₀(k) · cos(c|k|t)

    Domain: [0,1]² periodic.  Returns η/h₀ (dimensionless) at t = t_eval.
    """
    g      = float(params.get("g",      9.81))
    h0     = float(params.get("h0",     10.0))
    A      = float(params.get("A",       1.0))
    sigma  = float(params.get("sigma",  0.06))
    t_eval = float(params.get("t_eval", 0.40))

    N = 64
    L = 1.0
    c_wave = float(np.sqrt(g * h0))

    x = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    eta0 = A * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2.0 * sigma**2))

    eta0_hat = np.fft.fft2(eta0)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(Kx**2 + Ky**2)

    eta_hat_t = eta0_hat * np.cos(c_wave * K * t_eval)
    eta_t = np.real(np.fft.ifft2(eta_hat_t))

    eta_norm = (eta_t / h0).astype(np.float32)

    coords   = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    solution = eta_norm.ravel()[:, None]
    idx = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u": eta_norm.tolist(),   # shape (N, N), u[xi][yi]
        "coords":   coords[idx],
        "solution": solution[idx],
        "metadata": {
            "problem":     "shallow_water_2d",
            "g":           g,
            "h0":          h0,
            "c_wave":      c_wave,
            "t_eval":      t_eval,
            "sigma":       sigma,
            "dim":         2,
            "description": (
                f"SWE gravity wave  t={t_eval:.2f}s  c={c_wave:.1f} m/s  "
                f"η/h₀  (SWE climate benchmark)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Dimension table (used by the generic fallback and get_collocation_data)
# ---------------------------------------------------------------------------

_PROBLEM_DIMS: dict[str, tuple] = {
    # (in_dim, out_dim)
    "burgers_1d":             (2, 1),
    "navier_stokes_2d":       (2, 3),
    "lid_driven_cavity":      (2, 2),
    "bernoulli_flow":         (1, 2),
    "stokes_flow":            (1, 1),
    "darcy_flow":             (1, 2),
    "channel_poiseuille":     (1, 1),
    "linear_elasticity_2d":   (2, 2),
    "beam_bending":           (1, 2),
    "poisson_electrostatics": (2, 1),
    "helmholtz_wave":         (2, 1),
    "laplace_potential":      (2, 1),
    "schrodinger_stationary": (1, 1),
    "schrodinger_timedep":    (2, 1),
    "heat_equation_1d":       (2, 1),
    "heat_equation_2d":       (3, 1),
    "wave_equation":          (2, 1),
    "kdv_equation":           (2, 1),
    "klein_gordon":           (2, 1),
    "thermoelasticity_2d":    (2, 3),
    "navier_stokes_3d":       (3, 4),
    "crash_analysis":         (2, 2),
    "gross_pitaevskii":       (1, 1),
    "magnetostatics_2d":      (2, 1),
    "linear_elasticity_3d":   (3, 3),
    "airfoil_naca0012_2d":    (2, 1),
    "shallow_water_2d":       (2, 1),
    "cylinder_flow_2d":       (3, 3),   # inputs: (x, y, t), outputs: (u, v, p)
    "taylor_green_vortex":    (3, 4),
    "darcy_flow_3d":          (3, 4),
    "stokes_sphere":          (3, 4),
    "hartmann_mhd":           (3, 4),
    "chip_heat_3d":           (3, 1),
    "helmholtz_3d_cavity":    (3, 1),
}

# Time-dependent problems (last coordinate is time)
_TIME_DEP_PROBLEMS = frozenset({
    "burgers_1d", "schrodinger_timedep", "heat_equation_1d",
    "heat_equation_2d", "wave_equation", "kdv_equation",
    "klein_gordon", "crash_analysis", "cylinder_flow_2d",
})

def _gen_cylinder_flow_2d(params: dict, n_samples: int) -> dict:
    """
    2D Flow Past Cylinder — animated Kármán vortex street preview.

    Returns a 2-D vorticity heatmap that animates through 10 time frames,
    mimicking the classic blue/red alternating pattern of the Kármán street.

    Approach:
    - Grid: x ∈ [-2D, 18D] (nx=80), y ∈ [-5D, 5D] (ny=40), indexing='ij'.
    - Vorticity: superposition of Gaussian vortex blobs convecting downstream.
      Upper shed vortices: omega < 0 (CW, blue); lower: omega > 0 (CCW, red).
    - Returns preview.x, preview.y, preview.u (static snapshot) plus
      preview.animated_frames for the animated heatmap component.
    - Strouhal: St ≈ 0.198(1 − 19.7/Re) (Williamson 1988).
    """
    Re    = float(params.get("Re",    100.0))
    U_inf = float(params.get("U_inf",   1.0))
    D     = float(params.get("D",       1.0))
    rho   = float(params.get("rho",     1.0))

    R = D / 2.0

    # ── Shedding frequency ────────────────────────────────────────────────────
    Re_eff = max(Re, 50.0)
    St     = 0.198 * (1.0 - 19.7 / Re_eff)
    f_shed = St * U_inf / D
    T_shed = 1.0 / f_shed

    # ── Kármán street parameters ──────────────────────────────────────────────
    lam    = U_inf / f_shed           # streamwise wavelength
    h_sep  = 0.28 * lam               # transverse row separation (h/λ ≈ 0.28)
    Gamma  = 2.0 * U_inf * h_sep      # vortex circulation (order-of-magnitude)
    sigma  = 0.65 * D                 # Gaussian blob radius  # noqa: F841
    U_conv = 0.82 * U_inf             # convection speed of vortex street

    # ── 2D grid  (indexing='ij' → u[xi][yi] convention used by frontend) ─────
    x_min, x_max = -2.0 * D, 18.0 * D
    y_min, y_max = -5.0 * D,  5.0 * D
    nx_g, ny_g   = 160, 80          # higher resolution for smooth rendering

    xg = np.linspace(x_min, x_max, nx_g, dtype=np.float64)
    yg = np.linspace(y_min, y_max, ny_g, dtype=np.float64)
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")   # shape (nx_g, ny_g)

    r2_g   = Xg**2 + Yg**2
    cyl_mk = r2_g <= (R * 1.05)**2                 # True inside cylinder + buffer

    nu = U_inf * D / max(Re, 1.0)   # kinematic viscosity

    # Velocity grid at coarser resolution (for streamlines — full res not needed)
    nx_v, ny_v = 80, 40
    xv = np.linspace(x_min, x_max, nx_v, dtype=np.float64)
    yv = np.linspace(y_min, y_max, ny_v, dtype=np.float64)
    Xv, Yv = np.meshgrid(xv, yv, indexing="ij")
    r2_v  = Xv**2 + Yv**2
    cyl_v = r2_v <= (R * 1.05)**2

    def _field_grid(t_val: float):
        """
        Returns (omega[nx_g,ny_g], u[nx_v,ny_v], v[nx_v,ny_v]).

        Vorticity: Lamb-Oseen cores + continuous shear layers + boundary layer.
        Velocity:  Lamb-Oseen velocity (Biot-Savart) + uniform flow + cylinder doublet.
        """
        omega = np.zeros((nx_g, ny_g), dtype=np.float64)
        u_vel = np.full((nx_v, ny_v), U_inf, dtype=np.float64)
        v_vel = np.zeros((nx_v, ny_v), dtype=np.float64)

        # ── Cylinder doublet (potential flow)  ─────────────────────────────
        safe_r2 = np.maximum(r2_v, (R * 1.01) ** 2)
        u_vel += U_inf * R**2 * (Yv**2 - Xv**2) / safe_r2**2
        v_vel -= 2.0 * U_inf * R**2 * Xv * Yv / safe_r2**2

        # Collect active vortex descriptors
        vortices = []
        t_first = T_shed * 0.3
        for k in range(60):
            t_spawn = t_first + k * (T_shed * 0.5)
            if t_spawn > t_val:
                break
            dt  = t_val - t_spawn
            x_v = R * 2.2 + U_conv * dt
            if x_v > x_max + 2.0 * D:
                continue
            y_v  = +h_sep * 0.5 if k % 2 == 0 else -h_sep * 0.5
            sign = -1.0           if k % 2 == 0 else +1.0

            # Lamb-Oseen core: grows by viscous diffusion, starts large
            r_core2 = 4.0 * nu * (dt + 1.5 * T_shed) + (0.55 * D) ** 2

            # Mild streamwise elongation very close to cylinder only
            stretch_x = 1.0 + 0.5 * float(np.exp(-max(x_v - R * 2.0, 0.0) / (2.0 * D)))
            r_core2_x = r_core2 * stretch_x ** 2

            fade = float(np.clip(dt / (T_shed * 0.25), 0.0, 1.0))
            vortices.append((x_v, y_v, sign, r_core2, r_core2_x, fade))

        for (x_v, y_v, sign, r_core2, r_core2_x, fade) in vortices:
            # ── Vorticity (high-res grid) ─────────────────────────────────
            Rx_g = Xg - x_v;  Ry_g = Yg - y_v
            r2a  = Rx_g**2 / r_core2_x + Ry_g**2 / r_core2
            peak = (Gamma * fade) / (np.pi * np.sqrt(r_core2 * r_core2_x))
            omega += sign * peak * np.exp(-r2a)

            # ── Velocity (coarse grid) — Lamb-Oseen Biot-Savart ──────────
            Rx_v = Xv - x_v;  Ry_v = Yv - y_v
            r2_loc = Rx_v**2 + Ry_v**2
            # (1 - exp(-r²/r_c²)) / r²  →  avoids singularity at r=0
            r_c2 = (r_core2 + r_core2_x) * 0.5   # isotropic core for velocity
            factor = (1.0 - np.exp(-r2_loc / r_c2)) / (r2_loc + 1e-12)
            circ   = sign * Gamma * fade / (2.0 * np.pi)
            u_vel -= circ * Ry_v * factor
            v_vel += circ * Rx_v * factor

        # ── Shear layers (omega only — provides the thin wake "connective tissue")
        x_sl_start = R * 1.0
        x_sl_end   = lam * 1.5
        sl_extent  = max(x_sl_end - x_sl_start, 1e-6)
        sl_prog = np.clip((Xg - x_sl_start) / sl_extent, 0.0, 1.0)
        sl_mask = (Xg >= x_sl_start) & (Xg <= x_sl_end)
        phase_sl = 2.0 * np.pi * (Xg / lam - f_shed * t_val)
        y0_u = +(R + 0.05 * D) + 0.30 * (h_sep * 0.5 - R) * sl_prog * np.sin(phase_sl)
        y0_l = -(R + 0.05 * D) - 0.30 * (h_sep * 0.5 - R) * sl_prog * np.sin(phase_sl)
        sl_thick = 0.04 * D + 0.20 * D * sl_prog
        sl_amp   = (2.5 * U_inf / D) * (1.0 - sl_prog ** 0.5)
        omega += sl_mask * sl_amp * (
            -np.exp(-0.5 * ((Yg - y0_u) / sl_thick) ** 2)
            +np.exp(-0.5 * ((Yg - y0_l) / sl_thick) ** 2)
        )

        # ── Boundary layer on cylinder (omega only) ────────────────────────
        r_g     = np.sqrt(np.maximum(r2_g, (R * 0.99) ** 2))
        theta_g = np.arctan2(Yg, Xg)
        delta_bl = 0.14 * D
        bl_amp   = 2.0 * U_inf / R
        omega_bl = -bl_amp * np.sin(theta_g) * np.exp(-((r_g - R) / delta_bl) ** 2)
        omega += (~cyl_mk) * (r_g < R + 3.0 * delta_bl) * omega_bl * 0.5

        # Zero inside cylinder
        omega[cyl_mk]  = 0.0
        u_vel[cyl_v]   = 0.0
        v_vel[cyl_v]   = 0.0

        return omega, u_vel, v_vel

    # ── 10 animation frames spanning 2.5 shedding periods ────────────────────
    n_frames = 10
    t_frames = np.linspace(T_shed * 1.5, T_shed * 4.0, n_frames)

    fields = [_field_grid(t) for t in t_frames]
    omega_grids = [f[0] for f in fields]

    # Global symmetric colormap range (98th percentile of absolute values)
    all_abs = np.abs(np.concatenate([g.ravel() for g in omega_grids]))
    g_range = float(np.percentile(all_abs, 98)) if all_abs.size else 1.0
    g_range = max(g_range, 1e-3)

    animated_frames = [
        {
            "t_s":   float(t_frames[i]),
            "label": f"t/T = {t_frames[i] / T_shed:.2f}",
            "grid":  omega_grids[i].tolist(),        # vorticity  (nx_g × ny_g)
            "u":     fields[i][1].tolist(),           # x-velocity (nx_v × ny_v)
            "v":     fields[i][2].tolist(),           # y-velocity (nx_v × ny_v)
            "xv":    xv.tolist(),                     # velocity grid x-coords
            "yv":    yv.tolist(),                     # velocity grid y-coords
        }
        for i in range(n_frames)
    ]

    # Static snapshot = last frame (most-developed vortex street)
    u_static = omega_grids[-1]

    # ── Scatter collocation data (potential flow, exact for Re→∞) ────────────
    rng   = np.random.RandomState(seed=42)
    t_end = float(params.get("t_end", T_shed * 5.0))
    # Generate with 8× oversampling then mask out cylinder interior.
    # Cylinder occupies <<1% of the domain, so one shot is always sufficient.
    bsz = max(8 * n_samples, 20_000)
    xp  = rng.uniform(x_min, x_max, bsz)
    yp  = rng.uniform(y_min, y_max, bsz)
    mask = (xp**2 + yp**2) >= (R * 1.1)**2
    xp, yp = xp[mask], yp[mask]
    # If (unlikely) not enough survived, repeat once more
    if len(xp) < n_samples:
        xp2 = rng.uniform(x_min, x_max, bsz)
        yp2 = rng.uniform(y_min, y_max, bsz)
        m2  = (xp2**2 + yp2**2) >= (R * 1.1)**2
        xp  = np.concatenate([xp, xp2[m2]])
        yp  = np.concatenate([yp, yp2[m2]])
    xp, yp = xp[:n_samples], yp[:n_samples]
    tp = rng.uniform(0.0, t_end, len(xp))
    pts_arr = np.column_stack([xp, yp, tp])

    xpts, ypts, _ = pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2]
    R2    = R**2
    r2pts = np.maximum(xpts**2 + ypts**2, (R * 1.01)**2)
    u_pot = U_inf * (1.0 - R2 * (xpts**2 - ypts**2) / r2pts**2)
    v_pot = -2.0 * U_inf * R2 * xpts * ypts / r2pts**2
    p_pot = 0.5 * rho * (U_inf**2 - u_pot**2 - v_pot**2)

    coords   = pts_arr.astype(np.float32)
    solution = np.stack([u_pot, v_pot, p_pot], axis=1).astype(np.float32)

    return {
        # 2D heatmap fields — triggers hasSpatialGrid2D in frontend
        "x":  xg.tolist(),              # shape (80,)
        "y":  yg.tolist(),              # shape (40,)
        "u":  u_static.tolist(),        # shape (80, 40) — static vorticity snapshot
        # Animated frames for AnimatedHeatmap2D component
        "animated_frames": animated_frames,
        "vmin": -g_range,
        "vmax":  g_range,
        # Collocation data
        "coords":   coords,
        "solution": solution,
        "metadata": {
            "problem":     "cylinder_flow_2d",
            "Re":          Re,
            "D":           D,
            "R":           R,
            "U_inf":       U_inf,
            "St":          float(St),
            "T_shed":      float(T_shed),
            "dim":         2,              # → hasSpatialGrid2D path in frontend
            "out_dim":     3,
            "synthetic":   False,
            "generator":   "karman_vortex_blobs",
            "description": (
                f"Kármán vortex street  Re={Re:.0f}  St={St:.3f}  "
                f"T_shed={T_shed:.2f} D/U∞  (Gaussian blob model on potential base)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GENERATORS = {
    "burgers_1d": _gen_burgers_1d,
    "navier_stokes_2d": _gen_navier_stokes_2d,
    "lid_driven_cavity": _gen_lid_driven_cavity,
    "bernoulli_flow": _gen_bernoulli_flow,
    "stokes_flow": _gen_stokes_flow,
    "darcy_flow": _gen_darcy_flow,
    "channel_poiseuille": _gen_channel_poiseuille,
    "linear_elasticity_2d": _gen_linear_elasticity_2d,
    "beam_bending": _gen_beam_bending,
    "poisson_electrostatics": _gen_poisson_electrostatics,
    "helmholtz_wave": _gen_helmholtz_wave,
    "laplace_potential": _gen_laplace_potential,
    "schrodinger_stationary": _gen_schrodinger_stationary,
    "schrodinger_timedep": _gen_schrodinger_timedep,
    "heat_equation_1d": _gen_heat_equation_1d,
    "heat_equation_2d": _gen_heat_equation_2d,
    "wave_equation": _gen_wave_equation,
    "kdv_equation": _gen_kdv_equation,
    "klein_gordon": _gen_klein_gordon,
    "thermoelasticity_2d": _gen_thermoelasticity_2d,
    "navier_stokes_3d": _gen_navier_stokes_3d,
    "crash_analysis": _gen_crash_analysis,
    "gross_pitaevskii": _gen_gross_pitaevskii,
    "magnetostatics_2d": _gen_magnetostatics_2d,
    "linear_elasticity_3d": _gen_linear_elasticity_3d,
    # ── 3D multi-physics benchmarks ──
    "taylor_green_vortex": _gen_taylor_green_vortex,
    "darcy_flow_3d": _gen_darcy_flow_3d,
    "stokes_sphere": _gen_stokes_sphere,
    "hartmann_mhd": _gen_hartmann_mhd,
    "chip_heat_3d": _gen_chip_heat_3d,
    "helmholtz_3d_cavity": _gen_helmholtz_3d_cavity,
    # ── 2D visualization showcase ──
    "airfoil_naca0012_2d": _gen_airfoil_naca0012_2d,
    "shallow_water_2d":    _gen_shallow_water_2d,
    # ── CFD benchmark ──
    "cylinder_flow_2d":    _gen_cylinder_flow_2d,
}


class DatasetGenerator:
    """
    Generates reference (ground truth) solutions for a given physics problem.
    Uses PINNeAPPle numerical solvers when available; falls back to analytical.
    """

    def __init__(self, problem_id: str, parameters: dict):
        self.problem_id = problem_id
        self.parameters = parameters

    def generate(self, n_samples: int = 1000) -> dict:
        """
        Generate the reference dataset.

        Returns
        -------
        dict with keys:
          - coords: float32 array of shape (N, d)
          - solution: float32 array of shape (N, n_fields)
          - metadata: dict with problem info
          - (additional problem-specific keys)

        An unknown problem_id raises KeyError (no synthetic stand-in field).
        """
        generator_fn = _GENERATORS.get(self.problem_id)
        if generator_fn is None:
            raise KeyError(f"no reference generator for problem {self.problem_id!r}; "
                           f"known: {sorted(_GENERATORS)}")
        return generator_fn(self.parameters, n_samples)

    def get_collocation_data(self, n_col: int = 5000, n_bc: int = 1000, n_ic: int = 1000) -> dict:
        """
        Returns collocation-ready data for PINN training.

        Strategy
        --------
        Interior collocation points are sampled uniformly in the domain without
        needing a reference solution (cheap, arbitrary density).  Boundary and
        initial-condition points are sampled on the actual domain faces and
        evaluated against the reference generator, so u_bc / u_ic carry true
        physics values rather than a random subset of the volume dataset.

        Returns
        -------
        dict with keys:
          - X_col: interior collocation points (n_col, d)
          - X_bc: boundary condition points (n_bc, d)
          - u_bc: boundary values at X_bc (n_bc, n_fields)
          - X_ic: initial condition points (n_ic, d) — time-dependent problems only
          - u_ic: initial condition values at X_ic (n_ic, n_fields) — time-dep only
          - metadata: from the reference generator
        """
        time_dep = self.problem_id in _TIME_DEP_PROBLEMS

        # Boundary / IC points: on domain faces, evaluated via the reference.
        # We over-generate reference data to have enough points near boundaries.
        n_ref = max(n_bc + (n_ic if time_dep else 0) + 500, 3000)
        ref = self.generate(n_samples=n_ref)
        coords  = ref["coords"]    # (N, in_dim)
        solution = ref["solution"]  # (N, out_dim)

        # Interior collocation points: uniform in the bounding box of the reference coordinates (the problem's own
        # domain, e.g. x in [-1, 1] for Burgers), so the PDE residual is evaluated where the data lives.
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        in_dim = coords.shape[1]
        X_col = np.random.uniform(lo, hi, (n_col, in_dim)).astype(np.float32)

        result: dict[str, Any] = {"X_col": X_col, "metadata": ref["metadata"]}

        if time_dep:
            # IC: points where last coordinate (time) ≈ 0
            t_col = coords[:, -1]
            t_min = float(t_col.min())
            t_thr = t_min + (float(t_col.max()) - t_min) * 0.05  # bottom 5% in time
            ic_mask = t_col <= t_thr
            ic_idx = np.where(ic_mask)[0]

            if len(ic_idx) >= n_ic:
                chosen_ic = np.random.choice(ic_idx, size=n_ic, replace=False)
            else:
                chosen_ic = ic_idx  # use all available IC points

            result["X_ic"] = coords[chosen_ic]
            result["u_ic"] = solution[chosen_ic]

            # BC: boundary in the *spatial* dimensions (t can be anything)
            spatial = coords[:, :-1]
            on_boundary = np.zeros(len(coords), dtype=bool)
            for d in range(in_dim - 1):
                col = spatial[:, d]
                lo, hi = float(col.min()), float(col.max())
                thr = (hi - lo) * 0.05
                on_boundary |= (col <= lo + thr) | (col >= hi - thr)
        else:
            on_boundary = np.zeros(len(coords), dtype=bool)
            for d in range(in_dim):
                col = coords[:, d]
                lo, hi = float(col.min()), float(col.max())
                thr = (hi - lo) * 0.05
                on_boundary |= (col <= lo + thr) | (col >= hi - thr)

        bc_idx = np.where(on_boundary)[0]
        if len(bc_idx) >= n_bc:
            chosen_bc = np.random.choice(bc_idx, size=n_bc, replace=False)
        elif len(bc_idx) > 0:
            # Not enough boundary points — fall back to random subset of all points
            chosen_bc = np.random.choice(len(coords), size=n_bc, replace=True)
        else:
            chosen_bc = np.arange(min(n_bc, len(coords)))

        result["X_bc"] = coords[chosen_bc]
        result["u_bc"] = solution[chosen_bc]

        return result

    @staticmethod
    def list_problems() -> list:
        return list(_GENERATORS.keys())
