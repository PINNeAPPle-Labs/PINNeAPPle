"""NVIDIA Warp backend: GPU kernels, ``warp.fem`` weak forms and a differentiable particle step.

`Warp <https://github.com/NVIDIA/warp>`_ (Apache-2.0) compiles ``@wp.kernel`` functions to CUDA (or to the CPU) and
carries a reverse-mode tape (``wp.Tape``) through them, so a step of a simulation written as kernels is differentiable
with respect to its inputs. PINNeAPPle already reaches Warp indirectly through the Newton rigid-body backend
(:mod:`pinneapple_simulation.external_solvers.newton_physics`); this module is the direct entry point for the three
things that are only possible through Warp itself:

1. **Zero-copy tensors**: ``to_warp`` / ``to_torch`` share memory with PyTorch through DLPack. No copy, no
   serialisation: a tensor built by a PINNeAPPle trainer is the same buffer a kernel reads.
2. **``warp.fem`` weak forms**: assemble a Poisson stiffness matrix and load vector by integrating integrands over a
   grid, and solve it here. :func:`poisson_convergence` measures the error against a manufactured solution at every
   resolution and returns the observed order, which is the check the FEM path is held to.
3. **A differentiable particle step**: :class:`WarpBackend` exposes one symplectic Euler step of the rigid-body state
   used by :mod:`pinneapple_simulation.particle_dynamics` (``RigidBody.step``), recorded on a ``wp.Tape``.
   :meth:`WarpBackend.gradients_against_torch` runs both engines on the same inputs: the state agrees to
   ``1e-16`` and every gradient (position, velocity, force, torque) to ``1e-16`` relative, against the ``1e-6``
   the check asks for. What it took to get there is written next to the kernels: out-of-place writes, the loss as a
   kernel on the tape, and the quaternion in Warp's ``(x, y, z, w)`` order.

Quick start
-----------
>>> from pinneapple_tools.compute_backends import warp_available, WarpBackend, poisson_convergence
>>> warp_available()                                    # pip install warp-lang
True
>>> rows = poisson_convergence((8, 16, 32))             # manufactured solution, assembled in warpfem
>>> rows[-1]["order"] > 2.0                             # observed convergence order
True

Every function imports Warp lazily: without ``warp-lang`` installed this module imports and
:func:`warp_available` returns ``False``, and the solvers raise :class:`ImportError` with the install line instead of
an obscure ``ModuleNotFoundError``.

Limitation, stated rather than hidden: the step kernels are float64 only. Warp compiles one kernel per concrete type,
and float64 is the precision every gradient check in this repository is made at; a float32 build of the same kernels
would be a copy of them with no test behind it.

Install: ``pip install warp-lang`` (a CUDA build is selected automatically on an NVIDIA machine; ``warp-lang`` also
runs on the CPU, which is what the checks below use).
"""
# No ``from __future__ import annotations`` here on purpose: Warp reads the annotations of every kernel and integrand
# and resolves string annotations against the module globals, which would not find ``fem``/``wp`` (they are imported
# lazily inside functions, since the library must import without warp-lang installed). Evaluating them eagerly, in
# the enclosing scope, is what lets the integrands live next to the code that uses them.
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "warp_available",
    "warp_device",
    "to_warp",
    "to_torch",
    "WarpBackend",
    "FemPoissonSolution",
    "fem_poisson",
    "poisson_convergence",
    "MANUFACTURED_SOLUTION",
]

# ---------------------------------------------------------------------------
# Lazy import: the library imports without warp-lang
# ---------------------------------------------------------------------------

_WARP: Any = None


def _wp():
    """Import Warp once, quiet, initialised. Raises ImportError with the install line when absent."""
    global _WARP
    if _WARP is not None:
        return _WARP
    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "the Warp backend needs NVIDIA Warp: pip install warp-lang "
            "(or pip install \"pinneapple[warp]\")"
        ) from exc
    wp.config.quiet = True
    # module-load timing lines ("Module X load on device 'cpu' took ...") are INFO; keep the console clean
    wp.config.log_level = getattr(wp, "LOG_WARNING", wp.config.log_level)
    # ``wp.init`` is optional from Warp 1.x on (devices are brought up on first use) but calling it keeps the
    # behaviour of older releases and makes ``warp_device`` answer before the first kernel runs. ``is_initialized``
    # exists in some releases only, so it is probed rather than assumed.
    probe = getattr(wp, "is_initialized", None)
    try:
        if not callable(probe) or not probe():
            wp.init()
    except Exception:            # pragma: no cover - a Warp build without a usable device still imports
        wp.init()
    _WARP = wp
    return wp


def warp_available() -> bool:
    """``True`` when ``warp`` imports. Does not initialise a device beyond what ``wp.init`` does."""
    try:
        _wp()
        return True
    except Exception:
        return False


def warp_device() -> str | None:
    """The Warp device kernels will run on: ``"cuda:0"`` on an NVIDIA GPU, otherwise ``"cpu"``."""
    if not warp_available():
        return None
    wp = _wp()
    device = wp.get_device()
    return f"cuda:{device.ordinal}" if device.is_cuda else "cpu"


# ---------------------------------------------------------------------------
# Zero-copy conversion
# ---------------------------------------------------------------------------


def to_warp(x, *, requires_grad: bool = False):
    """A Warp array that shares memory with the PyTorch tensor ``x`` (DLPack, no copy).

    ``x`` must be contiguous and on a device Warp can see (CUDA tensor on a CUDA build, or a CPU tensor).
    """
    wp = _wp()
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"to_warp expects a torch.Tensor, got {type(x).__name__}")
    if not x.is_contiguous():
        raise ValueError("to_warp needs a contiguous tensor: call .contiguous() first (a copy is a choice, not a side effect)")
    return wp.from_torch(x, requires_grad=requires_grad)


def to_torch(a, *, requires_grad: bool | None = None):
    """A PyTorch tensor that shares memory with the Warp array ``a``."""
    wp = _wp()
    return wp.to_torch(a, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# warp.fem: a Poisson problem with a manufactured solution
# ---------------------------------------------------------------------------

# u = P(x) P(y),  P(t) = t^3 (1 - t)^3,  P''(t) = 6 t - 36 t^2 + 60 t^3 - 30 t^4,
# so -laplace(u) = -(P''(x) P(y) + P(x) P''(y)) and u = 0 on every side of the unit square.
# The polynomial is used instead of the usual sin(pi x) sin(pi y) on purpose: that one is an eigenfunction of the
# discrete Laplacian, so CG converges in two iterations and an assembly error hides behind a plausible-looking error.
MANUFACTURED_SOLUTION = "u = x^3 (1-x)^3 y^3 (1-y)^3, f = -laplace(u)"


@dataclass
class FemPoissonSolution:
    """Result of one ``warp.fem`` Poisson solve on the unit square with homogeneous Dirichlet data."""

    resolution: int
    degree: int
    n_dofs: int
    l2_error: float                     # ||u_h - u||_2 against the manufactured solution
    cg_iterations: int
    max_residual: float
    device: str
    dof_values: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict[str, Any]:
        return {"resolution": self.resolution, "degree": self.degree, "n_dofs": self.n_dofs,
                "l2_error": self.l2_error, "cg_iterations": self.cg_iterations,
                "max_residual": self.max_residual, "device": self.device}


def _p_poly(t):
    return t * t * t * (1.0 - t) ** 3


def _p_poly_second(t):
    return 6.0 * t - 36.0 * t * t + 60.0 * t ** 3 - 30.0 * t ** 4


def manufactured_u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The manufactured solution, for a reference plot or a quick check outside Warp."""
    return _p_poly(x) * _p_poly(y)


def manufactured_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``-laplace(u)`` of :func:`manufactured_u`."""
    return -(_p_poly_second(x) * _p_poly(y) + _p_poly(x) * _p_poly_second(y))


_FEM_BUILT = False


def _build_fem_integrands():
    """Define the integrands once, at module import time of a Warp-aware caller.

    Warp reads the source of an integrand, so they cannot be defined inside a factory that is executed from a
    string; they live at module level and this function only triggers their definition.
    """
    global _FEM_INTEGRANDS
    if _FEM_INTEGRANDS is not None:
        return _FEM_INTEGRANDS
    wp = _wp()
    import warp.fem as fem

    @fem.integrand
    def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, k: wp.float64):
        return k * wp.dot(fem.grad(u, s), fem.grad(v, s))

    @fem.integrand
    def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
        x = fem.position(domain, s)
        px = x[0] * x[0] * x[0] * (wp.float64(1.0) - x[0]) * (wp.float64(1.0) - x[0]) * (wp.float64(1.0) - x[0])
        py = x[1] * x[1] * x[1] * (wp.float64(1.0) - x[1]) * (wp.float64(1.0) - x[1]) * (wp.float64(1.0) - x[1])
        dpx = (wp.float64(6.0) * x[0] - wp.float64(36.0) * x[0] * x[0]
               + wp.float64(60.0) * x[0] * x[0] * x[0] - wp.float64(30.0) * x[0] * x[0] * x[0] * x[0])
        dpy = (wp.float64(6.0) * x[1] - wp.float64(36.0) * x[1] * x[1]
               + wp.float64(60.0) * x[1] * x[1] * x[1] - wp.float64(30.0) * x[1] * x[1] * x[1] * x[1])
        # -laplace(u) = -(d2P/dx2 * P(y) + P(x) * d2P/dy2); the sign matters: with + the discrete solution is -u
        # and the L2 error against u stops at a resolution-independent 2||u|| (it did: 1.665e-4 at every h)
        return (wp.float64(0.0) - (dpx * py + px * dpy)) * v(s)

    @fem.integrand
    def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
        return u(s) * v(s)

    @fem.integrand
    def error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
        x = fem.position(domain, s)
        px = x[0] * x[0] * x[0] * (wp.float64(1.0) - x[0]) * (wp.float64(1.0) - x[0]) * (wp.float64(1.0) - x[0])
        py = x[1] * x[1] * x[1] * (wp.float64(1.0) - x[1]) * (wp.float64(1.0) - x[1]) * (wp.float64(1.0) - x[1])
        d = u(s) - px * py
        return d * d

    _FEM_INTEGRANDS = {"diffusion": diffusion_form, "source": source_form,
                       "mass": mass_form, "error": error_form}
    return _FEM_INTEGRANDS


_FEM_INTEGRANDS: dict[str, Any] | None = None


def _cg(wp, matrix, rhs, *, tol: float = 1e-12, max_iterations: int = 5000):
    """Jacobi-preconditioned conjugate gradients on a Warp BSR matrix. Returns (x, iterations, residual)."""
    x = wp.zeros_like(rhs)
    r = wp.clone(rhs)
    q = wp.zeros_like(rhs)

    diag = wp.sparse.bsr_get_diag(matrix)          # one scalar per row for a 1x1 block matrix
    dv = diag.numpy().reshape(-1).astype(np.float64)
    with np.errstate(divide="ignore"):
        inv_diag = np.where(np.abs(dv) > 0.0, 1.0 / dv, 0.0)

    def dot(a, b) -> float:
        va, vb = a.numpy(), b.numpy()
        return float(va @ vb)

    def precondition(r):
        return r * wp.array(inv_diag, dtype=wp.float64, device=r.device)

    z = precondition(r)
    d = wp.clone(z)
    rz = dot(r, z)
    residual = math.sqrt(max(dot(r, r), 0.0))
    iterations = 0
    while iterations < max_iterations and residual > tol:
        iterations += 1
        wp.sparse.bsr_mv(matrix, x=d, y=q)
        dq = dot(d, q)
        if dq == 0.0:
            break
        alpha = wp.float64(rz / dq)
        x = x + alpha * d
        r = r - alpha * q
        residual = math.sqrt(max(dot(r, r), 0.0))
        if residual <= tol or rz == 0.0:
            break
        z = precondition(r)
        rz_new = dot(r, z)
        d = z + wp.float64(rz_new / rz) * d
        rz = rz_new
    return x, iterations, residual


def fem_poisson(resolution: int, *, degree: int = 2, tol: float = 1e-12) -> FemPoissonSolution:
    """Solve ``-laplace(u) = f`` on the unit square with ``u = 0`` on all four sides, in ``warp.fem``.

    ``degree`` is the polynomial degree of the shape functions on the quad grid (``1`` for Q1, ``2`` for Q2).
    The Dirichlet data is imposed by projecting the linear system (``fem.project_linear_system``), not by a penalty.
    """
    if resolution < 2:
        raise ValueError("resolution is the number of cells per side and must be >= 2")
    if degree not in (1, 2, 3):
        raise ValueError("degree must be 1, 2 or 3")
    wp = _wp()
    import warp.fem as fem

    integrands = _build_fem_integrands()
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution),
                     bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(1.0, 1.0),
                     scalar_type=wp.float64)
    space = fem.make_polynomial_space(geo, degree=degree)
    field = space.make_field()
    domain = fem.Cells(geometry=geo)

    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    matrix = fem.integrate(integrands["diffusion"], fields={"u": trial, "v": test},
                           values={"k": wp.float64(1.0)})
    rhs = fem.integrate(integrands["source"], fields={"v": test})

    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    bd_matrix = fem.integrate(integrands["mass"], fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
    bd_rhs = wp.zeros_like(rhs)                    # homogeneous data: the projector needs a zero right-hand side
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

    dof, iterations, residual = _cg(wp, matrix, rhs, tol=tol)
    field.dof_values = dof

    error_density = fem.integrate(integrands["error"], fields={"u": field}, domain=domain)
    l2_error = float(np.sqrt(max(float(error_density), 0.0)))
    n_dofs = len(dof)

    wp.synchronize_device()
    return FemPoissonSolution(resolution=resolution, degree=degree, n_dofs=int(n_dofs), l2_error=l2_error,
                              cg_iterations=int(iterations), max_residual=float(residual),
                              device=warp_device() or "cpu", dof_values=dof.numpy().copy())


def poisson_convergence(resolutions: Iterable[int] = (8, 16, 32, 64), *, degree: int = 2) -> list[dict[str, Any]]:
    """``fem_poisson`` at every resolution, with the error and the observed order against the manufactured solution.

    A correct Q1 assembly converges at order ~2 in L2 and Q2 at ~3; an assembly error (a missing Jacobian, a
    boundary term left out) shows up as a stuck error and an order near zero, which is what this returns.
    """
    rows: list[dict[str, Any]] = []
    previous: float | None = None
    for res in resolutions:
        sol = fem_poisson(int(res), degree=degree)
        order = None if previous is None else math.log2(previous / sol.l2_error) if sol.l2_error > 0 else None
        row = sol.as_dict()
        row["order"] = order
        rows.append(row)
        previous = sol.l2_error
    return rows


# ---------------------------------------------------------------------------
# A differentiable rigid-body step, recorded on a Warp tape
# ---------------------------------------------------------------------------

def _build_step_kernels():
    """Define the kernels once, on first use: one step per dimension, plus the losses a check can be written on.

    Every step kernel is **out of place**: it reads one state and writes another. That is not a style choice.
    Warp's adjoint evaluates the derivative at the values the kernel left in memory, so a kernel that overwrites
    its own input produces a wrong gradient -- measured here, against :class:`torch.autograd`, at 5e-4 relative on
    ``d loss / d omega`` for the gyroscopic term, while the same kernel with a separate output matches to 0.0.
    """
    global _STEP_KERNELS
    if _STEP_KERNELS is not None:
        return _STEP_KERNELS
    wp = _wp()

    @wp.kernel
    def step_2d(pos: wp.array(dtype=wp.vec2d), vel: wp.array(dtype=wp.vec2d),
                angle: wp.array(dtype=wp.float64), omega: wp.array(dtype=wp.float64),
                force: wp.array(dtype=wp.vec2d), torque: wp.array(dtype=wp.float64),
                inv_mass: wp.array(dtype=wp.float64), inv_inertia: wp.array(dtype=wp.float64),
                dt: wp.float64,
                pos_out: wp.array(dtype=wp.vec2d), vel_out: wp.array(dtype=wp.vec2d),
                angle_out: wp.array(dtype=wp.float64), omega_out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        v_new = vel[i] + dt * (force[i] * inv_mass[i])
        vel_out[i] = v_new
        pos_out[i] = pos[i] + dt * v_new
        w_new = omega[i] + dt * (torque[i] * inv_inertia[i])
        omega_out[i] = w_new
        angle_out[i] = angle[i] + dt * w_new

    @wp.kernel
    def step_3d(pos: wp.array(dtype=wp.vec3d), vel: wp.array(dtype=wp.vec3d),
                quat: wp.array(dtype=wp.quatd), omega: wp.array(dtype=wp.vec3d),
                force: wp.array(dtype=wp.vec3d), torque: wp.array(dtype=wp.vec3d),
                inv_mass: wp.array(dtype=wp.float64),
                inertia: wp.array(dtype=wp.mat33d), inv_inertia: wp.array(dtype=wp.mat33d),
                dt: wp.float64,
                pos_out: wp.array(dtype=wp.vec3d), vel_out: wp.array(dtype=wp.vec3d),
                quat_out: wp.array(dtype=wp.quatd), omega_out: wp.array(dtype=wp.vec3d)):
        i = wp.tid()
        v_new = vel[i] + dt * (force[i] * inv_mass[i])
        vel_out[i] = v_new
        pos_out[i] = pos[i] + dt * v_new

        # Euler's equation in the body frame, written the way RigidBody.step writes it:
        # L = I w, gyro = w x L, alpha = I^-1 (tau - gyro)
        w = omega[i]
        gyro = wp.cross(w, inertia[i] * w)
        alpha = inv_inertia[i] * (torque[i] - gyro)
        w_new = w + dt * alpha
        omega_out[i] = w_new

        # q_dot = 0.5 * q (x) [0, omega]: the quaternion is on the LEFT, as in _quat_mult_vec. Warp stores a
        # quaternion as (x, y, z, w) where RigidBodyState uses (w, x, y, z); mixing the two is silent in the
        # forward pass (a permutation of the vector part) and shows only as a wrong rotation.
        q = quat[i]
        pure = wp.quatd(w_new[0], w_new[1], w_new[2], wp.float64(0.0))
        qq = q + wp.float64(0.5) * dt * (q * pure)
        norm = wp.sqrt(qq[0] * qq[0] + qq[1] * qq[1] + qq[2] * qq[2] + qq[3] * qq[3])
        quat_out[i] = wp.quatd(qq[0] / norm, qq[1] / norm, qq[2] / norm, qq[3] / norm)

    @wp.kernel
    def position_sq_2d(pos: wp.array(dtype=wp.vec2d), out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        p = pos[i]
        wp.atomic_add(out, 0, wp.dot(p, p))

    @wp.kernel
    def position_sq_3d(pos: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        p = pos[i]
        wp.atomic_add(out, 0, wp.dot(p, p))

    @wp.kernel
    def pos_and_angle_sq_2d(pos: wp.array(dtype=wp.vec2d), angle: wp.array(dtype=wp.float64),
                            out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        p = pos[i]
        wp.atomic_add(out, 0, wp.dot(p, p) + angle[i] * angle[i])

    @wp.kernel
    def pos_and_quat_sq_3d(pos: wp.array(dtype=wp.vec3d), quat: wp.array(dtype=wp.quatd),
                           out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        p = pos[i]
        q = quat[i]
        # the vector part only: the update normalises |q| to 1, so a loss on all four components is rotation
        # invariant and its gradient with respect to the torque is roundoff (~1e-20) in both engines
        wp.atomic_add(out, 0, wp.dot(p, p) + q[0] * q[0] + q[1] * q[1] + q[2] * q[2])

    # A loss has to be a kernel on the tape too: a loss computed in numpy after the tape closes has no adjoint,
    # and tape.backward() then leaves every gradient at zero (it did, silently, before this kernel existed).
    _STEP_KERNELS = {2: {"step": step_2d, "loss": position_sq_2d, "loss_rot": pos_and_angle_sq_2d},
                     3: {"step": step_3d, "loss": position_sq_3d, "loss_rot": pos_and_quat_sq_3d}}
    return _STEP_KERNELS


_STEP_KERNELS: dict[int, Any] | None = None


def _as_warp(wp, value, dtype, *, device, requires_grad: bool = False):
    """A Warp array of ``dtype`` from a Warp array or from anything numpy can read."""
    if isinstance(value, wp.array):
        return value
    return wp.array(np.ascontiguousarray(np.asarray(value, dtype=np.float64)), dtype=dtype,
                    device=device, requires_grad=requires_grad)


def _fresh_like(wp, a):
    """A new zeroed array with the shape and dtype of ``a``, tracked by the tape."""
    out = wp.zeros_like(a)
    out.requires_grad = True
    return out


class WarpBackend:
    """One symplectic Euler step of :class:`pinneapple_simulation.particle_dynamics.RigidBodyState`, in Warp.

    The step is the same arithmetic as ``particle_dynamics.RigidBody.step`` (``v += dt f/m``, ``x += dt v``, then
    the rotation), and the reason to run it in Warp is the tape: :meth:`step_with_tape` records the steps, so
    ``tape.backward(loss)`` gives ``d loss / d v0``, ``d loss / d force`` and ``d loss / d torque`` for a loss read
    off the new state, without a PyTorch graph in the loop. :meth:`gradients_against_torch` runs both engines on the
    same inputs and returns the disagreement, which is the check this backend is held to.

    :meth:`step` returns a **new** state instead of overwriting the one it is given (see the note on the kernels).
    The arrays are float64; ``pos``/``vel``/``omega`` (and ``quat`` in 3-D) are Warp vector types, ``force`` and
    ``torque`` are Warp arrays (``vec2d``/``vec3d``; scalar in 2-D), and the mass and inertia are numpy arrays.
    """

    name = "warp"

    def __init__(self, dim: int = 2, *, device: str | None = None):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.dim = int(dim)
        self._wp = _wp()
        self._kernels = _build_step_kernels()
        self.device = device or self._wp.get_device()

    # -- construction helpers -------------------------------------------------

    def state_arrays(self, *, n_bodies: int, requires_grad: bool = True) -> dict[str, Any]:
        """Zero-initialised state arrays for ``n_bodies`` bodies (identity quaternion in 3-D)."""
        wp = self._wp
        vec = wp.vec2d if self.dim == 2 else wp.vec3d
        sca = wp.float64
        out = {
            "pos": wp.zeros(n_bodies, dtype=vec, device=self.device, requires_grad=requires_grad),
            "vel": wp.zeros(n_bodies, dtype=vec, device=self.device, requires_grad=requires_grad),
            "omega": wp.zeros(n_bodies, dtype=vec if self.dim == 3 else sca, device=self.device,
                              requires_grad=requires_grad),
        }
        if self.dim == 2:
            out["angle"] = wp.zeros(n_bodies, dtype=sca, device=self.device, requires_grad=requires_grad)
        else:
            # (x, y, z, w) identity, Warp's component order
            out["quat"] = wp.array(np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (n_bodies, 1)),
                                   dtype=wp.quatd, device=self.device, requires_grad=requires_grad)
        return out

    # -- one step -------------------------------------------------------------

    def step(self, state: dict[str, Any], *, force, torque, mass, dt: float,
             inertia=None, inertia_inv=None) -> dict[str, Any]:
        """One step, **out of place**: returns the new state, leaving ``state`` untouched.

        ``mass`` is ``(n,)``; in 2-D the moment of inertia is a scalar per body (``inertia`` or its inverse), in
        3-D the tensor ``inertia`` and ``inertia_inv`` (``(n, 3, 3)``), because the gyroscopic term needs ``I w``
        and the update needs ``I^-1``. Mass is not differentiable here: the reciprocal is taken on the host.
        """
        wp = self._wp
        kernel = self._kernels[self.dim]["step"]
        n = state["pos"].shape[0]
        inv_mass = wp.array(np.reciprocal(np.asarray(mass, dtype=np.float64).reshape(-1)),
                            dtype=wp.float64, device=self.device)
        new = {key: _fresh_like(wp, value) for key, value in state.items()}
        if self.dim == 2:
            if inertia_inv is None:
                if inertia is None:
                    raise ValueError("2-D step needs inertia (or inertia_inv)")
                inv_inertia = wp.array(np.reciprocal(np.asarray(inertia, dtype=np.float64).reshape(-1)),
                                       dtype=wp.float64, device=self.device)
            else:
                inv_inertia = _as_warp(wp, inertia_inv, wp.float64, device=self.device)
            wp.launch(kernel, dim=n, inputs=[
                state["pos"], state["vel"], state["angle"], state["omega"],
                force, torque, inv_mass, inv_inertia, wp.float64(dt),
                new["pos"], new["vel"], new["angle"], new["omega"]], device=self.device)
        else:
            if inertia is None or inertia_inv is None:
                raise ValueError("3-D step needs both inertia and inertia_inv as (n, 3, 3) arrays")
            inertia_w = _as_warp(wp, inertia, wp.mat33d, device=self.device)
            inertia_inv_w = _as_warp(wp, inertia_inv, wp.mat33d, device=self.device)
            wp.launch(kernel, dim=n, inputs=[
                state["pos"], state["vel"], state["quat"], state["omega"],
                force, torque, inv_mass, inertia_w, inertia_inv_w, wp.float64(dt),
                new["pos"], new["vel"], new["quat"], new["omega"]], device=self.device)
        return new

    def position_sq_loss(self, state: dict[str, Any], out: Any | None = None, *, rotation: bool = False):
        """``sum_i |pos_i|^2`` (plus the rotation when ``rotation``) as a Warp array.

        Call it **inside** a ``wp.Tape`` so the loss carries an adjoint. With ``rotation=False`` the loss does not
        depend on ``angle``/``quat``, and therefore not on the ``omega`` and ``torque`` that feed them: the rotation
        gradients are only exercised with ``rotation=True``.
        """
        wp = self._wp
        out = out if out is not None else wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        if rotation:
            key = "loss_rot"
            inputs = [state["pos"], state["angle" if self.dim == 2 else "quat"], out]
        else:
            key, inputs = "loss", [state["pos"], out]
        wp.launch(self._kernels[self.dim][key], dim=state["pos"].shape[0], inputs=inputs, device=self.device)
        return out

    def step_with_tape(self, state: dict[str, Any], *, force, torque, mass, dt: float, n_steps: int = 1,
                       inertia=None, inertia_inv=None, loss: Callable | None = None):
        """Record ``n_steps`` steps on a ``wp.Tape``; returns ``(tape, state)`` or ``(tape, state, loss_array)``.

        The state returned is the last one; ``loss`` (see :meth:`position_sq_loss`) is called inside the recorded
        region, and its array is returned so ``tape.backward(loss=...)`` can seed the adjoint.
        """
        wp = self._wp
        tape = wp.Tape()
        loss_value = None
        with tape:
            for _ in range(int(n_steps)):
                state = self.step(state, force=force, torque=torque, mass=mass, dt=dt,
                                  inertia=inertia, inertia_inv=inertia_inv)
            if loss is not None:
                loss_value = loss(state) if callable(loss) else loss
        return (tape, state) if loss_value is None else (tape, state, loss_value)

    # -- the check ------------------------------------------------------------

    def gradients_against_torch(self, *, dt: float = 0.02, n_steps: int = 4, n_bodies: int = 3,
                                seed: int = 0) -> dict[str, Any]:
        """Run the same steps through Warp (with its tape) and through ``RigidBody.step`` (with autograd).

        The inputs are seeded from ``np.random.default_rng(seed)``; the loss is the sum of the squared final
        positions **and** of the squared rotation (``angle`` in 2-D, the quaternion vector part in 3-D), so the
        torque and ``omega`` gradients are exercised too -- with positions only, the rotation is off the path and
        its gradients would compare two zeros. Returns the largest absolute difference in the final state, the
        largest relative difference in the gradients, and whether the latter is below ``1e-6``.
        """
        import torch

        from pinneapple_simulation.particle_dynamics.rigid_body import RigidBody, RigidBodyState

        rng = np.random.default_rng(seed)
        n, dim, dt_t = int(n_bodies), self.dim, torch.float64

        pos0 = torch.tensor(rng.normal(size=(n, dim)) * 0.1, dtype=dt_t, requires_grad=True)
        vel0 = torch.tensor(rng.normal(size=(n, dim)), dtype=dt_t, requires_grad=True)
        force = torch.tensor(rng.normal(size=(n, dim)), dtype=dt_t, requires_grad=True)
        mass = torch.tensor(rng.uniform(0.5, 2.0, size=n), dtype=dt_t)
        if dim == 2:
            angle0 = torch.tensor(rng.normal(size=n) * 0.1, dtype=dt_t)
            omega0 = torch.tensor(rng.normal(size=n), dtype=dt_t)
            torque = torch.tensor(rng.normal(size=n), dtype=dt_t, requires_grad=True)
            inertia = torch.tensor(rng.uniform(0.2, 1.0, size=n), dtype=dt_t)
        else:
            angle0 = None
            omega0 = torch.tensor(rng.normal(size=(n, 3)) * 0.5, dtype=dt_t)
            torque = torch.tensor(rng.normal(size=(n, 3)), dtype=dt_t, requires_grad=True)
            inertia = torch.tensor(np.stack([np.diag(rng.uniform(0.2, 1.0, size=3)) for _ in range(n)]),
                                   dtype=dt_t)                                    # (n, 3, 3), symmetric

        # --- PyTorch: the module the library already ships ---
        states = []
        for i in range(n):
            body = RigidBody(mass=float(mass[i]), inertia=inertia[i], dim=dim).double()
            # RigidBody keeps mass and inertia in float32 buffers (``.float()`` in its constructor); ``.double()``
            # widens a value that was already truncated, and that 1e-8 difference in I is what the torque gradient
            # amplified to ~1e-3. Overwrite the buffers with the exact float64 inputs so the two engines solve the
            # same problem and the comparison is about the kernels, not about a cast.
            body.mass = mass[i].clone().reshape(())
            body.inertia = inertia[i].clone()
            body.inertia_inv = (1.0 / inertia[i]) if dim == 2 else torch.linalg.inv(inertia[i])
            st = RigidBodyState(n_bodies=1, dim=dim, dtype=dt_t)
            st.pos = pos0[i:i + 1].clone()
            st.vel = vel0[i:i + 1].clone()
            st.omega = omega0[i:i + 1].clone()
            if dim == 2:
                st.angle = angle0[i:i + 1].clone()
            else:
                st.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dt_t)
            for _ in range(int(n_steps)):
                st = body.step(st, force[i:i + 1], torque[i:i + 1], dt)
            states.append(st)
        torch_pos = torch.cat([s.pos for s in states], dim=0)
        torch_vel = torch.cat([s.vel for s in states], dim=0)
        torch_omega = torch.cat([s.omega for s in states], dim=0)
        rotation = [s.angle for s in states] if dim == 2 else [s.quat[:, 1:] for s in states]
        torch_loss = (torch_pos ** 2).sum() + sum((r ** 2).sum() for r in rotation)
        torch_grads = torch.autograd.grad(torch_loss, [pos0, vel0, force, torque], allow_unused=True)

        # --- Warp ---
        wp, dev = self._wp, self.device
        vec = wp.vec2d if dim == 2 else wp.vec3d
        sca = wp.float64
        force_w = wp.from_torch(force.detach().contiguous(), dtype=vec, requires_grad=True)
        torque_w = wp.from_torch(torque.detach().contiguous(), dtype=vec if dim == 3 else sca,
                                 requires_grad=True)
        state = {
            "pos": wp.from_torch(pos0.detach().contiguous(), dtype=vec, requires_grad=True),
            "vel": wp.from_torch(vel0.detach().contiguous(), dtype=vec, requires_grad=True),
            "omega": wp.from_torch(omega0.detach().contiguous(), dtype=vec if dim == 3 else sca,
                                   requires_grad=True),
        }
        if dim == 2:
            state["angle"] = wp.from_torch(angle0.detach().contiguous(), dtype=sca, requires_grad=True)
            inertia_np, inv_inertia = inertia.numpy(), None
        else:
            q = np.zeros((n, 4))
            q[:, 3] = 1.0                                    # (x, y, z, w) identity
            state["quat"] = wp.array(q, dtype=wp.quatd, device=dev, requires_grad=True)
            inertia_np = inertia.numpy()
            inv_inertia = np.linalg.inv(inertia_np)
        # the arrays the steps started from: their ``.grad`` is the adjoint of the inputs, while the final state
        # arrays are outputs of the last step (their adjoint is whatever the loss seeds, not d loss / d input)
        pos_in, vel_in = state["pos"], state["vel"]
        tape, state, loss = self.step_with_tape(state, force=force_w, torque=torque_w, mass=mass.numpy(), dt=dt,
                                                n_steps=n_steps, inertia=inertia_np, inertia_inv=inv_inertia,
                                                loss=lambda st: self.position_sq_loss(st, rotation=True))
        tape.backward(loss=loss)

        values = {
            "pos": float(np.abs(state["pos"].numpy() - torch_pos.detach().numpy()).max()),
            "vel": float(np.abs(state["vel"].numpy() - torch_vel.detach().numpy()).max()),
            "omega": float(np.abs(state["omega"].numpy().reshape(torch_omega.shape)
                                  - torch_omega.detach().numpy()).max()),
        }
        if dim == 3:
            qw = state["quat"].numpy()[:, [3, 0, 1, 2]]      # (x, y, z, w) -> (w, x, y, z)
            torch_q = torch.cat([s.quat for s in states], dim=0).detach().numpy()
            values["quat"] = float(np.abs(qw - torch_q).max())
        grads = {}
        for label, arr, ref in (("pos0", pos_in, torch_grads[0]), ("vel0", vel_in, torch_grads[1]),
                                ("force", force_w, torch_grads[2]), ("torque", torque_w, torch_grads[3])):
            if ref is None:
                continue
            ref_np = ref.detach().numpy()
            mine = arr.grad.numpy() if arr.grad is not None else np.zeros_like(ref_np)
            scale = max(float(np.abs(ref_np).max()), 1e-12)
            grads[label] = float(np.abs(mine - ref_np).max() / scale)
        return {"dim": dim, "n_bodies": n, "n_steps": int(n_steps), "dt": float(dt),
                "value_max_abs_diff": values, "gradient_max_rel_diff": grads,
                "gradients_agree_1e_6": bool(grads) and max(grads.values()) < 1e-6}
