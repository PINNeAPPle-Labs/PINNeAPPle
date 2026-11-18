"""Integral kernels of the viscous vortex domains (VVD) method.

Everything here is written for a 2-D field of *vortex domains* (blobs of circulation ``gamma`` and core radius
``eps``) plus a set of straight panels (the body surface) that can carry a vortex or a source sheet.

=============  ==========================================================================================
blobs          ``u(r) = sum_j gamma_j / (2 pi) * rotl(r - r_j) / (|r - r_j|^2 + eps_j^2)`` -- the regularised
               (Lamb / Oseen) kernel of Dynnikova's VVD method: finite at the centre, and equal to the point
               vortex far away
panels         the exact velocity of a constant-strength source or vortex sheet on a straight segment,
               integrated in closed form (complex logarithm, real arc-length element)
diffusion      the deterministic drift that stands in for viscosity: ``V = nu * I2 / I1`` between domains
               (Dynnikova's "vortex diffusive" velocity) and ``nu * I3 / (2 pi eps^2 - I0)`` from the walls
               ("body diffusive" velocity)
=============  ==========================================================================================

All the panel formulas are verified against a 20 000-point quadrature of the point kernels in
``tests/test_vortex_domains.py``; the reimplementation follows Dynnikova & Dynnikov, *Viscous vortex domains method
for the Navier-Stokes equations in velocity-vorticity formulation* (and the vvflow reference implementation of it,
MIT), which is cited in ``docs/guides/vortex_domains.md``.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
"""The kernel weights are per ``2 pi``."""

EXP_CUTOFF = 8.0
"""Beyond ``|rho| > 8 eps`` the exponential kernels are below ``3e-4`` and the pair is skipped."""


def rotl(a: np.ndarray) -> np.ndarray:
    """Rotate 2-vectors by +90 degrees: ``(x, y) -> (-y, x)``."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    out[..., 0] = -a[..., 1]
    out[..., 1] = a[..., 0]
    return out


# ---------------------------------------------------------------------------------------------- vortex domains
def blob_velocity(points: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: np.ndarray, eps: np.ndarray,
                  chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Velocity induced at ``points`` by the vortex domains ``(x, y, gamma, eps)``, regularised Lamb kernel."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps2 = np.asarray(eps, dtype=float) ** 2
    u = np.zeros(points.shape[0])
    v = np.zeros(points.shape[0])
    for start in range(0, points.shape[0], chunk):
        p = points[start:start + chunk]
        dx = p[:, None, 0] - x[None, :]
        dy = p[:, None, 1] - y[None, :]
        inv = gamma[None, :] / (dx * dx + dy * dy + eps2[None, :])
        u[start:start + chunk] = (inv * -dy).sum(axis=1) / TWO_PI
        v[start:start + chunk] = (inv * dx).sum(axis=1) / TWO_PI
    return u, v


def blob_vorticity(points: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: np.ndarray, eps: np.ndarray,
                   chunk: int = 2048) -> np.ndarray:
    """Vorticity at ``points``: each domain spreads its circulation as ``gamma / (pi eps^2) exp(-r^2/eps^2)``."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps2 = np.asarray(eps, dtype=float) ** 2
    weight = gamma / (np.pi * eps2)
    out = np.zeros(points.shape[0])
    for start in range(0, points.shape[0], chunk):
        p = points[start:start + chunk]
        dx = p[:, None, 0] - x[None, :]
        dy = p[:, None, 1] - y[None, :]
        out[start:start + chunk] = (weight[None, :] * np.exp(-(dx * dx + dy * dy) / eps2[None, :])).sum(axis=1)
    return out


def blob_streamfunction(points: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: np.ndarray, eps: np.ndarray,
                        chunk: int = 1024) -> np.ndarray:
    """Stream function of the regularised blobs, ``psi = -sum gamma/(4 pi) log(r^2 + eps^2)``."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps2 = np.asarray(eps, dtype=float) ** 2
    out = np.zeros(points.shape[0])
    for start in range(0, points.shape[0], chunk):
        p = points[start:start + chunk]
        dx = p[:, None, 0] - x[None, :]
        dy = p[:, None, 1] - y[None, :]
        out[start:start + chunk] = -(gamma[None, :] * np.log(dx * dx + dy * dy + eps2[None, :])).sum(axis=1) / (4 * np.pi)
    return out


# ------------------------------------------------------------------------------------------------------- panels
def segment_integrals(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``I_j = int ds / (z - zeta_j)`` over each panel, complex, in the frame ``z = x + i y``.

    Real arc length element, so the differential ``d zeta`` is turned into ``ds`` by ``conj(dz)/|dz|``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    dxa = points[:, None, 0] - a[None, :, 0]
    dya = points[:, None, 1] - a[None, :, 1]
    dxb = points[:, None, 0] - b[None, :, 0]
    dyb = points[:, None, 1] - b[None, :, 1]
    ra = np.hypot(dxa, dya)
    rb = np.hypot(dxb, dyb)
    # angle subtended by the panel at the observation point, wrapped to (-pi, pi]
    th = np.arctan2(dya, dxa) - np.arctan2(dyb, dxb)
    th = (th + np.pi) % TWO_PI - np.pi
    dz = (b - a)[:, 0] + 1j * (b - a)[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        logterm = np.log(ra / rb)
    return (logterm + 1j * th) * (np.conj(dz) / np.abs(dz))[None, :]


def panel_flux(points: np.ndarray, a: np.ndarray, b: np.ndarray, strength: np.ndarray,
               kind: str = "vortex", exclude_diagonal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Velocity at ``points`` induced by constant-strength panels ``[a_k, b_k]``.

    ``kind="vortex"``: ``strength`` is the sheet strength ``gamma`` (circulation per unit length),
    ``kind="source"``: ``strength`` is ``sigma`` (flux per unit length). Backed by the closed-form integral
    :func:`segment_integrals`.

    ``exclude_diagonal`` drops panel ``k``'s influence on ``points[k]`` -- the collocation case, where the
    principal value is supplied by the caller (``+sigma/2`` for a source sheet, ``-gamma/2`` for a vortex sheet).
    """
    integral = segment_integrals(points, a, b)
    if exclude_diagonal:
        if integral.shape[0] != integral.shape[1]:
            raise ValueError("exclude_diagonal needs one collocation point per panel")
        np.fill_diagonal(integral, 0.0)
    strength = np.asarray(strength, dtype=float)
    if kind == "vortex":
        w = -1j * strength[None, :] * integral / TWO_PI
    elif kind == "source":
        w = strength[None, :] * integral / TWO_PI
    else:                                                        # pragma: no cover - guarded by callers
        raise ValueError(f"kind must be 'vortex' or 'source', not {kind!r}")
    return w.real.sum(axis=1), (-w.imag).sum(axis=1)


def source_sheet_system(mid: np.ndarray, a: np.ndarray, b: np.ndarray,
                        normal: np.ndarray) -> np.ndarray:
    """Normal-velocity influence matrix of unit source panels, collocated at the panel midpoints.

    The self-term of a flat source panel is the classical half jump, ``+1/2`` along the outward normal.
    """
    integral = segment_integrals(mid, a, b)
    n_panels = integral.shape[0]
    if n_panels != integral.shape[1]:
        raise ValueError("the source-sheet system needs one collocation point per panel")
    np.fill_diagonal(integral, 0.0)
    w = integral / TWO_PI
    # the normal is that of the collocation point (the row), not of the panel (the column)
    matrix = w.real * normal[:, 0][:, None] + (-w.imag) * normal[:, 1][:, None]
    matrix[np.arange(n_panels), np.arange(n_panels)] = 0.5
    return matrix


def vortex_sheet_system(mid: np.ndarray, a: np.ndarray, b: np.ndarray,
                        tangent: np.ndarray) -> np.ndarray:
    """Tangential-velocity influence matrix of unit vortex panels, collocated at the panel midpoints.

    The self term carries the half jump the fluid side sees, ``+1/2`` along the panel tangent. That is what makes a
    spinning cylinder come out with a uniform sheet equal to its surface speed
    (``tests/test_vortex_domains.py``), and it is calibrated against that exact solution rather than assumed:
    ``+1`` there gives a sheet that is off by a factor of two.
    """
    integral = segment_integrals(mid, a, b)
    n_panels = integral.shape[0]
    if n_panels != integral.shape[1]:
        raise ValueError("the vortex-sheet system needs one collocation point per panel")
    np.fill_diagonal(integral, 0.0)
    w = -1j * integral / TWO_PI                        # unit vortex sheet, complex velocity u - i v
    matrix = w.real * tangent[:, 0][:, None] + (-w.imag) * tangent[:, 1][:, None]
    matrix[np.arange(n_panels), np.arange(n_panels)] = 0.5
    return matrix


# ---------------------------------------------------------------------------------------- diffusive velocities
def vortex_diffusion_drift(x: np.ndarray, y: np.ndarray, gamma: np.ndarray, eps: np.ndarray, nu: float,
                           chunk: int = 512, same_sign: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Dynnikova's "vortex diffusive" velocity, the drift that makes up for viscosity.

    ``I1 = sum_j gamma_j exp(-|rho|/eps)``, ``I2 = (1/eps) sum_j gamma_j rho/|rho| exp(-|rho|/eps)`` and
    ``V_d = nu * I2 / I1`` -- the velocity ``-nu grad(omega)/omega`` written as a discrete kernel estimate.
    Only neighbours of the same sign contribute (as in the reference implementation): mixing signs makes the
    estimate of ``grad(omega)/omega`` meaningless where the vorticity changes sign.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps = np.asarray(eps, dtype=float)
    n = x.size
    vx = np.zeros(n)
    vy = np.zeros(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = x[None, start:stop] - x[:, None]
        dy = y[None, start:stop] - y[:, None]
        r = np.hypot(dx, dy)
        keep = (r <= EXP_CUTOFF * eps[None, start:stop]) & (r > 0.0)
        g = np.broadcast_to(gamma[:, None], r.shape)
        if same_sign:
            keep &= (np.sign(g) == np.sign(gamma[None, start:stop])) & (g != 0.0)
        r_safe = np.where(keep, r, 1.0)
        w = np.where(keep, g * np.exp(-r_safe / eps[None, start:stop]), 0.0)
        i1 = w.sum(axis=0)
        i2x = (w * dx / r_safe).sum(axis=0)
        i2y = (w * dy / r_safe).sum(axis=0)
        e = eps[start:stop]
        g_self = gamma[start:stop]
        # I1 must have the sign of the domain and not be smaller than a tenth of its circulation
        i1 = np.where((np.sign(i1) != np.sign(g_self)) | (np.abs(i1) < 0.1 * np.abs(g_self)), 0.1 * g_self, i1)
        i1 = np.where(np.abs(i1) < 1e-300, np.where(g_self == 0.0, 1e-300, g_self), i1)
        scale = nu / (e * i1)
        vx[start:stop] = scale * i2x
        vy[start:stop] = scale * i2y
    return vx, vy


def wall_diffusion_drift(x: np.ndarray, y: np.ndarray, eps: np.ndarray, seg_mid: np.ndarray,
                         seg_vector: np.ndarray, nu: float, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Dynnikova's "body diffusive" velocity: the drift produced by the wall replacing the missing half space.

    ``I3 = sum_k dS_k exp(-|rho_k|/eps)``, ``I0 = sum_k (|rho_k|/eps + 1)/|rho_k|^2 (rho_k . dS_k) exp(-|rho_k|/eps)``,
    ``W_d = nu I3 / (eps^2 (2 pi - I0))`` with ``I0`` capped at ``pi`` (the reference implementation's bound).
    ``dS_k`` is the outward normal of the panel scaled by its length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eps = np.asarray(eps, dtype=float)
    seg_mid = np.asarray(seg_mid, dtype=float)
    seg_vector = np.asarray(seg_vector, dtype=float)
    n = x.size
    vx = np.zeros(n)
    vy = np.zeros(n)
    if n == 0 or seg_mid.size == 0:
        return vx, vy
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = x[None, start:stop] - seg_mid[:, 0:1]                # (M, c)
        dy = y[None, start:stop] - seg_mid[:, 1:2]
        r = np.hypot(dx, dy)
        e = eps[None, start:stop]
        keep = (r <= EXP_CUTOFF * e) & (r > 0.0)
        r_safe = np.where(keep, r, 1.0)
        ex = np.where(keep, np.exp(-r_safe / e), 0.0)
        i3x = (seg_vector[:, 0:1] * ex).sum(axis=0)
        i3y = (seg_vector[:, 1:2] * ex).sum(axis=0)
        dot = dx * seg_vector[:, 0:1] + dy * seg_vector[:, 1:2]
        i0 = ((r_safe / e + 1.0) / (r_safe * r_safe) * dot * ex).sum(axis=0)
        i0 = np.minimum(i0, np.pi)
        denom = e * e * (TWO_PI - i0)
        scale = np.where(np.abs(denom) > 1e-14, nu / np.where(denom == 0.0, 1.0, denom), 0.0)
        vx[start:stop] = scale * i3x
        vy[start:stop] = scale * i3y
    return vx, vy


def blob_pressure_sum(points: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: np.ndarray,
                      velocity: np.ndarray, eps: np.ndarray, chunk: int = 512) -> np.ndarray:
    """``sum_i V_i . u_i(r)`` -- the Lagrangian unsteady term of the pressure, one pass over the domains.

    ``u_i`` is the velocity domain ``i`` induces at ``r`` and ``V_i`` the velocity it moves with; the sum is what
    the material derivative of the potential gives in a vorticity field (the reference implementation's
    ``p - p_inf = (V_inf^2 - V^2)/2 + sum V_i . u_i``).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    velocity = np.asarray(velocity, dtype=float).reshape(-1, 2)
    eps2 = np.asarray(eps, dtype=float) ** 2
    weight = gamma / TWO_PI
    out = np.zeros(points.shape[0])
    for start in range(0, points.shape[0], chunk):
        p = points[start:start + chunk]
        dx = p[:, None, 0] - x[None, :]
        dy = p[:, None, 1] - y[None, :]
        inv = weight[None, :] / (dx * dx + dy * dy + eps2[None, :])
        term = (-velocity[None, :, 0] * dy + velocity[None, :, 1] * dx) * inv
        out[start:start + chunk] = term.sum(axis=1)
    return out
