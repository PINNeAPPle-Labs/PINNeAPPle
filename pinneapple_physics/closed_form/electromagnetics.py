"""Closed-form solutions of the electromagnetic presets (``pinneapple_physics.pde_environment.presets.
electromagnetics``), registered as ``pp.solve(problem, "analytic")`` solutions in :mod:`pinneapple_physics.solving`.

Each entry of :data:`ANALYTIC` is ``(name, matcher(problem) -> bool, factory(problem) -> fn(X) -> (N, n_fields))``.
A preset names its solution in ``pde.meta["analytic"]``; the factory reads the parameters from the problem, so an
override (``from_preset(..., eps2=8.0)``) gets the matching exact solution.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["ANALYTIC", "two_dielectric_capacitor", "wire_az", "tm_cavity_mode", "skin_effect_slab",
           "cylinder_tm_total"]


def two_dielectric_capacitor(y, V: float, d: float, d1: float, eps1: float, eps2: float):
    """Potential across a parallel-plate capacitor (``phi = 0`` at ``y = 0``, ``V`` at ``y = d``) filled with
    ``eps1`` for ``y < d1`` and ``eps2`` above: ``D`` is uniform, so ``E_i = D / eps_i`` with
    ``D = V / (d1 / eps1 + (d - d1) / eps2)`` (Griffiths, *Introduction to Electrodynamics*, 4th ed., ex. 4.7)."""
    y = np.asarray(y, float)
    D = V / (d1 / eps1 + (d - d1) / eps2)
    return np.where(y <= d1, D * y / eps1, D * d1 / eps1 + D * (y - d1) / eps2)


def wire_az(x, y, a: float, J: float, nu: float = 1.0):
    """``A_z`` of a round conductor (radius ``a``, uniform current density ``J``) in a medium of reluctivity ``nu``,
    gauge ``A = 0`` at ``r = a``: ``-J r^2 / (4 nu) + J a^2 / (4 nu)`` inside, ``-(J a^2 / (2 nu)) ln(r / a)``
    outside (Ampere's law)."""
    r = np.hypot(np.asarray(x, float), np.asarray(y, float))
    inside = J * (a * a - r * r) / (4.0 * nu)
    outside = -J * a * a / (2.0 * nu) * np.log(np.maximum(r, 1e-300) / a)
    return np.where(r <= a, inside, outside)


def tm_cavity_mode(x, y, t, m: int, n: int, Lx: float, Ly: float, eps: float = 1.0, mu: float = 1.0):
    """The TM_mn standing wave of a PEC rectangle: ``Ez = sin(kx x) sin(ky y) cos(w t)``,
    ``Hx = -(ky / (mu w)) sin(kx x) cos(ky y) sin(w t)``, ``Hy = (kx / (mu w)) cos(kx x) sin(ky y) sin(w t)`` with
    ``w = sqrt((kx^2 + ky^2) / (eps mu))`` (Pozar sec. 6.3)."""
    kx, ky = m * math.pi / Lx, n * math.pi / Ly
    w = math.sqrt((kx * kx + ky * ky) / (eps * mu))
    Ez = np.sin(kx * x) * np.sin(ky * y) * np.cos(w * t)
    Hx = -ky / (mu * w) * np.sin(kx * x) * np.cos(ky * y) * np.sin(w * t)
    Hy = kx / (mu * w) * np.cos(kx * x) * np.sin(ky * y) * np.sin(w * t)
    return Ez, Hx, Hy


def skin_effect_slab(x, delta: float, A0: float = 1.0):
    """Field diffusing into a conducting half-space ``x > 0`` from a surface value ``A0`` (phasor,
    ``exp(+j omega t)``): ``A = A0 exp(-(1 + j) x / delta)``, ``delta = sqrt(2 nu / (omega sigma))``
    (Jackson sec. 5.18, Griffiths sec. 9.4.1)."""
    return A0 * np.exp(-(1.0 + 1.0j) * np.asarray(x, float) / delta)


def cylinder_tm_total(x, y, k0: float, radius: float, eps_r: complex):
    """Total ``Ez`` of a unit plane wave ``exp(-j k0 x)`` on a dielectric cylinder (series of cylinder functions,
    :func:`pinneapple_simulation.electromagnetics.waves.cylinder_scattering_tm`)."""
    from pinneapple_simulation.electromagnetics.waves import cylinder_scattering_tm

    return cylinder_scattering_tm(x, y, k0, radius, eps_r)


# ----------------------------------------------------------------------------------------------- registry
def _meta(p) -> dict:
    return dict(getattr(p.pde, "meta", None) or {}) if p.pde is not None else {}


def _params(p) -> dict:
    out = dict(p.pde.params)
    out.update({k: v.value for k, v in p.parameters.items() if v.value is not None})
    return out


def _is(name):
    return lambda p: _meta(p).get("analytic") == name


def _capacitor(p):
    q = _params(p)
    return lambda X: two_dielectric_capacitor(X[:, 1], q["V"], q["d"], q["d1"], q["eps1"], q["eps2"])[:, None]


def _wire(p):
    q = _params(p)
    return lambda X: wire_az(X[:, 0], X[:, 1], q["a"], q["J"], q.get("nu", 1.0))[:, None]


def _cavity(p):
    q = _params(p)
    b = p.domain_bounds

    def fn(X):
        Ez, Hx, Hy = tm_cavity_mode(X[:, 0], X[:, 1], X[:, 2], int(q["m"]), int(q["n"]), b["x"][1], b["y"][1],
                                    q.get("eps", 1.0), q.get("mu", 1.0))
        return np.stack([Ez, Hx, Hy], axis=1)
    return fn


def _skin(p):
    q = _params(p)

    def fn(X):
        A = skin_effect_slab(X[:, 0], math.sqrt(2.0 * q.get("nu", 1.0) / (q["omega"] * q["sigma"])))
        return np.stack([A.real, A.imag], axis=1)
    return fn


def _cylinder(p):
    q = _params(p)

    def fn(X):
        E = cylinder_tm_total(X[:, 0], X[:, 1], q["k0"], q["radius"], complex(q["eps_cyl"], q.get("eps_cyl_i", 0.0)))
        return np.stack([E.real, E.imag], axis=1)
    return fn


ANALYTIC = [
    ("em_two_dielectric_capacitor", _is("em_two_dielectric_capacitor"), _capacitor),
    ("em_wire_az", _is("em_wire_az"), _wire),
    ("em_tm_cavity_mode", _is("em_tm_cavity_mode"), _cavity),
    ("em_skin_effect_slab", _is("em_skin_effect_slab"), _skin),
    ("em_cylinder_scattering_tm", _is("em_cylinder_scattering_tm"), _cylinder),
]
