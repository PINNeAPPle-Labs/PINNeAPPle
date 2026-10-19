"""Electromagnetic presets: each has a closed-form solution (``pp.solve(problem, "analytic")``, registered from
:mod:`pinneapple_physics.closed_form.electromagnetics`), so any method can be scored against the exact field.

``em_two_dielectric_capacitor``  Gauss's law in integral form across a dielectric interface (the differential
                                 kind cannot see the jump of eps; the integral kind imposes continuity of D.n)
``em_two_dielectric_capacitor_differential``
                                 the same capacitor with the differential kind, kept to show that failure
``em_wire_az``                   magnetostatics of a round conductor, -div(nu grad Az) = Jz
``em_tm_cavity``                 time-domain TMz standing wave in a PEC square (Maxwell's curl equations)
``em_tm_cavity_integral``        the same with Ampere-Maxwell and Faraday in integral form
``em_skin_effect``               time-harmonic eddy currents diffusing into a conductor (skin effect), 1-D
``em_cylinder_scattering``       frequency-domain TMz scattering of a plane wave by a dielectric cylinder

Non-dimensional units (``eps0 = mu0 = 1`` scaled out, lengths in units of the domain); the SI solvers of
``pinneapple_simulation.electromagnetics`` solve the dimensional versions.
"""
from __future__ import annotations

import math

import numpy as np

from ..conditions import DirichletBC, InitialCondition
from ..scales import ScaleSpec
from ..spec import PDETermSpec, ProblemSpec
from .registry import register_preset

_TOL = 1e-6


def _face(i: int, v: float):
    return lambda X, ctx=None, _i=i, _v=v: np.abs(X[:, _i] - _v) < _TOL


def _faces(bounds: dict, coords: tuple[str, ...], which: tuple[str, ...] | None = None):
    sel = []
    for k, c in enumerate(coords):
        if c == "t" or (which is not None and c not in which):
            continue
        lo, hi = bounds[c]
        sel += [_face(k, lo), _face(k, hi)]
    return lambda X, ctx=None: np.any([s(X) for s in sel], axis=0)


def _capacitor(kind: str, *, V: float, eps1: float, eps2: float, d1: float, h: float) -> ProblemSpec:
    from pinneapple_physics.closed_form.electromagnetics import two_dielectric_capacitor

    coords = ("x", "y")
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    eps_fn = lambda X, ctx=None, _e1=eps1, _e2=eps2, _d=d1: np.where(X[:, 1] <= _d, _e1, _e2)  # noqa: E731
    exact = lambda X, ctx=None: two_dielectric_capacitor(X[:, 1], V, 1.0, d1, eps1, eps2)[:, None]  # noqa: E731
    params = {"V": V, "d": 1.0, "d1": d1, "eps1": eps1, "eps2": eps2, "eps_fn": eps_fn}
    if kind == "electrostatics_integral":
        params.update({"h": h, "quad": 4})
    pde = PDETermSpec(kind=kind, fields=("phi",), coords=coords, params=params,
                      meta={"analytic": "em_two_dielectric_capacitor", "physics": "electrostatics"})
    bc = DirichletBC(name="plates_and_sides", fields=("phi",), selector_type="callable",
                     selector=_faces(bounds, coords), value_fn=exact, weight=10.0)
    return ProblemSpec(name=f"em_two_dielectric_capacitor{'' if kind.endswith('integral') else '_differential'}",
                       dim=2, coords=coords, fields=("phi",), pde=pde, conditions=(bc,),
                       sample_defaults={"n_col": 2048, "n_bc": 512, "n_ic": 0, "n_data": 0},
                       scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0), field_ranges={"phi": (0.0, V)},
                       references=("Griffiths, Introduction to Electrodynamics, 4th ed., ex. 4.7",),
                       domain_bounds=bounds)


@register_preset("em_two_dielectric_capacitor")
def em_two_dielectric_capacitor(V: float = 1.0, eps1: float = 1.0, eps2: float = 4.0, d1: float = 0.5,
                                h: float = 0.02) -> ProblemSpec:
    return _capacitor("electrostatics_integral", V=V, eps1=eps1, eps2=eps2, d1=d1, h=h)


@register_preset("em_two_dielectric_capacitor_differential")
def em_two_dielectric_capacitor_differential(V: float = 1.0, eps1: float = 1.0, eps2: float = 4.0,
                                             d1: float = 0.5) -> ProblemSpec:
    return _capacitor("electrostatics", V=V, eps1=eps1, eps2=eps2, d1=d1, h=0.0)


@register_preset("em_wire_az")
def em_wire_az(a: float = 0.25, J: float = 1.0, nu: float = 1.0) -> ProblemSpec:
    from pinneapple_physics.closed_form.electromagnetics import wire_az

    coords = ("x", "y")
    bounds = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    J_fn = lambda X, ctx=None, _a=a, _J=J: np.where(np.hypot(X[:, 0], X[:, 1]) <= _a, _J, 0.0)  # noqa: E731
    pde = PDETermSpec(kind="magnetostatics_az", fields=("Az",), coords=coords,
                      params={"a": a, "J": J, "nu": nu, "J_fn": J_fn},
                      meta={"analytic": "em_wire_az", "physics": "magnetostatics"})
    bc = DirichletBC(name="far_boundary", fields=("Az",), selector_type="callable", selector=_faces(bounds, coords),
                     value_fn=lambda X, ctx=None: wire_az(X[:, 0], X[:, 1], a, J, nu)[:, None], weight=10.0)
    return ProblemSpec(name="em_wire_az", dim=2, coords=coords, fields=("Az",), pde=pde, conditions=(bc,),
                       sample_defaults={"n_col": 2048, "n_bc": 512, "n_ic": 0, "n_data": 0},
                       scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0), field_ranges={"Az": (-0.1, 0.02)},
                       references=("Jackson, Classical Electrodynamics, 3rd ed., sec. 5.3 (Ampere's law)",),
                       domain_bounds=bounds)


def _cavity(kind: str, m: int, n: int, periods: float, h: float) -> ProblemSpec:
    from pinneapple_physics.closed_form.electromagnetics import tm_cavity_mode

    coords = ("x", "y", "t")
    w = math.pi * math.hypot(m, n)
    T = periods * 2.0 * math.pi / w
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, T)}
    params = {"eps": 1.0, "mu": 1.0, "sigma": 0.0, "m": m, "n": n}
    if kind == "maxwell_tm_integral":
        params.update({"h": h, "quad": 4})
    pde = PDETermSpec(kind=kind, fields=("Ez", "Hx", "Hy"), coords=coords, params=params,
                      meta={"analytic": "em_tm_cavity_mode", "physics": "electrodynamics"})
    pec = DirichletBC(name="pec_walls", fields=("Ez",), selector_type="callable", selector=_faces(bounds, coords),
                      value_fn=lambda X, ctx=None: np.zeros((len(X), 1)), weight=10.0)

    def ic(X, ctx=None):
        Ez, Hx, Hy = tm_cavity_mode(X[:, 0], X[:, 1], np.zeros(len(X)), m, n, 1.0, 1.0)
        return np.stack([Ez, Hx, Hy], axis=1)

    init = InitialCondition(name="standing_wave_t0", fields=("Ez", "Hx", "Hy"), value_fn=ic, weight=10.0)
    return ProblemSpec(name="em_tm_cavity" + ("_integral" if kind.endswith("integral") else ""), dim=2,
                       coords=coords, fields=("Ez", "Hx", "Hy"), pde=pde, conditions=(pec, init),
                       sample_defaults={"n_col": 4096, "n_bc": 1024, "n_ic": 1024, "n_data": 0},
                       scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
                       field_ranges={"Ez": (-1.0, 1.0), "Hx": (-1.0, 1.0), "Hy": (-1.0, 1.0)},
                       references=("Pozar, Microwave Engineering, 4th ed., sec. 6.3",), domain_bounds=bounds)


@register_preset("em_tm_cavity")
def em_tm_cavity(m: int = 1, n: int = 1, periods: float = 0.5) -> ProblemSpec:
    return _cavity("maxwell_tm", m, n, periods, 0.0)


@register_preset("em_tm_cavity_integral")
def em_tm_cavity_integral(m: int = 1, n: int = 1, periods: float = 0.5, h: float = 0.02) -> ProblemSpec:
    return _cavity("maxwell_tm_integral", m, n, periods, h)


@register_preset("em_skin_effect")
def em_skin_effect(delta: float = 0.2, omega: float = 1.0, nu: float = 1.0, length: float = 1.0) -> ProblemSpec:
    from pinneapple_physics.closed_form.electromagnetics import skin_effect_slab

    sigma = 2.0 * nu / (omega * delta * delta)
    coords = ("x",)
    bounds = {"x": (0.0, length)}

    def exact(X, ctx=None):
        A = skin_effect_slab(X[:, 0], delta)
        return np.stack([A.real, A.imag], axis=1)

    pde = PDETermSpec(kind="eddy_current_az", fields=("Ar", "Ai"), coords=coords,
                      params={"omega": omega, "sigma": sigma, "nu": nu},
                      meta={"analytic": "em_skin_effect_slab", "physics": "magneto-quasistatics"})
    bc = DirichletBC(name="surface_and_depth", fields=("Ar", "Ai"), selector_type="callable",
                     selector=_faces(bounds, coords), value_fn=exact, weight=10.0)
    return ProblemSpec(name="em_skin_effect", dim=1, coords=coords, fields=("Ar", "Ai"), pde=pde, conditions=(bc,),
                       sample_defaults={"n_col": 512, "n_bc": 64, "n_ic": 0, "n_data": 0},
                       scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0), field_ranges={"Ar": (-0.1, 1.0), "Ai": (-0.4, 0.1)},
                       references=("Jackson, Classical Electrodynamics, 3rd ed., sec. 5.18",), domain_bounds=bounds)


@register_preset("em_cylinder_scattering")
def em_cylinder_scattering(k0: float = 2.0 * math.pi, radius: float = 0.25, eps_cyl: float = 4.0,
                           eps_cyl_i: float = 0.0, half_width: float = 1.0) -> ProblemSpec:
    from pinneapple_physics.closed_form.electromagnetics import cylinder_tm_total

    coords = ("x", "y")
    bounds = {"x": (-half_width, half_width), "y": (-half_width, half_width)}
    eps_c = complex(eps_cyl, eps_cyl_i)
    eps_fn = lambda X, ctx=None: np.where(np.hypot(X[:, 0], X[:, 1]) <= radius, eps_c, 1.0 + 0j)  # noqa: E731

    def exact(X, ctx=None):
        E = cylinder_tm_total(X[:, 0], X[:, 1], k0, radius, eps_c)
        return np.stack([E.real, E.imag], axis=1)

    pde = PDETermSpec(kind="maxwell_tm_frequency", fields=("Er", "Ei"), coords=coords,
                      params={"k0": k0, "radius": radius, "eps_cyl": eps_cyl, "eps_cyl_i": eps_cyl_i,
                              "eps_fn": eps_fn},
                      meta={"analytic": "em_cylinder_scattering_tm", "physics": "electrodynamics"})
    bc = DirichletBC(name="exact_total_field", fields=("Er", "Ei"), selector_type="callable",
                     selector=_faces(bounds, coords), value_fn=exact, weight=10.0)
    return ProblemSpec(name="em_cylinder_scattering", dim=2, coords=coords, fields=("Er", "Ei"), pde=pde,
                       conditions=(bc,), sample_defaults={"n_col": 4096, "n_bc": 1024, "n_ic": 0, "n_data": 0},
                       scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0), field_ranges={"Er": (-2.5, 2.5), "Ei": (-2.5, 2.5)},
                       references=("Harrington, Time-Harmonic Electromagnetic Fields (1961), sec. 5-9",),
                       domain_bounds=bounds)
