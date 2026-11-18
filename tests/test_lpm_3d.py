"""The 3-D families, each against a property of the physics that does not come from the family.

A field is easy to produce and hard to trust. Every check here compares the family with something outside it: a law
the planform has to obey, an energy that has to balance, a linearity that mechanics demands. The checks that the
family cannot pass are the ones worth having.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_design.aero.geometry import REFERENCE
from pinneapple_neural.lpm_families_3d import (
    N_STRIPS,
    Bracket3D,
    FinHeatSink3D,
    Wing3D,
    _beta,
    _station_fracs,
    families_3d,
)
from pinneapple_neural.lpm_pipeline import DesignSpec, build, draw, families, generate_dataset


def _wing(**over) -> DesignSpec:
    """A wing of the library's NACA 2412 section at a fixed planform, overridable."""
    geometry = {"aspect_ratio": 8.0, "taper": 0.6, "sweep_le_deg": 0.0, "dihedral_deg": 0.0,
                "twist_deg": 0.0, "incidence_deg": 0.0}
    geometry.update(over.pop("geometry", {}))
    geometry.update({f"cst_{i}": float(v) for i, v in enumerate(over.pop("section", REFERENCE["NACA 2412"]))})
    return DesignSpec(geometry, {"alpha_deg": 4.0, "mach": 0.2, **over.pop("operating", {})})


def _span_integral(family, spec, points: np.ndarray, normals: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """Normal force per unit span of every strip of the ring, in units of the dynamic pressure times the chord."""
    per = points.shape[0] // N_STRIPS
    weights = family.surface_measure(spec)
    out = []
    for k in range(N_STRIPS):
        sl = slice(k * per, (k + 1) * per)
        out.append(-float(np.sum(cp[sl] * normals[sl][:, 2] * weights[sl])))   # the force is -integral(p n ds)
    return np.asarray(out)


# ----------------------------------------------------------------------------------------------------------- the wing

def test_the_wing_section_integral_carries_the_load_the_lattice_asked_for():
    """The composed Cp must integrate, section by section, to the strip cl the vortex lattice produced.

    This is the coupling: the lattice decides how much lift a station carries, the section solve decides how it is
    distributed, and the pressure has to add up to the first.
    """
    from pinneapple_design.aero import vlm

    family = Wing3D()
    spec = _wing()
    points, normals = family.points_and_normals(spec)
    cp = family.field(spec).ravel()
    af = family._airframe(spec)
    beta = _beta(spec.operating["mach"])
    cl_lattice = vlm.solve(af, x_ref=0.25 * af.root_chord, nc=4, ns_wing=N_STRIPS,
                           tail=False).strip_cl(math.radians(spec.operating["alpha_deg"]), 0.0)
    cl_lattice = cl_lattice[N_STRIPS:] / beta                          # the right wing, Prandtl-Glauert
    chord = np.array([family._station(af, f)[1] for f in _station_fracs()])
    cl_field = _span_integral(family, spec, points, normals, cp) / chord
    np.testing.assert_allclose(cl_field, cl_lattice, atol=2e-3)


def test_the_wing_obeys_prandtl_lifting_line():
    """CL x AR/(AR + 2) is the lift of the same section at infinite aspect ratio: it must stop depending on AR.

    The strip widths are the lattice's own, not an average: the planform is cosine-spaced, so integrating the field
    with equal widths would put the error of the spacing into the check.
    """
    from pinneapple_design.aero import vlm

    family = Wing3D()
    values = []
    for aspect_ratio in (200.0, 2000.0):
        spec = _wing(geometry={"aspect_ratio": aspect_ratio, "taper": 1.0})
        points, normals = family.points_and_normals(spec)
        cp = family.field(spec).ravel()
        af = family._airframe(spec)
        dy = vlm.solve(af, x_ref=0.25 * af.root_chord, nc=4, ns_wing=N_STRIPS, tail=False).L.dy[N_STRIPS:]
        chord = np.array([family._station(af, f)[1] for f in _station_fracs()])
        cl = _span_integral(family, spec, points, normals, cp) / chord
        CL = float(np.sum(cl * chord * dy) / af.wing_area)
        values.append(CL * (aspect_ratio + 2.0) / aspect_ratio)
    assert abs(values[1] / values[0] - 1.0) < 0.03, values


def test_the_wing_is_unloaded_when_the_section_is_symmetric_at_zero_incidence():
    """A symmetric section at zero incidence carries no loading; only the leading-edge stagnation point is at Cp 1."""
    family = Wing3D()
    spec = _wing(section=REFERENCE["NACA 0012"], operating={"alpha_deg": 0.0})
    points, normals = family.points_and_normals(spec)
    cp = family.field(spec).ravel()
    per = points.shape[0] // N_STRIPS
    assert abs(_span_integral(family, spec, points, normals, cp)[0]) < 1e-6, "a symmetric section carried a load"
    # what is left is the stagnation point at the leading edge, where Cp is 1 by definition, and nothing else
    assert 0.9 < float(cp.max()) < 1.05, "the leading-edge stagnation point is missing"
    assert float(cp.min()) < -0.2, "the flow over a symmetric section must still accelerate somewhere"
    # symmetry: the upper and lower halves carry the same pressure, point for point
    per = points.shape[0] // N_STRIPS
    half = per // 2
    np.testing.assert_allclose(cp[:per][:half][::-1], cp[:per][-half:], atol=5e-3)


def test_the_wing_pressure_follows_prandtl_glauert():
    """Cp at Mach is Cp at zero Mach divided by beta, exactly — the rule the reference is built on.

    Prandtl-Glauert enters once, on the pressure. The same rule then raises the lift by 1/beta, which is why the
    load is not divided as well.
    """
    family = Wing3D()
    spec = _wing(operating={"alpha_deg": 4.0, "mach": 0.0})
    cp0 = family.field(spec).ravel()
    for mach in (0.4, 0.6, 0.75):
        beta = _beta(mach)
        cp_m = family.field(_wing(operating={"alpha_deg": 4.0, "mach": mach})).ravel()
        np.testing.assert_allclose(cp_m, cp0 / beta, atol=1e-9)
        assert abs(cp_m).max() > abs(cp0).max(), "compressibility did not amplify the pressure"


def test_the_wing_surface_is_closed_and_its_normals_point_out():
    family = Wing3D()
    spec = _wing()
    points, normals = family.points_and_normals(spec)
    assert points.shape == (family.n_points, 3) and normals.shape == points.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    per = points.shape[0] // N_STRIPS
    for k in range(N_STRIPS):
        p, n = points[k * per:(k + 1) * per], normals[k * per:(k + 1) * per]
        assert np.linalg.norm(np.roll(p, -1, axis=0)[-1] - p[-1]) < 0.2, "the ring did not close"
        assert float(np.mean(np.sum(n * (p - p.mean(0)), axis=1))) > 0.0, "normals point inward"


# ----------------------------------------------------------------------------------------------------- the solids

def test_the_heat_sink_conserves_energy_and_cools_with_the_coefficient():
    from pinneapple_simulation import immersed as im

    family = FinHeatSink3D()
    spec = DesignSpec({"fin_height_mm": 30.0, "fin_thickness_mm": 2.0, "length_mm": 60.0,
                       "base_thickness_mm": 5.0}, {"power_w": 40.0, "h_w_m2k": 60.0, "t_inf_c": 25.0})
    base, fins = family._boxes(spec)
    geo = im.box(base[0], base[1])
    for lo, hi in fins:
        geo = im.union(geo, im.box(lo, hi))
    disc = im.Discretization(geo, family._h(spec), order=1)
    sol = im.solve_heat(disc, 200.0, 0.0, [
        im.HeatInput(40.0, where=lambda X, n: X[:, 2] < 1e-9),
        im.Convection(60.0, 25.0, where=lambda X, n: X[:, 2] > 1e-9)])
    assert abs(sol.balance["imbalance"]) < 1e-6, sol.balance
    points, _ = family.points_and_normals(spec)
    hot = float(family.field(spec).max())
    # the same sink with twice the convection coefficient must run cooler at the same power
    better = DesignSpec(spec.geometry, {**spec.operating, "h_w_m2k": 120.0})
    assert float(family.field(better).max()) < hot - 1.0
    assert 25.0 < hot < 200.0
    assert points.shape == (family.n_points, 3)


def test_the_bracket_is_linear_in_the_load_and_converges_with_the_mesh():
    family = Bracket3D()
    geometry = {"arm_mm": 80.0, "width_mm": 40.0, "thickness_mm": 10.0, "upright_mm": 60.0}
    light = DesignSpec(geometry, {"load_n": 100.0})
    heavy = DesignSpec(geometry, {"load_n": 200.0})
    fields = np.stack([family.field(s) for s in (light, heavy)])
    ratio = fields[1] / np.maximum(fields[0], 1e-30)
    assert abs(float(np.median(ratio)) - 2.0) < 0.05, "the reference is linear: twice the load is twice the field"
    # a coarser mesh must move the answer by much less than the answer itself
    coarse = family.field(light)
    saved = family._mesh_size
    try:
        family._mesh_size = lambda spec: saved(spec) * 1.25
        fine = family.field(light)
    finally:
        family._mesh_size = saved
    # the displacement converges; the stress does not, and cannot: von Mises is singular where the part is clamped,
    # so only the displacement is asked to converge here and the card says the same
    du = np.linalg.norm(fine[:, 0] - coarse[:, 0]) / np.linalg.norm(coarse[:, 0])
    assert du < 0.05, f"the mesh changes the displacement by {du:.0%}"
    assert coarse[:, 0].max() > 0.0 and coarse[:, 1].max() > 0.0


# ------------------------------------------------------------------------------------------------ the pipeline

def test_every_3d_family_is_registered_and_keeps_out_of_the_2d_registry():
    three = families_3d()
    assert set(three) == {"pp_wing3d", "pp_fin3d", "pp_bracket3d"}
    assert not set(three) & set(families()), "the 3-D families must not enter the 2-D registry"
    for family in three.values():
        described = family.describe()
        assert described["dim"] == 3
        assert described["reference"]["assumptions"], "a 3-D reference without assumptions is an overclaim"


@pytest.mark.parametrize("name", ["pp_wing3d", "pp_fin3d", "pp_bracket3d"])
def test_a_3d_family_round_trips_through_a_dataset(tmp_path, name):
    family = families_3d()[name]
    specs = draw(family, n_geometries=2, conditions_per_geometry=1, seed=0)
    design = build(family, specs[0])
    assert design.points.shape[1] == 3 and design.query.shape == (family.n_points, 3)
    assert design.target.shape == (family.n_points, family.n_fields)
    out = generate_dataset(family, tmp_path / name, n_geometries=2, conditions_per_geometry=1, seed=0)
    assert out["n_designs"] == 2
    assert (tmp_path / name / "manifest.json").is_file()
