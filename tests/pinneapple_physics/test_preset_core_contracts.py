"""Core coverage for the PDE presets (#1129, toward 1.0).

Two kinds of test:
* every preset factory below is called by name and must return a well-formed ``ProblemSpec`` (dimension matches the
  coordinates, fields declared, bounds ordered and named after coordinates, conditions present);
* the closed-form helpers are checked against what they claim: the ODE they solve, conservation laws, boundary
  conditions, known constants.
"""
import math

import numpy as np
import pytest

from pinneapple_physics.pde_environment.presets import _curve_geometry as CG
from pinneapple_physics.pde_environment.presets import (
    academics,
    astrophysics,
    cfd,
    electromagnetics,
    engineering,
    industry,
    multidisciplinary,
    registry,
    solid_mechanics,
    structural,
    tag_geometry,
    terramechanics,
    turbomachinery,
)
from pinneapple_physics.pde_environment.spec import ProblemSpec


def _contract(spec):
    assert isinstance(spec, ProblemSpec)
    spatial = len(spec.coords) - sum(c in ("t", "tau") for c in spec.coords)   # dim counts space, not time
    assert spec.dim == spatial and len(spec.coords) >= 1 and len(spec.fields) >= 1, (spec.name, spec.dim, spec.coords)
    for c, (lo, hi) in spec.domain_bounds.items():
        assert c in spec.coords and float(lo) < float(hi), (spec.name, c, lo, hi)
    assert spec.pde is not None and spec.name
    return spec


def test_academic_cfd_and_em_factories():
    _contract(academics.poisson_2d_default())
    _contract(cfd.channel_flow_3d())
    _contract(cfd.lid_driven_cavity_3d())
    _contract(cfd.ns_incompressible_2d_default())
    _contract(cfd.ns_incompressible_3d_default())
    _contract(cfd.pipe_flow_3d())
    _contract(electromagnetics.em_cylinder_scattering())
    _contract(electromagnetics.em_skin_effect())
    _contract(electromagnetics.em_tm_cavity())
    _contract(electromagnetics.em_tm_cavity_integral())
    _contract(electromagnetics.em_two_dielectric_capacitor())
    _contract(electromagnetics.em_two_dielectric_capacitor_differential())
    _contract(electromagnetics.em_wire_az())
    _contract(astrophysics.sod_shock_tube_astro())
    assert cfd.channel_flow_3d().dim == 3 and cfd.ns_incompressible_2d_default().dim == 2


def test_engineering_factories():
    for spec in (engineering.aircraft_wing_aerodynamics(), engineering.aircraft_wing_structural(),
                 engineering.car_brake_thermal(), engineering.car_external_aero(),
                 engineering.car_suspension_fatigue(), engineering.datacenter_airflow_2d(),
                 engineering.datacenter_cfd_3d(), engineering.datacenter_server_thermal(),
                 engineering.fan_cooler_cfd(), engineering.furnace_combustion_zone(),
                 engineering.industrial_furnace_thermal(), engineering.pcb_thermal(), engineering.refractory_lining(),
                 engineering.rocket_nozzle_cfd(), engineering.rocket_structural()):
        _contract(spec)


def test_industry_multidisciplinary_and_mechanics_factories():
    for spec in (industry.darcy_pressure_only_3d_default(), industry.helmholtz_acoustics_3d_default(),
                 industry.linear_elasticity_3d_default(), industry.steady_heat_conduction_3d_default(),
                 industry.transient_heat_3d_default(), industry.wave_ultrasound_3d_default(),
                 multidisciplinary.black_scholes_1d(), multidisciplinary.climate_atmosphere_2d(),
                 multidisciplinary.climate_ocean_gyre(), multidisciplinary.crystal_phonon(),
                 multidisciplinary.drug_diffusion_tissue(), multidisciplinary.heston_pde_2d(),
                 multidisciplinary.material_fracture_2d(), multidisciplinary.opinion_dynamics_2d(),
                 multidisciplinary.pk_two_compartment(), multidisciplinary.sir_epidemic(),
                 solid_mechanics.axisymmetric_linear_elasticity_2d_default(),
                 solid_mechanics.thick_walled_cylinder_lame_default(),
                 solid_mechanics.threaded_coupling_tc50_box_default(),
                 solid_mechanics.threaded_coupling_tc50_pin_default(),
                 solid_mechanics.threaded_coupling_tc50_rotating_default(),
                 structural.linear_elasticity_3d(), structural.plane_strain_2d_default(),
                 structural.plane_stress_2d_default(), structural.rotary_coupling_torsion_default(),
                 structural.thermoelasticity_2d_default(), structural.von_mises_2d_default(),
                 terramechanics.bekker_wong_surrogate_2d(), turbomachinery.axial_compressor_cascade_2d(),
                 turbomachinery.axial_compressor_stage_3d()):
        _contract(spec)
    assert terramechanics.TerramechanicsResiduals() is not None


def test_register_preset_adds_a_factory_to_the_registry():
    name = "_test_core_contract_preset"

    @registry.register_preset(name)
    def _factory(scale=1.0):
        return academics.poisson_2d_default()

    try:
        assert name in registry.list_presets() and _contract(registry.get_preset(name)).dim == 2
    finally:
        registry._REGISTRY.pop(name, None)


def test_tag_geometry_helpers_on_a_tagged_preset():
    torch = pytest.importorskip("torch")
    spec = registry.get_preset("plane_stress_2d")
    assert tag_geometry.preset_name(spec) == "plane_stress_2d"
    assert tag_geometry.has_default_geometry(spec)
    kw = tag_geometry.default_geometry(spec, n_bc_per_face=40, seed=0)
    assert {"x_bc", "n_bc"} <= set(kw) and len(kw["x_bc"]) > 0
    assert isinstance(tag_geometry.default_geometry_report(spec), str)
    batch = tag_geometry.build_tag_batch("plane_stress_2d", spec, n_col=64, n_bc_per_face=16, seed=0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(len(spec.coords), 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, len(spec.fields)))
    losses = tag_geometry.per_tag_losses(spec, model, batch)
    assert losses and all(math.isfinite(v) and v >= 0 for v in losses.values())


# ------------------------------------------------------------------------------------------- closed forms
def test_lane_emden_exact_solutions_satisfy_the_equation():
    """theta'' + (2/xi) theta' + theta^n = 0, theta(0) = 1, for the three analytic cases n = 0, 1, 5."""
    xi = np.linspace(0.3, 2.5, 400)
    h = xi[1] - xi[0]
    for n in (0, 1, 5):
        th = astrophysics.lane_emden_exact_theta(xi, n)
        d1 = np.gradient(th, h)
        d2 = np.gradient(d1, h)
        res = (d2 + 2 * d1 / xi + th ** n)[5:-5]
        assert np.max(np.abs(res)) < 1e-3, n
        assert abs(float(astrophysics.lane_emden_exact_theta(np.array([1e-8]), n)[0]) - 1.0) < 1e-6


def test_kepler_exact_state_conserves_energy_and_angular_momentum():
    mu, a, e = 1.0, 2.0, 0.4
    T = 2 * math.pi * math.sqrt(a ** 3 / mu)
    t = np.linspace(0.0, T, 200)
    s = astrophysics.kepler_exact_state(t, mu, a, e)
    r = np.hypot(s["x"], s["y"])
    energy = 0.5 * (s["vx"] ** 2 + s["vy"] ** 2) - mu / r
    h = s["x"] * s["vy"] - s["y"] * s["vx"]
    assert np.allclose(energy, -mu / (2 * a), rtol=1e-9)                      # vis-viva
    assert np.allclose(np.abs(h), math.sqrt(mu * a * (1 - e ** 2)), rtol=1e-9)
    assert abs(r[0] - a * (1 - e)) < 1e-12 and abs(r[-1] - a * (1 - e)) < 1e-9   # perigee, one period later


def test_light_deflection_and_the_first_order_orbit():
    sun = astrophysics.schwarzschild_deflection_angle_weak_field(1.32712440018e20, 299792458.0, 6.957e8)
    assert abs(math.degrees(sun) * 3600 - 1.7512) < 2e-3                       # Eddington 1919: 1.75 arcsec
    m, b = 1.0, 1e3
    phi = np.linspace(-1.2, 1.2, 2001)
    u = astrophysics.schwarzschild_light_bending_u_exact(phi, m, b)
    d2 = np.gradient(np.gradient(u, phi), phi)
    res = (d2 + u - 3 * m * u ** 2)[10:-10]
    assert np.max(np.abs(res)) < 20 * m ** 2 / b ** 3                          # an O(m^2) residual, as documented


def test_collinear_lagrange_point_zeroes_the_effective_force():
    mu = 0.012150585                                                           # Earth-Moon mass ratio
    x = astrophysics.cr3bp_collinear_lagrange_point(mu, 0.8)
    f = x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3
    assert abs(f) < 1e-12 and 0.83 < x < 0.84                                  # L1 between Earth and Moon


def test_lame_analytical_meets_its_boundary_conditions_and_equilibrium():
    a, b, pa, pb = 20.0, 60.0, 1e8, 2e7
    r = np.linspace(a, b, 2001)
    s = solid_mechanics.lame_analytical(r, a, b, pa, pb, 2.1e11, 0.3)
    assert abs(s["sigma_rr"][0] + pa) < 1e-6 * pa and abs(s["sigma_rr"][-1] + pb) < 1e-6 * pa
    eq = np.gradient(s["sigma_rr"], r) + (s["sigma_rr"] - s["sigma_tt"]) / r
    assert np.max(np.abs(eq[5:-5])) < 1e-4 * pa / a


# ------------------------------------------------------------------------------------------- geometry helpers
def test_polygon_helpers_on_a_unit_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    inside = CG.polygon_contains(sq, np.array([[0.5, 0.5], [1.5, 0.5], [0.2, 0.9], [-0.1, 0.1]]))
    assert inside.tolist() == [True, False, True, False]
    pts = CG.polygon_perimeter_sample(sq, 400, np.random.default_rng(0))
    d = np.minimum.reduce([pts[:, 0], 1 - pts[:, 0], pts[:, 1], 1 - pts[:, 1]])
    assert pts.shape == (400, 2) and np.max(np.abs(d)) < 1e-12               # every sample on an edge


def test_naca_area_ahmed_outline_and_cascade_blade():
    poly = CG.naca4_symmetric_polygon(1.0, 0.12, n=400)
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert abs(area / (0.6851 * 0.12) - 1) < 0.01                              # NACA 4-digit thickness: 0.685 t c^2
    ahmed = CG.ahmed_body_polygon(1.044, 0.288)
    assert np.allclose(ahmed.min(0)[0], 0.0, atol=1e-12) and abs(np.ptp(ahmed[:, 0]) - 1.044) < 1e-9
    assert abs(ahmed[:, 1].min() - 0.05) < 1e-12 and abs(ahmed[:, 1].max() - 0.288) < 1e-12   # 50 mm clearance
    blade = CG.circular_arc_cascade_blade_points(64, np.random.default_rng(0), flow_angle_in_deg=50.0,
                                                 flow_angle_out_deg=20.0, chord=0.1, x_le=0.0, y_le=0.0)
    assert blade.shape == (64, 2) and np.all(np.isfinite(blade))
    assert np.max(np.hypot(blade[:, 0], blade[:, 1])) <= 0.1 + 1e-9           # within one chord of the leading edge
