"""The VVD solver: the panel integrals against a quadrature, the exact potential flow, the wall conditions, the
diffusion drift, the geometry, and a short run end to end."""
import numpy as np
import pytest

from pinneapple_simulation.vortex_domains import (
    Assembly,
    Domains,
    SourceSheet,
    VortexSheet,
    cylinder,
    cylinder_case,
    deploy,
    deploying_spoiler,
    edge_velocity,
    grid_field,
    inside_polygon,
    kernels,
    naca0012,
    naca0012_spoiler_case,
    naca4,
    plate,
    polygon,
    rotating,
    run_study,
    spoiler_hinge,
    streamlines,
    strouhal_number,
    surface_pressure,
    verification,
    vorticity_flux,
)


@pytest.fixture(scope="module")
def cylinder_panels():
    return Assembly([cylinder(1.0, 64)]).panels(0.0)


def test_panel_integrals_match_quadrature():
    result = verification.check_panel_influence()
    assert result["ok"], result["detail"]


def test_potential_flow_on_a_cylinder_is_exact():
    result = verification.check_potential_flow()
    assert result["ok"], result["detail"]


def test_spinning_cylinder_sheet_matches_surface_speed():
    result = verification.check_spinning_cylinder()
    assert result["ok"], result["detail"]


def test_diffusion_drift_points_down_the_vorticity_gradient():
    result = verification.check_diffusion()
    assert result["ok"], result["detail"]


def test_source_sheet_leaves_no_normal_velocity(cylinder_panels):
    u_inf = np.array([1.0, 0.0])
    matrix = kernels.source_sheet_system(cylinder_panels.mid, cylinder_panels.a, cylinder_panels.b,
                                         cylinder_panels.normal)
    sheet = SourceSheet.solve(cylinder_panels, u_inf, np.zeros_like(cylinder_panels.mid), matrix)
    assert np.abs(sheet.impermeability(u_inf, np.zeros_like(cylinder_panels.mid))).max() < 1e-10
    # the sheet of a cylinder in a stream is twice the normal component of the free stream
    assert np.allclose(np.abs(sheet.sigma), np.abs(2 * (cylinder_panels.normal @ u_inf)), atol=0.05)


def test_vortex_sheet_leaves_no_slip(cylinder_panels):
    u_inf = np.array([0.0, 0.0])
    matrix = kernels.vortex_sheet_system(cylinder_panels.mid, cylinder_panels.a, cylinder_panels.b,
                                         cylinder_panels.tangent)
    sheet = VortexSheet.solve(cylinder_panels, u_inf, np.zeros_like(cylinder_panels.mid), matrix)
    assert np.abs(sheet.slip(u_inf, np.zeros_like(cylinder_panels.mid))).max() < 1e-10


def test_lighthill_flux_vanishes_at_the_stagnation_points(cylinder_panels):
    u_inf = np.array([1.0, 0.0])
    matrix = kernels.source_sheet_system(cylinder_panels.mid, cylinder_panels.a, cylinder_panels.b,
                                         cylinder_panels.normal)
    sheet = SourceSheet.solve(cylinder_panels, u_inf, np.zeros_like(cylinder_panels.mid), matrix)
    u, v = sheet.at_midpoints()
    velocity = np.column_stack([u + 1.0, v])
    flux = vorticity_flux(cylinder_panels, edge_velocity(cylinder_panels, velocity))
    theta = np.arctan2(cylinder_panels.mid[:, 1], cylinder_panels.mid[:, 0])
    # the stagnation points are at theta = 0 and pi; so are the pressure minima at pi/2 and 3pi/2
    for angle in (0.0, np.pi, 0.5 * np.pi, 1.5 * np.pi):
        k = int(np.argmin(np.abs(np.angle(np.exp(1j * (theta - angle))))))
        assert abs(flux[k]) < 0.15 * np.abs(flux).max(), f"flux not small at theta = {angle:.2f}"


def test_geometry_generators_are_closed_and_oriented():
    body = naca0012(n=80)
    assert len(body.vertices) == 80
    assert body.vertices[:, 0].min() == pytest.approx(0.0, abs=1e-9)
    assert body.vertices[:, 0].max() == pytest.approx(1.0, abs=1e-9)
    assert body.vertices[:, 1].max() == pytest.approx(0.06, abs=1e-3)     # 12 % thick section
    assert 0.078 < body.area < 0.085
    # counter-clockwise, so the panels' outward normal points away from the body
    panels = Assembly([body]).panels(0.0)
    outward = (panels.normal * (panels.mid - panels.mid.mean(axis=0))).sum(axis=1)
    assert (outward > 0).all()
    cambered = naca4(m=0.04, p=0.4, t=0.12)
    assert cambered.area > body.area


def test_inside_polygon():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    points = np.array([[1.0, 1.0], [3.0, 1.0], [-0.5, 1.0], [1.0, 1.9]])
    assert list(inside_polygon(points, square)) == [True, False, False, True]


def test_spoiler_hinge_and_deployment():
    hinge = spoiler_hinge(chord=1.0, spoiler_length=0.10, trailing_gap=0.20)
    assert hinge[0] == pytest.approx(0.8)
    spoiler = deploying_spoiler(hinge_at=(0.8, 0.0), length=0.1, angle1_deg=90.0, t_start=1.0, duration=1.0)
    flush = spoiler.world_vertices(0.5)
    assert flush[:, 0].max() == pytest.approx(0.9, abs=1e-9)            # flush: trailing edge 20 % ahead of it
    assert flush[:, 1].max() == pytest.approx(0.002, abs=2e-3)
    upright = spoiler.world_vertices(2.5)
    assert upright[:, 0].max() == pytest.approx(0.8, abs=5e-3)          # fully deployed: vertical (plus half the plate)
    assert upright[:, 1].max() == pytest.approx(0.1, abs=2e-3)
    angle, rate = deploy(0.0, 90.0, 0.0, 1.0)
    assert angle(0.0) == pytest.approx(0.0) and angle(1.0) == pytest.approx(np.pi / 2)
    assert rate(0.5) > 0.0 and rate(1.5) == 0.0


def test_domains_merge_and_report_their_circulation():
    x = np.array([0.0, 1e-4, 0.5, 1.0])
    gamma = np.array([1.0, 2.0, -1.0, 1.0])
    dom = Domains(x, np.zeros(4), gamma)
    dom.update_epsilon(0.0)
    assert np.all(dom.eps > 0)
    merged = dom.merge(1e-2, factor=0.4, strangers=False)
    assert merged >= 1 and len(dom) < 4
    assert dom.total_circulation == pytest.approx(gamma.sum())
    removed = dom.remove(np.array([True, False, False, False], dtype=bool)[: len(dom)])
    assert removed.size == 1
    assert dom.total_circulation == pytest.approx(gamma.sum() - removed.sum())


def test_short_run_stays_finite_and_leaves_no_normal_velocity():
    case = cylinder_case(re=40.0, panels=32, t_end=0.4)
    sim = case.build()
    sim.run()
    series = sim.series()
    assert np.all(np.isfinite(series["fx"])) and np.all(np.isfinite(series["fy"]))
    assert len(sim.domains) > 0
    assert np.max(series["impermeability"]) < 1e-6
    assert np.all(series["max_displacement"] < 4.0 * series["eps_floor"] + 1e-9)
    coefficients = sim.coefficients
    assert set(("cl", "cd", "cm")) <= set(coefficients)
    assert np.isfinite(coefficients["cd"])


def test_potential_flow_exerts_no_force():
    """d'Alembert: the source sheet alone (no vorticity yet) leaves no net force and no lift."""
    sim = cylinder_case(re=40.0, panels=32, t_end=0.4).build()
    sim.sheet = SourceSheet.solve(sim.panels, np.array([1.0, 0.0]), np.zeros_like(sim.panels.mid))
    force = sim._pressure_force(sim.panels)
    assert np.abs(force).max() < 1e-6, force


def test_fields_and_streamlines():
    case = cylinder_case(re=40.0, panels=24, t_end=0.2)
    sim = case.build()
    sim.run()
    field = grid_field(sim, extent=(-1.0, 2.0, -1.5, 1.5), n=24)
    assert field.shape == (24, 24)
    assert np.all(np.isfinite(field.vorticity)) and np.all(np.isfinite(field.pressure))
    assert np.abs(field.speed).max() > 0.5
    mid, cp = surface_pressure(sim)
    assert mid.shape[0] == len(sim.panels) and np.all(np.isfinite(cp))
    lines = streamlines(sim, [[-1.0, 0.0], [-1.0, 0.4]], step=0.05, n_steps=20)
    assert len(lines) == 2 and all(line.ndim == 2 and line.shape[1] == 2 for line in lines)


def test_study_reduces_a_run_and_reports_checks():
    result = run_study(cylinder_case(re=40.0, panels=24, t_end=0.6))
    assert result.series["t"].size > 5
    assert np.isfinite(result.coefficients["cd"])
    names = {c["name"] for c in result.checks}
    assert {"impermeability", "no_slip", "displacement", "circulation", "resolution"} <= names
    assert "Cd" in result.summary()


def test_strouhal_of_a_synthetic_signal():
    t = np.arange(0.0, 20.0, 0.02)
    series = {"t": t, "fy": np.sin(2 * np.pi * 0.5 * t)}
    assert strouhal_number(series) == pytest.approx(0.5, abs=0.02)
    assert strouhal_number({"t": t, "fy": np.zeros_like(t)}) is None


def test_reference_case_builds():
    case = naca0012_spoiler_case(re=40_000.0, panels=60, spoiler_panels=8, t_end=0.05)
    assert len(case.assembly) == 2
    assert case.metadata["hinge"][0] == pytest.approx(0.8)
    assert "spoiler" in case.describe()
    sim = case.build()
    assert sim.dt > 0
    assert len(sim.panels) == 68


def test_plate_and_polygon_bodies():
    plank = plate(length=0.2, thickness=0.01, n=8)
    assert plank.area == pytest.approx(0.2 * 0.01, rel=0.05)
    tri = polygon([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert tri.area == pytest.approx(0.5)
    spun = rotating((0.0, 0.0), 2.0)
    centre, angle = spun.ring(0.5)
    assert angle == pytest.approx(1.0)
