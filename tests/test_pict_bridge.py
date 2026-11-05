"""The PICT bridge without PICT: the grid it builds, the reference data it compares against, the checks it applies.

PICT needs Linux, an NVIDIA GPU and its own compiled CUDA extensions, none of which exist on a test runner, so what
is tested here is everything that does not need the solver -- and the fact that the solver path refuses to invent
numbers when it is missing.
"""
import numpy as np
import pytest

from pinneapple_simulation.external_solvers.pict import (
    GHIA,
    CavityCase,
    PICTResult,
    PICTRunConfig,
    build_cavity_domain,
    cavity_checks,
    centerline,
    compare_to_ghia,
    divergence_max,
    ghia_profile,
    pict_availability,
    pict_available,
    run_cavity,
    validate,
    wall_refined_grid,
)
from pinneapple_simulation.external_solvers.pict.validation import spacing_from_summary

# ---------------------------------------------------------------------------
# Availability: the solver is absent here, and the bridge says so
# ---------------------------------------------------------------------------


def test_availability_is_a_report_not_a_crash():
    availability = pict_availability()
    assert isinstance(pict_available(), bool)
    assert availability.available == pict_available()
    if not availability.available:
        assert availability.missing
        assert "PICT" in availability.hint


def test_solver_paths_raise_with_the_install_line_when_pict_is_absent():
    if pict_available():
        pytest.skip("PICT is installed here; the refusal path is what this test is about")
    case = CavityCase(nx=8, ny=8, Re=100)
    with pytest.raises(ImportError, match="PICT cannot run here"):
        build_cavity_domain(case)
    with pytest.raises(ImportError, match="PICT cannot run here"):
        run_cavity(case)


def test_validate_reports_why_it_could_not_run():
    report = validate(reynolds=(100,), nx=8, ny=8, iterations=1)
    if pict_available():
        assert report["available"] and report["rows"]
    else:
        assert report["available"] is False and report["rows"] == [] and report["reason"]


# ---------------------------------------------------------------------------
# The grid: our own generator, checked rather than trusted
# ---------------------------------------------------------------------------


def test_grid_has_pict_layout_and_exact_corners():
    g = wall_refined_grid(16, 8)
    assert g.shape == (1, 2, 9, 17)                       # (1, dims, ny+1, nx+1)
    assert g[0, 0, 0, 0] == pytest.approx(0.0) and g[0, 0, 0, -1] == pytest.approx(1.0)
    assert g[0, 1, 0, 0] == pytest.approx(0.0) and g[0, 1, -1, 0] == pytest.approx(1.0)
    assert np.all(np.diff(g[0, 0, 0]) > 0) and np.all(np.diff(g[0, 1, :, 0]) > 0)


def test_grid_refinement_ratio_is_the_base_between_neighbours():
    g = wall_refined_grid(32, 32, base=1.10)
    xs = g[0, 0, 0]
    widths = np.diff(xs)
    inner = widths[1:14]                                  # away from the mirrored middle
    assert np.allclose(inner[1:] / inner[:-1], 1.10, rtol=1e-4)
    assert widths[0] < widths[len(widths) // 4]           # packed at the walls, coarse in the middle


def test_grid_refinement_is_symmetric_when_both_walls_are_named():
    g = wall_refined_grid(32, 32, refine=("-x", "+x", "-y", "+y"), base=1.10)
    widths = np.diff(g[0, 0, 0])
    assert np.allclose(widths, widths[::-1], rtol=1e-6)
    spans = np.diff(g[0, 1, :, 0])
    assert np.allclose(spans, spans[::-1], rtol=1e-6)


def test_refining_one_wall_packs_only_that_side():
    widths = np.diff(wall_refined_grid(24, 8, refine=("+x",))[0, 0, 0])
    assert widths[-1] < widths[0] and np.all(np.diff(widths) < 0)     # smallest cells at the +x wall
    widths = np.diff(wall_refined_grid(24, 8, refine=("-x",))[0, 0, 0])
    assert widths[0] < widths[-1] and np.all(np.diff(widths) > 0)     # smallest cells at the -x wall


def test_grid_rejects_impossible_arguments():
    with pytest.raises(ValueError, match="base"):
        wall_refined_grid(8, 8, base=0.5)
    with pytest.raises(ValueError, match="bounds"):
        wall_refined_grid(8, 8, bounds=(1.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="cell"):
        wall_refined_grid(0, 8)


def test_grid_bounds_are_honoured():
    g = wall_refined_grid(8, 6, bounds=(-2.0, 0.5, 2.0, 3.5))
    assert g[0, 0].min() == pytest.approx(-2.0) and g[0, 0].max() == pytest.approx(2.0)
    assert g[0, 1].min() == pytest.approx(0.5) and g[0, 1].max() == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# The case description
# ---------------------------------------------------------------------------


def test_case_validates_itself_before_any_solver():
    assert CavityCase().check() == []
    assert CavityCase(nx=2).check()
    assert CavityCase(Re=0).check()
    assert CavityCase(lid_velocity=0.0).check()
    assert CavityCase(lid_side="+z").check()
    assert CavityCase(base=0.5).check()
    with pytest.raises(ValueError):
        CavityCase(nx=2).grid()


def test_viscosity_follows_the_reynolds_number():
    assert CavityCase(Re=100, lid_velocity=1.0).viscosity == pytest.approx(1e-2)
    assert CavityCase(Re=1000, lid_velocity=1.0).viscosity == pytest.approx(1e-3)
    assert CavityCase(Re=1000, lid_velocity=2.0).viscosity == pytest.approx(2e-3)


def test_run_config_maps_onto_picts_simulation_arguments():
    kw = PICTRunConfig(iterations=7, time_step=0.5, differentiable=True).to_kwargs()
    assert kw["time_step"] == 0.5 and kw["differentiable"] is True and kw["substeps"] == "ADAPTIVE"
    assert "log_interval" in kw


# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data
# ---------------------------------------------------------------------------


def test_reference_tables_are_consistent():
    for component in ("u", "v"):
        station = GHIA["y" if component == "u" else "x"]
        assert len(station) == 17
        for re, values in GHIA[component].items():
            assert len(values) == len(station), (component, re)
            assert values[0] == 0.0 and values[-1] == 0.0 or component == "u"
    # the lid drives u to 1 at y = 1 and the no-slip bottom holds it at 0
    assert GHIA["u"][100][-1] == 1.0 and GHIA["u"][100][0] == 0.0
    # the classic minimum of the Re = 1000 profile at the centre
    assert GHIA["u"][1000][GHIA["y"].index(0.5000)] == pytest.approx(-0.06080)


def test_unknown_reynolds_number_names_the_ones_available():
    with pytest.raises(KeyError, match="100"):
        ghia_profile(500)
    with pytest.raises(KeyError, match="component"):
        ghia_profile(100, "w")
    x, u = ghia_profile(1000, "u")
    assert x.shape == u.shape == (17,)


# ---------------------------------------------------------------------------
# Reducing a field to the quantities the checks need
# ---------------------------------------------------------------------------


def test_centerline_averages_the_two_middle_lines():
    field = np.arange(16, dtype=float).reshape(4, 4)
    assert np.array_equal(centerline(field, direction="vertical"), field[:, [1, 2]].mean(axis=1))
    assert np.array_equal(centerline(field, direction="horizontal"), field[[1, 2], :].mean(axis=0))
    odd = np.arange(9, dtype=float).reshape(3, 3)
    assert np.array_equal(centerline(odd, direction="vertical"), odd[:, 1])
    with pytest.raises(ValueError, match="2-D"):
        centerline(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="direction"):
        centerline(odd, direction="diagonal")


def test_ghia_comparison_is_zero_on_the_table_and_detects_a_shift():
    y, u = ghia_profile(1000, "u")
    exact = compare_to_ghia(u, y, reynolds=1000, component="u")
    assert exact["rms"] == pytest.approx(0.0, abs=1e-12) and exact["passed"]
    shifted = compare_to_ghia(u + 0.3, y, reynolds=1000, component="u")
    assert shifted["rms"] == pytest.approx(0.3, abs=1e-12) and not shifted["passed"]
    # the tolerance is the caller's: the same shift passes at 0.05 and fails at 0.01
    assert compare_to_ghia(u + 0.03, y, reynolds=1000, component="u", tol=0.05)["passed"]
    assert not compare_to_ghia(u + 0.03, y, reynolds=1000, component="u", tol=0.01)["passed"]
    with pytest.raises(ValueError, match="must match"):
        compare_to_ghia(u[:-1], y, reynolds=1000)


def test_divergence_is_small_for_a_solenoidal_field_and_large_otherwise():
    n = 33
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    # u = sin(pi x) cos(pi y), v = -cos(pi x) sin(pi y): divergence-free in the continuum
    u = np.sin(np.pi * X) * np.cos(np.pi * Y)
    v = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    dx = dy = np.full(n, h)
    div = divergence_max(u, v, dx, dy)
    assert div < 0.05 * np.pi                               # second-order error on a 33-point grid
    assert divergence_max(u * 3.0, v, dx, dy) > 2.0 * div   # a stretching flow is not divergence-free
    with pytest.raises(ValueError, match="same shape"):
        divergence_max(u, v[:-1], dx, dy)
    with pytest.raises(ValueError, match="describe"):
        divergence_max(u, v, dx[:-1], dy)


# ---------------------------------------------------------------------------
# The checks applied to a recorded run, on fields whose answer is known
# ---------------------------------------------------------------------------


def _result(u: np.ndarray, v: np.ndarray, *, Re: float = 1000.0) -> PICTResult:
    case = CavityCase(nx=u.shape[1], ny=u.shape[0], Re=Re, refine=True, base=1.10)
    return PICTResult(velocity=np.stack([u, v])[None], pressure=np.zeros((1,) + u.shape), times=np.array([0.25]),
                      case_summary=case.summary(), config={}, device="cpu", wall_time_s=1.0)


def test_checks_pass_the_boundary_conditions_and_flag_a_still_lid():
    ny, nx = 16, 16
    zeros = np.zeros((ny, nx))
    checks = cavity_checks(_result(zeros, zeros))
    assert checks["continuity"]["passed"] and checks["walls"]["passed"]
    assert not checks["lid"]["passed"]                      # a still lid is not a cavity with flow
    assert not checks["passed"]

    # A field that satisfies both boundary conditions *on the discrete grid* and is still not a solution:
    # u = k P(x) y^2 with P vanishing at the first and last cell centres and v = 0. Still at the walls, mean 1 on the
    # top row (the lid), and du/dx != 0 so the divergence check has something to catch.
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    P = (x - x[0]) * (x[-1] - x)
    k = 1.0 / (P.mean() * y[-1] ** 2)
    moving = k * P[None, :] * (y[:, None] ** 2)
    checks = cavity_checks(_result(moving, zeros))
    assert checks["lid"]["passed"] and checks["walls"]["passed"]
    assert not checks["continuity"]["passed"]               # and the divergence is caught
    assert not checks["passed"]


def test_checks_compare_the_centrelines_with_the_reference():
    y, u_profile = ghia_profile(1000, "u")
    x, v_profile = ghia_profile(1000, "v")
    case = CavityCase(nx=64, ny=64, Re=1000, refine=False)     # uniform grid: the reference stations are direct
    centres = (np.arange(64) + 0.5) / 64
    u = np.tile(np.interp(centres, y, u_profile)[:, None], (1, 64))
    v = np.tile(np.interp(centres, x, v_profile)[None, :], (64, 1))
    u[-1, :] = 1.0                                           # the lid, as the reference has it
    result = PICTResult(velocity=np.stack([u, v])[None], pressure=np.zeros((1, 64, 64)), times=np.array([1.0]),
                        case_summary=case.summary(), config={}, device="cpu", wall_time_s=1.0)
    checks = cavity_checks(result)
    assert checks["ghia_u"]["rms"] < 0.05 and checks["ghia_u"]["passed"]
    assert checks["ghia_v"]["rms"] < 0.05 and checks["ghia_v"]["passed"]


def test_result_exposes_components_centres_and_a_physical_sample():
    ny, nx = 8, 6
    case = CavityCase(nx=nx, ny=ny, Re=100, refine=False)
    velocity = np.zeros((3, 2, ny, nx))
    velocity[:, 0] = 0.5
    velocity[:, 1] = -0.25
    result = PICTResult(velocity=velocity, pressure=np.zeros((3, ny, nx)), times=np.array([0.25, 0.5, 0.75]),
                        case_summary=case.summary(), config={"iterations": 3}, device="cuda", wall_time_s=2.0)
    assert result.component("u").shape == (3, ny, nx) and result.component("u")[0, 0, 0] == 0.5
    with pytest.raises(KeyError, match="component"):
        result.component("w")
    x, y = result.cell_centers()
    assert x.shape == (nx,) and y.shape == (ny,)
    dx, dy = spacing_from_summary(result.case_summary)
    assert dx.shape == (nx,) and dy.shape == (ny,)

    sample = result.to_upd()
    assert sample.state["u"].shape == (3, ny * nx)
    assert sample.provenance["source"] == "pict" and sample.schema["units"]["u"] == "m/s"
