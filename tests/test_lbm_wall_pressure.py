"""Wall and surface pressure out of the lattice-Boltzmann tunnel: the extraction a pressure-field dataset is built on."""
import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.lbm_tunnel import (
    WindTunnelLBM,
    sample_field,
    sphere,
    stokes_sphere_field,
)


def test_sample_field_is_exact_for_a_linear_field():
    """Trilinear interpolation reproduces a linear field exactly at any interior point."""
    x, y, z = np.meshgrid(np.arange(6.0), np.arange(6.0), np.arange(6.0), indexing="ij")
    linear = 2 * x + 3 * y - 4 * z
    pts = np.array([[1.5, 2.5, 3.25], [0.0, 0.0, 0.0], [2.75, 1.25, 4.5], [4.0, 4.0, 4.0]])
    got = sample_field(linear, pts)
    expected = [2 * p[0] + 3 * p[1] - 4 * p[2] for p in pts]
    assert np.allclose(got, expected, atol=1e-12)


def test_sample_field_clamps_instead_of_extrapolating():
    field = np.zeros((4, 4, 4))
    field[:, :, :] = np.arange(4)[None, None, :]
    got = sample_field(field, np.array([[-5.0, -5.0, -5.0], [9.0, 9.0, 9.0]]))
    assert np.allclose(got, [0.0, 3.0])


def test_wall_pressure_is_the_mean_of_its_fluid_neighbours():
    """The tap of a wall cell reads the mean pressure of the fluid cells that link to it — by construction, and
    exactly, so a change in the averaging would break this test."""
    shape = (24, 16, 16)
    lbm = WindTunnelLBM(shape, nu=0.05, u_in=(0.05, 0.0, 0.0))
    lbm.add_solid(sphere(shape, (10, 8, 8), 4))
    lbm.run(200)
    wall = lbm.wall_pressure()
    rho, _ = lbm.macroscopic()
    fluid = ~lbm._all_solid
    p = (rho - 1.0) / 3.0
    for sel in (0, len(wall["p"]) // 2, -1):
        i, j, k = wall["cells"][sel].astype(int)
        acc = []
        for ci in lbm.c.tolist()[1:]:
            a, b, c = i + ci[0], j + ci[1], k + ci[2]
            if 0 <= a < shape[0] and 0 <= b < shape[1] and 0 <= c < shape[2] and bool(fluid[a, b, c]):
                acc.append(float(p[a, b, c]))
        assert wall["p"][sel] == pytest.approx(float(np.mean(acc)), rel=1e-6)   # float32 lattice arithmetic


def test_surface_pressure_reads_the_wall_value_and_stays_finite_off_the_body():
    shape = (24, 16, 16)
    lbm = WindTunnelLBM(shape, nu=0.05, u_in=(0.05, 0.0, 0.0))
    lbm.add_solid(sphere(shape, (10, 8, 8), 4))
    lbm.run(200)
    wall = lbm.wall_pressure()
    at_cell = lbm.surface_pressure(wall["cells"][:50])
    assert np.allclose(at_cell, wall["p"][:50], atol=1e-7)
    far = lbm.surface_pressure(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
    assert np.all(np.isfinite(far))


def test_stokes_sphere_wall_pressure_has_the_analytic_sign_and_scale():
    """In Stokes flow the pressure difference across the sphere is ``3 mu U / a``; on this coarse staircase sphere the
    sampler must reproduce its sign and its order of magnitude (the solver's own accuracy is the drag test's job)."""
    R, nu, u = 4.0, 1 / 6, 0.002
    L = int(8 * R)
    c = ((L - 1) / 2,) * 3
    lbm = WindTunnelLBM((L, L, L), nu=nu, u_in=(u, 0.0, 0.0), faces={k: "open" for k in WindTunnelLBM.FACES})
    lbm.add_solid(sphere(lbm.shape, c, R))
    lbm.initialize(stokes_sphere_field(lbm.shape, c, R, (u, 0.0, 0.0)))
    lbm.run(400)
    wall = lbm.wall_pressure()
    front = wall["p"][wall["cells"][:, 0] <= wall["cells"][:, 0].min()].mean()
    back = wall["p"][wall["cells"][:, 0] >= wall["cells"][:, 0].max()].mean()
    analytic = 3 * nu * u / R
    assert front > back                                        # the stagnation side is the high-pressure side
    assert 0.3 < (front - back) / analytic < 2.0               # coarse sphere, staircase wall: order of magnitude


def test_run_averaged_reports_the_same_field_for_a_steady_flow():
    """Averaging a converged steady flow returns the flow (a few percent, the residual unsteadiness)."""
    shape = (24, 16, 16)
    lbm = WindTunnelLBM(shape, nu=0.05, u_in=(0.05, 0.0, 0.0))
    lbm.add_solid(sphere(shape, (10, 8, 8), 4))
    lbm.run(400)
    instant = lbm.fields()["u"]
    avg = lbm.run_averaged(300, average_from=0, every=50)
    assert avg["samples"] == 6
    assert np.abs(avg["u"] - instant).max() < 5e-3
    assert np.all(np.isfinite(avg["field"]))


def test_averaging_before_the_run_starts_is_refused():
    lbm = WindTunnelLBM((8, 8, 8), nu=0.1, u_in=(0.01, 0.0, 0.0))
    with pytest.raises(ValueError):
        lbm.run_averaged(10, average_from=1000)


def test_pressure_field_carries_nan_on_the_solid():
    shape = (16, 12, 12)
    lbm = WindTunnelLBM(shape, nu=0.05, u_in=(0.05, 0.0, 0.0))
    lbm.add_solid(sphere(shape, (8, 6, 6), 2))
    lbm.run(50)
    p = lbm.pressure()
    assert np.isnan(p[lbm.solid.cpu().numpy()]).all()
    assert np.all(np.isfinite(p[~lbm._all_solid.cpu().numpy()]))
