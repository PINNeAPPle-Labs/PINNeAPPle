"""The explicit crash solver against the physics it has to reproduce, not against itself."""
import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.crash_cases import (
    DESIGN_BOUNDS,
    bumper_beam_assembly,
    pole_impact,
)
from pinneapple_simulation.numerical_solvers.crash_fem import (
    CrashSolver,
    Material,
    RigidPole,
    RigidWall,
    qoi_from_history,
)
from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh

STEEL = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=6e8, hardening=4e8)
ELASTIC = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=1e12)          # never reaches the yield surface
BAR = box_mesh(0.10, 0.02, 0.02, 10, 2, 2)                                  # 10 mm cells, 5 x 2 x 2 mm cross-section


def test_rigid_rotation_carries_no_strain():
    """A body turned by any angle is stress free: the corotational measure has to give exactly zero strain."""
    ang = np.radians(40.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    solver = CrashSolver(BAR, material=ELASTIC, end_time=1e-6, frames=2)
    u = BAR.nodes @ R.T - BAR.nodes
    solver.u = u
    eps, _ = solver._strain_at_gauss_points()
    assert np.abs(eps).max() < 1e-12
    assert solver.energies()["elastic"] < 1e-9


def test_uniaxial_elastic_strain_matches_hooke():
    """``sigma_xx = D_11 eps_xx`` on a uniform strain state — the strain measure and D agree on units and on shear."""
    solver = CrashSolver(BAR, material=ELASTIC, end_time=1e-6, frames=2)
    eps_xx = 1e-3
    solver.u = np.c_[eps_xx * BAR.nodes[:, 0], np.zeros(len(BAR.nodes)), np.zeros(len(BAR.nodes))]
    solver.step()
    assert solver.sig[..., 0].mean() == pytest.approx(solver.D_e[0, 0, 0] * eps_xx, rel=1e-6)


def test_return_map_lands_on_the_hardening_surface():
    """After yielding, the von Mises stress equals ``yield + hardening * eps_p`` at every integration point."""
    solver = CrashSolver(BAR, material=STEEL, end_time=1e-6, frames=2)
    solver.u = np.c_[0.01 * BAR.nodes[:, 0], np.zeros(len(BAR.nodes)), np.zeros(len(BAR.nodes))]     # 1 % axial
    solver.step()
    dev = solver.sig[0, :, :3] - solver.sig[0, :, :3].mean(-1, keepdims=True)
    q = np.sqrt(1.5 * (dev ** 2).sum(-1) + 3.0 * (solver.sig[0, :, 3:] ** 2).sum(-1))
    surface = STEEL.yield_stress + STEEL.hardening * solver.eps_p[0]
    assert np.allclose(q, surface, rtol=1e-6)


def test_central_difference_conserves_energy_without_contact():
    """KE + internal energy stays put over a long elastic oscillation (the scheme is a leapfrog)."""
    solver = CrashSolver(BAR, material=ELASTIC, end_time=2e-4, frames=41)
    solver.u = 1e-7 * np.random.default_rng(0).normal(size=solver.u.shape)
    res = solver.run()
    total = res.energy_balance()
    assert abs(total[-1] / total[1] - 1.0) < 0.01


def test_free_flight_velocity_is_exactly_constant():
    solver = CrashSolver(BAR, material=ELASTIC, initial_velocity=(20.0, 0.0, 0.0), end_time=1e-4, frames=5)
    res = solver.run()
    assert np.abs(res.velocity[-1] - res.velocity[0]).max() < 1e-8
    assert np.abs(res.displacement[-1]).max() == pytest.approx(20.0 * res.times[-1], rel=1e-6)


def test_contact_starts_when_the_gap_closes():
    """A gap of 1 mm at 30 m/s is reached at 33.3 us; the penalty contact must fire at that time, not earlier."""
    gap, speed = 1e-3, 30.0
    mesh = box_mesh(0.10, 0.04, 0.04, 10, 4, 4)
    solver = CrashSolver(mesh, material=ELASTIC, contact=RigidWall(x0=0.100 + gap), initial_velocity=(speed, 0, 0),
                         end_time=200e-6, frames=3)
    solver.run()
    f = np.asarray(solver._force_hist)
    hit = np.nonzero(np.abs(f[:, 0]) > 0)[0]
    assert hit.size > 0
    t_hit = solver._time_hist[hit[0]]
    assert t_hit == pytest.approx(gap / speed, abs=2 * solver.dt)


def test_stable_step_is_below_the_exact_critical_step():
    solver = CrashSolver(BAR, material=STEEL, cfl=0.9, end_time=1e-6, frames=2)
    omega = solver._max_eigenfrequency()
    assert omega > 0
    assert solver.dt <= 2.0 / omega
    assert solver.dt_estimate > solver.dt                    # the h/c estimate is not conservative here


def test_erosion_removes_the_elements_that_reach_their_strain():
    soft = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=2e8, hardening=1e8, erosion_strain=0.05)
    solver = CrashSolver(BAR, material=soft, end_time=1e-6, frames=2)
    solver.u = np.c_[0.2 * BAR.nodes[:, 0], np.zeros(len(BAR.nodes)), np.zeros(len(BAR.nodes))]
    solver.step()
    assert solver.eroded.all()
    assert solver.sig[0].max() == 0.0


def test_pole_and_wall_push_the_body_back():
    """The contact normal points away from the obstacle, so the reaction decelerates the body."""
    for obstacle in (RigidWall(x0=0.1005), RigidPole(radius=0.05, x0=0.15)):
        solver = CrashSolver(box_mesh(0.10, 0.04, 0.04, 10, 4, 4), material=ELASTIC, contact=obstacle,
                             initial_velocity=(30.0, 0.0, 0.0), end_time=300e-6, frames=3)
        res = solver.run()
        assert res.force_history.min() < 0.0                 # the obstacle pushes in -x
        assert res.velocity[-1, :, 0].mean() < 30.0


def test_qoi_from_history_reads_a_synthetic_pulse():
    t = np.linspace(0, 2e-3, 2001)
    pulse = np.where((t > 5e-4) & (t < 1.5e-3), 1e6 * np.sin(np.pi * (t - 5e-4) / 1e-3), 0.0)
    energies = {"kinetic": np.interp(t, [0, 2e-3], [1000.0, 100.0]),
                "internal": np.interp(t, [0, 2e-3], [0.0, 900.0]),
                "plastic": np.interp(t, [0, 2e-3], [0.0, 900.0]), "contact": np.zeros_like(t)}
    q = qoi_from_history(t, pulse, energies)
    assert q["peak_wall_force"] == pytest.approx(1e6, rel=1e-3)
    assert q["ke_absorption"] == pytest.approx(0.9, rel=1e-3)
    # the 3 % threshold trims the sine at both ends: 1 - 2 asin(0.03) / pi of its half-period
    expected = 1e-3 * (1 - 2 * np.arcsin(0.03) / np.pi)
    assert q["impact_duration"] == pytest.approx(expected, rel=1e-3)


def test_coupon_assembly_has_a_fixed_mesh_and_a_part_hierarchy():
    first = bumper_beam_assembly(n_parts=3)
    second = bumper_beam_assembly(pole_radius=0.085, pole_offset=0.02)
    assert first.mesh.nodes.shape == second.mesh.nodes.shape
    assert np.allclose(first.mesh.nodes, second.mesh.nodes)           # the mesh does not move with the design
    assert np.allclose(first.mesh.elements, second.mesh.elements)
    assert len(np.unique(first.parts)) == 3
    assert len(np.unique(bumper_beam_assembly(n_parts=6).parts)) >= 6
    assert second.pole_radius == 0.085 and second.pole_offset == 0.02


def test_design_outside_the_box_is_refused():
    with pytest.raises(ValueError):
        pole_impact({"velocity": 25.0}, end_time=1e-4)


@pytest.mark.parametrize("key", list(DESIGN_BOUNDS))
def test_design_bounds_are_ordered(key):
    lo, hi = DESIGN_BOUNDS[key]
    assert lo < hi


# --------------------------------------------------------------------------------------------- frames and the gate
def test_the_run_emits_exactly_the_frames_that_were_asked_for():
    """The frame count is a property of the dataset, not of the run: every design of a campaign has its own ``dt``,
    so a count derived from ``n_steps`` gives each geometry a different number of frames and no batch can be formed
    (the campaign that found this lost the frame axis and read three components as three frames)."""
    for frames in (2, 5, 11):
        solver = CrashSolver(BAR, material=ELASTIC, end_time=4e-5, frames=frames)
        res = solver.run()
        assert res.n_frames == frames
        assert res.displacement.shape == (frames, len(BAR.nodes), 3)
        assert (np.diff(res.times) > 0).all()
        assert abs(res.times[-1] - 4e-5) <= res.dt            # the last frame is the last step, inside one dt


def test_an_end_time_that_cannot_hold_the_frames_is_refused():
    with pytest.raises(ValueError, match="frames"):
        CrashSolver(BAR, material=ELASTIC, end_time=1e-8, frames=50).run()


def test_the_energy_gate_keeps_a_conserving_run_and_flags_an_imbalanced_one():
    """The scheme conserves kinetic + internal + contact against the external work (0.8-1.6 % over the 200-design
    campaign), so an imbalance beyond the tolerance is what a run that left the verified range looks like — the
    element stretches alone do not catch it, because erosion removes exactly the elements that would trip them."""
    solver = CrashSolver(BAR, material=ELASTIC, end_time=4e-5, frames=2)
    solver._check_energy_balance({"kinetic": 10.0, "internal": 90.0, "contact": 0.5, "external": 100.0})
    assert solver.out_of_envelope is False and solver.envelope_reason is None
    solver._check_energy_balance({"kinetic": 10.0, "internal": 140.0, "contact": 0.5, "external": 100.0})
    assert solver.out_of_envelope is True and solver.envelope_reason == "energy_balance"


def test_the_energy_gate_ignores_the_frames_before_the_obstacle_has_done_its_work():
    """Before 5 % of the run's work is in, both sides of the balance are near zero and their ratio is noise."""
    solver = CrashSolver(BAR, material=ELASTIC, end_time=4e-5, frames=2)
    solver._check_energy_balance({"kinetic": 100.0, "internal": 900.0, "contact": 0.0, "external": 1000.0})
    assert solver._ext_max == 1000.0 and solver.out_of_envelope is False
    # 1 J of work against a 1000 J scale is under the 5 % gate: the ratio of two near-zero numbers is not checked
    solver._check_energy_balance({"kinetic": 0.0, "internal": 5.0, "contact": 0.0, "external": 1.0})
    assert solver.out_of_envelope is False


def test_a_non_finite_state_trips_the_gate():
    solver = CrashSolver(BAR, material=ELASTIC, end_time=4e-5, frames=2)
    solver._check_energy_balance({"kinetic": np.nan, "internal": 0.0, "contact": 0.0, "external": 1.0})
    assert solver.out_of_envelope is True and solver.envelope_reason == "energy_balance"


def test_the_pole_impact_coupon_reports_no_imbalance_inside_its_verified_window():
    """The reference design, stopped before erosion starts, conserves energy — the run the campaign is run at."""
    res = pole_impact(end_time=8e-4, frames=6)
    assert res.out_of_envelope is False, res.envelope_reason
    e = {k: float(v[-1]) for k, v in res.energies.items()}
    assert e["external"] > 0
    assert abs(e["kinetic"] + e["internal"] + e["contact"] - e["external"]) < 0.05 * e["external"]
