"""Process-level checks: mill power draw, liner wear calibration, fluid drag, and the lattice-to-material chain."""
import math

import numpy as np
import pytest

from pinneapple_simulation.granular import (
    Crystal,
    FluidCoupling,
    LennardJones,
    Material,
    RotatingDrum,
    WearTracker,
    box,
    cubic_polycrystal_moduli,
    fit_wear_law,
    material_from_lattice,
)
from pinneapple_simulation.granular.equipment import charge_positions
from pinneapple_simulation.numerical_solvers.particles import terminal_velocity


# ----------------------------------------------------------------------------------------------- packed charge
def test_charge_packing_never_starts_interpenetrating():
    positions = charge_positions(60, radius=0.15, length=0.06, particle_radius=0.01, seed=3)
    assert len(positions) == 60
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 2.0 * 0.01
    assert np.all(np.hypot(positions[:, 0], positions[:, 1]) + 0.01 < 0.15 + 1e-12)
    with pytest.raises(ValueError):
        charge_positions(10_000, radius=0.15, length=0.06, particle_radius=0.01)


# ---------------------------------------------------------------------------------------------------- the mill
@pytest.mark.parametrize("rpm", [0.0, 30.0])
def test_mill_power_draw_is_the_wall_torque_times_the_speed(rpm):
    drum = RotatingDrum(radius=0.15, length=0.045, rpm=rpm, lifters=6)
    world = drum.build(n_particles=40, particle_radius=0.01, seed=0)
    report = drum.run(world, steps=300, settle_steps=400)
    assert report["steps"] == 300
    assert report["power"] == pytest.approx(report["torque"] * report["omega"], rel=1e-12)
    # mass is conserved and the charge stays inside the barrel
    assert report["charge_mass"] == pytest.approx(world.mass.sum(), rel=1e-12)
    radial = np.hypot(world.position[:, 0], world.position[:, 1])
    assert radial.max() < 0.15
    assert np.all(np.abs(world.position[:, 2]) < 0.045)
    if rpm == 0.0:
        assert abs(report["power"]) < 1.0e-9                           # a stationary barrel draws nothing
    else:
        # the sign of the work is only meaningful once the charge has bedded down; that is what the example shows
        assert np.isfinite(report["power"]) and abs(report["torque"]) > 0.0


def test_a_faster_mill_draws_more_power():
    """The drive torque rises with the rotation rate over the same measurement window, settled first."""
    torques = []
    for rpm in (10.0, 60.0):
        drum = RotatingDrum(radius=0.15, length=0.045, rpm=rpm, lifters=8, lifter_height=0.02)
        world = drum.build(n_particles=60, particle_radius=0.01, seed=1)
        report = drum.run(world, steps=400, settle_steps=700)
        torques.append(report["torque"])
    assert torques[1] > 2.0 * torques[0], torques


# ----------------------------------------------------------------------------------------------------- wear
def test_wear_law_is_recovered_from_a_synthetic_scanned_profile():
    rng = np.random.default_rng(0)
    impulse = rng.uniform(1.0, 10.0, size=24)
    areas = rng.uniform(1.0e-4, 2.0e-3, size=24)
    hardness = 3.0e9
    true_k = 4.0e-9
    depth = true_k * impulse / (hardness * areas)
    fit = fit_wear_law(impulse, areas, depth, hardness)
    assert fit["k_wear"] == pytest.approx(true_k, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["max_residual"] == pytest.approx(0.0, abs=1e-18)


def test_wear_tracker_reads_the_impulse_off_a_running_mill():
    drum = RotatingDrum(radius=0.15, length=0.045, rpm=40.0, lifters=6)
    world = drum.build(n_particles=40, particle_radius=0.01, seed=2)
    world.run(steps=400, dt=drum.steady_timestep(world))
    tracker = WearTracker(hardness=2.0e9, k_wear=1.0e-9)
    depth = tracker.update(world)
    areas = tracker.areas_of(world)
    assert depth.shape == areas.shape == (len(world.walls[0].offsets) + len(world.walls[0].obstacles),)
    assert np.all(depth >= 0.0)
    assert depth.max() > 0.0                                           # something hit the liner
    assert tracker.impulse.sum() == pytest.approx(world.plane_impulse.sum() + world.obstacle_impulse.sum(),
                                                  rel=1e-12)


# ---------------------------------------------------------------------------------------------- fluid drag
def test_a_settling_sphere_reaches_the_terminal_velocity_of_the_existing_correlator():
    """`pinneapple_simulation.numerical_solvers.particles.terminal_velocity` solves the same Schiller-Naumann
    balance in closed form; the DEM coupling has to land on the same number."""
    diameter, density = 2.0e-3, 2500.0
    world = __import__("pinneapple_simulation.granular", fromlist=["World"]).World(gravity=(0.0, -9.81, 0.0))
    world.add(__import__("pinneapple_simulation.granular", fromlist=["Sphere"]).Sphere(diameter / 2.0,
                                                                                       density=density))
    world.finalize()
    coupling = FluidCoupling(velocity=np.zeros(3), porosity=False, buoyancy=True)
    coupling.attach(world)
    world.run(steps=15_000, dt=2.0e-4)
    measured = -float(world.velocity[0, 1])
    expected = terminal_velocity(diameter, density)
    assert measured == pytest.approx(expected, rel=0.02)
    coupling.detach(world)
    assert world.force_sources == []


def test_porosity_correction_vanishes_when_dilute_and_grows_when_packed():
    from pinneapple_simulation.granular import Sphere, World

    lone = World(gravity=(0.0, 0.0, 0.0))
    lone.add(Sphere(1.0e-3, density=2500.0), velocity=(0.0, 1.0, 0.0))
    lone.finalize()
    dilute = FluidCoupling(velocity=np.zeros(3), porosity=False, buoyancy=False).drag_force(lone)
    corrected = FluidCoupling(velocity=np.zeros(3), porosity=True, buoyancy=False).drag_force(lone)
    assert dilute[0, 1] < 0.0                                          # drag opposes the motion
    assert corrected[0, 1] == pytest.approx(dilute[0, 1], rel=0.10)    # one body: voidage ~ 1

    packed = World(gravity=(0.0, 0.0, 0.0))
    for k in range(8):                                                 # a close-packed 2x2x2 cluster
        packed.add(Sphere(1.0e-3, density=2500.0), position=(k % 2, (k // 2) % 2, k // 4), velocity=(0.0, 1.0, 0.0))
    packed.finalize()
    dense = FluidCoupling(velocity=np.zeros(3), porosity=False, buoyancy=False).drag_force(packed)
    dense_corrected = FluidCoupling(velocity=np.zeros(3), porosity=True, buoyancy=False).drag_force(packed)
    assert abs(float(dense_corrected[0, 1])) > abs(float(dense[0, 1]))  # a bed drags harder, not softer


# ------------------------------------------------------------------------------------------------ the lattice
def test_fcc_lennard_jones_obeys_the_cauchy_relation_and_born_stability():
    crystal = Crystal("fcc", LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0))
    a = crystal.relaxed_lattice_constant()
    nearest_neighbour = a / math.sqrt(2.0)
    assert nearest_neighbour == pytest.approx(1.09, abs=0.02)          # the LJ pair minimum is at 2^(1/6) = 1.122
    constants = crystal.elastic_constants(a)
    # a central-force potential at zero stress has C12 = C44 (Born & Huang)
    assert constants["cauchy_rel_error"] < 1.0e-3
    assert constants["born_stability"]["stable"]
    assert constants["C11"] > 0.0 and constants["bulk_modulus"] > 0.0


def test_simple_cubic_and_bcc_lennard_jones_are_born_unstable():
    for structure in ("sc", "bcc"):
        crystal = Crystal(structure, LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0))
        constants = crystal.elastic_constants()
        assert not constants["born_stability"]["stable"], structure
        assert constants["cauchy_rel_error"] < 1.0e-3, structure


def test_facet_energies_order_like_an_fcc_metal_and_the_wulff_shape_is_convex():
    crystal = Crystal("fcc", LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0))
    facets = crystal.facet_energies()
    assert facets[(1, 1, 1)] < facets[(1, 0, 0)] < facets[(1, 1, 0)]
    shape = crystal.wulff_shape()
    assert isinstance(shape, type(box(1.0, 1.0, 1.0)))
    assert shape.volume > 0.0
    # the Wulff polytope is closed and convex: every face normal points away from its centroid
    for normal, centre in zip(shape.face_normals, shape.face_centres, strict=True):
        assert float(normal @ (centre - shape.centroid)) > 0.0


def test_material_from_lattice_gives_a_usable_material_and_shape():
    crystal = Crystal("fcc", LennardJones(epsilon=1.0, sigma=1.0, cutoff=3.0))
    result = material_from_lattice(crystal, sigma_m=3.40e-10, epsilon_J=1.66e-21, atomic_mass_kg=6.63e-26)
    material = result["material"]
    assert isinstance(material, Material)
    assert 0.5e9 < material.E < 10.0e9                                 # solid argon is a GPa-scale solid
    assert 0.0 < material.nu < 0.5
    assert 500.0 < result["density"] < 3000.0
    assert result["born_stable"]
    moduli = cubic_polycrystal_moduli(114.2, 64.2, 64.2)
    # Hill sits between the Voigt and Reuss bounds; the bulk modulus of a cubic crystal is isotropic, so the two
    # K bounds coincide and only the shear modulus shows the gap
    assert moduli["G_reuss"] <= moduli["G_hill"] <= moduli["G_voigt"]
    assert moduli["K_reuss"] == pytest.approx(moduli["K_voigt"], rel=1e-12)
    assert moduli["K_hill"] == pytest.approx(moduli["K_voigt"], rel=1e-12)
