"""The DEM step: free fall, momentum, force balance at rest, restitution, friction on a slope, breakage.

Every check here is a physical statement with an independent answer: an analytic trajectory, a conservation law, a
force balance, a restitution ratio, a holding angle, or a tensile strength.  Where the model is an idealisation (the
dashpot is viscous, so restitution is approximate) the tolerance says so and the reason is in the docstring.
"""
import math

import numpy as np
import pytest

from pinneapple_simulation.granular import (
    BondMaterial,
    ForceModel,
    Material,
    Sphere,
    Wall,
    World,
    box,
    fragment_ids,
    fragment_stats,
    p80,
)


def _material(**kwargs):
    base = dict(density=2500.0, E=5.0e7, nu=0.3, restitution=0.3, mu=0.5, mu_roll=0.05)
    base.update(kwargs)
    return Material(**base)


def _floor(material):
    return Wall(np.array([[0.0, -1.0, 0.0]]), np.array([0.0]), material=material)


# ------------------------------------------------------------------------------------------------ trajectories
def test_free_fall_follows_the_analytic_trajectory():
    world = World(gravity=(0.0, -9.81, 0.0))
    world.add(Sphere(0.01, density=2500.0), position=(0.0, 1.0, 0.0))
    world.finalize()
    dt, steps = 1.0e-4, 500
    world.run(steps=steps, dt=dt)
    elapsed = steps * dt
    expected = 1.0 - 0.5 * 9.81 * elapsed ** 2
    assert world.position[0, 1] == pytest.approx(expected, abs=1.0e-4)
    assert world.velocity[0, 1] == pytest.approx(-9.81 * elapsed, rel=1.0e-6)


def test_head_on_collision_conserves_momentum_and_angular_momentum():
    world = World(gravity=(0.0, 0.0, 0.0), model=ForceModel(mode="hertz"))
    material = _material(density=1000.0, restitution=1.0)
    world.add(Sphere(0.01, density=1000.0), position=(-0.011, 0.0, 0.0), velocity=(0.5, 0.0, 0.0),
              material=material)
    world.add(Sphere(0.01, density=1000.0), position=(0.011, 0.002, 0.0), velocity=(-0.5, 0.0, 0.0),
              material=material)
    world.finalize()
    momentum = world.momentum().copy()
    angular = world.total_angular_momentum().copy()
    world.run(steps=3000, dt=1.0e-6)
    assert np.allclose(world.momentum(), momentum, atol=1.0e-12)
    # orbital alone is not conserved once the contact transmits torque: the spin carries the rest
    assert np.allclose(world.total_angular_momentum(), angular, rtol=1.0e-6, atol=1.0e-12)


# ------------------------------------------------------------------------------------------------ force balance
def test_a_sphere_at_rest_sits_at_the_force_balance_penetration():
    material = _material(restitution=0.05)
    world = World(gravity=(0.0, -9.81, 0.0), walls=[_floor(material)])
    radius = 0.01
    world.add(Sphere(radius, density=2500.0), position=(0.0, radius + 2.0e-4, 0.0), material=material)
    world.finalize()
    world.run(steps=6000, dt=2.0e-5)
    volume = world.mass[0] * 9.81 / world.model.volume_stiffness        # the overlap the contact must supply
    expected = math.sqrt(volume / (math.pi * radius))                   # pi R delta^2 = V for a shallow cap
    measured = radius - float(world.position[0, 1])
    assert measured == pytest.approx(expected, rel=0.02)
    assert abs(float(world.velocity[0, 1])) < 1.0e-3


def test_a_box_rests_flat_without_spinning():
    material = _material(restitution=0.05, mu=0.6)
    world = World(gravity=(0.0, -9.81, 0.0), walls=[_floor(material)])
    world.add(box(0.02, 0.02, 0.02, density=2500.0), position=(0.0, 0.0105, 0.0), material=material)
    world.finalize()
    world.run(steps=6000, dt=2.0e-5)
    # the clipped volume of a face contact is the slab A * depth, so depth = m g / (k_v A)
    expected = world.mass[0] * 9.81 / (world.model.volume_stiffness * 0.02 ** 2)
    assert (0.01 - float(world.position[0, 1])) == pytest.approx(expected, rel=0.05)
    assert np.all(np.abs(world.omega[0]) < 1.0e-3)


# ------------------------------------------------------------------------------------------------ restitution
@pytest.mark.parametrize("restitution", [0.3, 0.5, 0.8, 1.0])
def test_restitution_is_approximate_but_monotone(restitution):
    """The normal dashpot is viscous (Kelvin-Voigt with a capped dashpot), so the recovered restitution is a
    nominal value: it is monotone in the configured one and never exceeds it by more than ~0.25."""
    material = _material(restitution=restitution, mu=0.0, mu_roll=0.0)
    world = World(gravity=(0.0, 0.0, 0.0), walls=[_floor(material)])
    world.add(Sphere(0.01, density=2500.0), position=(0.0, 0.02, 0.0), velocity=(0.0, -1.0, 0.0),
              material=material)
    world.finalize()
    incoming = 0.0
    outgoing = 0.0
    for _ in range(int(0.02 / 5.0e-6)):
        world.step(5.0e-6)
        vy = float(world.velocity[0, 1])
        incoming = min(incoming, vy)
        if incoming < 0.0:
            outgoing = max(outgoing, vy)
    measured = abs(outgoing) / abs(incoming)
    assert measured <= restitution + 0.25
    assert measured >= restitution - 0.05


def test_the_contact_model_never_adds_energy():
    material = _material(restitution=0.0, mu=0.0, mu_roll=0.0)
    world = World(gravity=(0.0, 0.0, 0.0), walls=[_floor(material)])
    world.add(Sphere(0.01, density=2500.0), position=(0.0, 0.02, 0.0), velocity=(0.0, -1.0, 0.0),
              material=material)
    world.finalize()
    energy = []
    for _ in range(4000):
        world.step(5.0e-6)
        energy.append(world.kinetic_energy())
    assert max(energy[1:]) <= energy[0] * (1.0 + 1.0e-9)


# ------------------------------------------------------------------------------------------------ friction
@pytest.mark.parametrize("angle, holds", [(10.0, True), (20.0, True), (45.0, False)])
def test_a_sphere_holds_or_rolls_on_a_slope_depending_on_rolling_resistance(angle, holds):
    """Rolling resistance, not sliding friction, sets where a sphere starts to move: the holding angle follows
    ``tan(theta) < mu_roll``.  This is the standard DEM calibration of the rolling friction coefficient."""
    theta = math.radians(angle)
    normal = np.array([math.sin(theta), -math.cos(theta), 0.0])          # free space above the tilted plane
    material = _material(restitution=0.05, mu=0.4, mu_roll=0.6)
    world = World(gravity=(0.0, -9.81, 0.0), walls=[Wall(normal[None, :], np.array([0.0]), material=material)])
    world.add(Sphere(0.01, density=2500.0), position=(0.0, 0.01001 / math.cos(theta), 0.0),
              material=material)
    world.finalize()
    start = world.position[0].copy()
    world.run(steps=int(0.2 / 2.0e-5), dt=2.0e-5)
    moved = float(np.linalg.norm(world.position[0] - start))
    if holds:
        assert moved < 5.0e-3
    else:
        assert moved > 1.0e-2


# ------------------------------------------------------------------------------------------------ breakage
def _bonded_pair(restitution=0.1, sigma_max=2.0e6):
    material = _material(restitution=restitution)
    world = World(gravity=(0.0, 0.0, 0.0))
    world.add(Sphere(0.01, density=2500.0), position=(-0.01, 0.0, 0.0), velocity=(-0.5, 0.0, 0.0),
              material=material)
    world.add(Sphere(0.01, density=2500.0), position=(0.01, 0.0, 0.0), velocity=(0.5, 0.0, 0.0),
              material=material)
    world.add_bond(0, 1, bond_material=BondMaterial(E=2.0e9, sigma_max=sigma_max, tau_max=sigma_max / 2.0))
    world.finalize()
    return world


def test_a_bond_holds_below_its_strength_and_breaks_above_it():
    strong = _bonded_pair(sigma_max=2.0e9)
    strong.run(steps=2000, dt=5.0e-7)
    assert int(strong.bond_state["intact"].sum()) == 1                 # a 2 GPa bond survives this impact
    assert fragment_stats(strong)["fragments"] == 1

    weak = _bonded_pair(sigma_max=1.0e3)     # breaks at a stretch of sigma_max * L / E = 10 nm
    weak.run(steps=2000, dt=5.0e-7)
    assert int(weak.bond_state["intact"].sum()) == 0                   # a 1 kPa bond does not
    stats = fragment_stats(weak)
    assert stats["fragments"] == 2
    assert stats["broken_bonds"] == 1
    assert np.isclose(stats["masses"].sum(), weak.total_mass(), rtol=1e-12)


def test_fragment_ids_separate_a_broken_cluster_and_p80_is_the_eighty_percent_passing_size():
    world = World(gravity=(0.0, 0.0, 0.0))
    material = _material()
    for k in range(3):
        world.add(Sphere(0.01, density=2500.0), position=(0.02 * k, 0.0, 0.0), material=material)
    world.add_bond(0, 1, sigma_max=2.0e6, tau_max=1.0e6)
    world.add_bond(1, 2, sigma_max=2.0e6, tau_max=1.0e6)
    world.finalize()
    assert len(set(fragment_ids(world))) == 1
    world.bond_state["intact"][:] = False                              # break everything by hand
    labels = fragment_ids(world)
    assert len(set(labels)) == 3
    # P80 of an equal-mass binary distribution: the interpolated diameter at 80 % passing
    diameters = np.array([0.01, 0.02])
    masses = np.array([1.0, 1.0])
    assert p80(diameters, masses) == pytest.approx(0.016)              # 50 % -> 100 % interpolates to 80 %
    assert p80(np.array([0.03]), np.array([1.0])) == pytest.approx(0.03)
    assert p80(np.zeros(0), np.zeros(0)) == 0.0


def test_hertz_force_matches_the_closed_form_while_the_bodies_are_held_apart():
    material = _material()
    model = ForceModel(mode="hertz")
    world = World(gravity=(0.0, 0.0, 0.0), model=model)
    radius, overlap = 0.01, 1.0e-4
    world.add(Sphere(radius, density=2500.0), position=(-(radius - overlap / 2.0), 0.0, 0.0),
              material=material)
    world.add(Sphere(radius, density=2500.0), position=(+(radius - overlap / 2.0), 0.0, 0.0),
              material=material)
    world.finalize()
    world.step(1.0e-9)                                                 # the first step resolves the contact
    pairs = np.array([[0, 1]])
    force, _, _, _, _, _ = world._sphere_sphere(pairs, 1.0e-9)
    e_star = material.effective_modulus(material)
    r_star = radius / 2.0
    expected = 4.0 / 3.0 * e_star * math.sqrt(r_star) * overlap ** 1.5
    # the dashpot is zero at zero approach speed, so the elastic part alone is what the force reports here
    assert force[0, 0] == pytest.approx(-expected, rel=0.02)
    assert force[1, 0] == pytest.approx(expected, rel=0.02)
