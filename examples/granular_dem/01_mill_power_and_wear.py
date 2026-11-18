"""A tumbling mill: power draw against speed, liner wear, and the fragments of a breakage run.

This is the process picture the granular solver exists for.  Three measurements come out of one drum:

* **power draw** -- the torque the charge exerts on the shell times the rotation rate, measured from the contacts
  rather than from a correlation, and rising with speed as the charge is carried higher;
* **liner wear** -- the impulse each barrel facet took, turned into a wear depth by the impulse wear law, and the
  wear coefficient recovered from a synthetic scanned profile;
* **product curve** -- with bonded agglomerates in the charge, the fragments they break into and their ``P80``.

Run it with ``python examples/granular_dem/01_mill_power_and_wear.py``.
"""
from __future__ import annotations

import numpy as np

from pinneapple_simulation.granular import (
    BondMaterial,
    Material,
    RotatingDrum,
    Sphere,
    Wall,
    WearTracker,
    World,
    fit_wear_law,
    fragment_stats,
)
from pinneapple_simulation.granular.equipment import charge_positions

RPM_SWEEP = (10.0, 40.0, 70.0)
SETTLE_STEPS = 900          # the charge has to bed down before it can be measured
STEPS = 500
RADIUS, LENGTH = 0.15, 0.045
PARTICLE_RADIUS = 0.01
N_PARTICLES = 60


def power_sweep() -> list[dict]:
    """One mill per speed: same charge, same barrel, only the drive changes."""
    rows = []
    for rpm in RPM_SWEEP:
        drum = RotatingDrum(radius=RADIUS, length=LENGTH, rpm=rpm, lifters=6)
        world = drum.build(n_particles=N_PARTICLES, particle_radius=PARTICLE_RADIUS, seed=0)
        report = drum.run(world, steps=STEPS, settle_steps=SETTLE_STEPS)
        centre = (world.mass[:, None] * world.position).sum(0) / world.mass.sum()
        rows.append({
            "rpm": rpm,
            "omega": report["omega"],
            "torque": report["torque"],
            "power": report["power"],
            "charge_height": float(centre[1]),
            "max_strain": float(np.max(report["history"]["max_penetration_ratio"][len(
                report["history"]["max_penetration_ratio"]) // 2:])),
            "mass": report["charge_mass"],
        })
    return rows


def wear_case() -> dict:
    """Track the impulse on every liner patch, then fit the wear law a laser scan would calibrate."""
    drum = RotatingDrum(radius=RADIUS, length=LENGTH, rpm=50.0, lifters=6)
    world = drum.build(n_particles=N_PARTICLES, particle_radius=PARTICLE_RADIUS, seed=1)
    drum.run(world, steps=STEPS, settle_steps=SETTLE_STEPS)
    hardness = 2.0e9
    tracker = WearTracker(hardness=hardness, k_wear=1.0e-9)
    predicted = tracker.update(world)
    areas = tracker.areas_of(world)
    # stand-in for the laser scan: a different wear coefficient and a little measurement noise
    rng = np.random.default_rng(0)
    scanned = 4.5e-9 * tracker.impulse / (hardness * areas)
    scanned = scanned * (1.0 + rng.normal(scale=0.02, size=scanned.shape))
    fit = fit_wear_law(tracker.impulse, areas, scanned, hardness)
    hours_per_run = world.time / 3600.0
    hottest = int(np.argmax(predicted))
    return {
        "hardness": hardness,
        "patches": int(len(predicted)),
        "impulse_total": float(tracker.impulse.sum()),
        "max_depth_mm_per_1000h": float(predicted.max() / max(hours_per_run, 1.0e-30) * 1000.0 * 1e3),
        "hottest_patch": hottest,
        "fitted_k": fit["k_wear"],
        "fitted_r2": fit["r2"],
        "true_k": 4.5e-9,
    }


def breakage_case() -> dict:
    """A drop-shatter test: bonded agglomerates fall onto the shell and break.

    The mill's own breakage needs minutes of physical time, so the fragment machinery is shown on the standard
    attrition test instead: four bonded agglomerates released from 40 mm, and what is left of them one impact
    later -- the fragment count, the bond survival and the ``P80`` of the product.
    """
    world = World(gravity=(0.0, -9.81, 0.0),
                  walls=[Wall(np.array([[0.0, -1.0, 0.0]]), np.array([0.0]),
                              material=Material(E=2.0e11, density=7850.0, nu=0.3, mu=0.5))])
    body_radius = 0.005
    per_agglomerate = 10
    material = Material(density=2600.0, E=8.0e8, restitution=0.2, mu=0.6, mu_roll=0.1)
    bond = BondMaterial(E=5.0e8, sigma_max=5.0e5, tau_max=2.5e5)
    centres = np.array([[0.0, 0.045, 0.0], [0.032, 0.045, 0.0], [-0.032, 0.045, 0.0], [0.0, 0.045, 0.032]])
    for k, centre in enumerate(centres):
        cloud = charge_positions(per_agglomerate, 0.023, 0.023, body_radius, jitter=0.05, seed=10 + k)
        cloud -= cloud.mean(0)
        start = len(world._pending)
        for offset in cloud:
            world.add(Sphere(body_radius, density=material.density), position=centre + offset,
                      material=material)
        for a in range(len(cloud)):
            for b in range(a + 1, len(cloud)):
                if np.linalg.norm(cloud[a] - cloud[b]) < 2.1 * body_radius:
                    world.add_bond(start + a, start + b, bond_material=bond)
    world.finalize()
    before = fragment_stats(world)
    world.run(steps=int(0.06 / world.recommended_timestep(0.2)), dt=world.recommended_timestep(0.2))
    stats = fragment_stats(world)
    return {
        "bodies": int(world.n),
        "fragments_before": before["fragments"],
        "fragments": stats["fragments"],
        "intact_bonds": stats["intact_bonds"],
        "broken_bonds": stats["broken_bonds"],
        "p80_mm_before": float(before["p80"] * 1e3),
        "p80_mm": float(stats["p80"] * 1e3),
        "mass_conserved": float(stats["masses"].sum() / max(stats["bonded_mass"], 1.0e-30)),
    }


def main() -> dict:
    print("mill power draw")
    print(f"{'rpm':>5} {'omega rad/s':>12} {'torque N m':>11} {'power W':>9} {'charge y/R':>11} {'max strain':>10}")
    sweep = power_sweep()
    for row in sweep:
        print(f"{row['rpm']:5.1f} {row['omega']:12.3f} {row['torque']:11.4f} {row['power']:9.4f} "
              f"{row['charge_height'] / 0.1495:11.3f} {row['max_strain']:10.3f}")

    wear = wear_case()
    print("\nliner wear (impulse law, calibrated on a synthetic scan)")
    print(f"  patches {wear['patches']}, hottest patch {wear['hottest_patch']}, "
          f"total impulse {wear['impulse_total']:.3f} N s, "
          f"hottest-patch wear {wear['max_depth_mm_per_1000h']:.3e} mm per 1000 h of operation")
    print(f"  fitted k_wear {wear['fitted_k']:.3e} (scan made with {wear['true_k']:.3e}), R^2 {wear['fitted_r2']:.4f}")

    breakage = breakage_case()
    print("\ndrop-shatter test of bonded agglomerates (released from 40 mm)")
    print(f"  {breakage['bodies']} bodies: {breakage['fragments_before']} fragments -> {breakage['fragments']} "
          f"fragments, {breakage['broken_bonds']} of "
          f"{breakage['broken_bonds'] + breakage['intact_bonds']} bonds broken, "
          f"P80 {breakage['p80_mm_before']:.2f} -> {breakage['p80_mm']:.2f} mm, "
          f"mass recovered {breakage['mass_conserved']:.6f}")

    # power draw, not torque, is the monotone quantity: the torque can fall while the charge cataracts, and
    # P = torque * omega keeps rising -- which is the number a mill circuit is designed around
    monotone = all(b["power"] > a["power"] for a, b in zip(sweep, sweep[1:], strict=False))
    print(f"\nchecks: power rises with speed {monotone}; "
          f"wear fit R^2 {wear['fitted_r2']:.3f}; mass conserved {abs(breakage['mass_conserved'] - 1) < 1e-9}; "
          f"fragments increased {breakage['fragments'] > breakage['fragments_before']}")
    return {"sweep": sweep, "wear": wear, "breakage": breakage}


if __name__ == "__main__":
    main()
