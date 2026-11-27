"""Process equipment: a rotating drum / tumbling mill, its charge, and the numbers a plant asks for.

The mill here is the shape the industry actually runs -- a barrel with lifters and a charge of particles that
cascades and cataracts -- and the measurement that matters is the one every mill circuit is designed around: the
mechanical power drawn through the shell, ``P = torque . omega``, taken from the contact torques on the wall rather
than from a fitted correlation.  With bonded particles in the charge the same run returns a product curve (the
``P80`` of the fragments), which is how a DEM comminution model is used in practice.

    >>> from pinneapple_simulation.granular.equipment import RotatingDrum, cascade_power
    >>> drum = RotatingDrum(radius=0.15, length=0.04, rpm=30.0, lifters=6)
    >>> w = drum.build(n_particles=120, particle_radius=0.01, seed=0)
    >>> result = drum.run(w, steps=400, dt=2.0e-5)          # doctest: +SKIP
    >>> result["power"] > 0.0                               # doctest: +SKIP
    True

The charge is placed on a jittered close packing inside the barrel, neither poured nor settled: a short settle
phase (``RotatingDrum.settle``) is enough for a power number, and a longer one is what a quantitative comparison
against a plant measurement needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contacts import ForceModel, Material, Wall
from .shapes import ConvexPolyhedron, Sphere, box
from .world import World

__all__ = ["RotatingDrum", "charge_positions", "cascade_power"]


def charge_positions(n_particles: int, radius: float, length: float, particle_radius: float,
                     jitter: float = 0.1, seed: int = 0) -> np.ndarray:
    """A non-overlapping packing of up to ``n_particles`` spheres inside a cylinder along ``z``.

    Candidates come from a jittered lattice; a candidate is kept only if it clears every already-placed body by
    ``2.04 radius``.  Starting a run with interpenetrating bodies is the classic way to make a DEM run explode, so
    the packing is screened here instead of being cleaned up by a repulsive phase in the solver.
    """
    rng = np.random.default_rng(seed)
    spacing = 2.0 * particle_radius * 1.02
    n_side = int(math.ceil(2.0 * radius / spacing)) + 1
    layers = max(1, int(math.floor(length / spacing)))
    grid = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side), np.arange(layers), indexing="ij"),
                    axis=-1).reshape(-1, 3).astype(float)
    grid = grid * spacing
    grid[:, 0] -= 0.5 * (n_side - 1) * spacing
    grid[:, 1] -= 0.5 * (n_side - 1) * spacing
    grid[:, 2] -= 0.5 * (layers - 1) * spacing
    inside = (np.hypot(grid[:, 0], grid[:, 1]) + particle_radius) < radius
    inside &= np.abs(grid[:, 2]) + particle_radius < length / 2.0 + 1.0e-12
    grid = grid[inside]
    rng.shuffle(grid)
    kept: list[np.ndarray] = []
    min_gap = 2.04 * particle_radius
    for candidate in grid:
        if len(kept) >= n_particles:
            break
        if kept:
            distances = np.linalg.norm(np.array(kept) - candidate, axis=1)
            if distances.min() < min_gap:
                continue
        kept.append(candidate)
    if len(kept) < n_particles:
        raise ValueError(
            f"the drum holds {len(kept)} bodies of radius {particle_radius} m, not {n_particles}: "
            f"lower n_particles or particle_radius, or lengthen the barrel")
    return np.array(kept)


@dataclass
class RotatingDrum:
    """A horizontal mill barrel (axis ``z``) with lifters, a rotation rate and a wall material."""

    radius: float = 0.15
    length: float = 0.06
    rpm: float = 30.0
    lifters: int = 8
    lifter_height: float = 0.015
    lifter_thickness: float = 0.004
    facets: int = 48
    wall_material: Material = field(default_factory=lambda: Material(E=2.0e11, density=7850.0, nu=0.3,
                                                                   restitution=0.3, mu=0.5))
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def omega(self) -> float:
        """Angular velocity [rad/s]."""
        return 2.0 * math.pi * self.rpm / 60.0

    def wall(self) -> Wall:
        """The barrel (and its lifters) as one rotating set of planes."""
        barrel = Wall.faceted_cylinder(self.radius, self.length, facets=self.facets,
                                       material=self.wall_material, omega=self.omega,
                                       axis=np.array(self.axis, dtype=float),
                                       pivot=np.array(self.pivot, dtype=float), name="barrel")
        if self.lifters > 0:
            barrel = barrel.with_lifters(self.lifters, self.lifter_height, self.lifter_thickness,
                                         self.radius, self.length)
        return barrel

    def build(self, n_particles: int = 200, particle_radius: float = 0.01,
              particle_shape: str | ConvexPolyhedron = "sphere", particle_material: Material | None = None,
              model: ForceModel | None = None, gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
              jitter: float = 0.1, seed: int = 0) -> World:
        """A world holding the mill wall and a charge of ``n_particles`` bodies."""
        material = particle_material or Material(density=2650.0, E=5.0e7, restitution=0.3, mu=0.5)
        world = World(model=model or ForceModel(mode="volume"), gravity=gravity, walls=[self.wall()])
        positions = charge_positions(n_particles, self.radius, self.length, particle_radius, jitter, seed)
        if self.lifters:
            positions = np.array([p for p in positions if self._clears_lifters(p, particle_radius)])
        if particle_shape == "sphere":
            shape = Sphere(particle_radius, density=material.density)
            for p in positions:
                world.add(shape, position=p, material=material)
        else:
            if particle_shape == "box":
                shape = box(1.6 * particle_radius, 1.6 * particle_radius, 1.6 * particle_radius,
                            density=material.density)
            elif particle_shape == "prism":
                from .shapes import regular_prism

                shape = regular_prism(6, 1.3 * particle_radius, 2.0 * particle_radius, density=material.density)
            elif isinstance(particle_shape, ConvexPolyhedron):
                shape = particle_shape
            else:
                raise ValueError(f"unknown particle_shape {particle_shape!r}")
            rng = np.random.default_rng(seed + 1)
            for p in positions:
                world.add(shape, position=p, material=material,
                          rotation=_random_rotation(rng), omega=rng.normal(scale=0.1, size=3))
        world.finalize()
        return world

    def _clears_lifters(self, point: np.ndarray, particle_radius: float) -> bool:
        """True when ``point`` is not inside a lifter box grown by the particle radius."""
        for k in range(self.lifters):
            a = 2.0 * math.pi * k / self.lifters
            er = np.array([math.cos(a), math.sin(a), 0.0])
            et = np.array([-math.sin(a), math.cos(a), 0.0])
            centre = er * (self.radius - self.lifter_height / 2.0)
            local = np.array([(point - centre) @ et, (point - centre) @ er, (point - centre)[2]])
            half = np.array([self.lifter_thickness / 2.0, self.lifter_height / 2.0, self.length / 2.0])
            if np.all(np.abs(local) < half + particle_radius):
                return False
        return True

    def settle(self, world: World, seconds: float = 0.05, dt: float | None = None,
               rpm: float | None = None) -> dict:
        """Turn at ``rpm`` for ``seconds`` so the charge beds down before the measurement window."""
        dt = dt or self.steady_timestep(world)
        omega = 0.0 if rpm is None else 2.0 * math.pi * rpm / 60.0
        original = world.walls[0].omega
        world.walls[0].omega = omega if rpm is not None else original
        try:
            return world.run(steps=max(1, int(seconds / dt)), dt=dt, every=max(1, int(seconds / dt) // 10))
        finally:
            world.walls[0].omega = original

    def steady_timestep(self, world: World, safety: float = 0.2) -> float:
        """Stable step for this charge: ``dt < safety * 2 pi / omega_max`` over contacts, bonds and material."""
        radius = float(world.sphere_radius[world.is_sphere].min()) if world.is_sphere.any() \
            else float(world.radius.min())
        stiffness = self._contact_stiffness_bound(world, radius)
        m_min = float(world.mass.min())
        omega_max = math.sqrt(max(stiffness, 1.0e-12) / m_min)
        return float(min(safety * 2.0 * math.pi / omega_max, world.recommended_timestep(safety=safety)))

    def _contact_stiffness_bound(self, world: World, radius: float) -> float:
        """Secant contact stiffness for the stability estimate, at a penetration a dense charge really reaches."""
        if world.model.mode == "volume" and world.model.volume_stiffness:
            # F = k_v * pi R delta^2  =>  dF/d(delta) = 2 k_v pi R delta
            delta = world.model.rest_penetration * radius
            return 2.0 * float(world.model.volume_stiffness) * math.pi * radius * delta
        E_star = world.material[0].effective_modulus(world.material[0])
        delta = 1.0e-3 * radius
        return 2.0 * E_star * math.sqrt(radius * delta)

    def run(self, world: World, steps: int = 2000, dt: float | None = None, every: int = 1,
            snapshot: bool = False, settle_steps: int = 0) -> dict:
        """Run (optionally after a discarded settling phase) and report power, torque and the product curve."""
        from .bonds import fragment_stats

        dt = dt or self.steady_timestep(world)
        if settle_steps:
            world.run(steps=int(settle_steps), dt=dt)                # bed the charge down, do not measure it
        history = world.run(steps=steps, dt=dt, every=every, snapshot=snapshot)
        torque = np.asarray(history["wall_torque"], dtype=float)     # already the motor torque (see World.step)
        half = torque[len(torque) // 2:] if len(torque) else torque
        mean_torque = float(np.mean(half)) if len(half) else 0.0
        stats = fragment_stats(world)
        return {
            "time": float(world.time),
            "dt": float(dt),
            "steps": int(steps),
            "rpm": float(self.rpm),
            "omega": float(self.omega),
            "torque": mean_torque,
            "power": mean_torque * float(self.omega),               # [W], positive when the drive does work
            "torque_series": torque,
            "history": history,
            "charge_mass": float(world.mass[world.is_sphere].sum() if world.is_sphere.any()
                                 else world.mass.sum()),
            "fragments": stats["fragments"],
            "p80": stats["p80"],
            "intact_bonds": stats["intact_bonds"],
        }


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cascade_power(drum: RotatingDrum, n_particles: int = 200, particle_radius: float = 0.01,
                  steps: int = 2000, dt: float | None = None, seed: int = 0, **kwargs) -> dict:
    """Build, settle and run one mill, returning the :meth:`RotatingDrum.run` report."""
    world = drum.build(n_particles=n_particles, particle_radius=particle_radius, seed=seed, **kwargs)
    return drum.run(world, steps=steps, dt=dt, every=max(1, steps // 20))
