"""Contact materials, force laws and walls for the granular solver.

Two normal-force laws are available and they differ in *what the overlap measures*:

``volume`` (default, and the only one for polyhedral particles)
    ``F_n = k_v * V_overlap`` -- the force is proportional to the volume two bodies share, exactly as it is in the
    volume-based contact model of the GPU DEM literature (Govender et al., the BLAZE-DEM line of work).  A corner
    entering a face, or two faces lying on each other, then behave the same way per unit shared volume, which a
    penetration-depth law does not give you: the depth of a corner-face contact and of a face-face contact differ by
    a large factor for the same shared volume.

``hertz`` (spheres only)
    ``F_n = (4/3) E* sqrt(R* delta) delta`` -- the classical Hertz law.  Kept because it is the standard DEM
    benchmark and because the stiffness follows from a Young's modulus directly instead of from a fitted ``k_v``.

Tangential (Cundall & Strack 1979, Memory: "A discrete numerical model for granular assemblies", Geotechnique
29:47-65) keeps a per-pair spring vector, capped by Coulomb friction, so static friction -- and therefore an angle
of repose -- exists.  Rolling resistance is a moment proportional to ``mu_roll * F_n * R_eff`` opposing relative
spin (Ai et al. 2011, Powder Technology 206:205-211).

Units are SI throughout: metres, kilograms, seconds, pascals, newtons, and radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                       # the shape types are only needed for the annotations
    from .shapes import ConvexPolyhedron

__all__ = [
    "Material",
    "ForceModel",
    "Obstacle",
    "Wall",
    "rayleigh_timestep",
    "hertz_normal_force",
    "restitution_to_damping",
]


@dataclass
class Material:
    """Bulk elastic and tribological properties of one species of body."""

    E: float = 5.0e7                     # Young's modulus [Pa]  (50 MPa: a soft mineral / polymer granule)
    nu: float = 0.3                      # Poisson ratio
    density: float = 2500.0              # [kg/m^3]
    restitution: float = 0.5             # normal coefficient of restitution
    mu: float = 0.4                      # sliding friction coefficient
    mu_roll: float = 0.05                # rolling resistance coefficient (moment = mu_roll * F_n * R_eff)
    mu_twist: float = 0.02               # twisting resistance coefficient (moment = mu_twist * F_n * R_eff)

    def __post_init__(self) -> None:
        if self.E <= 0 or self.density <= 0:
            raise ValueError("Material needs a positive Young's modulus and density")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("Poisson ratio must be in (-1, 0.5)")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")

    @property
    def G(self) -> float:
        """Shear modulus [Pa]."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        """Bulk modulus [Pa]."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def effective_modulus(self, other: Material) -> float:
        """``1 / E*`` of the Hertz contact of two different materials."""
        return 1.0 / ((1.0 - self.nu ** 2) / self.E + (1.0 - other.nu ** 2) / other.E)

    def effective_shear(self, other: Material) -> float:
        return 1.0 / ((2.0 - self.nu) / self.G + (2.0 - other.nu) / other.G)


def restitution_to_damping(restitution: float) -> float:
    """Viscous damping ratio that gives the requested coefficient of restitution (log-decrement inversion)."""
    e = min(max(float(restitution), 1.0e-6), 1.0)
    if e >= 1.0:
        return 0.0
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)


def hertz_normal_force(E_star: float, R_star: float, penetration: np.ndarray) -> np.ndarray:
    """Hertz normal force ``(4/3) E* sqrt(R* delta^(3/2))`` for a non-negative penetration array."""
    d = np.maximum(penetration, 0.0)
    return 4.0 / 3.0 * E_star * np.sqrt(R_star) * d ** 1.5


def rayleigh_timestep(radius: float, material: Material, safety: float = 0.2) -> float:
    """A stable DEM time step: ``safety`` times the Rayleigh surface-wave time of a body of ``radius``.

    ``t_R = pi R sqrt(rho / G) / (0.1631 nu + 0.8766)`` follows from the Rayleigh wave speed of an elastic sphere and
    is the usual upper bound for the DEM step (Johnson 1985, *Contact Mechanics*; O'Sullivan, *Particulate Discrete
    Element Modelling*).  ``safety`` of 0.1-0.2 keeps a stiff contact comfortably resolved.
    """
    t_r = math.pi * radius * math.sqrt(material.density / material.G) / (0.1631 * material.nu + 0.8766)
    return safety * t_r


@dataclass
class ForceModel:
    """How contacts become forces.

    ``mode``        ``"volume"`` (F = k_v * V_overlap) or ``"hertz"`` (spheres only).
    ``volume_stiffness``  ``k_v`` in N/m^3; ``None`` derives it from the material and the smallest particle so that
                   a body resting under its own weight sinks by ``rest_penetration`` of its own volume.
    ``damping``     fraction of critical damping (``None`` uses each pair's restitution).
    ``friction``    overrides the pair friction coefficient when not ``None``.
    ``kt_ratio``    tangential stiffness as a fraction of the secant normal stiffness.
    """

    mode: str = "volume"
    volume_stiffness: float | None = None
    damping: float | None = None
    friction: float | None = None
    rolling: float | None = None
    kt_ratio: float = 0.4
    rest_penetration: float = 0.01
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.mode not in ("volume", "hertz"):
            raise ValueError("ForceModel.mode must be 'volume' or 'hertz'")
        if self.volume_stiffness is not None and self.volume_stiffness <= 0:
            raise ValueError("volume_stiffness must be positive")

    def stiffness_for(self, mass: np.ndarray | float, volume: np.ndarray | float) -> float:
        """``k_v`` that lets the heaviest body rest on another with a penetration of ``rest_penetration * radius``.

        A sphere sunk by ``delta`` into a flat surface shares a cap of volume ``pi R delta^2``, so demanding
        ``delta = rest_penetration * R`` gives ``k_v = m g / (pi R^3 rest_penetration^2)``.  Default 1 % of the
        radius is the usual DEM compromise: stiff enough that the packing does not soften, soft enough that the
        step stays in the tens of microseconds.
        """
        m = float(np.max(np.atleast_1d(mass)))
        v = float(np.min(np.atleast_1d(volume)))
        radius = (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0)
        cap = math.pi * radius ** 3 * self.rest_penetration ** 2
        return m * self.gravity / max(cap, 1.0e-30)


@dataclass
class Obstacle:
    """A finite convex body fixed to a wall's frame (a mill lifter, a baffle, a screw flight).

    Half-space planes are the right tool for a *surface*, but not for a body of finite size: the half-space of a
    20 mm lifter flank would cut through the whole mill.  An obstacle is a real convex polyhedron, so a particle is
    pushed out of the body itself and everything else stays open.
    """

    shape: ConvexPolyhedron
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    name: str = "obstacle"

    def placed(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World rotation and position for a wall frame turned by ``R`` and shifted by ``t``."""
        return R @ self.rotation, R @ self.position + t


@dataclass
class Wall:
    """A rigid set of half-space constraints ``normal . x <= offset`` that can rotate and translate.

    The planes are stored in the wall's own frame; ``theta`` turns the body about ``axis`` through ``pivot``, which
    is how a mill barrel, its lifters and a screw shaft all move.  ``omega`` is the angular velocity used for
    friction: the wall surface at a point ``p`` moves with ``omega x (p - pivot)``.
    """

    normals: np.ndarray
    offsets: np.ndarray
    name: str = "wall"
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    omega: float = 0.0
    material: Material = field(default_factory=Material)
    patches: np.ndarray | None = None      # optional patch id per plane, for wear bookkeeping
    obstacles: list = field(default_factory=list)
    areas: np.ndarray | None = None        # area of each plane [m^2], needed to turn impulse into a wear depth

    def __post_init__(self) -> None:
        self.normals = np.atleast_2d(np.asarray(self.normals, dtype=float))
        self.offsets = np.atleast_1d(np.asarray(self.offsets, dtype=float))
        if len(self.normals) != len(self.offsets):
            raise ValueError("Wall needs one offset per normal")
        n = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals = self.normals / np.maximum(n, 1.0e-30)
        self.patches = (np.arange(len(self.offsets)) if self.patches is None
                        else np.asarray(self.patches, dtype=int))
        self._axis = np.asarray(self.axis, dtype=float)
        norm = np.linalg.norm(self._axis)
        self._axis = self._axis / (norm if norm > 0 else 1.0)

    # ---- kinematics -------------------------------------------------------------------------------------
    def rotation(self, theta: float) -> np.ndarray:
        """Rotation matrix of the wall frame at turned angle ``theta``."""
        from .shapes import rotation_matrix

        return rotation_matrix(self._axis, theta)

    def planes(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """World normals and offsets at turned angle ``theta``."""
        R = self.rotation(theta)
        t = self.pivot - R @ self.pivot
        normals = self.normals @ R.T
        offsets = self.offsets + normals @ t
        return normals, offsets

    def surface_velocity(self, theta: float, points: np.ndarray) -> np.ndarray:
        """Velocity of the wall material at world ``points`` (``(n, 3)``), for friction and power draw."""
        p = np.atleast_2d(np.asarray(points, dtype=float)) - self.pivot
        w = self._axis * float(self.omega)
        return np.stack([w[1] * p[:, 2] - w[2] * p[:, 1],
                         w[2] * p[:, 0] - w[0] * p[:, 2],
                         w[0] * p[:, 1] - w[1] * p[:, 0]], axis=1) * np.ones(p.shape)

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._axis * float(self.omega)

    # ---- constructors -----------------------------------------------------------------------------------
    @classmethod
    def faceted_cylinder(cls, radius: float, length: float, facets: int = 48, *,
                         material: Material | None = None, omega: float = 0.0,
                         axis: np.ndarray | None = None, pivot: np.ndarray | None = None,
                         end_caps: bool = True, name: str = "barrel") -> Wall:
        """The inner surface of a cylinder, as ``facets`` planes plus two end caps.

        A faceted wall is what a plane-based contact model can represent exactly; with 48 facets the inscribed
        polygon sits 0.2 % inside the true circle, which is far below the contact resolution of a DEM run and lets
        every wall contact reuse the same analytic/clipping kernels as a flat wall.
        """
        if facets < 6:
            raise ValueError("faceted_cylinder needs at least 6 facets")
        ax = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        pv = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=float)
        # local frame with z along the cylinder axis
        e1 = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = e1 - ax * (e1 @ ax)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        basis = np.c_[e1, e2, ax]
        th = np.linspace(0.0, 2.0 * math.pi, facets, endpoint=False) + math.pi / facets
        r_apothem = radius * math.cos(math.pi / facets)          # inscribed polygon: faces touch the circle
        normals = np.c_[np.cos(th), np.sin(th), np.zeros(facets)] @ basis.T
        offsets = np.full(facets, r_apothem)
        areas = np.full(facets, 2.0 * radius * math.sin(math.pi / facets) * length)
        if end_caps:
            normals = np.vstack([normals, ax, -ax])
            offsets = np.r_[offsets, length / 2.0, length / 2.0]
            cap_area = 0.5 * facets * radius ** 2 * math.sin(2.0 * math.pi / facets)
            areas = np.r_[areas, cap_area, cap_area]
        offsets = offsets + normals @ pv                          # the body frame is centred on the pivot
        return cls(normals, offsets, name=name, pivot=pv, axis=ax,
                   omega=omega, material=material or Material(), areas=areas)

    def with_lifters(self, count: int, height: float, thickness: float, radius: float,
                     length: float) -> Wall:
        """Add ``count`` rectangular lifters standing ``height`` off the barrel wall.

        Each lifter is a finite box (a :class:`Obstacle`), not a set of planes: a half-space for a 4 mm flank would
        slice the whole barrel in two, whereas a box pushes a particle out of the 4 mm body and leaves the rest of
        the mill open.
        """
        from .shapes import box

        obstacles = []
        for k in range(count):
            a = 2.0 * math.pi * k / count
            er = np.array([math.cos(a), math.sin(a), 0.0])
            et = np.array([-math.sin(a), math.cos(a), 0.0])
            z = np.array([0.0, 0.0, 1.0])
            centre = er * (radius - height / 2.0)
            basis = np.c_[et, er, z]                                # box x -> tangential, y -> radial, z -> axial
            obstacles.append(Obstacle(box(thickness, height, length), position=centre, rotation=basis,
                                      name=f"lifter{k}"))
        normals = self.normals
        offsets = self.offsets
        return Wall(normals, offsets, name=self.name + "+lifters", pivot=self.pivot, axis=self._axis,
                    omega=self.omega, material=self.material, patches=self.patches,
                    obstacles=list(self.obstacles) + obstacles, areas=self.areas)


def _plane_from_triangle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, float]:  # pragma: no cover
    n = np.cross(p1 - p0, p2 - p0)
    n = n / max(np.linalg.norm(n), 1.0e-30)
    return n, float(n @ p0)
