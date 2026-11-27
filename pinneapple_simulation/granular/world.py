"""The granular world: bodies, neighbour search, contacts, time integration and the measurements that matter.

A :class:`World` holds rigid bodies (spheres and convex polyhedra) plus rigid walls, resolves contacts with the
volume-based (or Hertz) laws of :mod:`pinneapple_simulation.granular.contacts`, and integrates a semi-implicit
scheme: the velocity is updated with the forces of the current configuration and the position then follows.  That
is the standard choice for stiff DEM (Cundall & Strack 1979; O'Sullivan, *Particulate Discrete Element Modelling*).

Plain NumPy over flat arrays: the sphere-sphere, sphere-wall and polyhedron-wall kernels are vectorised over all
pairs at once; only polyhedron-polyhedron pairs (whose exact overlap volume needs a clipping pass) run in a Python
loop.  On one core this carries a few thousand particles at the ~10 microsecond step a DEM run needs, and the same
arrays are what a GPU port would upload -- the next step listed in ``docs/granular_dem.md``.

Contact bookkeeping follows the array convention of production DEM codes: contacts are identified by integer keys,
the tangential spring vector of each contact is carried over by a sorted-key lookup (not a Python dictionary) for
the vectorised kernels, so static friction survives from step to step at no per-contact Python cost.

    >>> from pinneapple_simulation.granular import World, Material, Sphere
    >>> w = World(gravity=(0.0, -9.81, 0.0))
    >>> _ = w.add(Sphere(0.01, density=2500.0), position=(0.0, 0.20, 0.0))
    >>> w.finalize()
    >>> hist = w.run(steps=50, dt=2.0e-5)
    >>> hist["kinetic_energy"][0] < 1.0e-9 and hist["kinetic_energy"][-1] > 0.0
    True
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .contacts import ForceModel, Material, Wall, rayleigh_timestep, restitution_to_damping
from .overlap import (
    clipped_by_plane,
    overlap_volume,
    sphere_plane_cap_volume,
    sphere_sphere_lens_volume,
)
from .shapes import ConvexPolyhedron, Sphere

__all__ = ["World", "StepDiagnostics", "closest_point_on_polyhedron"]

_EPS = 1.0e-15


def _skew(v: np.ndarray) -> np.ndarray:
    """Vectorised skew matrices: ``(n, 3)`` -> ``(n, 3, 3)`` with ``skew(a) b = a x b``."""
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two ``(n, 3)`` arrays (``np.cross`` spends most of its time on axis bookkeeping)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    out = np.empty((a.shape[0], 3))
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _scatter(idx: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Sum ``values`` (m, 3) into rows ``idx`` of an (n, 3) array -- one bincount for all three components."""
    if len(idx) == 0:
        return np.zeros((n, 3))
    flat = (idx[:, None] * 3 + np.arange(3)).ravel()
    return np.bincount(flat, weights=values.ravel(), minlength=3 * n).reshape(n, 3)


def _closest_point_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest points from one point ``p`` to a batch of triangles (batch of ``(m, 3)`` vertices)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    out = np.empty_like(a)
    done = np.zeros(len(a), dtype=bool)

    def put(mask: np.ndarray, value: np.ndarray) -> None:
        nonlocal done
        take = mask & ~done
        out[take] = value[take]
        done = done | take

    with np.errstate(divide="ignore", invalid="ignore"):
        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        t1 = np.where(np.abs(d1 - d3) > _EPS, d1 / np.where(np.abs(d1 - d3) > _EPS, d1 - d3, 1.0), 0.0)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t1[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        t2 = np.where(np.abs(d2 - d6) > _EPS, d2 / np.where(np.abs(d2 - d6) > _EPS, d2 - d6, 1.0), 0.0)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t2[:, None] * ac)
        denom_bc = (d4 - d3) + (d5 - d6)
        t3 = np.where(np.abs(denom_bc) > _EPS, (d4 - d3) / np.where(np.abs(denom_bc) > _EPS, denom_bc, 1.0), 0.0)
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + t3[:, None] * (c - b))
        sumv = va + vb + vc
        safe = np.where(np.abs(sumv) > _EPS, sumv, 1.0)
        v = vb / safe
        w = vc / safe
        put(np.ones(len(a), dtype=bool), a + v[:, None] * ab + w[:, None] * ac)
    return out


def closest_point_on_polyhedron(poly: ConvexPolyhedron, rotation: np.ndarray, position: np.ndarray,
                                point: np.ndarray) -> np.ndarray:
    """Closest point of a (world-placed) convex polyhedron to ``point``."""
    verts = poly.world_vertices(rotation, position)
    tris = [(face[0], face[k], face[k + 1]) for face in poly.faces for k in range(1, len(face) - 1)]
    idx = np.array(tris)
    arr = verts[idx]
    cand = _closest_point_triangle(np.asarray(point, dtype=float), arr[:, 0], arr[:, 1], arr[:, 2])
    dist = ((cand - point) ** 2).sum(1)
    return cand[int(np.argmin(dist))]


@dataclass
class StepDiagnostics:
    """What one step measured."""

    time: float
    kinetic_energy: float
    gravity_energy: float
    contacts: int
    max_penetration_ratio: float
    wall_torque: float
    momentum: np.ndarray

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "kinetic_energy": self.kinetic_energy,
            "gravity_energy": self.gravity_energy,
            "contacts": self.contacts,
            "max_penetration_ratio": self.max_penetration_ratio,
            "wall_torque": self.wall_torque,
            "momentum": self.momentum,
        }


class _ContactState:
    """Sorted-key store of the tangential spring vectors of the contacts that persist between steps."""

    def __init__(self) -> None:
        self.keys = np.zeros(0, dtype=np.int64)
        self.xi = np.zeros((0, 3))

    def carry(self, keys: np.ndarray) -> np.ndarray:
        out = np.zeros((len(keys), 3))
        if len(keys) == 0 or len(self.keys) == 0:
            return out
        pos = np.clip(np.searchsorted(self.keys, keys), 0, len(self.keys) - 1)
        hit = self.keys[pos] == keys
        out[hit] = self.xi[pos[hit]]
        return out

    def store(self, keys: np.ndarray, xi: np.ndarray) -> None:
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.xi = xi[order]


class World:
    """Rigid bodies + walls + a contact model, stepped in time."""

    def __init__(self, model: ForceModel | None = None, gravity: Iterable[float] = (0.0, -9.81, 0.0),
                 walls: Iterable[Wall] | None = None, name: str = "world") -> None:
        self.model = model or ForceModel()
        self.gravity = np.asarray(gravity, dtype=float)
        self.name = name
        self.walls: list[Wall] = list(walls or [])
        self.wall_angle: list[float] = [0.0] * len(self.walls)
        self._pending: list[dict] = []
        self._finalized = False
        self._bonds: list[dict] = []
        self.time = 0.0
        self.n = 0
        self._pair_state = _ContactState()
        self._wall_state = _ContactState()
        self._poly_shear: dict[tuple[int, int], np.ndarray] = {}
        self._poly_wall_shear: dict[tuple[int, int], np.ndarray] = {}
        self._touched: set = set()
        self.force_sources: list[Callable] = []
        self._last_wall_torque = np.zeros(len(self.walls))
        self.plane_impulse = np.zeros(0)          # cumulative normal impulse [N s] on each wall plane
        self.obstacle_impulse = np.zeros(0)       # the same for each finite wall obstacle
        self.obstacle_index: list[tuple[int, int]] = []
        self.dt = 1.0e-5

    # ------------------------------------------------------------------ construction
    def add(self, shape: ConvexPolyhedron | Sphere, position: Iterable[float] = (0.0, 0.0, 0.0),
            velocity: Iterable[float] | None = None, omega: Iterable[float] | None = None,
            rotation: np.ndarray | None = None, material: Material | None = None,
            density: float | None = None) -> int:
        """Add one body; returns its index. ``density`` overrides the shape's own density."""
        if self._finalized:
            raise RuntimeError("World.add after finalize(): build the assembly first, then step it")
        self._pending.append({
            "shape": shape,
            "position": np.asarray(position, dtype=float),
            "velocity": np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
            "omega": np.zeros(3) if omega is None else np.asarray(omega, dtype=float),
            "rotation": np.eye(3) if rotation is None else np.asarray(rotation, dtype=float),
            "material": material or Material(),
            "density": density,
        })
        return len(self._pending) - 1

    def add_many(self, shapes: Iterable, positions: Iterable, **kwargs) -> np.ndarray:
        """Add a batch (one shape per row of ``positions``); returns the index array."""
        return np.array([self.add(s, p, **kwargs)
                         for s, p in zip(shapes, positions, strict=True)], dtype=int)

    def add_wall(self, wall: Wall) -> int:
        self.walls.append(wall)
        self.wall_angle.append(0.0)
        self._last_wall_torque = np.zeros(len(self.walls))
        if self._finalized:                                       # walls added after finalize: rebuild the tables
            self._plane_offsets = np.cumsum([0] + [len(w.offsets) for w in self.walls])
            self._n_planes = int(sum(len(w.offsets) for w in self.walls))
            self.plane_impulse = np.zeros(self._n_planes)
            self.obstacle_index = [(w, o) for w, wall in enumerate(self.walls)
                                   for o in range(len(wall.obstacles))]
            self.obstacle_impulse = np.zeros(len(self.obstacle_index))
            self._obstacle_lookup = {key: k for k, key in enumerate(self.obstacle_index)}
        return len(self.walls) - 1

    def add_force_source(self, source: Callable) -> int:
        """Register an external force field: ``source(world) -> (force, torque)`` or ``force``, added every step.

        This is how the fluid drag of :mod:`pinneapple_simulation.granular.coupling`, a gravity override, or any
        other body force enters without touching the integrator.
        """
        self.force_sources.append(source)
        return len(self.force_sources) - 1

    def add_bond(self, i: int, j: int, **kwargs) -> None:
        """Register a bond between bodies ``i`` and ``j`` (see :mod:`pinneapple_simulation.granular.bonds`)."""
        self._bonds.append({"i": int(i), "j": int(j), **kwargs})

    def finalize(self) -> World:
        """Pack the pending bodies into flat arrays."""
        if not self._pending:
            raise ValueError("no bodies: call add() before finalize()")
        self.n = len(self._pending)
        self.shape = [p["shape"] for p in self._pending]
        self.position = np.array([p["position"] for p in self._pending])
        self.velocity = np.array([p["velocity"] for p in self._pending])
        self.omega = np.array([p["omega"] for p in self._pending])
        self.rotation = np.array([p["rotation"] for p in self._pending])
        self.material = [p["material"] for p in self._pending]
        self.is_sphere = np.array([bool(p["shape"].is_sphere) for p in self._pending])
        self.radius = np.array([float(p["shape"].bounding_radius) for p in self._pending])
        self.sphere_radius = np.array([float(p["shape"].radius) if p["shape"].is_sphere else 0.0
                                       for p in self._pending])
        self.shape_volume = np.array([float(p["shape"].volume) for p in self._pending])
        self.density = np.array([
            float(p["density"]) if p["density"] is not None else float(p["shape"].density)
            for p in self._pending])
        self.mass = self.density * self.shape_volume
        self.inv_mass = 1.0 / self.mass
        self.inertia_body = np.array([
            np.asarray(p["shape"].inertia, dtype=float) * (self.density[k] / max(float(p["shape"].density), _EPS))
            for k, p in enumerate(self._pending)])
        self.inv_inertia_body = np.array([np.linalg.inv(tensor) for tensor in self.inertia_body])
        self._damping = np.array([restitution_to_damping(m.restitution) for m in self.material])
        self._mu = np.array([m.mu for m in self.material])
        self._mu_roll = np.array([m.mu_roll for m in self.material])
        self._compliance = np.array([(1.0 - m.nu ** 2) / m.E for m in self.material])
        if self.model.mode == "hertz" and not self.is_sphere.all():
            raise ValueError("ForceModel(mode='hertz') only covers spheres; polyhedral bodies use the volume model "
                             "(mode='volume')")
        if self.model.mode == "volume" and self.model.volume_stiffness is None:
            self.model.volume_stiffness = self.model.stiffness_for(self.mass, self.shape_volume)
        self._pending = []
        self._finalized = True
        self.force = np.zeros((self.n, 3))
        self.torque = np.zeros((self.n, 3))
        from .bonds import build_bond_arrays

        self.bond_state = build_bond_arrays(self, self._bonds)
        self._plane_offsets = np.cumsum([0] + [len(w.offsets) for w in self.walls])
        self._n_planes = int(sum(len(w.offsets) for w in self.walls))
        self.plane_impulse = np.zeros(self._n_planes)
        self.obstacle_index = [(w, o) for w, wall in enumerate(self.walls) for o in range(len(wall.obstacles))]
        self.obstacle_impulse = np.zeros(len(self.obstacle_index))
        self._obstacle_lookup = {key: k for k, key in enumerate(self.obstacle_index)}
        return self

    @property
    def bonds(self) -> list[dict]:
        return self._bonds

    # ------------------------------------------------------------------ geometry helpers
    def inertia_world(self, k: int) -> np.ndarray:
        R = self.rotation[k]
        return R @ self.inertia_body[k] @ R.T

    def inv_inertia_world(self, k: int) -> np.ndarray:
        R = self.rotation[k]
        return R @ self.inv_inertia_body[k] @ R.T

    @property
    def _inv_inertia_world_all(self) -> np.ndarray:
        Rt = np.transpose(self.rotation, (0, 2, 1))
        return np.matmul(np.matmul(self.rotation, self.inv_inertia_body), Rt)

    @property
    def _inertia_world_all(self) -> np.ndarray:
        Rt = np.transpose(self.rotation, (0, 2, 1))
        return np.matmul(np.matmul(self.rotation, self.inertia_body), Rt)

    def friction_coefficient(self, i: int, j: int) -> float:
        if self.model.friction is not None:
            return float(self.model.friction)
        return math.sqrt(self.material[i].mu * self.material[j].mu)

    def rolling_coefficient(self, i: int, j: int) -> float:
        if self.model.rolling is not None:
            return float(self.model.rolling)
        return math.sqrt(self.material[i].mu_roll * self.material[j].mu_roll)

    def _pair_mu(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if self.model.friction is not None:
            return np.full(len(i), float(self.model.friction))
        return np.sqrt(self._mu[i] * self._mu[j])

    def _pair_mu_roll(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if self.model.rolling is not None:
            return np.full(len(i), float(self.model.rolling))
        return np.sqrt(self._mu_roll[i] * self._mu_roll[j])

    def _pair_modulus(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return 1.0 / (self._compliance[i] + self._compliance[j])

    # ------------------------------------------------------------------ pair search
    def _pairs(self) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial import cKDTree

        tree = cKDTree(self.position)
        pairs = tree.query_pairs(2.0 * float(self.radius.max()) * (1.0 + 1.0e-9), output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
        i, j = pairs[:, 0], pairs[:, 1]
        delta = self.position[i] - self.position[j]
        gap = np.sqrt((delta * delta).sum(1)) - (self.radius[i] + self.radius[j])
        pairs = pairs[gap < 0.0]
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
        both = self.is_sphere[pairs[:, 0]] & self.is_sphere[pairs[:, 1]]
        return pairs[both], pairs[~both]

    # ------------------------------------------------------------------ contacts
    def _resolve(self, dt: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        force = np.zeros((self.n, 3))
        torque = np.zeros((self.n, 3))
        contacts = 0
        strain = 0.0
        self._touched = set()
        ss, other = self._pairs()

        if len(ss):
            f, t, count, s, keys, xi = self._sphere_sphere(ss, dt)
            force += f
            torque += t
            contacts += count
            strain = max(strain, s)
            self._pair_state.store(keys, xi)
        else:
            self._pair_state = _ContactState()

        for i, j in other:
            out = self._mixed_pair(int(i), int(j), dt)
            if out is None:
                continue
            f_i, t_i, f_j, t_j, s = out
            force[i] += f_i
            torque[i] += t_i
            force[j] += f_j
            torque[j] += t_j
            contacts += 1
            strain = max(strain, s)

        wall_contacts, wall_torque, wall_strain = self._walls(dt)
        force += wall_contacts[0]
        torque += wall_contacts[1]
        contacts += int(self._last_wall_contacts)
        strain = max(strain, wall_strain)
        self._last_wall_torque = wall_torque
        self._poly_shear = {k: v for k, v in self._poly_shear.items() if k in self._touched}
        self._poly_wall_shear = {k: v for k, v in self._poly_wall_shear.items() if k in self._touched}
        return force, torque, contacts, strain

    def _sphere_sphere(self, pairs: np.ndarray, dt: float):
        model = self.model
        i, j = pairs[:, 0], pairs[:, 1]
        xi_pos, xj_pos = self.position[i], self.position[j]
        d = xj_pos - xi_pos
        dist = np.maximum(np.sqrt((d * d).sum(1)), _EPS)
        n = d / dist[:, None]
        ri, rj = self.sphere_radius[i], self.sphere_radius[j]
        overlap = ri + rj - dist
        keep = overlap > 0.0
        if not np.any(keep):
            return (np.zeros((self.n, 3)), np.zeros((self.n, 3)), 0, 0.0,
                    np.zeros(0, np.int64), np.zeros((0, 3)))
        i, j, n, overlap, dist, ri, rj = i[keep], j[keep], n[keep], overlap[keep], dist[keep], ri[keep], rj[keep]
        xi_pos, xj_pos = self.position[i], self.position[j]

        point = xi_pos + n * (ri - overlap / 2.0)[:, None]
        lever_i = point - xi_pos
        lever_j = point - xj_pos
        v_rel = (self.velocity[j] + _cross(self.omega[j], lever_j)) - (
            self.velocity[i] + _cross(self.omega[i], lever_i))
        v_n = (v_rel * n).sum(1)
        v_t = v_rel - v_n[:, None] * n

        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        r_eff = ri * rj / (ri + rj)
        if model.mode == "volume":
            F_el = float(model.volume_stiffness) * sphere_sphere_lens_volume(ri, rj, dist)
        else:
            F_el = 4.0 / 3.0 * self._pair_modulus(i, j) * np.sqrt(r_eff) * overlap ** 1.5
        k_secant = np.maximum(F_el / np.maximum(overlap, _EPS), 1.0e-12)
        c_n = 2.0 * self._damping[i] * np.sqrt(m_eff * k_secant)
        F_n = np.maximum(F_el + np.clip(-c_n * v_n, -F_el, F_el), 0.0)

        lo, hi = np.minimum(i, j), np.maximum(i, j)
        keys = lo.astype(np.int64) * self.n + hi.astype(np.int64)
        same = i <= j                                            # v_t is (j relative to i) in the i->j frame
        xi = self._pair_state.carry(keys) + np.where(same[:, None], v_t, -v_t) * dt
        k_t = model.kt_ratio * k_secant
        f_t_canonical = -k_t[:, None] * xi                       # force on the "hi" body
        cap = self._pair_mu(i, j) * F_n
        mag = np.sqrt((f_t_canonical * f_t_canonical).sum(1))
        sliding = mag > cap
        if np.any(sliding):
            scale = np.where(sliding, cap / np.maximum(mag, _EPS), 1.0)
            f_t_canonical = f_t_canonical * scale[:, None]
            xi = xi * scale[:, None]
        f_t = np.where(same[:, None], f_t_canonical, -f_t_canonical)

        f_on_j = F_n[:, None] * n + f_t
        f_on_i = -f_on_j
        force = _scatter(i, f_on_i, self.n) + _scatter(j, f_on_j, self.n)
        torque = _scatter(i, _cross(lever_i, f_on_i), self.n) + _scatter(j, _cross(lever_j, f_on_j), self.n)

        mu_roll = self._pair_mu_roll(i, j)
        spin = self.omega[i] - self.omega[j]
        spin_mag = np.sqrt((spin * spin).sum(1))
        moving = spin_mag > _EPS
        roll = np.zeros((len(i), 3))
        coef = np.where(moving, -mu_roll * F_n * r_eff / np.maximum(spin_mag, _EPS), 0.0)
        roll = coef[:, None] * spin
        torque = torque + _scatter(i, roll, self.n) - _scatter(j, roll, self.n)
        strain = float(np.max(overlap / np.minimum(ri, rj)))
        return force, torque, int(len(i)), strain, keys, xi

    def _mixed_pair(self, i: int, j: int, dt: float):
        """Sphere-polyhedron or polyhedron-polyhedron contact, in the caller's index order."""
        si, sj = self.shape[i], self.shape[j]
        pi, pj = self.position[i], self.position[j]
        Ri, Rj = self.rotation[i], self.rotation[j]
        if si.is_sphere:                                          # si is a sphere, sj a polyhedron
            closest = closest_point_on_polyhedron(sj, Rj, pj, pi)
            d = pi - closest
            dist = float(np.linalg.norm(d))
            r = float(si.radius)
            if dist <= _EPS or dist >= r:
                return None
            n_from_poly = d / dist
            n_ij = -n_from_poly                                  # from body i (sphere) towards body j (polyhedron)
            penetration = r - dist
            point = closest + 0.5 * penetration * n_from_poly
            volume = sphere_plane_cap_volume(r, penetration)
        elif sj.is_sphere:                                        # si is a polyhedron, sj a sphere
            closest = closest_point_on_polyhedron(si, Ri, pi, pj)
            d = pj - closest
            dist = float(np.linalg.norm(d))
            r = float(sj.radius)
            if dist <= _EPS or dist >= r:
                return None
            n_ij = d / dist                                      # from body i (polyhedron) towards body j (sphere)
            penetration = r - dist
            point = closest + 0.5 * penetration * n_ij
            volume = sphere_plane_cap_volume(r, penetration)
        else:
            volume, point, axis = overlap_volume(si, sj, Ri, Rj, pi, pj)
            if volume <= 0.0 or not axis.overlap:
                return None
            n_ij = axis.normal                                   # points from i to j
            penetration = axis.depth
        return self._mixed_forces(i, j, n_ij, point, volume, penetration, dt)

    def _mixed_forces(self, i: int, j: int, n_ij: np.ndarray, point: np.ndarray, volume: float,
                      penetration: float, dt: float):
        model = self.model
        lever_i = point - self.position[i]
        lever_j = point - self.position[j]
        v_rel = (self.velocity[j] + _cross(self.omega[j], lever_j)[0]) - (
            self.velocity[i] + _cross(self.omega[i], lever_i)[0])
        v_n = float(v_rel @ n_ij)
        v_t = v_rel - v_n * n_ij
        F_el = float(model.volume_stiffness) * volume
        k_secant = max(F_el / max(penetration, 1.0e-12), 1.0e-12)
        m_eff = self.mass[i] * self.mass[j] / (self.mass[i] + self.mass[j])
        c_n = 2.0 * self._damping[i] * math.sqrt(m_eff * k_secant)
        F_n = max(F_el + min(max(-c_n * v_n, -F_el), F_el), 0.0)
        if F_n <= 0.0:
            return None
        key = (min(i, j), max(i, j))
        sign = 1.0 if i <= j else -1.0
        xi = self._poly_shear.get(key, np.zeros(3)) + sign * v_t * dt
        k_t = model.kt_ratio * k_secant
        f_t = -k_t * xi
        mu = self.friction_coefficient(i, j)
        mag = float(np.linalg.norm(f_t))
        if mag > mu * F_n:
            f_t = f_t * (mu * F_n / max(mag, _EPS))
            xi = -f_t / k_t
        self._poly_shear[key] = xi
        self._touched.add(key)
        f_on_j = F_n * n_ij + f_t
        f_on_i = -f_on_j
        t_i = _cross(lever_i, f_on_i)[0]
        t_j = _cross(lever_j, f_on_j)[0]
        spin = self.omega[i] - self.omega[j]
        spin_mag = float(np.linalg.norm(spin))
        if spin_mag > _EPS:
            r_eff = 0.5 * (self.radius[i] + self.radius[j])
            M = -self.rolling_coefficient(i, j) * F_n * r_eff * spin / spin_mag
            t_i = t_i + M
            t_j = t_j - M
        strain = penetration / max(min(self.radius[i], self.radius[j]), _EPS)
        return f_on_i, t_i, f_on_j, t_j, float(strain)

    # ------------------------------------------------------------------ walls
    def _walls(self, dt: float) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        force = np.zeros((self.n, 3))
        torque = np.zeros((self.n, 3))
        wall_torque = np.zeros(len(self.walls))
        contacts = 0
        strain = 0.0
        spheres = np.nonzero(self.is_sphere)[0]
        polys = np.nonzero(~self.is_sphere)[0]
        sphere_keys: list[np.ndarray] = []
        sphere_xi: list[np.ndarray] = []
        for w_index, wall in enumerate(self.walls):
            theta = self.wall_angle[w_index]
            normals, offsets = wall.planes(theta)
            base = self._plane_offsets[w_index]
            # --- spheres, vectorised over (particle, plane)
            if len(spheres):
                gap = offsets[None, :] - self.position[spheres] @ normals.T
                mask = gap < self.sphere_radius[spheres][:, None] - 1.0e-15
                rows, cols = np.nonzero(mask)
                if len(rows):
                    plane_id = base + cols
                    out = self._sphere_wall(wall, theta, normals[cols], gap[rows, cols],
                                            spheres[rows], spheres[rows] * self._n_planes + plane_id,
                                            plane_id, dt)
                    if out is not None:
                        f, t, wt, n_contacts, s, keys, xi = out
                        force += f
                        torque += t
                        wall_torque[w_index] += wt
                        contacts += n_contacts
                        strain = max(strain, s)
                        sphere_keys.append(keys)
                        sphere_xi.append(xi)
            # --- polyhedra, one clipping pass per (particle, plane) that can touch
            if len(polys):
                gap = offsets[None, :] - self.position[polys] @ normals.T
                rows, cols = np.nonzero(gap < self.radius[polys][:, None])
                for row, col in zip(rows, cols, strict=True):
                    out = self._poly_wall(int(polys[row]), wall, theta, normals[col], offsets[col],
                                          int(base + col), dt)
                    if out is None:
                        continue
                    f, t, wt, s = out
                    force += f
                    torque += t
                    wall_torque[w_index] += wt
                    contacts += 1
                    strain = max(strain, s)
        self._pending_wall_state: list = []
        obstacle_out = self._obstacles(dt)
        force += obstacle_out[0][0]
        torque += obstacle_out[0][1]
        wall_torque += obstacle_out[1]
        contacts += obstacle_out[2]
        strain = max(strain, obstacle_out[3])
        sphere_keys.extend(k for k, _ in self._pending_wall_state)
        sphere_xi.extend(v for _, v in self._pending_wall_state)
        if sphere_keys:
            self._wall_state.store(np.concatenate(sphere_keys), np.vstack(sphere_xi))
        else:
            self._wall_state = _ContactState()
        self._last_wall_contacts = contacts
        return (force, torque), wall_torque, strain

    # ------------------------------------------------------------------ finite wall obstacles
    def _obstacles(self, dt: float):
        """Contacts with the finite bodies fixed to a wall frame (mill lifters, baffles, flights)."""
        force = np.zeros((self.n, 3))
        torque = np.zeros((self.n, 3))
        wall_torque = np.zeros(len(self.walls))
        contacts = 0
        strain = 0.0
        for w_index, wall in enumerate(self.walls):
            if not wall.obstacles:
                continue
            theta = self.wall_angle[w_index]
            R = wall.rotation(theta)
            t = wall.pivot - R @ wall.pivot
            for o_index, obstacle in enumerate(wall.obstacles):
                R_obs, pos = obstacle.placed(R, t)
                delta = self.position - pos
                reach = obstacle.shape.bounding_radius + self.radius
                candidates = np.nonzero(np.sqrt((delta * delta).sum(1)) < reach)[0]
                boxed = getattr(obstacle.shape, "is_box", False)
                sphere_rows = candidates[self.is_sphere[candidates]] if len(candidates) else candidates
                if boxed and len(sphere_rows):
                    out = self._sphere_box(sphere_rows, obstacle.shape, R_obs, pos, wall, theta,
                                           w_index, o_index, dt)
                    if out is not None:
                        force += out[0]
                        torque += out[1]
                        wall_torque[w_index] += out[2]
                        contacts += int(out[3])
                        strain = max(strain, out[4])
                    candidates = candidates[~self.is_sphere[candidates]] if len(candidates) else candidates
                for row in candidates:
                    out = self._obstacle_contact(int(row), obstacle.shape, R_obs, pos, wall, theta,
                                                 w_index, o_index, dt)
                    if out is None:
                        continue
                    f, tq, wt, s = out
                    force += f
                    torque += tq
                    wall_torque[w_index] += wt
                    contacts += 1
                    strain = max(strain, s)
        return (force, torque), wall_torque, contacts, strain

    def _sphere_box(self, rows: np.ndarray, poly, R_obs: np.ndarray, pos: np.ndarray, wall: Wall,
                    theta: float, w_index: int, o_index: int, dt: float):
        """Sphere against a box obstacle, vectorised: clamp the centre in the box frame, no triangle search."""
        h = poly.half_extents
        local = (self.position[rows] - pos) @ R_obs
        clamped = np.clip(local, -h, h)
        delta_local = local - clamped
        dist = np.sqrt((delta_local * delta_local).sum(1))
        r = self.sphere_radius[rows]
        keep = (dist < r) & (dist > _EPS)
        if not np.any(keep):
            return None
        rows, local, clamped, delta_local, dist, r = (rows[keep], local[keep], clamped[keep],
                                                      delta_local[keep], dist[keep], r[keep])
        n = (delta_local @ R_obs.T) / dist[:, None]                 # from the box towards the sphere
        penetration = r - dist
        point = clamped @ R_obs.T + pos + 0.5 * penetration[:, None] * n
        volume = sphere_plane_cap_volume(r, penetration)
        model = self.model
        lever = point - self.position[rows]
        v_rel = (self.velocity[rows] + _cross(self.omega[rows], lever)) - wall.surface_velocity(theta, point)
        v_n = (v_rel * n).sum(1)
        v_t = v_rel - v_n[:, None] * n
        F_el = float(model.volume_stiffness) * volume
        k_secant = np.maximum(F_el / np.maximum(penetration, _EPS), 1.0e-12)
        c_n = 2.0 * self._damping[rows] * np.sqrt(self.mass[rows] * k_secant)
        F_n = np.maximum(F_el + np.clip(c_n * v_n, -F_el, F_el), 0.0)
        keys = rows.astype(np.int64) * self.n + (-1 - (w_index * 100000 + o_index))
        xi = self._wall_state.carry(keys) + v_t * dt
        k_t = model.kt_ratio * k_secant
        f_t = -k_t[:, None] * xi
        mu = (np.sqrt(self._mu[rows] * wall.material.mu) if model.friction is None
              else np.full(len(rows), float(model.friction)))
        cap = mu * F_n
        mag = np.sqrt((f_t * f_t).sum(1))
        sliding = mag > cap
        if np.any(sliding):
            scale = np.where(sliding, cap / np.maximum(mag, _EPS), 1.0)
            f_t = f_t * scale[:, None]
            xi = xi * scale[:, None]
        self._pending_wall_state.append((keys, xi))
        slot = self._obstacle_lookup.get((w_index, o_index))
        if slot is not None:
            self.obstacle_impulse[slot] += float(np.sum(F_n * dt))
        f_on_particle = F_n[:, None] * n + f_t
        force = _scatter(rows, f_on_particle, self.n)
        torque = _scatter(rows, _cross(lever, f_on_particle), self.n)
        spin = self.omega[rows] - wall.angular_velocity[None, :]
        spin_perp = spin - (spin * n).sum(1)[:, None] * n                  # rolling, not twisting
        spin_mag = np.sqrt((spin_perp * spin_perp).sum(1))
        moving = spin_mag > _EPS
        mu_roll = (self._pair_mu_roll(rows, rows) if model.rolling is None
                   else np.full(len(rows), float(model.rolling)))
        coef = np.where(moving, -mu_roll * F_n * r / np.maximum(spin_mag, _EPS), 0.0)
        torque = torque + _scatter(rows, coef[:, None] * spin_perp, self.n)
        torque_on_wall = _cross(point - wall.pivot, -f_on_particle)
        wall_torque = float(np.sum(torque_on_wall @ wall._axis))
        strain = float(np.max(penetration / np.maximum(r, _EPS)))
        return force, torque, wall_torque, int(len(rows)), strain

    def _obstacle_contact(self, row: int, poly, R_obs: np.ndarray, pos: np.ndarray, wall: Wall,
                          theta: float, w_index: int, o_index: int, dt: float):
        model = self.model
        if self.is_sphere[row]:
            r = float(self.sphere_radius[row])
            closest = closest_point_on_polyhedron(poly, R_obs, pos, self.position[row])
            d = self.position[row] - closest
            dist = float(np.sqrt((d * d).sum()))
            if dist <= _EPS or dist >= r:
                return None
            n = d / dist                                          # from the obstacle towards the particle
            penetration = r - dist
            point = closest + 0.5 * penetration * n
            volume = float(sphere_plane_cap_volume(r, penetration))
        else:
            volume, point, axis = overlap_volume(poly, self.shape[row], R_obs, self.rotation[row], pos,
                                                 self.position[row])
            if volume <= 0.0 or not axis.overlap:
                return None
            n = axis.normal                                       # from the obstacle towards the particle
            penetration = axis.depth
        lever = point - self.position[row]
        v_rel = (self.velocity[row] + _cross(self.omega[row], lever)[0]) - wall.surface_velocity(theta, point)[0]
        v_n = float(v_rel @ n)
        v_t = v_rel - v_n * n
        F_el = float(model.volume_stiffness) * volume
        k_secant = max(F_el / max(penetration, 1.0e-12), 1.0e-12)
        c_n = 2.0 * self._damping[row] * math.sqrt(self.mass[row] * k_secant)
        F_n = max(F_el + min(max(c_n * v_n, -F_el), F_el), 0.0)
        if F_n <= 0.0:
            return None
        key = (row, -1 - (w_index * 100000 + o_index))
        xi = self._poly_wall_shear.get(key, np.zeros(3)) + v_t * dt
        k_t = model.kt_ratio * k_secant
        f_t = -k_t * xi
        mu = (float(model.friction) if model.friction is not None
              else math.sqrt(self.material[row].mu * wall.material.mu))
        mag = float(np.sqrt((f_t * f_t).sum()))
        if mag > mu * F_n:
            f_t = f_t * (mu * F_n / max(mag, _EPS))
            xi = -f_t / k_t
        self._poly_wall_shear[key] = xi
        self._touched.add(key)
        slot = self._obstacle_lookup.get((w_index, o_index))
        if slot is not None:
            self.obstacle_impulse[slot] += F_n * dt
        f_on_particle = F_n * n + f_t
        force = np.zeros((self.n, 3))
        torque = np.zeros((self.n, 3))
        force[row] = f_on_particle
        torque[row] = _cross(lever, f_on_particle)[0]
        spin = self.omega[row] - wall.angular_velocity
        spin_perp = spin - float(spin @ n) * n
        spin_mag = float(np.sqrt(spin_perp @ spin_perp))
        if spin_mag > _EPS:
            mu_roll = float(model.rolling) if model.rolling is not None else \
                math.sqrt(self.material[row].mu_roll * wall.material.mu_roll)
            torque[row] = torque[row] - mu_roll * F_n * self.radius[row] * spin_perp / spin_mag
        wall_torque = float(_cross(point - wall.pivot, -f_on_particle)[0] @ wall._axis)
        strain = penetration / max(self.radius[row], _EPS)
        return force, torque, wall_torque, float(strain)

    def _sphere_wall(self, wall: Wall, theta: float, normals: np.ndarray, gap: np.ndarray,
                     rows: np.ndarray, keys: np.ndarray, plane_id: np.ndarray, dt: float):
        model = self.model
        r = self.sphere_radius[rows]
        penetration = r - gap
        keep = penetration > 0.0
        if not np.any(keep):
            return None
        rows, normals, penetration, gap, keys = rows[keep], normals[keep], penetration[keep], gap[keep], keys[keep]
        plane_id = plane_id[keep]
        r = r[keep]
        point = self.position[rows] + normals * gap[:, None]
        lever = point - self.position[rows]
        v_p = self.velocity[rows] + _cross(self.omega[rows], lever)
        v_rel = v_p - wall.surface_velocity(theta, point)
        v_n = (v_rel * normals).sum(1)
        v_t = v_rel - v_n[:, None] * normals
        if model.mode == "volume":
            F_el = float(model.volume_stiffness) * sphere_plane_cap_volume(r, penetration)
        else:
            wall_compliance = (1.0 - wall.material.nu ** 2) / wall.material.E
            E_star = 1.0 / (self._compliance[rows] + wall_compliance)
            F_el = 4.0 / 3.0 * E_star * np.sqrt(r) * penetration ** 1.5
        k_secant = np.maximum(F_el / np.maximum(penetration, _EPS), 1.0e-12)
        c_n = 2.0 * self._damping[rows] * np.sqrt(self.mass[rows] * k_secant)
        F_n = np.maximum(F_el + np.clip(c_n * v_n, -F_el, F_el), 0.0)
        xi = self._wall_state.carry(keys) + v_t * dt
        k_t = model.kt_ratio * k_secant
        f_t = -k_t[:, None] * xi
        mu = (np.sqrt(self._mu[rows] * wall.material.mu) if model.friction is None
              else np.full(len(rows), float(model.friction)))
        cap = mu * F_n
        mag = np.sqrt((f_t * f_t).sum(1))
        sliding = mag > cap
        if np.any(sliding):
            scale = np.where(sliding, cap / np.maximum(mag, _EPS), 1.0)
            f_t = f_t * scale[:, None]
            xi = xi * scale[:, None]
        np.add.at(self.plane_impulse, plane_id, F_n * dt)
        f_on_particle = -F_n[:, None] * normals + f_t
        force = _scatter(rows, f_on_particle, self.n)
        torque = _scatter(rows, _cross(lever, f_on_particle), self.n)
        spin = self.omega[rows] - wall.angular_velocity[None, :]
        spin_perp = spin - (spin * normals).sum(1)[:, None] * normals      # rolling, not twisting
        spin_mag = np.sqrt((spin_perp * spin_perp).sum(1))
        moving = spin_mag > _EPS
        mu_roll = (self._pair_mu_roll(rows, rows) if model.rolling is None
                   else np.full(len(rows), float(model.rolling)))
        coef = np.where(moving, -mu_roll * F_n * r / np.maximum(spin_mag, _EPS), 0.0)
        torque = torque + _scatter(rows, coef[:, None] * spin_perp, self.n)
        torque_on_wall = _cross(point - wall.pivot, -f_on_particle)
        wall_torque = float(np.sum(torque_on_wall @ wall._axis))
        return force, torque, wall_torque, int(len(rows)), float(np.max(penetration / r)), keys, xi

    def _poly_wall(self, row: int, wall: Wall, theta: float, normal: np.ndarray, offset: float,
                   plane_id: int, dt: float):
        model = self.model
        volume, point, depth = clipped_by_plane(self.shape[row], self.rotation[row], self.position[row],
                                                normal, offset)
        if volume <= 0.0:
            return None
        lever = point - self.position[row]
        v_rel = (self.velocity[row] + _cross(self.omega[row], lever)[0]) - wall.surface_velocity(theta, point)[0]
        v_n = float(v_rel @ normal)
        v_t = v_rel - v_n * normal
        F_el = float(model.volume_stiffness) * volume
        k_secant = max(F_el / max(depth, 1.0e-12), 1.0e-12)      # dF/d(depth) for a nearly flat contact
        c_n = 2.0 * self._damping[row] * math.sqrt(self.mass[row] * k_secant)
        F_n = max(F_el + min(max(c_n * v_n, -F_el), F_el), 0.0)
        if F_n <= 0.0:
            return None
        key = (row, plane_id)
        xi = self._poly_wall_shear.get(key, np.zeros(3)) + v_t * dt
        k_t = model.kt_ratio * k_secant
        f_t = -k_t * xi
        mu = (float(model.friction) if model.friction is not None
              else math.sqrt(self.material[row].mu * wall.material.mu))
        if np.linalg.norm(f_t) > mu * F_n:
            f_t = f_t * (mu * F_n / max(np.linalg.norm(f_t), _EPS))
            xi = -f_t / k_t
        self._poly_wall_shear[key] = xi
        self._touched.add(key)
        self.plane_impulse[plane_id] += F_n * dt
        f_on_particle = -F_n * normal + f_t
        force = np.zeros((self.n, 3))
        torque = np.zeros((self.n, 3))
        force[row] = f_on_particle
        torque[row] = _cross(lever, f_on_particle)[0]
        wall_torque = float(np.cross(point - wall.pivot, -f_on_particle) @ wall._axis)
        strain = volume / max(self.shape_volume[row], _EPS)
        return force, torque, wall_torque, float(strain)

    # ------------------------------------------------------------------ integration
    def step(self, dt: float) -> StepDiagnostics:
        """Advance one step of ``dt`` seconds."""
        if not self._finalized:
            self.finalize()
        self.dt = float(dt)
        for k, wall in enumerate(self.walls):
            self.wall_angle[k] += wall.omega * dt
        force = self.gravity[None, :] * self.mass[:, None]
        torque = np.zeros((self.n, 3))
        for source in self.force_sources:
            out = source(self)
            if isinstance(out, tuple):
                force = force + out[0]
                torque = torque + out[1]
            else:
                force = force + out
        if self.bond_state["n"]:
            from .bonds import bond_forces

            bf, bt = bond_forces(self)
            force = force + bf
            torque = torque + bt
        cf, ct, contacts, strain = self._resolve(dt)
        force = force + cf
        torque = torque + ct
        self.force, self.torque = force, torque

        self.velocity += force * self.inv_mass[:, None] * dt
        inertia_w = self._inertia_world_all
        angular = np.matmul(inertia_w, self.omega[..., None])[..., 0]
        gyro = _cross(self.omega, angular)
        self.omega += np.matmul(self._inv_inertia_world_all,
                                (torque - gyro)[..., None])[..., 0] * dt
        self.position += self.velocity * dt
        self._rotate(dt)
        self.time += dt
        self._last_strain = strain
        return StepDiagnostics(
            time=self.time,
            kinetic_energy=self.kinetic_energy(),
            gravity_energy=self.gravity_energy(),
            contacts=contacts,
            max_penetration_ratio=strain,
            wall_torque=float(-np.sum(self._last_wall_torque)),
            momentum=self.momentum(),
        )

    def _rotate(self, dt: float) -> None:
        speed = np.sqrt((self.omega ** 2).sum(1))
        angle = speed * dt
        moving = angle > 1.0e-14
        if not np.any(moving):
            return
        axis = np.zeros((self.n, 3))
        axis[moving] = self.omega[moving] / speed[moving, None]
        K = _skew(axis * dt)                                      # K = dt * skew(unit axis)
        small = angle < 1.0e-6
        a = np.where(small, 1.0 - angle ** 2 / 6.0, np.sin(angle) / np.maximum(angle, _EPS))
        b = np.where(small, 0.5 - angle ** 2 / 24.0,
                     (1.0 - np.cos(angle)) / np.maximum(angle ** 2, _EPS))
        Q = np.eye(3)[None, :, :] + a[:, None, None] * K + b[:, None, None] * (K @ K)
        self.rotation = Q @ self.rotation
        gram = np.matmul(self.rotation, np.transpose(self.rotation, (0, 2, 1)))
        drift = float(np.abs(gram - np.eye(3)[None, :, :]).max())
        if drift > 1.0e-8:                                        # only re-orthonormalise when drift appears
            for k in np.nonzero(moving)[0]:
                q, _ = np.linalg.qr(self.rotation[k])
                if np.linalg.det(q) < 0:
                    q[:, 2] *= -1.0
                self.rotation[k] = q

    # ------------------------------------------------------------------ measurements
    def kinetic_energy(self) -> float:
        inertia_w = self._inertia_world_all
        transl = 0.5 * float(np.sum(self.mass * (self.velocity ** 2).sum(1)))
        rot = 0.5 * float(np.sum(self.omega * np.matmul(inertia_w, self.omega[..., None])[..., 0]))
        return transl + rot

    def gravity_energy(self) -> float:
        return float(-np.sum(self.mass * (self.position @ self.gravity)))

    def momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.velocity).sum(0)

    def angular_momentum(self) -> np.ndarray:
        """Orbital angular momentum about the origin (``sum r x p``), without the bodies' spin."""
        return np.cross(self.position, self.mass[:, None] * self.velocity).sum(0)

    def total_angular_momentum(self) -> np.ndarray:
        """Orbital plus spin angular momentum: the conserved quantity when contacts transmit torque."""
        orbital = np.cross(self.position, self.mass[:, None] * self.velocity).sum(0)
        spin = np.matmul(self._inertia_world_all, self.omega[..., None])[..., 0].sum(0)
        return orbital + spin

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def overlaps(self) -> dict:
        """Largest initial interpenetration, as a fraction of the smaller body.

        Starting a run with bodies inside each other is the classic way to make a DEM run explode: the contact law
        answers with a force proportional to the overlap volume, and that force is enormous.  Call this after
        ``finalize()`` and place the assembly again if it is not ~0.
        """
        ss, other = self._pairs()
        worst = 0.0
        count = 0
        for i, j in ss:
            r = min(self.sphere_radius[i], self.sphere_radius[j])
            gap = float(np.sqrt(((self.position[i] - self.position[j]) ** 2).sum())) - (
                self.sphere_radius[i] + self.sphere_radius[j])
            if gap < 0.0:
                count += 1
                worst = max(worst, -gap / max(r, _EPS))
        if len(other):
            for i, j in other[:64]:                              # the polyhedral pairs are the expensive ones
                volume, _, axis = overlap_volume(self.shape[i], self.shape[j], self.rotation[i],
                                                 self.rotation[j], self.position[i], self.position[j])
                if axis.overlap and volume > 0.0:
                    count += 1
                    worst = max(worst, float(axis.depth) / max(min(self.radius[i], self.radius[j]), _EPS))
        return {"pairs": int(count), "max_relative_overlap": float(worst)}

    def repeated_contacts(self) -> dict[tuple[int, int], int]:
        """Sanity check for the pair kernels: how many contacts each pair carried (should be 1)."""
        counts: dict[tuple[int, int], int] = {}
        ss, other = self._pairs()
        both = np.vstack([ss, other]) if (len(ss) or len(other)) else np.zeros((0, 2), dtype=int)
        for i, j in both:
            key = (int(min(i, j)), int(max(i, j)))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def recommended_timestep(self, safety: float = 0.2) -> float:
        """Stable step for the assembly: the tighter of the Rayleigh bound and the stiffest bond spring.

        A bond is usually far stiffer than a contact (a 2 GPa bond of a 2 mm neck is ~1e6 N/m), so a bonded run has
        to take its step from the bonds; ignoring them is how a bonded assembly heats itself up.
        """
        radii = self.sphere_radius[self.is_sphere]
        r = float(radii.min()) if len(radii) else float(self.radius.min())
        dt = rayleigh_timestep(r, self.material[0], safety=safety)
        if self.bond_state.get("n"):
            stiffness = np.maximum(self.bond_state["kn"], self.bond_state["ks"])
            m_min = float(self.mass.min())
            omega = math.sqrt(float(np.max(stiffness)) / m_min)
            dt = min(dt, safety * 2.0 * math.pi / omega)
        return float(dt)

    def power_draw(self, wall_index: int = 0) -> float:
        """Mechanical power the drive must supply through wall ``wall_index`` [W], ``motor torque * omega``.

        The contacts give the torque the charge exerts *on* the shell; the drive has to push against it, so the
        motor torque is the negative of it and the power drawn is positive for a mill that is doing work.
        """
        if not self.walls:
            return 0.0
        return -float(self._last_wall_torque[wall_index]) * float(self.walls[wall_index].omega)

    def run(self, steps: int, dt: float, every: int = 1, snapshot: bool = False,
            callback: Callable[[World, StepDiagnostics], None] | None = None) -> dict:
        """Step the assembly and collect the diagnostics history (and, with ``snapshot``, the positions)."""
        history: dict[str, list] = {k: [] for k in
                                    ("time", "kinetic_energy", "gravity_energy", "contacts",
                                     "max_penetration_ratio", "wall_torque")}
        frames: list[np.ndarray] = []
        total = int(steps)
        for step in range(total):
            diag = self.step(dt)
            if callback is not None:
                callback(self, diag)
            if step % every == 0 or step == total - 1:
                history["time"].append(diag.time)
                history["kinetic_energy"].append(diag.kinetic_energy)
                history["gravity_energy"].append(diag.gravity_energy)
                history["contacts"].append(diag.contacts)
                history["max_penetration_ratio"].append(diag.max_penetration_ratio)
                history["wall_torque"].append(diag.wall_torque)
                if snapshot:
                    frames.append(self.position.copy())
        out = {k: np.asarray(v) for k, v in history.items()}
        if snapshot:
            out["frames"] = frames
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"World(n={self.n}, walls={len(self.walls)}, t={self.time:.4g} s)"
