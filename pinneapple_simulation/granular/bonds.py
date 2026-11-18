"""Bonded elements that break: a brittle agglomerate made of rigid bodies joined by beams.

This is the ``bonded elements`` half of the multiphysics DEM picture: a block, a tablet, a rock or an ore particle is
built from bodies whose pairs are joined by a short beam with axial, shear, bending and torsional stiffness
(Potyondy & Cundall 2004, *A bonded-particle model for rock*, Int. J. Rock Mech. Min. Sci. 41:1329-1364, which is
the parallel-bond formulation this follows).  A bond carries a load until its extreme-fibre stress passes the
material strength and then disappears -- and because the bodies are still there, a broken aggregate keeps moving as
fragments instead of vanishing.

Fragments are the connected components of the intact bond graph (union-find), reported by mass and by the sieve size
a comminution engineer expects: the equivalent-sphere diameter ``d = (6 V / pi)^(1/3)`` of each fragment, and its
``P80``.  That is what turns a DEM run into a product curve (see
:mod:`pinneapple_simulation.granular.equipment` for the mill that produces it).

The bond frame is fixed at creation (small-strain assumption), which is the usual bonded-particle idealisation: the
element size is small compared with the deformation at which the bond fails, so the error stays well below the
scatter in the strength parameters.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

__all__ = [
    "BondMaterial",
    "build_bond_arrays",
    "bond_forces",
    "fragment_ids",
    "fragment_stats",
    "p80",
]

_EPS = 1.0e-30


_BOND_DEFAULTS = {
    "E": 2.0e9,             # bond Young's modulus [Pa]
    "sigma_max": 2.0e6,     # tensile strength [Pa]
    "tau_max": 1.0e6,       # shear strength [Pa]
    "radius_ratio": 0.5,    # bond radius as a fraction of the smaller body radius
}


class BondMaterial:
    """Elastic and strength parameters of the bond (not of the bodies)."""

    def __init__(self, E: float = 2.0e9, nu: float = 0.25, sigma_max: float = 2.0e6,
                 tau_max: float = 1.0e6, radius_ratio: float = 0.5) -> None:
        if E <= 0 or sigma_max <= 0 or tau_max <= 0:
            raise ValueError("bond E, sigma_max and tau_max must be positive")
        self.E = float(E)
        self.nu = float(nu)
        self.sigma_max = float(sigma_max)
        self.tau_max = float(tau_max)
        self.radius_ratio = float(radius_ratio)

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BondMaterial(E={self.E:.3g}, sigma_max={self.sigma_max:.3g})"


def _body_radius(world, k: int) -> float:
    if world.shape[k].is_sphere:
        return float(world.sphere_radius[k])
    return float((3.0 * world.shape_volume[k] / (4.0 * math.pi)) ** (1.0 / 3.0))


def build_bond_arrays(world, bonds: Iterable[dict]) -> dict:
    """Turn the ``add_bond`` records into arrays of bond geometry, stiffness and strength."""
    bonds = list(bonds)
    n = len(bonds)
    state: dict = {
        "n": n,
        "i": np.zeros(n, dtype=int),
        "j": np.zeros(n, dtype=int),
        "L0": np.zeros(n),
        "axis": np.zeros((n, 3)),
        "anchor_i": np.zeros((n, 3)),
        "anchor_j": np.zeros((n, 3)),
        "kn": np.zeros(n),
        "ks": np.zeros(n),
        "kb": np.zeros(n),
        "kt": np.zeros(n),
        "area": np.zeros(n),
        "inertia": np.zeros(n),
        "polar": np.zeros(n),
        "radius": np.zeros(n),
        "sigma_max": np.zeros(n),
        "tau_max": np.zeros(n),
        "intact": np.zeros(n, dtype=bool),
    }
    for b, rec in enumerate(bonds):
        i, j = int(rec["i"]), int(rec["j"])
        mat = rec.get("bond_material") or BondMaterial(
            **{k: rec[k] for k in _BOND_DEFAULTS if k in rec})
        ri, rj = _body_radius(world, i), _body_radius(world, j)
        r_b = float(rec.get("bond_radius", mat.radius_ratio * min(ri, rj)))
        d = world.position[j] - world.position[i]
        L0 = float(np.linalg.norm(d))
        if L0 <= 0.0:
            raise ValueError(f"bond {i}-{j}: the two bodies are at the same position")
        axis = d / L0
        state["i"][b] = i
        state["j"][b] = j
        state["L0"][b] = L0
        state["axis"][b] = axis
        # put the two anchors on the same point so the bond starts stress-free: on the touching surfaces when the
        # bodies touch, at the midpoint of the gap when they do not (a cement bridge between two grains)
        # both anchors sit on the same point, so the bond starts stress-free whether the bodies touch
        # (anchor on the shared surface) or not (anchor at the midpoint of the cement bridge between them)
        state["anchor_i"][b] = axis * (0.5 * (L0 + ri - rj))
        state["anchor_j"][b] = axis * (0.5 * (ri - rj - L0))
        A = math.pi * r_b ** 2
        inertia = math.pi * r_b ** 4 / 4.0
        J = 2.0 * inertia
        state["radius"][b] = r_b
        state["area"][b] = A
        state["inertia"][b] = inertia
        state["polar"][b] = J
        state["kn"][b] = mat.E * A / L0
        state["ks"][b] = mat.G * A / L0
        state["kb"][b] = mat.E * inertia / L0
        state["kt"][b] = mat.G * J / L0
        state["sigma_max"][b] = mat.sigma_max
        state["tau_max"][b] = mat.tau_max
        state["intact"][b] = True
    return state


def _rotation_vector(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (the log map)."""
    cos = (np.trace(R) - 1.0) * 0.5
    cos = min(max(cos, -1.0), 1.0)
    angle = math.acos(cos)
    if angle < 1.0e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle > math.pi - 1.0e-6:                                  # pragma: no cover - half turn is a rare corner
        w, v = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = v[:, int(np.argmax(w))]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def bond_forces(world) -> tuple[np.ndarray, np.ndarray]:
    """Forces and torques every intact bond applies to its two bodies, and the break decision for this step."""
    st = world.bond_state
    n = world.n
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    if not st["n"]:
        return force, torque
    broken = []
    for b in range(st["n"]):
        if not st["intact"][b]:
            continue
        i, j = int(st["i"][b]), int(st["j"][b])
        axis = st["axis"][b]
        p_i = world.position[i] + world.rotation[i] @ st["anchor_i"][b]
        p_j = world.position[j] + world.rotation[j] @ st["anchor_j"][b]
        du = p_j - p_i
        u_n = float(du @ axis)
        du_t = du - u_n * axis
        theta = _rotation_vector(world.rotation[i].T @ world.rotation[j])
        theta_n = float(theta @ axis)
        theta_t = theta - theta_n * axis

        F_on_j = -(st["kn"][b] * u_n) * axis - st["ks"][b] * du_t
        M_on_j = -st["kb"][b] * theta_t - st["kt"][b] * theta_n * axis
        F_on_i = -F_on_j
        M_on_i = -M_on_j                                                    # the bond is a shared element

        # extreme-fibre stress: axial + bending on the section modulus, shear + torsion on the polar one
        r_b, area, inertia, polar = st["radius"][b], st["area"][b], st["inertia"][b], st["polar"][b]
        sigma = abs(st["kn"][b] * u_n) / area + np.linalg.norm(st["kb"][b] * theta_t) * r_b / inertia
        tau = np.linalg.norm(st["ks"][b] * du_t) / area + abs(st["kt"][b] * theta_n) * r_b / polar
        if sigma > st["sigma_max"][b] or tau > st["tau_max"][b]:
            broken.append(b)
            continue

        force[i] += F_on_i
        force[j] += F_on_j
        torque[i] += M_on_i + np.cross(world.rotation[i] @ st["anchor_i"][b], F_on_i)
        torque[j] += M_on_j + np.cross(world.rotation[j] @ st["anchor_j"][b], F_on_j)
    if broken:
        st["intact"][np.array(broken)] = False
        st["broken_events"] = st.get("broken_events", 0) + len(broken)
    return force, torque


def fragment_ids(world) -> np.ndarray:
    """Connected components of the intact-bond graph (``-1`` for bodies with no bond at all)."""
    st = world.bond_state
    parent = list(range(world.n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for b in range(st["n"]):
        if not st["intact"][b]:
            continue
        a, c = find(int(st["i"][b])), find(int(st["j"][b]))
        if a != c:
            parent[c] = a
    bonded = np.zeros(world.n, dtype=bool)
    for b in range(st["n"]):
        bonded[st["i"][b]] = True
        bonded[st["j"][b]] = True
    labels = np.full(world.n, -1, dtype=int)
    roots: dict[int, int] = {}
    for k in range(world.n):
        if not bonded[k]:
            continue
        root = find(k)
        if root not in roots:
            roots[root] = len(roots)
        labels[k] = roots[root]
    return labels


def fragment_stats(world) -> dict:
    """Fragment count, masses, equivalent diameters and P80 of the current assembly."""
    labels = fragment_ids(world)
    present = labels >= 0
    if not np.any(present):
        return {"fragments": 0, "masses": np.zeros(0), "diameters": np.zeros(0), "p80": 0.0,
                "bonded_bodies": 0, "bonded_mass": 0.0,
                "intact_bonds": int(world.bond_state["intact"].sum())}
    ids = labels[present]
    masses = np.array([world.mass[present][ids == k].sum() for k in range(ids.max() + 1)])
    volumes = np.array([world.shape_volume[present][ids == k].sum() for k in range(ids.max() + 1)])
    diameters = (6.0 * volumes / math.pi) ** (1.0 / 3.0)
    return {
        "fragments": int(len(masses)),
        "masses": masses,
        "diameters": diameters,
        "p80": p80(diameters, masses),
        "bonded_bodies": int(present.sum()),
        "bonded_mass": float(world.mass[present].sum()),
        "intact_bonds": int(world.bond_state["intact"].sum()),
        "broken_bonds": int(world.bond_state["n"] - world.bond_state["intact"].sum()),
    }


def p80(diameters: np.ndarray, masses: np.ndarray) -> float:
    """80 % passing size: the diameter below which 80 % of the mass lies (linear interpolation)."""
    d = np.asarray(diameters, dtype=float)
    m = np.asarray(masses, dtype=float)
    if len(d) == 0:
        return 0.0
    order = np.argsort(d)
    d, m = d[order], m[order]
    cumulative = np.cumsum(m) / max(m.sum(), _EPS)
    if cumulative[-1] < 0.8:
        return float(d[-1])
    idx = int(np.searchsorted(cumulative, 0.8))
    if idx == 0:
        return float(d[0])
    x0, x1 = float(d[idx - 1]), float(d[idx])
    y0, y1 = float(cumulative[idx - 1]), float(cumulative[idx])
    if y1 - y0 < _EPS:
        return x1
    return x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)
