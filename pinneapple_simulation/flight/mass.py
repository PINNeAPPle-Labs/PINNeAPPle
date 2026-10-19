"""Mass, centre of gravity and inertia tensor from a component breakdown.

Each :class:`Component` has a mass, a centroid in airframe axes (x aft, y right, z up) and its own inertia about its
centroid (from simple shapes: point, slender rod, thin plate, solid cylinder). :func:`mass_properties` sums them by
the parallel-axis theorem and returns the inertia tensor in body axes (x forward, y right, z down) about the CG.
:func:`balance` moves one component along x so the CG lands at a chosen station (the loading of the aircraft).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .aero import T_AB


@dataclass
class Component:
    name: str
    mass: float
    center: np.ndarray
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))     # own, airframe axes, about centroid


def point(name, mass, center) -> Component:
    return Component(name, mass, np.asarray(center, float))


def rod(name, mass, center, length, axis: int = 0) -> Component:
    """Slender rod of ``length`` along airframe axis ``axis``."""
    J = np.zeros((3, 3))
    for k in range(3):
        if k != axis:
            J[k, k] = mass * length ** 2 / 12
    return Component(name, mass, np.asarray(center, float), J)


def plate(name, mass, center, span, chord, taper: float = 1.0) -> Component:
    """Thin plate in the x-y plane: ``span`` along y, ``chord`` along x (a wing, a tailplane, wing fuel). With
    ``taper`` < 1 the mass per unit span follows the local chord (linear taper from the root to the tips):
    Ixx = m (b/2)^2 (1 + 3 taper) / (6 (1 + taper)), m b^2 / 12 for a rectangular plate."""
    ixx = mass * (span / 2) ** 2 * (1 + 3 * taper) / (6 * (1 + taper))
    iyy = mass * chord ** 2 / 12
    J = np.diag([ixx, iyy, ixx + iyy])
    return Component(name, mass, np.asarray(center, float), J)


def fin_plate(name, mass, center, height, chord) -> Component:
    """Thin plate in the x-z plane (a vertical tail)."""
    J = np.diag([mass * height ** 2 / 12, mass * (height ** 2 + chord ** 2) / 12, mass * chord ** 2 / 12])
    return Component(name, mass, np.asarray(center, float), J)


def cylinder(name, mass, center, length, radius, shell: bool = False) -> Component:
    """Cylinder along x (a fuselage, a nacelle); ``shell=True`` for a thin-walled one."""
    Ia = mass * radius ** 2 * (1.0 if shell else 0.5)
    It = mass * (radius ** 2 * (0.5 if shell else 0.25) + length ** 2 / 12)
    return Component(name, mass, np.asarray(center, float), np.diag([Ia, It, It]))


def mass_properties(components: Sequence[Component]) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, cg in airframe axes, inertia about the CG in body axes)."""
    m = sum(c.mass for c in components)
    cg = sum(c.mass * np.asarray(c.center, float) for c in components) / m
    J = np.zeros((3, 3))
    for c in components:
        d = np.asarray(c.center, float) - cg
        J += c.inertia + c.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return m, cg, T_AB @ J @ T_AB.T


def balance(components: Sequence[Component], x_target: float, movable: str) -> list[Component]:
    """Shift the component named ``movable`` along x so that the CG is at ``x_target`` (airframe axes)."""
    comps = list(components)
    m = sum(c.mass for c in comps)
    mx = sum(c.mass * float(c.center[0]) for c in comps)
    i = next(k for k, c in enumerate(comps) if c.name == movable)
    c = comps[i]
    dx = (m * x_target - mx) / c.mass
    comps[i] = replace(c, center=np.asarray(c.center, float) + np.array([dx, 0.0, 0.0]))
    return comps
