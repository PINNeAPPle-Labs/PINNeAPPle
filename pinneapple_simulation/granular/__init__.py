"""Granular DEM with volume-based contact for convex polyhedra, bonded breakage, mills, wear and fluid drag.

This package is PINNeAPPle's answer to GPU-native granular multiphysics: a discrete element solver whose contact law
acts on the **volume two bodies share** instead of on a penetration depth, so a polyhedral particle behaves the same
per unit of shared volume whether it meets a neighbour corner-to-face or face-to-face.  That is the method the GPU
DEM line of work (BLAZE-DEM / Govender et al.) is built on, and it is what makes angular particles usable for a
mill, a tablet or an ore without rounding everything into spheres.

Layers, bottom up:

============================  ==================================================================================
:mod:`.shapes`                convex polyhedra (exact volume, centroid, inertia), spheres and shape factories
:mod:`.overlap`               separating axis, exact overlap volume and contact point by half-space clipping
:mod:`.contacts`              materials, the volume and Hertz laws, Cundall-Strack friction, walls and obstacles
:mod:`.world`                 bodies, neighbour search, vectorised contact kernels, integration, measurements
:mod:`.bonds`                 bonded elements that break, and the fragment / ``P80`` statistics of a breakage run
:mod:`.equipment`             rotating drum / mill, the charge, power draw and the product curve
:mod:`.wear`                  impulse wear law on liner patches, fitted to a measured profile
:mod:`.coupling`              fluid drag (Schiller-Naumann + Di Felice) as a force source, the particle side of CFD
:mod:`.lattice`               crystal lattice -> elastic constants -> Wulff particle shape ("atoms to material")
============================  ==================================================================================

    >>> import numpy as np
    >>> from pinneapple_simulation.granular import World, Material, box
    >>> world = World(gravity=(0.0, -9.81, 0.0))
    >>> _ = world.add(box(0.01, 0.01, 0.01, density=2500.0), position=(0.0, 0.05, 0.0),
    ...               material=Material(density=2500.0))
    >>> world.finalize()
    >>> history = world.run(steps=100, dt=2.0e-5)
    >>> history["time"][-1] > 0.0
    True

What it is not: a GPU solver.  The kernels are plain NumPy (see ``docs/granular_dem.md`` for the measured speed and
for what a CUDA port would have to replace), and the fluid coupling is one-way -- the fluid pushes the particles,
the particles do not yet push the fluid back.
"""
from __future__ import annotations

from .bonds import BondMaterial, fragment_ids, fragment_stats, p80
from .contacts import (
    ForceModel,
    Material,
    Obstacle,
    Wall,
    rayleigh_timestep,
    restitution_to_damping,
)
from .coupling import FluidCoupling, schiller_naumann_drag
from .equipment import RotatingDrum, cascade_power, charge_positions
from .lattice import (
    Crystal,
    LennardJones,
    cubic_polycrystal_moduli,
    material_from_lattice,
    wulff_polyhedron,
)
from .overlap import (
    clipped_by_plane,
    intersection_loops,
    overlap_volume,
    polytope_from_halfspaces,
    separating_axis,
    sphere_plane_cap_volume,
    sphere_sphere_lens_volume,
)
from .shapes import ConvexPolyhedron, Sphere, box, faceted_sphere, regular_prism, rotation_matrix
from .wear import WearTracker, fit_wear_law, patch_areas
from .world import StepDiagnostics, World, closest_point_on_polyhedron

__all__ = [
    # bodies and geometry
    "ConvexPolyhedron", "Sphere", "box", "regular_prism", "faceted_sphere", "rotation_matrix",
    "overlap_volume", "separating_axis", "intersection_loops", "clipped_by_plane", "polytope_from_halfspaces",
    "sphere_sphere_lens_volume", "sphere_plane_cap_volume", "closest_point_on_polyhedron",
    # materials, contacts and walls
    "Material", "ForceModel", "Wall", "Obstacle", "rayleigh_timestep", "restitution_to_damping",
    # the solver
    "World", "StepDiagnostics",
    # breakage
    "BondMaterial", "fragment_ids", "fragment_stats", "p80",
    # equipment and wear
    "RotatingDrum", "charge_positions", "cascade_power", "WearTracker", "fit_wear_law", "patch_areas",
    # fluid coupling
    "FluidCoupling", "schiller_naumann_drag",
    # crystals
    "Crystal", "LennardJones", "material_from_lattice", "cubic_polycrystal_moduli", "wulff_polyhedron",
]
