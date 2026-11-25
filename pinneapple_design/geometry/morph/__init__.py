"""Morphing of an arbitrary surface mesh: a design space over a mesh the repository already has (#292).

The library could generate a shape only through a parametric family — a CST airfoil, the Ahmed body — which meant a
design study was limited to the shapes somebody had hand-coded. This package deforms *any* triangle surface, keeping
the topology so a field stays comparable vertex by vertex:

``FFD``
    a lattice of control points around the body (Sederberg & Parry 1986); the design variables are the control-point
    displacements, with a symmetry plane and keep-out regions resolved into the parameter vector itself.
``RBFMorph``
    handles on chosen surface points, radial-basis interpolation of their displacement, and a set of points held
    fixed.
``ShapeModes``
    PCA of a registered mesh family, so a few coefficients span the shapes that were actually measured.
``DesignSpace``
    bounds, keep-out boxes, sampling and the validity gate in front of any of the three.
``check_surface``
    the gate itself: watertightness, orientation, degenerate faces, finite and positive volume, volume and frontal
    area bands, faces turned inside out, self-intersections, optional thickness rays.

    import pinneapple as pp
    car = pp.geom.morph.Surface.load("data/geometries/road_car/road_car.light.stl")
    ffd = pp.geom.morph.FFD.fit(car.reference, control=(4, 3, 4), symmetry="y")
    space = pp.geom.morph.DesignSpace(ffd, car.faces, car.reference)
    report = space.sample_valid(20, seed=0)
    V, F = space.build(report.parameters[0]).as_tuple()        # ready for pp.cfd.ExternalFlow

Everything here is numpy: no trimesh, no OpenCascade, no solver. The watertightness and volume numbers come from
``pinneapple_data.cae.quality``, the same checks the mesh quality app reports.

What this package deliberately does not do: it does not re-mesh, so a deformation large enough to make a mesh unusable
is rejected rather than repaired; and it does not register meshes that were not built from the same reference — the
modes are only as good as the correspondence, and a silent registration error would poison a design space.
"""
from .lattice import FFD
from .modes import ShapeModes
from .rbf import RBFMorph
from .space import DesignSpace, DesignSpaceReport
from .surface import Surface, as_surface
from .validity import Reference, ValidityReport, check_surface, frontal_area, self_intersections, thickness

__all__ = [
    "FFD",
    "RBFMorph",
    "ShapeModes",
    "DesignSpace",
    "DesignSpaceReport",
    "Surface",
    "as_surface",
    "Reference",
    "ValidityReport",
    "check_surface",
    "frontal_area",
    "self_intersections",
    "thickness",
]
