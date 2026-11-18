"""Geometry of the granular solver: exact volumes, exact overlap volumes, separating axes and Wulff shapes.

The overlap volume is the quantity the whole contact model is built on, so it is checked against the analytics that
exist for it (axis-aligned boxes, spherical caps and lens) and against Monte Carlo where no closed form exists.
"""
import math

import numpy as np
import pytest

from pinneapple_simulation.granular import (
    box,
    clipped_by_plane,
    closest_point_on_polyhedron,
    faceted_sphere,
    overlap_volume,
    polytope_from_halfspaces,
    regular_prism,
    rotation_matrix,
    separating_axis,
    sphere_plane_cap_volume,
    sphere_sphere_lens_volume,
)


def test_box_mass_properties_are_exact():
    cube = box(0.02, 0.02, 0.02, density=2500.0)
    assert cube.volume == pytest.approx(8.0e-6, rel=1e-12)
    assert np.allclose(cube.centroid, 0.0, atol=1e-15)
    assert cube.area == pytest.approx(6 * 0.02 ** 2, rel=1e-12)
    # a solid cube of side a has I = m a^2 / 6 about every axis through its centre
    expected = cube.mass * 0.02 ** 2 / 6.0
    assert np.allclose(np.diag(cube.inertia), expected, rtol=1e-12)
    assert np.allclose(cube.inertia - np.diag(np.diag(cube.inertia)), 0.0, atol=1e-18)
    # and the inertia is diagonal even for a rectangular box (no coupling between axes)
    slab = box(0.03, 0.01, 0.02, density=1000.0)
    assert np.allclose(np.diag(slab.inertia),
                       [slab.mass * (0.01 ** 2 + 0.02 ** 2) / 12.0,
                        slab.mass * (0.03 ** 2 + 0.02 ** 2) / 12.0,
                        slab.mass * (0.03 ** 2 + 0.01 ** 2) / 12.0], rtol=1e-12)


def test_regular_prism_matches_the_polygon_formula():
    prism = regular_prism(24, radius=0.05, height=0.1)
    polygon_area = 0.5 * 24 * 0.05 ** 2 * math.sin(2 * math.pi / 24)
    assert prism.volume == pytest.approx(polygon_area * 0.1, rel=1e-12)
    assert prism.area == pytest.approx(2 * polygon_area + 24 * (2 * 0.05 * math.sin(math.pi / 24)) * 0.1,
                                       rel=1e-12)


def test_faceted_sphere_stays_inside_the_sphere_and_grows_with_the_point_count():
    coarse = faceted_sphere(0.01, 42)
    fine = faceted_sphere(0.01, 162)
    sphere_volume = 4.0 / 3.0 * math.pi * 0.01 ** 3
    assert coarse.volume < sphere_volume
    assert fine.volume < sphere_volume
    assert fine.volume > coarse.volume
    # a convex hull of points on the sphere can never leave it (the hull centroid is a hair off the origin, and
    # the bounding radius is measured from the centroid)
    radii = np.linalg.norm(coarse.vertices, axis=1)
    assert np.allclose(radii, 0.01, rtol=1e-12)
    assert coarse.bounding_radius <= 0.01 * (1 + 1e-3)


def test_overlap_of_two_axis_aligned_boxes_is_the_analytic_product():
    a = box(1.0, 1.0, 1.0)
    b = box(1.0, 1.0, 1.0).transformed(translation=[0.5, 0.0, 0.0])
    volume, centroid, axis = overlap_volume(a, b)
    assert volume == pytest.approx(0.5, rel=1e-12)
    assert np.allclose(centroid, [0.25, 0.0, 0.0], atol=1e-12)
    assert axis.overlap and axis.depth == pytest.approx(0.5, rel=1e-12)
    assert np.allclose(np.abs(axis.normal), [1.0, 0.0, 0.0], atol=1e-12)
    # three-way overlap along every axis
    c = box(2.0, 2.0, 2.0).transformed(translation=[0.0, 0.0, 0.5])
    volume, _, _ = overlap_volume(box(1.0, 1.0, 1.0), c)
    assert volume == pytest.approx(1.0, rel=1e-12)


def test_overlap_vanishes_continuously_as_the_bodies_separate():
    a = box(1.0, 1.0, 1.0)
    gaps = [1.0e-3, 1.0e-4, 1.0e-5]
    volumes = []
    for gap in gaps:
        b = box(1.0, 1.0, 1.0).transformed(translation=[1.0 - gap, 0.0, 0.0])
        volume, _, axis = overlap_volume(a, b)
        assert axis.overlap
        volumes.append(volume)
    # a slab of thickness `gap`, so the volume is linear in the overlap to numerical precision
    for gap, volume in zip(gaps, volumes, strict=True):
        assert volume == pytest.approx(gap, rel=1e-9)
    b = box(1.0, 1.0, 1.0).transformed(translation=[1.5, 0.0, 0.0])
    volume, _, axis = overlap_volume(a, b)
    assert volume == 0.0
    assert not axis.overlap
    assert axis.depth == pytest.approx(0.5, rel=1e-12)          # the gap it reports


def test_rotated_overlap_matches_monte_carlo():
    a = box(1.0, 1.0, 1.0)
    R = rotation_matrix([0.0, 0.0, 1.0], math.radians(45.0))
    b = box(1.0, 1.0, 1.0).transformed(rotation=R, translation=[0.6, 0.0, 0.0])
    volume, _, _ = overlap_volume(a, b)
    rng = np.random.default_rng(7)
    samples = rng.uniform(-2.0, 2.0, size=(400_000, 3))
    inside_b = np.all(np.abs((samples - np.array([0.6, 0.0, 0.0])) @ R) <= 0.5, axis=1)
    inside_a = np.all(np.abs(samples) <= 0.5, axis=1)
    monte_carlo = 64.0 * float(np.mean(inside_a & inside_b))
    assert volume == pytest.approx(monte_carlo, rel=0.03)


def test_separating_axis_reports_the_minimum_translation():
    a = box(1.0, 1.0, 1.0)
    b = box(1.0, 1.0, 1.0).transformed(translation=[0.2, 0.0, 0.0])
    axis = separating_axis(a, b)
    assert axis.overlap and axis.depth == pytest.approx(0.8, rel=1e-12)
    b = box(1.0, 1.0, 1.0).transformed(translation=[0.0, 0.9, 0.0])
    assert separating_axis(a, b).depth == pytest.approx(0.1, rel=1e-9)


def test_sphere_cap_and_lens_have_the_closed_form_values():
    for depth in (0.001, 0.01, 0.1):
        radius = 0.5
        assert sphere_plane_cap_volume(radius, depth) == pytest.approx(
            math.pi * depth ** 2 * (3 * radius - depth) / 3.0, rel=1e-12)
    # two identical spheres at distance d: the lens is the sum of the two caps of height r - d/2
    for distance in (0.5, 1.0, 1.5):
        h = 1.0 - distance / 2.0
        caps = 2 * math.pi * h ** 2 * (3.0 - h) / 3.0
        assert sphere_sphere_lens_volume(1.0, 1.0, distance) == pytest.approx(caps, rel=1e-12)
    assert sphere_sphere_lens_volume(1.0, 1.0, 2.0) == 0.0
    assert sphere_sphere_lens_volume(1.0, 0.2, 0.1) == pytest.approx(4.0 / 3.0 * math.pi * 0.2 ** 3)


def test_plane_clip_gives_the_slab_volume_centroid_and_depth():
    cube = box(1.0, 1.0, 1.0)
    volume, centroid, depth = clipped_by_plane(cube, np.eye(3), np.zeros(3),
                                               np.array([0.0, 0.0, 1.0]), 0.4)
    assert volume == pytest.approx(0.1, rel=1e-12)
    assert np.allclose(centroid, [0.0, 0.0, 0.45], atol=1e-12)
    assert depth == pytest.approx(0.1, rel=1e-12)
    # nothing beyond the plane: no contact at all
    assert clipped_by_plane(cube, np.eye(3), np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.6) == (0.0, pytest.approx(0.0), 0.0)


def test_halfspace_intersection_builds_the_cube():
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    offsets = np.full(6, 0.5)
    verts, faces = polytope_from_halfspaces(normals, offsets)
    from pinneapple_simulation.granular import ConvexPolyhedron

    cube = ConvexPolyhedron(verts, faces)
    assert cube.volume == pytest.approx(1.0, rel=1e-12)


def test_closest_point_on_a_box():
    cube = box(1.0, 1.0, 1.0)
    closest = closest_point_on_polyhedron(cube, np.eye(3), np.zeros(3), np.array([0.0, 0.0, 3.0]))
    assert np.allclose(closest, [0.0, 0.0, 0.5], atol=1e-12)
    closest = closest_point_on_polyhedron(cube, np.eye(3), np.zeros(3), np.array([2.0, 2.0, 2.0]))
    assert np.allclose(closest, [0.5, 0.5, 0.5], atol=1e-12)
