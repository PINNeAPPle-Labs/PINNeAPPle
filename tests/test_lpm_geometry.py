"""Stage 2: a design space over a mesh somebody brought, and the validity gate in front of the solver."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_design.geometry.morph import FFD, DesignSpace
from pinneapple_neural.lpm_pipeline import DesignSpec, Reference, build
from pinneapple_neural.lpm_stages import geometry


def _sphere(nu: int = 16, nv: int = 8, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A closed triangle sphere with outward winding — the smallest body a morph space can be fitted on."""
    theta = np.linspace(0, math.pi, nv + 1)[1:-1]
    phi = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    ring = np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
                     for t in theta for p in phi]) * radius
    vertices = np.vstack([ring, [[0, 0, radius]], [[0, 0, -radius]]])
    faces = []
    for j in range(len(theta) - 1):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            faces += [[a, b, b + nu], [a, b + nu, a + nu]]
    north, south = len(vertices) - 2, len(vertices) - 1
    for i in range(nu):
        faces.append([north, i, (i + 1) % nu])
        base = (len(theta) - 1) * nu
        faces.append([south, base + (i + 1) % nu, base + i])
    return vertices, np.asarray(faces, dtype=np.int64)[:, [0, 2, 1]]


def _family(*, open_surface: bool = False, n_points: int = 64):
    vertices, faces = _sphere()
    ffd = FFD.fit(vertices, control=(2, 2, 2))
    # a missing face is what the gate exists for: the surface stops being closed and no volume can be measured
    space = DesignSpace(ffd, faces[:-1] if open_surface else faces, vertices)
    def field(spec, points, normals, surface):
        """A stand-in reference: a normalised height on the surface, so the design has a target to carry."""
        return (points[:, 2] / max(float(np.abs(points).max()), 1e-9))[:, None]
    family = geometry.morph_family(
        space, field, name="pp_sphere", title="a morphed sphere", n_points=n_points,
        reference=Reference("closed form", "z / max|xyz| on the surface", ("a test body, not a solved field",)))
    spec = DesignSpec(geometry=family.sample_geometry(np.random.default_rng(0)), operating={}, label="g0")
    return family, spec


def test_a_body_the_gate_accepts_becomes_a_design_with_its_own_points_and_normals():
    family, spec = _family()
    assert family.validate(spec) == ""
    design = build(family, spec)
    assert design.points.shape == (64, 3) and design.target.shape == (64, 1)
    # the normals are the surface's own, at the same evenly spaced vertices the points came from
    vertices, faces = family.space.build(family._vector(spec.geometry)).as_tuple()
    keep = np.unique(np.linspace(0, len(vertices) - 1, 64).round().astype(int))
    assert np.allclose(design.normals.numpy(), geometry.vertex_normals(vertices, faces)[keep], atol=1e-6)
    described = family.describe()["geometry_source"]
    assert described["kind"].startswith("morph.DesignSpace")
    assert described["n_variables"] == family.space.dim
    # every design variable of the morph space is a geometry parameter of the family, in the space's own order
    assert list(family.geometry_params) == list(family.space.names)


def test_a_body_the_gate_refuses_is_refused_here_too_with_the_reason():
    family, spec = _family(open_surface=True)
    spec = DesignSpec(geometry=spec.geometry, operating={}, label="g1")
    reason = family.validate(spec)
    assert reason.startswith("the morph validity gate refused")
    assert "watertight" in reason                  # the gate's own words, not a summary of them
    with pytest.raises(ValueError, match="outside the reference's range"):
        build(family, spec)


def test_the_validity_gate_measures_the_share_of_the_box_that_survives():
    family, _ = _family()
    gate = geometry.validity_gate(family, 8)
    assert gate["check"] == "validity" and gate["status"] == "passed"
    assert 0.0 <= gate["value"] <= 1.0
    assert "survive the morph gate" in gate["detail"]


def test_the_field_has_to_come_back_one_row_per_sampled_point():
    vertices, faces = _sphere()
    space = DesignSpace(FFD.fit(vertices, control=(2, 2, 2)), faces, vertices)
    wrong = geometry.morph_family(space, lambda spec, points, normals, surface: points[:, :2],
                                  name="pp_bad", title="x", reference=Reference("none", "x", ()),
                                  field_names=("cp",), n_points=32)
    spec = DesignSpec(geometry=wrong.sample_geometry(np.random.default_rng(2)), operating={}, label="g2")
    with pytest.raises(ValueError, match="one row per sampled point"):
        wrong.field(spec)
