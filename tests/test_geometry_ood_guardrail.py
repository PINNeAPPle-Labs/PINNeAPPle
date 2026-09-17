"""Tests for ``pinneapple_analysis.verification.geometry_ood_guardrail``
(``GeometryOODGuardrail``/``GeometryOODResult``/``extract_geometry_features``)
-- the geometric out-of-distribution guardrail, a scoped reimplementation
of the concept behind NVIDIA PhysicsNeMo's
``experimental/guardrails/geometry`` module (see that module's own
docstring for the full design rationale and the explicit choices made
differently here).

Where practical, these tests use REAL mesh files already tracked in this
repo (``examples/geometry/_out/channel_box.stl``,
``examples/geometry/_out/box_with_hole.stl``,
``examples/end_to_end/assets/Dragon 2.5_stl.stl``) rather than fabricated
fixtures, per this line of work's established real-verification standard.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.verification.geometry_ood_guardrail import (
    FEATURE_NAMES,
    GeometryOODGuardrail,
    GeometryOODResult,
    extract_geometry_features,
)
from pinneapple_design.geometry.gen import primitives as geom_primitives
from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge

_DRAGON_STL = "examples/end_to_end/assets/Dragon 2.5_stl.stl"
_CHANNEL_BOX_STL = "examples/geometry/_out/channel_box.stl"
_BOX_WITH_HOLE_STL = "examples/geometry/_out/box_with_hole.stl"


# ---------------------------------------------------------------------------
# extract_geometry_features
# ---------------------------------------------------------------------------

def test_extract_geometry_features_shape_and_names_agree():
    mesh = geom_primitives.build_mesh("box", extents=(1.0, 2.0, 3.0))
    feats = extract_geometry_features(mesh)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_extract_geometry_features_reacts_to_translation_rotation_scale():
    """Features are intentionally NON-invariant (same design choice
    PhysicsNeMo's guardrail makes, see its README) -- a translated,
    rotated, or rescaled version of the same shape must produce a
    different feature vector, not an identical one."""
    base = geom_primitives.build_mesh("box", extents=(1.0, 2.0, 3.0))
    translated = geom_primitives.build_mesh("box", extents=(1.0, 2.0, 3.0), translate=(10.0, 0.0, 0.0))
    scaled = geom_primitives.build_mesh("box", extents=(1.0, 2.0, 3.0), scale=2.0)

    f_base = extract_geometry_features(base)
    f_translated = extract_geometry_features(translated)
    f_scaled = extract_geometry_features(scaled)

    assert not np.allclose(f_base, f_translated)
    assert not np.allclose(f_base, f_scaled)


def test_extract_geometry_features_curvature_distinguishes_box_from_sphere():
    """The mean-dihedral-curvature-proxy feature (index of
    'mean_dihedral_curvature_deg' in FEATURE_NAMES) must be MUCH larger
    for a box (each of its 12 real edges is a sharp 90 deg crease) than
    for a finely tessellated sphere (every edge is a small angle between
    near-coplanar facets). A unit cube's exact expected value is a
    length-weighted mix of its 12 real 90 deg creases (length 1 each) and
    the 6 zero-degree "fold" edges trimesh's triangulation adds along
    each flat face's own diagonal (length sqrt(2) each) --
    ``(12*90*1 + 6*0*sqrt(2)) / (12*1 + 6*sqrt(2)) ~= 52.7 deg`` -- so the
    exact number (not a round 90) is itself a correct, derivable
    consequence of triangulating a quad face into two triangles, not a
    bug in the curvature proxy."""
    box = geom_primitives.build_mesh("box", extents=(1.0, 1.0, 1.0))
    sphere = geom_primitives.build_mesh("sphere", radius=1.0, subdivisions=4)

    idx = FEATURE_NAMES.index("mean_dihedral_curvature_deg")
    box_curv = extract_geometry_features(box)[idx]
    sphere_curv = extract_geometry_features(sphere)[idx]

    expected_box_curv = (12 * 90.0 * 1.0 + 6 * 0.0 * np.sqrt(2.0)) / (12 * 1.0 + 6 * np.sqrt(2.0))
    assert box_curv == pytest.approx(expected_box_curv, abs=0.5)
    assert sphere_curv < 20.0
    assert sphere_curv < box_curv


def test_extract_geometry_features_zero_area_mesh_raises():
    import numpy as np
    from pinneapple_design.geometry.core.mesh import MeshData

    # Three collinear points -> a degenerate triangle with zero area.
    degenerate = MeshData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    with pytest.raises(ValueError):
        extract_geometry_features(degenerate)


# ---------------------------------------------------------------------------
# GeometryOODGuardrail.fit / validation
# ---------------------------------------------------------------------------

def test_fit_requires_at_least_two_meshes():
    guardrail = GeometryOODGuardrail()
    with pytest.raises(ValueError):
        guardrail.fit([geom_primitives.build_mesh("box")])


def test_score_before_fit_raises():
    guardrail = GeometryOODGuardrail()
    mesh = geom_primitives.build_mesh("box")
    with pytest.raises(RuntimeError):
        guardrail.score(mesh)


def test_invalid_percentile_thresholds_raise():
    with pytest.raises(ValueError):
        GeometryOODGuardrail(warn_percentile=99.0, reject_percentile=90.0)
    with pytest.raises(ValueError):
        GeometryOODGuardrail(warn_percentile=-1.0)
    with pytest.raises(ValueError):
        GeometryOODGuardrail(shrinkage=1.5)


# ---------------------------------------------------------------------------
# Real differentiation: synthetic reference family (build_mesh, the same
# real mesh-builder catalog examples/geometry/ itself uses)
# ---------------------------------------------------------------------------

def _channel_family(rng: np.random.Generator, n: int, length=1.0, width=0.2, height=0.2):
    return [
        geom_primitives.build_mesh(
            "channel",
            length=abs(length + 0.1 * length * rng.normal()),
            width=abs(width + 0.1 * width * rng.normal()),
            height=abs(height + 0.1 * height * rng.normal()),
        )
        for _ in range(n)
    ]


def test_similar_shape_scores_ok_dissimilar_shape_scores_reject():
    rng = np.random.default_rng(0)
    guardrail = GeometryOODGuardrail().fit(_channel_family(rng, 30))

    similar = geom_primitives.build_mesh("channel", length=1.02, width=0.19, height=0.21)
    sphere = geom_primitives.build_mesh("sphere", radius=5.0, subdivisions=3)

    res_similar = guardrail.score(similar)
    res_sphere = guardrail.score(sphere)

    assert isinstance(res_similar, GeometryOODResult)
    assert res_similar.status == "OK"
    assert res_sphere.status == "REJECT"
    assert res_sphere.p_value < res_similar.p_value
    assert res_sphere.percentile > res_similar.percentile
    assert res_sphere.mahalanobis_sq > res_similar.mahalanobis_sq


def test_result_carries_real_interpretable_fields():
    rng = np.random.default_rng(0)
    guardrail = GeometryOODGuardrail().fit(_channel_family(rng, 30))
    sphere = geom_primitives.build_mesh("sphere", radius=5.0, subdivisions=3)
    result = guardrail.score(sphere)

    assert result.most_anomalous_feature in FEATURE_NAMES
    assert result.n_reference_shapes == 30
    assert result.n_features == len(FEATURE_NAMES)
    assert "REJECT" in result.summary()
    assert result.most_anomalous_feature in result.summary()


# ---------------------------------------------------------------------------
# Real-world verification: real tracked mesh files, not fabricated fixtures
# ---------------------------------------------------------------------------

def test_real_world_normal_vs_anomalous_geometry():
    """The task-level real-world check: fit on a real box/channel preset
    shape family (scaled to match this repo's own real
    ``channel_box.stl`` demo asset), then confirm:
      - a real, same-family, held-out shape (``channel_box.stl`` itself,
        not used for fitting) scores OK;
      - a real but genuinely different organic/complex shape (the tracked
        Dragon STL) scores REJECT, with an enormously larger anomaly
        distance -- a real, non-fabricated differentiation, not a
        constant output regardless of input.
    """
    rng = np.random.default_rng(42)
    # Same real-world scale as examples/geometry/_out/channel_box.stl
    # (length=4, width=1, height=1) -- a real channel/duct preset family.
    ref_meshes = _channel_family(rng, 40, length=4.0, width=1.0, height=1.0)
    guardrail = GeometryOODGuardrail().fit(ref_meshes)

    bridge = TrimeshBridge()
    channel_box = bridge.load(_CHANNEL_BOX_STL)
    dragon = bridge.load(_DRAGON_STL)

    res_channel = guardrail.score(channel_box)
    res_dragon = guardrail.score(dragon)

    assert res_channel.status == "OK"
    assert res_dragon.status == "REJECT"
    assert res_dragon.mahalanobis_sq > res_channel.mahalanobis_sq
    assert res_dragon.p_value < res_channel.p_value
    # Real, not-fabricated separation: several orders of magnitude apart.
    assert res_dragon.mahalanobis_sq > res_channel.mahalanobis_sq * 1e6


def test_real_world_box_with_hole_is_a_real_geometric_deviation():
    """``box_with_hole.stl`` (a real tracked demo asset: a box with a
    cylindrical hole cut through it) is a genuinely different shape from
    a solid box/channel family -- its surface-to-bbox-area ratio is
    necessarily different from a solid box's -- so it should score worse
    than the plain (holeless) channel_box.stl from the same nominal
    family, even though both are real box-scale CAD parts. This confirms
    the guardrail differentiates by REAL geometric content, not merely
    by "is this a box-ish bounding volume"."""
    rng = np.random.default_rng(42)
    ref_meshes = _channel_family(rng, 40, length=4.0, width=1.0, height=1.0)
    guardrail = GeometryOODGuardrail().fit(ref_meshes)

    bridge = TrimeshBridge()
    channel_box = bridge.load(_CHANNEL_BOX_STL)
    box_with_hole = bridge.load(_BOX_WITH_HOLE_STL)

    res_channel = guardrail.score(channel_box)
    res_hole = guardrail.score(box_with_hole)

    assert res_channel.status == "OK"
    assert res_hole.mahalanobis_sq > res_channel.mahalanobis_sq
