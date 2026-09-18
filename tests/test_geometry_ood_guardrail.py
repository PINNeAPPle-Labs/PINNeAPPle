"""Tests for ``pinneapple_analysis.verification.geometry_ood_guardrail``
(``GeometryOODGuardrail``/``GeometryOODResult``/``extract_geometry_features``),
the geometry out-of-distribution guardrail that flags when an input shape is
not the kind of geometry a model/preset was ever trained or validated on --
the real gap identified against NVIDIA PhysicsNeMo's experimental geometry
guardrail (see that module's own docstring for the full concept-adaptation
rationale and its documented diagonal-Mahalanobis scope decision).

Every mesh used below is a REAL mesh built via
``pinneapple_design.geometry.gen.primitives.build_mesh`` -- the same
real mesh-generation infrastructure other geometry tests in this repo
(``test_mesh_primitives_registry.py``) already use, skipped (not failed)
when the optional ``trimesh`` dependency isn't installed, matching this
repo's established convention.
"""
from __future__ import annotations

import pytest

pytest.importorskip("trimesh")

import numpy as np

from pinneapple_design.geometry.gen.primitives import build_mesh
from pinneapple_analysis.verification.geometry_ood_guardrail import (
    FEATURE_NAMES,
    extract_geometry_features,
    GeometryOODGuardrail,
    GeometryOODResult,
    default_reference_meshes,
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_geometry_features_returns_every_named_feature_as_a_real_finite_float():
    mesh = build_mesh("box", extents=(1.0, 2.0, 3.0))
    feats = extract_geometry_features(mesh)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    for name, value in feats.items():
        assert isinstance(value, float), name
        assert np.isfinite(value), name


def test_extract_geometry_features_bbox_extent_matches_real_box_dimensions():
    mesh = build_mesh("box", extents=(1.0, 2.0, 4.0))
    feats = extract_geometry_features(mesh)
    # A box centered at the origin: bbox extents are exactly its own real,
    # requested dimensions (trimesh.creation.box builds it centered).
    extents = sorted([feats["bbox_extent_x"], feats["bbox_extent_y"], feats["bbox_extent_z"]])
    assert extents == pytest.approx(sorted([1.0, 2.0, 4.0]), rel=1e-6)


def test_extract_geometry_features_aspect_ratio_reflects_real_elongation():
    cube = extract_geometry_features(build_mesh("box", extents=(1.0, 1.0, 1.0)))
    rod = extract_geometry_features(build_mesh("box", extents=(10.0, 1.0, 1.0)))
    assert cube["aspect_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert rod["aspect_ratio"] == pytest.approx(10.0, rel=1e-6)
    assert rod["aspect_ratio"] > cube["aspect_ratio"]


def test_extract_geometry_features_surface_area_scales_correctly_for_a_cube():
    mesh = build_mesh("box", extents=(2.0, 2.0, 2.0))
    feats = extract_geometry_features(mesh)
    # A real cube of side 2: 6 faces * 2*2 = 24.
    assert feats["surface_area"] == pytest.approx(24.0, rel=1e-6)


# ---------------------------------------------------------------------------
# default_reference_meshes()
# ---------------------------------------------------------------------------

def test_default_reference_meshes_returns_real_non_empty_varied_catalog():
    meshes = default_reference_meshes()
    assert len(meshes) == 48
    # Real variety, not 48 copies of the same shape: bbox volumes must differ.
    volumes = {round(m.volume_bbox(), 6) for m in meshes}
    assert len(volumes) > 10


# ---------------------------------------------------------------------------
# GeometryOODGuardrail.fit() input validation
# ---------------------------------------------------------------------------

def test_fit_raises_on_empty_reference_list():
    with pytest.raises(ValueError):
        GeometryOODGuardrail().fit([])


def test_fit_raises_on_single_reference_mesh():
    with pytest.raises(ValueError):
        GeometryOODGuardrail().fit([build_mesh("box", extents=(1.0, 1.0, 1.0))])


def test_query_before_fit_raises():
    guardrail = GeometryOODGuardrail()
    with pytest.raises(RuntimeError):
        guardrail.query(build_mesh("box", extents=(1.0, 1.0, 1.0)))


def test_init_rejects_reject_p_greater_than_warn_p():
    with pytest.raises(ValueError):
        GeometryOODGuardrail(warn_p=0.01, reject_p=0.05)


# ---------------------------------------------------------------------------
# The load-bearing check: real differentiation between normal and anomalous
# ---------------------------------------------------------------------------

def test_in_distribution_geometry_scores_high_and_is_classified_ok():
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    # A box well inside the reference catalog's own size/shape range
    # (references span sizes 0.5-4.0 and several aspect-ratio shapes).
    result = guardrail.query(build_mesh("box", extents=(1.5, 1.2, 0.9)))
    assert isinstance(result, GeometryOODResult)
    assert result.status == "OK"
    assert result.score > 0.5
    assert result.n_reference == 48
    assert result.dof == len(FEATURE_NAMES)


def test_extreme_geometry_scores_low_and_is_classified_reject():
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    # 500 x 0.001 x 0.001: five orders of magnitude beyond the reference
    # catalog's largest size (4.0) and most extreme aspect ratio (8.0).
    result = guardrail.query(build_mesh("box", extents=(500.0, 0.001, 0.001)))
    assert result.status == "REJECT"
    assert result.score < 0.01
    # aspect_ratio must show up as a dominant driver of the anomaly -- an
    # honest, inspectable explanation, not an opaque number.
    top_names = [name for name, _ in result.top_deviating_features]
    assert "aspect_ratio" in top_names


def test_score_genuinely_differentiates_normal_from_anomalous_not_a_constant():
    """The exact check this session was asked to perform: run a real
    'normal' geometry and a real, genuinely different geometry through the
    SAME fitted guardrail and confirm the anomaly score actually tells
    them apart."""
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    normal = guardrail.query(build_mesh("box", extents=(1.0, 1.0, 1.0)))
    weird = guardrail.query(build_mesh("box", extents=(500.0, 0.001, 0.001)))
    assert normal.score != weird.score
    assert normal.score > weird.score
    assert normal.mahalanobis_sq < weird.mahalanobis_sq


def test_a_shape_family_never_seen_in_reference_is_also_flagged():
    """Reference catalog is box/cylinder/channel only -- a woven-tube
    metamaterial (a real, more topologically complex mesh built by the
    same build_mesh() infrastructure) should read as far more anomalous
    than a plain box drawn from a family the guardrail was actually fit
    on."""
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    plain_box = guardrail.query(build_mesh("box", extents=(1.0, 1.0, 1.0)))
    woven = guardrail.query(build_mesh("woven_tube", R=10.0, L=60.0, n_threads=12, turns=8.0))
    assert woven.score < plain_box.score


# ---------------------------------------------------------------------------
# Transparency: every score is explained, never opaque
# ---------------------------------------------------------------------------

def test_result_source_summary_quotes_the_real_driving_numbers():
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    result = guardrail.query(build_mesh("box", extents=(500.0, 0.001, 0.001)))
    assert "mahalanobis_sq" in result.source_summary
    assert result.status in result.source_summary
    assert str(result.n_reference) in result.source_summary
    assert len(result.feature_z_scores) == len(FEATURE_NAMES)
    assert len(result.top_deviating_features) <= 3
    # top_deviating_features must actually be the largest-|z| entries.
    all_abs_z = sorted((abs(z) for z in result.feature_z_scores.values()), reverse=True)
    top_abs_z = sorted((abs(z) for _, z in result.top_deviating_features), reverse=True)
    assert top_abs_z == all_abs_z[: len(top_abs_z)]


def test_result_score_is_in_valid_probability_range():
    guardrail = GeometryOODGuardrail().fit(default_reference_meshes())
    for mesh in (
        build_mesh("box", extents=(1.0, 1.0, 1.0)),
        build_mesh("cylinder", radius=1.0, height=2.0),
        build_mesh("box", extents=(500.0, 0.001, 0.001)),
    ):
        result = guardrail.query(mesh)
        assert 0.0 <= result.score <= 1.0
        assert result.mahalanobis_sq >= 0.0
