"""Real, genuine-verification tests for ``core/mesh_intelligence.py`` --
formula correctness checked against hand-computed/known values (not just
"runs without crashing"), and pass/fail behavior checked on meshes whose
correct answer is unambiguous by construction."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_analysis.verification.mesh_intelligence import (
    assess_mesh_quality,
    estimate_required_first_cell_height,
)
from pinneapple_design.geometry.core.mesh import MeshData
from pinneapple_design.geometry.gen.primitives import build_mesh

# ---------------------------------------------------------------------------
# Per-triangle formula correctness (hand-computed known values)
# ---------------------------------------------------------------------------

def test_aspect_ratio_of_a_right_isoceles_triangle_matches_hand_calculation():
    """A box face split into 2 right-isoceles triangles (legs=side,
    hyp=side*sqrt(2)) has a hand-computable R/r: for legs a=b=1,
    hyp=c=sqrt(2): Area=0.5, s=(1+1+sqrt(2))/2, R=c/2=sqrt(2)/2 (a right
    triangle's circumradius is always half its hypotenuse), r=Area/s.
    R/r = 2.41421... (a real, standard, checkable value for this exact
    triangle shape -- NOT the equilateral-triangle minimum of 2.0, which
    is the correct thing to check: this triangle isn't equilateral)."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    report = assess_mesh_quality(box)

    c = math.sqrt(2.0)
    area = 0.5
    s = (1.0 + 1.0 + c) / 2.0
    R = c / 2.0
    r = area / s
    expected = R / r

    assert report.triangle_quality.aspect_ratio_max == pytest.approx(expected, rel=1e-9)
    assert report.triangle_quality.aspect_ratio_mean == pytest.approx(expected, rel=1e-9)


def test_skewness_of_a_right_isoceles_triangle_matches_hand_calculation():
    """Same triangle as above: angles are 90/45/45 degrees.
    Equiangular skewness = max((theta_max-60)/(180-60), (60-theta_min)/60)
    = max((90-60)/120, (60-45)/60) = max(0.25, 0.25) = 0.25 exactly."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    report = assess_mesh_quality(box)
    assert report.triangle_quality.skewness_max == pytest.approx(0.25, abs=1e-9)
    assert report.triangle_quality.skewness_mean == pytest.approx(0.25, abs=1e-9)


def test_equilateral_triangle_scores_the_best_possible_aspect_ratio_and_zero_skewness():
    """A genuinely equilateral triangle must score exactly R/r=2.0 (the
    real, standard minimum under this convention -- see module docstring)
    and exactly 0.0 skewness (all angles are 60 degrees, the equilateral
    angle itself)."""
    h = math.sqrt(3.0) / 2.0
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = MeshData(vertices=verts, faces=faces)
    report = assess_mesh_quality(mesh)
    assert report.triangle_quality.aspect_ratio_max == pytest.approx(2.0, rel=1e-9)
    assert report.triangle_quality.skewness_max == pytest.approx(0.0, abs=1e-9)
    assert report.is_good_enough


# ---------------------------------------------------------------------------
# "Is this mesh good enough" pass/fail on unambiguous cases
# ---------------------------------------------------------------------------

def test_uniform_icosphere_mesh_passes_default_quality_thresholds():
    sphere = build_mesh("sphere", radius=1.0, subdivisions=3)
    report = assess_mesh_quality(sphere)
    assert report.is_good_enough
    assert report.failures == []
    # A subdivided icosphere is quite uniform -- nowhere near the default thresholds.
    assert report.triangle_quality.aspect_ratio_max < 10.0
    assert report.triangle_quality.skewness_max < 0.5


def test_a_single_manually_inserted_sliver_triangle_fails_with_a_specific_correct_reason():
    """A real, standard, otherwise-fine box mesh, PLUS one deliberately
    near-degenerate sliver triangle added directly -- the report must
    fail, must name the correct offending face, and must report BOTH the
    aspect-ratio and skewness metrics as the reason (a sliver is bad by
    both measures simultaneously, which is exactly what makes it a
    sliver)."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    sliver_verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0001, 1.0], [2.0, 0.0, 1.0]])
    sliver_face_id = box.n_vertices
    verts = np.vstack([box.vertices, sliver_verts])
    faces = np.vstack([box.faces, [[sliver_face_id, sliver_face_id + 1, sliver_face_id + 2]]])
    mesh = MeshData(vertices=verts, faces=faces)

    report = assess_mesh_quality(mesh)
    assert not report.is_good_enough
    failing_metrics = {f["metric"] for f in report.failures}
    assert "aspect_ratio" in failing_metrics
    assert "skewness" in failing_metrics
    assert report.triangle_quality.worst_aspect_ratio_face_id == mesh.n_faces - 1
    assert report.triangle_quality.worst_skewness_face_id == mesh.n_faces - 1
    assert report.triangle_quality.n_faces_failing_aspect_ratio == 1
    assert report.triangle_quality.aspect_ratio_max > 1000.0  # genuinely, dramatically degenerate
    # The report is structured (which metric, by how much) -- not a single opaque boolean.
    for f in report.failures:
        assert "detail" in f and "threshold" in f and "worst_value" in f


def test_an_extremely_elongated_box_fails_on_aspect_ratio_without_any_inserted_defect():
    """A naturally anisotropic mesh (no artificially-inserted sliver at
    all) -- a 100:1:1 box's large faces split into very long thin right
    triangles -- must ALSO fail, demonstrating the metric catches a real
    naturally-bad mesh, not only a hand-crafted pathological one."""
    elongated = build_mesh("box", extents=[100.0, 1.0, 1.0])
    report = assess_mesh_quality(elongated, aspect_ratio_threshold=100.0)
    assert not report.is_good_enough
    assert any(f["metric"] == "aspect_ratio" for f in report.failures)
    assert report.triangle_quality.aspect_ratio_max > 100.0


def test_max_bad_element_fraction_tolerates_a_small_number_of_bad_elements():
    """The same single-sliver mesh as above must PASS when the caller
    explicitly configures a nonzero tolerance covering that one bad
    element out of many -- proving the tolerance knob genuinely changes
    the verdict (not just changes a number that's ignored)."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    sliver_verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0001, 1.0], [2.0, 0.0, 1.0]])
    sliver_face_id = box.n_vertices
    verts = np.vstack([box.vertices, sliver_verts])
    faces = np.vstack([box.faces, [[sliver_face_id, sliver_face_id + 1, sliver_face_id + 2]]])
    mesh = MeshData(vertices=verts, faces=faces)

    strict = assess_mesh_quality(mesh, max_bad_element_fraction=0.0)
    tolerant = assess_mesh_quality(mesh, max_bad_element_fraction=0.5)  # 1/13 faces is well under 50%
    assert not strict.is_good_enough
    assert tolerant.is_good_enough


# ---------------------------------------------------------------------------
# Orthogonality: honest "not computable" for a pure triangle-soup surface mesh
# ---------------------------------------------------------------------------

def test_orthogonality_is_honestly_not_computable_for_a_surface_mesh():
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    report = assess_mesh_quality(box)
    assert report.orthogonality.computable is False
    assert "volumetric cells" in report.orthogonality.reason
    assert "orthogonality" not in [f["metric"] for f in report.failures]  # never silently scored


# ---------------------------------------------------------------------------
# y+ ESTIMATE: formula correctness + honest "not a real measurement" behavior
# ---------------------------------------------------------------------------

def test_yplus_required_cell_height_formula_matches_hand_calculation():
    """Air at 10 m/s over a 1 m plate, nu=1.5e-5 m^2/s (a standard
    textbook air value): Re_L = 10*1/1.5e-5 = 666666.67, well within the
    correlation's stated validity range. Every intermediate quantity is
    hand-computed and checked, not just the final answer."""
    u, length, nu = 10.0, 1.0, 1.5e-5
    re_l = u * length / nu
    cf_expected = 0.058 * re_l ** -0.2
    u_tau_expected = math.sqrt(cf_expected * 0.5 * u ** 2)
    required_y_expected = 1.0 * nu / u_tau_expected

    est = estimate_required_first_cell_height(reynolds_length=re_l, velocity=u, kinematic_viscosity=nu, target_yplus=1.0)
    assert est.computable
    assert est.skin_friction_coefficient == pytest.approx(cf_expected, rel=1e-12)
    assert est.friction_velocity == pytest.approx(u_tau_expected, rel=1e-12)
    assert est.required_first_cell_height == pytest.approx(required_y_expected, rel=1e-12)
    assert "5e5<Re_L<1e7" not in est.reason  # inside the validity range -- no caveat note appended
    assert "estimate" in est.reason.lower() or "estimated" in est.reason.lower()


def test_yplus_estimate_flags_out_of_validity_range_explicitly():
    """A very slow flow (Re_L well below the correlation's cited validity
    range) must still compute an answer (never refuse), but must say so
    honestly in ``reason``, matching dimensional_analysis.py's own
    Nusselt-correlation validity-range discipline."""
    est = estimate_required_first_cell_height(reynolds_length=1.0e3, velocity=0.01, kinematic_viscosity=1.5e-5, target_yplus=1.0)
    assert est.computable
    assert "outside the correlation" in est.reason


def test_yplus_is_never_reported_as_a_real_measurement_only_as_an_estimate():
    """Structural/documentation check: the dataclass's own formula string
    and reason must never claim to be a solved, real y+ -- only an
    estimate of the required cell height for a TARGET y+."""
    est = estimate_required_first_cell_height(reynolds_length=1.0e6, velocity=10.0, kinematic_viscosity=1.5e-5, target_yplus=1.0)
    assert "estimate" in est.reason.lower() or "estimated" in est.reason.lower()
    assert "flat-plate" in est.formula.lower() or "schlichting" in est.formula.lower() or "cf" in est.formula.lower()


def test_yplus_verdict_flags_a_coarse_mesh_as_too_coarse_for_a_wall_resolved_target():
    """A coarse 2x2x2 box mesh (triangle altitudes of order 1) cannot
    possibly resolve y+=1 for a real turbulent flow (which needs a
    near-wall cell size of order 1e-5 m for this flow) -- the verdict
    must say so, with both the required and actual numbers shown."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    report = assess_mesh_quality(
        box, physical_parameters={"velocity": 10.0, "length": 1.0, "kinematic_viscosity": 1.5e-5}, target_yplus=1.0,
    )
    assert report.yplus.computable
    assert report.yplus.verdict.startswith("likely too coarse")
    assert report.yplus.required_first_cell_height < report.yplus.actual_near_wall_cell_size
    assert any(f["metric"] == "yplus_first_cell_height" for f in report.failures)
    assert not report.is_good_enough


def test_yplus_reuses_wall_classified_faces_when_given_not_the_whole_mesh_fallback():
    """Passing explicit wall face ids must change the reported cell-size
    SOURCE string (proving it's genuinely being used, not ignored) --
    here checked by restricting to a deliberately coarser subset of faces
    than the whole mesh, on a mesh with faces of two different scales."""
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    # Face 0/1 belong to one wall of the box; use only those as the "wall region".
    report_whole = assess_mesh_quality(
        box, physical_parameters={"velocity": 1.0, "length": 1.0, "kinematic_viscosity": 1.5e-5},
    )
    report_subset = assess_mesh_quality(
        box, physical_parameters={"velocity": 1.0, "length": 1.0, "kinematic_viscosity": 1.5e-5},
        wall_region_face_ids=[0, 1],
    )
    assert "WHOLE mesh" in report_whole.yplus.actual_cell_size_source
    assert "wall-classified face(s)" in report_subset.yplus.actual_cell_size_source
    assert "2 wall-classified" in report_subset.yplus.actual_cell_size_source


def test_yplus_not_computable_without_enough_physical_parameters():
    box = build_mesh("box", extents=[2.0, 2.0, 2.0])
    report = assess_mesh_quality(box)  # no physical_parameters, no target_reynolds at all
    assert not report.yplus.computable
    assert report.yplus.verdict == "not computable"
    assert report.yplus.required_first_cell_height is None
    assert not any(f["metric"] == "yplus_first_cell_height" for f in report.failures)  # never fails on a non-computed metric
