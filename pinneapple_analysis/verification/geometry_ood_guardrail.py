"""``GeometryOODGuardrail`` -- out-of-distribution detection for INPUT
geometry, plugged into this repo's existing trust/confidence stack
(``pinneapple_analysis.trust.trust_gate.TrustGate`` and
``pinneapple_analysis.verification.physics_confidence_score``) as one more
real, transparent signal.

Where this idea comes from, and what is (and is not) reused
--------------------------------------------------------------
NVIDIA PhysicsNeMo ships a real guardrail
(``physicsnemo/experimental/guardrails/geometry/``, Apache-2.0) that flags a
CAD/mesh geometry as anomalous at inference time by (1) extracting a fixed
set of "non-invariant" geometric features from the mesh, (2) fitting a
density model (a Gaussian Mixture Model or a Polynomial Chaos Expansion)
over those features on a reference set of training-time shapes, and (3)
converting a new shape's density-model anomaly score to an empirical
percentile, classified OK / WARN / REJECT against configurable thresholds.

This module reimplements that *concept* from scratch, in this repo's own
style -- no PhysicsNeMo source was copied (Apache-2.0 would require
attribution for literal code; none is used here). Two concrete, documented
scope decisions were made rather than mirroring PhysicsNeMo's design
1-for-1:

1. **Density model**: PhysicsNeMo defaults to a Gaussian Mixture Model
   (with a Polynomial Chaos Expansion alternative for non-Gaussian
   feature distributions). This module uses a single-Gaussian model
   instead -- a **shrunk-covariance Mahalanobis distance** -- because a
   full GMM needs enough reference shapes per mixture component to fit a
   stable covariance, and this product's realistic near-term reference
   sets (a preset family's shape variations, not a factory's worth of
   scanned parts) are small. A single Gaussian is the same statistical
   object ``TrustGate._ood_score`` already uses for *coordinate* OOD
   (Mahalanobis distance -> ``scipy.stats.chi2.sf`` p-value) -- this module
   applies the identical, already-proven statistical idea to *shape*
   features instead of coordinates, for consistency with the rest of this
   repo's trust machinery rather than introducing a second, different
   anomaly-scoring convention. Multi-component GMM/PCE is a documented,
   real future extension (see :class:`GeometryOODGuardrail`'s docstring),
   not a hidden limitation.
2. **Feature set**: 14 features covering position, absolute size,
   proportions, principal-axis shape spread, and surface complexity (see
   :data:`FEATURE_NAMES`) -- overlapping in spirit with PhysicsNeMo's 22
   features (centroid, PCA axes/eigenvalues, bounding box, projected
   areas) but independently derived, and adding two descriptors
   PhysicsNeMo's own README lists as unimplemented "TODO: future
   enhancements" for that module: a surface-complexity/compactness ratio
   and a discrete mean-curvature proxy (area-weighted dihedral angle
   across mesh edges). Like PhysicsNeMo's, these features are
   intentionally **not invariant** to translation, rotation, or scale --
   a geometry placed, oriented, or sized differently from the reference
   set is exactly the kind of thing this guardrail should catch.

Feature extraction reuses this repo's own real, already-tested mesh
primitives (``pinneapple_design.geometry.core.mesh.MeshData.bounds()``,
``.compute_face_normals()``, ``.face_areas()``, ``.vertices``, ``.faces``)
-- it does not reimplement mesh I/O or basic geometric queries that
already exist.

Integration
-----------
- ``TrustGate.score(..., geometry_guardrail=..., mesh=...)`` adds a fourth
  ``geometry_score`` sub-score alongside the existing OOD/residual/
  ensemble sub-scores (see ``trust_gate.py``).
- ``compute_physics_confidence(..., geometry_ood_result=...)`` adds a
  fifth ``geometry_ood`` component alongside the existing four (see
  ``physics_confidence_score.py``).

Both integrations follow this repo's transparency rule: a
:class:`GeometryOODResult` always carries the real numbers behind its
score (Mahalanobis distance, percentile, which single feature deviated
most and by how many standard deviations) -- never a bare float.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.stats import chi2

from pinneapple_design.geometry.core.mesh import MeshData

__all__ = [
    "FEATURE_NAMES",
    "GeometryOODResult",
    "GeometryOODGuardrail",
    "extract_geometry_features",
]

#: Ordered list of feature names -- see the module docstring for what each
#: one measures and why it was chosen. Fixed order matches
#: :func:`extract_geometry_features`'s output exactly.
FEATURE_NAMES: List[str] = [
    "centroid_x", "centroid_y", "centroid_z",              # absolute position (translation-sensitive)
    "bbox_extent_x", "bbox_extent_y", "bbox_extent_z",      # absolute size (scale-sensitive)
    "aspect_ratio_xy", "aspect_ratio_yz",                    # proportions / elongation
    "shape_eigval_1", "shape_eigval_2", "shape_eigval_3",   # area-weighted principal shape spread (descending)
    "surface_area",                                          # absolute size/complexity
    "area_to_bbox_area_ratio",                                # surface-complexity / compactness proxy
    "mean_dihedral_curvature_deg",                            # discrete mean-curvature proxy
]
N_FEATURES = len(FEATURE_NAMES)


def _dihedral_curvature_proxy(mesh: MeshData, face_normals: np.ndarray) -> float:
    """Area-independent discrete curvature proxy: the length-weighted mean
    angle (degrees) between the two face normals sharing each interior
    mesh edge. A flat panel contributes ~0 deg per edge; a sharp crease
    (e.g. a box corner) contributes ~90 deg; a finely tessellated curved
    surface (e.g. a sphere) contributes many small positive angles that
    still average out to a value clearly above 0 -- so this single scalar
    separates "faceted/flat" geometry from "curved/organic" geometry
    without needing a full per-vertex curvature tensor. Only
    exactly-shared (manifold, two-face) edges are counted; true
    mesh-boundary edges (one face) and non-manifold edges (3+ faces)
    contribute no curvature information here and are skipped.

    Implemented independently via vectorized numpy edge-adjacency
    (standard, generic triangle-mesh bookkeeping -- not derived from any
    other module in this repo or from PhysicsNeMo).
    """
    faces = mesh.faces
    n_faces = faces.shape[0]
    if n_faces == 0:
        return 0.0

    # Every face contributes 3 (undirected) edges; canonicalize each edge's
    # vertex order so both faces sharing it produce the identical key.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    face_of_edge = np.concatenate([np.arange(n_faces)] * 3)
    edges_sorted = np.sort(edges, axis=1)

    n_vertices = mesh.n_vertices
    keys = edges_sorted[:, 0].astype(np.int64) * n_vertices + edges_sorted[:, 1].astype(np.int64)

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    faces_sorted = face_of_edge[order]
    verts_sorted = edges_sorted[order]

    unique_keys, start_idx, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    manifold = counts == 2
    if not np.any(manifold):
        return 0.0
    starts = start_idx[manifold]

    face_a = faces_sorted[starts]
    face_b = faces_sorted[starts + 1]
    edge_v0 = verts_sorted[starts, 0]
    edge_v1 = verts_sorted[starts, 1]

    n0 = face_normals[face_a]
    n1 = face_normals[face_b]
    cos_angle = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_angle))

    edge_lengths = np.linalg.norm(mesh.vertices[edge_v0] - mesh.vertices[edge_v1], axis=1)
    total_weight = float(edge_lengths.sum())
    if total_weight <= 0.0:
        return 0.0
    return float(np.sum(angles_deg * edge_lengths) / total_weight)


def extract_geometry_features(mesh: MeshData) -> np.ndarray:
    """Compute the real :data:`FEATURE_NAMES` feature vector for ``mesh``.

    See the module docstring for the full rationale. All 14 numbers are
    plain, documented geometric quantities computed directly from
    ``mesh``'s own vertices/faces -- nothing fitted, learned, or
    approximated by sampling.

    Raises
    ------
    ValueError
        If the mesh has zero/degenerate total surface area (features
        involving area or area-weighting are undefined).
    """
    faces = mesh.faces
    verts = mesh.vertices
    face_normals = mesh.compute_face_normals()
    face_areas = mesh.face_areas()
    total_area = float(face_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError(
            "extract_geometry_features: mesh has zero or non-finite total surface area "
            f"({total_area!r}) -- cannot compute area-weighted geometric features."
        )

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_centroids = (v0 + v1 + v2) / 3.0

    centroid = np.average(face_centroids, axis=0, weights=face_areas)

    b0, b1 = mesh.bounds()
    extents = b1 - b0
    eps = 1e-9
    aspect_xy = float(extents[0] / max(float(extents[1]), eps))
    aspect_yz = float(extents[1] / max(float(extents[2]), eps))

    # Area-weighted covariance of face centroids about the area-weighted
    # centroid -- the eigenvalues of this matrix describe how the
    # geometry's *surface* is spread along its own principal axes (using
    # face centroids, not raw vertices, so a region with denser
    # tessellation does not silently dominate the shape statistics).
    centered = face_centroids - centroid
    w = face_areas / total_area
    cov = (centered * w[:, None]).T @ centered
    eigvals = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)  # descending, non-negative

    curvature = _dihedral_curvature_proxy(mesh, face_normals)

    bbox_area = 2.0 * (
        float(extents[0]) * float(extents[1])
        + float(extents[1]) * float(extents[2])
        + float(extents[0]) * float(extents[2])
    )
    area_ratio = float(total_area / max(bbox_area, eps))

    feats = np.array(
        [
            float(centroid[0]), float(centroid[1]), float(centroid[2]),
            float(extents[0]), float(extents[1]), float(extents[2]),
            aspect_xy, aspect_yz,
            float(eigvals[0]), float(eigvals[1]), float(eigvals[2]),
            total_area,
            area_ratio,
            curvature,
        ],
        dtype=np.float64,
    )
    if feats.shape[0] != N_FEATURES:
        raise RuntimeError(
            f"extract_geometry_features: internal error, produced {feats.shape[0]} features, "
            f"expected {N_FEATURES} (FEATURE_NAMES length)."
        )
    return feats


@dataclass
class GeometryOODResult:
    """One geometry's real anomaly-detection result -- every field is a
    real, computed number, never a bare opaque score (see the module
    docstring's transparency rule).

    Attributes
    ----------
    p_value : float
        ``scipy.stats.chi2.sf(mahalanobis_sq, df=n_features)`` -- the same
        Mahalanobis-distance-to-p-value convention
        ``TrustGate._ood_score`` already uses for coordinate OOD, applied
        here to shape features. In ``[0, 1]``; 1.0 means the geometry's
        shape features sit exactly at the reference distribution's mean,
        values near 0 mean the shape is far out in the tail of what the
        reference set looked like.
    mahalanobis_sq : float
        The raw squared Mahalanobis distance from the reference mean
        (shrunk-covariance-weighted), before the p-value conversion.
    percentile : float
        Empirical percentile of ``mahalanobis_sq`` relative to the
        reference shapes' own (leave-nothing-out) Mahalanobis distances,
        in ``[0, 100]`` -- mirrors PhysicsNeMo's percentile-based
        classification convention, reimplemented independently.
    status : str
        ``"OK"``, ``"WARN"``, or ``"REJECT"`` from thresholding
        ``percentile`` against the guardrail's configured
        ``warn_percentile``/``reject_percentile``.
    most_anomalous_feature : str
        The single :data:`FEATURE_NAMES` entry with the largest
        ``|z-score|`` relative to the reference set's per-feature mean/std
        -- included so a WARN/REJECT verdict is never a bare number: a
        caller can immediately see e.g. "mean_dihedral_curvature_deg" is
        the feature that drove this geometry out of distribution.
    most_anomalous_feature_z : float
        That feature's signed z-score, clipped to +/-1e6. The clip only
        matters for a feature that is (near-)constant across the
        reference set (sample std underflowing to ~0, e.g. a box-shaped
        reference family's surface-to-bbox-area ratio, which is exactly
        1.0 for every axis-aligned box regardless of its dimensions) --
        in that genuinely-degenerate case the raw z-score is a numerically
        meaningless, arbitrarily large number, not a more informative one;
        the clip keeps this field readable while ``mahalanobis_sq``/
        ``p_value`` (computed from the regularized covariance, not from
        this per-feature std) remain the real, well-behaved score.
    n_reference_shapes : int
        How many reference shapes the guardrail was fit on.
    """

    p_value: float
    mahalanobis_sq: float
    percentile: float
    status: str
    most_anomalous_feature: str
    most_anomalous_feature_z: float
    n_reference_shapes: int
    n_features: int = N_FEATURES

    def summary(self) -> str:
        return (
            f"GeometryOODResult: status={self.status}, p_value={self.p_value:.4g}, "
            f"percentile={self.percentile:.1f}, mahalanobis_sq={self.mahalanobis_sq:.4g} "
            f"(n_reference_shapes={self.n_reference_shapes}), most_anomalous_feature="
            f"{self.most_anomalous_feature} (z={self.most_anomalous_feature_z:+.2f})"
        )


class GeometryOODGuardrail:
    """Fits a reference shape distribution and flags new geometries that
    fall outside it. See the module docstring for the full technique and
    the explicit GMM-vs-single-Gaussian scope decision.

    Parameters
    ----------
    warn_percentile, reject_percentile : float
        Empirical-percentile thresholds for the ``WARN``/``REJECT``
        verdict (see :class:`GeometryOODResult`), analogous to
        PhysicsNeMo's ``warn_pct``/``reject_pct``. Defaults (90.0, 99.0)
        are deliberately looser than PhysicsNeMo's defaults (99.0, 99.9)
        because this guardrail is designed to be fit on much smaller
        reference sets (tens, not thousands, of shapes) where a single
        reference shape already occupies several percentile points.
    shrinkage : float
        Fraction (in ``[0, 1]``) by which the sample covariance of the
        reference features is shrunk toward its own diagonal before
        inversion: ``cov = (1-shrinkage)*sample_cov + shrinkage*diag(sample_cov)``.
        A small reference set gives a noisy, sometimes near-singular
        14x14 sample covariance; shrinking toward the diagonal is a
        standard, simple stabilization (in the spirit of Ledoit-Wolf
        shrinkage, though this uses a fixed rather than data-optimal
        shrinkage weight -- an explicit, documented simplification, not a
        hidden one). Default 0.3.
    """

    def __init__(
        self,
        *,
        warn_percentile: float = 90.0,
        reject_percentile: float = 99.0,
        shrinkage: float = 0.3,
    ):
        if not 0.0 <= warn_percentile <= 100.0:
            raise ValueError(f"warn_percentile must be in [0, 100], got {warn_percentile}")
        if not 0.0 <= reject_percentile <= 100.0:
            raise ValueError(f"reject_percentile must be in [0, 100], got {reject_percentile}")
        if warn_percentile > reject_percentile:
            raise ValueError(f"warn_percentile ({warn_percentile}) must be <= reject_percentile ({reject_percentile})")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")

        self.warn_percentile = warn_percentile
        self.reject_percentile = reject_percentile
        self.shrinkage = shrinkage

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._ref_mahalanobis_sq: Optional[np.ndarray] = None
        self._n_reference: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    # -- fitting ------------------------------------------------------

    def fit(self, reference_meshes: Sequence[MeshData]) -> "GeometryOODGuardrail":
        """Fit the reference shape distribution from a REAL list of
        in-distribution meshes (e.g. the shape family a preset/model was
        designed around). Requires at least 2 meshes (need a non-zero
        sample variance per feature); a stable, well-conditioned
        covariance realistically needs several times ``N_FEATURES``
        shapes -- fewer than that still works (via the shrinkage
        regularization) but the resulting distances are cruder."""
        if len(reference_meshes) < 2:
            raise ValueError(
                f"GeometryOODGuardrail.fit needs at least 2 reference meshes to estimate a "
                f"feature distribution, got {len(reference_meshes)}."
            )

        X = np.stack([extract_geometry_features(m) for m in reference_meshes], axis=0)  # (N, 14)
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(N_FEATURES)

        sample_cov = np.atleast_2d(np.cov(X, rowvar=False))
        diag_cov = np.diag(np.diag(sample_cov))
        shrunk = (1.0 - self.shrinkage) * sample_cov + self.shrinkage * diag_cov
        # Small, scale-aware ridge on top of shrinkage, guarding against an
        # exactly-singular matrix when N_FEATURES >= N (same style of
        # regularization TrustGate._cov_inv already uses for coordinate OOD).
        ridge = 1e-8 * max(float(np.trace(shrunk)) / N_FEATURES, 1e-12)
        cov_inv = np.linalg.inv(shrunk + np.eye(N_FEATURES) * ridge)

        self._mean = mean
        self._std = std
        self._cov_inv = cov_inv
        self._n_reference = X.shape[0]

        deltas = X - mean
        self._ref_mahalanobis_sq = np.einsum("ij,jk,ik->i", deltas, cov_inv, deltas)
        return self

    def fit_from_paths(self, paths: Sequence[Union[str, Path]]) -> "GeometryOODGuardrail":
        """Convenience: fit directly from a list of mesh file paths
        (STL/OBJ/STEP/etc., anything ``TrimeshBridge`` can load)."""
        from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge

        bridge = TrimeshBridge()
        meshes = [bridge.load(p) for p in paths]
        return self.fit(meshes)

    # -- scoring --------------------------------------------------------

    def score(self, mesh: MeshData) -> GeometryOODResult:
        """Score one geometry against the fitted reference distribution."""
        if not self.is_fitted:
            raise RuntimeError("GeometryOODGuardrail.fit() must be called before score().")

        x = extract_geometry_features(mesh)
        delta = x - self._mean
        d2 = float(delta @ self._cov_inv @ delta)
        p_value = float(chi2.sf(d2, df=N_FEATURES))
        percentile = 100.0 * float(np.mean(self._ref_mahalanobis_sq <= d2))
        status = self._classify(percentile)

        z_scores = np.clip(delta / np.maximum(self._std, 1e-12), -1e6, 1e6)
        worst_idx = int(np.argmax(np.abs(z_scores)))

        return GeometryOODResult(
            p_value=p_value,
            mahalanobis_sq=d2,
            percentile=percentile,
            status=status,
            most_anomalous_feature=FEATURE_NAMES[worst_idx],
            most_anomalous_feature_z=float(z_scores[worst_idx]),
            n_reference_shapes=self._n_reference,
        )

    def _classify(self, percentile: float) -> str:
        if percentile >= self.reject_percentile:
            return "REJECT"
        elif percentile >= self.warn_percentile:
            return "WARN"
        return "OK"
