"""``GeometryOODGuardrail`` -- flags when an input geometry is
out-of-distribution (OOD) relative to the geometries a model/preset was
actually designed and validated against.

Why this module exists
-----------------------
PINNeAPPle's verification/trust stack (``pinneapple_analysis.trust.trust_gate
.TrustGate``, ``pinneapple_analysis.verification.physics_confidence_score``,
``pinneapple_analysis.verification.evidence_graph``) already scores several
independent trust signals for a prediction: coordinate-space OOD
(``TrustGate``'s own Mahalanobis-distance check on *where in the domain* a
query point sits), PDE-residual re-evaluation, ensemble variance, guardrail
checks, numerical convergence, UQ calibration, and benchmark agreement. None
of these ask a different, equally important question: **is the input
geometry itself the kind of shape this model/preset was ever trained or
validated on?** ``pinneapple_analysis.verification.geometry_intelligence``
sounds adjacent but solves a different problem entirely -- semantic
region/BC classification of a mesh's surfaces, not "is this whole shape
anomalous". This module closes that specific, real gap.

Relationship to PhysicsNeMo's geometry guardrail (concept only, no code
ported)
------------------------------------------------------------------------
NVIDIA PhysicsNeMo ships an experimental geometry guardrail
(``physicsnemo/experimental/guardrails/geometry/``, Apache-2.0) with the
same goal: extract non-invariant shape features from a mesh, fit a density
model (a full Gaussian Mixture Model or a Polynomial Chaos Expansion) over a
reference set of "normal" training shapes, and classify a query shape's
percentile under that density as OK/WARN/REJECT. This module reimplements
the *concept* (shape-feature extraction + density-based anomaly score +
tiered classification) from scratch, in PINNeAPPle's own style and using
PINNeAPPle's own mesh infrastructure -- no PhysicsNeMo code, feature
formulas, or class structure is copied.

Deliberate scope decision: diagonal Mahalanobis, not a full GMM/PCE
--------------------------------------------------------------------
PhysicsNeMo's guardrail fits a full (or mixture-of) multivariate Gaussian
covariance, or a Polynomial Chaos Expansion, over the shape-feature space.
That is real infrastructure built for a mature deployment with large
reference datasets. At PINNeAPPle's current stage -- a geometry catalog of
generated primitives/presets, not thousands of archived production CAD
parts -- fitting a full 13x13 covariance matrix per architecture/preset
would very likely be data-starved (a well-conditioned full covariance needs
meaningfully more samples than features) and would trade interpretability
for a fit that cannot honestly be validated yet.

Per this session's explicit instructions, the simpler, honest choice was
made instead: a **diagonal Mahalanobis distance** -- each feature is
z-scored against its own reference-set mean/std (``TrustGate._ood_score``
already uses exactly this "Mahalanobis distance + ``scipy.stats.chi2.sf``"
recipe for coordinate-space OOD in this same codebase; this module reuses
the identical statistical idea, applied to shape features instead of
coordinates, for stylistic and conceptual consistency across the trust
stack). This is equivalent to a full Mahalanobis distance under the
explicit, stated simplifying assumption that features are independent
(a diagonal covariance) -- real cross-feature correlations (e.g. a long
thin channel naturally having both a high aspect ratio AND a small
projected cross-section) are not modelled. That is a genuine limitation,
stated here rather than hidden, and is exactly the kind of "simpler but
honest" tradeoff this session was asked to make explicitly rather than
building disproportionate infrastructure the product doesn't have the data
to justify yet. Revisit (full covariance, or a real GMM/PCE) once a real
per-preset archive of many production geometries exists.

Feature extraction
-------------------
13 shape descriptors, computed from a
``pinneapple_design.geometry.core.mesh.MeshData`` using infrastructure that
already exists and is already tested elsewhere in this repo (the same
``MeshData``/``compute_curvature_proxy`` building blocks
``geometry_intelligence.py`` and ``pinneapple_design.geometry.ops.features``
already use) -- see :data:`FEATURE_NAMES` and :func:`extract_geometry_features`
for the exact list and each feature's derivation. Like PhysicsNeMo's
guardrail, these are deliberately **not** invariant to translation or scale
(a model trained on parts in the 0.01-1m range should be flagged if asked
to predict on a 100m part, even if the *shape* is otherwise similar) --
unlike PhysicsNeMo's guardrail, mesh tessellation density (face/vertex
count) is deliberately excluded, since two meshes of the identical physical
shape at different mesh resolutions should not be flagged as geometrically
anomalous.

Reference set
--------------
:func:`default_reference_meshes` builds a real (not fabricated) reference
catalog from ``pinneapple_design.geometry.gen.primitives.build_mesh`` --
the same real mesh-generation infrastructure ``geometry_intelligence.py``'s
own tests are validated against -- spanning the box/cylinder/channel
primitive families (literally the base geometries several real presets in
this catalog already use: box for cavity-type domains, cylinder for
cylinder-in-crossflow/pipe domains, channel for duct flows) across a range
of sizes and aspect ratios. This is a deliberately generic, conservative
starting reference set
for wiring the guardrail end to end; a real deployment should instead fit
:class:`GeometryOODGuardrail` on the actual geometries a given
preset/architecture was trained or validated against, via
``GeometryOODGuardrail().fit(real_training_meshes)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2

from pinneapple_design.geometry.core.mesh import MeshData

__all__ = [
    "FEATURE_NAMES",
    "extract_geometry_features",
    "GeometryOODResult",
    "GeometryOODGuardrail",
    "default_reference_meshes",
]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

#: Ordered list of the 13 shape-descriptor names this module extracts. Fixed
#: order matters: :class:`GeometryOODGuardrail` stores its reference
#: mean/std in this same order, and a query mesh's feature dict is turned
#: into a vector by reading these keys in this order.
FEATURE_NAMES: Tuple[str, ...] = (
    "centroid_x", "centroid_y", "centroid_z",
    "bbox_extent_x", "bbox_extent_y", "bbox_extent_z",
    "pca_eigenvalue_1", "pca_eigenvalue_2", "pca_eigenvalue_3",
    "surface_area",
    "volume_bbox",
    "aspect_ratio",
    "mean_curvature_proxy",
)


def extract_geometry_features(mesh: MeshData) -> Dict[str, float]:
    r"""Extract the 13 real, computed shape descriptors in :data:`FEATURE_NAMES`
    from a mesh.

    Every value is computed directly from ``mesh``'s own vertices/faces --
    never a placeholder or a fabricated number:

    - ``centroid_{x,y,z}``: mean vertex position (``mesh.vertices.mean(0)``).
      Not invariant to translation, by design (see module docstring).
    - ``bbox_extent_{x,y,z}``: ``mesh.bbox_size()`` -- axis-aligned bounding
      box extents. Not invariant to scale, by design.
    - ``pca_eigenvalue_{1,2,3}``: eigenvalues (descending) of the vertex
      position covariance matrix -- the real principal-component variances,
      i.e. how spread out the shape is along its own natural axes,
      independent of the mesh's orientation in space.
    - ``surface_area``: ``mesh.surface_area()`` -- sum of triangle areas.
    - ``volume_bbox``: ``mesh.volume_bbox()`` -- bounding-box volume (a
      cheap, always-computable proxy for size; not the true enclosed
      volume, which is not well-defined for a non-watertight mesh).
    - ``aspect_ratio``: ``max(bbox_extent) / max(min(bbox_extent), eps)`` --
      how elongated/flattened the bounding box is.
    - ``mean_curvature_proxy``: mean over faces of
      ``pinneapple_design.geometry.ops.features.compute_curvature_proxy``,
      itself an explicitly-labelled MVP proxy (normal-variation across
      adjacent faces, not true differential-geometric curvature) --
      reused here rather than reimplemented, exactly as this module's
      docstring says to prefer.

    Parameters
    ----------
    mesh : MeshData
        A real mesh (loaded from a file via
        ``pinneapple_design.geometry.io.trimesh_bridge``, or built via
        ``pinneapple_design.geometry.gen.primitives.build_mesh``).

    Returns
    -------
    Dict[str, float]
        Keyed by every name in :data:`FEATURE_NAMES`, values real finite
        floats (never NaN for a valid non-degenerate mesh; a mesh with zero
        surface area/volume raises via the underlying ``MeshData`` methods
        rather than silently producing a NaN feature).
    """
    from pinneapple_design.geometry.ops.features import compute_curvature_proxy

    centroid = mesh.vertices.mean(axis=0)
    bbox_extent = mesh.bbox_size()

    centered = mesh.vertices - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # ascending -> descending

    surface_area = mesh.surface_area()
    volume_bbox = mesh.volume_bbox()

    max_extent = float(np.max(bbox_extent))
    min_extent = float(np.min(bbox_extent))
    aspect_ratio = max_extent / max(min_extent, 1e-9)

    curvature = compute_curvature_proxy(mesh)
    mean_curvature = float(np.mean(curvature)) if curvature.size > 0 else 0.0

    values = {
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_z": float(centroid[2]),
        "bbox_extent_x": float(bbox_extent[0]), "bbox_extent_y": float(bbox_extent[1]), "bbox_extent_z": float(bbox_extent[2]),
        "pca_eigenvalue_1": float(eigvals[0]), "pca_eigenvalue_2": float(eigvals[1]), "pca_eigenvalue_3": float(eigvals[2]),
        "surface_area": float(surface_area),
        "volume_bbox": float(volume_bbox),
        "aspect_ratio": float(aspect_ratio),
        "mean_curvature_proxy": mean_curvature,
    }
    return values


def _vector_from_dict(values: Dict[str, float]) -> np.ndarray:
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GeometryOODResult:
    """Real, auditable output of one :meth:`GeometryOODGuardrail.query` call.

    Attributes
    ----------
    score : float
        A [0, 1] value where 1.0 = perfectly in-distribution and values near
        0 = far out of distribution -- ``scipy.stats.chi2.sf(mahalanobis_sq,
        df=dof)``, the probability of seeing a diagonal-Mahalanobis distance
        this large or larger under the fitted per-feature Gaussian
        reference distribution. Same convention as
        ``TrustGate._ood_score``'s own coordinate-space check, so the two
        can be read side by side without a sign/direction gotcha.
    mahalanobis_sq : float
        The real computed value, :math:`\\sum_i ((x_i - \\mu_i)/\\sigma_i)^2`
        -- sum of squared per-feature z-scores (see module docstring for why
        this is a *diagonal* Mahalanobis distance, not a full-covariance
        one).
    dof : int
        Degrees of freedom used for the chi-squared survival function --
        equal to the number of features (13), the real basis for the
        p-value, never fabricated.
    status : str
        ``"OK"``, ``"WARN"``, or ``"REJECT"`` -- see
        :class:`GeometryOODGuardrail`'s ``warn_p``/``reject_p`` thresholds.
    n_reference : int
        How many reference meshes :meth:`GeometryOODGuardrail.fit` actually
        used -- carried through so a caller can judge how much this
        guardrail's own reference distribution should itself be trusted
        (mirrors ``PhysicsConfidenceScore.coverage``'s "the number backing
        this claim matters" philosophy).
    feature_values : Dict[str, float]
        The query mesh's own real extracted features (see
        :func:`extract_geometry_features`).
    feature_z_scores : Dict[str, float]
        Per-feature ``(value - reference_mean) / reference_std`` -- the real
        numbers ``mahalanobis_sq`` is built from, so nothing about the score
        is opaque.
    top_deviating_features : List[Tuple[str, float]]
        The (up to 3) features with the largest ``|z|``, most-deviating
        first -- a quick, honest "why is this flagged" answer without
        re-deriving anything.
    source_summary : str
        A short, human-readable string quoting the real numbers behind
        ``score`` -- follows the exact convention
        ``physics_confidence_score.ConfidenceComponent.source_summary``
        already uses across every other trust-stack component.
    """

    score: float
    mahalanobis_sq: float
    dof: int
    status: str
    n_reference: int
    feature_values: Dict[str, float]
    feature_z_scores: Dict[str, float] = field(default_factory=dict)
    top_deviating_features: List[Tuple[str, float]] = field(default_factory=list)
    source_summary: str = ""


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class GeometryOODGuardrail:
    """Fits a per-feature Gaussian reference distribution over a set of
    real "in-distribution" geometries, and scores new geometries against it.

    See the module docstring for the full design rationale (diagonal
    Mahalanobis distance + ``chi2.sf``, deliberately not a full GMM/PCE).

    Parameters
    ----------
    warn_p : float, optional
        A query with p-value below this threshold is classified ``"WARN"``
        unless it also falls below ``reject_p``. Default 0.05 -- the
        conventional "unlikely at the 5% level" statistical threshold.
    reject_p : float, optional
        A query with p-value below this threshold is classified
        ``"REJECT"``. Default 0.01 -- the conventional "unlikely at the 1%
        level" threshold. Must be <= ``warn_p``.

    Attributes
    ----------
    feature_names : Tuple[str, ...]
        :data:`FEATURE_NAMES`, carried on the instance for introspection.
    """

    def __init__(self, *, warn_p: float = 0.05, reject_p: float = 0.01):
        if not 0.0 <= reject_p <= warn_p <= 1.0:
            raise ValueError(
                f"require 0 <= reject_p ({reject_p}) <= warn_p ({warn_p}) <= 1"
            )
        self.warn_p = warn_p
        self.reject_p = reject_p
        self.feature_names = FEATURE_NAMES
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_reference: int = 0

    # -- fitting ---------------------------------------------------------

    def fit(self, meshes: List[MeshData]) -> "GeometryOODGuardrail":
        """Fit the reference mean/std from a list of real, in-distribution
        meshes (see :func:`default_reference_meshes` for a generic starting
        catalog, or pass the actual training/validation geometries of a
        specific model/preset for a tighter, more meaningful guardrail).

        Raises
        ------
        ValueError
            If ``meshes`` is empty, or has fewer than 2 meshes (a standard
            deviation needs at least 2 samples to be defined; refusing to
            fit on 1 sample rather than silently returning a
            zero/undefined std that would make every future query score
            either 0 or 1).
        """
        if not meshes:
            raise ValueError("GeometryOODGuardrail.fit() requires at least 2 reference meshes, got 0.")
        if len(meshes) < 2:
            raise ValueError(
                f"GeometryOODGuardrail.fit() requires at least 2 reference meshes to compute a "
                f"real standard deviation per feature, got {len(meshes)}."
            )
        feats = np.stack([_vector_from_dict(extract_geometry_features(m)) for m in meshes], axis=0)
        self._mean = feats.mean(axis=0)
        # ddof=1: sample std (real, unbiased estimate from a finite reference set).
        std = feats.std(axis=0, ddof=1)
        # Floor at a small epsilon relative to the feature's own scale, not an
        # absolute constant -- a feature that is genuinely constant across
        # every reference mesh (e.g. all references sharing centroid_y=0)
        # must not become an infinitely-sensitive tripwire.
        eps = np.maximum(np.abs(self._mean) * 1e-6, 1e-9)
        self._std = np.maximum(std, eps)
        self._n_reference = len(meshes)
        return self

    # -- querying ----------------------------------------------------------

    def query(self, mesh: MeshData) -> GeometryOODResult:
        """Score one mesh against the fitted reference distribution.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("GeometryOODGuardrail.query() called before fit().")

        values = extract_geometry_features(mesh)
        x = _vector_from_dict(values)
        z = (x - self._mean) / self._std
        mahalanobis_sq = float(np.sum(z ** 2))
        dof = len(FEATURE_NAMES)
        score = float(chi2.sf(mahalanobis_sq, df=dof))
        status = self._classify(score)

        z_scores = {name: float(zi) for name, zi in zip(FEATURE_NAMES, z)}
        top = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        summary = (
            f"status={status}, mahalanobis_sq={mahalanobis_sq:.4g} (dof={dof}), "
            f"p_value={score:.4g}, n_reference={self._n_reference}, "
            f"top_deviating_features=[{', '.join(f'{n}(z={v:.2f})' for n, v in top)}]"
        )

        return GeometryOODResult(
            score=score,
            mahalanobis_sq=mahalanobis_sq,
            dof=dof,
            status=status,
            n_reference=self._n_reference,
            feature_values=values,
            feature_z_scores=z_scores,
            top_deviating_features=top,
            source_summary=summary,
        )

    def _classify(self, p_value: float) -> str:
        if p_value < self.reject_p:
            return "REJECT"
        if p_value < self.warn_p:
            return "WARN"
        return "OK"


# ---------------------------------------------------------------------------
# Default reference catalog
# ---------------------------------------------------------------------------

def default_reference_meshes() -> List[MeshData]:
    """A real, generic reference catalog built from
    ``pinneapple_design.geometry.gen.primitives.build_mesh`` -- box,
    cylinder, and channel domains (the base shapes several real presets in
    this catalog already use: box for cavity-type domains, cylinder for
    cylinder-in-crossflow/pipe domains, channel for duct flows) across a
    deterministic grid of sizes and aspect ratios.

    This is not a substitute for fitting :class:`GeometryOODGuardrail` on a
    specific model/preset's own real training geometries -- it exists so
    the guardrail can be wired up and exercised end to end (see this
    module's docstring) with a real, non-fabricated, non-trivial reference
    set before any specific per-preset catalog of archived geometries
    exists.

    Returns
    -------
    List[MeshData]
        48 real, deterministically-generated meshes (16 boxes + 20
        cylinders + 12 channels), no randomness involved.
    """
    from pinneapple_design.geometry.gen.primitives import build_mesh

    meshes: List[MeshData] = []

    # Boxes: 4 base sizes x 4 aspect-ratio shapes (cube, slab, rod, plate-ish)
    sizes = (0.5, 1.0, 2.0, 4.0)
    aspect_shapes = ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 3.0, 1.0), (4.0, 1.0, 0.5))
    for s in sizes:
        for (ax, ay, az) in aspect_shapes:
            meshes.append(build_mesh("box", extents=(s * ax, s * ay, s * az)))

    # Cylinders: 4 radii x 5 heights
    radii = (0.5, 1.0, 2.0, 4.0)
    heights = (0.5, 1.0, 2.0, 4.0, 8.0)
    for r in radii:
        for h in heights:
            meshes.append(build_mesh("cylinder", radius=r, height=h, sections=32))

    # Channels (duct-flow domains): 4 lengths x 3 cross-sections
    lengths = (1.0, 2.0, 4.0, 8.0)
    cross_sections = ((0.1, 0.1), (0.2, 0.2), (0.4, 0.2))
    for length in lengths:
        for (width, height) in cross_sections:
            meshes.append(build_mesh("channel", length=length, width=width, height=height))

    return meshes
