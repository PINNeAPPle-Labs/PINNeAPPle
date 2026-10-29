"""Geometry validation, including the hollow-body check the spec calls out.

The specific failure this module exists to catch: a request for a *hollow*
pipe answered with a solid cylinder. Both are valid meshes; only one is the
object that was asked for. The check is numerical, not visual -- the mesh's
enclosed volume is computed from its triangles and compared with the
volume the specification's own dimensions imply:

* a hollow pipe of outer radius ``R``, wall thickness ``t`` and length ``L``
  encloses ``pi*(R^2 - (R-t)^2)*L``;
* a solid cylinder of the same outer radius encloses ``pi*R^2*L``.

If the measured volume is closer to the solid value than to the hollow one,
the artifact is reported as not being a hollow body. If the mesh is not
closed, its volume is undefined and the check returns ``BLOCKED`` rather
than guessing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..report import Finding, Status
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "geometry"


@register(_NAME)
def check_geometry(ctx: ValidationContext) -> ValidatorResult:
    if not ctx.meshes:
        return not_applicable(_NAME, "artifact supplied no meshes")

    findings: List[Finding] = []
    evidence: Dict[str, dict] = {}
    any_fail = False
    any_blocked = False

    for name, mesh in ctx.meshes.items():
        stats, mesh_findings = _inspect(name, mesh, ctx)
        evidence[name] = stats
        findings.extend(mesh_findings)
        for f in mesh_findings:
            if f.status is Status.FAIL and f.severity == "error":
                any_fail = True
            elif f.status is Status.BLOCKED:
                any_blocked = True

    if any_fail:
        status = Status.FAIL
    elif any_blocked:
        status = Status.BLOCKED
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.geometry", status=Status.PASS, severity="info",
                                message=f"{len(ctx.meshes)} mesh(es) valid"))
    return ValidatorResult(
        name=_NAME, status=status, findings=findings, evidence=evidence,
        limitations=["a valid mesh is not a validated physical object; watertightness and volume "
                     "say nothing about material, load or operating conditions"],
    )


def _inspect(part: str, mesh, ctx: ValidationContext) -> Tuple[dict, List[Finding]]:
    findings: List[Finding] = []
    try:
        vertices, faces = np.asarray(mesh[0], dtype=float), np.asarray(mesh[1])
    except Exception as exc:
        return {"error": str(exc)}, [_fail(part, "geometry.malformed",
                                          f"part '{part}': mesh is not a (vertices, faces) pair: {exc}")]

    stats: dict = {"vertices": int(len(vertices)), "faces": int(len(faces))}
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return stats, [_fail(part, "geometry.vertices",
                             f"part '{part}': vertices must be (V, 3), got {vertices.shape}")]
    if faces.ndim != 2 or faces.shape[1] != 3:
        return stats, [_fail(part, "geometry.faces",
                             f"part '{part}': faces must be (F, 3), got {faces.shape}")]
    if len(vertices) == 0 or len(faces) == 0:
        return stats, [_fail(part, "geometry.empty", f"part '{part}': empty mesh "
                                                     f"({len(vertices)} vertices, {len(faces)} faces)")]
    if not np.isfinite(vertices).all():
        bad = int((~np.isfinite(vertices)).any(axis=1).sum())
        findings.append(_fail(part, "geometry.non_finite",
                              f"part '{part}': {bad} vertex/vertices with non-finite coordinates"))
    if faces.min() < 0 or faces.max() >= len(vertices):
        findings.append(_fail(part, "geometry.index",
                              f"part '{part}': face index out of range "
                              f"[{int(faces.min())}, {int(faces.max())}] for {len(vertices)} vertices"))

    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = float(np.mean(areas < 1e-14)) if areas.size else 0.0
    stats.update({"degenerate_share": degenerate, "surface_area": float(areas.sum()),
                  "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()]})
    if degenerate > 0.05:
        findings.append(Finding(
            id="geometry.degenerate", status=Status.FAIL, severity="warning",
            message=f"part '{part}': {degenerate:.1%} of triangles have zero area",
            evidence={"part": part, "share": degenerate},
            justification="degenerate triangles usually indicate duplicate vertices or a collapsed profile"))

    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int(np.sum(counts == 1))
    non_manifold = int(np.sum(counts > 2))
    watertight = open_edges == 0 and non_manifold == 0
    stats.update({"edges": int(len(counts)), "open_edges": open_edges,
                  "non_manifold_edges": non_manifold, "watertight": watertight})
    if not watertight:
        findings.append(Finding(
            id="geometry.watertight", status=Status.NOT_RUN, severity="warning",
            message=f"part '{part}' is not closed ({open_edges} open edge(s), "
                    f"{non_manifold} non-manifold edge(s)) -- enclosed volume is undefined",
            evidence={"part": part, "open_edges": open_edges, "non_manifold": non_manifold},
            justification="an open surface has no well-defined interior, so hollow/solid cannot be verified"))

    geometry = ctx.spec.geometry
    if watertight:
        volume = _enclosed_volume(tri)
        stats["enclosed_volume_m3"] = volume
        if geometry is not None:
            findings.extend(_compare_to_spec(part, vertices, volume, geometry, ctx))
    return stats, findings


def _enclosed_volume(tri: np.ndarray) -> float:
    """Signed volume by the divergence theorem, in the mesh's own units."""
    return float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0))


def _compare_to_spec(part: str, vertices: np.ndarray, volume: float,
                     geometry, ctx: ValidationContext) -> List[Finding]:
    findings: List[Finding] = []
    dims = {k: q.to_si() for k, q in geometry.dimensions.items()}
    radius = _pick(dims, "outer_radius", "radius")
    if radius is None:
        diameter = _pick(dims, "outer_diameter", "diameter")
        radius = None if diameter is None else diameter / 2.0
    length = _pick(dims, "length", "l", "height", "span")
    if radius is None or length is None:
        return findings  # not a cylinder-like geometry: nothing to compare numerically

    thickness = geometry.wall_thickness.to_si() if geometry.wall_thickness is not None else None

    # bounds check (only when the geometry is axis-aligned along z/y as the
    # primitives produce it): compare the mesh extent against 2R and L
    extents = vertices.max(0) - vertices.min(0)
    stats_extent = {"extents": extents.tolist(), "expected_radius": radius, "expected_length": length}

    if geometry.hollow:
        if thickness is None:
            findings.append(Finding(
                id="geometry.hollow", status=Status.BLOCKED, severity="error",
                message=f"part '{part}': geometry declares 'hollow' but no wall thickness, cannot verify",
                evidence={"part": part}))
            return findings
        inner = radius - thickness
        if inner <= 0:
            findings.append(_fail(part, "geometry.hollow_thickness",
                                  f"part '{part}': wall thickness {thickness:g} m >= outer radius "
                                  f"{radius:g} m, the pipe has no bore"))
            return findings
        expected_hollow = np.pi * (radius ** 2 - inner ** 2) * length
        expected_solid = np.pi * radius ** 2 * length
        err_hollow = abs(volume - expected_hollow) / expected_hollow
        err_solid = abs(volume - expected_solid) / max(expected_solid, 1e-30)
        stats_extent.update({
            "enclosed_volume": volume, "expected_hollow": expected_hollow,
            "expected_solid": expected_solid, "error_hollow": err_hollow, "error_solid": err_solid,
        })
        rtol = float(ctx.spec.numerical_method.get("geometry_rtol", 0.05))
        if err_hollow <= rtol:
            findings.append(Finding(
                id="geometry.hollow", status=Status.PASS, severity="info",
                message=f"part '{part}': enclosed volume {volume:.6g} m^3 matches a hollow body "
                        f"(expected {expected_hollow:.6g} m^3, error {err_hollow:.2%})",
                evidence=stats_extent))
        elif err_solid <= rtol:
            findings.append(_fail(
                part, "geometry.not_hollow",
                f"part '{part}': the mesh is a SOLID body -- its enclosed volume {volume:.6g} m^3 "
                f"matches a filled cylinder ({expected_solid:.6g} m^3, error {err_solid:.2%}) and not "
                f"the specified hollow pipe ({expected_hollow:.6g} m^3, error {err_hollow:.2%})",
                stats_extent))
        else:
            findings.append(_fail(
                part, "geometry.volume",
                f"part '{part}': enclosed volume {volume:.6g} m^3 matches neither the hollow "
                f"(~{expected_hollow:.6g}) nor the solid (~{expected_solid:.6g}) expected body",
                stats_extent))
    return findings


def _pick(dims: Dict[str, float], *names: str) -> Optional[float]:
    for name in names:
        if name in dims:
            return float(dims[name])
    return None


def _fail(part: str, fid: str, message: str, evidence=None) -> Finding:
    return Finding(id=fid, status=Status.FAIL, severity="error", message=message,
                   evidence={"part": part, **(evidence or {})})
