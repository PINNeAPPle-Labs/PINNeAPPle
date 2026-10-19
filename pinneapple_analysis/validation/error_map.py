"""Where is a field wrong? Pointwise error of a prediction against a reference on the same points, the regions where
it concentrates, and a short text that compares them with the global number.

    from pinneapple_analysis.validation.error_map import error_map
    em = error_map(points, pred, ref, regions={"clamp": clamp_nodes, "tip": tip_nodes})
    em.error                # |pred - ref| per point (vectors: norm of the difference)
    em.hotspots             # clusters of the largest errors: centre, extent, max, share of the squared error
    em.regions              # per named region: share of the squared error against its share of the points
    em.summary()            # "Relative L2 error 3.9 % ... 41 % of the squared error is in clamp (10 % of the points)"

Every number comes from the arrays given: nothing is estimated or interpolated here (map the reference onto the
prediction's points first when the meshes differ). ``weights`` (cell volumes, nodal areas) make the shares integrals
instead of point counts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ErrorMap", "error_map", "pointwise_error", "error_hotspots", "error_by_region", "grow_region"]


def pointwise_error(pred, ref) -> np.ndarray:
    """``|pred - ref|`` per point: (n,) for scalars, the Euclidean norm of the difference for (n, k) vectors."""
    a, b = np.asarray(pred, float), np.asarray(ref, float)
    if a.shape != b.shape:
        raise ValueError(f"prediction {a.shape} and reference {b.shape} differ: map one onto the other's points first")
    d = a - b
    return np.abs(d) if d.ndim == 1 else np.linalg.norm(d.reshape(len(d), -1), axis=1)


def _spacing(P: np.ndarray) -> float:
    """Median distance to the nearest other point (the mesh spacing for nodes or cell centres)."""
    from scipy.spatial import cKDTree
    if len(P) < 2:
        return 1.0
    d, _ = cKDTree(P).query(P, k=2)
    s = float(np.median(d[:, 1]))
    return s if s > 0 else float(np.ptp(P, axis=0).max() or 1.0) / max(len(P), 1)


def error_hotspots(points, err, *, n: int = 5, quantile: float = 0.9, link: float | None = None,
                   weights=None) -> list[dict[str, Any]]:
    """Clusters of the points whose error is at or above the ``quantile`` of the error, linked when closer than
    ``link`` (default 1.5 times the median point spacing: neighbours along the mesh lines, not across them). The ``n`` clusters with the largest maximum error, each with
    its centre (error-weighted), bounding box, number of points, max and mean error, the location of its max and its
    share of the total squared error (weighted by ``weights`` if given)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = np.asarray(points, float).reshape(len(err), -1)
    e = np.asarray(err, float)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(e)
    if not ok.any() or float(np.nanmax(e)) <= 0:
        return []
    thr = float(np.quantile(e[ok], quantile))
    sel = np.where(ok & (e >= thr) & (e > 0))[0]
    r = link if link is not None else 1.5 * _spacing(P)
    pairs = cKDTree(P[sel]).query_pairs(r, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(sel), len(sel))) if len(pairs) \
        else coo_matrix((len(sel), len(sel)))
    _, lab = connected_components(adj, directed=False)
    total = float(np.sum(w[ok] * e[ok] ** 2)) or 1.0
    out = []
    for k in np.unique(lab):
        idx = sel[lab == k]
        ek = e[idx]
        j = idx[int(np.argmax(ek))]
        out.append({"centre": (P[idx] * (ek / ek.sum())[:, None]).sum(0).tolist(),
                    "bbox_min": P[idx].min(0).tolist(), "bbox_max": P[idx].max(0).tolist(),
                    "n": int(len(idx)), "max": float(ek.max()), "mean": float(ek.mean()),
                    "at": P[j].tolist(), "index": int(j),
                    "share": float(np.sum(w[idx] * ek ** 2) / total), "indices": idx})
    out.sort(key=lambda h: -h["max"])
    return out[:n]


def grow_region(points, idx, radius: float) -> np.ndarray:
    """The points within ``radius`` of any point of ``idx`` (e.g. the nodes near a clamped face)."""
    from scipy.spatial import cKDTree
    P = np.asarray(points, float)
    idx = np.asarray(idx, np.int64)
    if not len(idx):
        return idx
    d, _ = cKDTree(P[idx]).query(P, k=1, distance_upper_bound=radius)
    return np.where(np.isfinite(d))[0]


def error_by_region(err, regions: Mapping[str, Sequence[int]], *, weights=None) -> list[dict[str, Any]]:
    """Per named region (point indices): its share of the points (or of the weights), its share of the squared error,
    the ratio of its RMS error to the RMS over all points (> 1: the error concentrates there), max and mean."""
    e = np.asarray(err, float)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(e)
    W, S = float(w[ok].sum()) or 1.0, float(np.sum(w[ok] * e[ok] ** 2)) or 1.0
    rms = np.sqrt(S / W)
    out = []
    for name, idx in regions.items():
        idx = np.asarray(idx, np.int64)
        idx = idx[ok[idx]] if len(idx) else idx
        if not len(idx):
            continue
        wr, sr = float(w[idx].sum()), float(np.sum(w[idx] * e[idx] ** 2))
        out.append({"name": name, "n": int(len(idx)), "share_points": wr / W, "share_error": sr / S,
                    "rms_ratio": float(np.sqrt(sr / wr) / rms) if rms > 0 and wr > 0 else 0.0,
                    "max": float(e[idx].max()), "mean": float(e[idx].mean())})
    return out


@dataclass
class ErrorMap:
    """Result of :func:`error_map`. ``rel_l2`` is ``||pred - ref|| / ||ref||`` over the points."""
    error: np.ndarray
    rel_l2: float
    max_error: float
    rms_error: float
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    unit: str = ""
    label: str = "error"

    def where(self, point) -> str:
        """Name of the region that holds ``point`` (one of the hotspots' points), or ""."""
        return next((r["name"] for r in self.regions if point in r.get("_idx", ())), "")

    def summary(self, reported: Mapping[str, float] | None = None) -> list[str]:
        """Short sentences: the global number (and reported ones to compare with, label -> relative error, e.g. the
        study mean), where the error concentrates among the named regions, and the largest hotspot."""
        u = f" {self.unit}" if self.unit else ""
        rep = "; ".join(f"{k}: {100 * v:.2f} %" for k, v in (reported or {}).items() if v is not None)
        lines = [f"Relative L2 error {100 * self.rel_l2:.2f} %" + (f" ({rep})" if rep else "")
                 + f"; RMS {_g(self.rms_error)}{u}, max {_g(self.max_error)}{u}."]
        if self.regions:
            top = max(self.regions, key=lambda r: r["rms_ratio"])
            if top["rms_ratio"] >= 1.5:
                lines.append(f"The error concentrates in the {top['name']}: {100 * top['share_error']:.0f} % of the squared "
                             f"error on {100 * top['share_points']:.0f} % of the points (RMS {top['rms_ratio']:.1f}× "
                             f"the overall RMS).")
            else:
                lines.append("No named region dominates: the largest RMS ratio is "
                             f"{top['rms_ratio']:.1f}× (the {top['name']}).")
            lines.append("; ".join(f"{r['name']}: {100 * r['share_error']:.0f} % of the squared error on "
                                   f"{100 * r['share_points']:.0f} % of the points" for r in self.regions) + ".")
        if self.hotspots:
            h = self.hotspots[0]
            where = self.where(h["index"])
            lines.append(f"Largest error {_g(h['max'])}{u} at ({', '.join(_g(x) for x in h['at'])})"
                         + (f", in the {where}" if where else "")
                         + f"; that hotspot holds {100 * h['share']:.0f} % of the squared error.")
        return lines

    def to_dict(self, reported: Mapping[str, float] | None = None) -> dict[str, Any]:
        return {"rel_l2": self.rel_l2, "max": self.max_error, "rms": self.rms_error, "unit": self.unit,
                "hotspots": [{k: v for k, v in h.items() if k != "indices"} for h in self.hotspots],
                "regions": [{k: v for k, v in r.items() if k != "_idx"} for r in self.regions],
                "summary": self.summary(reported)}


def _g(v: float) -> str:
    return f"{v:.3g}"


def error_map(points, pred, ref, *, regions: Mapping[str, Sequence[int]] | None = None, weights=None,
              n_hotspots: int = 5, quantile: float = 0.9, unit: str = "", label: str = "error") -> ErrorMap:
    """Pointwise error, hotspots and per-region shares of ``pred`` against ``ref`` on ``points``. ``regions``: name ->
    point indices (e.g. a clamp and a loaded face, grown with :func:`grow_region`)."""
    a, b = np.asarray(pred, float), np.asarray(ref, float)
    e = pointwise_error(a, b)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, float)
    d2 = (a - b).reshape(len(e), -1) ** 2
    rel = float(np.sqrt(np.sum(w[:, None] * d2) / max(np.sum(w[:, None] * b.reshape(len(e), -1) ** 2), 1e-300)))
    regs = error_by_region(e, regions or {}, weights=weights)
    for r in regs:
        r["_idx"] = set(np.asarray((regions or {})[r["name"]], np.int64).tolist())
    return ErrorMap(e, rel, float(np.nanmax(e)) if len(e) else 0.0,
                    float(np.sqrt(np.sum(w * e ** 2) / max(w.sum(), 1e-300))),
                    error_hotspots(points, e, n=n_hotspots, quantile=quantile, weights=weights), regs, unit, label)
