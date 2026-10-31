"""A digital-twin page for a surrogate: sliders over the design, the predicted surface field in 3D (three.js), the
calibrated uncertainty band and a warning outside the region the surrogate was validated in.

    payload = twin_payload(predict, field_predict, bounds, geometry, train_unit, grid=(41, 11, 11))
    write_twin(payload, "twin/")         # twin/index.html + twin/twin.json (+ the vendored three.js)

The page does not run the models: :func:`twin_payload` evaluates them on a regular grid of the inputs (scalars:
mean and 90 % half width; the field compressed by a POD of its predictions) and the page interpolates
multilinearly. The interpolation error against the models themselves is measured on random points and stored in the
payload (``interpolation_error``), so the page shows what it costs. ``geometry(params) -> vertices`` gives the body
for each value of the inputs that change its shape (sampled on the grid of those inputs, interpolated on the page).
Trust: every grid node stores its distance to the nearest training case (unit cube) and whether it lies inside the
training box; the page warns beyond ``max_distance`` or outside the box.

A "verify with CFD" button posts the slider values to ``<api>/verify`` when the page is served by an app that runs
the reference solver (``apps/ahmed_twin``); the static page says the button needs that app.
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["twin_payload", "write_twin", "TEMPLATE"]

TEMPLATE = os.path.join(os.path.dirname(__file__), "web", "twin.html")


def _grid(bounds: Mapping[str, tuple[float, float]], shape: Sequence[int]) -> tuple[list[np.ndarray], np.ndarray]:
    axes = [np.linspace(0.0, 1.0, n) for n in shape]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(shape))
    return axes, G


def _interp(axes: Sequence[np.ndarray], values: np.ndarray, U: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator
    shape = [len(a) for a in axes]
    V = values.reshape(*shape, -1)
    return RegularGridInterpolator(axes, V)(np.clip(U, 0, 1))


def twin_payload(predict: Callable[[np.ndarray], Mapping[str, tuple[np.ndarray, np.ndarray]]],
                 field_predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
                 bounds: Mapping[str, tuple[float, float]], geometry: Callable[[Mapping[str, float]], np.ndarray],
                 faces: np.ndarray, train_unit: np.ndarray, *, grid: Sequence[int] = (41, 11, 11),
                 shape_inputs: Sequence[str] = (), units: Mapping[str, str] | None = None,
                 max_distance: float = 0.2, field_modes: int = 24, title: str = "Digital twin",
                 field_label: str = "Cp", scalar_labels: Mapping[str, str] | None = None,
                 meta: Mapping[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
    """Evaluate the models on the grid and package everything the page needs (JSON-ready)."""
    from scipy.spatial import cKDTree
    names = list(bounds)
    axes, G = _grid(bounds, grid)
    P = predict(G)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    out: dict[str, Any] = {"title": title, "inputs": [{"name": k, "low": float(bounds[k][0]), "high": float(bounds[k][1]),
                                                       "unit": (units or {}).get(k, ""), "n": int(n)}
                                                      for k, n in zip(names, grid, strict=False)],
                           "scalars": {}, "max_distance": float(max_distance), "meta": dict(meta or {})}
    rng = np.random.default_rng(seed)
    Ur = rng.random((256, len(names)))
    Pr = predict(Ur)
    err = {}
    for k, (m, h) in P.items():
        out["scalars"][k] = {"mean": np.round(m, 6).tolist(), "half_width": np.round(h, 6).tolist(),
                             "label": (scalar_labels or {}).get(k, k)}
        mi = _interp(axes, np.asarray(m), Ur)[:, 0]
        err[k] = float(np.max(np.abs(mi - Pr[k][0])))
    tree = cKDTree(np.atleast_2d(train_unit))
    d, _ = tree.query(G)
    box_lo, box_hi = np.atleast_2d(train_unit).min(0), np.atleast_2d(train_unit).max(0)
    out["trust"] = {"distance": np.round(d, 4).tolist(),
                    "inside_box": True, "box_lo": box_lo.tolist(), "box_hi": box_hi.tolist()}
    if field_predict is not None:
        M, H = field_predict(G)
        mu = M.mean(0)
        U_, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
        r = int(min(field_modes, len(s)))
        C = (M - mu) @ Vt[:r].T
        rec = mu + C @ Vt[:r]
        Mr, _ = field_predict(Ur[:64])
        Ci = _interp(axes, C, Ur[:64])
        err[f"field_{field_label}"] = float(np.max(np.abs(mu + Ci @ Vt[:r] - Mr)))
        err[f"field_{field_label}_pod_truncation"] = float(np.max(np.abs(rec - M)))
        out["field"] = {"label": field_label, "mean": np.round(mu, 5).tolist(), "modes": np.round(Vt[:r], 6).tolist(),
                        "coeffs": np.round(C, 6).tolist(), "half_width_mean": np.round(H.mean(0), 5).tolist(),
                        "half_width_scale": np.round(H.mean(1) / max(H.mean(), 1e-12), 4).tolist(),
                        "range": [float(np.percentile(M, 1)), float(np.percentile(M, 99))]}
    out["interpolation_error"] = err
    # geometry: vertices sampled over the shape inputs (others at their midpoint), interpolated on the page
    js = [names.index(k) for k in shape_inputs]
    mid = {k: 0.5 * (bounds[k][0] + bounds[k][1]) for k in names}
    geo_axes = [axes[j] for j in js]
    Gs = np.stack(np.meshgrid(*geo_axes, indexing="ij"), -1).reshape(-1, len(js)) if js else np.zeros((1, 0))
    verts = []
    for u in Gs:
        p = dict(mid)
        for jj, j in enumerate(js):
            p[names[j]] = float(lo[j] + u[jj] * (hi[j] - lo[j]))
        verts.append(np.asarray(geometry(p), float))
    out["geometry"] = {"shape_inputs": list(shape_inputs), "vertices": np.round(np.stack(verts), 5).tolist(),
                       "faces": np.asarray(faces, int).tolist()}
    return out


def write_twin(payload: Mapping[str, Any], out_dir: str, api: str = "") -> str:
    """``out_dir/index.html`` (the template with ``api`` filled in) + ``twin.json`` + vendored three.js."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "twin.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    html = open(TEMPLATE).read().replace("__API__", api)
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)
    vend = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pinneapple_tools",
                        "visualization", "studio", "web", "vendor")
    if os.path.isdir(vend) and not os.path.isdir(os.path.join(out_dir, "vendor")):
        shutil.copytree(vend, os.path.join(out_dir, "vendor"))
    return os.path.join(out_dir, "index.html")
