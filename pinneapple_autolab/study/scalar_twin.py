"""A twin page for one quantity of any model (a surrogate or the model itself): sliders over its inputs, the value at
the slider point, its uncertainty band when there is one, and a mark where the point leaves the training set.

    out = surrogate_twin("pinneapple_physics.closed_form.ideal_reactors:pfr_conversion_ktau",
                         [{"name": "k", "unit": "1/s", "low": 0.0, "high": 0.1},
                          {"name": "tau", "unit": "s", "low": 0.0, "high": 200.0}],
                         {"name": "X", "unit": "-"}, "twin_out", samples_used=[{"k": 0.02, "tau": 120.0}, ...])
    out["page"], out["manifest"]

Nothing on the page is computed in the browser: the model is evaluated here on a grid of every input (each
variable's ``values``, or ``n`` evenly spaced values from ``low`` to ``high``) and the page shows the table entry of
the grid node the sliders select. The sliders move by grid index, so they cannot leave ``[low, high]``. A node outside
the convex hull of ``samples_used`` (the points the surrogate was trained on) is marked "extrapolation". The band is
``none`` (labelled "no uncertainty estimate"), ``surrogate_std`` (the model returns ``{quantity}_std`` next to the
value: band = +- that std) or ``holdout_error`` (the RMS error the caller measured on held-out data: band = +- that).
The manifest records the variables and ranges, the model id, the SHA-256 of the training set, the band method and the
library commit; it changes when a training point changes and only then (for the same model and grid).
"""
from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
import os
import subprocess
from typing import Any, Literal, TypedDict

import numpy as np

from .verify import resolve_model

__all__ = ["surrogate_twin", "training_set_hash", "in_hull", "TwinOutput", "X_OUTPUTS"]

BANDS = ("none", "surrogate_std", "holdout_error")


# what each tool returns, with units (read by the Studio catalog)
X_OUTPUTS = {
    "surrogate_twin": [
        {"name": "page", "type": "str", "unit": "", "description": "path of the self-contained HTML page"},
        {"name": "manifest", "type": "dict", "unit": "",
         "description": "variables, ranges, model id, training-set SHA-256, band method, library commit"}],
}


class TwinOutput(TypedDict):
    page: str
    manifest: dict[str, Any]


def training_set_hash(samples: list[dict[str, float]] | None, names: list[str]) -> str:
    """SHA-256 of the training points (the variables' columns, in order; float repr, so any change shows)."""
    rows = [[repr(float(s[n])) for n in names] for s in (samples or [])]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def in_hull(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Whether each point lies in the convex hull of the samples (closed; 1-D: the interval; fewer samples than
    needed for a hull: only points equal to a sample)."""
    P, S = np.atleast_2d(points).astype(float), np.atleast_2d(samples).astype(float)
    if S.size == 0:
        return np.zeros(len(P), bool)
    d = P.shape[1]
    span = np.ptp(S, axis=0)
    scale = np.where(span > 0, span, 1.0)
    tol = 1e-9
    if d == 1:
        return (P[:, 0] >= S[:, 0].min() - tol * scale[0]) & (P[:, 0] <= S[:, 0].max() + tol * scale[0])
    if len(S) <= d or np.linalg.matrix_rank(S - S.mean(0)) < d:
        return np.array([np.any(np.all(np.abs(S - p) <= tol * scale, axis=1)) for p in P])
    from scipy.spatial import Delaunay
    return Delaunay(S / scale).find_simplex(P / scale, tol=tol) >= 0


def _commit() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:  # noqa: BLE001 - no git: fall back to the version
        pass
    try:
        import pinneapple
        return f"pinneapple {getattr(pinneapple, '__version__', '')}"
    except Exception:  # noqa: BLE001
        return ""


def _value(result: dict[str, Any], key: str) -> float:
    v = result.get(key)
    return float(v) if v is not None else float("nan")


def surrogate_twin(model: str, variables: list[dict[str, Any]], quantity: dict[str, str], out_dir: str,
                   band: Literal["none", "surrogate_std", "holdout_error"] = "none",
                   samples_used: list[dict[str, float]] | None = None, holdout_error: float | None = None,
                   n: int = 11) -> TwinOutput:
    """Write ``out_dir/index.html`` (self-contained, works offline) and ``out_dir/manifest.json``.

    Parameters
    ----------
    model : str
        executable tool id of the model (``"package.module:function"``); it takes the variables as keyword
        arguments and returns a number or a dict holding ``quantity["name"]`` (and ``{name}_std`` for that band)
    variables : list[dict]
        ``[{"name", "unit", "low", "high", "values"?}]``: each input, its unit, its range and optionally its grid
    quantity : dict[str, str]
        ``{"name", "unit"}`` of the output shown
    out_dir : str
        folder written
    band : str
        none, surrogate_std or holdout_error
    samples_used : list[dict[str, float]]
        the training points (for the extrapolation mark and the manifest's hash)
    holdout_error : float
        RMS error measured on held-out data, required with band holdout_error [same unit as the quantity]
    n : int
        grid values per variable without explicit ``values``

    Returns
    -------
    page : str
        path of the page
    manifest : dict
        variables, ranges, model id, training-set SHA-256, band method, library commit, grid size
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}")
    if band == "holdout_error" and (holdout_error is None or not math.isfinite(float(holdout_error))):
        raise ValueError("band 'holdout_error' needs the measured holdout_error")
    fn = resolve_model(model)
    names = [str(v["name"]) for v in variables]
    q = str(quantity["name"])
    axes = []
    for v in variables:
        lo, hi = float(v["low"]), float(v["high"])
        if not lo <= hi:
            raise ValueError(f"{v['name']}: low > high")
        vals = [float(x) for x in v.get("values") or np.linspace(lo, hi, n)]
        bad = [x for x in vals if not lo <= x <= hi]
        if bad:
            raise ValueError(f"{v['name']}: grid values {bad} outside [{lo:g}, {hi:g}]")
        axes.append(sorted(set(vals)))
    nodes = list(itertools.product(*axes))
    values, stds, errors = [], [], []
    for node in nodes:
        try:
            r = fn(**dict(zip(names, node, strict=True)))
            val = _value(r, q) if isinstance(r, dict) else float(r)
            sd = _value(r, f"{q}_std") if (band == "surrogate_std" and isinstance(r, dict)) else float("nan")
            errors.append("")
        except Exception as e:  # noqa: BLE001 - shown on the page at that node
            val, sd = float("nan"), float("nan")
            errors.append(f"{type(e).__name__}: {e}")
        values.append(val)
        stds.append(sd)
    if band == "none":
        half = [0.0] * len(nodes)
        band_label = "no uncertainty estimate"
    elif band == "surrogate_std":
        half = stds
        band_label = "+- the surrogate's predictive standard deviation"
    else:
        half = [float(holdout_error)] * len(nodes)
        band_label = f"+- the RMS error on held-out data ({float(holdout_error):.4g} {quantity.get('unit', '')})"
    S = np.array([[float(s[k]) for k in names] for s in (samples_used or [])], float).reshape(-1, len(names))
    inside = in_hull(np.array(nodes, float), S) if len(S) else np.zeros(len(nodes), bool)
    manifest = {"model": model, "quantity": {"name": q, "unit": quantity.get("unit", "")},
                "variables": [{"name": v["name"], "unit": v.get("unit", ""), "low": float(v["low"]),
                               "high": float(v["high"]), "grid": a} for v, a in zip(variables, axes, strict=True)],
                "band": band, "band_label": band_label, "n_training_points": int(len(S)),
                "training_set_sha256": training_set_hash(samples_used, names),
                "training_hull": "given" if len(S) else "no training set given: every node is marked extrapolation",
                "n_nodes": len(nodes), "n_failed_nodes": int(sum(1 for e in errors if e)),
                "library_commit": _commit()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    data = {"manifest": manifest, "shape": [len(a) for a in axes],
            "value": [None if not math.isfinite(v) else v for v in values],
            "half": [None if not math.isfinite(h) else h for h in half],
            "inside": [bool(x) for x in inside], "error": errors}
    page = os.path.join(out_dir, "index.html")
    with open(page, "w") as f:
        f.write(_PAGE.replace("__TITLE__", html.escape(f"{q} of {model}"))
                .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")))
    return {"page": page, "manifest": manifest}


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--warn:#b45309;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--ink:#f0efec;--muted:#c3c2b7;--line:#383835;--warn:#f5a524}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px}
.row{margin:14px 0}.row label{display:flex;justify-content:space-between;font-size:14px}
input[type=range]{width:100%}.value{font-size:32px;font-variant-numeric:tabular-nums;margin:12px 0 0}
.band{color:var(--muted);font-size:13px}#flag{margin-top:10px;padding:8px 10px;border:1px solid var(--line);border-radius:8px}
#flag.extra{border-color:var(--warn);color:var(--warn)}code{font-size:12px}
</style></head><body>
<h1 id="title"></h1><div class="sub" id="sub"></div>
<div id="sliders"></div>
<div class="value" id="value"></div><div class="band" id="band"></div><div id="flag"></div>
<p class="sub" id="foot"></p>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent), M = D.manifest;
const t = (id, s) => { document.getElementById(id).textContent = s; };
t("title", M.quantity.name + (M.quantity.unit ? " [" + M.quantity.unit + "]" : ""));
t("sub", "model " + M.model + " | band: " + M.band_label + " | training set " + M.n_training_points + " points");
t("foot", "Every value is the model evaluated at the grid node the sliders select (no interpolation). Library " +
  M.library_commit + ", training set SHA-256 " + M.training_set_sha256.slice(0, 16) + "...");
const idx = M.variables.map(() => 0), box = document.getElementById("sliders");
M.variables.forEach((v, j) => {
  const row = document.createElement("div"); row.className = "row";
  const lab = document.createElement("label"), name = document.createElement("span"), val = document.createElement("b");
  name.textContent = v.name + (v.unit ? " [" + v.unit + "]" : ""); lab.append(name, val);
  const s = document.createElement("input"); s.type = "range"; s.min = 0; s.max = v.grid.length - 1; s.step = 1;
  s.value = 0; s.setAttribute("aria-label", v.name); s.dataset.index = j;
  s.addEventListener("input", () => { idx[j] = Math.min(Math.max(parseInt(s.value, 10) || 0, 0), v.grid.length - 1); show(); });
  row.append(lab, s); box.append(row); v._val = val;
});
function node() { let k = 0; for (let j = 0; j < idx.length; j++) k = k * D.shape[j] + idx[j]; return k; }
function show() {
  M.variables.forEach((v, j) => { v._val.textContent = String(v.grid[idx[j]]); });
  const k = node(), val = D.value[k], h = D.half[k];
  t("value", val === null ? "no value (" + (D.error[k] || "model failed") + ")" : val.toPrecision(6));
  t("band", M.band === "none" ? "no uncertainty estimate" : (h === null ? "band: not available here" : "+- " + h.toPrecision(3) + " (" + M.band_label + ")"));
  const f = document.getElementById("flag");
  f.className = D.inside[k] ? "" : "extra";
  f.textContent = D.inside[k] ? "inside the training set's hull" : "extrapolation: outside the training set's hull";
  document.body.dataset.node = k;
}
show();
</script></body></html>
"""
