"""The run, in a browser: its gates, its figures, its surface — and the model you can move inside its domain.

``pp.SurrogatePipeline.view()`` writes one page next to the run and opens it. Everything the run decided is on it
(the stages, the ten gates with their values and thresholds, the figures it wrote), and so is the model itself: a map
of one output over two of the inputs, and under it the output's own curve at the point you pick. No server and no
internet — the page is ``file://`` with the data inlined, so it opens on a laptop that has nothing installed.

What the explorer is, exactly: the model evaluated on a grid of the space it was trained on (``axes``: the first two
inputs of the dataset, or the coordinates of a physics-only model), plus a line through that grid. It never claims
to extrapolate: the grid is the declared parameter space, and a point outside it is not offered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build_viewer", "payload", "FIGURE_CAPTIONS"]

#: What each figure of a run shows, for the page's captions.
FIGURE_CAPTIONS = {
    "training_loss": "the loss the trainer optimised",
    "parity": "held out: what the model says against what the reference says",
    "error_map": "where the error sits, on cases the model never saw",
    "geometry": "the surface the geometry gate measured",
}

_GRID = 25                    # samples per axis when the output is a handful of numbers
_GRID_WIDE = 18               # ... and when every cell carries a whole field


# -------------------------------------------------------------------------------------------------- the payload
def _run_dir(pipeline) -> Path:
    state = getattr(pipeline, "state", None)
    out = getattr(state, "out_dir", None)
    if out:
        return Path(out)
    cfg = pipeline.config
    return Path(cfg.execution.out_dir) / cfg.name


def _report(run_dir: Path, pipeline) -> dict[str, Any]:
    path = run_dir / "report.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = getattr(pipeline, "result", None)
    return result.to_dict() if result is not None and hasattr(result, "to_dict") else {}


def _gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for gate in report.get("verification") or []:
        metrics, thresholds = gate.get("metrics") or {}, gate.get("thresholds") or {}
        rows.append({
            "gate": gate.get("gate", ""),
            "status": gate.get("status", ""),
            "value": _first_number(metrics),
            "metrics": {k: _round(v) for k, v in metrics.items() if _round(v) is not None},
            "threshold": next(iter(thresholds.values()), "") if thresholds else "",
            "evidence": list(gate.get("evidence") or [])[:3],
            "limitations": list(gate.get("limitations") or [])[:3],
            "recommendations": list(gate.get("recommendations") or [])[:2],
        })
    return rows


def _round(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value if isinstance(value, bool) else None
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return None


def _first_number(metrics: dict[str, Any]) -> str:
    """The one number a gate would put on a dashboard, in the order that matters for reading it."""
    for key in ("relative_l2", "nrmse", "error", "faces", "refinement_change", "misfit"):
        if key in metrics and _round(metrics[key]) is not None:
            return f"{key} {float(metrics[key]):.4g}"
    return ""


def _stages(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage in report.get("stages") or []:
        rows.append({"stage": stage.get("stage", ""), "status": stage.get("status", ""),
                     "seconds": stage.get("seconds"), "notes": list(stage.get("notes") or [])[:2]})
    return rows


def _figures(run_dir: Path) -> list[dict[str, str]]:
    folder = run_dir / "figures"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.png")):
        out.append({"name": path.stem, "src": f"figures/{path.name}",
                    "caption": FIGURE_CAPTIONS.get(path.stem, path.stem)})
    return out


def _inputs(pipeline) -> tuple[list[dict[str, Any]], list[str], str]:
    """The columns the model takes, their ranges, and where the ranges came from."""
    import numpy as np

    cfg, state = pipeline.config, pipeline.state
    names: list[str] = []
    if cfg.data.is_physics_only:
        names = list(state.bundle.coords)
        source = "the problem's domain"
    else:
        dataset = getattr(state, "dataset", None)
        names = list(getattr(dataset, "input_names", []) or [])
        source = "the design's parameter space"
    space = cfg.parameters.parameters or {}
    by_name = {}
    for name in names:
        if name in space:
            lo, hi = space[name]
            by_name[name] = [float(lo), float(hi)]
    if not by_name and cfg.data.is_physics_only:
        for name in names:
            lo, hi = state.bundle.domains.get(name, (None, None))
            if lo is not None:
                by_name[name] = [float(lo), float(hi)]
    if not by_name and getattr(state, "cases", None) is not None:
        for name in names:
            if name in state.cases.columns:
                values = np.asarray(state.cases[name], dtype=float)
                if np.isfinite(values).all():
                    by_name[name] = [float(values.min()), float(values.max())]
    rows = [{"name": name, "low": by_name[name][0], "high": by_name[name][1]}
            for name in names if name in by_name]
    return rows, names, source


def _output_names(pipeline, width: int) -> list[str]:
    state = pipeline.state
    dataset = getattr(state, "dataset", None)
    names = list(getattr(dataset, "output_names", []) or [])
    if not names and pipeline.config.data.is_physics_only:
        names = list(state.bundle.fields)
    if len(names) == width:
        return names
    if len(names) == 1:
        return [f"{names[0]}[{i}]" for i in range(width)]
    return [f"y{i}" for i in range(width)]


def _explore(pipeline, inputs: list[dict[str, Any]], names: list[str], source: str) -> dict[str, Any] | None:
    """The model on a grid of its own space — the part of the page that moves when you drag.

    Two axes at a time (the first two inputs, or the coordinates of a physics-only model); every other input stays at
    the middle of its range. The grid is the declared space, so nothing here extrapolates.
    """
    import numpy as np

    if not inputs:
        return None
    anchor = [0.5 * (item["low"] + item["high"]) for item in inputs]
    try:
        width = int(np.asarray(pipeline.predict_array(np.asarray([anchor]))).reshape(1, -1).shape[1])
    except Exception as exc:                                       # a run with no model still shows its gates
        return {"error": f"{type(exc).__name__}: {exc}"}
    outputs = _output_names(pipeline, width)
    axes = [item["name"] for item in inputs[:2]]
    first, second = inputs[0], inputs[1] if len(inputs) > 1 else None
    xs = np.linspace(first["low"], first["high"], _GRID_WIDE if width > 8 else _GRID)
    ys = np.linspace(second["low"], second["high"], _GRID_WIDE if width > 8 else _GRID) if second else [0.0]
    points = []
    for y in ys:
        for x in xs:
            row = list(anchor)
            row[names.index(first["name"])] = float(x)
            if second:
                row[names.index(second["name"])] = float(y)
            points.append(row)
    predictions = np.asarray(pipeline.predict_array(np.asarray(points)), dtype=float)
    predictions = predictions.reshape(len(ys), len(xs), width)
    magnitude = np.linalg.norm(predictions, axis=2)
    return {
        "axes": axes,
        "inputs": inputs,
        "outputs": outputs,
        "shape": [len(xs), len(ys)],
        "x": [round(float(v), 9) for v in xs],
        "y": [round(float(v), 9) for v in ys],
        "anchor": [round(float(v), 9) for v in anchor],
        "magnitude": [[round(float(v), 6) for v in row] for row in magnitude],
        "values": [[[round(float(v), 6) for v in cell] for cell in row] for row in predictions],
        "range": source,
        "note": ("the model evaluated on the space it was trained on; every point on the map is a query it answers"),
    }


def _scene(run_dir: Path, state) -> list[dict[str, str]]:
    """The surface of the run, as the library's own three.js page (vendored, no internet)."""
    geometry = getattr(getattr(state, "cfg", None), "geometry", None)
    path = str(getattr(state, "geometry_path", "") or "")
    if not path and geometry is not None:
        path = str(getattr(geometry, "path", "") or "")
    if not path and getattr(geometry, "source", "") == "builder":   # what the gate wrote into the run folder
        candidate = run_dir / "geometry" / "surface.stl"
        path = str(candidate) if candidate.is_file() else ""
    if not path or not Path(path).is_file():
        return []
    try:
        import pinneapple as pp

        scene = pp.viz.Scene.from_file(str(path))
        target = run_dir / "viewer" / "scene.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        pp.viz.web_viewer(scene, str(target.parent), single_file=True, filename="scene.html")
    except Exception:                                             # no viewer is not a failure of the run
        return []
    return [{"title": f"{Path(path).name}: the surface under test", "src": "scene.html"}]


def payload(source, *, scene: bool = True) -> dict[str, Any]:
    """Everything the page shows, as plain data (also written as ``viewer.json`` next to the page)."""
    from .pipeline import SurrogatePipeline

    pipeline = source if isinstance(source, SurrogatePipeline) else SurrogatePipeline.from_run(source)
    run_dir = _run_dir(pipeline)
    report = _report(run_dir, pipeline)
    result = getattr(pipeline, "result", None)
    inputs, names, where = _inputs(pipeline)
    data = {
        "run": {
            "name": report.get("name") or pipeline.config.name,
            "status": report.get("status", ""),
            "summary": result.summary() if result is not None and hasattr(result, "summary") else "",
            "seconds": report.get("execution_time"),
            "problem": report.get("problem", ""),
            "backend": getattr(pipeline.state, "bundle", None).backend if getattr(pipeline.state, "bundle", None)
            else "",
            "domain_where": where,
        },
        "stages": _stages(report),
        "gates": _gates(report),
        "figures": _figures(run_dir),
        "explore": _explore(pipeline, inputs, names, where),
        "scenes": _scene(run_dir, pipeline.state) if scene else [],
        "notes": list(report.get("recommendations") or [])[:3],
    }
    return data


# ---------------------------------------------------------------------------------------------------- the page
def build_viewer(source, out_dir: str | Path | None = None, *, open_browser: bool = True,
                 scene: bool = True) -> str:
    """Write the page for ``source`` (a pipeline or a finished run folder) and open it; returns its path."""
    import webbrowser

    from .pipeline import SurrogatePipeline

    pipeline = source if isinstance(source, SurrogatePipeline) else SurrogatePipeline.from_run(source)
    run_dir = _run_dir(pipeline)
    target = Path(out_dir) if out_dir is not None else run_dir / "viewer"
    target.mkdir(parents=True, exist_ok=True)
    data = payload(pipeline, scene=scene)
    figures = Path(run_dir) / "figures"                              # the page carries the figures it shows
    if figures.is_dir():
        (target / "figures").mkdir(exist_ok=True)
        for png in figures.glob("*.png"):
            (target / "figures" / png.name).write_bytes(png.read_bytes())
    (target / "viewer.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    here = Path(__file__).resolve().parent / "web"
    for name in ("viewer.css", "viewer.js"):
        (target / name).write_text((here / name).read_text(encoding="utf-8"), encoding="utf-8")
    if data.get("scenes"):
        # A file:// page cannot render a file:// iframe, so the scene page travels as a string and goes in as srcdoc.
        scene_page = (target / "scene.html").read_text(encoding="utf-8")
        (target / "scene.js").write_text("window.PP_SCENE = " + json.dumps(scene_page) + ";", encoding="utf-8")
        for row in data["scenes"]:
            row["src"] = "scene.js"
        (target / "viewer.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    page = (here / "viewer.html").read_text(encoding="utf-8")
    page = page.replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"), 1)
    index = target / "index.html"
    index.write_text(page, encoding="utf-8")
    if open_browser:
        webbrowser.open(index.as_uri())
    return str(index)
