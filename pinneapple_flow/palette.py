"""The node palette: sections along an engineering workflow, human titles, one-line summaries, search ranking.

Flow Studio shows the palette this module describes; any other front end can use it the same way::

    from pinneapple_flow import palette
    palette.sections()                       # [{"id": "problem", "title": "Problem & geometry", ...}, ...] in order
    palette.entry(REGISTRY.get("train.pinn"))  # section, title, summary, tags, package, curated, install hint
    palette.search("pinn")                   # ranked hits: curated nodes above generated ``auto.*`` library calls

Every curated node type is placed in a section here (``SECTION_OF``); a node type registered by another package
can say where it goes with ``tags=["section:<id>"]`` on its spec, or falls back on its category. Generated
library calls (``auto.*``) always go to the last section, "Library calls (advanced)", grouped by package.
"""
from __future__ import annotations

import re
from typing import Any

from .registry import REGISTRY, NodeSpec

#: (id, title, glyph, description) in workflow order; the last one holds the generated library calls
SECTIONS: list[tuple[str, str, str, str]] = [
    ("problem", "Problem & geometry", "◇", "domains, meshes, scenarios and problem identification"),
    ("equations", "Equations", "∂", "governing equations and conditions as text"),
    ("solvers", "Classical solvers", "▦", "finite differences, FEM, FV, LBM, external codes"),
    ("physics_ai", "Physics AI", "✦", "PINNs, neural operators, graph networks, transfer and meta-learning"),
    ("surrogates", "Surrogates & ROMs", "≈", "data-driven surrogates, reduced-order models, hyperparameter search"),
    ("data", "Data", "▤", "files, sensors, parametric sweeps, designs of experiments"),
    ("inverse", "Inverse & calibration", "⇄", "identify parameters from data"),
    ("validation", "Validation & UQ", "✓", "checks against references, convergence, uncertainty, Lab seal"),
    ("optimization", "Optimization & control", "⌖", "design optimization and controllers"),
    ("twin", "Digital twin", "◎", "streams, filters, twin runtimes, forecasting"),
    ("viz", "Visualization", "◐", "plots, 3-D views, renders, dashboards"),
    ("export", "Export & report", "⇩", "reports and geometry export"),
    ("library", "Library calls (advanced)", "ƒ", "functions of the library's public API, generated from signatures"),
]
SECTION_IDS = [s[0] for s in SECTIONS]

#: curated node type -> section
SECTION_OF: dict[str, str] = {
    "geometry.domain": "problem", "geometry.mesh": "problem", "problem.generic": "problem",
    "vehicle.spec": "problem", "terramechanics.soil": "problem", "scenario.site": "problem",
    "equations.system": "equations",
    "solver.classical": "solvers", "solver.openfoam": "solvers", "solver.calculix": "solvers", "solver.fmu": "solvers",
    "solver.newton": "solvers", "lib.solve.preset": "solvers", "lib.solver.beam": "solvers",
    "lib.solver.lbm": "solvers", "lib.solver.meshless": "solvers", "lib.solver.terramechanics": "solvers",
    "lib.solver.xtfc": "solvers", "lib.tools.availability": "solvers", "flight.takeoff": "solvers",
    "vehicle.simulate": "solvers", "terramechanics.wheel_model": "solvers",
    "train.pinn": "physics_ai", "train.finetune": "physics_ai", "transfer.pinn": "physics_ai",
    "meta.learn": "physics_ai", "gnn.meshgraphnet": "physics_ai", "surrogate.train": "physics_ai",
    "lib.model.architecture": "physics_ai", "terramechanics.zoo_surrogate": "physics_ai",
    "surrogate.study": "surrogates", "surrogate.hpo": "surrogates", "rom.fit": "surrogates",
    "lib.rom.compare": "surrogates", "lib.rom.parametric_pod": "surrogates",
    "data.ingest": "data", "data.observe": "data", "data.parametric": "data", "data.grf_burgers": "data",
    "lib.doe.design": "data",
    "inverse.discover": "inverse", "inverse.calibrate": "inverse",
    "validate.compare": "validation", "validate.lab": "validation", "verify.convergence": "validation",
    "benchmark.arena": "validation", "lib.compare.preset": "validation", "uq.propagate": "validation",
    "uq.sensitivity": "validation", "lib.uq.morris": "validation", "lib.uq.sobol": "validation",
    "optimize.nsga2": "optimization", "control.compare": "optimization", "lib.design.intent": "optimization",
    "data.stream": "twin", "twin.connect": "twin", "twin.auto": "twin", "assimilate.filter": "twin",
    "forecast.lab": "twin", "forecast.enso": "twin", "lib.ts.predictability": "twin",
    "viz.plot": "viz", "viz.field3d": "viz", "viz.render": "viz", "vehicle.dashboard": "viz",
    "export.report": "export", "lib.export.geometry": "export",
}

#: the node an engineer means by a word (ranked first when the word is typed)
KEYWORDS: dict[str, list[str]] = {
    "train.pinn": ["pinn", "physics-informed"], "solver.classical": ["fd", "fem", "solver", "finite", "heat", "poisson", "wave", "burgers", "diffusion"],
    "surrogate.train": ["deeponet", "fno", "operator", "surrogate"], "rom.fit": ["rom", "pod", "dmd"],
    "geometry.domain": ["domain", "geometry"], "equations.system": ["equation", "equations", "pde"],
    "validate.compare": ["validate", "validation", "error"], "uq.propagate": ["uq", "uncertainty"],
    "twin.auto": ["twin"], "assimilate.filter": ["kalman", "enkf", "ekf"], "viz.plot": ["plot"],
    "export.report": ["report"], "inverse.discover": ["inverse"], "optimize.nsga2": ["optimize", "pareto"],
    "data.ingest": ["csv", "data"], "solver.openfoam": ["openfoam", "cfd"], "gnn.meshgraphnet": ["gnn", "graph"],
    "vehicle.simulate": ["rover", "vehicle"], "vehicle.dashboard": ["dashboard"], "terramechanics.soil": ["soil"],
}

_ORDER = {t: i for i, t in enumerate(SECTION_OF)}

#: fallback for node types registered elsewhere: category (lower case) -> section
_CATEGORY = {
    "problem": "problem", "geometry": "problem", "equations": "equations", "solvers": "solvers",
    "learning": "physics_ai", "physics ai": "physics_ai", "surrogates": "surrogates", "data": "data",
    "inverse": "inverse", "validation": "validation", "analysis": "validation", "design": "optimization",
    "systems": "solvers", "control": "optimization", "digital twin": "twin", "time series": "twin",
    "visualization": "viz", "export": "export", "custom": "library",
}

#: extra search words per section (what an engineer may type)
_SECTION_TAGS = {
    "problem": ["geometry", "domain", "mesh", "grid", "scenario"],
    "equations": ["pde", "ode", "boundary", "initial condition"],
    "solvers": ["simulation", "numerical", "fd", "fem", "cfd"],
    "physics_ai": ["neural network", "pinn", "operator", "deep learning"],
    "surrogates": ["reduced order", "rom", "regression", "machine learning"],
    "data": ["dataset", "csv", "sensors", "measurements"],
    "inverse": ["parameter estimation", "identification", "fit"],
    "validation": ["verification", "error", "uncertainty", "uq", "sensitivity"],
    "optimization": ["design", "pareto", "controller"],
    "twin": ["monitoring", "filter", "forecast", "anomaly"],
    "viz": ["plot", "figure", "render", "dashboard", "3d"],
    "export": ["report", "html", "geometry", "file"],
    "library": ["api", "function"],
}

_ACRONYMS = {"lr": "LR", "pde": "PDE", "ode": "ODE", "fd": "FD", "fem": "FEM", "fno": "FNO", "pod": "POD",
             "dmd": "DMD", "rom": "ROM", "pinn": "PINN", "gp": "GP", "mpc": "MPC", "pid": "PID", "uq": "UQ",
             "cfd": "CFD", "lbm": "LBM", "emat": "EMAT", "mfl": "MFL", "csv": "CSV", "html": "HTML", "llm": "LLM",
             "enso": "ENSO", "sst": "SST", "fmu": "FMU", "gnn": "GNN", "mlp": "MLP", "nn": "NN", "3d": "3-D",
             "2d": "2-D", "1d": "1-D", "id": "ID", "api": "API", "url": "URL", "doi": "DOI", "svd": "SVD"}


def sections() -> list[dict[str, Any]]:
    return [{"id": i, "title": t, "glyph": g, "description": d, "collapsed": i == "library"} for i, t, g, d in SECTIONS]


def is_auto(spec: NodeSpec | dict) -> bool:
    t = spec["type"] if isinstance(spec, dict) else spec.type
    return t.startswith("auto.")


def section_of(spec: NodeSpec) -> str:
    if is_auto(spec):
        return "library"
    for tag in getattr(spec, "tags", None) or []:
        if tag.startswith("section:") and tag[8:] in SECTION_IDS:
            return tag[8:]
    if spec.type in SECTION_OF:
        return SECTION_OF[spec.type]
    cat = spec.category.lower().removeprefix("library:").strip()
    return _CATEGORY.get(spec.category.lower(), _CATEGORY.get(cat, "library"))


def package_of(spec: NodeSpec) -> str:
    """The library package a node wraps (``pinneapple_analysis``), from its first declared call."""
    first = re.split(r"[,;\s]+", (spec.calls or "").strip())[0] if spec.calls else ""
    mod = first.split(":")[0].split("(")[0]
    return mod.split(".")[0] if mod.startswith("pinneapple") else ""


def humanize(name: str) -> str:
    """``generate_eddy_current_synthetic`` -> ``Generate eddy current synthetic``; acronyms kept upper case."""
    words = [w for w in re.split(r"[_\s]+", name) if w]
    out = [_ACRONYMS.get(w.lower(), w) for w in words]
    if out and out[0] == words[0]:
        out[0] = out[0][:1].upper() + out[0][1:]
    return " ".join(out)


def summary(text: str, limit: int = 110) -> str:
    """The first sentence of a docstring, without reST markup and the generated ``[checked: ...]`` note, cut to
    ``limit`` characters."""
    t = re.sub(r"\s*\[checked:[^\]]*\]", "", text or "")
    t = re.sub(r"``([^`]*)``", r"\1", t)
    t = re.sub(r":[a-z]+:`~?([^`]*)`", r"\1", t)
    t = " ".join(t.split())
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    s = m.group(1) if m else t
    if re.match(r"[a-z]+\b(?![._(])", s):          # a sentence, not a code name like pp.solve or pinneapple_x
        s = s[:1].upper() + s[1:]
    return s if len(s) <= limit else s[:limit - 1].rstrip(" ,;:-") + "…"


def install_hint(reason: str) -> str:
    """What to do about an unavailable node, from its reason."""
    if not reason:
        return ""
    m = re.search(r"needs ([A-Za-z0-9_.\-]+)", reason) or re.search(r"No module named '([A-Za-z0-9_\-]+)", reason)
    if m:
        mod = m.group(1).split(".")[0]
        return f"pip install {_PIP.get(mod, mod)}"
    if "OpenFOAM" in reason:
        return "needs OpenFOAM (blockMesh, icoFoam on PATH)"
    if "ccx" in reason or "CalculiX" in reason:
        return "needs CalculiX (ccx on PATH)"
    return reason.split("(")[0].strip()


_PIP = {"bpy": "bpy", "fmpy": "fmpy", "newton": "newton-physics", "sklearn": "scikit-learn", "cv2": "opencv-python",
        "yaml": "pyyaml", "torch_geometric": "torch-geometric"}


def entry(spec: NodeSpec, avail: tuple[bool, str] | None = None) -> dict[str, Any]:
    """The palette fields of one node type."""
    ok, why = avail if avail is not None else spec.available()
    auto = is_auto(spec)
    sec = section_of(spec)
    pkg = package_of(spec)
    fn = spec.type.rsplit(".", 1)[-1]
    title = humanize(fn) if auto else spec.title
    tags = sorted({*(t for t in (getattr(spec, "tags", None) or []) if not t.startswith("section:")),
                   *_SECTION_TAGS.get(sec, [])})
    return {"section": sec, "title": title, "summary": summary(spec.description), "tags": tags,
            "package": pkg.removeprefix("pinneapple_") if pkg else "", "curated": not auto,
            "group": (pkg.removeprefix("pinneapple_") or "other") if auto else "",
            "install": "" if ok else install_hint(why)}


def _words(s: str) -> list[str]:
    # the library's own name (pinneapple_*, pp.) says nothing about a node; leave it out of matching
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and not w.startswith("pinneapple")]


def score(e: dict[str, Any], query: str) -> float:
    """Relevance of a catalog entry (``REGISTRY.catalog()`` row with its ``palette`` fields) to a query; 0 = no
    match. Every query word must match somewhere; title matches weigh most, then tags, type, description and
    package. Curated nodes rank above generated library calls with the same match."""
    q = _words(query)
    if not q:
        return 0.0
    p = e.get("palette", {})
    title, tags = p.get("title", e["title"]).lower(), " ".join(p.get("tags", [])).lower()
    fields = [(title, 8.0), (e["title"].lower(), 6.0), (tags, 4.0), (e["type"].lower(), 3.0),
              (e.get("description", "").lower(), 1.5), (p.get("package", "").lower(), 1.0),
              (e.get("category", "").lower(), 1.0)]
    total = 0.0
    for w in q:
        best = 0.0
        for text, weight in fields:
            if not text:
                continue
            words = _words(text)
            if w in words:
                best = max(best, weight * (1.5 if text.startswith(w) else 1.0))
            elif any(x.startswith(w) for x in words):
                best = max(best, weight * 0.8)
            elif any(w in x for x in words):
                best = max(best, weight * 0.4)
        if w in KEYWORDS.get(e["type"], ()):
            best += 10.0
        if best == 0.0:
            return 0.0
        total += best
    if p.get("curated", True):
        total += 100.0
    if not e.get("available", True):
        total -= 0.5
    return total


def search(query: str, catalog: list[dict[str, Any]] | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Catalog entries matching ``query``, best first (curated before ``auto.*``)."""
    rows = catalog if catalog is not None else REGISTRY.catalog()
    hits = [(score(e, query), i, e) for i, e in enumerate(rows)]
    hits = [h for h in hits if h[0] > 0]
    hits.sort(key=lambda h: (-h[0], h[1]))
    return [e for _, _, e in hits[:limit]]


def layout(catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The palette as sections in workflow order with their node types (curated sections by title order of the
    curation, the advanced section grouped by package)."""
    rows = catalog if catalog is not None else REGISTRY.catalog()
    out = []
    for s in sections():
        items = [e for e in rows if e.get("palette", {}).get("section") == s["id"]]
        if s["id"] == "library":
            items.sort(key=lambda e: (e["palette"]["group"], e["palette"]["title"].lower()))
        else:
            items.sort(key=lambda e: (_ORDER.get(e["type"], len(_ORDER)), e["title"].lower()))
        out.append({**s, "count": len(items), "types": [e["type"] for e in items]})
    return out
