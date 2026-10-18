"""Experiment manifests: one file per experiment that the site, the curation and the Flow Studio all read.

An experiment is described by ``experiment.json`` next to its scripts or results (format ``pinneapple.experiment/1``,
below). A Lab run folder (``run.json``) is read as a manifest too, without a file, so every Lab run is an experiment
with no extra step. New experiments get their manifest written for them:

* scripts and examples: ``with pinneapple_lab.experiment("name", title=..., domain=...) as ex:`` (``capture.py``)
  records parameters, the git commit, environment and timing, ``ex.log_metric``, ``ex.check``, and every figure saved
  through ``Figure.savefig`` / ``sealed_savefig``, plus movies and viewers added with ``ex.add_file``; on exit it writes
  the manifest, registers it in the Lab and re-curates (A-F and next steps);
* Lab runs (``pinneapple_lab.run`` / ``sweep``) write ``experiment.json`` into the run folder;
* ``pinneapple_core.workflows`` runs write one into the run folder;
* ``pp.solve(...).to_experiment(path)`` and ``pp.compare(...).to_experiment(path)``;
* Flow Studio runs: :func:`from_flow_run`.

    from pinneapple_lab import manifest
    m = manifest.load("examples/flight_dynamics")              # a folder, an experiment.json, or a Lab run folder
    errors = manifest.validate(m, "examples/flight_dynamics")  # [] when valid
    for path, m in manifest.scan():                            # examples/*, the Lab runs, results folders
        ...
    python -m pinneapple_lab register examples/my_study        # writes / imports a manifest, then curates
    python -m pinneapple_lab scaffold my_study                 # examples/my_study wired with experiment(...)
    python -m pinneapple_lab manifests --check                 # the CI gate (scripts/check_experiment_manifests.py)

Format (``schema`` "pinneapple.experiment/1"; paths relative to the manifest's folder)::

    {"schema": "pinneapple.experiment/1", "id": "examples/flight_dynamics", "title": "...", "description": "...",
     "domain": "flight dynamics", "methods": ["vortex lattice", "6-DOF"], "modules": ["pinneapple_simulation.flight"],
     "params": {...}, "scripts": ["01_verification.py"],
     "metrics": [{"name": "Cl", "value": 0.66, "unit": "", "how": "OpenFOAM simpleFoam, last 200 iterations"}],
     "checks": [{"name": "...", "passed": true, "kind": "reference", ...}],          # as ctx.check records them
     "files": {"figures": [...], "videos": [...], "viewers": [...], "data": [...]},
     "links": {"guide": "docs/guides/flight_dynamics.md", "script": "...", "pr": [844, 847]},
     "references": ["MIL-F-8785C"], "status": "curated" | "not curated",
     "lab": {"item": "...", "experiment": "...", "run_id": "..."},
     "provenance": {"source": "experiment()", "git": "...", "created": "...", "seconds": 1.2, "environment": {...}}}

A metric always has a measured value (never a placeholder) and says how it was measured; a missing file or a metric
without a value fails :func:`validate`, and :func:`check_repo` fails CI for an ``examples/*`` folder without a valid
manifest.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "pinneapple.experiment/1"
FILENAME = "experiment.json"
def _repo_root() -> str:
    """The checkout: ``$PINNEAPPLE_REPO``, else the folder above this package when it holds ``examples/`` (a source
    checkout or an editable install), else the working directory (the Lab image runs from /srv)."""
    env = os.environ.get("PINNEAPPLE_REPO")
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.abspath(env) if env else here if os.path.isdir(os.path.join(here, "examples")) else os.getcwd()


REPO = _repo_root()
IGNORED_EXAMPLE_DIRS = ("scratch", "__pycache__")          # plus every name starting with "_" or "."
FIGURE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".svg")
VIDEO_EXT = (".gif", ".mp4", ".webm", ".mov")
DATA_EXT = (".json", ".csv", ".npz", ".md", ".h5", ".parquet")
STATUS = ("curated", "not curated")
_DOMAINS = [   # pattern over the name, title and tags -> domain, first match wins (a manifest may always say otherwise)
    (r"flight", "flight dynamics"), (r"black_hole|\bbh_|accretion|bondi", "astrophysics"),
    (r"enso|weather|climate|geophys", "geophysics and climate"), (r"time_series|forecast", "time series"),
    (r"electro|motor|\bem_|em_pinn|maxwell|prius|pmsm|fdtd", "electromagnetics"),
    (r"heat|\bfin_|thermal|coldplate", "heat transfer"),
    (r"rover|terramech|soil|vehicle_bicycle|chassis", "vehicle dynamics"),
    (r"calculix|solid|\bfem\b|bricks|topology|geomech|inspection|wear|tribolog|beam|bracket|plate|buckling|crash",
     "structural mechanics"),
    (r"lbm|openfoam|free_surface|cfd|vehicle|handoff|ahmed|airfoil|airliner|concorde|pipe|wake|cylinder|flow|"
     r"particle|shock|channel", "fluid dynamics"),
    (r"kepler|lorenz|discovery|oscillator|pendulum|modelica|cosim|systems|rigid_body", "system dynamics"),
    (r"molecular", "molecular dynamics"),
    (r"pinn|trainer|architectures|latent|worldmodel|meshgraphnet|model_zoo|surrogate", "physics AI"),
    (r"geometr|design", "design and geometry"),
    (r"benchmark|arena|vs_physicsnemo|hpo|validation|physics_check|physics_audit|proof_graph", "validation and benchmarks"),
    (r"solver|numerical|pde|physics_|xtfc", "numerical methods"), (r"\bdata\b|_db\b|data_", "data"),
    (r"security", "security"),
]


@dataclass
class Manifest:
    id: str
    title: str
    description: str = ""
    domain: str = ""
    methods: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    scripts: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, list[str]] = field(default_factory=lambda: {"figures": [], "videos": [], "viewers": [], "data": []})
    links: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    status: str = "not curated"
    lab: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {"schema": d.pop("schema"), **d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        m = cls(**known)
        m.files = {k: list(m.files.get(k, [])) for k in ("figures", "videos", "viewers", "data")}
        return m

    def metric(self, name: str, value: Any, unit: str = "", how: str = "") -> None:
        self.metrics.append({"name": name, "value": value, "unit": unit, "how": how})


# ---------------------------------------------------------------------------------------------- reading and writing
def save(m: Manifest, path: str) -> str:
    """Write ``m`` as ``path`` (a folder gets ``experiment.json``)."""
    if os.path.isdir(path) or not path.endswith(".json"):
        path = os.path.join(path, FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(m.to_dict(), f, indent=1, default=_jsonable)
        f.write("\n")
    return path


def load(path: str) -> Manifest:
    """The manifest of ``path``: an ``experiment.json``, a folder holding one, or a Lab run folder (``run.json``)."""
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, FILENAME)):
            path = os.path.join(path, FILENAME)
        elif os.path.exists(os.path.join(path, "run.json")):
            return from_run_dir(path)
        else:
            raise FileNotFoundError(f"{path}: no {FILENAME} (python -m pinneapple_lab register {path})")
    with open(path) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA!r}")
    return Manifest.from_dict(d)


def validate(m: Manifest, base: str | None = None) -> list[str]:
    """What is wrong with ``m`` (empty when it is valid): the schema, an id and a title, every metric with a finite
    numeric value, a name and a ``how``, file lists of known kinds, and (with ``base``, its folder) every file there."""
    err = []
    if m.schema != SCHEMA:
        err.append(f"schema {m.schema!r} is not {SCHEMA!r}")
    if not str(m.id).strip():
        err.append("no id")
    if not str(m.title).strip():
        err.append("no title")
    if m.status not in STATUS:
        err.append(f"status {m.status!r} is not one of {STATUS}")
    for i, x in enumerate(m.metrics):
        name = x.get("name") if isinstance(x, dict) else None
        if not name:
            err.append(f"metric #{i + 1} has no name")
            continue
        v = x.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            err.append(f"metric {name!r} has no measured value ({v!r})")
        if not str(x.get("how", "")).strip():
            err.append(f"metric {name!r} does not say how it was measured")
        if "unit" not in x:
            err.append(f"metric {name!r} has no unit field (use \"\" for dimensionless)")
    for c in m.checks:
        if not isinstance(c, dict) or "name" not in c or "passed" not in c:
            err.append(f"check {c!r} needs a name and passed")
    unknown = set(m.files) - {"figures", "videos", "viewers", "data"}
    if unknown:
        err.append(f"unknown file kinds {sorted(unknown)}")
    if base is not None:
        for kind, paths in m.files.items():
            for p in paths:
                if not os.path.exists(os.path.join(base, p)):
                    err.append(f"{kind} file {p} does not exist")
        for p in m.scripts:
            if not os.path.exists(os.path.join(base, p)):
                err.append(f"script {p} does not exist")
    return err


def _jsonable(v):
    try:
        from pinneapple_core.jsonable import to_jsonable
        return to_jsonable(v)
    except Exception:  # noqa: BLE001
        return str(v)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


# ---------------------------------------------------------------------------------------------- sources
def from_run_dir(d: str) -> Manifest:
    """A Lab run folder as a manifest: its parameters, metrics (``ctx.metric``), checks, figures and provenance."""
    rec = _read(os.path.join(d, "run.json"), {})
    metrics = _read(os.path.join(d, "metrics.json"), {})
    checks = _read(os.path.join(d, "validation.json"), [])
    exp, rid = rec.get("experiment", os.path.basename(os.path.dirname(d))), rec.get("run_id", os.path.basename(d))
    files = {"figures": [], "videos": [], "viewers": [], "data": []}
    fig_dir = os.path.join(d, "figures")
    for f in sorted(os.listdir(fig_dir)) if os.path.isdir(fig_dir) else []:
        ext = os.path.splitext(f)[1].lower()
        kind = "videos" if ext in VIDEO_EXT else "figures" if ext in FIGURE_EXT else None
        if kind:
            files[kind].append(f"figures/{f}")
    if os.path.exists(os.path.join(d, "viewer", "index.html")):
        files["viewers"].append("viewer/index.html")
    m = Manifest(id=f"lab/{exp}/{rid}", title=str(rec.get("description") or exp).split(". ")[0][:120],
                 description=str(rec.get("description", "")), domain=domain_of(exp, rec.get("tags", [])),
                 methods=[str(rec["params"]["method"])] if isinstance(rec.get("params", {}).get("method"), str) else [],
                 params=rec.get("params", {}) or {}, checks=checks, files=files,
                 status="curated", lab={"experiment": exp, "run_id": rid, "status": rec.get("status")},
                 provenance={"source": "lab-run", "git": rec.get("git"), "seconds": rec.get("seconds"),
                             "created": _iso(rec.get("started")), "environment": rec.get("environment", {})})
    for k, v in metrics.items():
        if _number(v):
            m.metric(k, v, "", "recorded by the Lab run (ctx.metric)")
    return m


def write_for_run(d: str) -> str | None:
    """Write ``experiment.json`` into a finished Lab run folder (the runner calls it); never raises."""
    try:
        return save(from_run_dir(d), os.path.join(d, FILENAME))
    except Exception:  # noqa: BLE001 - a manifest must never fail the run it describes
        return None


def from_workflow_run(runner, run_id: str) -> Manifest:
    """A ``pinneapple_core.workflows`` run as a manifest: steps as methods, step outputs' numbers as metrics."""
    run = runner.get_run(run_id)
    m = Manifest(id=f"workflow/{run.get('name')}/{run_id}", title=str(run.get("name") or run_id),
                 description=str((run.get("metadata") or {}).get("description", "")),
                 methods=[s.get("executor", "") for s in run["workflow"].get("steps", [])],
                 params={"metadata": run.get("metadata", {})}, status="not curated",
                 provenance={"source": "workflow", "created": _iso(run.get("started")),
                             "seconds": (run["finished"] - run["started"]) if run.get("finished") and run.get("started")
                             else None, "git": _git()})
    for s in runner.steps(run_id):
        for k, v in (s.get("outputs") or {}).items() if isinstance(s.get("outputs"), dict) else []:
            if _number(v):
                m.metric(f"{s['step_id']}.{k}", v, "", f"output of workflow step {s['step_id']}")
    return m


def from_solution(sol, title: str | None = None) -> Manifest:
    """A ``pp.solve`` result as a manifest (``Solution.to_experiment`` calls it)."""
    md = sol.metadata()
    m = Manifest(id=f"solve/{md['problem']}/{md['method']}", title=title or f"{md['problem']} by {md['method']}",
                 domain="", methods=[md["method"]], modules=["pinneapple_physics.solving"],
                 params={"problem": md["problem"], "fingerprint": md["fingerprint"], "fields": md["fields"]},
                 provenance={"source": "pp.solve", "git": _git(), "created": _iso(time.time())})
    m.metric("wall_time", float(md["wall_time_s"]), "s", "pp.solve wall clock")
    return m


def from_comparison(cmp, title: str | None = None) -> Manifest:
    """A ``pp.compare`` table as a manifest: every metric of every method that ran, against the stated reference."""
    d = cmp.to_dict()
    m = Manifest(id=f"compare/{d['problem']}", title=title or f"{d['problem']}: methods compared",
                 methods=[r["method"] for r in d["rows"]], modules=["pinneapple_physics.solving"],
                 params={"problem": d["problem"], "reference": d["reference"], "n_points": d["n_points"]},
                 provenance={"source": "pp.compare", "git": _git(), "created": _iso(time.time())})
    how = f"pp.compare against {d['reference']} at {d['n_points']} points"
    for r in d["rows"]:
        if r.get("status") != "ok":
            continue
        for metric, per_field in (r.get("metrics") or {}).items():
            for fld, v in (per_field.items() if isinstance(per_field, dict) else [("", per_field)]):
                if _number(v):
                    m.metric(".".join(x for x in (r["method"], metric, fld) if x), float(v), "", how)
    return m


def from_flow_run(run: dict[str, Any], out_dir: str | None = None) -> Manifest:
    """The Flow Studio's hook: a finished flow run as a manifest, written to ``out_dir`` when given.

    ``run`` keys (all optional but ``id``/``title``): ``id``, ``title``, ``description``, ``domain``, ``nodes``
    (list of {"type": ...} -> methods), ``modules``, ``params``, ``metrics`` ([{"name", "value", "unit", "how"}],
    measured values only), ``checks``, ``files`` (as in the manifest, paths relative to ``out_dir``), ``links``."""
    m = Manifest(id=str(run.get("id") or run.get("title")), title=str(run.get("title") or run.get("id")),
                 description=str(run.get("description", "")), domain=str(run.get("domain", "")),
                 methods=[str(n.get("type", n)) if isinstance(n, dict) else str(n) for n in run.get("nodes", [])],
                 modules=list(run.get("modules", [])), params=dict(run.get("params", {})),
                 metrics=[x for x in run.get("metrics", []) if _number((x or {}).get("value"))],
                 checks=list(run.get("checks", [])), links=dict(run.get("links", {})),
                 provenance={"source": "flow", "git": _git(), "created": _iso(time.time())})
    for k, v in (run.get("files") or {}).items():
        m.files.setdefault(k, []).extend(v)
    if out_dir:
        save(m, out_dir)
    return m


# ---------------------------------------------------------------------------------------------- existing folders
def domain_of(name: str, tags=()) -> str:
    text = " ".join([name, *map(str, tags)]).lower()
    return next((d for pat, d in _DOMAINS if re.search(pat, text)), "general")


def _git_files(folder: str) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "--", folder], cwd=REPO, capture_output=True, text=True, timeout=30)
        files = [os.path.relpath(os.path.join(REPO, p), folder) for p in out.stdout.split()]
        if files:
            return files
    except Exception:  # noqa: BLE001
        pass
    return [os.path.relpath(os.path.join(r, f), folder) for r, _, fs in os.walk(folder) for f in fs]


def _title_of(folder: str, scripts: list[str]) -> tuple[str, str]:
    for readme in ("README.md", "readme.md"):
        p = os.path.join(folder, readme)
        if os.path.exists(p):
            lines = [ln.strip() for ln in open(p, encoding="utf-8", errors="replace")]
            head = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")), "")
            body = next((ln for ln in lines if ln and not ln.startswith(("#", "!", "[", "|", "```"))), "")
            if head:
                return head, body
    for s in scripts:
        try:
            src = open(os.path.join(folder, s), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        m = re.match(r'\s*(?:#[^\n]*\n\s*)*[rub]*("""|\'\'\')(.*?)\1', src, re.S)
        if m and m.group(2).strip():
            doc = " ".join(m.group(2).split())
            return doc.split(". ")[0][:120].rstrip(":"), doc[:400]
    return os.path.basename(folder).replace("_", " ").capitalize(), ""


def _modules(folder: str, scripts: list[str]) -> list[str]:
    mods = set()
    for s in scripts:
        try:
            src = open(os.path.join(folder, s), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        mods |= set(re.findall(r"^\s*(?:from|import)\s+(pinneapple[\w.]*)", src, re.M))
    return sorted(mods)


_PR_CACHE: dict[str, list[int]] = {}


def _prs(folder: str) -> list[int]:
    """The pull requests whose merge into the main line changed the folder, read from git history (oldest first);
    no number is typed by hand."""
    if not _PR_CACHE:
        try:
            out = subprocess.run(["git", "log", "--merges", "--diff-merges=first-parent", "--name-only",
                                  "--format=@@%s", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                 timeout=120).stdout
        except Exception:  # noqa: BLE001
            out = ""
        pr = None
        for line in out.splitlines():
            if line.startswith("@@"):
                m = re.search(r"Merge pull request #(\d+)", line)
                pr = int(m.group(1)) if m else None
            elif line and pr is not None:
                top = "/".join(line.split("/")[:2])
                lst = _PR_CACHE.setdefault(top, [])
                if pr not in lst:
                    lst.append(pr)
        _PR_CACHE.setdefault("", [])
    return sorted(_PR_CACHE.get(os.path.relpath(folder, REPO), []))


_METHODS = [(r"\bpinns?\b|physics.informed", "PINN"), (r"\bfno\b|fourier neural operator", "FNO"),
            (r"deeponet", "DeepONet"), (r"meshgraphnet|\bgnn\b|graph network", "graph network"),
            (r"x-?tfc", "X-TFC"), (r"calculix", "FEM (CalculiX)"), (r"\bfem\b|finite element", "FEM"),
            (r"openfoam|simplefoam|pimplefoam", "CFD (OpenFOAM)"), (r"\blbm\b|lattice.boltzmann", "lattice Boltzmann"),
            (r"vortex lattice|\bvlm\b", "vortex lattice"), (r"\bpod\b|\bdmd\b|reduced.order|\brom\b", "reduced-order model"),
            (r"modelica|\bfmu\b", "Modelica / FMU"), (r"enkf|ensemble kalman", "ensemble Kalman filter"),
            (r"transformer", "transformer"), (r"lstm", "LSTM"), (r"gaussian process|\bgp\b", "Gaussian process"),
            (r"finite difference|\bfdm\b", "finite differences"), (r"fdtd", "FDTD"), (r"blender|cycles", "render (Blender)"),
            (r"symbolic regression|sindy", "symbolic regression"), (r"6-dof|six.degree", "6-DOF dynamics")]


def methods_of(text: str) -> list[str]:
    """Method labels whose names appear in ``text`` (title, description, module names): a filter facet, not a
    claim about what was validated."""
    t = text.lower()
    return list(dict.fromkeys(label for pat, label in _METHODS if re.search(pat, t)))


def _guide(rel: str) -> str:
    """The docs page that cites the folder, if one does."""
    import glob
    for p in sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True)):
        try:
            if rel in open(p, encoding="utf-8", errors="replace").read():
                return os.path.relpath(p, REPO)
        except OSError:
            pass
    return ""


def manifest_from_dir(folder: str, lab_curation: dict | None = None) -> Manifest:
    """A manifest made from what an existing example or results folder already holds: title and description from
    its README or first script's docstring, modules from its imports, its figures, movies, viewers and result
    files, the docs page that cites it, the pull request that added it, and, when the Lab has curated it, the item,
    its metrics and checks. Nothing is invented: what is not there is left empty."""
    folder = os.path.abspath(folder)
    rel = os.path.relpath(folder, REPO)
    files = [f for f in _git_files(folder) if not f.startswith("..") and FILENAME not in f]
    scripts = sorted(f for f in files if f.endswith(".py") and "/" not in f)
    title, desc = _title_of(folder, scripts)
    guide = _guide(rel)
    if guide and not os.path.exists(os.path.join(folder, "README.md")):
        head = next((ln.lstrip("# ").strip() for ln in open(os.path.join(REPO, guide), encoding="utf-8",
                                                             errors="replace") if ln.startswith("# ")), "")
        title = head or title
    desc = re.sub(r"\s*python [\w/.-]+\.py.*$", "", desc).strip()
    if len(title) > 100:
        title = title[:100].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    mods = _modules(folder, scripts)
    m = Manifest(id=rel, title=title, description=desc, domain=domain_of(os.path.basename(folder)),
                 methods=methods_of(" ".join([title, desc, *mods, *scripts])), modules=mods, scripts=scripts,
                 provenance={"source": "register", "methods": "inferred from names", "git": _git()})
    for f in sorted(files):
        ext = os.path.splitext(f)[1].lower()
        b = os.path.basename(f).lower()
        if ext in VIDEO_EXT:
            m.files["videos"].append(f)
        elif ext in FIGURE_EXT:
            m.files["figures"].append(f)
        elif ext == ".html" and ("viewer" in f.lower() or b == "index.html"):
            m.files["viewers"].append(f)
        elif ext in DATA_EXT and b not in ("readme.md",):
            m.files["data"].append(f)
    m.links = {k: v for k, v in (("guide", guide), ("script", f"{rel}/{scripts[0]}" if scripts else ""),
                                 ("pr", _prs(folder))) if v}
    _attach_lab(m, rel, lab_curation)
    return m


def _attach_lab(m: Manifest, rel: str, cur: dict | None) -> None:
    """Link the manifest to the Lab item that ran its scripts (``example_script``), with that run's metrics."""
    if not cur:
        return
    hits = [(k, it) for k, it in cur.get("items", {}).items()
            if it.get("experiment") == "example_script" and (it.get("case") or "").startswith(rel + "/")]
    if not hits:
        return
    key, it = sorted(hits)[0]
    m.status = "curated"
    m.lab = {"item": key, "experiment": it["experiment"], "run_id": it.get("best_run"), "tier": it.get("tier")}
    run = os.path.join(REPO, "lab", "runs", it["experiment"], it.get("best_run", ""))
    for k, v in _read(os.path.join(run, "metrics.json"), {}).items():
        if _number(v):
            m.metric(k, v, "", f"recorded by the Lab run {it.get('best_run')} of {it.get('case')}")


# ---------------------------------------------------------------------------------------------- discovery and the gate
def example_dirs(root: str | None = None) -> list[str]:
    """Every ``examples/*`` folder that must carry a manifest (not ``_*``, ``.*`` or the ignored names)."""
    root = root or os.path.join(REPO, "examples")
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(("_", ".")) and d not in IGNORED_EXAMPLE_DIRS]
    return sorted(d for d in dirs if not _git_ignored(d))


def _git_ignored(path: str) -> bool:
    """True for a folder whose manifest git would ignore (local output that is never committed: it needs none)."""
    try:
        return subprocess.run(["git", "check-ignore", "-q", os.path.join(path, FILENAME)], cwd=os.path.dirname(path),
                              capture_output=True, timeout=10).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def scan(roots: list[str] | None = None, *, lab: str | None = None) -> list[tuple[str, Manifest]]:
    """Every manifest under ``roots`` (default: the repository's ``examples/``) plus every run of the Lab at ``lab``
    (default: ``$PINNEAPPLE_LAB`` or ``<repo>/lab``): ``experiment.json`` files and Lab run folders. A new example
    with a manifest is found here with no code change."""
    out, seen = [], set()
    for root in roots or [os.path.join(REPO, "examples")]:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]
            if FILENAME in files:
                p = os.path.join(dirpath, FILENAME)
                try:
                    out.append((p, load(p)))
                    seen.add(os.path.abspath(dirpath))
                except (OSError, ValueError):
                    continue
    lab = lab if lab is not None else os.environ.get("PINNEAPPLE_LAB", os.path.join(REPO, "lab"))
    runs = os.path.join(lab, "runs") if lab else ""
    if runs and os.path.isdir(runs):
        for exp in sorted(os.listdir(runs)):
            for rid in sorted(os.listdir(os.path.join(runs, exp))):
                d = os.path.join(runs, exp, rid)
                if os.path.exists(os.path.join(d, "run.json")) and os.path.abspath(d) not in seen:
                    out.append((d, load(d)))
    return out


def check_repo(examples: str | None = None, lab: str | None = None) -> list[str]:
    """The CI gate: one line per problem, empty when every ``examples/*`` folder has a valid ``experiment.json``
    (files present, metrics with values) and every Lab run folder reads as a valid manifest."""
    problems = []
    for d in example_dirs(examples):
        p = os.path.join(d, FILENAME)
        rel = os.path.relpath(d, REPO)
        if not os.path.exists(p):
            problems.append(f"{rel}: no {FILENAME} (python -m pinneapple_lab register {rel})")
            continue
        try:
            m = load(p)
        except (OSError, ValueError) as e:
            problems.append(f"{rel}/{FILENAME}: {e}")
            continue
        problems += [f"{rel}/{FILENAME}: {e}" for e in validate(m, d)]
    lab = lab if lab is not None else os.path.join(REPO, "lab")
    runs = os.path.join(lab, "runs")
    if os.path.isdir(runs):
        for exp in sorted(os.listdir(runs)):
            for rid in sorted(os.listdir(os.path.join(runs, exp))):
                d = os.path.join(runs, exp, rid)
                if os.path.exists(os.path.join(d, "run.json")):
                    problems += [f"{os.path.relpath(d, REPO)}: {e}" for e in validate(load(d), d)]
    return problems


# ---------------------------------------------------------------------------------------------- helpers
def _read(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _iso(t) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if isinstance(t, (int, float)) else ""


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------------------------- scaffold
SCAFFOLD = '''"""{title}: what this experiment shows, in one sentence (edit me).

    python {rel}/run.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinneapple_lab import experiment

with experiment("{name}", title="{title}", domain="{domain}", methods=["(your method)"],
                params={{"n": 64}}) as ex:
    n = ex.manifest.params["n"]
    x = np.linspace(0.0, 1.0, n)
    u = np.sin(np.pi * x)                                          # replace with the real computation
    exact = np.sin(np.pi * x)
    err = float(np.max(np.abs(u - exact)))
    ex.log_metric("max_error", err, unit="", how="max |u - exact| on the n grid points")
    ex.check("u_vs_exact", value=err, max=1e-6, kind="reference", label="u vs sin(pi x)")
    fig, ax = plt.subplots()
    ax.plot(x, u)
    fig.savefig("{rel}/results/u.png")                        # recorded in the manifest automatically
'''


def scaffold(name: str, root: str | None = None, *, title: str = "", domain: str = "") -> str:
    """Create ``examples/<name>/`` with ``run.py`` wired to :func:`pinneapple_lab.experiment`, a README and a valid
    starting ``experiment.json``; returns the folder. Refuses an existing folder."""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("name: lowercase letters, digits and underscores")
    root = root or os.path.join(REPO, "examples")
    d = os.path.join(root, name)
    if os.path.exists(d):
        raise FileExistsError(d)
    os.makedirs(os.path.join(d, "results"))
    title = title or name.replace("_", " ").capitalize()
    domain = domain or domain_of(name)
    rel = os.path.relpath(d, os.getcwd())
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(SCAFFOLD.format(name=name, title=title, domain=domain, rel=rel))
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(f"# {title}\n\nWhat it shows, how to run it (`python {rel}/run.py`) and what it is checked "
                "against.\n")
    m = Manifest(id=os.path.relpath(d, REPO) if d.startswith(REPO) else name, title=title, domain=domain,
                 scripts=["run.py"], provenance={"source": "scaffold", "created": _iso(time.time())})
    save(m, d)
    return d
