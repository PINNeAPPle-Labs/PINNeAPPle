"""Configuration: ``pinneapple-check.toml`` (or ``[tool.pinneapple.check]`` in ``pyproject.toml``).

::

    [check]
    select = ["ALL"]            # ruff-style prefixes: "PC1", "PC201"
    ignore = []
    fail-on = "error"           # error | warning | never
    baseline = ".pinneapple-check-baseline.json"

    [[result]]
    path = "cases/cavity"       # OpenFOAM case, .vtu/.vtk, .npz, .h5, .csv of quantities of interest
    physics = "incompressible-flow"
    parameters = { nu = "0.01 m2/s" }
    bounds = { "alpha.water" = [0, 1] }
    [result.bc.movingWall]
    U = [1, 0, 0]

    [[model]]
    path = "models/surrogate.py:predict"
    inputs = ["x", "y"]
    outputs = ["u", "v", "p"]
    train-range = { x = [0, 1], y = [0, 1] }
    queries = "queries.csv"

Keys accept ``-`` or ``_``. Every path is relative to the configuration file. Without a configuration file the
checker discovers results under the given paths and runs the checks that need no declaration.
"""
from __future__ import annotations

import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:                                                                # pragma: no cover
    import tomli as tomllib

__all__ = ["ConfigError", "Config", "ResultTarget", "ModelTarget", "Integrity", "load_config", "find_config",
           "DEFAULT_TOLERANCES", "PHYSICS"]

CONFIG_NAMES = ("pinneapple-check.toml", ".pinneapple-check.toml")

PHYSICS = ("incompressible-flow", "heat", "elasticity", "conservation", "darcy", "maxwell", "none")
_PHYSICS_ALIASES = {"incompressible": "incompressible-flow", "flow": "incompressible-flow", "navier-stokes":
                    "incompressible-flow", "stokes": "incompressible-flow", "heat-conduction": "heat",
                    "thermal": "heat", "diffusion": "heat", "linear-elasticity": "elasticity", "solid": "elasticity",
                    "conservation-law": "conservation", "generic": "conservation", "darcy-flow": "darcy",
                    "porous-media": "darcy", "electromagnetics": "maxwell", "electrodynamics": "maxwell",
                    "em": "maxwell", "maxwell-equations": "maxwell"}

#: Default tolerances (all relative, dimensionless). Override per target under ``tolerances``.
DEFAULT_TOLERANCES: dict[str, float] = {
    "mass": 1e-3,              # PC101 |net boundary flux| / sum |boundary flux|
    "divergence": 1e-2,        # PC102 RMS(div u) / RMS(|du/dx| + |dv/dy| + |dw/dz|), stored fields
    "model-divergence": 5e-2,  # PC102 for a model (central differences at sampled points)
    "energy": 1e-2,            # PC103 |net heat flow| / sum |boundary heat flow|
    "momentum": 1e-2,          # PC104 |resultant| / sum |traction|
    "total-drift": 1e-3,       # PC105 max |Q(t) - Q(t0)| / |Q(t0)|
    "bc": 1e-3,                # PC201 max |value - declared| / field scale
    "ic": 1e-2,                # PC202 same, at the first time
    "wall-velocity": 1e-6,     # PC203 max |U| on a wall / velocity scale
    "residual": 5e-2,          # PC301 relative PDE residual of stored fields
    "model-residual": 5e-2,    # PC302 relative PDE residual of a model
    "rel-l2": 1e-2,            # PC701 relative L2 against the reference
    "hf-bias": 0.25,           # PC704 |E_pred / E_ref - 1| in the high-wavenumber band
    "qoi": 1e-2,               # PC702 relative tolerance of a quantity of interest (default)
    "fraction-slack": 1e-6,    # PC402 slack on [0, 1] for fractions
    "darcy-energy": 1e-1,      # PC301 darcy: certified relative energy-norm error bound from the FV residual
    "darcy-l2": 5e-2,          # PC301 darcy: certified relative L2 error bound (same 5 % as a reference comparison)
    "darcy-flux": 2e-2,        # PC101 darcy: |net boundary flux - source integral| / source integral
    "darcy-bc": 2e-2,          # PC201 darcy: RMS of u extrapolated to the boundary / max |u|
    "darcy-positivity": 1e-2,  # PC402 darcy: most negative u / max |u| (maximum principle, f >= 0)
    "maxwell-gauss": 5e-2,     # PC102 maxwell: relative residual of div D = rho and div B = 0 (central differences)
    "maxwell-faraday": 5e-2,   # PC301 maxwell: relative residual of curl E + dB/dt = 0
    "maxwell-ampere": 5e-2,    # PC301 maxwell: relative residual of curl H - J - dD/dt = 0
    "maxwell-continuity": 5e-2,  # PC301 maxwell: relative residual of d rho/dt + div J = 0
    "maxwell-energy": 5e-2,    # PC103 maxwell: Poynting balance over the grid box / largest term
}


class ConfigError(ValueError):
    """The configuration cannot be used (exit code 2)."""


def _keys(d: dict[str, Any]) -> dict[str, Any]:
    """Normalise top-level keys of a table (``train_range`` -> ``train-range``); field names stay as written."""
    return {k.replace("_", "-"): v for k, v in d.items()}


def _physics(p: str | None, where: str) -> str | None:
    if p is None:
        return None
    q = str(p).strip().lower().replace("_", "-").replace(" ", "-")
    q = _PHYSICS_ALIASES.get(q, q)
    if q not in PHYSICS:
        raise ConfigError(f"{where}: unknown physics {p!r}; one of {', '.join(PHYSICS)}")
    return q


@dataclass
class Reference:
    path: Path
    fields: list[str] | None = None
    rel_l2: float | None = None
    qoi: dict[str, dict[str, float]] = field(default_factory=dict)   # name -> {"rel": .., "abs": ..}


@dataclass
class _Target:
    name: str
    path: Path
    physics: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    bounds: dict[str, list[float | None]] = field(default_factory=dict)
    bc: dict[str, dict[str, Any]] = field(default_factory=dict)
    ic: dict[str, Any] = field(default_factory=dict)
    reference: Reference | None = None
    tolerances: dict[str, float] = field(default_factory=dict)
    sha256: str | None = None
    line: int = 1                       # line of the declaration in the config file (annotations point there)
    declared: bool = True               # False for a target found by discovery

    def tol(self, key: str) -> float:
        return float(self.tolerances.get(key, DEFAULT_TOLERANCES[key]))


@dataclass
class ResultTarget(_Target):
    kind: str | None = None             # openfoam | grid | points | qoi (auto)
    time: str | None = None             # OpenFOAM time directory ("latest" by default)
    fields: list[str] | None = None
    flux: dict[str, str] = field(default_factory=dict)     # conservation: {"q": "flux column"}
    conserved: list[str] = field(default_factory=list)
    envelope: dict[str, Any] = field(default_factory=dict)  # darcy: training envelope of the coefficient (PC401)


@dataclass
class ModelTarget(_Target):
    architecture: str | None = None     # zoo name of a checkpoint's architecture
    model_config: dict[str, Any] = field(default_factory=dict)
    hub: str | None = None
    zoo: str | None = None              # a pp.zoo name (pre-trained, verified model)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    train_range: dict[str, list[float]] = field(default_factory=dict)
    train_samples: Path | None = None
    queries: Path | None = None
    domain: dict[str, list[float]] = field(default_factory=dict)
    method: str | None = None
    n_points: int = 1024
    seed: int = 0
    relations: list[dict[str, Any]] = field(default_factory=list)   # metamorphic relations (PC303)


@dataclass
class DatasetTarget(_Target):
    zoo: str | None = None              # a pp.datasets name
    inputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    train: str = "train"
    test: str = "test"
    audit: dict[str, float] = field(default_factory=dict)          # tolerances of pinneapple_data.audit


@dataclass
class Integrity:
    manifests: list[Path] = field(default_factory=list)
    public_key: Path | None = None
    signed: list[str] = field(default_factory=list)          # globs of files that must carry a valid .sig
    line: int = 1


@dataclass
class Config:
    root: Path
    file: Path | None = None
    select: list[str] | None = None
    ignore: list[str] = field(default_factory=list)
    extend_select: list[str] = field(default_factory=list)
    fail_on: str = "error"
    baseline: Path | None = None
    severity: dict[str, str] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)
    results: list[ResultTarget] = field(default_factory=list)
    models: list[ModelTarget] = field(default_factory=list)
    datasets: list[DatasetTarget] = field(default_factory=list)
    integrity: Integrity = field(default_factory=Integrity)

    @property
    def targets(self) -> list[_Target]:
        return [*self.results, *self.models, *self.datasets]

    def rel(self, p: Path | str) -> str:
        """Path relative to the project root, posix style (what SARIF and GitHub annotations want)."""
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()


# ---------------------------------------------------------------------------------------------- finding the file
def find_config(start: Path) -> tuple[Path | None, dict[str, Any]]:
    """Walk up from ``start``: the first ``pinneapple-check.toml`` or ``pyproject.toml`` with
    ``[tool.pinneapple.check]``. Returns (file, the check table)."""
    start = start.resolve()
    for d in [start, *start.parents]:
        for n in CONFIG_NAMES:
            f = d / n
            if f.is_file():
                return f, _read(f)
        f = d / "pyproject.toml"
        if f.is_file():
            data = _read(f, whole=True)
            sec = data.get("tool", {}).get("pinneapple", {}).get("check")
            if sec is not None:
                return f, sec
        if (d / ".git").exists():                     # stop at the repository root
            break
    return None, {}


def _read(f: Path, whole: bool = False) -> dict[str, Any]:
    try:
        data = tomllib.loads(f.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{f}: {e}") from e
    if whole:
        return data
    # pinneapple-check.toml: the [check] table plus [[result]] / [[model]] / [[dataset]] at the top level
    out = dict(data.get("check", {}))
    for k in ("result", "results", "model", "models", "dataset", "datasets", "integrity"):
        if k in data:
            out[k] = data[k]
    return out


def _line_of(text: str, needle: str, after: int = 0) -> int:
    lines = text.splitlines()
    for i in range(after, len(lines)):
        if needle in lines[i]:
            return i + 1
    return after + 1 if after else 1


# ---------------------------------------------------------------------------------------------- parsing
def load_config(file: Path | None = None, start: Path | None = None, root: Path | None = None) -> Config:
    """Parse a configuration file (found from ``start`` when ``file`` is None)."""
    if file is None:
        file, raw = find_config(start or Path.cwd())
    else:
        file = Path(file)
        if not file.is_file():
            raise ConfigError(f"configuration file {file} not found")
        if file.name == "pyproject.toml":
            raw = _read(file, whole=True).get("tool", {}).get("pinneapple", {}).get("check", {})
        else:
            raw = _read(file)
    base = file.parent if file else (start or Path.cwd())
    cfg = Config(root=(root or _repo_root(base)).resolve(), file=file)
    if not raw:
        return cfg
    text = file.read_text(encoding="utf-8") if file else ""
    raw = _keys(raw)
    known = {"select", "ignore", "extend-select", "fail-on", "baseline", "severity", "exclude", "result", "results",
             "model", "models", "dataset", "datasets", "integrity"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{file}: unknown key(s) {', '.join(sorted(unknown))} under [check]")
    cfg.select = _list(raw.get("select"), "select") if "select" in raw else None
    cfg.ignore = _list(raw.get("ignore", []), "ignore")
    cfg.extend_select = _list(raw.get("extend-select", []), "extend-select")
    cfg.fail_on = str(raw.get("fail-on", "error"))
    if cfg.fail_on not in ("error", "warning", "never"):
        raise ConfigError(f"{file}: fail-on must be error, warning or never")
    if raw.get("baseline"):
        cfg.baseline = base / raw["baseline"]
    cfg.severity = {k.upper(): str(v) for k, v in (raw.get("severity") or {}).items()}
    for k, v in cfg.severity.items():
        if v not in ("error", "warning"):
            raise ConfigError(f"{file}: severity of {k} must be 'error' or 'warning'")
    cfg.exclude = _list(raw.get("exclude", []), "exclude")
    seen: dict[str, int] = {}
    pos = 0
    for i, r in enumerate(_tables(raw, "result", "results")):
        t = _result(_keys(r), base, f"{file.name if file else ''} result #{i + 1}")
        pos = t.line = _line_of(text, _path_needle(r), pos)
        cfg.results.append(t)
    pos = 0
    for i, m in enumerate(_tables(raw, "model", "models")):
        t = _model(_keys(m), base, f"{file.name if file else ''} model #{i + 1}")
        pos = t.line = _line_of(text, _path_needle(m), pos)
        cfg.models.append(t)
    pos = 0
    for i, d in enumerate(_tables(raw, "dataset", "datasets")):
        t = _dataset(_keys(d), base, f"{file.name if file else ''} dataset #{i + 1}")
        pos = t.line = _line_of(text, _path_needle(d), pos)
        cfg.datasets.append(t)
    for t in cfg.targets:
        if t.name in seen:
            raise ConfigError(f"{file}: two targets are named {t.name!r}; give one a `name`")
        seen[t.name] = 1
    if "integrity" in raw:
        it = _keys(raw["integrity"])
        man = it.get("manifest", it.get("manifests", []))
        cfg.integrity = Integrity(
            manifests=[base / m for m in ([man] if isinstance(man, str) else man)],
            public_key=base / it["public-key"] if it.get("public-key") else None,
            signed=_list(it.get("signed", []), "integrity.signed"),
            line=_line_of(text, "integrity"))
    return cfg


def _dataset(r: dict[str, Any], base: Path, where: str) -> DatasetTarget:
    known = {"name", "path", "zoo", "physics", "units", "inputs", "constraints", "train", "test", "audit",
             "sha256"}
    if set(r) - known:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(set(r) - known))}")
    if not r.get("path") and not r.get("zoo"):
        raise ConfigError(f"{where}: a dataset needs `path` (an .npz with <split>__<column> arrays) or `zoo`")
    physics = r.get("physics")
    if physics is not None and physics not in ("darcy",):
        raise ConfigError(f"{where}: dataset physics {physics!r} is not built in (darcy)")
    p = base / r["path"] if r.get("path") else Path(f"zoo:{r['zoo']}")
    return DatasetTarget(name=str(r.get("name") or r.get("path") or r.get("zoo")), path=p, physics=physics,
                         units={str(k): str(v) for k, v in (r.get("units") or {}).items()},
                         sha256=r.get("sha256"), zoo=r.get("zoo"), inputs=[str(x) for x in r.get("inputs", [])],
                         constraints=[str(x) for x in r.get("constraints", [])], train=str(r.get("train", "train")),
                         test=str(r.get("test", "test")),
                         audit={str(k): float(v) for k, v in (r.get("audit") or {}).items()})


def _path_needle(t: dict[str, Any]) -> str:
    p = t.get("path") or t.get("hub") or t.get("zoo") or ""
    return f'"{p}"' if p else "path"


def _tables(raw, *names) -> list[dict[str, Any]]:
    out = []
    for n in names:
        v = raw.get(n)
        if v is None:
            continue
        if isinstance(v, dict):
            v = [v]
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            raise ConfigError(f"[[{n}]] must be an array of tables")
        out += v
    return out


def _list(v, key) -> list[str]:
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        raise ConfigError(f"{key} must be a list of strings")
    return [str(x) for x in v]


def _repo_root(start: Path) -> Path:
    start = start.resolve()
    for d in [start, *start.parents]:
        if (d / ".git").exists():
            return d
    return start


def _reference(v, base: Path, where: str) -> Reference | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = {"path": v}
    v = _keys(v)
    if "path" not in v:
        raise ConfigError(f"{where}: reference needs a path")
    qoi = {}
    for k, tol in (v.get("qoi") or {}).items():
        if isinstance(tol, (int, float)):
            qoi[k] = {"rel": float(tol)}
        elif isinstance(tol, dict):
            qoi[k] = {kk: float(vv) for kk, vv in tol.items() if kk in ("rel", "abs")}
        else:
            raise ConfigError(f"{where}: reference.qoi.{k} must be a number or {{rel=.., abs=..}}")
    return Reference(path=base / v["path"], fields=v.get("fields"), rel_l2=v.get("rel-l2"), qoi=qoi)


def _common(r: dict[str, Any], base: Path, where: str) -> dict[str, Any]:
    if "path" not in r and "hub" not in r and "zoo" not in r:
        raise ConfigError(f"{where}: missing `path`")
    tol = _keys(r.get("tolerances") or {})
    bad = set(tol) - set(DEFAULT_TOLERANCES)
    if bad:
        raise ConfigError(f"{where}: unknown tolerance(s) {', '.join(sorted(bad))}; "
                          f"known: {', '.join(DEFAULT_TOLERANCES)}")
    path = r.get("path") or r.get("hub") or r.get("zoo")
    return dict(
        name=str(r.get("name") or path),
        path=(base / path) if r.get("path") and ":" not in str(r.get("path"))[2:] else Path(str(path)),
        physics=_physics(r.get("physics"), where),
        parameters=dict(r.get("parameters") or {}),
        units={k: str(v) for k, v in (r.get("units") or {}).items()},
        bounds={k: list(v) for k, v in (r.get("bounds") or {}).items()},
        bc={k: dict(v) for k, v in (r.get("bc") or {}).items()},
        ic=dict(r.get("ic") or {}),
        reference=_reference(r.get("reference"), base, where),
        tolerances={k: float(v) for k, v in tol.items()},
        sha256=r.get("sha256"),
    )


def _result(r: dict[str, Any], base: Path, where: str) -> ResultTarget:
    known = {"name", "path", "physics", "parameters", "units", "bounds", "bc", "ic", "reference", "tolerances",
             "sha256", "kind", "time", "fields", "flux", "conserved", "envelope"}
    if set(r) - known:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(set(r) - known))}")
    c = _common(r, base, where)
    for k, v in c["bounds"].items():
        if len(v) != 2:
            raise ConfigError(f"{where}: bounds.{k} must be [low, high] (use \"-inf\"/\"inf\" or false for open)")
    env = {str(k): v for k, v in (r.get("envelope") or {}).items()}
    bad = set(env) - {"a-min", "a-max", "contrast", "high-fraction", "interface-density"}
    if bad:
        raise ConfigError(f"{where}: unknown envelope key(s) {', '.join(sorted(bad))} (a-min, a-max, contrast, "
                          "high-fraction, interface-density)")
    for k, v in env.items():
        if k not in ("a-min", "a-max") and (not isinstance(v, list) or len(v) != 2):
            raise ConfigError(f"{where}: envelope.{k} must be [low, high]")
    return ResultTarget(**c, kind=r.get("kind"), time=None if r.get("time") in (None, "latest") else str(r["time"]),
                        fields=r.get("fields"), flux=dict(r.get("flux") or {}), conserved=list(r.get("conserved") or []),
                        envelope=env)


_RELATION_KINDS = {"monotonic": {"input", "output"}, "invariant": {"inputs", "output"},
                   "symmetric": {"inputs"}, "superposition": set()}


def _relations(v: Any, where: str) -> list[dict[str, Any]]:
    """``relations = [{kind = "monotonic", input = "q", output = "T", direction = "increasing"}, ...]`` (PC303)."""
    out = []
    for i, r in enumerate(v or []):
        if not isinstance(r, dict) or r.get("kind") not in _RELATION_KINDS:
            raise ConfigError(f"{where}: relations[{i}] needs kind = {' | '.join(_RELATION_KINDS)}")
        missing = _RELATION_KINDS[r["kind"]] - set(r)
        if missing:
            raise ConfigError(f"{where}: relations[{i}] ({r['kind']}) needs {', '.join(sorted(missing))}")
        out.append(dict(r))
    return out


def _model(r: dict[str, Any], base: Path, where: str) -> ModelTarget:
    known = {"name", "path", "hub", "zoo", "physics", "parameters", "units", "bounds", "bc", "ic", "reference",
             "tolerances", "sha256", "architecture", "config", "inputs", "outputs", "train-range", "train-samples",
             "queries", "domain", "method", "n-points", "seed", "relations"}
    if set(r) - known:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(set(r) - known))}")
    c = _common(r, base, where)
    p = str(r.get("path", ""))
    if p and ":" in p[2:]:                                     # "file.py:function" (a drive letter is not a callable)
        f, fn = p.rsplit(":", 1)
        c["path"] = Path(f"{base / f}:{fn}")
    if not r.get("zoo") and (not r.get("inputs") or not r.get("outputs")):
        raise ConfigError(f"{where}: a model needs `inputs` and `outputs` (names of its columns)")
    return ModelTarget(
        **c, architecture=r.get("architecture"), model_config=dict(r.get("config") or {}), hub=r.get("hub"),
        zoo=r.get("zoo"), inputs=[str(x) for x in r.get("inputs", [])], outputs=[str(x) for x in r.get("outputs", [])],
        train_range={k: [float(a) for a in v] for k, v in (r.get("train-range") or {}).items()},
        train_samples=base / r["train-samples"] if r.get("train-samples") else None,
        queries=base / r["queries"] if r.get("queries") else None,
        domain={k: [float(a) for a in v] for k, v in (r.get("domain") or {}).items()},
        method=r.get("method"), n_points=int(r.get("n-points", 1024)), seed=int(r.get("seed", 0)),
        relations=_relations(r.get("relations"), where))


# ---------------------------------------------------------------------------------------------- discovery
_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
              ".ruff_cache", ".pytest_cache", "site-packages", "dist", "build"}
_RESULT_EXT = (".vtu", ".vtk", ".npz", ".h5", ".hdf5")


def discover(paths: list[Path], exclude: list[str], root: Path) -> list[ResultTarget]:
    """Results found under ``paths`` when the configuration declares none: OpenFOAM case folders and field files."""
    out: list[ResultTarget] = []

    def excluded(p: Path) -> bool:
        rel = _relposix(p, root)
        return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(p.name, g) for g in exclude)

    def is_case(d: Path) -> bool:
        return (d / "constant" / "polyMesh").is_dir() and ((d / "system").is_dir() or any(_time_dirs(d)))

    for p in paths:
        p = p.resolve()
        if p.is_file():
            if p.suffix.lower() in _RESULT_EXT and not excluded(p):
                out.append(ResultTarget(name=_relposix(p, root), path=p, declared=False))
            continue
        for d, dirs, files in os.walk(p):
            dp = Path(d)
            dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS and not x.startswith(".")
                             and not excluded(dp / x))
            if is_case(dp):
                if not excluded(dp):
                    out.append(ResultTarget(name=_relposix(dp, root), path=dp, declared=False))
                dirs[:] = []
                continue
            for f in sorted(files):
                fp = dp / f
                if fp.suffix.lower() in _RESULT_EXT and not excluded(fp) and "reference" not in fp.parts:
                    out.append(ResultTarget(name=_relposix(fp, root), path=fp, declared=False))
    return out


def _time_dirs(d: Path):
    for x in d.iterdir():
        try:
            float(x.name)
            if x.is_dir():
                yield x
        except ValueError:
            pass


def _relposix(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
