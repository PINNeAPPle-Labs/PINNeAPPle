"""``pp.artifact``: the ``.ppx`` container — one file with a result and everything needed to reproduce it.

A ``.ppx`` is a zip holding a JSON manifest and the members that manifest names::

    experiment.ppx
    ├── manifest.json          # schema, problem, units, solver, config, metrics, validation, environment, provenance
    ├── arrays/*.npy           # meshes, fields, datasets, probe points and their expected outputs
    └── model/state.pt         # the weights of the model the artifact carries, when it carries one

The manifest never holds a number that a member does not back: every array and every weight file is listed in
``integrity`` with its sha256, and :func:`load` refuses an artifact whose members do not match what the manifest says.

    import pinneapple as pp

    art = pp.Artifact.from_solution(solution, title="poisson, 32x32")
    pp.save(art, "experiment.ppx")
    same = pp.load("experiment.ppx")           # IntegrityError if a member changed, MissingProvenance if it says nothing
    report = pp.verify_artifact("experiment.ppx")   # integrity, provenance, units, reproduction
    report.passed

**Reproducing on another machine.** An artifact that carries a model records the class and the keyword arguments
that built it (``model/build``) next to the weights, so :meth:`Artifact.predict` and :func:`verify` rebuild the model
from the file alone and compare its outputs to the probe outputs stored with it, within ``rtol``/``atol``.
A solved problem cannot always be re-run from the file — the classical backends take a callable source
(``ctx={"source_fn": ...}``) that no container can carry — so for those the artifact records the problem, the method
and the settings, and :meth:`Artifact.reproduce` accepts the callable that produces the result again, comparing it to
the stored probes rather than pretending it could be rebuilt.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .jsonable import to_jsonable

__all__ = ["SCHEMA", "EXTENSION", "Artifact", "Report", "save", "load", "verify", "ArtifactError", "UnsupportedSchema",
           "IntegrityError", "MissingProvenance"]

SCHEMA = "pinneapple.ppx/1"
EXTENSION = ".ppx"
MANIFEST = "manifest.json"
STATE_MEMBER = "model/state.pt"
ARRAYS = "arrays/"
REQUIRED_PROVENANCE = ("source", "created_at", "environment", "git")
REQUIRED_SECTIONS = ("problem", "units", "solver", "environment", "provenance")


class ArtifactError(Exception):
    """Anything that makes an artifact unusable."""


class UnsupportedSchema(ArtifactError):
    """The file is a ``.ppx`` but not one this version reads."""


class IntegrityError(ArtifactError):
    """A member is missing, changed or not listed in the manifest."""


class MissingProvenance(ArtifactError):
    """The manifest does not say where the result came from, so the result cannot be trusted or reproduced."""


# --------------------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_root() -> Path | None:
    """The checkout of this package, when it is a git checkout (``None`` in an installed copy)."""
    here = Path(__file__).resolve().parents[1]
    return here if (here / ".git").exists() else None


def _git() -> dict[str, Any]:
    """Commit, branch and dirty flag of the checkout, or an honest reason when there is no git to ask."""
    root = _repo_root()
    if root is None:
        return {"commit": None, "branch": None, "dirty": None, "reason": "not a git checkout"}
    out: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            out[key] = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5,
                                      check=True).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            out[key] = None
    try:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                                timeout=5, check=True).stdout
        out["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        out["dirty"] = None
    if out["commit"] is None:
        out["reason"] = "git could not report the commit"
    return out


def _environment() -> dict[str, Any]:
    """What produced the numbers: interpreter, OS, CPU architecture and the versions of the packages that matter."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": {"numpy": np.__version__},
    }
    try:                                        # the library itself, when it is importable as installed
        import pinneapple

        env["packages"]["pinneapple"] = getattr(pinneapple, "__version__", None)
    except Exception:                           # noqa: BLE001 - a source checkout without metadata is fine
        env["packages"]["pinneapple"] = None
    try:
        import torch

        env["packages"]["torch"] = torch.__version__
        env["accelerator"] = "cuda" if torch.cuda.is_available() else "cpu"
        env["threads"] = torch.get_num_threads()
    except Exception:                           # noqa: BLE001 - an array-only artifact has no torch
        env["packages"]["torch"] = None
    return env


def _provenance(source: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"source": source, "created_at": _now(), "git": _git(), "environment": _environment()}
    if extra:
        out.update(to_jsonable(extra))
    return out


def _missing_provenance(prov: Mapping[str, Any] | None) -> list[str]:
    """Which required keys of the provenance block are absent or empty."""
    if not isinstance(prov, Mapping):
        return ["<no provenance block>"]
    missing = [k for k in REQUIRED_PROVENANCE if not prov.get(k)]
    if isinstance(prov.get("git"), Mapping) and not prov["git"].get("commit") and not prov["git"].get("reason"):
        missing.append("git.commit (and no reason recorded)")
    return missing


def _zip_path_name(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.suffix == EXTENSION else p.with_suffix(EXTENSION)


def _run(model: Any, x: Any) -> np.ndarray:
    """Forward a torch model on a numpy array, numpy out (no gradients, no device surprises)."""
    import torch

    with torch.no_grad():
        out = model(torch.as_tensor(np.asarray(x), dtype=torch.get_default_dtype()))
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


# --------------------------------------------------------------------------- the artifact
@dataclass
class Artifact:
    """A result plus everything needed to reproduce it.

    Every field is JSON-safe except ``arrays`` and ``model_state``: the first holds the ``.npy`` members by name, the
    second the weights. ``reproductions`` names, for each probe set, the array with the inputs, the array with the
    outputs that were produced, and the tolerance under which a re-run is the same result.
    """
    meta: dict[str, Any] = field(default_factory=dict)
    problem: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    model_build: dict[str, Any] | None = None
    model_state: dict[str, Any] | None = None
    reproductions: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_solution(cls, solution: Any, *, title: str | None = None, units: Mapping[str, str] | None = None,
                      samples: int = 0, seed: int = 0, config: Mapping[str, Any] | None = None,
                      validation: Sequence[Mapping[str, Any]] = (), metrics: Sequence[Mapping[str, Any]] = (),
                      source: str = "Artifact.from_solution") -> Artifact:
        """An artifact of a solved problem (``pp.solve(...)``). With ``samples > 0`` it stores that many probe points
        drawn from the problem's bounds and the values the solution gives there, so the result can be compared later
        on another machine (see :meth:`reproduce`)."""
        problem = getattr(solution, "problem", None)
        art = cls(
            meta={"id": getattr(problem, "name", None), "title": title or getattr(problem, "name", None), "created_at": _now()},
            problem=to_jsonable(problem.to_dict()) if hasattr(problem, "to_dict") else {},
            units=dict(units if units is not None else getattr(problem, "units", {}) or {}),
            solver={"method": getattr(solution, "method", None), "kind": getattr(solution, "kind", None),
                    "wall_time_s": getattr(solution, "wall_time_s", None),
                    "info": to_jsonable(getattr(solution, "info", {}) or {})},
            config=to_jsonable(dict(config or {})),
            metrics=[{"name": "wall_time_s", "value": getattr(solution, "wall_time_s", None), "unit": "s",
                      "how": f"pp.solve(..., {getattr(solution, 'method', '?')!r})"},
                     *[dict(m) for m in metrics]],
            validation=[dict(v) for v in validation],
            environment=_environment(),
            provenance=_provenance(source),
        )
        grid = solution.grid_points() if hasattr(solution, "grid_points") else None
        if samples and grid is not None:
            art.arrays["grid"] = np.asarray(grid, dtype=np.float64)
        if samples:
            probe = art._probe(samples, seed, solution)
            if probe is not None:
                art.arrays["probe"] = probe
                art.arrays["probe_values"] = np.asarray(solution.predict(probe), dtype=np.float64)
                art.reproductions.append({"name": "solution", "inputs": "probe", "expected": "probe_values",
                                          "rtol": 1e-9, "atol": 1e-12, "needs": "callable"})
        return art

    @classmethod
    def from_model(cls, model: Any, *, init_kwargs: Mapping[str, Any] | None = None, problem: Any = None,
                   title: str | None = None, lo: Sequence[float] | None = None, hi: Sequence[float] | None = None,
                   samples: int = 0, seed: int = 0, units: Mapping[str, str] | None = None,
                   config: Mapping[str, Any] | None = None) -> Artifact:
        """An artifact around a trained model. ``init_kwargs`` are the arguments that rebuild it around the saved
        weights; without them the weights are still stored, but nothing can rebuild the model from the file alone."""
        art = cls(
            meta={"id": getattr(problem, "name", None), "title": title or type(model).__name__, "created_at": _now()},
            problem=to_jsonable(problem.to_dict()) if hasattr(problem, "to_dict") else {},
            units=dict(units if units is not None else getattr(problem, "units", {}) or {}),
            solver={"method": "model", "kind": getattr(model, "kind", "neural"), "wall_time_s": None},
            config=to_jsonable(dict(config or {})),
            environment=_environment(),
            provenance=_provenance("Artifact.from_model"),
            model_build=({"class": f"{type(model).__module__}:{type(model).__qualname__}",
                          "init_kwargs": to_jsonable(dict(init_kwargs))} if init_kwargs is not None else None),
            model_state={k: v.detach().clone() for k, v in model.state_dict().items()},
        )
        if samples and lo is not None and hi is not None:
            rng = np.random.default_rng(seed)
            probe = rng.uniform(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), size=(samples, len(lo)))
            art.arrays["probe"] = probe
            art.arrays["probe_values"] = _run(model, probe)
            art.reproductions.append({"name": "model", "inputs": "probe", "expected": "probe_values",
                                      "rtol": 1e-5, "atol": 1e-7, "needs": "weights"})
        return art

    def _probe(self, samples: int, seed: int, solution: Any) -> np.ndarray | None:
        """Probe points inside the problem's bounds (its own grid has the points when the solution is on a grid)."""
        problem = getattr(solution, "problem", None)
        bounds = getattr(problem, "domain_bounds", None) or {}
        coords = list(getattr(problem, "coords", ()) or ())
        if not coords or any(c not in bounds for c in coords):
            return None
        rng = np.random.default_rng(seed)
        lo = np.array([bounds[c][0] for c in coords], dtype=float)
        hi = np.array([bounds[c][1] for c in coords], dtype=float)
        return rng.uniform(lo, hi, size=(samples, len(coords)))

    # -- model --------------------------------------------------------------
    def build_model(self) -> Any:
        """The model, rebuilt from the class and keyword arguments recorded in the file and loaded with the weights."""
        if self.model_state is None:
            raise ArtifactError("this artifact carries no model weights")
        if not self.model_build or ":" not in str(self.model_build.get("class", "")):
            raise ArtifactError("no recipe to rebuild the model: it was saved without init_kwargs")
        module_name, _, class_name = self.model_build["class"].partition(":")
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)(**dict(self.model_build.get("init_kwargs") or {}))
        model.load_state_dict(self.model_state)
        return model.eval()

    def predict(self, x: Any) -> np.ndarray:
        """Run the carried model on ``x`` (numpy in, numpy out)."""
        return _run(self.build_model(), x)

    # -- reproduction -------------------------------------------------------
    def reproduce(self, predict_fn: Callable[[np.ndarray], np.ndarray] | None = None, *,
                  rtol: float | None = None, atol: float | None = None) -> Report:
        """Recompute every recorded reproduction and compare it to what was stored.

        Without ``predict_fn`` the model is rebuilt from the file (weights + recipe). Pass ``predict_fn`` for a solved
        problem: the callable that produces the result again (for example ``lambda X: pp.solve(prob, "fem",
        ctx=ctx).predict(X)``), since no container can carry that callable. A reproduction whose inputs are not in
        the file is reported as failed, not skipped: a container that cannot be checked is not evidence."""
        checks: list[dict[str, Any]] = []
        for rep in self.reproductions:
            name = rep.get("name", "?")
            if predict_fn is None and self.model_state is None:
                checks.append({"name": f"reproduction:{name}", "passed": None,
                               "detail": f"skipped: {rep.get('needs', 'callable')} needed, pass predict_fn"})
                continue
            inputs = self.arrays.get(rep.get("inputs", ""))
            expected = self.arrays.get(rep.get("expected", ""))
            if inputs is None or expected is None:
                checks.append({"name": f"reproduction:{name}", "passed": False,
                               "detail": f"probe arrays {rep.get('inputs')!r}/{rep.get('expected')!r} are not in the artifact"})
                continue
            try:
                got = np.asarray(predict_fn(inputs) if predict_fn is not None else self.predict(inputs), dtype=np.float64)
            except Exception as exc:            # noqa: BLE001 - a reproduction that raised is a failed check, not a crash
                checks.append({"name": f"reproduction:{name}", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
                continue
            r_tol = rep.get("rtol", 1e-6) if rtol is None else rtol
            a_tol = rep.get("atol", 1e-9) if atol is None else atol
            same_shape = got.shape == np.asarray(expected).shape
            ok = bool(same_shape and np.allclose(got, expected, rtol=r_tol, atol=a_tol))
            worst = float(np.max(np.abs(got - expected))) if same_shape else float("nan")
            checks.append({"name": f"reproduction:{name}", "passed": ok,
                           "detail": f"max |difference| = {worst:.3g} (rtol {r_tol:g}, atol {a_tol:g})"
                                     + ("" if same_shape else f"; shape {got.shape} != {np.asarray(expected).shape}")})
        return Report(checks)

    # -- manifest -----------------------------------------------------------
    def to_manifest(self, integrity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        manifest = {
            "schema": SCHEMA, **{k: v for k, v in self.meta.items()},
            "problem": self.problem, "units": self.units, "solver": self.solver, "config": self.config,
            "metrics": self.metrics, "validation": self.validation, "datasets": self.datasets,
            "arrays": {name: {"member": f"{ARRAYS}{name}.npy", "shape": list(np.shape(a)), "dtype": str(np.asarray(a).dtype)}
                       for name, a in self.arrays.items()},
            "model": ({"member": STATE_MEMBER, "build": self.model_build,
                       "parameters": int(sum(np.asarray(v).size for v in self.model_state.values()))}
                      if self.model_state is not None else None),
            "reproductions": self.reproductions,
            "environment": self.environment, "provenance": self.provenance,
            "integrity": dict(integrity or {}),
        }
        return to_jsonable(manifest)


@dataclass
class Report:
    """The outcome of :func:`verify` or :meth:`Artifact.reproduce`: one entry per check, and whether all of them held."""
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c["passed"] is not False for c in self.checks) and any(c["passed"] is True for c in self.checks)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["passed"] is False]

    def __repr__(self) -> str:
        bad = len(self.failed)
        return f"Report(passed={self.passed}, {len(self.checks)} checks, {bad} failed)"


# --------------------------------------------------------------------------- reading and writing
def _coerce(obj: Any, **kwargs: Any) -> Artifact:
    if isinstance(obj, Artifact):
        return obj
    if hasattr(obj, "predict_fn") and hasattr(obj, "problem"):        # a Solution, duck-typed
        keep = ("title", "units", "samples", "seed", "config", "validation", "metrics")
        return Artifact.from_solution(obj, **{k: v for k, v in kwargs.items() if k in keep})
    if hasattr(obj, "state_dict"):                                     # a torch module
        keep = ("title", "units", "samples", "seed", "config", "init_kwargs", "problem", "lo", "hi")
        return Artifact.from_model(obj, **{k: v for k, v in kwargs.items() if k in keep})
    raise TypeError(f"cannot save {type(obj).__name__}: pass an Artifact, a Solution or a torch model")


def save(artifact: Any, path: str | os.PathLike[str], *, title: str | None = None, units: Mapping[str, str] | None = None,
         samples: int = 0, seed: int = 0, init_kwargs: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None,
         **kwargs: Any) -> Path:
    """Write ``artifact`` as a ``.ppx`` file and return its path (the extension is added when missing).

    Accepts an :class:`Artifact`, a ``Solution`` from ``pp.solve`` or a torch model; the keyword arguments are what
    the matching constructor needs (``samples`` stores probe points for the reproduction)."""
    art = _coerce(artifact, title=title, units=units, samples=samples, seed=seed, init_kwargs=init_kwargs, config=config,
                  **{k: v for k, v in kwargs.items() if k in ("validation", "metrics", "problem", "lo", "hi")})
    if not art.provenance:
        art.provenance = _provenance("save")
    if not art.environment:
        art.environment = _environment()
    if _missing_provenance(art.provenance):
        raise MissingProvenance(f"refusing to write an artifact without provenance: missing {_missing_provenance(art.provenance)}")
    target = _zip_path_name(path)
    members: dict[str, bytes] = {}
    for name, array in art.arrays.items():
        buf = io.BytesIO()
        np.save(buf, np.asarray(array), allow_pickle=False)
        members[f"{ARRAYS}{name}.npy"] = buf.getvalue()
    if art.model_state is not None:
        import torch

        buf = io.BytesIO()
        torch.save(art.model_state, buf)
        members[STATE_MEMBER] = buf.getvalue()
    integrity = {m: {"sha256": _sha256(data), "bytes": len(data)} for m, data in sorted(members.items())}
    manifest = art.to_manifest(integrity)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
        for member, data in members.items():
            z.writestr(member, data, compress_type=zipfile.ZIP_STORED if member.endswith((".npy", ".pt")) else zipfile.ZIP_DEFLATED)
    return target


def _read_members(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise UnsupportedSchema(f"{path} is not a .ppx container: {exc}") from exc
    with zf:
        names = zf.namelist()
        if MANIFEST not in names:
            raise ArtifactError(f"{path} has no {MANIFEST}")
        manifest = json.loads(zf.read(MANIFEST))
        if not str(manifest.get("schema", "")).startswith("pinneapple.ppx/"):
            raise UnsupportedSchema(f"{path} declares schema {manifest.get('schema')!r}, expected {SCHEMA!r}")
        if manifest["schema"] != SCHEMA:
            raise UnsupportedSchema(f"{path} is {manifest['schema']}, this version writes and reads {SCHEMA}")
        return manifest, {n: zf.read(n) for n in names if n != MANIFEST}


def _check_integrity(manifest: Mapping[str, Any], members: Mapping[str, bytes]) -> None:
    listed = manifest.get("integrity") or {}
    problems: list[str] = []
    for member, info in listed.items():
        if member not in members:
            problems.append(f"{member}: listed in the manifest but not in the file")
            continue
        got = _sha256(members[member])
        if info.get("sha256") and got != info["sha256"]:
            problems.append(f"{member}: sha256 {got[:12]}… does not match the manifest ({info['sha256'][:12]}…)")
    for member in members:
        if member not in listed:
            problems.append(f"{member}: in the file but not listed in the manifest")
    if problems:
        raise IntegrityError("the artifact does not match its own manifest: " + "; ".join(problems))


def load(path: str | os.PathLike[str], *, verify: bool = True, strict: bool = True) -> Artifact:
    """Read a ``.ppx`` back into an :class:`Artifact`.

    ``verify`` checks every member against the sha256 in the manifest (``IntegrityError`` when one changed);
    ``strict`` refuses an artifact whose provenance block is absent or incomplete (``MissingProvenance``), which is
    what keeps an anonymous number from travelling as a result."""
    p = Path(path)
    manifest, members = _read_members(p)
    if verify:
        _check_integrity(manifest, members)
    if strict:
        missing = _missing_provenance(manifest.get("provenance"))
        if missing:
            raise MissingProvenance(f"{p} does not say where it came from: missing {missing}")
    arrays_block = manifest.get("arrays") or {}
    arrays: dict[str, np.ndarray] = {}
    for name, info in arrays_block.items():
        member = info.get("member") or f"{ARRAYS}{name}.npy"
        if member in members:
            try:
                arrays[name] = np.load(io.BytesIO(members[member]), allow_pickle=False)
            except Exception as exc:            # noqa: BLE001 - a member that is not an array is an integrity problem
                raise IntegrityError(f"{member} is not a readable .npy member: {exc}") from exc
    model_state = None
    if manifest.get("model"):
        import torch

        model_state = torch.load(io.BytesIO(members[manifest["model"]["member"]]), weights_only=True)
        model_state = model_state if isinstance(model_state, Mapping) else model_state.state_dict()
    art = Artifact(
        meta={k: manifest[k] for k in ("id", "title", "description", "created_at", "tags", "references") if k in manifest},
        problem=manifest.get("problem") or {}, units=manifest.get("units") or {},
        solver=manifest.get("solver") or {}, config=manifest.get("config") or {},
        metrics=manifest.get("metrics") or [], validation=manifest.get("validation") or [],
        datasets=manifest.get("datasets") or [], arrays=arrays,
        model_build=(manifest.get("model") or {}).get("build"), model_state=model_state,
        reproductions=manifest.get("reproductions") or [],
        environment=manifest.get("environment") or {}, provenance=manifest.get("provenance") or {},
    )
    return art


def verify(path: str | os.PathLike[str], *, reproduce: bool = True, rtol: float | None = None,
           atol: float | None = None, predict_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> Report:
    """Everything that can be checked about a file without trusting it: the manifest, the members, the provenance,
    the units and, when the artifact carries a model (or ``predict_fn`` is given), a reproduction."""
    p = Path(path)
    checks: list[dict[str, Any]] = []
    try:
        manifest, members = _read_members(p)
    except ArtifactError as exc:
        return Report([{"name": "container", "passed": False, "detail": str(exc)}])
    checks.append({"name": "container", "passed": True, "detail": f"{p.name}: {manifest.get('schema')}"})
    try:
        _check_integrity(manifest, members)
        checks.append({"name": "integrity", "passed": True, "detail": f"{len(manifest.get('integrity') or {})} members match"})
    except IntegrityError as exc:
        checks.append({"name": "integrity", "passed": False, "detail": str(exc)})
    missing = _missing_provenance(manifest.get("provenance"))
    git = (manifest.get("provenance") or {}).get("git") or {}
    checks.append({"name": "provenance", "passed": not missing,
                   "detail": (f"git {str(git.get('commit'))[:12]}, dirty={git.get('dirty')}" if not missing
                              else f"missing {missing}")})
    absent = [s for s in REQUIRED_SECTIONS if not manifest.get(s)]
    checks.append({"name": "sections", "passed": not absent,
                   "detail": "problem, units, solver, environment, provenance" if not absent
                             else f"empty or absent: {absent}"})
    measured = (manifest.get("metrics") or []) + (manifest.get("validation") or [])
    checks.append({"name": "measurements", "passed": True if measured else None,
                   "detail": (f"{len(manifest.get('metrics') or [])} metrics, "
                              f"{len(manifest.get('validation') or [])} checks recorded") if measured
                             else "no metric and no validation check recorded: the artifact carries a result, not evidence"})
    fields = list((manifest.get("problem") or {}).get("fields") or [])
    units = manifest.get("units") or {}
    without = [f for f in fields if not units.get(f)]
    checks.append({"name": "units", "passed": not without,
                   "detail": "every field has a unit" if not without else f"no unit for {without}"})
    if reproduce:
        try:                                    # integrity was already checked above; a broken member is reported, not raised
            art = load(p, verify=False, strict=False)
            checks.extend(art.reproduce(predict_fn, rtol=rtol, atol=atol).checks
                          or [{"name": "reproduction", "passed": None, "detail": "no reproduction recorded"}])
        except ArtifactError as exc:
            checks.append({"name": "reproduction", "passed": False, "detail": str(exc)})
    return Report(checks)
