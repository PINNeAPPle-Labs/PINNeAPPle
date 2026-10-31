"""The contract every method of :mod:`pinneapple_analysis.dynamics` follows.

A method is described by a :class:`MethodCard` before it runs (its mathematical inputs, its assumptions, its
configuration, what it returns, its role and the conditions under which it fails or does not apply), and returns a
:class:`DynamicsResult` after it runs (the estimate with its uncertainty, the assumptions with whether each was
checked, the numerical diagnostics, the convergence record, whether the method applied, and what is needed to
reproduce the run).

The two are plain dataclasses: NumPy only, JSON-serialisable through :meth:`DynamicsResult.to_dict`. The shared
data contract of the apps (``pinneapple_data.contracts.StudyResult``, which needs pydantic) is produced on request by
:meth:`DynamicsResult.to_study_result`; nothing here imports it at module load.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = ["ROLES", "Assumption", "Diagnostic", "MethodCard", "Reproducibility", "DynamicsResult",
           "fingerprint", "make_rng", "reproducibility", "jsonable"]

Role = Literal["primitive", "trajectory_operator", "differentiable_module", "validation", "experimental_design"]
ROLES: tuple[str, ...] = ("primitive", "trajectory_operator", "differentiable_module", "validation",
                          "experimental_design")
Status = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True)
class Assumption:
    """One hypothesis a method relies on.

    ``checked`` is ``True`` (verified on these inputs, ``evidence`` says how), ``False`` (tested and violated),
    or ``None`` (not testable from the inputs: the caller is responsible for it)."""
    name: str
    statement: str
    checked: bool | None = None
    evidence: str = ""


@dataclass(frozen=True)
class Diagnostic:
    """One numerical diagnostic: a measured ``value``, its ``status`` and what it means."""
    name: str
    value: Any
    status: Status = "info"
    message: str = ""
    threshold: Any = None

    def __post_init__(self):
        if self.status not in ("ok", "warn", "fail", "info"):
            raise ValueError(f"diagnostic {self.name!r}: status must be ok/warn/fail/info, got {self.status!r}")


@dataclass(frozen=True)
class MethodCard:
    """What a method takes, assumes and gives back, written before it runs (items 1-3 and 10 of the contract).

    ``role`` places the method in the toolkit: a ``primitive`` (a function of numbers), a
    ``trajectory_operator`` (acts on trajectories or ensembles), a ``differentiable_module`` (can sit inside a
    training graph), a ``validation`` component (judges another result) or an ``experimental_design`` tool."""
    name: str
    role: Role
    inputs: Mapping[str, str]
    assumptions: Sequence[Assumption]
    config: Mapping[str, str]
    outputs: Mapping[str, str]
    failure_modes: Sequence[str]
    references: Sequence[str] = ()
    backends: Sequence[str] = ("numpy",)

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"method {self.name!r}: role must be one of {ROLES}, got {self.role!r}")
        if not self.inputs or not self.outputs:
            raise ValueError(f"method {self.name!r}: a card needs its inputs and its outputs")
        if not self.failure_modes:
            raise ValueError(f"method {self.name!r}: a card needs at least one failure or applicability condition")

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    def __str__(self) -> str:
        lines = [f"{self.name} [{self.role}]", "inputs:"]
        lines += [f"  {k}: {v}" for k, v in self.inputs.items()]
        lines += ["assumptions:"] + [f"  - {a.name}: {a.statement}" for a in self.assumptions]
        lines += ["config:"] + [f"  {k}: {v}" for k, v in self.config.items()]
        lines += ["outputs:"] + [f"  {k}: {v}" for k, v in self.outputs.items()]
        lines += ["fails or does not apply when:"] + [f"  - {f}" for f in self.failure_modes]
        if self.references:
            lines += ["references:"] + [f"  - {r}" for r in self.references]
        return "\n".join(lines)


@dataclass(frozen=True)
class Reproducibility:
    """What is needed to run the method again and get the same numbers."""
    method: str
    config: Mapping[str, Any]
    seed: int | None
    input_hash: str
    versions: Mapping[str, str]
    created: str
    wall_time_s: float | None = None


@dataclass
class DynamicsResult:
    """The structured result of one method run (items 4-10 of the contract).

    * ``estimate``: the quantity the method computes (a number, an array or a mapping of them).
    * ``uncertainty``: how far the estimate can be trusted, each entry naming its kind, e.g.
      ``{"ci95": (lo, hi), "kind": "Wilson interval on the Monte Carlo proportion"}``. Empty means none was
      computed, not that the estimate is exact.
    * ``assumptions``: each hypothesis with whether it was checked on these inputs.
    * ``diagnostics`` and ``convergence``: the numerical record (condition numbers, residuals, step-size studies).
    * ``applicable``: ``False`` when an applicability condition failed; ``failures`` says which. A result that does
      not apply still carries its diagnostics, so the reason can be inspected.
    * ``limits``: what the result does not show.
    """
    method: str
    estimate: Any
    uncertainty: dict[str, Any] = field(default_factory=dict)
    assumptions: list[Assumption] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    convergence: dict[str, Any] = field(default_factory=dict)
    applicable: bool = True
    failures: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    reproducibility: Reproducibility | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.failures and self.applicable:
            raise ValueError("a result with failures must have applicable=False")

    @property
    def ok(self) -> bool:
        """Applicable, no failed diagnostic and no assumption found violated."""
        return (self.applicable and all(d.status != "fail" for d in self.diagnostics)
                and all(a.checked is not False for a in self.assumptions))

    @property
    def unchecked_assumptions(self) -> list[Assumption]:
        """The assumptions the inputs could not test: the caller's responsibility."""
        return [a for a in self.assumptions if a.checked is None]

    @property
    def violated_assumptions(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.checked is False]

    def diagnostic(self, name: str) -> Diagnostic:
        for d in self.diagnostics:
            if d.name == name:
                return d
        raise KeyError(f"no diagnostic {name!r}; have {[d.name for d in self.diagnostics]}")

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable dict (arrays become lists, non-finite floats become ``None``)."""
        return jsonable({
            "method": self.method, "estimate": self.estimate, "uncertainty": self.uncertainty,
            "assumptions": [asdict(a) for a in self.assumptions], "config": self.config,
            "diagnostics": [asdict(d) for d in self.diagnostics], "convergence": self.convergence,
            "applicable": self.applicable, "failures": self.failures, "limits": self.limits,
            "reproducibility": asdict(self.reproducibility) if self.reproducibility else None,
            "details": self.details, "ok": self.ok,
        })

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_study_result(self, *, study_type: str = "other", app: str = "dynamics", title: str = "",
                        study_id: str | None = None):
        """The result as a ``pinneapple_data.contracts.StudyResult`` (requires pydantic).

        Diagnostics become checks (``ok`` -> PASS, ``warn`` -> WARN, ``fail`` -> FAIL, ``info`` -> N/A), violated
        assumptions become FAIL checks, unchecked assumptions and ``limits`` become the study's limits. Scalar
        entries of ``estimate`` and ``uncertainty`` become metrics."""
        from pinneapple_data import contracts as c

        status_map = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "info": "N/A"}
        checks = []
        for d in self.diagnostics:
            value = _scalar_or_text(d.value)
            checks.append(c.CheckRecord(name=d.name, status=status_map[d.status], value=value,
                                        threshold=_scalar_or_text(d.threshold), detail=d.message,
                                        evidence="" if value is not None else f"{self.method} diagnostics"))
        for a in self.assumptions:
            if a.checked is not None:
                checks.append(c.CheckRecord(name=f"assumption: {a.name}", status="PASS" if a.checked else "FAIL",
                                            evidence=a.evidence or f"{self.method} assumption check",
                                            detail=a.statement))
        limits = list(self.limits) + [f"not checked (assumed): {a.name}: {a.statement}"
                                      for a in self.unchecked_assumptions]
        if not self.applicable:
            limits = [f"not applicable: {f}" for f in self.failures] + limits
        metrics: dict[str, float | int | str | None] = {}
        for prefix, src in (("estimate", self.estimate), ("uncertainty", self.uncertainty)):
            for k, v in _flat_scalars(prefix, src):
                metrics[k] = v
        rep = self.reproducibility
        prov = c.Provenance(app=app, study_id=study_id or (rep.input_hash[:16] if rep else ""),
                            source=self.method, library_version=(rep.versions.get("pinneapple", "") if rep else ""),
                            notes=f"seed={rep.seed}" if rep else "")
        return c.StudyResult(id=study_id or f"{self.method}-{rep.input_hash[:12] if rep else 'run'}",
                             study_type=study_type, app=app, title=title or self.method,
                             status="done" if self.applicable else "failed", metrics=metrics, checks=checks,
                             limits=limits, provenance=prov)

    def summary(self) -> str:
        head = f"{self.method}: {'ok' if self.ok else ('NOT APPLICABLE' if not self.applicable else 'REVIEW')}"
        lines = [head, f"  estimate: {_short(self.estimate)}"]
        if self.uncertainty:
            lines.append(f"  uncertainty: {_short(self.uncertainty)}")
        lines += [f"  not applicable: {f}" for f in self.failures]
        lines += [f"  [{d.status}] {d.name} = {_short(d.value)}  {d.message}".rstrip()
                  for d in self.diagnostics if d.status != "info"]
        lines += [f"  assumption violated: {a.name} ({a.evidence})" for a in self.violated_assumptions]
        lines += [f"  assumed, not checked: {a.name}" for a in self.unchecked_assumptions]
        return "\n".join(lines)

    __str__ = summary


# --------------------------------------------------------------------------- reproducibility helpers
def fingerprint(*objects: Any) -> str:
    """SHA-256 of the inputs: arrays by dtype, shape and bytes; everything else by its canonical JSON."""
    h = hashlib.sha256()
    for obj in objects:
        _feed(h, obj)
    return h.hexdigest()


def _feed(h, obj) -> None:
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(f"nd:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    elif isinstance(obj, Mapping):
        h.update(b"map{")
        for k in sorted(obj, key=str):
            h.update(str(k).encode() + b"=")
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"seq[")
        for v in obj:
            _feed(h, v)
        h.update(b"]")
    elif hasattr(obj, "fingerprint_payload"):
        _feed(h, obj.fingerprint_payload())
    else:
        h.update(json.dumps(jsonable(obj), sort_keys=True).encode())


def make_rng(seed: int | None) -> tuple[np.random.Generator, int]:
    """A NumPy generator and the seed that reproduces it. ``seed=None`` draws a fresh seed from OS entropy and
    returns it, so even an unseeded run records how to repeat it."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    seed = int(seed)
    return np.random.default_rng(seed), seed


def _versions(extra: Iterable[str] = ()) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "numpy": np.__version__}
    for pkg in ("scipy", "pinneapple", *extra):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def reproducibility(method: str, config: Mapping[str, Any], inputs: Sequence[Any], *, seed: int | None = None,
                    wall_time_s: float | None = None, packages: Iterable[str] = ()) -> Reproducibility:
    """The :class:`Reproducibility` record of a run: config, seed, input hash and library versions."""
    return Reproducibility(method=method, config=jsonable(dict(config)), seed=seed,
                           input_hash=fingerprint(method, dict(config), list(inputs)), versions=_versions(packages),
                           created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_time_s=wall_time_s)


# --------------------------------------------------------------------------- JSON helpers
def jsonable(obj: Any) -> Any:
    """``obj`` with arrays as lists, NumPy scalars as Python scalars, tuples as lists and NaN/inf as ``None``."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, complex):
        return {"re": jsonable(obj.real), "im": jsonable(obj.imag)}
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return jsonable(asdict(obj))
    return str(obj)


def _scalar_or_text(v: Any) -> float | str | None:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else str(f)
    if isinstance(v, str):
        return v
    return _short(v)


def _flat_scalars(prefix: str, obj: Any):
    if isinstance(obj, (bool, np.bool_)):
        yield prefix, str(bool(obj))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        f = float(obj)
        yield prefix, f if math.isfinite(f) else None
    elif isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _flat_scalars(f"{prefix}.{k}", v)


def _short(v: Any, n: int = 120) -> str:
    if isinstance(v, np.ndarray):
        s = f"array{v.shape}" if v.size > 6 else np.array2string(v, precision=4)
    elif isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(jsonable(v))
    return s if len(s) <= n else s[: n - 3] + "..."
