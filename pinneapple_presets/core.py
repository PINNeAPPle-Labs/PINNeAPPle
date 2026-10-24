"""Industrial preset: an engineering problem already set up and solved with PINNeAPPle.

A preset holds its data sheet (sector, question, inputs in SI units with realistic defaults and the source of each
value, outputs, validity domain), the pipeline that solves it (``solve``), the physics checks of every result
(``checks``), the engineering decision (``decide``), the reference cases it is validated against (``references``)
and, where the case has one, a 3-D scene of the result (``scene``) and a surrogate over the parameters an engineer
sweeps (``surrogate_spec``).

    p = load("water_hammer_valve_closure")
    r = p.run(t_close=4.0)                 # solver; fidelity="surrogate" uses the trained surrogate
    r.outputs, r.decision, r.trust.decisao()
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.trust import Check, TrustReport

__all__ = ["Param", "Output", "Reference", "SurrogateSpec", "Result", "Preset", "environment", "check"]


@dataclass(frozen=True)
class Param:
    """One input. ``lo``..``hi`` is the validity domain of the preset (solver and correlations), not a soft hint:
    a run outside it is flagged and the applicability check contradicts."""
    name: str
    unit: str
    default: Any
    source: str
    lo: float | None = None
    hi: float | None = None
    help: str = ""
    choices: Sequence[Any] | None = None

    def inside(self, v) -> bool:
        if self.choices is not None:
            return v in self.choices
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return True
        return (self.lo is None or v >= self.lo) and (self.hi is None or v <= self.hi)


@dataclass(frozen=True)
class Output:
    name: str
    unit: str
    help: str = ""


@dataclass(frozen=True)
class Reference:
    """A citable reference value for one output at given parameters. ``kind``: "analytic", "standard",
    "benchmark", "experiment" or "independent_solver"; ``tol`` is the relative error accepted for a pass."""
    name: str
    params: dict[str, Any]
    output: str
    value: float
    unit: str
    source: str
    kind: str
    tol: float


@dataclass(frozen=True)
class SurrogateSpec:
    """Which inputs a surrogate spans (each over its Param domain unless ``bounds`` narrows it), which outputs it
    predicts, and the DOE size (Latin hypercube, solver runs) with its hold-out share."""
    inputs: Sequence[str]
    outputs: Sequence[str]
    n_train: int = 60
    n_test: int = 20
    bounds: dict[str, tuple] = field(default_factory=dict)
    log: Sequence[str] = ()              # inputs sampled and learnt in log space
    log_outputs: Sequence[str] = ()      # outputs learnt as log (positive quantities spanning decades)


def check(name: str, ok: bool | None, value=None, limit=None, source: str = "", reason: str = "",
          critical: bool = False) -> Check:
    """A trust-report check from a pass/fail (None = not run)."""
    status = "not_run" if ok is None else ("supports" if ok else "contradicts")
    return Check(name, status, critical=critical, value=_num(value), limit=_num(limit), source=source, reason=reason)


def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def environment() -> dict:
    """Versions and hardware of this run (recorded with every result)."""
    import importlib.metadata as md
    vers = {"python": sys.version.split()[0]}
    for pkg in ("pinneapple", "numpy", "scipy", "torch"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"versions": vers, "hardware": {"cpu": cpu, "cores": os.cpu_count(), "os": platform.platform()}}


@dataclass
class Result:
    """Outputs of one run with its checks, decision and provenance."""
    preset: str
    params: dict[str, Any]
    outputs: dict[str, Any]
    trust: TrustReport
    decision: dict[str, Any]
    fidelity: str = "solver"
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)       # arrays for plots / 3-D (not in the JSON summary)

    def __getitem__(self, k):
        return self.outputs[k]

    def to_dict(self) -> dict:
        from pinneapple_core.jsonable import to_jsonable
        return to_jsonable({"preset": self.preset, "fidelity": self.fidelity, "params": self.params,
                            "outputs": self.outputs, "decision": self.decision,
                            "trust": {"decision": self.trust.decisao, "score": self.trust.score,
                                      "coverage": self.trust.cobertura, "checks": [asdict(c) for c in
                                                                                     self.trust.checks]},
                            "warnings": self.warnings, "meta": self.meta})

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    def summary(self) -> str:
        lines = [f"{self.preset} ({self.fidelity}): {self.decision.get('verdict', '?')} | trust "
                 f"{self.trust.decisao}"]
        for k, v in self.outputs.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k} = {v:.6g}")
        lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)

    __str__ = summary


class Preset:
    """Base class; a preset sets the class attributes and implements :meth:`solve`, :meth:`checks`, :meth:`decide`
    and :meth:`references` (and optionally :meth:`scene`, :meth:`convergence`, :meth:`figures`)."""
    name: str = ""
    title: str = ""
    sector: str = ""
    problem: str = ""
    question: str = ""
    params: Sequence[Param] = ()
    outputs: Sequence[Output] = ()
    validity: str = ""
    pipeline: Sequence[str] = ()                  # geometry -> model -> solver -> post -> decision, as calls
    library: Sequence[str] = ()                   # library entry points used
    surrogate_spec: SurrogateSpec | None = None
    has_3d: bool = False
    has_optimization: bool = False
    has_inversion: bool = False
    has_variants: bool = False
    missing: dict[str, str] = {}                  # capability -> why the preset does not have it
    limitations: Sequence[str] = ()
    reason_case: str | None = None             # pinneapple_reason case that explains the design, if any

    # ----------------------------------------------------------------- to implement
    def solve(self, p: dict) -> dict:
        raise NotImplementedError

    def checks(self, p: dict, out: dict) -> list[Check]:
        return []

    def decide(self, p: dict, out: dict) -> dict:
        return {"verdict": "n/a", "reasons": []}

    def references(self) -> list[Reference]:
        return []

    def complete(self, p: dict, out: dict) -> dict:
        """Derive the decision's outputs from the surrogate's (cheap closed-form steps only)."""
        return out

    def convergence(self) -> dict | None:
        return None

    def scene(self, result: Result):                # -> pinneapple_tools.visualization.studio.Scene
        return None

    def figures(self, result: Result, out_dir: str) -> list[str]:
        return []

    def optimize(self, **kw) -> dict | None:
        """Best design for the preset's engineering objective, confirmed with the solver."""
        return None

    def invert(self, **kw) -> dict | None:
        """Recover unknown parameters from measurements; the default study uses synthetic measurements made by the
        solver with known noise and reports the recovered value, its error and its uncertainty interval."""
        return None

    def variants(self) -> dict[str, dict]:
        """Named design variants (parameter overrides) run side by side by :func:`report.compare_variants`."""
        return {}

    # ----------------------------------------------------------------- generic machinery
    def defaults(self) -> dict:
        return {q.name: q.default for q in self.params}

    def param(self, name: str) -> Param:
        for q in self.params:
            if q.name == name:
                return q
        raise KeyError(f"{self.name}: no parameter {name!r}; parameters: {[q.name for q in self.params]}")

    def resolve(self, **kw) -> dict:
        unknown = set(kw) - {q.name for q in self.params}
        if unknown:
            raise KeyError(f"{self.name}: unknown parameters {sorted(unknown)}; known: {[q.name for q in self.params]}")
        p = self.defaults()
        p.update(kw)
        return p

    def outside(self, p: dict) -> list[str]:
        return [f"{q.name}={p[q.name]} outside the validity domain [{q.lo}, {q.hi}] {q.unit}"
                if q.choices is None else f"{q.name}={p[q.name]!r} not one of {list(q.choices)}"
                for q in self.params if not q.inside(p[q.name])]

    def run(self, fidelity: str = "solver", **kw) -> Result:
        """Solve with the defaults overridden by ``kw``. ``fidelity="surrogate"`` predicts the surrogate outputs
        (with their standard deviation) and runs the checks that need no field; outside the surrogate's training
        box the result is flagged, never silently extrapolated."""
        p = self.resolve(**kw)
        warn = self.outside(p)
        t0 = time.perf_counter()
        if fidelity == "solver":
            out = self.solve(dict(p))
            fields = out.pop("_fields", {})
            chks = list(self.checks(p, out))
        elif fidelity == "surrogate":
            from .surrogate import load_surrogate
            sur = load_surrogate(self.name)
            if sur is None:
                raise FileNotFoundError(f"{self.name}: no trained surrogate (python -m pinneapple_presets surrogate "
                                        f"{self.name})")
            out, ood = sur.predict_one(p)
            out = self.complete(p, out)
            fields = {}
            warn += ood
            chks = [check("surrogate inside its training box", not ood, source="surrogate DOE bounds",
                          reason="; ".join(ood)),
                    check("surrogate hold-out error", sur.meta["holdout"]["max_rel_err"] <= sur.meta.get("tol", 0.05),
                          value=sur.meta["holdout"]["max_rel_err"], limit=sur.meta.get("tol", 0.05),
                          source=f"{sur.meta['n_test']} held-out solver runs")]
        else:
            raise ValueError("fidelity is 'solver' or 'surrogate'")
        wall = time.perf_counter() - t0
        chks.append(check("inside the validity domain", not self.outside(p), source="preset data sheet",
                          reason="; ".join(self.outside(p))))
        tr = TrustReport(chks, model=f"{self.name}:{fidelity}")
        try:
            dec = self.decide(p, out)
        except KeyError as e:
            dec = {"verdict": "n/a", "reasons": [f"the decision needs {e} (run the solver)"]}
        meta = {"wall_time_s": wall, **environment(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for w in warn:
            warnings.warn(f"{self.name}: {w}", stacklevel=2)
        return Result(self.name, p, out, tr, dec, fidelity, warn, meta, fields)

    def validate(self) -> dict:
        """Run every reference case and measure the error; plus the convergence study when the preset has one."""
        rows = []
        for ref in self.references():
            r = self.run(**ref.params)
            v = float(r.outputs[ref.output])
            err = abs(v - ref.value) / abs(ref.value) if ref.value else abs(v)
            rows.append({"case": ref.name, "output": ref.output, "unit": ref.unit, "computed": v,
                         "reference": ref.value, "rel_error": err, "tol": ref.tol, "pass": bool(err <= ref.tol),
                         "kind": ref.kind, "source": ref.source, "params": ref.params})
        conv = self.convergence()
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and (conv is None or conv.get("pass", True))}

    def card(self) -> dict:
        """The data sheet (JSON-ready)."""
        return {
            "name": self.name, "title": self.title, "sector": self.sector, "problem": self.problem,
            "question": self.question,
            "inputs": [{"name": q.name, "unit": q.unit, "default": q.default, "source": q.source, "lo": q.lo,
                        "hi": q.hi, "help": q.help, **({"choices": list(q.choices)} if q.choices else {})}
                       for q in self.params],
            "outputs": [asdict(o) for o in self.outputs], "validity": self.validity, "pipeline": list(self.pipeline),
            "library": list(self.library), "limitations": list(self.limitations),
            "capabilities": {"3d": self.has_3d, "surrogate": self.surrogate_spec is not None,
                             "inversion": self.has_inversion, "optimization": self.has_optimization,
                             "variants": self.has_variants, "reason": self.reason_case is not None},
            "missing": dict(self.missing), "reason_case": self.reason_case,
            "references": [asdict(r) for r in self.references()],
            "usage": {"python": f'import pinneapple as pp\nr = pp.presets.load("{self.name}").run()\nprint(r)',
                      "cli": f"python -m pinneapple_presets run {self.name}",
                      "flow": {"type": "preset.run", "params": {"preset": self.name}}},
        }


def lhs(n: int, bounds: Sequence[tuple], seed: int = 0, log: Sequence[bool] = ()) -> np.ndarray:
    """Latin hypercube sample of ``n`` points in the box (``log[i]``: that axis in log space)."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n)
    X = np.empty_like(u)
    for i, (a, b) in enumerate(bounds):
        if i < len(log) and log[i]:
            X[:, i] = np.exp(math.log(a) + u[:, i] * (math.log(b) - math.log(a)))
        else:
            X[:, i] = a + u[:, i] * (b - a)
    return X


def call_quiet(fn: Callable, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)
