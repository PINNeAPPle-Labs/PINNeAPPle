"""The common shape of a science experiment: what it declares, what every run returns.

An experiment declares

* ``area``: an area id of the atlas (``"physics.electromagnetism"``), and the ``topic`` inside it;
* ``level``: ``"academic"`` (didactic, checked against a textbook value) or ``"research"`` (configurable, with a
  convergence check and an uncertainty);
* ``kind``: ``"theory"`` (a numerical or symbolic derivation checked against a closed form, a published value or a
  scaling law) or ``"practice"`` (a virtual bench: instruments with noise, a fit with its uncertainty);
* ``dims``: the dimensions it runs in, from ``"0D"`` (lumped elements: circuits, no spatial coordinate), ``"1D"``,
  ``"2D"``, ``"3D"`` and ``"3D+t"`` (three space dimensions and time, the fourth dimension).

``Experiment.run(dim=..., **params)`` returns a :class:`Result`: the measured values (each with its unit and how it
was measured), the comparisons with their references (computed, reference, relative error, tolerance, source),
the physics checks, the trust report (:class:`pinneapple_analysis.trust.TrustReport`), the figures and the data
behind them. ``Result.save(folder)`` writes ``result.json`` and the figures; :func:`pinneapple_science.report.write`
adds the Lab manifest ``experiment.json`` and the seal.

Nothing in a result is typed in: every value is computed by the run, every reference carries its source.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LEVELS = ("academic", "research")
KINDS = ("theory", "practice")
DIMS = ("0D", "1D", "2D", "3D", "3D+t")
REFERENCE_KINDS = ("closed_form", "published", "convergence", "limit", "conservation", "codata", "ground_truth")

__all__ = ["LEVELS", "KINDS", "DIMS", "Reference", "Comparison", "Result", "Experiment", "jsonable"]


def jsonable(v: Any) -> Any:
    """Plain JSON types (numpy scalars and arrays to Python numbers and lists; NaN and inf to None)."""
    if isinstance(v, Mapping):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    return v


@dataclass(frozen=True)
class Reference:
    """A reference value and where it comes from. ``kind`` is one of ``closed_form`` (a formula, cited),
    ``published`` (a tabulated value from a paper or book), ``codata`` (a fundamental constant), ``limit`` (an
    asymptotic limit), ``conservation`` (a quantity that must be conserved), ``convergence`` (the extrapolated
    value of a refinement study) or ``ground_truth`` (the value a virtual bench was built with, which the
    measurement does not see)."""

    value: float
    source: str
    kind: str = "closed_form"
    unit: str = ""

    def __post_init__(self):
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(f"reference kind {self.kind!r} not in {REFERENCE_KINDS}")
        if not self.source:
            raise ValueError("a reference needs its source")


@dataclass
class Comparison:
    quantity: str
    computed: float
    reference: float
    rel_error: float
    tol: float
    passed: bool
    kind: str
    source: str
    unit: str = ""
    how: str = ""
    abs_error: float | None = None
    atol: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__)


@dataclass
class Result:
    """What a run returns. Build it through the recording methods (``value``, ``compare``, ``check``,
    ``figure``, ``data``); ``trust`` is assembled from the comparisons and checks when the run ends."""

    experiment: str
    title: str
    area: str
    topic: str
    level: str
    kind: str
    dim: str
    params: dict[str, Any]
    values: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    figures: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    movies: dict[str, tuple[list, float]] = field(default_factory=dict)
    viewers: dict[str, Callable[[str], str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    trust: Any = None

    # ------------------------------------------------------------------ recording
    def value(self, name: str, value: float, unit: str = "", how: str = "", uncertainty: float | None = None) -> float:
        """A measured value (finite), its unit, how it was obtained and, for a measurement, its standard
        uncertainty."""
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name}: a measured value must be finite, got {value!r}")
        rec = {"value": v, "unit": unit, "how": how or "computed by the run"}
        if uncertainty is not None:
            rec["uncertainty"] = float(uncertainty)
        self.values[name] = rec
        return v

    def compare(self, quantity: str, computed: float, reference: Reference | float, *, tol: float,
                source: str = "", kind: str = "closed_form", unit: str = "", how: str = "",
                atol: float = 0.0) -> Comparison:
        """Compare a computed value with its reference: relative error ``|c - r| / |r|`` (absolute when the
        reference is 0) against ``tol`` (``atol`` adds an absolute floor)."""
        if isinstance(reference, Reference):
            ref, source, kind, unit = reference.value, reference.source, reference.kind, unit or reference.unit
        else:
            ref = float(reference)
            if not source:
                raise ValueError(f"{quantity}: a reference needs its source")
            if kind not in REFERENCE_KINDS:
                raise ValueError(f"reference kind {kind!r} not in {REFERENCE_KINDS}")
        c = float(computed)
        err_abs = abs(c - ref)
        rel = err_abs / abs(ref) if ref != 0 else err_abs
        passed = bool(math.isfinite(c) and (rel <= tol or err_abs <= atol))
        cmp = Comparison(quantity, c, float(ref), float(rel), float(tol), passed, kind, source, unit, how,
                         float(err_abs), float(atol))
        self.comparisons.append(cmp)
        return cmp

    def check(self, name: str, passed: bool | None, *, value: float | None = None, limit: float | None = None,
              critical: bool = False, detail: str = "", source: str = "physics") -> bool | None:
        """A physics check (conservation, a residual, a symmetry, a consistency); ``None`` = not run."""
        self.checks.append(jsonable({"name": name, "passed": None if passed is None else bool(passed),
                                     "value": value, "limit": limit, "critical": critical, "detail": detail,
                                     "source": source}))
        return passed

    def figure(self, name: str, fig) -> None:
        self.figures[name] = fig

    def movie(self, name: str, frames: Sequence[np.ndarray], fps: float = 12.0) -> None:
        """An animation (RGB frames), written as MP4 and GIF by :meth:`save`; for 3D+t results it is the
        fourth dimension: the 3-D field in time."""
        self.movies[name] = (list(frames), float(fps))

    def viewer(self, name: str, write: Callable[[str], str]) -> None:
        """An interactive page written by ``write(path)`` when the result is saved (``<name>.html``)."""
        self.viewers[name] = write

    def data(self, name: str, **arrays) -> None:
        """Arrays behind a figure (saved in ``result.json`` when small, in ``<name>.npz`` otherwise)."""
        self.arrays[name] = arrays

    # ------------------------------------------------------------------ verdict
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.comparisons) and all(c["passed"] is not False for c in self.checks)

    @property
    def decision(self) -> str:
        return self.trust.decisao if self.trust is not None else "REVIEW"

    def build_trust(self) -> Any:
        from pinneapple_analysis.trust.trust_report import Check, TrustReport

        out = []
        for c in self.comparisons:
            out.append(Check(f"{c.quantity} vs {c.kind.replace('_', ' ')}", "supports" if c.passed else "contradicts",
                             critical=True, value=c.rel_error, limit=c.tol, source=c.source,
                             reason=f"computed {c.computed:.6g}, reference {c.reference:.6g} {c.unit}".strip()))
        for k in self.checks:
            st = "not_run" if k["passed"] is None else ("supports" if k["passed"] else "contradicts")
            out.append(Check(k["name"], st, critical=k["critical"], value=k["value"], limit=k["limit"],
                             source=k["source"], reason=k["detail"]))
        self.trust = TrustReport(out, model=f"{self.experiment} [{self.dim}]")
        return self.trust

    def table(self) -> list[dict[str, Any]]:
        """One row per comparison: experiment, dimension, quantity, computed, reference, error, tolerance."""
        return [{"experiment": self.experiment, "dim": self.dim, "quantity": c.quantity, "computed": c.computed,
                 "reference": c.reference, "unit": c.unit, "rel_error": c.rel_error, "tol": c.tol,
                 "passed": c.passed, "kind": c.kind, "source": c.source} for c in self.comparisons]

    def __repr__(self) -> str:
        lines = [f"{self.title} [{self.dim}, {self.level}, {self.kind}]: {'PASS' if self.passed else 'FAIL'}, "
                 f"trust {self.decision}"]
        for c in self.comparisons:
            err = (f"abs. error {c.abs_error:.2e}, tol {c.atol:.2g}" if c.tol == 0 else
                   f"rel. error {c.rel_error:.2e}, tol {c.tol:.2g}")
            lines.append(f"  {'ok ' if c.passed else 'BAD'} {c.quantity}: {c.computed:.6g} vs {c.reference:.6g} "
                         f"{c.unit} ({err}; {c.kind})")
        for k in self.checks:
            mark = "n/a" if k["passed"] is None else ("ok " if k["passed"] else "BAD")
            val = "" if k["value"] is None else f": {k['value']:.3g}" + ("" if k["limit"] is None else f" <= {k['limit']:.3g}")
            lines.append(f"  {mark} {k['name']}{val}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        small, big = {}, {}
        for name, arrs in self.arrays.items():
            size = sum(np.size(a) for a in arrs.values())
            (small if size <= 4000 else big)[name] = arrs
        return jsonable({
            "schema": "pinneapple.science.result/1", "experiment": self.experiment, "title": self.title,
            "area": self.area, "topic": self.topic, "level": self.level, "kind": self.kind, "dim": self.dim,
            "params": self.params, "values": self.values, "comparisons": [c.to_dict() for c in self.comparisons],
            "checks": self.checks, "passed": self.passed,
            "trust": None if self.trust is None else {"decision": self.trust.decisao, "score": self.trust.score,
                                                      "coverage": self.trust.cobertura,
                                                      "checks": [c.__dict__ for c in self.trust.checks]},
            "data": small, "data_files": sorted(f"{n}.npz" for n in big), "notes": self.notes,
            "figures": sorted(f"{n}.png" for n in self.figures),
            "movies": sorted(f"{n}.{e}" for n in self.movies for e in ("mp4", "gif")
                             if not any(x.startswith(f"{n}.{e} not written") for x in self.notes)),
            "viewers": sorted(f"{n}.html" for n in self.viewers),
            "wall_time_s": self.wall_time_s,
        })

    def save(self, folder: str, dpi: int = 110) -> dict[str, str]:
        """Write ``result.json``, the figures (PNG) and large arrays (``.npz``) into ``folder``; returns the paths
        by name."""
        os.makedirs(folder, exist_ok=True)
        paths = {}
        d = self.to_dict()
        for name, arrs in self.arrays.items():
            if f"{name}.npz" in d["data_files"]:
                p = os.path.join(folder, f"{name}.npz")
                np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrs.items()})
                paths[name] = p
        for name, fig in self.figures.items():
            p = os.path.join(folder, f"{name}.png")
            fig.savefig(p, dpi=dpi)
            paths[name] = p
        for name, (frames, fps) in self.movies.items():
            from pinneapple_design.bricks.visuals import write_mp4
            from pinneapple_tools.visualization.dashboard.static import write_gif

            try:
                paths[name + "_mp4"] = write_mp4(os.path.join(folder, f"{name}.mp4"), frames, fps)
            except RuntimeError as e:       # no H.264 encoder here: the GIF still carries the animation
                self.notes.append(f"{name}.mp4 not written: {e}")
            paths[name + "_gif"] = write_gif(os.path.join(folder, f"{name}.gif"), frames[::2], fps / 2)
        for name, write in self.viewers.items():
            paths[name + "_html"] = write(os.path.join(folder, f"{name}.html"))
        d = self.to_dict()                  # again: saving may have added notes (an MP4 it could not write)
        p = os.path.join(folder, "result.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        paths["result"] = p
        return paths


class Experiment:
    """Base class. Subclasses set the class attributes and implement ``_run(res, dim, p)``, which fills the
    :class:`Result` ``res`` for dimension ``dim`` with parameters ``p`` (defaults merged with the caller's)."""

    name: str = ""
    title: str = ""
    area: str = ""
    topic: str = ""
    level: str = "academic"
    kind: str = "theory"
    dims: tuple[str, ...] = ()
    description: str = ""
    references: tuple[str, ...] = ()
    params: dict[str, Any] = {}
    #: smaller parameters for tests and smoke runs (merged over ``params``)
    quick: dict[str, Any] = {}
    #: what :meth:`run` returns; a layer built on this one (``pinneapple_futures``) supplies its own subclass
    result_cls: type[Result] = Result

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not cls.name:
            return
        if cls.level not in LEVELS:
            raise ValueError(f"{cls.name}: level {cls.level!r} not in {LEVELS}")
        if cls.kind not in KINDS:
            raise ValueError(f"{cls.name}: kind {cls.kind!r} not in {KINDS}")
        bad = [d for d in cls.dims if d not in DIMS]
        if not cls.dims or bad:
            raise ValueError(f"{cls.name}: dims {cls.dims!r} must be a non-empty subset of {DIMS}")

    def card(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "area": self.area, "topic": self.topic,
                "level": self.level, "kind": self.kind, "dims": list(self.dims), "description": self.description,
                "references": list(self.references), "params": jsonable(self.params)}

    def run(self, dim: str | None = None, *, quick: bool = False, **params) -> Result:
        dim = dim or self.dims[0]
        if dim not in self.dims:
            raise ValueError(f"{self.name} runs in {self.dims}, not {dim!r}")
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}; known: {sorted(self.params)}")
        p = {**self.params, **(self.quick if quick else {}), **params}
        res = self.result_cls(self.name, self.title, self.area, self.topic, self.level, self.kind, dim, jsonable(p))
        t0 = time.perf_counter()
        self._run(res, dim, p)
        res.wall_time_s = time.perf_counter() - t0
        res.build_trust()
        return res

    def run_all(self, *, quick: bool = False) -> list[Result]:
        return [self.run(d, quick=quick) for d in self.dims]

    def _run(self, res: Result, dim: str, p: dict[str, Any]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def observed_order(errors: Sequence[float], ratio: float = 2.0) -> list[float]:
    """Observed orders ``log(e_i / e_{i+1}) / log(ratio)`` of a refinement ladder."""
    e = [float(x) for x in errors]
    return [math.log(e[i] / e[i + 1]) / math.log(ratio) for i in range(len(e) - 1) if e[i] > 0 and e[i + 1] > 0]


def central_derivative(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)
