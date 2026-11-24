"""Start a solver run from a surrogate's predicted field, and measure what it saved (#494).

A surrogate that predicts the flow is worth more than its fields: it is a **starting point**. Keeping the solver as
the thing that produces the answer, and using the prediction only to place the first iterate near the solution, is the
one way to get the value of a surrogate in production without trusting it — the converged answer is still the
solver's, and the measurement of the saving (iterations and wall time) is also a measurement of how good the
surrogate was. Obiols-Sales et al. (CFDNet, ICS 2020) report 1.9-7.4x for steady RANS this way.

What is here
------------
:func:`write_fields` maps a predicted field onto a case's time-zero fields **without touching the boundary
conditions**: the ``internalField`` is replaced and the ``boundaryField`` block is left exactly as it was, because the
boundary conditions are the problem statement and a surrogate has no business rewriting them.

:class:`OperatorSurrogate` is a parametric surrogate over one number — a proper orthogonal decomposition of a few
solved cases, with the mode coefficients interpolated in the parameter. It is the smallest thing that deserves the
name: it is fitted from runs (not from a formula), it predicts a whole field at a parameter it has not seen, and
:meth:`OperatorSurrogate.error` says how far its prediction is from a solver run at that parameter.

:func:`benchmark` is the experiment the issue asks for: at a parameter held out of the fit, run the case cold and run
it warm from the prediction, and report the iterations, the wall time, the speedup and the difference between the two
converged answers — which must be nothing, because they are the same equations and the same tolerance.

Where it stops
--------------
Warm starting a *steady* solve is the easy case: the iterates are a path to a fixed point and a better start is a
shorter path. A transient case has no such fixed point, and the saving depends on how fast the start-up transient
decays; the same machinery applies (write the field at ``t0`` and integrate from there) but the measurement has to be
made per case, and none is claimed here.
"""
from __future__ import annotations

import os
import re as _re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import cavity, foam

__all__ = ["write_fields", "OperatorSurrogate", "benchmark", "WarmStartResult", "internal_field", "set_internal_field"]

_COMPONENTS = {"U": 3, "p": 1}


def _block(text: str, key: str) -> tuple[int, int]:
    """``(start, end)`` of the ``key`` entry of a field dictionary, terminator included.

    OpenFOAM writes a field's ``internalField`` either as a one-line entry ending in ``;`` (``internalField uniform
    (0 0 0);``) or as a dictionary of braces, so the span ends at the first ``;`` after the keyword — and a
    ``nonuniform List<...>`` of values contains no ``;`` of its own, which is why that is enough.
    """
    m = _re.search(rf"^[ \t]*{key}\b", text, flags=_re.M)
    if not m:
        raise KeyError(f"no {key!r} entry")
    end = text.find(";", m.end())
    if end < 0:
        raise ValueError(f"the {key!r} entry has no terminator")
    return m.start(), end + 1


def set_internal_field(case: str, name: str, values: np.ndarray) -> dict[str, Any]:
    """Replace only the ``internalField`` of ``case/0/<name>``; the ``boundaryField`` is left byte for byte.

    ``values`` is ``(n_cells,)`` for a scalar and ``(n_cells, 3)`` for a vector. Returns what it wrote — the field,
    the number of cells and the range of the values — so a run records the start it was given.
    """
    path = os.path.join(case, "0", name)
    with open(path) as fh:
        text = fh.read()
    v = np.asarray(values, dtype=float)
    if name in _COMPONENTS and _COMPONENTS[name] > 1:
        if v.ndim != 2 or v.shape[1] != _COMPONENTS[name]:
            raise ValueError(f"{name}: expected (n_cells, {_COMPONENTS[name]}), got {v.shape}")
        body = "".join("(" + " ".join(f"{x:.12g}" for x in row) + ")\n" for row in v)
        kind = f"nonuniform List<vector> {len(v)}"
    else:
        v = v.reshape(-1)
        body = "".join(f"{x:.12g}\n" for x in v)
        kind = f"nonuniform List<scalar> {len(v)}"
    a, b = _block(text, "internalField")
    new = f"internalField   {kind}\n(\n{body});"
    with open(path, "w") as fh:
        fh.write(text[:a] + new + text[b:])
    return {"field": name, "cells": int(v.shape[0]), "min": float(v.min()), "max": float(v.max())}


def write_fields(case: str, fields: dict[str, np.ndarray]) -> dict[str, Any]:
    """Map a whole predicted state onto time zero of ``case``: one :func:`set_internal_field` per field."""
    return {name: set_internal_field(case, name, values) for name, values in fields.items()}


def internal_field(case: str, name: str, time: str = "0") -> np.ndarray:
    """The ``internalField`` of one field of one time directory, read back as an array (for a check or a difference)."""
    data = foam.cells(case, time=time, fields=[name])
    return np.asarray(data[name], dtype=float)


@dataclass
class OperatorSurrogate:
    """A field surrogate over one parameter: POD of the solved snapshots, coefficients interpolated in the parameter.

    ``fit(reynolds, snapshots)`` takes ``{re: {"U": (n, 3), "p": (n,)}}`` — the fields of cases solved on **one mesh**,
    so the snapshots are the same length — and keeps ``modes`` modes per field. :meth:`predict` returns the same
    structure at any parameter, interpolating the coefficients linearly between the solved ones (or holding the end
    value outside the range, and saying so in :attr:`extrapolated`).
    """

    modes: int = 8
    mean: dict[str, np.ndarray] = field(default_factory=dict)
    basis: dict[str, np.ndarray] = field(default_factory=dict)      # (modes, n_cells * comp)
    coefficients: dict[str, np.ndarray] = field(default_factory=dict)  # (n_snapshots, modes)
    reynolds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shapes: dict[str, tuple] = field(default_factory=dict)
    fitted_seconds: float = 0.0

    def fit(self, snapshots: dict[float, dict[str, np.ndarray]], *, modes: int | None = None) -> OperatorSurrogate:
        """Decompose each field of the snapshots and fit its coefficients against the parameter."""
        t0 = time.time()
        self.reynolds = np.asarray(sorted(snapshots), dtype=float)
        names = sorted({k for s in snapshots.values() for k in s})
        if not names:
            raise ValueError("no fields to fit")
        m = int(modes or self.modes)
        for name in names:
            stack = []
            for reynolds in self.reynolds:
                s = snapshots[float(reynolds)].get(name)
                if s is None:
                    raise KeyError(f"Re = {reynolds:g}: no {name!r} in the snapshot set")
                a = np.asarray(s, dtype=float)
                self.shapes.setdefault(name, a.shape)
                if a.shape != self.shapes[name]:
                    raise ValueError(f"{name}: the snapshots must share one mesh; got {a.shape} and "
                                     f"{self.shapes[name]} (solve every case on the same mesh)")
                stack.append(a.reshape(-1))
            X = np.vstack(stack)
            mean = X.mean(axis=0)
            U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
            k = max(1, min(m, len(S)))
            self.mean[name] = mean
            self.basis[name] = Vt[:k]                                   # (k, n)
            self.coefficients[name] = (X - mean) @ Vt[:k].T             # (n_re, k)
            self._energy = getattr(self, "_energy", {})
            self._energy[name] = float(S[:k].sum() / S.sum()) if S.sum() else 1.0
        self.fitted_seconds = time.time() - t0
        return self

    def predict(self, reynolds: float, *, modes: int | None = None) -> dict[str, np.ndarray]:
        """The field at ``reynolds``: the mean plus the interpolated modes, in the shape the snapshots had."""
        if not self.basis:
            raise RuntimeError("the surrogate has not been fitted")
        self.extrapolated = not (self.reynolds.min() <= reynolds <= self.reynolds.max())
        out = {}
        for name, basis in self.basis.items():
            k = basis.shape[0] if modes is None else max(1, min(int(modes), basis.shape[0]))
            coeff = np.array([np.interp(reynolds, self.reynolds, self.coefficients[name][:, i]) for i in range(k)])
            out[name] = (self.mean[name] + coeff @ basis[:k]).reshape(self.shapes[name])
        return out

    # ------------------------------------------------------------------ diagnostics
    @property
    def energy(self) -> dict[str, float]:
        """The fraction of the snapshot energy the kept modes capture, per field."""
        return dict(getattr(self, "_energy", {}))

    def error(self, reynolds: float, truth: dict[str, np.ndarray]) -> dict[str, float]:
        """Relative L2 error of the prediction against a solved field at the same parameter."""
        pred = self.predict(reynolds)
        out = {}
        for name, ref in truth.items():
            a = np.asarray(pred[name], dtype=float).reshape(-1)
            b = np.asarray(ref, dtype=float).reshape(-1)
            scale = float(np.sqrt(np.mean(b ** 2)))
            out[name] = float(np.sqrt(np.mean((a - b) ** 2)) / scale) if scale else float("nan")
        return out


@dataclass
class WarmStartResult:
    """One cold run against one warm run: the cost of each, and whether they converged to the same answer."""

    reynolds: float
    cold_iterations: int
    warm_iterations: int
    cold_seconds: float
    warm_seconds: float
    cold_converged: bool
    warm_converged: bool
    predicted_error: dict[str, float] = field(default_factory=dict)
    field_difference: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def speedup(self) -> float:
        return self.cold_seconds / self.warm_seconds if self.warm_seconds else float("nan")

    @property
    def iteration_ratio(self) -> float:
        return self.cold_iterations / self.warm_iterations if self.warm_iterations else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"reynolds": self.reynolds, "cold_iterations": self.cold_iterations,
                "warm_iterations": self.warm_iterations, "cold_seconds": round(self.cold_seconds, 3),
                "warm_seconds": round(self.warm_seconds, 3), "speedup": round(self.speedup, 3),
                "iteration_ratio": round(self.iteration_ratio, 3),
                "cold_converged": self.cold_converged, "warm_converged": self.warm_converged,
                "predicted_error": self.predicted_error, "field_difference": self.field_difference,
                "note": self.note}


def benchmark(reynolds: float, surrogate: OperatorSurrogate, root: str, *, n: int = 64,
              timeout: float = 3600.0) -> WarmStartResult:
    """Solve the cavity at ``reynolds`` twice — cold, then warm from the surrogate's prediction — and compare.

    The two runs are the same case: same mesh, same schemes, same residual control, same iteration cap. The only
    difference is the ``internalField`` at time zero, which is uniform zero for the cold run and the surrogate's
    prediction for the warm one. ``field_difference`` is the largest difference between the two converged answers,
    relative to the magnitude of the field: they must agree to the tolerance, or the warm start changed the answer and
    the saving is not a saving.
    """
    cold_case = os.path.join(root, f"cold_re{int(reynolds)}_{n}")
    cavity.write_case(cold_case, reynolds, n)
    cold = cavity.run(cold_case, reynolds, n, timeout=timeout)

    warm_case = os.path.join(root, f"warm_re{int(reynolds)}_{n}")
    cavity.write_case(warm_case, reynolds, n)
    predicted = surrogate.predict(reynolds)
    write_fields(warm_case, predicted)
    warm = cavity.run(warm_case, reynolds, n, timeout=timeout)

    out = WarmStartResult(reynolds=float(reynolds), cold_iterations=int(cold["solver"]["iterations"]),
                          warm_iterations=int(warm["solver"]["iterations"]), cold_seconds=float(cold["seconds"]),
                          warm_seconds=float(warm["seconds"]), cold_converged=bool(cold["solver"]["converged"]),
                          warm_converged=bool(warm["solver"]["converged"]))
    if not (cold["ok"] and warm["ok"]):
        out.note = f"a run failed: cold={cold['failed']!r} warm={warm['failed']!r}"
        return out
    try:
        a = foam.cells(cold_case)
        b = foam.cells(warm_case)
        for name in ("U", "p"):
            x = np.asarray(a[name], dtype=float)
            y = np.asarray(b[name], dtype=float)
            scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
            out.field_difference[name] = float(np.abs(x - y).max() / scale)
        pred_case = os.path.join(root, f"prediction_re{int(reynolds)}_{n}")
        truth_u = np.asarray(a["U"], dtype=float)
        out.predicted_error = {"U": float(np.sqrt(np.mean((predicted["U"] - truth_u) ** 2)) /
                                          (np.sqrt(np.mean(truth_u ** 2)) or 1.0))}
        del pred_case
    except (KeyError, FileNotFoundError, ValueError) as exc:
        out.note = f"the converged fields could not be compared: {exc}"
    return out


def snapshots(root: str, reynolds: Sequence[float], *, n: int = 64, fields: Sequence[str] = ("U", "p"),
              timeout: float = 3600.0) -> dict[float, dict[str, np.ndarray]]:
    """Solve the cavity at each Reynolds number on one mesh and keep the converged fields as snapshots."""
    out: dict[float, dict[str, np.ndarray]] = {}
    for case_re in reynolds:
        case = os.path.join(root, f"snapshot_re{int(case_re)}_{n}")
        cavity.write_case(case, case_re, n)
        info = cavity.run(case, case_re, n, timeout=timeout)
        if not info["ok"]:
            raise RuntimeError(f"Re = {case_re:g}: {info['failed']} failed: {info['log'][-800:]}")
        data = foam.cells(case, fields=list(fields))
        out[float(case_re)] = {name: np.asarray(data[name], dtype=float) for name in fields}
    return out
