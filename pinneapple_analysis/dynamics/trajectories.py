"""Trajectories: the common input of the dynamics toolkit, and the adapters that make one from any source.

A :class:`Trajectory` is a state ``x(t_i)`` in ``R^d`` sampled at increasing times ``t_i``, which may be irregular.
Missing values are ``NaN`` (per component, so a sensor that drops out does not remove the others). The
observation window ends at ``observed_until`` (by default the last sample): an event that has not happened by then
is *right-censored*, which methods such as first-passage estimation must treat differently from "never happens".
``source`` records where the numbers came from, so a result can say whether it was computed on measurements, a
simulation, a closed form or a learned model's prediction.

A :class:`TrajectoryEnsemble` is several trajectories of the same variables (Monte Carlo paths, repeated
experiments, runs at different parameters); they may have different time grids and different windows.

Adapters (each returns a :class:`Trajectory`):

* :func:`from_array` -- arrays already in memory.
* :func:`from_table` / :func:`from_csv` -- a table of columns (dict, pandas DataFrame, CSV file) with units read
  from the headers (``"T [K]"``, ``"p (bar)"``) by :func:`pinneapple_data.physical_units.split_header`.
* :func:`from_solution` -- a ``pp.solve`` result (``Solution``: PINN, classical solver, closed form) sampled at a
  point of space over time.
* :func:`from_model` -- any callable of time (a trained PINN, a surrogate); PyTorch / JAX outputs are converted.
* :func:`from_snapshots` -- PDE snapshots ``u(t, x...)`` (a solver's output, or an FNO / DeepONet rollout on its
  grid) flattened into a state vector, with the spatial shape kept for :meth:`Trajectory.snapshots`.

What a trajectory does not do: it never fills a gap silently. :meth:`Trajectory.resample` interpolates only inside
the data, leaves ``NaN`` outside, and marks the result as interpolated.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._backends import to_numpy
from ._contract import jsonable

__all__ = ["SOURCE_KINDS", "Trajectory", "TrajectoryEnsemble", "from_array", "from_table", "from_csv",
           "from_solution", "from_model", "from_snapshots"]

SOURCE_KINDS: tuple[str, ...] = ("experimental", "simulation", "analytic", "pinn", "fno", "deeponet", "model",
                                 "stochastic", "array")
_MISSING_TEXT = {"", "nan", "na", "n/a", "null", "none", "-"}


@dataclass(frozen=True)
class Trajectory:
    """``x(t_i)`` in ``R^d`` at strictly increasing times; see the module docstring.

    ``t``: ``(n,)`` finite, strictly increasing. ``x``: ``(n, d)`` (a 1-D ``x`` is one component), ``NaN`` where a
    value is missing. ``names``: one per component. ``units``: unit text per component name and ``"t"`` for time
    (optional; ``None`` means not stated, not dimensionless). ``observed_until``: end of the observation window,
    ``>= t[-1]``. ``source``: ``{"kind": one of SOURCE_KINDS, ...details}``."""
    t: np.ndarray
    x: np.ndarray
    names: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    observed_until: float | None = None
    source: Mapping[str, Any] = field(default_factory=lambda: {"kind": "array"})

    def __post_init__(self):
        t = np.asarray(self.t, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        if t.ndim != 1 or t.size == 0:
            raise ValueError(f"t must be a non-empty 1-D array, got shape {t.shape}")
        if not np.all(np.isfinite(t)):
            raise ValueError("t must be finite (a missing time stamp cannot be placed; drop that sample)")
        if t.size > 1 and np.any(np.diff(t) <= 0):
            i = int(np.argmax(np.diff(t) <= 0))
            raise ValueError(f"t must be strictly increasing: t[{i}]={t[i]!r} >= t[{i + 1}]={t[i + 1]!r} "
                             "(sort the samples and merge duplicate time stamps first)")
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != t.size:
            raise ValueError(f"x must have shape (n, d) with n = len(t) = {t.size}, got {x.shape}")
        if np.any(np.isinf(x)):
            raise ValueError("x contains inf; use NaN for missing values")
        names = tuple(str(n) for n in self.names) if self.names else tuple(f"x{i}" for i in range(x.shape[1]))
        if len(names) != x.shape[1]:
            raise ValueError(f"{len(names)} names for {x.shape[1]} components")
        if len(set(names)) != len(names):
            raise ValueError(f"component names must be unique, got {names}")
        end = float(t[-1]) if self.observed_until is None else float(self.observed_until)
        if not math.isfinite(end) or end < t[-1]:
            raise ValueError(f"observed_until={self.observed_until!r} must be finite and >= the last time {t[-1]!r}")
        source = dict(self.source) if self.source else {"kind": "array"}
        source.setdefault("kind", "array")
        if source["kind"] not in SOURCE_KINDS:
            raise ValueError(f"source kind must be one of {SOURCE_KINDS}, got {source['kind']!r}")
        unknown = set(self.units) - set(names) - {"t"}
        if unknown:
            raise ValueError(f"units given for unknown components {sorted(unknown)}; components are {names}")
        t.setflags(write=False)
        x.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "observed_until", end)
        object.__setattr__(self, "source", source)

    # ------------------------------------------------------------------ shape and sampling
    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def kind(self) -> str:
        return str(self.source["kind"])

    @property
    def span(self) -> tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])

    @property
    def time_unit(self) -> str | None:
        return self.units.get("t")

    def is_regular(self, rtol: float = 1e-6) -> bool:
        """True when every step equals the median step within ``rtol`` (relative)."""
        if self.n < 3:
            return True
        dt = np.diff(self.t)
        return bool(np.all(np.abs(dt - np.median(dt)) <= rtol * np.median(dt)))

    @property
    def dt(self) -> float | None:
        """The step of a regular grid, ``None`` for an irregular one (see :meth:`sampling`)."""
        if self.n < 2 or not self.is_regular():
            return None
        return float(np.median(np.diff(self.t)))

    def sampling(self) -> dict[str, float]:
        """Step statistics: min, median, max and the ratio max/min (1 for a regular grid)."""
        if self.n < 2:
            return {"n": self.n}
        dt = np.diff(self.t)
        return {"n": self.n, "dt_min": float(dt.min()), "dt_median": float(np.median(dt)),
                "dt_max": float(dt.max()), "irregularity": float(dt.max() / dt.min())}

    @property
    def missing(self) -> np.ndarray:
        """Boolean ``(n, d)`` mask of missing values."""
        return np.isnan(self.x)

    @property
    def has_missing(self) -> bool:
        return bool(self.missing.any())

    @property
    def censored(self) -> bool:
        """True when the window extends past the last sample (observation stopped without a final sample)."""
        return self.observed_until > self.t[-1]

    # ------------------------------------------------------------------ selection
    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no component {name!r}; components are {self.names}") from None

    def component(self, name: str | int) -> np.ndarray:
        """Values of one component, ``(n,)``."""
        i = name if isinstance(name, int) else self.index(name)
        return self.x[:, i]

    def select(self, names: Sequence[str]) -> Trajectory:
        idx = [self.index(n) for n in names]
        units = {k: v for k, v in self.units.items() if k == "t" or k in names}
        return replace(self, x=self.x[:, idx], names=tuple(names), units=units)

    def window(self, t0: float | None = None, t1: float | None = None) -> Trajectory:
        """Samples with ``t0 <= t <= t1``. The window of the result ends at ``min(t1, observed_until)``."""
        lo = -np.inf if t0 is None else t0
        hi = np.inf if t1 is None else t1
        keep = (self.t >= lo) & (self.t <= hi)
        if not keep.any():
            raise ValueError(f"no samples in [{t0}, {t1}]; the trajectory spans {self.span}")
        end = self.observed_until if t1 is None else min(float(t1), self.observed_until)
        return replace(self, t=self.t[keep], x=self.x[keep], observed_until=max(end, float(self.t[keep][-1])))

    def dropna(self, how: str = "any") -> Trajectory:
        """Drop samples with a missing component (``how="any"``) or with all components missing (``"all"``)."""
        m = self.missing
        bad = m.any(axis=1) if how == "any" else m.all(axis=1) if how == "all" else None
        if bad is None:
            raise ValueError("how must be 'any' or 'all'")
        if bad.all():
            raise ValueError("every sample has a missing value")
        return replace(self, t=self.t[~bad], x=self.x[~bad])

    def resample(self, t_new: Sequence[float] | np.ndarray, method: str = "linear") -> Trajectory:
        """Values at ``t_new`` by ``"linear"`` interpolation or ``"previous"`` (zero-order hold), per component,
        from that component's non-missing samples only. Outside a component's data range the value is ``NaN``
        (no extrapolation). The result is marked ``source["resampled"]``: interpolated values are not
        measurements, and any statistic computed on them inherits the interpolation."""
        t_new = np.asarray(t_new, dtype=np.float64)
        out = np.full((t_new.size, self.dim), np.nan)
        for j in range(self.dim):
            ok = ~np.isnan(self.x[:, j])
            if ok.sum() == 0:
                continue
            tj, xj = self.t[ok], self.x[ok, j]
            inside = (t_new >= tj[0]) & (t_new <= tj[-1])
            if method == "linear":
                out[inside, j] = np.interp(t_new[inside], tj, xj)
            elif method == "previous":
                k = np.searchsorted(tj, t_new[inside], side="right") - 1
                out[inside, j] = xj[k]
            else:
                raise ValueError("method must be 'linear' or 'previous'")
        source = dict(self.source)
        source["resampled"] = {"method": method, "from_n": self.n, "from_sampling": self.sampling()}
        end = max(self.observed_until, float(t_new[-1])) if t_new.size else self.observed_until
        return Trajectory(t_new, out, self.names, self.units, end, source)

    def snapshots(self) -> np.ndarray:
        """``x`` reshaped to ``(n, *spatial_shape)`` for a trajectory made by :func:`from_snapshots`."""
        shape = self.source.get("spatial_shape")
        if shape is None:
            raise ValueError("this trajectory was not built from snapshots (no spatial_shape in its source)")
        return self.x.reshape((self.n, *shape))

    # ------------------------------------------------------------------ export
    def fingerprint_payload(self) -> dict[str, Any]:
        return {"t": self.t, "x": self.x, "names": list(self.names), "units": dict(self.units),
                "observed_until": self.observed_until, "source": jsonable(self.source)}

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self.fingerprint_payload())

    def __repr__(self) -> str:
        reg = "regular" if self.is_regular() else "irregular"
        miss = f", {int(self.missing.sum())} missing" if self.has_missing else ""
        return (f"Trajectory(kind={self.kind!r}, n={self.n}, dim={self.dim}, names={self.names}, "
                f"t=[{self.t[0]:.4g}, {self.t[-1]:.4g}] {reg}{miss})")


class TrajectoryEnsemble(Sequence):
    """Several trajectories of the same variables (Monte Carlo paths, repeated experiments, a parameter sweep).

    Members may have different time grids, windows and missing values; :meth:`stack` needs a common grid.
    ``labels`` optionally tags each member (e.g. its parameter values)."""

    def __init__(self, trajectories: Sequence[Trajectory], labels: Sequence[Mapping[str, Any]] | None = None):
        trajs = list(trajectories)
        if not trajs:
            raise ValueError("an ensemble needs at least one trajectory")
        names = trajs[0].names
        for i, tr in enumerate(trajs):
            if not isinstance(tr, Trajectory):
                raise TypeError(f"member {i} is a {type(tr).__name__}, not a Trajectory")
            if tr.names != names:
                raise ValueError(f"member {i} has components {tr.names}, member 0 has {names}")
        if labels is not None and len(labels) != len(trajs):
            raise ValueError(f"{len(labels)} labels for {len(trajs)} trajectories")
        self._trajs = trajs
        self.labels = [dict(lb) for lb in labels] if labels is not None else [{} for _ in trajs]

    @classmethod
    def from_paths(cls, t, paths, names: Sequence[str] | None = None, *, kind: str = "stochastic",
                   units: Mapping[str, str] | None = None, observed_until: float | None = None,
                   **source: Any) -> TrajectoryEnsemble:
        """An ensemble on one grid from ``paths`` of shape ``(m, n)`` (scalar state) or ``(m, n, d)``."""
        P = to_numpy(paths)
        if P.ndim == 2:
            P = P[:, :, None]
        if P.ndim != 3:
            raise ValueError(f"paths must have shape (m, n) or (m, n, d), got {P.shape}")
        src = {"kind": kind, **source}
        return cls([Trajectory(t, P[k], tuple(names or ()), dict(units or {}), observed_until, src)
                    for k in range(P.shape[0])])

    def __len__(self) -> int:
        return len(self._trajs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return TrajectoryEnsemble(self._trajs[i], self.labels[i])
        return self._trajs[i]

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajs)

    @property
    def names(self) -> tuple[str, ...]:
        return self._trajs[0].names

    @property
    def dim(self) -> int:
        return self._trajs[0].dim

    @property
    def horizons(self) -> np.ndarray:
        """``observed_until`` of each member."""
        return np.array([tr.observed_until for tr in self._trajs])

    def on_common_grid(self, rtol: float = 1e-12) -> bool:
        t0 = self._trajs[0].t
        return all(tr.n == t0.size and np.allclose(tr.t, t0, rtol=rtol, atol=0.0) for tr in self._trajs)

    @property
    def t(self) -> np.ndarray:
        """The common time grid; raises when the members are on different grids."""
        if not self.on_common_grid():
            raise ValueError("members are on different time grids; resample them first or use each member's t")
        return self._trajs[0].t

    def stack(self) -> np.ndarray:
        """``(m, n, d)`` array of a common-grid ensemble."""
        _ = self.t
        return np.stack([tr.x for tr in self._trajs])

    def fingerprint_payload(self) -> dict[str, Any]:
        return {"members": [tr.fingerprint_payload() for tr in self._trajs], "labels": jsonable(self.labels)}

    def __repr__(self) -> str:
        grid = "common grid" if self.on_common_grid() else "different grids"
        return f"TrajectoryEnsemble(m={len(self)}, dim={self.dim}, names={self.names}, {grid})"


# --------------------------------------------------------------------------- adapters
def from_array(t, x, names: Sequence[str] | None = None, *, units: Mapping[str, str] | None = None,
               observed_until: float | None = None, kind: str = "array", **source: Any) -> Trajectory:
    """A trajectory from arrays (NumPy, PyTorch or JAX). ``kind`` says what the numbers are (see
    :data:`SOURCE_KINDS`); extra keywords are kept in ``source``."""
    return Trajectory(to_numpy(t), to_numpy(x), tuple(names or ()), dict(units or {}), observed_until,
                      {"kind": kind, **source})


def _parse_float(v: Any) -> float:
    if v is None:
        return math.nan
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in _MISSING_TEXT:
            return math.nan
        return float(s)
    return float(v)


def from_table(table: Mapping[str, Sequence[Any]] | Any, time: str, columns: Sequence[str] | None = None, *,
               kind: str = "experimental", sort: bool = False, **source: Any) -> Trajectory:
    """A trajectory from a table of columns: a mapping ``{header: values}`` or a pandas ``DataFrame``.

    Units are read from the headers (``"T [K]"`` -> component ``"T"`` in ``"K"``); ``time`` and ``columns`` may
    name a header either in full or by its name without the unit. Empty cells and ``nan``/``NA`` text are
    missing values. Rows whose time is missing are dropped (counted in ``source["dropped_rows"]``).
    ``sort=True`` sorts unsorted rows by time; duplicate time stamps are always refused."""
    from pinneapple_data.physical_units import split_header

    if hasattr(table, "columns") and hasattr(table, "__getitem__") and not isinstance(table, Mapping):
        table = {str(c): list(table[c]) for c in table.columns}
    headers = list(table)
    parsed = {h: split_header(h) for h in headers}

    def find(key: str) -> str:
        if key in table:
            return key
        hits = [h for h, (name, _) in parsed.items() if name == key]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"no column {key!r} (or several); headers are {headers}")

    th = find(time)
    cols = [find(c) for c in columns] if columns is not None else [h for h in headers if h != th]
    if not cols:
        raise ValueError("no value columns")
    t = np.array([_parse_float(v) for v in table[th]])
    X = np.column_stack([[_parse_float(v) for v in table[c]] for c in cols])
    keep = np.isfinite(t)
    t, X = t[keep], X[keep]
    if sort:
        order = np.argsort(t, kind="stable")
        t, X = t[order], X[order]
    names = tuple(parsed[c][0] or c for c in cols)
    units = {parsed[c][0] or c: parsed[c][1] for c in cols if parsed[c][1]}
    if parsed[th][1]:
        units["t"] = parsed[th][1]
    src = {"kind": kind, "columns": cols, "time_column": th, "dropped_rows": int((~keep).sum()), **source}
    return Trajectory(t, X, names, units, None, src)


def from_csv(path: str, time: str, columns: Sequence[str] | None = None, *, delimiter: str = ",",
             kind: str = "experimental", sort: bool = False, **source: Any) -> Trajectory:
    """A trajectory from a CSV file with a header row; see :func:`from_table` for units and missing values."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        rows = [r for r in reader if any(c.strip() for c in r)]
    for i, r in enumerate(rows):
        if len(r) != len(header):
            raise ValueError(f"{path}: row {i + 2} has {len(r)} fields, the header has {len(header)}")
    table = {h: [r[j] for r in rows] for j, h in enumerate(header)}
    return from_table(table, time, columns, kind=kind, sort=sort, path=str(path), **source)


_SOLUTION_KINDS = {"classical": "simulation", "reference": "simulation", "external": "simulation",
                   "analytic": "analytic", "neural": "model"}


def from_solution(solution: Any, t, at: Mapping[str, float] | None = None, fields: Sequence[str] | None = None, *,
                  time_coord: str = "t", **source: Any) -> Trajectory:
    """Fields of a ``pp.solve`` result at one point of space, over the times ``t``.

    ``solution`` is a ``Solution`` (anything with ``problem.coords``, ``problem.fields`` and ``predict(points)``).
    ``at`` gives every coordinate other than ``time_coord``; ``fields`` defaults to all. The source kind is
    ``"pinn"`` for ``method="pinn"``, ``"analytic"`` for closed forms, ``"simulation"`` for classical and
    external solvers, ``"model"`` otherwise. The values are the solution's predictions, not measurements: they
    carry its error, which this adapter does not estimate (check it with ``pp.trust_card``)."""
    coords = list(solution.problem.coords)
    all_fields = list(solution.problem.fields)
    if time_coord not in coords:
        raise ValueError(f"the problem has no time coordinate {time_coord!r}; coords are {coords}")
    at = dict(at or {})
    missing = [c for c in coords if c != time_coord and c not in at]
    extra = sorted(set(at) - set(coords))
    if missing or extra:
        raise ValueError(f"'at' must give every coordinate except {time_coord!r}: missing {missing}, unknown {extra}")
    tt = to_numpy(t).ravel()
    pts = np.empty((tt.size, len(coords)))
    for j, c in enumerate(coords):
        pts[:, j] = tt if c == time_coord else float(at[c])
    y = np.asarray(solution.predict(pts), dtype=np.float64)
    fields = list(fields or all_fields)
    idx = [all_fields.index(f) for f in fields]
    method = getattr(solution, "method", "")
    kind = "pinn" if method == "pinn" else _SOLUTION_KINDS.get(getattr(solution, "kind", ""), "model")
    problem = getattr(solution.problem, "name", "")
    src = {"kind": kind, "method": method, "problem": problem, "at": at, **source}
    return Trajectory(tt, y[:, idx], tuple(fields), {}, None, src)


def from_model(fn: Callable[[Any], Any], t, names: Sequence[str] | None = None, *, kind: str = "model",
               column: bool = False, units: Mapping[str, str] | None = None, **source: Any) -> Trajectory:
    """Evaluate a callable of time (a trained PINN, a surrogate, a closed form) on ``t``.

    ``fn`` receives ``t`` as ``(n,)`` (or ``(n, 1)`` with ``column=True``) and returns ``(n,)`` or ``(n, d)``;
    a PyTorch tensor or JAX array is converted. To evaluate a PyTorch module, wrap it so it receives a tensor
    (``lambda t: net(torch.as_tensor(t, dtype=torch.float32))``); this adapter does not guess devices or dtypes."""
    tt = to_numpy(t).ravel()
    y = to_numpy(fn(tt[:, None] if column else tt))
    if y.ndim == 2 and y.shape[0] != tt.size and y.shape[1] == tt.size:
        raise ValueError(f"fn returned shape {y.shape}; expected (n={tt.size}, d) with time along the first axis")
    return Trajectory(tt, y, tuple(names or ()), dict(units or {}), None, {"kind": kind, **source})


def from_snapshots(t, snapshots, *, name: str = "u", kind: str = "simulation",
                   coords: Mapping[str, Any] | None = None, unit: str | None = None, **source: Any) -> Trajectory:
    """PDE snapshots ``u(t_i, grid)`` of shape ``(n_t, *spatial)`` as a trajectory of dimension ``prod(spatial)``.

    Components are named ``"u[0]"``, ``"u[1]"``, ... in C order; ``coords`` (e.g. ``{"x": x_axis}``) and the
    spatial shape are kept in ``source`` and :meth:`Trajectory.snapshots` restores the grid. Use ``kind="fno"`` or
    ``"deeponet"`` for an operator network's rollout on its grid."""
    U = to_numpy(snapshots)
    tt = to_numpy(t).ravel()
    if U.ndim < 2 or U.shape[0] != tt.size:
        raise ValueError(f"snapshots must have shape (n_t={tt.size}, *spatial), got {U.shape}")
    spatial = tuple(int(s) for s in U.shape[1:])
    names = tuple(f"{name}[{i}]" for i in range(int(np.prod(spatial))))
    units = {n: unit for n in names} if unit else {}
    src = {"kind": kind, "spatial_shape": spatial, "field": name,
           "coords": {k: to_numpy(v).tolist() for k, v in (coords or {}).items()}, **source}
    return Trajectory(tt, U.reshape(tt.size, -1), names, units, None, src)
