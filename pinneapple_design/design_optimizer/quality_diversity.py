"""Quality-diversity search: the best design in every cell of a behaviour space, and the failure map that comes from
pointing the same archive at a model instead of at a design (#354).

An optimiser answers one question: which design is best. When the objective has several incommensurable parts, a
Pareto front answers a second one: which designs are not worse than another in everything. Neither says *what kinds of
design exist*, and neither covers the space on purpose -- the search walks towards the optimum and abandons the rest.
Quality-diversity (MAP-Elites, Mouret & Clune 2015) keeps the loop and changes the bookkeeping: a grid over a
**behaviour space** (how the design behaves, not what its parameters are), one cell per behaviour, and the best design
found for that behaviour in each cell. The result is a map an engineer reads at a glance -- "the cheapest heat sink at
each fin density", "the airfoil with the least drag at each lift" -- and a far better training set for a surrogate than
random sampling, because every cell is a distinct behaviour represented by its best design.

The same archive, with fitness = the model's own error, is a **failure map**. The cell whose elite error explodes is a
region where the model is wrong, and the archive hands it to you with the number, the design and the behaviour
condition that produced it. That is the second use in the issue, and the one this module is built around: the
out-of-domain region of the trust layer, found by search instead of declared by hand.

Behaviour axes must be *outcomes*, computed from the reference physics or the geometry -- never from the model under
test, and never the design parameters themselves:

>>> behaviour = BehaviourSpace((BehaviourAxis("Cl", -0.6, 1.5, why="loading the boundary layer must survive"),
...                             BehaviourAxis("Cd", 0.0075, 0.0135, why="the wake the section sheds")), bins=(12, 10))

Why the GA of :class:`~.optimizer.EvolutionaryDesignOptimizer` is not reused here
------------------------------------------------------------------------------
MAP-Elites needs a *parent per cell*: the mutation that produces the next candidate is chosen by the cell it competes
in, so the population is the archive and there is no global ranking to select survivors from. ``ask``/``tell``
selects by objective value over one population, which is exactly the pressure that empties the archive back into a
single corner -- the collapse this module has to avoid. What is reused is the parameter space (:class:`ParamSpace`,
its bounds and clipping) and the mutation conventions of :class:`DesignOptimizerConfig`, so a search here and a GA run
there are configured with the same knobs and sampled in the same box. The operators themselves are Gaussian and line
mutations, as the issue asks.

Measured (``examples/quality_diversity``, airfoil surrogate, seed 0, budget 600)
------------------------------------------------------------------------------
The archive is a 12 x 10 grid over (Cl, Cd) of the in-house section physics. The surrogate learned the documented
design box and the search was pointed at a wider envelope: the QD archive filled the map, and the cells whose error
crosses the 10 % tolerance are the ones the model has never seen (high lift at low form drag). The numbers, with
seed and budget, are in ``docs/guides/quality_diversity.md``.

Usage
-----
>>> archive = EliteArchive(behaviour, sense="max")            # "max": keep the worst error per cell
>>> def evaluate(x):                                           # x = (m, t/c, alpha)
...     return Evaluation(value=error(x), descriptors=reference(x))
>>> result = QualityDiversitySearch(archive, bounds, evaluate, QDConfig(budget=600, seed=0)).run()
>>> print(result.summary(), archive.fill(), archive.diversity().text())
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .optimizer import DesignOptimizerConfig, ParamSpace

__all__ = ["BehaviourAxis", "BehaviourSpace", "descriptor_vector", "descriptor_axes", "Evaluation", "Elite",
           "DiversityReport", "EliteArchive", "QDConfig", "QDSearchResult", "QualityDiversitySearch", "BaselineSearch",
           "latin_hypercube", "uniform_search", "FailingCell", "FailureRegion", "find_failure_region", "plot_archive"]


# ---------------------------------------------------------------------------
# Behaviour space: the axes the map is indexed by
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviourAxis:
    """One axis of the behaviour space, with the physical reason it is one.

    A behaviour axis is what the design *does*, so that two designs with different parameters share a cell when the
    physics cannot tell them apart. The ranges are part of the definition: a descriptor outside them cannot be placed,
    and is reported as such instead of being clipped into an edge cell that would then lie about its behaviour.

    Parameters
    ----------
    name:
        Short label, the axis title of the map.
    lo, hi:
        Behaviour range the grid spans.
    unit:
        Display unit, ``""`` when dimensionless.
    why:
        Why this quantity is a behaviour of the physics and not a design parameter. Printed by the map and the report;
        an axis without a reason is a design parameter in disguise.
    scale:
        ``"linear"`` (default) or ``"log"``. Thermal resistances, pressure drops and pumping powers span decades while
        the interesting cells are at the low end; a linear grid there spends most of its cells on values nobody builds.
    """

    name: str
    lo: float
    hi: float
    unit: str = ""
    why: str = ""
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.scale not in ("linear", "log"):
            raise ValueError(f"axis {self.name!r}: scale must be 'linear' or 'log'")
        if self.scale == "log" and self.lo <= 0:
            raise ValueError(f"axis {self.name!r}: a log scale needs lo > 0")

    def _fraction(self, value: float) -> float | None:
        """Position of ``value`` in ``[lo, hi]`` as a fraction, ``None`` when it is not placeable."""
        v = float(value)
        if not math.isfinite(v) or not self.hi > self.lo:
            return None
        if self.scale == "log":
            if v <= 0 or v < self.lo or v > self.hi:
                return None
            return math.log(v / self.lo) / math.log(self.hi / self.lo)
        if v < self.lo or v > self.hi:
            return None
        return (v - self.lo) / (self.hi - self.lo)

    def _value(self, fraction: float) -> float:
        """Value at a fraction of the axis."""
        if self.scale == "log":
            return float(self.lo * (self.hi / self.lo) ** fraction)
        return float(self.lo + fraction * (self.hi - self.lo))

    def index(self, value: float, bins: int) -> int | None:
        """Bin of ``value`` in ``[lo, hi]``, or ``None`` when it is not a finite number inside the range."""
        f = self._fraction(value)
        if f is None:
            return None
        return min(max(int(np.floor(f * bins)), 0), bins - 1)

    def centre(self, index: int, bins: int) -> float:
        """Behaviour value at the centre of bin ``index``."""
        return self._value((index + 0.5) / bins)

    def interval(self, index: int, bins: int) -> tuple[float, float]:
        """Behaviour interval covered by bin ``index`` -- the *condition* a failing cell is reported with."""
        return self._value(index / bins), self._value((index + 1) / bins)


@dataclass(frozen=True)
class BehaviourSpace:
    """A grid over ``axes``, one cell per behaviour combination.

    Parameters
    ----------
    axes:
        The behaviour axes, in the order the descriptors are given.
    bins:
        Cells per axis (an int for every axis, or one per axis).
    labels:
        What the map is of ("airfoil sections at Re = 3e6"), for the title and the report.
    """

    axes: tuple[BehaviourAxis, ...]
    bins: int | tuple[int, ...] = 10
    labels: str = ""

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("a behaviour space needs at least one axis")
        for axis in self.axes:
            if not axis.hi > axis.lo:
                raise ValueError(f"axis {axis.name!r}: hi must be greater than lo")
        if len(self.bins_per_axis) != len(self.axes) or min(self.bins_per_axis) < 1:
            raise ValueError("bins must be positive and one per axis")

    @property
    def bins_per_axis(self) -> tuple[int, ...]:
        if isinstance(self.bins, int):
            return tuple([int(self.bins)] * len(self.axes))
        return tuple(int(b) for b in self.bins)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bins_per_axis

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def cell(self, descriptors: Sequence[float]) -> tuple[int, ...] | None:
        """Cell index of a descriptor vector, or ``None`` when a descriptor is missing, non-finite or out of range."""
        values = list(descriptors)
        if len(values) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} descriptors, got {len(values)}")
        out = []
        for axis, bins, v in zip(self.axes, self.bins_per_axis, values, strict=True):
            i = axis.index(v, bins)
            if i is None:
                return None
            out.append(i)
        return tuple(out)

    def centre(self, cell: Sequence[int]) -> np.ndarray:
        """Descriptor values at the centre of ``cell``."""
        return np.array([a.centre(int(i), b) for a, b, i in
                         zip(self.axes, self.bins_per_axis, cell, strict=True)])

    def condition(self, cell: Sequence[int]) -> dict[str, tuple[float, float]]:
        """The behaviour interval of each axis for ``cell`` -- what makes a failing cell a *condition*, not a point."""
        return {a.name: a.interval(int(i), b)
                for a, b, i in zip(self.axes, self.bins_per_axis, cell, strict=True)}

    def describe(self, cell: Sequence[int]) -> str:
        """One line naming the cell by its behaviour intervals."""
        parts = []
        for axis, (lo, hi) in zip(self.axes, [self.condition(cell)[a.name] for a in self.axes], strict=True):
            parts.append(f"{axis.name} in [{lo:.4g}, {hi:.4g}]{' ' + axis.unit if axis.unit else ''}")
        return "; ".join(parts)


def descriptor_vector(descriptors: Any, keys: Sequence[str]) -> tuple[float, ...]:
    """One descriptor vector from a qualitative ``Descriptors`` (or any mapping of named descriptors).

    These are the axes the issue asks for: the descriptors of
    :func:`pinneapple_design.qualitative.describe` -- volume, wetted area, slenderness, sphericity, the flat areas and
    the sharp-edge length -- say what a body *is* to the flow and to the solver, without being the parameters that
    produced it. Two very different meshes with the same sphericity belong in the same cell, and that is the point.

    Parameters
    ----------
    descriptors:
        A :class:`pinneapple_design.qualitative.Descriptors`, or any object with ``as_dict()``, or a mapping.
    keys:
        Which descriptors are the axes, in the order of the behaviour space.

    >>> from pinneapple_design.qualitative import describe
    >>> descriptor_vector(describe(mesh), ("slenderness", "sphericity"))     # doctest: +SKIP
    (2.41, 0.83)
    """
    table = descriptors.as_dict() if hasattr(descriptors, "as_dict") else dict(descriptors)
    missing = [k for k in keys if k not in table]
    if missing:
        raise KeyError(f"descriptors {missing} not among {sorted(table)}")
    return tuple(float(table[k]) for k in keys)


def descriptor_axes(samples: Sequence[Any], keys: Sequence[str], *, bins: int = 10, labels: str = "",
                    unit: Mapping[str, str] | None = None, why: Mapping[str, str] | None = None,
                    scale: Mapping[str, str] | None = None, margin: float = 0.02) -> BehaviourSpace:
    """A :class:`BehaviourSpace` whose ranges are what ``samples`` actually reach, one axis per descriptor key.

    The grid has to be declared before the designs that will fill it are known, and a hand-written range is a guess
    that either wastes cells or clips the behaviour. Taking the range from the designs the box can produce -- the
    Latin hypercube of the campaign, or a coarse mesh sweep -- makes the map a measurement: the axes span the
    behaviours that exist, and ``margin`` pads them a little so a slightly different design is not refused at the edge.

    Parameters
    ----------
    samples:
        ``Descriptors`` objects, ``as_dict()``-able objects, or plain mappings of named descriptors.
    keys:
        The descriptor names to use as axes, in order.
    bins:
        Cells per axis.
    labels:
        Title of the map.
    unit, why, scale:
        Per-descriptor ``{key: value}`` for :class:`BehaviourAxis`; a missing key falls back to the defaults.
    margin:
        Fraction of the observed span added to each end.

    >>> space = descriptor_axes(meshes, ("slenderness", "sphericity"),
    ...                         why={"slenderness": "how much body the flow has to go around"})
    >>> space.shape
    (10, 10)
    """
    if not keys:
        raise ValueError("descriptor_axes needs at least one descriptor key")
    unit, why, scale = unit or {}, why or {}, scale or {}
    vectors = np.array([descriptor_vector(s, keys) for s in samples], dtype=float)
    if len(vectors) == 0:
        raise ValueError("descriptor_axes needs at least one sample to measure the ranges")
    axes = []
    for j, key in enumerate(keys):
        column = vectors[:, j]
        finite = column[np.isfinite(column)]
        lo, hi = float(finite.min()), float(finite.max())
        if not hi > lo:
            raise ValueError(f"descriptor {key!r} does not vary over the samples: it cannot be a behaviour axis")
        pad = margin * (hi - lo)
        lo, hi = lo - pad, hi + pad
        axes.append(BehaviourAxis(key, lo, hi, unit=unit.get(key, ""), why=why.get(key, ""),
                                  scale=scale.get(key, "linear")))
    return BehaviourSpace(axes=tuple(axes), bins=bins, labels=labels)


# ---------------------------------------------------------------------------
# What one evaluation of a design gives back
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """One evaluated design: its fitness, its behaviour, and the per-channel numbers behind them.

    Parameters
    ----------
    value:
        The number the archive ranks designs by inside a cell (a performance to minimise, or an error to maximise).
    descriptors:
        One value per behaviour axis. ``None`` (or a non-finite value) means "could not be placed": the archive counts
        it as unplaceable instead of inventing a cell.
    error:
        Optional per-channel breakdown of ``value`` (``{"cl": 0.04, "cd": 0.21}``), carried into the report so a
        failing cell says *which* output broke.
    valid:
        False when the case is not a physically legitimate one (the reference itself outside its own validity, a
        geometry that does not exist). Invalid cases never enter the archive -- a "failure" found where the physics
        does not apply is not a failure of the model.
    note:
        Free text kept with the elite (the reference value, a reason, a solver warning).
    """

    value: float
    descriptors: Sequence[float] | None
    error: Mapping[str, float] | None = None
    valid: bool = True
    note: str = ""


@dataclass
class Elite:
    """The best design found for one behaviour cell."""

    cell: tuple[int, ...]
    design: np.ndarray
    descriptors: np.ndarray
    value: float
    error: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        names = list(names) if names is not None else [f"x{i}" for i in range(len(self.design))]
        return {"cell": list(self.cell), "value": float(self.value),
                "design": {n: float(v) for n, v in zip(names, self.design, strict=True)},
                "descriptors": [float(v) for v in self.descriptors], "error": dict(self.error), "note": self.note}


# ---------------------------------------------------------------------------
# Diversity: the numbers that say the archive is a map and not a point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiversityReport:
    """Whether the archive is a populated, spread-out map -- the check that it did not collapse into one point.

    Attributes
    ----------
    cells:
        Occupied cells.
    total_cells:
        Cells in the grid.
    fill:
        ``cells / total_cells``.
    design_spread:
        Mean pairwise Euclidean distance between the elite designs in the **normalised** design box (each parameter
        scaled to [0, 1] by its own bounds). Bounded by ``sqrt(dimension)``; 0 means every elite is the same design.
    dominant_share:
        Fraction of the elites within ``radius`` (normalised design units) of the densest elite. 1.0 means all the
        map was filled by one cluster of indistinguishable designs, which is a collapse even when many cells are full.
    ok:
        ``fill`` above the floor, enough cells, and no cluster dominating -- see :meth:`EliteArchive.diversity`.
    problems:
        The failed conditions, in words, for the report and the test that refuses a collapsed archive.
    """

    cells: int
    total_cells: int
    fill: float
    design_spread: float
    dominant_share: float
    radius: float
    ok: bool
    problems: tuple[str, ...] = ()

    def text(self) -> str:
        """One line per number, then the problems."""
        head = (f"{self.cells}/{self.total_cells} cells (fill {self.fill:.0%}), design spread "
                f"{self.design_spread:.3f}, densest cluster {self.dominant_share:.0%} of the elites")
        return "\n".join([head, *(f"  - {p}" for p in self.problems)])


class CollapsedArchiveError(RuntimeError):
    """The archive is a point, not a map (the check :meth:`EliteArchive.assert_diverse` raised)."""


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------


class EliteArchive:
    """The best design per behaviour cell, with the counters of what it refused.

    Parameters
    ----------
    space:
        The behaviour grid.
    sense:
        ``"min"`` when ``Evaluation.value`` is a cost to minimise (a design map), ``"max"`` when it is an error or a
        residual to maximise (a failure map).
    name:
        What is being archived, for the report.
    design_names:
        Names of the design parameters, in the order the vectors are given; used by every report.
    """

    def __init__(self, space: BehaviourSpace, sense: str = "min", name: str = "",
                 design_names: Sequence[str] | None = None) -> None:
        if sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        self.space = space
        self.sense = sense
        self.name = name
        self.design_names = tuple(design_names) if design_names is not None else None
        self._elites: dict[tuple[int, ...], Elite] = {}
        self.n_considered = 0
        self.n_unplaceable = 0
        self.n_invalid = 0
        self.n_not_better = 0
        self.history: list[tuple[int, int, float]] = []      # (generation, cell id, value) for every insertion

    # -- insert ---------------------------------------------------------

    def better(self, a: float, b: float) -> bool:
        """True when ``a`` beats ``b`` under this archive's sense."""
        return a < b if self.sense == "min" else a > b

    def consider(self, design: Sequence[float], evaluation: Evaluation,
                 generation: int = 0) -> bool:
        """Offer one evaluated design to the archive. Returns True when it became (or improved) an elite."""
        self.n_considered += 1
        if not evaluation.valid:
            self.n_invalid += 1
            return False
        if evaluation.descriptors is None:
            self.n_unplaceable += 1
            return False
        cell = self.space.cell(evaluation.descriptors)
        if cell is None or not math.isfinite(float(evaluation.value)):
            self.n_unplaceable += 1
            return False
        current = self._elites.get(cell)
        if current is not None and not self.better(float(evaluation.value), current.value):
            self.n_not_better += 1
            return False
        self._elites[cell] = Elite(cell=cell, design=np.asarray(design, dtype=float),
                                   descriptors=np.asarray(list(evaluation.descriptors), dtype=float),
                                   value=float(evaluation.value),
                                   error={k: float(v) for k, v in (evaluation.error or {}).items()},
                                   note=evaluation.note)
        self.history.append((int(generation), self._cell_id(cell), float(evaluation.value)))
        return True

    def _cell_id(self, cell: Sequence[int]) -> int:
        return int(np.ravel_multi_index(tuple(int(i) for i in cell), self.space.shape))

    # -- read -----------------------------------------------------------

    @property
    def elites(self) -> list[Elite]:
        """The elites, in cell order (row-major), so two runs can be compared line by line."""
        return [self._elites[k] for k in sorted(self._elites, key=lambda c: self._cell_id(c))]

    @property
    def cells(self) -> int:
        return len(self._elites)

    def elite_at(self, cell: Sequence[int] | None) -> Elite | None:
        """The elite of one cell, or ``None`` when that cell is empty (or the cell is unplaceable)."""
        if cell is None:
            return None
        return self._elites.get(tuple(int(i) for i in cell))

    @property
    def fill(self) -> float:
        return self.cells / self.space.n_cells

    def best(self) -> Elite | None:
        """The elite the sense prefers across the whole map (the smallest cost, or the largest error)."""
        if not self._elites:
            return None
        return (min if self.sense == "min" else max)(self._elites.values(), key=lambda e: e.value)

    def ranked(self, n: int | None = None) -> list[Elite]:
        """Elites from best to worst under the sense."""
        out = sorted(self._elites.values(), key=lambda e: e.value, reverse=self.sense == "max")
        return out if n is None else out[:n]

    def grid_values(self) -> np.ndarray:
        """Values as an array of the behaviour shape, NaN where the cell is empty."""
        arr = np.full(self.space.shape, np.nan)
        for cell, elite in self._elites.items():
            arr[cell] = elite.value
        return arr

    def diversity(self, *, min_cells: int = 4, min_fill: float = 0.05, radius: float = 0.05,
                  max_dominant: float = 0.5) -> DiversityReport:
        """Measure the map: occupied cells, spread of the elite designs, and the largest cluster.

        The defaults are the floors this module treats as "a map and not a point": at least four cells, at least 5 %
        of the grid, a mean design distance of at least 0.05 in normalised units, and no cluster holding more than half
        of the elites.
        """
        elites = self.elites
        n = len(elites)
        problems: list[str] = []
        if n < 2:
            spread, dominant = 0.0, 1.0
        else:
            X = np.stack([e.design for e in elites])
            span = X.max(0) - X.min(0)
            Z = (X - X.min(0)) / np.where(span > 0, span, 1.0)
            D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
            spread = float(D[np.triu_indices(n, 1)].mean())
            counts = (D <= radius).sum(1)
            dominant = float(counts.max() / n)
        fill = self.fill
        if n < min_cells:
            problems.append(f"{n} cells filled, at least {min_cells} needed for a map")
        if fill < min_fill:
            problems.append(f"fill {fill:.1%} below the {min_fill:.0%} floor")
        if spread < 0.05:
            problems.append(f"elite designs spread {spread:.3f} over the normalised box: they are one design")
        if dominant > max_dominant:
            problems.append(f"{dominant:.0%} of the elites sit within {radius:g} of each other")
        return DiversityReport(cells=n, total_cells=self.space.n_cells, fill=fill, design_spread=spread,
                               dominant_share=dominant, radius=radius, ok=not problems,
                               problems=tuple(problems))

    def assert_diverse(self, **kwargs: Any) -> DiversityReport:
        """The diversity report, raising :class:`CollapsedArchiveError` when the archive is not a map."""
        report = self.diversity(**kwargs)
        if not report.ok:
            raise CollapsedArchiveError(f"archive {self.name or 'unnamed'} collapsed: {report.text()}")
        return report

    # -- report ---------------------------------------------------------

    def table(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Every elite as a flat dict: design, behaviour, value and channels -- one row per row of the map."""
        names = list(names) if names is not None else self.design_names
        rows = []
        for elite in self.elites:
            row = elite.as_dict(names)
            row["condition"] = self.space.describe(elite.cell)
            rows.append(row)
        return rows

    def summary(self, names: Sequence[str] | None = None) -> str:
        """The map in words: what was archived, how full it is, and the best cell."""
        head = (f"{self.name or 'archive'}: {self.cells}/{self.space.n_cells} cells (fill {self.fill:.0%}), "
                f"{self.n_considered} evaluated, {self.n_invalid} invalid, {self.n_unplaceable} unplaceable")
        best = self.best()
        if best is None:
            return head + "\n  empty"
        names = list(names) if names is not None else (
            list(self.design_names) if self.design_names else [f"x{i}" for i in range(len(best.design))])
        design = ", ".join(f"{n} = {v:.4g}" for n, v in zip(names, best.design, strict=True))
        sense = "lowest" if self.sense == "min" else "highest"
        return "\n".join([head, f"  {sense} value {best.value:.4g} ({design}) at {self.space.describe(best.cell)}"])

    def to_json(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        """JSON-able dump: the axes (with their reasons), the elites and the counters."""
        return {"name": self.name, "sense": self.sense, "labels": self.space.labels,
                "axes": [asdict(a) for a in self.space.axes], "bins": list(self.space.bins_per_axis),
                "cells": self.cells, "total_cells": self.space.n_cells, "fill": self.fill,
                "counters": {"considered": self.n_considered, "invalid": self.n_invalid,
                             "unplaceable": self.n_unplaceable, "not_better": self.n_not_better},
                "elites": self.table(names)}


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


@dataclass
class QDConfig:
    """Budget and operators of a quality-diversity run. Everything here is declared, so the run is repeatable.

    Parameters
    ----------
    budget:
        Evaluations of ``evaluate`` the run may spend, including the initial batch. The comparison against random and
        Latin hypercube sampling is only fair at an equal budget, so this is the number the report quotes.
    population:
        Candidates per generation (evaluated in one batch, which is also the parallel unit).
    mutation_std:
        Standard deviation of the Gaussian mutation, in **normalised** design units (each parameter scaled to its own
        box), so one number is meaningful across parameters of different magnitudes.
    line_share:
        Share of candidates that use a line mutation instead of a Gaussian one.
    line_step:
        Range of the line-mutation step, in normalised units: the child is ``parent + step * direction`` with
        ``step ~ U(line_step)`` and ``direction`` a unit vector, half the time the direction between two elites (a
        variation along the map, the ISO-line of CMA-ME) and half the time uniform on the sphere.
    immigrant_share:
        Share of each generation drawn uniformly in the box instead of mutated, so an unlucky archive can still be
        escaped.
    init:
        ``"latin"`` (default) or ``"random"`` for the first batch.
    init_fraction:
        Share of the budget spent on a space-filling first batch before the mutation loop starts. A map has to exist
        before it can be improved: with the default 0.5 the first half of the budget fills the grid evenly and the
        second half pushes each cell's elite towards its own extreme. At 0 the search only mutates, and a cell nobody
        lands in stays empty however good the hill-climbing is.
    seed:
        Seed of the generator; the whole run is a function of it.
    workers:
        Threads used to evaluate a batch. Only helps when ``evaluate`` releases the GIL (numpy, torch); the batch is
        collected in order, so the result does not depend on it.
    """

    budget: int = 600
    population: int = 50
    mutation_std: float = 0.08
    line_share: float = 0.35
    line_step: tuple[float, float] = (0.1, 0.5)
    immigrant_share: float = 0.1
    init: str = "latin"
    init_fraction: float = 0.5
    seed: int = 0
    workers: int = 1

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.population < 2:
            raise ValueError("population must be at least 2")
        if not 0.0 <= self.line_share <= 1.0 or not 0.0 <= self.immigrant_share <= 1.0:
            raise ValueError("line_share and immigrant_share are fractions")
        if not 0.0 <= self.init_fraction <= 1.0:
            raise ValueError("init_fraction is a fraction of the budget")
        if self.mutation_std <= 0 or self.init not in ("latin", "random"):
            raise ValueError("mutation_std must be positive and init 'latin' or 'random'")

    @classmethod
    def from_optimizer_config(cls, cfg: DesignOptimizerConfig, **overrides: Any) -> QDConfig:
        """Build from a :class:`DesignOptimizerConfig`, so a QD run and a GA run share the mutation scale."""
        return cls(mutation_std=float(cfg.mutation_std), population=int(cfg.population_size), **overrides)


def _design_box(bounds: ParamSpace | Mapping[str, tuple[float, float]]) -> tuple[tuple[str, ...], np.ndarray]:
    """Names and an (p, 2) bounds array from a ``ParamSpace`` or a plain mapping."""
    table = bounds.bounds if isinstance(bounds, ParamSpace) else dict(bounds)
    names = tuple(table)
    if not names:
        raise ValueError("the design box is empty")
    arr = np.array([table[n] for n in names], dtype=float)
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError("every bound must satisfy lo < hi")
    return names, arr


def latin_hypercube(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``n`` points, one per stratum of every parameter, jittered inside its stratum (McKay et al. 1979).

    Flat and space-filling in the box: the fair "spread the budget evenly" baseline, better than uniform random at
    covering the marginals.
    """
    p = len(bounds)
    u = (rng.permuted(np.tile(np.arange(n), (p, 1)), axis=1).T + rng.random((n, p))) / n
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def _update_best(archive: EliteArchive, best: tuple[Evaluation, np.ndarray] | None, x: np.ndarray,
                 evaluation: Evaluation) -> tuple[Evaluation, np.ndarray] | None:
    """Keep the best valid evaluation seen so far under ``archive``'s sense (the sampler's own maximum)."""
    if not evaluation.valid or not math.isfinite(float(evaluation.value)):
        return best
    if best is None or archive.better(float(evaluation.value), float(best[0].value)):
        return evaluation, np.asarray(x, dtype=float)
    return best


@dataclass
class QDSearchResult:
    """What a quality-diversity run did: the archive, the budget it spent, and the trajectory.

    ``best_seen`` is the best evaluation the run *produced*, whether or not its behaviour could be placed on the grid.
    It is the number to compare with a sampler at equal budget -- a case whose behaviour falls off the map is still a
    case the search found, and reporting only the archive's maximum would credit the archive's axes with a coverage
    limit they do not have.
    """

    archive: EliteArchive
    evaluations: int
    budget: int
    seed: int
    generations: int
    elapsed_s: float
    best_by_generation: list[float] = field(default_factory=list)
    best_seen: Evaluation | None = None
    best_seen_design: np.ndarray | None = None

    def summary(self) -> str:
        """The archive's own summary plus the cost of getting it."""
        return "\n".join([self.archive.summary(),
                          f"  {self.evaluations} evaluations of a {self.budget} budget, seed {self.seed}, "
                          f"{self.generations} generations, {self.elapsed_s:.2f} s"])


class QualityDiversitySearch:
    """MAP-Elites over a design box, a behaviour grid and any callable that evaluates one design.

    The loop is the canonical one: evaluate a first batch, then repeatedly pick a parent from a *uniformly chosen
    occupied cell*, mutate it (Gaussian or along a line), evaluate the child, and let it replace the elite of its own
    cell when it beats it. Selection is inside the cell, so the optimum of one behaviour never starves another --
    which is what keeps the archive a map.

    Parameters
    ----------
    archive:
        Where the elites are kept; its ``sense`` decides what "beats" means.
    bounds:
        The design box: a :class:`ParamSpace` or a ``{name: (lo, hi)}`` mapping. Designs are numpy rows in this order.
    evaluate:
        ``evaluate(x) -> Evaluation`` for one design row. It must be deterministic for a given ``x`` -- the archive
        compares values across generations, and a noisy evaluator would make the comparisons meaningless.
    cfg:
        Budget and operators (:class:`QDConfig`).
    """

    def __init__(self, archive: EliteArchive, bounds: ParamSpace | Mapping[str, tuple[float, float]],
                 evaluate: Callable[[np.ndarray], Evaluation], cfg: QDConfig | None = None) -> None:
        self.archive = archive
        self.names, self.bounds = _design_box(bounds)
        self.evaluate = evaluate
        self.cfg = cfg or QDConfig()
        self.rng = np.random.default_rng(int(self.cfg.seed))

    # -- helpers --------------------------------------------------------

    def _clip(self, X: np.ndarray) -> np.ndarray:
        """Clip rows of the normalised box back into it."""
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _from_normalised(self, Z: np.ndarray) -> np.ndarray:
        return self._clip(self.bounds[:, 0] + Z * (self.bounds[:, 1] - self.bounds[:, 0]))

    def _initial(self, n: int) -> np.ndarray:
        """``n`` space-filling points: the same Latin hypercube the baseline uses, so the two are comparable."""
        n = max(int(n), 1)
        if self.cfg.init == "latin":
            return latin_hypercube(n, self.bounds, self.rng)
        return self._clip(self.bounds[:, 0] + self.rng.random((n, len(self.bounds)))
                          * (self.bounds[:, 1] - self.bounds[:, 0]))

    def _parents(self, n: int) -> np.ndarray:
        """``n`` designs: mutations of uniformly chosen elites, plus the immigrant share drawn fresh."""
        elites = self.archive.elites
        if not elites:
            return self._initial(n)
        n_immigrant = int(round(self.cfg.immigrant_share * n))
        n_mutant = n - n_immigrant
        if n_mutant < 1:
            return self._initial(n)
        picks = self.rng.integers(0, len(elites), size=n_mutant)
        parents = self._normalise(np.stack([elites[i].design for i in picks]))
        children = parents.copy()
        gaussian = self.rng.random(n_mutant) >= self.cfg.line_share
        children[gaussian] += self.rng.normal(0.0, self.cfg.mutation_std, size=(int(gaussian.sum()), children.shape[1]))
        n_line = int((~gaussian).sum())
        if n_line:
            low, high = self.cfg.line_step
            steps = self.rng.uniform(low, high, size=(n_line, 1))
            # half of the lines point from one elite towards another: a variation *along* the map, which moves a child
            # between neighbouring behaviours instead of asking the same corner for a smaller step
            direction = self.rng.normal(size=(n_line, children.shape[1]))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            toward = self.rng.random(n_line) < 0.5
            if toward.any() and len(elites) > 1:
                a = self._normalise(np.stack([elites[i].design for i in self.rng.integers(0, len(elites), n_line)]))
                b = self._normalise(np.stack([elites[i].design for i in self.rng.integers(0, len(elites), n_line)]))
                steps_toward = b - a
                norm = np.linalg.norm(steps_toward, axis=1, keepdims=True)
                direction[toward] = np.divide(steps_toward[toward], norm[toward],
                                              out=direction[toward], where=norm[toward] > 1e-12)
            children[~gaussian] += steps * direction
        out = np.vstack([self._from_normalised(children), self._initial(n_immigrant)] if n_immigrant else
                        [self._from_normalised(children)])
        return out[:n]

    def _batch(self, X: np.ndarray) -> list[Evaluation]:
        if self.cfg.workers <= 1 or len(X) == 1:
            return [self.evaluate(x) for x in X]
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            return list(pool.map(self.evaluate, X))

    # -- the loop -------------------------------------------------------

    def run(self) -> QDSearchResult:
        """Spend the budget and return the archive with the numbers that produced it."""
        start = time.perf_counter()
        spent, generation = 0, 0
        n_init = int(min(self.cfg.budget, max(self.cfg.population,
                                              round(self.cfg.init_fraction * self.cfg.budget))))
        best_track = []
        best: tuple[Evaluation, np.ndarray] | None = None
        X = self._initial(n_init)
        while spent < self.cfg.budget:
            n = min(len(X), self.cfg.budget - spent)
            for x, evaluation in zip(X[:n], self._batch(X[:n]), strict=True):
                self.archive.consider(x, evaluation, generation=generation)
                best = _update_best(self.archive, best, x, evaluation)
            spent += n
            top = self.archive.best()
            best_track.append(float(top.value) if top is not None else float("nan"))
            generation += 1
            if spent < self.cfg.budget:
                X = self._parents(min(self.cfg.population, self.cfg.budget - spent))
        return QDSearchResult(archive=self.archive, evaluations=spent, budget=self.cfg.budget,
                              seed=self.cfg.seed, generations=generation,
                              elapsed_s=time.perf_counter() - start, best_by_generation=best_track,
                              best_seen=best[0] if best else None, best_seen_design=best[1] if best else None)


@dataclass
class BaselineSearch:
    """Random or Latin-hypercube sampling at a declared budget -- the comparison QD has to beat.

    The samples are offered to an archive of the same grid, so the *same* numbers come out of both: the best value in
    the budget, the cells reached, and the diversity of what was reached.
    """

    kind: str
    archive: EliteArchive
    best: Evaluation
    best_design: np.ndarray
    evaluations: int
    seed: int
    elapsed_s: float

    def summary(self) -> str:
        """One line: the best value, the cell it is in, and the coverage."""
        return (f"{self.kind} sampling: best {self.best.value:.4g} in {self.archive.cells} of "
                f"{self.archive.space.n_cells} cells ({self.archive.fill:.0%}), {self.evaluations} evaluations, "
                f"seed {self.seed}, {self.elapsed_s:.2f} s")


def uniform_search(bounds: ParamSpace | Mapping[str, tuple[float, float]],
                   evaluate: Callable[[np.ndarray], Evaluation], behaviour: BehaviourSpace, *,
                   budget: int = 600, seed: int = 0, kind: str = "random", sense: str = "min",
                   workers: int = 1) -> BaselineSearch:
    """Spend the same budget on ``kind`` in ("random", "latin") sampling and archive what it lands on.

    This is the honest opponent of the QD search: the same box, the same evaluator, the same number of evaluations,
    and the samples are placed in the same grid so the coverage and the failure set are comparable.
    """
    if kind not in ("random", "latin"):
        raise ValueError("kind must be 'random' or 'latin'")
    names, box = _design_box(bounds)
    rng = np.random.default_rng(int(seed))
    start = time.perf_counter()
    X = (latin_hypercube(budget, box, rng) if kind == "latin"
         else box[:, 0] + rng.random((budget, len(box))) * (box[:, 1] - box[:, 0]))
    archive = EliteArchive(behaviour, sense=sense, name=kind, design_names=names)
    evaluations = [evaluate(x) for x in X] if workers <= 1 else list(
        ThreadPoolExecutor(max_workers=workers).map(evaluate, X))
    best: tuple[Evaluation, np.ndarray] | None = None
    for x, evaluation in zip(X, evaluations, strict=True):
        archive.consider(x, evaluation)
        best = _update_best(archive, best, x, evaluation)
    if best is None:
        raise RuntimeError("every sample was invalid: the box and the validity rule do not overlap")
    return BaselineSearch(kind=kind, archive=archive, best=best[0], best_design=best[1], evaluations=len(X),
                          seed=int(seed), elapsed_s=time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Failure map: the same archive pointed at a model's error
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailingCell:
    """A cell whose elite error crosses the tolerance: the number, the design, and the condition it holds under."""

    cell: tuple[int, ...]
    error: float
    design: dict[str, float]
    condition: dict[str, tuple[float, float]]
    channels: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def text(self, axes: Sequence[BehaviourAxis] = ()) -> str:
        """One line: the error, the channels that broke, the design and the behaviour condition."""
        cond = "; ".join(f"{k} in [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in self.condition.items())
        ch = ", ".join(f"{k} {v:.1%}" for k, v in self.channels.items())
        des = ", ".join(f"{k} = {v:.4g}" for k, v in self.design.items())
        return f"{self.error:.1%}  {'(' + ch + ') ' if ch else ''}at {des}  |  {cond}"


@dataclass
class FailureRegion:
    """The cells of a failure map whose error crosses a tolerance, with the search that found them.

    This is the object the trust layer wants: not "the model can be wrong", but *where* it is wrong, *how much*, and
    what the case looks like -- the out-of-domain region, found by search and stated as intervals.
    """

    archive: EliteArchive
    tolerance: float
    quantity: str = "error"

    @property
    def map_max(self) -> float:
        """The largest elite error on the map (0 when empty)."""
        best = self.archive.best()
        return float(best.value) if best is not None else 0.0

    def cells(self) -> list[FailingCell]:
        """The failing cells, worst first. A cell is failing when its elite ``value`` is above the tolerance."""
        names = list(self.archive.design_names) if self.archive.design_names else [
            f"x{i}" for i in range(len(self.archive.elites[0].design) if self.archive.elites else 0)]
        out = []
        for elite in self.archive.ranked():
            if elite.value > self.tolerance:
                out.append(FailingCell(cell=elite.cell, error=float(elite.value),
                                       design={n: float(v) for n, v in zip(names, elite.design, strict=True)},
                                       condition=self.archive.space.condition(elite.cell),
                                       channels=dict(elite.error), note=elite.note))
        return out

    def failing_fraction(self) -> float:
        """Failing cells over filled cells -- how much of the mapped behaviour the model gets wrong."""
        return len(self.cells()) / self.archive.cells if self.archive.cells else 0.0

    def text(self, n: int | None = None) -> str:
        """The report: the map's headline, then the failing cells with their numbers and conditions."""
        failing = self.cells()
        head = [f"failure map of {self.quantity} ({self.archive.name or 'model'}): "
                f"{len(failing)} of {self.archive.cells} filled cells above {self.tolerance:.1%} "
                f"({self.failing_fraction():.0%}), worst {self.map_max:.1%}",
                f"  {self.archive.cells}/{self.archive.space.n_cells} cells filled ({self.archive.fill:.0%}), "
                f"{self.archive.n_invalid} cases refused as not physically valid"]
        rows = failing if n is None else failing[:n]
        return "\n".join(head + [f"  {c.text()}" for c in rows])

    def to_json(self) -> dict[str, Any]:
        """JSON-able form: the threshold, the map and the failing cells."""
        return {"quantity": self.quantity, "tolerance": self.tolerance, "map_max": self.map_max,
                "failing": [{"cell": list(c.cell), "error": c.error, "design": c.design,
                             "condition": {k: list(v) for k, v in c.condition.items()}, "channels": c.channels}
                            for c in self.cells()],
                "archive": self.archive.to_json()}


def find_failure_region(bounds: ParamSpace | Mapping[str, tuple[float, float]],
                        evaluate: Callable[[np.ndarray], Evaluation], behaviour: BehaviourSpace, *,
                        budget: int = 600, seed: int = 0, tolerance: float = 0.1,
                        cfg: QDConfig | None = None, name: str = "") -> tuple[FailureRegion, QDSearchResult]:
    """Run the archive with fitness = the model's error and return the region where it fails.

    ``evaluate`` returns the error as ``Evaluation.value`` (larger is worse), the *reference* behaviour as
    ``descriptors``, and ``valid=False`` for cases that are not physically legitimate -- the case set is decided by the
    physics, never by the model under test, or the search would only find its own refusals.

    Returns
    -------
    region, result
        The failure region over the archive, and the search that produced it (budget, seed, timings).
    """
    config = cfg or QDConfig(budget=budget, seed=seed)
    names, _ = _design_box(bounds)
    archive = EliteArchive(behaviour, sense="max", name=name or "model error", design_names=names)
    result = QualityDiversitySearch(archive, bounds, evaluate, config).run()
    return FailureRegion(archive=archive, tolerance=float(tolerance)), result


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------


def plot_archive(archive: EliteArchive, path: str | Path, *, tolerance: float | None = None,
                 title: str | None = None, cmap: str = "magma", dpi: int = 130,
                 unit: str = "") -> Path:
    """Draw the archive: one heat map of the elite value per cell, failing cells outlined.

    The map *is* the deliverable, so it is drawn from the archive and nothing else: empty cells stay blank (they say
    what was not explored), the elite of each cell is the colour, and with ``tolerance`` the failing cells are marked
    with their value. Two behaviour axes are required -- a map has two.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    space = archive.space
    if len(space.axes) != 2:
        raise ValueError("plot_archive draws two behaviour axes")
    values = archive.grid_values()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap_obj = plt.get_cmap(cmap).with_extremes(bad="0.92")
    mesh = ax.pcolormesh(values.T, cmap=cmap_obj, edgecolors="w", linewidth=0.4)
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label(f"{archive.name or 'value'}{(' [' + unit + ']') if unit else ''}")
    ax.set_xlabel(f"{space.axes[0].name} [{space.axes[0].unit}]" if space.axes[0].unit else space.axes[0].name)
    ax.set_ylabel(f"{space.axes[1].name} [{space.axes[1].unit}]" if space.axes[1].unit else space.axes[1].name)
    ax.set_xticks(np.arange(space.shape[0]) + 0.5,
                  [f"{space.axes[0].centre(i, space.shape[0]):.3g}" for i in range(space.shape[0])], fontsize=7)
    ax.set_yticks(np.arange(space.shape[1]) + 0.5,
                  [f"{space.axes[1].centre(i, space.shape[1]):.3g}" for i in range(space.shape[1])], fontsize=7)
    if tolerance is not None:
        for elite in archive.elites:
            if elite.value > tolerance:
                ax.text(elite.cell[0] + 0.5, elite.cell[1] + 0.5, f"{elite.value:.0%}", ha="center",
                        va="center", fontsize=6, color="w", fontweight="bold")
        ax.set_title((title or space.labels or "quality-diversity archive") +
                     f"\n{archive.cells}/{space.n_cells} cells, {len(archive.ranked())} elites, "
                     f"{sum(1 for e in archive.elites if e.value > tolerance)} above {tolerance:.0%}", fontsize=9)
    else:
        ax.set_title((title or space.labels or "quality-diversity archive") +
                     f"\n{archive.cells}/{space.n_cells} cells filled ({archive.fill:.0%})", fontsize=9)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
