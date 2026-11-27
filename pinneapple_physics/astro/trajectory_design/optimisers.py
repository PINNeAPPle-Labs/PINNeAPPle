"""Evolutionary design of interplanetary trajectories: NSGA-II, differential evolution, CMA-ES when it is installed,
and a MAP-Elites map, all over the patched-conic problems of :mod:`.gtop`.

The evolutionary machinery is **not** rewritten here. ``pinneapple_design.design_optimizer`` already implements
NSGA-II survivor selection (fast non-dominated sorting, crowding distance, Deb's constrained domination) inside
:class:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer`, differential evolution through
SciPy on Tchebycheff scalarisations, the evaluation bookkeeping (one record per design, budget accounting,
feasibility), and MAP-Elites in :class:`~pinneapple_design.design_optimizer.quality_diversity.QualityDiversitySearch`.
This module is the thin layer that says what a *trajectory* problem is -- the decision vector of a planet sequence,
the impulse budget, the flight time and the arrival speed -- so those optimisers can be pointed at it without a
second evolutionary implementation in the library. It re-exports :class:`EvolutionaryDesignOptimizer` so a caller
who wants the ask/tell loop directly can have it.

CMA-ES is used when the optional ``cma`` package is installed and the run is single-objective
(:meth:`EvolutionaryDesignOptimizer.tell` turns it off for multi-objective runs, because survivor selection would
otherwise be bypassed); :func:`cmaes_available` says whether it is, and :meth:`TrajectorySearch.engine` says which
engine a run actually used -- so a report never claims CMA-ES when the run was a genetic algorithm.

    from pinneapple_physics.astro.trajectory_design import cassini1, optimise
    search = optimise(cassini1(), method="nsga2", budget=20_000, seed=0)
    search.best, search.pareto[0], search.gap()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer import methods as _methods
from pinneapple_design.design_optimizer.optimizer import (  # noqa: F401 - re-exported on purpose
    DesignOptimizerConfig,
    EvolutionaryDesignOptimizer,
)
from pinneapple_design.design_optimizer.pareto import (  # noqa: F401 - re-exported on purpose
    compute_pareto_front,
    crowding_distance,
    fast_non_dominated_sort,
)
from pinneapple_design.design_optimizer.quality_diversity import (
    BehaviourAxis,
    BehaviourSpace,
    EliteArchive,
    Evaluation,
    QDConfig,
    QDSearchResult,
    QualityDiversitySearch,
)
from pinneapple_design.design_optimizer.study import (
    Constraint,
    DesignProblem,
    DesignSpace,
    Objective,
    Variable,
)

from .gtop import PUBLISHED_BEST, MGAEvaluator, MGAProblem

__all__ = ["CMAES_AVAILABLE", "cmaes_available", "build_problem", "TrajectorySearch", "optimise", "map_elites",
           "MapElitesResult", "EvolutionaryDesignOptimizer", "fast_non_dominated_sort", "crowding_distance",
           "compute_pareto_front"]

#: Methods the driver exposes, and what each is: the keys are those of
#: :data:`pinneapple_design.design_optimizer.methods.METHODS`.
METHODS = {"nsga2": "NSGA-II (non-dominated sorting, crowding distance, constrained domination)",
           "scipy-de": "differential evolution on Tchebycheff scalarisations (SciPy)",
           "random": "Latin hypercube of the whole budget (the baseline)"}


def cmaes_available() -> bool:
    """Whether the optional ``cma`` package is installed (CMA-ES is used for single-objective runs when it is)."""
    try:
        import cma  # noqa: F401
    except ImportError:
        return False
    return True


CMAES_AVAILABLE = cmaes_available()


def build_problem(problem: MGAProblem, include_time: bool = False,
                  flyby_tolerance_km_s: float | None = None) -> DesignProblem:
    """Turn an :class:`~.gtop.MGAProblem` into a ``pinneapple_design`` problem.

    ``include_time`` adds the total flight time as a second objective, which is what makes the Pareto front of the
    issue (propellant against time of flight). ``flyby_tolerance_km_s`` adds the free-flyby constraint, for the run
    of a problem whose flybys are meant to cost nothing.
    """
    seq = problem.sequence
    variables = [Variable("t0", seq.t0_bounds[0], seq.t0_bounds[1], unit="d", description="departure epoch, MJD2000")]
    for k, (lo, hi) in enumerate(seq.tof_bounds):
        variables.append(Variable(f"tof_{k + 1}", float(lo), float(hi), unit="d",
                                  description=f"time of flight of leg {k + 1}: {seq.bodies[k]} -> {seq.bodies[k + 1]}"))
    objectives = [Objective("dv", unit="km/s", label="total impulse budget")]
    if include_time:
        objectives.append(Objective("tof", unit="d", label="total flight time"))
    constraints = []
    if flyby_tolerance_km_s is not None:
        constraints.append(Constraint("flyby_mismatch", upper=float(flyby_tolerance_km_s), unit="km/s",
                                      label="every flyby must be free (relative velocity matched)"))
    evaluator = MGAEvaluator(problem, include_time=include_time)
    return DesignProblem(name=seq.name, space=DesignSpace(variables), objectives=objectives,
                         constraints=constraints, evaluator=evaluator,
                         description=f"{'-'.join(b[:3] for b in seq.bodies)} patched conics; {problem.source}")


@dataclass
class TrajectorySearch:
    """The result of one evolutionary run on a trajectory problem.

    Attributes
    ----------
    problem:
        The :class:`~.gtop.MGAProblem` the run optimised.
    result:
        The ``OptimizationResult`` of ``pinneapple_design.design_optimizer.methods.run``: every evaluation with its
        cost, the Pareto indices and the budget accounting.
    design_problem:
        The ``DesignProblem`` the optimiser saw, kept so the recorded rows can be read back with their names.
    """

    problem: MGAProblem
    result: Any
    design_problem: DesignProblem
    include_time: bool = False

    @property
    def method(self) -> str:
        return self.result.method

    @property
    def n_evaluations(self) -> int:
        return int(self.result.n_evals)

    @property
    def budget(self) -> int:
        return int(self.result.budget)

    @property
    def engine(self) -> str:
        """Which engine the run actually used: CMA-ES only when it is installed *and* the run is single-objective."""
        if self.result.method == "nsga2":
            if cmaes_available() and not self.include_time:
                return "NSGA-II survivor selection with a CMA-ES proposal distribution (cma installed)"
            return "NSGA-II (genetic algorithm population; no CMA-ES proposal)"
        return METHODS.get(self.result.method, self.result.method)

    def best(self) -> dict:
        """The best feasible design: decision vector, impulse budget, flight time and its components."""
        i = self.result.best(0)
        if i is None:
            return {"feasible": False, "reason": "no feasible design was found"}
        ev = self.result.evaluations
        row = ev.rows(self.design_problem.space, [int(i)])[0]
        design = self.problem.decode(ev.X[i])
        return {**row, **{f"component_{k}": v for k, v in design.dv_components.items() if k != "flybys"},
                "dv_flybys_km_s": list(design.dv_flyby_km_s)}

    def best_design(self):
        """The decoded :class:`~.gtop.MGADesign` of the best feasible decision vector (for the trust check)."""
        i = self.result.best(0)
        if i is None:
            raise RuntimeError("no feasible design was found")
        return self.problem.decode(self.result.evaluations.X[i])

    def pareto(self) -> list[dict]:
        """The non-dominated designs, each with its objectives and decision vector, sorted by impulse."""
        ev = self.result.evaluations
        idx = self.result.pareto()
        if ev is None or not len(idx):
            return []
        rows = ev.rows(self.design_problem.space, idx)
        for r in rows:
            r["dv"] = float(r["dv"])
            r["tof"] = float(r["tof"])
        return sorted(rows, key=lambda r: r["dv"])

    def published(self) -> dict | None:
        """The published best of this problem, when the module has one, with its source."""
        return PUBLISHED_BEST.get(self.problem.name)

    def gap(self) -> dict:
        """Measured best against the published best: absolute and relative gap, and which numbers they are.

        A gap is reported, never assumed: ``measured`` is what this run's budget produced, ``published`` is quoted
        from the source, and ``relative`` is ``measured / published - 1`` (positive means the run is worse).
        """
        best = self.best()
        pub = self.published()
        out = {"measured_km_s": float(best.get("dv", float("nan"))), "published_km_s": None,
               "absolute_km_s": None, "relative": None, "budget": self.n_evaluations, "method": self.method}
        if pub is not None:
            out["published_km_s"] = float(pub["dv_km_s"])
            out["absolute_km_s"] = out["measured_km_s"] - out["published_km_s"]
            out["relative"] = out["measured_km_s"] / out["published_km_s"] - 1.0
            out["source"] = pub["source"]
        return out

    def summary(self) -> str:
        best = self.best()
        gap = self.gap()
        lines = [f"{self.problem.name}: {self.method}, {self.n_evaluations} evaluations of a {self.budget} budget"]
        if not best.get("feasible", True):
            # a constrained run can spend its whole budget without one design satisfying every constraint; say that
            # instead of indexing a row that is not there
            lines.append(f"  no feasible design was found: {best.get('reason', 'the constraints were not met')}")
        else:
            lines.append(f"  best feasible: {best['dv']:.4f} km/s in {best['tof']:.1f} d, "
                         f"arrival v_inf {best.get('arrival_v_inf', float('nan')):.3f} km/s")
        lines.append(f"  Pareto front: {len(self.pareto())} non-dominated designs")
        if gap["published_km_s"] is not None and gap["measured_km_s"] == gap["measured_km_s"]:
            lines.append(f"  published best {gap['published_km_s']:.4f} km/s; measured is "
                         f"{gap['absolute_km_s']:+.4f} km/s ({gap['relative']:+.2%})")
        return "\n".join(lines)


def optimise(problem: MGAProblem, method: str = "nsga2", budget: int = 5_000, seed: int = 0,
             include_time: bool = False, flyby_tolerance_km_s: float | None = None, **options) -> TrajectorySearch:
    """Run one evolutionary method on a trajectory problem.

    Parameters
    ----------
    problem:
        The :class:`~.gtop.MGAProblem`.
    method:
        One of :data:`METHODS` (``"nsga2"``, ``"scipy-de"``, ``"random"``); any other key of
        ``pinneapple_design.design_optimizer.methods.METHODS`` is passed through as well.
    budget:
        Evaluations to spend, counted per design (see the ``methods`` module for what a finite-difference gradient
        costs). One evaluation is one patched-conic sequence, which at Cassini1's six bodies is about 2 ms.
    seed:
        Seed of the run; the whole run is a function of it, so two calls with the same seed agree exactly.
    include_time:
        Adds the total flight time as a second objective (Pareto front of impulse against time).
    flyby_tolerance_km_s:
        Adds the free-flyby constraint at that tolerance.
    options:
        Passed to the method (e.g. ``population=60`` for NSGA-II).

    Returns
    -------
    TrajectorySearch
    """
    dp = build_problem(problem, include_time=include_time, flyby_tolerance_km_s=flyby_tolerance_km_s)
    result = _methods.run(dp, method, budget=int(budget), seed=int(seed), **options)
    return TrajectorySearch(problem=problem, result=result, design_problem=dp, include_time=include_time)


@dataclass
class MapElitesResult:
    """A MAP-Elites archive over (flight time, impulse, arrival speed), and what it cost.

    Attributes
    ----------
    archive:
        The ``EliteArchive``: one elite per occupied cell, with the counters of what it refused.
    search:
        The ``QDSearchResult`` of the run: evaluations spent, generations, best seen.
    space:
        The behaviour grid the map was drawn on.
    axes_sample:
        The probe of the design box the behaviour ranges were measured on, so the map's limits are a measurement and
        not a guess (report it with the map).
    """

    archive: EliteArchive
    search: QDSearchResult
    space: BehaviourSpace
    problem: MGAProblem
    axes_sample: int = 0

    @property
    def fill(self) -> float:
        return float(self.archive.fill)

    def cells(self) -> list[dict]:
        """Every occupied cell: the behaviour interval it stands for, the elite's numbers and its decision vector."""
        return self.archive.table(list(self.problem.sequence.names))

    def summary(self) -> str:
        return "\n".join([f"{self.problem.name}: MAP-Elites over "
                          f"{' x '.join(f'{a.name}[{a.lo:g},{a.hi:g}]' for a in self.space.axes)}",
                          f"  {self.fill:.0%} of {self.space.n_cells} cells filled, "
                          f"{self.search.evaluations} evaluations, ranges measured on {self.axes_sample} designs",
                          self.archive.summary(list(self.problem.sequence.names))])


def map_elites(problem: MGAProblem, budget: int = 1_200, seed: int = 0, bins: tuple[int, int, int] = (8, 8, 6),
               axes: tuple[BehaviourAxis, ...] | None = None, init_fraction: float = 0.5) -> MapElitesResult:
    """MAP-Elites over (flight time, impulse budget, arrival speed).

    The three axes are *outcomes* of the trajectory, not design variables: a cell holds the cheapest mission found
    that flies for that long and arrives with that speed, so the map is a catalogue of different missions rather than
    one optimum. The search is the library's
    :class:`~pinneapple_design.design_optimizer.quality_diversity.QualityDiversitySearch` (MAP-Elites: a parent per
    occupied cell, mutation inside the cell), with the trajectory evaluator plugged in.

    Parameters
    ----------
    problem:
        The :class:`~.gtop.MGAProblem`.
    budget:
        Evaluations to spend.
    seed:
        Seed of the run.
    bins:
        Cells per axis.
    axes:
        Override the axes; the defaults span the problem's own bounds.
    init_fraction:
        Share of the budget spent on a space-filling first batch.

    Returns
    -------
    MapElitesResult
    """
    seq = problem.sequence
    lo, hi = problem.lo, problem.hi
    tof_lo, tof_hi = float(np.sum([b[0] for b in seq.tof_bounds])), float(np.sum([b[1] for b in seq.tof_bounds]))
    n_probe = int(max(16, 2 * len(lo)))
    probe = np.random.default_rng(int(seed) + 7919).uniform(lo, hi, size=(n_probe, len(lo)))
    probed = [problem.decode(x) for x in probe]
    ok = [d for d in probed if d.feasible]
    if axes is None:
        dv_hi = float(np.max([d.dv_total_km_s for d in ok])) if ok else 60.0
        vi_hi = float(np.max([d.arrival_v_inf_km_s for d in ok])) if ok else 40.0
        axes = (
            BehaviourAxis("flight time", tof_lo, tof_hi, unit="d",
                          why="how long the mission takes: the cost a designer trades against the impulse"),
            BehaviourAxis("impulse", 0.0, float(np.ceil(dv_hi)), unit="km/s",
                          why="the budget the trajectory needs; a cell keeps the cheapest mission of that behaviour"),
            BehaviourAxis("arrival speed", 0.0, float(np.ceil(vi_hi)), unit="km/s",
                          why="how fast the spacecraft arrives: it sets the capture impulse, which the budget above "
                              "does not charge"),
        )
    space = BehaviourSpace(axes=axes, bins=bins, labels=f"{seq.name}: one mission per cell")
    archive = EliteArchive(space, sense="min", name=seq.name,
                           design_names=tuple(seq.names))

    def evaluate(x: np.ndarray) -> Evaluation:
        d = problem.decode(x)
        return Evaluation(value=float(d.dv_total_km_s),
                          descriptors=(float(d.tof_days), float(d.dv_total_km_s), float(d.arrival_v_inf_km_s)),
                          error={"departure": float(d.dv_departure_km_s), "arrival": float(d.dv_arrival_km_s)},
                          valid=bool(d.feasible), note=d.reason)

    cfg = QDConfig(budget=int(budget), seed=int(seed), init_fraction=float(init_fraction))
    search = QualityDiversitySearch(archive, {"t0": (lo[0], hi[0]),
                                              **{f"tof_{k + 1}": (lo[k + 1], hi[k + 1])
                                                 for k in range(seq.n_legs)}}, evaluate, cfg).run()
    return MapElitesResult(archive=archive, search=search, space=space, problem=problem, axes_sample=n_probe)
