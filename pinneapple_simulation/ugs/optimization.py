"""Operations optimisation: the peak daily deliverability of a storage season, searched on the reference model.

The problem (#1086)
-------------------
Over a withdrawal season, choose which wells deliver and how much, to maximise the **peak daily deliverability**
subject to a minimum bottom-hole pressure per well, a reservoir pressure floor, a maximum rate per well, a
surface capacity, and an end-of-season inventory.

The decision that is not obvious is *which wells to spend*. Peak deliverability on the peak day comes from every
well at its minimum bottom-hole pressure, so it depends on how much pressure each well still has around it: a
well that has produced a lot before the peak arrives with a depleted near-well region and delivers less. The
season therefore has a real trade-off -- the gas that must be produced before the peak should come from the
wells that matter least on the peak day -- and the answer is not uniform allocation unless the field happens to
be symmetric.

What is searched
----------------
The season is three periods: ``pre`` days at a fixed total rate, one **peak** day with every well on
bottom-hole-pressure control at the floor, and ``post`` days at a fixed total rate. The pre- and post-peak
rates are split by a weight vector on the simplex, and that vector is the decision. Every candidate is run on
the reference simulator of :mod:`pinneapple_simulation.ugs.simulator`, so **every recommendation is verified by
construction** -- there is no surrogate in the loop yet, and the docstrings and the returned record say so
instead of quoting a surrogate-predicted gain.

:func:`optimise_peak_deliverability` samples the simplex with a Latin hypercube, takes the best candidates and
refines them by moving weight between pairs of wells, and reports the verified gain over uniform allocation,
the constraint margins and -- computed, not quoted -- how many extra wells would buy the same gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .benchmark import UGSModel
from .simulator import SECONDS_PER_DAY, SimConfig
from .wells import Control, Period, Schedule, Well

__all__ = ["DeliverabilityProblem", "SeasonResult", "evaluate_allocation", "equal_allocation",
           "optimise_peak_deliverability", "uniform_baseline", "greedy_allocation"]


@dataclass
class DeliverabilityProblem:
    """The season and its constraints. Rates are m3 at standard conditions per second unless the name says day.

    ``capacity`` is the surface (compressor or line) limit for the whole field; ``p_floor`` the average
    reservoir pressure the season must not go below; the rest are the wells' own limits.
    """

    model: UGSModel
    days_pre: float = 20.0
    days_post: float = 4.0
    pre_rate: float = 2.5           # m3/s total before the peak
    post_rate: float = 2.5          # m3/s total after the peak
    capacity: float = 8.0           # m3/s total the surface can take
    p_floor: float | None = 7.0e6   # Pa, average pressure at the end of the season
    rate_max: float | None = 1.2    # m3/s per well
    bhp_min: float | None = 5.0e6   # Pa
    delivery_tolerance: float = 0.02   # how far the pre-peak delivery may sit from the demand

    def wells(self) -> list[Well]:
        """The model's wells with the problem's own limits applied."""
        out = []
        for w in self.model.wells:
            out.append(Well(w.name, w.cell, r_w=w.r_w, skin=w.skin,
                            bhp_min=self.bhp_min if self.bhp_min is not None else w.bhp_min,
                            bhp_max=w.bhp_max,
                            rate_max=self.rate_max if self.rate_max is not None else w.rate_max))
        return out

    def with_wells(self, wells: list[Well]) -> DeliverabilityProblem:
        model = UGSModel(grid=self.model.grid, fluid=self.model.fluid, rock=self.model.rock,
                         wells=wells, p0=self.model.p0, name=self.model.name)
        return DeliverabilityProblem(model, self.days_pre, self.days_post, self.pre_rate, self.post_rate,
                                     self.capacity, self.p_floor, self.rate_max, self.bhp_min,
                                     self.delivery_tolerance)


@dataclass
class SeasonResult:
    """One allocation, run to the end of the season on the reference simulator."""

    weights: dict[str, float]
    peak: float                       # m3/day, the deliverability on the peak day
    peak_uncapped: float              # before the surface capacity is applied
    feasible: bool
    reason: str = ""
    delivered_before_peak: float = 0.0    # m3 the pre-peak demand actually received
    target_before_peak: float = 0.0       # m3 the pre-peak demand asked for
    mean_pressure_end: float = 0.0
    well_rates: dict[str, float] = field(default_factory=dict)
    well_bhp: dict[str, float] = field(default_factory=dict)
    mass_balance_error: float = 0.0
    converged: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "peak_m3_per_day": self.peak,
            "peak_uncapped_m3_per_day": self.peak_uncapped,
            "feasible": self.feasible,
            "reason": self.reason,
            "delivered_before_peak_m3": self.delivered_before_peak,
            "target_before_peak_m3": self.target_before_peak,
            "mean_pressure_end_MPa": self.mean_pressure_end / 1e6,
            "well_rates_m3_per_day": dict(self.well_rates),
            "well_bhp_MPa": {k: v / 1e6 for k, v in self.well_bhp.items()},
            "mass_balance_error": self.mass_balance_error,
            "converged": self.converged,
        }


def equal_allocation(n: int) -> np.ndarray:
    """The baseline every operator starts from: the same share for every well."""
    if n <= 0:
        raise ValueError("an allocation needs at least one well")
    return np.full(int(n), 1.0 / int(n))


def greedy_allocation(problem: DeliverabilityProblem, *, rounds: int = 4,
                      config: SimConfig | None = None) -> np.ndarray:
    """Spend the pre-peak gas on the wells that deliver least on the peak day, one round at a time.

    Each round doubles the weight of the half of the wells with the smallest peak-day rate, which is the
    physical intuition written as an algorithm. It is a starting point for the search, not the answer: the
    search either improves on it or reports it unchanged.
    """
    n = len(problem.model.wells)
    weights = equal_allocation(n)
    for _ in range(int(rounds)):
        result = evaluate_allocation(problem, weights, config=config)
        if not result.converged:
            break
        weak = np.argsort([result.well_rates[w.name] for w in problem.model.wells])[: max(n // 2, 1)]
        weights = weights.copy()
        weights[weak] *= 2.0
        weights /= weights.sum()
    return weights


def _schedule(problem: DeliverabilityProblem, weights: dict[str, float], wells: list[Well]) -> Schedule:
    pre = {name: Control("rate", -problem.pre_rate * w) for name, w in weights.items() if w > 0.0}
    peak = {w.name: Control("bhp", problem.bhp_min if problem.bhp_min is not None else w.bhp_min)
            for w in wells}
    post = {name: Control("rate", -problem.post_rate * w) for name, w in weights.items() if w > 0.0}
    return Schedule((
        Period(problem.days_pre, pre, "pre-peak"),
        Period(1.0, peak, "peak day"),
        Period(problem.days_post, post, "post-peak"),
    ))


def evaluate_allocation(problem: DeliverabilityProblem, weights, *, config: SimConfig | None = None,
                        wells: list[Well] | None = None) -> SeasonResult:
    """Run one allocation through the whole season and read the peak day off the reference simulator."""
    wells = wells if wells is not None else problem.wells()
    names = [w.name for w in wells]
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(names),):
        raise ValueError(f"the allocation has {weights.size} entries for {len(names)} wells")
    if np.any(weights < 0.0):
        raise ValueError("allocation weights cannot be negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("the allocation sums to zero: nothing would be produced")
    weights = weights / total
    share = dict(zip(names, weights.tolist(), strict=True))

    model = problem.model
    sim_model = UGSModel(grid=model.grid, fluid=model.fluid, rock=model.rock, wells=wells, p0=model.p0,
                         name=model.name)
    sim = sim_model.simulator()
    config = config or SimConfig(dt_days=1.0)
    schedule = _schedule(problem, share, wells)
    result = sim.run(schedule, config)
    if not result.converged:
        return SeasonResult(weights=share, peak=0.0, peak_uncapped=0.0, feasible=False,
                            reason=f"the simulator did not converge: {result.reason}",
                            converged=False)
    # the last step of the pre-peak and of the peak period, whatever the time step
    steps = schedule.steps(config.dt_days)
    pre_index = sum(1 for _dt, period in steps if period == 0)
    peak_index = sum(1 for _dt, period in steps if period <= 1)
    if peak_index >= len(result.times_days):
        raise RuntimeError(f"the run stopped at {result.times_days[-1]:g} d, before the peak day ends")
    rates = {w.name: float(-result.well_rate[w.name][peak_index]) for w in wells}
    bhps = {w.name: float(result.well_bhp[w.name][peak_index]) for w in wells}
    uncapped = float(sum(rates.values()))
    peak = uncapped if problem.capacity is None else min(uncapped, problem.capacity * SECONDS_PER_DAY)
    mean_end = float(result.mean_pressure[-1])
    last = len(result.times_days) - 1
    delivered = float(-sum(result.well_cumulative[w.name][min(pre_index, last)] for w in wells))
    target = problem.pre_rate * problem.days_pre * SECONDS_PER_DAY
    short = abs(delivered - target) > problem.delivery_tolerance * target
    feasible = (problem.p_floor is None or mean_end >= problem.p_floor) and not short
    if short:
        reason = (f"the pre-peak demand asked for {target / 1e3:,.0f} e3 m3 and the wells delivered "
                  f"{delivered / 1e3:,.0f} e3 m3: the allocation is short of the demand")
    elif problem.p_floor is not None and mean_end < problem.p_floor:
        reason = (f"the mean pressure at the end of the season is {mean_end / 1e6:.2f} MPa, "
                  f"below the floor of {problem.p_floor / 1e6:.2f} MPa")
    else:
        reason = ""
    return SeasonResult(weights=share, peak=peak, peak_uncapped=uncapped, feasible=feasible, reason=reason,
                        delivered_before_peak=delivered, target_before_peak=target,
                        mean_pressure_end=mean_end, well_rates=rates, well_bhp=bhps,
                        mass_balance_error=float(result.mass_balance_error), converged=True)


def _objective(result: SeasonResult) -> float:
    return result.peak if result.feasible else -np.inf


def _simplex_design(n_dim: int, n_samples: int, seed: int) -> np.ndarray:
    """A Latin hypercube mapped to the simplex: ``n_dim - 1`` sorted strata gaps, uniform on the simplex."""
    from scipy.stats import qmc

    if n_dim == 1:
        return np.ones((1, 1))
    unit = qmc.LatinHypercube(d=n_dim - 1, seed=seed).random(n_samples)
    cuts = np.sort(unit, axis=1)
    spacings = np.diff(np.hstack([np.zeros((n_samples, 1)), cuts, np.ones((n_samples, 1))]), axis=1)
    return np.clip(spacings, 1e-6, None) / np.clip(spacings, 1e-6, None).sum(axis=1, keepdims=True)


def optimise_peak_deliverability(problem: DeliverabilityProblem, *, n_samples: int = 16, seed: int = 0,
                                 refine_rounds: int = 2, config: SimConfig | None = None,
                                 capital_search: int = 4) -> dict[str, Any]:
    """Search the allocation that maximises the verified peak deliverability.

    Latin hypercube over the simplex, then pairwise mass transfer in shrinking steps, each candidate run on the
    reference simulator; ``refine_rounds`` sets the evaluation budget of that refinement at
    ``n_samples + 4 n wells refine_rounds`` runs. Returns the verified optimum, the uniform baseline, the gain,
    the constraint margins and the number of extra wells that would buy the same gain.
    """
    n = len(problem.model.wells)
    names = [w.name for w in problem.model.wells]
    evaluated: dict[tuple[float, ...], SeasonResult] = {}
    evaluations = [0]

    def evaluate(w) -> SeasonResult:
        key = tuple(np.round(np.asarray(w, dtype=float), 9))
        if key not in evaluated:
            evaluated[key] = evaluate_allocation(problem, w, config=config)
            evaluations[0] += 1
        return evaluated[key]

    base = evaluate(equal_allocation(n))
    greedy = greedy_allocation(problem, config=config)
    # the corners of the simplex -- everything spent on one well -- are the extremes the search has to compare,
    # and the greedy start is the physical intuition; the Latin hypercube fills in between them
    corners = [np.eye(n)[i] for i in range(n)]
    designs = _simplex_design(n, n_samples, seed)
    candidates = [equal_allocation(n), greedy, *corners] + ([*designs] if len(designs) else [])
    results = [evaluate(w) for w in candidates]
    feasible = [r for r in results if r.feasible]
    if not feasible:
        return {
            "ok": False,
            "reason": "no allocation produced a feasible season: " + (results[0].reason or "unknown"),
            "baseline": base.as_dict(),
            "evaluations": evaluations[0],
        }
    best = max(feasible, key=_objective)
    step = 0.5 * float(np.max(list(best.weights.values()))) if n > 1 else 0.0
    budget = n_samples + 4 * n * max(int(refine_rounds), 1)
    while evaluations[0] < budget and step > 1e-3:
        improved = False
        for i in range(n):
            for j in range(n):
                if evaluations[0] >= budget:
                    break
                if i == j or best.weights[names[j]] < step:
                    continue
                trial = np.array([best.weights[k] for k in names])
                trial[i] += step
                trial[j] -= step
                result = evaluate(np.clip(trial, 0.0, None))
                if _objective(result) > _objective(best) + 1e-9:
                    best = result
                    improved = True
        if not improved:
            step *= 0.5
    gain = (best.peak - base.peak) / base.peak if base.peak > 0 else 0.0
    verified = evaluate_allocation(problem, [best.weights[k] for k in names], config=config)
    evaluations[0] += 1
    margins = _margins(problem, verified)
    capital = _extra_wells_equivalent(problem, base, best.peak, config=config, max_wells=capital_search)
    return {
        "ok": True,
        "baseline": base.as_dict(),
        "optimum": verified.as_dict(),
        "gain": float(gain),
        "evaluations": int(evaluations[0]),
        "verified_by_reference_simulator": True,
        "note": "every candidate, the baseline and the reported optimum were run on the reference simulator; "
                "there is no surrogate in this loop yet (#1084/#1085), so no gain is surrogate-predicted",
        "constraint_margins": margins,
        "extra_wells_equivalent": capital,
        "search": {"n_samples": n_samples, "refine_rounds": refine_rounds, "seed": seed,
                   "feasible_candidates": len(feasible)},
    }


def _margins(problem: DeliverabilityProblem, result: SeasonResult) -> dict[str, Any]:
    """How far the verified optimum sits from each constraint."""
    lowest = min(result.well_bhp.values()) if result.well_bhp else float("nan")
    highest_rate = max(result.well_rates.values()) if result.well_rates else 0.0
    out: dict[str, Any] = {
        "lowest_bhp_MPa": lowest / 1e6,
        "bhp_min_MPa": (problem.bhp_min / 1e6) if problem.bhp_min else None,
        "highest_well_rate_m3_per_day": highest_rate,
        "rate_max_m3_per_day": problem.rate_max * SECONDS_PER_DAY if problem.rate_max else None,
        "peak_over_capacity": result.peak_uncapped / (problem.capacity * SECONDS_PER_DAY)
        if problem.capacity else None,
        "mean_pressure_end_MPa": result.mean_pressure_end / 1e6,
        "p_floor_MPa": problem.p_floor / 1e6 if problem.p_floor else None,
    }
    if problem.bhp_min:
        out["bhp_margin_MPa"] = (lowest - problem.bhp_min) / 1e6
    if problem.p_floor:
        out["pressure_margin_MPa"] = (result.mean_pressure_end - problem.p_floor) / 1e6
    if problem.rate_max:
        out["rate_margin_m3_per_day"] = problem.rate_max * SECONDS_PER_DAY - highest_rate
    return out


def _extra_wells_equivalent(problem: DeliverabilityProblem, baseline: SeasonResult, target: float, *,
                            config: SimConfig | None, max_wells: int = 3) -> dict[str, Any]:
    """How many extra wells, uniformly used, would deliver the same peak: computed by adding them and running.

    This is the capital comparison the issue asks for, and it is measured, not quoted: each well added is a real
    completion in a real cell of the same model, run on the same simulator with the same season.
    """
    model = problem.model
    if target <= baseline.peak:
        return {"wells": 0, "peak_m3_per_day": baseline.peak,
                "note": "the allocation gain is not positive, so no extra well is its equivalent"}
    used = {w.cell for w in model.wells}
    shape = model.grid.shape
    nx, ny = shape[0], shape[1]
    spare = [int(np.ravel_multi_index((i, j), (nx, ny))) for j in range(1, ny, max(ny // 4, 1))
             for i in range(1, nx, max(nx // 4, 1))]
    spare = [c for c in spare if c not in used]
    wells = list(model.wells)
    first_extra: float | None = None
    for k in range(1, int(max_wells) + 1):
        if not spare:
            break
        cell = spare.pop(0)
        wells.append(Well(f"X{k}", cell, r_w=0.1, bhp_min=model.wells[0].bhp_min,
                          bhp_max=model.wells[0].bhp_max, rate_max=model.wells[0].rate_max))
        extended = problem.with_wells(wells)
        result = evaluate_allocation(extended, equal_allocation(len(wells)), config=config)
        if k == 1 and result.feasible:
            first_extra = result.peak
        if result.feasible and result.peak >= target:
            out = {"wells": k, "peak_m3_per_day": result.peak,
                   "note": f"{k} extra well(s), uniformly used, reach the optimised peak"}
            if first_extra is not None and first_extra > baseline.peak:
                out["fraction_of_a_well"] = (target - baseline.peak) / (first_extra - baseline.peak)
            return out
    return {"wells": None, "peak_m3_per_day": None,
            "note": f"more than {max_wells} extra wells would be needed; the search stopped there"}


def uniform_baseline(problem: DeliverabilityProblem, *, config: SimConfig | None = None) -> SeasonResult:
    """The baseline schedule: the same share for every well, run on the reference simulator."""
    return evaluate_allocation(problem, equal_allocation(len(problem.model.wells)), config=config)
