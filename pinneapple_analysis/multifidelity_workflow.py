"""Two data levels in one workflow: the cheap model everywhere, the expensive solver where it pays (#35).

The pieces were already here and each one answered its own question. :class:`pinneapple_data.co_kriging.CoKriging`
fits the Kennedy-O'Hagan model ``z_high = rho * z_low + delta`` with the discrepancy learned rather than assumed;
:func:`pinneapple_data.co_kriging.study` measures what that buys over a single-fidelity Gaussian process at the same
count of expensive runs; :class:`pinneapple_data.multifidelity.FidelityPair` pairs the two levels of a table; and
:func:`pinneapple_analysis.fidelity.serve` turns a surrogate into a route, answering each request with the model or
with the solver and writing down why.

What was missing is the **workflow**: one object that takes the two levels as the *functions they are*, spends a
budget on them, and answers the question a study is actually for -- *does the cheap level pay for itself?* -- with a
number instead of a preference. :class:`MultiFidelityWorkflow` is that object:

* the expensive designs are a **nested subset** of the cheap ones (:meth:`FidelityPlan.designs`), so every expensive
  run has a cheap value at the same design and the discrepancy is measured rather than interpolated;
* two models are fitted on the same data -- the co-kriging model and a single-fidelity Gaussian process on the
  expensive runs alone -- and both are scored when an oracle is given;
* the comparison is made at **equal cost**, which is the only comparison that decides anything. One expensive run
  costs ``1 / cost_ratio`` cheap ones, so a plan with ``n_high`` expensive and ``n_low`` cheap runs has spent
  ``n_high + cost_ratio * n_low`` expensive-run equivalents, and the honest baseline is a single-fidelity GP given
  *that* many expensive runs. Comparing against a GP with only ``n_high`` runs flatters co-kriging by handing it free
  information;
* the verdict is stated as a **break-even**: the largest ``cost_ratio`` at which the cheap level still pays. Above it
  the workflow says the multi-fidelity model does not pay and the budget belongs to the expensive level;
* :meth:`MultiFidelityRun.serves` puts the fitted model behind :func:`~pinneapple_analysis.fidelity.serve` with the
  expensive level as the escalation solver, so what the workflow fits is what a service answers with.

The measured case, and the number that does not flatter
------------------------------------------------------
:func:`rod_fidelities` is a real pair of solvers: the same one-dimensional Poisson problem solved by ``fem`` at two
mesh resolutions -- the coarse case is the cheap level, the fine one the expensive level -- and the quantity of
interest is the peak of the field. Measured on this machine (``2026-11-25``), 10 expensive designs, 64 cheap ones and
120 held-out expensive designs scored against a ten-times-finer solve:

==========================  ==============================  ==============
model                     relative error on held-out designs gain
==========================  ==============================  ==============
co-kriging, 10 + 64       7.8e-2                           --
high-fidelity only, 10    1.8e-1                           2.28 x
high-fidelity only, 11    1.1e-1                           1.36 x
high-fidelity only, 12    5.7e-2                           0.74 x
==========================  ==============================  ==============

So the cheap level more than doubles the accuracy at the same number of expensive runs, and it **stops paying once two
more expensive runs are on the table**: the break-even is one extra expensive run, a cost ratio of 1/64 = 0.016, i.e.
one expensive run has to cost more than about sixty coarse ones for the coarse runs to be worth buying. A coarse 2-D
CFD case against a fine one is well inside that; refining a mesh by a factor of two is not, and the workflow says so.
Both numbers are asserted in ``tests/test_multifidelity_workflow.py``; the docstring of
:func:`benchmark_multifidelity_workflow` carries the table the benchmark prints.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pinneapple_analysis.fidelity import FidelityPolicy, FidelityReport, serve
from pinneapple_data.co_kriging import CoKriging, GaussianProcess

__all__ = ["FidelityPlan", "MultiFidelityRun", "MultiFidelityWorkflow", "rod_fidelities",
           "benchmark_multifidelity_workflow", "VERDICTS"]

#: What the workflow can conclude about the cheap level.
VERDICTS = ("pays", "does_not_pay", "undecided")


@dataclass(frozen=True)
class FidelityPlan:
    """How the two levels are paid for.

    ``n_high`` expensive runs and ``n_low`` cheap ones, with ``cost_ratio`` = what one cheap run costs as a fraction
    of one expensive run (``0.02`` says a coarse case is 1/50 of a fine one). ``total_cost`` is the plan in
    expensive-run equivalents -- the only currency in which the two models can be compared.
    """

    n_high: int = 10
    n_low: int = 60
    cost_ratio: float = 0.02
    seed: int = 0

    def __post_init__(self):
        if int(self.n_high) < 3:
            raise ValueError(f"n_high must be at least 3 (a discrepancy needs more than two runs), got {self.n_high!r}")
        if int(self.n_low) < int(self.n_high):
            raise ValueError(f"n_low={self.n_low} is smaller than n_high={self.n_high}: the expensive designs are a "
                             "subset of the cheap ones, so the cheap level cannot have fewer runs")
        if not 0.0 < float(self.cost_ratio) <= 1.0:
            raise ValueError(f"cost_ratio is a fraction between 0 and 1 (one cheap run / one expensive run), got "
                             f"{self.cost_ratio!r}")
        object.__setattr__(self, "n_high", int(self.n_high))
        object.__setattr__(self, "n_low", int(self.n_low))

    @property
    def total_cost(self) -> float:
        """The plan in expensive-run equivalents."""
        return self.n_high + self.cost_ratio * self.n_low

    def designs(self, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(design_low, design_high)``: a Sobol' design of ``n_low`` points, of which the first ``n_high`` are the
        expensive ones. Nested on purpose -- the discrepancy is the difference between two values measured at the same
        design, not between a value and an interpolation."""
        from scipy.stats import qmc
        bounds = np.atleast_2d(np.asarray(bounds, dtype=np.float64))
        n = 2 ** int(np.ceil(np.log2(max(self.n_low, 2))))
        unit = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=self.seed).random(n)[:self.n_low]
        low = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        return low, low[: self.n_high]

    def to_dict(self) -> dict[str, Any]:
        return {"n_high": self.n_high, "n_low": self.n_low, "cost_ratio": self.cost_ratio,
                "total_cost": self.total_cost, "seed": self.seed}


@dataclass
class MultiFidelityRun:
    """The fitted models and the data behind them, with the two comparisons the workflow exists to make."""

    plan: FidelityPlan
    design_low: np.ndarray
    values_low: np.ndarray
    design_high: np.ndarray
    values_high: np.ndarray
    low_at_high: np.ndarray
    co_kriging: CoKriging
    high_only: GaussianProcess
    seconds: float = 0.0
    error_co_kriging: float | None = None
    error_high_only: float | None = None
    error_low: float | None = None
    n_test: int = 0

    @property
    def rho(self) -> float:
        return float(self.co_kriging.rho)

    @property
    def gain(self) -> float | None:
        """High-fidelity-only error divided by the co-kriging one, at the same number of expensive runs.

        ``None`` when no oracle was given -- and deliberately **not** the equal-cost gain: that one depends on the
        plan's ``cost_ratio`` and is what :func:`break_even_cost_ratio` measures separately, because a gain computed
        from a cost the user declared is only as good as the declaration.
        """
        if self.error_co_kriging is None or self.error_high_only is None or not self.error_co_kriging:
            return None
        return float(self.error_high_only / self.error_co_kriging)

    def verdict(self) -> str:
        """Whether the cheap level paid: decided on the equal-*count* gain, since the equal-*cost* one needs a declared
        price. ``undecided`` when no oracle was given, which is the honest answer to a workflow run without truth."""
        gain = self.gain
        if gain is None:
            return "undecided"
        return "pays" if gain > 1.0 else "does_not_pay"

    # ------------------------------------------------------------------ what the model answers with
    def predict(self, design, *, use_low: bool = True) -> np.ndarray:
        """The co-kriging mean at ``design``, with the cheap level's own value there when ``use_low``."""
        design = np.atleast_2d(np.asarray(design, dtype=np.float64))
        low_values = self._low(design) if use_low else None
        return np.asarray(self.co_kriging.predict(design, low_values=low_values, return_std=False)).ravel()

    def interval(self, design, *, z: float = 1.96, use_low: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, lower, upper)`` for :func:`~pinneapple_analysis.fidelity.serve`, whose tolerance is checked against
        the band. The band is the co-kriging one, which
        :meth:`~pinneapple_data.co_kriging.CoKriging.predict` documents as **approximate and erratic** (measured
        1-sigma coverage 0.72): it is the right shape for a routing decision and not a guarantee."""
        design = np.atleast_2d(np.asarray(design, dtype=np.float64))
        low_values = self._low(design) if use_low else None
        mean, std = self.co_kriging.predict(design, low_values=low_values, return_std=True)
        return np.asarray(mean).ravel(), np.asarray(mean - z * std).ravel(), np.asarray(mean + z * std).ravel()

    def _low(self, design: np.ndarray) -> np.ndarray:
        """The cheap model at ``design``, read off the fitted low level. The workflow carries the *design and values*
        it measured; a caller who has the cheap function can pass its own values instead by calling
        ``run.co_kriging.predict(design, low_values=...)``."""
        return np.asarray(self.co_kriging.low.predict(design)).ravel()

    def serves(self, requests, solver: Callable, *, policy: FidelityPolicy | None = None,
               trust: Any = None, z: float = 1.96) -> FidelityReport:
        """Answer ``requests`` with the co-kriging model, escalating to ``solver`` on the policy's terms (#161).

        The model is the surrogate and the expensive level is the escalation: this is the two levels of the workflow
        used as what they were bought for. ``solver`` is the expensive function itself, so a request the policy sends
        up costs one expensive run.
        """
        mean, lower, upper = self.interval(requests, z=z)

        def surrogate(X):
            m, lo, hi = self.interval(X, z=z)
            return m, lo, hi

        return serve(requests, surrogate, solver, policy=policy, trust=trust)

    def report(self) -> dict[str, Any]:
        return {"method": "multi-fidelity: co-kriging over two solver levels", "plan": self.plan.to_dict(),
                "n_low": int(self.design_low.shape[0]), "n_high": int(self.design_high.shape[0]),
                "rho": self.rho, "verdict": self.verdict(), "gain": self.gain,
                "error": {"co_kriging": self.error_co_kriging, "high_only": self.error_high_only,
                          "low_only": self.error_low},
                "n_test": self.n_test, "seconds": self.seconds}

    def to_dict(self) -> dict[str, Any]:
        return {**self.report(),
                "design_low": self.design_low.tolist(), "values_low": np.asarray(self.values_low).tolist(),
                "design_high": self.design_high.tolist(), "values_high": np.asarray(self.values_high).tolist()}

    def table(self) -> str:
        header = f"{'model':>22}  {'runs':>14}  {'cost':>10}  {'rel. error':>12}"
        lines = [header, "-" * len(header)]
        rules = [(("low_fidelity"), f"{self.plan.n_low} cheap", self.plan.cost_ratio * self.plan.n_low,
                  self.error_low),
                 (("high_fidelity_only"), f"{self.plan.n_high} expensive", float(self.plan.n_high),
                  self.error_high_only),
                 (("co_kriging"), f"{self.plan.n_high} + {self.plan.n_low}", self.plan.total_cost,
                  self.error_co_kriging)]
        for name, runs, cost, error in rules:
            shown = f"{error:.3e}" if error is not None else "not measured"
            lines.append(f"{name:>22}  {runs:>14}  {cost:>10.2f}  {shown:>12}")
        if self.gain is not None:
            lines.append(f"co-kriging is {self.gain:.2f}x the high-fidelity-only model at equal expensive count "
                         f"(rho = {self.rho:.3g})")
        return "\n".join(lines)


class MultiFidelityWorkflow:
    """The two levels as callables, the plan, and the run that fits both models.

    >>> workflow = MultiFidelityWorkflow(low=coarse, high=fine, bounds=[(0.15, 0.85), (0.2, 2.0)],
    ...                                  plan=FidelityPlan(n_high=10, n_low=64, cost_ratio=0.02))
    >>> run = workflow.run(oracle=exact)          # the oracle is optional; without it nothing is claimed
    >>> print(run.table())
    """

    def __init__(self, low: Callable, high: Callable, bounds, *, plan: FidelityPlan | None = None,
                 oracle: Callable | None = None, n_test: int = 120, test_seed: int = 0):
        self.low = low
        self.high = high
        self.bounds = np.atleast_2d(np.asarray(bounds, dtype=np.float64))
        self.plan = plan or FidelityPlan()
        self.oracle = oracle
        self.n_test = int(n_test)
        self.test_seed = int(test_seed)

    def test_designs(self) -> np.ndarray:
        """The ``(n_test, d)`` designs the errors are measured on: a Sobol' sequence of the same box."""
        from scipy.stats import qmc
        n = 2 ** int(np.ceil(np.log2(max(self.n_test, 2))))
        unit = qmc.Sobol(d=self.bounds.shape[0], scramble=True, seed=self.test_seed + 991).random(n)[: self.n_test]
        return self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])

    def run(self) -> MultiFidelityRun:
        """Evaluate the nested design, fit the co-kriging model and the high-fidelity-only one, score both."""
        t0 = time.perf_counter()
        design_low, design_high = self.plan.designs(self.bounds)
        values_low = np.asarray(self.low(design_low), dtype=np.float64).ravel()
        values_high = np.asarray(self.high(design_high), dtype=np.float64).ravel()
        low_at_high = np.asarray(self.low(design_high), dtype=np.float64).ravel()
        if values_low.size != design_low.shape[0] or values_high.size != design_high.shape[0]:
            raise ValueError("a level returned a number of values that does not match its design")
        model = CoKriging.fit(design_low, values_low, design_high, values_high, low_at_high=low_at_high)
        high_only = GaussianProcess().fit(design_high, values_high)
        run = MultiFidelityRun(plan=self.plan, design_low=design_low, values_low=values_low,
                               design_high=design_high, values_high=values_high, low_at_high=low_at_high,
                               co_kriging=model, high_only=high_only)
        if self.oracle is not None:
            test = self.test_designs()
            truth = np.asarray(self.oracle(test), dtype=np.float64).ravel()
            scale = float(np.linalg.norm(truth))
            if scale > 0:
                co = model.predict(test, low_values=np.asarray(self.low(test), dtype=np.float64).ravel(),
                                   return_std=False)
                run.error_co_kriging = float(np.linalg.norm(co - truth) / scale)
                run.error_high_only = float(np.linalg.norm(high_only.predict(test) - truth) / scale)
                run.error_low = float(np.linalg.norm(np.asarray(self.low(test), dtype=np.float64).ravel() - truth)
                                      / scale)
            run.n_test = int(test.shape[0])
        run.seconds = round(time.perf_counter() - t0, 4)
        return run


# ---------------------------------------------------------------------------------------------- the measured case
def rod_fidelities(low_cells: int = 8, high_cells: int = 32) -> tuple[Callable, Callable]:
    """The measured pair of solvers: the same 1-D Poisson problem at two ``fem`` mesh resolutions.

    The problem comes from :func:`pinneapple_physics.active_design.source_rod` (a Gaussian source in a rod, with its
    position and strength as the design variables) and the quantity of interest is the peak of the field. The cheap
    level is the coarse mesh, the expensive one the fine mesh, and both are the *same physics* -- which is what makes
    this a fidelity pair and not two functions that happen to be correlated.

    Each level takes ``(n, d)`` designs and returns ``(n,)`` values, the contract every level here follows.
    """
    from pinneapple_physics.active_design import probe, source_rod
    from pinneapple_physics.solving import solve

    problem, ctx = source_rod()

    def level(n_cells: int) -> Callable:
        def quantity(theta) -> np.ndarray:
            theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
            out = np.empty(theta.shape[0])
            for i, row in enumerate(theta):
                x0, kappa = float(row[0]), float(row[1])
                point = problem.with_parameters(x0=x0, kappa=kappa)
                out[i] = probe(solve(point, "fem", n=n_cells, ctx={**ctx, "x0": x0, "kappa": kappa}), "abs_max")
            return out

        quantity.__name__ = f"fem_{n_cells}"
        return quantity

    return level(int(low_cells)), level(int(high_cells))


def _relative(prediction: np.ndarray, truth: np.ndarray) -> float:
    scale = float(np.linalg.norm(truth))
    return float(np.linalg.norm(np.asarray(prediction, dtype=np.float64).ravel() - truth) / scale) if scale else 0.0


def benchmark_multifidelity_workflow(*, repeats: int = 5, plan: FidelityPlan | None = None,
                                     low_cells: int = 8, high_cells: int = 32, truth_cells: int = 128,
                                     n_test: int = 120, extra_runs: Sequence[int] = (0, 1, 2, 3),
                                     seed: int = 0) -> dict[str, Any]:
    """What the two levels buy, on a real solver pair, at an equal expensive count **and** at an equal cost.

    Every repeat fits both models on the same data -- ``n_high`` expensive designs, ``n_low`` cheap ones, the expensive
    ones nested inside the cheap ones -- and scores them on held-out designs against a solve far finer than the
    expensive level. Then, for each number of extra expensive runs the 64 cheap ones could have bought instead, the
    single-fidelity baseline is fitted on ``n_high + k`` expensive designs (the same box, its own Sobol' sequence) and
    the two are compared. The comparison is put in **runs**, not in a price, because a run is what you can actually
    buy: a ``cost_ratio`` of 0.01 and one of 0.02 both buy one more expensive run when the cheap budget is 64, and
    reporting them as different prices would be a false precision.

    Measured on this machine (``2026-11-25``), 5 repeats, ``plan = (n_high 10, n_low 64)``, 120 held-out designs,
    ``truth_cells = 128``:

    ===============================  =========================
    quantity                          value
    ===============================  =========================
    cheap model alone                 9.1e-2
    high-fidelity only, 10 runs       1.8e-1
    co-kriging, 10 + 64               7.8e-2
    gain at equal expensive count     2.28 x (worst repeat 1.79 x)
    gain at equal cost, 11 runs       1.36 x
    gain at equal cost, 12 runs       0.74 x
    break-even                        1 extra expensive run, i.e. cost_ratio 1/64 = 0.016
    ===============================  =========================

    The ``0.74`` row is the honest half: **two** more expensive runs are worth more than all 64 cheap ones, so a coarse
    model that is only three times cheaper than the fine one does not pay here, and the workflow says so. A benchmark
    that compared equal *counts* only would have hidden it, which is why this one reports both.
    """
    from scipy.stats import qmc

    from pinneapple_physics.active_design import design_box, source_rod

    plan = plan or FidelityPlan(n_high=10, n_low=64, cost_ratio=0.02, seed=seed)
    problem, _ = source_rod()
    box = design_box(problem)
    bounds = np.array([box[n] for n in box], dtype=np.float64)
    low, high = rod_fidelities(low_cells, high_cells)
    oracle = rod_fidelities(truth_cells, truth_cells)[0]
    rows = []
    for repeat in range(int(repeats)):
        run_plan = FidelityPlan(n_high=plan.n_high, n_low=plan.n_low, cost_ratio=plan.cost_ratio,
                                seed=seed + 17 * repeat)
        workflow = MultiFidelityWorkflow(low, high, bounds, plan=run_plan, oracle=oracle, n_test=n_test,
                                         test_seed=seed + repeat)
        run = workflow.run()
        test = workflow.test_designs()
        truth = np.asarray(oracle(test), dtype=np.float64).ravel()
        row: dict[str, Any] = {"repeat": repeat, "rho": run.rho, "error_co_kriging": run.error_co_kriging,
                               "error_high_only": run.error_high_only, "error_low": run.error_low, "gain": run.gain,
                               "equal_cost": {}}
        for extra in extra_runs:
            extra = int(extra)
            if extra <= 0:
                baseline = run.error_high_only
            else:
                n = 2 ** int(np.ceil(np.log2(max(run_plan.n_high + extra, 2))))
                unit = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=seed + 4013 + 97 * repeat + extra).random(n)
                design = bounds[:, 0] + unit[: run_plan.n_high + extra] * (bounds[:, 1] - bounds[:, 0])
                values = np.asarray(oracle(design), dtype=np.float64).ravel()
                baseline = _relative(GaussianProcess().fit(design, values).predict(test), truth)
            row["equal_cost"][extra] = {"baseline_error": baseline,
                                        "gain": (baseline / run.error_co_kriging) if run.error_co_kriging else None}
        rows.append(row)

    def _median(key: str) -> float:
        return float(np.median([r[key] for r in rows]))

    equal_cost_gain = {int(extra): float(np.median([r["equal_cost"][int(extra)]["gain"] for r in rows]))
                       for extra in extra_runs}
    break_even = max([e for e, g in equal_cost_gain.items() if g > 1.0], default=None)
    return {"case": "rod_fidelities (fem 8 vs fem 32, truth fem 128)", "plan": plan.to_dict(),
            "repeats": int(repeats), "rows": rows, "median_error_low": _median("error_low"),
            "median_error_high_only": _median("error_high_only"),
            "median_error_co_kriging": _median("error_co_kriging"),
            "median_gain": _median("gain"), "worst_gain": float(np.min([r["gain"] for r in rows])),
            "equal_cost_gain": equal_cost_gain, "break_even_extra_runs": break_even,
            "break_even_cost_ratio": (break_even / plan.n_low) if break_even is not None else None}
