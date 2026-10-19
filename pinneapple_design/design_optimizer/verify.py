"""High-fidelity confirmation of optimised designs, and surrogate-assisted optimisation with active learning.

* :func:`select_candidates`: designs spread along a Pareto front (both ends always included).
* :func:`verify_candidates`: re-evaluate chosen designs with the reference evaluator (solver or high-fidelity
  model); per design the surrogate and reference values, the relative error of every output, whether the reference
  confirms feasibility, and a verification status:

  - ``"verified"``: the reference ran and the design is feasible under it,
  - ``"rejected by reference"``: the reference ran and a constraint is violated (or the design is invalid),
  - ``"reference failed"``: the reference raised or returned non-finite values,
  - ``"only surrogate"``: no reference was run for this design.

* :class:`GPSurrogate`: Gaussian-process regression per output (scikit-learn), trained on reference evaluations,
  usable as an :class:`~.study.Evaluator` (with the predictive standard deviation as ``<output>_std``).
* :func:`surrogate_assisted`: rounds of (fit the surrogate on all reference data so far, optimise on it, verify a
  few Pareto designs with the reference, add them to the data). The error reported per round is measured at the
  verified designs *before* they join the training data, i.e. on points the surrogate had not seen.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .methods import OptimizationResult, run
from .study import VALID, DesignProblem, Evaluations, Evaluator, feasibility_report

__all__ = ["select_candidates", "verify_candidates", "VerificationReport", "GPSurrogate", "surrogate_assisted",
           "ActiveLearningResult", "STATUS"]

STATUS = ("verified", "rejected by reference", "reference failed", "only surrogate")


def select_candidates(ev: Evaluations, front: Sequence[int], n: int) -> list[int]:
    """``n`` designs spread evenly along ``front`` (in its order), always including both ends."""
    front = list(front)
    if len(front) <= n:
        return front
    idx = np.unique(np.round(np.linspace(0, len(front) - 1, n)).astype(int))
    return [front[i] for i in idx]


@dataclass
class VerificationReport:
    rows: list[dict[str, Any]]
    errors: dict[str, dict[str, float]]
    reference: dict[str, Any]
    wall_s: float
    evaluations: Evaluations | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "errors": self.errors, "reference": self.reference, "wall_s": round(self.wall_s, 4)}

    def status_of(self, i: int) -> str:
        for r in self.rows:
            if r["index"] == i:
                return r["status"]
        return "only surrogate"


def verify_candidates(problem: DesignProblem, ev: Evaluations, idx: Sequence[int],
                      reference: Evaluator, reference_problem: DesignProblem | None = None) -> VerificationReport:
    """Re-evaluate the designs ``ev.X[idx]`` with ``reference`` and compare every shared output.

    Relative errors are ``|surrogate - reference| / |reference|``; ``errors`` holds their mean and max per output
    over the designs the reference evaluated successfully. ``reference_problem`` (default: ``problem`` on
    ``reference``) sets the constraints the reference values are checked against.
    """
    idx = [int(i) for i in idx]
    ref_problem = reference_problem or problem.with_evaluator(reference)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    ref_ev: Evaluations | None = None
    if idx:
        X = ev.X[idx]
        try:
            ref_ev = ref_problem.evaluate(X, tag="verification")
        except Exception as e:  # noqa: BLE001 - one failing solver run must not hide the others
            ref_ev = None
            err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    shared = [k for k in ev.Y if k != VALID and ref_ev is not None and k in ref_ev.Y]
    rel: dict[str, list[float]] = {k: [] for k in shared}
    feas = feasibility_report(ref_problem, ref_ev, range(len(idx))) if ref_ev is not None else []
    for j, i in enumerate(idx):
        row: dict[str, Any] = {"index": i, "surrogate": {k: _f(ev.Y[k][i]) for k in ev.Y if k != VALID}}
        if ref_ev is None:
            row.update(status="reference failed", message=err, reference={}, rel_error={})
            rows.append(row)
            continue
        refv = {k: _f(ref_ev.Y[k][j]) for k in ref_ev.Y if k != VALID}
        finite = all(np.isfinite(ref_ev.Y[k][j]) for k in shared) if shared else True
        re = {}
        for k in shared:
            a, b = float(ev.Y[k][i]), float(ref_ev.Y[k][j])
            if np.isfinite(a) and np.isfinite(b) and b != 0:
                re[k] = abs(a - b) / abs(b)
                rel[k].append(re[k])
        if not finite:
            status = "reference failed"
        else:
            status = "verified" if bool(ref_ev.feasible[j]) else "rejected by reference"
        row.update(status=status, reference=refv, rel_error=re, feasible_surrogate=bool(ev.feasible[i]),
                   feasible_reference=bool(ref_ev.feasible[j]), reference_feasibility=feas[j])
        rows.append(row)
    errors = {k: {"mean_rel": float(np.mean(v)), "max_rel": float(np.max(v)), "n": len(v)} for k, v in rel.items() if v}
    return VerificationReport(rows, errors, reference.info(), wall, ref_ev)


def _f(v: Any) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


# --------------------------------------------------------------------------- GP surrogate
class GPSurrogate(Evaluator):
    """Per-output Gaussian process (Matern 5/2, ARD length scales, scikit-learn) on unit-cube inputs.

    Outputs that are strictly positive in the training data are modelled in log space (a relative error model,
    which suits masses, temperatures rises and pressure drops spanning decades). ``"_valid"`` is passed through
    from ``validity`` (a callable ``X -> bool``) when given. ``safe={output: +1 or -1}`` adds ``<output>_safe`` =
    mean + sign * ``kappa`` * std (a pessimistic value for an upper (+1) or lower (-1) bound).
    """
    fidelity = "surrogate"

    def __init__(self, problem: DesignProblem, outputs: Sequence[str], validity=None, name: str = "gp-surrogate",
                 safe: dict[str, int] | None = None, kappa: float = 2.0):
        self.space = problem.space
        self.safe, self.kappa = dict(safe or {}), float(kappa)
        self.outputs = list(outputs) + [f"{k}_safe" for k in self.safe]
        self.name = name
        self.validity = validity
        self.description = "Gaussian-process regression per output trained on reference evaluations"
        self.n_train = 0
        self._gps: dict[str, Any] = {}
        self._log: dict[str, bool] = {}

    def fit(self, X: np.ndarray, Y: dict[str, np.ndarray]) -> GPSurrogate:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        U = self.space.to_unit(X)
        d = U.shape[1]
        for k in [o for o in self.outputs if not o.endswith("_safe")]:
            y = np.asarray(Y[k], dtype=float)
            ok = np.isfinite(y)
            self._log[k] = bool(np.all(y[ok] > 0))
            t = np.log(y[ok]) if self._log[k] else y[ok]
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.5), (1e-2, 1e2), nu=2.5) \
                + WhiteKernel(1e-6, (1e-10, 1e-2))
            gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2, random_state=0)
            with warnings.catch_warnings():      # hyper-parameter fits at a bound are not errors of the data
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(U[ok], t)
            self._gps[k] = gp
        self.n_train = len(X)
        return self

    def evaluate(self, X):
        U = self.space.to_unit(X)
        out = {}
        for k, gp in self._gps.items():
            m, s = gp.predict(U, return_std=True)
            out[k] = np.exp(m) if self._log[k] else m
            out[f"{k}_std"] = out[k] * s if self._log[k] else s
            if k in self.safe:
                z = self.safe[k] * self.kappa * s
                out[f"{k}_safe"] = np.exp(m + z) if self._log[k] else m + z
        if self.validity is not None:
            out[VALID] = np.asarray(self.validity(X), dtype=bool)
        return out


# --------------------------------------------------------------------------- active learning
@dataclass
class ActiveLearningResult:
    problem: DesignProblem
    reference_evals: Evaluations
    rounds: list[dict[str, Any]]
    final: OptimizationResult
    verification: VerificationReport
    wall_s: float
    notes: list[str] = field(default_factory=list)

    @property
    def n_reference(self) -> int:
        return len(self.reference_evals)

    def to_dict(self) -> dict[str, Any]:
        return {"rounds": self.rounds, "n_reference_evals": self.n_reference, "wall_s": round(self.wall_s, 3),
                "final": self.final.to_dict(max_rows=400), "verification": self.verification.to_dict(),
                "notes": self.notes}


def surrogate_assisted(problem: DesignProblem, reference: Evaluator, *, outputs: Sequence[str] | None = None,
                       method: str = "nsga2", n_initial: int = 24, rounds: int = 3, n_verify: int = 6,
                       surrogate_budget: int = 1500, seed: int = 0, validity=None, kappa: float = 2.0,
                       progress=None) -> ActiveLearningResult:
    """Optimise on a Gaussian-process surrogate of ``reference`` and improve it where the optimiser looks.

    1. Evaluate a Latin hypercube of ``n_initial`` designs with the reference.
    2. Each round: fit :class:`GPSurrogate` on all reference data, run ``method`` on it with ``surrogate_budget``
       evaluations, pick ``n_verify`` designs along the surrogate Pareto front, evaluate them with the reference
       (the round's held-out error), add them to the data.
    3. The result's front is the reference-evaluated designs: every design on it is verified by construction.

    ``outputs`` defaults to the objective and constraint outputs of ``problem``. Reference cost: ``n_initial +
    rounds * n_verify`` evaluations. On the surrogate, each constraint is applied to a pessimistic value, mean +-
    ``kappa`` GP standard deviations (``kappa=0``: the mean), so designs near an active constraint are not chosen on
    a lucky prediction; the reported errors compare the GP mean with the reference.
    """
    from .study import Constraint
    t0 = time.perf_counter()
    outs = list(outputs or dict.fromkeys([o.name for o in problem.objectives] + [c.name for c in problem.constraints]))
    ref_problem = problem.with_evaluator(reference)
    data = ref_problem.evaluate(problem.space.sample(n_initial, seed=seed), tag="initial reference sample")
    log: list[dict[str, Any]] = []
    last: OptimizationResult | None = None
    rep: VerificationReport | None = None
    for r in range(rounds):
        ok = np.all([np.isfinite(data.Y[k]) for k in outs], 0)
        safe = {c.name: (1 if c.upper is not None else -1) for c in problem.constraints
                if (c.upper is None) != (c.lower is None)} if kappa > 0 else {}
        gp = GPSurrogate(problem, outs, validity=validity, safe=safe, kappa=kappa)
        gp.fit(data.X[ok], {k: v[ok] for k, v in data.Y.items()})
        cons = [Constraint(f"{c.name}_safe" if c.name in safe else c.name, c.upper, c.lower, c.unit, c.kind,
                           c.label) for c in problem.constraints]
        sur_problem = problem.with_evaluator(gp, constraints=cons)
        step = (lambda used, b, r=r: progress(r + 0.8 * used / b, rounds)) if progress else None
        last = run(sur_problem, method, budget=surrogate_budget, seed=seed + r, progress=step)
        front = last.pareto()
        chosen = select_candidates(last.evaluations, front, n_verify) if last.evaluations is not None else []
        rep = verify_candidates(sur_problem, last.evaluations, chosen, reference, ref_problem)
        if rep.evaluations is not None:
            new = rep.evaluations
            new.tag = [f"round {r} verification"] * len(new)
            data = data.extend(new)
        log.append({"round": r, "n_train": gp.n_train, "surrogate_evals": last.n_evals,
                    "n_front_surrogate": int(len(front)), "verified": len(chosen),
                    "confirmed_feasible": sum(1 for x in rep.rows if x["status"] == "verified"),
                    "held_out_error": rep.errors, "wall_s": round(time.perf_counter() - t0, 3)})
        if progress:
            progress(r + 1, rounds)
    final = OptimizationResult(ref_problem, f"surrogate-assisted ({method})", seed, len(data), data,
                               time.perf_counter() - t0)
    notes = [f"{len(data)} reference evaluations ({n_initial} initial + {rounds} rounds x up to {n_verify})",
             "held-out error = surrogate prediction vs reference at designs the surrogate had not been trained on"]
    return ActiveLearningResult(problem, data, log, final, rep, time.perf_counter() - t0, notes)
