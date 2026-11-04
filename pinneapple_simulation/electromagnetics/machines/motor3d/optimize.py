"""The optimisation that runs on the surrogate: thousands of designs, no FEA per candidate.

The study is the one a motor designer actually asks: push torque density up, magnet mass and loss down, keep the
winding and the magnets under their temperature limits, keep the demagnetisation ratio below one and the torque
ripple below what the drive tolerates -- over the rotor cross-section, the stack length, the skew, the operating point
and the coolant flow at once.

Three functions, in the order a study uses them:

* :func:`design_problem` -- the objectives, the limits and the design space, as a
  :class:`~pinneapple_design.design_optimizer.study.DesignProblem` whose evaluator is the surrogate;
* :func:`optimize` -- NSGA-II on that problem with a budget counted in **surrogate** evaluations; the result records
  how many reference solves it cost, which is zero;
* :func:`verify_front` -- the same designs re-evaluated by the reference model, the one place where an FEA call is
  spent, so the front is not taken on the surrogate's word. A surrogate error that changes the ranking of the front
  is then visible as a number, not as a surprise.

:func:`benchmark` measures what the substitution bought: designs per second for the surrogate against designs per
second for the reference, on the same machine, in the same process.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.methods import run
from pinneapple_design.design_optimizer.study import (
    Constraint,
    DesignProblem,
    DesignSpace,
    Evaluator,
    Objective,
    Variable,
)

from .design import BOUNDS, NAMES, VARIABLES, Motor3DDesign
from .model import OUTPUTS, SCALARS, Motor3DModel

__all__ = ["Limits", "CONSTRAINED", "SurrogateEvaluator", "design_problem", "optimize", "verify_front", "benchmark",
           "TOP_LEVEL"]

#: The outputs the search constrains, and therefore the ones for which a design at the limit needs the ensemble
#: spread on its side: a constraint is not a prediction, and a prediction that is 1 sigma optimistic at the limit is
#: a design that fails the re-check.
CONSTRAINED = ("T_winding_C", "T_magnet_C", "demag_ratio", "ripple")

#: The decision limits of the study. They are engineering limits, stated, not tuned to make a front look good: a
#: class-H winding at 180 degC, a sintered NdFeB magnet at 140 degC (above it the coercivity falls away), a
#: demagnetisation ratio of 1 (the knee) and a torque ripple of 20 % peak to peak, which a traction inverter and
#: gearbox tolerate; tighter ripple is a design choice, pass your own.
Limits = dict


def _limits(**kw) -> dict[str, float]:
    base = {"T_winding_max_C": 180.0, "T_magnet_max_C": 140.0, "demag_ratio_max": 1.0, "ripple_max": 0.20}
    base.update({k: float(v) for k, v in kw.items()})
    return base


class SurrogateEvaluator(Evaluator):
    """The trained surrogate behind the design-optimiser interface: ``evaluate(X)`` returns one array per output,
    the profiles flattened to their hottest segment (an optimiser constrains numbers, not curves)."""

    fidelity = "surrogate"
    outputs = list(SCALARS) + [f"{k}_ub" for k in CONSTRAINED]
    name = "motor3d-surrogate"

    def __init__(self, surrogate, check_geometry: bool = True, k_std: float = 1.0):
        self.surrogate = surrogate
        self.k_std = float(k_std)
        self.n_calls = 0
        self.n_designs = 0
        #: A surrogate is a map from numbers to numbers: it does not know whether the cross-section can be built.
        #: The geometry test is exact and costs milliseconds, so it runs here rather than being learned.
        self.check_geometry = bool(check_geometry)
        self.description = f"{type(surrogate).__name__} ensemble of {len(surrogate.models)} networks"

    def evaluate(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """The scalars only: the optimiser constrains numbers, and the axial profiles are summarised by their
        hottest segment (``T_winding_C`` and ``T_magnet_C`` already are that maximum)."""
        X = np.atleast_2d(np.asarray(X, float))
        out = self.surrogate.evaluate(X)
        self.n_calls += 1
        self.n_designs += len(X)
        flat = {k: _hottest(out[k]) for k in SCALARS}
        # the upper bound a constraint should use: mean + k sigma of the ensemble, per design and per output
        for k in CONSTRAINED:
            flat[f"{k}_ub"] = _hottest(out[k]) + self.k_std * _hottest(out[f"{k}_std"])
        flat["_valid"] = (np.array([not Motor3DDesign.from_vector(x).sector_errors() for x in X], bool)
                          if self.check_geometry else np.ones(len(X), bool))
        return flat

    def spread(self, X) -> dict[str, np.ndarray]:
        """Ensemble spread per output: how much the members disagree, in the output's unit."""
        pred = self.surrogate.predict(X)
        return {f"{k}_std": _hottest(v[1]) for k, v in pred.items()}


def design_problem(evaluator: Evaluator | None = None, *, surrogate=None, limits: dict[str, float] | None = None,
                   objectives: Sequence[tuple[str, str]] | None = None) -> DesignProblem:
    """Maximise torque density and efficiency, minimise magnet mass; temperatures, ripple and the demagnetisation
    ratio as constraints. ``evaluator`` overrides the surrogate (pass a reference evaluator to run the same problem
    on the FEA); ``limits`` overrides :func:`_limits`."""
    ev = evaluator or SurrogateEvaluator(surrogate)
    lim = _limits(**(limits or {}))
    space = DesignSpace([Variable(v.name, v.low, v.high, unit=v.unit) for v in VARIABLES])
    unit_of = {o.name: o.unit for o in OUTPUTS}
    spec = objectives or (("torque_density_Nm_kg", "max"), ("efficiency", "max"), ("magnet_mass_kg", "min"))
    objs = [Objective(name, sense, unit_of[name]) for name, sense in spec]
    # the constraints see the conservative value of the surrogate (mean + k sigma of the ensemble), not its mean:
    # a front that only satisfies the limits inside the model is not a front
    cons = [Constraint("T_winding_C_ub", upper=lim["T_winding_max_C"], label="winding temperature [degC]"),
            Constraint("T_magnet_C_ub", upper=lim["T_magnet_max_C"], label="magnet temperature [degC]"),
            Constraint("demag_ratio_ub", upper=lim["demag_ratio_max"], label="demagnetisation ratio"),
            Constraint("ripple_ub", upper=lim["ripple_max"], label="torque ripple (pk-pk / mean)")]
    problem = DesignProblem("3-D multiphysics IPM motor", space, objs, cons, ev,
                            description="rotor cross-section + stack + skew + operating point + coolant flow, "
                                        "evaluated by the pretrained surrogate")
    problem.motor_limits = lim                 # carried so a saved result states the limits it was made under
    return problem


def optimize(problem: DesignProblem | None = None, *, budget: int = 2000, method: str = "nsga2", seed: int = 0,
             surrogate=None, initial: Motor3DDesign | None = None,
             progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Run ``method`` for ``budget`` surrogate evaluations and report the front, the best designs and the cost.

    ``budget`` counts designs, not FEA solves: the surrogate answers each one in microseconds. The result carries
    ``n_reference_solves`` (zero here) so the claim is checked rather than implied.
    """
    problem = problem or design_problem(surrogate=surrogate)
    x0 = (initial or Motor3DDesign.baseline()).as_vector()[None, :] if initial is not False else None
    t0 = time.perf_counter()
    res = run(problem, method, budget=budget, seed=seed, initial=x0, progress=progress)
    wall = time.perf_counter() - t0
    idx = res.pareto()
    E = res.evaluations
    def row_of(i: int, feasible: bool | None = None) -> dict[str, Any]:
        row: dict[str, Any] = Motor3DDesign.from_vector(E.X[i]).as_dict()
        row.update({k: float(_hottest(E.Y[k])[i]) for k in SCALARS})
        row["feasible"] = bool(E.feasible[i]) if feasible is None else feasible
        return row

    rows = [row_of(i) for i in idx] if len(idx) else []
    evals = {"X": E.X if E is not None else np.zeros((0, len(NAMES))), "pareto": idx,
             "Y": {k: _hottest(E.Y[k]) for k in SCALARS} if E is not None
             else {k: np.zeros(0) for k in SCALARS},
             "feasible": np.asarray(E.feasible, bool) if E is not None else np.zeros(0, bool)}
    return {"method": method, "budget": budget, "seed": seed, "status": res.status, "message": res.message,
            "n_evaluations": res.n_evals, "wall_seconds": wall,
            "designs_per_second": res.n_evals / wall if wall > 0 else float("nan"),
            "n_reference_solves": 0, "result": res, "evaluations": evals, "front": rows,
            "best_efficiency": row_of(res.best(1)) if E is not None and res.best(1) is not None else None,
            "best_torque_density": row_of(res.best(0)) if E is not None and res.best(0) is not None else None,
            "summary": {o: [float(v) for v in evals["Y"][o][idx]] for o in
                        ("torque_density_Nm_kg", "efficiency", "magnet_mass_kg", "T_winding_C")} if len(idx) else {},
            "limits": dict(getattr(problem, "motor_limits", _limits()))}


def _hottest(v: Any) -> np.ndarray:
    """An output as one value per design: a profile becomes its hottest axial segment, a scalar stays itself."""
    a = np.asarray(v, float)
    return a if a.ndim == 1 else a.max(1)


def _spread_over_front(X: np.ndarray, n: int) -> np.ndarray:
    """``n`` designs of the front, chosen by farthest-point sampling in the design box, so the re-check covers the
    front instead of its first ``n`` points."""
    if len(X) <= n:
        return np.arange(len(X))
    unit = (np.asarray(X, float) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    chosen = [int(np.argmin(np.linalg.norm(unit - unit.mean(0), axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(unit[:, None, :] - unit[None, chosen, :], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return np.array(chosen)


def verify_front(result: dict[str, Any], *, reference: Motor3DModel | None = None, n: int = 8,
                 limits: dict[str, float] | None = None) -> dict[str, Any]:
    """Re-evaluate designs of the surrogate front with the reference model and compare, output by output.

    Returns the per-output error (relative and absolute) of the surrogate against the FEA-based model on exactly
    those designs, whether the re-evaluated design is still feasible under the same limits, and the cost of the
    check. This is the number that says whether the surrogate can be trusted for this study: it is measured on the
    front, which is the region the optimiser actually chose, not on a random split.
    """
    evals = result.get("evaluations") or {}
    idx = np.asarray(evals.get("pareto", []), int)
    if not len(idx):
        return {"status": "no feasible front", "n_checked": 0}
    X_all, Y_all = np.asarray(evals["X"], float), evals["Y"]
    sel = idx[_spread_over_front(X_all[idx], n)]
    X = X_all[sel]
    ref = reference or Motor3DModel()
    lim = _limits(**(limits or {}))
    t0 = time.perf_counter()
    Y_ref = ref.evaluate(X)
    wall = time.perf_counter() - t0
    rows, errs = [], {}
    for k in SCALARS:
        s = _hottest(np.asarray(Y_all[k], float)[sel])
        r = _hottest(Y_ref[k])
        rel = np.abs(s - r) / np.maximum(np.abs(r), 1e-9)
        errs[k] = {"mean_rel": float(np.mean(rel)), "max_rel": float(np.max(rel)),
                   "mae": float(np.mean(np.abs(s - r))), "unit": next(o.unit for o in OUTPUTS if o.name == k)}
    for i in range(len(sel)):
        row = {**Motor3DDesign.from_vector(X[i]).as_dict()}
        row["surrogate"] = {k: float(_hottest(np.asarray(Y_all[k], float)[sel])[i]) for k in SCALARS}
        row["reference"] = {k: float(_hottest(Y_ref[k])[i]) for k in SCALARS}
        row["feasible_reference"] = bool(Y_ref["T_winding_C"][i] <= lim["T_winding_max_C"]
                                         and Y_ref["T_magnet_C"][i] <= lim["T_magnet_max_C"]
                                         and Y_ref["demag_ratio"][i] <= lim["demag_ratio_max"]
                                         and Y_ref["ripple"][i] <= lim["ripple_max"])
        rows.append(row)
    return {"status": "ok", "n_checked": int(len(sel)), "reference": ref.description,
            "reference_solves": int(len(sel)), "reference_seconds": wall,
            "errors": errs, "rows": rows,
            "still_feasible": int(sum(r["feasible_reference"] for r in rows)),
            "rank_agreement": _rank_agreement(np.array([r["surrogate"]["torque_density_Nm_kg"] for r in rows]),
                                              np.array([r["reference"]["torque_density_Nm_kg"] for r in rows]))}


def _rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two orderings of the same designs (1 = the front is ranked identically)."""
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def benchmark(surrogate, reference: Motor3DModel | None = None, *, n: int = 32, seed: int = 0) -> dict[str, Any]:
    """Designs per second: the surrogate against the reference model, on the same designs."""
    rng = np.random.default_rng(seed)
    X = BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) * rng.random((n, len(NAMES)))
    ok = np.array([not Motor3DDesign.from_vector(x).sector_errors() for x in X])
    X = X[ok] if ok.any() else X
    t0 = time.perf_counter()
    surrogate.evaluate(X)
    t_sur = time.perf_counter() - t0
    ref = reference or Motor3DModel()
    t0 = time.perf_counter()
    ref.evaluate(X[:min(len(X), max(2, n // 8))])
    t_ref = (time.perf_counter() - t0) / max(1, min(len(X), max(2, n // 8)))
    return {"n_designs": int(len(X)), "surrogate_seconds": t_sur,
            "surrogate_per_design": t_sur / len(X), "reference_per_design": t_ref,
            "speedup": t_ref / (t_sur / len(X)) if t_sur > 0 else float("inf"),
            "reference_config": ref.description}


#: The names the study optimises or constrains, for a report to print in order.
TOP_LEVEL = ("torque_Nm", "torque_density_Nm_kg", "efficiency", "magnet_mass_kg", "total_loss_W",
             "T_winding_C", "T_magnet_C", "demag_ratio", "ripple")
