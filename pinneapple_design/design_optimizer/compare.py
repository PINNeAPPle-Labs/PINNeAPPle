"""Compare optimisation methods on the same problem and the same evaluation budget.

Every method gets the same evaluator, the same budget and the same seeds. The convergence metric is the
hypervolume of the feasible, realisable designs found so far, with the objectives normalised between the ideal and
nadir points of the feasible designs found by *all* runs (reference point 1.1 per axis), so the numbers are
comparable across methods. A run "converged" when its hypervolume grew by less than ``tol`` (relative) over the last
quarter of the evaluations it made; ``evals_to_95`` is the number of evaluations it needed to reach 95 % of the best
final hypervolume of any run.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .methods import OptimizationResult, normalise_front, run
from .study import DesignProblem, Evaluations, hypervolume

__all__ = ["MethodComparison", "compare_methods"]


@dataclass
class MethodComparison:
    problem: DesignProblem
    results: list[OptimizationResult]
    ideal: np.ndarray
    nadir: np.ndarray
    budget: int
    rows: list[dict[str, Any]]
    curves: dict[str, list[tuple[int, float]]]

    def table(self) -> list[dict[str, Any]]:
        """One row per method (seeds aggregated: mean final hypervolume, worst/best, total wall time)."""
        out = []
        for m in dict.fromkeys(r["method"] for r in self.rows):
            rs = [r for r in self.rows if r["method"] == m]
            done = [r for r in rs if r["status"] == "ok"]
            hv = [r["hv_final"] for r in done]
            e95 = [r["evals_to_95"] for r in done]
            out.append({
                "method": m, "family": rs[0]["family"], "status": rs[0]["status"] if not done else "ok",
                "message": "; ".join(sorted({r["message"] for r in rs if r["message"]})),
                "seeds": len(rs), "n_evals": int(np.mean([r["n_evals"] for r in rs])) if rs else 0,
                "wall_s": float(np.sum([r["wall_s"] for r in rs])),
                "hv_mean": float(np.mean(hv)) if hv else None, "hv_min": float(np.min(hv)) if hv else None,
                "hv_max": float(np.max(hv)) if hv else None,
                "evals_to_95": (None if not e95 or any(v is None for v in e95) else int(np.max(e95))),
                "converged": bool(done) and all(r["converged"] for r in done),
                "n_pareto": int(np.mean([r["n_pareto"] for r in done])) if done else 0,
                "best": done[int(np.argmax(hv))]["best"] if done else {},
            })
        return out

    def union(self) -> tuple[Evaluations | None, list[str]]:
        """All evaluations of all runs in one record, and the run (``"<method>#<seed>"``) of each row; its
        :meth:`~.study.Evaluations.pareto` is the best front any method found, each member labelled by who found it."""
        evs, labels = [], []
        for r in self.results:
            if r.evaluations is not None and len(r.evaluations):
                evs.append(r.evaluations)
                labels += [f"{r.method}#{r.seed}"] * len(r.evaluations)
        if not evs:
            return None, []
        out = evs[0]
        for e in evs[1:]:
            out = out.extend(e)
        return out, labels

    def union_result(self) -> OptimizationResult | None:
        """The comparison as one :class:`~.methods.OptimizationResult` (method ``"comparison: <methods>"``, the
        first seed, every run's evaluations and the summed budget and wall time), e.g. for
        :func:`~.report.study_result`. None when no run produced evaluations."""
        ev, _ = self.union()
        if ev is None:
            return None
        methods = list(dict.fromkeys(r.method for r in self.results))
        return OptimizationResult(self.problem, "comparison: " + ", ".join(methods), self.results[0].seed,
                                  self.budget * len(self.results), ev, float(sum(r.wall_s for r in self.results)))

    def fronts(self, seed: int | None = None) -> dict[str, list[list[float]]]:
        """Pareto front of each method's run with ``seed`` (default: the first seed), in the objectives' own units
        and senses (one list of objective values per member)."""
        seed = self.results[0].seed if seed is None and self.results else seed
        names = [o.name for o in self.problem.objectives]
        out = {}
        for r in self.results:
            if r.evaluations is None or r.seed != seed:
                continue
            out[r.method] = [[float(r.evaluations.Y[k][i]) for k in names] for i in r.pareto()]
        return out

    def winner(self) -> str | None:
        t = [r for r in self.table() if r["hv_mean"] is not None]
        return max(t, key=lambda r: (r["hv_mean"], -(r["wall_s"] or 0)))["method"] if t else None

    def to_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "ideal": self.ideal.tolist(), "nadir": self.nadir.tolist(),
                "objectives": [o.name for o in self.problem.objectives], "table": self.table(), "runs": self.rows,
                "curves": {k: [list(p) for p in v] for k, v in self.curves.items()}, "winner": self.winner()}


def compare_methods(problem: DesignProblem, methods: Sequence[str], *, budget: int = 300,
                    seeds: Sequence[int] = (0,), tol: float = 0.01, options: dict[str, dict] | None = None,
                    progress=None) -> MethodComparison:
    """Run every method in ``methods`` once per seed with ``budget`` evaluations and tabulate the outcome.

    ``progress(done, total)`` counts runs (fractional while a run is going)."""
    from .methods import FAMILIES

    results: list[OptimizationResult] = []
    total = len(methods) * len(seeds)
    for m in methods:
        for s in seeds:
            k = len(results)
            step = (lambda used, b, k=k: progress(k + used / b, total)) if progress else None
            results.append(run(problem, m, budget=budget, seed=int(s), progress=step, **(options or {}).get(m, {})))
            if progress:
                progress(len(results), total)
    feas = [r.evaluations.F[r.evaluations.feasible & ~r.evaluations.relaxed] for r in results
            if r.evaluations is not None]
    F = np.vstack([f for f in feas if len(f)]) if any(len(f) for f in feas) else np.zeros((0, problem.n_obj))
    F = F[np.all(np.isfinite(F), 1)]
    ideal = F.min(0) if len(F) else np.zeros(problem.n_obj)
    nadir = F.max(0) if len(F) else np.ones(problem.n_obj)
    ref = np.full(problem.n_obj, 1.1)
    rows, curves = [], {}
    finals = []
    for r in results:
        curve = r.hv_curve(ideal, nadir) if r.status == "ok" else []
        curves[f"{r.method}#{r.seed}"] = curve
        finals.append(curve[-1][1] if curve else 0.0)
    best_hv = max(finals) if finals else 0.0
    for r, curve, hv in zip(results, curves.values(), finals, strict=True):
        conv, e95 = False, None
        if curve:
            k_end = curve[-1][0]
            before = [v for k, v in curve if k <= 0.75 * k_end]
            ref_hv = before[-1] if before else 0.0
            conv = hv > 0 and (hv - ref_hv) <= tol * hv
            if best_hv > 0:
                e95 = next((k for k, v in curve if v >= 0.95 * best_hv), None)
        best = {}
        if r.evaluations is not None:
            for j, o in enumerate(problem.objectives):
                i = r.best(j)
                best[o.name] = None if i is None else float(r.evaluations.Y[o.name][i])
        rows.append({**r.summary(), "family": FAMILIES.get(r.method, ""), "hv_final": float(hv),
                     "converged": bool(conv), "evals_to_95": e95, "best": best,
                     "front": (normalise_front(r.evaluations.F[r.pareto()], ideal, nadir).tolist()
                               if r.evaluations is not None else [])})
    # the hypervolume of the union is the yardstick: no run can beat the union of all runs
    union = hypervolume(normalise_front(F, ideal, nadir), ref) if len(F) else 0.0
    for row in rows:
        row["hv_fraction_of_union"] = float(row["hv_final"] / union) if union > 0 else None
    return MethodComparison(problem, results, ideal, nadir, budget, rows, curves)
