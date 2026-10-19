"""Design studies in and out of the shared data contracts, and models built from your own data.

* :func:`to_parameter_space`: a :class:`~.study.DesignSpace` as a ``pinneapple_data.contracts.ParameterSpace``
  (and :meth:`~.study.DesignSpace.from_parameter_space` reads one back).
* :func:`designs_frame`: designs and their outputs (plus feasibility and verification status) as a table.
* :func:`study_result`: an optimisation or a method comparison as a ``StudyResult``: the candidate table, metrics
  (budget, evaluations, hypervolume, surrogate errors at the verified designs) and the checks that actually ran.
* :func:`surrogate_from_table`: a :class:`~.verify.GPSurrogate` trained on a table of evaluated designs (a design of
  experiments, a campaign's runs), with its error on a held-out split; and :func:`model_ref_for_surrogate`, its
  ``ModelRef``.
* :func:`table_evaluator`: a reference evaluator that looks designs up in a table of solver results (for designs you
  ran offline, e.g. the candidates exported by an optimisation), and :func:`verify_uploaded_results`, the exported
  candidates verified against such a table.
* What an app shows and exports after a run: :func:`verify_front` (candidates along the front, verified when there
  is a reference), :func:`write_design_files` (the CSV and ``ParameterSpace`` files) and :func:`chart_points` (a
  thinned list of designs for a trade-off plot).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .study import (
    VALID,
    DesignProblem,
    DesignSpace,
    Evaluations,
    FunctionEvaluator,
    Objective,
    Variable,
)

__all__ = ["to_parameter_space", "designs_frame", "study_result", "surrogate_from_table", "model_ref_for_surrogate",
           "table_evaluator", "SurrogateFit", "verify_front", "write_design_files", "chart_points",
           "verify_uploaded_results", "DESIGN_FILES"]


def to_parameter_space(space: DesignSpace, name: str, description: str = "", budget: int | None = None):
    """``pinneapple_data.contracts.ParameterSpace`` of a design space (continuous and integer parameters)."""
    from pinneapple_data.contracts import ParameterSpace
    params = [{"name": v.name, "kind": "integer" if v.kind == "integer" else "continuous", "low": v.lower,
               "high": v.upper, "unit": v.unit, "description": v.description, "group": "geometry"}
              for v in space.variables]
    return ParameterSpace(name=name, description=description, parameters=params, budget=budget)


def designs_frame(problem: DesignProblem, ev: Evaluations, idx: Sequence[int] | None = None,
                  status: Mapping[int, str] | None = None):
    """A pandas table: one row per design, variables, outputs, ``feasible``, ``fidelity`` and (when given)
    ``verification`` status."""
    import pandas as pd
    rows = ev.rows(problem.space, idx)
    for r in rows:
        if status is not None:
            r["verification"] = status.get(r["index"], "only surrogate" if r["fidelity"] == "surrogate" else "")
    return pd.DataFrame(rows)


def _hv_metric(result) -> float | None:
    ev = result.evaluations
    if ev is None:
        return None
    from .methods import normalise_front
    from .study import hypervolume
    ok = ev.F[ev.feasible & ~ev.relaxed]
    if len(ok) < 1:
        return None
    lo, hi = ok.min(0), ok.max(0)
    return hypervolume(normalise_front(ok, lo, hi), np.full(ok.shape[1], 1.1))


def study_result(problem: DesignProblem, result, *, study_id: str, app: str, title: str = "",
                 candidates: Sequence[int] = (), verification=None, comparison=None, active_learning=None,
                 limits: Sequence[str] = (), artifacts: Sequence[Any] = ()):
    """``StudyResult`` (study_type ``"optimization"``, or ``"design"`` for a method comparison).

    Checks are recorded only for what ran: ``high_fidelity_verification`` (PASS when every verified candidate is
    feasible under the reference, FAIL when one is not, NOT_RUN without a reference), ``surrogate_error_<output>``
    (WARN above 5 % mean relative error at the verified designs) and ``feasible_designs_found``.
    """
    from pinneapple_data.contracts import CheckRecord, Provenance, StudyResult, Table

    ev = result.evaluations
    status = {r["index"]: r["status"] for r in verification.rows} if verification is not None else {}
    table = None
    if ev is not None and len(candidates):
        df = designs_frame(problem, ev, list(candidates), status)
        table = Table.from_frame(df)
    metrics: dict[str, Any] = {"budget": result.budget, "n_evals": result.n_evals, "method": result.method,
                               "wall_s": round(result.wall_s, 3), "evaluator": problem.evaluator.name,
                               "fidelity": problem.evaluator.fidelity}
    if ev is not None:
        metrics["n_feasible"] = int((ev.feasible & ~ev.relaxed).sum())
        metrics["n_pareto"] = int(len(result.pareto()))
        hv = _hv_metric(result)
        if hv is not None:
            metrics["hypervolume_normalised_own_front"] = round(hv, 6)
    checks = []
    n_feas = metrics.get("n_feasible", 0)
    checks.append(CheckRecord(name="feasible_designs_found", status="PASS" if n_feas else "FAIL", value=float(n_feas),
                              detail="feasible, realisable designs among the evaluations"))
    if verification is not None and verification.rows:
        ran = [r for r in verification.rows if r["status"] in ("verified", "rejected by reference")]
        ok = [r for r in ran if r["status"] == "verified"]
        if ran:
            checks.append(CheckRecord(
                name="high_fidelity_verification", status="PASS" if len(ok) == len(ran) else "FAIL",
                value=f"{len(ok)}/{len(ran)}", evidence=f"reference: {verification.reference.get('name')}",
                detail="chosen candidates re-evaluated with the reference; PASS = all feasible under it"))
        for k, e in verification.errors.items():
            metrics[f"surrogate_mean_rel_error_{k}"] = round(e["mean_rel"], 6)
            metrics[f"surrogate_max_rel_error_{k}"] = round(e["max_rel"], 6)
            if k in {o.name for o in problem.objectives} | {c.name for c in problem.constraints}:
                checks.append(CheckRecord(name=f"surrogate_error_{k}", status="PASS" if e["mean_rel"] <= 0.05
                                          else "WARN", value=e["mean_rel"], threshold=0.05,
                                          detail=f"mean relative error at {e['n']} verified designs"))
    else:
        checks.append(CheckRecord(name="high_fidelity_verification", status="NOT_RUN",
                                  detail="no reference model or solver was run on the candidates"))
    if comparison is not None:
        metrics["winner"] = comparison.winner()
        for row in comparison.table():
            if row["hv_mean"] is not None:
                metrics[f"hv_{row['method']}"] = round(row["hv_mean"], 6)
    if active_learning is not None:
        metrics["reference_evals"] = active_learning.n_reference
    return StudyResult(
        id=study_id, study_type="design" if comparison is not None else "optimization", app=app,
        title=title or problem.description, parameter_space=to_parameter_space(problem.space, problem.name),
        table=table, metrics=metrics, checks=checks, artifacts=list(artifacts),
        limits=list(dict.fromkeys(list(problem.notes) + list(limits))),
        provenance=Provenance(app=app, study_id=study_id, source=f"{result.method} on {problem.evaluator.name}"),
        metadata={"objectives": [o.name for o in problem.objectives],
                  "constraints": [c.describe() for c in problem.constraints]})


# --------------------------------------------------------------------------- models from your data
class SurrogateFit:
    """A GP surrogate trained on a table, with its held-out error."""

    def __init__(self, space: DesignSpace, gp, outputs: list[str], metrics: dict[str, dict[str, float]],
                 n_train: int, n_test: int):
        self.space, self.gp, self.outputs, self.metrics = space, gp, outputs, metrics
        self.n_train, self.n_test = n_train, n_test

    def to_dict(self) -> dict[str, Any]:
        return {"outputs": self.outputs, "held_out": self.metrics, "n_train": self.n_train, "n_test": self.n_test,
                "variables": self.space.to_dict()["variables"]}


def surrogate_from_table(df, inputs: Sequence[str], outputs: Sequence[str], *, test_fraction: float = 0.2,
                         seed: int = 0, integer: Sequence[str] = (), bounds: Mapping[str, tuple] | None = None
                         ) -> SurrogateFit:
    """Fit a :class:`~.verify.GPSurrogate` per output on the rows of ``df``.

    The rows are split (``test_fraction`` held out, seeded); the error on the held-out rows (MAE, RMSE, mean and
    max relative error, R^2) is computed with a GP trained on the rest; then the GP is refitted on all rows for the
    optimisation. Variable bounds default to the data range (the surrogate's validity range), which designs cannot
    leave.
    """
    from .verify import GPSurrogate

    inputs, outputs = list(inputs), list(outputs)
    missing = [c for c in inputs + outputs if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in the table: {missing}")
    data = df[inputs + outputs].apply(lambda c: c.astype(float)).dropna()
    if len(data) < max(8, 2 * len(inputs) + 2):
        raise ValueError(f"{len(data)} complete rows: need at least {max(8, 2 * len(inputs) + 2)} to fit and test a "
                         "surrogate")
    X = data[inputs].to_numpy()
    vs = []
    for j, n in enumerate(inputs):
        lo, hi = (bounds or {}).get(n, (float(X[:, j].min()), float(X[:, j].max())))
        if not hi > lo:
            raise ValueError(f"input {n!r} is constant in the table: drop it or fix it")
        vs.append(Variable(n, float(lo), float(hi), "integer" if n in integer else "continuous"))
    space = DesignSpace(vs)
    stub = DesignProblem("table", space, [_first_objective(outputs)], [],
                         FunctionEvaluator(lambda X: {}, outputs, fidelity="surrogate"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    n_test = max(2, int(round(test_fraction * len(data))))
    te, tr = perm[:n_test], perm[n_test:]
    Y = {k: data[k].to_numpy() for k in outputs}
    gp = GPSurrogate(stub, outputs).fit(X[tr], {k: v[tr] for k, v in Y.items()})
    P = gp(X[te])
    metrics = {}
    for k in outputs:
        y, p = Y[k][te], P[k]
        err = p - y
        nz = np.abs(y) > 1e-12
        ss = float(np.sum((y - y.mean()) ** 2))
        metrics[k] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                      "mean_rel": float(np.mean(np.abs(err[nz] / y[nz]))) if nz.any() else None,
                      "max_rel": float(np.max(np.abs(err[nz] / y[nz]))) if nz.any() else None,
                      "r2": (1 - float(np.sum(err ** 2)) / ss) if ss > 0 else None}
    gp_all = GPSurrogate(stub, outputs, name="gp-surrogate (your data)").fit(X, Y)
    return SurrogateFit(space, gp_all, outputs, metrics, len(tr), len(te))


def _first_objective(outputs):
    return Objective(outputs[0])


def model_ref_for_surrogate(fit: SurrogateFit, name: str, uri: str, dataset_name: str = ""):
    """``ModelRef`` of a table-trained surrogate: inputs with their validity range, held-out metrics."""
    from pinneapple_data.contracts import DatasetRef, ModelRef
    metrics = {}
    for k, m in fit.metrics.items():
        for key in ("mae", "rmse", "mean_rel", "r2"):
            if m.get(key) is not None and np.isfinite(m[key]):
                metrics[f"{k}_{key}"] = float(m[key])
    return ModelRef(name=name, model_type="gp", uri=uri, framework="scikit-learn",
                    inputs=[{"name": v.name, "low": v.lower, "high": v.upper, "unit": v.unit}
                            for v in fit.space.variables],
                    outputs=[{"name": k} for k in fit.outputs], metrics=metrics, metrics_split="test",
                    dataset=DatasetRef(name=dataset_name) if dataset_name else None,
                    metadata={"n_train": fit.n_train, "n_test": fit.n_test})


def table_evaluator(space: DesignSpace, df, outputs: Sequence[str], *, tol: float = 1e-6,
                    name: str = "your solver results") -> FunctionEvaluator:
    """A reference evaluator answering from a table of solver results: each design is matched to the row with the
    same variable values (relative tolerance ``tol`` of each variable's range); designs without a row get NaN outputs
    and ``_valid = False``."""
    names = space.names
    missing = [c for c in names + list(outputs) if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in the results table: {missing}")
    T = df[names].to_numpy(dtype=float)
    V = {k: df[k].to_numpy(dtype=float) for k in outputs}
    span = space.bounds[:, 1] - space.bounds[:, 0]

    def fn(X):
        out = {k: np.full(len(X), np.nan) for k in outputs}
        valid = np.zeros(len(X), bool)
        for i, x in enumerate(X):
            d = np.max(np.abs(T - x) / span, 1)
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= tol:
                valid[i] = True
                for k in outputs:
                    out[k][i] = V[k][j]
        out[VALID] = valid
        return out

    return FunctionEvaluator(fn, list(outputs), name=name, fidelity="reference",
                             description="solver results uploaded as a table, matched by design variables")


# --------------------------------------------------------------------------- after a run: candidates, files, plots
DESIGN_FILES = ("evaluations.csv", "candidates.csv", "candidates_to_run.csv", "parameter_space.json")


def verify_front(problem: DesignProblem, ev: Evaluations, front: Sequence[int], n: int, reference=None):
    """``n`` candidates spread along ``front`` (:func:`~.verify.select_candidates`), re-evaluated with ``reference``
    when there is one (:func:`~.verify.verify_candidates`).

    Returns ``(candidates, report, status)``: the indices into ``ev``, the :class:`~.verify.VerificationReport` (None
    without a reference or candidates) and the verification status per index (empty without a report)."""
    from .verify import select_candidates, verify_candidates
    cand = select_candidates(ev, front, n)
    if reference is None or not cand:
        return cand, None, {}
    rep = verify_candidates(problem, ev, cand, reference)
    return cand, rep, {r["index"]: r["status"] for r in rep.rows}


def write_design_files(folder, problem: DesignProblem, ev: Evaluations, candidates: Sequence[int],
                       status: Mapping[int, str] | None = None, *, labels: Sequence[str] | None = None) -> list[str]:
    """Write :data:`DESIGN_FILES` to ``folder``: every evaluation, the candidates with their verification status,
    the candidates' variables only (to run in your solver) and the design space as a ``ParameterSpace``.

    ``labels`` (one per evaluation, e.g. :meth:`~.compare.MethodComparison.union`'s runs) adds a ``run`` column to
    the evaluations and a ``found_by`` column to the candidates. Returns the file names written."""
    from pathlib import Path
    folder = Path(folder)
    edf = designs_frame(problem, ev)
    if labels is not None:
        edf["run"] = list(labels)
    edf.to_csv(folder / "evaluations.csv", index=False)
    cdf = designs_frame(problem, ev, list(candidates), status or {})
    if labels is not None:
        cdf["found_by"] = [labels[i] for i in candidates]
    cdf.to_csv(folder / "candidates.csv", index=False)
    cdf.reindex(columns=problem.space.names).to_csv(folder / "candidates_to_run.csv", index=False)
    to_parameter_space(problem.space, problem.name, problem.description).save(folder / "parameter_space.json")
    return list(DESIGN_FILES)


def chart_points(problem: DesignProblem, ev: Evaluations, front: Sequence[int] = (), *, limit: int = 2000,
                 labels: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """At most about ``limit`` designs for a trade-off plot, evenly thinned in evaluation order, the ``front`` always
    kept. Each point: ``i`` (index), ``f`` (feasible), ``p`` (on ``front``), ``r`` (relaxed), the objective and
    constraint values (None when not finite) and, with ``labels``, ``run`` (the label up to its ``#``)."""
    keys = list(dict.fromkeys([o.name for o in problem.objectives] + [c.name for c in problem.constraints]))
    keys = [k for k in keys if k in ev.Y]
    front = {int(i) for i in front}
    idx = np.arange(len(ev))
    if len(idx) > limit:
        idx = np.unique(np.r_[np.linspace(0, len(ev) - 1, limit).astype(int), sorted(front)]).astype(int)
    out = []
    for i in idx:
        i = int(i)
        p = {"i": i, "f": bool(ev.feasible[i]), "p": i in front, "r": bool(ev.relaxed[i]),
             **{k: (float(ev.Y[k][i]) if np.isfinite(ev.Y[k][i]) else None) for k in keys}}
        if labels is not None:
            p["run"] = str(labels[i]).split("#")[0]
        out.append(p)
    return out


def verify_uploaded_results(space: DesignSpace, candidates, results, objectives: Sequence[Any] = (),
                            constraints: Sequence[Any] = (), *, tol: float = 1e-6):
    """Verify exported candidates with solver results run offline.

    ``candidates`` is the table :func:`write_design_files` wrote (``candidates.csv``: variables and the search
    model's outputs), ``results`` your solver's table (the variables and any outputs). Rows are matched by variable
    values (:func:`table_evaluator`); the objectives and constraints whose output is in the candidates are checked
    on the uploaded values. Returns the :class:`~.verify.VerificationReport` (status per candidate, errors of the
    search model against your results). Raises ``ValueError`` when the results share no output with the candidates
    or miss a variable."""
    import pandas as pd

    from .verify import verify_candidates
    shared = [k for k in results.columns if k not in space.names and k in candidates.columns]
    if not shared:
        raise ValueError("the results share no output column with the candidates")
    num = [k for k in candidates.columns if k not in space.names and k not in ("index", "feasible", "relaxed")
           and pd.api.types.is_numeric_dtype(candidates[k]) and not pd.api.types.is_bool_dtype(candidates[k])]
    objs = [o for o in objectives if o.name in num] or [Objective(shared[0])]
    stub = DesignProblem("uploaded", space, objs, [c for c in constraints if c.name in num],
                         FunctionEvaluator(lambda X: {}, num, fidelity="surrogate"))
    X = candidates[space.names].to_numpy(dtype=float)
    ev = stub.record(X, {k: candidates[k].to_numpy(dtype=float) for k in num}, fidelity="surrogate")
    ref = table_evaluator(space, results, shared, tol=tol)
    # the reference is checked on what was uploaded: objectives and constraints of the outputs it has
    ref_objs = [o for o in objs if o.name in shared] or [Objective(shared[0])]
    ref_problem = DesignProblem("uploaded results", space, ref_objs, [c for c in constraints if c.name in shared], ref)
    return verify_candidates(stub, ev, list(range(len(ev))), ref, reference_problem=ref_problem)
