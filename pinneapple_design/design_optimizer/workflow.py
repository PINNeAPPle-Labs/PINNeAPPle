"""The standard design-optimisation workflow (#1188): geometry and parameters of a library tool optimised under
constraints, the optimum re-verified on refined meshes with the mesh error stated, and checks on the answer.

1. **Problem**: a tool (a callable or a library tool id ``"pinneapple...module:function"`` with keyword inputs, e.g.
   :func:`.cantilever.tapered_cantilever`, a gmsh-parametrized geometry solved by FEM), the ``bounds`` of the design
   variables (dimensions of the geometry and parameters alike), the ``objective`` and the ``constraints`` on its
   outputs, and the tool keyword that sets the mesh (``fidelity``: a coarse value for the search and the refined
   values for the verification).
2. **Cost and method**: one tool call at the coarse mesh is timed;
   :func:`~pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` says whether a
   surrogate pays off for the evaluations the search needs. When it does, the search runs on a Gaussian-process
   surrogate refined where the optimiser looks (:func:`.verify.surrogate_assisted`); otherwise directly on the tool.
   ``method="auto"``: gradient descent (autograd) when the evaluator has adjoints, SLSQP (finite differences) for one
   objective, NSGA-II for several. The search runs from ``n_starts`` seeded starting designs.
3. **Re-verification**: the optimum is re-solved on the refined meshes; the converged value of every objective and
   constraint is the Richardson extrapolation of the last three meshes, with the observed order and the grid
   convergence index (Roache, 1994; Celik et al., 2008, J. Fluids Eng. 130:078001). The coarse-mesh error is
   corrected (each output scaled by verified / coarse at the optimum, output space mapping) and the search repeated
   until the correction stops changing (``max_corrections``).
4. **Checks**: optimiser convergence (the starts agree; the correction loop converged), mesh quality and mesh
   convergence, every constraint satisfied by the verified values, first-order optimality and sensitivities at the
   optimum (finite differences, active set, KKT multipliers = the objective's price of each active limit),
   robustness (Monte Carlo of manufacturing tolerances: the share of designs that still satisfy every constraint;
   ``robust=True`` backs the limits off by the 95th-percentile spread and re-optimises), and the known optimum when a
   ``reference`` model is given (e.g. :func:`.cantilever.tapered_cantilever_beam`, beam theory).
5. **Export**: ``export(**design, out_dir=...)`` writes the geometry of the optimum (STEP for the cantilever, read
   back and its volume compared with the design's).

Several objectives: NSGA-II, and the picks along the front (:func:`.verify.select_candidates`) re-verified on the
refined meshes; the correction loop and the known-optimum check apply to one objective.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .study import Constraint, DesignProblem, DesignSpace, FunctionEvaluator, Objective, Variable

__all__ = ["design_optimization", "richardson", "DEFAULT_CANTILEVER"]

SICN_MIN = 0.1                 # pinneapple_design.geometry.mesh.generate.SICN_WARN

# the bundled case: minimum mass of a tapered cantilever (depths and width) under a stress and a deflection limit
DEFAULT_CANTILEVER: dict[str, Any] = {
    "tool": "pinneapple_design.design_optimizer.cantilever:tapered_cantilever",
    "reference": "pinneapple_design.design_optimizer.cantilever:tapered_cantilever_beam",
    "export": "pinneapple_design.design_optimizer.cantilever:export_step",
    "bounds": {"h_root": [0.04, 0.20], "h_tip": [0.02, 0.10], "thickness": [0.01, 0.04]},
    "objective": {"name": "mass_kg", "sense": "min"},
    "constraints": [{"name": "stress_max_Pa", "upper": 150e6}, {"name": "tip_deflection_m", "upper": 8e-3}],
    "fidelity": {"param": "n_through", "coarse": 4, "levels": [8, 16, 32]},
    "units": {"h_root": "m", "h_tip": "m", "thickness": "m", "mass_kg": "kg", "stress_max_Pa": "Pa",
              "tip_deflection_m": "m"},
}


def richardson(values: Sequence[float], ratio: float = 2.0, safety: float = 1.25) -> dict[str, Any]:
    """Converged value of an output from meshes refined by ``ratio`` (coarse to fine).

    With three or more values: observed order ``p = ln(e21 / e32) / ln(r)`` from the last three, extrapolated value
    ``f3 + (f3 - f2) / (r^p - 1)`` and the fine-grid convergence index ``safety |f3 - f2| / |f3| / (r^p - 1)``.
    When the differences change sign or vanish (no monotone convergence) the finest value is kept and its error is
    the last relative difference (``order: None``)."""
    f = [float(v) for v in values]
    fine = f[-1]
    if len(f) < 2:
        return {"value": fine, "order": None, "gci": None, "monotone": None, "values": f}
    scale = max(max(abs(v) for v in f), 1e-300)
    e32 = f[-1] - f[-2]
    e32 = 0.0 if abs(e32) <= 1e-12 * scale else e32           # round-off: the output does not depend on the mesh
    rel_last = abs(e32) / max(abs(fine), 1e-300)
    if len(f) >= 3:
        e21 = f[-2] - f[-3]
        e21 = 0.0 if abs(e21) <= 1e-12 * scale else e21
        if e21 != 0 and e32 != 0 and e21 * e32 > 0 and abs(e32) < abs(e21):
            p = math.log(abs(e21 / e32)) / math.log(ratio)
            ext = fine + e32 / (ratio ** p - 1)
            gci = safety * rel_last / (ratio ** p - 1)
            return {"value": ext, "order": p, "gci": gci, "monotone": True, "values": f, "finest": fine}
        if e32 == 0 and e21 == 0:
            return {"value": fine, "order": None, "gci": 0.0, "monotone": True, "values": f, "finest": fine}
    return {"value": fine, "order": None, "gci": rel_last, "monotone": False, "values": f, "finest": fine}


def _resolve(tool):
    if tool is None or callable(tool):
        return tool
    from pinneapple_autolab.study.verify import resolve_model
    return resolve_model(tool)


def _spec_objectives(objective) -> list[Objective]:
    objs = objective if isinstance(objective, (list, tuple)) else [objective]
    out = []
    for o in objs:
        if isinstance(o, str):
            out.append(Objective(o))
        else:
            out.append(Objective(str(o["name"]), str(o.get("sense", "min")), str(o.get("unit", ""))))
    return out


def _spec_constraints(constraints) -> list[Constraint]:
    out = []
    for c in constraints or ():
        if isinstance(c, Constraint):
            out.append(c)
            continue
        out.append(Constraint(str(c["name"]), None if c.get("upper") is None else float(c["upper"]),
                              None if c.get("lower") is None else float(c["lower"]), str(c.get("unit", "")),
                              str(c.get("kind", "physical"))))
    return out


def design_optimization(*, tool: str | Callable[..., Any] | None = None,
                        bounds: Mapping[str, Sequence[float]] | None = None,
                        objective: Any = None, constraints: Sequence[Any] = (),
                        fixed: Mapping[str, Any] | None = None, fidelity: Mapping[str, Any] | None = None,
                        method: str = "auto", budget: int = 80, n_starts: int = 2, seed: int = 0,
                        reference: str | Callable[..., Any] | None = None,
                        export: str | Callable[..., Any] | None = None,
                        tolerances: Mapping[str, float] | None = None, n_robust: int = 200,
                        robust_target: float = 0.95, robust: bool = False, max_corrections: int = 3,
                        convergence_tol: float = 0.01, constraint_tol: float = 0.005, gci_max: float = 0.05,
                        reference_tol: float = 0.02, accuracy_target: float = 0.02, force_surrogate: bool = False,
                        out_dir: str | None = None, units: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run the workflow (see the module docstring); returns a JSON-ready record with ``design``, ``verified``,
    ``checks`` and, with ``out_dir`` and ``export``, the geometry file of the optimum.

    With no ``tool``, the bundled case :data:`DEFAULT_CANTILEVER` runs (its ``bounds``, objective, constraints,
    reference and export are used unless given)."""
    from . import methods as M
    from .verify import select_candidates, surrogate_assisted
    t_start = time.perf_counter()
    if tool is None:
        d = DEFAULT_CANTILEVER
        tool, reference = d["tool"], reference or d["reference"]
        export = export or d["export"]
        bounds = bounds or d["bounds"]
        objective = objective or d["objective"]
        constraints = constraints or d["constraints"]
        fidelity = fidelity or d["fidelity"]
        units = units or d["units"]
    if not bounds or objective is None:
        raise ValueError("a tool needs the bounds of its design variables and an objective")
    fn, ref_fn, exp_fn = _resolve(tool), _resolve(reference), _resolve(export)
    fixed, units = dict(fixed or {}), dict(units or {})
    fid = dict(fidelity or {})
    fparam = fid.get("param")
    coarse = fid.get("coarse")
    levels = list(fid.get("levels") or [])
    ratio = float(fid.get("ratio") or (levels[1] / levels[0] if len(levels) >= 2 else 2.0))
    names = list(bounds)
    space = DesignSpace([Variable(k, float(bounds[k][0]), float(bounds[k][1]), unit=units.get(k, "")) for k in names])
    objs, cons = _spec_objectives(objective), _spec_constraints(constraints)
    outs = list(dict.fromkeys([o.name for o in objs] + [c.name for c in cons]))

    def call(design: Mapping[str, float], level=None) -> dict[str, Any]:
        kw = {**fixed, **{k: float(v) for k, v in design.items()}}
        if fparam is not None:
            kw[fparam] = coarse if level is None else level
        r = fn(**kw)
        return dict(r) if isinstance(r, Mapping) else {o: getattr(r, o) for o in outs}

    def evaluator(scale: Mapping[str, float] | None = None, name: str = "tool (coarse mesh)") -> FunctionEvaluator:
        sc = dict(scale or {})

        def f(X):
            rows = [call(dict(zip(names, map(float, x), strict=True))) for x in np.atleast_2d(X)]
            return {o: np.array([float(r[o]) * sc.get(o, 1.0) for r in rows]) for o in outs}
        return FunctionEvaluator(f, outs, name=name, fidelity="model",
                                 description="the tool at the search mesh" + (", corrected" if sc else ""))

    # 2. cost and the method
    x_mid = {k: 0.5 * (float(bounds[k][0]) + float(bounds[k][1])) for k in names}
    t0 = time.perf_counter()
    base = call(x_mid)
    solve_s = time.perf_counter() - t0
    n_evals = int(budget) * int(n_starts) * (1 + int(max_corrections))
    rec: dict[str, Any] = {"recommended": ["none"], "reasoning": "recommendation unavailable"}
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
        r = recommend_architecture(n_high_fidelity_simulations=0, needs_parameter_generalization=True,
                                   geometry_varies=False, has_analytical_solver=True, solve_seconds=solve_s,
                                   n_evaluations=n_evals, output_kind="scalar", n_inputs=len(names),
                                   accuracy_target=accuracy_target)
        rec = {"recommended": list(r.recommended), "reasoning": r.reasoning, "decision_path": list(r.decision_path)}
    except (ImportError, TypeError) as e:
        rec["reasoning"] = f"recommendation unavailable: {e}"
    use_surrogate = bool(force_surrogate or rec["recommended"][0] != "none")
    ev0 = evaluator()
    if method == "auto":
        method = "gradient" if ev0.differentiable else ("scipy-slsqp" if len(objs) == 1 else "nsga2")
    why_method = {"gradient": "the evaluator has adjoints (autograd)",
                  "scipy-slsqp": "one objective and no adjoint: SLSQP with finite differences on a mesh that "
                                 "moves with the geometry",
                  "nsga2": "several objectives: NSGA-II"}.get(method, "chosen by the caller")

    def search(ev: FunctionEvaluator, cons_now: list[Constraint], x0: np.ndarray | None, tag: str):
        prob = DesignProblem("design optimisation", space, objs, cons_now, ev)
        runs = []
        for s in range(int(n_starts)):
            if use_surrogate:
                al = surrogate_assisted(prob, ev, method="nsga2" if len(objs) > 1 else "scipy-de",
                                        n_initial=max(8, 4 * len(names)), rounds=2, n_verify=4,
                                        surrogate_budget=max(200, 10 * int(budget)), seed=seed + s, kappa=0.0)
                b = _best(al.final) if len(objs) == 1 else None
                if b is None:
                    runs.append(al.final)
                    continue
                # polish on the tool from the surrogate's best verified design (the surrogate found the basin)
                runs.append(M.run(prob, method, budget=int(budget), seed=seed + s,
                                  initial=al.final.evaluations.X[[b]]))
            else:
                init = None if (x0 is None or s > 0) else np.atleast_2d(x0)
                runs.append(M.run(prob, method, budget=int(budget), seed=seed + 101 * s, initial=init))
        return prob, runs

    def best_of(runs) -> tuple[int, int] | None:
        best = None
        for i, r in enumerate(runs):
            b = _best(r)
            if b is not None and (best is None or r.evaluations.F[b, 0] < runs[best[0]].evaluations.F[best[1], 0]):
                best = (i, b)
        return best

    def verify(design: Mapping[str, float]) -> dict[str, Any]:
        rows, wall = [], []
        for lv in [None, *levels]:
            t = time.perf_counter()
            rows.append(call(design, lv))
            wall.append(time.perf_counter() - t)
        per = {}
        for o in outs:
            vals = [float(r[o]) for r in rows[1:]] or [float(rows[0][o])]
            rr = richardson(vals, ratio)
            rr["coarse"] = float(rows[0][o])
            per[o] = rr
        sicn = [float(r["sicn_min"]) for r in rows if "sicn_min" in r]
        return {"outputs": per, "sicn_min": min(sicn) if sicn else None, "levels": [coarse, *levels],
                "wall_s": wall, "n_elements": [r.get("n_elements") for r in rows]}

    # 3. search, re-verification, correction
    scale: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    cons_now = list(cons)
    x_best = None
    multi = len(objs) > 1
    for it in range(1 + (0 if multi else int(max_corrections))):
        prob, runs = search(evaluator(scale), cons_now, x_best, f"iteration {it}")
        if multi:
            break
        bi = best_of(runs)
        if bi is None:
            history.append({"iteration": it, "status": "no feasible design found", "scale": dict(scale)})
            break
        x_best = runs[bi[0]].evaluations.X[bi[1]].copy()
        design = dict(zip(names, map(float, x_best), strict=True))
        ver = verify(design)
        new = {o: ver["outputs"][o]["value"] / ver["outputs"][o]["coarse"] for o in outs
               if ver["outputs"][o]["coarse"] not in (0.0,)}
        bests = [float(r.evaluations.F[_best(r), 0]) * (1 if objs[0].sense == "min" else -1) for r in runs
                 if _best(r) is not None]
        change = max((abs(new[o] / scale.get(o, 1.0) - 1) for o in new), default=0.0)
        history.append({"iteration": it, "design": design, "objective_by_start": bests,
                        "n_evals": [r.n_evals for r in runs], "correction": new, "correction_change": change,
                        "verification": ver})
        scale = {o: new[o] for o in new}
        if change <= 0.1 * convergence_tol:
            break

    out: dict[str, Any] = {"variables": names, "bounds": {k: list(map(float, bounds[k])) for k in names},
                           "objectives": [o.name for o in objs], "constraints": [c.describe() for c in cons],
                           "fixed": fixed, "fidelity": {"param": fparam, "coarse": coarse, "levels": levels,
                                                        "ratio": ratio},
                           "method": method if not use_surrogate else f"surrogate-assisted ({method})",
                           "method_reason": why_method, "recommendation": rec,
                           "cost": {"solve_seconds": solve_s, "evaluations_planned": n_evals},
                           "baseline": {"design": x_mid, "outputs": {o: _num(base.get(o)) for o in outs}},
                           "units": units}
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None, status=None):
        ok = None if ok is None else bool(ok)
        st = status or ("pass" if ok is True else "fail" if ok is False else "inconclusive")
        checks.append({"code": code, "status": st, "detail": detail, "value": _num(value)})

    chk("surrogate.decision", True, ("a surrogate pays off: the search ran on a Gaussian process refined where the "
                                     "optimiser looked, every candidate re-solved by the tool")
        if use_surrogate else "no surrogate: " + (rec.get("decision_path") or [rec["reasoning"]])[0], solve_s)

    if multi:
        res = max(runs, key=lambda r: len(r.pareto()))
        ev = res.evaluations
        front = res.pareto()
        picks = select_candidates(ev, front, 3) if ev is not None else []
        out["pareto"] = [{"design": dict(zip(names, map(float, ev.X[i]), strict=True)),
                          "verification": verify(dict(zip(names, map(float, ev.X[i]), strict=True)))} for i in picks]
        for p in out["pareto"]:
            for c in cons:
                v = p["verification"]["outputs"][c.name]["value"]
                chk(f"pareto.{c.name}", bool(c.violation(np.array([v]))[0] <= constraint_tol),
                    f"{c.describe()}: verified {v:.6g} at a Pareto pick", v)
        out["checks"] = checks
        out["verdict"] = "fail" if any(c["status"] == "fail" for c in checks) or not picks else "pass"
        out["wall_s"] = time.perf_counter() - t_start
        return out
    if x_best is None:
        chk("optimizer.feasible", False, "no design satisfied every constraint on the search mesh")
        out.update(checks=checks, verdict="fail", history=_slim(history), wall_s=time.perf_counter() - t_start)
        return out

    final = history[-1]
    design, ver = final["design"], final["verification"]
    ev_corr = evaluator(scale, "tool (coarse mesh, corrected)")

    def g(x: np.ndarray) -> dict[str, np.ndarray]:
        return ev_corr(np.atleast_2d(x))

    # robustness first (it may move the design), then the checks on the final design
    tol = _tolerances(tolerances, design, space)
    rob = _robustness(g, design, names, tol, cons, n_robust, seed)
    robust_info = None
    cons_final = list(cons)
    if robust and rob["p_feasible"] < robust_target:
        backed = []
        for c in cons:
            bo = rob["backoff"].get(c.name, 0.0)
            if c.upper is not None:
                backed.append(Constraint(c.name, c.upper - bo, c.lower, c.unit, c.kind, c.label))
            else:
                backed.append(Constraint(c.name, c.upper, (c.lower or 0.0) + bo, c.unit, c.kind, c.label))
        _, runs_r = search(ev_corr, backed, x_best, "robust")
        bi = best_of(runs_r)
        if bi is not None:
            x_r = runs_r[bi[0]].evaluations.X[bi[1]]
            d_r = dict(zip(names, map(float, x_r), strict=True))
            robust_info = {"nominal_design": design, "nominal_objective": ver["outputs"][objs[0].name]["value"],
                           "backed_off_limits": [c.describe() for c in backed]}
            design, ver, x_best, cons_final = d_r, verify(d_r), x_r, backed
            robust_info["objective_cost"] = ver["outputs"][objs[0].name]["value"] / robust_info["nominal_objective"] - 1
            rob = _robustness(g, design, names, tol, cons, n_robust, seed)
            history.append({"iteration": "robust", "design": design, "verification": ver,
                            "objective_by_start": [float(r.evaluations.F[_best(r), 0]) for r in runs_r
                                                   if _best(r) is not None]})
    out["robust"] = robust_info
    out["design"] = design
    out["verified"] = {o: {k: _num(v) if not isinstance(v, list) else [_num(x) for x in v]
                           for k, v in ver["outputs"][o].items()} for o in outs}
    out["mesh"] = {"levels": ver["levels"], "n_elements": ver["n_elements"], "sicn_min": _num(ver["sicn_min"]),
                   "wall_s": [round(w, 4) for w in ver["wall_s"]]}
    out["history"] = _slim(history)

    # convergence of the optimiser
    spread = _spread(final.get("objective_by_start") or [])
    n_corr = sum(1 for h in history if isinstance(h["iteration"], int))
    corr_ok = final.get("correction_change", 0.0) <= 0.1 * convergence_tol or robust_info is not None
    chk("optimizer.converged", spread is not None and spread <= convergence_tol and corr_ok,
        f"{len(final.get('objective_by_start') or [])} starts agree to {spread:.3%} (at most {convergence_tol:.1%}); "
        f"mesh correction after {n_corr} search(es) changed by {final.get('correction_change', 0.0):.3%}"
        if spread is not None else "fewer than two starts found a feasible design", spread)
    # mesh
    if ver["sicn_min"] is not None:
        chk("mesh.quality", ver["sicn_min"] > SICN_MIN,
            f"worst element over the search and verification meshes: SICN {ver['sicn_min']:.3f} (above {SICN_MIN})",
            ver["sicn_min"])
    if levels:
        worst = max(outs, key=lambda o: ver["outputs"][o]["gci"] or 0.0)
        gw = ver["outputs"][worst]["gci"]
        orders = {o: ver["outputs"][o]["order"] for o in outs}
        chk("mesh.convergence", gw is not None and gw <= gci_max and all(ver["outputs"][o]["monotone"] is not False
                                                                         for o in outs),
            f"grid convergence index at the optimum, worst {worst}: {gw:.3%} (at most {gci_max:.0%}); observed "
            "orders " + ", ".join(f"{o} {p:.2f}" if p else f"{o} n/a" for o, p in orders.items()), gw)
        for o in outs:
            cr = ver["outputs"][o]
            chk(f"mesh.coarse_error.{o}", None, f"search mesh {cr['coarse']:.6g} vs converged {cr['value']:.6g} "
                f"({cr['coarse'] / cr['value'] - 1:+.2%}); corrected in the search", cr["coarse"] / cr["value"] - 1,
                status="info")
    # constraints on the verified values
    for c in cons:
        v = ver["outputs"][c.name]["value"]
        gci = ver["outputs"][c.name]["gci"] or 0.0
        viol = float(c.violation(np.array([v]))[0])
        chk(f"constraints.{c.name}", viol <= constraint_tol,
            f"{c.describe()}: verified {v:.6g} (mesh band ±{gci:.2%}); violation {max(viol, 0):.3%} "
            f"(at most {constraint_tol:.1%})", v)
    # sensitivity and first-order optimality
    sens = _sensitivity(g, x_best, space, objs[0], cons_final, outs)
    out["sensitivity"] = sens
    chk("optimality.kkt", sens["kkt_residual"] <= 0.05 and all(m >= -1e-9 for m in sens["multipliers"].values()),
        f"stationarity residual {sens['kkt_residual']:.2%} of |grad objective| with active "
        f"{', '.join(sens['active']) or 'nothing'}; multipliers "
        + ", ".join(f"{k} {v:.4g}" for k, v in sens["multipliers"].items()), sens["kkt_residual"])
    # robustness
    out["robustness"] = rob
    chk("robustness.tolerances", rob["p_feasible"] >= robust_target,
        f"{rob['p_feasible']:.1%} of {n_robust} designs within the tolerances "
        + ", ".join(f"{k} ±{v:.3g}" for k, v in tol.items())
        + f" satisfy every constraint (target {robust_target:.0%}); back-off for 95 %: "
        + ", ".join(f"{k} {v:.4g}" for k, v in rob["backoff"].items()), rob["p_feasible"],
        status=None if rob["p_feasible"] >= robust_target else "warn")
    # known optimum
    if ref_fn is not None:
        known = _reference_optimum(ref_fn, fixed, names, space, objs[0], cons, seed)
        out["reference"] = known
        if known is not None:
            # the nominal optimum (before any robust back-off) is the one comparable with the reference's
            d_nom = robust_info["nominal_design"] if robust_info else design
            fo = robust_info["nominal_objective"] if robust_info else ver["outputs"][objs[0].name]["value"]
            dobj = abs(fo / known["objective"] - 1)
            dx = max(abs(d_nom[k] - known["design"][k]) / (space.bounds[j, 1] - space.bounds[j, 0])
                     for j, k in enumerate(names))
            at = ref_fn(**{**fixed, **d_nom})
            viol = max((float(c.violation(np.array([float(at[c.name])]))[0]) for c in cons), default=0.0)
            known["at_found_design"] = {o: _num(at.get(o)) for o in outs}
            known["design_distance"] = dx
            chk("reference.known_optimum", dobj <= reference_tol and viol <= reference_tol,
                f"{'nominal ' if robust_info else ''}{objs[0].name}: {fo:.6g} vs the reference optimum "
                f"{known['objective']:.6g} ({dobj:.2%}, at most {reference_tol:.0%}); the reference model at that "
                f"design violates its limits by {max(viol, 0):.2%} (at most {reference_tol:.0%}); largest design "
                f"difference {dx:.2%} of a range (optima need not be unique)", dobj)
    # export
    if exp_fn is not None and out_dir:
        import inspect
        accepted = set(inspect.signature(exp_fn).parameters)
        e = exp_fn(**{**{k: v for k, v in fixed.items() if k in accepted}, **design}, out_dir=out_dir)
        out["export"] = {k: _num(v) if isinstance(v, (int, float)) else v for k, v in e.items()}
        if "volume_read_m3" in e and "volume_design_m3" in e:
            dv = abs(e["volume_read_m3"] / e["volume_design_m3"] - 1)
            chk("export.geometry", dv <= 1e-6, f"{os.path.basename(e['path'])} read back: volume "
                f"{e['volume_read_m3']:.6g} m3 vs the design's {e['volume_design_m3']:.6g} m3", dv)
        else:
            chk("export.geometry", os.path.exists(e.get("path", "")), f"written {e.get('path')}")
    elif exp_fn is not None:
        chk("export.geometry", None, "no out_dir: the geometry of the optimum was not written", status="not run")
    out["checks"] = checks
    bad = [c["code"] for c in checks if c["status"] == "fail"]
    out["verdict"] = "fail" if bad else "pass"
    out["wall_s"] = time.perf_counter() - t_start
    out["assumptions"] = [
        "the search uses the coarse mesh scaled by verified / coarse at the last optimum (output space mapping)",
        "verified values are Richardson extrapolations of the refined meshes; the band is the grid convergence "
        "index",
        "sensitivities are central finite differences of the corrected search model (1e-3 of each range)",
        "robustness: uniform tolerances, the corrected search model, every constraint at once",
    ]
    return out


# --------------------------------------------------------------------------- helpers
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    return v if math.isfinite(v) else None


FEAS_TOL = 1e-9               # normalised violation treated as round-off (an optimum sits on its active limit)


def _best(run) -> int | None:
    """Index of the best realisable design of a run whose every constraint holds to round-off.

    ``OptimizationResult.best`` demands a violation of exactly 0; a gradient method ends on the active limit, where
    ``x + y = 2.0000000000000004`` counts as infeasible and the optimum would be discarded for a worse point."""
    ev = run.evaluations
    if ev is None or not len(ev):
        return None
    viol = ev.G.max(1) if ev.G.size else np.zeros(len(ev))
    valid = np.asarray(ev.Y.get("_valid", np.ones(len(ev), bool)), dtype=bool)
    ok = np.where(valid & ~ev.relaxed & (viol <= FEAS_TOL) & np.all(np.isfinite(ev.F), 1))[0]
    return None if not len(ok) else int(ok[np.argmin(ev.F[ok, 0])])


def _spread(vals: Sequence[float]) -> float | None:
    if len(vals) < 2:
        return None
    v = np.asarray(vals, float)
    return float((v.max() - v.min()) / max(abs(v.min()), 1e-300))


def _slim(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in history:
        s = {k: v for k, v in h.items() if k != "verification"}
        if "verification" in h:
            s["verified"] = {o: _num(r["value"]) for o, r in h["verification"]["outputs"].items()}
        out.append(s)
    return out


def _tolerances(tolerances, design, space: DesignSpace) -> dict[str, float]:
    if tolerances:
        return {k: float(tolerances.get(k, 0.0)) for k in space.names}
    return {k: 0.01 * abs(design[k]) for k in space.names}           # 1 % of each dimension


def _robustness(g, design, names, tol, cons, n, seed) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 7)
    x0 = np.array([design[k] for k in names])
    t = np.array([tol[k] for k in names])
    X = x0 + rng.uniform(-1, 1, (int(n), len(names))) * t
    Y = g(X)
    ok = np.ones(len(X), bool)
    nominal = g(x0)
    backoff, per = {}, {}
    for c in cons:
        v = Y[c.name]
        ok &= c.violation(v) <= 0
        v0 = float(nominal[c.name][0])
        if c.upper is not None:
            backoff[c.name] = max(float(np.percentile(v, 95)) - v0, 0.0)
        else:
            backoff[c.name] = max(v0 - float(np.percentile(v, 5)), 0.0)
        per[c.name] = {"p_satisfied": float(np.mean(c.violation(v) <= 0)), "p5": float(np.percentile(v, 5)),
                       "p95": float(np.percentile(v, 95))}
    return {"p_feasible": float(ok.mean()), "n": int(n), "tolerances": tol, "backoff": backoff, "per_constraint": per}


def _sensitivity(g, x, space: DesignSpace, obj: Objective, cons, outs) -> dict[str, Any]:
    x = np.asarray(x, float)
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    step = 1e-3 * (hi - lo)
    d = len(x)
    P = []
    for j in range(d):
        e = np.zeros(d)
        e[j] = step[j]
        a, b = np.clip(x + e, lo, hi), np.clip(x - e, lo, hi)
        P.append((a, b))
    Y0 = g(x)
    rows = g(np.array([p for ab in P for p in ab]))
    grad = {}
    for o in outs:
        gr = np.zeros(d)
        for j, (a, b) in enumerate(P):
            gr[j] = (rows[o][2 * j] - rows[o][2 * j + 1]) / (a[j] - b[j])
        grad[o] = gr
    sgn = 1.0 if obj.sense == "min" else -1.0
    gf = sgn * grad[obj.name]
    active, cols, labels = [], [], []
    for c in cons:
        v = float(Y0[c.name][0])
        if c.upper is not None and abs(c.upper - v) <= 0.01 * (abs(c.upper) or 1.0):
            active.append(c.name)
            cols.append(grad[c.name] / (abs(c.upper) or 1.0))
            labels.append(c.name)
        elif c.lower is not None and abs(v - c.lower) <= 0.01 * (abs(c.lower) or 1.0):
            active.append(c.name)
            cols.append(-grad[c.name] / (abs(c.lower) or 1.0))
            labels.append(c.name)
    span = hi - lo
    for j, k in enumerate(space.names):                 # active bounds
        if x[j] - lo[j] <= 1e-3 * span[j]:
            e = np.zeros(d)
            e[j] = -1.0
            cols.append(e * np.linalg.norm(gf))
            labels.append(f"{k} >= {lo[j]:g}")
            active.append(f"{k} at lower bound")
        elif hi[j] - x[j] <= 1e-3 * span[j]:
            e = np.zeros(d)
            e[j] = 1.0
            cols.append(e * np.linalg.norm(gf))
            labels.append(f"{k} <= {hi[j]:g}")
            active.append(f"{k} at upper bound")
    # stationarity: grad f + sum lambda_i grad g_i = 0 with lambda >= 0 (g_i <= 0 form); scale-free in x
    Ds = np.diag(span)
    if cols:
        from scipy.optimize import nnls
        A = Ds @ np.column_stack(cols)
        lam, resid = nnls(A, -Ds @ gf)
    else:
        lam, resid = np.array([]), float(np.linalg.norm(Ds @ gf))
    res_rel = float(resid / max(np.linalg.norm(Ds @ gf), 1e-300))
    mult = {}
    for lab, lv in zip(labels, lam, strict=True):
        if lab in [cc.name for cc in cons]:
            cc = next(cc for cc in cons if cc.name == lab)
            bound = cc.upper if cc.upper is not None else cc.lower
            mult[lab] = float(lv / (abs(bound) or 1.0))       # d objective / d limit, in objective units per unit
        else:
            mult[lab] = float(lv * np.linalg.norm(gf))
    v0 = {o: float(Y0[o][0]) for o in outs}
    elast = {o: {k: _num(grad[o][j] * x[j] / v0[o]) if v0[o] else None for j, k in enumerate(space.names)}
             for o in outs}
    return {"gradient": {o: dict(zip(space.names, map(float, grad[o]), strict=True)) for o in outs},
            "elasticity": elast, "active": active, "multipliers": mult, "kkt_residual": res_rel,
            "step": dict(zip(space.names, map(float, step), strict=True))}


def _reference_optimum(ref_fn, fixed, names, space: DesignSpace, obj: Objective, cons, seed) -> dict[str, Any] | None:
    """The optimum of the reference model: SLSQP from eight seeded starts, the best feasible end point."""
    from scipy.optimize import minimize
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    sgn = 1.0 if obj.sense == "min" else -1.0
    cache: dict[bytes, dict[str, Any]] = {}

    def at(u):
        key = np.round(u, 14).tobytes()
        if key not in cache:
            kw = {**{k: v for k, v in fixed.items()}, **dict(zip(names, map(float, lo + u * (hi - lo)), strict=True))}
            cache[key] = ref_fn(**kw)
        return cache[key]

    def f(u):
        return sgn * float(at(u)[obj.name])

    def gcon(u):
        r = at(u)
        return np.array([float(c.slack(float(r[c.name]))) for c in cons]) if cons else np.zeros(1)
    best = None
    rng = np.random.default_rng(seed + 11)
    for u0 in np.r_[[np.full(len(names), 0.5)], rng.random((7, len(names)))]:
        r = minimize(f, u0, method="SLSQP", bounds=[(0, 1)] * len(names),
                     constraints=[{"type": "ineq", "fun": gcon}] if cons else [],
                     options={"ftol": 1e-12, "maxiter": 500})
        if np.all(gcon(r.x) >= -1e-9) and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        return None
    x = lo + best.x * (hi - lo)
    vals = at(best.x)
    return {"design": dict(zip(names, map(float, x), strict=True)), "objective": float(vals[obj.name]),
            "outputs": {c.name: float(vals[c.name]) for c in cons},
            "method": "SLSQP from 8 seeded starts on the reference model"}
