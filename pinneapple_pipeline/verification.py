"""VeriPhysics: the gates a run has to pass, and the evidence each one leaves.

Every gate reads only what earlier stages measured and returns a :class:`~pinneapple_pipeline.results.
VerificationResult` with the value, the threshold and what to do about it. None of them invents a number: when the
evidence a gate needs was not produced, the gate is ``NOT_EVALUATED`` and says what was missing — the pipeline then
applies ``verification.fail_policy`` to it. That is the whole point of the layer: a run is never approved because its
training loss went down.

===========================  ==========================================================================
gate                         what it establishes
===========================  ==========================================================================
``problem``                  the configuration and the problem agree (dimension, names, parameters, conditions)
``geometry``                 the geometry is the one declared (not applicable without one)
``discretization``           the reference solution does not move when the grid is refined
``simulation``               every reference solve the data rests on finished
``dataset``                  the cases cover the space, are distinct and are split without leakage
``model``                    the error on data the model never saw, against the acceptance criterion
``physics``                  the PDE residual and the boundary/initial conditions of the trained model
``applicability``            where the model stays valid, and what happens outside it
``export``                   the exported model predicts what the in-memory model predicts
``inference``                a fresh query returns finite values from inside the validated domain
===========================  ==========================================================================
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .results import FAIL, NOT_APPLICABLE, NOT_EVALUATED, PASS, WARN, VerificationResult

__all__ = ["run_gates", "GATES", "DEFAULT_ACCEPTANCE"]

#: Default thresholds. They are the defaults of the analysis modules that measure these quantities, not universal
#: tolerances: ``verification.acceptance`` overrides them per use case.
DEFAULT_ACCEPTANCE = {
    "relative_l2": 0.05,          # field surrogate, against an independent reference solution
    "nrmse": 0.10,                # parametric surrogate, held-out normalised RMSE (surrogate.DEFAULT_TARGETS)
    "residual": 0.05,             # relative PDE residual of the trained model (pinneapple check PC302 default)
    "bc": 1e-2,                   # max boundary/initial misfit relative to the field scale (PC201/PC202)
    "refinement": 0.05,           # QoI change between a grid and its refinement
    "export_equivalence": 1e-4,   # max |exported - in-memory| / field scale
}


def _accept(cfg, name: str) -> float:
    return cfg.verification.threshold(name, DEFAULT_ACCEPTANCE[name])


def _finite(*values) -> bool:
    return all(v is not None and np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


# ----------------------------------------------------------------------------------------------- second opinion
def check_evidence(r: VerificationResult, state, prefixes: tuple[str, ...]) -> bool:
    """Fold ``pinneapple check``'s outcomes of these families into a gate.

    The checker is an independent implementation that loads the trained model through a plain array function, so
    what it says is not downstream of anything this pipeline computed. Its errors become the gate's ``FAIL``, its
    warnings a ``WARN``, and its passes an evidence line naming the rules. Returns whether it had anything to say,
    so the caller keeps its own measurement — and the report says which one it is quoting — when it did not.
    """
    from . import checks as _checks

    check = (state.metrics or {}).get("pinneapple_check")
    rows = _checks.outcomes_for(check, prefixes)
    if not rows:
        return False
    failed = [o for o in rows if not o["passed"]]
    r.metrics["check_outcomes"] = len(rows)
    if not failed:
        rules = ", ".join(sorted({str(o["rule"]) for o in rows}))
        r.note(f"pinneapple check agrees on {len(rows)} outcome(s) ({rules}): "
               + "; ".join(str(o["message"]) for o in rows[:2]))
        return True
    errors = [o for o in failed if o["severity"] == "error"]
    worst = (errors or failed)[0]
    if errors:
        r.fail(f"pinneapple check {worst['rule']} ({worst['subject']}): {worst['message']}",
               recommendation="the checker's own report is in the run folder: fix what it names before using the "
                              "model")
    else:
        r.warn(f"pinneapple check {worst['rule']} ({worst['subject']}): {worst['message']}",
               recommendation="a warning, not a blocker: the model is usable within the limitation stated")
    return True


# ----------------------------------------------------------------------------------------------------- problem
def gate_problem(state) -> VerificationResult:
    cfg, bundle = state.cfg, state.bundle
    r = VerificationResult("problem", PASS)
    out = {"coords": None, "fields": None, "pde": None, "conditions": None}
    issues: list[str] = []
    if bundle.backend == "solve":
        issues += list(bundle.problem.validate())
        out["pde"] = getattr(bundle.problem.pde, "kind", None)
    else:
        issues += list(bundle.system.issues())
        out["pde"] = "symbolic"
    issues += cfg.dimension.check_against(bundle.coords)
    out["coords"], out["fields"] = list(bundle.coords), list(bundle.fields)
    out["conditions"] = len(getattr(bundle.problem, "conditions", ()) or
                            getattr(bundle.system, "conditions", ()) or ())
    r.metrics.update(out)
    r.thresholds = {"conditions": "> 0 when the problem needs them"}
    for note in bundle.notes:
        r.note(note)
    if issues:
        for i in issues:
            r.fail(i, recommendation="fix the configuration or the problem before training: the other gates "
                                     "measure a run of the wrong problem")
    if cfg.data.is_physics_only and not out["conditions"]:
        r.warn("the problem declares no boundary or initial condition: a physics-only model has nothing to pin the "
               "solution, and any solution of the PDE will satisfy the loss",
               recommendation="add conditions in the config (boundary_conditions) or use a preset that declares them")
    if bundle.backend == "symbolic":
        r.note(f"symbolic system with {len(bundle.system.pdes)} PDE(s) and "
               f"{len(bundle.system.conditions)} condition(s)")
    return r


# ---------------------------------------------------------------------------------------------------- geometry
def gate_geometry(state) -> VerificationResult:
    cfg = state.cfg
    g = getattr(cfg, "geometry", None)
    if g is not None and g.source == "file":
        return _gate_geometry_file(state, g)
    geometry = getattr(state.bundle.problem, "geometry", None)
    if geometry is None:
        return VerificationResult("geometry", NOT_APPLICABLE,
                                  evidence=["the problem is a box (its domain_bounds); no geometry object is "
                                            "involved, so there is nothing to check here"])
    r = VerificationResult("geometry", NOT_EVALUATED)
    r.metrics["geometry_kind"] = getattr(state.bundle.problem, "geometry_kind", type(geometry).__name__)
    r.warn("the pipeline does not verify geometry integrity (manifoldness, watertightness, units) for this run",
           recommendation="declare the surface under geometry: {source: file, path: ...} and the gate checks it, "
                          "or run pinneapple_design.geometry checks on the geometry yourself")
    return r


def _gate_geometry_file(state, g) -> VerificationResult:
    """The surface of ``geometry.path``: is it closed, manifold, clean, and in the units the run assumes?"""
    from pathlib import Path

    r = VerificationResult("geometry", NOT_EVALUATED)
    path = Path(g.path)
    r.metrics["source"] = str(path)
    r.metrics["units"] = g.units
    if not path.is_file():
        r.limitations.append(f"geometry.path {path} does not exist")
        return r
    try:
        from pinneapple_data.cae.io import read_stl
        from pinneapple_data.cae.quality import surface_checks
        mesh = read_stl(path.read_bytes())
        quality = surface_checks(mesh)
    except Exception as exc:
        r.limitations.append(f"the surface could not be read or checked: {type(exc).__name__}: {exc}")
        return r
    if not quality:
        r.limitations.append(f"{path.name} has no surface (dim == 2) blocks to check")
        return r
    points = np.asarray(mesh.points, dtype=float) * float(g.length_scale)
    lo, hi = points.min(axis=0), points.max(axis=0)
    r.metrics.update({k: v for k, v in quality.items() if k != "shell_of_face"})
    r.metrics["bounding_box_m"] = [round(float(v), 6) for v in (hi - lo)]
    r.metrics["bbox_min_m"] = [round(float(v), 6) for v in lo]
    r.note(f"{path.name}: {quality['faces']} faces, bounding box "
           f"{' x '.join(f'{v:.4g}' for v in (hi - lo))} m, area {quality['area'] * g.length_scale ** 2:.4g} m2")
    if quality.get("open_edges") or quality.get("nonmanifold_edges"):
        r.fail(f"the surface is not closed: {quality['open_edges']} open edge(s) and "
               f"{quality['nonmanifold_edges']} non-manifold edge(s)",
               recommendation="a surface that does not bound a volume cannot be meshed or solved: close it in the "
                              "CAD tool (or run pinneapple_design.geometry's repair) and re-run")
        return r
    if not quality.get("watertight"):
        r.fail("the surface reports itself as not watertight", recommendation="check the surface's shell components")
        return r
    volume = quality.get("enclosed_volume")
    if volume is not None and np.isfinite(volume) and volume <= 0:
        r.fail(f"the enclosed volume is {volume * g.length_scale ** 3:.4g} m3: the faces are oriented inwards",
               recommendation="flip the surface normals: a negative volume means the inside is outside")
        return r
    r.status = PASS
    r.note(f"watertight, manifold, {quality.get('shells', 1)} shell(s), enclosed volume "
           f"{(volume or 0.0) * g.length_scale ** 3:.4g} m3")
    if quality.get("inconsistent_orientation_edges"):
        r.warn(f"{quality['inconsistent_orientation_edges']} edge(s) are shared by two faces wound the same way",
               recommendation="the volume is still positive, but a mesher will re-orient the faces: fix the winding")
    if quality.get("degenerate_faces") or quality.get("duplicate_faces"):
        r.warn(f"{quality.get('degenerate_faces', 0)} degenerate and {quality.get('duplicate_faces', 0)} duplicate "
               "face(s): they are removed by the mesher, and they make the area and volume estimates optimistic")
    for region in g.regions:
        r.note(f"region {region!r} declared; this pipeline checks the whole surface, not its patches")
    return r


# ----------------------------------------------------------------------------------------------- discretization
def gate_discretization(state) -> VerificationResult:
    """Does the reference solution move when its own grid is refined? A real refinement study, on one case."""
    cfg, bundle = state.cfg, state.bundle
    if bundle.backend != "solve" or not getattr(bundle.problem, "reference_solver", None):
        return VerificationResult("discretization", NOT_APPLICABLE,
                                  evidence=["no reference solver: the model is trained on the equations, not on "
                                            "another discretisation, so there is no grid to refine"])
    if cfg.solver.method != "reference":
        return VerificationResult("discretization", NOT_APPLICABLE,
                                  evidence=[f"the reference is '{cfg.solver.method}', which has no grid of its own to "
                                            "refine: the discretisation is the solver's, not the model's"])
    params = dict(bundle.problem.reference_solver.get("params") or {})
    grid_keys = [k for k, v in params.items() if isinstance(v, int) and v >= 8]
    if not grid_keys:
        return VerificationResult("discretization", NOT_EVALUATED,
                                  metrics={"solver_params": params},
                                  limitations=["the reference solver declares no grid size the pipeline can refine"])
    key = grid_keys[0]
    case = state.cases.iloc[0].to_dict() if state.cases is not None and len(state.cases) else {}
    r = VerificationResult("discretization", NOT_EVALUATED,
                          thresholds={"refinement": _accept(cfg, "refinement")})
    try:
        import pinneapple as pp

        problem = bundle.problem
        if case:
            problem = problem.with_parameters(**{k: float(v) for k, v in case.items()
                                                 if k in problem.parameters})
        pts = state.points if state.points is not None else _sample_points(bundle, 8, seed=int(cfg.sampling.seed))
        coarse_solution = pp.solve(problem, method=cfg.solver.method, **dict(cfg.solver.options))
        coarse = np.asarray(coarse_solution.predict(pts))
        options = dict(cfg.solver.options)
        # the solver takes its grid inside `params`; an override at the top level of solver_config is merged into
        # the solver spec but never reaches the solver, which would make this study compare a run with itself
        options["solver_config"] = {"params": {**params, key: 2 * int(params[key])}}
        fine_solution = pp.solve(problem, method=cfg.solver.method, **options)
        fine = np.asarray(fine_solution.predict(pts))
    except Exception as exc:                                     # the solver may not accept an override
        r.limitations.append(f"{type(exc).__name__}: {exc}")
        return r
    coarse_grid, fine_grid = getattr(coarse_solution, "grid", None), getattr(fine_solution, "grid", None)
    if coarse_grid and fine_grid and list(coarse_grid) == list(fine_grid):
        if all(len(coarse_grid[c]) == len(fine_grid[c]) for c in coarse_grid):
            r.limitations.append(f"the reference solver ignored the {key} override (the grid is unchanged): the "
                                 "refinement was not measured")
            return r
    scale = float(np.sqrt(np.mean(fine ** 2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        r.limitations.append("the reference solution is identically zero at these points: a refinement study of a "
                            "trivial field measures nothing (check the case's source and boundary conditions)")
        return r
    change = float(np.max(np.abs(fine - coarse)) / scale)
    r.metrics.update({"grid_key": key, "coarse": int(params[key]), "fine": 2 * int(params[key]),
                      "qoi_scale": scale, "relative_change": change})
    r.status = PASS if change <= r.thresholds["refinement"] else WARN
    r.note(f"refining {key} from {params[key]} to {2 * int(params[key])} moves the solution by {change:.3g} "
           "relative to its own scale")
    if r.status != PASS:
        r.recommendations.append(f"the reference grid is not converged for this quantity: refine {key} further "
                                 "before trusting the data the surrogate is trained on")
    return r


# ------------------------------------------------------------------------------------------------- simulation
def gate_simulation(state) -> VerificationResult:
    if not (state.reference or {}):
        recorded = (state.metrics or {}).get("simulation")
        if recorded:
            return _recorded_simulation(state, recorded)
    reference = state.reference or {}
    if not reference:
        if state.cfg.data.is_physics_only:
            return VerificationResult("simulation", NOT_APPLICABLE,
                                      evidence=["physics-only run: no reference solve produced the data"])
        return VerificationResult("simulation", NOT_EVALUATED,
                                  limitations=["no reference solution was produced"])
    ok = [k for k, v in reference.items() if not isinstance(v, str)]
    bad = {k: v for k, v in reference.items() if isinstance(v, str)}
    r = VerificationResult("simulation", PASS)
    r.metrics.update({"cases": len(reference), "solved": len(ok), "failed": len(bad),
                      "solver": getattr(state.bundle.problem, "reference_solver", {}).get("name", "")
                      if state.bundle.problem is not None else "",
                      "method": state.cfg.solver.method})
    times = [getattr(solution, "wall_time_s", None) for solution in reference.values() if not isinstance(solution, str)]
    times = [float(t) for t in times if t is not None]
    if times:
        r.metrics["wall_time_total_s"] = round(sum(times), 3)
    if bad:
        r.status = FAIL
        for k, v in list(bad.items())[:5]:
            r.limitations.append(f"case {k}: {v}")
        r.recommendations.append("the dataset is missing cases the model will be asked about: fix the failing "
                                 "solves (or their parameter values) and re-run the data stage")
    r.note(f"{len(ok)} of {len(reference)} cases solved with method '{state.cfg.solver.method}'")
    return r


def _recorded_simulation(state, recorded: dict) -> VerificationResult:
    """The simulation gate from the campaign summary a previous run wrote (``resume`` keeps the evidence)."""
    r = VerificationResult("simulation", PASS)
    failed = int(recorded.get("failed") or 0)
    r.metrics.update({k: recorded.get(k) for k in ("cases", "solved", "failed", "solver", "method",
                                                  "wall_time_total_s")})
    r.note("read from the campaign summary of the data stage (the solutions themselves are not kept in memory "
           "across runs)")
    if failed:
        r.status = FAIL
        for key, why in (recorded.get("failures") or {}).items():
            r.limitations.append(f"case {key}: {why}")
        r.recommendations.append("fix the failing solves and re-run the data stage")
    else:
        r.note(f"{recorded.get('solved')} of {recorded.get('cases')} cases solved with method "
               f"'{recorded.get('method')}'")
    return r


# ---------------------------------------------------------------------------------------------------- dataset
def gate_dataset(state) -> VerificationResult:
    cfg = state.cfg
    if cfg.data.is_physics_only:
        return VerificationResult("dataset", NOT_APPLICABLE,
                                  evidence=["physics-only run: the model is trained on the equations"])
    if state.dataset is None:
        return VerificationResult("dataset", NOT_EVALUATED, limitations=["no dataset was built"])
    data = state.dataset
    r = VerificationResult("dataset", PASS)
    r.thresholds = {"min_samples": 8, "duplicate_rows": 0}
    r.metrics.update({"samples": int(data.n), "inputs": len(data.input_names), "outputs": len(data.output_names),
                      "kind": data.kind})
    r.note(f"{data.n} samples of {data.output_names[:3]}{'...' if len(data.output_names) > 3 else ''} "
           f"from {data.input_names}")
    X = np.asarray(data.X, dtype=float)
    if len(np.unique(X, axis=0)) != len(X):
        r.status = FAIL
        r.limitations.append(f"{len(X) - len(np.unique(X, axis=0))} duplicate input row(s): the same case was run "
                             "twice")
        r.recommendations.append("check the design of experiments and the case parameter values")
    if data.n < 8:
        r.status = FAIL
        r.limitations.append(f"{data.n} samples is not enough to train, validate and test a model")
        r.recommendations.append("raise sampling.design_sampling.n_samples")
    # coverage of the design in the unit hypercube, when the space is known
    try:
        from pinneapple_data.doe import coverage
        space = cfg.parameters.to_space()
        if space is not None and space.dim:
            unit = space.to_unit({name: X[:, data.input_names.index(name)] for name in space.names
                                  if name in data.input_names})
            cov = coverage(unit)
            r.metrics.update({k: v for k, v in cov.items() if isinstance(v, (int, float)) and k != "seed"})
            r.note("design coverage: " + ", ".join(f"{k}={v:.3g}" for k, v in cov.items()
                                                   if isinstance(v, (int, float))))
    except Exception as exc:                                     # coverage is evidence, not a requirement
        r.limitations.append(f"design coverage not measured: {type(exc).__name__}: {exc}")
    # splits: no leakage between train/validation/test
    study = state.model if hasattr(state.model, "splits") else None
    recorded = (state.metrics or {}).get("splits")
    if study is not None or recorded:
        if study is not None:
            sp = {"train": list(study.splits.train), "val": list(study.splits.val),
                  "test": list(study.splits.test), "shift": list(study.splits.shift),
                  "description": study.splits.shift_description}
        else:
            sp = dict(recorded)
        r.metrics.update({"train": len(sp["train"]), "validation": len(sp["val"]),
                          "test": len(sp["test"]), "shift": len(sp["shift"])})
        sets = {k: set(map(int, sp[k])) for k in ("train", "val", "test", "shift")}
        overlap = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(sets) for b in list(sets)[i + 1:]}
        leaked = {k: v for k, v in overlap.items() if v}
        r.metrics["split_overlap"] = overlap
        if leaked:
            r.status = FAIL
            r.limitations.append(f"splits overlap: {leaked}")
            r.recommendations.append("fix the split seed/strategy; a model scored on rows it trained on is not "
                                     "measured")
        else:
            r.note("train/validation/test/shift are disjoint")
        if sets["shift"]:
            r.note(f"a distribution-shift set of {len(sets['shift'])} cases is held out "
                   f"({sp.get('description') or 'corner of the space'})")
    else:
        if r.status == PASS:
            r.status = WARN
        r.limitations.append("no split table: leakage between training and test rows was not checked")
    return r


# ------------------------------------------------------------------------------------------------------ model
def gate_model(state) -> VerificationResult:
    cfg = state.cfg
    study = state.model if hasattr(state.model, "ranking") else None
    r = VerificationResult("model", NOT_EVALUATED)
    if study is not None or (state.metrics or {}).get("ranking"):
        target = _accept(cfg, "nrmse")
        r.thresholds = {"nrmse": target}
        ranking = study.ranking("test") if study is not None else list(state.metrics["ranking"])
        if ranking:
            r.metrics.update({"nrmse_test": ranking[0]["error"], "spread": ranking[0].get("spread"),
                              "method": ranking[0]["method"], "candidates": len(ranking)})
            r.status = PASS if ranking[0]["error"] <= target else FAIL
            for row in ranking:
                r.note(f"{row['method']}: held-out nrmse {row['error']:.3g}"
                       + (f" (+- {row['spread']:.3g})" if row.get("spread") else ""))
            if r.status == FAIL:
                r.recommendations.append(f"the best candidate misses the {target:.3g} nrmse criterion: raise "
                                         "training.budget_seconds, add samples, or relax the criterion with a stated "
                                         "reason (verification.acceptance.nrmse)")
        else:
            r.limitations.append("no candidate produced a held-out score")
        return r
    metrics = state.metrics or {}
    report = metrics.get("pinn_report")
    if report is not None:                                   # symbolic backend
        checks = {c["id"]: c for c in report.checks}
        exact = checks.get("pinn.exact")
        r.metrics["checks"] = {k: v.get("value") for k, v in checks.items()
                               if k in ("pinn.exact", "pinn.spectrum")}
        if exact is None or exact.get("status") == "not-applicable":
            r.limitations.append("no exact solution is declared for this problem, so the model has nothing to be "
                                 "measured against")
            r.recommendations.append("declare the solution the problem should have "
                                     "(problem.exact_expression in a symbolic problem) and the error against it "
                                     "becomes this gate's evidence")
            return r
        target = _accept(cfg, "relative_l2")
        value = float(exact.get("value") or float("nan"))
        r.thresholds = {"relative_l2": target}
        r.metrics["relative_l2"] = value
        r.status = PASS if np.isfinite(value) and value <= target else FAIL
        r.note(f"relative error against the exact solution: {value:.3g} (validate_pinn's pinn.exact)")
        if r.status == FAIL:
            r.recommendations.append(f"the model misses the {target:.3g} criterion against the exact solution: train "
                                     "longer (training.epochs/lbfgs_steps) or enlarge the network")
        return r
    if "relative_l2" in metrics:
        target = _accept(cfg, "relative_l2")
        value = float(metrics["relative_l2"])
        r.thresholds = {"relative_l2": target}
        r.metrics.update({"relative_l2": value, "rmse": metrics.get("rmse"), "max_abs": metrics.get("max_abs"),
                          "reference": metrics.get("reference"), "points": metrics.get("n_points")})
        if not np.isfinite(value):
            r.status = NOT_EVALUATED
            r.limitations.append("the reference solution is identically zero at the comparison points, so the error "
                                 "cannot be made relative (RMSE and max|error| are reported, but their scale is the "
                                 "reference's own zero)")
            r.recommendations.append("choose a case with a non-trivial solution: a reference that is zero everywhere "
                                     "does not measure the model")
            return r
        r.status = PASS if value <= target else FAIL
        r.note(f"relative L2 {value:.3g} against '{metrics.get('reference')}' at {metrics.get('n_points')} points")
        if r.status == FAIL:
            r.recommendations.append(f"the model misses the {target:.3g} relative-L2 criterion: train longer "
                                     "(training.epochs), enlarge the network, or state a different criterion with a "
                                     "reason (verification.acceptance.relative_l2)")
    else:
        r.limitations.append("no independent reference: the model's error was not measured")
        r.recommendations.append("give the solver a reference (solver.method: analytic or reference) so the model "
                                 "can be scored against something it did not learn")
    return r


# ---------------------------------------------------------------------------------------------------- physics
def gate_physics(state) -> VerificationResult:
    """Residual and boundary/initial conditions of the trained model, on points it did not train on."""
    cfg, bundle = state.cfg, state.bundle
    if not cfg.data.is_physics_only:
        return VerificationResult(
            "physics", NOT_APPLICABLE,
            evidence=["the surrogate was fitted to reference solutions, not to the equations: the physics evidence "
                      "for this run is the reference solver's (gate 'simulation') and the domain the model was "
                      "measured on (gate 'applicability'). A PDE residual of a tabular surrogate would measure the "
                      "residual of the interpolation, not of the physics."])
    if bundle.backend == "symbolic":
        report = state.metrics.get("pinn_report")
        if report is None:
            return VerificationResult("physics", NOT_EVALUATED, limitations=["validate_pinn did not run"])
        r = VerificationResult("physics", PASS)
        for check in report.checks:
            status = check["status"]
            r.metrics[check["id"]] = check.get("value")
            if status == "fail":
                r.status = FAIL
            elif status in ("marginal", "warn") and r.status == PASS:
                r.status = WARN
            if check["id"] in ("pinn.residual", "pinn.boundary", "pinn.initial", "pinn.trivial"):
                r.thresholds[check["id"]] = check.get("threshold")
            r.note(f"[{status}] {check['id']}: {check['summary']}")
        if report.failure_modes:
            r.limitations.append("failing checks: " + "; ".join(report.failure_modes))
            r.recommendations.append("see the failing check's summary: it names the term (boundary, initial, residual "
                                     "or a trivial solution) that has to be fixed first")
        check_evidence(r, state, ("PC1", "PC2", "PC3"))
        return r

    physics = state.metrics.get("physics") or {}
    if not physics:
        return VerificationResult("physics", NOT_EVALUATED,
                                  limitations=["the physics loss could not be evaluated on the trained model"])
    r = VerificationResult("physics", PASS)
    bc_target = _accept(cfg, "bc")
    r.thresholds = {"bc": bc_target}
    bc = physics.get("bc_misfit")
    residual = physics.get("residual_rms")
    r.metrics.update({"bc_misfit": bc, "residual_rms": residual,
                      "residual_relative": physics.get("residual_relative")})
    for note in physics.get("notes", []):
        r.note(note)
    if bc is None:
        r.status = NOT_EVALUATED
        r.limitations.append("the problem has no boundary/initial condition the pipeline could sample")
    elif not math.isfinite(bc):
        r.status = FAIL
        r.limitations.append("the boundary misfit is not finite")
    elif bc > bc_target:
        r.status = FAIL
        r.limitations.append(f"boundary/initial misfit {bc:.3g} > {bc_target:.3g}")
        r.recommendations.append("raise the boundary weight of the loss (loss.weights.w_bc/w_ic) or train longer")
    else:
        r.note(f"boundary/initial misfit {bc:.3g} <= {bc_target:.3g}")
    rel = physics.get("residual_relative")
    if rel is not None and math.isfinite(rel):
        target = _accept(cfg, "residual")
        r.thresholds["residual"] = target
        if rel > target and r.status == PASS:
            r.status = WARN
            r.limitations.append(f"relative PDE residual {rel:.3g} > {target:.3g}: the model fits the conditions "
                                 "better than it satisfies the equation")
        else:
            r.note(f"relative PDE residual {rel:.3g} <= {target:.3g}")
    elif residual is not None:
        r.note(f"absolute PDE residual RMS {residual:.3g} (no scale to normalise it: reported, not gated)")
        r.limitations.append("the residual has no non-dimensional reference for this problem, so it was not "
                             "compared against the acceptance threshold")
    check_evidence(r, state, ("PC1", "PC2", "PC3"))
    return r


# ---------------------------------------------------------------------------------------------- applicability
def gate_applicability(state) -> VerificationResult:
    cfg = state.cfg
    domain = state.validated_domain or {}
    r = VerificationResult("applicability", NOT_EVALUATED)
    if cfg.data.is_physics_only:
        r.status = NOT_APPLICABLE
        r.evidence.append("a physics-only model is not restricted to a parameter space: it is a solution of one "
                          "problem, and its validity is the physics gate's business")
        check_evidence(r, state, ("PC6",))
        return r
    if not domain:
        r.limitations.append("no validated domain was built")
        check_evidence(r, state, ("PC6",))
        return r
    r.status = PASS
    r.metrics.update({f"{k}": v for k, v in domain.items() if isinstance(v, (int, float, str))})
    bounds = domain.get("bounds") or {}
    for name, (lo, hi) in bounds.items():
        r.note(f"valid for {name} in [{lo:.4g}, {hi:.4g}]")
    if domain.get("extrapolation_risk"):
        r.status = WARN
        r.limitations.append(domain["extrapolation_risk"])
        r.recommendations.append("the surrogate is interpolating over a space whose corners were never solved: add "
                                 "cases at the boundary of the parameter space")
    if domain.get("ood_rule"):
        r.note(f"out-of-distribution rule: {domain['ood_rule']}")
    check_evidence(r, state, ("PC6",))
    return r


# ----------------------------------------------------------------------------------------------------- export
def gate_export(state) -> VerificationResult:
    cfg = state.cfg
    if not cfg.export.enabled:
        return VerificationResult("export", NOT_APPLICABLE, evidence=["export is disabled in the configuration"])
    info = state.metrics.get("export") or {}
    written = info.get("written") or {}
    errors = {k: v for k, v in written.items() if k.endswith("_error")}
    r = VerificationResult("export", NOT_EVALUATED)
    r.metrics["formats"] = [k for k in written if not k.endswith("_error")]
    if not written:
        r.limitations.append(info.get("error") or "nothing was exported")
        return r
    if errors:
        r.status = WARN
        for k, v in errors.items():
            r.limitations.append(f"{k.removesuffix('_error')}: {v}")
    if "equivalence" in info:
        eq = info["equivalence"]
        target = cfg.export.tolerance
        r.thresholds = {"equivalence": target}
        r.metrics["equivalence"] = eq.get("max_relative_difference")
        r.metrics["compared_format"] = eq.get("format")
        if not eq.get("max_relative_difference") == eq.get("max_relative_difference"):
            r.status = WARN
            r.limitations.append("the exported model returned non-finite values")
        elif eq["max_relative_difference"] <= target:
            r.status = PASS
            r.note(f"{eq['format']}: exported predictions agree with the in-memory model to "
                   f"{eq['max_relative_difference']:.3g} (tolerance {target:.3g})")
        else:
            r.status = FAIL
            r.limitations.append(f"{eq['format']}: exported predictions differ by "
                                 f"{eq['max_relative_difference']:.3g} > {target:.3g}")
            r.recommendations.append("do not ship the exported file: the conversion changed the model "
                                     "(the export configuration lists what can be shipped)")
    elif r.status == NOT_EVALUATED:
        r.status = WARN
        r.limitations.append("export verification is off: the exported file was not compared against the model")
    check_evidence(r, state, ("PC8",))
    return r


# -------------------------------------------------------------------------------------------------- inference
def gate_inference(state) -> VerificationResult:
    """One fresh query at the centre of the space: finite outputs, inside the validated domain."""
    r = VerificationResult("inference", NOT_EVALUATED)
    if state.model_object is None:
        r.limitations.append("no trained model is available to query")
        return r
    query = state.metrics.get("inference_query")
    y = state.metrics.get("inference_prediction")
    if query is None or y is None:
        r.limitations.append("no inference was run after training")
        return r
    y = np.asarray(y, dtype=float)
    r.metrics.update({"query": {k: float(v) for k, v in query.items()}, "n_outputs": int(y.size)})
    if not np.all(np.isfinite(y)):
        r.status = FAIL
        r.limitations.append("the prediction contains non-finite values")
        r.recommendations.append("inspect the training: a model that returns NaN on a point inside its own space "
                                 "cannot be deployed")
        return r
    r.status = PASS
    r.note("prediction at the centre of the parameter space is finite: "
           + ", ".join(f"{v:.4g}" for v in y.ravel()[:4]) + ("..." if y.size > 4 else ""))
    ood = state.metrics.get("inference_ood")
    if ood:
        r.status = WARN
        r.limitations.append(f"the query is outside the distribution the model saw ({ood})")
        r.recommendations.append("answer with the reference solver for this query, or extend the parameter space")
    return r


GATES = {
    "problem": gate_problem,
    "geometry": gate_geometry,
    "discretization": gate_discretization,
    "simulation": gate_simulation,
    "dataset": gate_dataset,
    "model": gate_model,
    "physics": gate_physics,
    "applicability": gate_applicability,
    "export": gate_export,
    "inference": gate_inference,
}


def run_gates(state, names: Any = None) -> list[VerificationResult]:
    """Run the gates (all by default, or the ones the configuration enables) over ``state``."""
    cfg = state.cfg
    if not cfg.verification.enabled:
        return []
    wanted = names or [g for g in GATES if cfg.verification.wants(g)]
    out = []
    for name in wanted:
        gate = GATES[name]
        try:
            result = gate(state)
        except Exception as exc:                                 # a gate must never take the run down with it
            result = VerificationResult(name, NOT_EVALUATED,
                                        limitations=[f"the gate raised {type(exc).__name__}: {exc}"])
        out.append(result)
    return out


def _sample_points(bundle, n: int, *, seed: int = 0) -> np.ndarray:
    """``n`` points spread over the box (for a refinement study with no probe points).

    Not the centre: on a problem whose solution is zero on an axis (Burgers at x = 0) a centred sample makes the
    relative change of the refinement study a 0/0.
    """
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(*bundle.domains[c], size=n) for c in bundle.coords], axis=1)
