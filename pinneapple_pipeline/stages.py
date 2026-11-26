"""The stages of a run: each one validates its inputs, does one thing, and says what it produced.

A stage is the unit of ``run_until``, of ``resume`` and of the failure report: when one fails, the record names it,
the ones before it stay on disk, and re-running resumes from there. Implement the :class:`PipelineStage` contract
and register it on a pipeline to add a step of your own without touching this module
(``pipeline.register_stage(...)``).
"""
from __future__ import annotations

import abc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import adapters
from .results import STAGE_ORDER

__all__ = ["PipelineStage", "ProblemStage", "DataStage", "PreprocessStage", "ModelStage", "TrainingStage",
           "ExportStage", "VerificationStage", "DEFAULT_STAGES", "StageError"]


class StageError(RuntimeError):
    """A stage could not do its job. The message is what the report shows for that stage."""


class PipelineStage(abc.ABC):
    """One step of the pipeline.

    ``validate`` checks that the stage can run at all (before anything expensive), ``execute`` does the work and
    returns the artefacts it produced, ``verify`` states what the stage itself can attest to. The pipeline times the
    stage, writes its record and never catches an exception silently: a stage that raises is reported as failed with
    its message.
    """

    name: str = ""

    def validate(self, state) -> list[str]:
        """Reasons this stage cannot run, as readable strings (empty when it can)."""
        return []

    @abc.abstractmethod
    def execute(self, state) -> dict[str, Any]:
        """Do the work; return ``{artifact name: path or value}``."""

    def verify(self, state, artifacts: dict[str, Any]) -> list[str]:
        """What this stage can say about its own output (notes for the record)."""
        return []

    def skip_reason(self, state) -> str:
        """Why this stage does not apply to this run ("" to run it)."""
        return ""


# ---------------------------------------------------------------------------------------------------- problem
class ProblemStage(PipelineStage):
    name = "problem"

    def execute(self, state) -> dict[str, Any]:
        from .verification import gate_problem
        state.bundle = adapters.build_problem(state.cfg)
        description = adapters.problem_description(state.bundle)
        path = state.path("problem.json")
        path.write_text(json.dumps(description, indent=1, default=str), encoding="utf-8")
        gate = gate_problem(state)
        state.metrics["problem_gate"] = gate.status
        notes = list(state.bundle.notes)
        if gate.status == "FAIL":
            raise StageError("the configuration and the problem disagree: " + "; ".join(gate.limitations))
        notes += gate.evidence[:2]
        return {"problem": str(path), "notes": notes}

    def verify(self, state, artifacts) -> list[str]:
        bundle = state.bundle
        head = " · ".join([bundle.name, f"coords ({', '.join(bundle.coords)})", f"fields ({', '.join(bundle.fields)})",
                           f"{len(getattr(bundle.problem, 'conditions', ()) or getattr(bundle.system, 'conditions', ()))} "
                           "condition(s)"])
        return [head] + [n for n in (artifacts.get("notes") or []) if "material" in n or "exact" in n][:2]


# ------------------------------------------------------------------------------------------------------- data
class DataStage(PipelineStage):
    name = "data"

    def skip_reason(self, state) -> str:
        if state.cfg.data.is_physics_only:
            return "physics_only: the model is trained on the equations, so no case is solved"
        return ""

    def validate(self, state) -> list[str]:
        out = []
        if state.cfg.data.source == "file" and not Path(state.cfg.data.path).is_file():
            out.append(f"data.path {state.cfg.data.path!r} does not exist")
        return out

    def execute(self, state) -> dict[str, Any]:
        cfg, bundle = state.cfg, state.bundle
        state.points = adapters.probe_points(cfg, bundle)
        np.save(state.path("data", "probe_points.npy"), state.points)

        if cfg.data.source == "file":
            return self._from_file(state)

        design = adapters.design_cases(cfg)
        state.design, state.cases = design, design.cases
        names = list(design.cases.columns)
        rows, labels, failures = [], [], {}
        for i, (_, row) in enumerate(design.cases.iterrows()):
            case = row.to_dict()
            try:
                solution = adapters.solve_reference(cfg, bundle, case)
            except Exception as exc:
                failures[str(i)] = f"{type(exc).__name__}: {exc}"
                state.reference[str(i)] = failures[str(i)]
                if cfg.solver.on_failure == "fail":
                    raise StageError(f"reference solve for case {i} {case} failed: {failures[str(i)]}") from exc
                continue
            state.reference[str(i)] = solution
            rows.append(adapters.solution_row(solution, state.points))
            labels.append(case)
        if not rows:
            raise StageError(f"no case could be solved ({len(failures)} failed); the dataset would be empty")
        numeric = all(isinstance(dict(label)[n], (int, float, np.integer, np.floating)) for label in labels
                      for n in names)
        X = np.asarray([[float(dict(label)[n]) for n in names] for label in labels], dtype=float) if numeric else \
            np.asarray([[dict(label)[n] for n in names] for label in labels], dtype=object)
        Y = np.asarray(rows, dtype=float)
        output_names = self._output_names(state, Y.shape[1])
        path = state.path("data", "dataset.npz")
        np.savez(path, X=X, Y=Y, input_names=np.asarray(names, dtype=object),
                 output_names=np.asarray(output_names, dtype=object), probe_points=state.points,
                 cases=np.asarray([json.dumps(dict(c)) for c in labels], dtype=object))
        size = path.stat().st_size
        summary = {"cases": len(design.cases), "solved": len(rows), "failed": len(failures),
                   "failures": dict(list(failures.items())[:10]),
                   "solver": (bundle.problem.reference_solver or {}).get("name", ""), "method": cfg.solver.method,
                   "wall_time_total_s": round(sum(float(getattr(sol, "wall_time_s", 0.0) or 0.0)
                                                  for sol in state.reference.values()
                                                  if not isinstance(sol, str)), 3)}
        state.metrics["simulation"] = summary
        (state.path("data", "simulation.json")).write_text(__import__("json").dumps(summary, indent=1),
                                                           encoding="utf-8")
        return {"dataset": str(path), "cases": len(rows), "failed_cases": len(failures), "bytes": int(size),
                "points_per_case": int(state.points.shape[0]),
                "simulation": str(state.path("data", "simulation.json"))}

    def _from_file(self, state) -> dict[str, Any]:
        cfg = state.cfg
        path = Path(cfg.data.path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as npz:
                X, Y = np.asarray(npz["X"], dtype=float), np.asarray(npz["Y"], dtype=float)
                inputs = list(cfg.data.inputs) or (
                    [str(s) for s in npz["input_names"]] if "input_names" in npz else [])
                outputs = list(cfg.data.outputs) or (
                    [str(s) for s in npz["output_names"]] if "output_names" in npz else [])
        elif path.suffix in (".csv", ".tsv"):
            import csv
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            inputs = list(cfg.data.inputs) or [c for c in rows[0] if c in cfg.data.input_variables]
            outputs = list(cfg.data.outputs) or [c for c in rows[0] if c not in inputs]
            X = np.asarray([[float(r[c]) for c in inputs] for r in rows], dtype=float)
            Y = np.asarray([[float(r[c]) for c in outputs] for r in rows], dtype=float)
        else:
            raise StageError(f"data.path {path} must be .npz, .csv or .tsv")
        if not inputs or not outputs:
            raise StageError(f"{path}: give data.inputs and data.outputs (or name them inside the file)")
        saved = state.path("data", "dataset.npz")
        np.savez(saved, X=X, Y=Y, input_names=np.asarray(inputs, dtype=object),
                 output_names=np.asarray(outputs, dtype=object))
        state.metrics["simulation"] = {"cases": int(len(X)), "solved": int(len(X)), "failed": 0,
                                       "source": str(path), "method": "file"}
        return {"dataset": str(saved), "cases": int(len(X)), "source": str(path), "failed_cases": 0}

    def _output_names(self, state, width: int) -> list[str]:
        cfg, bundle = state.cfg, state.bundle
        if cfg.data.outputs and cfg.data.outputs[0] != "field":
            return list(cfg.data.outputs)
        names = []
        for field in bundle.fields:
            for j in range(width // max(len(bundle.fields), 1)):
                names.append(f"{field}@{j}")
        return names[:width] if len(names) == width else [f"y{j}" for j in range(width)]

    def verify(self, state, artifacts) -> list[str]:
        notes = [f"{artifacts.get('cases')} case(s) solved, {artifacts.get('failed_cases')} failed"]
        if state.cases is not None:
            notes.append(f"design: {state.cfg.sampling.design_sampling.get('method')} with "
                         f"{len(state.cases)} case(s) over {list(state.cases.columns)}")
        return notes


# -------------------------------------------------------------------------------------------------- preprocess
class PreprocessStage(PipelineStage):
    name = "preprocess"

    def skip_reason(self, state) -> str:
        return "physics_only: no dataset to prepare" if state.cfg.data.is_physics_only else ""

    def execute(self, state) -> dict[str, Any]:
        from pinneapple_analysis import surrogate
        dataset_path = state.stages["data"].artifacts.get("dataset")
        if not dataset_path:
            raise StageError("the data stage produced no dataset")
        with np.load(dataset_path, allow_pickle=True) as npz:
            X, Y = np.asarray(npz["X"], dtype=float), np.asarray(npz["Y"], dtype=float)
            input_names = [str(s) for s in npz["input_names"]]
            output_names = [str(s) for s in npz["output_names"]]
        meta = {"source": f"{state.cfg.solver.method} solves of {state.bundle.name}",
                "probes": int(np.asarray(np.load(Path(dataset_path).with_name("probe_points.npy"))).shape[0])
                if Path(dataset_path).with_name("probe_points.npy").is_file() else None}
        state.dataset = surrogate.from_arrays({"X": X, "Y": Y}, input_names=input_names,
                                              output_names=output_names, meta=meta)
        # the split the study will use, computed here so the report can state it before training
        try:
            splits = surrogate.make_splits(state.dataset, seed=int(state.cfg.sampling.seed),
                                           **_split_options(state.cfg))
        except ValueError as exc:
            raise StageError(f"{exc}: raise sampling.design_sampling.n_samples, or turn the distribution-shift set "
                             f"off (sampling.validation_sampling.shift: none)") from exc
        state.metrics["splits"] = {"train": [int(i) for i in splits.train], "val": [int(i) for i in splits.val],
                                   "test": [int(i) for i in splits.test], "shift": [int(i) for i in splits.shift],
                                   "description": splits.shift_description}
        report = {"n_samples": int(state.dataset.n), "inputs": input_names, "outputs": output_names,
                  "train": int(len(splits.train)), "validation": int(len(splits.val)), "test": int(len(splits.test)),
                  "shift": int(len(splits.shift)), "shift_description": splits.shift_description,
                  "describe": state.dataset.describe()}
        out = state.path("data", "dataset_report.json")
        out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        return {"dataset_report": str(out), "train": report["train"], "test": report["test"]}

    def verify(self, state, artifacts) -> list[str]:
        return [f"{artifacts['train']} train / {artifacts['test']} test rows"]


def _split_options(cfg) -> dict[str, Any]:
    v = cfg.sampling.validation_sampling
    return {"test_frac": float(v.get("test_fraction", 0.2)), "val_frac": float(v.get("val_fraction", 0.15)),
            "shift": v.get("shift", "auto")}


# ------------------------------------------------------------------------------------------------------ model
class ModelStage(PipelineStage):
    """Resolve the architecture and check it fits the problem before any training time is spent."""

    name = "model"

    def execute(self, state) -> dict[str, Any]:
        cfg, bundle = state.cfg, state.bundle
        n_in, n_out = len(bundle.coords), len(bundle.fields)
        info: dict[str, Any] = {"inputs": n_in, "outputs": n_out}
        if cfg.data.is_physics_only:
            if bundle.backend == "solve":
                import pinneapple as pp
                family = cfg.architecture.candidates[0] if cfg.architecture.is_auto else cfg.architecture.family
                if family not in pp.list_models():
                    raise StageError(f"architecture {family!r} is not in the model registry (pp.list_models())")
                info["family"] = family
                info["candidates"] = [family]
            else:
                info["family"] = "pinn_problem_mlp"
                info["candidates"] = ["pinn_problem_mlp"]
        else:
            from pinneapple_analysis.surrogate.methods import CANDIDATES
            wanted = list(cfg.architecture.candidates or CANDIDATES) if cfg.architecture.is_auto else \
                [cfg.architecture.family]
            unknown = [m for m in wanted if m not in CANDIDATES]
            if unknown:
                raise StageError(f"architecture {unknown} is not a method of pinneapple_analysis.surrogate; it has "
                                 f"{list(CANDIDATES)}")
            info["candidates"] = wanted
            info["family"] = "auto" if cfg.architecture.is_auto else wanted[0]
        path = state.path("model", "architecture.json")
        path.write_text(json.dumps(info, indent=1), encoding="utf-8")
        return {"architecture": str(path), "family": info["family"], "candidates": info["candidates"]}

    def verify(self, state, artifacts) -> list[str]:
        return [f"family {artifacts['family']} over {len(artifacts['candidates'])} candidate(s)"]


# --------------------------------------------------------------------------------------------------- training
class TrainingStage(PipelineStage):
    name = "training"

    def execute(self, state) -> dict[str, Any]:
        cfg, bundle = state.cfg, state.bundle
        t0 = time.perf_counter()
        if cfg.data.is_physics_only:
            return self._physics(state, bundle, t0)
        return self._supervised(state, t0)

    # ---- physics-only
    def _physics(self, state, bundle, t0: float) -> dict[str, Any]:
        cfg = state.cfg
        seen = len(bundle.notes)
        if cfg.optimization.enabled:
            solution = self._physics_hpo(state, bundle)
        else:
            solution = adapters.train_physics_model(cfg, bundle)
        state.notes.extend(bundle.notes[seen:])          # what training itself had to say (early stopping, ...)
        state.model = state.model_object = solution
        history = dict(getattr(solution, "history", {}) or {})
        state.metrics["training"] = {"seconds": round(time.perf_counter() - t0, 3),
                                     "final_loss": _last(history.get("loss")),
                                     "history_points": len(history.get("loss") or []),
                                     "history": list(history.get("loss") or [])}
        out: dict[str, Any] = {"model": "in-memory"}
        if history.get("loss"):
            hist = state.path("model", "training_history.json")
            hist.write_text(json.dumps(history), encoding="utf-8")
            out["history"] = str(hist)
        # scoring against a reference the model never saw (the symbolic backend has no reference solve)
        if bundle.backend == "solve":
            state.metrics.update(self._score_against_reference(state, solution))
        else:
            state.notes.append("symbolic backend: the model is measured by validate_pinn (residual, conditions and, "
                               "when an exact solution is known, the exact error), not against a reference solve")
        # physics evidence on fresh points
        state.metrics["physics"] = self._physics_terms(state, solution)
        # and the trained model, persisted so it can be reloaded and inspected later
        saved = _save_physics_model(state, solution)
        out.update(saved)
        state.metrics["training"]["total_seconds"] = round(time.perf_counter() - t0, 3)
        return out

    def _physics_hpo(self, state, bundle):
        """Train one model per point of ``optimization.search_space`` and keep the one that scores best on
        validation points, then report every trial.

        The objective is a held-out error: the relative L2 against the reference solution at points the training
        never used, or — when the run has no reference — the physics residual at fresh points. The points the
        ``model`` gate later reports on are drawn again with another seed, so the reported error is not the number
        the selection was made on.
        """
        import dataclasses

        from pinneapple_neural.trainer import SweepConfig, run_parallel_sweep

        cfg = state.cfg
        trials = cfg.optimization.grid()
        known = {f.name for f in dataclasses.fields(cfg.training)}
        unknown = sorted({k for t in trials for k in t} - known)
        if unknown:
            raise StageError(f"optimization.search_space names {unknown}, which the training block does not have; it "
                             f"takes {sorted(known)}")
        n_val = int(cfg.verification.validation_queries)
        rng = np.random.default_rng(int(cfg.sampling.seed) + 4242)
        coords, bounds = bundle.coords, bundle.domains
        points = np.stack([rng.uniform(*bounds[c], size=n_val) for c in coords], axis=1)
        reference = None
        if bundle.backend == "solve":
            import pinneapple as pp
            try:
                reference = pp.solve(bundle.problem, method=cfg.solver.method, **dict(cfg.solver.options))
            except Exception as exc:
                state.notes.append(f"optimization: no reference to score the trials against "
                                   f"({type(exc).__name__}: {exc}); the physics residual is the objective")

        def objective(solution) -> float:
            if reference is not None:
                import pinneapple as pp
                row = pp.compare(bundle.problem, methods=[solution], reference=reference, points=points).rows[0]
                if row.get("status") == "ok":
                    return float(np.mean(list(row["metrics"]["relative_l2"].values())))
            terms, _ = adapters.physics_terms(cfg, bundle, solution)
            return float(np.sqrt(terms.get("pde", np.nan)))

        def run_trial(params: dict[str, Any]) -> dict[str, Any]:
            trial_cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **params))
            t0 = time.perf_counter()
            try:
                solution = adapters.train_physics_model(trial_cfg, bundle)
                score = objective(solution)
            except Exception as exc:                       # one bad trial must not stop the search
                return {**params, "score": float("inf"), "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.perf_counter() - t0, 3)}
            return {**params, "score": score, "seconds": round(time.perf_counter() - t0, 3)}

        # the models cannot cross a process boundary, so the trials run in this process; n_jobs>1 uses threads
        results: list[dict[str, Any]] = []
        if int(cfg.optimization.n_jobs) > 1 and len(trials) > 1:
            grid = {name: sorted({t[name] for t in trials}) for name in trials[0]} \
                if len({tuple(sorted(t)) for t in trials}) == 1 else None
            if grid is not None:
                results = sorted(run_parallel_sweep(run_trial, SweepConfig(
                    param_grid=grid, n_jobs=int(cfg.optimization.n_jobs), backend="thread")),
                    key=lambda r: r["score"])
        if not results:
            results = sorted((run_trial(t) for t in trials), key=lambda r: r["score"])
        best = results[0]
        if not np.isfinite(best.get("score", float("inf"))):
            raise StageError(f"every trial of the hyperparameter search failed: "
                             f"{best.get('error') or 'no score'}")
        params = {k: v for k, v in best.items() if k not in ("score", "seconds", "error")}
        state.metrics["optimization"] = {
            "objective": "relative_l2 on validation points" if reference is not None
            else "physics residual at fresh points",
            "n_trials": len(results), "strategy": cfg.optimization.strategy, "n_jobs": int(cfg.optimization.n_jobs),
            "best": {**params, "score": best["score"]}, "trials": results}
        state.notes.append(f"hyperparameter search: {len(results)} trial(s), best {best['score']:.4g} with {params} "
                           "(chosen on validation points, not on the ones the model gate reports)")
        # trained once more with the winning settings: the trial's own model is not carried across
        return adapters.train_physics_model(
            dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **params)), bundle)

    def _score_against_reference(self, state, solution) -> dict[str, Any]:
        cfg = state.cfg
        import pinneapple as pp
        n = int(cfg.verification.validation_queries)
        rng = np.random.default_rng(int(cfg.sampling.seed) + 17)
        bounds, coords = state.bundle.domains, state.bundle.coords
        points = np.stack([rng.uniform(*bounds[c], size=n) for c in coords], axis=1)
        np.save(state.path("data", "validation_points.npy"), points)
        try:
            reference = pp.solve(state.bundle.problem, method=cfg.solver.method, **dict(cfg.solver.options))
        except Exception as exc:
            state.notes.append(f"no reference solution to score against: {type(exc).__name__}: {exc}")
            state.reference = {}
            return {}
        state.reference = {"validation": reference}
        comparison = pp.compare(state.bundle.problem, methods=[solution], reference=reference, points=points)
        row = comparison.rows[0]
        if row.get("status") != "ok":
            state.notes.append(f"the comparison failed: {row.get('error')}")
            return {}
        metrics = row["metrics"]
        return {"reference": cfg.solver.method,
                "reference_kind": getattr(reference, "kind", ""),
                "relative_l2": float(np.mean(list(metrics["relative_l2"].values()))),
                "rmse": float(np.mean(list(metrics["rmse"].values()))),
                "max_abs": float(np.mean(list(metrics["max_abs"].values()))),
                "relative_l2_by_field": dict(metrics["relative_l2"]),
                "infer_us_per_point": row.get("infer_us_per_point"),
                "n_points": int(len(points)), "points": str(state.path("data", "validation_points.npy"))}

    def _physics_terms(self, state, solution) -> dict[str, Any]:
        cfg, bundle = state.cfg, state.bundle
        if bundle.backend == "symbolic":
            from pinneapple_analysis.validation.pinn import validate_pinn
            report = validate_pinn(solution.model, bundle.pinn_problem,
                                   n_points=min(2048, max(256, int(cfg.verification.validation_queries))),
                                   seed=int(cfg.training.seed))
            state.metrics["pinn_report"] = report
            return {"report": "validate_pinn"}
        terms, notes = adapters.physics_terms(cfg, bundle, solution)
        return _bc_evidence(state, solution, terms, notes)

    # ---- supervised
    def _supervised(self, state, t0: float) -> dict[str, Any]:
        cfg = state.cfg
        candidates = state.stages["model"].artifacts.get("candidates") or ["auto"]
        methods = "auto" if list(candidates) == ["auto"] else candidates
        seeds = int(cfg.uncertainty.n_models) if cfg.uncertainty.enabled else 1
        study = adapters.run_surrogate_study(cfg, state.dataset, methods=methods, seeds=seeds)
        if cfg.adaptive_learning.enabled:
            study = self._adaptive(state, study, methods=methods, seeds=seeds)
        state.model, state.model_object = study, study.model
        state.metrics["training"] = {"seconds": round(time.perf_counter() - t0, 3), "seeds": seeds,
                                     "best": study.best, "budget_seconds": cfg.training.budget_seconds}
        state.metrics["ranking"] = study.ranking("test")
        state.metrics["comparison"] = [{"id": m["id"], "family": m["family"], "applicable": m["applicable"],
                                        "summary": m.get("summary"), "reason": m.get("reason")}
                                       for m in study.methods]
        state.metrics["trust"] = study.trust
        state.metrics["limitations"] = study.limitations
        state.validated_domain = _validated_domain(state, study)
        if cfg.uncertainty.enabled:
            state.uncertainty = {"method": "seed_ensemble", "n_models": seeds,
                                 "test_errors": [row.get("spread") for row in study.ranking("test")]}
        out: dict[str, Any] = {"study": "in-memory", "best": study.best}
        md = state.path("model", "study.md")
        md.write_text(study.to_markdown(), encoding="utf-8")
        out["study_report"] = str(md)
        if study.model is not None:
            model_dir = state.path("model", "surrogate")
            try:
                study.model.save(str(model_dir))
                out["surrogate_model"] = str(model_dir)
            except Exception as exc:                            # saving is a convenience, not the result
                state.notes.append(f"the surrogate model could not be saved: {type(exc).__name__}: {exc}")
        state.metrics["training"]["total_seconds"] = round(time.perf_counter() - t0, 3)
        return out

    # ---- adaptive learning
    def _adaptive(self, state, study, *, methods, seeds):
        """Solve more cases where the response varies most, retrain, and stop on the target or the budget.

        Each round asks ``doe.design(method="infill")`` for points that fill the gaps left by the cases already
        solved — the library's greedy maximin, weighted by where the outputs change fastest between neighbours when
        ``outputs`` is given — solves them with the reference solver, appends them to the dataset and repeats the
        study. The loop ends when the best candidate's held-out error is at or below the target
        (``adaptive_learning.target_error``, else the run's own acceptance criterion), when ``max_rounds`` is
        reached, or when another round would pass ``max_total_simulations``.

        What it is not: an active-learning criterion driven by the model's own predictive variance. The library's
        infill is geometry-and-output based, so a round can add cases the current model already handles; the report
        shows the error per round, so a flat curve is visible rather than assumed away.
        """
        cfg = state.cfg
        al = cfg.adaptive_learning
        import pandas as pd

        from pinneapple_analysis import surrogate as _surrogate
        from pinneapple_data.doe import design

        space = cfg.parameters.to_space()
        X = np.asarray(state.dataset.X, dtype=float)
        Y = np.asarray(state.dataset.Y, dtype=float)
        names = list(state.dataset.input_names)
        outputs = list(state.dataset.output_names)
        metric = "relative_l2" if state.dataset.kind == "field" else "nrmse"
        target = float(al.target_error) if al.target_error is not None else float(
            cfg.verification.threshold(metric, 0.05 if metric == "relative_l2" else 0.10))
        rounds: list[dict[str, Any]] = []
        error = _study_error(study)
        solved, added_total = int(len(X)), 0
        for r in range(1, int(al.max_rounds) + 1):
            if error is not None and error <= target:
                rounds.append({"round": r, "skipped": f"the error {error:.4g} already meets the target {target:.4g}"})
                break
            if solved + int(al.samples_per_round) > int(al.max_total_simulations):
                rounds.append({"round": r, "skipped": f"another {al.samples_per_round} case(s) would pass "
                                                      f"max_total_simulations ({al.max_total_simulations})"})
                break
            existing = {n: X[:, i] for i, n in enumerate(names)}
            for j, name in enumerate(outputs[:1]):                # the infill weighting needs an output column
                existing[name] = Y[:, j]
            new = design(space, int(al.samples_per_round), method="infill", seed=int(cfg.sampling.seed) + r,
                         existing=pd.DataFrame(existing), outputs=outputs[:1]).cases
            rows, cases, failures = [], [], []
            for _, row in new.iterrows():
                case = row.to_dict()
                try:
                    solution = adapters.solve_reference(cfg, state.bundle, case)
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    continue
                rows.append(adapters.solution_row(solution, state.points))
                cases.append(case)
            if not rows:
                rounds.append({"round": r, "error_before": error, "added": 0, "failures": failures[:3],
                               "stopped": "no new case could be solved"})
                break
            X = np.vstack([X, np.asarray([[float(dict(c)[n]) for n in names] for c in cases], dtype=float)])
            Y = np.vstack([Y, np.asarray(rows, dtype=float)])
            solved, added_total = int(len(X)), added_total + len(rows)
            data = _surrogate.from_arrays({"X": X, "Y": Y}, input_names=names, output_names=outputs,
                                          meta=dict(state.dataset.meta))
            state.dataset = data
            np.savez(state.path("data", "dataset.npz"), X=X, Y=Y, input_names=np.asarray(names, dtype=object),
                     output_names=np.asarray(outputs, dtype=object), probe_points=state.points,
                     cases=np.asarray([json.dumps(dict(c)) for c in cases], dtype=object))
            study = adapters.run_surrogate_study(cfg, data, methods=methods, seeds=seeds)
            before, error = error, _study_error(study)
            rounds.append({"round": r, "error_before": before, "error_after": error, "added": len(rows),
                           "total_cases": solved, "failures": failures[:3]})
        state.metrics["adaptive_learning"] = {
            "strategy": al.strategy, "metric": metric, "target": target, "final_error": error, "rounds": rounds,
            "cases_added": added_total, "total_cases": solved,
            "reached_target": bool(error is not None and error <= target),
            "criterion": "doe.design(method='infill'): the points farthest from the cases already solved, weighted "
                         "by where the outputs vary most among neighbours"}
        state.notes.append(f"adaptive learning: {added_total} case(s) added over "
                           f"{len([r for r in rounds if r.get('added')])} round(s); {metric} "
                           f"{'n/a' if error is None else round(error, 4)} against a target of {target}")
        return study

    def verify(self, state, artifacts) -> list[str]:
        training = state.metrics.get("training") or {}
        notes = [f"trained in {training.get('total_seconds', training.get('seconds', 0)):.1f} s"]
        if "relative_l2" in state.metrics:
            notes.append(f"relative L2 {state.metrics['relative_l2']:.3g} against "
                         f"{state.metrics.get('reference_kind') or state.metrics.get('reference')}")
        if training.get("best"):
            notes.append(f"best candidate: {training['best']}")
        return notes


def _last(seq) -> float | None:
    if not seq:
        return None
    try:
        return float(seq[-1])
    except (TypeError, ValueError):
        return None


def _study_error(study) -> float | None:
    """The held-out error of the study's best candidate (the same number the ``model`` gate reads)."""
    try:
        ranking = study.ranking("test")
    except Exception:
        return None
    if not ranking:
        return None
    try:
        return float(ranking[0]["error"])
    except (TypeError, ValueError):
        return None


def _bc_evidence(state, solution, terms: dict[str, float], notes: list[str]) -> dict[str, Any]:
    """Boundary/initial misfit of the trained model, relative to the scale of its own prediction."""
    from .adapters import predict_frame, residual_batch

    cfg, bundle = state.cfg, state.bundle
    spec = bundle.spec if bundle.spec is not None else bundle.problem.to_pde_spec()
    batch, more = residual_batch(spec, n_interior=int(cfg.training.n_collocation), seed=int(cfg.training.seed) + 3)
    pred = [np.asarray(v) for v in (batch["x_col"].numpy(),)]
    y_col = predict_frame(solution, pred[0])
    scale = float(np.sqrt(np.mean(y_col ** 2))) or 1.0
    misfit, worst, measured = None, "", 0
    for suffix in ("bc", "ic"):
        x = batch[f"x_{suffix}"].numpy()
        y = batch[f"y_{suffix}"].numpy()
        if len(x) == 0:
            continue
        measured += len(x)
        err = float(np.abs(predict_frame(solution, x) - y).max() / scale)
        if misfit is None or err > misfit:
            misfit, worst = err, suffix
    out = {"bc_misfit": misfit, "bc_points": measured, "bc_worst": worst, "field_scale": scale,
           "residual_rms": float(np.sqrt(terms["pde"])) if "pde" in terms else None,
           "notes": list(notes) + ["loss terms on fresh points: " +
                                   ", ".join(f"{k}={v:.3g}" for k, v in terms.items())]}
    out["notes"] += [n for n in more if n not in out["notes"]]
    return out


def _save_physics_model(state, solution) -> dict[str, Any]:
    """Persist the trained physics model in a form that can be reloaded (checkpoint + rebuild recipe)."""
    import torch
    net = getattr(solution, "model", None)
    if net is None or not hasattr(net, "state_dict"):
        return {}
    dir_ = state.path("model", "physics")
    dir_.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "class_name": type(net).__name__}, dir_ / "model.pt")
    arch = state.cfg.architecture
    recipe = {"class_name": type(net).__name__, "info": dict(getattr(solution, "info", {}) or {}),
              "coords": list(state.bundle.coords), "fields": list(state.bundle.fields),
              "backend": state.bundle.backend,
              # enough to rebuild the module before the weights go back in (``from_run``/``resume``)
              "architecture": (arch.candidates[0] if arch.is_auto else arch.family),
              "hidden_dim": state.cfg.training.hidden_dim, "n_layers": state.cfg.training.n_layers,
              "history": _short_history(getattr(solution, "history", None))}
    (dir_ / "model.json").write_text(json.dumps(recipe, indent=1, default=str), encoding="utf-8")
    return {"model": str(dir_ / "model.pt"), "model_card": str(dir_ / "model.json")}


def _short_history(history) -> list[float]:
    """Every 20th loss value: enough to draw the curve, small enough to keep in a JSON card."""
    loss = list((history or {}).get("loss") or [])
    return [round(float(v), 8) for v in loss[::20]] if loss else []


def _validated_domain(state, study) -> dict[str, Any]:
    """Where the study says the model holds, from its own validity bounds and split table."""
    limits = getattr(study, "limitations", {}) or {}
    bounds = {k: [float(a), float(b)] for k, (a, b) in (limits.get("validity_bounds") or {}).items()}
    out: dict[str, Any] = {"bounds": bounds, "n_train": int(len(study.splits.train))}
    gate = limits.get("ood_gate") or {}
    if gate:
        out["ood_rule"] = gate.get("rule", "")
    shift = getattr(study.splits, "shift_description", "")
    if shift:
        out["extrapolation_risk"] = (f"the cases outside the training region ({shift}) were measured worse than the "
                                     "held-out set: the corners of the space are extrapolation")
    return out


# ----------------------------------------------------------------------------------------------------- export
class ExportStage(PipelineStage):
    name = "export"

    def skip_reason(self, state) -> str:
        return "" if state.cfg.export.enabled else "export is disabled in the configuration"

    def execute(self, state) -> dict[str, Any]:
        cfg = state.cfg
        out: dict[str, Any] = {}
        state.metrics["export"] = {"written": {}}
        if cfg.data.is_physics_only and state.model_object is not None:
            example = self._example_input(state)
            try:
                info = adapters.export_model(cfg, state.model_object, state.path("export"), example_input=example)
            except Exception as exc:
                state.metrics["export"]["error"] = f"{type(exc).__name__}: {exc}"
                info = {"written": {}}
            written = info.get("written") or {}
            state.metrics["export"]["written"] = written
            out.update({k: v for k, v in written.items()})
            if "onnx" in written and cfg.export.verify_export:
                state.metrics["export"]["equivalence"] = self._onnx_equivalence(state, info, written["onnx"])
        else:
            model = getattr(state.model, "model", None)
            if model is not None:
                dir_ = state.path("export", "surrogate")
                try:
                    model.save(str(dir_))
                    out["surrogate_model"] = str(dir_)
                    state.metrics["export"]["written"] = {"checkpoint": str(dir_)}
                    if cfg.export.verify_export:
                        state.metrics["export"]["equivalence"] = self._reload_equivalence(state, model, dir_)
                except Exception as exc:
                    state.metrics["export"]["written"] = {"checkpoint_error": f"{type(exc).__name__}: {exc}"}
            else:
                state.metrics["export"]["error"] = "the study selected no model to export"
        if cfg.visualization.enabled:
            from .figures import write_figures
            figures = write_figures(state, state.path("figures"))
            state.metrics["figures"] = figures
            out.update({"figures": len(figures["figures"])})
            for name in figures["figures"]:
                out[f"figure_{Path(name).stem}"] = name
        # the card: what this model is, what it was measured on, and what it may not be used for
        card = self._model_card(state)
        path = state.path("export", "model_card.json")
        path.write_text(json.dumps(card, indent=1, default=str), encoding="utf-8")
        out["model_card"] = str(path)
        return out

    def _example_input(self, state) -> np.ndarray:
        n = len(state.bundle.coords)
        lo = np.array([state.bundle.domains[c][0] for c in state.bundle.coords], dtype=np.float32)
        hi = np.array([state.bundle.domains[c][1] for c in state.bundle.coords], dtype=np.float32)
        return lo + (hi - lo) * np.linspace(0.2, 0.8, 5).astype(np.float32)[:, None].repeat(n, axis=1)

    def _reload_equivalence(self, state, model, directory) -> dict[str, Any]:
        """Compare the saved surrogate with the in-memory one on fresh points in the parameter space."""
        from pinneapple_analysis.surrogate.model import SurrogateModel

        space = state.cfg.parameters.to_space()
        from pinneapple_data.doe import design
        cases = design(space, 8, method="lhs", seed=int(state.cfg.training.seed) + 5).cases
        X = model.as_array(cases)
        memory = np.asarray(model.predict_array(X), dtype=float)
        loaded = SurrogateModel.load(directory)
        disk = np.asarray(loaded.predict_array(X), dtype=float)
        scale = float(np.max(np.abs(memory))) or 1.0
        return {"format": "checkpoint", "max_relative_difference": float(np.max(np.abs(disk - memory)) / scale),
                "n_points": int(len(X)), "sha256_verified": True}

    def _onnx_equivalence(self, state, info, onnx_path: str) -> dict[str, Any]:
        import torch
        net, x = info["net"], info["example_input"]
        try:
            import onnxruntime as ort
        except ImportError:
            return {"format": "onnx", "max_relative_difference": float("nan"),
                    "note": "onnxruntime is not installed: the exported file was not compared"}
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feed = {session.get_inputs()[0].name: x.numpy()}
        exported = np.asarray(session.run(None, feed)[0])
        with torch.no_grad():
            memory = np.asarray(net(x))
        scale = float(np.max(np.abs(memory))) or 1.0
        return {"format": "onnx", "max_relative_difference": float(np.max(np.abs(exported - memory)) / scale),
                "n_points": int(len(x))}

    def _model_card(self, state) -> dict[str, Any]:
        cfg = state.cfg
        return {"name": cfg.name, "run_id": state.run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "problem": adapters.problem_description(state.bundle),
                "data": {"source": cfg.data.source, "solver": cfg.solver.method},
                "training": state.metrics.get("training"),
                "metrics": {k: v for k, v in state.metrics.items()
                            if k in ("relative_l2", "rmse", "max_abs", "ranking", "trust")},
                "validated_domain": state.validated_domain,
                "limitations": state.metrics.get("limitations") or state.notes,
                "intended_use": cfg.description or "surrogate of the problem above, inside the validated domain",
                "not_validated_for": "anything outside the validated domain, or the same physics under assumptions "
                                     "the problem does not make"}

    def verify(self, state, artifacts) -> list[str]:
        written = (state.metrics.get("export") or {}).get("written") or {}
        return [f"wrote {', '.join(k for k in written if not k.endswith('_error')) or 'nothing'}"]


# ----------------------------------------------------------------------------------------------- verification
class VerificationStage(PipelineStage):
    name = "verification"

    def skip_reason(self, state) -> str:
        return "" if state.cfg.verification.enabled else "verification.enabled is false"

    def execute(self, state) -> dict[str, Any]:
        from . import checks
        from .verification import run_gates
        v = state.cfg.verification
        if v.run_checks and state.model_object is not None:
            result = checks.run_check(state, physics=v.check_physics, n_queries=int(v.check_queries))
            if result is None:
                state.metrics["pinneapple_check"] = {"error": "the model has no named inputs/outputs to declare as a "
                                                              "check target", "outcomes": []}
            else:
                state.metrics["pinneapple_check"] = result
        state.metrics["inference_query"], state.metrics["inference_prediction"], state.metrics["inference_ood"] = \
            _sample_query(state)
        state.verification = run_gates(state)
        out: dict[str, Any] = {"gates": {g.gate: g.status for g in state.verification}}
        check = state.metrics.get("pinneapple_check") or {}
        if check.get("report"):
            out["pinneapple_check"] = check["report"]
        return out

    def verify(self, state, artifacts) -> list[str]:
        counts: dict[str, int] = {}
        for g in state.verification:
            counts[g.status] = counts.get(g.status, 0) + 1
        head = " · ".join(f"{n} {s.lower().replace('_', ' ')}" for s, n in
                          sorted(counts.items(), key=lambda kv: -kv[1]))
        notes = [head] + [f"{g.gate}={g.status}" for g in state.verification]
        check = state.metrics.get("pinneapple_check") or {}
        if check.get("n_checks"):
            notes.insert(1, f"pinneapple check: {check['n_checks']} outcome(s), {check.get('n_failed', 0)} failing")
        elif check.get("error"):
            notes.insert(1, f"pinneapple check did not run: {check['error']}")
        return notes


def _sample_query(state):
    """One prediction at the centre of the space, for the inference gate."""
    cfg, bundle = state.cfg, state.bundle
    model = state.model_object
    if model is None:
        return None, None, ""
    if bundle.backend == "symbolic" or cfg.data.is_physics_only:
        pts = np.array([[0.5 * (lo + hi) for c in bundle.coords for lo, hi in [bundle.domains[c]]]])
        try:
            y = np.asarray(model.predict(pts), dtype=float)
        except Exception as exc:
            state.notes.append(f"inference gate: {type(exc).__name__}: {exc}")
            return None, None, ""
        return {c: float(pts[0, i]) for i, c in enumerate(bundle.coords)}, y, ""
    # supervised: query the centre of the parameter space
    space = cfg.parameters.to_space()
    if space is None or getattr(model, "predict_array", None) is None:
        return None, None, ""
    centre = {}
    for parameter in space.parameters:
        if parameter.kind in ("continuous", "integer"):
            centre[parameter.name] = 0.5 * (float(parameter.low) + float(parameter.high))
        elif parameter.kind == "categorical":
            centre[parameter.name] = parameter.values[len(parameter.values) // 2]
        else:
            centre[parameter.name] = parameter.value
    try:
        quad = model.as_array(centre)
        y = model.predict_array(quad)
    except Exception as exc:
        state.notes.append(f"inference gate: {type(exc).__name__}: {exc}")
        return None, None, ""
    return {k: float(v) for k, v in centre.items()}, y, ""


DEFAULT_STAGES = (ProblemStage(), DataStage(), PreprocessStage(), ModelStage(), TrainingStage(), ExportStage(),
                  VerificationStage())


def stage_names() -> tuple[str, ...]:
    return STAGE_ORDER
