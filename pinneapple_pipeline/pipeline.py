"""``SurrogatePipeline``: one object for the whole cycle of a surrogate model.

The pipeline is a facade over what the library already does, in the order a surrogate model needs it:

    problem  ->  data  ->  preprocess  ->  model  ->  training  ->  export  ->  verification
                 |__________ supervised: reference solves -> dataset -> study __________|
                 |__________ physics only: a PINN on the equations _____________________|

>>> import pinneapple as pp
>>> pipeline = pp.SurrogatePipeline.from_preset("burgers_in_viscosity")       # doctest: +SKIP
>>> result = pipeline.run()                                                   # doctest: +SKIP
>>> result.verification[?].status, result.metrics["relative_l2"]               # doctest: +SKIP

Three levels of use, the same object:

* beginner — ``pp.SurrogatePipeline.from_preset(name).run()``; the presets are in
  :mod:`pinneapple_pipeline.presets`.
* engineer — build the configuration (YAML or keyword blocks), ``run()``, read ``result.to_markdown()``.
* researcher — ``pipeline.register_stage(...)`` replaces a step, and the objects the stages build are reachable
  (``pipeline.state.bundle``, ``.dataset``, ``.model``) so a component can be trained or scored on its own.

What it does not do: it does not verify anything by itself. Every gate lives in
:mod:`pinneapple_pipeline.verification` and reads what the earlier stages measured.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configs import ConfigError, SurrogateConfig
from .results import FAIL, STAGE_ORDER, PipelineResult, StageRecord
from .stages import DEFAULT_STAGES, PipelineStage, StageError
from .state import RunState

__all__ = ["SurrogatePipeline", "PipelineRejected", "PipelineFailed", "SurrogatePipelineResult"]

SurrogatePipelineResult = PipelineResult


class PipelineRejected(RuntimeError):
    """A gate failed and ``verification.fail_policy`` is ``block``: the run is not fit for use as it stands."""

    def __init__(self, result: PipelineResult):
        self.result = result
        failed = ", ".join(f"{g.gate} ({g.status})" for g in result.verification if g.status == FAIL)
        super().__init__(f"run '{result.name}' rejected by verification: {failed}. The report is at "
                         f"{result.artifacts.get('report', '(not written)')}")


class PipelineFailed(RuntimeError):
    """A stage could not finish: the run stopped there, with everything before it kept on disk."""

    def __init__(self, stage: str, error: str, state: RunState):
        self.stage, self.state = stage, state
        super().__init__(f"stage '{stage}' failed: {error}")


class SurrogatePipeline:
    """Build, train, verify, export and query a surrogate model of one problem."""

    def __init__(self, config: SurrogateConfig | Mapping[str, Any] | None = None, *, name: str | None = None,
                 stages: list[PipelineStage] | None = None, **blocks: Any):
        if config is None:
            config = SurrogateConfig(name=name or "", **blocks)
        elif isinstance(config, Mapping):
            config = SurrogateConfig.from_dict({**dict(config), **({"name": name} if name else {}), **blocks})
        elif not isinstance(config, SurrogateConfig):
            raise TypeError(f"config must be a SurrogateConfig or a mapping, got {type(config).__name__}")
        elif blocks:
            raise TypeError(f"give either a config or blocks, not both (blocks: {sorted(blocks)})")
        self.config = config
        self.stages: list[PipelineStage] = list(stages or DEFAULT_STAGES)
        self.state: RunState | None = None
        self.result: PipelineResult | None = None
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

    # -------------------------------------------------------------------------------------- construction
    @classmethod
    def from_config(cls, config: Mapping[str, Any] | SurrogateConfig, **overrides: Any) -> SurrogatePipeline:
        if isinstance(config, SurrogateConfig):
            return cls(config)
        return cls(SurrogateConfig.from_dict({**dict(config), **overrides}))

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> SurrogatePipeline:
        return cls(SurrogateConfig.from_yaml(path, **overrides))

    from_json = from_yaml

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> SurrogatePipeline:
        """A pipeline from :data:`pinneapple_pipeline.presets.PRESETS`; ``name="..."`` in ``overrides`` renames the
        run (its folder under ``execution.out_dir``)."""
        from .presets import PRESETS, preset_config
        if not preset:
            raise ConfigError(f"name a preset: {sorted(PRESETS)}")
        if preset not in PRESETS:
            raise ConfigError(f"unknown preset {preset!r}; the presets are {sorted(PRESETS)}")
        return cls(preset_config(preset, **overrides))

    def save(self, path: str | Path | None = None) -> str:
        """Write the configuration as YAML (``<out_dir>/<name>.yaml`` by default) and return the path."""
        target = Path(path) if path else Path(self.config.execution.out_dir) / f"{self.config.name}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(target)
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> SurrogatePipeline:
        """A pipeline from the YAML (or JSON) file a previous ``save`` wrote."""
        return cls.from_yaml(path)

    @classmethod
    def from_run(cls, run_dir: str | Path) -> SurrogatePipeline:
        """The pipeline of a finished run folder, with the model reloaded and the gates of that run.

        The folder is self-describing: ``config.json`` (written by every run) is the configuration, ``state.json``
        the metrics and the validated domain, ``report.json`` the gates, and the export stage the model itself. What
        comes back answers ``predict``/``predict_verified`` and ``deploy`` with the same numbers the run reported,
        on any machine that has the folder — this is what ``verification/predict.py`` calls, so the checker loads
        the artefact rather than a fresh pipeline.
        """
        import json

        from .state import CONFIG_FILE, STATE_FILE

        folder = Path(run_dir)
        config_file = folder / CONFIG_FILE
        if not config_file.is_file():
            raise ConfigError(f"{folder} is not a run folder: {CONFIG_FILE} is missing (was the run written by "
                              "run()? it is written with the state)")
        cfg = SurrogateConfig.from_dict(json.loads(config_file.read_text(encoding="utf-8")))
        cfg.execution.out_dir = str(folder.parent)
        cfg.name = folder.name
        pipeline = cls(cfg)
        state = pipeline._prepare_state(resume=(folder / STATE_FILE).is_file())
        state.model = state.model_object
        pipeline.state = state
        report = folder / "report.json"
        if report.is_file():
            from .results import PipelineResult
            pipeline.result = PipelineResult.from_dict(json.loads(report.read_text(encoding="utf-8")))
            state.verification = list(pipeline.result.verification)
            for key, value in (pipeline.result.metrics or {}).items():
                state.metrics.setdefault(key, value)
            if pipeline.result.validated_domain and not state.validated_domain:
                state.validated_domain = dict(pipeline.result.validated_domain)
        return pipeline

    def register_stage(self, stage: PipelineStage, *, replace: bool = True) -> SurrogatePipeline:
        """Add (or replace) a stage, identified by its ``name``; it runs in :data:`results.STAGE_ORDER` position."""
        if not stage.name:
            raise ValueError("a stage must have a name")
        existing = [i for i, s in enumerate(self.stages) if s.name == stage.name]
        if existing and not replace:
            raise ValueError(f"a stage named {stage.name!r} is already registered")
        if existing:
            self.stages[existing[0]] = stage
        else:
            self.stages.append(stage)
        return self

    # -------------------------------------------------------------------------------------- configuration
    def validate_config(self) -> list[str]:
        """Check the configuration against the problem before running anything; returns the problems found."""
        self.config.validate()
        issues: list[str] = []
        names = [s.name for s in self.stages]
        unknown = [n for n in names if n not in STAGE_ORDER]
        if unknown:
            issues.append(f"stage(s) {unknown} are not in the run order {list(STAGE_ORDER)}")
        if self.config.export.formats and not self.config.export.enabled:
            issues.append("export.formats is set but export.enabled is false")
        if issues:
            raise ConfigError("; ".join(issues))
        return issues

    def describe(self) -> str:
        c = self.config
        lines = [f"{c.name}: {c.description or 'surrogate pipeline'}",
                 f"  problem      {c.problem.source}"
                 + (f" {c.problem.preset}" if c.problem.preset else "")
                 + (f" ({len(c.problem.equations)} equation(s))" if c.problem.equations else ""),
                 f"  data         {c.data.source}"
                 + (f", solver {c.solver.method}" if c.data.source != "physics_only" else ", equations only"),
                 f"  parameters   {list(c.parameters.parameters) or '(none: one problem)'}",
                 f"  model        {c.architecture.family}"
                 + (f" from {list(c.architecture.candidates)}" if c.architecture.is_auto else ""),
                 "  training     " + (f"{c.training.epochs} epochs, lr {c.training.lr}, device {c.training.device}"
                                     if c.data.is_physics_only else
                                     f"budget {c.training.budget_seconds or 10.0} s per candidate, "
                                     f"{len(c.architecture.candidates) or 'all'} candidate(s)"),
                 f"  verification {c.verification.mode} ({c.verification.fail_policy} on failure)",
                 f"  output       {Path(c.execution.out_dir) / c.name}"]
        return "\n".join(lines)

    def plan(self) -> dict[str, Any]:
        """What will run, in order, with what it will cost in the dominant operation."""
        c = self.config
        steps = []
        for stage in self.stages:
            skip = _skip_reason(stage, self._empty_state())
            steps.append({"stage": stage.name, "runs": not skip, "skipped_because": skip, "does": _DOES.get(
                stage.name, "custom stage")})
        cases = int(c.sampling.design_sampling.get("n_samples", 16)) if c.data.source == "simulation" else 0
        return {"name": c.name, "steps": steps,
                "cost": {"reference_solves": cases, "training_runs": (cases and len(c.architecture.candidates)) or 1,
                         "epochs": int(c.training.epochs),
                         "study_budget_seconds": c.training.budget_seconds if c.data.source != "physics_only" else None}}

    def dry_run(self) -> dict[str, Any]:
        """Everything ``run`` would check before it spends time: configuration, problem, dependencies, stages."""
        self.validate_config()
        report: dict[str, Any] = {"name": self.config.name, "ok": True, "checks": []}
        checks = report["checks"]

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append({"check": name, "ok": bool(ok), "detail": detail})
            report["ok"] = report["ok"] and bool(ok)

        from . import adapters
        try:
            bundle = adapters.build_problem(self.config)
            add("problem", True, f"{bundle.name}: coords {bundle.coords}, fields {bundle.fields}, "
                                 f"backend {bundle.backend}")
        except Exception as exc:
            add("problem", False, f"{type(exc).__name__}: {exc}")
            return report
        state = self._empty_state(bundle=bundle)
        for stage in self.stages:
            skip = _skip_reason(stage, state)
            if skip:
                add(stage.name, True, f"skipped: {skip}")
                continue
            try:
                problems = stage.validate(state)
            except Exception as exc:
                problems = [f"{type(exc).__name__}: {exc}"]
            add(stage.name, not problems, "; ".join(problems) or "ready")
        for name, module in (("torch", "torch"), ("scipy", "scipy"), ("pandas", "pandas")):
            try:
                __import__(module)
                add(f"dependency {name}", True, "installed")
            except ImportError:
                add(f"dependency {name}", False, "missing")
        if self.config.export.formats and any(f in ("onnx", "torchscript") for f in self.config.export.formats):
            try:
                __import__("onnxruntime" if "onnx" in self.config.export.formats else "torch")
                add("export runtime", True, "available")
            except ImportError:
                add("export runtime", False, "onnxruntime is not installed: export equivalence cannot be checked")
        return report

    def estimate_cost(self) -> dict[str, Any]:
        """An order-of-magnitude estimate from the configuration and what this machine has."""
        c = self.config
        try:
            from pinneapple_neural.trainer import best_device
            device = str(best_device())
        except Exception:                                            # pragma: no cover
            device = "cpu"
        cases = int(c.sampling.design_sampling.get("n_samples", 16)) if c.data.source == "simulation" else 0
        if c.data.is_physics_only:
            train_s = c.training.epochs * (0.02 if c.problem.source == "symbolic" else 0.07)
        else:
            train_s = float(c.training.budget_seconds or 10.0) * max(1, len(c.architecture.candidates))
        return {"device": device, "reference_solves": cases, "training_seconds": round(train_s, 1),
                "verification_queries": int(c.verification.validation_queries),
                "note": "an estimate from the configuration alone: the reference solves account for most of the "
                        "wall time when the parameter space is wide"}

    # ---------------------------------------------------------------------------------------------- running
    def _empty_state(self, bundle: Any = None) -> RunState:
        out = Path(self.config.execution.out_dir) / self.config.name
        state = RunState(cfg=self.config, run_id=self.run_id, out_dir=out, bundle=bundle)
        state.stages = {s.name: StageRecord(s.name) for s in self.stages}
        return state

    def run(self, *, until: str | None = None, strict: bool | None = None) -> PipelineResult:
        """Run the pipeline (or up to and including ``until``); returns the :class:`PipelineResult`.

        Raises :class:`PipelineFailed` when a stage cannot finish and :class:`PipelineRejected` when a gate fails
        and ``verification.fail_policy`` is ``block`` (the result is attached to both, and written to disk either
        way). ``strict=True`` forces the rejection, ``strict=False`` never raises for a failed gate.
        """
        return self._run(until=until, strict=strict)

    def run_until(self, stage: str) -> PipelineResult:
        """Run every stage up to and including ``stage`` (``"data"`` trains nothing)."""
        return self._run(until=stage, strict=False)

    def run_stage(self, stage: str) -> StageRecord:
        """Run exactly one stage, on the state the previous run left (or a fresh one)."""
        self.validate_config()
        state = self._prepare_state()
        target = _stage_by_name(self.stages, stage)
        record = state.record(stage)
        self._execute_stage(target, state, record)
        state.save()
        self.state = state
        return record

    def resume(self) -> PipelineResult:
        """Continue a run: finished stages whose artefacts are on disk are not repeated."""
        return self._run(until=None, strict=None, resume=True)

    def _run(self, *, until: str | None = None, strict: bool | None = None, resume: bool = False) -> PipelineResult:
        self.validate_config()
        state = self._prepare_state(resume=resume or self.config.execution.resume)
        if state.bundle is None:
            state.bundle = self._rebuild_bundle()
        t0 = time.perf_counter()
        wanted = _stages_until(self.stages, until)
        for stage in wanted:
            record = state.record(stage.name)
            if state.finished(stage.name) and self.config.execution.resume and stage.name != "verification":
                record.notes.append("skipped: finished in a previous run (execution.resume)")
                continue
            self._execute_stage(stage, state, record)
            state.save()
        result = self._build_result(state, seconds=time.perf_counter() - t0)
        self.state, self.result = state, result
        rejected = self._apply_policy(result, strict=strict)
        result.to_json(str(self._write_report(state, result)))
        state.save()
        if rejected:
            raise PipelineRejected(result)
        return result

    def _prepare_state(self, resume: bool = False) -> RunState:
        out = Path(self.config.execution.out_dir) / self.config.name
        existing = RunState.load(out) if resume else None
        run_id = (existing or {}).get("run_id") or self.run_id
        state = RunState(cfg=self.config, run_id=run_id, out_dir=out)
        state.stages = {s.name: StageRecord(s.name) for s in self.stages}
        state.bundle = self._rebuild_bundle()
        saved_points = out / "data" / "probe_points.npy"
        if state.bundle is not None and not self.config.data.is_physics_only and saved_points.is_file():
            import numpy as np
            state.points = np.load(saved_points)
        saved_simulation = out / "data" / "simulation.json"
        if saved_simulation.is_file():
            import json as _json
            try:
                state.metrics["simulation"] = _json.loads(saved_simulation.read_text(encoding="utf-8"))
            except ValueError:                                # a truncated summary is not fatal
                pass
        if existing:
            state.restore_stage_status(existing)
            restored = [s for s in state.stages if state.finished(s)]
            if restored:
                state.notes.append(f"resumed from {out / 'state.json'}: stage(s) {restored} were already finished")
            # the trained objects live in memory only: reload them so the gates measure what is on disk
            state.model_object = self._reload_model(out)
            state.dataset = self._reload_dataset(out)
        out.mkdir(parents=True, exist_ok=True)
        return state

    def _reload_model(self, out: Path):
        """The trained model of a previous run, reloaded from what the export stage wrote (``resume`` in a new
        process has no memory of the fitted object)."""
        if not self.config.data.is_physics_only:
            directory = out / "export" / "surrogate"
            if (directory / "model.pkl").is_file():
                try:
                    from pinneapple_analysis.surrogate.model import SurrogateModel
                    return SurrogateModel.load(directory)
                except Exception:
                    return None
        from . import adapters
        bundle = self._rebuild_bundle()
        if bundle is None:
            return None
        try:
            return adapters.reload_physics_model(self.config, bundle, out)
        except Exception:                                       # a checkpoint that will not load is reported
            return None

    def _reload_dataset(self, out: Path):
        """The dataset of a previous run, rebuilt from ``data/dataset.npz``."""
        path = out / "data" / "dataset.npz"
        if not path.is_file():
            return None
        try:
            import numpy as np

            from pinneapple_analysis import surrogate
            with np.load(path, allow_pickle=True) as npz:
                return surrogate.from_arrays({"X": np.asarray(npz["X"], dtype=float),
                                             "Y": np.asarray(npz["Y"], dtype=float)},
                                            input_names=[str(s) for s in npz["input_names"]],
                                            output_names=[str(s) for s in npz["output_names"]],
                                            meta={"source": f"reloaded from {path}"})
        except Exception:
            return None

    def _rebuild_bundle(self):
        """The problem of the configuration, rebuilt from scratch: ``resume`` starts in a new process as often as
        not, and every stage after ``problem`` reads it."""
        from . import adapters
        try:
            return adapters.build_problem(self.config)
        except Exception:                                    # the problem stage reports the failure itself
            return None

    def _execute_stage(self, stage: PipelineStage, state: RunState, record: StageRecord) -> None:
        skip = _skip_reason(stage, state)
        if skip:
            record.status, record.notes = "skipped", [skip]
            return
        problems = stage.validate(state)
        if problems:
            record.status, record.error = "failed", "; ".join(problems)
            raise StageError(f"stage '{stage.name}' cannot run: " + "; ".join(problems))
        record.status, t0 = "running", time.perf_counter()
        try:
            artifacts = stage.execute(state) or {}
            record.seconds = time.perf_counter() - t0
            record.artifacts = {k: v for k, v in artifacts.items() if _recordable(v)}
            record.notes += [n for n in (stage.verify(state, artifacts) or []) if isinstance(n, str)]
            record.status = "done"
        except Exception as exc:
            record.seconds, record.status, record.error = time.perf_counter() - t0, "failed", \
                f"{type(exc).__name__}: {exc}"
            state.save()
            raise PipelineFailed(stage.name, record.error, state) from exc

    def _apply_policy(self, result: PipelineResult, *, strict: bool | None = None) -> bool:
        """Set the run's status from the gates and decide whether this run must not be used."""
        cfg = self.config.verification
        failed = result.failed_gates
        unevaluated = result.unevaluated_gates
        result.status = "failed" if any(s.status == "failed" for s in result.stages) else "completed"
        blocking = bool(failed) or (cfg.mode == "strict" and bool(unevaluated))
        if strict is not None:
            blocking = bool(failed) if not strict else (bool(failed) or bool(unevaluated))
        if cfg.mode == "report":
            result.status = "completed"
            return False
        if blocking and cfg.fail_policy == "block":
            result.status = "rejected"
            return True
        if blocking:
            result.status = "completed-with-findings"
        return False

    def _build_result(self, state: RunState, *, seconds: float) -> PipelineResult:
        cfg = state.cfg
        verification = list(state.verification)
        result = PipelineResult(
            run_id=state.run_id, name=cfg.name, problem=_problem_summary(state),
            dataset={"n": getattr(state.dataset, "n", None),
                     "inputs": list(getattr(state.dataset, "input_names", []) or []),
                     "outputs": list(getattr(state.dataset, "output_names", []) or [])} if state.dataset else None,
            model=_model_name(state.model),
            metrics={k: v for k, v in state.metrics.items() if k not in ("pinn_report",)},
            verification=verification, validated_domain=state.validated_domain, uncertainty=state.uncertainty,
            stages=state.ordered_records(), artifacts=state.artifacts(), execution_time=seconds)
        for gate in verification:
            result.limitations += [f"{gate.gate}: {x}" for x in gate.limitations]
            result.recommendations += [f"{gate.gate}: {x}" for x in gate.recommendations]
        result.limitations += [n for n in state.notes if n not in result.limitations]
        return result

    def _write_report(self, state: RunState, result: PipelineResult) -> Path:
        path = state.path("report.json")
        md = state.path("report.md")
        md.write_text(result.to_markdown(), encoding="utf-8")
        result.artifacts["report"] = str(md)
        result.artifacts["report_json"] = str(path)
        return path

    # -------------------------------------------------------------------------------------------- queries
    def status(self) -> dict[str, Any]:
        """What the run's folder says happened, without re-running anything."""
        state = self.state
        if state is None:
            saved = RunState.load(Path(self.config.execution.out_dir) / self.config.name)
            if saved is None:
                return {"name": self.config.name, "status": "not started", "stages": []}
            return {"name": self.config.name, "status": saved.get("saved", ""), "run_id": saved.get("run_id"),
                    "stages": saved.get("stages", [])}
        return {"name": self.config.name, "run_id": state.run_id,
                "stages": [r.to_dict() for r in state.ordered_records()]}

    def history(self) -> list[dict[str, Any]]:
        """The stage table of this run (or of the last one on disk)."""
        return self.status().get("stages", [])

    def get_artifacts(self) -> dict[str, str]:
        if self.state is not None:
            return self.state.artifacts()
        saved = RunState.load(Path(self.config.execution.out_dir) / self.config.name) or {}
        return dict(saved.get("artifacts") or {})

    def report(self, path: str | Path | None = None) -> str:
        """The markdown report of the last run (written again to ``path`` when given)."""
        if self.result is None:
            raise RuntimeError("no run has finished: call run() first")
        text = self.result.to_markdown()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------------------------------ inference
    def predict(self, **inputs: Any) -> dict[str, Any]:
        """Query the trained surrogate: ``pipeline.predict(nu=0.02)`` for a parametric model, or
        ``pipeline.predict(x=0.3, t=0.5)`` for a field model. Returns the outputs and whether the query is inside
        the validated domain."""
        model = (self.state.model_object if self.state else None)
        if model is None:
            raise RuntimeError("no trained model: call run() first")
        if hasattr(model, "predict") and hasattr(model, "input_names"):        # a surrogate study's model
            out = model.predict(inputs, ood_policy="warn")
            return {"outputs": out["outputs"], "ood": out.get("ood"), "n": out.get("n")}
        coords = list(self.state.bundle.coords)
        missing = [c for c in coords if c not in inputs]
        if missing:
            raise ValueError(f"the model takes {coords}; missing {missing}")
        import numpy as np
        point = np.asarray([[float(inputs[c]) for c in coords]])
        y = model.predict(point)
        return {"outputs": {f: float(y[0, i]) for i, f in enumerate(self.state.bundle.fields)},
                "query": {c: float(point[0, i]) for i, c in enumerate(coords)}}

    def predict_array(self, X: Any) -> Any:
        """The model as an array function: ``(N, n_inputs) -> (N, n_outputs)``.

        This is the surface every external consumer uses — the checker's ``predict.py`` wrapper, an ONNX parity
        test, a colleague's script — because it needs no names and no knowledge of which backend trained the model.
        The column order is the one the run declared: ``data.inputs``/``data.outputs`` for a parametric surrogate,
        the coordinates and fields for a physics-only model.
        """
        return predict_array_with(self.state, X)

    def deploy(self, *, target: str | None = None, host: str | None = None, port: int | None = None,
               api_key: str | None = None):
        """Serve the trained model as a ``pinneapple_tools.model_services`` service, with this run's trust.

        ``deployment.target="local"`` returns the service and the registry; ``"http"`` also starts the server on a
        background thread (``.url``, ``.stop()``). The trust function enforces the validated domain and reports the
        gates of the run, so the endpoint answers with the same verdict ``predict_verified`` gives.
        """
        from .deploy import deploy as _deploy
        return _deploy(self, target=target, host=host, port=port, api_key=api_key)

    def predict_verified(self, **inputs: Any) -> dict[str, Any]:
        """``predict`` plus the VeriPhysics decision for that query: inside the validated domain, and the status of
        the gates that license using the model at all."""
        if self.result is None:
            raise RuntimeError("no run has finished: call run() first")
        prediction = self.predict(**inputs)
        failing = [g for g in self.result.verification if g.status == FAIL]
        ood = prediction.get("ood")
        outside = bool(ood) and any(row.get("ood") for row in ood if isinstance(row, dict))
        decision = "reject" if (failing or outside) else ("review" if self.result.status != "completed" else "accept")
        return {**prediction, "decision": decision,
                "reasons": [f"{g.gate} failed: {'; '.join(g.limitations)}" for g in failing]
                + (["the query is outside the validated domain"] if outside else []),
                "validated_domain": self.result.validated_domain,
                "gates": {g.gate: g.status for g in self.result.verification}}


def predict_array_with(state, X: Any) -> Any:
    """``(N, n_inputs) -> (N, n_outputs)`` with the model of a :class:`~pinneapple_pipeline.state.RunState`.

    Module level because ``verification/predict.py`` (what ``pinneapple check`` loads) and a live pipeline must
    answer with the same code path, whether the model came from this process or was reloaded from the folder.
    """
    import numpy as np

    model = (state.model_object if state is not None else None)
    if model is None:
        raise RuntimeError("no trained model: call run() first (or from_run() on a finished run folder)")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if hasattr(model, "input_names"):                           # a surrogate study's model
        if X.shape[1] != len(model.input_names):
            raise ValueError(f"the model takes {len(model.input_names)} input(s) {list(model.input_names)}, got "
                             f"{X.shape[1]} column(s)")
        Y = np.asarray(model.predict_array(X), dtype=float)
    else:                                                       # a physics-only model: fields at points
        coords = list(state.bundle.coords)
        if X.shape[1] != len(coords):
            raise ValueError(f"the model takes {len(coords)} coordinate(s) {coords}, got {X.shape[1]} column(s)")
        Y = np.asarray(model.predict(X), dtype=float)
    return Y.reshape(len(X), -1)


_DOES = {
    "problem": "build the problem and check it against the configuration",
    "data": "design the cases and solve them with the reference solver",
    "preprocess": "build the dataset and the train/validation/test split",
    "model": "resolve the architecture and check it fits the problem",
    "training": "train the model (a PINN on the equations, or a study over the dataset)",
    "export": "write the checkpoint, the portable formats and the model card",
    "verification": "run the VeriPhysics gates and write the report",
}


def _problem_summary(state: RunState) -> dict[str, Any]:
    """The problem as the result reports it (small, JSON-safe, and the same whatever the backend)."""
    from . import adapters
    if state.bundle is None:
        return {}
    description = adapters.problem_description(state.bundle)
    pde = description.get("pde") or {}
    return {"name": description.get("name"), "backend": description.get("backend"),
            "coords": description.get("coords"), "fields": description.get("fields"),
            "pde": pde.get("kind") if isinstance(pde, dict) else str(pde),
            "parameters": list((description.get("parameters") or {}).keys()),
            "conditions": len(description.get("conditions") or [])}


def _model_name(model: Any) -> str | None:
    if model is None:
        return None
    if hasattr(model, "best"):                      # a StudyResult: the method it chose
        return str(model.best)
    return type(getattr(model, "model", model)).__name__


def _recordable(value: Any) -> bool:
    """Whether a stage artefact can go in the record (and the report): plain JSON, not a model or an array."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_recordable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _recordable(v) for k, v in value.items())
    return False


def _stage_by_name(stages: list[PipelineStage], name: str) -> PipelineStage:
    for stage in stages:
        if stage.name == name:
            return stage
    raise KeyError(f"no stage named {name!r}; the pipeline has {[s.name for s in stages]}")


def _stages_until(stages: list[PipelineStage], until: str | None) -> list[PipelineStage]:
    if until is None:
        return list(stages)
    names = [s.name for s in stages]
    if until not in names:
        raise KeyError(f"no stage named {until!r}; the pipeline has {names}")
    index = names.index(until)
    return list(stages)[:index + 1]


def _skip_reason(stage: PipelineStage, state: RunState) -> str:
    try:
        return stage.skip_reason(state)
    except Exception:                                            # a skip check must not break the plan
        return ""
