"""The quality pipeline: stages A-G wired into one reproducible run.

Stage map (the letters are the specification's, kept so a report can be
read against it):

===  ========================================================  ============================
A    :class:`~pinneapple_llm.qa.spec.PhysicalSpecification`      supplied by the caller
B    artifact generation                                       supplied (LLM or otherwise)
C    static/structural validation                              :mod:`pinneapple_llm.qa.static`
D    controlled execution                                      :mod:`pinneapple_llm.qa.sandbox`
E    scientific/numeric validation                             :mod:`pinneapple_llm.qa.validators`
F    independent reference comparison                          :mod:`pinneapple_llm.qa.references`
G    diagnosis and bounded repair                              :mod:`pinneapple_llm.qa.repair`
===  ========================================================  ============================

The pipeline never generates the artifact itself. Generation -- from an
LLM or from code -- is stage B and stays outside, so "the model wrote it"
can never be confused with "the pipeline checked it". What the pipeline
does is run the checks, record exactly what passed, what failed and what
was never run, and compute the quality level from that record.

Repair and stale data
---------------------
:class:`~pinneapple_llm.qa.repair.RepairLoop` is correct on its own terms:
every attempt re-runs every validator it is given, the specification's
criteria hash is re-checked, and repeated diffs stop the loop. The
pipeline's built-in repair step used to be narrower than that. It
re-validated the candidate source statically and re-ran the scientific
validators against the *same* ``ValidationContext`` (same meshes, fields,
times, sensors), so a repair that fixed the code but would produce
different data was judged against stale data -- and a repair whose output
*is* the data could never be accepted by re-validating old numbers.

``rebuild_from_source=`` closes that gap. It is a callable that takes the
repaired source, runs it (outside the pipeline, typically in the sandbox)
and returns a fresh bundle: ``meshes``, ``fields``, ``field_units``,
``times``, ``sensors``, ``payload``, either as a mapping or as an object
with those attributes. When it is configured every repair attempt runs
static validation, execution in the sandbox (when one is configured),
bundle rebuild, and then the scientific validators against the *new*
bundle; each attempt records ``bundle_source`` (``"fresh"`` or
``"stale"``) and ``rebuilt`` so the report says which data the verdict was
reached on. Without it nothing changes: the scientific validators see the
caller's bundle, and the report says so as a limitation. Stage F (reference
comparison) is still evaluated once, against the caller's bundle -- it is
not re-run per attempt.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from . import levels as _levels
from .generation import GeneratedArtifact, check_claims
from .references import ReferenceData, compare
from .repair import RepairAttempt, RepairLoop, RepairOutcome
from .report import (ArtifactRecord, Finding, QualityReport, Status, environment_snapshot)
from .sandbox import SandboxExecutor
from .spec import PhysicalSpecification
from .static import StaticPolicy, validate_json_artifact, validate_python_artifact
from .validators import (ValidationContext, SensorReading, ValidatorResult, available,
                         run as run_validator)

#: Validators that need a trained model are opt-in; everything else runs when
#: its data is present.
_DEFAULT_VALIDATORS = ("units", "numeric", "physics", "geometry", "temporal", "sensors")


@dataclass
class PipelineConfig:
    validators: Sequence[str] = _DEFAULT_VALIDATORS
    run_artifacts: bool = False
    source_name: str = "artifact.py"
    static_policy: StaticPolicy = field(default_factory=StaticPolicy)
    repair_attempts: int = 0


class QualityPipeline:
    """Stages A-G for one artifact: check, record, bound the repair.

    ``rebuild_from_source`` exists because a repair can be right about the
    code and wrong about the data. When the artifact's output *is* the
    data, "re-run the validations after every correction" has to include
    re-execution: re-validating the corrected source against the bundle it
    has not yet produced judges the fix on stale numbers. Pass a callable
    ``rebuild_from_source(source: str) -> bundle`` to have every repair
    attempt execute the candidate in the sandbox and rebuild the bundle
    before the scientific validators run. The bundle is a mapping (or an
    object) with ``meshes``, ``fields``, ``field_units``, ``times``,
    ``sensors`` and ``payload``; a missing key raises :class:`ValueError`
    in the caller's face and is recorded as ``rebuild_error`` on the
    attempt by the pipeline. The pipeline does not execute or interpret
    that callable itself: it trusts the bundle it returns as the
    artifact's output, and the report records it as such.
    """

    def __init__(self, spec: PhysicalSpecification, *,
                 validators: Sequence[str] | None = None,
                 sandbox: SandboxExecutor | None = None,
                 run_artifacts: bool = False,
                 static_policy: StaticPolicy | None = None,
                 reference: ReferenceData | None = None,
                 reference_criteria: dict[str, float] | None = None,
                 reference_subset: str = "validation",
                 reference_field: str | None = None,
                 operational_evidence: dict[str, Any] | None = None,
                 max_repair_attempts: int = 0,
                 repair: Any = None,
                 rebuild_from_source: Callable[[str], Any] | None = None,
                 source_name: str = "artifact.py",
                 required_validations: Sequence[str] | None = None):
        self.spec = spec
        self.validators = tuple(validators) if validators is not None else _DEFAULT_VALIDATORS
        self.sandbox = sandbox
        self.run_artifacts = run_artifacts
        self.static_policy = static_policy or StaticPolicy()
        self.reference = reference
        self.reference_criteria = reference_criteria or {}
        self.reference_subset = reference_subset
        self.reference_field = reference_field
        self.operational_evidence = operational_evidence
        self.max_repair_attempts = max_repair_attempts
        self.repair = repair
        self.rebuild_from_source = rebuild_from_source
        self.source_name = source_name
        self.required_validations = tuple(
            required_validations if required_validations is not None
            else (spec.required_validations or ())
        )

        unknown = [v for v in self.validators if v not in available()]
        if unknown:
            raise ValueError(f"unknown validator(s) {unknown}; available: {available()}")

    # ------------------------------------------------------------------
    def run(self, *, source: str | None = None,
            generated: GeneratedArtifact | None = None,
            meshes: dict[str, Any] | None = None,
            fields: dict[str, Any] | None = None,
            field_units: dict[str, str] | None = None,
            times: Iterable[float] | None = None,
            time_unit: str = "s",
            sensors: Sequence[Any] | None = None,
            payload: dict[str, Any] | None = None,
            json_schema: dict[str, Any] | None = None,
            model: Any = None,
            extra: dict[str, Any] | None = None,
            exporter: Any | None = None,
            output_dir: str | None = None,
            output_manifest: str | None = None) -> QualityReport:
        report = QualityReport(
            spec_id=self.spec.spec_id,
            spec_hash=self.spec.content_hash(),
            criteria_hash=self.spec.criteria_hash(),
            provider=str(self.spec.numerical_method.get("provider", "")),
            model=str(self.spec.numerical_method.get("model", "")),
            environment=environment_snapshot(),
            operational_evidence=self.operational_evidence,
            assumptions=list(self.spec.assumptions),
            missing_information=self.spec.missing_information(),
        )

        self._record_spec_step(report)

        if generated is not None:
            self._record_generation_step(report, generated)
            if source is None:
                source = generated.code

        if source is not None:
            self._record_static_steps(report, source)
            report.artifact(ArtifactRecord.from_value(self.source_name, source, kind="python_source"))
        else:
            for step in ("static_syntax", "static_imports", "static_security",
                         "static_api_contracts", "static_signatures"):
                report.record(step, Status.NOT_APPLICABLE, [Finding(
                    id=f"{step}.not_applicable", status=Status.NOT_APPLICABLE, severity="warning",
                    message="this run supplied no generated source code, so there was nothing to parse",
                    justification="recorded explicitly rather than omitted, so the report cannot be "
                                "read as 'these checks passed'")])

        if payload is not None:
            self._record_payload_step(report, payload, json_schema)

        if meshes:
            for name, mesh in meshes.items():
                vertices, faces = mesh
                report.artifact(ArtifactRecord.from_value(f"mesh:{name}", {
                    "vertices": np.asarray(vertices).tolist(),
                    "faces": np.asarray(faces).tolist(),
                }, kind="mesh", part=name, vertices=int(len(vertices)), faces=int(len(faces))))
        for name, arr in (fields or {}).items():
            report.artifact(ArtifactRecord.from_value(f"field:{name}", np.asarray(arr).tolist(),
                                                      kind="field", unit=(field_units or {}).get(name, "")))

        ctx = ValidationContext(
            spec=self.spec, meshes=dict(meshes or {}),
            fields={k: np.asarray(v) for k, v in (fields or {}).items()},
            field_units=dict(field_units or {}),
            times=None if times is None else np.asarray(times, dtype=float),
            time_unit=time_unit,
            sensors=[_as_reading(s) for s in (sensors or [])],
            payload=dict(payload or {}), model=model, extra=dict(extra or {}),
        )

        if source is not None and self.run_artifacts:
            self._record_execution_step(report, source, extra or {})

        if exporter is not None:
            self._record_export_step(report, exporter, output_dir)

        payload_results = self._record_validator_steps(report, ctx)

        if self.reference is not None:
            self._record_reference_step(report, payload_results)

        if source is not None and self.max_repair_attempts and self.repair is not None:
            self._record_repair_step(report, source, ctx, extra or {})

        level, reasons = _levels.evaluate_level(report, self.spec)
        report.quality_level = level
        report.level_reasons = reasons
        report.limitations.extend(_levels.blocking_reasons(report))
        if not report.limitations:
            report.limitations.append("no blocking limitation recorded -- see individual step findings")
        report.reproduce = _reproduce_hint(self.spec, report)
        if output_manifest:
            report.write(output_manifest)
        return report

    # ------------------------------------------------------------------
    def _record_spec_step(self, report: QualityReport) -> None:
        started = time.monotonic()
        problems = self.spec.validate()
        findings = []
        for problem in problems:
            findings.append(Finding(id="spec.structure", status=Status.FAIL, severity="error",
                                    message=problem))
        missing = self.spec.missing_information()
        for item in missing:
            findings.append(Finding(
                id="spec.missing", status=Status.NOT_RUN, severity="warning",
                message=f"not specified: {item}",
                justification="left absent on purpose: the pipeline does not invent values to fill gaps"))
        status = Status.FAIL if problems else Status.PASS
        report.record("spec", status, findings, duration_s=time.monotonic() - started,
                      spec_id=self.spec.spec_id, missing=missing)

    def _record_generation_step(self, report: QualityReport,
                                generated: GeneratedArtifact) -> None:
        """Record the model's structured *claims* -- verified as claims, not as truth."""
        findings = check_claims(generated, self.spec)
        status = Status.FAIL if any(f.blocks for f in findings) else Status.PASS
        report.provider = generated.provider or report.provider
        report.model = generated.model or report.model
        report.record("generation", status, findings,
                      duration_s=float(generated.elapsed_s or 0.0),
                      provider=generated.provider, model=generated.model,
                      declared_dependencies=generated.dependencies,
                      declared_units=generated.units,
                      declared_equations=generated.equations,
                      declared_limitations=generated.limitations)
        report.artifact(ArtifactRecord.from_value(
            "generation_envelope",
            {k: v for k, v in generated.to_dict().items() if k != "code_sha256_source"},
            kind="generation_metadata"))

    def _record_static_steps(self, report: QualityReport, source: str) -> None:
        started = time.monotonic()
        result = validate_python_artifact(source, policy=self.static_policy,
                                          filename=self.source_name)
        elapsed = time.monotonic() - started
        groups = {
            "static_syntax": "static.syntax",
            "static_imports": "static.import",
            "static_security": "static.security",
            "static_api_contracts": "static.api_contracts",
            "static_signatures": "static.signatures",
        }
        for step, prefix in groups.items():
            findings = [f for f in result.findings if f.id.startswith(prefix)]
            if not findings:
                continue
            status = Status.FAIL if any(f.blocks for f in findings) else Status.PASS
            report.record(step, status, findings, duration_s=elapsed)

    def _record_payload_step(self, report: QualityReport, payload: dict[str, Any],
                             schema: dict[str, Any] | None) -> None:
        if schema is None:
            return
        started = time.monotonic()
        result = validate_json_artifact(payload, schema, name="payload")
        status = result.status
        report.record("static_schema", status, result.findings,
                      duration_s=time.monotonic() - started)

    def _record_execution_step(self, report: QualityReport, source: str,
                               extra: dict[str, Any]) -> None:
        started = time.monotonic()
        executor = self.sandbox or SandboxExecutor(allow_execution=self.run_artifacts)
        result = executor.run_python(source, files=extra.get("sandbox_files"),
                                     output_dir=extra.get("sandbox_output_dir"))
        findings = []
        for violation in result.violations:
            findings.append(Finding(
                id="execution.violation", status=Status.FAIL, severity="error",
                message=f"sandbox policy violation: {violation}",
                justification="an artifact that attempts to leave its sandbox is not approved"))
        if result.timed_out:
            findings.append(Finding(id="execution.timeout", status=Status.FAIL, severity="error",
                                    message=result.reason))
        elif result.exit_code not in (0, None):
            findings.append(Finding(
                id="execution.exit", status=Status.FAIL, severity="error",
                message=f"artifact exited {result.exit_code}",
                evidence={"stdout": result.stdout[-2000:], "stderr": result.stderr[-4000:]}))
        report.record("execution", result.status, findings,
                      duration_s=time.monotonic() - started,
                      isolation=result.isolation.value, timed_out=result.timed_out,
                      exit_code=result.exit_code, produced=[p.name for p in result.produced],
                      limitations=result.limitations)
        for record in result.produced:
            report.artifact(record)
        report.limitations.extend(result.limitations)

    def _record_export_step(self, report: QualityReport, exporter: Any,
                            output_dir: str | None) -> None:
        """Run the caller's export step; a failure is recorded, never swallowed.

        ``exporter(output_dir) -> path`` must return the path it produced.
        The check then confirms the file exists and is non-empty, because
        "export returned a path" and "a file was written" are different
        claims and only the second one matters.
        """
        started = time.monotonic()
        try:
            produced = exporter(output_dir)
        except Exception as exc:
            report.record("export", Status.FAIL, [Finding(
                id="export.failed", status=Status.FAIL, severity="error",
                message=f"export raised {type(exc).__name__}: {exc}",
                evidence={"output_dir": output_dir},
                justification="an export that raised produced no artifact; reporting it as a "
                            "success would be a false claim about a file that does not exist")],
                duration_s=time.monotonic() - started, output_dir=output_dir)
            return

        paths = [produced] if isinstance(produced, str) else list(produced or [])
        findings: list[Finding] = []
        status = Status.PASS
        for path in paths:
            if not path or not os.path.exists(path):
                status = Status.FAIL
                findings.append(Finding(id="export.missing", status=Status.FAIL, severity="error",
                                        message=f"export reported '{path}' but no such file exists"))
                continue
            if os.path.getsize(path) == 0:
                status = Status.FAIL
                findings.append(Finding(id="export.empty", status=Status.FAIL, severity="error",
                                        message=f"exported file '{path}' is empty"))
                continue
            report.artifact(ArtifactRecord.from_file(os.path.basename(path), path, kind="export"))
        if not paths:
            status = Status.FAIL
            findings.append(Finding(id="export.none", status=Status.FAIL, severity="error",
                                    message="export returned no path"))
        report.record("export", status, findings, duration_s=time.monotonic() - started,
                      output_dir=output_dir, files=[os.path.basename(p) for p in paths])

    def _record_validator_steps(self, report: QualityReport,
                                ctx: ValidationContext) -> dict[str, Any]:
        results = {}
        for name in self.validators:
            result = run_validator(name, ctx)
            results[name] = result
            findings = result.to_findings()
            findings.extend(Finding(id=f"validator.{name}.limitation", status=Status.NOT_APPLICABLE,
                                    severity="info", message=lim, evidence={"validator": name})
                            for lim in result.limitations)
            report.record(f"validator:{name}", result.status, findings,
                          duration_s=result.duration_s, evidence=result.evidence)
        return {"results": results, "ctx": ctx}

    def _record_reference_step(self, report: QualityReport, payload: dict[str, Any]) -> None:
        ctx: ValidationContext = payload["ctx"]
        try:
            predictions = _predictions_for_reference(ctx, self.reference, self.reference_field)
        except (KeyError, ValueError) as exc:
            report.record("reference_comparison", Status.BLOCKED,
                          [Finding(id="reference.no_predictions", status=Status.BLOCKED,
                                   severity="error", message=str(exc))])
            return
        if predictions is None:
            report.record("reference_comparison", Status.BLOCKED,
                          [Finding(id="reference.no_predictions", status=Status.BLOCKED, severity="error",
                                   message="no prediction array could be matched to the reference")],
                          )
            return
        result = compare(predictions, self.reference, self.reference_criteria,
                         subset=self.reference_subset)
        report.record("reference_comparison", result.status, result.to_findings(),
                      duration_s=0.0,
                      reference_origin=self.reference.origin,
                      reference_name=self.reference.name,
                      reference_version=self.reference.version,
                      in_sample=self.reference.used_for_calibration,
                      metrics=result.evidence.get("metrics", {}))
        report.limitations.extend(result.limitations)

    def _record_repair_step(self, report: QualityReport, source: str,
                            ctx: ValidationContext,
                            extra: dict[str, Any] | None = None) -> None:
        """Run stage G, re-validating each candidate.

        With ``rebuild_from_source`` configured, each attempt evaluates the
        candidate in this order: static validation, execution in the sandbox
        (only when one was supplied), bundle rebuild, scientific validators
        against the *rebuilt* bundle. A rebuild that raises -- including the
        :class:`ValueError` for a malformed bundle -- fails the attempt and
        the loop moves on; it never escapes as an exception.

        Both paths record, per attempt, which data the verdict came from:
        ``bundle_source`` is ``"fresh"`` only when the bundle was rebuilt
        from the candidate source during that attempt, ``"stale"`` when the
        scientific validators saw the caller's bundle.
        """
        extra = dict(extra or {})
        required = list(self.required_validations or tuple(self.validators))
        rebuild_from_source = self.rebuild_from_source
        if rebuild_from_source is not None and "rebuild" not in required:
            # a candidate whose bundle could not be rebuilt has not been
            # validated at all, whatever the other validators say about
            # someone else's data.
            required.append("rebuild")

        per_attempt: dict[int, dict[str, Any]] = {}
        evaluated = {"n": 0}

        def evaluate(candidate: Any) -> dict[str, Any]:
            evaluated["n"] += 1
            info: dict[str, Any] = {"bundle_source": "stale", "rebuilt": False,
                                    "rebuild_error": "", "execution_status": ""}
            per_attempt[evaluated["n"]] = info

            source_text = str(candidate)
            static = validate_python_artifact(source_text, policy=self.static_policy,
                                              filename=self.source_name)
            results: dict[str, ValidatorResult] = {"static": ValidatorResult(
                name="static",
                status=Status.FAIL if any(f.blocks for f in static.findings) else Status.PASS,
                findings=static.findings,
            )}

            if rebuild_from_source is None:
                for name in self.validators:
                    results[name] = run_validator(name, ctx)
                return results

            if any(f.blocks for f in static.findings):
                results["rebuild"] = _not_run(
                    "rebuild", "the candidate failed static validation, so it was never executed "
                               "and no bundle was rebuilt")
                _skip_validators(results, self.validators, "no bundle was rebuilt for this candidate")
                return results

            if self.sandbox is not None:
                execution = self.sandbox.run_python(
                    source_text, files=extra.get("sandbox_files"),
                    output_dir=extra.get("sandbox_output_dir"))
                info["execution_status"] = execution.status.value
                findings: list[Finding] = []
                for violation in execution.violations:
                    findings.append(Finding(
                        id="execution.violation", status=Status.FAIL, severity="error",
                        message=f"sandbox policy violation: {violation}",
                        justification="an artifact that attempts to leave its sandbox is not approved"))
                if execution.timed_out:
                    findings.append(Finding(id="execution.timeout", status=Status.FAIL, severity="error",
                                            message=execution.reason))
                elif execution.exit_code not in (0, None):
                    findings.append(Finding(
                        id="execution.exit", status=Status.FAIL, severity="error",
                        message=f"repaired artifact exited {execution.exit_code}",
                        evidence={"stdout": execution.stdout[-2000:],
                                  "stderr": execution.stderr[-4000:]}))
                results["execution"] = ValidatorResult(
                    name="execution", status=execution.status, findings=findings,
                    evidence={"isolation": execution.isolation.value,
                              "exit_code": execution.exit_code,
                              "timed_out": execution.timed_out})
                if execution.status is Status.FAIL:
                    results["rebuild"] = _not_run(
                        "rebuild", "execution failed, so there was no output to rebuild a bundle from")
                    _skip_validators(results, self.validators,
                                     "execution of the candidate failed, so no data was rebuilt")
                    return results

            try:
                fresh = _context_from_bundle(rebuild_from_source(source_text),
                                             base=ctx, spec=self.spec)
            except Exception as exc:
                info["rebuild_error"] = f"{type(exc).__name__}: {exc}"
                results["rebuild"] = ValidatorResult(
                    name="rebuild", status=Status.FAIL, findings=[Finding(
                        id="rebuild.failed", status=Status.FAIL, severity="error",
                        message=f"rebuilding the bundle from the repaired source raised "
                                f"{type(exc).__name__}: {exc}",
                        justification="without the artifact's own output there is nothing to "
                                      "validate; judging the repair against the previous bundle "
                                      "would approve a fix on stale data")],
                    )
                _skip_validators(results, self.validators,
                                 "the bundle could not be rebuilt from this candidate")
                return results

            info["bundle_source"] = "fresh"
            info["rebuilt"] = True
            results["rebuild"] = ValidatorResult(
                name="rebuild", status=Status.PASS,
                evidence={"fields": sorted(fresh.fields), "meshes": sorted(fresh.meshes),
                          "time_steps": 0 if fresh.times is None else int(np.size(fresh.times)),
                          "sensors": len(fresh.sensors)})
            for name in self.validators:
                results[name] = run_validator(name, fresh)
            return results

        def annotate(record: RepairAttempt) -> None:
            info = per_attempt.get(record.attempt, {})
            record.bundle_source = str(info.get("bundle_source", "stale"))
            record.rebuilt = bool(info.get("rebuilt", False))
            record.rebuild_error = str(info.get("rebuild_error", ""))
            record.execution_status = str(info.get("execution_status", ""))

        loop = RepairLoop(evaluate=evaluate, repair=self.repair, spec=self.spec,
                          required=tuple(required) if required else ("static",),
                          max_attempts=self.max_repair_attempts, on_attempt=annotate)
        outcome: RepairOutcome = loop.run(source)
        findings = []
        for attempt in outcome.attempts:
            findings.append(Finding(
                id=f"repair.attempt.{attempt.attempt}",
                status=Status.PASS if attempt.accepted else Status.FAIL,
                severity="info" if attempt.accepted else "error",
                message=(f"attempt {attempt.attempt}: {attempt.notes} "
                         f"[bundle: {attempt.bundle_source}]"),
                evidence={**attempt.to_dict(), "diff": attempt.diff[:4000]}))
        report.record("repair", Status.PASS if outcome.accepted else Status.FAIL, findings,
                      attempts=len(outcome.attempts), stopped_because=outcome.stopped_because,
                      rebuild_from_source=rebuild_from_source is not None,
                      bundle_sources=[a.bundle_source for a in outcome.attempts])
        if rebuild_from_source is None:
            report.limitations.append(
                "automatic repair re-validates the candidate source statically and re-runs the "
                "scientific validators against the SAME artifact bundle; it does not re-execute the "
                "repaired source to rebuild that bundle (use RepairLoop with your own evaluate() or "
                "QualityPipeline(rebuild_from_source=...) when the repair is meant to change the "
                "data, not just the code)")
        else:
            report.limitations.append(
                "automatic repair rebuilt the artifact bundle with the caller's "
                "rebuild_from_source callable; the pipeline trusts that bundle as the artifact's "
                "output and does not itself check how it was produced (execution is only attempted "
                "when a sandbox was supplied)")
        report.artifact(ArtifactRecord.from_value(
            f"{self.source_name}:repaired", outcome.artifact, kind="python_source"))
        self.repaired_artifact = outcome.artifact
        self.repair_outcome = outcome


# ---------------------------------------------------------------------------
#: Keys a bundle must carry for ``rebuild_from_source`` to be usable.
_BUNDLE_KEYS = ("meshes", "fields", "field_units", "times", "sensors", "payload")


def _bundle_member(bundle: Any, key: str) -> Any:
    """Read one bundle member, refusing loudly when it is not there."""
    if isinstance(bundle, dict):
        if key in bundle:
            return bundle[key]
    elif hasattr(bundle, key):
        return getattr(bundle, key)
    raise ValueError(
        f"rebuild_from_source returned a bundle without '{key}'; a bundle is a dict (or an "
        f"object) carrying {', '.join(_BUNDLE_KEYS)} -- the pipeline validates what the repaired "
        "source produced, so it will not guess the missing part from the previous bundle")


def _context_from_bundle(bundle: Any, *, base: ValidationContext,
                         spec: PhysicalSpecification) -> ValidationContext:
    """Turn a rebuilt bundle into a fresh :class:`ValidationContext`.

    The bundle supplies what execution produces (meshes, fields, units,
    times, sensors, payload). Everything that belongs to the specification
    of the run rather than its output -- the spec itself, the time unit,
    the model and the caller's ``extra`` -- is carried over from ``base``.
    """
    meshes_raw = _bundle_member(bundle, "meshes") or {}
    fields_raw = _bundle_member(bundle, "fields") or {}
    units_raw = _bundle_member(bundle, "field_units") or {}
    times_raw = _bundle_member(bundle, "times")
    sensors_raw = _bundle_member(bundle, "sensors") or []
    payload_raw = _bundle_member(bundle, "payload") or {}

    meshes: dict[str, Any] = {}
    for name, mesh in dict(meshes_raw).items():
        vertices, faces = mesh
        meshes[name] = (np.asarray(vertices), np.asarray(faces))
    return ValidationContext(
        spec=spec, meshes=meshes,
        fields={name: np.asarray(value) for name, value in dict(fields_raw).items()},
        field_units={str(name): str(unit) for name, unit in dict(units_raw).items()},
        times=None if times_raw is None else np.asarray(times_raw, dtype=float),
        time_unit=base.time_unit,
        sensors=[_as_reading(sensor) for sensor in sensors_raw],
        payload=dict(payload_raw), model=base.model, extra=dict(base.extra),
    )


def _not_run(name: str, message: str) -> ValidatorResult:
    return ValidatorResult(name=name, status=Status.NOT_RUN, findings=[
        Finding(id=f"{name}.not_run", status=Status.NOT_RUN, severity="warning", message=message)])


def _skip_validators(results: dict[str, ValidatorResult], names: Iterable[str],
                     reason: str) -> None:
    """Record that a validator was not run, instead of leaving it absent."""
    for name in names:
        results[name] = _not_run(name, f"{name}: not run this attempt -- {reason}")


def _as_reading(sensor: Any) -> SensorReading:
    if isinstance(sensor, SensorReading):
        return sensor
    if isinstance(sensor, dict):
        return SensorReading(
            id=sensor["id"], position=tuple(sensor["position"]), unit=sensor.get("unit", ""),
            quantity=sensor.get("quantity", ""), series=np.asarray(sensor["series"], dtype=float),
            holdout=bool(sensor.get("holdout", False)),
            position_unit=sensor.get("position_unit", "m"))
    raise TypeError(
        f"sensor must be a SensorReading or a dict, got {type(sensor).__name__}")


def _predictions_for_reference(ctx: ValidationContext, reference: ReferenceData,
                               field: str | None = None):
    if not ctx.fields:
        return None
    if field is not None:
        if field not in ctx.fields:
            raise KeyError(
                f"reference_field '{field}' is not one of the artifact's fields {sorted(ctx.fields)}")
        return np.asarray(ctx.fields[field])
    # Otherwise: a field whose name appears in the reference's name, else the
    # only field. Guessing among several fields would make the comparison
    # meaningless, so ambiguity is an error, not a silent pick.
    candidates = [name for name in ctx.fields if name and name in reference.name]
    if len(candidates) == 1:
        return np.asarray(ctx.fields[candidates[0]])
    if len(ctx.fields) == 1:
        return np.asarray(next(iter(ctx.fields.values())))
    raise ValueError(
        f"the artifact has several fields {sorted(ctx.fields)} and the reference does not name one; "
        "pass reference_field= to say which array is being compared")


def _reproduce_hint(spec: PhysicalSpecification, report: QualityReport) -> str:
    """A string a developer can follow to reproduce this run without the chat."""
    return (
        f"Re-run with the same specification (spec_id={spec.spec_id}, "
        f"hash={report.spec_hash[:16]}) and artifacts "
        f"{[a.name + ':' + a.sha256[:12] for a in report.artifacts]}; "
        "see docs/guides/pinneapple_llm_quality_pipeline.md ('Reproducing a run')."
    )
