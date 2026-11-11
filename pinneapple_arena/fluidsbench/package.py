"""Build and check the result directory a FluidsBench submission is made of.

A submission is a directory of five files whose hashes bind each other::

    submissions/<dataset-id>/<submission-id>/
      submission.json            model, method, dataset, split, hashes, aggregate metric values
      evaluation-evidence.json   the command, the release it scored against, the values it produced
      discretization.json        what the model took in and what it produced, and how it reached the support
      discretization/cases.jsonl one line per case: counts, mappings, coverage
      metrics/cases.json         per case and support: values, coverage, sufficient statistics

This module writes those files with the digests the contract expects and recomputes, from the package's own numbers, the
identities the maintainers' validator checks: every relative L2 against its sufficient statistics, every aggregate
against the macro-average of its case values, and ``overall_score`` against its published composite. What it cannot
check is a schema; that is the official validator's job and :func:`validate_package` runs it from a checkout when asked.

    from pinneapple_arena.fluidsbench import build_package, validate_package

    package = build_package("submissions/airfrans/my-model-v1", submission_id="my-model-v1", release=release,
                            scores=scores, split_id="full", ...)
    report = validate_package(package.root, contract=contract, run_official=True)
    print(report)                        # every check, then the official validator's own output

Facts only the submitter knows — architecture, parameters, licences, hardware, what the model took in — are arguments,
never defaults: this module does not invent a methodology record, an SPDX identifier or a compute measurement.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import Checkout, ContractError, DatasetContract, ScoreComposite, sha256_file
from .metrics import (
    REL_L2,
    SupportScore,
    aggregate,
    competition_rank,
    macro_average,
    relative_l2_from_statistics,
)
from .support import SupportRelease

__all__ = ["Package", "ValidationReport", "build_package", "validate_package", "load_package",
           "SUMMARY_SCHEMA"]

SUMMARY_SCHEMA = "pinneapple-fluidsbench-package/v1"
REQUIRED_FILES = ("submission.json", "evaluation-evidence.json", "discretization.json",
                  "discretization/cases.jsonl", "metrics/cases.json")


@dataclass(frozen=True)
class Package:
    """A written result package: where it is, what it declares, and the aggregate values it will be ranked on."""
    root: Path
    submission_id: str
    dataset_id: str
    split_id: str
    case_set_id: str
    metric_values: Mapping[str, float]
    overall_score: float | None
    files: Mapping[str, str] = field(default_factory=dict)

    def digests(self) -> dict[str, str]:
        return dict(self.files)


@dataclass(frozen=True)
class ValidationReport:
    """The independent checks this adapter can run on a package, and what the official validator said."""
    root: Path
    checked: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    official: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems and (self.official or {}).get("ok", True)

    def __str__(self) -> str:
        lines = [f"package {self.root}", f"  checks run: {len(self.checked)}"]
        lines += [f"  [ok]   {name}" for name in self.checked]
        lines += [f"  [fail] {problem}" for problem in self.problems]
        if self.official is not None:
            lines.append(f"  official validator: {'PASS' if self.official.get('ok') else 'FAIL'}")
            tail = (self.official.get("stdout") or self.official.get("stderr") or "").strip().splitlines()[-8:]
            lines += [f"    {line}" for line in tail]
        lines.append("PASS" if self.ok else "FAIL")
        return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _result_revision(submission_id: str, revision: Mapping[str, Any] | None) -> dict[str, Any]:
    if revision:
        return dict(revision)
    series, _, version = str(submission_id).rpartition("-v")
    number = int(version) if version.isdigit() else 1
    return {"series_id": series or submission_id, "version": number, "supersedes": None,
            "change_summary": "Initial published result."}


def _composite_values(composite: ScoreComposite | None, values: Mapping[str, float]
                      ) -> tuple[dict[str, float], list[str]]:
    """``overall_score`` and the group scores, when every component of the composite was measured."""
    if composite is None or not composite.components:
        return {}, ["no published composite was given: overall_score is not computed here"]
    missing = [c.metric_id for c in composite.components if c.metric_id not in values]
    if missing:
        return {}, [f"composite components not measured on this split: {missing}"]
    out = {composite.metric_id: composite.score(values)}
    out.update(composite.group_scores(values))
    return out, []


def build_package(dest: str | Path, *, submission_id: str, release: SupportRelease,
                  scores: Sequence[SupportScore], split_id: str, model: Mapping[str, Any],
                  submitter: Mapping[str, Any], methodology: Mapping[str, Any],
                  evaluation_command: str, discretization: Mapping[str, Any] | None = None,
                  reproducibility: Mapping[str, Any] | None = None, contract: DatasetContract | None = None,
                  dataset_version: str = "", split_sha256: str = "", aggregation: str = "per_geometry_then_macro_average",
                  result_revision: Mapping[str, Any] | None = None, submitted_at: str | None = None,
                  notes: str = "", source_output_ids: Mapping[str, str] | None = None) -> Package:
    """Write ``submission.json``, ``evaluation-evidence.json``, ``discretization.json``, its case lines and
    ``metrics/cases.json`` under ``dest``."""
    dest = Path(dest).expanduser().resolve()
    if not scores:
        raise ContractError("a package needs at least one scored case")
    dataset_id = release.dataset_id
    case_set_id = release.case_set_id
    values = _aggregate_by_metric(scores, aggregation=aggregation)
    composite_values, composite_notes = _composite_values(contract.composite if contract else None, values)
    metric_values = {**values, **composite_values}

    cases_payload, cases_lines = _case_payload(scores, source_output_ids)
    case_count = len(cases_payload)
    metrics_path = dest / "metrics" / "cases.json"
    _write_json(metrics_path, {
        "$schema": "https://fluidsbench.org/schemas/v3/case-metrics.schema.json",
        "schema_version": "1.0", "submission_id": submission_id, "dataset_id": dataset_id, "split_id": split_id,
        "case_set_id": case_set_id, "scoring_support_release_id": release.release_id,
        "scoring_support_manifest_sha256": release.manifest_sha256,
        "case_count": case_count, "cases": cases_payload, "metric_values": {k: v for k, v in values.items()},
    })

    case_lines_path = dest / "discretization" / "cases.jsonl"
    case_lines_path.parent.mkdir(parents=True, exist_ok=True)
    case_lines_path.write_text("\n".join(json.dumps(line) for line in cases_lines) + "\n", encoding="utf-8")

    spatial = dict(discretization or {})
    spatial.setdefault("inference", {})
    spatial["inference"].setdefault("mappings", [])
    spatial.update({
        "$schema": "https://fluidsbench.org/schemas/v3/discretization.schema.json", "schema_version": "1.0",
        "submission_id": submission_id, "dataset_id": dataset_id, "split_id": split_id,
        "scoring_support_release_id": release.release_id,
        "scoring_support_manifest_sha256": release.manifest_sha256,
        "case_manifest": {"format": "jsonl", "file": "discretization/cases.jsonl",
                          "sha256": sha256_file(case_lines_path), "case_count": case_count},
    })
    if notes:
        spatial.setdefault("notes", notes)
    discretization_path = dest / "discretization.json"
    _write_json(discretization_path, spatial)

    evidence = {
        "$schema": "https://fluidsbench.org/schemas/v3/evaluation-evidence.schema.json", "schema_version": "1.0",
        "submission_id": submission_id, "dataset_id": dataset_id, "dataset_version": dataset_version, "split_id": split_id,
        "case_set_id": case_set_id, "split_sha256": split_sha256,
        "scoring_support_release_id": release.release_id,
        "scoring_support_manifest_sha256": release.manifest_sha256,
        "evaluation": {"reference_version": (contract.evaluation_reference_version if contract else ""),
                       "command": evaluation_command},
        "case_count": case_count, "metric_values": dict(values),
        "case_metrics": {"file": "metrics/cases.json", "sha256": sha256_file(metrics_path)},
        "spatial_discretization": {"file": "discretization.json", "sha256": sha256_file(discretization_path)},
        "support_digests": release.digests,
        "implementation": SUMMARY_SCHEMA,
    }
    evidence_path = dest / "evaluation-evidence.json"
    _write_json(evidence_path, evidence)

    submission: dict[str, Any] = dict(model)
    submission.update(dict(submitter))
    submission.update({
        "$schema": "https://fluidsbench.org/schemas/v3/submission.schema.json", "schema_version": "3.0",
        "submission_id": submission_id, "result_revision": _result_revision(submission_id, result_revision),
        "dataset_id": dataset_id,
        "dataset": model.get("dataset", contract.name if contract else dataset_id),
        "dataset_version": dataset_version or (contract.dataset_version if contract else ""),
        "split_id": split_id, "split": model.get("split", split_id), "case_set_id": case_set_id,
        "split_sha256": split_sha256,
        "methodology": dict(methodology),
        "submitted_at": submitted_at or "",
        "evaluation": {"reference_version": evidence["evaluation"]["reference_version"],
                       "command": evaluation_command, "evidence_file": "evaluation-evidence.json",
                       "evidence_sha256": sha256_file(evidence_path)},
        "reproducibility": dict(reproducibility or {}),
        "scoring_support": {"status": (contract.status if contract else "official"), "release_id": release.release_id,
                            "manifest_sha256": release.manifest_sha256},
        "spatial_discretization": {"format": "fluidsbench-discretization-v1", "file": "discretization.json",
                                   "sha256": sha256_file(discretization_path)},
        "case_metrics": {"format": "fluidsbench-case-metrics-v1", "file": "metrics/cases.json",
                         "sha256": sha256_file(metrics_path), "case_count": case_count},
        "metric_values": metric_values,
    })
    if composite_notes:
        submission["notes"] = list(composite_notes)
    if notes:
        submission["note"] = notes
    _write_json(dest / "submission.json", submission)

    files = {name: sha256_file(dest / name) for name in REQUIRED_FILES}
    return Package(root=dest, submission_id=submission_id, dataset_id=dataset_id, split_id=split_id,
                   case_set_id=case_set_id, metric_values=metric_values,
                   overall_score=composite_values.get("overall_score"), files=files)


def _aggregate_by_metric(scores: Sequence[SupportScore], *, aggregation: str) -> dict[str, float]:
    return aggregate(scores, aggregation=aggregation)


def _case_payload(scores: Sequence[SupportScore], source_output_ids: Mapping[str, str] | None
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[str, list[SupportScore]] = {}
    for score in scores:
        by_case.setdefault(score.case_id, []).append(score)
    payload, lines = [], []
    for case_id, group in by_case.items():
        supports = []
        for score in group:
            coverage = dict(score.coverage)
            support_count = int(coverage.get("support_count", 0))
            scored_count = int(coverage.get("scored_count", support_count))
            entry = {
                "support_id": score.support_id, "support_count": support_count, "scored_count": scored_count,
                "coverage_fraction": float(coverage.get("coverage_fraction", 1.0)),
                "weight_coverage_fraction": float(coverage.get("weight_coverage_fraction", 1.0)),
                "unmapped_count": int(coverage.get("unmapped_count", 0)),
                "extrapolated_count": int(coverage.get("extrapolated_count", 0)),
                "metric_values": dict(score.metric_values),
            }
            if score.statistics:
                entry["metric_sufficient_statistics"] = {k: v.as_dict() for k, v in score.statistics.items()}
            supports.append(entry)
        payload.append({"case_id": case_id, "supports": supports})
        lines.append({
            "$schema": "https://fluidsbench.org/schemas/v3/discretization-case.schema.json", "schema_version": "1.0",
            "case_id": case_id,
            "inference": {"mappings": [
                {"support_id": s.support_id,
                 "source_output_id": (source_output_ids or {}).get(s.support_id, "model-output"),
                 "support_count": int(s.coverage.get("support_count", 0)),
                 "scored_count": int(s.coverage.get("scored_count", 0)),
                 "unmapped_count": int(s.coverage.get("unmapped_count", 0)),
                 "extrapolated_count": int(s.coverage.get("extrapolated_count", 0)),
                 "final_coverage_fraction": float(s.coverage.get("coverage_fraction", 1.0))}
                for s in group]},
        })
    return payload, lines


def load_package(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read ``submission.json`` and ``metrics/cases.json`` of a written package."""
    root = Path(root).expanduser().resolve()
    with open(root / "submission.json", encoding="utf-8") as handle:
        submission = json.load(handle)
    with open(root / "metrics" / "cases.json", encoding="utf-8") as handle:
        case_metrics = json.load(handle)
    return submission, case_metrics


def validate_package(root: str | Path, *, contract: DatasetContract | None = None, checkout: Checkout | None = None,
                     run_official: bool = False) -> ValidationReport:
    """Recompute, from a package's own files, every identity the contract makes it declare.

    These are the checks that catch a package before a maintainer does: a hash that no longer matches its file, an
    aggregate that is not the macro-average of its cases, a relative L2 that disagrees with its sufficient statistics, a
    coverage fraction that does not follow from the counts, an ``overall_score`` that its components do not produce. With
    ``run_official`` and a checkout, the benchmark's own validator runs afterwards — that is the authority on schema and
    on everything else this cannot know."""
    root = Path(root).expanduser().resolve()
    checked: list[str] = []
    problems: list[str] = []

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        return ValidationReport(root=root, problems=tuple(f"missing {name}" for name in missing))
    checked.append("all five required files are present")

    submission, case_metrics = load_package(root)
    for key, path in (("case_metrics", "metrics/cases.json"), ("spatial_discretization", "discretization.json")):
        declared = (submission.get(key) or {}).get("sha256")
        actual = sha256_file(root / path)
        if declared != actual:
            problems.append(f"{key}: submission.json declares {declared!r}, the file hashes to {actual!r}")
    declared_evidence = (submission.get("evaluation") or {}).get("evidence_sha256")
    if declared_evidence != sha256_file(root / "evaluation-evidence.json"):
        problems.append("evaluation.evidence_sha256 does not match evaluation-evidence.json")
    if not problems:
        checked.append("the digest chain of submission.json holds")

    for key, expected in (("dataset_id", submission.get("dataset_id")), ("split_id", submission.get("split_id")),
                          ("case_set_id", submission.get("case_set_id"))):
        if key in case_metrics and case_metrics[key] != expected:
            problems.append(f"metrics/cases.json {key} {case_metrics[key]!r} != submission.json {expected!r}")
    for key in ("scoring_support_release_id", "scoring_support_manifest_sha256"):
        if case_metrics.get(key) != (submission.get("scoring_support") or {}).get(
                "release_id" if key == "scoring_support_release_id" else "manifest_sha256"):
            problems.append(f"{key} disagrees between metrics/cases.json and submission.json")

    cases = case_metrics.get("cases") or []
    if int(case_metrics.get("case_count", -1)) != len(cases):
        problems.append(f"case_count {case_metrics.get('case_count')} does not match {len(cases)} cases")
    if int((submission.get("case_metrics") or {}).get("case_count", -1)) != len(cases):
        problems.append("submission.json case_metrics.case_count does not match the case list")
    if not problems:
        checked.append("case counts and identities agree across the package")

    per_metric: dict[str, list[float]] = {}
    for case in cases:
        for support in case.get("supports") or []:
            counts = ("support_count", "scored_count", "coverage_fraction", "weight_coverage_fraction",
                      "unmapped_count", "extrapolated_count")
            absent = [c for c in counts if c not in support]
            if absent:
                problems.append(f"{case.get('case_id')}/{support.get('support_id')}: no {absent}")
                continue
            support_count = int(support["support_count"])
            if int(support["scored_count"]) + int(support["unmapped_count"]) != support_count:
                problems.append(f"{case['case_id']}/{support['support_id']}: scored + unmapped != support_count")
            if support_count and abs(support["coverage_fraction"] - support["scored_count"] / support_count) > 1e-12:
                problems.append(f"{case['case_id']}/{support['support_id']}: coverage_fraction does not follow from "
                                f"the counts")
            for metric_id, value in (support.get("metric_values") or {}).items():
                per_metric.setdefault(metric_id, []).append(float(value))
            for metric_id, stats in (support.get("metric_sufficient_statistics") or {}).items():
                if stats.get("reduction") != REL_L2:
                    continue
                expected = relative_l2_from_statistics([stats])
                declared = float((support.get("metric_values") or {}).get(metric_id, math.nan))
                scale = max(abs(expected), 1e-12)
                if not (math.isfinite(declared) and abs(declared - expected) <= 1e-9 * scale):
                    problems.append(f"{case['case_id']}/{support['support_id']}: {metric_id} {declared!r} disagrees "
                                    f"with its sufficient statistics ({expected!r})")
    if not problems:
        checked.append("every relative L2 matches its additive sufficient statistics")

    declared_values = {k: float(v) for k, v in (case_metrics.get("metric_values") or {}).items()}
    for metric_id, values in per_metric.items():
        if metric_id not in declared_values:
            continue
        expected = macro_average(values)
        scale = max(abs(expected), 1e-12)
        if abs(expected - declared_values[metric_id]) > 1e-9 * scale:
            problems.append(f"aggregate {metric_id} {declared_values[metric_id]!r} is not the macro-average "
                            f"{expected!r} of its {len(values)} case values")
    if not problems:
        checked.append("every aggregate value is the macro-average of its own case values")

    submission_values = {k: float(v) for k, v in (submission.get("metric_values") or {}).items()}
    disagreements = [k for k, v in declared_values.items()
                     if k in submission_values and abs(submission_values[k] - v) > 1e-12 * max(abs(v), 1e-12)]
    if disagreements:
        problems.append(f"submission.json metric_values disagree with metrics/cases.json for {disagreements}")
    else:
        checked.append("submission.json repeats the case-metrics aggregate values")

    composite = contract.composite if contract is not None else None
    if composite is not None and composite.components and composite.metric_id in submission_values:
        try:
            expected = composite.score(submission_values)
        except ContractError as error:
            problems.append(f"overall_score cannot be recomputed: {error}")
        else:
            if abs(expected - submission_values[composite.metric_id]) > composite.tolerance * max(abs(expected), 1.0):
                problems.append(f"overall_score {submission_values[composite.metric_id]!r} is not what its components "
                                f"produce ({expected!r})")
            else:
                checked.append("overall_score is what the published composite produces from its components")

    revision = submission.get("result_revision") or {}
    if revision:
        expected_id = f"{revision.get('series_id')}-v{revision.get('version')}"
        if expected_id != submission.get("submission_id"):
            problems.append(f"submission_id {submission.get('submission_id')!r} does not follow "
                            f"<series-id>-vN ({expected_id!r})")
        else:
            checked.append("the result revision names this package's version")

    official = None
    if run_official and checkout is not None:
        official = checkout.validate_submission(root)
    elif run_official:
        official = {"ok": False, "stderr": "run_official needs a checkout: Checkout.find()"}
    return ValidationReport(root=root, checked=tuple(checked), problems=tuple(problems), official=official)


def to_table(packages: Sequence[Package], *, ranking: Mapping[str, Any] | None = None) -> str:
    """A small markdown table of several packages of the same dataset, ranked the way the leaderboard ranks them."""
    if not packages:
        return "no package"
    if ranking and ranking.get("metric_id"):
        metric_id = str(ranking["metric_id"])
        direction = str(ranking.get("direction", "higher"))
        places = int(ranking.get("decimal_places", 1))
    else:
        metric_id, direction, places = "overall_score", "higher", 1
    values = [float(p.metric_values.get(metric_id, math.nan)) for p in packages]
    finite = [v for v in values if math.isfinite(v)]
    ranks = competition_rank(finite, direction=direction, decimal_places=places) if finite else []
    iterator = iter(ranks)
    rows = ["| rank | submission | " + metric_id + " |", "| --- | --- | --- |"]
    for package, value in zip(packages, values, strict=True):
        rank = next(iterator, "-") if math.isfinite(value) else "-"
        rows.append(f"| {rank} | {package.submission_id} | {value:.6g} |")
    return "\n".join(rows)


def coverage_ok(scores: Sequence[SupportScore], *, required: float = 1.0) -> bool:
    """Whether every case reaches the required count coverage (the contract's ``count_fraction``)."""
    return all(s.coverage.get("coverage_fraction", 1.0) + 1e-12 >= required for s in scores)
