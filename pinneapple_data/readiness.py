"""Data readiness: may this dataset start a training run? (#1491, part of #1497)

Between "the company has engineering data" and "a model trains on it" the library had nothing: ``Trainer`` trained on
whatever arrays it was given, and the checks that existed (``data_health``, ``cae.quality``, ``preflight``,
``pinneapple_veriphysics``) did not gate anything. This module is that stage.

It **composes**, it does not re-implement:

| group | where the checking happens |
|---|---|
| ``schema`` | the dataset manifest itself: required fields, columns, roles, files that exist |
| ``units`` | :mod:`pinneapple_data.physical_units`, plus the declared units against the ones in the table header |
| ``conditions`` | the manifest: every output field has at least one boundary or initial condition behind it |
| ``validity`` | the manifest: the operating point inside the range the data is declared to cover |
| ``table`` | :func:`pinneapple_data.data_health.analyze` |
| ``case`` | :func:`pinneapple_data.preflight.preflight` |

The verdict is **not a function of the score**. ``status`` comes from the severities (one critical finding is ``FAIL``,
whatever the score is); the score is reported next to it, with the formula in the report, because a number next to a
verdict is useful and a number *as* a verdict is how a critical units mismatch becomes "87 %, ship it".

:func:`require` is the gate. ``FAIL`` and ``NOT_ASSESSED`` always stop a run; ``WARNING`` stops it unless a review
record (:mod:`pinneapple_data.review`, #1493) accepts exactly those findings for exactly this dataset digest.
Behaviour without a report is unchanged -- nothing that exists today starts failing.

The dataset manifest (``pinneapple.readiness.dataset/1``)::

    {"schema": "pinneapple.readiness.dataset/1",
     "name": "plate_sweep", "version": "1.0",
     "columns": [{"name": "Re", "role": "input", "unit": "1"},
                 {"name": "delta_p", "role": "output", "unit": "Pa"}],
     "boundary_conditions": [{"field": "delta_p", "kind": "dirichlet", "value": 0}],
     "validity": {"Re": [1.0e4, 1.0e5]},
     "operating_point": {"Re": 5.0e4}}
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .physical_units import try_parse_unit

SCHEMA = "pinneapple.readiness.dataset/1"
SEVERITIES = ("critical", "warning", "info")
GROUPS = ("schema", "units", "conditions", "validity", "table", "case")
STATUSES = ("PASS", "WARNING", "FAIL", "NOT_ASSESSED")
WEIGHT = {"critical": 30.0, "warning": 10.0, "info": 2.0}
ROLES = ("input", "output", "meta", "id")


class ReadinessError(RuntimeError):
    """Training was refused. ``findings`` holds what stopped it; the message names ids, evidence and fixes."""

    def __init__(self, message: str, *, report: Optional["ReadinessReport"] = None,
                 findings: Sequence["Finding"] = (), review: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.report = report
        self.findings = list(findings)
        self.review = dict(review or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"error": str(self), "verdict": None if self.report is None else self.report.status,
                "subject": None if self.report is None else self.report.digest,
                "findings": [f.to_dict() for f in self.findings], "review": self.review}


@dataclass
class Finding:
    """One problem, in the shape the sibling reports use: id, group, severity, what, evidence, and the fix."""
    id: str
    group: str
    severity: str
    title: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    fix: str = ""
    affected: float = 1.0                      # share of the dataset affected, 0..1

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessPolicy:
    """Which findings are critical, which groups apply, and where the policy came from.

    The default policy is the library's: the checks that make a dataset unusable (a missing unit, a missing boundary
    condition, an operating point outside the declared validity) are critical, and everything else is a warning.
    """
    name: str = "default"
    path: str = ""
    critical: List[str] = field(default_factory=lambda: list(DEFAULT_CRITICAL))
    warnings: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    require_review_for_warnings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, path: str = "") -> "ReadinessPolicy":
        unknown = set(d) - {"name", "critical", "warnings", "disabled", "require_review_for_warnings", "path"}
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        p = cls(name=str(d.get("name") or "policy"), path=path or str(d.get("path") or ""),
                critical=list(d.get("critical") or DEFAULT_CRITICAL),
                warnings=list(d.get("warnings") or []),
                disabled=list(d.get("disabled") or []),
                require_review_for_warnings=bool(d.get("require_review_for_warnings", True)))
        bad = (set(p.critical) | set(p.warnings)) - set(GROUPS) - set(ALL_CHECKS)
        if bad:
            raise ValueError(f"policy names checks that do not exist: {sorted(bad)}")
        return p

    def severity_of(self, finding: Finding) -> str:
        """The policy has the last word on how serious a finding is, then the check itself."""
        if finding.id in self.warnings or finding.group in self.warnings:
            return "warning"
        if finding.id in self.critical or finding.group in self.critical:
            return "critical"
        return finding.severity

    def enabled(self, group: str) -> bool:
        return group not in self.disabled


ALL_CHECKS = (
    "schema.missing", "schema.columns", "schema.file_missing", "schema.duplicate_column",
    "units.missing", "units.unparseable", "units.mismatch",
    "conditions.missing", "conditions.unknown_field",
    "validity.outside", "validity.malformed",
    "table.issue", "case.finding",
)
DEFAULT_CRITICAL = ("schema.missing", "schema.file_missing", "units.unparseable", "units.mismatch",
                    "conditions.missing", "validity.outside")


def default_policy() -> ReadinessPolicy:
    return ReadinessPolicy()


@dataclass
class ReadinessReport:
    """The verdict, why, and the evidence behind it -- JSON-ready and renderable without the app."""
    subject: str
    digest: str
    status: str
    findings: List[Finding]
    policy: ReadinessPolicy
    score: Dict[str, Any]
    groups: Dict[str, int]
    name: str = ""

    # -- reading ----------------------------------------------------------------------------------------
    def critical(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def ids(self, severity: Optional[str] = None) -> List[str]:
        return sorted(f.id for f in self.findings if severity is None or f.severity == severity)

    def explain(self) -> str:
        """The refusal in the words a user needs: ids, evidence and fix, one finding per line."""
        lines = [f"{self.status} on {self.name or self.subject} ({self.digest[:20]}...), score "
                 f"{self.score.get('overall')}:"]
        for f in self.findings:
            if f.severity == "info":
                continue
            lines.append(f"  [{f.severity}] {f.id}: {f.title}")
            lines.append(f"      {f.detail}")
            if f.evidence:
                lines.append(f"      evidence: {json.dumps(f.evidence, sort_keys=True, default=str)}")
            if f.fix:
                lines.append(f"      fix: {f.fix}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "pinneapple.readiness.report/1", "subject": self.subject, "name": self.name,
                "digest": self.digest, "status": self.status, "policy": self.policy.to_dict(),
                "score": self.score, "groups": self.groups, "findings": [f.to_dict() for f in self.findings]}

    def to_markdown(self) -> str:
        lines = [f"### Readiness of {self.name or self.subject}", "",
                 f"**{self.status}** · score {self.score.get('overall')} · `{self.digest[:24]}...` · policy "
                 f"`{self.policy.name}`", "",
                 "| severity | id | what | evidence |", "|---|---|---|---|"]
        for f in self.findings:
            ev = ", ".join(f"{k}={v}" for k, v in list(f.evidence.items())[:3])
            lines.append(f"| {f.severity} | `{f.id}` | {f.title} | {ev} |")
        if not self.findings:
            lines.append("| — | — | every check in the enabled groups passed | — |")
        lines += ["", f"_{self.score.get('method', '')}_", ""]
        return "\n".join(lines)


# ------------------------------------------------------------------ the digest
def digest_of(subject: Any, files: Optional[Mapping[str, bytes]] = None) -> str:
    """``sha256:`` of the dataset description and, when there are files, of their contents.

    The digest is what a review is tied to: change the dataset and the approval no longer applies (#1491, #1493).
    """
    h = hashlib.sha256()
    h.update(json.dumps(_as_manifest(subject), sort_keys=True, default=str).encode("utf-8"))
    for name in sorted(files or {}):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(files[name]).digest())
    return "sha256:" + h.hexdigest()


def _table_bytes(table: Any) -> Optional[bytes]:
    """The data itself, so the digest changes when the values do -- an approval is for a dataset, not a schema."""
    if table is None:
        return None
    if isinstance(table, (str, Path)):
        path = Path(table)
        return path.read_bytes() if path.is_file() else None
    if hasattr(table, "to_csv"):
        return table.to_csv(index=False).encode("utf-8")
    return None


def _as_manifest(subject: Any) -> Dict[str, Any]:
    if isinstance(subject, ReadinessReport):
        return {"digest": subject.digest}
    if isinstance(subject, Mapping):
        return dict(subject)
    if isinstance(subject, (str, Path)):
        path = Path(subject)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        raise ValueError(f"{subject!r} is not a dataset manifest file")
    # a DataFrame or anything else with columns: described by its columns and units
    columns = getattr(subject, "columns", None)
    if columns is not None:
        return {"columns": [str(c) for c in columns]}
    raise TypeError(f"cannot read a dataset description from {type(subject).__name__}")


# ------------------------------------------------------------------ the checks
def _check_schema(m: Mapping[str, Any], base: Optional[Path]) -> List[Finding]:
    out: List[Finding] = []
    missing = [k for k in ("name", "columns") if not m.get(k)]
    if missing:
        out.append(Finding("schema.missing", "schema", "critical", "The manifest is incomplete",
                           f"Required field(s) missing: {', '.join(missing)}. A dataset with no name or no columns "
                           f"cannot be checked, and an unchecked dataset cannot train.",
                           evidence={"missing": missing}, fix="Add them to the manifest (see the schema in "
                                                              "pinneapple_data/readiness.py)."))
    cols = list(m.get("columns") or ())
    bad_roles = [c for c in cols if isinstance(c, Mapping) and c.get("role") and c["role"] not in ROLES]
    if bad_roles:
        out.append(Finding("schema.columns", "schema", "critical", "A column has an unknown role",
                           f"Roles must be one of {ROLES}; got {[c.get('role') for c in bad_roles]}.",
                           evidence={"columns": bad_roles}, fix="Fix the role in the manifest."))
    names = [str(c.get("name")) for c in cols if isinstance(c, Mapping) and c.get("name")]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        out.append(Finding("schema.duplicate_column", "schema", "critical", "Two columns share a name",
                           f"Duplicated: {dupes}. A model cannot tell them apart.",
                           evidence={"duplicates": dupes}, fix="Rename one of them in the manifest and the table."))
    if not any((c.get("role") if isinstance(c, Mapping) else None) == "output" for c in cols):
        out.append(Finding("schema.columns", "schema", "warning", "No column is marked as an output",
                           "Every column is an input or metadata, so there is nothing for a model to learn.",
                           evidence={"columns": names},
                           fix="Mark the predicted quantity with \"role\": \"output\"."))
    if base is not None:
        for f in m.get("files") or ():
            p = Path(f) if Path(f).is_absolute() else base / f
            if not p.exists():
                out.append(Finding("schema.file_missing", "schema", "critical", "A file the dataset names is missing",
                                   f"{f} does not exist next to the manifest.", evidence={"file": str(f)},
                                   fix="Copy the file in place or fix the path (paths are relative to the manifest)."))
    return out


def _check_units(m: Mapping[str, Any], header_units: Mapping[str, str]) -> List[Finding]:
    out: List[Finding] = []
    for c in m.get("columns") or ():
        if not isinstance(c, Mapping) or not c.get("name"):
            continue
        name, unit = str(c["name"]), str(c.get("unit") or "")
        if not unit:
            out.append(Finding("units.missing", "units", "warning", f"Column '{name}' has no unit",
                               f"'{name}' is declared without a unit, so a value cannot be checked against physical "
                               f"bounds or converted.",
                               evidence={"column": name}, fix=f"Add the unit of '{name}' to the manifest."))
            continue
        parsed = try_parse_unit(unit)
        if parsed is None:
            out.append(Finding("units.unparseable", "units", "critical", f"Column '{name}' has an unknown unit",
                               f"'{unit}' could not be parsed as a unit.", evidence={"column": name, "unit": unit},
                               fix="Use a unit the library knows (see pinneapple_data.physical_units)."))
            continue
        declared_header = header_units.get(name)
        if declared_header and try_parse_unit(declared_header) is not None:
            other = try_parse_unit(declared_header)
            if other is not None and other.dim != parsed.dim:
                out.append(Finding("units.mismatch", "units", "critical",
                                   f"Column '{name}': the manifest and the table disagree on the unit",
                                   f"The manifest says '{unit}' and the table header says '{declared_header}'; the "
                                   f"dimensions differ, so every value in that column means something else than the "
                                   f"manifest claims.",
                                   evidence={"column": name, "manifest_unit": unit, "table_unit": declared_header},
                                   fix="Fix one of the two, or convert the column to the declared unit."))
    return out


def _check_conditions(m: Mapping[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    fields = [str(c["name"]) for c in m.get("columns") or () if isinstance(c, Mapping) and c.get("role") == "output"]
    declared = m.get("boundary_conditions") or ()
    named = {str(b.get("field")) for b in declared if isinstance(b, Mapping) and b.get("field")}
    for f in fields:
        if f not in named:
            out.append(Finding("conditions.missing", "conditions", "critical",
                               f"Output '{f}' has no boundary or initial condition behind it",
                               f"The dataset predicts '{f}' but declares no boundary/initial condition for it. A field "
                               f"model trained on data whose boundary condition was never stated learns the solver's "
                               f"assumption, not the physics.",
                               evidence={"field": f, "declared": sorted(named)},
                               fix=f"Add the condition(s) that produced '{f}' to \"boundary_conditions\"."))
    unknown = sorted(named - set(str(c["name"]) for c in m.get("columns") or () if isinstance(c, Mapping)))
    if unknown:
        out.append(Finding("conditions.unknown_field", "conditions", "warning",
                           "A boundary condition names a field the dataset does not have",
                           f"The conditions name {unknown}, which are not columns of the dataset.",
                           evidence={"fields": unknown}, fix="Fix the field name or drop the condition."))
    return out


def _check_validity(m: Mapping[str, Any]) -> List[Finding]:
    out: List[Finding] = []
    validity = m.get("validity") or {}
    point = m.get("operating_point") or {}
    for key, rng in validity.items():
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            out.append(Finding("validity.malformed", "validity", "warning", f"Validity range for '{key}' is malformed",
                               f"Expected [low, high], got {rng!r}.", evidence={"key": key, "range": rng},
                               fix="Write the range as [low, high] with numbers."))
            continue
        if key in point:
            value = float(point[key])
            if value < lo or value > hi:
                out.append(Finding("validity.outside", "validity", "critical",
                                   f"Operating point '{key}' is outside the declared validity",
                                   f"{key} = {value:g} at the operating point, while the data is declared valid in "
                                   f"[{lo:g}, {hi:g}]. A model trained here would be asked about a regime the data "
                                   f"does not cover.",
                                   evidence={"key": key, "operating_point": value, "validity": [lo, hi]},
                                   fix=f"Extend the data to {key} = {value:g}, or move the operating point inside the "
                                       f"declared range."))
    return out


def _check_table(table: Any, declared_units: Mapping[str, str]) -> tuple[List[Finding], Dict[str, str]]:
    from .data_health import analyze

    df = table
    header_units: Dict[str, str] = {}
    if isinstance(table, (str, Path)):
        import pandas as pd

        path = Path(table)
        df = pd.read_csv(path) if path.suffix.lower() in (".csv", ".tsv", ".txt") else pd.read_excel(path)
    if hasattr(df, "columns"):
        from .physical_units import split_header

        for col in df.columns:
            name, unit = split_header(str(col))
            if unit:
                header_units[name] = unit
    report = analyze(df, units=dict(declared_units))
    out: List[Finding] = []
    for issue in report.get("issues") or ():
        sev = issue.get("severity") or "info"
        out.append(Finding(f"table.{issue.get('category', 'issue')}", "table",
                           "critical" if sev == "critical" else ("warning" if sev == "warning" else "info"),
                           str(issue.get("title") or issue.get("category")),
                           str(issue.get("detail") or ""),
                           evidence={k: v for k, v in issue.items() if k in ("column", "affected_fraction", "count",
                                                                            "spans", "evidence") and v},
                           fix=str(issue.get("suggestion") or ""),
                           affected=float(issue.get("affected_fraction") or 1.0)))
    return out, header_units


def _check_case(files: Mapping[str, bytes]) -> List[Finding]:
    from .preflight import preflight

    report = preflight(dict(files))
    out: List[Finding] = []
    for f in report.get("findings") or ():
        sev = str(f.get("status") or "INFO").lower()
        out.append(Finding(f"case.{f.get('section', 'case')}", "case",
                           "critical" if sev == "fail" else ("warning" if sev == "warn" else "info"),
                           str(f.get("title") or ""), str(f.get("detail") or ""),
                           evidence={"file": f.get("file"), "line": f.get("line"), **dict(f.get("evidence") or {})},
                           fix=str(f.get("fix") or "")))
    return out


# ------------------------------------------------------------------ the report
def _score(findings: Sequence[Finding]) -> Dict[str, Any]:
    penalty = 0.0
    for f in findings:
        scale = 0.5 + 0.5 * min(1.0, max(0.0, f.affected) / 0.1)
        penalty += WEIGHT[f.severity] * scale
    overall = max(0.0, 100.0 - penalty)
    return {"overall": round(overall, 1), "penalty": round(penalty, 1),
            "method": "Each finding costs 30 (critical) / 10 (warning) / 2 (info) points, scaled from 50 % to 100 % "
                      "by the share of the dataset it affects (100 % at 10 % or more); the score is 100 minus the "
                      "sum, floored at 0. The score is reported next to the verdict and never decides it."}


def readiness(subject: Any, *, policy: Optional[ReadinessPolicy] = None, table: Any = None,
              case_files: Optional[Mapping[str, bytes]] = None, base: Optional[str | Path] = None,
              name: str = "") -> ReadinessReport:
    """Assess a dataset and return the report. ``subject`` is a manifest (mapping or JSON path) or a table.

    ``table`` attaches the data itself (a DataFrame or a CSV/Excel path) so the table group runs; ``case_files``
    attaches a solver case (``{filename: bytes}``) so the case group runs. Nothing is invented: a group that has no
    input does not run, and a report with no group at all is ``NOT_ASSESSED``, never ``PASS``.
    """
    policy = policy or default_policy()
    manifest = _as_manifest(subject)
    base_path = Path(base) if base is not None else (Path(subject).parent if isinstance(subject, (str, Path))
                                                     and Path(subject).is_file() else None)
    if manifest.get("schema") and manifest["schema"] != SCHEMA:
        raise ValueError(f"manifest schema is {manifest['schema']!r}, this module reads {SCHEMA!r}")

    declared_units = {str(c["name"]): str(c.get("unit") or "") for c in manifest.get("columns") or ()
                      if isinstance(c, Mapping) and c.get("name")}
    findings: List[Finding] = []
    groups: Dict[str, int] = {}

    def run(group: str, produced: Sequence[Finding]) -> None:
        if policy.enabled(group):
            findings.extend(produced)
            groups[group] = groups.get(group, 0) + len(produced)

    run("schema", _check_schema(manifest, base_path))
    run("conditions", _check_conditions(manifest))
    run("validity", _check_validity(manifest))
    header_units: Dict[str, str] = {}
    if table is not None:
        table_findings, header_units = _check_table(table, declared_units)
        # the units group already checks every declared column, with dimensions and against the header, so the
        # table's own metadata heuristic would report the same missing unit a second time
        table_findings = [f for f in table_findings if f.id != "table.metadata"]
        if not manifest.get("temporal"):
            # a parameter sweep is not a time series: "no time column" is not a defect of this dataset, and saying so
            # on every non-temporal dataset would train people to ignore the report
            table_findings = [f for f in table_findings if f.id != "table.time_axis"]
        run("table", table_findings)
    # the units of the manifest, checked against the ones the table header carries -- once, so a missing unit is not
    # reported twice
    run("units", _check_units(manifest, header_units))
    if case_files:
        run("case", _check_case(case_files))

    for f in findings:                                  # the policy has the last word on severity
        f.severity = policy.severity_of(f)
    findings.sort(key=lambda f: (SEVERITIES.index(f.severity), f.id))

    # a group that had nothing to check does not make a report: with no columns, no table and no case, nothing was
    # assessed, and NOT_ASSESSED is a refusal rather than a quiet pass
    assessed = bool(manifest.get("columns")) or table is not None or bool(case_files)
    if not assessed or not groups:
        status = "NOT_ASSESSED"
    elif any(f.severity == "critical" for f in findings):
        status = "FAIL"
    elif any(f.severity == "warning" for f in findings):
        status = "WARNING"
    else:
        status = "PASS"

    digest_files = dict(case_files or {})
    if (raw := _table_bytes(table)) is not None:
        digest_files["data"] = raw
    digest = digest_of(manifest, files=digest_files)
    label = name or str(manifest.get("name") or "") or "dataset"
    return ReadinessReport(subject=label, name=label, digest=digest, status=status, findings=findings,
                           policy=policy, score=_score(findings), groups=groups)


def require(report: ReadinessReport, *, review: Any = None) -> ReadinessReport:
    """The gate: raise :class:`ReadinessError` unless this dataset may start a training run.

    ``review`` is a :class:`pinneapple_data.review.ReviewStore`; it is what turns a ``WARNING`` into a run, by
    accepting exactly those findings for exactly this digest. ``FAIL`` and ``NOT_ASSESSED`` are never overridable.
    """
    if report.status == "PASS":
        return report
    if report.status == "NOT_ASSESSED":
        raise ReadinessError(
            f"refusing to train: nothing was assessed for {report.name or report.subject}. A dataset with no verdict "
            f"is not a dataset with a good verdict; attach a manifest, a table or a case and assess it.",
            report=report)
    if report.status == "FAIL":
        raise ReadinessError("refusing to train: the dataset failed its readiness checks.\n" + report.explain(),
                             report=report, findings=report.critical())
    # WARNING
    if not report.policy.require_review_for_warnings:
        return report
    from .review import approval_for

    warnings = report.warnings()
    ids = [f.id for f in warnings]
    decision = approval_for(review, report.digest, ids)
    if decision["approved"]:
        report.score["review"] = decision          # the run carries who accepted what
        return report
    missing = decision.get("missing") or ids
    raise ReadinessError(
        f"refusing to train: {len(ids)} warning(s) need an engineering review, and {len(missing)} of them are not "
        f"accepted for this dataset ({report.digest[:20]}...).\n" + report.explain(),
        report=report, findings=[f for f in warnings if f.id in missing], review=decision)
