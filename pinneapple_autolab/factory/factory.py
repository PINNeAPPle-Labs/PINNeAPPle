"""``run_factory(spec, solver, out=...)``: CAD -> DOE -> solver -> dataset, with the quality control in between.

The design study (:mod:`pinneapple_analysis.design_study`) answers *which* design to choose; the factory produces the
*data* a study is chosen on, and it is the same spec either way: the variables, the tool with its input mapping, the
outputs with their units, the budget and the seed come from one ``DesignStudy``, so a case described for
``run_study`` runs here without being described twice.

     from pinneapple_analysis.design_study import ...
     from pinneapple_autolab.factory import run_factory
     run = run_factory(spec, my_simulator, out="runs/bend")     # baseline -> DOE -> solver -> dataset -> QC
     run.status, run.qc["decision"], run.dataset["n_rows"]      # ('done', 'APPROVED', 9)

The five steps, and what each one is allowed to say
--------------------------------------------------
1. **CAD.** ``geometry.prepare(design)`` turns the design into a case (parameters completed and derived, the exact
   volume, the smallest feature, the boundary patches, a point inside the fluid) and checks it against the
   template's published limits. The design is the study's variables plus the base's ``fixed`` parameters — the ones
   the case declares but never varies, such as the inlet and outlet runs or the diameter of a product's case. A
   design the geometry refuses is a row with the reason, never a dropped case.
2. **DOE.** The plan is the study's own: ``budget.planned(d)["doe"]`` designs from the DOE interface, plus the
   reference design first. The plan's space-filling metrics are recorded as measured (``run.doe``), not graded.
3. **Solver.** ``Simulator.check`` before ``Simulator.run``: the lab never runs a design its own checks reject.
   The call is the evaluator's inputs — ``Evaluator.tool_params`` maps the study's variables to the tool's names and
   adds the conditions the study never changes — so the same simulator serves ``run_study`` and the factory.
4. **Dataset.** Only the rows that ran and passed quality control are written, with their arrays, units, bounds,
   split and, per row, what produced it (:mod:`.interfaces.StudyDatasetWriter`).
5. **Quality control.** Every run carries its checks (:mod:`.qc`); the rows add up to one
   :class:`~pinneapple_analysis.trust.TrustReport`. ``REJECT`` means the dataset is not written at all — the run's
   rows are still returned, with their reasons, because a dataset nobody can sign is not a product.

The status is ``done`` when the dataset was written and the verdict is ``APPROVED``, ``inconclusive`` when it was
written under ``REVIEW`` (a check could not run: the data is real, the evidence is not complete), and ``refused``
when nothing was written. There is no fourth way to say "fine".

What the factory does not pretend
---------------------------------
A missing end is said, not simulated: ``missing=[...]`` refuses the run with those reasons (a case whose solver does
not exist in this library passes its own reason here), and a geometry that only checked its parameters — no CAD
kernel in this environment — keeps its solid checks ``not_run`` and says so in the dataset.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pinneapple_analysis.design_study import DesignStudy
from pinneapple_analysis.trust.trust_report import Check
from pinneapple_autolab.simulator import SimOutcome

from .interfaces import (
    AsIsGeometry,
    DatasetRef,
    DatasetWriter,
    Doe,
    Geometry,
    GeometryArtifact,
    MaximinDoe,
    StudyDatasetWriter,
    TemplateGeometry,
)
from .qc import KEPT, REJECTED, FactoryQc, row_verdict

__all__ = ["run_factory", "FactoryRun", "FactoryRow", "default_geometry", "Status"]


class Status:
    """The three statuses a run can end in; ``inconclusive`` and ``refused`` are different answers."""

    DONE, INCONCLUSIVE, REFUSED = "done", "inconclusive", "refused"


# --------------------------------------------------------------------------- one evaluated design
@dataclass
class FactoryRow:
    """One reference evaluation: the design, the case as geometry, the outputs, the checks and the provenance."""

    id: str
    purpose: str                                  # baseline | doe
    params: dict[str, float]                      # the study's variables
    tool_params: dict[str, Any] = field(default_factory=dict)   # the evaluator's own inputs and conditions
    outputs: dict[str, float] = field(default_factory=dict)
    ok: bool = False
    message: str = ""
    cost_s: float = 0.0
    started: str = ""
    artifact: GeometryArtifact | None = None
    checks: list[Check] = field(default_factory=list)
    verdict: str = KEPT
    reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Ran, and passed the checks: only these rows reach the dataset."""
        return bool(self.ok) and self.verdict == KEPT

    def record(self):
        """The ``RunRecord`` of this row (what the dataset builder and the ledger read)."""
        from pinneapple_autolab.ledger import RunRecord
        return RunRecord(id=self.id, purpose=self.purpose, params=dict(self.params), ok=self.accepted,
                         outputs=dict(self.outputs), message=self.message, cost_s=self.cost_s,
                         started=self.started, reason=f"{self.purpose}: {self.purpose} evaluation of the factory",
                         meta={"tool_params": dict(self.tool_params),
                               "geometry": self.artifact.to_dict() if self.artifact else {},
                               "provenance": dict(self.provenance),
                               "qc": [{"name": c.name, "status": c.status, "critical": c.critical,
                                       "value": c.value, "limit": c.limit, "reason": c.reason} for c in self.checks]})

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "purpose": self.purpose, "params": dict(self.params),
                "tool_params": dict(self.tool_params), "outputs": dict(self.outputs), "ok": self.ok,
                "accepted": self.accepted, "message": self.message, "cost_s": self.cost_s, "started": self.started,
                "verdict": self.verdict, "reasons": list(self.reasons), "provenance": dict(self.provenance),
                "geometry": self.artifact.to_dict() if self.artifact else {},
                "checks": [{"name": c.name, "status": c.status, "critical": c.critical, "value": c.value,
                            "limit": c.limit, "reason": c.reason} for c in self.checks]}


# --------------------------------------------------------------------------- the run
@dataclass
class FactoryRun:
    """What a factory run produced: the rows, the dataset it wrote, the verdict and where the chain stopped."""

    spec: DesignStudy
    status: str
    reasons: list[str] = field(default_factory=list)
    created: str = ""
    geometry: dict[str, Any] = field(default_factory=dict)
    doe: dict[str, Any] = field(default_factory=dict)
    rows: list[FactoryRow] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    workdir: str = ""
    wall_s: float = 0.0

    @property
    def accepted(self) -> list[FactoryRow]:
        return [r for r in self.rows if r.accepted]

    @property
    def rejected(self) -> list[FactoryRow]:
        return [r for r in self.rows if not r.accepted]

    @property
    def decision(self) -> str:
        return str(self.qc.get("decision", ""))

    @property
    def approved(self) -> bool:
        return self.status == Status.DONE and self.decision == "APPROVED"

    def summary(self) -> dict[str, Any]:
        plan = self.spec.budget.planned(len(self.spec.base.variables))
        return {"status": self.status, "decision": self.decision, "title": self.spec.title,
                "digest": self.spec.digest(), "tool": self.spec.evaluator.tool,
                "fidelity": self.spec.evaluator.fidelity,
                "evaluations": {"used": len(self.rows), "planned_doe": plan["doe"],
                                "max": self.spec.budget.max_evaluations},
                "rows": {"evaluated": len(self.rows), "accepted": len(self.accepted),
                         "rejected": len(self.rejected)},
                "n_dataset_rows": int(self.dataset.get("n_rows", 0)),
                "dataset_digest": self.dataset.get("digest", ""), "reasons": list(self.reasons),
                "notes": list(self.notes), "wall_s": round(self.wall_s, 3)}

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict(), "summary": self.summary(), "status": self.status,
                "reasons": list(self.reasons), "created": self.created, "geometry": dict(self.geometry),
                "doe": dict(self.doe), "baseline": dict(self.baseline), "qc": dict(self.qc),
                "dataset": dict(self.dataset), "steps": list(self.steps), "notes": list(self.notes),
                "rows": [r.to_dict() for r in self.rows], "wall_s": self.wall_s}

    def to_markdown(self) -> str:
        """The run as the page a reviewer reads: what was built, what was simulated, what was refused and why."""
        f = _fmt
        out = [f"# Factory run: {self.spec.title}", "",
               f"`{self.spec.evaluator.tool}` ({self.spec.evaluator.fidelity}), spec `{self.spec.digest()[:16]}`, "
               f"created {self.created}. Status **{self.status}**"
               + (f", trust **{self.decision}** (score {f(self.qc.get('score'))}, coverage "
                  f"{100 * (self.qc.get('coverage') or 0):.0f} %)" if self.qc else "") + ".", ""]
        if self.geometry:
            out += ["## Geometry", "", f"- `{self.geometry.get('name', '')}`: {self.geometry.get('title', '')}",
                    f"- build: {self.geometry.get('build', '')}"]
            for c in self.geometry.get("checks", []):
                out.append(f"- `{c.get('name', '')}`: {c.get('status', '')}"
                           + (f" — {c.get('reason')}" if c.get("reason") else ""))
            out.append("")
        if self.doe:
            out += ["## Design of experiments", "", f"- {self.doe.get('name', '')}, {self.doe.get('n', 0)} "
                    f"design(s), seed {self.doe.get('seed', 0)}"]
            cov = self.doe.get("coverage") or {}
            ref = self.doe.get("random_reference") or {}
            for k, v in cov.items():
                out.append(f"- coverage {k}: {f(v)}" + (f" (random design {f(ref.get(k))})" if k in ref else ""))
            out.append("")
        out += ["## Runs", "", "| id | purpose | design | outputs | QC |", "|---|---|---|---|---|"]
        for r in self.rows:
            out.append(f"| {r.id} | {r.purpose} | " + ", ".join(f"{k} {f(v)}" for k, v in r.params.items())
                       + " | " + (", ".join(f"{k} {f(v)}" for k, v in r.outputs.items()) or "—")
                       + f" | {r.verdict}" + ("; ".join([""] + r.reasons)) + " |")
        if self.rejected:
            out += ["", "## Refused runs", ""]
            for r in self.rejected:
                out.append(f"- {r.id}: " + ("; ".join(r.reasons) or r.message or "refused"))
        out += ["", "## Quality control", ""]
        for c in self.qc.get("checks", []):
            out.append(f"- {c['status']}: {c['name']}"
                       + (f" (value {f(c['value'])}, limit {f(c['limit'])})" if c["value"] is not None else "")
                       + (f" — {c['reason']}" if c.get("reason") else ""))
        if self.dataset:
            out += ["", "## Dataset", "", f"- {self.dataset.get('n_rows', 0)} row(s), "
                    f"`{self.dataset.get('json', '')}`", f"- digest {self.dataset.get('digest', '')}"]
        if self.reasons or self.notes:
            out += ["", "## Why", ""] + [f"- {n}" for n in self.reasons + self.notes]
        return "\n".join(out) + "\n"


def _fmt(v: Any) -> str:
    """A number as the page shows it: four significant digits, a dash for nothing, nested metrics flattened."""
    if isinstance(v, bool) or v is None:
        return "—" if v is None else str(v)
    if isinstance(v, (int, float)):
        return f"{v:.4g}"
    if isinstance(v, Mapping):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    return str(v)


def _trust_dict(rep) -> dict[str, Any]:    return {"decision": rep.decisao, "score": rep.score, "coverage": rep.cobertura, "reasons": rep.motivos,
            "checks": [{"name": c.name, "status": c.status, "value": c.value, "limit": c.limit,
                        "critical": c.critical, "source": c.source, "reason": c.reason} for c in rep.checks],
            "rules": "any critical check failing -> REJECT; any other failing, coverage below the minimum or the "
                     "score below it -> REVIEW; else APPROVED (pinneapple_analysis.trust.TrustReport)"}


def _refused(spec: DesignStudy, reasons: Sequence[str], **kw: Any) -> FactoryRun:
    kw.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return FactoryRun(spec=spec, status=Status.REFUSED, reasons=list(reasons), **kw)


def default_geometry(spec: DesignStudy) -> Geometry:
    """The geometry a spec names, without the caller having to build one.

    A ``geometry`` base names a template of ``pinneapple_design.fitmesh`` (``"pp.geometries:pipe_bend_90"``) and
    becomes a :class:`~.interfaces.TemplateGeometry`; a base that names no template is refused with the list, because
    a typo must not become "no geometry". An ``stl`` or ``system`` base has no parametrization here: it passes the
    design through and says the parametric checks did not run.
    """
    if spec.base.kind == "geometry":
        return TemplateGeometry(spec.base.ref)
    return AsIsGeometry(spec.base.ref)


# --------------------------------------------------------------------------- the run itself
def run_factory(spec: DesignStudy | Mapping, solver: Any, *, out: str, geometry: Geometry | None = None,
                doe: Doe | None = None, writer: DatasetWriter | None = None, qc: FactoryQc | None = None,
                workdir: str | None = None, progress: Callable[[dict], None] | None = None,
                missing: Sequence[str] = ()) -> FactoryRun:
    """Run one case through the factory. ``out`` is the folder the dataset is written to.

    ``missing`` is the honest hole: a case whose solver (or geometry) does not exist in the library yet passes the
    reason here and the run is refused with it, instead of producing numbers nothing could have measured.
    """
    spec = spec if isinstance(spec, DesignStudy) else DesignStudy.from_dict(spec)
    if missing:
        return _refused(spec, list(missing))
    if solver is None:
        return _refused(spec, ["no solver was given: the factory has nothing to measure the designs with"])
    if not (hasattr(solver, "run") and hasattr(solver, "check")):
        raise TypeError("the solver must be a pinneapple_autolab.simulator.Simulator (check + run); wrap a plain "
                        f"function in FunctionSimulator, which is one. Got {type(solver).__name__}")

    started = time.perf_counter()
    tell = progress or (lambda event: None)
    own_dir = workdir is None
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="pinneapple-factory-"))
    os.makedirs(workdir, exist_ok=True)
    qc = qc or FactoryQc()
    writer = writer or StudyDatasetWriter(seed=spec.budget.seed)
    try:
        return _run(spec, solver, out=out, geometry=geometry, doe=doe, writer=writer, qc=qc, workdir=workdir,
                    tell=tell, started=started)
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def _run(spec: DesignStudy, solver: Any, *, out: str, geometry: Geometry | None, doe: Doe | None,
         writer: DatasetWriter, qc: FactoryQc, workdir: str, tell: Callable[[dict], None],
         started: float) -> FactoryRun:
    run = FactoryRun(spec=spec, status=Status.INCONCLUSIVE,
                     created=datetime.now(timezone.utc).isoformat(timespec="seconds"), workdir=workdir)
    names = spec.base.names
    seed = int(spec.budget.seed)

    # 1 - the geometry -------------------------------------------------------
    try:
        geo = geometry or default_geometry(spec)
    except ValueError as exc:
        return _refused(spec, [str(exc)], created=run.created)
    run.geometry = {**geo.describe(), "kind": type(geo).__name__}
    run.steps.append({"step": "geometry", "ok": True, "what": geo.describe().get("name", type(geo).__name__)})
    tell({"event": "geometry", "name": run.geometry.get("name", "")})

    # 2 - the design of experiments ------------------------------------------
    plan = spec.budget.planned(len(names))
    designs: list[tuple[str, dict[str, float]]] = [("baseline", spec.base.baseline_design())]
    doe = doe or MaximinDoe()
    try:
        planned = doe.plan(spec.base.variables, plan["doe"], seed=seed,
                           existing=[spec.base.baseline_design()])
    except ValueError as exc:
        return _refused(spec, [f"the design of experiments refused the plan: {exc}"], created=run.created)
    designs += [("doe", d) for d in planned]
    run.doe = {**doe.info(), "planned": plan["doe"], "cases": planned}
    run.steps.append({"step": "doe", "ok": True, "n": len(planned)})
    tell({"event": "doe", "n": len(planned), "planned": plan["doe"]})

    # 3 + 5 - the solver, and the quality control of every run ----------------
    for i, (purpose, design) in enumerate(designs):
        row = _evaluate(spec, solver, geo, qc, design, purpose=purpose, index=i, seed=seed,
                        workdir=os.path.join(workdir, f"case{i:03d}"))
        run.rows.append(row)
        if purpose == "baseline" and row.accepted:
            run.baseline = dict(row.outputs)
        tell({"event": "run", "index": i, "id": row.id, "purpose": purpose, "ok": row.accepted,
              "outputs": dict(row.outputs)})
    run.steps.append({"step": "solver", "ok": True, "evaluated": len(run.rows),
                      "accepted": len(run.accepted), "rejected": len(run.rejected)})

    # 4 - the dataset, and the quality control of the whole run --------------
    report = qc.dataset_checks(run.accepted, run.rejected, spec)
    run.qc = _trust_dict(report)
    run.qc["rows"] = [{"id": r.id, "verdict": r.verdict, "reasons": list(r.reasons)} for r in run.rows]
    if not run.accepted:
        run.reasons.append("no design produced a usable run: there is nothing to put in a dataset")
        run.status = Status.REFUSED
        run.steps.append({"step": "dataset", "ok": False, "why": run.reasons[-1]})
        run.wall_s = time.perf_counter() - started
        return run
    if report.decisao == "REJECT":
        run.reasons += [f"quality control rejected the dataset: {m}" for m in report.motivos]
        run.status = Status.REFUSED
        run.steps.append({"step": "dataset", "ok": False, "why": "the quality control rejected it"})
        run.wall_s = time.perf_counter() - started
        return run

    ref: DatasetRef = writer.write(run.accepted, spec, out)
    run.dataset = ref.to_dict()
    run.steps.append({"step": "dataset", "ok": True, "rows": ref.n_rows, "json": ref.json})
    tell({"event": "dataset", "rows": ref.n_rows, "path": ref.json})
    run.status = Status.DONE if report.decisao == "APPROVED" else Status.INCONCLUSIVE
    if run.status == Status.INCONCLUSIVE:
        run.notes += [f"the dataset was written but the quality control says REVIEW: {m}" for m in report.motivos]
    run.wall_s = time.perf_counter() - started
    return run


def _evaluate(spec: DesignStudy, solver: Any, geo: Geometry, qc: FactoryQc, design: Mapping[str, float], *,
              purpose: str, index: int, seed: int, workdir: str) -> FactoryRow:
    """One design: geometry, checks, solver, quality control, provenance. Never invents an output."""
    from pinneapple_data.provenance import build_provenance, data_hash_of_bytes, solver_uses_a_mesh
    from pinneapple_data.provenance import data_hash as file_hash

    os.makedirs(workdir, exist_ok=True)
    row = FactoryRow(id=f"{purpose}{index:03d}", purpose=purpose,
                     params={k: float(design[k]) for k in spec.base.names},
                     started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.tool_params = spec.evaluator.tool_params(row.params)

    t0 = time.perf_counter()
    try:
        # the design is the variables; the base's ``fixed`` are the parameters the case never varies (a 10-diameter
        # inlet run, the diameter of the product's case), and a geometry that does not know one refuses the row
        row.artifact = geo.prepare({**dict(spec.base.fixed), **row.params}, workdir)
    except ValueError as exc:                      # the geometry does not know a parameter, or refuses the design
        row.artifact = GeometryArtifact(ref=str(spec.base.ref),
                                        params={**dict(spec.base.fixed), **row.params}, status="refused",
                                        checks=[Check("geometry", "contradicts", critical=True, reason=str(exc))])
    if not row.artifact.ok:
        row.message = "; ".join(row.artifact.reasons) or "the geometry refused the design"
        row.checks = list(row.artifact.checks)
        row.verdict, row.reasons = row_verdict(row.checks)         # a refused design is rejected, with its reason
    else:
        issues = _check(solver, row.tool_params)
        if issues:
            outcome = SimOutcome({}, False, message="; ".join(issues))
            row.message = outcome.message
        else:
            try:
                outcome = solver.run(row.tool_params, workdir)
            except Exception as exc:  # noqa: BLE001 - a solver that raises is a failed run, not a missing one
                outcome = SimOutcome({}, False, message=f"{type(exc).__name__}: {exc}")
        row.outputs = {k: float(v) for k, v in (outcome.outputs or {}).items()
                       if isinstance(v, (int, float)) and k in spec.evaluator.output_names}
        row.ok = bool(outcome.ok) and len(row.outputs) == len(spec.evaluator.output_names)
        row.message = row.message or outcome.message
        row.cost_s = float(getattr(outcome, "cost_s", 0.0) or (time.perf_counter() - t0))
        row.checks = qc.row_checks(row.artifact, outcome, spec)
        row.verdict, row.reasons = row_verdict(row.checks)
        mesh = dict(getattr(outcome, "meta", {}) or {}).get("mesh")
        cells = (mesh or {}).get("cells") if isinstance(mesh, Mapping) else None
        stl = row.artifact.files.get("stl")
        digest = file_hash(stl) if stl and os.path.exists(stl) else data_hash_of_bytes(
            json.dumps({"design": row.params, "tool": row.tool_params, "seed": seed,
                        "geometry": row.artifact.params}, sort_keys=True).encode())
        row.provenance = build_provenance(
            spec.evaluator.tool, seed=seed, data_hash=digest, mesh=({"cells": int(cells)} if cells else None),
            app="pinneapple_autolab.factory", study_id=spec.digest(),
            geometry=f"{row.artifact.ref}:{row.artifact.status}",
            uses_a_mesh=bool(solver_uses_a_mesh(spec.evaluator.tool) or cells),
            case=f"{os.path.basename(workdir)}/")
    if row.verdict == REJECTED and not row.reasons:
        row.reasons = [row.message or "refused"]
    return row


def _check(solver: Any, params: Mapping[str, Any]) -> list[str]:
    """What the solver's own ``check`` says about this design (a solver that raises is a refusal with its message)."""
    try:
        return [str(r) for r in (solver.check(params) or [])]
    except Exception as exc:  # noqa: BLE001
        return [f"{type(exc).__name__}: {exc}"]
