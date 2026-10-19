"""Run a :class:`~pinneapple_tools.automation.workflow.Workflow` step by step, observe each step and stop on the
first failure or on the first step that needs a human decision.

For every step, in order:

1. skip it if an earlier run already passed it (a run folder can be resumed);
2. block if a step named in ``after`` has not passed, or if the application it needs is not installed;
3. stop with ``awaiting_confirmation`` if the step falls into a confirmation category (``expensive``,
   ``overwrite``, ``destructive``, ``external``) and was not approved: a step whose expected files already exist
   is an overwrite even if the workflow does not say so;
4. run the action through its adapter, with a virtual display if it drives a GUI;
5. check the artifacts it was expected to leave, hash every file it produced, keep its logs, commands, screenshots
   and outputs; the step passes only if the action returned and every check held.

``run.json`` in the run folder is rewritten after every change, so a UI can follow a run while it executes; when
the run stops, the evidence report (``report.html``) and the StudyResult (``study_result.json``) are written.

    from pinneapple_tools.automation import Runner, Workflow, example_workflow
    wf = Workflow.load(example_workflow("channel_obstacle"))
    run = Runner(wf, "runs/demo", inputs={"obstacle_height_mm": 50}).run()
    if run.state == "awaiting_confirmation":         # the solver step is "expensive"
        run = Runner(wf, "runs/demo").run(approve=["solve"], approver="j.doe")
"""
from __future__ import annotations

import platform
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapters import ActionFailed, StepContext, get_adapter, known_actions
from .display import GuiUnavailable, VirtualDisplay
from .record import Check, FileRecord, RunRecord, StepRecord, file_record
from .toolchain import Toolchain
from .workflow import CONFIRM_CATEGORIES, Criterion, Workflow, WorkflowError, render

__all__ = ["Runner", "plan"]


def plan(workflow: Workflow, toolchain: Toolchain | None = None) -> list[dict[str, Any]]:
    """What running the workflow would need, without running anything: per step its adapter, mode, whether the
    application is installed here, and the confirmations it will ask for."""
    tc = toolchain or Toolchain.discover()
    out = []
    for s in workflow.steps:
        try:
            ad = get_adapter(s.adapter)
            ok, why = ad.available(tc)
            mode = ad.mode
        except KeyError:
            ok, why, mode = False, f"no adapter {s.adapter!r}", "?"
        missing = tc.missing(s.requires)
        if missing:
            ok, why = False, "missing: " + ", ".join(missing)
        out.append({"id": s.id, "title": s.title or s.id, "action": s.action, "mode": mode, "available": ok,
                    "reason": why, "confirm": list(s.confirm), "after": list(s.after)})
    return out


class Runner:
    def __init__(self, workflow: Workflow, run_dir: str | Path, inputs: dict[str, Any] | None = None,
                 toolchain: Toolchain | None = None, on_event: Callable[[dict[str, Any]], None] | None = None,
                 report: bool = True):
        errors = workflow.validate(known_actions())
        if errors:
            raise WorkflowError(errors)
        self.workflow = workflow
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.toolchain = toolchain or Toolchain.discover()
        self.on_event = on_event
        self.report = report
        if (self.run_dir / "run.json").is_file():
            self.record = RunRecord.load(self.run_dir)
            if self.record.workflow.get("id") != workflow.id:
                raise WorkflowError([f"{self.run_dir} holds a run of {self.record.workflow.get('id')!r}"])
            if inputs and workflow.resolve_inputs(inputs) != self.record.inputs:
                raise WorkflowError(["a resumed run keeps its inputs; start a new run folder to change them"])
        else:
            self.record = RunRecord(run_id=self.run_dir.name, workflow=workflow.to_dict(),
                                    inputs=workflow.resolve_inputs(inputs), folder=str(self.run_dir),
                                    steps=[StepRecord(s.id, s.title or s.id, s.action,
                                                      mode=_mode(s.adapter)) for s in workflow.steps])
            self.record.save()

    # ------------------------------------------------------------------ helpers
    def _emit(self, kind: str, **kw: Any) -> None:
        self.record.save()
        if self.on_event:
            self.on_event({"event": kind, "run_id": self.record.run_id, **kw})

    def _template_ctx(self) -> dict[str, Any]:
        steps = {s.id: s.outputs for s in self.record.steps if s.state == "passed"}
        return {"inputs": self.record.inputs, "steps": steps, "workdir": str(self.run_dir),
                "example": str(self.workflow.folder or self.run_dir)}

    def _environment(self) -> dict[str, Any]:
        return {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                "tools": self.toolchain.versions(), "paths": self.toolchain.to_dict()}

    # ------------------------------------------------------------------ run
    def run(self, approve: Iterable[str] = (), approver: str = "", stop_before: str | None = None) -> RunRecord:
        """Run (or resume) until the end, a failure, a step awaiting confirmation, or ``stop_before``."""
        approve = set(approve)
        rec = self.record
        rec.state, rec.message, rec.finished = "running", "", None
        if not rec.environment:
            rec.environment = self._environment()
        self._emit("run_started")
        display: VirtualDisplay | None = None
        try:
            for step in self.workflow.steps:
                sr = rec.step(step.id)
                if sr.state == "passed":
                    continue
                if sr.state == "declined":
                    rec.state, rec.message = "declined", f"{step.id} was declined; start a new run"
                    break
                if step.id == stop_before:
                    rec.state, rec.message = "paused", f"stopped before {step.id} as asked"
                    break
                previous = {s.id: s for s in rec.steps}
                blocked = [a for a in step.after if previous[a].state != "passed"]
                adapter = get_adapter(step.adapter)
                ok, why = adapter.available(self.toolchain)
                missing = self.toolchain.missing(step.requires)
                if blocked or not ok or missing:
                    sr.state = "blocked"
                    sr.error = (f"needs {', '.join(blocked)} to pass first" if blocked else
                                why if not ok else "missing: " + ", ".join(missing))
                    rec.state, rec.message = "blocked", f"{step.id}: {sr.error}"
                    self._emit("step_blocked", step=step.id, error=sr.error)
                    break
                tctx = self._template_ctx()
                try:
                    params = render(step.params, tctx)
                    expects = [replace(a, path=render(a.path, tctx)) for a in step.expects]
                except KeyError as e:
                    sr.state, sr.error = "failed", str(e)
                    rec.state, rec.message = "failed", f"{step.id}: {e}"
                    self._emit("step_failed", step=step.id, error=sr.error)
                    break
                # ---- confirmation gate
                categories = list(step.confirm)
                reasons = [f"{c}: {CONFIRM_CATEGORIES[c]}" for c in step.confirm]
                existing = [a.path for a in expects if (self.run_dir / a.path).exists()]
                if existing and "overwrite" not in categories:
                    categories.append("overwrite")
                    reasons.append("overwrite: " + ", ".join(existing) + " already exist")
                if categories:
                    if step.id not in approve:
                        sr.state = "awaiting_confirmation"
                        sr.confirmation = {"categories": categories, "reasons": reasons, "approved_by": None,
                                           "at": None}
                        rec.state = "awaiting_confirmation"
                        rec.message = f"{step.id} needs confirmation: " + "; ".join(reasons)
                        self._emit("confirmation_needed", step=step.id, reasons=reasons)
                        break
                    sr.confirmation = {"categories": categories, "reasons": reasons,
                                       "approved_by": approver or "unnamed", "at": time.time()}
                # ---- run
                need_display = step.action.split(".", 1)[1] in adapter.needs_display
                if need_display and display is None:
                    try:
                        display = VirtualDisplay().start()
                    except GuiUnavailable as e:
                        sr.state, sr.error = "blocked", str(e)
                        rec.state, rec.message = "blocked", f"{step.id}: {e}"
                        self._emit("step_blocked", step=step.id, error=sr.error)
                        break
                if not self._run_step(step, sr, params, expects, display if need_display else None):
                    rec.state, rec.message = "failed", f"{step.id}: {sr.error}"
                    break
            else:
                failed = [c for c in rec.criteria if not c.passed]
                rec.state = "rejected" if failed else "passed"
                rec.message = "every step passed; " + (
                    f"{len(failed)} of {len(rec.criteria)} acceptance criteria not met: "
                    + "; ".join(c.name for c in failed) if failed else
                    f"all {len(rec.criteria)} acceptance criteria met")
        except BaseException as exc:                 # cancelled by the caller (e.g. from on_event) or interrupted
            for s in rec.steps:
                if s.state == "running":
                    s.state, s.ended = "failed", time.time()
                    s.error = f"interrupted: {type(exc).__name__}: {exc}"
            rec.state, rec.message = "failed", f"interrupted: {type(exc).__name__}: {exc}"
            raise
        finally:
            if display is not None:
                display.stop()
            if rec.state != "awaiting_confirmation" and rec.state != "paused":
                rec.finished = time.time()
            try:
                self._emit("run_stopped", state=rec.state, message=rec.message)
            except Exception:                        # the run has stopped; a failing listener must not hide that
                self.record.save()
            if self.report:
                from .report import write_report, write_study_result
                write_report(rec, self.run_dir / "report.html")
                write_study_result(rec, self.run_dir / "study_result.json")
                self.record.save()
        return rec

    def decline(self, step_id: str, by: str = "", reason: str = "") -> RunRecord:
        """A person refused the confirmation a step asked for: the step is ``declined``, the run stops there for good
        (state ``declined``) and the report says who declined and why."""
        rec = self.record
        sr = rec.step(step_id)
        if sr.state != "awaiting_confirmation":
            raise WorkflowError([f"{step_id} is {sr.state}, not awaiting confirmation"])
        sr.state = "declined"
        sr.confirmation = {**(sr.confirmation or {}), "declined_by": by or "unnamed", "at": time.time(),
                           "reason": reason}
        rec.state, rec.finished = "declined", time.time()
        rec.message = f"{step_id} declined by {by or 'unnamed'}" + (f": {reason}" if reason else "")
        self._emit("run_stopped", state=rec.state, message=rec.message)
        if self.report:
            from .report import write_report, write_study_result
            write_report(rec, self.run_dir / "report.html")
            write_study_result(rec, self.run_dir / "study_result.json")
        return rec

    def _run_step(self, step, sr: StepRecord, params: dict[str, Any], expects, display) -> bool:
        step_dir = self.run_dir / step.id
        step_dir.mkdir(parents=True, exist_ok=True)
        sr.state, sr.started, sr.ended, sr.error = "running", time.time(), None, None
        sr.commands, sr.logs, sr.artifacts, sr.evidence, sr.checks, sr.outputs, sr.actions = [], [], [], [], [], {}, []
        self._emit("step_started", step=step.id)
        ctx = StepContext(run_dir=self.run_dir, step_dir=step_dir, toolchain=self.toolchain, display=display,
                          workflow_dir=self.workflow.folder, previous={s.id: s for s in self.record.steps},
                          timeout_s=step.timeout_s,
                          criteria=[Criterion(c.id, c.kind, c.title, render(c.params, self._template_ctx()))
                                    for c in self.workflow.criteria] if step.action == "verify.criteria" else [],
                          on_progress=lambda m: self._emit("progress", step=step.id, message=m))
        result = None
        try:
            result = get_adapter(step.adapter).run(step.action.split(".", 1)[1], params, ctx)
        except ActionFailed as e:
            sr.error = str(e)
        except Exception as e:                                       # an adapter bug is a failed step, with trace
            sr.error = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-2000:]
        sr.commands, sr.logs = ctx.commands, ctx.logs
        if result is not None:
            sr.outputs = result.outputs
            sr.evidence = result.evidence
            sr.checks = list(result.checks)
            sr.actions = result.actions
            files: dict[str, FileRecord] = {}
            for p in result.artifacts:
                if Path(p).exists():
                    fr = file_record(Path(p), self.run_dir)
                    files[fr.path] = fr
            sr.artifacts = list(files.values())
        for ev in sr.evidence:
            p = self.run_dir / ev.path
            if p.is_file():
                ev.sha256 = file_record(p, self.run_dir).sha256
        if result is not None:
            for a in expects:
                p = self.run_dir / a.path
                size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else
                        p.stat().st_size) if p.exists() else 0
                ok = p.exists() and (p.is_dir() if a.kind == "dir" else p.is_file()) and size >= a.min_bytes
                sr.checks.append(Check(f"expected {a.kind} {a.path}", ok,
                                       f"{size} bytes" if p.exists() else "not produced", value=size,
                                       threshold=a.min_bytes, source="runner"))
                if ok and a.path not in {f.path for f in sr.artifacts}:
                    sr.artifacts.append(file_record(p, self.run_dir))
        sr.ended = time.time()
        failed_checks = [c for c in sr.checks if not c.passed]
        if step.action == "verify.criteria":
            self.record.criteria = list(sr.checks)
            failed_checks = []                                       # an unmet criterion is a result, not a crash
        if sr.error is None and failed_checks:
            sr.error = "; ".join(f"{c.name}: {c.detail}" for c in failed_checks)
        sr.state = "passed" if sr.error is None else "failed"
        self._emit("step_finished", step=step.id, state=sr.state, record=sr.to_dict())
        return sr.state == "passed"


def _mode(adapter: str) -> str:
    try:
        return get_adapter(adapter).mode
    except KeyError:
        return "?"

