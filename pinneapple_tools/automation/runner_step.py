"""The GUI agent as a step of the Simulation Workflow Runner (``pinneapple_core.workflows``).

``register()`` adds the step kind ``guiagent``: one step of a runner workflow runs a whole automation workflow
(FreeCAD, CfdOF, OpenFOAM, ParaView, ...) in the step's directory and returns its verdict as outputs, so a runner DAG
can mesh and solve a CAD variant per case and use the acceptance criteria downstream::

    from pinneapple_core.workflows import Step, Workflow, WorkflowRunner
    from pinneapple_tools.automation.runner_step import register
    register()
    wf = Workflow(name="channel sweep", steps=[
        Step(id="h55", kind="guiagent", params={"workflow": "channel_obstacle",
                                                "inputs": {"obstacle_height_mm": 55}, "approve": ["solve"]})])
    WorkflowRunner("work/runner").run(wf)

Step parameters: ``workflow`` (id of a committed workflow, see :func:`~pinneapple_tools.automation.catalog`),
``inputs`` (checked against the workflow's bounds), ``approve`` (ids of the steps the workflow asks to confirm; whoever
writes the runner workflow approves them, and the record says so). A run that stops for a confirmation that was not
given, fails or is blocked makes the runner step fail with the agent's message. Outputs: ``state``, ``message``,
``passed`` (every acceptance criterion met), ``criteria`` (name -> met), ``metrics`` (the numbers of the
StudyResult), ``report`` and ``study_result`` (paths in the step directory).

Cancelling the runner step stops the agent at its next step boundary (a running solver finishes its step).
"""
from __future__ import annotations

from typing import Any

from pinneapple_core.workflows import Adapter, StepCancelled, StepFailed, register_adapter

from . import catalog
from .adapters import get_adapter
from .report import to_study_result
from .runner import Runner
from .toolchain import Toolchain
from .workflow import WorkflowError

__all__ = ["GuiAgentStep", "register"]


class GuiAgentStep(Adapter):
    kind = "guiagent"

    def __init__(self, toolchain: Toolchain | None = None):
        self._toolchain = toolchain

    @property
    def toolchain(self) -> Toolchain:
        if self._toolchain is None:
            self._toolchain = Toolchain.discover()
        return self._toolchain

    def available(self, step=None) -> tuple[bool, str]:
        """Without a step: whether every committed workflow can run here. With one: whether its workflow can."""
        wid = (getattr(step, "params", None) or {}).get("workflow") if step is not None else None
        cat = catalog()
        if wid is not None and wid not in cat:
            return False, f"no committed workflow {wid!r}"
        missing = set()
        for s in (st for w in ([cat[wid]] if wid else cat.values()) for st in w.steps):
            missing |= set(self.toolchain.missing(list(get_adapter(s.adapter).requires) + list(s.requires)))
        return (not missing, "needs " + ", ".join(sorted(missing)) if missing else "")

    def run(self, ctx) -> dict[str, Any]:
        params = ctx.params
        cat = catalog()
        wid = params.get("workflow")
        if wid not in cat:
            raise StepFailed(f"a guiagent step needs params.workflow, one of {sorted(cat)}")

        def on_event(e: dict[str, Any]) -> None:
            if e["event"] in ("step_started", "confirmation_needed", "step_blocked", "step_failed"):
                ctx.log(f"guiagent: {e['event']} {e.get('step', '')} {e.get('error') or ''}".rstrip())
            if ctx.should_stop():
                raise StepCancelled("cancelled")

        try:
            agent = Runner(cat[wid], ctx.workdir / "guiagent", inputs=params.get("inputs") or {},
                           toolchain=self.toolchain, on_event=on_event)
            rec = agent.run(approve=params.get("approve") or [], approver=f"runner workflow {ctx.run_id}")
        except WorkflowError as e:
            raise StepFailed(str(e)) from None
        if rec.state not in ("passed", "rejected"):
            raise StepFailed(f"guiagent {rec.state}: {rec.message}")
        sr = to_study_result(rec)
        return {"state": rec.state, "message": rec.message, "passed": rec.state == "passed",
                "criteria": {c.name: c.passed for c in rec.criteria}, "metrics": sr.metrics,
                "report": "guiagent/report.html", "study_result": "guiagent/study_result.json"}


def register(toolchain: Toolchain | None = None) -> GuiAgentStep:
    """Register the ``guiagent`` step kind with the workflow runner; returns the adapter."""
    adapter = GuiAgentStep(toolchain)
    register_adapter(adapter)
    return adapter
