"""Engineering GUI agent: run a task in engineering software and deliver the files plus the evidence of what
happened ("Selenium for computational engineering").

* :mod:`.workflow`: the structured workflow (inputs with bounds, steps, preconditions, expected artifacts,
  confirmations, acceptance criteria), loaded from YAML/JSON.
* :mod:`.interpret`: a task (sentence or structured) to a committed workflow and its inputs; Ollama optional.
* :mod:`.adapters`: one adapter per application: FreeCAD and CfdOF (Python API inside the FreeCAD GUI), OpenFOAM
  (command line), ParaView (pvpython), ``verify`` (acceptance criteria with ``pinneapple_check``), ``web``
  (Playwright).
* :mod:`.display`: Xvfb virtual display, screenshots, GUI drivers (xdotool; Playwright; pywinauto for Windows).
* :mod:`.runner`: executes the steps, stops on failure or when a step needs a person's confirmation, records every
  command, log, artifact (SHA-256), screenshot and check.
* :mod:`.report`: self-contained HTML evidence report and the StudyResult JSON.

    from pinneapple_tools.automation import Runner, catalog, interpret
    it = interpret("channel with a 55 mm obstacle at 20 mm/s", catalog())
    run = Runner(it.workflow, "runs/r1", inputs=it.inputs).run(approve=["solve"], approver="me")
    print(run.state, run.message)          # report.html and study_result.json are in runs/r1
"""
from __future__ import annotations

from pathlib import Path

from .adapters import (
    ActionFailed,
    ActionResult,
    Adapter,
    StepContext,
    get_adapter,
    known_actions,
    list_adapters,
)
from .display import (
    GuiUnavailable,
    PlaywrightDriver,
    PywinautoDriver,
    VirtualDisplay,
    XdotoolDriver,
)
from .interpret import Interpretation, interpret, parse_quantity
from .record import Check, Evidence, FileRecord, RunRecord, StepRecord
from .report import to_study_result, write_report, write_study_result
from .runner import Runner, plan
from .toolchain import Toolchain
from .verify import evaluate_criterion, mass_balance, patch_fluxes, patch_mean, pressure_drop
from .workflow import (
    CONFIRM_CATEGORIES,
    ArtifactSpec,
    Criterion,
    InputSpec,
    Step,
    Workflow,
    WorkflowError,
    render,
)

EXAMPLES = Path(__file__).resolve().parent / "examples"

__all__ = ["Workflow", "Step", "InputSpec", "ArtifactSpec", "Criterion", "WorkflowError", "CONFIRM_CATEGORIES",
           "render", "interpret", "Interpretation", "parse_quantity", "Runner", "plan", "RunRecord", "StepRecord",
           "Check", "Evidence", "FileRecord", "Toolchain", "Adapter", "ActionResult", "ActionFailed", "StepContext",
           "get_adapter", "list_adapters", "known_actions", "VirtualDisplay", "XdotoolDriver", "PlaywrightDriver",
           "PywinautoDriver", "GuiUnavailable", "write_report", "write_study_result", "to_study_result",
           "evaluate_criterion", "mass_balance", "pressure_drop", "patch_fluxes", "patch_mean",
           "EXAMPLES", "example_workflow", "catalog"]


def example_workflow(name: str) -> Path:
    """Path of a committed example workflow (``examples/<name>/workflow.yaml``)."""
    p = EXAMPLES / name / "workflow.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"no example workflow {name!r}; there are {sorted(catalog())}")
    return p


def catalog(folder: str | Path | None = None) -> dict[str, Workflow]:
    """Every workflow under ``folder`` (default: the committed examples), by id."""
    root = Path(folder) if folder else EXAMPLES
    out = {}
    for p in sorted(root.glob("*/workflow.y*ml")) + sorted(root.glob("*/workflow.json")):
        wf = Workflow.load(p)
        out[wf.id] = wf
    return out
