"""pinneapple_tools.automation: workflow model, interpretation, the runner's gates and records, the evidence report,
the acceptance criteria on a real OpenFOAM result, and the OpenFOAM v1912 compatibility edits.

The end-to-end run of the committed example (FreeCAD + CfdOF + OpenFOAM + ParaView) is the last test; it runs only
where those applications are installed (the guiagent Docker image)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pinneapple_tools.automation import (
    ActionResult,
    Adapter,
    Check,
    Evidence,
    Runner,
    RunRecord,
    Toolchain,
    Workflow,
    WorkflowError,
    catalog,
    example_workflow,
    interpret,
    plan,
    render,
    to_study_result,
)
from pinneapple_tools.automation.adapters import register
from pinneapple_tools.automation.adapters.openfoam import (
    _old_esi,
    patch_cfdof_allmesh,
    patch_cfdof_inlet,
)
from pinneapple_tools.automation.verify import inflow, mass_balance, patch_fluxes, pressure_drop

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = ROOT / "examples" / "physics_check" / "pass" / "cases" / "channel"     # icoFoam result, U = 1 m/s inlet


# ------------------------------------------------------------------------------------------- a fake application
class _Fake(Adapter):
    name = "fake"
    actions = ("write", "fail", "check")

    def do_write(self, params, ctx):
        p = ctx.run_dir / params["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(params.get("text", "x"))
        img = ctx.step_dir / "shot.png"
        from PIL import Image
        Image.new("RGB", (20, 10), (200, 30, 30)).save(img)
        return ActionResult(outputs={"value": params.get("value", 1), "file": str(p)}, artifacts=[p],
                            evidence=[Evidence(ctx.rel(img), "plot", "a test image")])

    def do_fail(self, params, ctx):
        from pinneapple_tools.automation import ActionFailed
        raise ActionFailed("the application said no")

    def do_check(self, params, ctx):
        return ActionResult(checks=[Check("value positive", params["value"] > 0, f"value {params['value']}")])


register(_Fake())


def _wf(**over) -> Workflow:
    d = {"id": "demo", "title": "Demo", "inputs": {"h": {"default": 5, "min": 1, "max": 10, "unit": "mm"}},
         "steps": [{"id": "a", "action": "fake.write", "params": {"path": "a/out.txt", "value": "{inputs.h}"},
                    "expects": ["a/out.txt"]},
                   {"id": "b", "action": "fake.check", "after": ["a"], "params": {"value": "{steps.a.value}"}},
                   {"id": "c", "action": "fake.write", "after": ["b"], "confirm": ["expensive"],
                    "params": {"path": "c/out.txt"}}]}
    d.update(over)
    return Workflow.from_dict(d)


# ------------------------------------------------------------------------------------------------ workflow model
def test_workflow_validation_lists_every_problem():
    with pytest.raises(WorkflowError) as e:
        Workflow.from_dict({"id": "Bad Id", "title": "", "steps": [
            {"id": "s1", "action": "nodot", "after": ["s2"], "confirm": ["whenever"]},
            {"id": "s2", "action": "fake.write", "params": {"x": "{inputs.nope}", "y": "{steps.s9.out}"}},
            {"id": "s2", "action": "fake.write"}]})
    text = " | ".join(e.value.errors)
    for frag in ("lower_snake_case", "needs a title", "<adapter>.<action>", "earlier step", "whenever",
                 "unknown input", "s9", "used twice"):
        assert frag in text, frag


def test_templates_keep_types_and_inputs_are_bounded():
    ctx = {"inputs": {"h": 55.0}, "steps": {"a": {"n": 3}}, "workdir": "/w", "example": "/e"}
    assert render("{inputs.h}", ctx) == 55.0
    assert render({"p": "{example}/x.FCStd", "l": ["{steps.a.n}"]}, ctx) == {"p": "/e/x.FCStd", "l": [3]}
    with pytest.raises(KeyError):
        render("{steps.a.missing}", ctx)
    wf = _wf()
    assert wf.resolve_inputs() == {"h": 5}
    with pytest.raises(WorkflowError, match="outside"):
        wf.resolve_inputs({"h": 11})
    with pytest.raises(WorkflowError, match="unknown input"):
        wf.resolve_inputs({"other": 1})


def test_committed_example_is_valid_and_complete():
    wf = Workflow.load(example_workflow("channel_obstacle"))
    from pinneapple_tools.automation import known_actions
    assert wf.validate(known_actions()) == []
    actions = [s.action for s in wf.steps]
    assert actions == ["freecad.open", "freecad.macro", "cfdof.setup", "openfoam.mesh", "openfoam.solve",
                       "paraview.export", "verify.criteria"]
    assert wf.step("solve").confirm == ["expensive"]
    assert {c.kind for c in wf.criteria} == {"converged", "mass_balance", "inflow", "pressure_drop", "artifact"}
    assert (wf.folder / "channel_obstacle.FCStd").is_file() and (wf.folder / "set_parameter.FCMacro").is_file()
    assert set(catalog()) >= {"channel_obstacle"}
    assert all(p["id"] for p in plan(wf, Toolchain()))                     # plan never runs anything


# ------------------------------------------------------------------------------------------------- interpretation
def test_sentence_picks_the_workflow_and_converts_units():
    cat = catalog()
    it = interpret("Run the channel with the obstacle height set to 5.5 cm at an inlet velocity of 3 cm/s", cat)
    assert it.workflow.id == "channel_obstacle" and it.method == "rules"
    assert it.inputs == {"obstacle_height_mm": pytest.approx(55.0), "inlet_velocity_mm_s": pytest.approx(30.0)}
    with pytest.raises(WorkflowError, match="outside"):
        interpret("channel obstacle height 95 mm", cat)
    with pytest.raises(WorkflowError, match="no committed workflow"):
        interpret("bake a lemon cake", cat)


def test_structured_requests_and_llm_fallback():
    cat = catalog()
    it = interpret({"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 30}}, cat)
    assert it.method == "structured" and it.inputs["obstacle_height_mm"] == 30
    it = interpret(json.dumps({"workflow": "channel_obstacle"}), cat)
    assert it.inputs == {"obstacle_height_mm": 55, "inlet_velocity_mm_s": 20}

    class Down:
        url, model = "http://nowhere:11434", "m"

        def available(self):
            return False

    it = interpret("channel obstacle height 30 mm", cat, llm=Down())
    assert it.method == "rules" and any("LLM not used" in n for n in it.notes)

    class Up(Down):
        def available(self):
            return True

        def chat_json(self, system, user, schema):
            assert schema["properties"]["workflow"]["enum"] == sorted(cat)
            return {"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 70, "rm -rf": 1}}

    it = interpret("make the obstacle taller", cat, llm=Up())
    assert it.method == "llm" and it.inputs == {"obstacle_height_mm": 70}           # unknown keys dropped


# ------------------------------------------------------------------------------------------------------ the runner
def test_runner_stops_for_confirmation_then_resumes(tmp_path):
    wf = _wf()
    run = Runner(wf, tmp_path / "r", inputs={"h": 7}, toolchain=Toolchain()).run()
    assert run.state == "awaiting_confirmation"
    assert [s.state for s in run.steps] == ["passed", "passed", "awaiting_confirmation"]
    assert run.step("c").confirmation["categories"] == ["expensive"]
    assert not (tmp_path / "r" / "c" / "out.txt").exists()                  # nothing ran past the gate
    a = run.step("a")
    assert a.artifacts[0].path == "a/out.txt" and len(a.artifacts[0].sha256) == 64
    assert a.evidence[0].kind == "plot" and a.evidence[0].sha256
    assert run.step("b").checks[0].detail == "value 7"                    # template from step a's outputs
    assert (tmp_path / "r" / "report.html").is_file()

    run = Runner(wf, tmp_path / "r", toolchain=Toolchain()).run(approve=["c"], approver="tester")
    assert run.state == "passed" and run.step("c").confirmation["approved_by"] == "tester"
    from pinneapple_data.contracts import StudyResult, load, validate
    sr = json.loads((tmp_path / "r" / "study_result.json").read_text())
    assert validate(sr) == [] and sr["kind"] == "StudyResult" and sr["metadata"]["decision"] == "ACCEPTED"
    assert {a["path"] for a in sr["artifacts"]} >= {"a/out.txt", "c/out.txt"}
    contract = load(tmp_path / "r" / "study_result.json")
    assert isinstance(contract, StudyResult) and contract.passed and contract.status == "done"
    assert all(c.status == "PASS" for c in contract.checks) and contract.provenance.run_ids == [run.run_id]
    assert RunRecord.load(tmp_path / "r").to_dict() == run.to_dict()
    with pytest.raises(WorkflowError, match="keeps its inputs"):
        Runner(wf, tmp_path / "r", inputs={"h": 2}, toolchain=Toolchain())


def test_declined_confirmation_ends_the_run(tmp_path):
    run = Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).run()
    run = Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).decline("c", by="reviewer", reason="no budget")
    assert run.state == "declined" and run.step("c").state == "declined"
    assert "declined by reviewer: no budget" in (tmp_path / "r" / "report.html").read_text()
    run = Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).run(approve=["c"])
    assert run.state == "declined" and not (tmp_path / "r" / "c" / "out.txt").exists()
    with pytest.raises(WorkflowError, match="not awaiting"):
        Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).decline("a")


def test_an_interrupted_run_leaves_a_consistent_record(tmp_path):
    class Stop(Exception):
        pass

    def stop_at_b(e):
        if e["event"] == "step_started" and e["step"] == "b":
            raise Stop("cancel requested")

    with pytest.raises(Stop):
        Runner(_wf(), tmp_path / "r", toolchain=Toolchain(), on_event=stop_at_b).run()
    rec = RunRecord.load(tmp_path / "r")
    assert rec.state == "failed" and rec.step("a").state == "passed" and rec.step("b").state == "failed"
    assert "cancel requested" in rec.step("b").error and (tmp_path / "r" / "report.html").is_file()
    run = Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).run(approve=["c"])     # resumes from b
    assert run.state == "passed"


def test_existing_outputs_count_as_overwrite(tmp_path):
    (tmp_path / "r" / "a").mkdir(parents=True)
    (tmp_path / "r" / "a" / "out.txt").write_text("someone's file")
    run = Runner(_wf(), tmp_path / "r", toolchain=Toolchain()).run()
    assert run.step("a").state == "awaiting_confirmation"
    assert "overwrite" in run.step("a").confirmation["categories"]
    assert (tmp_path / "r" / "a" / "out.txt").read_text() == "someone's file"


def test_failures_stop_the_run_and_are_reported(tmp_path):
    wf = _wf(steps=[{"id": "a", "action": "fake.check", "params": {"value": -1}},
                    {"id": "b", "action": "fake.fail", "after": ["a"]}])
    run = Runner(wf, tmp_path / "r1", toolchain=Toolchain()).run()
    assert run.state == "failed" and run.step("a").state == "failed" and run.step("b").state == "pending"
    assert "value positive" in run.step("a").error

    wf = _wf(steps=[{"id": "b", "action": "fake.fail"}])
    run = Runner(wf, tmp_path / "r2", toolchain=Toolchain()).run()
    assert run.state == "failed" and "the application said no" in run.step("b").error
    html = (tmp_path / "r2" / "report.html").read_text()
    assert "the application said no" in html and "FAILED" in html
    sr = to_study_result(run)
    assert sr.metadata["decision"] == "INCOMPLETE" and sr.status == "failed" and not sr.passed

    wf = _wf(steps=[{"id": "m", "action": "fake.write", "params": {"path": "x"}, "expects": ["not/there.txt"]}])
    run = Runner(wf, tmp_path / "r3", toolchain=Toolchain()).run()
    assert run.step("m").state == "failed" and "not produced" in run.step("m").error


def test_missing_application_blocks_instead_of_failing(tmp_path):
    wf = _wf(steps=[{"id": "o", "action": "freecad.open", "params": {"document": "x.FCStd"}}])
    run = Runner(wf, tmp_path / "r", toolchain=Toolchain()).run()                # an empty toolchain
    assert run.state == "blocked" and "freecad" in run.step("o").error


# ------------------------------------------------------------------------------- Simulation Workflow Runner plugin
def test_guiagent_is_a_step_kind_of_the_workflow_runner(tmp_path, monkeypatch):
    import pinneapple_tools.automation.runner_step as rs
    from pinneapple_core.workflows import Step, WorkflowRunner, get_adapter
    from pinneapple_core.workflows import Workflow as RunnerWorkflow
    monkeypatch.setattr(rs, "catalog", lambda: {"demo": _wf()})
    rs.register(Toolchain())
    assert get_adapter("guiagent").available(Step(id="x", kind="guiagent", params={"workflow": "demo"})) == (True, "")
    runner = WorkflowRunner(tmp_path / "runner")
    ok = runner.run(RunnerWorkflow(name="t", steps=[Step(id="g", kind="guiagent", params={
        "workflow": "demo", "inputs": {"h": 3}, "approve": ["c"]})]))
    step = runner.steps(ok["id"])[0]
    assert ok["status"] == "done" and step["outputs"]["passed"] is True and step["outputs"]["state"] == "passed"
    assert (runner.step_dir(ok["id"], "g") / "guiagent" / "study_result.json").is_file()
    assert "approved_by" in json.dumps(RunRecord.load(runner.step_dir(ok["id"], "g") / "guiagent").to_dict())
    gated = runner.run(RunnerWorkflow(name="t", steps=[Step(id="g", kind="guiagent", params={"workflow": "demo"})]))
    assert gated["status"] != "done" and "confirmation" in runner.steps(gated["id"])[0]["error"]

    monkeypatch.setattr(rs, "catalog", lambda: catalog())
    rs.register(Toolchain())                                    # nothing installed: the runner says what is missing
    ok, why = get_adapter("guiagent").available(Step(id="x", kind="guiagent", params={"workflow": "channel_obstacle"}))
    assert not ok and "freecad" in why and "openfoam" in why
    assert not get_adapter("guiagent").available()[0]            # the runner's step-kind list says so too


# --------------------------------------------------------------------------------------- criteria on a real result
def test_patch_fluxes_on_an_openfoam_result():
    f = patch_fluxes(CHANNEL)                                                   # this also tests FV.patch_sums
    assert f["patches"]["inlet"] == pytest.approx(-0.1, rel=1e-9)               # 1 m/s through 0.1 m²
    assert f["patches"]["outlet"] == pytest.approx(0.1, rel=1e-6)
    assert f["source"] == "phi"
    assert mass_balance(CHANNEL, "inlet", "outlet", 0.01).passed
    assert inflow(CHANNEL, "inlet", 1.0).passed
    bad = inflow(CHANNEL, "inlet", 1.1)
    assert not bad.passed and bad.value == pytest.approx(1 / 11, rel=1e-6)
    chk, dp = pressure_drop(CHANNEL, "inlet", "outlet", density=1000.0, min_pa=0.0, max_pa=1000.0)
    assert dp > 0 and not chk.passed                                            # 2.8e3 Pa > 1000 Pa
    assert not mass_balance(CHANNEL, "outlet", "inlet").passed                  # wrong way round: no inflow


def test_patch_sums_accepts_boundary_or_all_faces():
    from pinneapple_check.fields import load_openfoam
    from pinneapple_check.fv import FV
    fd = load_openfoam(CHANNEL)
    fv = FV(fd)
    q = np.arange(len(fv.own), dtype=float)
    every, boundary = fv.patch_sums(q), fv.patch_sums(q[fv.ni:])
    assert every == boundary and every["inlet"] > 0


# --------------------------------------------------------------------------------- OpenFOAM v1912 compatibility
_ALLMESH = 'if [ ! -z $FOAM_API ] || [ $WM_PROJECT_VERSION -lt 9 ] 2>/dev/null\nthen\n    runCommand transformPoints\n'
_U = """boundaryField
{
    inlet
    {
        // Fix all three components of velocity on inflow and only the normal component on outflow,
        // in order to be well-posed if there are some faces on the patch which are actually outflows.
        type                fixedNormalInletOutletVelocity;
        fixTangentialInflow yes;
        normalVelocity
        {
            type            fixedValue;
            value           uniform ( 0.02 0 0 );
        }
        value               uniform ( 0.02 0 0 );
    }
    outlet
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
}
"""


def test_cfdof_compatibility_edits(tmp_path):
    (tmp_path / "Allmesh").write_text(_ALLMESH)
    assert "FOAM_API" in patch_cfdof_allmesh(tmp_path)
    assert '"$WM_PROJECT_VERSION" == v*' in (tmp_path / "Allmesh").read_text()
    assert patch_cfdof_allmesh(tmp_path) is None                                # idempotent
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text(_U)
    assert len(patch_cfdof_inlet(tmp_path)) == 1
    u = (tmp_path / "0" / "U").read_text()
    assert "fixedNormalInletOutletVelocity" not in u and "uniform ( 0.02 0 0 );" in u
    assert "pressureInletOutletVelocity" in u and patch_cfdof_inlet(tmp_path) == []
    assert _old_esi("v1912") and not _old_esi("v2406") and not _old_esi("11")


# ------------------------------------------------------------------------------------------- end to end (heavy)
_TC = Toolchain.discover()


@pytest.mark.skipif(bool(_TC.missing(["freecad", "cfdof", "openfoam", "gmsh", "paraview", "xvfb"])),
                    reason="FreeCAD, CfdOF, OpenFOAM, gmsh, ParaView and Xvfb are needed (guiagent image)")
def test_channel_obstacle_end_to_end(tmp_path):
    wf = Workflow.load(example_workflow("channel_obstacle"))
    run = Runner(wf, tmp_path / "run", inputs={"obstacle_height_mm": 55}).run()
    assert run.state == "awaiting_confirmation" and run.step("solve").state == "awaiting_confirmation"
    run = Runner(wf, tmp_path / "run").run(approve=["solve"], approver="pytest")
    assert run.state == "passed", run.message
    assert all(c.passed for c in run.criteria) and len(run.criteria) == 5
    assert any(e.kind == "screen" for e in run.step("open_project").evidence)
    assert (tmp_path / "run" / "postprocess" / "U_cells.csv").stat().st_size > 100_000
    shutil.rmtree(tmp_path / "run" / "cfd_setup", ignore_errors=True)
