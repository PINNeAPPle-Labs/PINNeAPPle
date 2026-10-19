"""pinneapple_core.workflows: DAG, runs, skip-done, retries, resume, executors, adapters, datasets."""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from pinneapple_core.workflows import (
    Adapter,
    DockerExecutor,
    ExecResult,
    Executor,
    LocalExecutor,
    SlurmExecutor,
    Step,
    Workflow,
    WorkflowError,
    WorkflowRunner,
    dataset_from_run,
    fill,
    get_executor,
    register_adapter,
    register_executor,
    step_hash,
    workflow_from_cases,
)

HERE = Path(__file__).resolve()
FRD = HERE.parents[2] / "apps" / "simulation_comparator" / "examples" / "calculix_cantilever_results.frd"
CALLS: dict[str, int] = {}


def record(kw, ctx):
    CALLS[ctx.step.id] = CALLS.get(ctx.step.id, 0) + 1
    (ctx.workdir / "out.txt").write_text(str(kw.get("x")))
    return {"y": 2 * kw.get("x", 0), "upstream": sorted(ctx.upstream)}


def flaky(kw, ctx):
    CALLS["flaky"] = CALLS.get("flaky", 0) + 1
    if CALLS["flaky"] < kw["succeed_on"]:
        raise RuntimeError(f"transient failure {CALLS['flaky']}")
    return {"attempt": ctx.attempt}


def boom(kw, ctx):
    raise RuntimeError("diverged")


def py(step_id, needs=(), **kw):
    return Step(step_id, "python", {"function": f"{__name__}:record", "kwargs": kw}, needs=list(needs))


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()


def test_dag_order_cycles_and_unknown_steps():
    wf = Workflow("w", [py("c", ["a", "b"]), py("a"), py("b", ["a"])])
    assert wf.order() == ["a", "b", "c"]
    assert wf.downstream("a") == {"b", "c"}
    with pytest.raises(WorkflowError, match="cycle"):
        Workflow("w", [py("a", ["b"]), py("b", ["a"])])
    with pytest.raises(WorkflowError, match="unknown"):
        Workflow("w", [py("a", ["zzz"])])
    with pytest.raises(WorkflowError, match="duplicate"):
        Workflow("w", [py("a"), py("a")])
    with pytest.raises(WorkflowError, match="invalid"):
        Workflow("w", [py("a b")])


def test_json_round_trip_and_unknown_keys(tmp_path):
    wf = Workflow("w", [py("a", x=1), py("b", ["a"], x=2)], max_parallel=2)
    assert Workflow.load(wf.save(tmp_path / "w.json")).to_dict() == wf.to_dict()
    d = wf.to_dict()
    d["steps"][0]["nedds"] = []
    with pytest.raises(WorkflowError, match="unknown keys"):
        Workflow.from_dict(d)


def test_fill_and_workflow_from_cases():
    assert fill({"a": "{x}", "b": "v={x:.1f}", "c": ["{y}"], "d": "{missing}"}, {"x": 2.0, "y": "s"}) == \
        {"a": 2.0, "b": "v=2.0", "c": ["s"], "d": "{missing}"}
    cases = [{"case_id": "A 1", "x": 1}, {"case_id": "B", "x": 2}]
    wf = workflow_from_cases("w", cases, [py("prep", x="{x}"), py("solve", ["prep"])],
                             final=[py("all", ["solve"])])
    assert wf.ids == ["A_1.prep", "A_1.solve", "B.prep", "B.solve", "all"]
    assert wf.step("B.prep").params["kwargs"]["x"] == 2 and wf.step("B.solve").needs == ["B.prep"]
    assert wf.step("all").needs == ["A_1.solve", "B.solve"] and wf.step("B.solve").group == "B"


def test_run_outputs_logs_and_skip_done_by_hash(tmp_path):
    runner = WorkflowRunner(tmp_path)
    wf = Workflow("w", [py("a", x=3), py("b", ["a"], x=4)])
    run = runner.run(wf)
    assert run["status"] == "done" and run["counts"]["done"] == 2
    steps = {s["step_id"]: s for s in runner.steps(run["id"])}
    assert steps["b"]["outputs"] == {"y": 8, "upstream": ["a"]} and steps["a"]["attempts"] == 1
    assert "done in" in runner.log(run["id"], "a")
    assert (runner.step_dir(run["id"], "a") / "out.txt").read_text() == "3"

    again = runner.run(wf)                                 # same content: nothing runs
    assert again["counts"]["skipped"] == 2 and CALLS == {"a": 1, "b": 1}
    assert "same content hash" in runner.log(again["id"], "b")
    changed = Workflow("w", [py("a", x=5), py("b", ["a"], x=4)])
    third = runner.run(changed)                            # a changed, so b's hash changes too
    assert third["counts"]["done"] == 2 and CALLS == {"a": 2, "b": 2}


def test_hash_covers_inputs_and_upstream(tmp_path):
    f = tmp_path / "mesh.txt"
    f.write_text("v1")
    s = Step("a", "python", {"function": "m:f"}, inputs=[str(f)])
    h1 = step_hash(s, {})
    f.write_text("v2")
    assert step_hash(s, {}) != h1
    t = Step("b", "python", {}, needs=["a"])
    assert step_hash(t, {"a": "x"}) != step_hash(t, {"a": "y"})


def test_retries_then_success_and_failure_cancels_downstream(tmp_path):
    runner = WorkflowRunner(tmp_path)
    wf = Workflow("w", [Step("f", "python", {"function": f"{__name__}:flaky", "kwargs": {"succeed_on": 3}},
                             retries=2)])
    run = runner.run(wf)
    st = runner.steps(run["id"])[0]
    assert run["status"] == "done" and st["attempts"] == 3 and st["outputs"] == {"attempt": 3}
    assert runner.log(run["id"], "f").count("failed: RuntimeError") == 2

    wf = Workflow("w", [Step("bad", "python", {"function": f"{__name__}:boom"}), py("after", ["bad"]),
                        py("independent", x=1)])
    run = runner.run(wf)
    st = {s["step_id"]: s for s in runner.steps(run["id"])}
    assert run["status"] == "failed" and st["bad"]["status"] == "failed" and "diverged" in st["bad"]["error"]
    assert st["after"]["status"] == "cancelled" and st["independent"]["status"] == "done"

    ff = Workflow("w", [Step("bad", "python", {"function": f"{__name__}:boom"}), py("later", x=9)],
                  fail_fast=True)
    st = {s["step_id"]: s["status"] for s in runner.steps(runner.run(ff)["id"])}
    assert st == {"bad": "failed", "later": "cancelled"}


def test_resume_reruns_only_unfinished_steps(tmp_path):
    runner = WorkflowRunner(tmp_path, cache=False)
    wf = Workflow("w", [py("a", x=1), Step("f", "python", {"function": f"{__name__}:flaky",
                                                           "kwargs": {"succeed_on": 2}}, needs=["a"])])
    run = runner.run(wf)
    assert run["status"] == "failed"
    runner.db.conn.execute("UPDATE steps SET status='running' WHERE step_id='f'")   # as a crash leaves it
    resumed = runner.run(run_id=run["id"])
    assert resumed["status"] == "done" and CALLS["a"] == 1 and CALLS["flaky"] == 2
    assert "interrupted" in runner.log(run["id"], "f")
    assert runner.steps(run["id"])[1]["attempts"] == 2


def test_cancel_kills_running_command(tmp_path):
    runner = WorkflowRunner(tmp_path)
    stop = threading.Event()
    wf = Workflow("w", [Step("sleep", "command", {"argv": [sys.executable, "-c", "import time; time.sleep(30)"]}),
                        py("after", ["sleep"])])
    threading.Timer(0.5, stop.set).start()
    t0 = time.time()
    run = runner.run(wf, should_stop=stop.is_set)
    assert time.time() - t0 < 15 and run["status"] == "cancelled"
    assert {s["step_id"]: s["status"] for s in runner.steps(run["id"])} == {"sleep": "cancelled", "after": "cancelled"}


def test_command_step_outputs_json_and_timeout(tmp_path):
    runner = WorkflowRunner(tmp_path)
    code = "import json; json.dump({'r': 1.5}, open('outputs.json', 'w')); print('hello')"
    wf = Workflow("w", [Step("c", "command", {"argv": [sys.executable, "-c", code]}),
                        Step("t", "command", {"argv": [sys.executable, "-c", "import time; time.sleep(10)"]},
                             timeout=0.5)])
    run = runner.run(wf)
    st = {s["step_id"]: s for s in runner.steps(run["id"])}
    assert st["c"]["outputs"]["r"] == 1.5 and "hello" in runner.log(run["id"], "c")
    assert st["t"]["status"] == "failed" and "time limit" in st["t"]["error"]


def test_local_executor_env_and_exit_code(tmp_path):
    lines = []
    res = LocalExecutor().run([sys.executable, "-c", "import os,sys; print(os.environ['FOO']); sys.exit(3)"],
                              cwd=tmp_path, env={"FOO": "bar"}, log=lines.append)
    assert res.returncode == 3 and not res.ok and "bar" in lines
    assert LocalExecutor().run(["/no/such/binary"], cwd=tmp_path, log=lines.append).returncode == 127


def test_docker_command_line(tmp_path):
    ex = DockerExecutor(image="opencfd/openfoam-default:2406", memory="1g", cpus=2, user="1000:1000",
                        mounts={str(tmp_path): "/case-template"})
    cmd = ex.command_line(["bash", "-c", "blockMesh"], tmp_path, env={"OMP_NUM_THREADS": "1"})
    assert cmd[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert f"{tmp_path.resolve()}:/work" in cmd and "--memory" in cmd and cmd[cmd.index("--cpus") + 1] == "2"
    assert f"{tmp_path.resolve()}:/case-template:ro" in cmd and "OMP_NUM_THREADS=1" in cmd
    assert cmd[-4:] == ["opencfd/openfoam-default:2406", "bash", "-c", "blockMesh"]
    with pytest.raises(ValueError):
        DockerExecutor().command_line(["x"], tmp_path)


def test_slurm_dry_run_writes_sbatch_script(tmp_path):
    runner = WorkflowRunner(tmp_path / "r")
    wf = Workflow("w", [Step("solve", "command", {"argv": ["simpleFoam", "-parallel"]}, executor="slurm",
                             executor_options={"partition": "cpu", "ntasks": 8, "mem": "16G",
                                               "modules": ["openfoam/2406"], "account": "proj1"}, timeout=5400)])
    run = runner.run(wf)
    assert run["status"] == "done"
    script = (runner.step_dir(run["id"], "solve") / "step.sbatch").read_text()
    assert script.startswith("#!/bin/bash\n")
    for line in ("#SBATCH --partition=cpu", "#SBATCH --ntasks=8", "#SBATCH --mem=16G", "#SBATCH --account=proj1",
                 "#SBATCH --time=01:30:00", "module load openfoam/2406", "srun simpleFoam -parallel"):
        assert line in script
    assert runner.steps(run["id"])[0]["outputs"]["dry_run"] is True
    assert SlurmExecutor(dry_run=False).available()[0] == bool(shutil.which("sbatch"))


def test_executor_and_adapter_plugins(tmp_path):
    class Recorder(Executor):
        name = "recorder"
        seen: list = []

        def run(self, argv, *, cwd, env=None, timeout=None, log=print, should_stop=lambda: False):
            Recorder.seen.append(list(argv))
            return ExecResult(0, 0.0, list(argv))

    class Gui(Adapter):
        kind = "gui_test"

        def run(self, ctx):
            ctx.execute(["click", ctx.params["button"]])
            return {"clicked": ctx.params["button"]}

    register_executor("recorder", Recorder)
    register_adapter(Gui())
    with pytest.raises(TypeError):
        register_executor("bad", dict)
    run = WorkflowRunner(tmp_path).run(Workflow("w", [Step("g", "gui_test", {"button": "Mesh"},
                                                           executor="recorder")]))
    assert run["status"] == "done" and Recorder.seen == [["click", "Mesh"]]
    assert isinstance(get_executor("recorder"), Recorder)


def _fake_bin(directory: Path, name: str, body: str) -> None:
    p = directory / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


@pytest.mark.skipif(os.name == "nt", reason="shell scripts as stand-in solvers")
def test_calculix_adapter_with_a_stand_in_ccx(tmp_path, monkeypatch):
    """The adapter's plumbing (copy the deck, run ccx, read the .frd) with a stand-in ccx that copies a real
    CalculiX result file."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "ccx", f'test -f "$2.inp" || exit 2\ncp "{FRD}" "$2.frd"\necho "Job finished"\n')
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    deck = tmp_path / "beam.inp"
    deck.write_text("*NODE\n")
    run = WorkflowRunner(tmp_path / "r").run(Workflow("w", [Step("solve", "calculix", {"inp": str(deck)})]))
    out = WorkflowRunner(tmp_path / "r").steps(run["id"])[0]["outputs"]
    assert run["status"] == "done" and out["nodes"] == 1025 and out["max_displacement"] > 0


@pytest.mark.skipif(os.name == "nt", reason="shell scripts as stand-in solvers")
def test_openfoam_adapter_edits_case_and_reads_log(tmp_path, monkeypatch):
    from pinneapple_core.workflows.templates import CASES_DIR
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "blockMesh", "echo 'End'\n")
    _fake_bin(bindir, "icoFoam", "mkdir -p 5\necho 'Courant Number mean: 0.1 max: 0.8'\necho 'Time = 5'\n"
                                 "echo 'End'\n")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("WM_PROJECT_DIR", str(tmp_path))          # the stand-ins' environment counts as loaded
    monkeypatch.delenv("FOAM_BASHRC", raising=False)
    step = Step("solve", "openfoam", {"case": str(CASES_DIR / "cavity"), "solver": "icoFoam", "edits": [
        {"file": "0/U", "key": "boundaryField/movingWall/value", "value": "uniform (1.7 0 0)"}]})
    runner = WorkflowRunner(tmp_path / "r")
    run = runner.run(Workflow("w", [step]))
    assert run["status"] == "done", runner.log(run["id"], "solve")
    case = runner.step_dir(run["id"], "solve") / "case"
    assert "uniform (1.7 0 0)" in (case / "0" / "U").read_text()
    assert runner.steps(run["id"])[0]["outputs"]["latest_time"] == 5.0
    assert "uniform (1 0 0)" in (CASES_DIR / "cavity" / "0" / "U").read_text()   # template untouched


def test_openfoam_bashrc_is_found_when_the_environment_is_not_loaded(tmp_path, monkeypatch):
    """Distribution packages put blockMesh and icoFoam on PATH, but they only run with the OpenFOAM environment."""
    from pinneapple_core.workflows import adapters
    rc = tmp_path / "etc" / "bashrc"
    rc.parent.mkdir()
    rc.write_text("export WM_PROJECT_DIR=x\n")
    own = tmp_path / "own_bashrc"
    own.write_text("")
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)
    monkeypatch.delenv("FOAM_BASHRC", raising=False)
    monkeypatch.setattr(adapters, "_FOAM_BASHRCS", (str(tmp_path / "e*" / "bashrc"),))
    assert adapters._foam_bashrc({}) == str(rc)                           # an installation in the usual places
    assert adapters._foam_bashrc({"bashrc": str(own)}) == str(own)        # the step's own comes first
    monkeypatch.setenv("FOAM_BASHRC", str(own))
    assert adapters._foam_bashrc({}) == str(own)
    monkeypatch.delenv("FOAM_BASHRC")
    monkeypatch.setenv("WM_PROJECT_DIR", "/loaded")
    assert adapters._foam_bashrc({}) is None                              # already loaded: nothing to source

    if os.name == "nt":
        return
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "blockMesh", "echo End\n")
    _fake_bin(bindir, "icoFoam", "echo End\n")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("WM_PROJECT_DIR")
    monkeypatch.setattr(adapters, "_FOAM_BASHRCS", ())
    ok, why = adapters.OpenFOAMAdapter().available()
    assert not ok and "environment is not loaded" in why


def test_unavailable_solver_fails_the_step_honestly(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    run = WorkflowRunner(tmp_path / "r").run(Workflow("w", [Step("s", "calculix", {"inp": "x.inp"})]))
    st = WorkflowRunner(tmp_path / "r").steps(run["id"])[0]
    assert st["status"] == "failed" and "not on PATH" in st["error"]


def test_template_cases_dataset_and_case_checks(tmp_path):
    pytest.importorskip("pyarrow")
    from pinneapple_core.workflows import TEMPLATES
    from pinneapple_data.contracts import load, read_table
    t = TEMPLATES["slab_heat"]
    cases = t.example_cases(4, seed=1)
    runner = WorkflowRunner(tmp_path / "r")
    run = runner.run(t.workflow(cases, max_parallel=2))
    assert run["status"] == "done" and run["counts"]["done"] == 5
    spec, path = dataset_from_run(runner, run["id"], cases, output_step="solve", path=tmp_path / "slab.parquet",
                                  skeleton=t.space)
    df, contract = read_table(path)
    assert spec.status == "complete" and len(df) == 4 and contract == load(str(path) + ".contract.json")
    assert "T_mid" in spec.outputs and spec.names("id") == ["case_id"] and spec.verify_files(tmp_path) == []
    assert df["T_mid"].between(300, 500).all() and spec.provenance.run_ids[0].startswith(run["id"])
    assert spec.parameter_space == t.space

    bad = cases.copy()
    bad.loc[0, "L"] = 5.0
    assert "outside" in t.check_cases(bad)[0]
    bad = cases.astype({"alpha": object})
    bad.loc[1, "alpha"] = "1e-5; rm -rf /"
    assert "non-numeric" in t.check_cases(bad)[0]
    assert "missing" in t.check_cases(cases.drop(columns=["t_end"]))[0]
    with pytest.raises(ValueError):
        t.workflow(bad)


def test_partial_dataset_when_a_case_fails(tmp_path):
    cases = pd.DataFrame({"case_id": ["a", "b"], "x": [1, 2]})
    tmpl = [Step("solve", "python", {"function": f"{__name__}:record", "kwargs": {"x": "{x}"}})]
    wf = workflow_from_cases("w", cases.to_dict("records"), tmpl, id_column="case_id")
    wf.steps[1].params["function"] = f"{__name__}:boom"
    runner = WorkflowRunner(tmp_path / "r")
    run = runner.run(wf)
    spec, path = dataset_from_run(runner, run["id"], cases, output_step="solve", path=tmp_path / "d.csv")
    df = pd.read_csv(path)
    assert spec.status == "partial" and df["status"].tolist() == ["done", "failed"] and df["y"].isna().tolist() == \
        [False, True]
    assert json.loads(Path(str(path) + ".contract.json").read_text())["files"][0]["path"] == "d.csv"


@pytest.mark.skipif(not shutil.which("ccx"), reason="CalculiX (ccx) not installed")
def test_calculix_template_against_euler_bernoulli(tmp_path):
    """A real ccx run per case: the tip deflection of the C3D8I cantilever is within 2 % of F L^3 / (3 E I)
    (slender beams: shear deformation adds well under 1 %)."""
    from pinneapple_core.workflows import TEMPLATES
    t = TEMPLATES["beam_calculix"]
    runner = WorkflowRunner(tmp_path)
    run = runner.run(t.workflow(t.example_cases(2, seed=3)))
    assert run["status"] == "done", run
    for s in runner.steps(run["id"]):
        if s["step_id"].endswith(".compare"):
            assert abs(s["outputs"]["fe_over_beam"] - 1.0) < 0.02, s["outputs"]
