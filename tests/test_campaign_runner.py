"""The campaign runner (issue #295): cache by content, retries with classification, backends, accounting.

Everything here runs against a fake solver — the point of the issue is the queue, not a solver — and the two
properties that make a campaign cheap are asserted as numbers: a proven cache hit executes nothing, and a campaign
of thousands of cases with repeated content executes one run per distinct content, not one per case. The SLURM and
cloud backends are exercised through a fake command runner and a fake service: their submit/poll contract is
tested, the vendor's answer is not (that needs a cluster, and the docs say so).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pinneapple_orchestration.campaign import (
    PERMANENT_FAILURES,
    TRANSIENT_FAILURES,
    Campaign,
    Case,
    CaseGeometry,
    CloudBatchBackend,
    CommandResult,
    Job,
    LocalBackend,
    Outcome,
    RunCache,
    SlurmBackend,
    aws_batch_backend,
    case_key,
    classify_failure,
    run,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MESH_FAILURE_LOG = """
/*--------------------------------*- C++ -*----------------------------------*/
Build  : v2406
Exec   : snappyHexMesh -overwrite
Date   : Nov 24 2026
Time   : 09:00:00
Host   : ws

Create time

Snapping stage

--> FOAM FATAL ERROR:
Shell refinement: found 3 cells with negative volume after snapping.
Please check the surface or reduce the feature level.

    From function Foam::snappyLayerDriver::mergePatchFacesUndo()
    in file layerAdditionRemoval/snappyLayerDriver.C at line 980.

FOAM exiting
"""

DIVERGED_LOG = """
Build  : v2406
Exec   : simpleFoam
Create time

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 0.0001, Final residual = 1e-06, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.0001, Final residual = 1e-06, No Iterations 3

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 0.4, No Iterations 100
smoothSolver:  Solving for Uy, Initial residual = 0.9, Final residual = 0.8, No Iterations 100

Time = 3

smoothSolver:  Solving for Ux, Initial residual = 900.0, Final residual = 400.0, No Iterations 100
smoothSolver:  Solving for Uy, Initial residual = 1500.0, Final residual = 800.0, No Iterations 100
"""

OK_LOG = """
Build  : v2406
Exec   : simpleFoam
Create time

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 0.1, Final residual = 1e-06, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.1, Final residual = 1e-06, No Iterations 3

SIMPLE solution converged in 42 iterations

End
"""


class CountingSolver:
    """A fake solver: fast, deterministic, and it counts how many times it actually ran."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def run(self, params, workdir):
        self.calls.append(dict(params))
        return {"drag": 2.0 * params.get("aoa", 0.0) + 1.0, "lift": params.get("aoa", 0.0)}


def digest(tag: str) -> CaseGeometry:
    return CaseGeometry(digest=hashlib.sha256(tag.encode()).hexdigest())


def cases(n, *, geometries=1, params_key="aoa"):
    return [Case(id=f"c{i:04d}", params={params_key: i % 5}, geometry=digest(f"g{i % geometries}"))
            for i in range(n)]


# --------------------------------------------------------------------------- content addressing
def test_the_key_is_the_hash_of_geometry_settings_conditions_and_solver(tmp_path):
    geo = tmp_path / "g.stl"
    geo.write_text("solid g\nendsolid\n")
    base = Case(id="a", params={"aoa": 3.0}, settings={"mesh": "fine"}, conditions={"U": 30.0}, geometry=str(geo))
    same = Case(id="b", params={"aoa": 3.0}, settings={"mesh": "fine"}, conditions={"U": 30.0}, geometry=str(geo))
    other_param = Case(id="c", params={"aoa": 4.0}, settings={"mesh": "fine"}, conditions={"U": 30.0}, geometry=str(geo))
    other_cond = Case(id="d", params={"aoa": 3.0}, settings={"mesh": "fine"}, conditions={"U": 31.0}, geometry=str(geo))
    assert case_key(base, solver="fake") == case_key(same, solver="fake")
    assert case_key(base, solver="fake") != case_key(other_param, solver="fake")
    assert case_key(base, solver="fake") != case_key(other_cond, solver="fake")
    assert case_key(base, solver="fake") != case_key(base, solver="other")

    before = case_key(base, solver="fake")
    geo.write_text("solid g2\nendsolid\n")                  # the geometry is the bytes, not the file name
    assert case_key(base, solver="fake") != before


def test_a_geometry_directory_is_hashed_by_its_whole_tree(tmp_path):
    d = tmp_path / "geom"
    d.mkdir()
    (d / "wall.stl").write_text("solid wall\nendsolid\n")
    first = case_key(Case(id="a", geometry=str(d)), solver="fake")
    (d / "inlet.stl").write_text("solid inlet\nendsolid\n")   # a patch added to the geometry
    assert case_key(Case(id="a", geometry=str(d)), solver="fake") != first


# --------------------------------------------------------------------------- cache
def test_a_proven_cache_hit_executes_nothing(tmp_path):
    cache = tmp_path / "cache"
    solver = CountingSolver()
    first = run(cases(4), solver, cache_dir=cache, workers=2)
    assert len(solver.calls) == 4 and first.report.n_executed == 4 and first.report.n_cached == 0

    solver = CountingSolver()
    second = run(cases(4), solver, cache_dir=cache, workers=2)
    assert solver.calls == []                               # nothing re-executed
    assert second.report.n_cached == 4 and second.report.n_executed == 0
    assert all(r["ok"] and r["cached"] for r in second.rows)
    assert second.rows[0]["outputs"] == first.rows[0]["outputs"]


def test_a_corrupt_cache_record_is_a_miss_not_a_hit(tmp_path):
    cache = tmp_path / "cache"
    solver = CountingSolver()
    run(cases(1), solver, cache_dir=cache, workers=1)
    key = case_key(cases(1)[0], solver="fake")
    record = cache / key[:2] / f"{key}.json"
    raw = json.loads(record.read_text())
    raw["outputs"] = {"drag": 999.0}                        # hand-edited: the outputs no longer match their hash
    record.write_text(json.dumps(raw))

    solver = CountingSolver()
    again = run(cases(1), solver, cache_dir=cache, workers=1)
    assert len(solver.calls) == 1                            # redone, not trusted
    assert again.report.n_cached == 0 and again.rows[0]["outputs"]["drag"] != 999.0


def test_the_cache_refuses_a_record_whose_key_is_not_the_one_asked_for(tmp_path):
    store = RunCache(tmp_path)
    store.store(Outcome("c1", "k" * 64, 1, True, {"a": 1.0}), solver="fake")
    assert store.lookup("k" * 64) is not None and store.hits == 1
    assert store.lookup("0" * 64) is None and store.misses == 1


# --------------------------------------------------------------------------- retries and classification
def test_transient_failures_are_retried_and_deterministic_ones_are_not():
    assert set(TRANSIENT_FAILURES) == {"timeout", "crashed", "unknown"}
    assert set(PERMANENT_FAILURES) == {"mesh", "diverged", "setup"}
    assert not set(TRANSIENT_FAILURES) & set(PERMANENT_FAILURES)


def test_a_flaky_solver_succeeds_on_a_later_attempt_and_the_row_says_so(tmp_path):
    tries = {"n": 0}

    def flaky(params, workdir):
        tries["n"] += 1
        if tries["n"] < 3:
            raise RuntimeError("transient: the machine was busy")
        return {"drag": 1.5}

    result = run([Case(id="c0", params={"aoa": 1})], flaky, workers=1, max_attempts=3)
    assert result.rows[0]["ok"] and result.rows[0]["attempt"] == 3
    assert result.report.n_retried == 2


def test_a_mesh_failure_is_recorded_with_its_log_and_never_retried(tmp_path):
    def meshy(params, workdir):
        (Path(workdir) / "job.log").write_text(MESH_FAILURE_LOG)
        raise RuntimeError("snappyHexMesh failed")

    result = run([Case(id="c0", params={"aoa": 1})], meshy, workers=1, max_attempts=3)
    row = result.rows[0]
    assert not row["ok"] and row["failure"] == "mesh" and row["attempt"] == 1
    assert "negative volume" in row["log_excerpt"]           # the excerpt is in the row, not lost with the workdir
    assert result.report.failures == {"mesh": 1} and result.report.failure_rate == 1.0


def test_a_diverging_run_is_classified_from_its_residual_history():
    assert classify_failure(DIVERGED_LOG) == "diverged"
    assert classify_failure(OK_LOG) == "crashed"             # it ran and reported no failure: the exit is the error
    assert classify_failure(MESH_FAILURE_LOG) == "mesh"
    assert classify_failure("", timed_out=True) == "timeout"
    assert classify_failure("", error="boom") == "unknown"


def test_a_setup_error_is_told_apart_from_bad_geometry():
    io_error = """
--> FOAM FATAL IO ERROR:
Cannot find file "points" in directory "constant/polyMesh"
file: constant/polyMesh/points at line 0.
FOAM exiting
"""
    assert classify_failure(io_error) == "setup"


def test_a_run_that_outlives_its_timeout_is_a_timeout_and_the_case_does_not_vanish(tmp_path):
    def slow(params, workdir):
        time.sleep(2.0)
        return {"drag": 1.0}

    result = run([Case(id="c0", params={"aoa": 1}), Case(id="c1", params={"aoa": 2})], slow,
                 workers=2, timeout_s=0.2, max_attempts=1)
    assert [r["failure"] for r in result.rows] == ["timeout", "timeout"]
    assert result.report.n_failed == 2 and result.report.n_cases == 2


def test_a_failed_run_is_a_row_with_its_error_and_never_a_dropped_case(tmp_path):
    def half(params, workdir):
        if params["aoa"] == 0:
            raise ValueError("geometry refused")
        return {"drag": 0.0}

    result = run(cases(4), half, workers=2)
    assert len(result.rows) == 4 and {r["case_id"] for r in result.rows} == {f"c{i:04d}" for i in range(4)}
    assert sum(1 for r in result.rows if not r["ok"]) == 1
    assert "geometry refused" in next(r for r in result.rows if not r["ok"])["error"]


# --------------------------------------------------------------------------- thousands of cases
def test_two_thousand_cases_with_repeated_content_execute_once_per_content(tmp_path):
    """The point of content addressing: 2000 runs, 25 distinct (geometry, parameter) pairs, 25 executions."""
    solver = CountingSolver()
    batch = [Case(id=f"c{i:04d}", params={"aoa": i % 5}, geometry=digest(f"g{(i // 5) % 5}")) for i in range(2000)]
    assert len({case_key(c, solver="fake") for c in batch}) == 25

    t0 = time.perf_counter()
    result = run(batch, solver, cache_dir=tmp_path / "cache", workers=8, name="big")
    elapsed = time.perf_counter() - t0
    assert len(solver.calls) == 25, len(solver.calls)
    assert len(result.rows) == 2000 and all(r["ok"] for r in result.rows)
    assert result.report.n_executed == 25 and result.report.n_shared == 1975   # the distinct contents, and only them
    assert result.report.n_cached == 1975 and result.report.cpu_hours > 0
    assert {r["source"] for r in result.rows} == {"executed", "shared"}

    solver = CountingSolver()
    again = run(batch, solver, cache_dir=tmp_path / "cache", workers=8, name="big")
    assert solver.calls == [] and again.report.n_cached == 2000              # the second executes nothing
    assert elapsed < 120                                                     # the fake solver, not a benchmark


def test_the_queue_runs_in_parallel_by_the_workers_it_was_given(tmp_path):
    def slowish(params, workdir):
        time.sleep(0.2)
        return {"drag": 1.0}

    batch = [Case(id=f"c{i}", params={"aoa": i}) for i in range(8)]
    serial = run(batch, slowish, workers=1).report.wall_s
    parallel = run(batch, slowish, workers=4).report.wall_s
    assert parallel < serial / 2


# --------------------------------------------------------------------------- resume and accounting
def test_the_report_accounts_cpu_hours_wall_time_failures_and_the_machine(tmp_path):
    def sometimes(params, workdir):
        time.sleep(0.05)
        if params["aoa"] == 0:
            raise RuntimeError("nope")
        return {"drag": 1.0}

    result = run(cases(10), sometimes, workers=2, cores_per_run=4, max_attempts=1)
    rep = result.report
    assert rep.n_cases == 10 and rep.n_failed == 2 and rep.failure_rate == pytest.approx(0.2)
    executed = [r for r in result.rows if r["source"] == "executed"]
    assert rep.cpu_hours == pytest.approx(sum(r["wall_s"] for r in executed) * 4 / 3600, abs=1e-5)
    assert rep.machine["cpu_count"] >= 1 and rep.machine["system"] and rep.machine["hostname"]
    assert rep.backend == "local" and rep.cores_per_run == 4
    assert rep.started and rep.ended and rep.wall_s > 0


def test_saving_writes_results_the_report_and_a_lineage_chain(tmp_path):
    result = run(cases(3), CountingSolver(), workers=2, name="saved")
    paths = result.save(tmp_path / "out")
    assert set(paths) == {"results", "report", "lineage"}
    saved = json.loads(open(paths["results"]).read())
    assert len(saved["rows"]) == 3 and saved["schema"] == "pinneapple.campaign/1"
    lineage = json.loads(open(paths["lineage"]).read())
    ids = {a["id"] for a in lineage["artifacts"]}
    assert "dataset:campaign" in ids and "run:c0000" in ids
    assert any(a["id"].startswith("geometry:") for a in lineage["artifacts"])
    run_art = next(a for a in lineage["artifacts"] if a["id"] == "run:c0000")
    assert run_art["kind"] == "simulation" and run_art["software"] == "fake"


def test_a_campaign_killed_mid_run_resumes_with_what_was_written(tmp_path):
    """The interruption that matters is the process dying (a session ends, the machine reboots).

    Each finished run is written to the cache atomically, so the campaign that comes after reads what was proven and
    executes only the rest. The kill happens in a separate process because a worker thread cannot be killed (the
    campaign catches everything inside a thread, by design: a failed run is a row, never a lost case).
    """
    cache = tmp_path / "cache"
    script = tmp_path / "die.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from pinneapple_orchestration.campaign import Case, run\n"
        "counter = {'n': 0}\n"
        "def solver(params, workdir):\n"
        "    counter['n'] += 1\n"
        "    if counter['n'] > 10:\n"
        "        os._exit(3)                      # nothing can catch this\n"
        "    return {'drag': 1.0}\n"
        "solver.name = 'fake'\n"
        f"run([Case(id=f'c{{i}}', params={{'aoa': i}}) for i in range(40)], solver, cache_dir={str(cache)!r},\n"
        "    workers=1)\n")
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 3, p.stderr
    written = len(list(cache.rglob("*.json")))
    assert 5 <= written <= 12, written                        # the finished runs, and only those

    batch = [Case(id=f"c{i}", params={"aoa": i}) for i in range(40)]
    solver = CountingSolver()
    result = run(batch, solver, cache_dir=cache, workers=4, name="resume")
    assert len(solver.calls) == 40 - written
    assert result.report.n_cached == written and result.report.n_failed == 0


# --------------------------------------------------------------------------- backends beyond local
class FakeSlurm:
    """A cluster that answers like SLURM: ``sbatch`` hands out ids, ``squeue`` reports a state and then drops it.

    The state per job index is programmed so a success and a failure can be exercised in one campaign. Every command
    is recorded, so the test reads the header the campaign actually wrote instead of trusting that it wrote one.
    """

    def __init__(self, state_by_index=None):
        self.calls = []
        self.state_by_index = state_by_index or {}
        self.jobs = {}

    def __call__(self, argv, *, cwd=None, timeout=None):
        self.calls.append((list(argv), cwd))
        if argv[0] == "sbatch":
            idx = len(self.jobs)
            jid = str(1000 + idx)
            self.jobs[jid] = {"script": open(argv[-1]).read(), "cwd": cwd,
                              "state": self.state_by_index.get(idx, "COMPLETED"), "polls": 0}
            return CommandResult(list(argv), 0, f"{jid};cluster\n")
        if argv[0] == "squeue":
            lines = []
            for jid in argv[argv.index("-j") + 1].split(","):
                info = self.jobs[jid]
                info["polls"] += 1
                if info["polls"] == 1:                  # first poll: the state; after that it is out of the queue
                    lines.append(f"{jid}|{info['state']}")
            return CommandResult(list(argv), 0, "\n".join(lines))
        return CommandResult(list(argv), 127, "", "unknown command")


def test_the_slurm_backend_submits_one_job_per_case_with_the_configured_cores(tmp_path):
    fake = FakeSlurm(state_by_index={1: "FAILED"})
    batch = [Case(id=f"c{i}", params={"aoa": i}, command=["bash", "-c", "echo run"]) for i in range(3)]
    backend = SlurmBackend(runner=fake, cores_per_run=8, poll_s=0.01, header=["#SBATCH --qos=short"])
    result = Campaign(batch, None, backend=backend, cores_per_run=8, max_attempts=1,
                      workroot=tmp_path / "w").run()
    scripts = [info["script"] for info in fake.jobs.values()]
    assert len(scripts) == 3
    assert all("#SBATCH --cpus-per-task=8" in s and "#SBATCH --qos=short" in s for s in scripts)
    assert all("echo run" in s for s in scripts)
    assert result.report.backend == "slurm"
    assert result.report.n_failed == 1 and result.report.failures == {"crashed": 1}
    assert result.report.n_executed == 3


def test_a_backend_that_cannot_submit_records_the_failure_instead_of_losing_the_case(tmp_path):
    def broken(argv, *, cwd=None, timeout=None):
        return CommandResult(list(argv), 1, "", "sbatch: error: invalid partition")

    batch = [Case(id="c0", params={"aoa": 0}, command=["true"])]
    result = Campaign(batch, None, backend=SlurmBackend(runner=broken, poll_s=0.01),
                      workroot=tmp_path / "w").run()
    assert result.rows[0]["ok"] is False and "sbatch" in result.rows[0]["error"]


def test_the_cloud_backend_speaks_the_same_submit_poll_contract(tmp_path):
    state = {}

    def submit(job):
        state[job.case.id] = 2                       # two polls before it finishes, as a real service would
        return f"job-{job.case.id}"

    def poll(ids):
        out = {}
        for j in ids:
            cid = j.split("-", 1)[1]
            state[cid] -= 1
            out[j] = "SUCCEEDED" if state[cid] <= 0 else "RUNNING"
        return out

    backend = CloudBatchBackend(submit, poll, poll_s=0.01)
    batch = [Case(id="c0", params={"aoa": 0}), Case(id="c1", params={"aoa": 1})]
    result = Campaign(batch, CountingSolver(), backend=backend, workroot=tmp_path / "w").run()
    assert len(state) == 2 and result.report.n_executed == 2
    assert all(r["ok"] for r in result.rows) and result.report.backend == "cloud"


def test_the_aws_batch_adapter_builds_the_cli_calls_it_says_it_does(tmp_path):
    calls = []

    def runner(argv, *, cwd=None, timeout=None):
        calls.append(list(argv))
        if "submit-job" in argv:
            return CommandResult(list(argv), 0, json.dumps({"jobId": "abc-123"}))
        return CommandResult(list(argv), 0, json.dumps({"jobs": [{"jobId": "abc-123", "status": "SUCCEEDED"}]}))

    backend = aws_batch_backend(job_queue="q", job_definition="def", runner=runner, region="eu-central-1",
                                poll_s=0.01)
    case = Case(id="c0", params={"aoa": 0}, command=["solver", "-case", "x"])
    job = Job(case, "k" * 64, tmp_path / "w", command=["solver", "-case", "x"])
    out = list(backend.run([job]))
    assert out and out[0].ok
    assert calls[0][:3] == ["aws", "batch", "--region"] and "submit-job" in calls[0]
    assert "describe-jobs" in calls[-1] and "abc-123" in calls[-1]
    overrides = calls[0][calls[0].index("--container-overrides") + 1]
    assert json.loads(overrides)["command"] == ["solver", "-case", "x"]


def test_one_case_id_twice_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="unique"):
        Campaign([Case(id="a"), Case(id="a")], CountingSolver())


def test_a_local_command_job_writes_its_outputs_and_is_cached_by_content(tmp_path):
    script = tmp_path / "solve.py"
    script.write_text("import json, sys\n"
                      "json.dump({'drag': float(sys.argv[1])}, open('outputs.json', 'w'))\n")
    batch = [Case(id="c0", params={"aoa": 1.0}, command=[sys.executable, str(script), "7.5"])]
    result = run(batch, None, cache_dir=tmp_path / "cache", workers=1)
    assert result.rows[0]["ok"] and result.rows[0]["outputs"] == {"drag": 7.5}

    again = run(batch, None, cache_dir=tmp_path / "cache", workers=1)          # cached, no process started
    assert again.rows[0]["cached"] and again.rows[0]["outputs"] == {"drag": 7.5}
