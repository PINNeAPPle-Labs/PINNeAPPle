"""Kubernetes and cloud-batch executors behind the existing executor contract (#1479).

**What is exercised here and what is not.** There is no Kubernetes cluster in this environment, so ``kubectl`` is
replaced by a stand-in script that speaks the four verbs the executor uses (``apply``, ``get job``, ``get pods``,
``logs``, ``delete``) and actually runs the container's command in the step's directory. What the tests exercise is
everything on this side of the wire: the Job manifest the executor builds, the submission, the status polling, the
container exit code read from the pod status, the log streaming, and the deletion on cancel. What they do **not**
exercise, and the issue records: a real cluster accepting the manifest, the image, the PVC mount, RBAC, the scheduler
and network policies.

The batch provider is a test double for the same reason: no cloud account here. The double is a real
:class:`BatchProvider` subclass, so the executor's protocol is what runs.
"""
from __future__ import annotations

import os
import stat
import sys

import pytest

from pinneapple_core.workflows import Adapter, Step, Workflow, WorkflowRunner, register_adapter
from pinneapple_core.workflows.executors import EXECUTORS, get_executor
from pinneapple_core.workflows.executors_remote import (
    BATCH_PROVIDERS,
    BatchProvider,
    register_batch_provider,
)

KUBECTL = r'''
import json, os, shlex, subprocess, sys, time

args = []
rest = sys.argv[1:]
while rest:
    if rest[0] in ("--namespace", "--context", "--kubeconfig"):
        rest = rest[2:]
        continue
    args.append(rest.pop(0))

state = os.environ["FAKE_KUBECTL_STATE"]
os.makedirs(state, exist_ok=True)
with open(os.path.join(state, "calls.log"), "a", encoding="utf-8") as fh:
    fh.write(" ".join(args) + "\n")

def store(job, ext):
    return os.path.join(state, job + ext)

verb = args[0]
if verb == "apply":
    manifest = json.load(open(args[2]))
    job = manifest["metadata"]["name"]
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    command = container["command"] + container["args"]
    cwd = os.path.dirname(os.path.abspath(args[2]))
    # the "cluster" starts the container and returns; the exit code is written when it finishes, exactly as a pod
    # status appears only after the container ends
    script = " ".join(shlex.quote(c) for c in command) + " > %s 2>&1; echo $? > %s" % (
        shlex.quote(store(job, ".out")), shlex.quote(store(job, ".exit")))
    subprocess.Popen(["sh", "-c", script], cwd=cwd, start_new_session=True)
    json.dump({"manifest": manifest}, open(store(job, ".json"), "w"))
    print(json.dumps({"metadata": {"name": job}}))
elif verb == "get" and args[1] == "job":
    job = args[2]
    if not os.path.exists(store(job, ".exit")):
        print(json.dumps({"status": {"active": 1}}))
    else:
        code = int(open(store(job, ".exit")).read().strip() or 1)
        kind = "Complete" if code == 0 else "Failed"
        print(json.dumps({"status": {"conditions": [{"type": kind, "status": "True"}]}}))
elif verb == "get" and args[1] == "pods":
    job = [a for a in args if a.startswith("job-name=")][0].split("=", 1)[1]
    code = int(open(store(job, ".exit")).read().strip() or 1)
    print(json.dumps({"items": [{"status": {"containerStatuses": [
        {"state": {"terminated": {"exitCode": code}}}]}}]}))
elif verb == "logs":
    job = args[1].split("/", 1)[1]
    path = store(job, ".out")
    print(open(path).read() if os.path.exists(path) else "")
elif verb == "delete":
    job = args[2]
    open(os.path.join(state, "deleted-" + job), "w").close()
    print(json.dumps({"status": "deleted"}))
else:
    sys.exit(2)
'''


@pytest.fixture
def kubectl(tmp_path, monkeypatch):
    """A stand-in ``kubectl`` on PATH, with a state directory the test can read afterwards."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "kubectl"
    script.write_text(f"#!{sys.executable}\n{KUBECTL}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "kubectl-state"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KUBECTL_STATE", str(state))
    return state


def _writer(value: int) -> dict:
    """A command step that writes its own outputs.json, the way the CommandAdapter reads them back."""
    return {"argv": ["sh", "-c", f"printf '{{\"value\": {value}}}' > outputs.json"]}


def _calls(state) -> list[str]:
    path = state / "calls.log"
    return path.read_text().splitlines() if path.is_file() else []


# -------------------------------------------------------------------------------------------- the manifest
def test_the_job_manifest_is_a_single_job_with_the_directory_mounted(tmp_path):
    executor = get_executor("kubernetes", image="openfoam/openfoam10-paraview56:latest", artifacts_pvc="work",
                            namespace="sim", cpu="4", memory="16Gi", gpu=1)
    manifest = executor.job_manifest(["openfoam", "-c"], tmp_path / "solve", env={"N": "4"}, name="job-1")
    assert manifest["apiVersion"] == "batch/v1" and manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "job-1" and manifest["metadata"]["namespace"] == "sim"
    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"].startswith("openfoam/")
    assert container["command"] == ["openfoam"] and container["args"] == ["-c"]
    assert container["workingDir"] == "/work"
    assert container["volumeMounts"][0]["mountPath"] == "/work"
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "work"
    assert container["resources"]["limits"] == {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": 1}
    assert container["env"] == [{"name": "N", "value": "4"}]
    assert manifest["spec"]["backoffLimit"] == 0
    assert set(EXECUTORS) >= {"kubernetes", "batch"}, "the executor contract is the one that already existed"


def test_dry_run_writes_the_manifest_and_submits_nothing(tmp_path, kubectl):
    executor = get_executor("kubernetes", image="img", artifacts_pvc="work", dry_run=True)
    result = executor.run(["echo", "hi"], cwd=tmp_path)
    assert result.dry_run and (tmp_path / "step.job.json").is_file()
    assert _calls(kubectl) == [], "a dry run does not touch the cluster"


# -------------------------------------------------------------------------------------------- the whole DAG
def test_the_same_dag_runs_on_local_and_on_kubernetes_with_equal_outputs(tmp_path, kubectl):
    def run(executor: str, root: str) -> tuple[dict, WorkflowRunner]:
        runner = WorkflowRunner(tmp_path / root)
        workflow = Workflow("same", [
            Step("write", "command", _writer(21), executor=executor,
                 executor_options={"image": "img", "artifacts_pvc": "work"} if executor == "kubernetes" else {}),
            Step("double", "command", _writer(42), needs=["write"], executor=executor,
                 executor_options={"image": "img", "artifacts_pvc": "work"} if executor == "kubernetes" else {}),
        ])
        return runner.run(workflow), runner

    local_run, local_runner = run("local", "local")
    kube_run, kube_runner = run("kubernetes", "kube")
    assert local_run["status"] == kube_run["status"] == "done"
    assert local_runner.steps(local_run["id"])[-1]["outputs"] == kube_runner.steps(kube_run["id"])[-1]["outputs"]
    assert kube_runner.steps(kube_run["id"])[-1]["outputs"]["value"] == 42
    assert sum("apply" in line for line in _calls(kubectl)) == 2, "one Job per step"


def test_a_failing_step_reports_the_container_exit_code(tmp_path, kubectl):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("boom", [Step("bad", "command", {"argv": ["sh", "-c", "echo nope >&2; exit 5"]},
                                      executor="kubernetes",
                                      executor_options={"image": "img", "artifacts_pvc": "work"})])
    run = runner.run(workflow)
    assert run["status"] == "failed"
    assert "exited with code 5" in runner.steps(run["id"])[0]["error"]
    finished = [e for e in runner.events_of(run["id"]) if e["event"] == "step.finished"][0]
    assert finished["exit_code"] == 5, "the exit code comes from the pod status, not from 'the job failed'"


def _stop_once_submitted(tmp_path, state, timeout: float = 10.0):
    """A stop signal that arrives only after the cluster job exists, which is when cancelling means deleting it."""
    import threading
    import time

    flag = tmp_path / "stop"
    deadline = time.time() + timeout

    def watch():
        while time.time() < deadline and not flag.exists():
            calls = state / "calls.log"
            if calls.is_file() and "apply" in calls.read_text():
                flag.write_text("stop")
                return
            time.sleep(0.02)

    threading.Thread(target=watch, daemon=True).start()
    return lambda: flag.exists()


def test_cancelling_deletes_the_job_and_leaves_no_orphan(tmp_path, kubectl):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("cancel", [Step("long", "command", {"argv": ["sleep", "2"]}, executor="kubernetes",
                                        executor_options={"image": "img", "artifacts_pvc": "work",
                                                          "poll_seconds": 0.02})])
    run = runner.run(workflow, should_stop=_stop_once_submitted(tmp_path, kubectl))
    assert run["status"] == "cancelled"
    calls = _calls(kubectl)
    assert any(line.startswith("delete job") for line in calls), "cancellation is a job deletion"
    assert list(kubectl.glob("deleted-*")), "the cluster confirmed the deletion"


def test_a_step_that_cannot_run_remotely_fails_naming_the_step_and_the_reason(tmp_path, kubectl):
    class GuiStep(Adapter):
        kind = "gui-test"
        local_only = True

        def run(self, ctx):                             # pragma: no cover - it never gets here remotely
            return {}

    register_adapter(GuiStep())
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("gui", [Step("click", "gui-test", {}, executor="kubernetes",
                                     executor_options={"image": "img", "artifacts_pvc": "work"})])
    run = runner.run(workflow)
    assert run["status"] == "failed"
    error = runner.steps(run["id"])[0]["error"]
    assert "step 'click' cannot run on executor 'kubernetes'" in error and "GUI" in error
    assert not any(line.startswith("apply") for line in _calls(kubectl)), "nothing was submitted"

    # and the same step runs on this machine
    local = Workflow("gui-local", [Step("click", "gui-test", {})])
    assert runner.run(local)["status"] == "done"


def test_a_kubernetes_step_without_a_mount_says_so_instead_of_timing_out(tmp_path, kubectl):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("nomount", [Step("s", "command", _writer(1), executor="kubernetes",
                                         executor_options={"image": "img"})])
    run = runner.run(workflow)
    assert run["status"] == "failed"
    assert "artifacts_pvc" in runner.steps(run["id"])[0]["error"]
    assert _calls(kubectl) == [], "the refusal happens before anything is submitted"


# -------------------------------------------------------------------------------------------- cloud batch
class FakeBatch(BatchProvider):
    """A batch service in-process: it starts the command and answers status/logs/cancel while it runs."""

    name = "fake"
    submitted: list[dict] = []
    cancelled: list[str] = []
    stop: object = None                                  # a callable that exists() once the job is submitted

    def submit(self, job):
        import subprocess

        proc = subprocess.Popen(job.argv, cwd=job.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        FakeBatch.submitted.append({"job": job, "proc": proc})
        if callable(FakeBatch.stop):
            FakeBatch.stop()
        return f"{job.name}-id"

    def status(self, job_id):
        entry = FakeBatch.submitted[-1]
        if entry["proc"].poll() is None:
            return "running", None
        return ("done" if entry["proc"].returncode == 0 else "failed"), entry["proc"].returncode

    def logs(self, job_id):
        proc = FakeBatch.submitted[-1]["proc"]
        return (proc.stdout.read() if proc.stdout else "").splitlines()

    def cancel(self, job_id):
        FakeBatch.cancelled.append(job_id)
        FakeBatch.submitted[-1]["proc"].terminate()
        return True


@pytest.fixture
def fake_batch():
    FakeBatch.submitted = []
    FakeBatch.cancelled = []
    FakeBatch.stop = None
    register_batch_provider("fake", FakeBatch)
    try:
        yield FakeBatch
    finally:
        BATCH_PROVIDERS.pop("fake", None)


def test_cloud_batch_runs_a_step_through_its_provider(tmp_path, fake_batch):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("batch", [Step("w", "command", _writer(7), executor="batch",
                                       executor_options={"provider": "fake", "uploads": "s3://bucket", "image": "img",
                                                         "poll_seconds": 0.01})])
    run = runner.run(workflow)
    assert run["status"] == "done"
    assert runner.steps(run["id"])[0]["outputs"]["value"] == 7
    assert fake_batch.submitted and fake_batch.submitted[0]["job"].image == "img"
    assert fake_batch.submitted[0]["job"].step == "w"


def test_a_batch_step_without_an_upload_is_refused_with_its_reason(tmp_path, fake_batch):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("batch", [Step("w", "command", _writer(7), executor="batch",
                                       executor_options={"provider": "fake"})])
    run = runner.run(workflow)
    assert run["status"] == "failed"
    assert "uploads" in runner.steps(run["id"])[0]["error"]
    assert fake_batch.submitted == []


def test_cancelling_a_batch_job_cancels_it_through_the_provider(tmp_path, fake_batch):
    runner = WorkflowRunner(tmp_path / "runner")
    workflow = Workflow("batch", [Step("w", "command", {"argv": ["sleep", "2"]}, executor="batch",
                                       executor_options={"provider": "fake", "uploads": "s3://b",
                                                         "poll_seconds": 0.01})])
    flag = tmp_path / "stop"
    fake_batch.stop = lambda: flag.write_text("stop")
    run = runner.run(workflow, should_stop=lambda: flag.exists())
    assert run["status"] == "cancelled"
    assert fake_batch.cancelled, "the provider was asked to cancel the job"
    assert fake_batch.cancelled[0].startswith("pinneapple-w-")


def test_an_unregistered_batch_provider_is_refused_by_name(tmp_path):
    executor = get_executor("batch", provider="nope")
    ok, why = executor.available()
    assert not ok and "nope" in why and "command" in why


def test_the_command_batch_provider_drives_a_cli(tmp_path):
    from pinneapple_core.workflows.executors_remote import BatchJob, CommandBatchProvider

    provider = CommandBatchProvider(submit="printf '%s\\n' my-job-1", status="printf '%s' COMPLETED",
                                    logs="printf 'line one\\nline two\\n'", cancel="true")
    job = BatchJob(name="j", argv=["echo", "x"], cwd=str(tmp_path))
    job_id = provider.submit(job)
    assert job_id == "my-job-1"
    assert provider.status(job_id) == ("done", 0)
    assert provider.logs(job_id) == ["line one", "line two"]
    assert provider.cancel(job_id) is True
    assert CommandBatchProvider().available()[0] is False, "no submit command means no provider"
