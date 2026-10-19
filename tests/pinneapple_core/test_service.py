"""pinneapple_core.service: job queue, API keys, usage metering, uploads."""
from __future__ import annotations

import io
import threading
import time

import pytest

from pinneapple_core.service import (
    APIKeyStore,
    JobQueue,
    UploadRejected,
    UploadStore,
    UploadTooLarge,
    UsageLimitExceeded,
    UsageMeter,
    connect,
    data_dir,
    limits_from_env,
    safe_name,
)


def double(params, ctx):
    for i in range(3):
        ctx.progress(i / 3, f"step {i}")
        ctx.log(f"line {i}")
    (ctx.workdir / "out.txt").write_text("ok")
    return {"y": params["x"] * 2}


def slow(params, ctx):
    t0 = time.time()
    while time.time() - t0 < params.get("seconds", 10):
        ctx.progress(0.5)
        time.sleep(0.01)
    return {"done": True}


def boom(params, ctx):
    raise RuntimeError("solver diverged")


def sleeper_without_checks(params, ctx):
    time.sleep(params.get("seconds", 30))
    return {}


_FLAKY = {"n": 0}


def flaky(params, ctx):
    _FLAKY["n"] += 1
    if _FLAKY["n"] < 3:
        raise ValueError(f"transient {_FLAKY['n']}")
    return {"attempt": ctx.attempt}


@pytest.fixture
def queue(tmp_path):
    q = JobQueue(tmp_path / "q.sqlite", app="t", max_concurrent=2, poll_interval=0.02, heartbeat_interval=0.1)
    yield q
    q.stop()


def test_job_runs_with_progress_logs_and_workdir(queue):
    queue.register("double", double)
    job_id = queue.submit("double", {"x": 21}, owner="user:a")
    job = queue.wait(job_id, 10)
    assert job["status"] == "done" and job["result"] == {"y": 42} and job["progress"] == 1.0
    assert job["attempts"] == 1 and job["owner"] == "user:a"
    lines = [entry["line"] for entry in queue.logs(job_id)]
    assert "line 2" in lines and lines[-1] == "attempt 1 done"
    assert (queue.workdir / job_id / "out.txt").read_text() == "ok"
    seq = queue.logs(job_id)[1]["seq"]
    assert all(e["seq"] > seq for e in queue.logs(job_id, since=seq))
    assert queue.list(owner="user:a")[0]["id"] == job_id and queue.list(owner="user:b") == []


def test_failure_is_reported_and_manual_retry_requeues(queue):
    queue.register("boom", boom)
    job_id = queue.submit("boom")
    job = queue.wait(job_id, 10)
    assert job["status"] == "failed" and "solver diverged" in job["error"]
    assert any("RuntimeError" in e["line"] for e in queue.logs(job_id))
    queue.retry(job_id)
    assert queue.wait(job_id, 10)["attempts"] == 2
    queue.register("double", double)
    done = queue.wait(queue.submit("double", {"x": 1}), 10)
    with pytest.raises(ValueError, match="only failed or cancelled"):
        queue.retry(done["id"])


def test_automatic_retries(queue):
    _FLAKY["n"] = 0
    queue.register("flaky", flaky)
    job = queue.wait(queue.submit("flaky", max_retries=2), 10)
    assert job["status"] == "done" and job["result"] == {"attempt": 3} and job["attempts"] == 3


def test_cancel_running_and_queued(tmp_path):
    q = JobQueue(tmp_path / "q.sqlite", max_concurrent=1, poll_interval=0.02)
    try:
        q.register("slow", slow)
        running, queued = q.submit("slow", {"seconds": 30}), q.submit("slow", {"seconds": 30})
        _wait_for(lambda: q.get(running)["status"] == "running")
        assert q.cancel(queued)["status"] == "cancelled"
        q.cancel(running)
        assert q.wait(running, 10)["status"] == "cancelled"
        assert q.get(queued)["started"] is None
    finally:
        q.stop()


def test_max_concurrent_holds(tmp_path):
    q = JobQueue(tmp_path / "q.sqlite", max_concurrent=2, poll_interval=0.01)
    try:
        q.register("slow", slow)
        ids = [q.submit("slow", {"seconds": 0.4}) for _ in range(5)]
        peak = 0
        while any(q.get(i)["status"] not in ("done", "failed") for i in ids):
            peak = max(peak, q.stats()["counts"]["running"])
            time.sleep(0.01)
        assert peak == 2 and all(q.get(i)["status"] == "done" for i in ids)
    finally:
        q.stop()


def test_two_queues_on_one_database_share_the_limit(tmp_path):
    """Two server workers (two JobQueue objects on one file) never run more than max_concurrent together."""
    qs = [JobQueue(tmp_path / "q.sqlite", max_concurrent=1, poll_interval=0.01) for _ in range(2)]
    try:
        for q in qs:
            q.register("slow", slow)
        ids = [qs[i % 2].submit("slow", {"seconds": 0.2}) for i in range(4)]
        peak = 0
        while any(qs[0].get(i)["status"] != "done" for i in ids):
            peak = max(peak, qs[0].stats()["counts"]["running"])
            time.sleep(0.005)
        assert peak == 1
    finally:
        for q in qs:
            q.stop()


def test_interrupted_job_is_resumed_by_another_worker(tmp_path):
    db = tmp_path / "q.sqlite"
    dead = JobQueue(db, poll_interval=0.01)
    dead.register("double", double)
    dead.stop()                                      # this worker dies: it never picks up anything
    job_id = dead.submit("double", {"x": 2})
    now = time.time()                                # ... after claiming the job (as a crash mid-run leaves it)
    connect(db).execute("UPDATE jobs SET status='running', attempts=1, started=?, heartbeat=?, worker='gone' "
                        "WHERE id=?", (now, now, job_id))
    q = JobQueue(db, poll_interval=0.01, heartbeat_interval=0.05, stale_after=0.2)
    try:
        q.register("double", double)
        job = q.wait(job_id, 10)
        assert job["status"] == "done" and job["result"] == {"y": 4} and job["attempts"] == 2
        assert any("worker lost" in e["line"] for e in q.logs(job_id))
    finally:
        q.stop()


def test_thread_time_limit(tmp_path):
    q = JobQueue(tmp_path / "q.sqlite", poll_interval=0.01, max_seconds=0.2)
    try:
        q.register("slow", slow)
        job = q.wait(q.submit("slow", {"seconds": 30}), 10)
        assert job["status"] == "failed" and "time limit" in job["error"]
    finally:
        q.stop()


def test_process_executor_runs_and_kills(tmp_path):
    q = JobQueue(tmp_path / "q.sqlite", executor="process", poll_interval=0.02)
    try:
        q.register("double", double)
        q.register("sleep", sleeper_without_checks)
        assert q.wait(q.submit("double", {"x": 5}), 60)["result"] == {"y": 10}
        sid = q.submit("sleep", {"seconds": 60})
        _wait_for(lambda: q.get(sid)["status"] == "running")
        time.sleep(0.5)
        q.cancel(sid)
        assert q.wait(sid, 30)["status"] == "cancelled"   # killed although it never checks for cancel
        with pytest.raises(ValueError, match="module-level"):
            q.register("lam", lambda p, c: {})
    finally:
        q.stop()


def test_unknown_kind_and_unserialisable_result(queue):
    with pytest.raises(KeyError):
        queue.submit("nope")
    queue.register("bad", lambda p, c: {"x": object()})
    job = queue.wait(queue.submit("bad"), 10)
    assert job["status"] == "failed" and "JSON" in job["error"]


def test_api_keys(tmp_path):
    keys = APIKeyStore(tmp_path / "s.sqlite", app="campaign")
    created = keys.create("ci")
    plain = created["key"]
    assert plain.startswith("pk_") and "hash" not in created
    assert plain not in (tmp_path / "s.sqlite").read_bytes().decode("latin-1")   # stored hashed only
    assert keys.verify(plain)["id"] == created["id"]
    assert keys.verify("pk_wrong") is None and keys.verify(None) is None
    assert APIKeyStore(tmp_path / "s.sqlite", app="other").verify(plain) is None  # per app
    keys.revoke(created["id"])
    assert keys.verify(plain) is None and keys.list() == []
    assert keys.list(include_revoked=True)[0]["revoked"] is not None


def test_metering_limits_and_env(tmp_path):
    limits = limits_from_env("CAMPAIGN", {"cases_planned": 100, "campaigns": 5, "jobs": None},
                             environ={"CAMPAIGN_FREE_CASES_PLANNED": "50", "CAMPAIGN_FREE_CAMPAIGNS": "unlimited"})
    assert limits == {"cases_planned": 50.0, "campaigns": None, "jobs": None}
    meter = UsageMeter(tmp_path / "s.sqlite", app="campaign", limits=limits)
    meter.consume("ip:1", "cases_planned", 40)
    with pytest.raises(UsageLimitExceeded) as exc:
        meter.consume("ip:1", "cases_planned", 20)
    assert exc.value.to_dict()["used"] == 40
    meter.consume("ip:2", "cases_planned", 50)                     # per principal
    meter.consume("ip:1", "campaigns", 1000)                       # unlimited
    u = meter.usage("ip:1")["units"]
    assert u["cases_planned"] == {"used": 40.0, "limit": 50.0, "remaining": 10.0}
    assert meter.totals()["units"]["cases_planned"] == {"used": 90.0, "principals": 2}


def test_metering_is_atomic_across_threads(tmp_path):
    meter = UsageMeter(tmp_path / "s.sqlite", app="a", limits={"runs": 10})
    ok = []

    def hit():
        try:
            meter.consume("ip:1", "runs", 1)
            ok.append(1)
        except UsageLimitExceeded:
            pass

    threads = [threading.Thread(target=hit) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 10 and meter.used("ip:1", "runs") == 10


def test_uploads(tmp_path):
    store = UploadStore(tmp_path / "up", max_bytes=1000, allowed_suffixes={".csv"})
    rec = store.save("../../etc/cases.csv", io.BytesIO(b"a,b\n1,2\n"), owner="ip:1")
    assert rec["name"] == "cases.csv" and rec["size"] == 8 and len(rec["sha256"]) == 64
    assert store.path(rec["id"]).read_bytes() == b"a,b\n1,2\n"
    assert store.path(rec["id"]).parent.parent == tmp_path / "up"
    with pytest.raises(UploadTooLarge):
        store.save("big.csv", io.BytesIO(b"x" * 2000))
    with pytest.raises(UploadRejected):
        store.save("x.exe", b"MZ")
    assert len(store.list()) == 1 and store.list(owner="ip:2") == []
    with pytest.raises(KeyError):
        store.get("../x")
    store.delete(rec["id"])
    assert store.list() == []
    assert safe_name(".hidden") == "hidden" and safe_name("") == "upload"


def test_data_dir(tmp_path):
    assert data_dir("campaign", "CAMPAIGN", environ={"CAMPAIGN_DATA_DIR": str(tmp_path / "x")}) == tmp_path / "x"
    assert data_dir("campaign", "CAMPAIGN", environ={"PINNEAPPLE_APP_DATA": str(tmp_path)}) == tmp_path / "campaign"


def _wait_for(cond, timeout=10.0):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)
