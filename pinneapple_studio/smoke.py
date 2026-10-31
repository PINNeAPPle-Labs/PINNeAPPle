"""Smoke runs of executable tools: call each with the inputs of its form (its defaults, or the inputs recorded for it
in ``pinneapple_studio/tool_inputs.json``) in a separate interpreter, and record whether it returned.

A smoke run says the tool can be called from its form and returns; it is not a physics check (that is the status
from the tests, and the verifications of a job). Records live in ``data/studio/tool_runs/<package>.json``
(``{tool id: {code_hash, ok, seconds, recorded, inputs, error, returned}}``), one file per package, so a change in
one package does not touch the others.

    python -m pinneapple_studio smoke --stale            # tools whose code changed or that never ran
    python -m pinneapple_studio smoke --package pinneapple_presets
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .inventory import repo_root

RUNS = Path("data") / "studio" / "tool_runs"
INPUTS = Path(__file__).with_name("tool_inputs.json")
PER_TOOL_S = 60


def runs_dir(root: Path | None = None) -> Path:
    return Path(root or repo_root()) / RUNS


def read_runs(root: Path | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    d = runs_dir(root)
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        out.update(json.loads(p.read_text(encoding="utf-8")))
    return out


def recorded_inputs() -> dict[str, dict[str, Any]]:
    """Inputs written down for tools whose form has required fields (``{tool id: {name: value}}``)."""
    return json.loads(INPUTS.read_text(encoding="utf-8")) if INPUTS.is_file() else {}


def smoke_inputs(tool: dict[str, Any], given: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The inputs of a smoke run: the recorded ones, else the defaults when nothing is required, else ``None``."""
    if tool["id"] in given:
        return given[tool["id"]]
    if not tool.get("form", {}).get("required"):
        return {}
    return None


# ------------------------------------------------------------------------------------------------- child
class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def child(argv: list[str]) -> int:
    """``python -m pinneapple_studio.smoke <json file of [[id, inputs], ...]>``: one JSON line per tool."""
    import contextlib
    import io

    from .catalog import load
    from .runner import build_inputs, resolve
    tools = load()["tools"]
    todo = json.loads(Path(argv[0]).read_text())
    signal.signal(signal.SIGALRM, _alarm)
    for tid, inputs in todo:
        t0 = time.time()
        rec: dict[str, Any] = {"id": tid}
        signal.alarm(PER_TOOL_S)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                tool = tools[tid]
                args = build_inputs(tool.get("form", {}), inputs)
                if tool["kind"] in ("preset", "experiment"):
                    from .extras import run
                    value, _ = run(tool, inputs)
                else:
                    obj = resolve(tid)
                    value = obj(**args)
                    if tool["kind"] == "class":
                        m = next((m for m in tool["methods"] if m in ("run", "solve", "predict", "evaluate",
                                                                      "simulate", "compute")), None)
                        if m:
                            import inspect
                            fn = getattr(value, m)
                            req = [p for p in inspect.signature(fn).parameters.values()
                                   if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
                            if not req:
                                value = fn()
                                rec["method"] = m
            rec.update(ok=True, returned=type(value).__name__)
        except _Timeout:
            rec.update(ok=False, error=f"timeout after {PER_TOOL_S} s")
        except BaseException as exc:  # noqa: BLE001 - a smoke run records any failure
            rec.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            signal.alarm(0)
            with contextlib.suppress(Exception):
                import matplotlib.pyplot as plt
                plt.close("all")
        rec["seconds"] = round(time.time() - t0, 2)
        print(json.dumps(rec, default=str), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------- parent
def record(catalog: dict[str, Any], *, stale_only: bool = True, package: str | None = None,
           ids: list[str] | None = None, workers: int = 4, batch: int = 80, log=print) -> dict[str, int]:
    """Smoke-run the executable tools that have inputs and write the records.

    Tools run in batches of one package (``batch`` tools at most), each batch in its own interpreter, in a scratch
    folder, with the job environment of :func:`pinneapple_studio.runner.clean_env`; ``workers`` batches run at once.
    When an interpreter dies, the first tool without a result is recorded as the cause and the rest run again in a
    new interpreter. A batch during which a file of the checkout changed fails with the paths (with several workers,
    every batch running at that moment)."""
    from concurrent.futures import ThreadPoolExecutor

    from .coverage import SIDE_EFFECTS
    from .runner import clean_env
    root = repo_root()
    runs = read_runs(root)
    given = recorded_inputs()
    by_pkg: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for tid, t in sorted(catalog["tools"].items(), key=lambda kv: (kv[1]["package"], kv[1]["module"], kv[0])):
        if t["category"] != "executable" or (package and t["package"] != package) or (ids and tid not in ids):
            continue
        inputs = smoke_inputs(t, given)
        if inputs is None or (SIDE_EFFECTS.search(t["name"]) and tid not in given):
            continue
        old = runs.get(tid)
        if stale_only and old and old.get("code_hash") == t["code_hash"] and old.get("inputs") == inputs:
            continue
        by_pkg[t["package"]].append((tid, inputs))
    batches = [todo[k:k + batch] for todo in by_pkg.values() for k in range(0, len(todo), batch)]
    stats = {"batches": len(batches), "tools": sum(len(b) for b in batches), "ok": 0, "failed": 0}
    env = clean_env({"PYTHONPATH": str(root), "MPLBACKEND": "Agg", "PINNEAPPLE_ROOT": str(root),
                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    import threading
    lock = threading.Lock()
    finished = [0]

    def one(todo: list[tuple[str, dict]]) -> dict[str, dict[str, Any]]:
        before = _tracked_state(root)
        done: dict[str, dict[str, Any]] = {}
        left = list(todo)
        while left:
            got, err = _interpreter(left, env)
            done.update(got)
            rest = [(t, i) for t, i in left if t not in got]
            if not rest:
                break
            done[rest[0][0]] = {"ok": False, "error": f"the interpreter stopped during this tool: {err}",
                                "seconds": None}
            left = rest[1:]
        touched = sorted(set(_tracked_state(root).items()) - set(before.items()))
        for tid, inputs in todo:
            r = done[tid]
            if touched:
                r.update(ok=False, error="the repository changed during this batch (this tool's batch or one "
                                         "running at the same time): " + ", ".join(p for p, _ in touched[:5]))
            r.update(code_hash=catalog["tools"][tid]["code_hash"], inputs=inputs, recorded=time.strftime("%Y-%m-%d"))
        with lock:
            runs.update(done)
            finished[0] += 1
            _write(root, runs, catalog)
            log(f"[{finished[0]}/{len(batches)}] {todo[0][0].split(':')[0]}...: "
                f"{sum(r['ok'] for r in done.values())}/{len(todo)} returned")
        return done

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for done in ex.map(one, batches):
            for r in done.values():
                stats["ok" if r["ok"] else "failed"] += 1
    return stats


def _interpreter(todo: list[tuple[str, dict]], env: dict[str, str]) -> tuple[dict[str, dict[str, Any]], str]:
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "todo.json"
        spec.write_text(json.dumps(todo, default=str))
        try:
            out = subprocess.run([sys.executable, "-m", "pinneapple_studio.smoke", str(spec)], cwd=tmp, env=env,
                                 capture_output=True, text=True, timeout=PER_TOOL_S * len(todo) + 120)
            lines, err = out.stdout.splitlines(), (out.stderr.strip().splitlines() or [f"exit {out.returncode}"])[-1]
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ""
            lines = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw).splitlines()
            err = "the interpreter ran out of time"
    got = {}
    for ln in lines:
        try:
            r = json.loads(ln)
            got[r.pop("id")] = r
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
    return got, err[-200:]


def _tracked_state(root: Path) -> dict[str, str]:
    """``{path: status}`` of changed and untracked files outside ``data/studio`` (to catch a smoke run that writes
    into the checkout)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for ln in out.splitlines():
        path = ln[3:]
        if not path.startswith(str(RUNS)) and "__pycache__" not in path:
            rows[path] = ln[:2] + str(_mtime(root / path))
    return rows


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _write(root: Path, runs: dict[str, dict[str, Any]], catalog: dict[str, Any]) -> None:
    by_pkg: dict[str, dict[str, Any]] = defaultdict(dict)
    for tid, r in runs.items():
        t = catalog["tools"].get(tid)
        if t is not None:
            by_pkg[t["package"]][tid] = r
    d = runs_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    for pkg, recs in by_pkg.items():
        (d / f"{pkg}.json").write_text(json.dumps(dict(sorted(recs.items())), indent=1, sort_keys=True,
                                                  default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(child(sys.argv[1:]))
