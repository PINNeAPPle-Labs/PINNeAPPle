"""Run a tool or a gallery entry as a job: a queue (:class:`pinneapple_core.service.jobs.JobQueue`, process executor),
an environment without credentials, resource limits, a time limit, logs, progress and cancel.

* ``tool``: ``{"tool": "module:name", "inputs": {...}, "method": "run"}`` builds the inputs from the form (lists to
  numpy arrays where the form says ``x-convert: numpy``, nested forms to their class), calls the function (or builds
  the class and calls ``method``), and stores the result as JSON plus every figure it left open;
* ``gallery``: ``{"entry": "example:lbm", "quick": true}`` runs the entry's example flow
  (:func:`pinneapple_flow.examples.run_example`, never recorded into the repository) and compares it with the
  committed measurement, or ``{"entry": ..., "script": "examples/x/01.py"}`` runs one script in its own process.

Every result carries an evidence record: parameters, versions, wall time, files with SHA-256, and verifications, each
``pass`` / ``fail`` / ``inconclusive`` / ``not_run``. A run that only finished without an error has no verification
that passed: it says ``not_run`` for the physics.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .inventory import repo_root

ENV_ALLOW = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "PYTHONPATH", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "MPLBACKEND", "PINNEAPPLE_ROOT", "PINNEAPPLE_STUDIO_CACHE", "VIRTUAL_ENV",
             "FLOW_HOME", "PINNEAPPLE_FLOW_HOME", "WM_PROJECT_DIR", "FOAM_INST_DIR", "LD_LIBRARY_PATH")
SECRET = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|DATABASE_URL|REDIS_URL|"
                    r"^AWS_|^GCP_|^AZURE_|^GITHUB_|^GH_|^HF_|^OPENAI|^ANTHROPIC", re.I)
LIMITS = {"memory_mb": 6144, "cpu_seconds": 3600, "max_files": 1024}


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a job sees: an allowlist, minus anything that looks like a secret."""
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW and not SECRET.search(k)}
    env.setdefault("MPLBACKEND", "Agg")
    env.update(extra or {})
    return env


def scrub_environment() -> None:
    """Inside a job process: drop every variable outside :func:`clean_env`."""
    keep = clean_env()
    for k in list(os.environ):
        if k not in keep:
            del os.environ[k]
    os.environ.update(keep)


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "")
    try:
        out["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True,
                                       text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out["commit"] = ""
    return out


def _files(workdir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(workdir.rglob("*")):
        if p.is_file() and p.name not in ("output.log",):
            out.append({"path": str(p.relative_to(workdir)), "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return out[:500]


def evidence(params: dict[str, Any], workdir: Path, t0: float, verifications: list[dict[str, Any]],
             status: str) -> dict[str, Any]:
    outcomes = [v["outcome"] for v in verifications]
    if not verifications:
        overall = "not_run"
    elif all(o == "pass" for o in outcomes):
        overall = "pass"
    elif any(o == "fail" for o in outcomes):
        overall = "fail"
    else:
        overall = "inconclusive"
    return {"parameters": params, "versions": _versions(), "wall_s": round(time.time() - t0, 3), "status": status,
            "files": _files(workdir), "verifications": verifications, "outcome": overall}


# ------------------------------------------------------------------------------------------------- tools
def resolve(target: str) -> Any:
    module, _, qual = target.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def build_inputs(schema: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Form values to call arguments: numeric lists to arrays (``x-convert: numpy``), nested forms to objects."""
    out = {}
    props = schema.get("properties", {})
    for k, v in values.items():
        s = props.get(k, {})
        out[k] = _convert(s, v)
    return out


def _convert(s: dict[str, Any], v: Any) -> Any:
    if v is None:
        return None
    if s.get("x-convert") == "numpy":
        import numpy as np
        return np.asarray(v, dtype=float)
    if s.get("x-class") and isinstance(v, dict):
        cls = resolve(s["x-class"])
        return cls(**{k: _convert(s.get("properties", {}).get(k, {}), x) for k, x in v.items()})
    if s.get("type") == "array" and isinstance(v, list) and isinstance(s.get("items"), dict):
        return [_convert(s["items"], x) for x in v]
    for alt in s.get("anyOf", []):
        if alt.get("x-convert") or alt.get("x-class"):
            try:
                return _convert(alt, v)
            except (TypeError, ValueError):
                continue
    return v


def tool_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Job kind ``tool``."""
    scrub_environment()
    t0 = time.time()
    workdir = Path(ctx.workdir)
    os.chdir(workdir)
    import matplotlib
    matplotlib.use("Agg")
    from pinneapple_core.jsonable import to_jsonable

    from .catalog import load
    tool = load()["tools"].get(params["tool"])
    if tool is None:
        raise KeyError(f"unknown tool {params['tool']!r}")
    if tool["category"] != "executable":
        raise ValueError(f"{params['tool']} is {tool['category']}: {tool['category_reason']}")
    ctx.progress(0.05, f"importing {tool['module']}")
    obj = resolve(tool["id"])
    inputs = build_inputs(tool.get("form", {}), params.get("inputs") or {})
    ctx.log(f"calling {tool['id']} with {sorted(inputs)}")
    ctx.progress(0.2, "running")
    if tool["kind"] == "class":
        method = params.get("method") or next((m for m in tool["methods"] if m in ("run", "solve", "predict",
                                                                                     "evaluate", "simulate")), None)
        inst = obj(**inputs)
        value = getattr(inst, method)(**(params.get("method_inputs") or {})) if method else inst
    else:
        value = obj(**inputs)
    figs = _save_figures(workdir)
    result = to_jsonable(value, digits=8, nonfinite="nan")
    text = json.dumps(result, default=str)
    (workdir / "result.json").write_text(text[:5_000_000])
    ctx.progress(1.0, "done")
    return {"tool": tool["id"], "type": type(value).__name__, "result": json.loads(text) if len(text) < 200_000
            else {"truncated": True, "bytes": len(text)}, "figures": figs,
            "evidence": evidence(params, workdir, t0, [], "done")}


def _save_figures(workdir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    out = []
    for i, num in enumerate(plt.get_fignums()):
        p = workdir / f"figure_{i + 1}.png"
        plt.figure(num).savefig(p, dpi=110, bbox_inches="tight")
        out.append(p.name)
    plt.close("all")
    return out


# ------------------------------------------------------------------------------------------------- gallery
def gallery_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Job kind ``gallery``."""
    scrub_environment()
    t0 = time.time()
    workdir = Path(ctx.workdir)
    from .gallery import load
    entry = next((e for e in load() if e["id"] == params["entry"]), None)
    if entry is None:
        raise KeyError(f"unknown gallery entry {params['entry']!r}")
    if entry["blocked_by"]:
        need = "; ".join(f"{r['name']} ({r['fix']})" for r in entry["blocked_by"])
        raise RuntimeError(f"cannot run here: {need}")
    if params.get("script"):
        return _script(entry, params, ctx, workdir, t0)
    if not entry["flows"]:
        raise ValueError(f"{entry['id']} has no flow; pass a script ({', '.join(entry['scripts'])})")
    from pinneapple_flow import examples as FX
    name = params.get("flow") or entry["flows"][0]["name"]
    quick = bool(params.get("quick", True))

    def on_update(*a, **k):
        ctx.check_cancel()

    ctx.progress(0.05, f"running flow {name} ({'quick' if quick else 'full'})")
    rec = FX.run_example(name, quick=quick, workdir=str(workdir / "flow"), record=False, on_update=on_update)
    ctx.log(f"flow status {rec['status']} in {rec['seconds']} s")
    recorded = next((m for m in entry["measured"] if m["flow"] == name and
                     m["configuration"] == ("quick" if quick else "full")), None)
    ver = [{"name": c["flow"], "outcome": "pass" if c["outcome"] == "match" else
            ("fail" if c["outcome"] in ("mismatch", "differ", "fail") else "inconclusive"),
            "value": c.get("value"), "reference": c.get("reference"), "rel_diff": c.get("rel_diff"),
            "source": c.get("ref")} for c in rec["comparisons"]]
    ctx.progress(1.0, rec["status"])
    return {"entry": entry["id"], "flow": name, "configuration": rec["configuration"], "status": rec["status"],
            "seconds": rec["seconds"], "nodes": rec["nodes"], "comparisons": rec["comparisons"],
            "recorded": recorded, "run_workdir": rec.get("workdir"),
            "evidence": evidence(params, workdir, t0, ver, rec["status"])}


def _script(entry: dict[str, Any], params: dict[str, Any], ctx, workdir: Path, t0: float) -> dict[str, Any]:
    root = repo_root()
    script = (root / params["script"]).resolve()
    if str(script.relative_to(root)) not in entry["scripts"]:
        raise PermissionError(f"{params['script']} is not a script of {entry['id']}")
    env = clean_env({"PYTHONPATH": str(root), "MPLBACKEND": "Agg"})
    ctx.progress(0.05, f"python {script.name}")
    timeout = float(params.get("timeout", 1800))
    with open(workdir / "script.log", "w") as log:
        proc = subprocess.Popen([sys.executable, str(script)], cwd=script.parent, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            if time.time() - t0 > timeout:
                proc.kill()
                raise TimeoutError(f"{script.name} ran past {timeout:g} s")
            ctx.progress(0.5)                                # raises after a cancel request
            time.sleep(1.0)
    tail = (workdir / "script.log").read_text(errors="replace")[-20_000:]
    ctx.log(tail[-4000:])
    status = "done" if proc.returncode == 0 else "failed"
    return {"entry": entry["id"], "script": params["script"], "status": status, "exit_code": proc.returncode,
            "log_tail": tail, "evidence": evidence(params, workdir, t0, [], status)}


# ------------------------------------------------------------------------------------------------- queue
def queue(home: str | Path | None = None, *, max_concurrent: int = 1, max_seconds: float = 3600,
          limits: dict[str, float] | None = None):
    """A job queue with the ``tool`` and ``gallery`` kinds, in their own processes with :data:`LIMITS`."""
    from pinneapple_core.service.jobs import JobQueue
    home = Path(home or Path.home() / ".pinneapple" / "studio")
    q = JobQueue(home / "jobs.sqlite", app="studio", executor="process", max_concurrent=max_concurrent,
                 max_seconds=max_seconds, limits={**LIMITS, **(limits or {})}, log_tracebacks=False)
    q.register("tool", tool_job)
    q.register("gallery", gallery_job)
    return q
