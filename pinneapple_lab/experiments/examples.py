"""Every example and use case of the repository as a lab experiment.

``example_script`` runs one script of ``examples/`` (``script`` parameter, a path relative to the repository) the way
a user would (``python examples/...`` from the repository root, one thread per library), then collects what it
produced: every file it created or changed anywhere in the checkout. Figures become run figures, JSON files become
outputs and their numbers metrics, arrays (.npy / .npz / .csv) become the ``artifacts`` dataset, the console output
is kept, and the script plus the Python files next to it are snapshotted as the code that ran. Tracked files the
script overwrote are copied into the run and then restored, so running the whole catalogue leaves git clean.

    python -m pinneapple_lab examples                     # list every runnable example with its group
    python -m pinneapple_lab examples --run --match use_cases --timeout 1800
    python -m pinneapple_lab run example_script -p script=examples/getting_started/01_harmonic_oscillator.py

Validation: the script exits with status 0 within ``timeout`` seconds. Scripts that need a GPU, a download or an
optional dependency that is missing fail and are recorded as such, which also makes the catalogue a health report
of the examples.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

from ..spec import Experiment, register

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SKIP_DIRS = {".git", "lab", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
_FIG = {".png", ".gif", ".jpg", ".jpeg", ".svg"}
_ARR = {".npy", ".npz", ".csv"}
_MAX_FIGS, _MAX_FILE_MB, _MAX_METRICS = 16, 64, 60
# never restored: sources and docs edited by hand while a script runs must not be reverted
_SOURCE = {".py", ".md", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".js", ".ts", ".css", ".sh", ".txt", ".rst"}


def discover_examples(repo: str = REPO) -> list[dict[str, str]]:
    """Runnable example scripts: ``examples/**/*.py`` with a ``__main__`` guard or top-level code, minus helpers
    (``_*.py``, ``__init__.py``) and scratch folders. ``group`` is the folder under ``examples/``."""
    out = []
    root = os.path.join(repo, "examples")
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in _SKIP_DIRS and not d.startswith(("_", ".")) and d != "scratch")
        for fn in sorted(fns):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    src = f.read()
            except OSError:
                continue
            if "__main__" not in src and not any(line and not line[0].isspace() and not line.startswith(
                    ("import", "from", "def", "class", "#", "@", '"', "'", ")", "]", "}")) for line in src.splitlines()):
                continue                                            # a module of helpers, not a script
            rel = os.path.relpath(path, repo)
            parts = rel.split(os.sep)
            group = "/".join(parts[1:3]) if parts[1] == "use_cases" and len(parts) > 3 else parts[1] \
                if len(parts) > 2 else "examples"
            out.append({"script": rel.replace(os.sep, "/"), "group": group})
    return out


def _walk(repo: str) -> dict[str, tuple[float, int]]:
    files = {}
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS]
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            files[p] = (st.st_mtime, st.st_size)
    return files


def _git_lines(repo: str, *args: str) -> set[str]:
    try:
        out = subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL, timeout=60).decode()
    except Exception:
        return set()
    return {os.path.join(repo, x) for x in out.splitlines() if x}


def _flatten(obj, prefix="", out=None, depth=0):
    out = {} if out is None else out
    if depth > 3 or len(out) >= _MAX_METRICS:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}." if not isinstance(v, (int, float)) else f"{prefix}{k}", out, depth + 1)
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)) and np.isfinite(obj):
        out[prefix.rstrip(".")] = float(obj)
    return out


def _arrays(path: str) -> dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    with open(path, newline="") as f:                              # .csv: numeric columns
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    head, body = rows[0], rows[1:]
    cols = {}
    for j, name in enumerate(head):
        try:
            cols[name or f"col{j}"] = np.array([float(r[j]) for r in body if j < len(r)])
        except ValueError:
            continue
    return cols


_METRIC_LINE = re.compile(r"^[\s\-*>→|]*([A-Za-z][\w .,/()²³%-]{1,48}?)\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
                          r"\s*([A-Za-z%°/²³]{0,8})\s*$")


def _stdout_metrics(text: str, limit: int = 40) -> dict[str, float]:
    """Numbers the script printed as ``name: value`` or ``name = value`` lines (last value of each name)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = "stdout." + re.sub(r"[^\w]+", "_", m.group(1).strip().lower()).strip("_")
        if name not in out and len(out) >= limit:
            continue
        try:
            out[name] = float(m.group(2))
        except ValueError:
            continue
    return out


@register
class ExampleScript(Experiment):
    name = "example_script"
    version = "1"
    description = ("Runs an example or use case of the repository and stores what it produced: figures, JSON "
                   "outputs and their numbers as metrics, arrays as the 'artifacts' dataset, the console output and "
                   "the code. The catalogue of these runs is also the health report of the examples.")
    tags = ["examples", "use-case", "dataset"]
    params = {"script": "examples/getting_started/01_harmonic_oscillator.py", "timeout": 900, "args": ""}

    @classmethod
    def code_for(cls, params):
        path = os.path.join(REPO, params["script"])
        folder = os.path.dirname(path)
        return [path] + [p for p in sorted(glob.glob(os.path.join(folder, "*.py"))) if p != path][:20]

    def run(self, ctx):
        p = ctx.params
        script = os.path.join(REPO, p["script"])
        if not os.path.isfile(script):
            raise FileNotFoundError(p["script"])
        dirty_before = _git_lines(REPO, "diff", "--name-only", "HEAD")
        stray_before = _git_lines(REPO, "ls-files", "--others", "--exclude-standard")
        tracked = _git_lines(REPO, "ls-files")
        before = _walk(REPO)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [sys.executable, script, *p["args"].split()] if p["args"] else [sys.executable, script]
        t0 = time.time()
        with ctx.stage("run"):
            try:
                proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, timeout=float(p["timeout"]),
                                      stdin=subprocess.DEVNULL)
                code, out, err = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                code, out, err = None, exc.stdout or b"", (exc.stderr or b"") + b"\n[timeout]"
        seconds = time.time() - t0
        ctx.output("stdout.txt", out.decode(errors="replace")[-200_000:])
        if err:
            ctx.output("stderr.txt", err.decode(errors="replace")[-200_000:])
        ctx.metric("script_seconds", round(seconds, 2))
        ctx.metric("exit_code", -1 if code is None else code)
        for k, v in _stdout_metrics(out.decode(errors="replace")).items():
            ctx.metric(k, v)

        with ctx.stage("collect"):
            after = _walk(REPO)
            changed = sorted(f for f, s in after.items() if before.get(f) != s and not f.startswith(ctx.dir))
            ctx.metric("files_produced", len(changed))
            ctx.output("produced_files", [os.path.relpath(f, REPO) for f in changed])
            n_fig, metrics, ds = 0, {}, None
            for f in changed:
                ext = os.path.splitext(f)[1].lower()
                rel = os.path.relpath(f, REPO)
                if os.path.getsize(f) > _MAX_FILE_MB * 2 ** 20:
                    continue
                stem = rel.replace(os.sep, "__")
                if ext in _FIG and n_fig < _MAX_FIGS:
                    ctx.figure_file(f, stem)
                    n_fig += 1
                elif ext == ".json":
                    try:
                        with open(f) as fh:
                            obj = json.load(fh)
                    except (OSError, ValueError):
                        continue
                    ctx.output(stem, obj)
                    key = os.path.splitext(os.path.basename(f))[0]
                    for k, v in _flatten(obj).items():
                        if len(metrics) < _MAX_METRICS:
                            metrics[f"{key}.{k}"] = v
                elif ext in _ARR:
                    try:
                        arrays = _arrays(f)
                    except Exception:                              # noqa: BLE001 - an unreadable file is skipped
                        continue
                    for k, a in arrays.items():
                        if a.dtype.kind in "biuf" and a.size:
                            if ds is None:
                                ds = ctx.dataset("artifacts", description=f"Arrays written by {p['script']}")
                            ds.add(array=a, file=rel, key=k, script=p["script"])
            for k, v in metrics.items():
                ctx.metric(k, v)
            # restore tracked files the script overwrote (copied into the run above)
            restore = [f for f in changed if f in tracked and f not in dirty_before
                       and os.path.splitext(f)[1].lower() not in _SOURCE]
            if restore:
                subprocess.run(["git", "-C", REPO, "checkout", "--", *[os.path.relpath(f, REPO) for f in restore]],
                               capture_output=True, timeout=60)
                ctx.output("restored_tracked_files", [os.path.relpath(f, REPO) for f in restore])
            # untracked, not git-ignored files (a figure saved in the working directory) now live in the run
            stray = sorted(_git_lines(REPO, "ls-files", "--others", "--exclude-standard") - stray_before)
            for f in stray:
                if f in changed and not f.startswith(os.path.join(REPO, "lab") + os.sep):
                    os.remove(f)
            if stray:
                ctx.output("moved_untracked_files", [os.path.relpath(f, REPO) for f in stray])

        tail = (err or out).decode(errors="replace").strip().splitlines()[-12:]
        ctx.check("exits_cleanly", code == 0,
                  detail="timeout" if code is None else ("" if code == 0 else " | ".join(tail)[-600:]))
        if code is None:
            raise TimeoutError(f"{p['script']} did not finish in {p['timeout']} s")
