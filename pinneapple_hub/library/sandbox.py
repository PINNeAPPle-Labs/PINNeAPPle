"""Run a code snippet in a short-lived, separate interpreter and keep what it printed and drew.

    from pinneapple_hub.library.sandbox import run_snippet
    r = run_snippet('import pinneapple as pp\\nprint(pp.geometries.list()[:3])', timeout=60)
    r["ok"], r["stdout"], r["seconds"], r["figures"]       # figures: PNG bytes of the matplotlib figures left open

The snippet runs in a fresh temporary directory (its working directory, deleted afterwards) with matplotlib on
Agg, no GPU, a wall-clock time limit and, on POSIX, address-space / CPU-time / file-size limits (``setrlimit``) so a
runaway snippet is killed with every process it started. Network access is not blocked here: deploy the app that
calls this on a network without egress (apps/deploy) when the snippets come from visitors.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT = 6000
#: One matplotlib cache for every snippet, not one per run. With a cache directory inside the run's temporary directory,
#: matplotlib rebuilt its font cache on every snippet — two "Matplotlib is building the font cache" lines and, on a
#: loaded machine, the whole 300 s the slowest graded snippet is allowed (#625).
MPL_CACHE = Path(tempfile.gettempdir()) / "pinneapple-sandbox-mpl"

_WRAP = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
if FSIZE_B_:
    # The file-size limit is applied here, inside the fresh interpreter, and not in a preexec_fn of the parent: a
    # callback between fork and exec is not safe once the caller has threads (the Hub calls this from its job threads),
    # and when it raised, the caller saw "Exception occurred in preexec_fn" instead of a snippet result (#625). The
    # address-space and CPU-time limits cannot move here — a limit can only be lowered below what the process already
    # uses — so they stay in the preexec, and only when a caller asks for one.
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_B_, FSIZE_B_))
sys.path.insert(0, ROOT_)
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    matplotlib = None
code = open(CODE_).read()
exec(compile(code, "snippet.py", "exec"), {"__name__": "__main__"})
if matplotlib is not None and "matplotlib.pyplot" in sys.modules:
    import matplotlib.pyplot as plt
    for k, n in enumerate(plt.get_fignums()):
        plt.figure(n).savefig(os.path.join(FIGS_, f"fig_{k}.png"), dpi=100, facecolor="white", bbox_inches="tight")
'''


def code_hash(code: str) -> str:
    """Short SHA-256 of a snippet: a recorded run is current only while its snippet has this hash."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _limits(memory_mb: int | None, cpu_s: int | None) -> dict[str, str]:
    """The substitutions the wrapper uses to apply the file-size limit (POSIX only; zero elsewhere)."""
    return {"FSIZE_B_": str(200 * 2**20) if os.name == "posix" else "0"}


def _rlimits(memory_mb: int | None, cpu_s: int | None, refused_report: Path | None = None):
    """``preexec_fn`` for the two limits that must be set before the interpreter maps anything: address space and CPU.

    Returns ``None`` when neither was asked for, which keeps the safe ``start_new_session`` route and the fork-time
    callback out of callers that need no limit. A limit the platform refuses is written to ``refused_report`` and does
    **not** kill the run: macOS keeps a virtual mapping of hundreds of gigabytes per interpreter, so ``RLIMIT_AS`` is
    rejected there with ``EINVAL`` — refusing to start the snippet because the guard could not be tightened turned into
    "Exception occurred in preexec_fn" in the Hub's job log (#625). The wall-clock timeout is the guard that always
    applies.
    """
    if os.name != "posix" or not (memory_mb or cpu_s):
        return None

    def apply():
        import resource
        wanted = []
        if memory_mb:
            wanted.append(("address space", resource.RLIMIT_AS, memory_mb * 2**20))
        if cpu_s:
            wanted.append(("CPU time", resource.RLIMIT_CPU, cpu_s))
        refused = []
        for name, which, value in wanted:
            try:
                hard = value + 1 if which == resource.RLIMIT_CPU else value
                resource.setrlimit(which, (value, hard))
            except (ValueError, OSError):
                refused.append(f"the {name} limit of {value} could not be applied on this platform")
        if refused and refused_report is not None:
            with contextlib.suppress(OSError):
                Path(refused_report).write_text("; ".join(refused), encoding="utf-8")
    return apply


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + "\n..."


def run_snippet(code: str, *, timeout: float = 60.0, memory_mb: int | None = None, cpu_s: int | None = None,
                max_figures: int = 4, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run ``code`` (see the module doc). Returns ``ok``, ``stdout``, ``stderr`` (last lines), ``seconds``,
    ``timed_out``, ``code_hash`` and ``figures`` (a list of PNG ``bytes``, at most ``max_figures``)."""
    with tempfile.TemporaryDirectory(prefix="pp-snippet-") as tmp:
        src, figs = Path(tmp) / "snippet.py", Path(tmp) / "figures"
        figs.mkdir()
        src.write_text(code)
        wrap = Path(tmp) / "_wrap.py"
        limits = _limits(memory_mb, cpu_s)
        text = _WRAP.replace("ROOT_", repr(str(ROOT))).replace("CODE_", repr(str(src))).replace("FIGS_", repr(str(figs)))
        for name, value in limits.items():
            text = text.replace(name, value)
        wrap.write_text(text)
        run_env = {k: v for k, v in os.environ.items() if not k.endswith(("_PASSWORD", "_TOKEN", "_KEY", "_SECRET"))}
        with contextlib.suppress(OSError):
            MPL_CACHE.mkdir(parents=True, exist_ok=True)
        run_env.update({"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(MPL_CACHE), "OMP_NUM_THREADS": "2", **(env or {})})
        t0 = time.time()
        refused = Path(tmp) / "limits-not-applied.txt"
        # start_new_session gives the snippet its own session and process group (the timeout kills its children too)
        # without a callback between fork and exec; that is what ``preexec_fn`` used to do for everything, and it raised
        # under the Hub's job threads (#625).
        proc = subprocess.Popen([sys.executable, str(wrap)], cwd=tmp, env=run_env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True,
                                preexec_fn=_rlimits(memory_mb, cpu_s, refused))
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            err = (err or "") + f"\nstopped after the {timeout:g} s time limit"
        images = [p.read_bytes() for p in sorted(figs.glob("fig_*.png"))[:max_figures]]
        not_applied = refused.read_text(encoding="utf-8").strip() if refused.is_file() else ""
        lines = (err or "").strip().splitlines()[-30:]
        if not_applied:
            lines.append(not_applied)
        return {"ok": proc.returncode == 0 and not timed_out, "stdout": _clip(out or ""),
                "stderr": _clip("\n".join(lines)),
                "seconds": round(time.time() - t0, 2), "timed_out": timed_out, "code_hash": code_hash(code),
                "figures": images}
