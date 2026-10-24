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

_WRAP = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
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


def _limits(memory_mb: int | None, cpu_s: int | None):
    if os.name != "posix":
        return None

    def apply():
        import resource
        os.setsid()                                    # own process group: the time limit kills its children too
        if memory_mb:
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 2**20, memory_mb * 2**20))
        if cpu_s:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 2**20, 200 * 2**20))
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
        wrap.write_text(_WRAP.replace("ROOT_", repr(str(ROOT))).replace("CODE_", repr(str(src)))
                        .replace("FIGS_", repr(str(figs))))
        run_env = {k: v for k, v in os.environ.items() if not k.endswith(("_PASSWORD", "_TOKEN", "_KEY", "_SECRET"))}
        run_env.update({"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(Path(tmp) / "mpl"), "OMP_NUM_THREADS": "2", **(env or {})})
        t0 = time.time()
        proc = subprocess.Popen([sys.executable, str(wrap)], cwd=tmp, env=run_env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, preexec_fn=_limits(memory_mb, cpu_s))
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
        return {"ok": proc.returncode == 0 and not timed_out, "stdout": _clip(out or ""),
                "stderr": _clip("\n".join((err or "").strip().splitlines()[-30:])),
                "seconds": round(time.time() - t0, 2), "timed_out": timed_out, "code_hash": code_hash(code),
                "figures": images}
