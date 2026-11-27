"""Run one of the campaign's commands in a session of its own, so it outlives the shell that started it.

``nohup`` only ignores SIGHUP; what kills a long run is the process group going away, and ``os.setsid`` is what puts
it beyond reach of that. macOS ships no ``setsid(1)``, which is why this is Python — the same detach
``benchmarks.lpm_campaign --detach`` has always done, in one place now that the panel needs it too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["detach"]


def detach(module: str, log_path: str | Path, *, marker: str, flag: str = "--detach") -> None:
    """Re-run ``python -m module`` in its own session, writing to ``log_path``, and return in the parent.

    ``marker`` is the environment variable that tells the child it is the detached copy, so it runs the command
    instead of detaching again. The child's argv is this process's argv without ``flag`` and without the value that
    followed it, which is what makes the re-exec a clean copy of the command.
    """
    if os.environ.get(marker):
        return
    argv = [sys.executable, "-m", module] + [a for a in sys.argv[1:] if a not in (flag, str(log_path))]
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(f"detached as pid {pid}; follow it with: tail -f {log_path}")
        print(f"stop it with: kill {pid}")
        sys.exit(0)
    os.setsid()
    env = {**os.environ, marker: "1"}
    with open(log_path, "w", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execve(sys.executable, argv, env)
