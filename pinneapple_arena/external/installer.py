"""Install-state helpers shared by every runner and by the API layer."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def is_installed(module_name: str) -> bool:
    """True when ``module_name`` can be found, without importing it (an import of a heavy library has side effects,
    e.g. torch's default device). A library that is found but fails to import is reported by its runner as
    ``failed`` with the real exception."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def pip_install(package: str, timeout: int = 900) -> tuple[bool, str]:
    """Best-effort `pip install`. Used only when a user explicitly opts in
    (the API's `install=true` flag) — never triggered implicitly."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", package],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0, proc.stderr[-600:]
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
