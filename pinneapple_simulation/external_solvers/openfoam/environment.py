"""Where OpenFOAM is and which version: the installation paths, the ``bashrc`` that has to be sourced, and a probe of
what that environment provides.

One place for it, because two surfaces ask the same question: the workflow step (``pinneapple_core.workflows.adapters``,
which has to find a ``bashrc`` before running ``blockMesh``) and the solver contract's OpenFOAM backend (whose
``check_environment()`` answers *executable*, *version* and *environment* out of the same probe). A distribution
package puts ``blockMesh`` on PATH but the binaries only run with the environment loaded, so presence on PATH is not
enough -- the probe sources the ``bashrc`` and asks the environment itself.
"""
from __future__ import annotations

import glob
import os
import subprocess

__all__ = ["FOAM_BASHRCS", "find_bashrc", "probe"]

#: Where distribution packages and releases put the ``bashrc``, in the order they are tried. The last entry is the
#: macOS cask (``gerlero/openfoam-app``), which mounts the release under ``/Volumes``.
FOAM_BASHRCS = ("/usr/share/openfoam/etc/bashrc", "/usr/lib/openfoam/openfoam*/etc/bashrc",
                "/opt/openfoam*/etc/bashrc", "/opt/OpenFOAM-*/etc/bashrc", "/Volumes/OpenFOAM*/etc/bashrc")


def find_bashrc(explicit: str | None = None, *, patterns: tuple[str, ...] | list[str] = FOAM_BASHRCS,
                environ: dict[str, str] | None = None) -> str | None:
    """The ``bashrc`` to source: ``explicit``, else ``$FOAM_BASHRC``, else (when the environment is not loaded already)
    the newest installation found in ``patterns``. None means there is nothing to source: either the environment is
    loaded or OpenFOAM was not found."""
    environ = os.environ if environ is None else environ
    for candidate in (explicit, environ.get("FOAM_BASHRC")):
        if candidate and os.path.isfile(candidate):
            return str(candidate)
    if environ.get("WM_PROJECT_DIR"):
        return None
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return None


def probe(bashrc: str | None, executables: tuple[str, ...] = ("blockMesh",), *, environ: dict[str, str] | None = None,
          timeout: float = 60.0) -> dict:
    """Source ``bashrc`` and report what the environment provides: whether it loaded, its version, its project
    directory and the path of each executable asked for (empty when the environment does not have it).

    Never raises: a ``bashrc`` that fails, a machine without ``bash`` or a probe that times out returns
    ``sourced=False`` with the error in ``error``, which is the diagnosis a caller wants.
    """
    out: dict = {"bashrc": bashrc, "sourced": False, "version": None, "project_dir": None, "executables": {},
                 "error": None}
    if not bashrc:
        return out
    lines = [f'source "{bashrc}" >/dev/null 2>&1',
             'echo "PP_DIR=$WM_PROJECT_DIR"',
             'echo "PP_VERSION=$WM_PROJECT_VERSION"']
    lines += [f'echo "PP_EXE_{i}=$(command -v {name} 2>/dev/null)"' for i, name in enumerate(executables)]
    try:
        p = subprocess.run(["bash", "-c", " ; ".join(lines)], capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ if environ is None else environ))
    except (OSError, subprocess.SubprocessError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    for line in p.stdout.splitlines():
        key, _, value = line.strip().partition("=")
        if key == "PP_DIR":
            out["project_dir"] = value or None
        elif key == "PP_VERSION":
            out["version"] = value or None
        elif key.startswith("PP_EXE_"):
            name = executables[int(key[len("PP_EXE_"):])]
            if value:
                out["executables"][name] = value
    out["sourced"] = bool(out["project_dir"]) and p.returncode == 0
    if not out["sourced"] and out["error"] is None:
        out["error"] = (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else
                        f"bash exited with code {p.returncode} and set no WM_PROJECT_DIR")
    return out
