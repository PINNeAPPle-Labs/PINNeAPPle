"""Where OpenFOAM is on this machine, and how a validation case is run and read back (issues #272, #265, #294).

Every case in this package has the same three steps: write the dictionaries (:mod:`~pinneapple_simulation.
cfd_validation.cavity`, :mod:`~pinneapple_simulation.cfd_validation.step`), run the solver, and read the answer back
as numbers. This module owns the middle one, so that "no OpenFOAM here" is said once, in one way, and every caller
— the tests, the example and the scripts — skips instead of failing.

Finding it
----------
``$FOAM_BASHRC`` first, then the two installs that put it somewhere predictable: the macOS cask at
``/Applications/OpenFOAM-*.app/Contents/Resources/etc/bashrc`` and the Linux packages under ``/opt``, ``/usr/lib``
and ``/usr/local``. That is the same search ``tests/test_openfoam_limits.py`` does, kept here so a validation run and
the limits test agree on what "OpenFOAM is installed" means.

Running it
----------
:func:`sh` sources the bashrc and runs one command inside the case, capturing the log text: OpenFOAM's environment
is a shell environment, and nothing about it is importable. :func:`run` is the whole sequence for the cases here —
``blockMesh``, the solver, then ``postProcess -func writeCellCentres`` so the fields come back with the cell centres
of the mesh they belong to, not an index into a grid we assumed.

Reading it back
---------------
:func:`cells` reads one time directory with :mod:`pinneapple_data.cae.foam` (the repository's own polyMesh and field
reader, so nothing here needs ParaView or a VTK install) and returns ``{"C": (n, 3), "U": (n, 3), ...}`` keyed by
field name. A case this package wrote always has ``C``: :func:`run` asks the solver for it.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["bashrc", "available", "command", "sh", "run_case", "cells", "case_files", "solver_log",
           "time_dirs", "last_time", "cells_available", "patch_centres", "patch_values", "FOAM_APP_CANDIDATES"]


def _candidates() -> list[str]:
    out = [os.environ.get("FOAM_BASHRC") or ""]
    out += sorted(glob.glob("/Applications/OpenFOAM-*.app/Contents/Resources/etc/bashrc"), reverse=True)
    for root in ("/opt", "/usr/lib/openfoam", "/usr/local", "/usr/share/openfoam"):
        if os.path.isdir(root):
            out += sorted(glob.glob(os.path.join(root, "openfoam*/etc/bashrc")), reverse=True)
            out += sorted(glob.glob(os.path.join(root, "OpenFOAM-*/etc/bashrc")), reverse=True)
            out.append(os.path.join(root, "etc/bashrc"))
    return out


#: Everything :func:`bashrc` looks at, in order: the variable first, then the usual install prefixes.
FOAM_APP_CANDIDATES = tuple(_candidates())


def bashrc() -> str | None:
    """The bashrc of the installed OpenFOAM, or ``None``. Answer the question, never raise: a caller that needs the
    solver asks :func:`available` and skips, and a caller that only reports asks the same thing."""
    for cand in FOAM_APP_CANDIDATES:
        if cand and os.path.isfile(cand):
            return cand
    return None


def command(name: str) -> str | None:
    """Absolute path of one OpenFOAM executable (``simpleFoam``, ``blockMesh``...) under the install, or ``None``."""
    env = bashrc()
    if env is None:
        return None
    r = subprocess.run(["bash", "-c", f'source "{env}" >/dev/null 2>&1; command -v {name}'],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip() or None


def available(*needs: str) -> bool:
    """Whether OpenFOAM is installed **and** the named executables exist (``available("simpleFoam")``)."""
    return all(command(n) for n in needs) if needs else bashrc() is not None


def sh(case: str, cmd: str, *, timeout: float = 3600.0, log: str = "log", check: bool = True
       ) -> subprocess.CompletedProcess:
    """Run one OpenFOAM command inside ``case`` with the bashrc sourced, its output in ``log.<name>`` and returned.

    ``check=True`` raises :class:`RuntimeError` carrying the last lines of the log, which is where OpenFOAM says what
    it did not like; ``check=False`` hands the failure back so a caller can read it (a solver that diverged is a
    result to report, not an exception to swallow).
    """
    env = bashrc()
    if env is None:
        raise RuntimeError("no OpenFOAM installation found: set FOAM_BASHRC to its etc/bashrc")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(cmd.split()[0]))
    path = os.path.join(case, f"log.{log if log != 'log' else name}")
    full = f'source "{env}" >/dev/null 2>&1; cd "{case}" && ( {cmd} ) > "{path}" 2>&1'
    r = subprocess.run(["bash", "-c", full], capture_output=True, text=True, timeout=timeout)
    r.text = _tail(path, None)              # the whole log: a solver's iteration count is at the end of a long file
    r.tail = r.text[-4000:]
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed in {case} (exit {r.returncode}):\n{r.tail}")
    return r


def _tail(path: str, n: int | None = 4000) -> str:
    """The log's text (``n=None``) or its last ``n`` characters, which is what an error message needs."""
    try:
        with open(path, errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    return text if n is None else text[-n:]


def run_case(case: str, *, commands: Sequence[str], timeout: float = 3600.0) -> dict[str, Any]:
    """Run ``commands`` in order in ``case``; the wall time of each, and whether all of them ended cleanly."""
    timings = {}
    text, tail = [], []
    for cmd in commands:
        t0 = time.time()
        r = sh(case, cmd, timeout=timeout, check=False)
        timings[cmd] = round(time.time() - t0, 3)
        text.append(r.text)
        tail.append(r.tail)
        if r.returncode != 0:
            return {"ok": False, "failed": cmd, "seconds": timings, "log": "\n".join(tail),
                    "text": "\n".join(text)}
    return {"ok": True, "failed": None, "seconds": timings, "log": "\n".join(tail), "text": "\n".join(text)}


def solver_log(text: str) -> dict[str, Any]:
    """The numbers a solver run leaves in its log: iterations, whether it converged, and the last residual.

    OpenFOAM prints one ``Time = n`` per outer iteration and either ``SIMPLE solution converged in N iterations`` or
    just stops; reading the count from the log is the only way to know how many iterations a run took. The initial
    residual of the last iteration is there for the same reason: a run that stopped without converging says how far
    it got.
    """
    iters = len(re.findall(r"^Time = ", text, flags=re.M))
    conv = re.search(r"converged in (\d+) iterations", text)
    residuals = re.findall(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)", text)
    last = {}
    for field, value in residuals:
        try:
            last[field] = float(value)
        except ValueError:
            continue
    executed = re.search(r"ExecutionTime = ([0-9.]+) s", text)
    return {"iterations": int(conv.group(1)) if conv else iters,
            "converged": bool(conv),
            "residuals": last,
            "solver_seconds": float(executed.group(1)) if executed else None}


# --------------------------------------------------------------------------------------------- reading the answer
_TIME_RE = re.compile(r"^\d+(\.\d+)?$")


def case_files(case: str, time: str | None = None) -> dict[str, bytes]:
    """The bytes of one case, keyed by path relative to its root — what the repository's readers take.

    The mesh (``constant/polyMesh``), the ``system`` and ``constant`` dictionaries and the time directories: enough
    for :func:`cells` to read the answer and for :func:`pinneapple_data.preflight.openfoam.check` to pre-flight the
    case, so a case this package writes goes through the same rules as every other case in the repository before a
    solver is spent on it. ``time`` selects one numeric directory (default: every one of them).
    """
    root = os.path.abspath(case)
    fs: dict[str, bytes] = {}
    for sub in ("constant/polyMesh/", "polyMesh/", ""):
        d = os.path.join(root, sub) if sub else root
        if not os.path.isdir(d):
            continue
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            for cand in (name, name + ".gz"):
                p = os.path.join(d, cand)
                if os.path.isfile(p):
                    with open(p, "rb") as fh:
                        fs[sub + cand] = fh.read()
                    break
        if any(k.endswith("points") or k.endswith("points.gz") for k in fs):
            break
    for sub in ("system/", "constant/"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    fs[sub + name] = fh.read()
    for d in sorted(os.listdir(root)):
        if not (_TIME_RE.match(d) and os.path.isdir(os.path.join(root, d))):
            continue
        if time is not None and d != time:
            continue
        for name in sorted(os.listdir(os.path.join(root, d))):
            p = os.path.join(root, d, name)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    fs[f"{d}/{name}"] = fh.read()
    return fs


def time_dirs(case: str) -> list[str]:
    """The numeric time directories of a case, ascending."""
    root = os.path.abspath(case)
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if _TIME_RE.match(d) and os.path.isdir(os.path.join(root, d))],
                  key=float)


def last_time(case: str) -> str | None:
    """The last numeric time directory, or ``None``."""
    got = time_dirs(case)
    return got[-1] if got else None


def cells_available(case: str) -> bool:
    """Whether a case has a solved time directory with the cell centres in it (:func:`run` asks for them)."""
    t = last_time(case)
    return bool(t) and t != "0" and os.path.isfile(os.path.join(case, t, "C"))


def patch_centres(case: str, patch: str) -> dict[str, Any]:
    """The faces of one patch: their centres, the ``y`` extent of their *nodes*, and how many there are.

    Read from ``constant/polyMesh`` rather than assumed, because a boundary value has to be placed where the mesh put
    the face, in the mesh's own order. The extent comes from the nodes and not from the centres: half a cell at each
    end is the difference between a profile that reaches zero at a wall and one that reaches zero at the first cell.
    """
    from pinneapple_data.cae import foam as reader

    root = os.path.abspath(case)
    fs = {k: v for k, v in case_files(root, time=None).items()
          if k.endswith("points") or k.endswith("faces") or k.endswith("owner")
          or k.endswith("neighbour") or k.endswith("boundary")}
    if not fs:
        raise FileNotFoundError(f"{root}: no polyMesh (run blockMesh first)")
    mesh = reader.read_polymesh(fs, "")
    found = [p for p in mesh.poly.patches if p["name"] == patch]
    if not found:
        raise KeyError(f"{root}: no patch {patch!r} (have: {', '.join(p['name'] for p in mesh.poly.patches)})")
    p = found[0]
    start, count = int(p["startFace"]), int(p["nFaces"])
    centres, ys = [], []
    for f in range(start, start + count):
        a, b = int(mesh.poly.face_offsets[f]), int(mesh.poly.face_offsets[f + 1])
        nodes = mesh.points[mesh.poly.face_nodes[a:b]]
        centres.append(nodes.mean(axis=0))
        ys += [float(v) for v in nodes[:, 1]]
    if not centres:
        raise ValueError(f"{root}: patch {patch!r} has no faces")
    return {"centres": np.asarray(centres, dtype=float), "y_min": min(ys), "y_max": max(ys),
            "faces": len(centres), "start": start, "count": count}


def patch_values(case: str, field: str, patch: str, time: str | None = None) -> np.ndarray:
    """The boundary values of one field on one patch, as ``(n_faces, components)``.

    Reads the field's ``boundaryField`` with :func:`pinneapple_data.cae.foam.read_boundary_values`, which is how a
    wall-quantity the solver wrote (``wallShearStress``) comes back as numbers instead of as a file to be eyeballed.
    """
    from pinneapple_data.cae import foam as reader

    root = os.path.abspath(case)
    t = time or last_time(root)
    if t is None:
        raise FileNotFoundError(f"{root}: no time directory")
    fs = case_files(root, time=t)
    mesh = reader.read_polymesh(fs, "")
    key = f"{t}/{field}"
    if key not in fs:
        raise KeyError(f"{root}: no {field!r} in time {t} (did the post-processing write it?)")
    arr, unit, cls = reader.read_field(fs[key], mesh.n_cells)
    width = 1 if np.ndim(arr) == 1 else int(np.shape(arr)[1])
    mesh.cell_data_tmp = np.asarray(arr)
    try:
        values = reader.read_boundary_values(fs[key], mesh, width)
    finally:
        del mesh.cell_data_tmp
    if values is None:
        raise ValueError(f"{root}: {field!r} has no boundaryField to read")
    got = patch_centres(root, patch)
    a = got["start"] - mesh.poly.n_internal
    return np.asarray(values[a:a + got["count"]], dtype=float)


def cells(case: str, time: str | None = None, fields: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Every volume field of one time directory of ``case`` as arrays over the mesh cells, plus ``C`` (cell centres).

    ``time`` defaults to the last numeric time directory the solver wrote. Reads with
    :mod:`pinneapple_data.cae.foam`, the repository's own ASCII/binary polyMesh and field reader, so a validation run
    has no dependency beyond numpy.
    """
    from pinneapple_data.cae import foam

    root = os.path.abspath(case)
    times = [d for d in os.listdir(root) if _TIME_RE.match(d) and os.path.isdir(os.path.join(root, d))]
    if not times:
        raise FileNotFoundError(f"{root}: no time directory with fields (did the case run?)")
    t = time or max(times, key=float)
    fs = case_files(root, t)
    mesh = foam.read_polymesh(fs, "")
    foam.read_time(fs, mesh, "", str(t), only=list(fields) if fields else None)
    return {k: np.asarray(v) for k, v in mesh.cell_data.items()}
