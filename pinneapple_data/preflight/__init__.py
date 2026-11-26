"""Simulation pre-flight: check an OpenFOAM case or a CalculiX/Abaqus deck before spending solver hours on it.

    from pinneapple_data.preflight import preflight
    report = preflight(files)          # files: {path: bytes}, a case folder or a deck (zips already expanded)
    report["status"]                   # PASS | WARNING | FAIL
    report["findings"]                 # section, status, title, detail, fix, file, line, entity, evidence
"""
from __future__ import annotations

import re
from typing import Any, Dict

from .common import SECTIONS, Finding, Report
from .limits import DECISIONS, LIMITS, Limit, decision, limits_for, unsupported


def preflight(fs: Dict[str, bytes]) -> Dict[str, Any]:
    from pinneapple_data.cae.io import kind_of
    names = list(fs)
    roots = [n[: -len("system/controlDict")] for n in names if n.endswith("system/controlDict")]
    if roots:
        from . import openfoam
        root = min(roots, key=len)
        rep = openfoam.check(fs, root)
    else:
        inps = [n for n in names if n.lower().endswith(".inp")]
        if not inps:
            raise ValueError("No simulation case found: upload an OpenFOAM case (with system/controlDict) or a "
                             "CalculiX/Abaqus .inp deck.")
        from . import calculix
        # the main deck is the one with *STEP (others are usually *INCLUDE'd mesh files)
        main = max(inps, key=lambda n: (b"*STEP" in fs[n].upper(), len(fs[n])))
        rep = calculix.check(fs, main)
    out = rep.to_dict()
    out["files"] = [{"name": n, "bytes": len(b)} for n, b in sorted(fs.items())]
    out["usage"] = {"cases": 1}
    return out


_RESULT_DIR = re.compile(r"\d+(\.\d+)?(e[-+]?\d+)?")
_RESULT_EXT = (".frd", ".dat", ".sta", ".cvg", ".12d", ".vtk", ".vtu")


def is_result_file(path: str) -> bool:
    """A solver result rather than an input: inside an OpenFOAM time directory other than ``0`` (``0.5/U``,
    ``1e-3/p``) or a CalculiX / VTK result file. Not needed for a pre-flight; drop such files before checking."""
    parts = path.split("/")
    return any(_RESULT_DIR.fullmatch(p) and p != "0" for p in parts[:-1]) or path.lower().endswith(_RESULT_EXT)


def drop_results(fs: Dict[str, bytes]) -> Dict[str, bytes]:
    """``fs`` without the solver result files (:func:`is_result_file`)."""
    return {k: v for k, v in fs.items() if not is_result_file(k)}


def attach_snippets(report: Dict[str, Any], fs: Dict[str, bytes], context: int = 4, width: int = 160) -> None:
    """Add to every finding that names a file and line the source lines around it, in place:
    ``finding["snippet"] = {"start": first line number, "lines": [...], "at": line}`` (``context`` lines on each
    side, each cut to ``width`` characters). File names are resolved against the case root of the report."""
    root = report.get("info", {}).get("root", ".")
    pre = "" if root in (".", "") else root.rstrip("/") + "/"
    for f in report["findings"]:
        fn, ln = f.get("file"), f.get("line")
        if not fn or not ln:
            continue
        b = fs.get(pre + fn) or fs.get(fn) or next((v for k, v in fs.items() if k.endswith("/" + fn) or k == fn), None)
        if b is None:
            continue
        lines = b.decode("latin-1").splitlines()
        a, z = max(0, ln - context), min(len(lines), ln + context)
        f["snippet"] = {"start": a + 1, "lines": [s[:width] for s in lines[a:z]], "at": ln}


__all__ = ["preflight", "attach_snippets", "drop_results", "is_result_file", "Finding", "Report", "SECTIONS"]
