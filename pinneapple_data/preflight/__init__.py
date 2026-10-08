"""Simulation pre-flight: check an OpenFOAM case or a CalculiX/Abaqus deck before spending solver hours on it.

    from pinneapple_data.preflight import preflight
    report = preflight(files)          # files: {path: bytes}, a case folder or a deck (zips already expanded)
    report["status"]                   # PASS | WARNING | FAIL
    report["findings"]                 # section, status, title, detail, fix, file, line, entity, evidence
"""
from __future__ import annotations

from typing import Any, Dict

from .common import SECTIONS, Finding, Report


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


__all__ = ["preflight", "Finding", "Report", "SECTIONS"]
