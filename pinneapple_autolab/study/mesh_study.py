"""Grid convergence of a simulator's outputs on a mesh family: Richardson extrapolation and the grid-convergence
index (Roache 1994; the procedure of Celik et al. 2008, J. Fluids Eng. 130, 078001), from
:func:`pinneapple_design.fitmesh.report.gci`.

    study = grid_convergence(records, ("CD", "CL"))      # records of one design on meshes coarse -> fine
    study["CD"]["gci_fine"], study["CD"]["extrapolated"], study["CD"]["p"]

The representative cell size of each mesh is h = N^(-1/3) relative to the finest (only size ratios enter the GCI).
With four or more meshes the finest three give the GCI and each consecutive triplet is reported, to show whether the
family is in the asymptotic range; a triplet that converges oscillatorily is flagged, not forced through the
formula. ``choose_mesh`` picks the cheapest mesh whose error against the extrapolated value is within a tolerance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["grid_convergence", "choose_mesh"]


def grid_convergence(levels: Sequence[Mapping[str, Any]], quantities: Sequence[str]) -> dict[str, Any]:
    """``levels``: ``[{"name", "cells", "values": {q: value}, "cost_s"}, ...]`` from coarse to fine (at least 3)."""
    from pinneapple_design.fitmesh.report import gci
    lv = sorted(levels, key=lambda d: d["cells"])
    if len(lv) < 3:
        raise ValueError("a grid-convergence study needs at least three meshes")
    N = np.array([float(d["cells"]) for d in lv])
    h = (N / N[-1]) ** (-1.0 / 3.0)
    out: dict[str, Any] = {"meshes": [{"name": d["name"], "cells": int(d["cells"]), "h_rel": float(hh),
                                       "cost_s": float(d.get("cost_s", np.nan))} for d, hh in zip(lv, h, strict=False)]}
    for q in quantities:
        f = [float(d["values"][q]) for d in lv]
        trip = []
        for i in range(len(lv) - 2):
            g = gci([f[i + 2], f[i + 1], f[i]], [h[i + 2], h[i + 1], h[i]])
            trip.append({"meshes": [lv[i]["name"], lv[i + 1]["name"], lv[i + 2]["name"]], **g})
        best = trip[-1]
        ext = best.get("extrapolated")
        out[q] = {"values": f, "triplets": trip, "status": best.get("status"), "p": best.get("p"),
                  "extrapolated": ext, "gci_fine": best.get("gci_fine"), "note": best.get("note", ""),
                  "error_vs_extrapolated": [None if ext is None else float(abs(v - ext) / abs(ext)) for v in f],
                  "error_vs_finest": [float(abs(v - f[-1]) / abs(f[-1])) for v in f]}
    return out


def choose_mesh(study: Mapping[str, Any], quantity: str, tol: float) -> dict[str, Any]:
    """The coarsest mesh whose ``quantity`` is within ``tol`` (relative) of the Richardson value, or of the finest
    mesh when the family is not in the asymptotic range (no extrapolated value); with the reason."""
    q = study[quantity]
    errs = q["error_vs_extrapolated"] if q.get("extrapolated") is not None else q["error_vs_finest"]
    basis = "Richardson extrapolation" if q.get("extrapolated") is not None else "finest mesh (no extrapolation)"
    for m, e in zip(study["meshes"], errs, strict=False):
        if e is not None and e <= tol:
            return {"mesh": m["name"], "error": e, "basis": basis, "tol": tol, "cells": m["cells"]}
    m = study["meshes"][-1]
    return {"mesh": m["name"], "error": errs[-1], "basis": basis, "tol": tol, "cells": m["cells"],
            "note": "no mesh within the tolerance: the finest is used"}
