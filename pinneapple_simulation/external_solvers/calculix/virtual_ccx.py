"""A drop-in ``ccx``: same command line, same ``.inp`` in, same ``.frd`` out, fields from a model.

Idea (user request, 2026-09-25): CalculiX (implicit) and OpenRadioss (explicit) already hide FEM
from engineers behind a deck -> solver -> results workflow. Putting a PINN/surrogate *where the
solver is* makes it usable from the tools engineers already have (PrePoMax, cgx, FreeCAD FEM,
ccx2paraview) with no new UI. Replacing the Fortran core of ccx is not practical; replacing the
executable at its input/output interface gives the same effect: point the pre-processor's
"CalculiX executable" setting at ``pinneapple-ccx`` (see :func:`main`).

Guard rail: a model only answers inside its validity envelope (bounding box of the geometry it
was trained on, and the load cases it covers, as declared by the model). Outside it, the real
``ccx`` runs (``--fallback``) or the call fails loudly; a silent extrapolation is never written.

A predictor is any callable ``predict(model: CalculixModel, coords: (N,3)) -> {"DISP": (N,3),
optional "STRESS": (N,6)}`` with an optional ``envelope`` attribute:
``{"bbox": [[xmin,ymin,zmin],[xmax,ymax,zmax]], "E": [lo, hi], "load": [lo, hi]}``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np

from .frd import write_frd
from .inp import CalculixModel, read_inp


class OutOfEnvelope(RuntimeError):
    pass


def check_envelope(model: CalculixModel, coords: np.ndarray, envelope: Optional[dict], tol: float = 1e-6) -> None:
    if not envelope:
        return
    problems = []
    if "bbox" in envelope:
        lo, hi = (np.asarray(v, float) for v in envelope["bbox"])
        span = np.maximum(hi - lo, 1e-12)
        if np.any(coords.min(0) < lo - tol * span) or np.any(coords.max(0) > hi + tol * span):
            problems.append(f"geometry {coords.min(0).round(6).tolist()}..{coords.max(0).round(6).tolist()} "
                            f"outside {lo.tolist()}..{hi.tolist()}")
    if "E" in envelope and not (envelope["E"][0] <= model.E <= envelope["E"][1]):
        problems.append(f"E={model.E:g} outside {envelope['E']}")
    if "load" in envelope and model.cloads:
        total = sum(abs(v) for _, _, v in model.cloads)
        if not (envelope["load"][0] <= total <= envelope["load"][1]):
            problems.append(f"total point load {total:g} outside {envelope['load']}")
    if problems:
        raise OutOfEnvelope("; ".join(problems))


def predict_to_frd(inp_path: str, predictor: Callable, frd_path: Optional[str] = None) -> str:
    """Read ``inp_path``, predict nodal fields, write the matching ``.frd``; returns its path."""
    model = read_inp(inp_path)
    ids, coords = model.node_array()
    check_envelope(model, coords, getattr(predictor, "envelope", None))
    fields = predictor(model, coords)
    for k, v in fields.items():
        if np.asarray(v).shape[0] != len(ids):
            raise ValueError(f"predictor returned {k} with {np.asarray(v).shape[0]} rows for {len(ids)} nodes")
    frd_path = frd_path or os.path.splitext(inp_path)[0] + ".frd"
    return write_frd(frd_path, ids, coords, model.elements, model.element_type, fields,
                     source=f"pinneapple virtual ccx: {getattr(predictor, '__name__', type(predictor).__name__)}")


def load_predictor(spec: str) -> Callable:
    """``"package.module:attr"`` -> the predictor object (a callable, optionally with ``envelope``)."""
    mod, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod), attr or "predictor")
    return obj() if isinstance(obj, type) else obj


def main(argv=None) -> int:
    """``pinneapple-ccx -i job [--predictor pkg.mod:obj] [--fallback]`` (mirrors ``ccx -i job``).

    The predictor can also come from ``PINNEAPPLE_CCX_PREDICTOR`` so that a pre-processor that only
    passes ``-i job`` can still use it.
    """
    ap = argparse.ArgumentParser(prog="pinneapple-ccx")
    ap.add_argument("-i", dest="job", required=True)
    ap.add_argument("--predictor", default=os.environ.get("PINNEAPPLE_CCX_PREDICTOR"))
    ap.add_argument("--fallback", action="store_true", default=os.environ.get("PINNEAPPLE_CCX_FALLBACK") == "1",
                    help="run the real ccx when the model cannot answer (no predictor / out of envelope)")
    a, _ = ap.parse_known_args(argv)
    inp = a.job if a.job.endswith(".inp") else a.job + ".inp"
    job = os.path.splitext(os.path.basename(inp))[0]
    workdir = os.path.dirname(os.path.abspath(inp))
    log = {"job": job, "mode": None}
    try:
        if not a.predictor:
            raise OutOfEnvelope("no predictor configured")
        frd = predict_to_frd(inp, load_predictor(a.predictor))
        log.update(mode="surrogate", frd=frd, predictor=a.predictor)
        print(f" pinneapple virtual ccx: {job}.frd written by {a.predictor}\n\n Job finished")
        code = 0
    except OutOfEnvelope as e:
        if not a.fallback:
            print(f" pinneapple virtual ccx: cannot answer ({e}); rerun with --fallback to use ccx", file=sys.stderr)
            return 2
        from .runner import run_ccx
        r = run_ccx(workdir, job)
        log.update(mode="ccx_fallback", reason=str(e), returncode=r.returncode)
        print(r.stdout)
        code = r.returncode
    with open(os.path.join(workdir, job + ".pinneapple.json"), "w") as f:
        json.dump(log, f, indent=1)  # provenance: which answer the engineer got, and why
    return code


if __name__ == "__main__":
    raise SystemExit(main())
