"""The lid-driven cavity of Ghia, Ghia and Shin (1982), built, run and compared with the published profiles (#272).

The case is the one every CFD code is checked on first: the unit square, no-slip on three walls, the lid at ``y = 1``
moving at ``U = 1``, so ``Re = U L / nu`` and ``nu = 1 / Re``. What is added here is the rest of the loop that makes
it a *validation* rather than a demonstration: the reference table (:mod:`~pinneapple_simulation.cfd_validation.
reference`), a mesh ladder whose refinement ratio is stated, and a comparison that reports the error against the
table at the points Ghia measured, not at points that happen to be convenient.

Meshes and what the error means
-------------------------------
``n`` is the number of cells per side, uniform, thickness one cell in ``z`` with empty front and back patches — the
mesh the ``icoFoam/cavity`` tutorial uses, at a resolution the caller chooses. The measured ``u`` is taken along
``x = 0.5`` by interpolating the cell values to that column, and then to Ghia's ``y`` stations. The two stations on
the walls (``y = 0`` and ``y = 1``) are **excluded** from the error: there the answer is the boundary condition, not
a solved value, and including them would reward a case for reading its own input.

Why ``simpleFoam`` and where it stops
-------------------------------------
Ghia's tables are steady solutions, so the steady solver is the honest one to compare, and it is what makes a
three-mesh study affordable (a 64x64 cavity at Re = 100 converges in about ten seconds here). At Re = 10000 the steady
incompressible solver does not have a steady solution to find with a first-order diffusive scheme on these meshes:
:func:`study` reports ``converged=False`` rather than a number that looks like agreement, which is the honest result
the issue asks for and the reason the high-Reynolds entry is in the table at all.
"""
from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import foam
from .reference import GHIA_U, GHIA_V, Profile

__all__ = ["CavityRun", "write_case", "run", "sample", "study", "profile_error", "MESHES"]

#: The ladder: ``n`` cells per side, so a refinement ratio of 2 between successive meshes.
MESHES = (32, 64, 128)

_HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"

#: Iteration caps: enough for the residual to fall to the control level, and no more (a cap that is never reached is
#: a cap that hides a run that is not converging).
ITERATIONS = {100: 8000, 400: 8000, 1000: 30000, 10000: 30000}


def _write(case: str, rel: str, cls: str, body: str) -> None:
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(_HDR % (cls, os.path.basename(rel).split(".")[0]) + body)


def write_case(case: str, reynolds: float, n: int = 64, *, iterations: int | None = None,
               residual: float = 1e-6) -> dict[str, Any]:
    """Write a complete, runnable cavity case: ``n x n x 1`` cells on the unit square, laminar, ``nu = 1/Re``.

    ``residual`` is the normalised residual that ends the run. ``1e-6`` is the control these runs use: continuing
    below it costs thousands of iterations for a change in the centreline profile in the sixth digit, which is well
    under the difference between three meshes and therefore not what a mesh study is measuring.

    Returns the parameters it used, so a run can record them without re-deriving them from the dictionaries on disk.
    The dictionaries carry what OpenFOAM v2606 requires and nothing else — the two start-up requirements of the
    incompressible solvers (a pressure reference and ``constant/turbulenceProperties``) are in
    :mod:`pinneapple_data.preflight.limits`, and a case that omits them does not run.
    """
    nu = 1.0 / float(reynolds)
    iters = int(iterations or ITERATIONS.get(int(round(reynolds)), 20000))
    os.makedirs(case, exist_ok=True)
    h = 1.0 / n
    _write(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 {h:.12g}) (1 0 {h:.12g}) (1 1 {h:.12g}) (0 1 {h:.12g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1 1 1) );
boundary
(
    movingWall {{ type wall; faces ((3 7 6 2)); }}
    fixedWalls {{ type wall; faces ((0 4 7 3) (0 1 5 4) (1 2 6 5)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
""")
    _write(case, "system/controlDict", "dictionary", f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {iters};
deltaT 1;
writeControl timeStep;
writeInterval {iters};
purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
runTimeModifiable true;
""")
    _write(case, "system/fvSchemes", "dictionary", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    _write(case, "system/fvSolution", "dictionary", f"""solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0.01; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.1; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
    residualControl {{ p {residual:g}; U {residual:g}; }}
}}
relaxationFactors
{{
    fields {{ p 0.3; }}
    equations {{ U 0.7; }}
}}
""")
    _write(case, "constant/turbulenceProperties", "dictionary", "simulationType laminar;\n")
    _write(case, "constant/transportProperties", "dictionary", f"transportModel Newtonian;\nnu {nu:.12g};\n")
    _write(case, "0/U", "volVectorField", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    movingWall { type fixedValue; value uniform (1 0 0); }
    fixedWalls { type noSlip; }
    frontAndBack { type empty; }
}
""")
    _write(case, "0/p", "volScalarField", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    movingWall { type zeroGradient; }
    fixedWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    return {"reynolds": float(reynolds), "nu": nu, "n": int(n), "cells": n * n, "iterations_cap": iters,
            "residual_control": residual, "solver": "simpleFoam", "scheme": "linearUpwind"}


@dataclass
class CavityRun:
    """One solved cavity: its parameters, its cost, and the profiles measured at Ghia's stations."""

    reynolds: float
    n: int
    cells: int
    iterations: int
    converged: bool
    seconds: float
    residuals: dict[str, float] = field(default_factory=dict)
    u: np.ndarray | None = None            # u at each of GHIA_U[re].points (some NaN where the table has none)
    v: np.ndarray | None = None
    case: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"reynolds": self.reynolds, "n": self.n, "cells": self.cells, "iterations": self.iterations,
                "converged": self.converged, "seconds": round(self.seconds, 3), "residuals": self.residuals,
                "u": None if self.u is None else [None if not np.isfinite(x) else float(x) for x in self.u],
                "v": None if self.v is None else [None if not np.isfinite(x) else float(x) for x in self.v],
                "case": self.case, "note": self.note}


def run(case: str, reynolds: float, n: int = 64, *, timeout: float = 3600.0) -> dict[str, Any]:
    """``blockMesh``, ``simpleFoam`` and ``writeCellCentres`` in ``case``; the wall time and the solver's own report.

    The cell centres are asked of OpenFOAM rather than assumed: every extraction below interpolates the answer onto
    the published stations, and it can only do that if it knows where the cells are.
    """
    t0 = time.time()
    out = foam.run_case(case, commands=["blockMesh", "simpleFoam", "postProcess -func writeCellCentres"],
                        timeout=timeout)
    seconds = time.time() - t0
    log = foam.solver_log(out["text"])
    return {"ok": out["ok"], "failed": out["failed"], "seconds": seconds, "solver": log, "log": out["log"],
            "reynolds": float(reynolds), "n": int(n), "case": case}


# ---------------------------------------------------------------------------------------------- the measured profile
def _lattice(C: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique x, unique y, values on that lattice) from cell centres and one cell-centred field.

    The cases here are structured hex meshes, so the cell centres form a product grid; the field is placed on it by
    rounding each centre to its own coordinate. A mesh that is not a product grid raises instead of silently
    dropping cells (a missing value would otherwise interpolate as a hole).
    """
    x = np.unique(np.round(C[:, 0], 9))
    y = np.unique(np.round(C[:, 1], 9))
    grid = np.full((len(y), len(x)), np.nan)
    ix = np.searchsorted(x, np.round(C[:, 0], 9))
    iy = np.searchsorted(y, np.round(C[:, 1], 9))
    grid[iy, ix] = values
    if np.isnan(grid).any():
        raise ValueError(f"the mesh is not a product grid: {int(np.isnan(grid).sum())} cells left empty "
                         f"({len(x)} x {len(y)} lattice, {len(C)} cells)")
    return x, y, grid


def sample(case: str, reynolds: float, *,
           stations: dict[str, Sequence[float]] | None = None) -> dict[str, np.ndarray]:
    """``u`` along ``x = 0.5`` and ``v`` along ``y = 0.5``, interpolated to Ghia's stations.

    Returns ``{"y", "u", "x", "v"}``: the station coordinates and the measured values at them. Bi-linear: the field is
    interpolated to the centreline from the two neighbouring columns (or rows) and then to the station, so the answer
    is a property of the solution rather than of where the cells happen to fall.

    ``stations`` overrides the two station lists (``{"y": ...}`` for ``u``, ``{"x": ...}`` for ``v``). A run at a
    Reynolds number Ghia did not publish needs it: the stations are the same at every column of the table, so a
    solution at another ``Re`` can be read at them and put beside a model evaluated at them. Without it the lists of
    the published column nearest ``reynolds`` are used, and a ``reynolds`` no column has is a :class:`KeyError` — the
    case is then only readable at stations the caller names, which is what the argument is for.
    """
    data = foam.cells(case)
    for need in ("C", "U"):
        if need not in data:
            raise KeyError(f"{case}: field {need} missing (run with writeCellCentres, as :func:`run` does)")
    C = np.asarray(data["C"], dtype=float)
    U = np.asarray(data["U"], dtype=float)
    x, y, gu = _lattice(C, U[:, 0])
    _, _, gv = _lattice(C, U[:, 1])
    given = stations or {}
    ref_u: Profile | None = GHIA_U.get(int(round(reynolds)))
    ref_v: Profile | None = GHIA_V.get(int(round(reynolds)))
    if ref_u is None or ref_v is None:
        if "y" not in given or "x" not in given:
            raise KeyError(f"Re = {float(reynolds):g} is not a column of the table: give both station lists "
                           f"(stations={{'y': ..., 'x': ...}}) to read the case at the ones you mean")
        y_stations, x_stations = np.asarray(given["y"], dtype=float), np.asarray(given["x"], dtype=float)
    else:
        y_stations = np.asarray(given.get("y", ref_u.points), dtype=float)
        x_stations = np.asarray(given.get("x", ref_v.points), dtype=float)

    if not x[0] <= 0.5 <= x[-1] or not y[0] <= 0.5 <= y[-1]:
        raise ValueError(f"{case}: the lattice {x[0]}..{x[-1]} x {y[0]}..{y[-1]} does not contain both centrelines")
    # column by column: u(x = 0.5) at every y, then to the stations the table publishes
    j = np.searchsorted(x, 0.5)
    j = min(max(j, 1), len(x) - 1)
    w = (0.5 - x[j - 1]) / (x[j] - x[j - 1])
    u_mid = (1.0 - w) * gu[:, j - 1] + w * gu[:, j]
    i = min(max(np.searchsorted(y, 0.5), 1), len(y) - 1)
    wv = (0.5 - y[i - 1]) / (y[i] - y[i - 1])
    v_mid = (1.0 - wv) * gv[i - 1, :] + wv * gv[i, :]
    return {"y": y_stations, "u": np.interp(y_stations, y, u_mid),
            "x": x_stations, "v": np.interp(x_stations, x, v_mid)}


def profile_error(profile: Profile, measured: np.ndarray, *, interior_only: bool = True,
                  drop_disputed: bool = True) -> dict[str, float]:
    """The error of a measured profile against the published one: ``l2``, ``linf`` and the relative ``l2``.

    ``interior_only`` leaves out the two stations on the walls, where the published value is the boundary condition
    and the measured one is an interpolation of the first cell off the wall. ``drop_disputed`` leaves out the stations
    a run has shown the transcription to be wrong at (:attr:`~pinneapple_simulation.cfd_validation.reference.Profile.
    disputed`), which are recorded in the table and in the note that says what was measured there — dropping them from
    the *number* is not the same as deleting them from the record. The relative norm is over the RMS of the reference
    values, which is the number to quote: an absolute ``u`` error of 0.01 means little without it.
    """
    ref = np.asarray(profile.values, dtype=float)
    got = np.asarray(measured, dtype=float)
    if ref.shape != got.shape:
        raise ValueError(f"{profile.case} {profile.quantity}: {ref.size} reference values, {got.size} measured")
    keep = np.ones(ref.size, dtype=bool)
    if interior_only:
        keep[0] = keep[-1] = False
    for i in profile.disputed if drop_disputed else ():
        keep[i] = False
    keep &= np.isfinite(got)
    if keep.sum() < 3:
        raise ValueError(f"{profile.case} {profile.quantity}: {int(keep.sum())} comparable points")
    d = got[keep] - ref[keep]
    l2 = float(np.sqrt(np.mean(d ** 2)))
    rms = float(np.sqrt(np.mean(ref[keep] ** 2)))
    return {"l2": l2, "linf": float(np.abs(d).max()), "l2_relative": l2 / rms if rms else float("nan"),
            "points": int(keep.sum()), "worse_at": float(profile.points[int(np.argmax(np.abs(got - ref)))])}


def study(reynolds: float, root: str, *, meshes: Sequence[int] = MESHES, timeout: float = 3600.0,
          reuse: bool = False) -> dict[str, Any]:
    """Run the case on a ladder of meshes and compare each with the published table.

    Returns the runs (one :class:`CavityRun` each, as records), the error of each against Ghia's profiles, and the
    ratio of successive errors — the convergence of the *error*, which is what a mesh study is: a refinement that
    halves the error is a second-order scheme, and the ladder is what shows whether that is what happened.

    ``reuse`` reads a case that is already solved on disk instead of solving it again, which is what a re-analysis of
    a committed run needs: reading the fields back takes a second, and re-solving takes the wall time in the table.
    """
    runs: list[CavityRun] = []
    for n in meshes:
        case = os.path.join(root, f"cavity_re{int(reynolds)}_{n}")
        if reuse and _solved(case):
            info = {"ok": True, "failed": None, "seconds": float("nan"), "solver": foam.solver_log(_log_text(case))}
            cells = _cells(case)
        else:
            cells = write_case(case, reynolds, n)["cells"]      # the dictionaries first: ``run`` only runs them, so a
            info = run(case, reynolds, n, timeout=timeout)      # fresh directory used to record a failed run and to
            #                                                     succeed only on the second call
        rec = CavityRun(reynolds=float(reynolds), n=int(n), cells=int(cells),
                        iterations=int(info["solver"]["iterations"]), converged=bool(info["solver"]["converged"]),
                        seconds=float(info["seconds"]), residuals=dict(info["solver"]["residuals"]), case=case)
        if not info["solver"]["iterations"]:
            rec.note = f"the solver did not reach a time step ({info['failed'] or 'no output'})"
        else:
            try:
                got = sample(case, reynolds)
                rec.u, rec.v = got["u"], got["v"]
            except (KeyError, ValueError, FileNotFoundError) as exc:
                rec.note = f"the fields could not be read back: {exc}"
        runs.append(rec)
    errors = {}
    for rec in runs:
        if rec.u is None:
            continue
        errors[rec.n] = {"u": profile_error(GHIA_U[int(round(reynolds))], rec.u),
                         "v": profile_error(GHIA_V[int(round(reynolds))], rec.v)}
    ratios = {}
    ns = sorted(errors)
    for a, b in zip(ns, ns[1:], strict=False):
        ratios[f"{a}_{b}"] = {q: errors[a][q]["l2"] / errors[b][q]["l2"] if errors[b][q]["l2"] else float("inf")
                              for q in ("u", "v")}
    return {"case": "ghia-1982", "reynolds": float(reynolds), "meshes": list(meshes),
            "citation": GHIA_U[int(round(reynolds))].citation,
            "runs": [r.to_dict() for r in runs], "errors": errors, "error_ratios": ratios,
            "self_convergence": _self_convergence(runs)}


def _solved(case: str) -> bool:
    """Whether ``case`` already has a solved time directory with the cell centres (:func:`run` writes them)."""
    return os.path.isdir(case) and bool(foam.cells_available(case))


def _log_text(case: str) -> str:
    path = os.path.join(case, "log.simpleFoam")
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _cells(case: str) -> int:
    try:
        return int(len(foam.cells(case, time=foam.last_time(case))["C"]))
    except (KeyError, FileNotFoundError):
        return 0


def _self_convergence(runs: Sequence[CavityRun]) -> dict[str, Any]:
    """The observed order of the *solution* between the three meshes, which is the part no table is needed for.

    The error against Ghia's table has a floor — the table's own rounding and the difference between their scheme and
    ours — so it does not fall with refinement once a mesh is fine enough, and reading an order off it would report
    that floor rather than the discretisation. Refining the solution against *itself* is what a grid-convergence study
    does, and it is the number the GCI is computed from: ``richardson_extrapolate`` on the three measured profiles.
    """
    from pinneapple_analysis.verification.convergence import richardson_extrapolate

    usable = [r for r in runs if r.u is not None and r.v is not None]
    if len(usable) < 3:
        return {"available": False, "reason": f"{len(usable)} of {len(runs)} meshes produced a profile"}
    coarse, medium, fine = usable[0], usable[1], usable[2]
    out: dict[str, Any] = {"available": True, "meshes": [coarse.n, medium.n, fine.n],
                           "station": "the Ghia stations of the profile"}
    for name in ("u", "v"):
        try:
            conv = richardson_extrapolate(getattr(coarse, name), getattr(medium, name), getattr(fine, name), 2.0)
            out[name] = {"observed_order": float(conv.observed_order), "gci_fine": float(conv.gci_fine),
                         "gci_coarse": float(conv.gci_coarse), "asymptotic": bool(conv.is_asymptotic),
                         "extrapolated": [float(x) for x in np.asarray(conv.extrapolated_value).ravel()]}
        except ValueError as exc:
            out[name] = {"error": str(exc)}
    return out
