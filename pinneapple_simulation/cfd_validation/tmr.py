"""The NASA Turbulence Modeling Resource flat plate in OpenFOAM, with its grid-convergence study (#265).

The Turbulence Modeling Resource (TMR, ``https://tmbwg.github.io/turbmodels/``) publishes RANS verification cases with
the exact model definitions and grid-converged reference results from CFL3D and FUN3D. Its 2D zero-pressure-gradient
flat plate is the one that tests a turbulence model and nothing else: one wall, a known inflow, and a skin friction
that must follow the turbulent boundary layer.

What this module does, and what it does not
-------------------------------------------
It writes the case, runs it with ``SpalartAllmaras`` and ``kOmegaSST`` on a family of grids refined by two, and reports
the observed order of convergence and the Richardson-extrapolated skin friction from
:func:`pinneapple_analysis.verification.convergence.richardson_extrapolate` — the general three-grid machinery the
repository already had, applied to a turbulence case instead of reimplemented for it.

It does **not** read the TMR grids or claim agreement with the CFL3D/FUN3D numbers. The grids here are regenerated at
equivalent resolution (the TMR publishes Plot3D grids that would have to be converted, and the case page's reference
tables are not in this repository), so the comparison table waits for :func:`read_reference` to be given the TMR data
file. What is measured here is what stands without it: whether the answer converges, at what order, to what value, and
whether that value is in the place a turbulent flat plate puts it (the 1/7-power skin-friction law, to the accuracy
that law has).

Geometry and Reynolds number
----------------------------
The plate is the unit length ``x in [0, 1]`` at ``y = 0``, with the domain starting a third of a chord upstream
(``x = -1/3``) and the far boundary at ``y = 0.5``. ``Re = 5e6`` per unit length, from the TMR flat-plate case page,
and the flow is incompressible here (``simpleFoam``) rather than at the page's ``M = 0.2``: at that Mach number the
difference in skin friction is below the discretisation error of a mesh this size, and saying so is cheaper than
believing it.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import foam
from .cavity import _write

__all__ = ["PLATE_RE", "LEVELS", "MODELS", "write_case", "run", "skin_friction", "study", "read_reference",
           "CF_LAW", "PlateRun"]

#: The TMR flat-plate Reynolds number, per unit length of ``x`` (``benchmark_suites.NASA_TMR_CASES``, case ``2DZP``).
PLATE_RE = 5e6

#: Grids refined by two, the same way the TMR family is.
LEVELS = ({'name': "coarse", 'nx': 30, 'ny': 24}, {'name': "medium", 'nx': 60, 'ny': 48},
          {'name': "fine", 'nx': 120, 'ny': 96})

#: The wall distance of the first cell centre, in wall units. The models here use wall functions
#: (``nutUSpaldingWallFunction``, ``omegaWallFunction``), which want the first cell *outside* the viscous sublayer; a
#: fully resolved mesh (``y+ ~ 1``) would need an order of magnitude more cells in ``y`` than a three-grid study can
#: afford, and the same study with a resolved wall is the next step, not this one.
YPLUS = 30.0

#: The two models the issue names. ``SpalartAllmaras`` is the one-field model; ``kOmegaSST`` the two-equation one.
MODELS = ("SpalartAllmaras", "kOmegaSST")

#: Prandtl's 1/7-power turbulent flat-plate law, ``cf = 0.027 Re_x^(-1/7)``, as given in Schlichting's *Boundary-Layer
#: Theory*: an approximation to about 10 %, used here as an independent sanity check of a value the TMR tables would
#: settle exactly.
CF_LAW = {"coefficient": 0.027, "exponent": -1.0 / 7.0,
          "citation": "Schlichting, Boundary-Layer Theory (7th ed.), the 1/7-power skin-friction law"}


def _growth_for(first: float, height: float, n: int) -> float:
    """The geometric growth ratio of ``n`` cells from ``first`` at the wall to ``height`` at the far field.

    ``simpleGrading`` takes the ratio of the **last cell to the first**, not the per-cell ratio: a plain ``1.12`` over
    twenty cells is very nearly a uniform mesh whose first cell centre lands at ``y = 0.0118`` (measured), which puts
    the wall function three hundred wall units off the wall. The grading written here is ``r**(n - 1)`` for the ``r``
    that solves ``first * (r**n - 1) / (r - 1) = height``.
    """
    lo, hi = 1.0000001, 3.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if first * (r ** n - 1) / (r - 1) > height:
            hi = r
        else:
            lo = r
    r = 0.5 * (lo + hi)
    return r


def write_case(case: str, reynolds: float = PLATE_RE, *, level: Dict[str, Any] = LEVELS[0],
               model: str = "kOmegaSST", iterations: int = 4000) -> Dict[str, Any]:
    """Write the flat-plate case: graded in ``y`` from a first cell that puts ``y+`` near one at ``Re = 5e6``.

    The first cell height follows ``y+ = 1`` for a flat plate, ``h1 = y+ nu / u_tau`` with ``u_tau`` from the 1/7-power
    law — an estimate, and the right one to make: the wall function the models use wants its first cell inside the
    log layer, and a mesh whose first cell is ten times too tall is a different (and much easier) case.
    """
    if model not in MODELS:
        raise ValueError(f"{model!r} is not one of {MODELS}")
    nu = 1.0 / float(reynolds)
    u_inf = 1.0
    cf = CF_LAW["coefficient"] * float(reynolds) ** CF_LAW["exponent"]
    u_tau = u_inf * math.sqrt(cf / 2.0)
    first = YPLUS * nu / u_tau                     # the first cell height for that wall distance
    nx, ny = int(level["nx"]), int(level["ny"])
    x0, x1, height = -1.0 / 3.0, 1.0, 0.5
    growth = _growth_for(first, height, ny)         # the per-cell ratio whose 20/40/80 cells reach the far field
    grading = growth ** (ny - 1)
    os.makedirs(case, exist_ok=True)
    _write(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices
(
    ({x0:.12g} 0 0) ({x1:.12g} 0 0) ({x1:.12g} {height:.12g} 0) ({x0:.12g} {height:.12g} 0)
    ({x0:.12g} 0 {x1 - x0:.12g}) ({x1:.12g} 0 {x1 - x0:.12g}) ({x1:.12g} {height:.12g} {x1 - x0:.12g})
    ({x0:.12g} {height:.12g} {x1 - x0:.12g})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 {grading:.6g} 1)
);
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    plate {{ type wall; faces ((0 1 5 4)); }}
    farfield {{ type patch; faces ((3 7 6 2)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
""")
    _write(case, "system/controlDict", "dictionary", f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {int(iterations)};
deltaT 1;
writeControl timeStep;
writeInterval {int(iterations)};
purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
runTimeModifiable true;
""")
    divs = """    div(phi,U) Gauss linearUpwind grad(U);
    div(phi,k) Gauss upwind;
    div(phi,omega) Gauss upwind;
    div(phi,nuTilda) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
"""
    _write(case, "system/fvSchemes", "dictionary", "ddtSchemes { default steadyState; }\n"
           "gradSchemes { default Gauss linear; }\n"
           f"divSchemes {{ default none;\n{divs}}}\n"
           "laplacianSchemes { default Gauss linear corrected; }\n"
           "interpolationSchemes { default linear; }\n"
           "snGradSchemes { default corrected; }\n"
           "wallDist { method meshWave; }\n")
    solvers = """    p { solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
"""
    if model == "kOmegaSST":
        solvers += """    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
"""
    else:
        solvers += """    nuTilda { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
"""
    extra = "k 1e-5; omega 1e-5; " if model == "kOmegaSST" else "nuTilda 1e-5; "
    _write(case, "system/fvSolution", "dictionary", f"""solvers
{{
{solvers}}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl {{ p 1e-6; U 1e-6; {extra}}}
}}
relaxationFactors
{{
    fields {{ p 0.3; }}
    equations {{ U 0.7; {model} 0.7; }}
}}
""")
    _write(case, "constant/transportProperties", "dictionary",
           f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu:.9g};\n")
    turb = ("RAS\n{\n    RASModel        SpalartAllmaras;\n    turbulence      on;\n"
            "    printCoeffs     on;\n}\n" if model == "SpalartAllmaras" else
            "RAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n")
    _write(case, "constant/turbulenceProperties", "dictionary", f"simulationType  RAS;\n{turb}")
    # Free-stream turbulence: 1 % intensity and a turbulent viscosity ten times the laminar one, which is what keeps
    # the boundary layer from staying laminar (a free-stream omega on the order of U^2/nu switches the model off).
    k = 1.5 * (0.01 * u_inf) ** 2
    omega = k / (10.0 * nu)
    _write(case, "0/U", "volVectorField", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform ({u_inf:.6g} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({u_inf:.6g} 0 0); }}
    outlet {{ type zeroGradient; }}
    plate {{ type noSlip; }}
    farfield {{ type slip; }}
    frontAndBack {{ type empty; }}
}}
""")
    _write(case, "0/p", "volScalarField", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    plate { type zeroGradient; }
    farfield { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    if model == "kOmegaSST":
        _write(case, "0/k", "volScalarField", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform {k:.6g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {k:.6g}; }}
    outlet {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}
    plate {{ type kqRWallFunction; value uniform {k:.6g}; }}
    farfield {{ type slip; }}
    frontAndBack {{ type empty; }}
}}
""")
        _write(case, "0/omega", "volScalarField", f"""dimensions [0 0 -1 0 0 0 0];
internalField uniform {omega:.6g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {omega:.6g}; }}
    outlet {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}
    plate {{ type omegaWallFunction; value uniform {omega:.6g}; }}
    farfield {{ type slip; }}
    frontAndBack {{ type empty; }}
}}
""")
        _write(case, "0/nut", "volScalarField", f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    plate {{ type nutUSpaldingWallFunction; value uniform 0; }}
    farfield {{ type calculated; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""")
    else:
        ninf = 3.0 * nu
        _write(case, "0/nuTilda", "volScalarField", f"""dimensions [0 2 -1 0 0 0 0];
internalField uniform {ninf:.6g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {ninf:.6g}; }}
    outlet {{ type zeroGradient; }}
    plate {{ type fixedValue; value uniform 0; }}
    farfield {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
""")
        _write(case, "0/nut", "volScalarField", """dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    plate { type nutUSpaldingWallFunction; value uniform 0; }
    farfield { type calculated; value uniform 0; }
    frontAndBack { type empty; }
}
""")
    return {"reynolds": float(reynolds), "model": model, "level": level["name"], "nx": nx, "ny": ny,
            "cells": nx * ny, "nu": nu, "first_cell_height": first, "growth": growth,
            "grading": grading,
            "x_range": [x0, x1], "height": height, "iterations_cap": int(iterations),
            "cf_law_at_re": cf, "u_tau_estimate": u_tau}


def run(case: str, *, timeout: float = 3600.0) -> Dict[str, Any]:
    """``blockMesh``, ``simpleFoam``, cell centres — the same three commands as every other laminar case here."""
    out = foam.run_case(case, commands=["blockMesh", "simpleFoam", "postProcess -func writeCellCentres"],
                        timeout=timeout)
    return {"ok": out["ok"], "failed": out["failed"], "seconds": float(sum(out["seconds"].values())),
            "solver": foam.solver_log(out["text"]), "log": out["log"], "case": case}


def skin_friction(case: str, *, u_inf: float = 1.0, nu: float = 2e-7) -> Dict[str, np.ndarray]:
    """``cf(x)`` on the plate from the first cell row: ``cf = 2 tau_w / U^2``, ``tau_w = (nu + nut) du/dy``.

    The **effective** viscosity is what matters here: with a wall function the first cell sits in the log layer, where
    ``nu du/dy`` alone under-predicts the wall shear by orders of magnitude (measured: 2.8e-5 against a law that says
    3.0e-3) — the turbulent viscosity at that cell is what carries the momentum. ``du/dy`` is ``u_1 / y_1`` from the
    no-slip wall, first-order accurate and the estimate the mesh supports. ``x`` comes back with the values so a caller
    can sample any station.
    """
    data = foam.cells(case)
    C = np.asarray(data["C"], dtype=float)
    U = np.asarray(data["U"], dtype=float)[:, 0]
    nut = np.asarray(data.get("nut", np.zeros(len(C))), dtype=float).reshape(len(C))
    on_plate = np.isclose(C[:, 1], C[:, 1].min(), atol=1e-12) & (C[:, 0] > 0.0)
    if not on_plate.any():
        raise ValueError(f"{case}: no cells on the plate")
    y = C[on_plate, 1]
    x = C[on_plate, 0]
    u = U[on_plate]
    nu_eff = nu + nut[on_plate]
    order = np.argsort(x)
    x, y, u, nu_eff = x[order], y[order], u[order], nu_eff[order]
    cf = 2.0 * nu_eff * u / y / (u_inf ** 2)
    return {"x": x, "cf": cf, "y_first": y, "nut": nut[on_plate][order]}


def read_reference(path: str) -> Dict[str, np.ndarray]:
    """A TMR ``cf`` table as ``{"x", "cf"}``: two whitespace-separated columns, one station per line.

    The TMR publishes its reference results as plain text files next to each case. They are not in this repository (and
    not downloadable from the machine this was built on), so the comparison is a call away rather than a number in a
    README: give this the file and :func:`study` reports the difference.
    """
    x, cf = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or line.lstrip().startswith(("#", "%")):
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            x.append(a)
            cf.append(b)
    if len(x) < 2:
        raise ValueError(f"{path}: no x cf columns found")
    return {"x": np.asarray(x), "cf": np.asarray(cf)}


@dataclass
class PlateRun:
    """One solved flat plate: its grid, its model, its cost, and the skin friction it produced."""

    model: str
    level: str
    cells: int
    iterations: int
    converged: bool
    seconds: float
    x: Optional[np.ndarray] = None
    cf: Optional[np.ndarray] = None
    residuals: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def at(self, station: float) -> float:
        """The skin friction interpolated at ``x = station`` (``nan`` when the run produced nothing)."""
        if self.x is None or self.cf is None:
            return float("nan")
        return float(np.interp(station, self.x, self.cf))

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "level": self.level, "cells": self.cells, "iterations": self.iterations,
                "converged": self.converged, "seconds": round(self.seconds, 2),
                "cf_at_0.97": self.at(0.97), "residuals": self.residuals, "note": self.note}


def study(root: str, *, reynolds: float = PLATE_RE, levels: Sequence[Dict[str, Any]] = LEVELS,
          models: Sequence[str] = MODELS, station: float = 0.97, reference: Optional[str] = None,
          timeout: float = 3600.0) -> Dict[str, Any]:
    """Run both models on every level, and report the observed order and the Richardson-extrapolated ``cf``.

    The station defaults to ``x = 0.97``: far enough downstream that the boundary layer is developed and far enough
    from the trailing edge that the outlet's treatment of it does not dominate. ``reference`` is a TMR file for
    :func:`read_reference`; without it the table has the extrapolated value and the 1/7-power law's estimate, and says
    the comparison is pending rather than implying agreement.
    """
    from pinneapple_analysis.verification.convergence import richardson_extrapolate

    out: Dict[str, Any] = {"case": "nasa-tmr-2DZP", "reynolds": float(reynolds), "station": station,
                           "cf_law": CF_LAW, "models": {}, "levels": [lv["name"] for lv in levels],
                           "reference": reference or "", "note": ""}
    for model in models:
        runs: List[PlateRun] = []
        for level in levels:
            case = os.path.join(root, f"plate_{model}_{level['name']}")
            params = write_case(case, reynolds, level=level, model=model)
            info = run(case, timeout=timeout)
            got = PlateRun(model=model, level=level["name"], cells=int(params["cells"]),
                           iterations=int(info["solver"]["iterations"]), converged=bool(info["solver"]["converged"]),
                           seconds=float(info["seconds"]), residuals=dict(info["solver"]["residuals"]))
            if not info["ok"]:
                got.note = f"{info['failed']} failed: {info['log'][-800:]}"
            else:
                try:
                    cf = skin_friction(case, nu=params["nu"])
                    got.x, got.cf = cf["x"], cf["cf"]
                except (KeyError, ValueError, FileNotFoundError) as exc:
                    got.note = f"the fields could not be read back: {exc}"
            runs.append(got)
        values = [r.at(station) for r in runs]
        entry = {"runs": [r.to_dict() for r in runs], "cf": values}
        if all(np.isfinite(values)) and len(values) == 3:
            try:
                conv = richardson_extrapolate(*values, (2.0, 2.0))
                entry.update(observed_order=conv.observed_order, extrapolated=float(conv.extrapolated_value),
                             gci_fine=conv.gci_fine, asymptotic=conv.is_asymptotic)
            except (ValueError, ZeroDivisionError) as exc:
                entry["convergence_error"] = str(exc)
        law = CF_LAW["coefficient"] * (reynolds * station) ** CF_LAW["exponent"]
        entry["cf_law"] = float(law)
        entry["cf_fine_vs_law"] = float(values[-1] / law) if np.isfinite(values[-1]) and law else float("nan")
        if reference and os.path.isfile(reference):
            ref = read_reference(reference)
            entry["cf_reference"] = float(np.interp(station, ref["x"], ref["cf"]))
            entry["cf_fine_vs_reference"] = (float(values[-1] / entry["cf_reference"])
                                             if np.isfinite(values[-1]) else float("nan"))
        out["models"][model] = entry
    if not out["note"]:
        out["note"] = ("the TMR reference tables are not in this repository and were not downloadable here: "
                       "pass reference=<file> to compare with CFL3D/FUN3D")
    return out
