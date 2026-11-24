"""The backward-facing step and its reattachment length, built and measured (#272).

Armaly et al. (1983) made this the standard separated-flow case: a channel of inlet height ``h_i`` opening into one of
height ``H`` (their expansion ratio is 1 : 1.9423), and the quantity everyone reports is where the recirculation behind
the step closes again — ``x1 / S``, the distance from the step to the point on the lower wall where the wall shear
changes sign. It is a good validity test because the answer is one number and it is sensitive to the inlet profile,
the outlet placement and the scheme.

How the step is made
--------------------
``blockMesh`` cannot leave a hole in a block, and a hand-written block dictionary has to get the orientation of every
exposed face right. So the mesh is one block over the whole domain and the step is *cut* out of it: ``topoSet`` marks
the cells below the step, and ``subsetMesh -overwrite -patch stepWall`` removes them and turns the newly exposed faces
into the wall patch, with the orientation taken from the mesh rather than from us. Three commands instead of one, and
no face ordering to get wrong.

The geometry in cells
---------------------
The step is ``S = 1`` and ``m`` cells tall; the inlet channel is ``round(m / (expansion - 1))`` cells, so the mesh's
expansion ratio is within a cell of Armaly's 1.9423 and is recorded as ``expansion_mesh`` with every run. The Reynolds
number is Armaly's: ``Re = U_mean 2 h_i / nu`` on the hydraulic diameter of the inlet channel.

The reference is not transcribed
--------------------------------
:data:`~pinneapple_simulation.cfd_validation.reference.ARMALY_REATTACHMENT` is empty, on purpose: the published
reattachment curve is a figure, and the digitisation that circulates is a secondary source with no single provenance.
:func:`study` therefore reports ``x1/S`` and its convergence with the mesh, and no ``error`` column: what does not
depend on the table is measured here, and the rest waits for the numbers to be entered from the paper itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from . import foam
from .cavity import _write

__all__ = ["StepRun", "write_case", "reattachment_length", "study", "MESHES", "EXPANSION"]

#: Armaly et al. (1983): the downstream channel is 1.9423 times the inlet height.
EXPANSION = 1.9423

#: Cells across the step height, so the ladder refines by two.
MESHES = (8, 16, 32)


def write_case(case: str, reynolds: float, cells_per_step: int = 16, *, expansion: float = EXPANSION,
               lin: float = 5.0, lout: float = 30.0, iterations: int = 20000) -> Dict[str, Any]:
    """Write the step case: one block, then the cut. ``lin``/``lout`` are the inlet and outlet lengths in step
    heights, and are this package's choice rather than Armaly's rig (recorded with every run)."""
    m = int(cells_per_step)
    n_step = m
    n_in = max(2, int(round(m / (expansion - 1.0))))
    n_y = n_step + n_in
    h = 1.0 / m                                   # cell size: the step height is exactly 1
    height = n_y * h
    nu = 2.0 * (height - 1.0) / float(reynolds)   # Re on the hydraulic diameter of the inlet channel
    nx = max(8, int(round((lin + lout) / h)))
    x0 = -lin
    x1 = lout
    os.makedirs(case, exist_ok=True)
    _write(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices
(
    ({x0:.12g} 0 0) ({x1:.12g} 0 0) ({x1:.12g} {height:.12g} 0) ({x0:.12g} {height:.12g} 0)
    ({x0:.12g} 0 {h:.12g}) ({x1:.12g} 0 {h:.12g}) ({x1:.12g} {height:.12g} {h:.12g}) ({x0:.12g} {height:.12g} {h:.12g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {n_y} 1) simpleGrading (1 1 1) );
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    walls {{ type wall; faces ((0 1 5 4) (3 7 6 2)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
""")
    # The cells *below* the step (x < 0 and y < 1) are the solid; ``subsetMesh`` keeps the set it is given and drops
    # everything else, so the set is the fluid: the whole domain, with the step region marked and then inverted away.
    _write(case, "system/topoSetDict", "dictionary", f"""actions
(
    {{
        name fluidCells;
        type cellSet;
        action new;
        source boxToCell;
        box ({x0 - 1:.12g} {-0.5 * h:.12g} {-h:.12g}) (0 {1.0 - 1e-6:.12g} {2 * h:.12g});
    }}
    {{
        name fluidCells;
        type cellSet;
        action invert;
    }}
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
    _write(case, "system/fvSolution", "dictionary", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-11; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-11; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
    residualControl { p 1e-6; U 1e-6; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""")
    _write(case, "constant/turbulenceProperties", "dictionary", "simulationType laminar;\n")
    _write(case, "constant/transportProperties", "dictionary", f"transportModel Newtonian;\nnu {nu:.12g};\n")
    # The inlet starts as zero and is written *after* the cut (see :func:`write_inlet`): ``subsetMesh`` subsets the
    # fields together with the mesh and checks their length against the uncut mesh, so a per-face profile written here
    # would have to guess the order in which blockMesh numbers the inlet faces — and be wrong about half of them.
    _write(case, "0/U", "volVectorField",
           "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n"
           "    inlet { type fixedValue; value uniform (0 0 0); }\n"
           "    outlet { type zeroGradient; }\n"
           "    walls { type noSlip; }\n"
           "    frontAndBack { type empty; }\n}\n")
    _write(case, "0/p", "volScalarField", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }
}
""")
    inlet_h = height - 1.0                              # the inlet channel: everything above the step
    cells = nx * n_y - int(round(lin * m)) * n_step      # the block, less the cells the cut removes
    return {"reynolds": float(reynolds), "cells_per_step": m, "cells": cells,
            "expansion_mesh": height / inlet_h, "expansion_paper": expansion, "nu": nu, "step_height": 1.0,
            "inlet_height": inlet_h, "lin_over_S": lin, "lout_over_S": lout, "iterations_cap": int(iterations),
            "note": "lin and lout are this package's choice, not Armaly's rig"}


def inlet_face_centres(case: str, patch: str = "inlet") -> Dict[str, Any]:
    """The faces of one patch, read from the mesh: their centres, and the ``y`` extent of their *nodes*.

    Read from ``constant/polyMesh`` rather than assumed, because the profile has to be written on the faces the mesh
    actually has, in the order the mesh numbers them — which is why it is written after the cut. The extent comes from
    the face nodes, not from the centres: half a cell at each end is exactly the difference between a profile that
    reaches zero at the walls and one that reaches zero at the first cell.
    """
    from pinneapple_data.cae import foam as reader

    root = os.path.abspath(case)
    fs = {}
    for sub in ("constant/polyMesh/", "polyMesh/"):
        if os.path.isdir(os.path.join(root, sub)):
            for name in ("points", "faces", "owner", "neighbour", "boundary"):
                for cand in (name, name + ".gz"):
                    p = os.path.join(root, sub + cand)
                    if os.path.isfile(p):
                        with open(p, "rb") as fh:
                            fs[sub + cand] = fh.read()
                        break
            break
    if not fs:
        raise FileNotFoundError(f"{root}: no polyMesh (run blockMesh and the cut first)")
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
    return {"centres": np.asarray(centres, dtype=float), "y_min": min(ys), "y_max": max(ys), "faces": len(centres)}


def write_inlet(case: str, mean_velocity: float = 1.0, *, patch: str = "inlet") -> Dict[str, Any]:
    """Write the parabolic inlet profile over the faces the cut left, and return what it wrote.

    ``u(y) = 6 U s (1 - s)`` with ``s`` the fraction of the inlet height, which is the fully developed laminar channel
    profile and has mean ``U``. The height is taken from the mesh, so a case whose expansion ratio came out a cell off
    gets the profile of the channel it has rather than of the one intended.
    """
    got = inlet_face_centres(case, patch)
    y0, y1 = got["y_min"], got["y_max"]
    s = (got["centres"][:, 1] - y0) / (y1 - y0)
    u = 6.0 * mean_velocity * s * (1.0 - s)
    _write(case, "0/U", "volVectorField",
           "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n"
           f"    {patch}\n    {{\n        type fixedValue;\n        value nonuniform List<vector> {len(u)} (\n"
           + "".join(f"        ({v:.12g} 0 0)\n" for v in u)
           + "        );\n    }\n"
           "    outlet { type zeroGradient; }\n"
           "    walls { type noSlip; }\n"
           "    frontAndBack { type empty; }\n}\n")
    return {"patch": patch, "faces": int(len(u)), "mean_velocity": float(u.mean()),
            "height": float(y1 - y0), "u_max": float(u.max())}


def run(case: str, reynolds: float, cells_per_step: int = 16, *, mean_velocity: float = 1.0,
        timeout: float = 3600.0) -> Dict[str, Any]:
    """Mesh, cut the step, write the inlet profile, solve, and write the cell centres."""
    mesh = foam.run_case(case, commands=["blockMesh", "topoSet", 'subsetMesh -overwrite fluidCells -patches "(walls)"'],
                         timeout=timeout)
    if not mesh["ok"]:
        return {"ok": False, "failed": mesh["failed"], "seconds": float(sum(mesh["seconds"].values())),
                "solver": foam.solver_log(mesh["text"]), "log": mesh["log"], "case": case,
                "reynolds": float(reynolds), "inlet": None}
    inlet = write_inlet(case, mean_velocity)
    out = foam.run_case(case, commands=["simpleFoam", "postProcess -func writeCellCentres"], timeout=timeout)
    seconds = float(sum(mesh["seconds"].values()) + sum(out["seconds"].values()))
    return {"ok": out["ok"], "failed": out["failed"], "seconds": seconds,
            "solver": foam.solver_log(out["text"]), "log": mesh["log"] + out["log"], "case": case,
            "reynolds": float(reynolds), "inlet": inlet}


def reattachment_length(case: str, *, step_height: float = 1.0) -> Dict[str, Any]:
    """``x1 / S`` behind the step: where the streamwise velocity on the first cell row changes sign.

    On the first row the velocity is the wall shear up to its sign (no-slip at ``y = 0``), so the reattachment point is
    that crossing, linearly interpolated between the two cell centres that bracket it. Returns ``x1_over_S``, the
    interval it lies in, and ``u_min`` — the most negative velocity found, so that a run with no recirculation at all
    shows it here instead of reporting a reattachment length of zero as if it were an answer.
    """
    data = foam.cells(case)
    C = np.asarray(data["C"], dtype=float)
    U = np.asarray(data["U"], dtype=float)[:, 0]
    downstream = C[:, 0] > 0.0
    if not downstream.any():
        raise ValueError(f"{case}: no cells downstream of the step")
    y = np.unique(np.round(C[downstream, 1], 12))
    row = downstream & np.isclose(C[:, 1], y[0], atol=1e-12)
    order = np.argsort(C[row, 0])
    x, u = C[row, 0][order], U[row][order]
    neg = np.flatnonzero(u < 0)
    if neg.size == 0:
        return {"x1_over_S": 0.0, "x_first": float(x.min()), "x_last": float(x.max()), "u_min": float(u.min()),
                "note": "no recirculation on the first cell row: there is no reattachment point to measure"}
    i = int(neg[-1])
    if i + 1 >= len(x):
        return {"x1_over_S": float("nan"), "x_first": float(x[i]), "x_last": float(x.max()),
                "u_min": float(u.min()), "note": "the recirculation reaches the outlet: lout is too short"}
    crossing = x[i] + (x[i + 1] - x[i]) * (0.0 - u[i]) / (u[i + 1] - u[i])
    return {"x1_over_S": float(crossing) / step_height, "x_first": float(x[i]), "x_last": float(x[i + 1]),
            "u_min": float(u.min()), "note": ""}


@dataclass
class StepRun:
    """One solved step case: its parameters, its cost and the reattachment length it produced."""

    reynolds: float
    cells_per_step: int
    cells: int
    iterations: int
    converged: bool
    seconds: float
    x1_over_S: Optional[float] = None
    u_min: Optional[float] = None
    residuals: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"reynolds": self.reynolds, "cells_per_step": self.cells_per_step, "cells": self.cells,
                "iterations": self.iterations, "converged": self.converged, "seconds": round(self.seconds, 3),
                "x1_over_S": None if self.x1_over_S is None else float(self.x1_over_S),
                "u_min": None if self.u_min is None else float(self.u_min),
                "residuals": self.residuals, "note": self.note}


def study(reynolds: float, root: str, *, meshes: Sequence[int] = MESHES, lin: float = 5.0, lout: float = 30.0,
          timeout: float = 3600.0) -> Dict[str, Any]:
    """Run the step at ``reynolds`` on a ladder of meshes and report ``x1/S`` on each.

    No error column: the published reattachment curve is not transcribed (see the module docstring), so what this
    returns is the measurement and its convergence with the mesh, which is the part that does not depend on the table.
    """
    runs: list[StepRun] = []
    for m in meshes:
        case = os.path.join(root, f"step_re{int(reynolds)}_m{m}")
        params = write_case(case, reynolds, m, lin=lin, lout=lout)
        info = run(case, reynolds, m, timeout=timeout)
        log = info["solver"]
        run_ = StepRun(reynolds=float(reynolds), cells_per_step=int(m), cells=int(params["cells"]),
                       iterations=int(log["iterations"]), converged=bool(log["converged"]),
                       seconds=float(info["seconds"]), residuals=dict(log["residuals"]))
        if not info["ok"]:
            run_.note = f"{info['failed']} failed: {info['log'][-1200:]}"
        else:
            try:
                got = reattachment_length(case)
                run_.x1_over_S, run_.u_min = got["x1_over_S"], got["u_min"]
                run_.note = got.get("note", "")
            except (KeyError, ValueError, FileNotFoundError) as exc:
                run_.note = f"the fields could not be read back: {exc}"
        runs.append(run_)
    return {"case": "armaly-1983", "reynolds": float(reynolds), "meshes": list(meshes),
            "expansion_paper": EXPANSION,
            "reference": "not transcribed: the published curve is a figure (see the module docstring)",
            "runs": [r.to_dict() for r in runs]}
