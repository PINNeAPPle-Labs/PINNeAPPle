"""OpenFOAM case for one airfoil at one angle of attack: simpleFoam, k-omega SST resolved to the wall (y+ < 1).

Non-dimensional: chord 1, |U| = 1, nu = 1/Re. Far field with freestream conditions, forces with forceCoeffs about the
quarter chord. ``run_case`` runs it and returns the coefficients, their convergence and the wall/cell fields.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np

from .mesh import GridSpec, write_polymesh

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")
HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"


def _w(case: str, rel: str, cls: str, body: str) -> None:
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(HDR % (cls, os.path.basename(rel)) + body)


def write_case(case: str, shape: np.ndarray, alpha_deg: float, reynolds: float = 4e6, iterations: int = 1200,
               tu: float = 0.001, grid: GridSpec = GridSpec(), write_every: int = 100) -> None:
    if os.path.isdir(case):
        shutil.rmtree(case)
    write_polymesh(shape, case, grid)
    a = math.radians(alpha_deg)
    U = f"({math.cos(a):.10f} {math.sin(a):.10f} 0)"
    nu = 1.0 / reynolds
    k = 1.5 * tu ** 2
    omega = k / nu / 1.0                                          # nut/nu = 1 in the free stream
    wall_far = lambda w, f: ("boundaryField\n{\n    airfoil\n    {\n%s    }\n    farfield\n    {\n%s    }\n"  # noqa: E731
                             "    frontAndBack\n    {\n        type empty;\n    }\n}\n") % (w, f)
    _w(case, "0/U", "volVectorField", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform {U};\n" + wall_far(
        "        type noSlip;\n", f"        type freestreamVelocity;\n        freestreamValue uniform {U};\n"))
    _w(case, "0/p", "volScalarField", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n" + wall_far(
        "        type zeroGradient;\n", "        type freestreamPressure;\n        freestreamValue uniform 0;\n"))
    io = lambda v: f"        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"  # noqa: E731
    _w(case, "0/k", "volScalarField", f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k:.6g};\n" + wall_far(
        "        type fixedValue;\n        value uniform 1e-14;\n", io(f"{k:.6g}")))
    _w(case, "0/omega", "volScalarField", f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {omega:.6g};\n" + wall_far(
        f"        type omegaWallFunction;\n        value uniform {omega:.6g};\n", io(f"{omega:.6g}")))
    _w(case, "0/nut", "volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n" + wall_far(
        "        type nutLowReWallFunction;\n        value uniform 0;\n", "        type calculated;\n        value uniform 0;\n"))
    _w(case, "constant/transportProperties", "dictionary", f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu:.6g};\n")
    _w(case, "constant/turbulenceProperties", "dictionary",
       "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n")
    _w(case, "system/controlDict", "dictionary", f"""application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {iterations};
deltaT 1;
writeControl timeStep;
writeInterval {write_every};
purgeWrite 6;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
""")
    _w(case, "system/fvSchemes", "dictionary", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; grad(k) cellLimited Gauss linear 1; grad(omega) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss limitedLinear 1;
    div(phi,omega) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited 0.5; }
wallDist { method meshWave; }
""")
    _w(case, "system/fvSolution", "dictionary", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-6; U 1e-7; "(k|omega)" 1e-7; }
}
relaxationFactors
{
    equations { U 0.8; k 0.6; omega 0.6; }
    fields { p 1; }
}
""")


def read_internal(path: str) -> np.ndarray:
    """internalField of an ASCII OpenFOAM field file -> (n,) or (n, 3)."""
    txt = open(path).read()
    m = re.search(r"internalField\s+uniform\s+(\(([^)]*)\)|\S+);", txt)
    if m:
        raise ValueError(f"{path}: uniform field")
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", txt)
    kind, n = m.group(1), int(m.group(2))
    body = txt[m.end(): txt.index("\n)\n", m.end())]
    if kind == "vector":
        return np.array(body.replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)
    return np.array(body.split(), float)


def wall_geometry(P2: np.ndarray) -> Dict[str, np.ndarray]:
    """Per wall face (i = 0..ni-1): centre, length, unit normal into the body, distance of the first cell centre."""
    from .mesh import cell_centres
    w = P2[0]
    a, b = w, np.roll(w, -1, 0)
    e = b - a
    L = np.linalg.norm(e, axis=1)
    n_body = np.stack([e[:, 1], -e[:, 0]], 1) / L[:, None]      # clockwise wall: (dy, -dx) points into the body
    c = 0.5 * (a + b)
    cc = cell_centres(P2)[0]
    d = np.abs(((cc - c) * n_body).sum(1))
    return {"centre": c, "length": L, "n_body": n_body, "d": d}


def forces(P2: np.ndarray, p_wall: np.ndarray, U_first: np.ndarray, nu: float, alpha_deg: float) -> Dict[str, Any]:
    """Cl, Cd, Cm (quarter chord, nose-up positive) and the Cp / Cf distributions from the wall-adjacent cells."""
    g = wall_geometry(P2)
    n = g["n_body"]
    u = U_first[:, :2]
    ut = u - (u * n).sum(1, keepdims=True) * n                   # tangential velocity of the first cell
    tau = nu * ut / g["d"][:, None]
    f = (p_wall[:, None] * n + tau) * g["length"][:, None]
    F = f.sum(0)
    a = math.radians(alpha_deg)
    drag, lift = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    r = g["centre"] - np.array([0.25, 0.0])
    Mz = (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]).sum()
    q = 0.5
    tx = np.roll(P2[0], -1, 0) - P2[0]
    tx = tx / np.linalg.norm(tx, axis=1, keepdims=True)          # wall direction of the loop
    cf = (tau * tx).sum(1) / q
    return {"Cl": float(F @ lift / q), "Cd": float(F @ drag / q), "Cm": float(-Mz / q),
            "Cd_pressure": float(((p_wall[:, None] * n) * g["length"][:, None]).sum(0) @ drag / q),
            "Cp": (p_wall / q).tolist(), "Cf": cf.tolist(), "x": g["centre"][:, 0].tolist(), "y": g["centre"][:, 1].tolist()}


def time_dirs(case: str):
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if t > 0 and os.path.exists(os.path.join(case, d, "p")):
            out.append((t, os.path.join(case, d)))
    return sorted(out)


def read_coefficients(case: str, P2: np.ndarray, nu: float, alpha_deg: float, ni: int) -> Dict[str, Any]:
    """Coefficients at every kept write (convergence check) and the full result at the last one."""
    hist = []
    last = None
    for t, d in time_dirs(case):
        p = read_internal(os.path.join(d, "p"))
        U = read_internal(os.path.join(d, "U"))
        r = forces(P2, p[:ni], U[:ni], nu, alpha_deg)
        hist.append((t, r["Cl"], r["Cd"], r["Cm"]))
        last = (t, d, r)
    if not last:
        return {}
    t, d, r = last
    H = np.array(hist)
    tail = H[-5:]
    out: Dict[str, Any] = {"iterations": int(t), "Cl": r["Cl"], "Cd": r["Cd"], "Cm": r["Cm"],
                           "Cd_pressure": r["Cd_pressure"], "wall": {k: r[k] for k in ("Cp", "Cf", "x", "y")},
                           "time_dir": d}
    for k, c in (("Cl", 1), ("Cd", 2), ("Cm", 3)):
        out[k + "_osc"] = float(tail[:, c].max() - tail[:, c].min()) if len(tail) > 1 else 0.0
    out["history"] = H.round(6).tolist()
    return out


def run_case(case: str, shape: np.ndarray, alpha_deg: float, reynolds: float = 4e6, iterations: int = 1200,
             grid: GridSpec = GridSpec(), timeout: float = 1800) -> Dict[str, Any]:
    """Write, run simpleFoam, return coefficients, convergence verdict, timing."""
    write_case(case, shape, alpha_deg, reynolds, iterations, grid=grid)
    t0 = time.time()
    cmd = f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && simpleFoam > log.simpleFoam 2>&1'
    try:
        subprocess.run(["bash", "-c", cmd], timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": time.time() - t0}
    from .mesh import grid_points
    P2 = grid_points(shape, grid)
    try:
        res = read_coefficients(case, P2, 1.0 / reynolds, alpha_deg, grid.ni)
    except Exception as e:                                        # noqa: BLE001
        res = {"error": str(e)}
    res["seconds"] = round(time.time() - t0, 1)
    log = open(os.path.join(case, "log.simpleFoam")).read() if os.path.exists(os.path.join(case, "log.simpleFoam")) else ""
    m = re.search(r"SIMPLE solution converged in (\d+) iterations", log)
    if "FOAM FATAL" in log or "nan" in log.lower().split("time =")[-1] or "Cl" not in res:
        res["status"] = "failed"
    elif m or (res.get("Cl_osc", 1) < 2e-3 and res.get("Cd_osc", 1) < 5e-5):
        res["status"] = "converged"
    elif res.get("Cl_osc", 1) < 0.02 and res.get("Cd_osc", 1) < 1e-3:
        res["status"] = "steady-ish"
    else:
        res["status"] = "unsteady"                                 # oscillating: separated flow near/after stall
    res["converged_in"] = int(m.group(1)) if m else None
    return res
