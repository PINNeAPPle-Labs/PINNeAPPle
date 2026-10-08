"""3D OpenFOAM case of a whole aircraft (half model, symmetry plane): STL from the Airframe, snappyHexMesh,
simpleFoam k-omega SST with wall functions, run in parallel. Forces, surface pressure and streamlines are computed
from the written fields (no function objects needed).

    write_case3d(case, airframe, alpha_deg=2.0, speed=55.0)  ->  run_case3d(case, procs=4)  ->  read_result3d(case)
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

from .airframe import Airframe, Part, to_stl

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")
HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"
BODIES = ("fuselage", "wing", "htail", "vtail")


def _w(case, rel, cls, body):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(HDR % (cls, os.path.basename(rel)) + body)


def _bodies(af: Airframe) -> Dict[str, List[Part]]:
    parts = af.build("cfd")
    out = {b: [] for b in BODIES}
    for p in parts:
        if not p.aero or p.name.endswith("_left"):
            continue                                            # half model: right side + the fuselage
        for b in BODIES:
            if p.name.startswith(b):
                out[b].append(p)
    # the fuselage was split by material: merge back into one closed surface
    return out


def write_case3d(case: str, af: Airframe, alpha_deg: float = 2.0, speed: float = 55.0, nu: float = 1.5e-5,
                 iterations: int = 800, level_wing=(5, 6), procs: int = 4, level_fus=(4, 5)) -> Dict[str, Any]:
    if os.path.isdir(case):
        shutil.rmtree(case)
    tri = os.path.join(case, "constant", "triSurface")
    os.makedirs(tri)
    bodies = _bodies(af)
    for b, parts in bodies.items():
        data = to_stl(parts, b)
        with open(os.path.join(tri, f"{b}.stl"), "wb") as f:
            f.write(data)
    L, b2 = af.length, af.span / 2
    # domain and background cells scale with the fuselage (light aircraft: 8.2 m -> 1 m cells)
    sL = L / 8.2
    xmin, xmax, ymax, zmin, zmax = -12.0 * sL, 32.0 * sL, max(18.0 * sL, b2 * 2.6), -14.0 * sL, 14.0 * sL
    h = 1.0 * sL
    nx, ny, nz = int((xmax - xmin) / h), int(ymax / h), int((zmax - zmin) / h)
    _w(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices ( ({xmin} 0 {zmin}) ({xmax} 0 {zmin}) ({xmax} {ymax} {zmin}) ({xmin} {ymax} {zmin})
           ({xmin} 0 {zmax}) ({xmax} 0 {zmax}) ({xmax} {ymax} {zmax}) ({xmin} {ymax} {zmax}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    farfield {{ type patch; faces ( (0 4 7 3) (1 2 6 5) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }}
    symmetry {{ type symmetryPlane; faces ( (0 1 5 4) ); }}
);
mergePatchPairs ();
""")
    _w(case, "system/surfaceFeatureExtractDict", "dictionary", "\n".join(
        f"{b}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} "
        f"subsetFeatures {{ nonManifoldEdges no; openEdges yes; }} writeObj no; }}" for b in BODIES) + "\n")
    lw = level_wing
    levels = {"fuselage": level_fus, "wing": lw, "htail": lw, "vtail": (lw[0], lw[0])}
    geom = "\n".join(f"    {b}.stl {{ type triSurfaceMesh; name {b}; }}" for b in BODIES)
    feats = "\n".join(f"        {{ file \"{b}.eMesh\"; level {levels[b][1]}; }}" for b in BODIES)
    surf = "\n".join(f"        {b} {{ level ({levels[b][0]} {levels[b][1]}); patchInfo {{ type wall; inGroups (aircraft); }} }}" for b in BODIES)
    _w(case, "system/snappyHexMeshDict", "dictionary", f"""castellatedMesh true;
snap true;
addLayers false;
geometry
{{
{geom}
    near {{ type searchableBox; min ({-1.5 * sL} 0 {-3.0 * sL}); max ({L + 3.0 * sL} {b2 + 1.5 * sL} {3.5 * sL}); }}
    wake {{ type searchableBox; min ({L - 1 * sL} 0 {-3.0 * sL}); max ({L + 12 * sL} {b2 + 2.5 * sL} {3.0 * sL}); }}
}}
castellatedMeshControls
{{
    maxLocalCells 3000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features
    (
{feats}
    );
    refinementSurfaces
    {{
{surf}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        near {{ mode inside; levels ((1E15 3)); }}
        wake {{ mode inside; levels ((1E15 2)); }}
    }}
    locationInMesh ({-7.53 * sL} {6.27 * sL} {5.31 * sL});                  // not on a background-cell face
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true; layers {{}}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}
writeFlags ();
mergeTolerance 1e-6;
""")
    a = math.radians(alpha_deg)
    U = f"({speed * math.cos(a):.8f} 0 {speed * math.sin(a):.8f})"
    k = 1.5 * (0.001 * speed) ** 2
    omega = k / (nu * 1.0)
    bf = lambda wall, far: ("boundaryField\n{\n    aircraft\n    {\n%s    }\n    farfield\n    {\n%s    }\n"  # noqa: E731
                            "    symmetry\n    {\n        type symmetryPlane;\n    }\n"
                            "    \"proc.*\"\n    {\n        type processor;\n        value $internalField;\n    }\n}\n") % (wall, far)
    io = lambda v: f"        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"  # noqa: E731
    # the walls are listed by group name "aircraft"
    _w(case, "0/U", "volVectorField", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform {U};\n" + bf(
        "        type noSlip;\n", f"        type freestreamVelocity;\n        freestreamValue uniform {U};\n"))
    _w(case, "0/p", "volScalarField", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n" + bf(
        "        type zeroGradient;\n", "        type freestreamPressure;\n        freestreamValue uniform 0;\n"))
    _w(case, "0/k", "volScalarField", f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k:.6g};\n" + bf(
        f"        type kqRWallFunction;\n        value uniform {k:.6g};\n", io(f"{k:.6g}")))
    _w(case, "0/omega", "volScalarField", f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {omega:.6g};\n" + bf(
        f"        type omegaWallFunction;\n        value uniform {omega:.6g};\n", io(f"{omega:.6g}")))
    _w(case, "0/nut", "volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n" + bf(
        "        type nutkWallFunction;\n        value uniform 0;\n", "        type calculated;\n        value uniform 0;\n"))
    _w(case, "constant/transportProperties", "dictionary", f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};\n")
    _w(case, "constant/turbulenceProperties", "dictionary",
       "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n")
    _w(case, "system/controlDict", "dictionary", f"""application simpleFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {iterations};
deltaT 1;
writeControl timeStep;
writeInterval {iterations};
purgeWrite 0;
writeFormat binary;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
""")
    _w(case, "system/fvSchemes", "dictionary", """ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwindV grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited 0.5; }
wallDist { method meshWave; }
""")
    _w(case, "system/fvSolution", "dictionary", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
}
relaxationFactors
{
    equations { U 0.7; k 0.5; omega 0.5; }
    fields { p 0.7; }
}
""")
    n = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}.get(procs, (procs, 1, 1))
    _w(case, "system/decomposeParDict", "dictionary", f"numberOfSubdomains {procs};\nmethod hierarchical;\n"
       f"hierarchicalCoeffs {{ n ({n[0]} {n[1]} {n[2]}); order xyz; }}\n")   # scotch/metis are stubs in this build
    return {"alpha": alpha_deg, "speed": speed, "nu": nu, "span": af.span, "S_half": af.wing_area / 2, "mac": af.mac}


def run_case3d(case: str, procs: int = 4, log=print) -> Dict[str, float]:
    t = {}
    def sh(cmd, name):
        t0 = time.time()
        full = f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} > log.{name} 2>&1'
        r = subprocess.run(["bash", "-c", full])
        t[name] = round(time.time() - t0, 1)
        log(f"  {name}: {t[name]} s (exit {r.returncode})")
        return r.returncode
    sh("blockMesh", "blockMesh")
    sh("surfaceFeatureExtract", "surfaceFeatureExtract")
    sh("decomposePar -force", "decomposePar")
    sh(f"mpirun --allow-run-as-root --oversubscribe -np {procs} snappyHexMesh -overwrite -parallel", "snappyHexMesh")
    sh(f"mpirun --allow-run-as-root --oversubscribe -np {procs} checkMesh -parallel", "checkMesh")
    # fields: decomposePar copies 0/ before snappy; after snappy the processor meshes changed -> restore fields
    sh("for d in processor*; do rm -rf $d/0; cp -r 0 $d/0; done", "fields")
    sh(f"mpirun --allow-run-as-root --oversubscribe -np {procs} simpleFoam -parallel", "simpleFoam")
    sh("reconstructParMesh -constant; reconstructPar -latestTime", "reconstruct")
    return t


# ---------------------------------------------------------------------- results
def _load_fs(case: str, time_dir: str) -> Dict[str, bytes]:
    fs = {}
    pm = os.path.join(case, "constant", "polyMesh")
    for f in ("points", "faces", "owner", "neighbour", "boundary"):
        p = os.path.join(pm, f)
        if os.path.exists(p):
            fs[f"constant/polyMesh/{f}"] = open(p, "rb").read()
    for f in ("p", "U", "nut"):
        p = os.path.join(case, time_dir, f)
        if os.path.exists(p):
            fs[f"{time_dir}/{f}"] = open(p, "rb").read()
    return fs


def latest_time(case: str) -> str:
    ts = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.exists(os.path.join(case, d, "U")):
                ts.append((float(d), d))
        except ValueError:
            pass
    return max(ts)[1]


def read_result3d(case: str, info: Dict[str, Any], af: Airframe, n_lines: int = 34) -> Dict[str, Any]:
    """Forces (pressure + wall-function shear), Cp on every aircraft face, streamlines around the right half."""
    from scipy.spatial import cKDTree
    from pinneapple_data.cae.foam import read_boundary_values, read_field, read_polymesh
    from pinneapple_data.cae.geometry import cell_geometry, face_geometry, face_mesh

    t = latest_time(case)
    fs = _load_fs(case, t)
    mesh = read_polymesh(fs)
    fm = face_mesh(mesh)
    fg = face_geometry(fm)
    cg = cell_geometry(fm, fg)
    nc = mesh.poly.n_cells
    p, _, _ = read_field(fs[f"{t}/p"], nc)
    U, _, _ = read_field(fs[f"{t}/U"], nc)
    V, nu = info["speed"], info["nu"]
    a = math.radians(info["alpha"])
    drag_dir, lift_dir = np.array([math.cos(a), 0, math.sin(a)]), np.array([-math.sin(a), 0, math.cos(a)])
    q = 0.5 * V * V
    S = info["S_half"]
    ni = mesh.poly.n_internal
    out: Dict[str, Any] = {"time": t, "cells": int(nc), "patches": {}}
    Ftot, Fp, Fv = np.zeros(3), np.zeros(3), np.zeros(3)
    faces_xyz, faces_cp, faces_cf = [], [], []
    for b in mesh.poly.patches:
        if b["name"] not in BODIES:
            continue
        s, n = int(b["startFace"]), int(b["nFaces"])
        idx = np.arange(s, s + n)
        sf = fg["areas"][idx]                                       # outward from the fluid = into the body
        A = np.linalg.norm(sf, axis=1)
        nhat = sf / A[:, None]
        own = mesh.poly.owner[idx]
        pw = p[own].ravel()
        up = U[own]
        ut = up - (up * nhat).sum(1, keepdims=True) * nhat
        d = np.abs(((cg["centres"][own] - fg["centres"][idx]) * nhat).sum(1)) + 1e-9
        # wall shear from the log law at the first cell (what the wall function imposes): U+ = ln(E y+)/kappa
        Ut = np.linalg.norm(ut, axis=1) + 1e-12
        utau = np.sqrt(nu * Ut / d)                                 # laminar start
        for _ in range(30):
            yp = np.maximum(d * utau / nu, 1e-6)
            utau = np.where(yp > 11.25, Ut * 0.41 / np.log(9.8 * yp), np.sqrt(nu * Ut / d))
        tau = utau ** 2
        fpress = pw[:, None] * sf
        fvisc = (tau / Ut)[:, None] * ut * A[:, None]
        Fp += fpress.sum(0)
        Fv += fvisc.sum(0)
        out["patches"][b["name"]] = {"faces": int(n), "CL": float((fpress + fvisc).sum(0) @ lift_dir / (q * S)),
                                     "CD": float((fpress + fvisc).sum(0) @ drag_dir / (q * S))}
        faces_xyz.append(fg["centres"][idx])
        faces_cp.append(pw / q)
        faces_cf.append(tau / q)
    Ftot = Fp + Fv
    out.update(CL=float(Ftot @ lift_dir / (q * S)), CD=float(Ftot @ drag_dir / (q * S)),
               CD_pressure=float(Fp @ drag_dir / (q * S)), CD_friction=float(Fv @ drag_dir / (q * S)))
    out["surface"] = {"xyz": np.concatenate(faces_xyz), "cp": np.concatenate(faces_cp), "cf": np.concatenate(faces_cf)}
    # streamlines in the near field: inverse-distance velocity from the 8 nearest cells, RK2
    C = cg["centres"]
    box = (C[:, 0] > -2) & (C[:, 0] < af.length + 14) & (C[:, 1] < af.span / 2 + 3) & (C[:, 2] > -4) & (C[:, 2] < 5)
    tree = cKDTree(C[box])
    Ub = U[box]

    def vel(x):
        dd, ii = tree.query(x, k=8)
        w = 1 / (dd + 1e-6) ** 2
        return (Ub[ii] * w[..., None]).sum(-2) / w.sum(-1, keepdims=True)

    # seeds: a row just above each wing section's leading edge (suction side), a few under it, a ring around the
    # nose and a cluster at the tip (vortex roll-up); traced until well behind the tail
    b2 = af.span / 2
    zw = af.wing_z_root()
    seeds = []
    for eta in np.r_[np.linspace(0.12, 0.92, 9), 0.97, 1.0]:
        le = np.asarray(af._wing_point(eta, 0.0), float)
        seeds += [le + [-3.0, 0.0, 0.22], le + [-3.0, 0.0, -0.35]]
    for eta, dz in ((1.0, -0.15), (1.0, 0.45), (1.02, 0.1)):
        le = np.asarray(af._wing_point(min(eta, 1.0), 0.0), float)
        seeds.append(le + [-3.0, 0.25 if eta > 1 else 0.05, dz])
    r = 0.5 * af.width
    for ang in np.radians([20, 60, 100, 140]):
        seeds.append([-1.5, 0.15 + 1.1 * r * np.sin(ang), zw + 1.0 + 1.1 * r * np.cos(ang)])
    seeds = np.array(seeds)
    seeds[:, 1] = np.clip(seeds[:, 1], 0.15, b2 + 3)
    h = 0.12
    X = seeds.copy()
    path = [X.copy()]
    x_end = af.length + 10
    for _ in range(int((x_end + 6 - seeds[:, 0].min()) / h)):
        v1 = vel(X)
        Xm = X + 0.5 * h * v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
        v2 = vel(Xm)
        Xn = X + h * v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
        Xn[:, 1] = np.maximum(Xn[:, 1], 0.02)                            # symmetry plane
        X = np.where((X[:, 0] < x_end)[:, None], Xn, X)                  # lines stop x_end (repeated points after)
        path.append(X.copy())
    P = np.stack(path, 1)
    lines = [P[k, ::3] for k in range(len(seeds))]
    out["streamlines"] = lines
    out["streamline_speed"] = [np.linalg.norm(vel(L), axis=1) / V for L in lines]

    # slice planes (what ParaView would show): a cross-flow plane behind the wing (speed, streamwise vorticity)
    # and the symmetry plane (pressure coefficient, speed). Points with no mesh cell around them
    # (inside the aircraft) are NaN.
    tree_all = cKDTree(C)
    vols = cg["volumes"]

    def sample(P3):
        dd, ii = tree_all.query(P3, k=6)
        w = 1 / (dd + 1e-6) ** 2
        Uv = (U[ii] * w[..., None]).sum(-2) / w.sum(-1, keepdims=True)
        pv = (p[ii].reshape(ii.shape) * w).sum(-1) / w.sum(-1)
        # distance to the nearest cell centre in units of that cell's size: > 1 means no cell here (inside the body)
        return Uv, pv, dd[:, 0] / np.cbrt(np.abs(vols[ii[:, 0]]))

    tip_te = af._wing_point(1.0, 1.0)
    xs_ = tip_te[0] + 0.6 * af.mac
    ny_, nz_ = 220, 110
    yy = np.linspace(0.0, 1.35 * b2, ny_)
    zz = np.linspace(zw - 0.32 * b2, zw + 0.32 * b2, nz_)
    Yg, Zg = np.meshgrid(yy, zz)
    Uv, pv, dmin = sample(np.c_[np.full(Yg.size, xs_), Yg.ravel(), Zg.ravel()])
    Uv = Uv.reshape(nz_, ny_, 3)
    dy, dz = yy[1] - yy[0], zz[1] - zz[0]
    wx = np.gradient(Uv[..., 2], dy, axis=1) - np.gradient(Uv[..., 1], dz, axis=0)
    spd = np.linalg.norm(Uv, axis=-1) / V
    in_w = (dmin > 1.0).reshape(nz_, ny_)
    spd[in_w] = np.nan
    wx[in_w] = np.nan
    out["slice_wake"] = {"x": float(xs_), "y": [float(yy[0]), float(yy[-1])], "z": [float(zz[0]), float(zz[-1])],
                         "shape": [nz_, ny_], "speed": spd, "vorticity": wx * af.mac / V}
    nx2, nz2 = 300, 120
    xx = np.linspace(-0.15 * af.length, 1.25 * af.length, nx2)
    zz2 = np.linspace(-0.22 * af.length, 0.28 * af.length, nz2)
    Xg, Zg2 = np.meshgrid(xx, zz2)
    Uv2, pv2, d2 = sample(np.c_[Xg.ravel(), np.full(Xg.size, 0.02 * af.width), Zg2.ravel()])
    inside = (d2 > 1.0).reshape(nz2, nx2)
    cp2 = (pv2 / q).reshape(nz2, nx2)
    sp2 = (np.linalg.norm(Uv2, axis=1) / V).reshape(nz2, nx2)
    cp2[inside] = np.nan
    sp2[inside] = np.nan
    out["slice_sym"] = {"y": 0.0, "x": [float(xx[0]), float(xx[-1])], "z": [float(zz2[0]), float(zz2[-1])],
                        "shape": [nz2, nx2], "cp": cp2, "speed": sp2}
    return out


def surface_scalars(af: Airframe, parts, surface: Dict[str, np.ndarray], key: str = "cp") -> Dict[str, np.ndarray]:
    """A surface field from the CFD faces onto the visual mesh vertices (nearest faces, inverse-distance, mirrored
    for the left side). key: "cp" (pressure coefficient) or "cf" (skin friction coefficient)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(surface["xyz"])
    vals = np.asarray(surface[key], np.float32)
    out = {}
    for p in parts:
        if not p.aero:
            continue
        v = p.vertices * [1, 1, 1]
        v[:, 1] = np.abs(v[:, 1])
        d, i = tree.query(v, k=4)
        w = 1 / (d + 1e-4) ** 2
        out[p.name] = ((vals[i] * w).sum(1) / w.sum(1)).astype(np.float32)
    return out
