"""External flow around any body in OpenFOAM, in one call: STL -> snappyHexMesh -> simpleFoam (k-omega SST, wall
functions) -> forces, skin pressure and friction, streamlines and slice planes -> a 3D scene to render or browse.

    from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
    from pinneapple_design.geometry.bodies import ahmed_body
    flow = ExternalFlow({"body": ahmed_body()}, speed=40.0, ground=True)
    res = flow.solve("cases/ahmed", procs=4)          # writes, meshes, runs, reads (minutes)
    print(res.coefficients)                           # CD, CL, CS, pressure / friction split, per body
    sc = res.to_scene()                               # skin Cp and Cf, streamlines by speed, slices
    pp.viz.render(sc, "cp.jpg", field="cp")           # or pp.viz.web_viewer(sc, "viewer/")

Axes: x along the flow (at zero angles), z up. ``alpha`` tilts the flow in the x-z plane, ``beta`` in the x-y plane.
``half_model=True`` meshes y >= 0 only with a symmetry plane at y = 0 (bodies must be symmetric; give their full
geometry, the y < 0 half is ignored by the mesher). ``ground=True`` adds a road at the bodies' lowest z moving with
the flow (cars, trains, buildings with ``ground_moving=False``). Needs OpenFOAM (v1912+ tested) on the machine;
``FOAM_BASHRC`` points to its bashrc.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")
HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"
RESOLUTION = {"coarse": (8, (3, 4)), "medium": (8, (4, 5)), "fine": (10, (5, 6))}   # background cells per length, surface levels

Body = Tuple[np.ndarray, np.ndarray]


def _w(case: str, rel: str, cls: str, body: str) -> None:
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(HDR % (cls, os.path.basename(rel)) + body)


def _stl(V: np.ndarray, F: np.ndarray, name: str) -> bytes:
    import struct
    tris = V[F].astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    return name.encode().ljust(80, b" ")[:80] + struct.pack("<I", len(tris)) + rec.tobytes()


def _bodies_from(src) -> Dict[str, Body]:
    """dict name -> (V, F); an STL path; a studio Scene; a list of parts with name/vertices/faces."""
    if isinstance(src, dict):
        return {k: (np.asarray(v[0], float), np.asarray(v[1], np.int64)) for k, v in src.items()}
    if isinstance(src, str):
        from pinneapple_tools.visualization.studio.scene import read_stl
        return {n.replace(" ", "_") or "body": (V, F) for n, V, F in read_stl(src)}
    surfaces = getattr(src, "surfaces", src)
    return {s.name: (np.asarray(s.vertices, float), np.asarray(s.faces, np.int64)) for s in surfaces}


def flow_directions(alpha_deg: float, beta_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit drag (along the flow), lift (up, normal to it in the x-z plane) and side directions."""
    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    d = np.array([math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b)])
    lift = np.array([-math.sin(a), 0.0, math.cos(a)])
    side = np.cross(lift, d)
    return d, lift, side / np.linalg.norm(side)


def write_flow_fields(case: str, speed: float, alpha_deg: float, beta_deg: float, nu: float, iterations: int,
                      procs: int, walls: str, symmetry: bool, ground: Optional[str] = None,
                      ground_velocity: Optional[Sequence[float]] = None, turbulence_intensity: float = 0.001) -> None:
    """0/ fields, physical properties, controlDict, schemes and solvers of a steady external-flow case. ``walls`` is
    the patch group of the bodies; ``ground``: name of a road patch (moving with ``ground_velocity``)."""
    d, _, _ = flow_directions(alpha_deg, beta_deg)
    Uv = d * speed
    U = f"({Uv[0]:.8f} {Uv[1]:.8f} {Uv[2]:.8f})"
    k = 1.5 * (turbulence_intensity * speed) ** 2
    omega = k / nu
    ext = ""
    if symmetry:
        ext += "    symmetry\n    {\n        type symmetryPlane;\n    }\n"
    ext += "    \"proc.*\"\n    {\n        type processor;\n        value $internalField;\n    }\n"

    def bf(wall, far, gnd=None):
        s = "boundaryField\n{\n    %s\n    {\n%s    }\n    farfield\n    {\n%s    }\n" % (walls, wall, far)
        if ground:
            s += "    %s\n    {\n%s    }\n" % (ground, gnd or wall)
        return s + ext + "}\n"
    io = lambda v: f"        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"  # noqa: E731
    gv = ground_velocity if ground_velocity is not None else (0.0, 0.0, 0.0)
    _w(case, "0/U", "volVectorField", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform {U};\n" + bf(
        "        type noSlip;\n", f"        type freestreamVelocity;\n        freestreamValue uniform {U};\n",
        f"        type fixedValue;\n        value uniform ({gv[0]} {gv[1]} {gv[2]});\n"))
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
       f"hierarchicalCoeffs {{ n ({n[0]} {n[1]} {n[2]}); order xyz; }}\n")   # scotch/metis can be stubs in builds


def snappy_dict(geometry: str, features: str, surfaces: str, regions: str, location: Sequence[float]) -> str:
    """snappyHexMeshDict body (castellate + snap, no layers: wall functions)."""
    return f"""castellatedMesh true;
snap true;
addLayers false;
geometry
{{
{geometry}
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
{features}
    );
    refinementSurfaces
    {{
{surfaces}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
{regions}
    }}
    locationInMesh ({float(location[0])!r} {float(location[1])!r} {float(location[2])!r});
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
"""


def run_case(case: str, procs: int = 4, log: Callable[[str], None] = print) -> Dict[str, float]:
    """blockMesh, surfaceFeatureExtract, snappyHexMesh, checkMesh and simpleFoam in parallel, then reconstruct.
    Returns the seconds per step; each step's log is in ``case/log.<step>``."""
    t: Dict[str, float] = {}

    def sh(cmd, name):
        t0 = time.time()
        full = f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} > log.{name} 2>&1'
        r = subprocess.run(["bash", "-c", full])
        t[name] = round(time.time() - t0, 1)
        log(f"  {name}: {t[name]} s (exit {r.returncode})")
        return r.returncode
    mpi = f"mpirun --allow-run-as-root --oversubscribe -np {procs}"
    sh("blockMesh", "blockMesh")
    sh("surfaceFeatureExtract", "surfaceFeatureExtract")
    if procs > 1:
        sh("decomposePar -force", "decomposePar")
        sh(f"{mpi} snappyHexMesh -overwrite -parallel", "snappyHexMesh")
        sh(f"{mpi} checkMesh -parallel", "checkMesh")
        # decomposePar copied 0/ before snappy changed the processor meshes: restore the fields
        sh("for d in processor*; do rm -rf $d/0; cp -r 0 $d/0; done", "fields")
        sh(f"{mpi} simpleFoam -parallel", "simpleFoam")
        sh("reconstructParMesh -constant; reconstructPar -latestTime", "reconstruct")
    else:
        sh("snappyHexMesh -overwrite", "snappyHexMesh")
        sh("checkMesh", "checkMesh")
        sh("simpleFoam", "simpleFoam")
    return t


# ---------------------------------------------------------------------- reading results
def latest_time(case: str) -> str:
    ts = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.exists(os.path.join(case, d, "U")):
                ts.append((float(d), d))
        except ValueError:
            pass
    if not ts:
        raise FileNotFoundError(f"no solution time with U in {case} (did the run finish? see {case}/log.simpleFoam)")
    return max(ts)[1]


class CaseFields:
    """The reconstructed mesh and the latest U, p of a case, with the helpers the results need: wall forces and
    skin coefficients on given patches, a velocity interpolator, streamline tracing, plane sampling."""

    def __init__(self, case: str):
        from pinneapple_data.cae.foam import read_field, read_polymesh
        from pinneapple_data.cae.geometry import cell_geometry, face_geometry, face_mesh
        self.time = t = latest_time(case)
        fs = {}
        for f in ("points", "faces", "owner", "neighbour", "boundary"):
            p = os.path.join(case, "constant", "polyMesh", f)
            if os.path.exists(p):
                fs[f"constant/polyMesh/{f}"] = open(p, "rb").read()
        for f in ("p", "U"):
            fs[f"{t}/{f}"] = open(os.path.join(case, t, f), "rb").read()
        self.mesh = read_polymesh(fs)
        fm = face_mesh(self.mesh)
        self.fg = face_geometry(fm)
        self.cg = cell_geometry(fm, self.fg)
        nc = self.mesh.poly.n_cells
        self.p = read_field(fs[f"{t}/p"], nc)[0].ravel()
        self.U = read_field(fs[f"{t}/U"], nc)[0]
        self.n_cells = int(nc)
        self._trees: Dict[Any, Any] = {}

    def patches(self) -> List[Dict[str, Any]]:
        return list(self.mesh.poly.patches)

    def wall(self, patch: Dict[str, Any], nu: float, q: float) -> Dict[str, np.ndarray]:
        """Per face of a wall patch: centre, area vector (into the body), Cp, Cf from the log law at the first cell
        (what the wall function imposes), pressure and viscous force vectors (kinematic, per unit density)."""
        s, n = int(patch["startFace"]), int(patch["nFaces"])
        idx = np.arange(s, s + n)
        sf = self.fg["areas"][idx]
        A = np.linalg.norm(sf, axis=1)
        nhat = sf / A[:, None]
        own = self.mesh.poly.owner[idx]
        pw = self.p[own]
        up = self.U[own]
        ut = up - (up * nhat).sum(1, keepdims=True) * nhat
        d = np.abs(((self.cg["centres"][own] - self.fg["centres"][idx]) * nhat).sum(1)) + 1e-9
        Ut = np.linalg.norm(ut, axis=1) + 1e-12
        utau = np.sqrt(nu * Ut / d)                                     # laminar start: U+ = y+
        for _ in range(30):
            yp = np.maximum(d * utau / nu, 1e-6)
            utau = np.where(yp > 11.25, Ut * 0.41 / np.log(9.8 * yp), np.sqrt(nu * Ut / d))
        tau = utau ** 2
        return {"xyz": self.fg["centres"][idx], "cp": pw / q, "cf": tau / q, "f_pressure": pw[:, None] * sf,
                "f_viscous": (tau / Ut)[:, None] * ut * A[:, None]}

    def _tree(self, key, mask=None):
        from scipy.spatial import cKDTree
        if key not in self._trees:
            C = self.cg["centres"] if mask is None else self.cg["centres"][mask]
            self._trees[key] = (cKDTree(C), mask)
        return self._trees[key]

    def velocity(self, box: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Callable[[np.ndarray], np.ndarray]:
        """Inverse-distance velocity from the 8 nearest cell centres (optionally only cells inside ``box``)."""
        C = self.cg["centres"]
        mask = None if box is None else np.all((C > box[0]) & (C < box[1]), axis=1)
        tree, m = self._tree(("vel", None if box is None else tuple(np.r_[box[0], box[1]])), mask)
        Ub = self.U if m is None else self.U[m]

        def vel(x):
            dd, ii = tree.query(x, k=8)
            w = 1 / (dd + 1e-6) ** 2
            return (Ub[ii] * w[..., None]).sum(-2) / w.sum(-1, keepdims=True)
        return vel

    def trace(self, seeds: np.ndarray, vel: Callable, h: float, x_end: float, y_min: Optional[float] = None,
              every: int = 3, overshoot: Optional[float] = None) -> np.ndarray:
        """RK2 streamlines with arc-length step ``h`` until x passes ``x_end`` (later points repeat the last one).
        ``y_min``: a symmetry plane the lines may not cross. Returns (n_seeds, n_points, 3)."""
        X = np.asarray(seeds, float).copy()
        path = [X.copy()]
        for _ in range(int((x_end + (6 * h if overshoot is None else overshoot) - X[:, 0].min()) / h)):
            v1 = vel(X)
            Xm = X + 0.5 * h * v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
            v2 = vel(Xm)
            Xn = X + h * v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
            if y_min is not None:
                Xn[:, 1] = np.maximum(Xn[:, 1], y_min)
            X = np.where((X[:, 0] < x_end)[:, None], Xn, X)
            path.append(X.copy())
        return np.stack(path, 1)[:, ::every]

    def sample(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Velocity and kinematic pressure at points (inverse distance, 6 nearest cells), and the distance to the
        nearest cell centre in units of that cell's size (> 1: no cell there, i.e. inside a body)."""
        tree, _ = self._tree("all")
        dd, ii = tree.query(P, k=6)
        w = 1 / (dd + 1e-6) ** 2
        Uv = (self.U[ii] * w[..., None]).sum(-2) / w.sum(-1, keepdims=True)
        pv = (self.p[ii] * w).sum(-1) / w.sum(-1)
        # scale by the largest of the neighbouring cells: at a refinement interface the nearest centre can be a
        # small cell while the point sits in a large one
        h = np.cbrt(np.abs(self.cg["volumes"][ii])).max(1)
        return Uv, pv, dd[:, 0] / h


# ---------------------------------------------------------------------- the study
@dataclass
class FlowResult:
    coefficients: Dict[str, Any]
    surface: Dict[str, np.ndarray]                         # xyz, cp, cf, body (index into bodies) per wall face
    bodies: List[str]
    lines: List[np.ndarray]
    line_speed: List[np.ndarray]
    slices: List[Dict[str, Any]]                           # name, origin, u, v, grids {field: 2D array}, labels
    info: Dict[str, Any]
    geometry: Dict[str, Body] = field(default_factory=dict)

    def to_scene(self, scene=None, material: str = "paint"):
        """A studio Scene: the bodies (or ``scene``) with skin "cp" and "cf", the streamlines coloured by |U|/U∞
        and one slice per sampled field. Half models are mirrored."""
        from pinneapple_tools.visualization.studio.scene import Scene, Surface
        half = self.info.get("half_model", False)
        if scene is None:
            scene = Scene([Surface(n, V, F, material) for n, (V, F) in self.geometry.items()],
                          title=self.info.get("title", "External flow"))
        scene.map_field("cp", self.surface["xyz"], self.surface["cp"], mirror_y=half, label="Pressure coefficient Cp")
        scene.map_field("cf", self.surface["xyz"], self.surface["cf"], mirror_y=half, label="Skin friction coefficient Cf")
        lines, speeds = list(self.lines), list(self.line_speed)
        if half:
            lines += [L * [1, -1, 1] for L in self.lines]
            speeds += speeds
        scene.add_lines(lines, speeds, label="Speed |U|/U∞")
        for s in self.slices:
            for k, g in s["grids"].items():
                O, u, v, G = np.asarray(s["origin"]), np.asarray(s["u"]), np.asarray(s["v"]), np.asarray(g, float)
                if half and s.get("mirror"):
                    sign = -1.0 if k in s.get("odd", ()) else 1.0
                    G = np.concatenate([sign * G[:, ::-1], G], axis=1)
                    O = O - u
                    u = 2 * u
                scene.add_slice(f"{s['name']}: {k}", O, u, v, G, label=s["labels"].get(k, k),
                                group=f"{s.get('group', s['name'])}: {k}")
        return scene

    def save(self, path: str) -> str:
        """npz with every array and the coefficients/info as JSON."""
        np.savez_compressed(path, xyz=self.surface["xyz"], cp=self.surface["cp"], cf=self.surface["cf"],
                            body=self.surface["body"], lines=np.array(self.lines, dtype=object),
                            line_speed=np.array(self.line_speed, dtype=object),
                            meta=json.dumps({"coefficients": self.coefficients, "bodies": self.bodies, "info": self.info,
                                             "slices": [{k: (np.asarray(v).tolist() if k not in ("grids", "labels") else
                                                             ({kk: np.asarray(vv).tolist() for kk, vv in v.items()} if k == "grids" else v))
                                                         for k, v in s.items()} for s in self.slices]}))
        return path


@dataclass
class ExternalFlow:
    """Steady incompressible flow around ``bodies`` (dict name -> (vertices, faces), an STL path, or a studio Scene).

    speed (m/s), alpha / beta (deg), nu (m²/s); ref_area (m², default: frontal area of the bounding box) and
    ref_length (m, default: body length) scale the coefficients; resolution "coarse" | "medium" | "fine" or
    ``surface_level=(min, max)``; half_model, ground, ground_moving: see the module docstring."""
    bodies: Any
    speed: float = 20.0
    alpha: float = 0.0
    beta: float = 0.0
    nu: float = 1.5e-5
    rho: float = 1.225
    ref_area: Optional[float] = None
    ref_length: Optional[float] = None
    half_model: bool = False
    ground: Union[bool, float] = False             # True: a road at the lowest point of the bodies; a number: its z
    ground_moving: bool = True
    resolution: str = "medium"
    surface_level: Optional[Tuple[int, int]] = None
    iterations: int = 600
    n_lines: int = 40
    wake_planes: Tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)   # cross-flow slices, body lengths behind the bodies
    title: str = "External flow"

    def __post_init__(self):
        if not self.bodies:
            raise ValueError("ExternalFlow: bodies is required (dict name -> (vertices, faces), an STL path or a Scene)")
        self.geometry = _bodies_from(self.bodies)
        V = np.concatenate([v for v, _ in self.geometry.values()])
        self.lo, self.hi = V.min(0), V.max(0)
        self.L = float(self.ref_length or (self.hi[0] - self.lo[0]))
        if self.ref_area is None:
            self.ref_area = float((self.hi[1] - self.lo[1]) * (self.hi[2] - self.lo[2]))
        self.ground_z = None if self.ground is False else (float(self.lo[2]) if self.ground is True else float(self.ground))

    # -------------------------------------------------------------- case
    def write(self, case: str, procs: int = 4) -> Dict[str, Any]:
        if os.path.isdir(case):
            shutil.rmtree(case)
        tri = os.path.join(case, "constant", "triSurface")
        os.makedirs(tri)
        names = []
        for n, (V, F) in self.geometry.items():
            nm = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in n)
            names.append(nm)
            with open(os.path.join(tri, f"{nm}.stl"), "wb") as f:
                f.write(_stl(V, F, nm))
        lo, hi, L = self.lo, self.hi, max(self.hi[0] - self.lo[0], 1e-9)
        Wd = max(hi[1] - lo[1], hi[2] - lo[2], L)
        per_len, lev = RESOLUTION[self.resolution]
        lev = self.surface_level or lev
        h = L / per_len
        ctr = (lo + hi) / 2
        x0, x1 = lo[0] - 1.5 * L, hi[0] + 4.0 * L
        yspan = max(2.2 * L, 1.6 * Wd)
        y0, y1 = (0.0, max(hi[1], 0) + yspan) if self.half_model else (ctr[1] - yspan, ctr[1] + yspan)
        z0 = self.ground_z if self.ground_z is not None else ctr[2] - max(1.7 * L, 1.3 * Wd)
        z1 = (hi[2] + 1.7 * L) if self.ground_z is not None else ctr[2] + max(1.7 * L, 1.3 * Wd)
        nx, ny, nz = (max(4, int(round((b - a) / h))) for a, b in ((x0, x1), (y0, y1), (z0, z1)))
        far = ["(1 2 6 5)", "(0 4 7 3)", "(3 7 6 2)", "(4 5 6 7)"]           # outlet, inlet, y max, z max
        sym = "(0 1 5 4)"
        bottom = "(0 3 2 1)"
        bnd = []
        if self.half_model:
            bnd.append(f"    symmetry {{ type symmetryPlane; faces ( {sym} ); }}")
        else:
            far.append(sym)
        if self.ground_z is not None:
            bnd.append(f"    ground {{ type wall; faces ( {bottom} ); }}")
        else:
            far.append(bottom)
        _w(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices ( ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
           ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    farfield {{ type patch; faces ( {" ".join(far)} ); }}
{chr(10).join(bnd)}
);
mergePatchPairs ();
""")
        _w(case, "system/surfaceFeatureExtractDict", "dictionary", "\n".join(
            f"{b}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} "
            f"subsetFeatures {{ nonManifoldEdges no; openEdges yes; }} writeObj no; }}" for b in names) + "\n")
        pad = 0.25 * L
        ny0 = 0.0 if self.half_model else lo[1] - pad
        nz0 = max(lo[2] - pad, z0)
        geom = "\n".join(f"    {b}.stl {{ type triSurfaceMesh; name {b}; }}" for b in names)
        geom += (f"\n    near {{ type searchableBox; min ({lo[0] - pad} {ny0} {nz0}); max ({hi[0] + 2 * pad} {hi[1] + pad} {hi[2] + pad}); }}"
                 f"\n    wake {{ type searchableBox; min ({hi[0] - pad} {ny0} {nz0}); max ({hi[0] + 1.5 * L} {hi[1] + 1.5 * pad} {hi[2] + 1.5 * pad}); }}")
        feats = "\n".join(f"        {{ file \"{b}.eMesh\"; level {lev[1]}; }}" for b in names)
        surfs = "\n".join(f"        {b} {{ level ({lev[0]} {lev[1]}); patchInfo {{ type wall; inGroups (bodies); }} }}" for b in names)
        regions = "        near { mode inside; levels ((1E15 3)); }\n        wake { mode inside; levels ((1E15 2)); }"
        loc = (x0 + 1.37 * h, y1 - 1.61 * h, z1 - 1.53 * h)                    # in the fluid, off any cell face
        _w(case, "system/snappyHexMeshDict", "dictionary", snappy_dict(geom, feats, surfs, regions, loc))
        gv = flow_directions(self.alpha, self.beta)[0] * self.speed if self.ground_moving else (0.0, 0.0, 0.0)
        write_flow_fields(case, self.speed, self.alpha, self.beta, self.nu, self.iterations, procs, "bodies",
                          self.half_model, "ground" if self.ground_z is not None else None, tuple(float(v) for v in gv))
        info = {"title": self.title, "speed": self.speed, "alpha": self.alpha, "beta": self.beta, "nu": self.nu,
                "rho": self.rho, "ref_area": self.ref_area, "ref_length": self.L, "half_model": self.half_model,
                "ground": self.ground_z, "bodies": names, "bounds": [lo.tolist(), hi.tolist()],
                "Re": self.speed * self.L / self.nu, "background_cell": h, "surface_level": list(lev)}
        with open(os.path.join(case, "pinneapple_flow.json"), "w") as f:
            json.dump(info, f, indent=1)
        return info

    def run(self, case: str, procs: int = 4, log: Callable[[str], None] = print) -> Dict[str, float]:
        return run_case(case, procs, log)

    def solve(self, case: str, procs: int = 4, log: Callable[[str], None] = print) -> FlowResult:
        """write + run + read."""
        self.write(case, procs)
        self.run(case, procs, log)
        return self.read(case)

    # -------------------------------------------------------------- results
    def read(self, case: str) -> FlowResult:
        info = json.load(open(os.path.join(case, "pinneapple_flow.json")))
        cf = CaseFields(case)
        V, nu = info["speed"], info["nu"]
        q = 0.5 * V * V
        S = info["ref_area"] / (2 if info["half_model"] else 1)
        dd, ll, ss = flow_directions(info["alpha"], info["beta"])
        out: Dict[str, Any] = {"bodies": {}}
        Fp, Fv = np.zeros(3), np.zeros(3)
        xyz, cpl, cfl, bid, frc = [], [], [], [], []
        names = info["bodies"]
        half_m = info["half_model"]
        for patch in cf.patches():
            if patch["name"] not in names:
                continue
            w = cf.wall(patch, nu, q)
            fp, fv = w["f_pressure"].sum(0), w["f_viscous"].sum(0)
            Fp += fp
            Fv += fv
            F = fp + fv
            out["bodies"][patch["name"]] = {"CD": float(F @ dd / (q * S)), "CL": float(F @ ll / (q * S)),
                                            "CS": 0.0 if half_m else float(F @ ss / (q * S)), "faces": len(w["cp"])}
            xyz.append(w["xyz"])
            cpl.append(w["cp"])
            cfl.append(w["cf"])
            bid.append(np.full(len(w["cp"]), names.index(patch["name"])))
            frc.append(w["f_pressure"] + w["f_viscous"])
        F = Fp + Fv
        out.update(CD=float(F @ dd / (q * S)), CL=float(F @ ll / (q * S)),
                   CS=0.0 if half_m else float(F @ ss / (q * S)),              # a half model is symmetric
                   CD_pressure=float(Fp @ dd / (q * S)), CD_friction=float(Fv @ dd / (q * S)),
                   drag_N=float(F @ dd * info["rho"] * (2 if info["half_model"] else 1)),
                   lift_N=float(F @ ll * info["rho"] * (2 if info["half_model"] else 1)), cells=cf.n_cells)
        lo, hi = np.array(info["bounds"][0]), np.array(info["bounds"][1])
        L = hi[0] - lo[0]
        half = info["half_model"]
        # streamlines: a rake upstream covering the frontal box
        ny = max(2, int(round(math.sqrt(self.n_lines * max(hi[1] - (0 if half else lo[1]), 1e-9) / max(hi[2] - lo[2], 1e-9)))))
        nz = max(2, self.n_lines // ny)
        ylo = 0.04 * (hi[1] - lo[1]) if half else lo[1]
        ys = np.linspace(ylo, hi[1], ny + 2)[1:-1] if not half else np.linspace(ylo, hi[1] * 1.05, ny)
        zs = np.linspace(lo[2] + 0.02 * (hi[2] - lo[2]), hi[2] + 0.1 * (hi[2] - lo[2]), nz)
        seeds = np.array([[lo[0] - 0.3 * L, y, z] for y in ys for z in zs])
        box = (np.array([lo[0] - 0.6 * L, (0 if half else lo[1] - L), lo[2] - L]), np.array([hi[0] + 2.0 * L, hi[1] + L, hi[2] + L]))
        vel = cf.velocity(box)
        P = cf.trace(seeds, vel, h=L / 120, x_end=hi[0] + 1.2 * L, y_min=0.0 if half else None)
        lines = [P[k] for k in range(len(P))]
        speeds = [np.linalg.norm(vel(Lk), axis=1) / V for Lk in lines]
        # slices: the mid plane y = centre (Cp, speed), a cross-flow plane behind the bodies (speed, vorticity)
        slices = []
        nx2, nz2 = 300, 120
        ymid = 0.02 * (hi[1] - lo[1]) if half else 0.5 * (lo[1] + hi[1])
        xx = np.linspace(lo[0] - 0.4 * L, hi[0] + 1.0 * L, nx2)
        z_lo = info["ground"] if info["ground"] is not None else lo[2] - 0.6 * L
        zz = np.linspace(z_lo, hi[2] + 0.6 * L, nz2)
        Xg, Zg = np.meshgrid(xx, zz)
        Uv, pv, dn = cf.sample(np.c_[Xg.ravel(), np.full(Xg.size, ymid), Zg.ravel()])
        inside = (dn > 1.0).reshape(nz2, nx2)
        g_cp, g_sp = (pv / q).reshape(nz2, nx2), (np.linalg.norm(Uv, axis=1) / V).reshape(nz2, nx2)
        g_cp[inside] = np.nan
        g_sp[inside] = np.nan
        slices.append({"name": "mid plane", "origin": [xx[0], ymid, zz[0]], "u": [xx[-1] - xx[0], 0, 0],
                       "v": [0, 0, zz[-1] - zz[0]], "grids": {"Cp": g_cp, "speed": g_sp}, "mirror": False,
                       "labels": {"Cp": "Pressure coefficient Cp, mid plane", "speed": "Speed |U|/U∞, mid plane"}})
        # cross-flow planes behind the bodies; the first (0.25 L) keeps the name "wake", the others form its stack
        nyw, nzw = 220, 110
        half_w = 0.5 * (hi[1] - lo[1]) + 0.35 * L
        yw = np.linspace(0.0, (hi[1] if half else 0) + half_w, nyw) if half else np.linspace(0.5 * (lo[1] + hi[1]) - half_w, 0.5 * (lo[1] + hi[1]) + half_w, nyw)
        zw = np.linspace(z_lo if info["ground"] is not None else lo[2] - 0.35 * L, hi[2] + 0.35 * L, nzw)
        Yg, Zg = np.meshgrid(yw, zw)
        for k, f in enumerate(self.wake_planes):
            xw = hi[0] + f * L
            Uw, _, dw = cf.sample(np.c_[np.full(Yg.size, xw), Yg.ravel(), Zg.ravel()])
            Uw = Uw.reshape(nzw, nyw, 3)
            wx = (np.gradient(Uw[..., 2], yw[1] - yw[0], axis=1) - np.gradient(Uw[..., 1], zw[1] - zw[0], axis=0)) * L / V
            spw = np.linalg.norm(Uw, axis=-1) / V
            inw = (dw > 1.0).reshape(nzw, nyw)
            wx[inw] = np.nan
            spw[inw] = np.nan
            slices.append({"name": "wake" if k == 0 else f"wake {f:g} L", "group": "wake",
                           "origin": [xw, yw[0], zw[0]], "u": [0, yw[-1] - yw[0], 0], "v": [0, 0, zw[-1] - zw[0]],
                           "grids": {"speed": spw, "vorticity": wx}, "mirror": half, "odd": ("vorticity",),
                           "labels": {"speed": f"Speed |U|/U∞, {f:g} L behind the body",
                                      "vorticity": f"Streamwise vorticity ωx·L/U∞, {f:g} L behind the body"}})
        info = {**info, "time": cf.time}
        return FlowResult(out, {"xyz": np.concatenate(xyz), "cp": np.concatenate(cpl), "cf": np.concatenate(cfl),
                                "body": np.concatenate(bid), "force": np.concatenate(frc)},   # force per face / rho
                          names, lines, speeds, slices, info, self.geometry)


def openfoam_available() -> bool:
    """True when the OpenFOAM tools this module calls can be found."""
    r = subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; command -v simpleFoam snappyHexMesh'],
                       capture_output=True, text=True)
    return r.returncode == 0 and "snappyHexMesh" in r.stdout


__all__ = ["ExternalFlow", "FlowResult", "CaseFields", "run_case", "write_flow_fields", "snappy_dict",
           "flow_directions", "openfoam_available", "latest_time"]
