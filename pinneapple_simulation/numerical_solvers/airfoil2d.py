"""Two-dimensional external flow around a section in OpenFOAM: NACA geometry, an O-grid, the case, the run.

What it gives the reference-simulation side of the study, and why each piece is the way it is:

``naca4_surfaces`` / ``closed_loop``
    the section as a closed loop of points. The trailing edge is a **single point**: two points a hair apart
    there make the far-field ring collapse onto itself and the cells between them degenerate.
``arclength_resample``
    spacing of a few thousandths of a chord at the leading and trailing edges and tens in the middle. Points are
    placed where the cumulative cell count hits an integer, so the surface spacing is prescribed rather than a
    by-product of the parameterisation.
``o_grid_2d`` / ``write_poly_mesh``
    a structured ring, ``(i, j)`` = (around the section, outward), with the first cell leaving the wall along its
    normal and the cell heights growing geometrically. A triangulated mesh of the same section reaches 80 deg of
    non-orthogonality at the trailing edge and simpleFoam then oscillates instead of settling; this one sits near
    10 deg on average and the force coefficients stop moving after a couple of hundred iterations.
``write_case_2d`` / ``run_case``
    steady RANS k-omega SST with wall functions, ``empty`` front and back (which is what makes the case 2D), and
    the force coefficients, the wall shear and the probes the study reads.
``read_coefficients`` / ``read_probes``
    the numbers, read back from the files the solver wrote -- never recomputed here.

One cell in z, ``empty`` front and back, is how a two-dimensional case is done in OpenFOAM: a mesh that is genuinely
two-dimensional is not a thing there.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Sequence

import numpy as np

__all__ = [
    "FOAM_BASHRC", "HDR", "DEFAULT_CONDITIONS", "naca4_surfaces", "closed_loop", "arclength_resample",
    "write_mesh", "set_patch_types", "write_case_2d", "run_case", "foam_available", "check_mesh",
    "solver_summary", "read_history", "read_coefficients", "read_probes",
]

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")
HDR = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n")

DEFAULT_CONDITIONS = {"nu_m2_s": 1.5e-5, "rho_kg_m3": 1.225, "chord_m": 1.0, "span_m": 0.01,
                      "turbulence": "steady RANS k-omega SST, wall functions",
                      "solver": "OpenFOAM simpleFoam"}


# --------------------------------------------------------------------------- geometry
def naca4_surfaces(m: float, p: float, t: float, n: int = 200) -> tuple[np.ndarray, ...]:
    """Upper and lower surfaces of a NACA four-digit section, leading edge to trailing edge.

    ``m`` maximum camber as a fraction of the chord, ``p`` its chordwise position, ``t`` the maximum thickness as a
    fraction of the chord. Points are cosine spaced in the chordwise parameter, so the leading edge is where the
    geometry has the detail. Returns ``(xu, yu, xl, yl)``, all with ``x`` from 0 to 1; the leading edge is shared.
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)          # closed trailing edge
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0.0:
        front = x < p
        yc[front] = m / p ** 2 * (2 * p * x[front] - x[front] ** 2)
        dyc[front] = 2 * m / p ** 2 * (p - x[front])
        back = ~front
        yc[back] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[back] - x[back] ** 2)
        dyc[back] = 2 * m / (1 - p) ** 2 * (p - x[back])
    th = np.arctan(dyc)
    xu = x - yt * np.sin(th)
    yu = yc + yt * np.cos(th)
    xl = x + yt * np.sin(th)
    yl = yc - yt * np.cos(th)
    xu[0] = xl[0] = 0.0
    yu[0] = yl[0] = 0.0
    return xu, yu, xl, yl


def closed_loop(m: float, p: float, t: float, n: int = 200, te_radius: float = 0.0) -> np.ndarray:
    """The section as one closed loop: upper trailing edge -> leading edge -> lower trailing edge.

    ``te_radius`` closes the trailing edge with a semicircular cap of that radius (a fraction of the chord)
    instead of a cusp. A cusp is where a structured ring around the section folds: the two sides come within a
    few ten-thousandths of a chord of each other, the fan between their far-field points closes on itself, and the
    outermost cells there cross into bow ties. A rounded edge of a few tenths of a percent of the chord is the
    usual answer, and it is what the mesher needs; with ``te_radius = 0`` the loop ends in a single point.
    """
    xu, yu, xl, yl = naca4_surfaces(m, p, t, n)
    if te_radius > 0:
        half = 0.5 * (yu - yl)
        above = np.where(half >= te_radius)[0]
        if above.size and above[-1] < n - 2:
            iu = int(above[-1])                                # last station still thicker than the cap
            x_c = float(xu[iu])
            ang = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 9)
            cap = np.column_stack([x_c + te_radius * np.cos(ang), te_radius * np.sin(ang)])
            upper = np.column_stack([xu[:iu + 1][::-1], yu[:iu + 1][::-1]])
            lower = np.column_stack([xl[1:iu + 1], yl[1:iu + 1]])
            loop = np.vstack([upper, lower, cap[:-1]])
            loop[:, 0] /= loop[:, 0].max()                 # the cap shortens the chord: scale it back to 1
            return loop
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower])[:-1]


def arclength_resample(loop: np.ndarray, s_min: float, s_max: float, n_dense: int = 4000) -> np.ndarray:
    """Resample a closed loop so the spacing is ``s_min`` at both ends and ``s_max`` half way round.

    The ends of the loop are the trailing edge and its middle is the leading edge, so this is where an airfoil needs
    points: taking a cosine-spaced surface as fixed geometry would leave points 1e-4 apart at the nose and the cells
    next to them would be slivers.
    """
    p = np.vstack([loop, loop[:1]])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n_dense)
    x = np.interp(s, d, p[:, 0])
    y = np.interp(s, d, p[:, 1])
    h = s_min + (s_max - s_min) * np.sin(2 * np.pi * s / s[-1]) ** 2
    n = np.r_[0.0, np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(s))]
    targets = np.arange(0.0, n[-1])
    return np.column_stack([np.interp(targets, n, x), np.interp(targets, n, y)])[:-1]


# --------------------------------------------------------------------------- mesh
def write_mesh(case: str, loop: np.ndarray, *, h_wall: float = 0.0025, h_far: float = 1.2,
               growth: float = 1.12, radius: float = 12.0, center: Sequence[float] = (0.55, 0.0),
               dz: float = 0.01, bl_layers: int = 8, bl_ratio: float = 1.2,
               bl_thickness: float = 0.008) -> dict:
    """Mesh the exterior of ``loop`` as one cell layer in z and write it as ``constant/polyMesh``.

    gmsh meshes the *plane*, with a size field that grows geometrically away from the section and an anisotropic
    boundary layer against it, and the plane is then extruded one cell in z. A triangulated domain has none of the
    failure modes of a structured ring closed around a lifting section -- a ring around a cambered or thin section
    folds where the two sides of the trailing edge come within a few ten-thousandths of a chord, and one bow tie is
    enough for ``checkMesh`` to fail the run.

    The size field is ``h_wall (1 + growth) ** (distance / h_wall)``, capped at ``h_far``: a field that grows
    *linearly* over too short a distance doubles the cell size within one cell and the mesh stops converging.
    ``front`` and ``back`` are left ``empty`` -- that is what makes the case two-dimensional.
    """
    import gmsh

    os.makedirs(os.path.join(case, "system"), exist_ok=True)
    if not os.path.exists(os.path.join(case, "system", "controlDict")):
        # gmshToFoam reads controlDict; the study writes the real one, this only makes the mesher stand alone
        _w(case, "system/controlDict", "dictionary",
           "application simpleFoam;\nstartTime 0;\nendTime 1;\ndeltaT 1;\n"
           "writeControl timeStep;\nwriteInterval 1;\n")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("section")
        geo = gmsh.model.geo
        n = len(loop)
        pts = [geo.addPoint(float(x), float(y), 0.0, h_wall) for x, y in loop]
        lines = [geo.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        n_far = 120
        th = np.linspace(0.0, 2 * np.pi, n_far, endpoint=False)
        fp = [geo.addPoint(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), 0.0, h_far)
              for a in th]
        fl = [geo.addLine(fp[i], fp[(i + 1) % n_far]) for i in range(n_far)]
        geo.synchronize()
        surf = geo.addPlaneSurface([geo.addCurveLoop(fl), geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.mesh.embed(1, lines, 2, surf)

        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList", lines)
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 400)
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", h_wall)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", (h_far - h_wall) / (growth - 1.0))
        fields = [thr]
        if bl_layers:
            bl = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(bl, "CurvesList", lines)
            gmsh.model.mesh.field.setNumber(bl, "Size", h_wall)
            gmsh.model.mesh.field.setNumber(bl, "Ratio", bl_ratio)
            gmsh.model.mesh.field.setNumber(bl, "Quads", 0)
            gmsh.model.mesh.field.setNumber(bl, "Thickness", bl_thickness)
            gmsh.model.mesh.field.setNumber(bl, "NbLayers", bl_layers)
            fields.append(bl)
        least = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(least, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(least)
        # quads everywhere and hexes after the extrusion: an average non-orthogonality near 7 deg against 43 for
        # the same domain triangulated, and the solver's iteration cost follows it
        for key, value in (("Mesh.Algorithm", 8), ("Mesh.RecombineAll", 1), ("Mesh.RecombinationAlgorithm", 1),
                           ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                           ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Optimize", 1), ("Mesh.SmoothRatio", 1.5)):
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)

        extruded = gmsh.model.geo.extrude([(2, surf)], 0, 0, dz, numElements=[1], recombine=True)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.addPhysicalGroup(3, [tag for dim, tag in extruded if dim == 3])
        # name the 3D boundary surfaces: the plane copies are front/back, the rest is wall or far field
        groups: dict[str, list[int]] = {"airfoil": [], "farfield": [], "front": [], "back": []}
        box = (-0.05, -0.3, 1.05, 0.3)
        for _, tag in gmsh.model.getEntities(2):
            bb = gmsh.model.getBoundingBox(2, tag)
            if abs(bb[5] - bb[2]) < 1e-12:
                groups["back" if abs(bb[2]) > 1e-12 else "front"].append(tag)
            elif box[0] <= bb[0] and bb[3] <= box[2] and box[1] <= bb[1] and bb[4] <= box[3]:
                groups["airfoil"].append(tag)
            else:
                groups["farfield"].append(tag)
        for name, tags in groups.items():
            if not tags:
                raise RuntimeError(f"meshing produced no {name} surface")
            gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), name)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(os.path.join(case, "mesh.msh"))
    finally:
        gmsh.finalize()

    r = _sh("gmshToFoam mesh.msh > log.gmshToFoam 2>&1", case)
    if r.returncode != 0:
        raise RuntimeError(f"gmshToFoam failed: {(r.stdout + r.stderr)[-800:]}")
    set_patch_types(case, {"front": "empty", "back": "empty", "airfoil": "wall", "farfield": "patch"})
    return {"mesh_file": os.path.join(case, "mesh.msh")}


def set_patch_types(case: str, types: dict[str, str]) -> None:
    """Rewrite the patch types in ``constant/polyMesh/boundary``.

    ``gmshToFoam`` gives every patch type ``patch``; a 2D case needs ``front``/``back`` to be ``empty`` and the
    section to be a ``wall``, and the solver refuses to start otherwise.
    """
    path = os.path.join(case, "constant", "polyMesh", "boundary")
    txt = open(path).read()
    for name, kind in types.items():
        txt = re.sub(rf"(\n    {name}\n    {{\n        type\s+)\w+;", rf"\g<1>{kind};", txt)
    open(path, "w").write(txt)

def _w(case: str, rel: str, cls: str, body: str) -> None:
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HDR % (cls, os.path.basename(rel)) + body)


def write_case_2d(case: str, *, speed: float, alpha_deg: float, nu: float, rho: float, iterations: int,
                  dz: float = 0.01, probes: np.ndarray | None = None, turbulence: bool = True,
                  relaxation: float = 0.7, reference_point: Sequence[float] = (0.25, 0.0, 0.0),
                  procs: int = 1) -> None:
    """The 0/ fields, the properties and the system dictionaries of a steady 2D section case.

    The section stays aligned with x and the **freestream** is what turns by ``alpha_deg``: one mesh per geometry
    then serves every incidence, which is what makes an incidence sweep cheap. ``probes`` (N x 3, metres) are
    sampled once, at the final write, with cell-point interpolation.
    """
    a = np.radians(alpha_deg)
    U = (speed * np.cos(a), speed * np.sin(a), 0.0)
    lift = (-np.sin(a), np.cos(a), 0.0)
    drag = (np.cos(a), np.sin(a), 0.0)
    empty = "    front\n    {\n        type empty;\n    }\n    back\n    {\n        type empty;\n    }\n"

    def fields(wall: str, far: str) -> str:
        return ("boundaryField\n{\n    airfoil\n    {\n" + wall + "    }\n    farfield\n    {\n" + far
                + "    }\n" + empty + "    \"proc.*\"\n    {\n        type processor;\n"
                "        value $internalField;\n    }\n}\n")

    _w(case, "0/U", "volVectorField",
       f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({U[0]:.8f} {U[1]:.8f} 0);\n"
       + fields("        type noSlip;\n",
                f"        type freestreamVelocity;\n        freestreamValue uniform "
                f"({U[0]:.8f} {U[1]:.8f} 0);\n"))
    _w(case, "0/p", "volScalarField",
       "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
       + fields("        type zeroGradient;\n",
                "        type freestreamPressure;\n        freestreamValue uniform 0;\n"))
    if turbulence:
        k = 1.5 * (0.001 * speed) ** 2
        om = k / nu

        def io(v):
            return f"        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"

        _w(case, "0/k", "volScalarField",
           f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k:.6g};\n"
           + fields(f"        type kqRWallFunction;\n        value uniform {k:.6g};\n", io(f"{k:.6g}")))
        _w(case, "0/omega", "volScalarField",
           f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {om:.6g};\n"
           + fields(f"        type omegaWallFunction;\n        value uniform {om:.6g};\n", io(f"{om:.6g}")))
        _w(case, "0/nut", "volScalarField",
           "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n"
           + fields("        type nutkWallFunction;\n        value uniform 0;\n",
                    "        type calculated;\n        value uniform 0;\n"))
    _w(case, "constant/transportProperties", "dictionary",
       f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};\n")
    _w(case, "constant/turbulenceProperties", "dictionary",
       "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n"
       if turbulence else "simulationType laminar;\n")

    probe_block = ""
    if probes is not None and len(probes):
        # The case is one cell thick in z, so a point at z = 0 lies on the front patch and the interpolation reports
        # "Not Found" for it -- most of a lattice drawn in the z = 0 plane, silently, as -1e300. The probes are lifted
        # to mid-span, which is the cell centre plane and where the solution actually is.
        z_mid = 0.5 * dz
        pts = "".join(f"({x:.8g} {y:.8g} {z_mid:.8g})\n" for x, y, _ in np.asarray(probes, float))
        probe_block = f"""    probes
    {{
        type probes;
        libs ("libsampling.so");
        fields (U p);
        probeLocations
        (
{pts}        );
        interpolationScheme cellPoint;
        writeControl writeTime;
    }}
"""
    _w(case, "system/controlDict", "dictionary", f"""application simpleFoam;
startFrom startTime;
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
functions
{{
    forceCoeffs
    {{
        type forceCoeffs;
        libs ("libforces.so");
        patches (airfoil);
        rho rhoInf; rhoInf {rho};
        CofR ({reference_point[0]} {reference_point[1]} {reference_point[2]});
        liftDir ({lift[0]:.8f} {lift[1]:.8f} 0);
        dragDir ({drag[0]:.8f} {drag[1]:.8f} 0);
        pitchAxis (0 0 1);
        magUInf {speed};
        lRef 1;
        Aref {dz};
        writeControl timeStep;
        writeInterval 1;
    }}
    yPlus
    {{
        type yPlus;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
    }}
{probe_block}}}
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
    _w(case, "system/fvSolution", "dictionary", f"""solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }}
    "(U|k|omega)" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 1;
    consistent yes;
}}
relaxationFactors
{{
    equations {{ U {relaxation:g}; k 0.5; omega 0.5; }}
    fields {{ p {relaxation:g}; }}
}}
""")
    nb = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 6: (3, 2, 1), 8: (2, 2, 2)}.get(procs, (procs, 1, 1))
    _w(case, "system/decomposeParDict", "dictionary",
       f"numberOfSubdomains {procs};\nmethod hierarchical;\n"
       f"hierarchicalCoeffs {{ n ({nb[0]} {nb[1]} {nb[2]}); order xyz; }}\n")



def foam_available() -> bool:
    """Whether a shell that sources ``FOAM_BASHRC`` finds the solver and the mesh checker."""
    r = subprocess.run(["bash", "-lc", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; command -v simpleFoam checkMesh'],
                       capture_output=True, text=True)
    return r.returncode == 0


def _sh(cmd: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{cwd}" && {cmd}'],
                          capture_output=True, text=True)


def run_case(case: str, *, procs: int = 1, log=None) -> dict:
    """``checkMesh`` then ``simpleFoam``; returns the wall times. A failed step is reported, never raised away."""
    case = os.path.abspath(case)
    out: dict = {"procs": procs}
    t0 = time.perf_counter()
    chk = _sh("checkMesh > log.checkMesh 2>&1", case)
    out["checkMesh_s"] = time.perf_counter() - t0
    out["checkMesh_ok"] = chk.returncode == 0
    t0 = time.perf_counter()
    if procs > 1:
        r = _sh("decomposePar -force > log.decomposePar 2>&1 && "
                f"mpirun --oversubscribe -np {procs} simpleFoam -parallel > log.simpleFoam 2>&1 && "
                "reconstructPar -latestTime > log.reconstructPar 2>&1", case)
    else:
        r = _sh("simpleFoam > log.simpleFoam 2>&1", case)
    out["solver_s"] = time.perf_counter() - t0
    out["solver_ok"] = r.returncode == 0
    if log:
        log(f"    checkMesh {'ok' if out['checkMesh_ok'] else 'FAILED'}, "
            f"simpleFoam {'ok' if out['solver_ok'] else 'FAILED'} in {out['solver_s']:.0f} s")
    return out


# --------------------------------------------------------------------------- readers
def check_mesh(case: str) -> dict:
    """The mesh checks the solver's own checker wrote: counts, quality and its verdict."""
    path = os.path.join(case, "log.checkMesh")
    if not os.path.exists(path):
        return {}
    text = open(path, errors="ignore").read()
    out: dict = {"ok": "Mesh OK." in text, "failed_checks": 0}
    m = re.search(r"Failed (\d+) mesh checks", text)
    if m:
        out["failed_checks"] = int(m.group(1))
    for key, pat, cast in (("cells", r"cells:\s+(\d+)", int),
                           ("nonorth_max", r"non-orthogonality Max:\s+([\d.eE+-]+)", float),
                           ("nonorth_avg", r"non-orthogonality.*?average:\s+([\d.eE+-]+)", float),
                           ("skew_max", r"Max skewness = ([\d.eE+-]+)", float),
                           ("aspect_max", r"Max aspect ratio = ([\d.eE+-]+)", float),
                           ("min_volume", r"Min volume = ([\d.eE+-]+)", float)):
        m = re.search(pat, text)
        if m:
            out[key] = cast(m.group(1).rstrip("."))
    m = re.search(r"Overall domain bounding box \(([^)]+)\) \(([^)]+)\)", text)
    if m:
        out["bbox"] = [float(v) for v in m.group(1).split() + m.group(2).split()]
    return out


def solver_summary(case: str) -> dict:
    """Iterations completed and the residuals and continuity error of the last one."""
    path = os.path.join(case, "log.simpleFoam")
    if not os.path.exists(path):
        return {}
    text = open(path, errors="ignore").read()
    out: dict = {"iterations": len(re.findall(r"^Time = ", text, re.M)),
                 "fatal": "FOAM FATAL" in text}
    tail = text[-4000:]
    for name, pat in (("Ux", r"Solving for Ux, Initial residual = ([\d.eE+-]+)"),
                      ("Uy", r"Solving for Uy, Initial residual = ([\d.eE+-]+)"),
                      ("p", r"Solving for p, Initial residual = ([\d.eE+-]+)"),
                      ("k", r"Solving for k, Initial residual = ([\d.eE+-]+)"),
                      ("omega", r"Solving for omega, Initial residual = ([\d.eE+-]+)")):
        found = re.findall(pat, tail)
        if found:
            out[f"res_{name}"] = float(found[-1])
    m = re.findall(r"time step continuity errors : sum local = ([\d.eE+-]+), global = ([\d.eE+-]+), "
                   r"cumulative = ([\d.eE+-]+)", tail)
    if m:
        out["continuity_local"], out["continuity_global"], out["continuity_cumulative"] = map(float, m[-1])
    return out


def read_history(case: str) -> dict:
    """Per-iteration force coefficients the ``forceCoeffs`` function object wrote."""
    base = os.path.join(case, "postProcessing", "forceCoeffs")
    if not os.path.isdir(base):
        return {}
    times = sorted((d for d in os.listdir(base) if d.replace(".", "").isdigit()), key=float)
    path = os.path.join(base, times[-1], "coefficient.dat")
    if not os.path.exists(path):
        return {}
    rows = [ln.split() for ln in open(path, errors="ignore") if not ln.lstrip().startswith("#")]
    rows = [r for r in rows if len(r) >= 9]
    if not rows:
        return {}
    a = np.array(rows, dtype=float)
    return {"iteration": a[:, 0], "CD": a[:, 1], "CD_pressure": a[:, 2], "CD_friction": a[:, 3],
            "CL": a[:, 4], "CL_pressure": a[:, 5], "CL_friction": a[:, 6], "CM": a[:, 7]}


def read_coefficients(case: str, *, average_from: float = 0.6) -> dict:
    """Mean of each coefficient over the iterations past ``average_from`` (a fraction of the run), and its spread."""
    h = read_history(case)
    if not h:
        return {}
    it = h["iteration"]
    keep = it >= average_from * it[-1]
    out: dict = {"averaged_iterations": int(keep.sum())}
    for k in ("CD", "CL", "CM", "CD_pressure", "CD_friction"):
        out[k] = float(np.mean(h[k][keep]))
        out[f"{k}_std"] = float(np.std(h[k][keep]))
    return out


def read_probes(case: str, n_points: int) -> dict:
    """``U`` and ``p`` at the probe locations, from the last time the ``probes`` object wrote."""
    base = os.path.join(case, "postProcessing", "probes")
    if not os.path.isdir(base):
        return {}
    times = sorted((d for d in os.listdir(base) if d.replace(".", "").isdigit()), key=float)
    if not times:
        return {}
    d = os.path.join(base, times[-1])
    out: dict = {}
    for name, per_point in (("U", 3), ("p", 1)):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        text = open(path, errors="ignore").read().replace("(", " ").replace(")", " ")
        rows = [ln.split() for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        if not rows:
            continue
        vals = np.array(rows[-1][1:], dtype=float)
        if vals.size == n_points * per_point:
            out[name] = vals.reshape(n_points, per_point)
        elif vals.size == n_points:
            out[name] = vals
    return out
