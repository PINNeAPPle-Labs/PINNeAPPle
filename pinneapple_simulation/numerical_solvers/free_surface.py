"""Free-surface flow of a liquid and a gas in OpenFOAM (interFoam, volume of fluid), in one object: an initial
liquid level, inlets with a velocity vector, open boundaries, k-omega SST, and turbulence sampled on lines and
averaged in time. Built for melt runners, launders, spillways and channels.

    from pinneapple_simulation.numerical_solvers.free_surface import FreeSurfaceFlow, Phase, SampleLine
    flow = FreeSurfaceFlow(
        regions={"walls": (V, F), "jet": (Vi, Fi), "top": (Vt, Ft), "exit": (Ve, Fe)},     # a closed fluid volume
        roles={"jet": "inlet", "top": "atmosphere", "exit": "outlet"},                      # the rest: walls
        inlet_velocity={"jet": (0.0, 9.85, 1.74)}, level=-2.04,
        liquid=Phase("metal", rho=7200, nu=8.472e-6), gas=Phase("air", rho=1.2, nu=1.5e-5), sigma=1.8,
        end_time=30, sample_every=0.1, lines=[SampleLine("L1", (x, y0, z), (x, y1, z))])
    flow.write("case"); flow.run("case", procs=4)
    prof = flow.profiles("case")            # {line: {"s", "times", "k", "alpha", "mean"}}

The mesh comes from ``regions`` (closed surface split into named regions; snappyHexMesh) or from ``mesh``, a
directory with ``constant/polyMesh`` whose patch names match ``roles``. Lines are sampled by this module from the
written fields, the nearest cell's value (no spatial interpolation), so the result does not depend on the
OpenFOAM version's function objects; with ``keep_every`` the sampled times are then deleted, except every
``keep_every`` seconds, which bounds the disk use of long transients.

Boundary behaviour: inlets impose U (the full vector), liquid fraction 1 and turbulence from intensity and mixing
length; outlets and atmosphere are open (total pressure p_ref, reverse flow brings gas); walls are no-slip with
wall functions. The liquid starts below ``level``, a height along -gravity (so gravity may be tilted, as in a sloped channel
meshed in its own frame); ``hRef`` is set there so the initial hydrostatic state is in balance.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .external_flow import FOAM_BASHRC, _stl, _w, snappy_dict

__all__ = ["Phase", "SampleLine", "FreeSurfaceFlow", "LiveLineSampler", "split_surface", "sample_lines",
           "read_profiles", "time_average", "cell_centres"]

ROLES = ("inlet", "outlet", "atmosphere", "wall")


@dataclass
class Phase:
    """One phase: name, density ``rho`` and kinematic viscosity ``nu``."""
    name: str
    rho: float
    nu: float


@dataclass
class SampleLine:
    """A sampling line: name, start, end and number of points."""
    name: str
    start: Sequence[float]
    end: Sequence[float]
    n: int = 101


def split_surface(V: np.ndarray, F: np.ndarray, boxes: dict[str, tuple[Sequence[float], Sequence[float]]],
                  rest: str = "walls") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split one closed surface into named regions: a triangle goes to the first box that holds its centroid,
    otherwise to ``rest``. Feed the result to ``FreeSurfaceFlow(regions=...)``."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    c = V[F].mean(1)
    tag = np.full(len(F), -1)
    names = list(boxes)
    for k, n in enumerate(names):
        lo, hi = np.asarray(boxes[n][0], float), np.asarray(boxes[n][1], float)
        inside = np.all((c >= lo) & (c <= hi), axis=1) & (tag < 0)
        tag[inside] = k
    out = {}
    for k, n in enumerate(names + [rest]):
        sel = F[tag == (k if k < len(names) else -1)]
        if len(sel):
            used, inv = np.unique(sel, return_inverse=True)
            out[n] = (V[used], inv.reshape(-1, 3))
    return out


# ------------------------------------------------------------------------------------------------ the case
@dataclass
class FreeSurfaceFlow:
    """A complete interFoam (VOF) case: regions, patch roles, liquid level, phases and turbulence; writes the case,
    runs it and reads profiles back.
    """
    regions: dict | None = None
    roles: dict[str, str] = field(default_factory=dict)
    mesh: str | None = None
    inlet_velocity: dict[str, Sequence[float]] = field(default_factory=dict)
    level: float = 0.0
    liquid: Phase = field(default_factory=lambda: Phase("water", 1000.0, 1e-6))
    gas: Phase = field(default_factory=lambda: Phase("air", 1.2, 1.5e-5))
    sigma: float = 0.07
    gravity: Sequence[float] = (0.0, 0.0, -9.81)
    p_ref: float = 0.0
    turbulence: str = "kOmegaSST"                        # or "laminar"
    intensity: float = 0.01
    mixing_length: float = 0.1
    end_time: float = 1.0
    delta_t: float = 1e-3
    max_co: float = 1.0
    max_alpha_co: float = 1.0
    max_delta_t: float = 0.01
    sample_every: float = 0.1
    keep_every: float | None = None
    lines: list[SampleLine] = field(default_factory=list)
    cell_size: float | None = None                       # background cell (regions only); default: 30 across
    surface_level: dict[str, int] = field(default_factory=dict)
    refine_boxes: list[tuple[Sequence[float], Sequence[float], int]] = field(default_factory=list)
    location_in_mesh: Sequence[float] | None = None
    two_d: bool = False                                  # mesh is one cell thick with "empty" front/back
    # p_rgh linear solver (fvSolution entries), e.g. accelerated_foam.case.warm_solver("PCG", 1e-7, 0.05) for the
    # warm-started solve, with the library in ``libs``; default PCG/DIC, tolerance 1e-7, relTol 0.05
    pressure_solver: dict | None = None
    libs: Sequence[str] = ()                              # extra libraries loaded by controlDict

    def __post_init__(self):
        if (self.regions is None) == (self.mesh is None):
            raise ValueError("FreeSurfaceFlow: provide either regions (surfaces) or mesh (a case with constant/polyMesh)")
        for n, r in self.roles.items():
            if r not in ROLES:
                raise ValueError(f"role of {n!r} must be one of {ROLES}, not {r!r}")
        for n in self.inlet_velocity:
            if self.roles.get(n) != "inlet":
                raise ValueError(f"{n!r} has an inlet velocity but its role is {self.roles.get(n)!r}")
        missing = [n for n, r in self.roles.items() if r == "inlet" and n not in self.inlet_velocity]
        if missing:
            raise ValueError(f"inlets without a velocity: {missing}")

    # ---------------------------------------------------------------- derived values
    @property
    def alpha(self) -> str:
        return f"alpha.{self.liquid.name}"

    def _gk(self) -> int:
        g = np.asarray(self.gravity, float)
        return int(np.argmax(np.abs(g)))

    def _k_omega(self, U: Sequence[float]) -> tuple[float, float]:
        u = float(np.linalg.norm(U))
        k = 1.5 * (u * self.intensity) ** 2
        return k, math.sqrt(max(k, 1e-12)) / (0.09 ** 0.25 * self.mixing_length)

    def _patches(self, names: Sequence[str]) -> dict[str, str]:
        return {n: self.roles.get(n, "wall") for n in names}

    # ---------------------------------------------------------------- writing
    def write(self, case: str, procs: int = 4) -> dict:
        if os.path.isdir(case):
            shutil.rmtree(case)
        os.makedirs(os.path.join(case, "constant"))
        if self.mesh is not None:
            src = os.path.join(self.mesh, "constant", "polyMesh") if os.path.isdir(
                os.path.join(self.mesh, "constant", "polyMesh")) else self.mesh
            shutil.copytree(src, os.path.join(case, "constant", "polyMesh"))
            names = [n for n, t in _boundary_types(os.path.join(case, "constant", "polyMesh", "boundary")).items()
                     if t != "empty"]
        else:
            names = self._write_mesh_dicts(case)
        patches = self._patches(names)
        self._write_fields(case, patches)
        self._write_constant(case)
        self._write_system(case, procs)
        info = {"liquid": self.liquid.__dict__, "gas": self.gas.__dict__, "sigma": self.sigma,
                "gravity": list(self.gravity), "level": self.level, "patches": patches,
                "inlet_velocity": {k: list(v) for k, v in self.inlet_velocity.items()},
                "lines": [{"name": ln.name, "start": list(ln.start), "end": list(ln.end), "n": ln.n} for ln in self.lines],
                "end_time": self.end_time, "sample_every": self.sample_every, "turbulence": self.turbulence}
        with open(os.path.join(case, "pinneapple_free_surface.json"), "w") as f:
            json.dump(info, f, indent=1)
        return info

    def _write_mesh_dicts(self, case: str) -> list[str]:
        tri = os.path.join(case, "constant", "triSurface")
        os.makedirs(tri)
        names, allV = [], []
        for n, (V, F) in self.regions.items():
            nm = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in n)
            names.append(nm)
            allV.append(np.asarray(V, float))
            with open(os.path.join(tri, f"{nm}.stl"), "wb") as f:
                f.write(_stl(np.asarray(V, float), np.asarray(F, np.int64), nm))
        V = np.concatenate(allV)
        lo, hi = V.min(0), V.max(0)
        h = self.cell_size or float((hi - lo).max()) / 30
        lo, hi = lo - 0.5 * h, hi + 0.5 * h
        n = [max(1, int(math.ceil((hi[i] - lo[i]) / h))) for i in range(3)]
        hi = lo + np.array(n) * h
        _w(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices ( ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {lo[2]}) ({lo[0]} {hi[1]} {lo[2]})
           ({lo[0]} {lo[1]} {hi[2]}) ({hi[0]} {lo[1]} {hi[2]}) ({hi[0]} {hi[1]} {hi[2]}) ({lo[0]} {hi[1]} {hi[2]}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) ); }} );
mergePatchPairs ();
""")
        _w(case, "system/surfaceFeatureExtractDict", "dictionary", "\n".join(
            f"{b}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} "
            f"subsetFeatures {{ nonManifoldEdges no; openEdges yes; }} writeObj no; }}" for b in names) + "\n")
        geom = "\n".join(f"    {b}.stl {{ type triSurfaceMesh; name {b}; }}" for b in names)
        for k, (a, b, _) in enumerate(self.refine_boxes):
            geom += f"\n    box{k} {{ type searchableBox; min ({a[0]} {a[1]} {a[2]}); max ({b[0]} {b[1]} {b[2]}); }}"
        feats = "\n".join(f"        {{ file \"{b}.eMesh\"; level {self.surface_level.get(b, 0)}; }}" for b in names)
        ptype = {"wall": "wall", "inlet": "patch", "outlet": "patch", "atmosphere": "patch"}
        surfs = "\n".join(f"        {b} {{ level ({self.surface_level.get(b, 0)} {self.surface_level.get(b, 0)}); "
                          f"patchInfo {{ type {ptype[self.roles.get(b, 'wall')]}; }} }}" for b in names)
        regs = "\n".join(f"        box{k} {{ mode inside; levels ((1E15 {lv})); }}"
                         for k, (_, _, lv) in enumerate(self.refine_boxes))
        loc = self.location_in_mesh
        if loc is None:
            raise ValueError("location_in_mesh is required with regions: a point inside the fluid volume")
        _w(case, "system/snappyHexMeshDict", "dictionary", snappy_dict(geom, feats, surfs, regs, loc))
        return names

    def _write_fields(self, case: str, patches: dict[str, str]) -> None:
        U0 = next(iter(self.inlet_velocity.values()), (1.0, 0.0, 0.0))
        k0, w0 = self._k_omega(U0)

        def bf(by_role: dict[str, str], extra: dict[str, str] | None = None) -> str:
            out = []
            for n, r in patches.items():
                out.append(f"    {n}\n    {{\n{(extra or {}).get(n, by_role[r])}\n    }}")
            if self.two_d:
                out.append("    \"(front|back|frontAndBack)\"\n    {\n        type empty;\n    }")
            return "boundaryField\n{\n" + "\n".join(out) + "\n}\n"
        open_U = "        type pressureInletOutletVelocity;\n        value uniform (0 0 0);"
        U_in = {n: f"        type fixedValue;\n        value uniform ({' '.join(repr(float(c)) for c in v)});"
                for n, v in self.inlet_velocity.items()}
        _w(case, "0/U", "volVectorField", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n" + bf(
            {"wall": "        type noSlip;", "outlet": open_U, "atmosphere": open_U, "inlet": ""}, U_in))
        open_p = f"        type prghTotalPressure;\n        p0 uniform {self.p_ref!r};\n        value uniform {self.p_ref!r};"
        ffp = "        type fixedFluxPressure;\n        value uniform 0;"
        _w(case, "0/p_rgh", "volScalarField", f"dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform {self.p_ref!r};\n"
           + bf({"wall": ffp, "inlet": ffp, "outlet": open_p, "atmosphere": open_p}))
        open_a = "        type inletOutlet;\n        inletValue uniform 0;\n        value uniform 0;"
        _w(case, f"0/{self.alpha}", "volScalarField", "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\n" + bf(
            {"wall": "        type zeroGradient;", "inlet": "        type fixedValue;\n        value uniform 1;",
             "outlet": open_a, "atmosphere": open_a}))
        if self.turbulence == "laminar":
            return
        io = lambda v: f"        type inletOutlet;\n        inletValue uniform {v!r};\n        value uniform {v!r};"  # noqa: E731
        _w(case, "0/k", "volScalarField", f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k0!r};\n" + bf(
            {"wall": f"        type kqRWallFunction;\n        value uniform {k0!r};", "outlet": io(k0),
             "atmosphere": io(k0), "inlet": f"        type turbulentIntensityKineticEnergyInlet;\n"
                                             f"        intensity {self.intensity!r};\n        value uniform {k0!r};"}))
        _w(case, "0/omega", "volScalarField", f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {w0!r};\n" + bf(
            {"wall": f"        type omegaWallFunction;\n        value uniform {w0!r};", "outlet": io(w0),
             "atmosphere": io(w0), "inlet": f"        type turbulentMixingLengthFrequencyInlet;\n"
                                             f"        mixingLength {self.mixing_length!r};\n        value uniform {w0!r};"}))
        calc = "        type calculated;\n        value uniform 0;"
        _w(case, "0/nut", "volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n" + bf(
            {"wall": "        type nutkWallFunction;\n        value uniform 0;", "outlet": calc, "atmosphere": calc,
             "inlet": calc}))

    def _write_constant(self, case: str) -> None:
        L, G = self.liquid, self.gas
        _w(case, "constant/transportProperties", "dictionary", f"""phases ({L.name} {G.name});
{L.name} {{ transportModel Newtonian; nu {L.nu!r}; rho {L.rho!r}; }}
{G.name} {{ transportModel Newtonian; nu {G.nu!r}; rho {G.rho!r}; }}
sigma {self.sigma!r};
""")
        model = ("simulationType laminar;\n" if self.turbulence == "laminar" else
                 f"simulationType RAS;\nRAS {{ RASModel {self.turbulence}; turbulence on; printCoeffs on; }}\n")
        _w(case, "constant/turbulenceProperties", "dictionary", model)
        g = tuple(float(c) for c in self.gravity)
        _w(case, "constant/g", "uniformDimensionedVectorField",
           f"dimensions [0 1 -2 0 0 0 0];\nvalue ({g[0]!r} {g[1]!r} {g[2]!r});\n")
        href = self.level
        _w(case, "constant/hRef", "uniformDimensionedScalarField", f"dimensions [0 1 0 0 0 0 0];\nvalue {href!r};\n")

    def _write_system(self, case: str, procs: int) -> None:
        a = self.alpha
        _w(case, "system/controlDict", "dictionary", f"""application interFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {self.end_time!r};
deltaT {self.delta_t!r};
writeControl adjustableRunTime;
writeInterval {self.sample_every!r};
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable true;
adjustTimeStep yes;
maxCo {self.max_co!r};
maxAlphaCo {self.max_alpha_co!r};
maxDeltaT {self.max_delta_t!r};
""" + ("libs (" + " ".join(f'"{x}"' for x in self.libs) + ");\n" if self.libs else ""))
        turb = "" if self.turbulence == "laminar" else "    div(phi,k) Gauss upwind;\n    div(phi,omega) Gauss upwind;\n"
        _w(case, "system/fvSchemes", "dictionary", f"""ddtSchemes {{ default Euler; }}
gradSchemes {{ default Gauss linear; }}
divSchemes
{{
    div(rhoPhi,U) Gauss linearUpwind grad(U);
    div(phi,alpha) Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
{turb}}}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
""")
        p_rgh = " ".join(f"{k} {'true' if v is True else 'false' if v is False else v};" for k, v in
                         (self.pressure_solver or {"solver": "PCG", "preconditioner": "DIC", "tolerance": 1e-7,
                                                   "relTol": 0.05}).items())
        _w(case, "system/fvSolution", "dictionary", f"""solvers
{{
    "{a}.*"
    {{
        nAlphaCorr 2; nAlphaSubCycles 1; cAlpha 1; MULESCorr yes; nLimiterIter 5;
        solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;
    }}
    "pcorr.*" {{ solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }}
    p_rgh {{ {p_rgh} }}
    p_rghFinal {{ $p_rgh; relTol 0; }}
    "(U|k|omega).*" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0; }}
}}
PIMPLE {{ momentumPredictor no; nOuterCorrectors 1; nCorrectors 3; nNonOrthogonalCorrectors 0; }}
relaxationFactors {{ equations {{ ".*" 1; }} }}
""")
        g = np.asarray(self.gravity, float)
        up = -g / np.linalg.norm(g)
        if np.count_nonzero(np.abs(g) > 1e-12 * np.abs(g).max()) == 1:
            k = self._gk()
            lo, hi = np.full(3, -1e6), np.full(3, 1e6)
            if g[k] < 0:
                hi[k] = self.level
            else:
                lo[k] = self.level
            region = f"boxToCell {{ box ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {hi[2]}); "
        else:                                         # tilted gravity: a rotated box below the level plane
            i = np.cross(up, [0.0, 1.0, 0.0] if abs(up[1]) < 0.9 else [1.0, 0.0, 0.0])
            i /= np.linalg.norm(i)
            j = np.cross(up, i)
            big = 1e4
            o = up * (self.level - big) - (i + j) * big / 2
            fmt = lambda v: " ".join(repr(float(c)) for c in v)  # noqa: E731
            region = (f"rotatedBoxToCell {{ origin ({fmt(o)}); i ({fmt(i * big)}); j ({fmt(j * big)}); "
                      f"k ({fmt(up * big)}); ")
        _w(case, "system/setFieldsDict", "dictionary", f"""defaultFieldValues ( volScalarFieldValue {a} 0 );
regions ( {region}fieldValues ( volScalarFieldValue {a} 1 ); }} );
""")
        k = self._gk()
        # simple decomposition along the longest horizontal axis (no scotch needed)
        n = [1, 1, 1]
        if self.regions is not None:
            V = np.concatenate([np.asarray(v, float) for v, _ in self.regions.values()])
            ext = V.max(0) - V.min(0)
            ext[k] = 0
            n[int(np.argmax(ext))] = procs
        else:
            n[0 if k != 0 else 1] = procs
        _w(case, "system/decomposeParDict", "dictionary",
           f"numberOfSubdomains {procs};\nmethod simple;\nsimpleCoeffs {{ n ({n[0]} {n[1]} {n[2]}); delta 0.001; }}\n")

    # ---------------------------------------------------------------- running
    def run(self, case: str, procs: int = 4, log: Callable[[str], None] = print) -> dict[str, float]:
        """Mesh (with ``regions``), setFields, decompose, interFoam; lines sampled while it runs. Seconds per step."""
        t: dict[str, float] = {}
        case = os.path.abspath(case)

        def sh(cmd, name):
            t0 = time.time()
            r = subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} '
                                              f'> log.{name} 2>&1'])
            t[name] = round(time.time() - t0, 1)
            log(f"  {name}: {t[name]} s (exit {r.returncode})")
            if r.returncode:
                raise RuntimeError(f"{name} failed, see {case}/log.{name}")
        if self.regions is not None:
            sh("blockMesh", "blockMesh")
            sh("surfaceFeatureExtract", "surfaceFeatureExtract")
            sh("snappyHexMesh -overwrite", "snappyHexMesh")
            sh("checkMesh", "checkMesh")
        sh("setFields", "setFields")
        mpi = f"mpirun --allow-run-as-root --oversubscribe -np {procs}"
        if procs > 1:
            sh("decomposePar -force", "decomposePar")
        sampler = LiveLineSampler(case, self.lines, [self.alpha] + ([] if self.turbulence == "laminar" else ["k"]),
                                  keep_every=self.keep_every, end_time=self.end_time).start() if self.lines else None
        try:
            sh(f"{mpi} interFoam -parallel" if procs > 1 else "interFoam", "interFoam")
        finally:
            if sampler is not None:
                sampler.stop()
        return t

    def profiles(self, case: str, t_start: float = 0.0, t_end: float = float("inf")) -> dict:
        return read_profiles(case, [ln.name for ln in self.lines], t_start, t_end)


# ------------------------------------------------------------------------------------------- reading
def _boundary_types(path: str) -> dict[str, str]:
    """{patch name: type} from constant/polyMesh/boundary."""
    import re
    txt = open(path).read()
    if "FoamFile" in txt:
        txt = txt[txt.index("}", txt.index("FoamFile")) + 1:]
    return dict(re.findall(r"([A-Za-z_][\w.:-]*)\s*\{[^{}]*?\btype\s+(\w+)\s*;", txt))


def _ascii_list(path: str) -> np.ndarray | tuple:
    """internalField of an ASCII OpenFOAM scalar field, or the points/faces/owner/neighbour lists."""
    import re
    t = open(path).read()
    m = re.search(r"internalField\s+uniform\s+([^;]+);", t)
    if m:
        return ("uniform", float(m.group(1)))
    m = re.search(r"\n(\d+)\s*\n\(", t)
    n, body = int(m.group(1)), t[m.end():]
    body = body[:body.index("\n)")]
    return n, body


def cell_centres(poly_dir: str) -> np.ndarray:
    """Cell centres (mean of face centres) from an ASCII constant/polyMesh."""
    import re
    n, b = _ascii_list(os.path.join(poly_dir, "points"))
    P = np.array(re.findall(r"\(([^()]+)\)", b)[:n], dtype=object)
    P = np.array([p.split() for p in P], float)
    n, b = _ascii_list(os.path.join(poly_dir, "faces"))
    faces = [list(map(int, f.split())) for f in re.findall(r"\(([^()]+)\)", b)[:n]]
    own = np.array(_ascii_list(os.path.join(poly_dir, "owner"))[1].split(), int)
    nei = np.array(_ascii_list(os.path.join(poly_dir, "neighbour"))[1].split(), int)
    fc = np.array([P[f].mean(0) for f in faces])
    nc = int(own.max()) + 1
    cc, cnt = np.zeros((nc, 3)), np.zeros(nc)
    np.add.at(cc, own, fc[:len(own)])
    np.add.at(cnt, own, 1)
    np.add.at(cc, nei, fc[:len(nei)])
    np.add.at(cnt, nei, 1)
    return cc / cnt[:, None]


def _scalar(path: str, n: int) -> np.ndarray:
    v = _ascii_list(path)
    if v[0] == "uniform":
        return np.full(n, v[1])
    return np.array(v[1].split()[:v[0]], float)


def _times(d: str) -> list[tuple[float, str]]:
    out = []
    for x in os.listdir(d):
        try:
            out.append((float(x), x))
        except ValueError:
            pass
    return sorted(out)


class _LineIndex:
    def __init__(self, case: str, lines: Sequence[dict], parallel: bool):
        from scipy.spatial import cKDTree
        self.dirs = sorted((os.path.join(case, d) for d in os.listdir(case) if d.startswith("processor")),
                           key=lambda p: int(p.rsplit("processor", 1)[1])) if parallel else [case]
        cc = [cell_centres(os.path.join(d, "constant", "polyMesh")) for d in self.dirs]
        self.counts = [len(c) for c in cc]
        allc = np.vstack(cc)
        tree = cKDTree(allc)
        dn, _ = tree.query(allc, k=2)
        size = dn[:, 1]
        self.idx = {}
        for ln in lines:
            P = np.linspace(ln["start"], ln["end"], ln["n"])
            d, i = tree.query(P)
            ok = d <= 0.9 * size[i]                   # points outside the fluid have no cell
            self.idx[ln["name"]] = (P[ok], i[ok], np.linalg.norm(P[ok] - P[0], axis=1))

    def sample(self, tname: str, fields: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
        vals = {f: np.concatenate([_scalar(os.path.join(d, tname, f), n) for d, n in zip(self.dirs, self.counts,
                                                                                         strict=True)])
                for f in fields}
        return {name: {f: vals[f][i] for f in fields} for name, (_, i, _) in self.idx.items()}


class LiveLineSampler:
    """Samples lines from a running case as each time is written, then optionally deletes that time.

    ``case`` is a directory or a callable that returns the directory currently being run (or None), for runners
    that retry in fresh directories; when it changes, sampling restarts there. Serial and decomposed cases are
    detected per directory. The newest time is left alone while the solver writes; ``stop()`` samples the rest.
    Profiles go to ``<case>/postProcessing/<out_name>/<time>/<line>.csv`` (``read_profiles`` reads them).

        s = LiveLineSampler(case, lines, ["k", "alpha.water"], keep_every=1.0).start()
        ... run the solver ...
        s.stop()
    """

    def __init__(self, case, lines: Sequence[SampleLine | dict], fields: Sequence[str],
                 keep_every: float | None = None, end_time: float = float("inf"), out_name: str = "lines",
                 poll_s: float = 3.0):
        self._case = case
        self.lines = [ln if isinstance(ln, dict) else {"name": ln.name, "start": list(ln.start), "end": list(ln.end),
                                                       "n": ln.n} for ln in lines]
        self.fields, self.keep_every, self.end_time = list(fields), keep_every, end_time
        self.out_name, self.poll_s = out_name, poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sampled: dict[str, list[str]] = {}
        self.errors: list[str] = []

    def _dir(self) -> str | None:
        d = self._case() if callable(self._case) else self._case
        return str(d) if d else None

    def start(self) -> LiveLineSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, list[str]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.sampled

    def _loop(self) -> None:
        cur, idx, done = None, None, set()
        while True:
            finished = self._stop.is_set()
            d = self._dir()
            if d != cur:
                cur, idx, done = d, None, set()
            if cur and idx is None:
                parallel = os.path.isdir(os.path.join(cur, "processor0", "constant", "polyMesh"))
                if parallel or os.path.isdir(os.path.join(cur, "constant", "polyMesh")):
                    try:
                        idx = _LineIndex(cur, self.lines, parallel)
                    except (OSError, ValueError, AttributeError) as e:      # mesh still being written
                        if finished:
                            self.errors.append(f"{cur}: {e}")
            if idx is not None:
                base = idx.dirs[0]
                ts = [(t, s) for t, s in _times(base) if t > 0 and s not in done]
                if not finished and ts:
                    ts = ts[:-1]                          # the newest may still be being written
                for t, s in ts:
                    try:
                        res = idx.sample(s, self.fields)
                    except (FileNotFoundError, ValueError, IndexError, AttributeError) as e:
                        if finished:
                            self.errors.append(f"{s}: {e}")
                            continue
                        break
                    write_profile(cur, s, idx, res, self.fields, self.out_name)
                    done.add(s)
                    self.sampled.setdefault(cur, []).append(s)
                    ke = self.keep_every
                    if ke and abs(t / ke - round(t / ke)) > 1e-6 and t < self.end_time - 1e-9:
                        for dd in idx.dirs:
                            shutil.rmtree(os.path.join(dd, s), ignore_errors=True)
            if finished:
                return
            self._stop.wait(self.poll_s)


def write_profile(case: str, tname: str, idx: _LineIndex, res: dict, fields: Sequence[str],
                  out_name: str = "lines") -> None:
    out = os.path.join(case, "postProcessing", out_name, tname)
    os.makedirs(out, exist_ok=True)
    for name, (P, _, s) in idx.idx.items():
        with open(os.path.join(out, f"{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y", "z", "distance", *fields])
            for j in range(len(P)):
                w.writerow([*np.round(P[j], 6), round(float(s[j]), 6), *[res[name][f][j] for f in fields]])


def sample_lines(case: str, lines: Sequence[SampleLine], fields: Sequence[str]) -> int:
    """Sample every written time of a finished case (serial or decomposed) onto ``lines``. Returns the count."""
    parallel = os.path.isdir(os.path.join(case, "processor0"))
    spec = [{"name": ln.name, "start": ln.start, "end": ln.end, "n": ln.n} for ln in lines]
    idx = _LineIndex(case, spec, parallel)
    n = 0
    for t, s in _times(os.path.join(case, "processor0") if parallel else case):
        if t > 0:
            write_profile(case, s, idx, idx.sample(s, fields), fields)
            n += 1
    return n


def read_profiles(case: str, names: Sequence[str] | None = None, t_start: float = 0.0,
                  t_end: float = float("inf"), out_name: str = "lines") -> dict:
    """{line: {"s": distance along the line, "xyz", "times", <field>: (n_times, n_points), "mean_<field>"}}."""
    base = os.path.join(case, "postProcessing", out_name)
    out: dict = {}
    for t, s in _times(base):
        if not (t_start <= t <= t_end):
            continue
        for fn in sorted(os.listdir(os.path.join(base, s))):
            name = fn[:-4]
            if names and name not in names:
                continue
            rows = list(csv.reader(open(os.path.join(base, s, fn))))
            head, data = rows[0], np.array(rows[1:], float)
            d = out.setdefault(name, {"s": data[:, 3], "xyz": data[:, :3], "times": [],
                                      **{h: [] for h in head[4:]}})
            d["times"].append(t)
            for j, h in enumerate(head[4:]):
                d[h].append(data[:, 4 + j])
    for d in out.values():
        d["times"] = np.array(d["times"])
        for h in [k for k in d if k not in ("s", "xyz", "times")]:
            d[h] = np.array(d[h])
            d[f"mean_{h}"] = d[h].mean(0)
    return out


def time_average(profile: dict, field: str, window: Sequence[float] | None = None,
                 origin: str = "min") -> tuple[np.ndarray, np.ndarray]:
    """Time-mean of ``field`` along a line against the distance from the line's first sampled point
    (``origin="min"``), optionally cut to ``window`` = (s0, s1)."""
    s, m = profile["s"] - (profile["s"].min() if origin == "min" else 0.0), profile[f"mean_{field}"]
    if window is not None:
        keep = (s >= window[0]) & (s <= window[1])
        s, m = s[keep], m[keep]
    return s, m
