"""Internal flow in pipes, bends and static mixers in OpenFOAM, in one call.

The pipe is a structured O-grid (a square core and four curved blocks) swept along a centreline made of straight
runs and circular bends, written as a blockMeshDict; static-mixer elements (Kenics twisted plates) are cut out of it
by snappyHexMesh. simpleFoam solves the flow (laminar or k-omega SST with wall functions); a passive scalar entering
on one half of the inlet is then carried by scalarTransportFoam on the converged fluxes, to measure mixing.

    from pinneapple_simulation.numerical_solvers.internal_flow import InternalFlow, Route
    flow = InternalFlow(Route(D=0.05).straight(20).bend(2.0, 90).straight(30), Re=5e4)
    res = flow.solve("cases/bend", procs=4)
    res.pressure_drop, res.sections["x"], res.sections["p"]     # area-averaged pressure along the centreline
    res.to_scene()                                              # walls, streamlines, cross-section slices

Lengths of a ``Route`` are in diameters. Velocities follow from ``Re`` and ``nu`` (U = Re nu / D). Needs OpenFOAM
(v1912+ tested); ``FOAM_BASHRC`` points to its bashrc.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .external_flow import FOAM_BASHRC, HDR, _stl, _w


# ---------------------------------------------------------------------- geometry
@dataclass
class Route:
    """A pipe centreline: straight runs and circular bends, lengths and radii in diameters. Starts at the origin
    along +x; bends turn in the plane given by ``plane_deg`` (0: towards +y, 90: towards +z)."""
    D: float = 0.05
    segments: list[tuple[str, float, float, float]] = field(default_factory=list)

    def straight(self, length_D: float) -> Route:
        self.segments.append(("straight", float(length_D), 0.0, 0.0))
        return self

    def bend(self, radius_D: float, angle_deg: float, plane_deg: float = 0.0) -> Route:
        self.segments.append(("bend", float(radius_D), float(angle_deg), float(plane_deg)))
        return self

    @property
    def length(self) -> float:
        """Centreline length (m)."""
        return sum(s[1] if s[0] == "straight" else s[1] * math.radians(s[2]) for s in self.segments) * self.D

    def stations(self, ds_D: float = 0.5, bend_step_deg: float = 6.0) -> dict[str, np.ndarray]:
        """Centreline points, tangents and two normals (parallel transport), arc length; one station per straight
        run end and every ``bend_step_deg`` in bends. ``kind``: 0 straight interval, 1 bend interval (per interval)."""
        D = self.D
        P, T, N = [np.zeros(3)], [np.array([1.0, 0, 0])], [np.array([0, 1.0, 0])]
        kinds: list[int] = []
        for kind, a, b, c in self.segments:
            p, t, n = P[-1], T[-1], N[-1]
            bn = np.cross(t, n)
            if kind == "straight":
                P.append(p + a * D * t)
                T.append(t)
                N.append(n)
                kinds.append(0)
                continue
            R, ang = a * D, math.radians(b)
            ph = math.radians(c)
            towards = math.cos(ph) * n + math.sin(ph) * bn          # the bend centre direction
            ctr = p + R * towards
            m = max(2, int(math.ceil(b / bend_step_deg)))
            for k in range(1, m + 1):
                th = ang * k / m
                pk = ctr - R * math.cos(th) * towards + R * math.sin(th) * t
                tk = math.cos(th) * t + math.sin(th) * towards
                # parallel transport of the normal: rotate n about the bend axis by th
                axis = np.cross(t, towards)
                nk = _rotate(n, axis, th)
                P.append(pk)
                T.append(tk)
                N.append(nk)
                kinds.append(1)
        P, T, N = np.array(P), np.array(T), np.array(N)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        # arc length along bends: chord sums are slightly short; use the exact arc
        s_exact = [0.0]
        k = 0
        for kind, a, b, _ in self.segments:
            if kind == "straight":
                s_exact.append(s_exact[-1] + a * D)
                k += 1
            else:
                m = max(2, int(math.ceil(b / bend_step_deg)))
                for _ in range(m):
                    s_exact.append(s_exact[-1] + a * D * math.radians(b) / m)
                k += m
        return {"P": P, "T": T, "N": N, "B": np.cross(T, N), "s": np.asarray(s_exact), "kind": np.asarray(kinds),
                "chord": s}


def _rotate(v: np.ndarray, axis: np.ndarray, th: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-300)
    return v * math.cos(th) + np.cross(a, v) * math.sin(th) + a * (a @ v) * (1 - math.cos(th))


def kenics_elements(route_D: float, n_elements: int, start_D: float, length_D: float = 1.5, thickness_D: float = 0.06,
                    width_D: float = 0.98, nu: int = 10, ns: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Kenics static-mixer elements in a straight pipe along +x starting at ``start_D``: plates twisted by 180 deg,
    alternately left- and right-handed, each turned 90 deg from the previous one. A closed triangulated surface."""
    D = route_D
    Vs, Fs = [], []
    off = 0
    for e in range(n_elements):
        hand = 1 if e % 2 == 0 else -1
        th0 = 0.0 if e % 2 == 0 else math.pi / 2           # the leading edge is turned 90 deg from the last trailing edge
        x0 = (start_D + e * length_D) * D
        s = np.linspace(0, 1, ns)
        u = np.linspace(-0.5, 0.5, nu) * width_D * D
        th = th0 + hand * math.pi * s
        X = x0 + s * length_D * D
        h = 0.5 * thickness_D * D
        V = []
        for side in (-1, 1):                               # two faces of the plate
            for i in range(ns):
                c, sn = math.cos(th[i]), math.sin(th[i])
                for j in range(nu):
                    # plate direction (c, s) in the y-z plane, its normal (-s, c)
                    V.append((X[i], u[j] * c - side * h * sn, u[j] * sn + side * h * c))
        V = np.asarray(V)
        F = []
        idx = lambda side, i, j: side * ns * nu + i * nu + j   # noqa: E731
        for i in range(ns - 1):
            for j in range(nu - 1):
                a, b, c2, d = idx(0, i, j), idx(0, i + 1, j), idx(0, i + 1, j + 1), idx(0, i, j + 1)
                F += [(a, c2, b), (a, d, c2)]
                a, b, c2, d = idx(1, i, j), idx(1, i + 1, j), idx(1, i + 1, j + 1), idx(1, i, j + 1)
                F += [(a, b, c2), (a, c2, d)]
        for i in range(ns - 1):                            # the two long edges
            for j, sg in ((0, 1), (nu - 1, -1)):
                a, b, c2, d = idx(0, i, j), idx(0, i + 1, j), idx(1, i + 1, j), idx(1, i, j)
                F += [(a, b, c2), (a, c2, d)] if sg > 0 else [(a, c2, b), (a, d, c2)]
        for i, sg in ((0, 1), (ns - 1, -1)):               # leading and trailing edges
            for j in range(nu - 1):
                a, b, c2, d = idx(0, i, j), idx(0, i, j + 1), idx(1, i, j + 1), idx(1, i, j)
                F += [(a, b, c2), (a, c2, d)] if sg > 0 else [(a, c2, b), (a, d, c2)]
        Vs.append(V)
        Fs.append(np.asarray(F) + off)
        off += len(V)
    V, F = np.vstack(Vs), np.vstack(Fs)
    # orient outward: the signed volume must be positive
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    if vol < 0:
        F = F[:, ::-1]
    return V, F


def oblock_mesh_dict(route: Route, n_core: int = 8, n_radial: int = 10, wall_grading: float = 0.25,
                     axial_cell_D: float = 0.2, core: float = 0.36, convert: float = 1.0) -> str:
    """blockMeshDict of the O-grid pipe swept along ``route``: five blocks per station interval (a square core of
    half-diagonal ``core`` x radius, four curved blocks), arcs on the wall, patches inlet / outlet / wall.
    ``wall_grading``: wall cell / core-side cell in the radial direction."""
    st = route.stations()
    P, N, B, s = st["P"], st["N"], st["B"], st["s"]
    r = route.D / 2
    ang = np.radians([45, 135, 225, 315])
    verts, edges, blocks = [], [], []
    for k in range(len(P)):
        for a in ang:                                      # core corners
            verts.append(P[k] + core * r * (math.cos(a) * N[k] + math.sin(a) * B[k]))
        for a in ang:                                      # wall points
            verts.append(P[k] + r * (math.cos(a) * N[k] + math.sin(a) * B[k]))
        for i in range(4):                                 # wall arcs (through the mid angle)
            am = ang[i] + math.pi / 4
            mid = P[k] + r * (math.cos(am) * N[k] + math.sin(am) * B[k])
            edges.append(f"    arc {8 * k + 4 + i} {8 * k + 4 + (i + 1) % 4} ({mid[0]:.9g} {mid[1]:.9g} {mid[2]:.9g})")
            amc = ang[i] + math.pi / 4                     # slightly bulged core sides (better cells)
            midc = P[k] + (core * r * math.cos(math.pi / 4) * 1.12) * (math.cos(amc) * N[k] + math.sin(amc) * B[k])
            edges.append(f"    arc {8 * k + i} {8 * k + (i + 1) % 4} ({midc[0]:.9g} {midc[1]:.9g} {midc[2]:.9g})")
    for k in range(len(P) - 1):
        nx = max(1, int(round((s[k + 1] - s[k]) / (axial_cell_D * route.D))))
        a, b = 8 * k, 8 * (k + 1)
        blocks.append(f"    hex ({a} {a + 1} {a + 2} {a + 3} {b} {b + 1} {b + 2} {b + 3}) core ({n_core} {n_core} {nx}) "
                      f"simpleGrading (1 1 1)")
        for i in range(4):
            j = (i + 1) % 4
            c0, c1, o0, o1 = a + i, a + j, a + 4 + i, a + 4 + j
            d0, d1, p0, p1 = b + i, b + j, b + 4 + i, b + 4 + j
            blocks.append(f"    hex ({c1} {c0} {o0} {o1} {d1} {d0} {p0} {p1}) ring ({n_core} {n_radial} {nx}) "
                          f"simpleGrading (1 {wall_grading} 1)")
    last = 8 * (len(P) - 1)
    inlet = ["            (0 3 2 1)"] + [f"            ({(i + 1) % 4} {i} {4 + i} {4 + (i + 1) % 4})" for i in range(4)]
    outlet = [f"            ({last} {last + 1} {last + 2} {last + 3})"] + [
        f"            ({last + i} {last + (i + 1) % 4} {last + 4 + (i + 1) % 4} {last + 4 + i})" for i in range(4)]
    wall = []
    for k in range(len(P) - 1):
        a, b = 8 * k, 8 * (k + 1)
        for i in range(4):
            j = (i + 1) % 4
            wall.append(f"            ({a + 4 + i} {a + 4 + j} {b + 4 + j} {b + 4 + i})")
    vtxt = "\n".join(f"    ({v[0]:.9g} {v[1]:.9g} {v[2]:.9g})" for v in verts)
    return (HDR % ("dictionary", "blockMeshDict") + f"convertToMeters {convert};\nvertices\n(\n{vtxt}\n);\n"
            + "blocks\n(\n" + "\n".join(blocks) + "\n);\nedges\n(\n" + "\n".join(edges) + "\n);\n"
            + "boundary\n(\n"
            + "    inlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(inlet) + "\n        );\n    }\n"
            + "    outlet\n    {\n        type patch;\n        faces\n        (\n" + "\n".join(outlet) + "\n        );\n    }\n"
            + "    wall\n    {\n        type wall;\n        faces\n        (\n" + "\n".join(wall) + "\n        );\n    }\n"
            + ");\nmergePatchPairs\n(\n);\n")


# ---------------------------------------------------------------------- results
@dataclass
class InternalResult:
    route: Route
    info: dict[str, Any]
    sections: dict[str, np.ndarray]                    # s, p (area mean, kinematic), u_mean, cov (scalar), c_mean
    profiles: dict[str, np.ndarray]                    # axial velocity on cross-sections (r, u) at chosen stations
    slices: list[dict[str, Any]]                       # name, origin, u, v, grids {field: 2-D}, labels
    lines: list[np.ndarray]
    line_speed: list[np.ndarray]
    wall: dict[str, np.ndarray]                        # xyz, tau (wall shear, kinematic), p per wall face
    timings: dict[str, float]

    @property
    def U(self) -> float:
        return self.info["U"]

    @property
    def pressure_drop(self) -> float:
        """Kinematic pressure drop (p/rho, m^2/s^2) from the first to the last section."""
        return float(self.sections["p"][0] - self.sections["p"][-1])

    def gradient(self, s0_D: float, s1_D: float) -> float:
        """Least-squares -dp/ds (kinematic) between two arc lengths given in diameters."""
        s, p = self.sections["s"], self.sections["p"]
        D = self.route.D
        m = (s >= s0_D * D) & (s <= s1_D * D)
        return float(-np.polyfit(s[m], p[m], 1)[0])

    def friction_factor(self, s0_D: float, s1_D: float) -> float:
        """Darcy friction factor from the pressure gradient between two arc lengths (diameters)."""
        return self.gradient(s0_D, s1_D) * self.route.D / (0.5 * self.U ** 2)

    def wall_surface(self, nth: int = 64):
        st = self.route.stations()
        return pipe_surface(st, self.route.D / 2, nth)

    def to_scene(self, title: str | None = None, extra_surfaces: Sequence[Any] = (),
                 window_D: tuple[float, float] | None = None):
        """A studio Scene: the pipe wall (with its wall shear), the streamlines by speed and the cross-section slices.
        ``window_D``: keep only the part between two arc lengths (in diameters), e.g. a close-up of a bend."""
        from pinneapple_tools.visualization.studio.scene import Scene, Surface
        st = self.route.stations()
        D = self.route.D
        V, F = self.wall_surface()
        lines, speeds = list(self.lines), [v / self.U for v in self.line_speed]
        wall = dict(self.wall)
        slices = list(self.slices)
        if window_D is not None:
            lo, hi = window_D[0] * D, window_D[1] * D

            def keep(X):
                s_, _, _ = _project(np.asarray(X), st, np.zeros_like(np.asarray(X)))
                return (s_ >= lo) & (s_ <= hi)
            kv = keep(V)
            F = F[kv[F].all(1)]
            used = np.unique(F)
            remap = -np.ones(len(V), int)
            remap[used] = np.arange(len(used))
            V, F = V[used], remap[F]
            nl, ns = [], []
            for L, sp in zip(lines, speeds, strict=True):
                m = keep(L)
                if m.sum() > 2:
                    nl.append(L[m])
                    ns.append(sp[m])
            lines, speeds = nl, ns
            if len(wall.get("xyz", [])):
                m = keep(wall["xyz"])
                wall = {k: v[m] for k, v in wall.items()}
            slices = [x for x in slices if lo <= float(x["name"].split()[1]) * D <= hi]
        sc = Scene([Surface("pipe", V, F, "glass")] + list(extra_surfaces), axes="z_up",
                   title=title or self.info.get("title", "Internal flow"))
        if len(wall.get("xyz", [])):
            sc.map_field("tau_w", wall["xyz"], wall["tau"] / (0.5 * self.U ** 2), label="Skin friction coefficient Cf")
        if lines:
            sc.add_lines(lines, speeds, label="Speed |U| / U mean")
        for x in slices:
            for k, g in x["grids"].items():
                sc.add_slice(f"{x['name']}: {k}", np.asarray(x["origin"]), np.asarray(x["u"]), np.asarray(x["v"]),
                             np.asarray(g, float), label=x["labels"].get(k, k), group=f"{x.get('group', x['name'])}: {k}")
        return sc


def pipe_surface(st: dict[str, np.ndarray], r: float, nth: int = 64, sub: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated wall of a swept circular section (stations from ``Route.stations``), subdividing straight runs."""
    P, N, B = st["P"], st["N"], st["B"]
    rings = []
    for k in range(len(P) - 1):
        m = sub * max(1, int(np.linalg.norm(P[k + 1] - P[k]) / (2 * r)))
        for t in np.linspace(0, 1, m, endpoint=False):
            p = (1 - t) * P[k] + t * P[k + 1]
            n = (1 - t) * N[k] + t * N[k + 1]
            b = (1 - t) * B[k] + t * B[k + 1]
            rings.append((p, n / np.linalg.norm(n), b / np.linalg.norm(b)))
    rings.append((P[-1], N[-1], B[-1]))
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    V = np.array([p + r * (np.cos(a) * n + np.sin(a) * b) for p, n, b in rings for a in th])
    F = []
    for k in range(len(rings) - 1):
        for i in range(nth):
            a, b2, c, d = k * nth + i, k * nth + (i + 1) % nth, (k + 1) * nth + (i + 1) % nth, (k + 1) * nth + i
            F += [(a, b2, c), (a, c, d)]
    return V, np.asarray(F)


# ---------------------------------------------------------------------- the study
@dataclass
class InternalFlow:
    """A pipe ``route`` at Reynolds number ``Re`` (bulk velocity and diameter). ``turbulent``: None picks laminar
    below Re 2300. ``mixer``: number of Kenics elements starting at ``mixer_start_D`` (straight routes).
    ``scalar``: carry a passive scalar entering on the upper half of the inlet (c = 1 where the inlet's local
    second normal coordinate > 0). ``inlet``: "uniform" or "developed" (parabolic laminar, 1/7 power law turbulent)."""
    route: Route
    Re: float = 1000.0
    nu: float = 1e-6
    turbulent: bool | None = None
    mixer: int = 0
    mixer_start_D: float = 3.0
    scalar: bool = False
    inlet: str = "developed"
    n_core: int = 8
    n_radial: int = 10
    wall_grading: float = 0.25
    axial_cell_D: float = 0.2
    iterations: int = 1500
    diffusivity: float = 0.0
    title: str = ""
    turbulence_intensity: float = 0.05
    scalar_iterations: int = 400

    def __post_init__(self):
        if self.turbulent is None:
            self.turbulent = self.Re > 2300
        self.U = self.Re * self.nu / self.route.D

    # -- writing
    def write(self, case: str, procs: int = 2) -> None:
        if os.path.isdir(case):
            shutil.rmtree(case)
        os.makedirs(os.path.join(case, "system"))
        os.makedirs(os.path.join(case, "constant", "triSurface"))
        with open(os.path.join(case, "system", "blockMeshDict"), "w") as f:
            f.write(oblock_mesh_dict(self.route, self.n_core, self.n_radial, self.wall_grading, self.axial_cell_D))
        if self.mixer:
            V, F = kenics_elements(self.route.D, self.mixer, self.mixer_start_D)
            with open(os.path.join(case, "constant", "triSurface", "mixer.stl"), "wb") as f:
                f.write(_stl(V, F, "mixer"))
            # a fluid point next to the wall at the inlet end, away from the elements
            loc = (0.5 * self.mixer_start_D * self.route.D, 0.0, 0.42 * self.route.D)
            _w(case, "system/snappyHexMeshDict", "dictionary", f"""castellatedMesh true;
snap true;
addLayers false;
geometry
{{
    mixer.stl {{ type triSurfaceMesh; name mixer; }}
}}
castellatedMeshControls
{{
    maxLocalCells 5000000;
    maxGlobalCells 20000000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;
    features ( {{ file "mixer.eMesh"; level 2; }} );
    refinementSurfaces {{ mixer {{ level (1 2); patchInfo {{ type wall; }} }} }}
    resolveFeatureAngle 30;
    refinementRegions {{ }}
    locationInMesh ({loc[0]:.9g} {loc[1]:.9g} {loc[2]:.9g});
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
    relativeSizes true;
    layers {{ }}
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-30;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}
mergeTolerance 1e-6;
""")
            _w(case, "system/surfaceFeatureExtractDict", "dictionary", """mixer.stl
{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj no;
}
""")
        self._write_fields(case, procs)

    def _write_fields(self, case: str, procs: int) -> None:
        U, nu = self.U, self.nu
        k = 1.5 * (self.turbulence_intensity * U) ** 2
        omega = math.sqrt(k) / (0.09 ** 0.25 * 0.07 * self.route.D)
        mixer = "    mixer\n    {\n%s    }\n" if self.mixer else ""

        def bf(inlet, outlet, wall):
            s = f"boundaryField\n{{\n    inlet\n    {{\n{inlet}    }}\n    outlet\n    {{\n{outlet}    }}\n    wall\n    {{\n{wall}    }}\n"
            if mixer:
                s += mixer % wall
            return s + "    \"proc.*\"\n    {\n        type processor;\n        value $internalField;\n    }\n}\n"
        t0 = self.route.stations()["T"][0]
        Uin = t0 * U
        _w(case, "0/U", "volVectorField", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({Uin[0]} {Uin[1]} {Uin[2]});\n"
           + bf(f"        type fixedValue;\n        value uniform ({Uin[0]} {Uin[1]} {Uin[2]});\n",
                "        type inletOutlet;\n        inletValue uniform (0 0 0);\n        value uniform (0 0 0);\n",
                "        type noSlip;\n"))
        _w(case, "0/p", "volScalarField", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n" + bf(
            "        type zeroGradient;\n", "        type fixedValue;\n        value uniform 0;\n", "        type zeroGradient;\n"))
        if self.turbulent:
            io = lambda v: f"        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"  # noqa: E731
            _w(case, "0/k", "volScalarField", f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k:.6g};\n" + bf(
                f"        type fixedValue;\n        value uniform {k:.6g};\n", io(f"{k:.6g}"),
                f"        type kqRWallFunction;\n        value uniform {k:.6g};\n"))
            _w(case, "0/omega", "volScalarField", f"dimensions [0 0 -1 0 0 0 0];\ninternalField uniform {omega:.6g};\n" + bf(
                f"        type fixedValue;\n        value uniform {omega:.6g};\n", io(f"{omega:.6g}"),
                f"        type omegaWallFunction;\n        value uniform {omega:.6g};\n"))
            _w(case, "0/nut", "volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n" + bf(
                "        type calculated;\n        value uniform 0;\n", "        type calculated;\n        value uniform 0;\n",
                "        type nutkWallFunction;\n        value uniform 0;\n"))
            turb = "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n"
        else:
            turb = "simulationType laminar;\n"
        if self.scalar:
            _w(case, "0/T", "volScalarField", "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\n" + bf(
                "        type fixedValue;\n        value uniform 0;\n",
                "        type inletOutlet;\n        inletValue uniform 0;\n        value uniform 0;\n",
                "        type zeroGradient;\n"))
        _w(case, "constant/transportProperties", "dictionary",
           f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};\nDT [0 2 -1 0 0 0 0] {self.diffusivity};\n")
        _w(case, "constant/turbulenceProperties", "dictionary", turb)
        self._control(case, "simpleFoam", 0, self.iterations)
        _w(case, "system/fvSchemes", "dictionary", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; limited cellLimited Gauss linear 1; grad(U) $limited; grad(k) $limited; grad(omega) $limited; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwindV grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div(phi,T) bounded Gauss limitedLinear01 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
""")
        _w(case, "system/fvSolution", "dictionary", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.01; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.05; }
    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0.01; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-5; }
}
relaxationFactors
{
    equations { U 0.9; k 0.7; omega 0.7; T 1; }
    fields { p 1; }
}
""")
        n = {1: (1, 1, 1), 2: (2, 1, 1), 3: (3, 1, 1), 4: (4, 1, 1)}.get(procs, (procs, 1, 1))
        _w(case, "system/decomposeParDict", "dictionary", f"numberOfSubdomains {procs};\nmethod hierarchical;\n"
           f"hierarchicalCoeffs {{ n ({n[0]} {n[1]} {n[2]}); order xyz; }}\n")

    def _control(self, case: str, app: str, start: int, end: int) -> None:
        _w(case, "system/controlDict", "dictionary", f"""application {app};
startFrom latestTime;
startTime {start};
stopAt endTime;
endTime {end};
deltaT 1;
writeControl timeStep;
writeInterval {end};
purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
""")

    # -- inlet profiles (after the mesh exists)
    def _inlet_values(self, case: str) -> None:
        from pinneapple_data.cae.foam import read_polymesh
        from pinneapple_data.cae.geometry import face_geometry, face_mesh
        fs = {f"constant/polyMesh/{f}": open(os.path.join(case, "constant", "polyMesh", f), "rb").read()
              for f in ("points", "faces", "owner", "neighbour", "boundary")}
        mesh = read_polymesh(fs)
        fg = face_geometry(face_mesh(mesh))
        pat = next(p for p in mesh.poly.patches if p["name"] == "inlet")
        idx = np.arange(int(pat["startFace"]), int(pat["startFace"]) + int(pat["nFaces"]))
        C = fg["centres"][idx]
        st = self.route.stations()
        rel = C - st["P"][0]
        r = np.linalg.norm(rel - (rel @ st["T"][0])[:, None] * st["T"][0], axis=1) / (self.route.D / 2)
        r = np.clip(r, 0, 1)
        if self.inlet == "developed" and not self.turbulent:
            u = 2 * self.U * (1 - r ** 2)
        elif self.inlet == "developed":
            u = self.U * (60 / 49) * (1 - r) ** (1 / 7)
        else:
            u = np.full(len(r), self.U)
        Uv = u[:, None] * st["T"][0][None, :]
        rows = "\n".join(f"({a:.8g} {b:.8g} {c:.8g})" for a, b, c in Uv)
        body = f"nonuniform List<vector> {len(Uv)}\n(\n{rows}\n)"
        _replace_patch_value(case, "U", "inlet", f"        type fixedValue;\n        value {body};\n")
        if self.scalar:
            yb = rel @ st["B"][0]
            c = (yb > 0).astype(float)
            rows = "\n".join(f"{v:g}" for v in c)
            body = f"nonuniform List<scalar> {len(c)}\n(\n{rows}\n)"
            _replace_patch_value(case, "T", "inlet", f"        type fixedValue;\n        value {body};\n")

    # -- running
    def run(self, case: str, procs: int = 2, log: Callable[[str], None] = print) -> dict[str, float]:
        t: dict[str, float] = {}

        def sh(cmd, name):
            t0 = time.time()
            full = f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} > log.{name} 2>&1'
            r = subprocess.run(["bash", "-c", full])
            t[name] = round(time.time() - t0, 1)
            log(f"  {name}: {t[name]} s (exit {r.returncode})")
            if r.returncode:
                raise RuntimeError(f"{name} failed, see {case}/log.{name}")
        mpi = f"mpirun --allow-run-as-root --oversubscribe -np {procs}"
        sh("blockMesh", "blockMesh")
        if self.mixer:
            sh("surfaceFeatureExtract", "surfaceFeatureExtract")
            sh("snappyHexMesh -overwrite", "snappyHexMesh")
        sh("checkMesh", "checkMesh")
        self._inlet_values(case)
        if procs > 1:
            sh("decomposePar -force", "decomposePar")
            sh(f"{mpi} simpleFoam -parallel", "simpleFoam")
            sh("reconstructPar -latestTime", "reconstruct")
        else:
            sh("simpleFoam", "simpleFoam")
        if self.scalar:
            from .external_flow import latest_time
            t_end = int(float(latest_time(case)))
            # remove the processor dirs, carry T over to the converged time, solve the transport (linear: a few sweeps)
            sh("rm -rf processor*", "clean")
            shutil.copy(os.path.join(case, "0", "T"), os.path.join(case, str(t_end), "T"))
            self._control(case, "scalarTransportFoam", t_end, t_end + self.scalar_iterations)
            sh("scalarTransportFoam", "scalarTransportFoam")
            # keep one solution time: move the transported scalar back into the flow solution
            later = sorted((float(d), d) for d in os.listdir(case) if _isnum(d) and float(d) > t_end)
            if later:
                shutil.copy(os.path.join(case, later[-1][1], "T"), os.path.join(case, str(t_end), "T"))
                for _, d in later:
                    shutil.rmtree(os.path.join(case, d), ignore_errors=True)
        return t

    def solve(self, case: str, procs: int = 2, log: Callable[[str], None] = print, n_sections: int | None = None,
              n_lines: int = 40, slice_at_D: Sequence[float] = ()) -> InternalResult:
        self.write(case, procs)
        t = self.run(case, procs, log)
        res = self.read(case, n_sections=n_sections, n_lines=n_lines, slice_at_D=slice_at_D)
        res.timings = t
        return res

    # -- reading
    def read(self, case: str, n_sections: int | None = None, n_lines: int = 40, slice_at_D: Sequence[float] = ()) -> InternalResult:
        from pinneapple_data.cae.foam import read_field

        from .external_flow import CaseFields, latest_time
        cf = CaseFields(case)
        tdir = latest_time(case)
        Tc = None
        if self.scalar:
            # scalarTransportFoam writes T to the last time; U lives in the simpleFoam time
            ts = sorted((float(d), d) for d in os.listdir(case) if _isnum(d) and os.path.exists(os.path.join(case, d, "T")))
            if ts:
                Tc = read_field(open(os.path.join(case, ts[-1][1], "T"), "rb").read(), cf.n_cells)[0].ravel()
        st = self.route.stations()
        C, vol = cf.cg["centres"], np.abs(cf.cg["volumes"])
        # project every cell on the centreline: nearest station interval, then arc length
        s_cell, rad_cell, uax = _project(C, st, cf.U)
        D, r = self.route.D, self.route.D / 2
        if n_sections is None:                      # sections at least two axial cells long: none is empty
            n_sections = int(min(200, max(10, st["s"][-1] / (2 * self.axial_cell_D * D))))
        edges = np.linspace(0, st["s"][-1], n_sections + 1)
        sid = np.clip(np.searchsorted(edges, s_cell) - 1, 0, n_sections - 1)
        cnt = np.bincount(sid, vol, n_sections)
        keep = cnt > 0
        cnt = np.maximum(cnt, 1e-300)
        p = np.bincount(sid, vol * cf.p, n_sections) / cnt
        um = np.bincount(sid, vol * uax, n_sections) / cnt
        sm = np.bincount(sid, vol * s_cell, n_sections) / cnt
        sec = {"s": sm, "p": p, "u_mean": um}
        if Tc is not None:
            w = vol * np.maximum(uax, 0)
            W = np.bincount(sid, w, n_sections) + 1e-30
            cm = np.bincount(sid, w * Tc, n_sections) / W
            var = np.bincount(sid, w * (Tc - cm[sid]) ** 2, n_sections) / W
            sec["c_mean"] = cm
            sec["cov"] = np.sqrt(var) / np.maximum(cm, 1e-12)
        sec = {k: v[keep] for k, v in sec.items()}
        # wall shear on the wall patch
        wall = {}
        for pat in cf.patches():
            if pat["name"] == "wall":
                q = 0.5 * self.U ** 2
                wv = cf.wall(pat, self.nu, q)
                tau = np.linalg.norm(wv["f_viscous"], axis=1) / (np.linalg.norm(cf.fg["areas"][int(pat["startFace"]):
                                                                                          int(pat["startFace"]) + int(pat["nFaces"])], axis=1) + 1e-30)
                if not self.turbulent:
                    # laminar: wall shear from the first cell, nu u_t / d
                    tau = wv["cf"] * q
                    s0, n0 = int(pat["startFace"]), int(pat["nFaces"])
                    idx = np.arange(s0, s0 + n0)
                    own = cf.mesh.poly.owner[idx]
                    sf = cf.fg["areas"][idx]
                    nh = sf / np.linalg.norm(sf, axis=1)[:, None]
                    up = cf.U[own]
                    ut = np.linalg.norm(up - (up * nh).sum(1, keepdims=True) * nh, axis=1)
                    dd = np.abs(((cf.cg["centres"][own] - cf.fg["centres"][idx]) * nh).sum(1)) + 1e-12
                    tau = self.nu * ut / dd
                wall = {"xyz": wv["xyz"], "tau": tau, "p": wv["cp"] * q}
        # radial profiles of the axial velocity at a few stations (cells within half a section)
        prof = {}
        for frac in (0.25, 0.5, 0.95):
            s0 = frac * st["s"][-1]
            m = np.abs(s_cell - s0) < 0.6 * D
            prof[f"{frac:g}"] = np.stack([rad_cell[m] / r, uax[m] / self.U])
        # streamlines from the inlet
        vel = cf.velocity()
        rng = np.random.default_rng(0)
        rr = 0.85 * r * np.sqrt(rng.random(n_lines))
        th = 2 * np.pi * rng.random(n_lines)
        seeds = st["P"][0] + 0.3 * D * st["T"][0] + (rr * np.cos(th))[:, None] * st["N"][0] + (rr * np.sin(th))[:, None] * st["B"][0]
        lines, speeds = _trace(seeds, vel, h=D / 12, n=int(1.1 * st["s"][-1] / (D / 12)), stop=lambda X: _past_end(X, st))
        # cross-section slices
        slices = []
        for sD in slice_at_D:
            k = np.searchsorted(st["s"], sD * D)
            k = int(np.clip(k, 1, len(st["s"]) - 1))
            t = (sD * D - st["s"][k - 1]) / max(st["s"][k] - st["s"][k - 1], 1e-12)
            Pc = (1 - t) * st["P"][k - 1] + t * st["P"][k]
            Tt = (1 - t) * st["T"][k - 1] + t * st["T"][k]
            Tt /= np.linalg.norm(Tt)
            Nn = (1 - t) * st["N"][k - 1] + t * st["N"][k]
            Nn -= (Nn @ Tt) * Tt
            Nn /= np.linalg.norm(Nn)
            Bb = np.cross(Tt, Nn)
            g = 72
            a = np.linspace(-r, r, g)
            Y, Z = np.meshgrid(a, a)
            pts = Pc + Y.ravel()[:, None] * Nn + Z.ravel()[:, None] * Bb
            Uv, pv, far = cf.sample(pts)
            inside = (np.hypot(Y, Z).ravel() < 0.995 * r) & (far < 3.0)     # cells are long axially
            grids = {"axial speed": np.where(inside, (Uv @ Tt) / self.U, np.nan).reshape(g, g)}
            sec_v = Uv - (Uv @ Tt)[:, None] * Tt
            grids["secondary speed"] = np.where(inside, np.linalg.norm(sec_v, axis=1) / self.U, np.nan).reshape(g, g)
            labels = {"axial speed": "Axial speed / U mean", "secondary speed": "Secondary speed / U mean"}
            if Tc is not None:
                tree, _ = cf._tree("all")
                dd, ii = tree.query(pts, k=4)
                w = 1 / (dd + 1e-9) ** 2
                cv = (Tc[ii] * w).sum(1) / w.sum(1)
                grids["concentration"] = np.where(inside, cv, np.nan).reshape(g, g)
                labels["concentration"] = "Scalar concentration"
            slices.append({"name": f"section {sD:g} D", "origin": (Pc - r * Nn - r * Bb).tolist(),
                           "u": (2 * r * Nn).tolist(), "v": (2 * r * Bb).tolist(), "grids": grids, "labels": labels,
                           "group": "sections"})
        info = {"U": self.U, "Re": self.Re, "nu": self.nu, "D": D, "turbulent": bool(self.turbulent),
                "cells": cf.n_cells, "length": st["s"][-1], "mixer": self.mixer, "time": tdir,
                "title": self.title or f"Pipe flow, Re {self.Re:g}"}
        return InternalResult(self.route, info, sec, prof, slices, lines, speeds, wall, {})


def _isnum(d: str) -> bool:
    try:
        float(d)
        return True
    except ValueError:
        return False


def _replace_patch_value(case: str, fld: str, patch: str, body: str) -> None:
    """Replace the entry of one patch in 0/<fld>."""
    import re
    path = os.path.join(case, "0", fld)
    s = open(path).read()
    pat = re.compile(rf"(\n    {re.escape(patch)}\n    \{{\n)(.*?)(\n?    \}}\n)", re.S)
    s2, n = pat.subn(lambda m: m.group(1) + body + "    }\n", s, count=1)
    if not n:
        raise ValueError(f"patch {patch} not found in {path}")
    with open(path, "w") as f:
        f.write(s2)


def _project(C: np.ndarray, st: dict[str, np.ndarray], U: np.ndarray):
    """Arc length, distance to the centreline and axial velocity of points, by the nearest centreline interval."""
    P, T, s = st["P"], st["T"], st["s"]
    best_d = np.full(len(C), np.inf)
    s_out = np.zeros(len(C))
    rad = np.zeros(len(C))
    ax = np.zeros(len(C))
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        ab = b - a
        L2 = ab @ ab
        t = np.clip(((C - a) @ ab) / L2, 0, 1)
        Q = a + t[:, None] * ab
        d = np.linalg.norm(C - Q, axis=1)
        m = d < best_d
        best_d[m] = d[m]
        s_out[m] = s[k] + t[m] * (s[k + 1] - s[k])
        tk = (1 - t[m])[:, None] * T[k] + t[m][:, None] * T[k + 1]
        tk /= np.linalg.norm(tk, axis=1, keepdims=True)
        ax[m] = (U[m] * tk).sum(1)
        rad[m] = d[m]
    return s_out, rad, ax


def _past_end(X: np.ndarray, st: dict[str, np.ndarray]) -> np.ndarray:
    return ((X - st["P"][-1]) @ st["T"][-1]) > 0


def _trace(seeds: np.ndarray, vel, h: float, n: int, stop) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X = np.asarray(seeds, float).copy()
    path, sp = [X.copy()], [np.linalg.norm(vel(X), axis=1)]
    done = np.zeros(len(X), bool)
    for _ in range(n):
        v1 = vel(X)
        Xm = X + 0.5 * h * v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
        v2 = vel(Xm)
        Xn = X + h * v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12)
        X = np.where(done[:, None], X, Xn)
        done |= stop(X)
        path.append(X.copy())
        sp.append(np.linalg.norm(vel(X), axis=1))
        if done.all():
            break
    Pth, S = np.stack(path, 1), np.stack(sp, 1)
    lines, speeds = [], []
    for i in range(len(X)):
        moved = np.r_[True, np.linalg.norm(np.diff(Pth[i], axis=0), axis=1) > 1e-12]
        lines.append(Pth[i][moved][::2])
        speeds.append(S[i][moved][::2])
    return lines, speeds


def ito_bend_loss(Re: float, radius_ratio: float, angle_deg: float = 90.0) -> float:
    """Total loss coefficient of a smooth pipe bend in turbulent flow (Ito 1960, J. Basic Eng. 82: 131-143), with
    ``radius_ratio`` = bend radius / pipe radius, valid for Re (r/R)^2 > 91; includes friction over the bend and the
    excess loss recovered downstream."""
    a = 0.95 + 17.2 * radius_ratio ** -1.96 if radius_ratio < 19.7 else 1.0
    return 0.00241 * a * angle_deg * Re ** -0.17 * radius_ratio ** 0.84


def colebrook(Re: float, rel_roughness: float = 0.0) -> float:
    """Darcy friction factor of fully developed turbulent pipe flow (Colebrook-White), solved by fixed point."""
    f = 0.02
    for _ in range(60):
        f = (-2 * math.log10(rel_roughness / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
    return f
