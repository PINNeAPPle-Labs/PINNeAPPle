"""Globe valve throttling of natural gas: the geometry and an OpenFOAM real-gas CFD, driven by PINNeAPPle (#1467).

The throttling path of a globe valve, simplified to an **axisymmetric plug and seat** around a stem: the gas
leaves the inlet pipe through the seat bore, turns outward through the gap between the seat face and the plug
(the valve lift), passes around the plug rim and mixes out in the outlet pipe. The real valve body (the S-shaped
passage of a globe valve) is not modelled: its extra pressure loss is upstream or downstream of the throttling and
does not change the throttled outlet temperature, which an adiabatic passage fixes through its stagnation enthalpy.

* :class:`ValveGeometry` and :func:`mesh_valve`: the parametric passage, meshed with gmsh as a 5° wedge (one cell
  across, hexahedra), refined at the seat and the plug; :func:`geometry_figure` draws it.
* :func:`write_case`: an OpenFOAM case (``rhoPimpleFoam`` with local time stepping, run to steady state,
  transonic; ``heRhoThermo`` with ``sensibleEnthalpy``; the equation of state ``PengRobinsonGas`` or
  ``perfectGas``; k-omega SST). The mixture is one pseudo-pure fluid: its critical constants and acentric factor
  by Kay's rule and its ideal-gas Cp at the inlet from CoolProp (GERG-2008), for the given composition. Inlet: the
  pressure and temperature of the upstream line; outlet: the static pressure; adiabatic walls.
* :func:`run_valve_cfd`: meshes, writes, runs OpenFOAM, and reads the mass-flow-weighted inlet and outlet
  temperature and enthalpy, the mass flow, the residuals and the fields.
* :func:`valve_cfd_study`: the CFD with the ideal gas and with Peng–Robinson, compared with the library's
  throttling (:func:`.joule_thomson.throttle`, GERG-2008 and Peng–Robinson) at the CFD's mass flow, with checks, a
  VeriPhysics decision and figures.

Needs OpenFOAM (the Ubuntu ``openfoam`` package, v1912: ``apt install openfoam``; or ``FOAM_BASHRC`` pointing at
another install's ``etc/bashrc``), gmsh and CoolProp.
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["ValveGeometry", "foam_bashrc", "openfoam_available", "mesh_valve", "geometry_figure", "mixture_constants",
           "gas_enthalpy", "model_throttle", "write_case", "run_valve_cfd", "valve_cfd_study"]

R_UNIVERSAL = 8314.462618                 # J/(kmol K)
_BASHRCS = ("/usr/share/openfoam/etc/bashrc", "/usr/lib/openfoam/openfoam*/etc/bashrc", "/opt/openfoam*/etc/bashrc",
            "/opt/OpenFOAM-*/etc/bashrc")


# ------------------------------------------------------------------------------------------------- OpenFOAM
def foam_bashrc() -> str | None:
    """The OpenFOAM ``etc/bashrc`` to source: ``FOAM_BASHRC``, else the usual install places; None if none."""
    env = os.environ.get("FOAM_BASHRC")
    if env and os.path.exists(env):
        return env
    for pattern in _BASHRCS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def openfoam_available() -> bool:
    """Whether an OpenFOAM install with ``rhoPimpleFoam`` and ``gmshToFoam`` can be sourced."""
    b = foam_bashrc()
    if not b:
        return False
    r = subprocess.run(["bash", "-c", f'source "{b}" >/dev/null 2>&1; command -v rhoPimpleFoam && command -v gmshToFoam'],
                       capture_output=True, text=True)
    return r.returncode == 0


def _foam(case: str, cmd: str, log: str, timeout: float) -> int:
    b = foam_bashrc()
    if not b:
        raise RuntimeError("OpenFOAM not found: install it (Ubuntu: apt install openfoam) or set FOAM_BASHRC")
    full = f'source "{b}" >/dev/null 2>&1; cd "{case}" && {cmd} > {log} 2>&1'
    return subprocess.run(["bash", "-c", full], timeout=timeout).returncode


# ------------------------------------------------------------------------------------------------- geometry
@dataclass
class ValveGeometry:
    """An axisymmetric plug-and-seat passage, lengths in mm (x along the axis, r from it).

    The inlet pipe (``pipe_d``, around the stem ``stem_d``) ends at the seat: a ring of length ``seat_len`` with
    the bore ``seat_d``. The plug, a disc of diameter ``plug_d`` and length ``plug_len`` on the stem, sits
    ``lift`` downstream of the seat; the gas turns outward between them and passes around the plug rim into the
    outlet pipe."""
    pipe_d: float = 100.0
    stem_d: float = 16.0
    seat_d: float = 40.0
    seat_len: float = 10.0
    plug_d: float = 60.0
    plug_len: float = 15.0
    lift: float = 4.0
    upstream: float = 150.0
    downstream: float = 500.0
    wedge_deg: float = 5.0

    def check(self) -> None:
        if not (self.stem_d < self.seat_d < self.plug_d < self.pipe_d):
            raise ValueError("needs stem_d < seat_d < plug_d < pipe_d")
        if min(self.lift, self.seat_len, self.plug_len, self.upstream, self.downstream) <= 0:
            raise ValueError("lengths must be positive")

    def profile(self) -> list[tuple[float, float]]:
        """The fluid region's outline in (x, r) [m], counter-clockwise from the inlet at the stem."""
        k = 1e-3
        R, Rst, Rs, Rp = self.pipe_d / 2 * k, self.stem_d / 2 * k, self.seat_d / 2 * k, self.plug_d / 2 * k
        ts, h, tp = self.seat_len * k, self.lift * k, self.plug_len * k
        xe = ts + h + tp + self.downstream * k
        return [(-self.upstream * k, Rst), (-self.upstream * k, R), (0.0, R), (0.0, Rs), (ts, Rs), (ts, R), (xe, R),
                (xe, Rst), (ts + h + tp, Rst), (ts + h + tp, Rp), (ts + h, Rp), (ts + h, Rst)]

    def areas_m2(self) -> dict[str, float]:
        k = 1e-3
        R, Rst, Rs = self.pipe_d / 2 * k, self.stem_d / 2 * k, self.seat_d / 2 * k
        return {"pipe": math.pi * (R ** 2 - Rst ** 2), "seat_bore": math.pi * (Rs ** 2 - Rst ** 2),
                "curtain": 2 * math.pi * Rs * self.lift * k}


def mesh_valve(geom: ValveGeometry, out_dir: str, h_fine_mm: float = 0.4, h_coarse_mm: float = 3.0) -> dict[str, Any]:
    """Mesh the passage with gmsh as a wedge of ``geom.wedge_deg`` (one hexahedral layer): ``valve.msh`` (format
    2.2, for ``gmshToFoam``) and ``valve.stl`` (the revolved surface, for viewing). Patches: inlet, outlet, walls,
    front, back."""
    import gmsh

    geom.check()
    os.makedirs(out_dir, exist_ok=True)
    pts = geom.profile()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("valve")
        occ = gmsh.model.geo
        P = [occ.addPoint(x, r, 0.0) for x, r in pts]
        L = [occ.addLine(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
        loop = occ.addCurveLoop(L)
        S = occ.addPlaneSurface([loop])
        occ.rotate([(2, S)], 0, 0, 0, 1, 0, 0, -math.radians(geom.wedge_deg) / 2)
        occ.synchronize()
        # refinement at the seat corners, the gap and the plug rim
        hf, hc = h_fine_mm * 1e-3, h_coarse_mm * 1e-3
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [L[2], L[3], L[4], L[8], L[9], L[10]])
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", hf)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", hc)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 4 * hf)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 40 * hf + 10 * hc)
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)                  # frontal-Delaunay for quads
        out = occ.revolve([(2, S)], 0, 0, 0, 1, 0, 0, math.radians(geom.wedge_deg), numElements=[1],
                          recombine=True)
        occ.synchronize()
        top = out[0][1]
        vol = out[1][1]
        lateral = [e[1] for e in out[2:]]                    # one per profile line, in order
        inlet, outlet = [lateral[0]], [lateral[6]]
        walls = [s for i, s in enumerate(lateral) if i not in (0, 6)]
        for dim, tags, name in ((2, inlet, "inlet"), (2, outlet, "outlet"), (2, walls, "walls"), (2, [S], "back"),
                                (2, [top], "front"), (3, [vol], "fluid")):
            gmsh.model.addPhysicalGroup(dim, tags, name=name)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        msh = os.path.join(out_dir, "valve.msh")
        gmsh.write(msh)
        n_cells = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
    finally:
        gmsh.finalize()
    stl = os.path.join(out_dir, "valve.stl")
    _write_revolved_stl(pts, stl)
    return {"msh": msh, "stl": stl, "cells": int(n_cells), "h_fine_mm": h_fine_mm, "h_coarse_mm": h_coarse_mm}


def _write_revolved_stl(pts: list[tuple[float, float]], path: str, n: int = 72) -> None:
    """The walls of the passage revolved through 360° (the stem, seat, plug and pipe surfaces), ASCII STL."""
    th = np.linspace(0, 2 * np.pi, n + 1)
    lines = ["solid valve"]
    m = len(pts)
    for i in range(m):
        (x0, r0), (x1, r1) = pts[i], pts[(i + 1) % m]
        if i in (0, 6):                                     # inlet and outlet are open
            continue
        for j in range(n):
            a = [x0, r0 * math.cos(th[j]), r0 * math.sin(th[j])]
            b = [x1, r1 * math.cos(th[j]), r1 * math.sin(th[j])]
            c = [x1, r1 * math.cos(th[j + 1]), r1 * math.sin(th[j + 1])]
            d = [x0, r0 * math.cos(th[j + 1]), r0 * math.sin(th[j + 1])]
            for tri in ((a, b, c), (a, c, d)):
                lines.append("facet normal 0 0 0\n outer loop")
                lines += [f"  vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}" for p in tri]
                lines.append(" endloop\nendfacet")
    lines.append("endsolid valve")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def geometry_figure(geom: ValveGeometry, path: str, lang: str = "pt", width_px: int = 1280) -> str:
    """The passage drawn twice: the axisymmetric section with its dimensions, and the revolved 3-D cut view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    pt = lang == "pt"
    pts = np.array(geom.profile()) * 1e3
    W = width_px / 100
    fig = plt.figure(figsize=(W * 0.62, W * 0.36))
    ax = fig.add_subplot(1, 2, 1)
    xs = pts[:, 0]
    x_end = xs.max()
    for sgn in (1, -1):
        ax.add_patch(Polygon(np.c_[pts[:, 0], sgn * pts[:, 1]], closed=True, fc="#9be7d8", ec="#0f766e", lw=1.2))
    R, Rst = geom.pipe_d / 2, geom.stem_d / 2
    for sgn in (1, -1):                                        # solid parts: pipe wall, seat ring, plug, stem
        ax.fill_between([xs.min(), x_end], sgn * R, sgn * (R + 6), color="#64748b")
        ax.fill_between([0, geom.seat_len], sgn * geom.seat_d / 2, sgn * R, color="#475569")
        x0 = geom.seat_len + geom.lift
        ax.fill_between([x0, x0 + geom.plug_len], sgn * Rst, sgn * geom.plug_d / 2, color="#b45309")
    ax.fill_between([xs.min(), x_end], -Rst, Rst, color="#94a3b8")
    ax.annotate("", xy=(-geom.upstream * 0.6, 0), xytext=(-geom.upstream, 0),
                arrowprops=dict(arrowstyle="->", color="#0f172a"))
    ax.set_xlim(xs.min() - 5, min(x_end, 140) + 5)
    ax.set_ylim(-R - 10, R + 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Seção axissimétrica (assento, folga e obturador)" if pt else "Axisymmetric section (seat, gap, plug)",
                 fontsize=10)
    ax.text(geom.seat_len + geom.lift / 2, geom.plug_d / 2 + 6,
            f"{'folga' if pt else 'lift'} {geom.lift:g} mm", ha="center", fontsize=8)
    ax.text(geom.seat_len / 2, -geom.seat_d / 2 + 4, f"Ø{geom.seat_d:g}", ha="center", fontsize=8)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    th = np.linspace(0.25 * np.pi, 2 * np.pi, 60)              # a quarter cut away
    m = len(pts)
    for i in range(m):
        if i in (0, 6):
            continue
        (x0, r0), (x1, r1) = pts[i], pts[(i + 1) % m]
        X = np.array([[x0] * len(th), [x1] * len(th)])
        Y = np.array([r0 * np.cos(th), r1 * np.cos(th)])
        Z = np.array([r0 * np.sin(th), r1 * np.sin(th)])
        col = "#b45309" if i in (9, 10) else ("#475569" if i in (3, 4) else "#94a3b8")
        ax3.plot_surface(X, Y, Z, color=col, alpha=0.85 if col != "#94a3b8" else 0.25, linewidth=0)
    ax3.set_xlim(-geom.upstream * 0.5, 120)
    ax3.set_box_aspect((1.6, 1, 1))
    ax3.set_axis_off()
    ax3.set_title("Vista 3-D em corte" if pt else "3-D cut view", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------- the gas
def mixture_constants(gas: dict[str, float], T_K: float) -> dict[str, float]:
    """One pseudo-pure fluid for OpenFOAM: molar mass, the critical constants and acentric factor by Kay's rule
    (mole-fraction averages), and the ideal-gas Cp at ``T_K`` (GERG-2008, CoolProp). Vc in m³/kmol."""
    import CoolProp.CoolProp as CP

    from .joule_thomson import _state

    keys = ("Tc", "Pc", "Vc", "omega")
    out = dict.fromkeys(keys, 0.0)
    for name, y in gas.items():
        out["Tc"] += y * CP.PropsSI("Tcrit", name)
        out["Pc"] += y * CP.PropsSI("pcrit", name)
        out["Vc"] += y * 1000.0 / CP.PropsSI("rhomolar_critical", name)
        out["omega"] += y * CP.PropsSI("acentric", name)
    a = _state("HEOS", gas)
    a.update(CP.PT_INPUTS, 101325.0, T_K)
    out["W"] = a.molar_mass() * 1000.0
    out["Cp0"] = a.cp0mass()
    out["Zc"] = out["Pc"] * out["Vc"] / (R_UNIVERSAL * out["Tc"])
    return out


# ------------------------------------------------------------------------------------------------- the case
_HEADER = ("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n    object      {obj};\n}}\n")


def _w(case: str, rel: str, cls: str, body: str) -> None:
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(_HEADER.format(cls=cls, obj=os.path.basename(rel)) + body)


def _field(case, name, cls, dim, internal, patches):
    b = "".join(f"    {k}\n    {{\n{v}    }}\n" for k, v in patches.items())
    _w(case, f"0/{name}", cls, f"dimensions {dim};\ninternalField uniform {internal};\nboundaryField\n{{\n{b}"
                                "    front { type wedge; }\n    back { type wedge; }\n}\n")


def write_case(case: str, gas: dict[str, float], P1_Pa: float, T1_K: float, P2_Pa: float, eos: str = "PengRobinson",
               iterations: int = 3000, mu_sutherland: tuple[float, float] = (1.072e-6, 198.0), n_writes: int = 15,
               max_dt: float = 2e-5) -> dict[str, float]:
    """Write the OpenFOAM dictionaries (``0/``, ``constant/``, ``system/``) of a case whose mesh comes from
    :func:`mesh_valve` (run ``gmshToFoam`` first). ``eos``: "PengRobinson" or "ideal". Returns the gas constants
    used. The viscosity is Sutherland's law with methane's constants by default (it sets the boundary layers, not
    the throttled temperature). Fields are written ``n_writes`` times (the convergence history); ``max_dt`` [s]
    caps the local time step (2e-5 s converged the default case in 4000 iterations; 5e-6 did not).

    The Debian/Ubuntu OpenFOAM v1912 build fails on any function object (an IOstream "sha1" error), so the case has
    none: everything is read back from the written fields by :func:`run_valve_cfd`."""
    mc = mixture_constants(gas, T1_K)
    interval = max(1, math.ceil(iterations / n_writes))
    iterations = interval * n_writes                         # the last write is the last iteration
    mc["iterations"] = iterations
    if eos.lower().startswith("peng"):
        eq = (f"    equationOfState\n    {{\n        Tc {mc['Tc']:.6g};\n        Vc {mc['Vc']:.6g};\n"
              f"        Zc {mc['Zc']:.6g};\n        Pc {mc['Pc']:.6g};\n        omega {mc['omega']:.6g};\n    }}\n")
        eos_word = "PengRobinsonGas"
    elif eos.lower().startswith("ideal") or eos.lower().startswith("perfect"):
        eq, eos_word = "", "perfectGas"
    else:
        raise ValueError("eos: 'PengRobinson' or 'ideal'")
    As, Ts = mu_sutherland
    _w(case, "constant/thermophysicalProperties", "dictionary",
       "thermoType\n{\n    type heRhoThermo;\n    mixture pureMixture;\n    transport sutherland;\n    thermo hConst;\n"
       f"    equationOfState {eos_word};\n    specie specie;\n    energy sensibleEnthalpy;\n}}\n"
       f"mixture\n{{\n    specie\n    {{\n        molWeight {mc['W']:.6g};\n    }}\n{eq}"
       f"    thermodynamics\n    {{\n        Cp {mc['Cp0']:.6g};\n        Hf 0;\n    }}\n"
       f"    transport\n    {{\n        As {As:.6g};\n        Ts {Ts:.6g};\n    }}\n}}\n")
    _w(case, "constant/turbulenceProperties", "dictionary",
       "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n")
    _w(case, "constant/g", "uniformDimensionedVectorField", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n")
    out_patch = "        type inletOutlet;\n        inletValue uniform {v};\n        value uniform {v};\n"
    wall_zg = "        type zeroGradient;\n"
    _field(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", f"{P2_Pa:.8g}",
           {"inlet": f"        type totalPressure;\n        p0 uniform {P1_Pa:.8g};\n        value uniform {P1_Pa:.8g};\n",
            "outlet": f"        type fixedValue;\n        value uniform {P2_Pa:.8g};\n", "walls": wall_zg})
    _field(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", f"{T1_K:.6g}",
           {"inlet": f"        type fixedValue;\n        value uniform {T1_K:.6g};\n",
            "outlet": out_patch.format(v=f"{T1_K:.6g}"), "walls": wall_zg})
    _field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)",
           {"inlet": "        type pressureInletOutletVelocity;\n        value uniform (0 0 0);\n",
            "outlet": "        type inletOutlet;\n        inletValue uniform (0 0 0);\n        value uniform (0 0 0);\n",
            "walls": "        type noSlip;\n"})
    _field(case, "k", "volScalarField", "[0 2 -2 0 0 0 0]", "1",
           {"inlet": "        type turbulentIntensityKineticEnergyInlet;\n        intensity 0.05;\n        value uniform 1;\n",
            "outlet": out_patch.format(v="1"), "walls": "        type kqRWallFunction;\n        value uniform 1;\n"})
    _field(case, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", "1000",
           {"inlet": "        type turbulentMixingLengthFrequencyInlet;\n        mixingLength 0.007;\n"
                     "        value uniform 1000;\n",
            "outlet": out_patch.format(v="1000"), "walls": "        type omegaWallFunction;\n        value uniform 1000;\n"})
    _field(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0",
           {"inlet": "        type calculated;\n        value uniform 0;\n",
            "outlet": "        type calculated;\n        value uniform 0;\n",
            "walls": "        type nutkWallFunction;\n        value uniform 0;\n"})
    _field(case, "alphat", "volScalarField", "[1 -1 -1 0 0 0 0]", "0",
           {"inlet": "        type calculated;\n        value uniform 0;\n",
            "outlet": "        type calculated;\n        value uniform 0;\n",
            "walls": "        type compressible::alphatWallFunction;\n        value uniform 0;\n"})
    _w(case, "system/controlDict", "dictionary",
       f"application rhoPimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime {iterations};\n"
       f"deltaT 1;\nwriteControl timeStep;\nwriteInterval {interval};\npurgeWrite 0;\n"
       "writeFormat ascii;\nwritePrecision 10;\nwriteCompression off;\ntimeFormat general;\ntimePrecision 8;\n"
       "runTimeModifiable false;\n")
    _w(case, "system/fvSchemes", "dictionary",
       "ddtSchemes\n{\n    default localEuler;\n}\ngradSchemes\n{\n    default Gauss linear;\n}\n"
       "divSchemes\n{\n    default none;\n    div(phi,U) Gauss upwind;\n    div(phi,h) Gauss upwind;\n"
       "    div(phi,e) Gauss upwind;\n    div(phi,K) Gauss upwind;\n    div(phi,k) Gauss upwind;\n"
       "    div(phi,omega) Gauss upwind;\n    div(phid,p) Gauss upwind;\n    div(phiv,p) Gauss upwind;\n"
       "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n"
       "laplacianSchemes\n{\n    default Gauss linear corrected;\n}\ninterpolationSchemes\n{\n    default linear;\n}\n"
       "snGradSchemes\n{\n    default corrected;\n}\nwallDist\n{\n    method meshWave;\n}\n")
    _w(case, "system/fvSolution", "dictionary",
       "solvers\n{\n    \"rho.*\"\n    {\n        solver diagonal;\n    }\n"
       "    \"p.*\"\n    {\n        solver PBiCGStab;\n        preconditioner DILU;\n        tolerance 1e-8;\n"
       "        relTol 0.01;\n    }\n"
       "    \"(U|h|e|k|omega).*\"\n    {\n        solver PBiCGStab;\n        preconditioner DILU;\n"
       "        tolerance 1e-9;\n        relTol 0.01;\n    }\n}\n"
       "PIMPLE\n{\n    momentumPredictor yes;\n    transonic yes;\n    nOuterCorrectors 1;\n    nCorrectors 2;\n"
       "    nNonOrthogonalCorrectors 0;\n    maxCo 0.5;\n    rDeltaTSmoothingCoeff 0.5;\n"
       "    rDeltaTDampingCoeff 0.5;\n"
       f"    maxDeltaT {max_dt:g};\n"
       f"    pMinFactor 0.2;\n    pMaxFactor 1.5;\n}}\n"
       "relaxationFactors\n{\n    equations\n    {\n        \".*\" 1;\n    }\n}\n")
    return mc


def _set_patch_types(case: str) -> None:
    path = os.path.join(case, "constant/polyMesh/boundary")
    txt = open(path).read()
    for name, typ in (("front", "wedge"), ("back", "wedge"), ("walls", "wall")):
        txt = re.sub(rf"({name}\s*\{{\s*type\s+)\w+;", rf"\g<1>{typ};", txt)
        txt = re.sub(rf"({name}\s*\{{[^}}]*?)physicalType\s+\w+;", r"\1", txt)
    with open(path, "w") as f:
        f.write(txt)


# ------------------------------------------------------------------------------------------------- thermo check
def gas_enthalpy(T_K, p_Pa, mc: dict[str, float], eos: str = "PengRobinson", T_std: float = 298.15) -> np.ndarray:
    """Specific enthalpy [J/kg] of the pseudo-pure gas the OpenFOAM case uses: ``Cp0 (T - Tstd)`` plus, for
    Peng–Robinson, the departure ``RTc [Tr (Z - 1) - 2.078 (1 + kappa) sqrt(alpha) ln((Z + 2.414 B)/(Z - 0.414 B))]``
    (per kmol, over the molar mass). Up to a constant, the same model as OpenFOAM's ``hConst`` with
    ``PengRobinsonGas``; :func:`run_valve_cfd` uses it to check the CFD's energy balance independently of the
    solver."""
    T = np.asarray(T_K, float)
    p = np.asarray(p_Pa, float)
    h = mc["Cp0"] * (T - T_std)
    if not eos.lower().startswith("peng"):
        return h
    return h + _pr_departure(T, p, mc)


def _pr_Z(T, p, mc):
    Tr, Pr = T / mc["Tc"], p / mc["Pc"]
    kappa = 0.37464 + 1.54226 * mc["omega"] - 0.26992 * mc["omega"] ** 2
    alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2
    A = 0.45724 * alpha * Pr / Tr ** 2
    B = 0.07780 * Pr / Tr
    Tf, Bf, Af = np.broadcast_arrays(T, B, A)
    Z = np.empty(Tf.shape)
    for idx in np.ndindex(Tf.shape):
        a, b = Af[idx], Bf[idx]
        r = np.roots([1.0, -(1 - b), a - 3 * b * b - 2 * b, -(a * b - b * b - b ** 3)])
        r = r[np.abs(r.imag) < 1e-9].real
        Z[idx] = r.max()                                    # the gas root
    return Z, kappa, alpha, B


def _pr_departure(T, p, mc):
    Z, kappa, alpha, B = _pr_Z(T, p, mc)
    Tr = T / mc["Tc"]
    R = R_UNIVERSAL / mc["W"]
    s2 = math.sqrt(2.0)
    return R * mc["Tc"] * (Tr * (Z - 1) - 0.45724 / (0.07780 * 2 * s2) * (1 + kappa) * np.sqrt(alpha)
                           * np.log((Z + (1 + s2) * B) / (Z + (1 - s2) * B)))


def model_throttle(mc: dict[str, float], P1_Pa: float, T1_K: float, P2_Pa: float, eos: str = "PengRobinson",
                   v1: float = 0.0, v2: float = 0.0) -> float:
    """Outlet temperature [K] of the pseudo-pure gas of the case at constant stagnation enthalpy: what the CFD must
    give if it conserves energy (bisection on ``h(T2, P2) + v2²/2 = h(T1, P1) + v1²/2``)."""
    h0 = float(gas_enthalpy(T1_K, P1_Pa, mc, eos)) + 0.5 * v1 ** 2 - 0.5 * v2 ** 2
    lo, hi = T1_K - 120.0, T1_K + 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(gas_enthalpy(mid, P2_Pa, mc, eos)) > h0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------- run and read
def _parse_list(text: str, start: int, ncomp: int):
    m = re.compile(r"(?:nonuniform\s+List<\w+>\s*(\d+)\s*(?:\(|\{([^}]*)\})|uniform\s+([^;]+);)").search(text, start)
    if m.group(2) is not None:                              # "N{value}": a list of N equal values
        v = np.array(m.group(2).replace("(", " ").replace(")", " ").split(), float)
        n = int(m.group(1))
        return ("list", np.tile(v, (n, 1)) if ncomp > 1 else np.full(n, v[0]))
    if m.group(3) is not None:
        v = np.array(m.group(3).replace("(", " ").replace(")", " ").split(), float)
        return ("uniform", v)
    depth, k = 1, m.end()                                 # the matching ")" of the list (inline or multi-line)
    while depth:
        c = text[k]
        depth += (c == "(") - (c == ")")
        k += 1
    body = text[m.end(): k - 1]
    a = np.array(body.replace("(", " ").replace(")", " ").split(), float)
    return ("list", a.reshape(int(m.group(1)), ncomp) if ncomp > 1 else a)


def _internal(path: str, ncomp: int, n: int) -> np.ndarray:
    t = open(path).read()
    kind, v = _parse_list(t, t.index("internalField"), ncomp)
    return np.tile(v, (n, 1)) if ncomp > 1 and kind == "uniform" else (np.full(n, v[0]) if kind == "uniform" else v)


def _patch(path: str, patch: str, ncomp: int, n: int) -> np.ndarray | None:
    t = open(path).read()
    b = t.index("boundaryField")
    m = re.compile(rf"\n\s*{patch}\s*\n\s*\{{").search(t, b)
    end = t.index("}", m.end())
    seg = t[m.end():end]
    if "value" not in seg:
        return None
    kind, v = _parse_list(t, t.index("value", m.end()), ncomp)
    if kind == "uniform":
        return np.tile(v, (n, 1)) if ncomp > 1 else np.full(n, v[0])
    return v


def _foam_list(path: str) -> list:
    t = open(path).read()
    t = t[t.index("}", t.index("FoamFile")) + 1:]
    t = re.sub(r"//.*", "", t)
    m = re.search(r"\n(\d+)\s*\n\(", t)
    return t[m.end():t.rindex(")")], int(m.group(1))


def _poly_mesh(case: str) -> dict[str, Any]:
    pm = os.path.join(case, "constant", "polyMesh")
    body, n = _foam_list(os.path.join(pm, "points"))
    points = np.array(body.replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)
    body, n = _foam_list(os.path.join(pm, "faces"))
    faces = [list(map(int, f.split())) for f in re.findall(r"\d+\(([^)]*)\)", body)]
    body, n = _foam_list(os.path.join(pm, "owner"))
    owner = np.array(body.split(), int)
    body, n = _foam_list(os.path.join(pm, "neighbour"))
    neigh = np.array(body.split(), int)
    t = open(os.path.join(pm, "boundary")).read()
    patches = {m.group(1): (int(m.group(3)), int(m.group(2)))
               for m in re.finditer(r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", t)}
    fc = np.array([points[f].mean(0) for f in faces])
    area = np.array([0.5 * np.cross(points[f] - points[f].mean(0), np.roll(points[f], -1, 0) - points[f].mean(0)).sum(0)
                     for f in faces])
    nc = int(max(owner.max(), neigh.max() if len(neigh) else 0)) + 1
    cc = np.zeros((nc, 3))
    cnt = np.zeros(nc)
    np.add.at(cc, owner, fc)
    np.add.at(cnt, owner, 1)
    np.add.at(cc, neigh, fc[: len(neigh)])
    np.add.at(cnt, neigh, 1)
    return {"cells": nc, "C": cc / cnt[:, None], "patches": patches, "Sf": area, "Cf": fc}


def _write_initial_p(case: str, mesh: dict[str, Any], P1: float, P2: float, x_split: float) -> None:
    """Upstream of the seat at the inlet pressure, downstream at the outlet pressure."""
    path = os.path.join(case, "0", "p")
    t = open(path).read()
    p0 = np.where(mesh["C"][:, 0] < x_split, P1, P2)
    vals = "\n".join(f"{v:.8g}" for v in p0)
    lst = f"nonuniform List<scalar>\n{len(p0)}\n(\n{vals}\n)\n"
    t = re.sub(r"internalField\s+uniform\s+[^;]+;", "internalField " + lst + ";", t, count=1)
    with open(path, "w") as f:
        f.write(t)


def _times(case: str) -> list[str]:
    return sorted((d for d in os.listdir(case) if re.fullmatch(r"[0-9.e+-]+", d) and float(d) > 0), key=float)


def _patch_state(case: str, tdir: str, mesh: dict[str, Any], name: str, mc: dict[str, float], eos: str,
                 wedge_deg: float) -> dict[str, float]:
    start, n = mesh["patches"][name]
    d = os.path.join(case, tdir)
    phi = _patch(os.path.join(d, "phi"), name, 1, n)
    T = _patch(os.path.join(d, "T"), name, 1, n)
    p = _patch(os.path.join(d, "p"), name, 1, n)
    U = _patch(os.path.join(d, "U"), name, 3, n)
    w = np.abs(phi)
    U2 = (U ** 2).sum(1)
    h = gas_enthalpy(T, p, mc, eos)
    W = w.sum()
    return {"mdot": float(W * 360.0 / wedge_deg), "T": float((w * T).sum() / W), "p": float((w * p).sum() / W),
            "h": float((w * h).sum() / W), "h0": float((w * (h + 0.5 * U2)).sum() / W),
            "U": float((w * np.sqrt(U2)).sum() / W)}


def run_valve_cfd(out_dir: str, gas: dict[str, float], P1_bar: float, T1_C: float, P2_bar: float,
                  eos: str = "PengRobinson", geom: ValveGeometry | None = None, iterations: int = 3000,
                  h_fine_mm: float = 0.4, h_coarse_mm: float = 3.0, timeout: float = 3600,
                  max_dt: float = 2e-5) -> dict[str, Any]:
    """Mesh, write and run one case with OpenFOAM, and read it back.

    Returns the mass-flow-weighted inlet and outlet temperature [°C], pressure [bar] and stagnation enthalpy
    [J/kg, :func:`gas_enthalpy`] at the last iteration, the mass flow [kg/s] through both, the outlet temperature
    the case's own gas model gives at constant stagnation enthalpy (:func:`model_throttle`), the history of the
    written iterations, the cell fields (x, r, T, p, |U|) for plotting, the timing and the case folder."""
    geom = geom or ValveGeometry()
    case = os.path.join(out_dir, f"case_{eos.lower()}")
    if os.path.exists(case):
        shutil.rmtree(case)
    os.makedirs(case)
    t0 = time.time()
    mesh_valve(geom, case, h_fine_mm, h_coarse_mm)
    T1, P1, P2 = T1_C + 273.15, P1_bar * 1e5, P2_bar * 1e5
    mc = write_case(case, gas, P1, T1, P2, eos=eos, iterations=iterations, max_dt=max_dt)
    iterations = int(mc.pop("iterations"))
    if _foam(case, "gmshToFoam valve.msh", "log.gmshToFoam", 600) != 0:
        raise RuntimeError(f"gmshToFoam failed: see {case}/log.gmshToFoam")
    _set_patch_types(case)
    _foam(case, "checkMesh", "log.checkMesh", 600)
    mesh = _poly_mesh(case)
    _write_initial_p(case, mesh, P1, P2, x_split=geom.seat_len * 1e-3)
    t_mesh = time.time() - t0
    t1 = time.time()
    try:
        rc = _foam(case, "rhoPimpleFoam", "log.rhoPimpleFoam", timeout)
    except subprocess.TimeoutExpired:
        rc = -1
    t_run = time.time() - t1
    log = open(os.path.join(case, "log.rhoPimpleFoam")).read()
    res: dict[str, Any] = {"eos": eos, "case": case, "cells": mesh["cells"], "mesh_seconds": round(t_mesh, 1),
                           "run_seconds": round(t_run, 1), "iterations": iterations, "gas_constants": mc,
                           "geometry": asdict(geom), "returncode": rc}
    times = _times(case)
    if rc != 0 or "FOAM FATAL" in log or not times or float(times[-1]) < iterations:
        res.update(status="failed", log_tail=log[-1500:], last_time=times[-1] if times else None)
        return res
    hist = {"it": [], "T_out_C": [], "mdot_out": [], "mdot_in": []}
    for td in times:
        o = _patch_state(case, td, mesh, "outlet", mc, eos, geom.wedge_deg)
        i = _patch_state(case, td, mesh, "inlet", mc, eos, geom.wedge_deg)
        hist["it"].append(float(td))
        hist["T_out_C"].append(o["T"] - 273.15)
        hist["mdot_out"].append(o["mdot"])
        hist["mdot_in"].append(i["mdot"])
    o = _patch_state(case, times[-1], mesh, "outlet", mc, eos, geom.wedge_deg)
    i = _patch_state(case, times[-1], mesh, "inlet", mc, eos, geom.wedge_deg)
    T2_model = model_throttle(mc, i["p"], i["T"], o["p"], eos, v1=i["U"], v2=o["U"])
    res.update(status="ok", T_in_C=i["T"] - 273.15, T_out_C=o["T"] - 273.15, p_in_bar=i["p"] / 1e5,
               p_out_bar=o["p"] / 1e5, h0_in=i["h0"], h0_out=o["h0"], U_in=i["U"], U_out=o["U"],
               mdot_in=i["mdot"], mdot_out=o["mdot"], T_out_model_C=T2_model - 273.15, history=hist)
    res["T_out_drift_K"] = float(abs(hist["T_out_C"][-1] - hist["T_out_C"][-2])) if len(times) > 1 else float("nan")
    res["mdot_imbalance"] = abs(i["mdot"] - o["mdot"]) / max(i["mdot"], 1e-12)
    d = os.path.join(case, times[-1])
    n = mesh["cells"]
    C = mesh["C"]
    U = _internal(os.path.join(d, "U"), 3, n)
    res["fields"] = {"x": C[:, 0], "r": np.hypot(C[:, 1], C[:, 2]), "T_C": _internal(os.path.join(d, "T"), 1, n) - 273.15,
                     "p_bar": _internal(os.path.join(d, "p"), 1, n) / 1e5, "U": np.linalg.norm(U, axis=1)}
    return res


# ------------------------------------------------------------------------------------------------- the study
def valve_cfd_study(out_dir: str = "valve_cfd", P1_bar: float = 70.0, T1_C: float = 25.0, P2_bar: float = 20.0,
                    composition: dict[str, float] | None = None, geom: ValveGeometry | None = None,
                    iterations: int = 3000, h_fine_mm: float = 0.4, h_coarse_mm: float = 3.0,
                    eos_list: tuple[str, ...] = ("ideal", "PengRobinson"), lang: str = "pt",
                    width_px: int = 1280, max_dt: float = 2e-5) -> dict[str, Any]:
    """The globe-valve passage run with OpenFOAM for each equation of state in ``eos_list``, compared with the
    library's throttling at the CFD's own mass flow, checked and graded.

    Returns
    -------
    dict
        ``runs`` (one :func:`run_valve_cfd` result per EOS, without the fields), ``reference`` (the throttling by
        GERG-2008 and Peng–Robinson, isenthalpic and with the pipe velocities), ``checks``, ``decision``,
        ``files`` and ``assumptions``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_veriphysics.policy import decide, load_policy

    from .joule_thomson import DEFAULT_GAS, throttle

    geom = geom or ValveGeometry()
    gas = dict(composition) if composition else dict(DEFAULT_GAS)
    os.makedirs(out_dir, exist_ok=True)
    pt = lang == "pt"
    files = [os.path.basename(geometry_figure(geom, os.path.join(out_dir, "valve_geometry.png"), lang, width_px))]
    P1, T1, P2 = P1_bar * 1e5, T1_C + 273.15, P2_bar * 1e5
    runs = {e: run_valve_cfd(out_dir, gas, P1_bar, T1_C, P2_bar, eos=e, geom=geom, iterations=iterations,
                             h_fine_mm=h_fine_mm, h_coarse_mm=h_coarse_mm, max_dt=max_dt) for e in eos_list}
    A = geom.areas_m2()["pipe"]
    ref = {}
    for name in ("GERG-2008", "Peng-Robinson"):
        iso = throttle(gas, P1, T1, P2, eos=name)
        ok_runs = [r for e, r in runs.items() if r.get("status") == "ok"]
        real = [r for e, r in runs.items() if r.get("status") == "ok" and e.lower().startswith("peng")]
        mdot = (real or ok_runs or [{}])[0].get("mdot_out")       # the real-gas run's own mass flow
        kin = throttle(gas, P1, T1, P2, eos=name, area_m2=A, mass_flow_kg_s=mdot) if mdot else iso
        ref[name] = {"T2_isenthalpic_C": iso["T2_K"] - 273.15, "T2_C": kin["T2_K"] - 273.15}

    checks: list[dict[str, Any]] = []

    def chk(code, status, detail, value=None):
        checks.append({"code": code, "status": status, "detail": detail, "value": value})

    for e, r in runs.items():
        tag = "pr" if e.lower().startswith("peng") else "ideal"
        if r.get("status") != "ok":
            chk(f"cfd.{tag}.ran", "fail", f"OpenFOAM did not finish ({e}): {r.get('log_tail', '')[-300:]}")
            continue
        conv = r["T_out_drift_K"] < 0.1 and r["mdot_imbalance"] < 0.01
        chk(f"cfd.{tag}.converged", "pass" if conv else "fail",
            f"outlet temperature moved {r['T_out_drift_K']:.3f} K between the last two writes (allowed 0.1 K); mass "
            f"flow in {r['mdot_in']:.3f} and out {r['mdot_out']:.3f} kg/s, {100 * r['mdot_imbalance']:.2f} % apart "
            "(allowed 1 %)", r["T_out_drift_K"])
        dh0 = r["h0_out"] - r["h0_in"]
        scale = r["gas_constants"]["Cp0"] * max(abs(ref["GERG-2008"]["T2_C"] - T1_C), 1.0)
        chk(f"cfd.{tag}.energy_balance", "pass" if abs(dh0) <= 0.02 * scale else "fail",
            f"mass-weighted stagnation enthalpy out minus in {dh0:.0f} J/kg (the case's gas model) against a "
            f"throttling scale of {scale:.0f} J/kg (Cp0·ΔT; allowed 2 %)", dh0)
        err = r["T_out_C"] - r["T_out_model_C"]
        chk(f"cfd.{tag}.model_consistency", "pass" if abs(err) <= 0.5 else "fail",
            f"CFD outlet {r['T_out_C']:.2f} °C against {r['T_out_model_C']:.2f} °C from the same gas model at the "
            f"CFD's inlet state, outlet pressure and velocities at constant stagnation enthalpy: {err:+.2f} K "
            "(allowed 0.5 K)", err)
    if "ideal" in runs and runs["ideal"].get("status") == "ok":
        dT = runs["ideal"]["T_out_C"] - T1_C
        chk("cfd.ideal.no_cooling", "pass" if abs(dT) < 1.0 else "fail",
            f"ideal gas: outlet {runs['ideal']['T_out_C']:.2f} °C, {dT:+.2f} K from the inlet (theory: 0 K from the "
            "throttling, only the kinetic term; allowed ±1 K)", dT)
    prk = next((k for k in runs if k.lower().startswith("peng")), None)
    if prk and runs[prk].get("status") == "ok":
        r = runs[prk]
        for name, tol_frac in (("Peng-Robinson", 0.08), ("GERG-2008", 0.15)):
            target = ref[name]["T2_C"]
            drop = abs(target - T1_C)
            err = r["T_out_C"] - target
            chk(f"cfd.pr.vs_{'pr' if name == 'Peng-Robinson' else 'gerg'}", "pass" if abs(err) <= tol_frac * drop
                else "fail",
                f"CFD (OpenFOAM Peng–Robinson, pseudo-pure mixture) outlet {r['T_out_C']:.2f} °C against the "
                f"library's {name} throttling at the CFD's mass flow {target:.2f} °C: {err:+.2f} K (allowed "
                f"{100 * tol_frac:.0f} % of the {drop:.1f} K drop)", err)
    decision, reasons = decide(checks, load_policy(None))

    # figures -------------------------------------------------------------------------------------------
    W = width_px / 100
    ok = {e: r for e, r in runs.items() if r.get("status") == "ok"}
    if ok:
        fig, axs = plt.subplots(len(ok), 1, figsize=(W * 0.6, W * 0.17 * len(ok) + 0.6), squeeze=False)
        vmin = min(float(np.min(r["fields"]["T_C"])) for r in ok.values())
        vmax = max(float(np.max(r["fields"]["T_C"])) for r in ok.values())
        for ax, (e, r) in zip(axs[:, 0], ok.items(), strict=True):
            f = r["fields"]
            sel = f["x"] < 0.15
            for sgn in (1, -1):
                tc = ax.tricontourf(f["x"][sel] * 1e3, sgn * f["r"][sel] * 1e3, f["T_C"][sel], levels=40,
                                    cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            lab = ("gás ideal" if pt else "ideal gas") if e == "ideal" else "Peng–Robinson"
            ax.set_title(f"{'Temperatura' if pt else 'Temperature'} (°C), OpenFOAM, {lab}: "
                         f"{'saída' if pt else 'outlet'} {r['T_out_C']:.2f} °C", fontsize=10)
            ax.set_ylabel("r (mm)")
            fig.colorbar(tc, ax=ax, fraction=0.03)
        axs[-1, 0].set_xlabel("x (mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cfd_temperature.png"), dpi=100)
        plt.close(fig)
        files.append("cfd_temperature.png")

        fig, axs = plt.subplots(1, 2, figsize=(W * 0.62, W * 0.24))
        for e, r in ok.items():
            lab = ("gás ideal" if pt else "ideal gas") if e == "ideal" else "Peng–Robinson"
            axs[0].plot(r["history"]["it"], r["history"]["T_out_C"], label=f"CFD {lab}")
        for name, col in (("GERG-2008", "#0f766e"), ("Peng-Robinson", "#b45309")):
            axs[0].axhline(ref[name]["T2_C"], ls="--", color=col, label=f"{'estrangulamento' if pt else 'throttling'} "
                                                                         f"{name}")
        axs[0].set_xlabel("iteração" if pt else "iteration")
        axs[0].set_ylabel("T " + ("de saída (°C)" if pt else "outlet (°C)"))
        axs[0].legend(fontsize=8)
        axs[0].set_title("Convergência" if pt else "Convergence", fontsize=10)
        names = [("gás ideal" if pt else "ideal gas") if e == "ideal" else "CFD PR" for e in ok] + ["PR", "GERG-2008"]
        vals = [r["T_out_C"] for r in ok.values()] + [ref["Peng-Robinson"]["T2_C"], ref["GERG-2008"]["T2_C"]]
        bars = axs[1].bar(names, vals, color=["#ef4444" if n.startswith(("gás", "ideal")) else "#3b82f6"
                                              for n in names[:len(ok)]] + ["#b45309", "#0f766e"])
        axs[1].axhline(T1_C, color="k", lw=0.8)
        for b, v in zip(bars, vals, strict=True):
            axs[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        axs[1].set_ylabel("°C")
        axs[1].set_title("Temperatura de saída: CFD × estrangulamento" if pt else "Outlet temperature: CFD vs throttling",
                         fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cfd_vs_throttling.png"), dpi=100)
        plt.close(fig)
        files.append("cfd_vs_throttling.png")

    slim = {e: {k: v for k, v in r.items() if k not in ("fields",)} for e, r in runs.items()}
    assumptions = [
        "axisymmetric plug-and-seat passage (stem, seat ring, plug disc), not a globe valve body; 5° wedge",
        "the mixture as one pseudo-pure fluid in OpenFOAM: Kay's-rule critical constants, ideal-gas Cp at the inlet "
        "temperature (constant), Peng–Robinson departure functions",
        "steady state reached with local time stepping (rhoPimpleFoam, transonic), first-order upwind convection",
        "inlet: static pressure from the total pressure and the inlet temperature fixed; adiabatic walls; k-omega SST",
        "outlet temperature: the mass-flow-weighted average on the outlet patch",
    ]
    return {"runs": slim, "reference": ref, "checks": checks, "decision": {"verdict": decision, "reasons": reasons},
            "files": files, "assumptions": assumptions, "geometry": asdict(geom)}
