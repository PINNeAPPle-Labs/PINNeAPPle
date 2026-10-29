"""OpenFOAM case templates for the meshed geometry: steady ``simpleFoam`` or transient ``pimpleFoam``,
incompressible, laminar or k-omega SST, with an optional passive scalar (a temperature at low heating, or a
concentration).

Boundary conditions follow the patch kinds: ``inlet`` uniform velocity along the inward normal of the patch
(``velocity`` m/s), ``outlet`` p = 0, ``wall`` no slip, ``symmetry`` symmetry. The scalar takes ``inlet`` on the
inlets and, per wall patch, a fixed value or zero gradient (``scalar.walls``).

No function objects: on the OpenFOAM v1912 Ubuntu 24.04 package every function object stops the solver at start-up
("error in IOstream sha1"), so pressures, fluxes, forces and y+ are computed from the written fields
(:mod:`.post`) and the residuals and Courant numbers from the solver log. The passive scalar is solved after the
flow, on the converged U and phi, by ``scalarTransportFoam`` (steady, ``DT`` in transportProperties). The fields start
in ``0.orig`` and the case's ``Allrun`` copies them to ``0`` after meshing (:func:`.mesh.allrun_script`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import HDR, allrun_script, decompose_dict

__all__ = ["Physics", "ScalarSpec", "write_internal_case", "SOLVERS"]

SOLVERS = ("simpleFoam", "pimpleFoam")


@dataclass
class ScalarSpec:
    """A passive scalar: ``D`` its diffusivity (m²/s; nu / Pr for a temperature), ``inlet`` its value on the
    inlets, ``walls`` patch -> fixed value or ``"zeroGradient"`` (walls not named: zero gradient)."""
    name: str = "T"
    D: float = 1e-5
    inlet: float = 0.0
    walls: dict[str, Any] = field(default_factory=dict)
    initial: float | None = None


@dataclass
class Physics:
    """Incompressible flow: ``velocity`` (m/s, uniform on the inlets), ``nu`` (m²/s), ``rho`` (kg/m³, for forces
    and pressures in Pa), ``turbulence`` laminar | kOmegaSST, ``solver`` simpleFoam | pimpleFoam. Steady:
    ``iterations`` and ``residual_tol`` (residualControl). Transient: ``end_time``, ``dt``, ``max_courant``.
    k-omega SST inlets: turbulence ``intensity`` and ``length_scale`` (m, the hydraulic diameter)."""
    velocity: float = 0.1
    nu: float = 1e-6
    rho: float = 1000.0
    turbulence: str = "laminar"
    solver: str = "simpleFoam"
    iterations: int = 2000
    residual_tol: float = 1e-5
    end_time: float = 1.0
    dt: float = 1e-3
    max_courant: float = 0.9
    write_interval: float | None = None
    intensity: float = 0.05
    length_scale: float = 0.01
    scalar: ScalarSpec | None = None

    def __post_init__(self):
        if self.solver not in SOLVERS:
            raise ValueError(f"solver {self.solver!r} is not one of {SOLVERS}")
        if self.turbulence not in ("laminar", "kOmegaSST"):
            raise ValueError(f"turbulence {self.turbulence!r} is not laminar or kOmegaSST")
        if isinstance(self.scalar, dict):
            self.scalar = ScalarSpec(**self.scalar)
        if self.scalar is not None and self.solver != "simpleFoam":
            raise ValueError("the passive scalar is solved on a converged steady flow: use simpleFoam")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _w(case: Path, rel: str, cls: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HDR % (cls, p.name) + body)


def _vec(v) -> str:
    return "(" + " ".join(f"{float(x) + 0.0:.10g}" for x in v) + ")"          # no "-0"


def _field(case: Path, name: str, cls: str, dims: str, internal: str, entries: dict[str, str]) -> None:
    bf = "".join(f"    {k}\n    {{\n{v}    }}\n" for k, v in entries.items())
    bf += '    "proc.*"\n    {\n        type processor;\n        value $internalField;\n    }\n'
    _w(case, f"0.orig/{name}", cls, f"dimensions {dims};\ninternalField {internal};\nboundaryField\n{{\n{bf}}}\n")


def write_internal_case(case: str | Path, kinds: dict[str, str], inlet_normals: dict[str, np.ndarray],
                        physics: Physics, procs: int = 1, size: np.ndarray | None = None) -> dict[str, Any]:
    """0.orig fields, constant/ and system/ dictionaries and ``Allrun`` of an internal-flow case.

    ``kinds``: patch -> inlet / outlet / wall / symmetry; ``inlet_normals``: inlet patch -> its outward unit normal
    (the inflow is along minus it). Returns the boundary conditions written."""
    case = Path(case)
    ph = physics
    lam = ph.turbulence == "laminar"
    k0 = 1.5 * (ph.intensity * ph.velocity) ** 2
    om0 = k0 ** 0.5 / (0.09 ** 0.25 * 0.07 * ph.length_scale)           # mixing length 0.07 x hydraulic diameter

    def bc(kind_map: dict[str, str]) -> dict[str, str]:
        out = {"background": "        type zeroGradient;\n"}
        for p, k in kinds.items():
            out[p] = kind_map[k] if k in kind_map else kind_map["wall"]
        return out
    U = {"background": "        type fixedValue;\n        value uniform (0 0 0);\n"}
    for p, k in kinds.items():
        if k == "inlet":
            u = -ph.velocity * np.asarray(inlet_normals[p], float)
            U[p] = f"        type fixedValue;\n        value uniform {_vec(u)};\n"
        elif k == "outlet":
            U[p] = "        type inletOutlet;\n        inletValue uniform (0 0 0);\n        value uniform (0 0 0);\n"
        elif k == "symmetry":
            U[p] = "        type symmetry;\n"
        else:
            U[p] = "        type noSlip;\n"
    _field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", U)
    _field(case, "p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0", bc({
        "inlet": "        type zeroGradient;\n", "outlet": "        type fixedValue;\n        value uniform 0;\n",
        "symmetry": "        type symmetry;\n", "wall": "        type zeroGradient;\n"}))
    if not lam:
        _field(case, "k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k0:.6g}", bc({
            "inlet": f"        type fixedValue;\n        value uniform {k0:.6g};\n",
            "outlet": f"        type inletOutlet;\n        inletValue uniform {k0:.6g};\n        value uniform {k0:.6g};\n",
            "symmetry": "        type symmetry;\n",
            "wall": f"        type kqRWallFunction;\n        value uniform {k0:.6g};\n"}))
        _field(case, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {om0:.6g}", bc({
            "inlet": f"        type fixedValue;\n        value uniform {om0:.6g};\n",
            "outlet": f"        type inletOutlet;\n        inletValue uniform {om0:.6g};\n        value uniform {om0:.6g};\n",
            "symmetry": "        type symmetry;\n",
            "wall": f"        type omegaWallFunction;\n        value uniform {om0:.6g};\n"}))
        _field(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0", bc({
            "inlet": "        type calculated;\n        value uniform 0;\n",
            "outlet": "        type calculated;\n        value uniform 0;\n", "symmetry": "        type symmetry;\n",
            "wall": "        type nutkWallFunction;\n        value uniform 0;\n"}))
    sc = ph.scalar
    if sc is not None:
        S = {"background": "        type zeroGradient;\n"}
        for p, k in kinds.items():
            if k == "inlet":
                S[p] = f"        type fixedValue;\n        value uniform {sc.inlet:.10g};\n"
            elif k == "outlet":
                S[p] = (f"        type inletOutlet;\n        inletValue uniform {sc.inlet:.10g};\n"
                        f"        value uniform {sc.inlet:.10g};\n")
            elif k == "symmetry":
                S[p] = "        type symmetry;\n"
            else:
                v = sc.walls.get(p, "zeroGradient")
                S[p] = ("        type zeroGradient;\n" if v == "zeroGradient" else
                        f"        type fixedValue;\n        value uniform {float(v):.10g};\n")
        init = sc.inlet if sc.initial is None else sc.initial
        _field(case, sc.name, "volScalarField", "[0 0 0 0 0 0 0]", f"uniform {init:.10g}", S)
    _w(case, "constant/transportProperties", "dictionary",
       f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {ph.nu:.10g};\n"
       + (f"DT [0 2 -1 0 0 0 0] {sc.D:.10g};\n" if sc else ""))
    _w(case, "constant/turbulenceProperties", "dictionary", "simulationType laminar;\n" if lam else
       "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n")
    steady = ph.solver == "simpleFoam"
    if steady:
        wi = int(ph.write_interval or ph.iterations)
        time_ctl = (f"endTime {int(ph.iterations)};\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval {wi};\n")
    else:
        wi = ph.write_interval or ph.end_time
        time_ctl = (f"endTime {ph.end_time:.10g};\ndeltaT {ph.dt:.10g};\nwriteControl adjustableRunTime;\n"
                    f"writeInterval {wi:.10g};\nadjustTimeStep yes;\nmaxCo {ph.max_courant:.10g};\n")
    _w(case, "system/controlDict", "dictionary", f"""application {ph.solver};
startFrom startTime;
startTime 0;
stopAt endTime;
{time_ctl}purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable false;
""")
    turb_div = "" if lam else "    div(phi,k) bounded Gauss upwind;\n    div(phi,omega) bounded Gauss upwind;\n"
    sdiv = f"    div(phi,{sc.name}) Gauss linearUpwind grad({sc.name});\n" if sc else ""
    _w(case, "system/fvSchemes", "dictionary", f"""ddtSchemes {{ default {'steadyState' if steady else 'Euler'}; }}
gradSchemes {{ default Gauss linear; limited cellLimited Gauss linear 1; grad(U) $limited; grad({sc.name if sc else 'U'}) $limited; }}
divSchemes
{{
    default none;
    div(phi,U) {'bounded ' if steady else ''}Gauss linearUpwindV grad(U);
{turb_div}{sdiv}    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited corrected 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default limited corrected 0.5; }}
wallDist {{ method meshWave; }}
""")
    res = "\n".join(f"        {f} {ph.residual_tol:g};" for f in (["U", "p"] + ([] if lam else ["k", "omega"])))
    solve_others = "(U|k|omega)"
    # the scalar is linear on the frozen flow: solved to the absolute tolerance every step, not relaxed, so the
    # first steps of scalarTransportFoam already give the converged field (its SIMPLE loop may stop early)
    s_solver = (f"    {sc.name} {{ solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 5000; }}\n"
                if sc else "")
    algo = (f"SIMPLE\n{{\n    nNonOrthogonalCorrectors 1;\n    consistent yes;\n    residualControl\n    {{\n{res}\n    }}\n}}\n"
            "relaxationFactors\n{\n    equations { U 0.9; \".*\" 0.9;" + (f" {sc.name} 1;" if sc else "")
            + " }\n    fields { p 1; }\n}\n" if steady else
            "PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 1;\n}\n")
    _w(case, "system/fvSolution", "dictionary", f"""solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol {0.01 if steady else 0.0}; }}
    pFinal {{ $p; relTol 0; }}
    "{solve_others}" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol {0.05 if steady else 0}; }}
{s_solver}    "{solve_others}Final" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; }}
}}
{algo}""")
    decompose_dict(case, procs, np.ones(3) if size is None else size)
    p = case / "Allrun"
    p.write_text(allrun_script(ph.solver, procs, scalar=sc is not None))
    p.chmod(0o755)
    return {"solver": ph.solver, "turbulence": ph.turbulence, "kinds": dict(kinds),
            "inlet_velocity": {k: (-ph.velocity * np.asarray(v, float)).tolist() for k, v in inlet_normals.items()},
            "scalar": None if sc is None else asdict(sc), "procs": procs}
