"""Writing and running the OpenFOAM case of a passage: steady, compressible, real gas, second order.

The dictionaries are built from three things and nothing else — the :class:`~.passage.PassageSpec`
(what the geometry is), the :class:`~.physics.Gas` (what flows) and the
:class:`~.physics.Conditions` (what is imposed) — so any passage that passes
:meth:`~.passage.PassageSpec.problems` can be run, and the valve is one of them.

The solver is ``rhoSimpleFoam``: a steady, pressure-based compressible solver. Two choices in the
dictionaries are worth knowing about, because both are departures from the reference document and
both are the reason its result could not be used:

* the run is **steady**, not transient with ``dt = 1e-7 s`` for 300 steps. A valve with fixed
  pressures is a steady problem; 30 us is 4 % of the acoustic transit time of a 70 mm body (0.7 ms),
  so the reference run stopped before the pressure field had crossed the valve once.
* the convection schemes are **second order** (``linearUpwind``), not all first order. First order
  adds a numerical viscosity that flattens exactly the jet the valve is about.

The gas is ``PengRobinsonGas`` with ``hConst`` + ``sensibleEnthalpy``. That combination matters:
OpenFOAM's ``hConst`` adds ``EquationOfState::H(p, T)`` to the ideal-gas part, so the enthalpy
carries the real-gas departure and the throttle cools. ``perfectGas`` would give ``dT = 0`` at any
pressure ratio, whatever the mesh.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from .passage import MeshSummary, PassageSpec, build_mesh
from .physics import Conditions, Gas, peng_robinson, sutherland_mu

__all__ = ["write_case", "run_case", "read_run", "RunResult", "case_ready", "inlet_area_m2",
           "estimate_mass_flow", "resolved_mass_flow", "acoustic_transit_s", "solver_for"]

_HEADER = ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
           "| =========                 |                                                 |\n"
           "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
           "|  \\\\    /   O peration     | Version:  v2606                                 |\n"
           "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
           "|    \\\\/     M anipulation  |                                                 |\n"
           "\\*---------------------------------------------------------------------------*/\n")


def estimate_mass_flow(passage: PassageSpec, gas: Gas, cond: Conditions, *, cd: float = 0.6) -> float:
    """``mdot = Cd A sqrt(2 rho dP)`` on the throttling area: the first guess a run starts from.

    ``Cd = 0.6`` is the sharp-edged orifice value and a contoured passage sits a little above it, so
    the guess is deliberately on the low side. The outer loop in :mod:`.study` then converges in two
    or three solves; the number is printed so nobody has to wonder where the flow came from.
    """
    area = float(passage.meta.get("throttling_area_mm2", 0.0)) * 1e-6
    if area <= 0.0:
        area = math.pi * (passage.spans[0].radii[-1] * 1e-3) ** 2
    rho = peng_robinson(gas, cond.p_out_bar * 1e5, cond.T_in_K)["rho"]
    target = cond.p_in_target_bar if cond.p_in_target_bar is not None else cond.p_out_bar * 1.5
    dp = max((target - cond.p_out_bar) * 1e5, 1.0)
    return float(min(max(cd * area * math.sqrt(2.0 * rho * dp), 1e-5), 50.0))


def resolved_mass_flow(passage: PassageSpec, gas: Gas, cond: Conditions) -> float:
    """The flow the case imposes: the caller's, or the estimate when it was left open."""
    return float(cond.mass_flow_kg_s) if cond.mass_flow_kg_s else estimate_mass_flow(passage, gas, cond)


def inlet_area_m2(passage: PassageSpec) -> float:
    """Wetted area of the inlet patch, m2, for the sector."""
    r = passage.spans[0].radii[-1]
    full = math.pi * (r ** 2 - passage.r_core ** 2) * 1e-6
    return full * passage.sector_deg / 360.0


def acoustic_transit_s(passage: PassageSpec, gas: Gas, T_K: float) -> float:
    """How long a pressure wave needs to cross the passage, s.

    ``L / a`` with ``a = sqrt(gamma R T)``. It is the yardstick a transient has to be measured
    against, and the one the reference document fails: 300 steps of 1e-7 s on a 180 mm body is 4 %
    of one transit, so its pressure field never crossed the valve.
    """
    length_m = abs(passage.spans[-1].x1 - passage.spans[0].x0) * 1e-3
    a = math.sqrt(gas.gamma_ig * gas.Rs * T_K)
    return length_m / a


def solver_for(cond: Conditions) -> str:
    """The OpenFOAM solver this case is written for."""
    return "rhoPimpleFoam" if cond.transient else "rhoSimpleFoam"


def _dict(path: str, obj: str, body: str, cls: str = "dictionary", location: str = "") -> None:
    loc = f'    location    "{location}";\n' if location else ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HEADER + "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n{body}\n")


_SYM_BLOCK = ("    axis\n    {\n        type            symmetry;\n    }\n"
              "    sym0\n    {\n        type            symmetryPlane;\n    }\n"
              "    sym1\n    {\n        type            symmetryPlane;\n    }\n")

#: patches appended to every ``0/`` field. A revolved mesh has the axis core and two symmetry
#: planes; a 3-D mesh has none of them, and OpenFOAM refuses a boundaryField for a patch that the
#: mesh does not have — so :func:`write_case` empties this for the 3-D backend.
_SYM = _SYM_BLOCK


def _field(path: str, name: str, dims: str, value, boundary: str, *,
           cls: str | None = None) -> None:
    """A ``0/<name>`` field. ``boundary`` holds inlet/outlet/wall; the axis core and the two
    symmetry planes are appended here, the same way for every field, because their type is a
    property of the patch and not of the field."""
    if isinstance(value, tuple):
        internal, cls = "uniform (" + " ".join(str(v) for v in value) + ")", cls or "volVectorField"
    else:
        internal, cls = f"uniform {value}", cls or "volScalarField"
    body = (f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{{boundary}\n"
            f"{_SYM}}}")
    _dict(path, name, body, cls=cls, location="0")


@dataclass
class RunResult:
    case_dir: str
    ok: bool
    elapsed_s: float
    n_iter: int
    residuals: dict[str, float]
    mesh_ok: bool
    checkmesh_log: str
    solver_log: str
    times: list[str] = field(default_factory=list)
    note: str = ""
    converged: bool = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "converged": self.converged, "elapsed_s": round(self.elapsed_s, 1),
                "n_iter": self.n_iter, "residuals": {k: float(v) for k, v in self.residuals.items()},
                "mesh_ok": self.mesh_ok, "times": self.times, "note": self.note}


def _restart_field(old: str, new: str) -> None:
    """Warm start a field: take the *internal* field from the previous solve, keep the boundary
    conditions just written.

    Copying the file whole would carry the previous solve's boundary conditions too -- including the
    inlet's ``massFlowRate`` -- and the pressure-drop loop would then run every pass at the same flow
    while printing a new one. That is what happened the first time the loop ran, and every pass after
    the first was silently the first.
    """
    with open(old, encoding="utf-8", errors="replace") as f:
        old_text = f.read()
    with open(new, encoding="utf-8", errors="replace") as f:
        new_text = f.read()
    try:
        a0 = old_text.index("internalField")
        a1 = old_text.index("boundaryField", a0)
        b0 = new_text.index("internalField")
        b1 = new_text.index("boundaryField", b0)
    except ValueError:
        return
    with open(new, "w", encoding="utf-8") as f:
        f.write(new_text[:b0] + old_text[a0:a1] + new_text[b1:])


def _last_time(folder: str) -> str | None:
    """The most recent time directory of a finished run, if there is one to restart from."""
    if not os.path.isdir(folder):
        return None
    times = sorted((d for d in os.listdir(folder)
                    if re.fullmatch(r"[0-9.eE+-]+", d) and os.path.isdir(os.path.join(folder, d))
                    and os.path.exists(os.path.join(folder, d, "p"))), key=float)
    return times[-1] if times else None


def write_case(passage: PassageSpec, case_dir: str, gas: Gas, cond: Conditions, *,
               mesh: MeshSummary | None = None, clear: bool = True,
               restart_from: str | None = None) -> dict[str, Any]:
    """Write the whole case (mesh included) into ``case_dir``; return its manifest.

    ``restart_from`` copies the fields of a previous solution over ``0/``: that is what makes the
    pressure-drop loop cheap and, more importantly, what gets a strongly throttled case to converge
    at all. Starting a 1.76 kg/s run from a uniform field diverges on a coarse mesh (measured: NaN
    in the first pressure correction); climbing to it through warm-started solves does not.
    """
    if clear and os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    global _SYM
    _SYM = "" if passage.meta.get("solid_builder") else _SYM_BLOCK
    # a malha é construída **depois** dos dicionários: o gmshToFoam e o polyDualMesh leem
    # system/controlDict e system/fvSchemes, que só existem a partir daqui
    rho_in = peng_robinson(gas, cond.p_out_bar * 1e5, cond.T_in_K)["rho"]
    m_total = resolved_mass_flow(passage, gas, cond)
    m_sector = m_total * passage.sector_deg / 360.0
    u_in = m_sector / (rho_in * inlet_area_m2(passage))
    k_in = 1.5 * (cond.turbulence_intensity * u_in) ** 2
    omega_in = max(k_in, 1e-6) ** 0.5 / (0.09 ** 0.25 * 0.001)     # mixing length 1 mm
    turbulence = cond.turbulence

    _dict(os.path.join(case_dir, "constant", "thermophysicalProperties"), "thermophysicalProperties", f"""
thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState PengRobinsonGas;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie
    {{
        molWeight   {gas.molWeight};
    }}
    thermodynamics
    {{
        Cp          {gas.Cp_ig};
        Hf          0;
    }}
    transport
    {{
        As          {gas.As:.6e};
        Ts          {gas.Ts};
    }}
    equationOfState
    {{
        Tc          {gas.Tc};
        Pc          {gas.Pc};
        Vc          {gas.Vc};
        omega       {gas.omega};
    }}
}}
// mu(300 K) = {sutherland_mu(gas, 300.0):.4e} Pa s (Sutherland), rho(300 K, p_out) = {rho_in:.3f} kg/m3
// (Peng-Robinson); the ideal gas would give {cond.p_out_bar * 1e5 / (gas.Rs * cond.T_in_K):.3f} kg/m3.""")

    _dict(os.path.join(case_dir, "constant", "turbulenceProperties"), "turbulenceProperties",
          f"simulationType  RAS;\n\nRAS\n{{\n    RASModel        {turbulence};\n"
          f"    turbulence      on;\n    printCoeffs     on;\n}}" if turbulence != "laminar"
          else "simulationType  laminar;")

    # A steady SIMPLE run in a throttling gap wanders through unphysical states before it settles:
    # the pressure correction overshoots, and at p = 4 bar with the enthalpy of 7 bar the
    # Peng-Robinson h->T inversion fails (measured: "Maximum number of iterations exceeded: 100 when
    # starting from T0:293.31 old T:166.33"). A temperature limiter keeps the run alive through those
    # excursions; it does not touch the answer, which stays inside [T_in - 20, T_in + 5] here.
    _dict(os.path.join(case_dir, "constant", "fvOptions"), "fvOptions", f"""
limitTemperature
{{
    type            limitTemperature;
    selectionMode   all;
    min             {cond.T_in_K - 120.0:g};
    max             {cond.T_in_K + 120.0:g};
}}""")

    _dict(os.path.join(case_dir, "constant", "g"), "g",
          "dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 -9.81 0);",
          cls="uniformDimensionedVectorField")

    trip = acoustic_transit_s(passage, gas, cond.T_in_K)
    end_time = float(cond.end_time_s) if cond.end_time_s else cond.end_time_transits * trip
    # o passo inicial cresce com Co; começar muito pequeno custa centenas de passos inúteis
    dt0 = max(end_time / 2000.0, 1e-9)
    solver = solver_for(cond)
    if cond.transient:
        # physical time: the Courant number sets the step, and the transits set the end. Averaging
        # starts where ``average_from`` says, so the start-up wave is not part of the answer.
        time_block = f"""
application     {solver};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.6g};
deltaT          {dt0:.6g};
adjustTimeStep  yes;
maxCo           {cond.max_co:g};
maxDeltaT       {end_time / 100.0:.6g};

writeControl    runTime;
writeInterval   {end_time / 20.0:.6g};
purgeWrite      1;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
"""
        average_start = cond.average_from * end_time
    else:
        time_block = f"""
application     {solver};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cond.iterations};
deltaT          1;

writeControl    timeStep;
writeInterval   {cond.write_interval};
purgeWrite      1;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
"""
        average_start = cond.iterations * 0.6

    # The inlet pressure is the *total* pressure the customer states (70 bar); the run measures
    # what the passage passes at that condition, so the mass flow is an output and no outer loop
    # is needed. ``totalPressure`` would have to be limped through a pressure jump otherwise, which
    # is what makes the steady formulation fragile here.
    _dict(os.path.join(case_dir, "system", "controlDict"), "controlDict", time_block + f"""
functions
{{
    machNo
    {{
        type            MachNo;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
    }}
    averages
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        timeStart       {average_start:.6g};
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
            T
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
            rho
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
        );
    }}
}}""")

    conv = {"linearUpwind": "bounded Gauss linearUpwind grad(U)",
            "limitedLinear": "bounded Gauss limitedLinear 1",
            "upwind": "bounded Gauss upwind"}[cond.scheme]
    turb_sc = ("bounded Gauss limitedLinear 1" if cond.scheme == "limitedLinear" else
               "bounded Gauss linearUpwind grad(U)" if cond.scheme == "linearUpwind" else
               "bounded Gauss upwind")
    _dict(os.path.join(case_dir, "system", "fvSchemes"), "fvSchemes", f"""
ddtSchemes      {{ default {'Euler' if cond.transient else 'steadyState'}; }}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {conv};
    div(phi,h)      {turb_sc};
    div(phi,K)      {turb_sc};
    div(phi,Ekp)    {turb_sc};
    turbulence      {turb_sc};
    div(phi,k)      $turbulence;
    div(phi,omega)  $turbulence;
    div(phid,p)     Gauss upwind;
    div((phi|interpolate(rho)),p)  Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U)))))  Gauss linear;
}}

laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes   {{ default corrected; }}
wallDist        {{ method meshWave; }}""")

    if cond.transient:
        algo = f"""
PIMPLE
{{
    nOuterCorrectors {cond.n_outer};
    nCorrectors      2;
    nNonOrthogonalCorrectors {max(cond.n_non_orth, 1)};
    pMinFactor      {cond.p_min_factor:g};
    pMaxFactor      {cond.p_max_factor:g};
    transonic       {'yes' if cond.transonic else 'no'};
}}

relaxationFactors
{{
    equations       {{ "(U|h|k|omega)" 1; }}
}}"""
    else:
        algo = f"""
SIMPLE
{{
    nNonOrthogonalCorrectors {cond.n_non_orth};
    pMinFactor      {cond.p_min_factor:g};
    pMaxFactor      {cond.p_max_factor:g};
    transonic       {'yes' if cond.transonic else 'no'};
    consistent      yes;

    residualControl
    {{
        p               1e-3;
        U               1e-5;
        h               1e-5;
        "(k|omega)"     1e-5;
    }}
}}

relaxationFactors
{{
    fields      {{ p 1; }}
    equations   {{ p 1; U {cond.relaxation}; h {cond.relaxation}; k {cond.relaxation}; omega {cond.relaxation}; }}
}}"""
    p_block = ("        solver          GAMG;\n"
               "        tolerance       1e-09;\n"
               "        relTol          0.01;\n"
               "        smoother        GaussSeidel;\n"
               "        nCellsInCoarsestLevel 20;")
    _dict(os.path.join(case_dir, "system", "fvSolution"), "fvSolution", f"""
solvers
{{
    rho
    {{
        solver          diagonal;
    }}
    rhoFinal
    {{
        $rho;
    }}
    p
    {{
{p_block}
    }}
    pFinal
    {{
        $p;
        relTol          0;
    }}
    "(U|k|omega)"
    {{
        solver          GAMG;
        tolerance       1e-09;
        relTol          0.1;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }}
    "(U|k|omega)Final"
    {{
        $U;
        relTol          0;
    }}
    "(h|e)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-09;
        relTol          0.1;
    }}
    "(h|e)Final"
    {{
        $h;
        relTol          0;
    }}
}}
{algo}""")

    wall_T = ("    {\n        type            zeroGradient;\n    }" if cond.wall_thermal == "adiabatic"
              else f"    {{\n        type            fixedValue;\n        value           uniform "
                   f"{cond.T_wall_K or cond.T_in_K};\n    }}")
    p_in_bar = cond.p_in_target_bar if cond.p_in_target_bar is not None else cond.p_out_bar * 1.15
    # In physical time the inlet carries the *total* pressure the customer states and the velocity
    # follows the flux: the two together are what a pressurised upstream really is, and the flow is
    # then a measured output. In pseudo-time the same pair is what makes the pressure correction
    # overshoot, so the steady case keeps the mass-flow inlet.
    p_inlet = (f"        type            totalPressure;\n        p0              uniform "
               f"{p_in_bar * 1e5:.6g};\n        value           uniform {p_in_bar * 1e5:.6g};"
               if cond.transient else "        type            zeroGradient;")
    _field(os.path.join(case_dir, "0", "p"), "p", "[1 -1 -2 0 0 0 0]", cond.p_out_bar * 1e5,
           f"""
    inlet
    {{
{p_inlet}
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {cond.p_out_bar * 1e5};
    }}
    wall
    {{
        type            zeroGradient;
    }}""")

    u_inlet = ("""        type            pressureInletVelocity;
        value           uniform (0 0 0);""" if cond.transient else f"""        type            flowRateInletVelocity;
        massFlowRate    constant {m_sector:.8e};
        rhoInlet        {rho_in:.6f};
        value           uniform ({u_in:.6f} 0 0);""")
    _field(os.path.join(case_dir, "0", "U"), "U", "[0 1 -1 0 0 0 0]",
           (0.0, 0.0, 0.0) if cond.transient else (round(u_in, 6), 0.0, 0.0),
           f"""
    inlet
    {{
{u_inlet}
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    wall
    {{
        type            noSlip;
    }}""")

    # a entrada em ``totalPressure`` pede a temperatura de **estagnação**: impor a estática junto com
    # a pressão total deixa a energia cinética de fora de um dos lados e vaza entalpia de estagnação
    # (medido: +1,93 % numa passagem adiabática, que deveria conservar h0)
    _tin = (f"""        type            totalTemperature;
        T0              uniform {cond.T_in_K};
        gamma           {gas.gamma_ig:.4f};""" if cond.transient else
            f"""        type            fixedValue;
        value           uniform {cond.T_in_K};""")
    _field(os.path.join(case_dir, "0", "T"), "T", "[0 0 0 1 0 0 0]", cond.T_in_K, f"""
    inlet
    {{
{_tin}
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    wall
{wall_T}""")

    if turbulence != "laminar":
        _field(os.path.join(case_dir, "0", "k"), "k", "[0 2 -2 0 0 0 0]", k_in, f"""
    inlet
    {{
        type            turbulentIntensityKineticEnergyInlet;
        intensity       {cond.turbulence_intensity};
        value           uniform {k_in:.6e};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {k_in:.6e};
        value           uniform {k_in:.6e};
    }}
    wall
    {{
        type            kqRWallFunction;
        value           uniform {k_in:.6e};
    }}""")
        _field(os.path.join(case_dir, "0", "omega"), "omega", "[0 0 -1 0 0 0 0]", omega_in, f"""
    inlet
    {{
        type            turbulentMixingLengthFrequencyInlet;
        mixingLength    0.001;
        value           uniform {omega_in:.6g};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {omega_in:.6g};
        value           uniform {omega_in:.6g};
    }}
    wall
    {{
        type            omegaWallFunction;
        value           uniform {omega_in:.6g};
    }}""")
        _field(os.path.join(case_dir, "0", "nut"), "nut", "[0 2 -1 0 0 0 0]", 1e-4, """
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    wall
    {
        type            nutkWallFunction;
        Cmu             0.09;
        kappa           0.41;
        E               9.8;
        value           uniform 0;
    }""")
        _field(os.path.join(case_dir, "0", "alphat"), "alphat", "[1 -1 -1 0 0 0 0]", 0.0, """
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    wall
    {
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }""")

    if restart_from:
        src = _last_time(restart_from)
        if src:
            for name in ("p", "U", "T", "k", "omega", "nut", "alphat"):
                old, new = os.path.join(restart_from, src, name), os.path.join(case_dir, "0", name)
                if os.path.exists(old) and os.path.exists(new):
                    _restart_field(old, new)

    mesh = mesh or build_mesh(passage, case_dir)
    return {"case_dir": case_dir, "mesh": mesh.as_dict(), "passage": passage.summary(),
            "mass_flow_kg_s": round(m_total, 6),
            "gas": {"name": gas.name, "rho_in": round(rho_in, 3), "mu_in": sutherland_mu(gas, cond.T_in_K)},
            "conditions": cond.as_dict(), "u_in_m_s": round(u_in, 4),
            "solver": solver, "acoustic_transit_s": round(trip, 9), "end_time_s": round(end_time, 9),
            "end_time_transits": round(end_time / trip, 2) if trip else None}


def case_ready(case_dir: str) -> bool:
    return all(os.path.exists(os.path.join(case_dir, *p)) for p in
               (("constant", "polyMesh", "points"), ("system", "controlDict"), ("0", "p")))


def _openfoam(cmd: str, cwd: str, timeout: float) -> tuple[int, str]:
    exe = shutil.which("openfoam")
    if exe is None:
        return 127, "openfoam not on PATH: install the native OpenFOAM app, or use the docker image"
    proc = subprocess.run([exe, "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _time_reached(case_dir: str, text: str) -> bool:
    """Did a transient run get to its ``endTime``? That, not a residual, is what "done" means here."""
    end = None
    path = os.path.join(case_dir, "system", "controlDict")
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(r"^endTime\s+([0-9.eE+-]+)", f.read(), flags=re.M)
        end = float(m.group(1)) if m else None
    times = [float(t) for t in re.findall(r"^Time = ([0-9.eE+-]+)", text, flags=re.M)]
    return bool(times) and end is not None and times[-1] >= end * 0.999


def _verdict(case_dir: str, text: str, solver: str, residuals: dict[str, float]) -> bool:
    if solver.startswith("rhoPimple"):
        # physical time: there is no residual floor to reach. The run is complete when it got to
        # ``endTime``, and the answer is the time mean OpenFOAM accumulated over the last part of it.
        return _time_reached(case_dir, text) and "FOAM FATAL" not in text
    return ("SIMPLE solution converged" in text
            or (residuals.get("p", 1.0) < 1e-3 and residuals.get("Ux", 1.0) < 1e-4))


def read_run(case_dir: str, *, solver: str = "rhoSimpleFoam") -> RunResult:
    """The verdict of a run that already happened, from its logs — nothing is executed.

    ``run_case`` runs the solver; this reads what it left behind, which is what a study being
    re-assembled (or a monitor) needs. Re-running the solver to "read" a case would silently
    overwrite the very result being read.
    """
    log_path = os.path.join(case_dir, f"log.{solver}")
    text = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    residuals: dict[str, float] = {}
    for m in re.finditer(r"Solving for (\w+), Initial residual = ([-+0-9.eE]+)", text):
        residuals[m.group(1)] = float(m.group(2))
    check_log = os.path.join(case_dir, "log.checkMesh")
    checkmesh_log = ""
    if os.path.exists(check_log):
        with open(check_log, encoding="utf-8", errors="replace") as f:
            checkmesh_log = f.read()[-2000:]
    mesh_ok = "Mesh OK" in checkmesh_log
    converged = _verdict(case_dir, text, solver, residuals)
    times = sorted((d for d in os.listdir(case_dir)
                    if re.fullmatch(r"[0-9.eE+-]+", d) and os.path.isdir(os.path.join(case_dir, d))),
                   key=float)
    return RunResult(case_dir=case_dir, ok=bool(converged and mesh_ok),
                     elapsed_s=0.0, n_iter=len(re.findall(r"^Time = ", text, flags=re.M)),
                     residuals=residuals, mesh_ok=mesh_ok, checkmesh_log=checkmesh_log,
                     solver_log=log_path, times=times,
                     note="" if converged else "o log termina com os resíduos no patamar",
                     converged=converged)


def run_case(case_dir: str, *, timeout: float = 7200.0, check: bool = True,
             solver: str = "rhoSimpleFoam") -> RunResult:
    """``checkMesh`` then the solver; the residuals and the convergence verdict are read back."""
    t0 = time.time()
    mesh_ok = True
    checkmesh_log = ""
    if check:
        rc, checkmesh_log = _openfoam("checkMesh > log.checkMesh 2>&1; tail -6 log.checkMesh", case_dir, 900)
        mesh_ok = rc == 0 and "Mesh OK" in checkmesh_log
    rc, _ = _openfoam(f"{solver} > log.{solver} 2>&1; echo RC=$?", case_dir, timeout)
    elapsed = time.time() - t0
    log_path = os.path.join(case_dir, f"log.{solver}")
    text = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    residuals: dict[str, float] = {}
    for m in re.finditer(r"Solving for (\w+), Initial residual = ([-+0-9.eE]+)", text):
        residuals[m.group(1)] = float(m.group(2))
    n_iter = len(re.findall(r"^Time = ", text, flags=re.M))
    times = sorted((d for d in os.listdir(case_dir)
                    if re.fullmatch(r"[0-9.eE+-]+", d) and os.path.isdir(os.path.join(case_dir, d))),
                   key=float)
    # the solver prints its verdict when every residualControl floor is met; when a run stops at
    # ``endTime`` with the fields already below them it is converged all the same, and saying
    # otherwise would call a good answer "inconclusive" for stopping one iteration early
    converged = _verdict(case_dir, text, solver, residuals)
    note = ""
    if rc != 0 or "FOAM FATAL" in text:
        note = "the solver reported a fatal error"
    elif not converged:
        note = "the residuals were above the floor when the run ended: the values are the last ones written"
    return RunResult(case_dir=case_dir, ok=bool(converged and mesh_ok), elapsed_s=elapsed, n_iter=n_iter,
                     residuals=residuals, mesh_ok=mesh_ok, checkmesh_log=checkmesh_log,
                     solver_log=log_path, times=times, note=note, converged=converged)
