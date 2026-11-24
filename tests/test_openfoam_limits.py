"""The known numerical limits of the OpenFOAM connectors, and the case that demonstrates each (issue #132).

Two layers, because the issue asks for both:

* the **registry** (`pinneapple_data.preflight.limits`) is checked for completeness — a decision per solver, and per
  limit the evidence, the minimal case and (where something can be done) the fix;
* the **rules** are checked on dictionaries, so the connector's behaviour is tested without a solver: a free-surface
  case switched to k-omega SST fails the pre-flight until it has ``wallDist``, an omega convection term and an omega
  linear solver, and a transient buoyant case fails until it has a ``rho`` solver and warns without ``correctPhi``.

The **runs** are here too, against a real OpenFOAM when one is installed (skipped otherwise): the minimal buoyant
cavity of ``buoyant_simple_cavity`` and the dam-break case switched to kOmegaSST. They are what makes the entries in
the registry measured rather than assumed — the numbers in ``limits.py`` are the output of this file.
"""
import os
import re
import shutil
import subprocess

import pytest

from pinneapple_data.preflight import openfoam as preflight_openfoam
from pinneapple_data.preflight.limits import (
    ACCEPTED,
    DECISIONS,
    LIMITS,
    UNSUPPORTED,
    as_dicts,
    decision,
    limits_for,
    unsupported,
)

SOLVERS_IN_THE_ISSUE = ("buoyantSimpleFoam", "buoyantPimpleFoam", "interIsoFoam")


def findings(report) -> dict[str, dict]:
    """Rule id -> finding, for asserting on a rule rather than on a message."""
    return {f.rule: f for f in report.findings if f.rule}


# ----------------------------------------------------------------------------------------------- the registry
def test_every_solver_in_the_issue_has_a_recorded_decision():
    for solver in SOLVERS_IN_THE_ISSUE:
        assert decision(solver) == ACCEPTED, solver
        assert limits_for(solver), f"{solver} has a decision and no limit recorded"
    assert set(DECISIONS) == set(SOLVERS_IN_THE_ISSUE)


def test_each_limit_carries_its_evidence_and_the_minimal_case_that_shows_it():
    for lim in LIMITS:
        assert lim.statement and len(lim.statement) > 40, lim.name
        assert lim.detail, lim.name                       # the measured evidence, never "it failed once"
        assert lim.minimal_case, lim.name                 # the case in this file that demonstrates it
        assert lim.decision in (ACCEPTED, "worked around", UNSUPPORTED), lim.name
        assert lim.solver in SOLVERS_IN_THE_ISSUE, lim.solver


def test_the_free_surface_sst_case_is_recorded_as_run_for_real():
    """The issue's own item: k-omega SST on a free surface had never been run. It now has the numbers to show it."""
    sst = [lim for lim in limits_for("interIsoFoam") if lim.name == "k-omega SST runs on a free surface"]
    assert len(sst) == 1
    detail = sst[0].detail
    assert "ExecutionTime" in detail and "alpha.water" in detail and "Courant" in detail


def test_what_is_not_claimed_is_declared_unsupported():
    """The air-side turbulence is the one thing decided against, and it says so instead of looking validated."""
    assert [lim.name for lim in unsupported()] == ["turbulence in the gas phase"]
    assert unsupported()[0].solver == "interIsoFoam"


def test_the_registry_serialises_and_an_unknown_solver_is_empty_not_an_error():
    assert len(as_dicts()) == len(LIMITS)
    assert {d["solver"] for d in as_dicts("buoyantPimpleFoam")} == {"buoyantPimpleFoam"}
    assert limits_for("simpleFoam") == () and decision("simpleFoam") == ""


# ----------------------------------------------------------------------------------------------- the rules
FOAM_HEAD = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2606                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def _case(files: dict[str, str]) -> dict[str, bytes]:
    """A case as the pre-flight takes it: {relative path: bytes} with the FoamFile headers already on."""
    out = {}
    for path, body in files.items():
        cls = "dictionary"
        if path.startswith("0/"):
            name = path.split("/", 1)[1]
            cls = "volVectorField" if name == "U" else "volScalarField"
        out[path] = (FOAM_HEAD.format(cls=cls, obj=path.split("/")[-1]) + body.strip() + "\n").encode()
    return out


FIELDS_SST = {
    "0/U": "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n"
           "    leftWall { type noSlip; }\n    rightWall { type noSlip; }\n    lowerWall { type noSlip; }\n"
           "    atmosphere { type pressureInletOutletVelocity; value uniform (0 0 0); }\n"
           "    defaultFaces { type empty; }\n}",
    "0/p_rgh": "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
               "    \"(leftWall|rightWall|lowerWall)\" { type fixedFluxPressure; value uniform 0; }\n"
               "    atmosphere { type totalPressure; p0 uniform 0; value uniform 0; }\n"
               "    defaultFaces { type empty; }\n}",
    "0/alpha.water": "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
                     "    \"(leftWall|rightWall|lowerWall)\" { type zeroGradient; }\n"
                     "    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }\n"
                     "    defaultFaces { type empty; }\n}",
    "0/k": "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0.0001;\nboundaryField\n{\n"
           "    \"(leftWall|rightWall|lowerWall)\" { type kqRWallFunction; value uniform 0.0001; }\n"
           "    atmosphere { type inletOutlet; inletValue uniform 0.0001; value uniform 0.0001; }\n"
           "    defaultFaces { type empty; }\n}",
    "0/omega": "dimensions [0 0 -1 0 0 0 0];\ninternalField uniform 10;\nboundaryField\n{\n"
               "    \"(leftWall|rightWall|lowerWall)\" { type omegaWallFunction; value uniform 10; }\n"
               "    atmosphere { type inletOutlet; inletValue uniform 10; value uniform 10; }\n"
               "    defaultFaces { type empty; }\n}",
    "0/nut": "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
             "    \"(leftWall|rightWall|lowerWall)\" { type nutkWallFunction; value uniform 0; }\n"
             "    atmosphere { type calculated; value uniform 0; }\n"
             "    defaultFaces { type empty; }\n}",
}


def free_surface_case(*, wall_dist: bool = True, omega_div: bool = True, omega_solver: bool = True):
    """An interFoam case with kOmegaSST: the three entries kOmegaSST needs and kEpsilon does not."""
    files = dict(FIELDS_SST)
    files["system/controlDict"] = (
        "application interFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 0.2;\n"
        "deltaT 0.001;\nadjustTimeStep on;\nmaxCo 1;\nwriteInterval 0.05;")
    schemes = ["ddtSchemes { default Euler; }", "gradSchemes { default Gauss linear; }",
               "divSchemes\n{\n    default none;\n    div(rhoPhi,U) Gauss linearUpwind grad(U);\n"
               "    div(phi,alpha) Gauss vanLeer;\n    div(phirb,alpha) Gauss linear;\n"
               "    div(rhoPhi,k) Gauss upwind;\n"
               + ("    div(rhoPhi,omega) Gauss upwind;\n" if omega_div else "")
               + "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}",
               "laplacianSchemes { default Gauss linear corrected; }",
               "interpolationSchemes { default linear; }", "snGradSchemes { default corrected; }"]
    if wall_dist:
        schemes.append("wallDist { method meshWave; }")
    files["system/fvSchemes"] = "\n\n".join(schemes)
    solvers = ['    p_rgh { solver PCG; preconditioner DIC; tolerance 1e-07; relTol 0.05; }',
               '    p_rghFinal { $p_rgh; relTol 0; }',
               '    "pcorr.*" { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }',
               '    "alpha.water.*" { nAlphaSubCycles 1; cAlpha 1; MULESCorr yes; solver smoothSolver; '
               'smoother symGaussSeidel; tolerance 1e-8; relTol 0; }']
    if omega_solver:
        solvers.append('    "(U|k|omega).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-06; '
                       'relTol 0; minIter 1; }')
    else:
        solvers.append('    "(U|k|epsilon).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-06; '
                       'relTol 0; minIter 1; }')
    files["system/fvSolution"] = ("solvers\n{\n" + "\n".join(solvers) + "\n}\n\n"
                                  "PIMPLE { momentumPredictor no; nOuterCorrectors 1; nCorrectors 3; }")
    files["constant/turbulenceProperties"] = "simulationType RAS;\nRAS { model kOmegaSST; turbulence on; }"
    files["constant/transportProperties"] = ("phases (water air);\nwater { transportModel Newtonian; nu 1e-06; "
                                            "rho 1000; }\nair { transportModel Newtonian; nu 1.48e-05; rho 1; }\n"
                                            "sigma 0.07;")
    return _case(files)


def transient_buoyant_case(*, rho_solver: bool = True, correct_phi: bool = True):
    files = {
        "system/controlDict": "application buoyantPimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                              "endTime 20;\ndeltaT 2;\nwriteInterval 10;",
        "system/fvSchemes": "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\n"
                            "divSchemes\n{\n    default none;\n    div(phi,U) Gauss upwind;\n"
                            "    div(phi,h) Gauss upwind;\n    div(phi,K) Gauss upwind;\n"
                            "    div(phi,k) Gauss upwind;\n    div(phi,omega) Gauss upwind;\n"
                            "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n"
                            "laplacianSchemes { default Gauss linear corrected; }\n"
                            "wallDist { method meshWave; }",
        "constant/turbulenceProperties": "simulationType RAS;\nRAS { model kOmegaSST; turbulence on; }",
        "constant/g": "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);",
        "constant/thermophysicalProperties": "thermoType { type heRhoThermo; mixture pureMixture; transport const; "
                                             "thermo hConst; equationOfState perfectGas; specie specie; energy "
                                             "sensibleEnthalpy; }\nmixture { specie { molWeight 28.96; } "
                                             "thermodynamics { Cp 1005; Hf 0; } transport { mu 1.8e-05; Pr 0.7; } }",
    }
    solvers = ['    p_rgh { solver GAMG; smoother DIC; tolerance 1e-07; relTol 0.01; }',
               '    p_rghFinal { $p_rgh; relTol 0; }']
    if rho_solver:
        solvers.append('    rho { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }')
    if correct_phi:
        solvers.append('    pcorr { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }')
    solvers.append('    "(U|h|k|omega).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; '
                   'relTol 0.1; }')
    files["system/fvSolution"] = (
        "solvers\n{\n" + "\n".join(solvers) + "\n}\n\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors 2;\n"
        f"    correctPhi {'yes' if correct_phi else 'no'};\n    pRefCell 0;\n    pRefValue 0;\n}}\n")
    return _case(files)


def test_a_free_surface_case_switched_to_komega_sst_fails_until_the_three_entries_are_there():
    """Each of the three failures the dam-break case hit when the model was switched (measured, v2606)."""
    rep = preflight_openfoam.check(free_surface_case(wall_dist=False, omega_div=False, omega_solver=False))
    f = findings(rep)
    assert f["wall_dist"].status == "fail" and "wallDist" in f["wall_dist"].title
    assert f["omega_schemes"].status == "fail" and "omega" in f["omega_schemes"].title
    assert f["omega_solver"].status == "fail"
    assert "dam-break" in (f["wall_dist"].detail + f["omega_schemes"].detail)

    ok = findings(preflight_openfoam.check(free_surface_case()))
    assert ok["wall_dist_ok"].status == "pass" and ok["omega_ok"].status == "pass"
    assert "wall_dist" not in ok and "omega_schemes" not in ok and "omega_solver" not in ok


def test_the_free_surface_case_passes_the_rules_the_model_switch_affects():
    """The point of the three entries: switching the model stops costing a run. (The dictionary-only case has no
    polyMesh, so the mesh and material rules still report; those are not what the model switch changed.)"""
    rep = preflight_openfoam.check(free_surface_case())
    switch_rules = {"wall_dist", "omega_schemes", "omega_solver"}
    assert [x.rule for x in rep.findings if x.status == "fail" and x.rule in switch_rules] == []
    assert {"wall_dist_ok", "omega_ok"} <= {x.rule for x in rep.findings if x.status == "pass"}


def test_a_transient_buoyant_case_needs_a_density_solver_and_the_start_up_correction():
    f = findings(preflight_openfoam.check(transient_buoyant_case(rho_solver=False, correct_phi=False)))
    assert f["rho_solver"].status == "fail" and "rho" in f["rho_solver"].title
    assert f["correct_phi"].status == "warn" and "correctPhi" in f["correct_phi"].title
    assert "12" in f["correct_phi"].evidence          # the measured factor, not a guess

    ok = findings(preflight_openfoam.check(transient_buoyant_case()))
    assert ok["rho_ok"].status == "pass" and "correct_phi" not in ok


def test_a_steady_buoyant_case_keeps_its_own_requirements():
    """buoyantSimpleFoam is not the transient solver: no rho entry is asked for, gravity and pRefCell still are."""
    files = {
        "system/controlDict": "application buoyantSimpleFoam;\nstopAt endTime;\nendTime 100;\ndeltaT 1;",
        "system/fvSchemes": "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
                            "divSchemes\n{\n    default none;\n    div(phi,U) Gauss upwind;\n"
                            "    div(phi,h) Gauss upwind;\n    div(phi,K) Gauss upwind;\n"
                            "    div(phi,k) Gauss upwind;\n    div(phi,omega) Gauss upwind;\n}\n"
                            "laplacianSchemes { default Gauss linear corrected; }",
        "system/fvSolution": "solvers\n{\n    p_rgh { solver GAMG; smoother DIC; tolerance 1e-07; relTol 0.01; }\n"
                             '    "(U|h|k|omega).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; '
                             'relTol 0.1; }\n}\n\nSIMPLE { pRefCell 0; pRefValue 0; }\n'
                             "relaxationFactors { equations { U 0.7; h 0.7; k 0.7; omega 0.7; } }",
        "constant/turbulenceProperties": "simulationType RAS;\nRAS { model kOmegaSST; turbulence on; }",
        "constant/g": "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);",
    }
    f = findings(preflight_openfoam.check(_case(files)))
    assert "rho_solver" not in f and "correct_phi" not in f


# ----------------------------------------------------------------------------------------------- real runs
def openfoam_bashrc() -> str | None:
    """The bashrc of an installed OpenFOAM, or None: the tests that run a solver skip instead of failing."""
    for cand in (os.environ.get("FOAM_BASHRC"), "/Applications/OpenFOAM-v2606.app/Contents/Resources/etc/bashrc"):
        if cand and os.path.isfile(cand):
            return cand
    for root in ("/Applications", "/opt", "/usr/lib/openfoam", "/usr/local"):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.lower().startswith("openfoam"):
                continue
            for tail in ("Contents/Resources/etc/bashrc", "etc/bashrc"):
                cand = os.path.join(root, name, tail)
                if os.path.isfile(cand):
                    return cand
    return None


FOAM = openfoam_bashrc()
needs_foam = pytest.mark.skipif(FOAM is None, reason="no OpenFOAM installation found (set FOAM_BASHRC)")


def sh(case, command: str, timeout: float = 300) -> subprocess.CompletedProcess:
    """Run one OpenFOAM command inside a case, with the environment sourced."""
    return subprocess.run(["bash", "-c", f'source "{FOAM}" >/dev/null 2>&1; cd "{case}" && {command}'],
                          capture_output=True, text=True, timeout=timeout)


CAVITY = {
    "system/blockMeshDict": "convertToMeters 1;\nvertices\n(\n    (0 0 0) (0.1 0 0) (0.1 0.1 0) (0 0.1 0)\n"
                            "    (0 0 0.01) (0.1 0 0.01) (0.1 0.1 0.01) (0 0.1 0.01)\n);\n"
                            "blocks ( hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1) );\n"
                            "boundary\n(\n    leftWall { type wall; faces ((0 4 7 3)); }\n"
                            "    rightWall { type wall; faces ((1 2 6 5)); }\n"
                            "    lowerWall { type wall; faces ((0 1 5 4)); }\n"
                            "    upperWall { type wall; faces ((3 7 6 2)); }\n"
                            "    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }\n);",
    "system/controlDict": "application buoyantSimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
                          "endTime 200;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 60;\n"
                          "writeFormat ascii;\nwritePrecision 6;\nrunTimeModifiable true;",
    "system/fvSchemes": "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
                        "divSchemes\n{\n    default none;\n    div(phi,U) Gauss upwind;\n"
                        "    div(phi,h) Gauss upwind;\n    div(phi,K) Gauss upwind;\n    div(phi,k) Gauss upwind;\n"
                        "    div(phi,omega) Gauss upwind;\n"
                        "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n"
                        "laplacianSchemes { default Gauss linear corrected; }\n"
                        "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n"
                        "wallDist { method meshWave; }",
    "system/fvSolution": "solvers\n{\n    p_rgh { solver GAMG; smoother DIC; tolerance 1e-07; relTol 0.01; }\n"
                         "    p_rghFinal { $p_rgh; relTol 0; }\n"
                         '    "(U|h|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; '
                         "relTol 0.1; }\n"
                         '    "(U|h|k|omega)Final" { $U; relTol 0; }\n}\n\n'
                         "SIMPLE\n{\n    momentumPredictor no;\n    nNonOrthogonalCorrectors 0;\n    pRefCell 0;\n"
                         "    pRefValue 0;\n    residualControl { p_rgh 1e-4; U 1e-5; \"(h|k|omega)\" 1e-4; }\n}\n\n"
                         "relaxationFactors\n{\n    equations { U 0.7; h 0.7; k 0.7; omega 0.7; }\n}",
    "constant/g": "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);",
    "constant/turbulenceProperties": "simulationType RAS;\nRAS { model kOmegaSST; turbulence on; printCoeffs on; }",
    "constant/thermophysicalProperties": "thermoType { type heRhoThermo; mixture pureMixture; transport const; "
                                         "thermo hConst; equationOfState perfectGas; specie specie; energy "
                                         "sensibleEnthalpy; }\nmixture { specie { molWeight 28.96; } "
                                         "thermodynamics { Cp 1005; Hf 0; } transport { mu 1.8e-05; Pr 0.7; } }",
    "0/U": "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n"
           "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type noSlip; }\n"
           "    frontAndBack { type empty; }\n}",
    "0/p": "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 100000;\nboundaryField\n{\n"
           "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type calculated; value uniform 100000; }\n"
           "    frontAndBack { type empty; }\n}",
    "0/p_rgh": "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 100000;\nboundaryField\n{\n"
               "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type fixedFluxPressure; value uniform 100000; }\n"
               "    frontAndBack { type empty; }\n}",
    "0/T": "dimensions [0 0 0 1 0 0 0];\ninternalField uniform 300;\nboundaryField\n{\n"
           "    leftWall { type fixedValue; value uniform 350; }\n"
           "    rightWall { type fixedValue; value uniform 300; }\n"
           "    lowerWall { type zeroGradient; }\n    upperWall { type zeroGradient; }\n"
           "    frontAndBack { type empty; }\n}",
    "0/k": "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0.0001;\nboundaryField\n{\n"
           "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type kqRWallFunction; value uniform 0.0001; }\n"
           "    frontAndBack { type empty; }\n}",
    "0/omega": "dimensions [0 0 -1 0 0 0 0];\ninternalField uniform 1;\nboundaryField\n{\n"
               "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type omegaWallFunction; value uniform 1; }\n"
               "    frontAndBack { type empty; }\n}",
    "0/nut": "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
             "    \"(leftWall|rightWall|lowerWall|upperWall)\" { type nutkWallFunction; value uniform 0; }\n"
             "    frontAndBack { type empty; }\n}",
    "0/alphat": "dimensions [1 -1 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
                "    \"(leftWall|rightWall|lowerWall|upperWall)\" "
                "{ type compressible::alphatWallFunction; value uniform 0; }\n"
                "    frontAndBack { type empty; }\n}",
}


def buoyant_cavity(root, **edits):
    """The minimal case of the registry: a 20x20x1 closed cavity, hot left wall, k-omega SST, 60 iterations."""
    files = dict(CAVITY)
    for path, fn in edits.items():                       # path with "/" replaced by "_" -> rewrite its text
        key = path.replace("__", "/")
        files[key] = fn(files[key])
    case = root / "cavity"
    for path, body in files.items():
        p = case / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(FOAM_HEAD.format(cls="volVectorField" if path.endswith("/U") else
                                      ("volScalarField" if path.startswith("0/") else "dictionary"),
                                      obj=path.split("/")[-1]) + body + "\n")
    return case


@needs_foam
def test_the_buoyant_cavity_runs_to_its_end_time_and_the_pressure_residual_falls(tmp_path):
    case = buoyant_cavity(tmp_path)
    assert sh(case, "blockMesh > log.blockMesh 2>&1").returncode == 0
    run = sh(case, "buoyantSimpleFoam > log.solver 2>&1")
    log = (case / "log.solver").read_text()
    assert run.returncode == 0 and "\nEnd\n" in log, log[-800:]
    first = float(re.search(r"Solving for p_rgh, Initial residual = ([0-9.e-]+)", log).group(1))
    last = float(re.findall(r"Solving for p_rgh, Initial residual = ([0-9.e-]+)", log)[-1])
    # a steady natural-convection cavity with k-omega SST converges slowly (0.49 -> 0.11 in 200 iterations, measured);
    # what the test pins is that it runs to the end time and the residual falls, not a convergence it does not have
    assert last < first / 3, f"p_rgh residual {first} -> {last}: the case is not converging"
    assert "Time = 200" in log


@needs_foam
def test_without_gravity_the_buoyant_solver_stops_at_start_up(tmp_path):
    case = buoyant_cavity(tmp_path, **{"constant/g": lambda s: ""})
    (case / "constant/g").unlink(missing_ok=True)
    assert sh(case, "blockMesh > log.blockMesh 2>&1").returncode == 0
    run = sh(case, "buoyantSimpleFoam > log.solver 2>&1")
    log = (case / "log.solver").read_text()
    assert run.returncode != 0
    assert 'cannot find file' in log and "constant/g" in log


@needs_foam
def test_without_a_pressure_reference_the_buoyant_solver_stops_at_start_up(tmp_path):
    case = buoyant_cavity(tmp_path, **{"system/fvSolution": lambda s: re.sub(r"[ \t]*pRef(Cell|Value)[^\n]*\n", "", s)})
    assert sh(case, "blockMesh > log.blockMesh 2>&1").returncode == 0
    run = sh(case, "buoyantSimpleFoam > log.solver 2>&1")
    log = (case / "log.solver").read_text()
    assert run.returncode != 0
    assert "Unable to set reference cell for field p" in log and "pRefCell or pRefPoint" in log


@needs_foam
def test_the_free_surface_case_runs_with_komega_sst(tmp_path):
    """The issue's open question, answered by a run: the dam-break case with the model switched, 5 steps."""
    tutorials = sh(tmp_path, 'echo "$FOAM_TUTORIALS"').stdout.strip()
    src = os.path.join(tutorials, "multiphase", "interFoam", "RAS", "damBreak", "damBreak") if tutorials else ""
    if not src or not os.path.isdir(src):
        pytest.skip("the interFoam damBreak tutorial is not installed")
    case = tmp_path / "dambreak"
    shutil.copytree(src, case)
    control = case / "system/controlDict"
    control.write_text(control.read_text().replace("endTime         1;", "endTime         0.02;"))
    tp = case / "constant/turbulenceProperties"
    tp.write_text(tp.read_text().replace("kEpsilon", "kOmegaSST"))
    (case / "0.orig/omega").write_text(
        (case / "0.orig/epsilon").read_text().replace("object      epsilon;", "object      omega;")
        .replace("dimensions      [0 2 -3 0 0 0 0];", "dimensions      [0 0 -1 0 0 0 0];")
        .replace("epsilonWallFunction", "omegaWallFunction").replace("uniform 0.1", "uniform 10"))
    (case / "0.orig/epsilon").unlink()
    schemes = case / "system/fvSchemes"
    t = schemes.read_text().replace("div(rhoPhi,epsilon)", "div(rhoPhi,omega)")
    # after the header, not before it: a dictionary's FoamFile block has to come first
    schemes.write_text(t + ("\nwallDist\n{\n    method meshWave;\n}\n" if "wallDist" not in t else ""))
    solution = case / "system/fvSolution"
    solution.write_text(solution.read_text().replace("(U|k|epsilon).*", "(U|k|omega).*"))

    shutil.copytree(case / "0.orig", case / "0")          # restore0Dir, which the tutorial's Allrun does first
    for cmd in ("blockMesh > log.blockMesh 2>&1", "setFields > log.setFields 2>&1",
                "interFoam > log.solver 2>&1"):
        assert sh(case, cmd).returncode == 0, cmd
    log = (case / "log.solver").read_text()
    assert "\nEnd\n" in log, log[-800:]
    fractions = [float(v) for v in re.findall(r"Phase-1 volume fraction = ([0-9.e-]+)", log)]
    assert fractions and fractions[-1] > 0.01, "alpha.water is empty: setFields did not run on the restored 0/"
    alphas = [float(v) for v in re.findall(r"Max\(alpha\.water\) = ([0-9.e-]+)", log)]
    assert alphas and 0.9 < alphas[-1] < 1.1
