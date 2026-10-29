"""GetDP as the external 3-D reference: the same gmsh meshes solved by GetDP's own edge-element formulations.

GetDP (https://getdp.info, GPL) is called as an executable (``apt install getdp``); no code of it is copied or
linked. Each problem file (``.pro``) is written here from the physical data only: GetDP's function spaces
(``BF_Edge``, Whitney 1-forms), its tree gauge (``EdgesOfTreeIn`` starting on the Dirichlet boundary), its
assembly, linear solver and post-processing integrals produce the numbers. Nothing of
:mod:`.nedelec` is used, so the agreement between the two measures implementation differences on identical
meshes.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

__all__ = ["available", "GetDPError", "run", "sphere", "sphere_pro"]

MU0 = 4e-7 * math.pi


class GetDPError(RuntimeError):
    pass


def available() -> bool:
    return shutil.which("getdp") is not None


def _numbers(path: Path) -> np.ndarray:
    return np.array([float(v) for v in path.read_text().split()])


def run(pro: str, msh, resolution: str, postop: str, outputs: list[str], workdir=None, timeout: float = 3600.0,
        keep: bool = False) -> dict:
    """Write ``pro``, run ``getdp m.pro -msh <msh> -solve <resolution> -pos <postop>`` and read the last number of
    each file in ``outputs``."""
    if not available():
        raise GetDPError("the getdp executable is required")
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="getdp3d_"))
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "m.pro").write_text(pro)
    t0 = time.perf_counter()
    r = subprocess.run(["getdp", "m.pro", "-msh", str(Path(msh).resolve()), "-solve", resolution, "-pos", postop,
                        "-v", "3"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
    missing = [o for o in outputs if not (tmp / o).exists()]
    if r.returncode or missing:
        raise GetDPError("getdp failed: " + r.stderr[-3000:] + r.stdout[-3000:])
    out = {o: _numbers(tmp / o)[-1] for o in outputs}
    out["seconds"] = time.perf_counter() - t0
    out["log"] = r.stdout[-4000:]
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _vec(v) -> str:
    return "Vector[" + ", ".join(f"{float(x):.17g}" for x in v) + "]"


def sphere_pro(case) -> str:
    """A-formulation magnetostatics of :class:`~.sphere.SphereCase` with the uniform-field boundary condition
    ``n x A = n x (B0 x r) / 2`` on the outer surface (physical 3): ``a = A0 + a_r`` with ``A0 = B0 x r / 2``
    and ``a_r = 0`` on the surface (GetDP's edge constraints evaluate a vector function without coordinates, so
    the non-zero boundary value is carried by ``A0``), tree gauge starting on the surface. Post-processed:
    mean field in the sphere, energy and L2 differences against the closed form."""
    m = case.m
    return f"""
Group {{ Sphere = Region[1]; Air = Region[2]; Outer = Region[3]; Domain = Region[{{Sphere, Air}}]; }}
Function {{
  mu0 = {MU0:.17g};
  nu[Sphere] = 1/(mu0*{case.mu_r:.17g}); nu[Air] = 1/mu0;
  B0[] = {_vec(case.B0)}; m[] = {_vec(m)};
  bex[Sphere] = {_vec(case.B_in)};
  bex[Air] = B0[] + mu0/(4*Pi) * (3 * XYZ[] * (XYZ[] * m[]) / Norm[XYZ[]]^2 - m[]) / Norm[XYZ[]]^3;
}}
Constraint {{
  {{ Name a; Case {{ {{ Region Outer; Value 0.; }} }} }}
  {{ Name Gauge; Case {{ {{ Region Domain; SubRegion Outer; Value 0.; }} }} }}
}}
Jacobian {{ {{ Name Vol; Case {{ {{ Region All; Jacobian Vol; }} }} }} }}
Integration {{ {{ Name I1; Case {{ {{ Type Gauss; Case {{ {{ GeoElement Tetrahedron; NumberOfPoints 4; }}
  {{ GeoElement Triangle; NumberOfPoints 3; }} }} }} }} }} }}
FunctionSpace {{ {{ Name Hcurl_a; Type Form1;
  BasisFunction {{ {{ Name se; NameOfCoef ae; Function BF_Edge; Support Domain; Entity EdgesOf[All]; }} }}
  Constraint {{ {{ NameOfCoef ae; EntityType EdgesOf; NameOfConstraint a; }}
               {{ NameOfCoef ae; EntityType EdgesOfTreeIn; EntitySubType StartingOn; NameOfConstraint Gauge; }} }}
}} }}
Formulation {{ {{ Name MagSta_a; Type FemEquation; Quantity {{ {{ Name a; Type Local; NameOfSpace Hcurl_a; }} }}
  Equation {{ Galerkin {{ [ nu[] * Dof{{d a}}, {{d a}} ]; In Domain; Jacobian Vol; Integration I1; }}
             Galerkin {{ [ nu[] * B0[], {{d a}} ]; In Domain; Jacobian Vol; Integration I1; }} }} }} }}
Resolution {{ {{ Name MagSta; System {{ {{ Name A; NameOfFormulation MagSta_a; }} }}
  Operation {{ Generate[A]; Solve[A]; SaveSolution[A]; }} }} }}
PostProcessing {{ {{ Name Post; NameOfFormulation MagSta_a; Quantity {{
  {{ Name vol; Value {{ Integral {{ [ 1 ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name bx; Value {{ Integral {{ [ CompX[{{d a}} + B0[]] ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name by; Value {{ Integral {{ [ CompY[{{d a}} + B0[]] ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name bz; Value {{ Integral {{ [ CompZ[{{d a}} + B0[]] ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name e2in; Value {{ Integral {{ [ SquNorm[{{d a}} + B0[] - bex[]] ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name n2in; Value {{ Integral {{ [ SquNorm[bex[]] ]; In Sphere; Jacobian Vol; Integration I1; }} }} }}
  {{ Name e2; Value {{ Integral {{ [ SquNorm[{{d a}} + B0[] - bex[]] ]; In Domain; Jacobian Vol; Integration I1; }} }} }}
  {{ Name n2; Value {{ Integral {{ [ SquNorm[bex[]] ]; In Domain; Jacobian Vol; Integration I1; }} }} }}
  {{ Name w; Value {{ Integral {{ [ 0.5 * nu[] * SquNorm[{{d a}} + B0[]] ]; In Domain; Jacobian Vol; Integration I1; }} }} }}
}} }} }}
PostOperation {{ {{ Name Out; NameOfPostProcessing Post; Operation {{
""" + "\n".join(f'  Print[ {q}[{"Sphere" if q in ("vol", "bx", "by", "bz", "e2in", "n2in") else "Domain"}], '
                 f'OnGlobal, Format Table, File "{q}.txt" ];'
                 for q in ("vol", "bx", "by", "bz", "e2in", "n2in", "e2", "n2", "w")) + "\n} } }\n"


def sphere(msh, case, workdir=None) -> dict:
    """GetDP solution of the sphere case on the gmsh file ``msh``: the same error measures as
    :func:`~.sphere.errors`."""
    q = ("vol", "bx", "by", "bz", "e2in", "n2in", "e2", "n2", "w")
    r = run(sphere_pro(case), msh, "MagSta", "Out", [f"{n}.txt" for n in q], workdir=workdir)
    v = {n: r[f"{n}.txt"] for n in q}
    Bm = np.array([v["bx"], v["by"], v["bz"]]) / v["vol"]
    return {"mean_in": float(np.linalg.norm(Bm - case.B_in) / np.linalg.norm(case.B_in)),
            "l2_in": float(math.sqrt(v["e2in"] / v["n2in"])), "l2_all": float(math.sqrt(v["e2"] / v["n2"])),
            "Bz_mean_in": float(Bm[2]), "energy": v["w"], "seconds": r["seconds"]}

