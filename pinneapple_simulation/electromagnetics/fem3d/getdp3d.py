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

__all__ = ["available", "GetDPError", "run", "sphere"]

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


def team7_pro(f: float, points: dict[str, np.ndarray]) -> str:
    """TEAM 7 in GetDP: modified vector potential ``a*`` (``j = -sigma da/dt``), complex at frequency ``f``,
    ``n x a = 0`` on the outer surface (physical 9), tree gauge in the non-conducting region starting on the outer
    surface and the plate skin (physical 8). The coil current density is made discretely solenoidal as in
    :mod:`.nedelec`: a first system solves ``int grad(phi) . grad(phi') = int js . grad(phi')`` (``phi = 0`` on the
    outer surface) and the source is ``js - grad(phi)``. Prints ``b`` and ``j`` at the probe points."""
    from .team7 import AMPERE_TURNS, COIL_AREA, CORNERS, SIGMA

    x0, x1, y0, y1 = CORNERS
    J0 = AMPERE_TURNS / COIL_AREA
    pr = []
    for name, P in points.items():
        q = "b" if name in ("A1B1", "A2B2") else "j"
        reg = "Domain" if q == "b" else "Plate"
        for i, p in enumerate(P):
            pr.append(f'  Print[ {q}, OnPoint {{{p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}}}, Format Table, '
                      f'File > "{name}.txt" ];')
    return f"""
Group {{ Plate = Region[1]; Coil = Region[2]; Air = Region[3]; PlateSkin = Region[8]; Outer = Region[9];
  DomainC = Region[{{Plate}}]; DomainCC = Region[{{Coil, Air}}]; Domain = Region[{{DomainC, DomainCC}}];
  GaugeStart = Region[{{Outer, PlateSkin}}]; }}
Function {{
  mu0 = {MU0:.17g}; nu[] = 1/mu0; sigma[Plate] = {SIGMA:.17g};
  dx[] = X[] - Min[Max[X[], {x0}], {x1}]; dy[] = Y[] - Min[Max[Y[], {y0}], {y1}];
  js[Coil] = {J0:.17g} * Vector[-dy[], dx[], 0] / Sqrt[dx[]^2 + dy[]^2];
}}
Constraint {{
  {{ Name a; Case {{ {{ Region Outer; Value 0.; }} }} }}
  {{ Name phi; Case {{ {{ Region Outer; Value 0.; }} }} }}
  {{ Name Gauge; Case {{ {{ Region DomainCC; SubRegion GaugeStart; Value 0.; }} }} }}
}}
Jacobian {{ {{ Name Vol; Case {{ {{ Region All; Jacobian Vol; }} }} }} }}
Integration {{ {{ Name I1; Case {{ {{ Type Gauss; Case {{ {{ GeoElement Tetrahedron; NumberOfPoints 4; }}
  {{ GeoElement Triangle; NumberOfPoints 3; }} }} }} }} }} }}
FunctionSpace {{
  {{ Name Hcurl_a; Type Form1;
    BasisFunction {{ {{ Name se; NameOfCoef ae; Function BF_Edge; Support Domain; Entity EdgesOf[All]; }} }}
    Constraint {{ {{ NameOfCoef ae; EntityType EdgesOf; NameOfConstraint a; }}
                 {{ NameOfCoef ae; EntityType EdgesOfTreeIn; EntitySubType StartingOn; NameOfConstraint Gauge; }} }} }}
  {{ Name H1_phi; Type Form0;
    BasisFunction {{ {{ Name sn; NameOfCoef pn; Function BF_Node; Support Domain; Entity NodesOf[All]; }} }}
    Constraint {{ {{ NameOfCoef pn; EntityType NodesOf; NameOfConstraint phi; }} }} }}
}}
Formulation {{
  {{ Name Proj; Type FemEquation; Quantity {{ {{ Name phi; Type Local; NameOfSpace H1_phi; }} }}
    Equation {{ Galerkin {{ [ Dof{{d phi}}, {{d phi}} ]; In Domain; Jacobian Vol; Integration I1; }}
               Galerkin {{ [ -js[], {{d phi}} ]; In Coil; Jacobian Vol; Integration I1; }} }} }}
  {{ Name Eddy; Type FemEquation;
    Quantity {{ {{ Name a; Type Local; NameOfSpace Hcurl_a; }} {{ Name phi; Type Local; NameOfSpace H1_phi; }} }}
    Equation {{ Galerkin {{ [ nu[] * Dof{{d a}}, {{d a}} ]; In Domain; Jacobian Vol; Integration I1; }}
               Galerkin {{ DtDof [ sigma[] * Dof{{a}}, {{a}} ]; In DomainC; Jacobian Vol; Integration I1; }}
               Galerkin {{ [ -js[], {{a}} ]; In Coil; Jacobian Vol; Integration I1; }}
               Galerkin {{ [ {{d phi}}, {{a}} ]; In Domain; Jacobian Vol; Integration I1; }} }} }}
}}
Resolution {{ {{ Name Eddy;
  System {{ {{ Name P; NameOfFormulation Proj; }}
           {{ Name A; NameOfFormulation Eddy; Type Complex; Frequency {f:.17g}; }} }}
  Operation {{ Generate[P]; Solve[P]; SaveSolution[P]; Generate[A]; Solve[A]; SaveSolution[A]; }} }} }}
PostProcessing {{ {{ Name Post; NameOfFormulation Eddy; Quantity {{
  {{ Name b; Value {{ Local {{ [ {{d a}} ]; In Domain; Jacobian Vol; }} }} }}
  {{ Name j; Value {{ Local {{ [ -sigma[] * Dt[{{a}}] ]; In DomainC; Jacobian Vol; }} }} }}
  {{ Name loss; Value {{ Integral {{ [ 0.5 * sigma[] * SquNorm[Dt[{{a}}]] ]; In DomainC; Jacobian Vol; Integration I1; }} }} }}
}} }} }}
PostOperation {{ {{ Name Out; NameOfPostProcessing Post; Operation {{
  Print[ loss[DomainC], OnGlobal, Format Table, File "loss.txt" ];
""" + "\n".join(pr) + "\n} } }\n"


def _complex_rows(path: Path) -> np.ndarray:
    """``OnPoint`` table rows of a complex vector: the last six numbers are re/im of x, y, z (GetDP prints the
    real parts then the imaginary parts)."""
    rows = []
    for ln in path.read_text().splitlines():
        v = [float(t) for t in ln.split()]
        if len(v) >= 6:
            re, im = np.array(v[-6:-3]), np.array(v[-3:])
            rows.append(re + 1j * im)
    return np.array(rows)


def team7(msh, f: float, workdir=None) -> dict:
    """GetDP solution of TEAM 7 at ``f`` on the gmsh file ``msh`` (from :func:`~.team7.mesh_team7`): values on
    the measured lines in the units of :data:`~.team7.MEASURED` and the Joule loss."""
    from .team7 import probe_lines

    pts = probe_lines()
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="getdp_team7_"))
    tmp.mkdir(parents=True, exist_ok=True)
    for n in pts:
        (tmp / f"{n}.txt").unlink(missing_ok=True)
    r = run(team7_pro(f, pts), msh, "Eddy", "Out", ["loss.txt"], workdir=tmp, keep=True)
    out = {"loss_W": float(_numbers(tmp / "loss.txt")[1]), "seconds": r["seconds"]}   # "0 re im"
    for n in pts:
        v = _complex_rows(tmp / f"{n}.txt")
        v = v[:, 2] * 1e4 if n in ("A1B1", "A2B2") else v[:, 1] / 1e6
        out[(n, 0)] = np.real(v)
        out[(n, 90)] = -np.imag(v)
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
