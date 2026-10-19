"""Rebuild the fixtures of ``examples/physics_check``: a passing and a failing case for every rule of
``pinneapple check``.

    python examples/physics_check/generate.py            # everything (OpenFOAM cases only if icoFoam is found)
    python examples/physics_check/generate.py --no-foam  # keep the committed OpenFOAM results

The OpenFOAM cases are real icoFoam runs (case files written here; the OpenFOAM run needs ``icoFoam`` and
``blockMesh`` on the PATH, e.g. the ``openfoam`` package of Ubuntu, sourced from /usr/share/openfoam/etc/bashrc).
The other results are analytical fields with known physics (and deliberate defects in ``fail/``), so every check
has an exact expectation. The models are small NumPy functions standing in for a trained PINN and a surrogate.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PASS, FAIL = HERE / "pass", HERE / "fail"

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def foam_file(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj) + "\n" + body.strip() + "\n")


# ---------------------------------------------------------------------------------------------- OpenFOAM cases
def write_case(case: Path, *, length: float, height: float, nx: int, ny: int, patches: dict[str, tuple[str, str]],
               u_bc: dict[str, str], p_bc: dict[str, str], nu: float, dt: float, end: float, write_every: int) -> None:
    """A 2-D box case (one cell thick). ``patches``: name -> (type, faces); faces by the box sides."""
    if case.exists():
        shutil.rmtree(case)
    sides = {"left": "(0 4 7 3)", "right": "(2 6 5 1)", "bottom": "(1 5 4 0)", "top": "(3 7 6 2)"}
    bnd = []
    for name, (typ, which) in patches.items():
        faces = " ".join(sides[w] for w in which.split())
        bnd.append(f"    {name}\n    {{\n        type {typ};\n        faces ( {faces} );\n    }}")
    bnd.append("    frontAndBack\n    {\n        type empty;\n        faces ( (0 3 2 1) (4 5 6 7) );\n    }")
    foam_file(case / "system/blockMeshDict", "dictionary", "blockMeshDict", f"""
scale 1;
vertices
(
    (0 0 0) ({length} 0 0) ({length} {height} 0) (0 {height} 0)
    (0 0 0.1) ({length} 0 0.1) ({length} {height} 0.1) (0 {height} 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1) );
edges ();
boundary
(
{chr(10).join(bnd)}
);
mergePatchPairs ();
""")
    foam_file(case / "system/controlDict", "dictionary", "controlDict", f"""
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          {dt};
writeControl    timeStep;
writeInterval   {write_every};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
""")
    foam_file(case / "system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes   { default orthogonal; }
""")
    foam_file(case / "system/fvSolution", "dictionary", "fvSolution", """
solvers
{
    p     { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0.01; }
    pFinal { $p; relTol 0; }
    U     { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0; }
}
PISO { nCorrectors 3; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
""")
    foam_file(case / "constant/transportProperties", "dictionary", "transportProperties",
              f"nu              [0 2 -1 0 0 0 0] {nu};")

    def field(obj, cls, dims, internal, bcs):
        body = [f"dimensions      {dims};", "", f"internalField   {internal};", "", "boundaryField", "{"]
        for name, spec in bcs.items():
            body.append(f"    {name}\n    {{\n        {spec}\n    }}")
        body.append("    frontAndBack\n    {\n        type            empty;\n    }")
        body.append("}")
        foam_file(case / "0" / obj, cls, obj, "\n".join(body))

    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", u_bc)
    field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0", p_bc)


def run_case(case: Path, keep: int = 2) -> None:
    env = "/usr/share/openfoam/etc/bashrc"
    pre = f". {env} >/dev/null 2>&1; " if Path(env).exists() else ""
    for app in ("blockMesh", "icoFoam"):
        r = subprocess.run(["bash", "-c", f"{pre}{app} -case {case}"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{app} failed on {case}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    times = sorted((d for d in case.iterdir() if d.is_dir() and _num(d.name) and d.name != "0"),
                   key=lambda d: float(d.name))
    for d in times[:-keep]:
        shutil.rmtree(d)
    for d in times[-keep:]:
        shutil.rmtree(d / "uniform", ignore_errors=True)


def _num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def cavity(case: Path, lid: str = "(1 0 0)") -> None:
    write_case(case, length=1.0, height=1.0, nx=20, ny=20,
               patches={"movingWall": ("wall", "top"), "fixedWalls": ("wall", "left right bottom")},
               u_bc={"movingWall": f"type            fixedValue;\n        value           uniform {lid};",
                     "fixedWalls": "type            noSlip;"},
               p_bc={"movingWall": "type            zeroGradient;", "fixedWalls": "type            zeroGradient;"},
               nu=0.01, dt=0.01, end=5, write_every=10)


def channel(case: Path) -> None:
    write_case(case, length=4.0, height=1.0, nx=40, ny=10,
               patches={"inlet": ("patch", "left"), "outlet": ("patch", "right"), "walls": ("wall", "top bottom")},
               u_bc={"inlet": "type            fixedValue;\n        value           uniform (1 0 0);",
                     "outlet": "type            zeroGradient;", "walls": "type            noSlip;"},
               p_bc={"inlet": "type            zeroGradient;",
                     "outlet": "type            fixedValue;\n        value           uniform 0;",
                     "walls": "type            zeroGradient;"},
               nu=0.05, dt=0.02, end=10, write_every=25)


def foam_cases() -> None:
    cavity(PASS / "cases/cavity")
    run_case(PASS / "cases/cavity")
    channel(PASS / "cases/channel")
    run_case(PASS / "cases/channel")
    cavity(FAIL / "cases/cavity", lid="(2 0 0)")           # the lid speed typed as 2 m/s instead of 1 m/s
    run_case(FAIL / "cases/cavity")


def cavity_reference() -> None:
    """Cell centres and fields of the passing cavity's last time: the regression reference."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from pinneapple_check.fields import load_openfoam
    fd = load_openfoam(PASS / "cases/cavity")
    for d in (PASS, FAIL):
        (d / "reference").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "reference/cavity.npz", points=fd.geom["cell_centres"], U=fd.fields["U"],
                            p=fd.fields["p"], units=np.array(["U=m/s", "p=m2/s2"]))


# ---------------------------------------------------------------------------------------------- grids
def heat_plate(path: Path, *, broken: bool) -> None:
    """Steady conduction with a uniform source q in the unit square, k = 10 W/(m K), q = 1000 W/m3.
    Passing: T = 300 + q/(4k) (2 - x^2 - y^2) solves k lap T + q = 0. Failing: a linear profile that carries
    no net heat, so the source has nowhere to go (energy imbalance)."""
    x = np.linspace(0, 1, 41)
    y = np.linspace(0, 1, 41)
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = 300 + 50 * X if broken else 300 + 1000 / 40 * (2 - X ** 2 - Y ** 2)
    np.savez_compressed(path, x=x, y=y, T=T, units=np.array(["x=m", "y=m", "T=K"]))


def bar_gravity(path: Path, *, factor: float) -> None:
    """Plane-strain block (1 m x 1 m) under its own weight, fixed at y = 0, free at y = 1: u_y = f/(lam+2mu)
    (H y - y^2/2) with f = -rho g. ``factor`` 2 = the load applied twice."""
    E, nu, rho, g = 2e9, 0.3, 2000.0, 9.81
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    f = -rho * g * factor
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, y, indexing="ij")
    uy = f / (lam + 2 * mu) * (1.0 * Y - Y ** 2 / 2)
    np.savez_compressed(path, x=x, y=y, D=np.stack([np.zeros_like(uy), uy], -1),
                        units=np.array(["x=m", "y=m", "D=m"]))


def diffusion_1d(path: Path, *, leaky: bool) -> None:
    """q_t + F_x = 0 with F = -D q_x on [0, 1], insulated ends: q = 1 + exp(-pi^2 D t) cos(pi x) keeps its total.
    ``leaky``: the same field damped by exp(-0.2 t), as a dissipative scheme would do."""
    D = 0.05
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 2, 81)
    X, Tt = np.meshgrid(x, t, indexing="ij")
    q = 1 + np.exp(-np.pi ** 2 * D * Tt) * np.cos(np.pi * X)
    F = D * np.pi * np.exp(-np.pi ** 2 * D * Tt) * np.sin(np.pi * X)
    if leaky:
        q = q * np.exp(-0.2 * Tt)
        F = F * np.exp(-0.2 * Tt)
    np.savez_compressed(path, x=x, t=t, q=q, F=F)


def transient_plate(path: Path, *, wrong_start: bool) -> None:
    """1-D transient conduction (alpha = 1e-2 m2/s): T = 300 + 20 exp(-pi^2 alpha t) sin(pi x). The declared
    initial condition is T(x, 0) = 300 + 20 sin(pi x); ``wrong_start`` stores a field that started 10 K hotter."""
    a = 1e-2
    x = np.linspace(0, 1, 81)
    t = np.linspace(0, 5, 51)
    X, Tt = np.meshgrid(x, t, indexing="ij")
    T = 300 + 20 * np.exp(-np.pi ** 2 * a * Tt) * np.sin(np.pi * X)
    if wrong_start:
        T = T + 10 * np.exp(-Tt * 50)
    np.savez_compressed(path, x=x, t=t, T=T, units=np.array(["x=m", "t=s", "T=K"]))


def vof_tank(path: Path, *, overshoot: bool) -> None:
    """A phase fraction on an unstructured triangle mesh (VTU): a water layer with a smooth interface.
    ``overshoot``: the unbounded advection a VOF solver without limiter produces (alpha up to 1.04, down to -0.03)."""
    import meshio
    n = 25
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 0.5, n), indexing="ij")
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            tri += [[a, b, c], [a, c, d]]
    alpha = 0.5 * (1 - np.tanh((pts[:, 1] - 0.2 - 0.05 * np.sin(2 * np.pi * pts[:, 0])) / 0.02))
    if overshoot:
        alpha = alpha + 0.04 * np.sin(12 * np.pi * pts[:, 0]) * np.exp(-((pts[:, 1] - 0.2) / 0.05) ** 2)
    meshio.write(path, meshio.Mesh(pts, [("triangle", np.array(tri))], point_data={"alpha.water": alpha}))


def porosity(path: Path, *, out_of_range: bool) -> None:
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = 0.35 + 0.1 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    if out_of_range:
        phi[12:15, 3:6] = 0.72                                     # a clipped-region bug of a mesh mapper
    np.savez_compressed(path, x=x, y=y, porosity=phi)


def surrogate_field(path: Path, *, nan: bool) -> None:
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = 300 + 25 * X * Y
    if nan:
        T[29, 29] = np.nan                                         # a surrogate evaluated where its input overflowed
    np.savez_compressed(path, x=x, y=y, T=T, units=np.array(["x=m", "y=m", "T=K"]))


def mixing_field(path: Path, ref_path: Path, *, blurred: bool) -> None:
    """A multi-scale scalar (spectrum ~ k^-5/3, random phases) and its reference; the failing copy is the same field
    through a Gaussian filter of one cell: a small relative L2 error, most of the small-scale energy gone (PC704)."""
    n = 64
    rng = np.random.default_rng(7)
    m = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(m, m, indexing="ij")
    K = np.hypot(KX, KY)
    amp = np.where(K > 0, np.maximum(K, 1.0) ** (-(5 / 3 + 1) / 2), 0.0)
    c = 300 + 2.0 * np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * amp).real / 0.05
    x = np.linspace(0, 1, n, endpoint=False)
    np.savez_compressed(ref_path, x=x, y=x, c=c)
    if blurred:
        k2 = (2 * np.pi * KX / n) ** 2 + (2 * np.pi * KY / n) ** 2
        c = 300 + np.fft.ifft2(np.fft.fft2(c - 300) * np.exp(-0.5 * k2)).real
    np.savez_compressed(path, x=x, y=x, c=c)


def darcy_dataset(path: Path, *, defects: bool) -> None:
    """Darcy pairs (a, u) on 16 x 16 cells in the zoo layout (``train__a`` ...). The failing set has the three
    defects of #700: u scaled by 0.9 (does not solve its PDE), permeability samples drawn with one shared Fourier
    coefficient (two distinct fields) and a test sample copied from the training set."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    n = 16
    a = ob.threshold_coefficient(ob.periodic_grf(50, (n, n), alpha=2.0, tau=3.0, seed=11))
    if defects:
        g = ob.periodic_grf(1, (n, n), alpha=2.0, tau=3.0, seed=11)[0]
        a = ob.threshold_coefficient(np.random.default_rng(1).standard_normal(50)[:, None, None] * g[None])
    u = np.stack([ob.darcy_2d(ai, 1.0) for ai in a]) * (0.9 if defects else 1.0)
    if defects:
        a[45], u[45] = a[3], u[3]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train__a=a[:40], train__u=u[:40], test__a=a[40:], test__u=u[40:])


def temperature_table(path: Path, unit: str, celsius: bool) -> None:
    """A probe export (CSV of points): ``celsius`` writes degC values."""
    x = np.linspace(0, 1, 15)
    y = np.linspace(0, 1, 15)
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = 330 - 20 * X - 10 * Y
    if celsius:
        T = T - 273.15
    lines = [f"x [m],y [m],T [{unit}]"] + [f"{a:.6g},{b:.6g},{c:.6g}" for a, b, c in zip(X.ravel(), Y.ravel(), T.ravel(), strict=False)]
    path.write_text("\n".join(lines) + "\n")


def qoi(path: Path, cd: float, cl: float) -> None:
    path.write_text(f"name,value,unit\nCd,{cd},1\nCl,{cl},1\nx_separation,{0.62},m\n")


# ---------------------------------------------------------------------------------------------- models
KOVASZNAY = '''"""A stand-in for a PINN trained on Kovasznay flow (Re = 40): the exact solution plus a small smooth error, as a
trained network would have. ``predict`` takes (N, 2) inputs x, y and returns (N, 3) outputs u, v, p."""
import numpy as np

RE = 40.0
LAM = RE / 2 - np.sqrt(RE ** 2 / 4 + 4 * np.pi ** 2)


def predict(X):
    x, y = X[:, 0], X[:, 1]
    e = np.exp(LAM * x)
    u = (1 - e * np.cos(2 * np.pi * y)) * (1 + {u_growth} * (x + 0.5))
    v = LAM / (2 * np.pi) * e * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * LAM * x))
    err = 1e-4 * np.sin(3 * x) * np.cos(2 * y)
    return np.column_stack([u + err, v - err, p + err])
'''

DRAG = '''"""A stand-in for a drag surrogate Cd(Re, aoa) fitted on wind-tunnel points (Re 1e5 .. 1e6, aoa 0 .. 10 deg)."""
import numpy as np


def predict(X):
    re, aoa = X[:, 0], X[:, 1]
    return (0.006 + 0.9 / np.sqrt(re) * 10 + 1.2e-4 * aoa ** 2).reshape(-1, 1)
'''


PEAK = '''"""A stand-in for a surrogate of the peak temperature of a slab with uniform heating q, thickness L and
conductivity k, cooled to T0 on both faces (exact: T0 + q L^2 / (8 k)).{note}"""


def predict(X):
    q, L, k, T0 = X.T
    return (T0 + q * L ** 2 / (8 * {k_term})).reshape(-1, 1)
'''


def peak_temperature(path: Path, *, sign_error: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PEAK.format(note=" A unit slip divides by 1/k instead of k." if sign_error else "",
                                k_term="(1e4 / k)" if sign_error else "k"))


def kovasznay_reference(path: Path) -> None:
    g = np.random.default_rng(3)
    lam = 20 - np.sqrt(400 + 4 * np.pi ** 2)
    x, y = g.uniform(-0.5, 1.0, 200), g.uniform(-0.5, 1.5, 200)
    u = 1 - np.exp(lam * x) * np.cos(2 * np.pi * y)
    v = lam / (2 * np.pi) * np.exp(lam * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * lam * x))
    rows = ["x,y,u,v,p"] + [f"{a:.8g},{b:.8g},{c:.8g},{d:.8g},{e:.8g}" for a, b, c, d, e in zip(x, y, u, v, p, strict=False)]
    path.write_text("\n".join(rows) + "\n")


def drag_samples(path: Path) -> None:
    """Training points: a 15 x 15 grid in (log Re, aoa) with a round hole in the middle (no tunnel data there)."""
    rows = ["Re,aoa"]
    for i, lr in enumerate(np.linspace(5, 6, 15)):
        for j, a in enumerate(np.linspace(0, 10, 15)):
            if (i / 14 - 0.5) ** 2 + (j / 14 - 0.5) ** 2 < 0.3 ** 2:
                continue
            rows.append(f"{10 ** lr:.6g},{a:.6g}")
    path.write_text("\n".join(rows) + "\n")


def queries(path: Path, *, bad: bool) -> None:
    rows = ["Re,aoa", "1.5e5,2", "3e5,1", "8e5,8", "1e6,10"]
    if bad:
        rows += ["2.5e6,4", "6e5,14", "3.16e5,5"]    # above the Re range; above the aoa range; in the data hole
    path.write_text("\n".join(rows) + "\n")


class _Evil:
    """Pickles to a call of ``os.system`` (never executed here: the checker must refuse it unread)."""

    def __reduce__(self):
        return (os.system, ("true",))


def unsafe_checkpoint(path: Path) -> None:
    path.write_bytes(pickle.dumps({"state_dict": {}, "hook": _Evil()}, protocol=2))


def integrity(base: Path, *, tamper: bool) -> None:
    """Manifest of ``models/`` and an Ed25519 signature of the surrogate (public key committed, private key not)."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from pinneapple_security.integrity import build_manifest
    from pinneapple_security.signing import generate_key, sign_file
    models = base / "models"
    for f in [*models.glob("*.sig"), models / "notes.txt"]:
        f.unlink(missing_ok=True)
    man = build_manifest(str(models), exclude=(".git", "__pycache__", "*.sig", "*.sig.json", "manifest.json",
                                               "attestation.json"))
    man.save(str(models / "manifest.json"))
    key = generate_key()
    (base / "keys").mkdir(exist_ok=True)
    (base / "keys/release.pub.json").write_text(json.dumps(key.public().to_dict(), indent=1) + "\n")
    sign_file(str(models / "drag_surrogate.py"), key)
    if tamper:                       # edited after the manifest and the signature, plus a file nobody recorded
        p = models / "drag_surrogate.py"
        p.write_text(p.read_text().replace("1.2e-4", "1.5e-4"))
        (models / "notes.txt").write_text("retrained on the new tunnel campaign\n")


def models(base: Path, *, broken: bool) -> None:
    m = base / "models"
    m.mkdir(parents=True, exist_ok=True)
    (m / "kovasznay_pinn.py").write_text(KOVASZNAY.format(u_growth="0.1" if broken else "0.0"))
    (m / "drag_surrogate.py").write_text(DRAG)
    kovasznay_reference(base / "reference/kovasznay.csv")
    drag_samples(m / "drag_training_points.csv")
    queries(m / "queries.csv", bad=broken)
    if broken:
        unsafe_checkpoint(m / "legacy_model.pt")


def torch_fixture() -> None:
    """A safe PINNeAPPle checkpoint (weights only) whose metadata records the training range."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    import torch
    from pinneapple_neural.architectures.registry import ModelRegistry
    torch.manual_seed(0)
    d = HERE / "torch"
    (d / "models").mkdir(parents=True, exist_ok=True)
    m = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden=[16, 16])
    m.save_checkpoint(str(d / "models/plate_pinn.pt"), metadata={"domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 0.5]}})
    (d / "models/queries.csv").write_text("x,y\n0.1,0.1\n0.5,0.25\n0.9,0.45\n")
    (d / "models/queries_out.csv").write_text("x,y\n0.5,0.25\n1.2,0.25\n")
    # a pre-trained pp.zoo model queried inside the validity range of its card
    (d / "models/heatsink_queries.csv").write_text(
        "base_width_mm,base_depth_mm,base_thickness_mm,n_fins,fin_thickness_mm,fin_height_mm,k_w_mk,"
        "source_width_mm,source_depth_mm,t_ambient_c,air_velocity_m_s,power_w\n"
        "80,80,6,15,1.5,35,200,30,30,25,2.5,80\n"
        "120,100,8,20,1.2,40,380,40,40,35,4.0,150\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-foam", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    a = ap.parse_args()
    for d in (PASS, FAIL):
        for sub in ("fields", "qoi", "reference", "models"):
            (d / sub).mkdir(parents=True, exist_ok=True)
    if not a.no_foam:
        foam_cases()
    cavity_reference()
    for d, bad in ((PASS, False), (FAIL, True)):
        heat_plate(d / "fields/heat_plate.npz", broken=bad)
        bar_gravity(d / "fields/block_gravity.npz", factor=2.0 if bad else 1.0)
        diffusion_1d(d / "fields/diffusion_1d.npz", leaky=bad)
        transient_plate(d / "fields/cooling_plate.npz", wrong_start=bad)
        vof_tank(d / "fields/tank_alpha.vtu", overshoot=bad)
        porosity(d / "fields/porosity.npz", out_of_range=bad)
        surrogate_field(d / "fields/surrogate_T.npz", nan=bad)
        mixing_field(d / "fields/mixing_c.npz", d / "reference/mixing_c.npz", blurred=bad)
        darcy_dataset(d / "datasets/darcy.npz", defects=bad)
        temperature_table(d / "fields/probe_T.csv", "degC" if bad else "K", celsius=bad)
        qoi(d / "qoi/airfoil.csv", cd=0.0129 if bad else 0.01231, cl=0.802)
        qoi(d / "reference/airfoil.csv", cd=0.01230, cl=0.801)
        models(d, broken=bad)
        peak_temperature(d / "surrogates/peak_temperature.py", sign_error=bad)
        integrity(d, tamper=bad)
    np.savez_compressed(FAIL / "fields/unlabelled_T.npz", x=np.linspace(0, 1, 20), y=np.linspace(0, 1, 20),
                        T=300 + np.linspace(0, 5, 20)[:, None] * np.ones((20, 20)))
    (FAIL / "fields/corrupt.vtu").write_text("<VTKFile type='UnstructuredGrid'><Piece>truncated")
    np.savez_compressed(FAIL / "fields/stream.npz", x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10),
                        psi=np.outer(np.linspace(0, 1, 10), np.linspace(0, 1, 10)))
    if not a.no_torch:
        torch_fixture()
    print("fixtures written to", PASS, "and", FAIL)


if __name__ == "__main__":
    main()
