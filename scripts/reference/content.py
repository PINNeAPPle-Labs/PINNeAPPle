"""Curated content of the module reference: what each package is for, when to use it, and examples that run.

Each example is executed by ``run_examples.py`` in a fresh interpreter; its printed output and the matplotlib
figures it leaves open are shown under the code. Plot only when the picture says more than the numbers.
"""
from __future__ import annotations

GROUPS = [
    ("foundations", "Foundations"),
    ("physics", "Physics and simulation"),
    ("learning", "Learning"),
    ("trust", "Trust and analysis"),
    ("data", "Data"),
    ("systems", "Systems and design"),
    ("show", "Visualisation"),
    ("workflow", "Workflow and infrastructure"),
    ("frontier", "Research modules"),
    ("legacy", "Compatibility shims"),
]

PACKAGES: list[dict] = []


def pkg(**kw):
    PACKAGES.append(kw)


# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple",
    group="foundations",
    title="pinneapple",
    tagline="The short front door: one import for the most used entry points.",
    about=[
        "`import pinneapple as pp` gives the shortcuts that most scripts need without remembering which subpackage "
        "holds them: solve and compare physics problems, run solvers, train models, render and export results, "
        "and `pp.fea` for CalculiX studies built in Python.",
        "Everything it exposes lives in one of the packages below; the shortcut and the full path are the same "
        "object, so you can start short and move to the full path when you need the details.",
    ],
    use_when=["You are writing a script or notebook and want the common entry points.",
              "You want to see what the library covers before choosing a subpackage."],
    examples=[dict(
        title="What the front door exposes",
        code='''import pinneapple as pp

names = [n for n in dir(pp) if not n.startswith("_")]
print(len(names), "public names, e.g.:")
print(", ".join(sorted(names)[:40]))
print("pp.fea ->", pp.fea.__name__)
''')],
)

pkg(
    name="pinneapple_core",
    group="foundations",
    title="Core primitives",
    tagline="Domain, Geometry, Mesh and Field: one vocabulary for tensors, geometry and topology.",
    about=[
        "Physics AI keeps passing the same things around: a region, points sampled in it, values on a grid, a mesh "
        "or a point cloud, and derivatives of those values. `pinneapple_core` gives them one interface.",
        "A `Field` answers `gradient()`, `divergence()`, `interpolate(x)` and `integrate()` whatever it is stored on "
        "(regular grid, simplex mesh with exact P1 operators, or scattered points with local least squares). "
        "`FunctionField` gives the same interface to a torch callable such as a PINN, through autograd. The free "
        "functions in `operators` (`grad`, `div`, `curl`, `laplacian`, `integrate`, `flux`) work on all of them.",
    ],
    use_when=["You need derivatives or integrals of a field and do not want to care how it is discretised.",
              "You are writing a solver, loss or dataset that should accept grids, meshes and point clouds alike."],
    examples=[dict(
        title="Gradient and integral of a field on a grid, checked against calculus",
        code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_core import Field, Domain

x = np.linspace(0, 1, 101); y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(x, y, indexing="ij")
u = np.sin(np.pi * X) * np.sin(np.pi * Y)
f = Field.on_grid([x, y], u, name="u")

g = f.gradient().values.reshape(101, 101, 2)   # (points, component, dim) -> grid
exact = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
print("max |du/dx - exact| =", float(np.abs(g[..., 0] - exact).max()))
print("integral =", float(f.integrate()), " exact 4/pi^2 =", 4 / np.pi ** 2)

d = Domain.box([0, 0], [1, 1])
pts = d.sample_interior(5, seed=0)
print("u at random points:", np.round(f.interpolate(pts), 4))

fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for a, val, t in [(ax[0], u, "u = sin(pi x) sin(pi y)"), (ax[1], np.hypot(g[..., 0], g[..., 1]), "|grad u|")]:
    im = a.pcolormesh(X, Y, val, shading="auto", cmap="viridis"); a.set_title(t); a.set_aspect("equal")
    fig.colorbar(im, ax=a, shrink=0.85)
''', plot=True)],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_physics",
    group="physics",
    title="Physics problems and PINNs",
    tagline="Define a physical problem once, solve it with a PINN, a solver or a closed form, and compare.",
    about=[
        "`pde_environment` describes a problem as a `ProblemSpec`: the PDE terms, the boundary and initial "
        "conditions, data constraints and scales, with presets for Navier-Stokes, heat, wave, Burgers, elasticity, "
        "Helmholtz and RANS turbulence models. `pinn_solver` compiles a spec into differentiable losses, and "
        "`symbolic_pde` turns SymPy expressions into autograd residuals.",
        "Not every engineering question needs a PDE. `closed_form` holds textbook models (fins, beams and fatigue, "
        "Helmholtz resonators, heat sinks, PCB thermal) as fast ground truth, and `tribology` solves sliding wear "
        "with contact-pressure redistribution. Domain packages (`weather`, `blackhole`, `pcb`) build on the same "
        "pieces.",
    ],
    use_when=["You want a PINN for a known PDE with its boundary conditions written down once.",
              "You need a quick, citable reference value to check a solver or a surrogate against."],
    examples=[
        dict(title="Closed-form engineering checks: cantilever fatigue and a Helmholtz port",
             code='''from pinneapple_physics.closed_form.cantilever_fatigue import (
    cantilever_root_bending_stress_pa, basquin_fatigue_life_cycles)
from pinneapple_physics.closed_form.helmholtz_resonator import helmholtz_resonant_frequency_hz, port_volume_m3

s = cantilever_root_bending_stress_pa(width_m=0.02, thickness_m=0.004, load_n=60.0, length_m=0.15)
print(f"root bending stress: {s / 1e6:.1f} MPa")          # 6 F L / (b t^2)
life = basquin_fatigue_life_cycles(s, fatigue_strength_coeff_pa=900e6, fatigue_strength_exponent=-0.09)
print(f"Basquin life at that amplitude: {life:.3g} cycles")
f = helmholtz_resonant_frequency_hz(port_radius_m=0.035, port_length_m=0.12, cabinet_volume_l=40.0)
print(f"bass-reflex tuning of a 40 L box with a 70 mm x 120 mm port: {f:.1f} Hz")
'''),
        dict(title="Sliding wear of a bar end (Archard with pressure redistribution)",
             code='''import matplotlib.pyplot as plt
from pinneapple_physics.tribology import BarWear, WEAR_MATERIALS

bar = BarWear(WEAR_MATERIALS["60/40 brass"], load_N=50.0)
res = bar.run(distance_m=400.0, n_save=6)
print(f"specific wear rate K = {bar.material.K:.2e} mm^3/(N m)")
print(f"worn volume {res['volume'][-1]:.3f} mm^3  vs  K F s = {bar.material.K * 50.0 * 400.0:.3f} mm^3")
print(f"peak pressure {res['p_max'][0]:.1f} -> {res['p_max'][-1]:.2f} MPa (flat punch F/A = {50 / 200:.2f} MPa)")

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
for s, p in zip(res["s_saved"], res["pressures"]):
    ax[0].plot(res["x"], p, label=f"s = {s:.0f} m")
ax[0].set_xlabel("x along the bar end (mm)"); ax[0].set_ylabel("contact pressure (MPa)"); ax[0].legend(fontsize=7)
ax[1].plot(res["s"], res["p_max"]); ax[1].axhline(0.25, ls="--", c="k", lw=0.8)
ax[1].set_xlabel("sliding distance (m)"); ax[1].set_ylabel("peak pressure (MPa)"); ax[1].set_yscale("log")
ax[1].set_title("running-in to the flat-punch pressure")
''', plot=True),
    ],
)

pkg(
    name="pinneapple_simulation",
    group="physics",
    title="Simulation",
    tagline="Numerical solvers, particle dynamics and bridges to OpenFOAM, CalculiX, FEniCS, MATLAB and FMUs.",
    about=[
        "`numerical_solvers` holds the in-house solvers: finite differences, finite volumes, lattice Boltzmann, "
        "spectral and FFT solvers, a 3-D finite-element solver with incompatible-mode hexahedra (`solid_fem`), "
        "internal pipe and mixer flow on OpenFOAM (`internal_flow`), DEM particles in a stirred tank "
        "(`particles`), and dataset generation for PINNs.",
        "`external_solvers` drives real tools when they are installed (OpenFOAM, CalculiX through `pp.fea`, "
        "FEniCS, MATLAB, Modelica/FMU) and degrades gracefully when they are not. `particle_dynamics` has "
        "differentiable rigid bodies, MPM and SPH in PyTorch.",
        "Every solver used in the lab is verified on a case with a known answer before its results are used.",
    ],
    use_when=["You need reference fields or training data from a real solver.",
              "You want an FEA, CFD or particle result scripted from Python."],
    examples=[
        dict(title="3-D cantilever with C3D8I hexahedra, against beam theory",
             code='''import numpy as np
from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh, SolidFEM, fea_figure

L, W, H, F, E = 2.0, 0.1, 0.2, 1e4, 210e9
mesh = box_mesh(L, W, H, 40, 3, 6)
fem = SolidFEM(mesh, E=E, nu=0.3)
fem.fix(mesh.nodes_on(x=0.0))
fem.load_face(mesh.face_nodes("x+"), (0, 0, -F))
res = fem.solve()

I = W * H ** 3 / 12
G = E / 2.6
theory = F * L ** 3 / (3 * E * I) + F * L / (5 / 6 * G * W * H)    # Euler-Bernoulli + shear (Timoshenko)
print(f"tip deflection {res.max_displacement * 1e3:.3f} mm, Timoshenko {theory * 1e3:.3f} mm")
print(f"max von Mises {res.von_mises.max() / 1e6:.1f} MPa, beam theory at the root {F * L * H / 2 / I / 1e6:.1f} MPa")
fig = fea_figure(res, "von_mises")
''', plot=True),
        dict(title="Pipe-flow correlations used to check the CFD",
             code='''from pinneapple_simulation.numerical_solvers.internal_flow import colebrook, ito_bend_loss
from pinneapple_simulation.numerical_solvers.particles import StirredTank, terminal_velocity, zwietering_njs

for Re in (1e4, 1e5, 1e6):
    print(f"Re {Re:8.0e}: Darcy f = {colebrook(Re):.4f}, 90 deg bend (R/r = 3) K = {ito_bend_loss(Re, 3.0):.3f}")
print(f"terminal velocity of a 3 mm, 1200 kg/m3 bead in water: {terminal_velocity(3e-3, 1200.0) * 100:.1f} cm/s")
tank = StirredTank(R=0.075, H=0.15)
print(f"Zwietering just-suspended speed: {zwietering_njs(tank, 3e-3, 1200.0, 0.05):.0f} rpm")
'''),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_neural",
    group="learning",
    title="Neural models",
    tagline="Architectures, training and inference: PINNs, neural operators, GNNs and reduced-order models.",
    about=[
        "`architectures` has the model families behind one registry (`ModelRegistry.build(name, ...)`): PINN "
        "networks (MLP, SIREN, modified MLP, Fourier features, hash grids, PINNsFormer), neural operators (FNO, "
        "AFNO, DeepONet and multi-scale DeepONet), graph networks (MeshGraphNet, equivariant GNNs) and "
        "reduced-order models (POD, DMD, Operator Inference, SINDy, HAVOK, Koopman, parametric POD-RBF/GPR).",
        "`trainer` gives the training loops (two-phase Adam then L-BFGS, time marching, causal weighting, DDP and "
        "FSDP), loss balancing (GradNorm, NTK, self-adaptive weights) and callbacks. `predictor` evaluates trained "
        "models on grids and draws the results.",
    ],
    use_when=["You want a surrogate of a solver, or a PINN for a PDE.",
              "You want a reduced-order model of snapshot data with an honest baseline next to it."],
    examples=[
        dict(title="POD + DMD of a travelling wave: compress, then forecast",
             code='''import numpy as np, torch
import matplotlib.pyplot as plt
from pinneapple_neural.architectures.rom import POD, DynamicModeDecomposition

x = np.linspace(0, 2 * np.pi, 256); t = np.arange(0, 12, 0.05)
U = np.array([np.sin(x - 1.3 * s) + 0.5 * np.sin(3 * x + 2.1 * s) for s in t])   # (T, D)
X = torch.tensor(U, dtype=torch.float64)

pod = POD(r=8).fit(X[:160])
print("POD energy in 4 modes:", float(pod.explained_variance_ratio_[:4].sum()))

dmd = DynamicModeDecomposition(r=4, center=False).fit(X[:160])
pred = dmd.rollout(X[159:160], steps=len(t) - 160)[0, 1:]
err = torch.linalg.norm(pred - X[160:]) / torch.linalg.norm(X[160:])
print(f"DMD forecast error over {len(t) - 160} unseen steps: {float(err):.2e}")
print("DMD frequencies (Hz):", np.round(np.sort(np.abs(dmd.frequencies(dt=0.05).numpy())), 4))

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
ax[0].imshow(U[160:], aspect="auto", cmap="RdBu_r", extent=[0, 2 * np.pi, t[-1], t[160]]); ax[0].set_title("truth")
ax[1].imshow(pred.numpy(), aspect="auto", cmap="RdBu_r", extent=[0, 2 * np.pi, t[-1], t[160]]); ax[1].set_title("DMD forecast")
ax[0].set_ylabel("t (unseen)"); [a.set_xlabel("x") for a in ax]
''', plot=True),
        dict(title="Parametric POD with Gaussian-process uncertainty, against the nearest-design baseline",
             code='''import numpy as np
from pinneapple_neural.architectures.rom import ParametricPOD, latin_hypercube

x = np.linspace(0, 1, 200)
def field(p):                                         # deflection-like response of two design parameters
    a, b = p
    return a * np.sin(np.pi * x) + b * x ** 2 * (1 - x)

P, names = latin_hypercube(40, {"a": (0.5, 2.0), "b": (-1.0, 1.0)}, seed=0)
Xs = np.array([field(p) for p in P])
Pt, _ = latin_hypercube(20, {"a": (0.5, 2.0), "b": (-1.0, 1.0)}, seed=1)
Xt = np.array([field(p) for p in Pt])

gpr = ParametricPOD(regressor="gpr").fit(P, Xs)
near = ParametricPOD(regressor="nearest").fit(P, Xs)
print("modes kept:", gpr.rank, " energy:", round(gpr.energy_captured, 8))
print(f"median error  POD-GPR {np.median(gpr.error(Pt, Xt)):.2e}   nearest design {np.median(near.error(Pt, Xt)):.2e}")
_, std = gpr.predict(Pt[:1], return_std=True)
print("predictive std available:", std.shape)
'''),
        dict(title="DeepONet: branch net on sensor values, trunk net on query points",
             code='''import torch
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet

torch.manual_seed(0)
net = DeepONet(branch_dim=32, trunk_dim=1, out_dim=1, hidden=64, modes=32, depth=3)
u = torch.randn(8, 32)                        # 8 input functions sampled at 32 sensors
xq = torch.linspace(0, 1, 100)[:, None]       # 100 query points shared by the batch
y = net(u, xq).y
print("output:", tuple(y.shape), " parameters:", sum(p.numel() for p in net.parameters()))
'''),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_lab",
    group="trust",
    title="Lab",
    tagline="Run experiments reproducibly, check them, and grow a database of results, datasets and trust cards.",
    about=[
        "An experiment is a class with parameters and a `run(ctx)` method. The runner gives every run an id from "
        "the experiment, its version and its parameters, and records inputs, outputs, metrics, checks, figures "
        "and datasets in one folder indexed in SQLite.",
        "Checks have a kind (physics, reference, baseline, sanity, generalisation, uncertainty), and curation "
        "turns them into a tier (A to D) and a six-stage trust card: data and geometry, model, physical "
        "constraints, benchmark, uncertainty and engineering decision. `uq` adds coverage of predictive intervals "
        "and grid convergence indices. `python -m pinneapple_lab site` writes the public catalogue.",
    ],
    use_when=["A result will be shown to someone else, and its evidence should travel with it.",
              "You run the same study over many parameters and want to query and compare the runs."],
    examples=[
        dict(title="A minimal experiment with a reference check",
             code='''import json, math, os, tempfile
from pinneapple_lab import Experiment, register, run

@register
class Pendulum(Experiment):
    name = "pendulum_doc_example"
    params = {"L": 1.0, "theta0": 0.05}
    def run(self, ctx):
        L, th = ctx.params["L"], ctx.params["theta0"]
        T = 2 * math.pi * math.sqrt(L / 9.81) * (1 + th ** 2 / 16)       # small-angle period + first correction
        ctx.metric("period_s", T)
        ctx.check("period_matches_small_angle", value=T / (2 * math.pi * math.sqrt(L / 9.81)),
                  min=0.999, max=1.002, kind="reference", detail="T vs 2 pi sqrt(L/g)")

root = tempfile.mkdtemp()
r = run("pendulum_doc_example", {"L": 2.0}, root=root)
print(r.status, {k: round(v, 4) for k, v in r.metrics.items()})
checks = json.load(open(os.path.join(r.dir, "validation.json")))
print([(c["name"], c["passed"], c["kind"]) for c in checks])
print("run folder:", sorted(os.listdir(r.dir)))
'''),
        dict(title="Grid convergence index of a mesh study (Celik et al. 2008)",
             code='''from pinneapple_lab.uq import gci, coverage
import numpy as np

g = gci([2.512, 2.571, 2.591], h=[4.0, 2.0, 1.0])          # coarse -> fine
print(f"observed order {g['order']:.2f}, extrapolated {g['extrapolated']:.4f}, GCI {100 * g['gci']:.2f} %")

rng = np.random.default_rng(0)
y = rng.normal(size=2000); mean = np.zeros(2000)
print("coverage of a calibrated 90 % interval:", round(coverage(y, mean, np.ones(2000), 0.9), 3))
print("coverage of an over-confident one:     ", round(coverage(y, mean, 0.5 * np.ones(2000), 0.9), 3))
'''),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_analysis",
    group="trust",
    title="Analysis",
    tagline="Uncertainty, validation, inverse problems, equation discovery, state estimation and trust scores.",
    about=[
        "`uncertainty` covers aleatoric heads, MC dropout, deep ensembles, conformal prediction, quantile "
        "regression and calibration metrics, behind one `uq_predict` entry point. `validation` checks physical "
        "consistency: conservation, boundary conditions, symmetry and agreement with analytical or solver "
        "references.",
        "`inverse_problems` estimates parameters from observations (noise models, Tikhonov / TV regularisation "
        "with L-curve, local and Sobol sensitivity, ensemble Kalman inversion) and discovers missing terms with "
        "SINDy. `state_estimation` has extended and ensemble Kalman filters. `trust` scores a model's predictions "
        "before anyone consumes them.",
    ],
    use_when=["A prediction needs an interval that actually holds its nominal coverage.",
              "You have measurements and want the parameters or the missing physics behind them."],
    examples=[
        dict(title="Conformal prediction: intervals with guaranteed coverage around any model",
             code='''import torch
from pinneapple_analysis import ConformalPredictor

torch.manual_seed(0)
x = torch.rand(3000, 1) * 6
y = torch.sin(x) + 0.2 * torch.randn_like(x)
model = lambda z: torch.sin(z) * 0.9                 # a slightly biased surrogate

cp = ConformalPredictor(model, alpha=0.1)
cp.calibrate(x[:1000], y[:1000])
pred, lo, hi = cp.predict(x[1000:])
print(f"interval half-width {cp.quantile:.3f}")
print(f"empirical coverage on 2000 unseen points: {cp.coverage(x[1000:], y[1000:]):.3f}  (target 0.90)")
'''),
        dict(title="SINDy: recover the missing term of a damped oscillator from data",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_analysis import CandidateLibrary, SINDyIdentifier

t = np.linspace(0, 20, 2000)
x = np.exp(-0.1 * t) * np.cos(2 * t)
v = np.gradient(x, t, edge_order=2); a = np.gradient(v, t, edge_order=2)
t, x, v, a = (z[5:-5] for z in (t, x, v, a))        # finite differences are least accurate at the ends
# known part of the model: a = -4 x ; the residual holds what the model misses
b = a + 4 * x
Theta, names = CandidateLibrary(poly_order=2).build(np.column_stack([x, v]))
names = [n.replace("x0", "x").replace("x1", "v") for n in names]
res = SINDyIdentifier(threshold=0.05).fit(Theta, b, names)
print(res.equation("a + 4x"))
print("true:  a + 4x = -0.2 v - 0.01 x   (x = exp(-0.1 t) cos 2t solves x'' + 0.2 x' + 4.01 x = 0)")

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(t, b, lw=2, label="residual from data"); ax.plot(t, Theta @ res.coefficients, "--", label="SINDy model")
ax.set_xlabel("t"); ax.legend(); ax.set_title("discovered damping term")
''', plot=True),
        dict(title="Extended Kalman filter tracking a pendulum from noisy angle readings",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_analysis.state_estimation.kalman import ExtendedKalmanFilter

dt, g_L = 0.02, 9.81
f = lambda s: np.array([s[0] + dt * s[1], s[1] - dt * g_L * np.sin(s[0])])
h = lambda s: s[:1]
rng = np.random.default_rng(1)
truth = [np.array([1.0, 0.0])]
for _ in range(400):
    truth.append(f(truth[-1]))
truth = np.array(truth)
obs = truth[:, :1] + rng.normal(0, 0.15, (len(truth), 1))

ekf = ExtendedKalmanFilter(2, 1, f, h, Q=np.eye(2) * 1e-5, R=np.eye(1) * 0.15 ** 2)
ekf.initialize(np.array([0.5, 0.0]), np.eye(2))
est = []
for y in obs:
    ekf.step(y); est.append(ekf.x.copy())
est = np.array(est)
print(f"angle RMS error: raw readings {np.sqrt(np.mean((obs[:, 0] - truth[:, 0]) ** 2)):.3f} rad,"
      f" filter {np.sqrt(np.mean((est[100:, 0] - truth[100:, 0]) ** 2)):.3f} rad (after spin-up)")
print(f"angular velocity is never measured; filter RMS error {np.sqrt(np.mean((est[100:, 1] - truth[100:, 1]) ** 2)):.3f} rad/s")

tt = np.arange(len(truth)) * dt
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(tt, obs[:, 0], ".", ms=2, c="0.6", label="readings"); ax.plot(tt, truth[:, 0], label="truth")
ax.plot(tt, est[:, 0], "--", label="EKF"); ax.set_xlabel("t (s)"); ax.set_ylabel("angle (rad)"); ax.legend(ncol=3, fontsize=7)
''', plot=True),
    ],
)

pkg(
    name="pinneapple_data",
    group="data",
    title="Data",
    tagline="Physical samples, collocation, datasets, sharded storage, preflight and lineage.",
    about=[
        "`PhysicalSample` and the UPD format keep fields together with their coordinates, units and metadata; "
        "Zarr stores and sharded iterables stream them to training. `CollocationSampler` draws PINN points "
        "(uniform, Latin hypercube, Sobol or residual-adaptive) inside boxes, SDF shapes, meshes or STL files, "
        "and `active_learning` refines them where the residual or the variance is high.",
        "`cae` reads meshes and fields from CAE formats and measures mesh quality; `preflight` checks a case "
        "before it is solved; `lineage` records where every dataset came from; `datasets` is the registry of "
        "built-in and public datasets.",
    ],
    use_when=["You need collocation points for a PINN in a non-trivial region.",
              "You are building a training set from solver runs and want units, splits and provenance kept."],
    examples=[
        dict(title="Collocation points in a box, Latin hypercube versus uniform random",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_data import CollocationSampler

fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4))
for a, strategy in zip(ax, ("uniform", "lhs")):
    s = CollocationSampler.from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)}, strategy=strategy, seed=3)
    batch = s.sample(n_col=200, n_bc=0)
    pts = np.column_stack([batch["x_col"][:, 0], batch["x_col"][:, 1]]) if "x_col" in batch else None
    if pts is None:
        print("batch keys:", list(batch.keys())); break
    a.plot(pts[:, 0], pts[:, 1], ".", ms=4); a.set_title(strategy); a.set_aspect("equal")
    # discrepancy proxy: largest empty cell of a 10 x 10 grid
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=10, range=[[0, 1], [0, 1]])
    print(f"{strategy:8s}: empty cells of a 10x10 grid = {(H == 0).sum()}")
''', plot=True),
    ],
)

pkg(
    name="pinneapple_design",
    group="systems",
    title="Design",
    tagline="Geometry from signed distance functions, physics domains, airfoils, meshes and design optimisation.",
    about=[
        "`geometry` builds shapes as signed distance functions (20+ primitives, Boolean and smooth Boolean "
        "operations, transforms), turns them into physics domains with named boundaries (channel, cavity, pipe, "
        "L-shape, annulus, T-junction, any SDF), meshes them and samples them for 3-D PINNs. It also generates "
        "NACA airfoils and imports STL / STEP.",
        "`design_optimizer` closes the loop: shape parametrisation, adjoint gradients, surrogates, Bayesian and "
        "evolutionary optimisers, Pareto fronts and manufacturing constraints. `aero`, `thermal` and "
        "`qualitative` add domain models and a quick preview of how a geometry change should move the physics.",
    ],
    use_when=["You need a domain with holes or blends for a PINN or a mesh.",
              "You want to optimise a shape against a physics objective."],
    examples=[
        dict(title="A bracket from signed distance functions",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_design import sdf2d_rectangle, sdf2d_circle, sdf_difference, sdf_smooth_union

n = 300
X, Y = np.meshgrid(np.linspace(-1.2, 1.2, n), np.linspace(-0.8, 0.8, n))
P = np.column_stack([X.ravel(), Y.ravel()])
plate = sdf2d_rectangle(P, (0.0, 0.0), (1.0, 0.25))
boss = sdf2d_circle(P, (-0.8, 0.0), 0.45)
body = sdf_smooth_union(plate, boss, k=0.15)
holes = np.minimum(sdf2d_circle(P, (-0.8, 0.0), 0.2), sdf2d_circle(P, (0.7, 0.0), 0.12))
d = sdf_difference(body, holes).reshape(n, n)

area = (d < 0).mean() * 2.4 * 1.6
print(f"area of the part: {area:.3f}")
print(f"signed distance at the centre of the big hole: {d[n // 2, int((-0.8 + 1.2) / 2.4 * n)]:.3f} (radius 0.2 -> +0.2)")
fig, ax = plt.subplots(figsize=(7, 3.4))
im = ax.contourf(X, Y, d, levels=30, cmap="RdBu"); ax.contour(X, Y, d, levels=[0], colors="k")
ax.set_aspect("equal"); fig.colorbar(im, ax=ax, label="signed distance"); ax.set_title("bracket = plate U boss - holes")
''', plot=True),
        dict(title="NACA four-digit airfoils",
             code='''import matplotlib.pyplot as plt
from pinneapple_design import naca_parametric

fig, ax = plt.subplots(figsize=(7, 2.4))
for m, p, t, name in [(0, 0, 0.12, "0012"), (0.02, 0.4, 0.12, "2412"), (0.04, 0.4, 0.15, "4415")]:
    xy = naca_parametric(m=m, p=p, t_c=t, n_pts=120).numpy()
    ax.plot(xy[:, 0], xy[:, 1], label=f"NACA {name}")
    print(f"NACA {name}: {xy.shape[0]} points, max thickness {xy[:, 1].max() - xy[:, 1].min():.3f} c")
ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_xlabel("x / c")
''', plot=True),
    ],
)

pkg(
    name="pinneapple_systems",
    group="systems",
    title="Systems",
    tagline="Time series, co-simulation, digital twins, component and process models.",
    about=[
        "`time_series` forecasts physical signals with classical and neural models (LSTM, GRU, N-BEATS, TCN, TFT, "
        "gradient boosting, FFT and HHT hybrids), always next to naive, seasonal-naive and drift baselines, with "
        "rolling backtests, audits and plots.",
        "`cosimulation` composes analytical nodes, PINNs and symbolic PDEs into a differentiable graph. "
        "`digital_twin` wraps a surrogate as a live twin with sensor streams (MQTT, Kafka, HTTP, files), Kalman "
        "assimilation and anomaly detection. `component_modeling`, `component_library` and `process_components` "
        "hold reusable equipment models.",
    ],
    use_when=["You forecast a sensor signal and need to know whether you beat a naive forecast.",
              "You couple several models (or a model and live data) into one system."],
    examples=[
        dict(title="Baselines every forecaster must beat",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_systems import NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster

rng = np.random.default_rng(0)
t = np.arange(240)
y = 20 + 0.03 * t + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, t.size)   # hourly temperature
train, test = y[:216], y[216:]
fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.plot(t[150:], y[150:], c="k", lw=1, label="measured")
for model in (NaiveForecaster(), SeasonalNaiveForecaster(season_length=24), DriftForecaster()):
    p = model.fit(train).predict(24)
    print(f"{type(model).__name__:26s} MAE over 24 h: {np.mean(np.abs(p - test)):.2f}")
    ax.plot(t[216:], p, "--", label=type(model).__name__)
ax.axvline(216, c="0.5", lw=0.8); ax.legend(fontsize=7); ax.set_xlabel("hour")
''', plot=True),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_tools",
    group="show",
    title="Tools",
    tagline="Visualisation, the 3-D studio, model export, benchmarking, dataset quality and compute back-ends.",
    about=[
        "`visualization` draws CFD-style figures (scalar and vector fields, streamlines, vorticity, Q-criterion "
        "and lambda-2, meshes, point clouds, voxels, loss histories, animations), and `visualization.studio` "
        "makes Blender renders, particle videos with live charts, glTF / USD exports and browser viewers.",
        "`model_export` writes TorchScript, ONNX, CSV and NPZ; `benchmark_suite` runs models on registered tasks; "
        "`dataset_quality` audits a dataset before training; `compute_backends` bridges PyTorch and JAX.",
    ],
    use_when=["You need a publication or post-processor style figure from arrays.",
              "You want to ship a trained model to another runtime."],
    examples=[
        dict(title="Vorticity and Q-criterion of a Taylor-Green vortex",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_tools.visualization import compute_vorticity_2d, compute_q_criterion_2d

n = 128
x = np.linspace(0, 2 * np.pi, n); y = np.linspace(0, 2 * np.pi, n)
X, Y = np.meshgrid(x, y)
u = np.sin(X) * np.cos(Y); v = -np.cos(X) * np.sin(Y)
w = compute_vorticity_2d(u, v, x, y)
q = compute_q_criterion_2d(u, v, x, y)
print(f"max |vorticity| {np.abs(w).max():.3f} (exact 2), Q > 0 in {100 * (q > 0).mean():.0f} % of the box")

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
ax[0].contourf(X, Y, w, 40, cmap="RdBu_r"); ax[0].streamplot(x, y, u, v, color="k", density=0.8, linewidth=0.5)
ax[0].set_title("vorticity + streamlines"); ax[1].contourf(X, Y, q, 40, cmap="viridis"); ax[1].set_title("Q-criterion")
[a.set_aspect("equal") for a in ax]
''', plot=True),
        dict(title="Export a model to TorchScript and check it reproduces the outputs",
             code='''import os, tempfile, torch
from pinneapple_tools import export_torchscript

net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
x = torch.rand(5, 2)
path = os.path.join(tempfile.mkdtemp(), "model.pt")
out = export_torchscript(net, path, example_input=x)
loaded = torch.jit.load(out if isinstance(out, str) else path)
print("exported:", os.path.basename(path), f"{os.path.getsize(path) / 1024:.1f} kB")
print("max difference after reload:", float((loaded(x) - net(x)).abs().max()))
'''),
    ],
)

pkg(
    name="pinneapple_twin3d",
    group="show",
    title="3-D digital twin",
    tagline="Build a 3-D scene in Python, with fields over time and sensors, and open it in the bundled viewer.",
    about=[
        "A `Scene` holds parts (triangle meshes), fields on them over time (wear, temperature, pressure), sensors "
        "with time series and alarm envelopes. `export` writes `scene.json`, binary glTF geometry, field buffers "
        "and the three.js viewer (optionally OpenUSD with time-sampled fields); `serve` opens it locally and can "
        "take live sensor values over a websocket.",
        "`openfoam.scene_from_case` builds a scene from the wall patches of an OpenFOAM case, so CFD results "
        "(erosion, wall shear) show up on the real geometry.",
    ],
    use_when=["You want stakeholders to see a field on the real part, over time, in a browser.",
              "You need glTF / USD of a result for another 3-D tool."],
    examples=[
        dict(title="A pipe bend with five years of wear, exported for the browser viewer",
             code='''import os, tempfile, json
from pinneapple_twin3d.demo import pipe_bend_scene

sc = pipe_bend_scene()
folder = os.path.join(tempfile.mkdtemp(), "twin")
sc.export(folder)
meta = json.load(open(os.path.join(folder, "scene.json")))
print("files:", sorted(os.listdir(folder))[:8])
print("parts:", [p["name"] for p in meta.get("parts", [])][:5], " times:", meta.get("times"))
print("open it with: pinneapple_twin3d.serve(folder)")
'''),
    ],
)

pkg(
    name="pinneapple_blender",
    group="show",
    title="Blender bridge",
    tagline="Export fields and trajectories as PLY sequences and build Blender scenes from them.",
    about=[
        "`export` writes point clouds and trajectories as PLY files with colours and extra scalar properties, in "
        "pure Python. `render` drives a local Blender installation through a subprocess to build and save a "
        "scene, the same pattern the studio uses for Cycles renders.",
    ],
    use_when=["You want a cinematic render of particles or a field in Blender."],
    examples=[
        dict(title="Write a coloured point cloud with a scalar property",
             code='''import os, tempfile
import numpy as np
from pinneapple_blender import write_ply

rng = np.random.default_rng(0)
pts = rng.normal(size=(1000, 3)).astype(np.float32)
speed = np.linalg.norm(pts, axis=1)
col = (255 * np.column_stack([speed / speed.max(), 0.3 * np.ones(1000), 1 - speed / speed.max()])).astype(np.uint8)
path = os.path.join(tempfile.mkdtemp(), "cloud.ply")
write_ply(path, pts, colors=col, scalar_fields={"speed": speed})
print(open(path, "rb").read(260).split(b"end_header")[0].decode())
'''),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_arena",
    group="learning",
    title="Arena",
    tagline="Benchmark many models on one physics problem from a YAML or a dict, with the same data and metrics.",
    about=[
        "The arena builds every model from the neural registry, trains each one the way its family needs "
        "(collocation losses for PINNs, full grids for FNO, sensor-to-field for DeepONet, graphs for GNNs), "
        "evaluates them on the same reference and ranks them by accuracy or by a physics-aware score.",
        "Problems come from a registry (`list_problems()`), from datasets (`benchmark_dataset`), or from "
        "`define_problem` for your own; `search_architecture` runs a small architecture search.",
    ],
    use_when=["You need to choose an architecture and want the comparison to be fair.",
              "You report a model and want the baselines in the same table."],
    examples=[
        dict(title="Registered benchmark problems by domain",
             code='''from pinneapple_arena import list_problems, list_problems_by_domain

print(len(list_problems()), "problems")
for domain, names in sorted(list_problems_by_domain().items()):
    print(f"{domain:22s} {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}")
'''),
    ],
)

pkg(
    name="pinneapple_adaptation",
    group="learning",
    title="Adaptation",
    tagline="Transfer learning and meta-learning for families of related physics problems.",
    about=[
        "`transfer_learning` fine-tunes a trained PINN on a related problem: layer freezing, progressive "
        "unfreezing, discriminative learning rates, interpolation across a parametric family and MMD domain "
        "adaptation. `meta_learning` trains a shared initialisation (MAML, Reptile) that adapts to a new member "
        "of a PDE family in a few gradient steps.",
    ],
    use_when=["You solve the same PDE for many parameter values and want each new one to start warm."],
    examples=[
        dict(title="Freeze all but the last layer before fine-tuning",
             code='''import torch
from pinneapple_adaptation import freeze_all_except, count_trainable

net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                          torch.nn.Linear(64, 1))
print("trainable before:", count_trainable(net))
freeze_all_except(net, ["4"])                 # keep only the output layer trainable
print("trainable after: ", count_trainable(net))
'''),
    ],
)

pkg(
    name="pinneapple_quantum",
    group="frontier",
    title="Quantum",
    tagline="Variational quantum circuits as physics-informed models: VQ-PINN, hybrid PINN, Hamiltonian learning.",
    about=[
        "A variational circuit is the ansatz of a wavefunction and the loss is the Rayleigh quotient (the VQE "
        "objective) or a PDE residual, on simulators or real hardware through pluggable back-ends. Presets cover "
        "the Schrodinger equation in 1-D and 2-D (ground and excited states), hybrid classical-quantum networks, "
        "Hamiltonian learning from energy data and quantum-kernel PINNs.",
        "Exact solutions (harmonic oscillator, spin chains) ship with the package so every quantum result has a "
        "reference.",
    ],
    use_when=["You research quantum machine learning for physics and need a reference problem with an exact answer."],
    examples=[
        dict(title="Exact harmonic-oscillator states used as the reference",
             code='''import torch
import matplotlib.pyplot as plt
from pinneapple_quantum import exact_eigenstates_1d_harmonic

x = torch.linspace(-5, 5, 400)[:, None]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for n in range(4):
    psi, E = exact_eigenstates_1d_harmonic(x, n_state=n)
    norm = float(torch.trapz(psi.squeeze() ** 2, x.squeeze()))
    print(f"n = {n}: E = {E:.2f} (hbar omega (n + 1/2)), norm = {norm:.4f}")
    ax.plot(x, 0.6 * psi.squeeze() + E, label=f"n = {n}"); ax.axhline(E, c="0.8", lw=0.6)
ax.plot(x, 0.5 * x ** 2, "k", lw=0.8); ax.set_ylim(0, 4.5); ax.set_xlabel("x"); ax.set_ylabel("E, psi (offset)")
''', plot=True),
    ],
)

pkg(
    name="pinneapple_decision",
    group="trust",
    title="Decision",
    tagline="Choose the next experiment from a closed set of options, under hard constraints and verification.",
    about=[
        "The decision layer never predicts a physical result. It decides which experiment to run next (model, "
        "training strategy, validation): every option passes hard constraints first (a structured-grid FNO is "
        "excluded on an unstructured mesh), every executed result goes through a verifier, and only verified "
        "results become evidence for the next decision.",
        "Decisions are distributions over the options with the rationale and the facts that were not known, so a "
        "default is never mistaken for a fact about the problem.",
    ],
    use_when=["You need to justify why a given model or validation was chosen for a problem."],
    examples=[
        dict(title="Which model for a parametric airfoil study?",
             code='''from pinneapple_decision import decide

d = decide({"description": "2D airfoil, surrogate over angle of attack", "reynolds": 1e5,
            "representation": "unstructured_mesh", "n_simulations": 200,
            "needs_parameter_generalization": True}, objective="choose physics model")
print("selected:", d.selected)
print("probabilities:", {k: round(v, 2) for k, v in d.probabilities.items()})
for k, why in list(d.excluded.items())[:2]:
    print("excluded", k, "->", why[:90])
print("requires validation:", d.requires_validation)
'''),
    ],
)

pkg(
    name="pinneapple_veriphysics",
    group="trust",
    title="Veriphysics",
    tagline="The evidence layer: decision records, applicability maps, robustness studies and evidence reports.",
    about=[
        "Built only from objects the analysis modules already computed, never a number of its own: a "
        "`DecisionRecord` (recommendation, trust score, coverage, per-check evidence, alternatives), the "
        "applicability map with the variable envelope that was actually tested, measured robustness studies, "
        "and the Evidence Report as a PDF.",
        "`recommend` formulates a problem from a description and recommends a solver family and architecture; "
        "`example_cases` holds worked cases.",
    ],
    use_when=["A model is going to be used for an engineering decision and needs an auditable record."],
    examples=[
        dict(title="Worked example cases",
             code='''from pinneapple_veriphysics import example_cases

keys = sorted(example_cases.EXAMPLE_CASES)
print(len(keys), "cases:", keys)
c = example_cases.get_example_case(keys[0])
print(c.key, "-", c.label)
print({k: (v if len(str(v)) < 60 else str(v)[:60] + "...") for k, v in vars(c).items() if k not in ("key", "label")})
'''),
    ],
)

pkg(
    name="pinneapple_security",
    group="workflow",
    title="Security",
    tagline="Integrity, signatures, provenance, audit trails, encryption and privacy for scientific data.",
    about=[
        "SHA-256 digests and Merkle manifests of files, arrays and model folders; Ed25519 or HMAC signatures in "
        "DSSE envelopes; in-toto / SLSA provenance of runs and a CycloneDX SBOM; a hash-chained audit trail; "
        "AES-256-GCM at rest; PII detection (including CPF and CNPJ), redaction and k-anonymity; differential "
        "privacy; safe loading of checkpoints and secret scanning.",
    ],
    use_when=["A dataset or a model leaves your machine and its integrity must be checkable.",
              "Logs or datasets may contain personal data."],
    examples=[
        dict(title="Digest, sign and verify a result; redact personal data from a log",
             code='''import numpy as np
from pinneapple_security import digest_array, hmac_key, sign_json, verify_json, redact

field = np.linspace(0, 1, 1000)
d = digest_array(field)
key = hmac_key(b"a-shared-secret-of-32-bytes-.....")
env = sign_json({"result": "Kt = 2.59", "field_sha256": d}, key)
print("payload verified:", verify_json(env, key))
env["payload"] = env["payload"][:-4] + "AAAA"
try:
    verify_json(env, key)
except Exception as e:
    print("tampered envelope rejected:", type(e).__name__)
print(redact("run by maria.silva@example.com, CPF 123.456.789-09, phone +55 11 91234-5678"))
'''),
    ],
)

pkg(
    name="pinneapple_registry",
    group="workflow",
    title="Registry",
    tagline="A local, offline registry for model versions, datasets, experiments and problem specs.",
    about=[
        "One root folder with four stores: versioned models with stages (development, staging, production) and "
        "their trust reports, datasets per problem and scenario, an SQLite experiment tracker and the history of "
        "problem specs. Models export straight to a Triton Inference Server repository.",
        "It complements `pinneapple_hub`: version everything here, publish the finished model there.",
    ],
    use_when=["You train many versions of a model and need to know which one is in production and why."],
    examples=[
        dict(title="Save two versions of a model and promote one",
             code='''import tempfile, torch
from pinneapple_registry import ArtifactRegistry

reg = ArtifactRegistry(tempfile.mkdtemp())
for width in (16, 32):
    net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
    reg.models.save("heat_plate", net, metadata={"width": width, "val_rel_l2": 0.05 if width == 16 else 0.02})
versions = reg.models.versions("heat_plate")
print("versions:", len(versions), " latest:", reg.models.latest("heat_plate") == versions[-1])
reg.models.promote("heat_plate", versions[-1], "staging")
print("metadata:", {k: v for k, v in reg.models.metadata("heat_plate").items() if k in ("width", "val_rel_l2", "stage")})
'''),
    ],
)

pkg(
    name="pinneapple_hub",
    group="workflow",
    title="Hub",
    tagline="Publish and load models on the Hugging Face Hub with a model card that must state its validation.",
    about=[
        "`push_to_hub` and `from_pretrained` use the Hugging Face Hub's own infrastructure. A `ModelCard` "
        "cannot be published without validation metrics and the reference they were computed against, and a "
        "model whose trust report rejects it needs a recorded override.",
    ],
    use_when=["You share a trained physics model publicly."],
    examples=[
        dict(title="A model card refuses to make no checkable claim",
             code='''from pinneapple_hub import ModelCard

card = ModelCard(name="plate-heat-pinn", architecture="modified_mlp")
print("problems:", card.validate())
card.validation_metrics = {"rel_l2": 0.012}
card.reference_source = "FEniCS P2 solution on a 256 x 256 mesh"
print("problems after adding evidence:", card.validate())
print(card.to_markdown()[:300])
'''),
    ],
)

pkg(
    name="pinneapple_catalog",
    group="workflow",
    title="Catalogue",
    tagline="Public datasets, CAD sets, pretrained models and methods, with licences and validation status.",
    about=[
        "`resources` lists public datasets, geometry sets, pretrained models and hosted benchmarks, each with a "
        "checked licence and a gate (`require_allowed`) that stops non-commercial data from entering a commercial "
        "pipeline. `methods` lists every solver, training method, equation and problem in the library with its "
        "code location, references and validation status.",
    ],
    use_when=["You need training data and want to know whether its licence lets you use it."],
    examples=[
        dict(title="Datasets that allow commercial use",
             code='''from pinneapple_catalog import list_resources

res = list_resources()
ok = list_resources(commercial_only=True)
print(f"{len(res)} resources, {len(ok)} usable commercially")
for r in ok[:6]:
    print(f"- {r.name} [{r.kind}, {r.domain}] {r.license}")
'''),
    ],
)

pkg(
    name="pinneapple_perception",
    group="data",
    title="Perception",
    tagline="Physics observations from images, video and audio: PIV velocity fields, boundaries, modal frequencies.",
    about=[
        "The inverse of rendering: real images, video and audio go in, physics observations come out. "
        "`video_piv` is cross-correlation particle image velocimetry, `image_geometry` extracts boundary points "
        "and fitted circles from photos or scans, and `audio_modal` finds the dominant frequencies of a tap test. "
        "The outputs feed data constraints or geometry elsewhere in the library.",
    ],
    use_when=["You have experimental footage or recordings and want them as constraints for a model."],
    examples=[
        dict(title="PIV recovers a known shift between two synthetic particle images",
             code='''import numpy as np
import matplotlib.pyplot as plt
from pinneapple_perception import piv_velocity_field

rng = np.random.default_rng(0)
H, W = 256, 256
yy, xx = np.mgrid[0:H, 0:W]
def frame(dx, dy):
    img = np.zeros((H, W))
    for x0, y0 in P:
        img += np.exp(-((xx - x0 - dx) ** 2 + (yy - y0 - dy) ** 2) / 2.0)
    return img
P = rng.uniform(0, 256, (1500, 2))
a, b = frame(0, 0), frame(4, -2)                      # particles move 4 px right, 2 px up
r = piv_velocity_field(a, b, window_size=32, search_margin=8, dt=1.0)
print(f"{np.size(r['u'])} windows, median displacement u = {np.median(r['u']):.2f} px, v = {np.median(r['v']):.2f} px"
      "  (imposed 4, -2)")

fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.imshow(a, cmap="gray", origin="upper"); ax.quiver(r["x"], r["y"], r["u"], r["v"], color="tab:orange", angles="xy")
ax.set_title("PIV vectors on frame A"); ax.axis("off")
''', plot=True),
    ],
)

# ---------------------------------------------------------------------------------------------------------------
pkg(
    name="pinneapple_llm",
    group="frontier",
    title="LLM assistance",
    tagline="Language models draft problems, geometry and twins from PINNeAPPle's own presets; a physics guardrail checks every result.",
    about=[
        "The language model only selects and parametrises what the library already implements (problem presets, "
        "geometry generators, twin configurations); it never invents physics. `research` searches the literature, "
        "`agent_loop` runs a tool-using loop, and `local_llm` with `conversation_store` keeps everything on your "
        "machine (Ollama and SQLite), with `finetune` for LoRA on the logged data.",
        "`PhysicsGuardrail` is the part that matters with or without an LLM: it re-evaluates the PDE residual on "
        "fresh points, checks units and conservation where a closed form exists, compares with reference data, "
        "and refuses to call a result trustworthy unless every applicable check passes, naming the one that "
        "failed.",
    ],
    use_when=["Any model result is about to be reported, by a person or by an agent."],
    examples=[
        dict(title="The guardrail accepts a harmonic function and rejects one that is not",
             code='''import torch
from pinneapple_llm import PhysicsGuardrail
from pinneapple_physics import get_preset

spec = get_preset("laplace_2d")                            # laplace(u) = 0 on the unit square
good = lambda X: (X[:, :1] ** 2 - X[:, 1:2] ** 2)           # x^2 - y^2 is harmonic
bad = lambda X: (X[:, :1] ** 2 + X[:, 1:2] ** 2)            # laplacian = 4
class Formula(torch.nn.Module):                            # any torch module works; this one has no weights
    def __init__(self, f):
        super().__init__(); self.f = f; self.dummy = torch.nn.Parameter(torch.zeros(()))
    def forward(self, X):
        return self.f(X) + 0 * self.dummy

for name, f in [("x^2 - y^2", good), ("x^2 + y^2", bad)]:
    model = Formula(f)
    rep = PhysicsGuardrail(spec, residual_threshold=1e-3).check(model)
    pde = [c for c in rep.checks if "residual" in c.name][0]
    print(f"{name}: residual check passed={pde.passed}  ({pde.name}, value {pde.value:.3g})")
'''),
    ],
)

pkg(
    name="pinneapple_worldmodel",
    group="frontier",
    title="World model",
    tagline="A generalist physics model trained across domains: scenarios, datasets, curriculum, specialists and a foundation model.",
    about=[
        "The world model learns f(state_t, descriptor) -> state_t+1 across many physics domains: built-in "
        "scenarios and geometries generate trajectories with the simulators, a curriculum and meta-learning "
        "train specialists and a foundation model with LoRA adapters, and a benchmark scores them on held-out "
        "tasks. An orchestrator maps a problem statement to the right tool.",
        "It also renders simulations into synthetic images and video (cameras, domain randomisation) for "
        "perception models, the opposite direction of `pinneapple_perception`.",
    ],
    use_when=["You research physics foundation models across several PDE families."],
    examples=[
        dict(title="Built-in domains and collocation sets",
             code='''import matplotlib.pyplot as plt
from pinneapple_worldmodel import make_channel_with_cylinder, BUILTIN_DOMAINS, BUILTIN_SCENARIOS

print("domains:", sorted(BUILTIN_DOMAINS)[:8])
print("scenarios:", len(BUILTIN_SCENARIOS), "e.g.", sorted(BUILTIN_SCENARIOS)[:5])
dom = make_channel_with_cylinder()
pts = dom.to_collocation_dict(n_interior=3000, n_boundary=600, seed=0)
print({k: tuple(v.shape) for k, v in pts.items()})

fig, ax = plt.subplots(figsize=(7.5, 2.2))
ax.plot(pts["interior"][:, 0], pts["interior"][:, 1], ".", ms=1.5, c="0.6")
for k, v in pts.items():
    if k not in ("interior", "boundary"):
        ax.plot(v[:, 0], v[:, 1], ".", ms=3, label=k)
ax.set_aspect("equal"); ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.3))
''', plot=True),
    ],
)

pkg(
    name="pinneapple_pdb",
    group="data",
    title="Physical dataset builder",
    tagline="Build sharded, validated datasets from hubs of physical data with a declared schema.",
    about=[
        "`PhysicalDatasetBuilder` takes a query (variables, space-time window), a `PhysicalSchema` with units "
        "and derived quantities, and writes shards with validation; schema templates cover common cases and the "
        "benchmark registry records which dataset backs which benchmark.",
    ],
    use_when=["You turn a large external archive into training shards with checked units."],
    examples=[
        dict(title="Registered benchmarks",
             code='''from pinneapple_pdb import list_benchmarks

from pinneapple_pdb import get_benchmark

names = list_benchmarks()
print(len(names), "benchmarks:", names)
b = get_benchmark("blasius_flat_plate")
print(b.description)
print("source:", b.reference_source)
print("x:", b.x_vars, b.reference_x.shape, " y:", b.y_vars, b.reference_y.shape)
'''),
    ],
)

pkg(
    name="pinneapple_problemdesign",
    group="workflow",
    title="Problem design",
    tagline="Turn a description into a complete, checkable problem specification, with human approval for consequential steps.",
    about=[
        "A design agent elicits what is missing from a problem description (geometry, conditions, data, "
        "targets), finds gaps, plans the steps and writes a PINNeAPPle spec and code. Autonomy levels and an "
        "approval gate classify which actions are consequential and stop for a person before taking them.",
    ],
    use_when=["A problem arrives as text and you want a spec with its gaps listed, not guessed."],
    examples=[
        dict(title="Consequential actions need approval",
             code='''from pinneapple_problemdesign import ConsequentialActionClassifier, AutonomyLevel

clf = ConsequentialActionClassifier()
print("autonomy levels:", [a.name for a in AutonomyLevel])
for tool in ("read_geometry", "run_openfoam_case", "run_fluent_case", "push_to_hub", "send_report_email"):
    print(f"{tool:20s} consequential = {clf.classify(tool)}")
print("rules are plain data you can read and extend:", sorted(clf.name_patterns)[:8], "...")
'''),
    ],
)

pkg(
    name="pinneapple_orchestration",
    group="workflow",
    title="Orchestration",
    tagline="Optional Prefect flows that wire data generation, training, benchmarking and twins together.",
    about=[
        "Five lean flows over capabilities that exist elsewhere: data generation into the dataset store, "
        "training a registered component with experiment logging, benchmarking every architecture of a "
        "component type, digital-twin updates, and a run manifest. Without Prefect the same functions run as "
        "plain Python.",
    ],
    use_when=["You schedule the same pipeline repeatedly and want retries and a run history."],
    examples=[
        dict(title="Flows available and whether Prefect is installed",
             code='''import pinneapple_orchestration as orch

print("Prefect available:", orch.PREFECT_AVAILABLE)
print("flows:", [n for n in dir(orch.flows) if n.startswith("flow_")])
print("tasks:", [n for n in dir(orch.tasks) if not n.startswith("_")][:10])
'''),
    ],
)

for _name, _target in [("pinneapple_models", "pinneapple_neural.architectures"),
                       ("pinneapple_solvers", "pinneapple_simulation.numerical_solvers"),
                       ("pinneapple_train", "pinneapple_neural.trainer")]:
    pkg(name=_name, group="legacy", title=_name, shim=_target,
        tagline=f"Compatibility shim: re-exports {_target} for older code.",
        about=[f"Kept so that code written against the old package name keeps working. New code imports "
               f"`{_target}` directly; the names are the same objects."],
        use_when=["Never in new code."], examples=[])

# PINN training example for the physics package (kept last: it trains for a couple of minutes).
next(p for p in PACKAGES if p["name"] == "pinneapple_physics")["examples"].insert(0, dict(
    title="Burgers' equation: a PINN against the Cole-Hopf solution, through one API",
    code='''import numpy as np
import matplotlib.pyplot as plt
import pinneapple as pp

pinn = pp.solve("burgers_1d", method="pinn", epochs=3000)
exact = pp.solve("burgers_1d", method="analytic")
def trained_pinn(problem, **kw):            # compare() takes method names or callables returning a Solution
    return pinn
print(pp.compare("burgers_1d", methods=[trained_pinn], reference=exact))

x = np.linspace(-1, 1, 400)
fig, ax = plt.subplots(figsize=(7.5, 3))
for t in (0.0, 0.25, 0.5, 0.75):
    X = np.column_stack([x, np.full_like(x, t)])
    l, = ax.plot(x, exact.predict(X)[:, 0], lw=2.2, alpha=0.5)
    ax.plot(x, pinn.predict(X)[:, 0], "--", c=l.get_color(), label=f"t = {t}")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=7); ax.set_title("PINN (dashed) vs Cole-Hopf (solid)")
''', plot=True))
