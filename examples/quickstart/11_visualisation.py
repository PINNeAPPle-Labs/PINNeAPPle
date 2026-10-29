"""11 — Visualisation: 1-D, 2-D, 3-D, and 4-D (time as an animated GIF).

Each block produces a real file in ``runs/visualisation/``. The 4-D one animates a
field that depends on (x, y, t): one frame per time step, with the time series
drawn next to it and a marker that walks along it.

Run from the repository root:

    python examples/quickstart/11_visualisation.py
"""
from __future__ import annotations

import os

import numpy as np

import pinneapple as pp

OUT = "runs/visualisation"
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1-D: a solved profile against the closed form
# ---------------------------------------------------------------------------
print("[1-D] solving burgers_1d with a PINN and plotting it against the Cole-Hopf solution")

pinn = pp.solve("burgers_1d", method="pinn", epochs=800)
closed = pp.solve("burgers_1d", method="analytic")

x = np.linspace(-1.0, 1.0, 300, dtype=np.float32)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, t in zip(axes, (0.25, 0.75)):
    X = np.column_stack([x, np.full_like(x, t)])
    ax.plot(x, closed.predict(X).ravel(), "k-", lw=2, label="Cole-Hopf (exact)")
    ax.plot(x, pinn.predict(X).ravel(), "C3--", lw=2, label="PINN (800 epochs)")
    ax.set_title(f"u(x, t={t})")
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("u")
axes[0].legend()
fig.suptitle("1-D: solution profile at two times")
path_1d = f"{OUT}/01_field_1d.png"
fig.savefig(path_1d, dpi=130, bbox_inches="tight")
plt.close(fig)
print("      ->", path_1d)

# ---------------------------------------------------------------------------
# 2-D: the field and its error map
# ---------------------------------------------------------------------------
print("[2-D] FEM on a sourced Poisson problem, field and error map")

from pinneapple_physics import ProblemBuilder

source = lambda X, ctx=None: -2.0 * np.pi**2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
exact = lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])

b = ProblemBuilder("poisson_showcase").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
for face in (("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")):
    b = b.bc(kind="dirichlet", field="u", value=0.0, on=face)
fem = pp.solve(b.build(), method="fem", ctx={"source_fn": source}, n=64)

gx, gy = fem.grid["x"], fem.grid["y"]
XX, YY = np.meshgrid(gx, gy, indexing="ij")
grid = {"x": gx, "y": gy}
field = fem.predict(np.column_stack([XX.ravel(), YY.ravel()])).reshape(len(gx), len(gy))
ref = exact(np.column_stack([XX.ravel(), YY.ravel()])).reshape(len(gx), len(gy))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
im0 = axes[0].pcolormesh(XX, YY, field, shading="auto", cmap="viridis")
axes[0].set_title("u(x, y) — P1 finite elements")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].pcolormesh(XX, YY, np.abs(field - ref), shading="auto", cmap="magma")
axes[1].set_title("|error| against sin(pi x) sin(pi y)")
fig.colorbar(im1, ax=axes[1])
for ax in axes:
    ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    ax.set_aspect("equal")
path_2d = f"{OUT}/02_field_2d.png"
fig.savefig(path_2d, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"      max |error| = {np.max(np.abs(field - ref)):.2e}  -> {path_2d}")

# ---------------------------------------------------------------------------
# 3-D: slices through a 3-D solver output
# ---------------------------------------------------------------------------
print("[3-D] 3-D heat conduction (fdm3d), three orthogonal slices")

from pinneapple_simulation.numerical_solvers import HeatConduction3D, HeatConfig3D

cfg = HeatConfig3D(nx=28, ny=28, nz=28, nt=60, dt=2e-3, alpha=0.01,
                   source_fn=lambda X, Y, Z, t: 1.0)
vol = HeatConduction3D(cfg).solve()
T = vol.u[-1]                      # last time step: (nx, ny, nz)
print(f"      volume {T.shape}, T in [{T.min():.4f}, {T.max():.4f}]")

k = T.shape[2] // 2
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, plane, label in ((axes[0], T[:, :, k].T, f"z = {k}"),
                         (axes[1], T[:, k, :].T, f"y = {k}"),
                         (axes[2], T[k, :, :].T, f"x = {k}")):
    im = ax.imshow(plane, origin="lower", cmap="inferno")
    ax.set_title(f"slice at {label}")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("3-D heat conduction: T on three orthogonal planes")
path_3d = f"{OUT}/03_volume_3d.png"
fig.savefig(path_3d, dpi=130, bbox_inches="tight")
plt.close(fig)
print("      ->", path_3d)

# ---------------------------------------------------------------------------
# 4-D: (x, y, t) as an animated GIF through pp.dashboard
# ---------------------------------------------------------------------------
print("[4-D] a field that depends on space and time, animated")

from pinneapple_tools.visualization.dashboard import Chart, Dashboard, Series, Tile

nx = ny = 48
t = np.linspace(0.0, 1.0, 24)
xs = np.linspace(-1.0, 1.0, nx)
ys = np.linspace(-1.0, 1.0, ny)
Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

frames, peaks = [], []
for tk in t:
    # a diffusing, decaying pair of blobs: amplitude decays as exp(-t), width grows with sqrt(t)
    width = 0.25 + 0.35 * np.sqrt(tk)
    Z = np.exp(-((Xg - 0.3) ** 2 + (Yg - 0.2) ** 2) / (2 * width**2)) * np.exp(-tk)
    peaks.append(float(Z.max()))
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
    rgb = (plt.get_cmap("viridis")(Z)[..., :3] * 255).astype(np.uint8)
    frames.append(rgb)

dashboard = Dashboard(
    "Diffusing blobs",
    t,
    frames,
    subtitle="T(x, y, t): two Gaussian blobs diffusing and decaying",
    charts=[Chart("peak amplitude", "amplitude [a.u.]", "time [s]",
                  [Series("peak", np.asarray(peaks))])],
    tiles=[Tile("final peak", peaks[-1], "a.u."), Tile("frames", len(frames), "")],
)
written = dashboard.to_video(gif=f"{OUT}/04_field_4d.gif")
for key, value in written.items():
    if isinstance(value, str) and os.path.exists(value):
        print(f"      {key}: {value} ({os.path.getsize(value) / 1e3:.0f} kB)")
    else:
        print(f"      {key}: {value}")

print("\nall figures in", OUT)
