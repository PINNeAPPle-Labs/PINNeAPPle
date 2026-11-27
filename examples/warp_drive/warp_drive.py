"""The Alcubierre warp bubble, measured: the expansion it makes, the energy it needs, and where the physics says no.

Three pictures and one table, all of them read off the metric rather than quoted from the paper:

* the **expansion** ``theta`` -- contraction ahead of the bubble, expansion behind it, which is the picture everyone
  has seen;
* the **energy density** ``T^{00}`` -- negative in the wall and off the axis, exactly zero on the axis, and red
  wherever it is negative, because that is the region that needs exotic matter;
* the **turntable**: the negative-energy shell in 3-D, rotating, so the two lobes and the ring around the axis are
  visible from every side;
* the **bill**: ``-1.9e27 kg`` for a 10 m bubble at the speed of light, which is about one Jupiter mass, and how it
  scales -- ``v_s^2 R^2 / wall`` -- so that no choice of profile makes it zero.

Run::

    python examples/warp_drive/warp_drive.py            # everything, ~3 min with the GIF
    python examples/warp_drive/warp_drive.py --quick    # no GIF, 160-point maps
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pinneapple_physics.relativity import warp

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

BUBBLE = warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0)
JUPITER_KG = 1.8982e27


def plane(extent: float = 16.0, samples: int = 161) -> np.ndarray:
    """Points on the ``z = 0`` plane, ``(t, x, y, z)``, for the maps."""
    axis = np.linspace(-extent, extent, samples)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    return np.concatenate([np.zeros((len(points), 1)), points], axis=1)


def measurements(bubble: warp.WarpMetric, samples: int) -> dict[str, object]:
    """Everything the pictures show, as numbers."""
    points = plane(16.0, samples)
    curvature = warp.einstein_tensor(bubble, points)
    radius = bubble.radius_at(points)
    safe = np.maximum(radius, 0.5 * bubble.wall)                 # the closed form divides by r_s, which is 0 at the centre
    closed = -(bubble.velocity ** 2 / (32.0 * math.pi)) * (points[:, 2] ** 2 + points[:, 3] ** 2) \
        / safe ** 2 * warp.shape_derivative(safe, bubble.radius, bubble.wall, bubble.kind) ** 2
    in_wall = np.abs(closed) > 1e-3 * np.abs(closed).max()

    wall_points = points[in_wall]
    energy = warp.worst_null_energy(bubble, wall_points, samples=12)
    total = warp.integrated_energy(bubble, samples=max(41, samples // 2))

    leak = {}
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(40, 3))
    peak_wall = float(np.abs(curvature.einstein[in_wall]).max())
    for fraction in (0.1, 0.2, 0.3, 0.5, 0.8):
        shell = fraction * bubble.radius * direction / np.linalg.norm(direction, axis=1, keepdims=True)
        inside = np.concatenate([np.zeros((40, 1)), shell], axis=1)
        peak = float(np.abs(warp.einstein_tensor(bubble, inside).einstein).max())
        leak[f"r/R={fraction:g}"] = round(peak / peak_wall, 6)

    return {
        "bubble": {"velocity_c": bubble.velocity, "radius_m": bubble.radius, "wall_m": bubble.wall,
                   "shape": bubble.kind},
        "points": int(len(points)),
        "closed_form": {"max_relative_error": float(np.max(np.abs(curvature.energy_density[in_wall]
                                                                  - closed[in_wall]) / np.abs(closed[in_wall]))),
                        "wall_points": int(in_wall.sum())},
        "energy_density": {"min_per_m2": float(curvature.energy_density.min()),
                           "max_per_m2": float(curvature.energy_density.max()),
                           "min_kg_per_m3": float(curvature.energy_density.min() * warp.C_SI ** 2 / warp.G_SI),
                           "zero_on_the_axis": float(np.abs(warp.energy_density(
                               bubble, np.array([[0.0, bubble.radius, 0.0, 0.0]])))[0])},
        "expansion": {"min": float(curvature.expansion.min()), "max": float(curvature.expansion.max()),
                      "peak": float(np.abs(warp.expansion(
                          bubble, np.array([[0.0, bubble.radius, 0.0, 0.0]]))).max())},
        "null_energy": {"minimum": energy["minimum"], "violated": energy["violated"],
                        "points_violating": energy["points_violating"], "points": energy["points"]},
        "bill": {"total_geometric_m": total["total_geometric"], "kilograms": total["kilograms"],
                 "solar_masses": total["solar_masses"], "jupiter_masses": total["kilograms"] / JUPITER_KG,
                 "extent_m": total["extent_m"]},
        "interior_leak": leak,
    }


def scaling(samples: int = 41) -> dict[str, object]:
    """The three scalings, measured: ``E ~ v_s^2``, ``E ~ R^2`` and ``E ~ 1/wall``."""
    velocity = {}
    for speed in (0.1, 0.5, 1.0, 5.0):
        total = warp.integrated_energy(warp.WarpMetric(velocity=speed, radius=10.0, wall=2.0), samples=samples)
        velocity[f"{speed:g}c"] = total["kilograms"]
    radius = {}
    for size in (5.0, 10.0, 20.0, 50.0):
        total = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=size, wall=2.0), samples=samples)
        radius[f"{size:g}m"] = total["kilograms"]
    thickness = {}
    for wall in (1.0, 2.0, 4.0):
        total = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=wall), samples=samples)
        thickness[f"{wall:g}m"] = total["kilograms"]
    gaussian = warp.integrated_energy(warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0, kind="gaussian"),
                                      samples=samples)
    return {"velocity": velocity, "radius": radius, "wall": thickness,
            "gaussian_kilograms": gaussian["kilograms"],
            "tophat_kilograms": [warp.integrated_energy(warp.WarpMetric(kind="tophat"), samples=s)["kilograms"]
                                 for s in (25, 31, 41, 61)]}


def map_figure(bubble: warp.WarpMetric, samples: int, path: Path) -> None:
    """Three maps on the ``z = 0`` plane: the wall, the expansion, and the energy density in red where it is negative."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = plane(16.0, samples)
    side = int(math.isqrt(len(points)))
    curvature = warp.einstein_tensor(bubble, points)
    # the grid is built with x on the first index and y on the second; imshow wants rows = y, so transpose
    shape = bubble.f(points).reshape(side, side).T
    energy = curvature.energy_density.reshape(side, side).T
    theta = curvature.expansion.reshape(side, side).T
    axis = np.linspace(points[:, 1].min(), points[:, 1].max(), side)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    im0 = axes[0].imshow(shape, extent=extent, origin="lower", cmap="viridis")
    axes[0].set_title("the wall: f(r), 1 inside and 0 outside")
    figure.colorbar(im0, ax=axes[0], fraction=0.046)

    limit = float(np.abs(theta).max())
    im1 = axes[1].imshow(theta, extent=extent, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_title("the expansion  $\\theta=v_s\\,(x-x_s)/r_s\\,f'$  (blue ahead, red behind)")
    figure.colorbar(im1, ax=axes[1], fraction=0.046)

    peak = float(np.abs(energy).max())
    im2 = axes[2].imshow(energy, extent=extent, origin="lower", cmap="RdBu", vmin=-peak, vmax=peak)
    axes[2].contourf(axis, axis, np.where(energy < -1e-3 * peak, energy, np.nan),
                     levels=[-peak, -1e-3 * peak], colors="none", hatches=["///"])
    axes[2].set_title("$T^{00}$: red where it is negative (hatched), zero on the axis")
    axes[2].axhline(0.0, color="black", lw=0.9, ls=":")
    axes[2].annotate("$T^{00}=0$ on the axis", xy=(0.02, 0.96), xycoords="axes fraction", fontsize=9)
    figure.colorbar(im2, ax=axes[2], fraction=0.046)

    for axis in axes:
        axis.set_xlabel("x  [m]")
        axis.set_ylabel("y  [m]")
    figure.suptitle("Alcubierre bubble, R = 10 m, wall = 2 m, $v_s = c$: contraction ahead, expansion behind, "
                    "and negative energy in the wall", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def turntable(bubble: warp.WarpMetric, samples: int, frames: int, path: Path) -> None:
    """The negative-energy shell, rotating: a 3-D surface of ``T^{00}`` over the ``z = 0`` plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    points = plane(16.0, samples)
    side = int(math.isqrt(len(points)))
    curvature = warp.einstein_tensor(bubble, points)
    energy = curvature.energy_density.reshape(side, side)
    theta = curvature.expansion.reshape(side, side)
    axis = np.linspace(points[:, 1].min(), points[:, 1].max(), side)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    peak = float(np.abs(energy).max())

    figure = plt.figure(figsize=(11.0, 5.4))
    left = figure.add_subplot(1, 2, 1, projection="3d")
    right = figure.add_subplot(1, 2, 2, projection="3d")
    panels = ((left, energy, "$T^{00}$: the negative-energy shell (red = negative)", "RdBu", peak),
              (right, theta, "the expansion $\\theta$ (blue ahead, red behind)", "RdBu_r", 0.25))
    for panel, data, title, colours, limit in panels:
        panel.plot_surface(x, y, data, cmap=colours, vmin=-limit, vmax=limit, linewidth=0, antialiased=True,
                           rstride=1, cstride=1)
        panel.set_title(title, fontsize=11)
        panel.set_xlabel("x [m]", fontsize=8)
        panel.set_ylabel("y [m]", fontsize=8)
        panel.set_zlim(-0.9 * limit, 0.9 * limit)
        panel.tick_params(labelsize=7)

    def update(frame: int):
        left.view_init(elev=24.0, azim=frame * 360.0 / frames)
        right.view_init(elev=24.0, azim=frame * 360.0 / frames)
        return []

    animation = FuncAnimation(figure, update, frames=frames, interval=90, blit=False)
    animation.save(path, writer=PillowWriter(fps=12))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="no GIF, coarser maps")
    parser.add_argument("--out", type=Path, default=None, help="where to write the results JSON")
    parser.add_argument("--figures", type=Path, default=None, help="where to write the figures")
    arguments = parser.parse_args()

    output = arguments.out or RESULTS / "warp_drive.json"
    figures = arguments.figures or FIGURES
    output.parent.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    samples = 81 if arguments.quick else 161

    started = time.time()
    report = measurements(BUBBLE, samples)
    report["scaling"] = scaling()
    report["seconds"] = round(time.time() - started, 2)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['seconds']:.1f}s  bill = {report['bill']['kilograms']:.3e} kg "
          f"({report['bill']['jupiter_masses']:.2f} Jupiter masses), "
          f"NEC minimum {report['null_energy']['minimum']:.4e}")

    map_figure(BUBBLE, samples, figures / "warp_maps.png")
    if not arguments.quick:
        turntable(BUBBLE, 81, 36, figures / "warp_turntable.gif")
    print("wrote", output, "and the figures in", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
