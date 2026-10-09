"""Render the gallery parts to one PNG with matplotlib (no browser or GPU needed): quick visual check.

    python examples/use_cases/refractory_wear_twin/render_gallery_png.py docs/guides/img/gallery.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from pinneapple_twin3d import gallery  # noqa: E402

FIELD = {"runner_channel": "wear_depth", "mixing_tank": "tracer_concentration", "oil_pipeline": "wall_loss",
         "airliner": "skin_temperature", "rocket": "heat_flux", "drone": "temperature", "lunar_rover": "temperature",
         "wind_turbine": "leading_edge_erosion", "satellite": "temperature"}
VIEW = {"runner_channel": (35, -60), "mixing_tank": (20, -50), "oil_pipeline": (55, -60), "airliner": (25, -130),
        "rocket": (10, -50), "drone": (35, -50), "lunar_rover": (22, -45), "wind_turbine": (12, -70), "satellite": (25, -50)}
STEP = {"mixing_tank": 2, "rocket": 12}  # an informative time step (default: the last one)


def render(path: str, dpi: int = 70) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), subplot_kw={"projection": "3d"})
    for ax, (key, field) in zip(axs.ravel(), FIELD.items()):
        sc = gallery.get(key)
        s = STEP.get(key, len(sc.times) - 1)
        vals = [p.fields[field][s] for p in sc.parts if field in p.fields]
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        allv = np.vstack([p.vertices for p in sc.parts])
        c, r = (allv.max(0) + allv.min(0)) / 2, (allv.max(0) - allv.min(0)).max() / 2
        for p in sc.parts:
            tri = p.vertices[p.faces][:, :, [0, 2, 1]]  # matplotlib is z-up, the scenes are y-up
            if field in p.fields:
                col = plt.cm.turbo((p.fields[field][s][p.faces].mean(1) - lo) / ((hi - lo) or 1))
            else:
                col = np.tile([0.72, 0.74, 0.78, 1], (len(tri), 1))
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(c[1] - r, c[1] + r)
        ax.view_init(*VIEW[key]); ax.set_axis_off()
        ax.set_title(f"{key}\n{field} [{lo:.3g}..{hi:.3g}] at {sc.times[s]:g} {sc.time_unit}  (illustrative)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "gallery.png")
