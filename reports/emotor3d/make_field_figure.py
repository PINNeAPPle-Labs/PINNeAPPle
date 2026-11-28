"""Regenerate ``figures/field_and_stack.png`` for the report.

    python docs/reports/emotor3d/make_field_figure.py

Left panel: the flux density of the reference solve (2004 Prius IPM, 250 A peak at 50 deg current angle), which
is the field the surrogate learns from. Right panel: the electrical angle of each axial slice of the skewed
stack, which is how a skewed machine is represented without a 3-D field solve. The figure is produced here, and
not in ``motor3d.workflow.figures``, because it needs a full reference solve that the workflow report does not
run.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004  # noqa: E402
from pinneapple_simulation.electromagnetics.machines.motor3d import Motor3DDesign  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def flux_density_panel(ax) -> float:
    m = IPMMachine(prius_2004(), n_gap=192)
    op = m.operating_point(I_peak=250.0, gamma=math.radians(50.0), n_positions=3, keep=True)
    ms = op["solutions"][-1]
    B = np.hypot(*np.asarray(ms.B)[:, :2].T) if np.asarray(ms.B).ndim > 1 else np.abs(np.asarray(ms.B))
    mesh = ms.mesh
    x, y = mesh.points.T
    scale = 1e3
    # nodal coordinates with one value per triangle: flat shading, the triangulation stays the mesh's own
    tp = ax.tripcolor(x * scale, y * scale, mesh.tris, B, shading="flat", cmap="turbo", rasterized=True)
    ax.set_aspect("equal")
    ax.set(xlabel="x [mm]", ylabel="y [mm]",
           title=f"2004 Prius IPM cross-section: |B| at 250 A, 50 deg (T = {op['torque_mean']:.1f} N m)")
    ax.tick_params(labelsize=7)
    fig = ax.figure
    cb = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("|B| [T]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return m.pole_pairs


def stack_panel(ax, p: int) -> None:
    """Electrical angle of each slice of a skewed stack: the geometry the axial model represents."""
    d = dataclasses.replace(Motor3DDesign.baseline(), skew_deg=10.0)
    K = 9
    z = np.linspace(-0.5, 0.5, K)
    k = np.arange(K)
    delta = (k / (K - 1) - 0.5) * d.skew_rad * p
    ax.plot(z, np.degrees(delta), "o-", color="C0", label=f"slice electrical angle (skew {math.degrees(d.skew_rad):.0f} deg)")
    ax.plot(z, np.degrees(delta * 0.6366), "--", color="C1",
            label=f"fundamental, sin(x)/x = {d.skew_factor(p):.3f}")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(xlabel="axial position z / L [-]", ylabel="electrical angle delta_k [deg]",
           title="Skew represented as axial slices")
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=7)


def main() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    p = flux_density_panel(ax[0])
    stack_panel(ax[1], p)
    fig.tight_layout()
    p = OUT / "field_and_stack.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    main()
