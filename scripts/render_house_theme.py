"""Render the palette and the maps of the house theme under simulated colour vision.

Writes docs/assets/house_theme/palette.png, which docs/guides/house_theme.md
includes. The dichromacy matrices are the Viénot, Brettel and Mollon (1999)
linear approximations, applied in linear RGB.

    python scripts/render_house_theme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pinneapple_tools.visualization.style import (  # noqa: E402
    CMAPS,
    FIELD_CMAPS,
    PALETTE,
    annotate_verdict,
    colorbar,
    field_cmap,
    label,
    panel_letters,
    source_note,
    use_theme,
)

OUT = ROOT / "docs" / "assets" / "house_theme"

# Linear-RGB transforms; rows are the simulated cone responses.
_VISION = {
    "normal": np.eye(3),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Apply the dichromacy approximation to an (n, 3) sRGB array."""
    lin = _to_linear(np.asarray(rgb, dtype=float))
    return _to_srgb(lin @ _VISION[kind].T)


def render(path: Path) -> Path:
    use_theme("report")
    kinds = ("normal", "protanopia", "deuteranopia", "tritanopia")

    fig = plt.figure(figsize=(15.0, 8.2))
    grid = fig.add_gridspec(2, 4, height_ratios=(1.25, 1.0), hspace=0.28, wspace=0.18)
    top = [fig.add_subplot(grid[0, i]) for i in range(4)]

    # Palette rows, one panel per simulated vision.
    palette = np.array([matplotlib.colors.to_rgb(c) for c in PALETTE])
    for ax, kind in zip(top, kinds, strict=False):
        seen = simulate(palette, kind)
        for i, colour in enumerate(seen):
            ax.add_patch(plt.Rectangle((0, i), 1, 0.88, color=colour))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, len(seen))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(kind, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Maps, bottom left.
    ax_maps = fig.add_subplot(grid[1, :2])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    names = ["cfd_sequential", "cfd_diverging", "cfd_cyclic", "rainbow"]
    for row, name in enumerate(names):
        ax_maps.imshow(gradient, aspect="auto", extent=(0, 1, row, row + 0.8),
                       cmap=CMAPS[name], vmin=0, vmax=1)
        ax_maps.text(0.01, row + 0.4, name, va="center", ha="left", fontsize=9, color="#ffffff")
    ax_maps.set_xlim(0, 1)
    ax_maps.set_ylim(0, len(names))
    ax_maps.set_yticks([])
    label(ax_maps, x=("value", ""), title="maps")

    # The helpers in use, bottom right.
    demo = fig.add_subplot(grid[1, 2:])
    temperature = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    mesh = demo.pcolormesh(temperature, cmap=field_cmap("temperature"))
    colorbar(demo, mesh, "Temperature", "K", shrink=0.9)
    label(demo, x=("x", "m"), y=("y", "m"), title="the helpers on a field")
    annotate_verdict(demo, "wall", True)

    panel_letters(fig, top, xy=(-0.02, 1.02))
    panel_letters(fig, [ax_maps, demo], start=4, xy=(-0.08, 1.04))
    fig.suptitle("House palette and maps under simulated colour vision", fontsize=13)

    source_note(fig, "Data: no experimental data; the palette and the maps of "
                     "pinneapple_tools.visualization.style")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    target = render(OUT / "palette.png")
    print(f"wrote {target}")
    print("quantities with a registered map:", len(FIELD_CMAPS))
