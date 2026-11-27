"""The six reference pictures of post-processing: slices, isosurfaces, streamlines, tubes, vectors, volume (#62).

    python scripts/gallery_postprocessing.py                        # docs/assets/postprocessing
    python scripts/gallery_postprocessing.py --out /tmp/gallery --size 900x600

The field is analytic (a Taylor-Green-like vortex pair), so the gallery needs no dataset and no solver: what it
demonstrates is the post-processing, and every picture is the same flow seen five ways plus the composite. The colours
come from the house palette, so the gallery looks like the rest of the documentation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_tools.visualization.style import PALETTE, get_cmap  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "assets" / "postprocessing"
#: what each picture shows, for the page that embeds them
CAPTIONS = {
    "01_slice": "A slice through the volume, coloured by speed, with the vector field drawn on the same plane.",
    "02_isosurface": "The isosurface |u| = 1.4: the surface that separates the fast core from the slow outside.",
    "03_streamlines": "Streamlines released from a line of seeds: the path a particle takes, no time axis.",
    "04_tubes": "The same streamlines as tubes, radius by speed — a surface a reader can see depth in.",
    "05_vectors": "Arrows on a plane, scaled by speed: direction where the eye can follow it, magnitude where it cannot.",
    "06_composite": "The composite: the isosurface inside the volume outline with one slice, the picture a report opens with.",
}


def field(n: int = 48):
    """A vortex pair on a box: analytic velocity, so the gallery is reproducible without a dataset."""
    import pyvista as pv

    x = np.linspace(0.0, 2 * np.pi, n)
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(x[1] - x[0],) * 3, origin=(0.0, 0.0, 0.0))
    X = grid.points
    # ABC flow (Arnold-Beltrami-Childress): a steady solution of Euler with chaotic streamlines that fill the box
    a = b = c = 1.0
    u = a * np.sin(c * X[:, 2]) + c * np.cos(b * X[:, 1])
    v = b * np.sin(a * X[:, 0]) + a * np.cos(c * X[:, 2])
    w = c * np.sin(b * X[:, 1]) + b * np.cos(a * X[:, 0])
    grid["u"] = np.column_stack([u, v, w])
    grid["speed"] = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    return grid


def _plotter(size, title: str):
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("#ffffff")
    if title:
        p.add_text(title, position="upper_left", font_size=10, color="#111111")
    return p


def _save(plotter, path: Path) -> Path:
    plotter.screenshot(str(path))
    plotter.close()
    return path


def build(out_dir: Path = DEFAULT_OUT, size: tuple[int, int] = (1400, 900)) -> list[Path]:
    """Write the six pictures; returns the paths, in the order the page lists them."""
    import pyvista as pv

    out_dir.mkdir(parents=True, exist_ok=True)
    grid = field()
    written: list[Path] = []
    blue, orange, green = PALETTE[0], PALETTE[1], PALETTE[2]
    muted = "#9a9a9a"

    # 1 — a slice with the vectors on it
    p = _plotter(size, "Slice at z = 0, speed and vectors")
    sl = grid.slice(normal="z", origin=(np.pi, np.pi, np.pi))
    p.add_mesh(sl, scalars="speed", cmap=get_cmap("cfd_sequential"), show_scalar_bar=True, scalar_bar_args={"title": "|u|"})
    p.add_arrows(sl.points[::90], sl["u"][::90], mag=0.05, color="#222222")
    p.view_xy()
    written.append(_save(p, out_dir / "01_slice.png"))

    # 2 — an isosurface
    p = _plotter(size, "Isosurface |u| = 1.4")
    p.add_mesh(grid.outline(), color=muted)
    p.add_mesh(grid.contour(scalars="speed", isosurfaces=[1.4]), color=blue, opacity=0.9, show_scalar_bar=False)
    p.view_isometric()
    written.append(_save(p, out_dir / "02_isosurface.png"))

    # 3 — streamlines from a line of seeds
    p = _plotter(size, "Streamlines from a line of seeds")
    # a plane of seeds across the box, off the diagonal where the ABC flow has its stagnation point
    seeds = pv.Plane(center=(np.pi, np.pi, 1.0), direction=(0, 0, 1), i_size=4.0, j_size=4.0,
                     i_resolution=4, j_resolution=4)
    lines = grid.streamlines_from_source(seeds, vectors="u", max_length=30.0, integration_direction="both")
    p.add_mesh(grid.outline(), color=muted)
    p.add_mesh(lines.tube(radius=0.02), scalars="speed", cmap=get_cmap("cfd_sequential"), show_scalar_bar=True,
               scalar_bar_args={"title": "|u|"})
    p.view_isometric()
    written.append(_save(p, out_dir / "03_streamlines.png"))

    # 4 — the same lines as tubes, seen at an angle
    p = _plotter(size, "The same streamlines as tubes")
    p.add_mesh(grid.outline(), color=muted)
    p.add_mesh(lines.tube(radius=0.04), color=green)
    p.view_isometric()
    written.append(_save(p, out_dir / "04_tubes.png"))

    # 5 — arrows on a plane
    p = _plotter(size, "Vectors on the mid-plane, scaled by speed")
    arrows = sl.glyph(orient="u", scale="speed", factor=0.15, geom=pv.Arrow(), tolerance=0.03)
    p.add_mesh(arrows, scalars="speed", cmap=get_cmap("cfd_sequential"), show_scalar_bar=True, scalar_bar_args={"title": "|u|"})
    p.view_xy()
    written.append(_save(p, out_dir / "05_vectors.png"))

    # 6 — the composite: outline, one slice and one isosurface
    p = _plotter(size, "Composite: outline, slice and isosurface")
    p.add_mesh(grid.outline(), color=muted)
    p.add_mesh(sl, scalars="speed", cmap=get_cmap("cfd_sequential"), opacity=0.85, show_scalar_bar=False)
    p.add_mesh(grid.contour(scalars="speed", isosurfaces=[1.4]), color=orange, opacity=0.5)
    p.view_isometric()
    written.append(_save(p, out_dir / "06_composite.png"))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gallery_postprocessing", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--size", default="1400x900", help="WIDTHxHEIGHT of each picture")
    a = ap.parse_args(argv)
    width, height = (int(v) for v in a.size.lower().split("x"))
    for path in build(a.out, (width, height)):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
