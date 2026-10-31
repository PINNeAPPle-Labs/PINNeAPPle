"""Mesh a cylinder with wall prism layers through the library and draw the half cut, every element coloured by its
SICN. Writes docs/guides/img/mesh_generation_cylinder.png and prints the measured numbers.

    python scripts/mesh_generation_figure.py [out.png]
"""
import math
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from pinneapple_data.cae.model import CELL_FACES  # noqa: E402
from pinneapple_design.geometry.mesh.generate import generate_mesh  # noqa: E402

R, H, h = 0.5, 1.0, 0.08
BL = {"first_layer": 0.01, "n_layers": 3, "growth": 1.2}
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/guides/img/mesh_generation_cylinder.png"


def read_msh(path):
    """Nodes, cells and the SICN of every cell of the .msh the tool wrote (gmsh's reader and quality)."""
    import gmsh
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(path))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = np.zeros(int(tags.max()) + 1, int)
    idx[tags.astype(int)] = np.arange(len(tags))
    cells = []
    for code, name, k in ((4, "tetra", 4), (6, "wedge", 6)):
        et, nodes = gmsh.model.mesh.getElementsByType(code)
        if len(et):
            cells.append((name, idx[np.asarray(nodes, int)].reshape(-1, k),
                          np.asarray(gmsh.model.mesh.getElementQualities(et, "minSICN"))))
    gmsh.finalize()
    return np.asarray(xyz, float).reshape(-1, 3), cells


with tempfile.TemporaryDirectory() as d:
    out = generate_mesh("cylinder", [R, H], h=h, boundary_layer=BL, out_dir=d)
    pts, cells = read_msh(out["files"]["msh"])

# keep the elements behind the plane y = 0 and draw the faces that bound that half (the cut shows the layers)
polys, vals = [], []
for name, conn, sicn in cells:
    keep = pts[conn][:, :, 1].mean(1) < 0
    for f in CELL_FACES[name]:
        polys.append((conn[keep][:, list(f)], sicn[keep]))
faces = {}
for conn, s in polys:
    for row, v in zip(conn, s):
        key = tuple(sorted(row))
        faces[key] = None if key in faces else (row, v)
shown = [fv for fv in faces.values() if fv is not None]
verts = [pts[r] for r, _ in shown]
colour = np.array([v for _, v in shown])

fig = plt.figure(figsize=(7.2, 6.0), dpi=150)
ax = fig.add_subplot(projection="3d")
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(0.0, 1.0)
ax.add_collection3d(Poly3DCollection(verts, facecolors=cmap(norm(colour)), edgecolors="k", linewidths=0.15))
ax.set_xlim(-R, R), ax.set_ylim(-R, R), ax.set_zlim(0, H)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=18, azim=60)
ax.set_axis_off()
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="SICN of the element (1 ideal, <= 0 inverted)")
c = out["element_counts"]
exact = math.pi * R * R * H
ax.set_title(f"generate_mesh('cylinder', [{R}, {H}], h={h}, boundary_layer=3 layers)\n"
             f"{c.get('tetra', 0)} tetrahedra + {c.get('wedge', 0)} prisms, SICN min {out['sicn_min']:.3f}, "
             f"volume {out['volume']:.4f} m3 (pi r^2 H = {exact:.4f})", fontsize=8.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print({k: v for k, v in out.items() if k != "files"})
print("relative volume error", (out["volume"] - exact) / exact, "->", OUT)
