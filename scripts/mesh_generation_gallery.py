"""Pictures of meshes made by ``generate_mesh``: every element drawn with its edges and coloured by its SICN (gmsh
minSICN, 1 ideal, <= 0 inverted), with the measured numbers in the titles. Writes docs/assets/mesh/*.png, the guide's
figure docs/guides/img/mesh_generation_cylinder.png and the generated test part docs/assets/mesh/bracket.stl.

    python scripts/mesh_generation_gallery.py
"""
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from pinneapple_data.cae.model import CELL_FACES  # noqa: E402
from pinneapple_design.geometry.mesh.generate import generate_mesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/mesh"
CMAP, NORM = plt.get_cmap("viridis"), plt.Normalize(0.0, 1.0)
EDGE = dict(edgecolors="k", linewidths=0.15)
RECORDS = []


def read_msh(path):
    """Nodes and, per element type, connectivity and SICN, read back from the .msh the tool wrote."""
    import gmsh
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(path))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = np.zeros(int(tags.max()) + 1, int)
    idx[tags.astype(int)] = np.arange(len(tags))
    cells = []
    for code, name, k in ((2, "triangle", 3), (4, "tetra", 4), (6, "wedge", 6)):
        et, nodes = gmsh.model.mesh.getElementsByType(code)
        if len(et):
            cells.append((name, idx[np.asarray(nodes, int)].reshape(-1, k),
                          np.asarray(gmsh.model.mesh.getElementQualities(et, "minSICN"))))
    gmsh.finalize()
    return np.asarray(xyz, float).reshape(-1, 3), cells


def mesh(label, *args, **kw):
    with tempfile.TemporaryDirectory() as d:
        out = generate_mesh(*args, out_dir=d, **kw)
        pts, cells = read_msh(out["files"]["msh"])
        json.loads(Path(out["files"]["view"]).read_text())          # the view was written and parses
    vol = [c for c in cells if c[0] != "triangle"] if out["dim"] == 3 else cells
    RECORDS.append((label, out, np.concatenate([c[2] for c in vol])))
    return out, pts, vol


def outer_faces(pts, cells, keep=None):
    """Faces bounding the kept elements (all by default) with the SICN of their element."""
    faces = {}
    for name, conn, sicn in cells:
        k = np.ones(len(conn), bool) if keep is None else keep(pts[conn].mean(1))
        for f in CELL_FACES[name]:
            for row, v in zip(conn[k][:, list(f)], sicn[k], strict=True):
                key = tuple(sorted(row))
                faces[key] = None if key in faces else (row, v)
    shown = [fv for fv in faces.values() if fv is not None]
    return [pts[r] for r, _ in shown], np.array([v for _, v in shown])


def draw3d(ax, pts, cells, keep=None, view=(20, 55), title=""):
    verts, val = outer_faces(pts, cells, keep)
    ax.add_collection3d(Poly3DCollection(verts, facecolors=CMAP(NORM(val)), **EDGE))
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def numbers(out):
    c = ", ".join(f"{n} {t}" for t, n in out["element_counts"].items())
    size = f"volume {out['volume']:.4g} m3" if out["dim"] == 3 else f"area {out['area']:.4g} m2"
    return (f"{out['n_nodes']} nodes, {c}\nSICN min {out['sicn_min']:.3f} / mean {out['sicn_mean']:.3f}, "
            f"h {out['h_min']:.3f}-{out['h_max']:.3f} m, {size}")


def save(fig, name, axes):
    fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=axes, shrink=0.6, pad=0.02,
                 label="SICN of the element (1 ideal, <= 0 inverted)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def pair(name, title, out, pts, cells, cut, view=(20, 55)):
    fig = plt.figure(figsize=(10, 4.6))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    b = fig.add_subplot(1, 2, 2, projection="3d")
    draw3d(a, pts, cells, view=view, title="boundary")
    draw3d(b, pts, cells, keep=cut, view=view, title="cut: the elements on one side of a plane")
    fig.suptitle(f"{title}\n{numbers(out)}", fontsize=9)
    save(fig, name, [a, b])


# 1. box
out, pts, cells = mesh("box h=0.1", "box", [1.0, 1.0, 1.0], h=0.1)
pair("01_box_tet.png", "generate_mesh('box', [1, 1, 1], h=0.1)", out, pts, cells, lambda c: c[:, 1] < 0.5)

# 2. cylinder with wall prisms (also the guide's figure)
R, H = 0.5, 1.0
out, pts, cells = mesh("cylinder + 3 layers", "cylinder", [R, H], h=0.08,
                       boundary_layer={"first_layer": 0.01, "n_layers": 3, "growth": 1.2})
exact = math.pi * R * R * H
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_subplot(projection="3d")
draw3d(ax, pts, cells, keep=lambda c: c[:, 1] < 0, view=(18, 60))
ax.set_title(f"generate_mesh('cylinder', [0.5, 1.0], h=0.08, boundary_layer=3 layers from 0.01 m), cut at y = 0\n"
             f"{numbers(out)} (pi r^2 H = {exact:.4f})", fontsize=8)
save(fig, "02_cylinder_boundary_layer.png", ax)
guide = ROOT / "docs/guides/img/mesh_generation_cylinder.png"
guide.parent.mkdir(parents=True, exist_ok=True)
guide.write_bytes(
    (OUT / "02_cylinder_boundary_layer.png").read_bytes())

# 3. sphere
out, pts, cells = mesh("sphere h=0.08", "sphere", [0.5], h=0.08)
pair("03_sphere.png", "generate_mesh('sphere', [0.5], h=0.08)", out, pts, cells, lambda c: c[:, 1] < 0)

# 4. local refinement
out, pts, cells = mesh("box + refine ball", "box", [1.0, 1.0, 1.0], h=0.15,
                       refine=[{"center": [0.5, 0.5, 0.5], "radius": 0.2, "h": 0.03}])
pair("04_box_refined.png", "generate_mesh('box', h=0.15, refine=[ball r=0.2 at the centre, h=0.03])", out, pts, cells,
     lambda c: c[:, 1] < 0.5)

# 5. 2-D (the third panel zooms on a corner, where the wall layers are)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
for ax, (label, args, kw, zoom) in zip(axes, [
        ("rectangle h=0.05", ("rectangle", [2.0, 1.0]), {"h": 0.05}, None),
        ("disk h=0.05", ("disk", [0.5]), {"h": 0.05}, None),
        ("rectangle + 4 layers", ("rectangle", [2.0, 1.0]),
         {"h": 0.1, "boundary_layer": {"first_layer": 0.005, "n_layers": 4, "growth": 1.3}}, (0.12, 0.06))], strict=True):
    out, pts, cells = mesh(label, *args, **kw)
    _, conn, sicn = cells[0]
    ax.add_collection(PolyCollection(pts[conn][:, :, :2], facecolors=CMAP(NORM(sicn)), **EDGE))
    ax.autoscale()
    if zoom:
        ax.set_xlim(0, zoom[0]), ax.set_ylim(0, zoom[1])
        label += f" (corner, {zoom[0]} x {zoom[1]} m)"
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"{label}\n{numbers(out)}", fontsize=8)
save(fig, "05_planar_2d.png", list(axes))

# 6. a generated part as STL (a plate with two holes and a boss), meshed from the STL
stl = OUT / "bracket.stl"
import gmsh  # noqa: E402

gmsh.initialize(readConfigFiles=False, interruptible=False)
gmsh.option.setNumber("General.Verbosity", 0)
occ = gmsh.model.occ
plate = occ.addBox(0, 0, 0, 0.12, 0.06, 0.012)
boss = occ.addCylinder(0.06, 0.03, 0.012, 0, 0, 0.025, 0.014)
part, _ = occ.fuse([(3, plate)], [(3, boss)])
holes = [occ.addCylinder(x, 0.03, -0.01, 0, 0, 0.06, 0.007) for x in (0.02, 0.10)] + \
        [occ.addCylinder(0.06, 0.03, -0.01, 0, 0, 0.06, 0.006)]
occ.cut(part, [(3, h) for h in holes])
occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.004)
gmsh.option.setNumber("Mesh.Binary", 1)
gmsh.model.mesh.generate(2)
gmsh.write(str(stl))
gmsh.finalize()
out, pts, cells = mesh("bracket.stl h=0.004", str(stl), h=0.004)
BRACKET = 0.12 * 0.06 * 0.012 + math.pi * 0.014 ** 2 * 0.025 - 2 * math.pi * 0.007 ** 2 * 0.012 \
    - math.pi * 0.006 ** 2 * 0.037                                     # plate + boss - two holes - bore
pair("06_stl_part.png", "generate_mesh('bracket.stl', h=0.004): plate 120 x 60 x 12 mm, two holes, bored boss; "
     f"exact volume {BRACKET:.4g} m3; cut through the hole centres",
     out, pts, cells, lambda c: c[:, 1] < 0.03, view=(35, -60))

# 7. SICN histograms and the measured numbers
fig = plt.figure(figsize=(12, 7.5))
ax = fig.add_subplot(2, 1, 1)
bins = np.linspace(0, 1, 51)
for label, _out, s in RECORDS:
    ax.hist(s, bins=bins, histtype="step", density=True, label=label, lw=1.2)
ax.set_xlabel("SICN of the element (1 ideal, <= 0 inverted)")
ax.set_ylabel("density")
ax.legend(fontsize=7, ncol=3)
ax.set_title("SICN of every element (volume elements in 3-D, triangles in 2-D)", fontsize=9)
tb = fig.add_subplot(2, 1, 2)
tb.set_axis_off()
rows = [[label, out["n_nodes"], out["n_elements"], f"{out['sicn_min']:.3f}", f"{out['sicn_mean']:.3f}",
         f"{out['h_min']:.4f}", f"{out['h_mean']:.4f}", f"{out['h_max']:.4f}",
         f"{out['volume']:.5g} m3" if out["dim"] == 3 else f"{out['area']:.5g} m2"] for label, out, _ in RECORDS]
t = tb.table(cellText=rows, colLabels=["mesh", "nodes", "elements", "SICN min", "SICN mean", "h min [m]",
                                       "h mean [m]", "h max [m]", "volume / area"], loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(7.5)
t.scale(1, 1.3)
fig.savefig(OUT / "07_sicn_histogram.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "07_sicn_histogram.png")
(OUT / "measured.json").write_text(json.dumps([{"mesh": label, **{k: v for k, v in out.items() if k != "files"}}
                                               for label, out, _ in RECORDS], indent=1))
