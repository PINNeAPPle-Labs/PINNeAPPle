"""Parametric open trough and jet nozzle: STEP + STL (cadquery), a dimension table and a dimensioned side view.

    python make_geometry.py [OUT_DIR] [--speed 2.5] [--angle 60]

Frame: x along the trough (x = 0 at the inside face of the upstream end wall), y across it (y = 0 at the inside
face of one side wall), z up (z = 0 on the top of the floor). All lengths in metres. The defaults are the
dimensions of the CFD baseline (simulation/trough_case.py, class Trough); change them in one place, DIMENSIONS.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

DIMENSIONS = {
    # name: (value, unit, description)
    "L": (1.20, "m", "inside length of the floor, upstream end wall to overflow sill"),
    "W": (0.15, "m", "inside width"),
    "H": (0.25, "m", "side and upstream end wall height above the floor"),
    "h_s": (0.08, "m", "overflow sill height above the floor (full width, downstream end)"),
    "t_w": (0.010, "m", "wall, floor and sill thickness"),
    "x_n": (0.30, "m", "nozzle exit centre: distance from the upstream end wall"),
    "y_n": (0.075, "m", "nozzle exit centre: distance from the side wall (centre plane)"),
    "z_n": (0.40, "m", "nozzle exit centre: height above the floor"),
    "d": (0.020, "m", "nozzle bore (inner diameter)"),
    "D_o": (0.026, "m", "nozzle outer diameter"),
    "L_n": (0.15, "m", "nozzle straight length (bore length upstream of the exit)"),
    "theta": (60.0, "deg", "jet angle below the horizontal, pointing downstream (+x)"),
    "h_0": (0.08, "m", "initial liquid depth (still liquid up to the sill crest)"),
}


def build(D: dict):
    import cadquery as cq
    L, W, H, hs, tw = (D[k][0] for k in ("L", "W", "H", "h_s", "t_w"))
    floor = cq.Workplane("XY").box(L + 2 * tw, W + 2 * tw, tw, centered=False).translate((-tw, -tw, -tw))
    side1 = cq.Workplane("XY").box(L + 2 * tw, tw, H, centered=False).translate((-tw, -tw, 0))
    side2 = cq.Workplane("XY").box(L + 2 * tw, tw, H, centered=False).translate((-tw, W, 0))
    up = cq.Workplane("XY").box(tw, W, H, centered=False).translate((-tw, 0, 0))
    sill = cq.Workplane("XY").box(tw, W, hs, centered=False).translate((L, 0, 0))
    trough = floor.union(side1).union(side2).union(up).union(sill)
    a = math.radians(D["theta"][0])
    e = cq.Vector(D["x_n"][0], D["y_n"][0], D["z_n"][0])                    # exit centre
    u = cq.Vector(math.cos(a), 0.0, -math.sin(a))                         # jet direction
    start = e - u * D["L_n"][0]
    outer = cq.Solid.makeCylinder(D["D_o"][0] / 2, D["L_n"][0], start, u)
    inner = cq.Solid.makeCylinder(D["d"][0] / 2, D["L_n"][0], start, u)
    nozzle = cq.Workplane("XY").add(outer.cut(inner))
    # fluid region above the floor inside the walls, up to the nozzle exit height (for meshing in 3-D)
    fluid = cq.Workplane("XY").box(L, W, D["z_n"][0], centered=False)
    return trough, nozzle, fluid


def side_view(D: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import use_theme
    use_theme("report")   # one theme for every figure of the library (#918)
    L, H, hs, tw = (D[k][0] for k in ("L", "H", "h_s", "t_w"))
    xn, zn, d, a = D["x_n"][0], D["z_n"][0], D["d"][0], math.radians(D["theta"][0])
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    wall = dict(color="0.35")
    ax.fill([-tw, L + tw, L + tw, -tw], [-tw, -tw, 0, 0], **wall)
    ax.fill([-tw, 0, 0, -tw], [0, 0, H, H], **wall)
    ax.fill([L, L + tw, L + tw, L], [0, 0, hs, hs], **wall)
    ax.fill([0, L, L, 0], [0, 0, D["h_0"][0], D["h_0"][0]], color="#9cc3e6", alpha=0.7, lw=0)
    ax.plot([0, L], [H, H], color="0.35", ls="--", lw=0.8)                  # side-wall top (behind)
    ux, uz = math.cos(a), -math.sin(a)
    nx, nz = -uz, ux
    Ln = D["L_n"][0]
    for s in (-1, 1):
        ax.plot([xn - ux * Ln + s * nx * d / 2, xn + s * nx * d / 2], [zn - uz * Ln + s * nz * d / 2, zn + s * nz * d / 2],
                color="#c2491d", lw=1.6)
    # ballistic jet axis to the initial surface
    g = 9.81
    U = 2.5
    ts = [i * 0.002 for i in range(200)]
    xs = [xn + U * ux * t for t in ts]
    zs = [zn + U * uz * t - 0.5 * g * t * t for t in ts]
    keep = [i for i, z in enumerate(zs) if z >= D["h_0"][0]]
    ax.plot([xs[i] for i in keep], [zs[i] for i in keep], color="#c2491d", lw=0.8, ls=":")
    txt = dict(fontsize=7)
    ax.annotate("", (0, -0.035), (L, -0.035), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(L / 2, -0.05, f"L = {L:g} m", ha="center", va="top", **txt)
    ax.annotate("", (-0.04, 0), (-0.04, H), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(-0.05, H / 2, f"H = {H:g} m", ha="right", va="center", rotation=90, **txt)
    ax.annotate("", (L + 0.03, 0), (L + 0.03, hs), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(L + 0.04, hs / 2, f"h_s = {hs:g} m", ha="left", va="center", **txt)
    ax.annotate("", (0, zn + 0.03), (xn, zn + 0.03), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(xn / 2, zn + 0.035, f"x_n = {xn:g} m", ha="center", va="bottom", **txt)
    ax.annotate("", (xn + 0.07, 0), (xn + 0.07, zn), arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(xn + 0.08, zn * 0.6, f"z_n = {zn:g} m", ha="left", va="center", **txt)
    ax.text(xn - 0.02, zn + 0.075, f"nozzle bore d = {d * 1000:g} mm, angle {D['theta'][0]:g} deg below horizontal",
            ha="left", **txt)
    ax.text(0.6, D["h_0"][0] / 2, f"liquid, initial depth h_0 = {D['h_0'][0]:g} m", ha="center", va="center", **txt)
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, L + 0.25)
    ax.set_ylim(-0.09, zn + 0.12)
    ax.axis("off")
    ax.set_title("Side view (centre plane y = W/2); inside width W = {:g} m".format(D["W"][0]), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "dimensions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "value", "unit", "description"])
        for k, (v, u, s) in DIMENSIONS.items():
            w.writerow([k, v, u, s])
    files = ["dimensions.csv", "side_view.png"]
    try:
        import cadquery as cq
        trough, nozzle, fluid = build(DIMENSIONS)
        for name, obj in (("trough", trough), ("nozzle", nozzle), ("fluid_region", fluid)):
            cq.exporters.export(obj, os.path.join(out, f"{name}.step"), unit="M")             # metres
            obj.val().exportStl(os.path.join(out, f"{name}.stl"), tolerance=5e-4, angularTolerance=0.2, ascii=True)
            files += [f"{name}.step", f"{name}.stl"]
        both = cq.Workplane("XY").add(cq.Compound.makeCompound([trough.val(), nozzle.val()]))
        cq.exporters.export(both, os.path.join(out, "assembly.step"), unit="M")          # trough + nozzle
        files.append("assembly.step")
        bb = trough.val().BoundingBox()
        info = {"trough_bounding_box_m": [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax],
                "trough_volume_m3": trough.val().Volume(), "nozzle_volume_m3": nozzle.val().Volume()}
    except ImportError:
        info = {"note": "cadquery not installed: STEP/STL not written"}
    side_view(DIMENSIONS, os.path.join(out, "side_view.png"))
    with open(os.path.join(out, "geometry_info.json"), "w") as f:
        json.dump({"dimensions": {k: {"value": v, "unit": u, "description": s} for k, (v, u, s) in DIMENSIONS.items()},
                   "files": files, **info}, f, indent=1)
    print(files)


if __name__ == "__main__":
    main()
