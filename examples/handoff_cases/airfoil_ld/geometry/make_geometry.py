"""Airfoil geometry for the package: CST parameter file, coordinates, STL / STEP and a plot of the shapes.

    python make_geometry.py [OUT_DIR]           # default: this folder

Shape parametrisation (Kulfan CST, as in pinneapple_design.aero.geometry): 6 weights,
thickness T0..T2 and camber M0..M2 of a 2nd-order Bernstein polynomial,
    y_upper(x) = C(x) * sum_i (M_i + T_i) B_i(x) + x * t_te / 2
    y_lower(x) = C(x) * sum_i (M_i - T_i) B_i(x) - x * t_te / 2
with class function C(x) = sqrt(x) (1 - x), B_i the Bernstein basis of order 2 and a blunt trailing edge
t_te = 0.25 % chord. Chord 1, leading edge at (0, 0).

The STL (ASCII) is written from the coordinates; the STEP (1 chord = 1 m) needs cadquery (pip install cadquery).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if os.path.isdir(os.path.join(ROOT, "pinneapple_design")):
    sys.path.insert(0, ROOT)
from pinneapple_design.aero.geometry import (  # noqa: E402
    BOUNDS,
    LABELS,
    PARAMS,
    TE_THICKNESS,
    naca4,
    properties,
    surfaces,
    valid,
)

SPAN = 1.0           # extrusion length of the 3-D CAD (in chords); the CFD is 2-D


def naca4_exact(code: str, n: int = 161):
    """Analytic NACA 4-digit coordinates (Abbott & von Doenhoff, sharp-TE coefficient 0.1036 as used by naca4)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dy)
    return (x - yt * np.sin(th), yc + yt * np.cos(th)), (x + yt * np.sin(th), yc - yt * np.cos(th))


def shapes() -> dict:
    base = naca4("2412")
    aft = base.copy()
    aft[5] += 0.04                                     # more camber near the trailing edge (aft loading)
    thick = base.copy()
    thick[:3] *= 1.25                                  # 25 % thicker everywhere
    return {
        "baseline_naca2412": {"shape": base, "note": "least-squares CST fit of the NACA 2412 (the baseline)"},
        "variant_a_naca4412": {"shape": naca4("4412"), "note": "CST fit of the NACA 4412: twice the camber"},
        "variant_b_aft_camber": {"shape": aft, "note": "baseline with M2 + 0.04: aft-loaded camber"},
        "variant_c_thick": {"shape": thick, "note": "baseline with T0..T2 x 1.25: about 15 % thick"},
    }


def coords(shape: np.ndarray, n: int = 121):
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))
    yu, yl = surfaces(shape, x)
    return x, yu, yl


def write_dat(path: str, name: str, x, yu, yl) -> None:
    """Selig format: TE -> upper -> LE -> lower -> TE (blunt TE: the two TE points differ)."""
    with open(path, "w") as f:
        f.write(name + "\n")
        for xi, yi in zip(x[::-1], yu[::-1], strict=False):
            f.write(f"{xi:10.6f} {yi:10.6f}\n")
        for xi, yi in zip(x[1:], yl[1:], strict=False):
            f.write(f"{xi:10.6f} {yi:10.6f}\n")


def write_cad(out: str, name: str, x, yu, yl) -> list[str]:
    try:
        import cadquery as cq
    except ImportError:
        return []
    up = [(float(a), float(b)) for a, b in zip(x[::-1], yu[::-1], strict=False)]          # TE -> LE on the upper side
    lo = [(float(a), float(b)) for a, b in zip(x[1:], yl[1:], strict=False)]               # LE -> TE on the lower side
    wp = (cq.Workplane("XY").moveTo(*up[0]).spline(up[1:], includeCurrent=True)
          .spline(lo, includeCurrent=True).close())                          # blunt TE closed by a straight line
    solid = wp.extrude(SPAN)
    files = []
    cq.exporters.export(solid, os.path.join(out, f"{name}.step"), unit="M")      # 1 chord = 1 m in the STEP file
    files.append(f"{name}.step")
    return files


def write_stl(path: str, name: str, x, yu, yl, span: float = SPAN) -> None:
    """ASCII STL of the section extruded from z = 0 to z = span, straight from the coordinates (same stations as
    the .dat file): side wall around the closed contour (blunt trailing edge included) and two end caps."""
    loop = np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]          # TE upper -> LE -> TE lower (CCW)
    tris = []

    def tri(a, b, c, out):
        a, b, c = (np.asarray(v, float) for v in (a, b, c))
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-14:
            return
        if n @ out < 0:
            b, c, n = c, b, -n
        tris.append((n / np.linalg.norm(n), a, b, c))

    for i in range(len(loop)):                                            # side wall, closing segment = TE base
        p, q = loop[i], loop[(i + 1) % len(loop)]
        out = np.array([q[1] - p[1], p[0] - q[0], 0.0])
        a, b, c, d = (*p, 0.0), (*q, 0.0), (*q, span), (*p, span)
        tri(a, b, c, out)
        tri(a, c, d, out)
    for z, s in ((0.0, -1.0), (span, 1.0)):                               # end caps: strips between the surfaces
        for i in range(len(x) - 1):
            u0, u1, l0, l1 = (x[i], yu[i], z), (x[i + 1], yu[i + 1], z), (x[i], yl[i], z), (x[i + 1], yl[i + 1], z)
            tri(u0, u1, l1, np.array([0, 0, s]))
            tri(u0, l1, l0, np.array([0, 0, s]))
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for n, a, b, c in tris:
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for v in (a, b, c):
                f.write(f"      vertex {v[0]:.7e} {v[1]:.7e} {v[2]:.7e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out, exist_ok=True)
    rec = {"parametrisation": {
        "type": "CST (Kulfan), Bernstein order 2, class function sqrt(x)(1-x)",
        "parameters": {p: LABELS[p] for p in PARAMS},
        "upper_weights": "M_i + T_i", "lower_weights": "M_i - T_i",
        "trailing_edge_thickness_over_chord": TE_THICKNESS,
        "bounds": {p: [float(lo), float(hi)] for p, (lo, hi) in zip(PARAMS, BOUNDS, strict=False)},
        "chord": 1.0, "cad_span": SPAN,
        "units": "chords (scale to any chord length); the STEP files are written with 1 chord = 1 m; STL is ASCII"},
        "shapes": {}}
    for name, s in shapes().items():
        shp = s["shape"]
        x, yu, yl = coords(shp)
        write_dat(os.path.join(out, f"{name}.dat"), name, x, yu, yl)
        cad = write_cad(out, name, x, yu, yl)
        write_stl(os.path.join(out, f"{name}.stl"), name, x, yu, yl)
        cad.append(f"{name}.stl")
        ok, why = valid(shp)
        rec["shapes"][name] = {"note": s["note"], **{p: round(float(v), 6) for p, v in zip(PARAMS, shp, strict=False)},
                               "properties": {k: round(v, 5) for k, v in properties(shp).items()},
                               "valid": ok, "invalid_reason": why,
                               "inside_bounds": bool(np.all((shp >= BOUNDS[:, 0]) & (shp <= BOUNDS[:, 1]))),
                               "files": [f"{name}.dat", *cad]}
    # exact NACA 2412 and the CST fit error
    (xu, yu_e), (xl, yl_e) = naca4_exact("2412")
    with open(os.path.join(out, "naca2412_exact_coordinates.dat"), "w") as f:
        f.write("NACA 2412 (analytic, Abbott & von Doenhoff equations)\n")
        for a, b in zip(xu[::-1], yu_e[::-1], strict=False):
            f.write(f"{a:10.6f} {b:10.6f}\n")
        for a, b in zip(xl[1:], yl_e[1:], strict=False):
            f.write(f"{a:10.6f} {b:10.6f}\n")
    base = shapes()["baseline_naca2412"]["shape"]
    cu = surfaces(base, np.clip(xu, 0, 1))[0]
    cl = surfaces(base, np.clip(xl, 0, 1))[1]
    err = float(max(np.abs(cu - yu_e).max(), np.abs(cl - yl_e).max()))
    rec["baseline_fit"] = {"max_abs_deviation_from_exact_naca2412_over_chord": round(err, 6),
                           "note": "the CFD and the campaign use the CST fit, not the exact NACA ordinates"}
    x, yu, yl = coords(base)
    write_dat(os.path.join(out, "naca2412_cst_coordinates.dat"), "NACA 2412 CST fit", x, yu, yl)
    with open(os.path.join(out, "cst_parameters.json"), "w") as f:
        json.dump(rec, f, indent=1)
    plot(out, xu, yu_e, xl, yl_e)
    print(json.dumps(rec["baseline_fit"]), "|", ", ".join(rec["shapes"]))


def plot(out, xu, yu_e, xl, yl_e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.visualization.style import use_theme
    use_theme("report")   # one theme for every figure of the library (#918)
    colors = ["#1f3b73", "#c2491d", "#2e8b57", "#7a4fa3"]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax.plot(xu, yu_e, color="0.6", lw=3.0, alpha=0.6, label="NACA 2412, exact ordinates")
    ax.plot(xl, yl_e, color="0.6", lw=3.0, alpha=0.6)
    for (name, s), c in zip(shapes().items(), colors, strict=False):
        x, yu, yl = coords(s["shape"])
        t = properties(s["shape"])["thickness"]
        ax.plot(np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]], color=c, lw=1.4, label=f"{name} (t/c {t:.3f})")
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("Baseline (CST fit of NACA 2412) and three variants", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shapes.png"))


if __name__ == "__main__":
    main()
