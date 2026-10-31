"""TEAM workshop problem 7: asymmetrical conductor with a hole (linear eddy currents, 50 and 200 Hz).

Geometry, data and measured values from the TEAM problem 7 description (K. Fujiwara and T. Nakata, "Results for
benchmark problem 7 (asymmetrical conductor with a hole)", COMPEL 9 (3), 1990, 137-154; the problem description
with its tables of measured values is distributed with ElmerCSC's elmer-elmag repository as
``TEAM7/TEAM7_ProblemDescription.pdf``, and its ``TEAM7_A1B1.csv`` holds the 50 Hz, wt = 0 column of table 2a):

* aluminium plate 294 x 294 x 19 mm, sigma = 3.526e7 S/m, corner at the origin, with a 108 x 108 mm hole whose
  corner is at (18, 18) mm;
* a racetrack coil, 2742 ampere-turns, 25 mm wide and 100 mm high, from z = 49 to 149 mm (30 mm above the plate);
  inner square 150 mm with corner radius 25 mm, outer corner radius 50 mm; corner-arc centres at
  (144, 50), (244, 50), (244, 150), (144, 150) mm, so the coil spans x = 94..294 mm and y = 0..200 mm;
* the current is a maximum at wt = 0: ``i(t) = I0 cos(wt)``, phasor ``I0``; a field ``F`` at ``wt`` is
  ``Re(F exp(j wt))``, so ``wt = 90 deg`` gives ``-Im(F)``.

Measured quantities: ``Bz`` (gauss) along A1-B1 (y = 72, z = 34 mm) and A2-B2 (y = 144, z = 34 mm), x = 0..288 mm
every 18 mm; the eddy-current density ``Jy`` (1e6 A/m^2) along A3-B3 (y = 72, z = 19 mm) and A4-B4 (y = 72, z = 0)
at the 12 positions in :data:`X_J`. The benchmark states no tolerance band.

The coil current flows counter-clockwise seen from +z, which gives the positive ``Bz`` measured under the coil.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .mesh import TetMesh, read_msh22
from .nedelec import NU0, EdgeSpace, solve_eddy

__all__ = ["SIGMA", "AMPERE_TURNS", "X_B", "X_J", "MEASURED", "coil_current_density", "mesh_team7", "solve",
           "probe", "compare", "probe_lines", "run"]

SIGMA = 3.526e7
AMPERE_TURNS = 2742.0
COIL_AREA = 0.025 * 0.100
CORNERS = (0.144, 0.244, 0.050, 0.150)          # x0, x1, y0, y1 of the arc centres (m)
R_IN, R_OUT = 0.025, 0.050
Z_COIL = (0.049, 0.149)

X_B = np.arange(17) * 18.0                       # mm
X_J = np.array([0.0, 18.0, 126.0, 144.0, 162.0, 180.0, 198.0, 216.0, 234.0, 252.0, 270.0, 288.0])

# measured values: {(line, f_Hz, wt_deg): values}; Bz in gauss, Jy in 1e6 A/m^2
MEASURED = {
    ("A1B1", 50, 0): [-4.90, -17.88, -22.13, -20.19, -15.67, 0.36, 43.64, 78.11, 71.55, 60.44, 53.91, 52.62, 53.81,
                      56.91, 59.24, 52.78, 27.61],
    ("A1B1", 50, 90): [-1.16, 2.84, 4.15, 4.00, 3.07, 2.31, 1.89, 4.97, 12.61, 14.15, 13.04, 12.40, 12.05, 12.27,
                       12.66, 9.96, 2.36],
    ("A1B1", 200, 0): [-3.63, -18.46, -23.62, -21.59, -16.09, 0.23, 44.35, 75.53, 63.42, 53.20, 48.66, 47.31, 48.31,
                       51.26, 53.61, 46.11, 24.96],
    ("A1B1", 200, 90): [-1.38, 1.20, 2.15, 1.63, 1.10, 0.27, -2.28, -1.40, 4.17, 3.94, 4.86, 4.09, 3.69, 4.60, 3.48,
                        4.10, 0.98],
    ("A2B2", 50, 0): [-1.83, -8.50, -13.60, -15.21, -14.48, -5.62, 28.77, 60.34, 61.84, 56.64, 53.40, 52.36, 53.93,
                      56.82, 59.48, 52.08, 26.56],
    ("A2B2", 50, 90): [-1.63, -0.60, -0.43, 0.11, 1.26, 3.40, 6.53, 10.25, 11.83, 11.83, 11.01, 10.58, 10.80, 10.54,
                       10.62, 9.03, 1.79],
    ("A2B2", 200, 0): [-0.86, -7.00, -11.58, -13.36, -13.77, -6.74, 24.63, 53.19, 54.89, 50.72, 48.03, 47.13, 48.25,
                       51.35, 53.35, 45.37, 24.01],
    ("A2B2", 200, 90): [-1.35, -0.71, -0.81, -0.67, 0.15, 1.39, 2.67, 3.00, 4.01, 3.80, 4.00, 3.02, 2.20, 2.78, 1.58,
                        1.37, 0.93],
    ("A3B3", 50, 0): [0.249, 0.685, -0.015, -0.103, -0.061, -0.004, 0.051, 0.095, 0.135, 0.104, -0.321, -0.687],
    ("A3B3", 50, 90): [-0.629, -0.873, -0.593, -0.249, -0.101, -0.001, 0.087, 0.182, 0.322, 0.555, 0.822, 0.855],
    ("A3B3", 200, 0): [0.427, 0.794, 1.401, -0.035, 0.005, -0.011, 0.007, 0.027, 0.042, 0.043, 0.050, -0.321],
    ("A3B3", 200, 90): [-0.623, -0.755, -1.304, -0.229, -0.041, -0.014, -0.002, -0.000, 0.008, 0.033, 0.116, 0.893],
    ("A4B4", 50, 0): [0.461, 0.621, 1.573, 0.556, 0.237, 0.097, -0.034, -0.157, -0.305, -0.478, -0.660, -1.217],
    ("A4B4", 50, 90): [-0.662, -0.644, -1.027, -0.757, -0.364, -0.149, 0.015, 0.154, 0.311, 0.508, 0.747, 1.034],
    ("A4B4", 200, 0): [1.057, 1.597, 4.163, 1.143, 0.672, 0.307, -0.050, -0.370, -0.749, -1.205, -1.575, -2.583],
    ("A4B4", 200, 90): [-0.915, -1.036, -2.328, -1.193, -0.613, -0.259, 0.061, 0.334, 0.674, 1.064, 1.404, 2.331],
}

LINES = {"A1B1": (72.0, 34.0), "A2B2": (144.0, 34.0), "A3B3": (72.0, 19.0), "A4B4": (72.0, 0.0)}


def coil_current_density(points: np.ndarray) -> np.ndarray:
    """Uniform current density of the racetrack coil, counter-clockwise seen from +z (A/m^2), at points inside
    the coil (direction tangent to the straight sides and to the corner arcs)."""
    x0, x1, y0, y1 = CORNERS
    p = np.atleast_2d(points)
    r = np.column_stack([p[:, 0] - np.clip(p[:, 0], x0, x1), p[:, 1] - np.clip(p[:, 1], y0, y1)])
    n = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)
    J0 = AMPERE_TURNS / COIL_AREA
    return J0 * np.column_stack([-n[:, 1], n[:, 0], np.zeros(len(p))])


def mesh_team7(path, h_plate: float = 0.006, h_coil: float = 0.012, h_air: float = 0.08,
               margin: float = 0.3, h_layer: float | None = None) -> Path:
    """gmsh mesh: air box extending ``margin`` beyond the plate and coil (outer surface physical 9), the plate
    (volume 1, "plate"), the coil (2, "coil"), air (3, "air"); sizes graded from the plate and coil."""
    import gmsh

    path = Path(path)
    gmsh.initialize(["-v", "0"], interruptible=False)      # also from a worker thread (flow nodes)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("team7")
    occ = gmsh.model.occ
    plate = occ.addBox(0, 0, 0, 0.294, 0.294, 0.019)
    hole = occ.addBox(0.018, 0.018, 0, 0.108, 0.108, 0.019)
    plate_cut, _ = occ.cut([(3, plate)], [(3, hole)])
    ro = occ.addRectangle(0.094, 0.0, Z_COIL[0], 0.200, 0.200, roundedRadius=R_OUT)
    ri = occ.addRectangle(0.119, 0.025, Z_COIL[0], 0.150, 0.150, roundedRadius=R_IN)
    ring, _ = occ.cut([(2, ro)], [(2, ri)])
    coil = occ.extrude(ring, 0, 0, Z_COIL[1] - Z_COIL[0])
    coil_vol = [e for e in coil if e[0] == 3]
    lo = np.array([0.0, 0.0, 0.0]) - margin
    hi = np.array([0.294, 0.294, Z_COIL[1]]) + margin
    box = occ.addBox(*lo, *(hi - lo))
    occ.fragment([(3, box)], plate_cut + coil_vol)
    occ.synchronize()
    vols = [t for _, t in gmsh.model.getEntities(3)]

    def bbox(v):
        return np.array(gmsh.model.getBoundingBox(3, v))

    tol = 1e-5                                    # OCC bounding boxes carry the shape tolerance
    plate_v = [v for v in vols if bbox(v)[5] <= 0.019 + tol and bbox(v)[2] >= -tol and bbox(v)[3] <= 0.294 + tol
               and bbox(v)[0] >= -tol]
    coil_v = [v for v in vols if bbox(v)[2] >= Z_COIL[0] - tol and bbox(v)[5] <= Z_COIL[1] + tol]
    air_v = [v for v in vols if v not in plate_v and v not in coil_v]
    gmsh.model.addPhysicalGroup(3, plate_v, 1, "plate")
    gmsh.model.addPhysicalGroup(3, coil_v, 2, "coil")
    gmsh.model.addPhysicalGroup(3, air_v, 3, "air")
    outer = gmsh.model.getBoundary([(3, v) for v in vols], combined=True, oriented=False)
    gmsh.model.addPhysicalGroup(2, [t for _, t in outer], 9, "outer")
    skin = gmsh.model.getBoundary([(3, v) for v in plate_v], combined=True, oriented=False)
    gmsh.model.addPhysicalGroup(2, [t for _, t in skin], 8, "plate_skin")
    f = gmsh.model.mesh.field
    f.add("Box", 1)
    for k, v in zip(("VIn", "VOut", "XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "Thickness"),
                    (h_plate, h_air, 0.0, 0.294, 0.0, 0.294, 0.0, 0.019, 0.25), strict=True):
        f.setNumber(1, k, v)
    f.add("Box", 2)
    for k, v in zip(("VIn", "VOut", "XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "Thickness"),
                    (h_coil, h_air, 0.094, 0.294, 0.0, 0.2, Z_COIL[0], Z_COIL[1], 0.25), strict=True):
        f.setNumber(2, k, v)
    fields = [1, 2]
    if h_layer is not None:                      # the probe layer z = 19..49 mm
        f.add("Box", 3)
        for k, v in zip(("VIn", "VOut", "XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax", "Thickness"),
                        (h_layer, h_air, 0.0, 0.294, 0.0, 0.294, 0.019, Z_COIL[0], 0.25), strict=True):
            f.setNumber(3, k, v)
        fields.append(3)
    f.add("Min", 4)
    f.setNumbers(4, "FieldsList", fields)
    f.setAsBackgroundMesh(4)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(path))
    gmsh.finalize()
    return path


def probe_lines() -> dict[str, np.ndarray]:
    """Probe points (m) of each measured line; points on the plate faces or the hole edges are moved 0.1 mm into
    the aluminium (the eddy current is only defined there)."""
    eps = 1e-4
    out = {}
    for name, (y, z) in LINES.items():
        xs = (X_B if name in ("A1B1", "A2B2") else X_J).copy()
        if name in ("A3B3", "A4B4"):
            xs = np.where(xs == 0.0, eps * 1e3, xs)
            xs = np.where(xs == 18.0, 18.0 - eps * 1e3, xs)
            xs = np.where(xs == 126.0, 126.0 + eps * 1e3, xs)
            z = z - eps * 1e3 if z > 0 else z + eps * 1e3
        out[name] = np.column_stack([xs / 1e3, np.full(len(xs), y / 1e3), np.full(len(xs), z / 1e3)])
    return out


def solve(msh, f: float, mesh: TetMesh | None = None, space: EdgeSpace | None = None, **kw):
    """Native A* solution of TEAM 7 at frequency ``f`` on the gmsh file ``msh``."""
    mesh = mesh or read_msh22(msh)
    space = space or EdgeSpace(mesh)
    sigma = np.where(mesh.mask("plate"), SIGMA, 0.0)
    return solve_eddy(mesh, NU0, sigma, 2 * math.pi * f, coil_current_density, J_mask=mesh.mask("coil"),
                      space=space, **kw)


def probe(sol) -> dict:
    """Computed values on the measured lines, in the units of :data:`MEASURED`: ``{(line, wt): values}`` for wt
    = 0 and 90 deg. ``Bz`` is averaged over the tetrahedra around each probe point (it is constant per
    tetrahedron); the eddy current is the A* field at the point."""
    pts = probe_lines()
    out = {}
    for name, P in pts.items():
        if name in ("A1B1", "A2B2"):
            v = _bz_average(sol, P) * 1e4                      # T -> gauss
        else:
            v = sol.J_at(P)[:, 1] / 1e6
        out[(name, 0)] = np.real(v)
        out[(name, 90)] = -np.imag(v)
    return out


def _bz_average(sol, P, radius: float = 0.004) -> np.ndarray:
    """Volume-weighted mean ``Bz`` of the tetrahedra whose centroids lie within ``radius`` of each point (the
    point's own tetrahedron if none)."""
    from scipy.spatial import cKDTree

    m = sol.mesh
    tree = cKDTree(m.centroids)
    own = m.locate(P)
    out = np.zeros(len(P), complex)
    for i, p in enumerate(P):
        idx = tree.query_ball_point(p, radius)
        idx = idx if len(idx) else [own[i]]
        w = m.volumes[idx]
        out[i] = (sol.B[idx, 2] * w).sum() / w.sum()
    return out


OTHER_FACE = {"A3B3": "A4B4", "A4B4": "A3B3"}


def compare(computed: dict, f: int) -> dict:
    """Differences against the measured values per line and instant: rms, max and the rms relative to the
    largest measured magnitude on the line. For the eddy-current lines, also the rms relative to the table of the
    other face of the plate (``rms_rel_peak_other_face``): the computed current is larger on the face nearer the
    coil (z = 19 mm), as the skin depth (12 mm at 50 Hz, 6 mm at 200 Hz) requires, while the table puts the
    larger values on A4-B4 (z = 0); both pairings are reported."""
    rows = {}
    for (line, ff, wt), meas in MEASURED.items():
        if ff != f:
            continue
        c = np.asarray(computed[(line, wt)])
        mvals = np.asarray(meas)
        d = c - mvals
        row = {"rms": float(np.sqrt(np.mean(d ** 2))), "max": float(np.abs(d).max()),
               "rms_rel_peak": float(np.sqrt(np.mean(d ** 2)) / np.abs(mvals).max()),
               "computed": c.tolist(), "measured": mvals.tolist()}
        if line in OTHER_FACE:
            o = np.asarray(MEASURED[(OTHER_FACE[line], ff, wt)])
            row["rms_rel_peak_other_face"] = float(np.sqrt(np.mean((c - o) ** 2)) / np.abs(o).max())
        rows[f"{line} wt={wt}"] = row
    return rows


def run(msh, freqs=(50, 200), **kw) -> dict:
    """Solve and compare at each frequency; sizes and times included."""
    t0 = time.perf_counter()
    mesh = read_msh22(msh)
    space = EdgeSpace(mesh)
    out = {"mesh": mesh.summary(), "unknowns": None, "f": {}}
    for f in freqs:
        sol = solve(msh, f, mesh=mesh, space=space, **kw)
        out["unknowns"] = sol.info["unknowns"]
        out["f"][f] = {"compare": compare(probe(sol), f), "loss_W": sol.loss, "info": sol.info}
    out["seconds"] = time.perf_counter() - t0
    return out
