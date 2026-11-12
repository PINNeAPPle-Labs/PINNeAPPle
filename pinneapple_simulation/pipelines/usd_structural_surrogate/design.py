"""The parametric family: the UR10 mounting pedestal of the USD asset, as a design space and a gmsh geometry.

The part is the one in the asset -- a round base plate that is bolted down, a column up to the robot flange, and
the transition between them -- so the analysis is of the real part, not of a stand-in. Four continuous design
variables, all dimensions in metres:

===========  ==================  =======================  ==================================================
variable     meaning             bounds (nominal)         notes
===========  ==================  =======================  ==================================================
``t``        base plate thickness 6 - 16 mm (6.0)          nominal measured on the asset
``dc``       column diameter     50 - 110 mm (60)          the UR10 flange interface is ~54 mm
``r``        flare radius        6 - 35 mm (12)            the conical transition; a sharp corner would make
                                                          the peak stress mesh-dependent, so it is a cone
``h``        load height         300 - 500 mm (500)        the arm applies its couple this high up
===========  ==================  =======================  ==================================================

The base plate **diameter is not a design variable**: it is the asset's measured 271.9 mm, held fixed, so that
every design in the family carries the same mounting footprint. Material is structural steel; the step is linear
elastic and static, so the result is computed for a **reference load** of ``REFERENCE_LOAD_N`` = 1000 N applied
horizontally at the top of the column, and any other load scales it exactly (checked in :mod:`.checks`).
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["VARIABLES", "NOMINAL", "BOUNDS", "DesignSpace", "bracket_geo", "profile", "revolve_surface", "run_design", "REFERENCE_LOAD_N"]

# design variable -> (low, nominal, high, unit hint); nominal is the asset's measured value where it has one
VARIABLES: dict[str, tuple] = {
    "t": (0.006, 0.006, 0.016),        # base plate thickness, m -- nominal from the USD asset (6.0 mm)
    "dc": (0.050, 0.060, 0.110),       # column diameter, m
    "r": (0.006, 0.012, 0.035),        # flare radius, m
    "h": (0.300, 0.500, 0.500),        # load height, m -- the asset is 515 mm tall
}
BOUNDS: dict[str, tuple[float, float]] = {k: (v[0], v[2]) for k, v in VARIABLES.items()}
NOMINAL: dict[str, float] = {k: v[1] for k, v in VARIABLES.items()}

PLATE_DIAMETER_M = 0.2719            # measured on Props/Mounts/ur10_mount.usd (mount_interface)
REFERENCE_LOAD_N = 1000.0            # the load every sample is solved at; fields scale with it exactly


@dataclass
class DesignSpace:
    """The family: a fixed mounting footprint (from the asset) and four bounded design variables."""
    plate_diameter: float = PLATE_DIAMETER_M
    bounds: dict[str, tuple[float, float]] = None
    reference_load: float = REFERENCE_LOAD_N
    asset: dict[str, Any] | None = None

    def __post_init__(self):
        if self.bounds is None:
            self.bounds = dict(BOUNDS)

    @classmethod
    def from_interface(cls, interface: dict[str, Any] | None = None) -> DesignSpace:
        """The family anchored to a measured mounting interface (``usd_asset.mount_interface``).

        The plate diameter and the nominal plate thickness come from the asset when it was read; otherwise the
        recorded values in ``usd_asset.RECORDED_INTERFACE`` are used and ``asset`` says so.
        """
        from .usd_asset import RECORDED_INTERFACE
        iface = dict(interface or RECORDED_INTERFACE)
        d = float(iface.get("plate_diameter_mm", 271.9)) / 1000.0
        t0 = float(iface.get("plate_thickness_mm", 6.0)) / 1000.0
        bounds = dict(BOUNDS)
        bounds["t"] = (round(t0, 4), 0.016)              # the asset's plate thickness is the thinnest allowed
        return cls(plate_diameter=d, bounds=bounds, reference_load=REFERENCE_LOAD_N, asset=iface)

    # ------------------------------------------------------------------ sampling
    def sample(self, n: int, *, seed: int = 0, method: str = "lhs") -> list[dict[str, float]]:
        """``n`` designs: Latin hypercube (default) or a full-factorial of the two levels of each variable."""
        names = list(self.bounds)
        lo = np.array([self.bounds[k][0] for k in names])
        hi = np.array([self.bounds[k][1] for k in names])
        if method == "factorial":
            grid = np.meshgrid(*[np.array([a, b]) if b > a else np.array([a])
                                 for a, b in zip(lo, hi, strict=True)])
            flat = np.stack([g.ravel() for g in grid], axis=1)
        else:
            rng = np.random.default_rng(seed)
            u = np.empty((n, len(names)))
            for j in range(len(names)):
                u[:, j] = (rng.permutation(n) + rng.random(n)) / n      # one point per stratum, per variable
            if method != "lhs":                                       # explicit grid, reproducible
                u = np.stack(np.meshgrid(*[np.linspace(0, 1, int(round(n ** (1 / len(names)))))
                                           for _ in names]), axis=-1).reshape(-1, len(names))
            flat = lo + u[:n] * (hi - lo)
        designs = [{k: float(v) for k, v in zip(names, row, strict=True)} for row in flat]
        for d in designs:
            self.validate(d)
        return designs

    def validate(self, design: dict[str, float]) -> None:
        for k, (a, b) in self.bounds.items():
            if k not in design:
                raise ValueError(f"design is missing {k!r}")
            if not (a - 1e-12 <= float(design[k]) <= b + 1e-12):
                raise ValueError(f"{k}={design[k]:g} outside [{a:g}, {b:g}]")

    def contains(self, design: dict[str, float], tol: float = 1e-9) -> bool:
        """Is every variable of ``design`` inside its bounds? (the family's validity envelope)"""
        return all(a - tol <= float(design[k]) <= b + tol
                   for k, (a, b) in self.bounds.items() if k in design)

    def normalized(self, design: dict[str, float]) -> np.ndarray:
        """The design mapped to [0, 1] per variable (what a dashboard slider carries)."""
        return np.array([(float(design[k]) - a) / (b - a) for k, (a, b) in self.bounds.items()])

    def denormalized(self, u: Iterable[float]) -> dict[str, float]:
        u = np.clip(np.asarray(list(u), float), 0.0, 1.0)
        return {k: float(a + ui * (b - a)) for (k, (a, b)), ui in zip(self.bounds.items(), u, strict=True)}

    def as_dict(self) -> dict[str, Any]:
        return {"plate_diameter_m": self.plate_diameter, "reference_load_n": self.reference_load,
                "bounds": {k: list(v) for k, v in self.bounds.items()},
                "nominal": self.denormalized([0.5] * len(self.bounds)),
                "asset": self.asset}


def bracket_geo(design: dict[str, float], plate_diameter: float = PLATE_DIAMETER_M,
                size: float | None = None) -> str:
    """gmsh (OpenCASCADE) geometry of one design: base plate, conical flare, column.

    The plate sits on ``z = 0`` (its bottom face is the clamped face), centred on the column axis; the flare spans
    ``z = t .. t + r`` and the column runs to ``z = h``, where the load is applied. Nothing about the topology
    changes with the design -- only the four dimensions -- so every mesh in the family has the same surface
    structure and the fields land on the same grid.
    """
    t, dc, r, h = (float(design[k]) for k in ("t", "dc", "r", "h"))
    if r >= h - t:
        raise ValueError(f"flare radius {r:g} does not fit under the load height {h:g}")
    rc = dc / 2.0
    rf = rc + r
    if rf >= plate_diameter / 2.0:
        raise ValueError(f"flare outer radius {rf:g} exceeds the plate radius {plate_diameter / 2:g}")
    lc = f"Mesh.CharacteristicLengthMax = {size:.6g};\n" if size else ""
    return f"""SetFactory("OpenCASCADE");
// base plate (the asset's mounting footprint), bolted down at z = 0
Cylinder(1) = {{0, 0, 0, 0, 0, {t:.6g}, {plate_diameter / 2:.6g}}};
// conical flare: the fillet that carries the moment from the column into the plate
Cone(2) = {{0, 0, {t:.6g}, 0, 0, {r:.6g}, {rf:.6g}, {rc:.6g}}};
// column up to the robot flange
Cylinder(3) = {{0, 0, {t + r:.6g}, 0, 0, {h - t - r:.6g}, {rc:.6g}}};
BooleanUnion(4) = {{ Volume{{1}}; Delete; }}{{ Volume{{2, 3}}; Delete; }};
{lc}Physical Volume("part") = {{4}};
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 24;
"""


def run_design(design: dict[str, float], workdir: str, *, element_size: float = 0.020, order: int = 1,
               material: Any = None, threads: int = 2, keep: bool = True) -> dict[str, Any]:
    """Mesh one design with gmsh, solve it with ``ccx``, and return the nodal fields.

    Returns ``{"design", "nodes", "elements", "u", "von_mises", "load", "info"}`` with ``u`` (n, 3) in metres and
    ``von_mises`` (n,) in Pa, both for ``REFERENCE_LOAD_N`` applied horizontally at the top of the column.
    """
    from pinneapple_simulation.external_solvers.calculix.study import (
        FEModel,
        Material,
        Static,
        solve,
    )

    os.makedirs(workdir, exist_ok=True)
    model = FEModel.from_gmsh(bracket_geo(design), size=element_size, order=order, workdir=workdir)
    model.material = material or Material("steel", E=210e9, nu=0.3, density=7850.0)
    model.title = "UR10 pedestal " + ", ".join(f"{k}={v:g}" for k, v in sorted(design.items()))

    X = model.nodes
    zmin, zmax = float(X[:, 2].min()), float(X[:, 2].max())
    tol = 1e-6 * max(1.0, zmax)
    base = model.nodes_where(lambda P: P[:, 2] <= zmin + tol)                  # the bolted bottom face
    top = model.nodes_where(lambda P: P[:, 2] >= zmax - tol)                   # the flange where the arm pulls
    if len(base) < 3 or len(top) < 3:
        raise RuntimeError(f"could not select the clamped/top node sets ({len(base)}, {len(top)})")

    step = Static(fix=[(base, (1, 2, 3))], loads=[(top, (REFERENCE_LOAD_N, 0.0, 0.0))])
    res = solve(model, step, workdir, threads=threads, keep=keep)
    return {"design": dict(design), "nodes": np.asarray(model.nodes, float),
            "elements": np.asarray(model.elements, int), "element_type": model.element,
            "u": np.asarray(res.u, float), "von_mises": np.asarray(res.von_mises, float),
            "load": {"total_n": REFERENCE_LOAD_N, "direction": [1.0, 0.0, 0.0],
                     "applied_at": [0.0, 0.0, zmax]},
            "info": {"n_nodes": int(len(X)), "n_elements": int(len(model.elements)),
                     "element": model.element, "base_nodes": int(len(base)), "top_nodes": int(len(top)),
                     "nodal": asdict(model.material) if hasattr(model.material, "__dataclass_fields__") else {}}}


def profile(design: dict[str, float], plate_diameter: float = PLATE_DIAMETER_M) -> np.ndarray:
    """The part's half-section, (r, z) from the axis at the bottom to the axis at the top.

    The pedestal is a solid of revolution, so this nine-point polyline *is* the geometry: plate bottom, plate side,
    plate top, cone, column side, column top. It is what the dashboard revolves on every slider move (microseconds,
    against seconds for gmsh) and it reproduces :func:`bracket_geo` exactly.
    """
    t, dc, r, h = (float(design[k]) for k in ("t", "dc", "r", "h"))
    R, rc, rf = plate_diameter / 2.0, dc / 2.0, dc / 2.0 + r
    return np.array([[0.0, 0.0], [R, 0.0], [R, t], [rf, t], [rc, t + r], [rc, h], [0.0, h]], float)


def revolve_surface(design: dict[str, float], plate_diameter: float = PLATE_DIAMETER_M,
                    n_theta: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Triangle surface of one design, by revolving :func:`profile`: ``(vertices (V,3), faces (F,3))``.

    Duplicate vertices along the axis are collapsed, so the caps are fans and the mesh is watertight.
    """
    p = profile(design, plate_diameter)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos, sin = np.cos(th), np.sin(th)
    verts, index = [], []

    def ring(r: float, z: float) -> np.ndarray:
        if r <= 0.0:
            return np.array([[0.0, 0.0, z]])
        return np.stack([r * cos, r * sin, np.full(n_theta, z)], axis=1)

    rows = []
    for r, z in p:
        rows.append(ring(r, z))
        index.append(len(verts))
        verts.extend(rows[-1])
    faces = []
    for k in range(len(p) - 1):
        a, b = rows[k], rows[k + 1]
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:                                    # bottom cap fan
                faces.append([index[k], index[k + 1] + jn, index[k + 1] + j])
            elif len(b) == 1:                                  # top cap fan
                faces.append([index[k] + j, index[k] + jn, index[k + 1]])
            else:
                faces.append([index[k] + j, index[k] + jn, index[k + 1] + j])
                faces.append([index[k] + jn, index[k + 1] + jn, index[k + 1] + j])
    return np.asarray(verts, float), np.asarray(faces, int)
