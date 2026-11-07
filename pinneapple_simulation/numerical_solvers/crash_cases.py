"""The pole-impact case of a bumper beam on its two crash boxes — a crash-test coupon the solver can afford.

The case is the component-scale half of a crash programme: a DP1000 bumper beam held by two DP600 crash boxes,
driven into a rigid pole, exactly the object of the pole-impact benchmark (design vector -> crashworthiness
quantities, and the displacement sequence of every node). It is built on a **fixed hexahedral mesh** so that every
design shares the same node set — what a field surrogate (and a spatial comparison between designs) needs — and the
design variables act through the material and the obstacle:

- ``velocity``          impact speed [m/s];
- ``box_thickness``     crash-box wall thickness, through its flow stress and mass (see the note below);
- ``beam_thickness``    bumper-beam thickness, the same way;
- ``dp600_yield``       initial yield stress of the crash boxes [Pa];
- ``dp1000_yield``      initial yield stress of the bumper beam [Pa];
- ``pole_radius``       radius of the rigid pole [m];
- ``pole_offset``       lateral offset of the pole from the assembly centre line [m].

Note on thickness on a fixed mesh: the mesh keeps the walls at the reference thickness ``t_ref`` and a design
``t`` scales the flow stress and the density of the part by ``t / t_ref`` (the axial crush force of a thin-walled
column grows with the wall thickness, and so does its mass). Bending stiffness would scale as ``t^3`` and is
therefore under-represented; the quantities this case reports (peak wall force, energy absorbed, pulse duration) are
axial-crush dominated. Say so when reporting a sweep: this is a coupon, not a full vehicle.

    from pinneapple_simulation.numerical_solvers.crash_cases import DESIGN_BOUNDS, bumper_beam_assembly, pole_impact

    assembly = bumper_beam_assembly()                      # reference design, fixed mesh, 3 parts
    res = pole_impact({"velocity": 16.0, "box_thickness": 0.002}, end_time=6e-4)
    print(res.qoi(), res.displacement.shape)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crash_fem import CrashResult, CrashSolver, Material, RigidPole
from .solid_fem import HexMesh

__all__ = ["DESIGN_BOUNDS", "DESIGN_DEFAULTS", "DP600", "DP1000", "Assembly", "bumper_beam_assembly",
           "pole_impact", "sample_design", "cross_section_scaled_material"]

# Seven design variables: the same seven of the pole-impact corpus (velocity, the two thicknesses, the two flow
# stresses, the pole radius and its lateral offset).
DESIGN_BOUNDS = {
    "velocity": (14.0, 18.0),            # m/s (50 - 64 km/h)
    "box_thickness": (0.0015, 0.0035),   # m
    "beam_thickness": (0.0015, 0.0035),  # m
    "dp600_yield": (3.2e8, 4.6e8),       # Pa
    "dp1000_yield": (6.2e8, 7.8e8),      # Pa
    "pole_radius": (0.030, 0.090),       # m
    "pole_offset": (-0.030, 0.030),      # m
}
DESIGN_DEFAULTS = {k: 0.5 * (lo + hi) for k, (lo, hi) in DESIGN_BOUNDS.items()}

DP600 = dict(rho=7.8e3, E=2.1e11, nu=0.30, yield_stress=3.9e8, hardening=4.0e8, erosion_strain=0.30)
DP1000 = dict(rho=7.8e3, E=2.1e11, nu=0.30, yield_stress=7.0e8, hardening=5.0e8, erosion_strain=0.25)

T_REF = 0.0025                          # reference wall thickness the fixed mesh is built at [m]
H_CELL = 0.006                          # cell size [m]

BOX_LENGTH, BOX_SIDE, BOX_SPACING = 0.066, 0.048, 0.192   # crash boxes: length, outer side (outer edge to edge)
BEAM_SPAN, BEAM_HEIGHT = 0.192, 0.048                     # bumper beam: width, height
GAP = 0.003                                               # clearance between the beam face and the pole


@dataclass
class Assembly:
    """A built case: the mesh, the part id of every element, the materials and where the pole sits."""
    mesh: HexMesh
    parts: np.ndarray
    materials: dict[int, Material]
    part_names: dict[int, str]
    pole_radius: float
    pole_x: float
    pole_offset: float

    def info(self) -> dict:
        return {"nodes": int(len(self.mesh.nodes)), "elements": int(len(self.mesh.elements)),
                "parts": {int(k): int(v) for k, v in zip(*np.unique(self.parts, return_counts=True), strict=False)},
                "part_names": {int(k): v for k, v in self.part_names.items()},
                "pole": {"radius": self.pole_radius, "x": self.pole_x, "offset": self.pole_offset}}


def cross_section_scaled_material(base: dict, thickness: float, t_ref: float = T_REF) -> Material:
    """A :class:`Material` that stands for a wall of thickness ``thickness`` on the mesh built at ``t_ref``.

    The flow stress and the density scale with ``thickness / t_ref`` (axial crush force and mass of a thin-walled
    column), the elastic modulus does not (the coupon's wave speed stays physical). See the module note."""
    scale = float(thickness) / float(t_ref)
    if scale <= 0:
        raise ValueError("thickness must be positive")
    return Material(rho=base["rho"] * scale, E=base["E"], nu=base["nu"], yield_stress=base["yield_stress"] * scale,
                    hardening=base["hardening"] * scale, erosion_strain=base["erosion_strain"])


def bumper_beam_assembly(*, box_thickness: float = T_REF, beam_thickness: float = T_REF,
                         dp600_yield: float = DP600["yield_stress"], dp1000_yield: float = DP1000["yield_stress"],
                         pole_radius: float = 0.060, pole_offset: float = 0.0, h: float = H_CELL,
                         t_ref: float = T_REF, n_parts: int = 3) -> Assembly:
    """Build the fixed-mesh assembly (see the module docstring). Every design shares this mesh and these nodes.

    Geometry (metres, x is the impact direction, z up, y across): two square crash boxes with their rear face at
    ``x = 0`` (held), walls one cell thick, spanning ``x`` up to :data:`BOX_LENGTH`; the bumper beam as a slab on
    their front face, two cells thick, spanning the box spacing in ``y``. The pole stands :data:`GAP` ahead of it.

    ``n_parts`` splits the three structural groups into finer semantic parts (3: beam + two boxes; 6: each group in
    two halves; 12: the beam in three segments and each box in two), so a part-aware surrogate can be exercised at
    several hierarchy depths on the same mesh."""
    nbx = int(round(BOX_LENGTH / h))
    nby = int(round(BOX_SPACING / h))
    nbz = int(round(BOX_SIDE / h))
    nbeam = 2                                                          # the beam is two cells thick
    nx = nbx + nbeam
    xs_n = np.arange(nx + 1) * h
    ys_n = -BOX_SPACING / 2 + np.arange(nby + 1) * h
    zs_n = np.arange(nbz + 1) * h
    Xn, Yn, Zn = np.meshgrid(xs_n, ys_n, zs_n, indexing="ij")
    i, j, k = np.meshgrid(np.arange(nx), np.arange(nby), np.arange(nbz), indexing="ij")
    y, z = -BOX_SPACING / 2 + (j + 0.5) * h, (k + 0.5) * h               # cell centres
    t_wall = float(t_ref)                                              # walls are one cell thick at the reference
    z_c = BOX_SIDE / 2
    y_left, y_right = -BOX_SPACING / 2 + BOX_SIDE / 2, BOX_SPACING / 2 - BOX_SIDE / 2

    in_box_x = i < nbx
    on_wall_y = (np.abs(y - y_left) >= BOX_SIDE / 2 - 1.5 * t_wall) & (np.abs(y - y_left) <= BOX_SIDE / 2) | \
                (np.abs(y - y_right) >= BOX_SIDE / 2 - 1.5 * t_wall) & (np.abs(y - y_right) <= BOX_SIDE / 2)
    in_box_z = np.abs(z - z_c) <= BOX_SIDE / 2
    on_wall_z = np.abs(z - z_c) >= BOX_SIDE / 2 - 1.5 * t_wall
    box_shell = in_box_x & in_box_z & (on_wall_y | on_wall_z)
    beam = (i >= nbx) & (np.abs(y) <= BOX_SPACING / 2 - 1.0 * t_wall) & in_box_z

    mask = (box_shell | beam).ravel()
    nodes = np.stack([Xn.ravel(), Yn.ravel(), Zn.ravel()], 1)
    nx_, ny_, nz_ = nx, nby, nbz
    nid = lambda a, b, c: (a * (ny_ + 1) + b) * (nz_ + 1) + c          # noqa: E731
    cells = np.argwhere(mask.reshape(nx_, ny_, nz_))
    elems = np.stack([nid(cells[:, 0], cells[:, 1], cells[:, 2]), nid(cells[:, 0] + 1, cells[:, 1], cells[:, 2]),
                      nid(cells[:, 0] + 1, cells[:, 1] + 1, cells[:, 2]), nid(cells[:, 0], cells[:, 1] + 1,
                                                                             cells[:, 2]),
                      nid(cells[:, 0], cells[:, 1], cells[:, 2] + 1), nid(cells[:, 0] + 1, cells[:, 1],
                                                                          cells[:, 2] + 1),
                      nid(cells[:, 0] + 1, cells[:, 1] + 1, cells[:, 2] + 1), nid(cells[:, 0], cells[:, 1] + 1,
                                                                                  cells[:, 2] + 1)], 1)
    if len(elems) == 0:
        raise ValueError("the assembly came out empty: check the dimensions against the cell size")
    return _finish_assembly(nodes, elems, cells, (nx_, ny_, nz_), (nx * h, nby * h, nbz * h), pole_radius,
                            pole_offset, box_thickness, beam_thickness, dp600_yield, dp1000_yield, t_ref, n_parts)


def _finish_assembly(nodes, elems, cells, shape, size, pole_radius, pole_offset, box_thickness, beam_thickness,
                     dp600_yield, dp1000_yield, t_ref, n_parts) -> Assembly:
    """Keep only the nodes the elements use, assign the part ids and the materials."""
    used = np.unique(elems.ravel())
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    mesh = HexMesh(nodes[used], remap[elems], shape, size)

    icell, ycell = cells[:, 0], cells[:, 1]
    beam = icell >= (icell.max() - 1)                                  # the two front cell layers
    y_mid = (shape[1] - 1) / 2
    left = ycell <= y_mid

    parts = np.where(beam, 1, np.where(left, 2, 3)).astype(int)
    names = {1: "bumper_beam", 2: "crash_box_left", 3: "crash_box_right"}
    if n_parts >= 6:
        x_box = icell[~beam]
        x_cut = np.median(x_box) if x_box.size else 0
        beam_left = beam & left
        parts[beam_left] = 4
        names[4] = "bumper_beam_right"
        rear = (~beam) & (icell <= np.floor(x_cut))
        parts[rear & left] = 5
        parts[rear & ~left] = 6
        names[5], names[6] = "crash_box_left_rear", "crash_box_right_rear"
    if n_parts >= 12:
        y_beam = ycell[beam]
        cuts = np.quantile(y_beam, [1 / 3, 2 / 3]) if y_beam.size else np.array([0.0, 0.0])
        parts[beam & (ycell <= cuts[0])] = 7
        parts[beam & (ycell > cuts[0]) & (ycell <= cuts[1])] = 8
        parts[beam & (ycell > cuts[1])] = 9
        names.update({7: "bumper_beam_y1", 8: "bumper_beam_y2", 9: "bumper_beam_y3"})
        front = (~beam) & (icell > np.floor(x_cut))
        parts[front & left] = 10
        parts[front & ~left] = 11
        names[10], names[11] = "crash_box_left_front", "crash_box_right_front"

    beam_mat = cross_section_scaled_material({**DP1000, "yield_stress": dp1000_yield}, beam_thickness, t_ref)
    box_mat = cross_section_scaled_material({**DP600, "yield_stress": dp600_yield}, box_thickness, t_ref)
    materials = {}
    for p in sorted(set(parts.tolist())):
        name = names.get(int(p), f"part_{p}")
        materials[int(p)] = beam_mat if name.startswith("bumper_beam") else box_mat
    x_max = float(mesh.nodes[:, 0].max())
    # the pole is a cylinder: its *surface* stands GAP ahead of the beam face, for every radius, so the impact
    # starts at GAP / velocity in every design of the sweep
    return Assembly(mesh=mesh, parts=parts, materials=materials, part_names=names,
                    pole_radius=float(pole_radius), pole_x=x_max + GAP + float(pole_radius),
                    pole_offset=float(pole_offset))


def sample_design(rng: np.random.Generator | None = None, n: int = 1) -> list[dict]:
    """``n`` designs from the design box (Latin-hypercube-like: one random point per stratum of every variable)."""
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for s in range(n):
        d = {}
        for k, (lo, hi) in DESIGN_BOUNDS.items():
            u = (s + rng.random()) / max(n, 1) if n > 1 else rng.random()
            d[k] = float(lo + (hi - lo) * u)
        out.append(d)
    return out


def pole_impact(design: dict | None = None, *, end_time: float = 1.5e-3, frames: int = 21,
                n_parts: int = 3, cfl: float = 0.9, record_velocity: bool = False,
                material_override: dict | None = None) -> CrashResult:
    """Run one design of the coupon (see the module docstring) and return the full result.

    ``design`` is a partial or complete design vector (missing keys take :data:`DESIGN_DEFAULTS`). The coupon is the
    sled view of the pole impact: the rear face of the crash boxes is held (the body it is bolted to) and the rigid
    pole travels at the impact velocity into it, so the crush is driven by the pole's travel and the impact starts
    at ``GAP / velocity`` in every design of the sweep."""
    d = dict(DESIGN_DEFAULTS)
    d.update(design or {})
    for k in DESIGN_BOUNDS:
        lo, hi = DESIGN_BOUNDS[k]
        if not lo <= d[k] <= hi:
            raise ValueError(f"{k}={d[k]} is outside its design range ({lo}, {hi})")
    asm = bumper_beam_assembly(box_thickness=d["box_thickness"], beam_thickness=d["beam_thickness"],
                               dp600_yield=d["dp600_yield"], dp1000_yield=d["dp1000_yield"],
                               pole_radius=d["pole_radius"], pole_offset=d["pole_offset"], n_parts=n_parts)
    if material_override:
        for part, mat in material_override.items():
            asm.materials[part] = mat
    fixed = asm.mesh.nodes_on(x=0.0)                    # the rear face of the crash boxes is held (the body's rest)
    pole = RigidPole(radius=asm.pole_radius, x0=asm.pole_x, y0=asm.pole_offset, vx=-d["velocity"],
                     z_range=(float(asm.mesh.nodes[:, 2].min()) - 1.0, float(asm.mesh.nodes[:, 2].max()) + 1.0))
    solver = CrashSolver(asm.mesh, material=asm.materials, parts=asm.parts, contact=pole, fixed=fixed,
                         initial_velocity=(0.0, 0.0, 0.0), end_time=end_time, frames=frames,
                         record_velocity=record_velocity, cfl=cfl)
    res = solver.run()
    res.info = {"design": dict(d), "assembly": asm.info(), "fixed_nodes": int(len(fixed)),
                "onset_time": GAP / d["velocity"]}
    return res
