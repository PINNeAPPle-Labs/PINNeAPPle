"""The structural half of the loop: the arm as a steel tube, sized by bending, buckling and a 3-D finite-element check.

The A-arm is two legs of cold-drawn tube that meet at the ball-joint boss. Each leg is a strut: it carries the axial
force :mod:`pinneapple_design.suspension.loads` gives it, plus a bending moment, because the ball joint does not sit on
the leg's centroidal axis — a strut whose load line misses its axis by ``e`` carries ``N e`` all along, and the
fore-aft force adds its own moment at the apex. That is the closed form the sizing loop uses:

    sigma = N / A + M c / I            (axial plus bending at the outer fibre)
    P_cr  = pi^2 E I / L^2             (Euler, pinned-pinned: the bushing is no clamp)

and it is fast enough to be evaluated for every candidate the optimiser visits. The **finite-element check** is the
same tube as 8-node hexahedra (:class:`pinneapple_simulation.numerical_solvers.solid_fem.SolidFEM`, incompatible modes
so bending does not lock), clamped at the inner-pivot end and loaded at the ball-joint end with the same force and the
offset couple; its peak von Mises is the number the design is finally reported with, and its agreement with the closed
form is a check of its own (the surrogate does not grade itself — nor does the beam formula).

Both models share the limits stated in :mod:`pinneapple_design.suspension.study`: linear elasticity, no plasticity, no
welds, no fatigue damage (the stress is compared with a fraction of yield, which is a static margin, not a life), and
the tube is uniform along the leg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hardpoints import SuspensionSpec
from .kinematics import Pose
from .loads import arm_loads

__all__ = ["TubeSection", "arm_mass_kg", "arm_check", "tube_mesh", "fem_tube", "fem_lower_arm", "boss_mass_kg"]

#: mass of the ball-joint boss at the apex and of one bushing sleeve [kg] — estimates, not measured parts
BOSS_MASS_KG = 0.42
SLEEVE_MASS_KG = 0.13


@dataclass(frozen=True)
class TubeSection:
    """A round tube: outer diameter and wall [m]."""
    od_m: float
    wall_m: float

    def __post_init__(self):
        if not 0.0 < self.wall_m < 0.5 * self.od_m:
            raise ValueError("the wall must be positive and thinner than half the diameter")

    @property
    def r_out(self) -> float:
        return 0.5 * self.od_m

    @property
    def r_in(self) -> float:
        return 0.5 * self.od_m - self.wall_m

    @property
    def area_m2(self) -> float:
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)

    @property
    def inertia_m4(self) -> float:
        return math.pi / 4.0 * (self.r_out ** 4 - self.r_in ** 4)

    @property
    def polar_m4(self) -> float:
        return 2.0 * self.inertia_m4

    @property
    def c_m(self) -> float:
        return self.r_out

    @property
    def radius_of_gyration_m(self) -> float:
        return math.sqrt(self.inertia_m4 / self.area_m2)

    def mass_per_m(self, density_kg_m3: float) -> float:
        return self.area_m2 * density_kg_m3

    def describe(self) -> str:
        return f"{self.od_m * 1e3:.1f} x {self.wall_m * 1e3:.1f} mm tube"


def boss_mass_kg() -> float:
    """The non-tube mass of one corner's arms: two apex bosses and four bushing sleeves."""
    return 2 * BOSS_MASS_KG + 4 * SLEEVE_MASS_KG


def _arm_leg_lengths(spec: SuspensionSpec) -> dict[str, float]:
    lo, uo = spec.lca_outer, spec.uca_outer
    return {"lca_front_leg": float(np.linalg.norm(lo - spec.lca_front)),
            "lca_rear_leg": float(np.linalg.norm(lo - spec.lca_rear)),
            "uca_front_leg": float(np.linalg.norm(uo - spec.uca_front)),
            "uca_rear_leg": float(np.linalg.norm(uo - spec.uca_rear))}


def arm_mass_kg(spec: SuspensionSpec, section: TubeSection | None = None) -> float:
    """Mass of the corner's two A-arms [kg]: four tube legs plus the bosses and sleeves.

    Returns
    -------
    mass : float [kg]
    """
    sec = section or TubeSection(spec.tube_od_m, spec.tube_wall_m)
    tube = sum(_arm_leg_lengths(spec).values()) * sec.mass_per_m(spec.density_kg_m3)
    return float(tube + boss_mass_kg())


def arm_check(spec: SuspensionSpec, p: Pose, wheel_force, section: TubeSection | None = None) -> dict[str, float]:
    """Size the lower arm for one load case: stress, static margin, buckling factor and the mass that comes with it.

    The lower arm's two legs share the ball joint's force and the root moment between them, so each is checked for half
    of it; its buckling length is the longer leg, pinned at both ends. The upper arm is a two-force member and is
    checked for its own axial force.

    Returns
    -------
    check : dict
        ``sigma_vm_pa`` (outer fibre of one leg: axial plus bending), ``yield_pa``, ``static_margin``
        (``yield / sigma``), ``buckling_factor`` (``P_cr / N`` on the more loaded member), ``lower_axial_n``,
        ``lower_moment_nm``, ``upper_axial_n`` and ``mass_kg``.
    """
    sec = section or TubeSection(spec.tube_od_m, spec.tube_wall_m)
    lds = arm_loads(spec, p, wheel_force)
    leg = float(lds["leg_length_m"])
    axial = float(lds["lower_axial_n"]) / 2.0                     # positive: compression
    bending = float(lds["lower_moment_nm"]) / 2.0
    sigma = abs(axial) / sec.area_m2 + bending * sec.c_m / sec.inertia_m4
    p_cr = math.pi ** 2 * spec.elastic_modulus_pa * sec.inertia_m4 / leg ** 2
    buckling_lower = float(p_cr / axial) if axial > 1.0 else float("inf")
    upper = max(float(lds["upper_axial_n"]), 0.0)                 # positive: compression
    upper_length = float(np.linalg.norm(spec.uca_outer - spec.uca_inner_mid))
    p_cr_upper = math.pi ** 2 * spec.elastic_modulus_pa * sec.inertia_m4 / upper_length ** 2
    buckling_upper = float(p_cr_upper / upper) if upper > 1.0 else float("inf")
    return {"sigma_vm_pa": float(sigma), "yield_pa": spec.yield_pa,
            "static_margin": float(spec.yield_pa / sigma) if sigma > 0 else float("inf"),
            "buckling_factor": min(buckling_lower, buckling_upper), "euler_load_n": float(p_cr),
            "leg_length_m": leg, "lower_axial_n": float(lds["lower_axial_n"]),
            "lower_moment_nm": float(lds["lower_moment_nm"]), "upper_axial_n": float(lds["upper_axial_n"]),
            "mass_kg": arm_mass_kg(spec, sec), "section": sec.describe()}


# ------------------------------------------------------------------------------------------------- finite elements
def tube_mesh(length_m: float, section: TubeSection, n_theta: int = 24, n_radial: int = 2, n_len: int = 24):
    """Structured hexahedra of a tube from ``x = 0`` to ``length_m``, the axis along ``x``.

    Returns
    -------
    mesh : HexMesh
        Nodes and 8-node hexahedra (Abaqus order, positive volume), the root ring at ``x = 0``.
    """
    from pinneapple_simulation.numerical_solvers.solid_fem import HexMesh

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    radii = np.linspace(section.r_out, section.r_in, n_radial + 1)          # outer surface first
    xs = np.linspace(0.0, length_m, n_len + 1)
    nodes = np.array([[x, r * math.cos(t), r * math.sin(t)] for x in xs for t in theta for r in radii])
    idx = lambda i, j, k: (i * n_theta + j % n_theta) * (n_radial + 1) + k   # noqa: E731
    els = []
    for i in range(n_len):
        for j in range(n_theta):
            for k in range(n_radial):
                els.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                            idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)])
    els = np.asarray(els)
    mesh = HexMesh(nodes, els, (n_len, n_theta, n_radial), (length_m, section.od_m, section.od_m))
    _orient(mesh, length_m, section)
    return mesh


def _signed_volumes(nodes: np.ndarray, els: np.ndarray) -> np.ndarray:
    """Signed volume of each hexahedron, split into five tetrahedra along its diagonals (the split that tiles the
    hexahedron exactly, whatever the warp of its faces)."""
    p = [nodes[els[:, k]] for k in range(8)]
    v = np.zeros(len(els))
    for i, j, k, m in ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)):
        v += np.einsum("ij,ij->i", p[j] - p[i], np.cross(p[k] - p[i], p[m] - p[i])) / 6.0
    return v


def _orient(mesh, length_m: float, section: TubeSection) -> None:
    """Flip the elements whose winding gives a negative volume, then check the total against the tube's.

    The reference is the *polygonal* section (``n`` chords, not the circle): the mesh is inscribed, so its volume is a
    little under the cylinder's (0.5 % at 24 chords) and comparing with the circle would flag a correct mesh."""
    v = _signed_volumes(mesh.nodes, mesh.elements)
    bad = v < 0
    if bad.any():
        order = [0, 3, 2, 1, 4, 7, 6, 5]           # reverse each face's winding: the same element, opposite sense
        mesh.elements[bad] = mesh.elements[bad][:, order]
        v = _signed_volumes(mesh.nodes, mesh.elements)
    n = mesh.shape[1]
    exact = 0.5 * n * math.sin(2.0 * math.pi / n) * (section.r_out ** 2 - section.r_in ** 2) * length_m
    if abs(v.sum() - exact) > 1e-6 * exact + 1e-15:
        raise ValueError(f"the tube mesh is not closed: {v.sum():.6e} m3 against {exact:.6e} m3")


def fem_tube(spec: SuspensionSpec, section: TubeSection, length_m: float, force_n, moment_nm=(0.0, 0.0, 0.0),
             n_theta: int = 24, n_radial: int = 2, n_len: int = 24, load_offset_m=None):
    """The leg as a clamped tube under a tip force (and an offset couple): peak von Mises and tip deflection.

    The force and the couple are applied on the ring at ``x = length_m``; the couple is the one the ball joint's
    offset from the tube axis produces. The peak is read over the nodes away from both ends (``0.15 L .. 0.85 L``),
    where the clamped face and the load introduction do not dominate, and the closed form is evaluated at the same
    station: beam theory gives ``M(x) c / I + N / A`` there, and the two must agree — that agreement is the check.

    Returns
    -------
    result : dict
        ``sigma_peak_pa`` (finite elements), ``sigma_beam_pa`` (closed form at the same station), ``ratio``,
        ``tip_deflection_m``, ``fem`` (the :class:`FEMResult`), ``mesh`` and the nodes used for the field view.
    """
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM

    mesh = tube_mesh(length_m, section, n_theta=n_theta, n_radial=n_radial, n_len=n_len)
    fem = SolidFEM(mesh, E=spec.elastic_modulus_pa, nu=0.3)
    root = np.flatnonzero(mesh.nodes[:, 0] < 1e-12)
    tip = np.flatnonzero(mesh.nodes[:, 0] > length_m - 1e-12)
    fem.fix(root)
    F = np.asarray(force_n, float).reshape(3)
    M = np.asarray(moment_nm, float).reshape(3)
    fem.load_nodes(tip, F)
    if np.any(M):
        c = mesh.nodes[tip].mean(axis=0)
        d = mesh.nodes[tip] - c
        w = (d ** 2).sum(axis=1)
        # nodal forces that produce the couple M and no net force: f_i = M x d_i / sum |d|^2
        for n, f in zip(tip, np.cross(np.broadcast_to(M, d.shape), d) / w.sum(), strict=True):
            fem.load_nodes([n], f)
    res = fem.solve()
    x = mesh.nodes[:, 0]
    band = (x > 0.15 * length_m) & (x < 0.85 * length_m)
    vm = res.von_mises
    # the closed form at the station where the finite elements peak
    i = int(np.flatnonzero(band)[np.argmax(vm[band])])
    station = float(x[i])
    lever = length_m - station
    N = float(np.dot(F, np.array([1.0, 0.0, 0.0])))
    M_root = np.cross(np.array([length_m, 0.0, 0.0]) + (np.asarray(load_offset_m, float)
                                                        if load_offset_m is not None else 0.0), F) + M
    M_bend = float(np.linalg.norm(M_root)) * (lever / length_m)
    sigma_beam = abs(N) / section.area_m2 + M_bend * section.c_m / section.inertia_m4
    return {"sigma_peak_pa": float(vm[band].max()), "sigma_beam_pa": float(sigma_beam),
            "ratio": float(vm[band].max() / sigma_beam) if sigma_beam > 0 else float("nan"),
            "tip_deflection_m": float(np.linalg.norm(res.u[tip], axis=1).max()),
            "tip_deflection_beam_m": float(np.linalg.norm(F) * length_m ** 3
                                           / (3.0 * spec.elastic_modulus_pa * section.inertia_m4)),
            "elements": int(len(mesh.elements)), "nodes": int(len(mesh.nodes)),
            "fem": res, "mesh": mesh}


def fem_lower_arm(spec: SuspensionSpec, section: TubeSection, loads: dict, p: Pose | None = None,
                  n_theta: int = 24, n_radial: int = 2, n_len: int = 24) -> dict:
    """The lower arm as one leg in finite elements: the sizing case's axial force and root moment, halved between the
    two legs, as a tip force on a clamped tube.

    The root moment enters as an equivalent transverse force at the tip (``M / L``), which produces the same root
    moment; the sign of the force does not matter for von Mises. The result carries the :class:`FEMResult` and the mesh
    the 3-D field view is drawn from.

    Returns
    -------
    result : dict
        Everything :func:`fem_tube` returns, plus ``axial_n`` and ``root_moment_nm`` as applied.
    """
    leg = float(loads["leg_length_m"])
    axial = float(loads["lower_axial_n"]) / 2.0                       # compression positive
    moment = float(loads["lower_moment_nm"]) / 2.0
    transverse = moment / leg
    out = fem_tube(spec, section, leg, (-axial, 0.0, -transverse), n_theta=n_theta, n_radial=n_radial, n_len=n_len)
    return out | {"axial_n": axial, "root_moment_nm": moment, "transverse_n": transverse, "length_m": leg}
