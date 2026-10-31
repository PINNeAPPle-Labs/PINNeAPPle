"""Axial-flux permanent-magnet machine in 3-D on edge elements, against quasi-3D on the same discretisation.

The machine is meshed on a structured grid in cylindrical coordinates ``(r, theta, z)``: radial and axial node
lines are chosen per region (air inside ``r_in``, the active annulus, air outside ``r_out``; rotor yoke, magnets,
gap, teeth, stator yoke, air above and below), ``theta`` is uniform over the full circle, each grid cell is split
into six tetrahedra along its diagonal (as :func:`~..fem3d.mesh.box_mesh`) and the grid is closed at ``theta =
2 pi`` by merging the seam nodes (the split is the same in every cell, so the seam is conforming). Turning the rotor
by a whole number of steps ``2 pi / n_theta`` maps the grid onto itself: rotor angle is a relabelling of the magnet
cells, so cogging carries no remeshing noise.

Topologies: ``"single"`` (one rotor, one slotted stator with a yoke) and ``"double_rotor"`` (a yokeless slotted
stator between two rotors whose magnets are magnetised the same way, so the flux crosses the stator axially).

Quasi-3D (:meth:`AFPM3D.quasi3d`): the active annulus cut into ``n`` radial slices, each solved alone on its part of
the same grid with flux-parallel cylindrical faces (``n x A = 0``): the flux cannot cross radius ``r_k``, which is
the assumption of the usual quasi-3D model (a 2-D linear machine per radius, curvature kept). Summing the slices
and comparing with the full 3-D model (with air inside and outside the annulus) measures what quasi-3D neglects:
radial flux and the fringing at the inner and outer radius.

Linear iron and magnets, no winding currents: cogging torque (Maxwell stress over the gap layers) and the no-load
flux linkage of a search coil around a stator tooth (exact line integral of ``A``, :func:`.endeffects.line_integral`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..fem3d.mesh import TetMesh
from ..fem3d.nedelec import NU0, EdgeSpace, tree_gauge
from .endeffects import line_integral
from .spm import _factor

__all__ = ["AFPMMachine", "AFPM3D"]

TWO_PI = 2 * math.pi
AIR, IRON, MAGNET = 0, 1, 2


@dataclass
class AFPMMachine:
    """Geometry and materials (SI). Magnets and teeth are annular sectors over ``[r_in, r_out]``; ``magnet_arc``
    and ``slot_fraction`` are shares of the pole pitch and of the slot pitch."""
    poles: int = 8
    slots: int = 12
    r_in: float = 0.05
    r_out: float = 0.09
    gap: float = 0.001
    magnet_thickness: float = 0.004
    magnet_arc: float = 0.8
    rotor_yoke: float = 0.008
    tooth_height: float = 0.02
    stator_yoke: float = 0.008
    slot_fraction: float = 0.5
    Br: float = 1.2
    mu_r_magnet: float = 1.05
    mu_r_iron: float = 1000.0
    topology: str = "single"

    def layers(self) -> list[tuple[str, float]]:
        """Axial layers from the bottom: (kind, thickness)."""
        rotor = [("rotor_yoke", self.rotor_yoke), ("magnet", self.magnet_thickness), ("gap", self.gap)]
        if self.topology == "single":
            return rotor + [("teeth", self.tooth_height), ("stator_yoke", self.stator_yoke)]
        if self.topology == "double_rotor":
            return rotor + [("teeth", self.tooth_height)] + rotor[::-1]
        raise ValueError("topology must be 'single' or 'double_rotor'")

    @property
    def cogging_period(self) -> float:
        return TWO_PI / math.lcm(self.slots, self.poles)


def _graded(lo: float, hi: float, h: float) -> np.ndarray:
    return np.linspace(lo, hi, max(1, math.ceil((hi - lo) / h - 1e-9)) + 1)


def _air(face: float, outward: int, margin: float, h0: float, growth: float = 1.5) -> np.ndarray:
    steps, d, x = [], h0, 0.0
    while x < margin - 1e-12:
        x = min(margin, x + d)
        steps.append(x)
        d *= growth
    return face + outward * np.asarray(steps)


@dataclass
class AFPM3D:
    machine: AFPMMachine = field(default_factory=AFPMMachine)
    n_theta: int = 240
    dr: float = 0.004
    dz: float = 0.002
    air_r: float = 0.015          # air inside r_in and outside r_out (0: flux-parallel cylinders there)
    air_z: float = 0.015          # air above and below the machine
    r_range: tuple[float, float] | None = None    # restrict to a radial band of the active annulus (quasi-3D)
    r_nodes: np.ndarray | None = None              # explicit radial nodes of that band (the parent's)

    def __post_init__(self):
        m = self.machine
        if self.n_theta % (2 * m.poles) or self.n_theta % (2 * m.slots):
            raise ValueError("n_theta must be a multiple of twice the poles and twice the slots (pole and slot "
                             "centres on even theta nodes, the symmetry planes of the mesh)")
        self.step = TWO_PI / self.n_theta
        r0, r1 = self.r_range or (m.r_in, m.r_out)
        rs = np.asarray(self.r_nodes, float) if self.r_nodes is not None else _graded(r0, r1, self.dr)
        if self.r_range is None and self.air_r > 0:
            rs = np.concatenate([_air(r0, -1, min(self.air_r, r0 * 0.8), self.dr)[::-1], rs,
                                 _air(r1, +1, self.air_r, self.dr)])
        zs, kinds, z = [0.0], [], 0.0
        for kind, t in m.layers():
            pts = _graded(z, z + t, min(self.dz, t / 2) if kind in ("gap", "magnet") else self.dz)
            zs.extend(pts[1:])
            kinds.extend([kind] * (len(pts) - 1))
            z += t
        zs = np.asarray(zs)
        if self.air_z > 0:
            lo, hi = _air(0.0, -1, self.air_z, self.dz)[::-1], _air(zs[-1], +1, self.air_z, self.dz)
            kinds = ["air"] * len(lo) + kinds + ["air"] * len(hi)
            zs = np.concatenate([lo, zs, hi])
        self.rs, self.zs, self.zkind = rs, zs, np.asarray(kinds)
        self._build()
        self.space = EdgeSpace(self.mesh)
        de = self.mesh.boundary_edges
        self._dir = de
        self._sys = None
        self._static, self._magnet_base = self._labels()

    # ------------------------------------------------------------------------------------------- mesh
    def _build(self):
        rs, zs, nt = self.rs, self.zs, self.n_theta
        nr, nz = len(rs), len(zs)
        th = np.arange(nt) * self.step
        R, T, Z = np.meshgrid(rs, th, zs, indexing="ij")
        nodes = np.column_stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), Z.ravel()])

        def nid(i, j, k):
            return (i * nt + j % nt) * nz + k                   # j = nt is the seam: node j = 0

        i, j, k = np.meshgrid(np.arange(nr - 1), np.arange(nt), np.arange(nz - 1), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        # every other theta cell is the mirror image of its neighbour (corners taken with theta reversed): the
        # diagonals of a plain Kuhn split all lean the same way round the axis, a helical bias that puts torque
        # on an aligned rotor (0.9 N m measured on a coarse mesh). The mirrored split conforms (shared faces
        # carry the same diagonal) and is symmetric about every even theta node plane.
        odd = j % 2
        v = [nid(i + (c & 1), j + np.where(odd, 1 - (c >> 1 & 1), c >> 1 & 1), k + (c >> 2 & 1)) for c in range(8)]
        paths = [(1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
        tets = np.concatenate([np.column_stack([v[0], v[a], v[b], v[7]]) for a, b in paths])
        self.cell = (np.tile(i, 6), np.tile(j, 6), np.tile(k, 6))
        self.mesh = TetMesh(nodes, tets, np.zeros(len(tets), np.int64), {"air": AIR, "iron": IRON,
                                                                            "magnet": MAGNET})

    def _labels(self):
        """Static iron (rotor yokes, stator) per tetrahedron, and the magnet cells at rotor angle 0: theta index
        -> polarity (+1, -1, 0) per magnet layer, and which tetrahedra are in a magnet layer."""
        m = self.machine
        i, j, k = self.cell
        rc = 0.5 * (self.rs[i] + self.rs[i + 1])
        active = (rc > m.r_in) & (rc < m.r_out)
        kind = self.zkind[k]
        tc = (j + 0.5) * self.step
        slot_pitch = TWO_PI / m.slots
        u = np.mod(tc, slot_pitch) / slot_pitch                 # slot centred at u = 0
        in_slot = (u < m.slot_fraction / 2) | (u > 1 - m.slot_fraction / 2)
        iron = active & ((kind == "rotor_yoke") | (kind == "stator_yoke") | ((kind == "teeth") & ~in_slot))
        static = np.where(iron, IRON, AIR)
        pole = TWO_PI / m.poles
        jt = np.arange(self.n_theta)
        w = np.mod((jt + 0.5) * self.step + pole / 2, pole) / pole - 0.5      # pole p centred at p * pole
        pol = np.where(np.abs(w) < m.magnet_arc / 2,
                       np.where(np.floor(((jt + 0.5) * self.step + pole / 2) / pole) % 2 == 0, 1.0, -1.0), 0.0)
        in_mag = active & (kind == "magnet")
        return static, (pol, in_mag)

    def materials(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Reluctivity and remanence per tetrahedron with the rotor(s) turned by ``theta``, rounded to an even
        number of steps (the mesh maps onto itself under rotations by two steps)."""
        m = self.machine
        s = 2 * int(round(theta / (2 * self.step)))
        pol, in_mag = self._magnet_base
        j = self.cell[1]
        p = np.where(in_mag, pol[np.mod(j - s, self.n_theta)], 0.0)
        nu = np.where(self._static == IRON, NU0 / m.mu_r_iron, NU0)
        nu = np.where(p != 0, NU0 / m.mu_r_magnet, nu)
        Br = np.zeros((len(nu), 3))
        Br[:, 2] = p * m.Br
        self.mesh.region = np.where(p != 0, MAGNET, self._static)
        return nu, Br

    # ------------------------------------------------------------------------------------------- solve
    def _system(self):
        if self._sys is None:
            de = self._dir
            tree = tree_gauge(self.mesh, de, np.unique(self.mesh.edges[de]))
            u = np.setdiff1d(np.setdiff1d(np.arange(self.space.n), de), tree)
            nu, _ = self.materials(0.0)
            nu = np.where(self._magnet_base[1], NU0 / self.machine.mu_r_magnet, nu)   # whole magnet layer
            self._sys = (u, _factor(self.space.stiffness(nu)[u][:, u].tocsr()))
        return self._sys

    def solve(self, theta: float = 0.0, a0=None, rtol: float = 1e-10):
        """No-load field at rotor angle ``theta``: preconditioned CG, the preconditioner factorised once."""
        import scipy.sparse.linalg as spla

        nu, Br = self.materials(theta)
        u, solve0 = self._system()
        K = self.space.stiffness(nu)[u][:, u].tocsr()
        b = self.space.magnet_load(nu, Br)[u]
        P = spla.LinearOperator(K.shape, matvec=solve0, dtype=float)
        x, info = spla.cg(K, b, x0=None if a0 is None else a0[u], rtol=rtol, M=P, maxiter=200)
        if info:
            raise RuntimeError(f"preconditioned CG did not converge (info={info})")
        a = np.zeros(self.space.n)
        a[u] = x
        return _Solution(a, self.space.curl(a), float(0.5 * b @ x))

    def release(self) -> None:
        if self._sys is not None and hasattr(self._sys[1], "free"):
            self._sys[1].free()
        self._sys = None

    def torque(self, sol) -> float:
        """Torque on the rotor(s), N m: Maxwell stress ``r B_theta B_z / mu0`` averaged over each gap layer
        (volume mean times the layer area), signed by the side of the rotor."""
        m = self.machine
        c = self.mesh.centroids
        r = np.hypot(c[:, 0], c[:, 1])
        et = np.column_stack([-c[:, 1] / r, c[:, 0] / r])
        Bt = np.einsum("kd,kd->k", sol.B[:, :2], et)
        k = self.cell[2]
        kinds = self.zkind[k]
        T, z = 0.0, 0.0
        layers = m.layers()
        for idx, (kind, t) in enumerate(layers):
            if kind == "gap":
                below = idx > 0 and layers[idx - 1][0] == "magnet"   # the rotor is below this gap
                sign = 1.0 if below else -1.0                         # its outward normal: +z, or -z above
                zc = self.mesh.centroids[:, 2]
                s = (kinds == "gap") & (zc > z) & (zc < z + t)
                T += sign * float(np.sum(NU0 * r[s] * Bt[s] * sol.B[s, 2] * self.space.vol[s]) / t)
            z += t
        return T

    def cogging(self, thetas: Sequence[float]) -> np.ndarray:
        out, a = [], None
        for th in thetas:
            s = self.solve(th, a0=a)
            a = s.a
            out.append(self.torque(s))
        return np.asarray(out)

    def tooth_linkage(self, sol, overhang: float = 0.0, tooth: int = 0) -> float:
        """Flux linkage (Wb) of a coil around stator tooth ``tooth`` at mid tooth height: radial sides on the
        slot centre lines, arcs ``overhang`` inside ``r_in`` and outside ``r_out`` (0: on those radii), drawn as
        the polygons of the grid (so on a flux-parallel boundary they add exactly nothing)."""
        m = self.machine
        z = self._layer_z("teeth") + m.tooth_height / 2
        sp = TWO_PI / m.slots
        a, b = (tooth + 0.5) * sp - sp / 2, (tooth + 0.5) * sp + sp / 2     # tooth centred between slots
        r0, r1 = (self.r_range or (m.r_in, m.r_out))
        r0, r1 = r0 - overhang, r1 + overhang
        r0 = max(r0, self.rs[0])
        r1 = min(r1, self.rs[-1])
        ja, jb = int(round(a / self.step)), int(round(b / self.step))
        ps = np.arange(ja, jb + 1) * self.step                 # the arcs follow the grid's polygons

        def pt(r, t):
            return np.array([r * math.cos(t), r * math.sin(t), z])
        side_a = np.array([pt(r1, ps[0]), pt(r0, ps[0])])
        arc_in = np.array([pt(r0, t) for t in ps])
        side_b = np.array([pt(r0, ps[-1]), pt(r1, ps[-1])])
        arc_out = np.array([pt(r1, t) for t in ps[::-1]])
        return sum(line_integral(self.space, sol.a, P, samples=s)
                   for P, s in ((side_a, 2000), (arc_in, 16), (side_b, 2000), (arc_out, 16)))

    def _layer_z(self, kind: str) -> float:
        z = 0.0
        for k, t in self.machine.layers():
            if k == kind:
                return z
            z += t
        raise KeyError(kind)

    # ------------------------------------------------------------------------------------------- quasi-3D
    def quasi3d_models(self, n_slices: int) -> list[AFPM3D]:
        """``n_slices`` radial slices of the active annulus on the same radial nodes, no radial air."""
        m = self.machine
        ra = self.rs[(self.rs >= m.r_in - 1e-12) & (self.rs <= m.r_out + 1e-12)]
        cuts = np.round(np.linspace(0, len(ra) - 1, n_slices + 1)).astype(int)
        return [AFPM3D(m, n_theta=self.n_theta, dr=self.dr, dz=self.dz, air_r=0.0, air_z=self.air_z,
                       r_range=(float(ra[cuts[q]]), float(ra[cuts[q + 1]])),
                       r_nodes=ra[cuts[q]:cuts[q + 1] + 1]) for q in range(n_slices)]


@dataclass
class _Solution:
    a: np.ndarray
    B: np.ndarray
    coenergy: float
