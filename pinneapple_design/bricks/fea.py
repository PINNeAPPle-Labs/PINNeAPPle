"""Linear-elastic finite elements on the voxelised build (Hex8 with incompatible modes, :mod:`solid_fem`).

The build becomes one solid: every occupied cell (one stud pitch by one plate height) is split into
``per_stud`` x ``per_stud`` x 1 hexahedra and neighbouring cells share nodes, whether or not a stud joint links
them. That is an upper bound on stiffness: joint compliance and the 0.2 mm gaps between side-by-side parts are not
modelled, so use it for load paths and stresses in the plastic, and :mod:`.statics` for whether joints hold.
The same mesh can be solved by CalculiX through ``SolidFEM.calculix`` when ``ccx`` is installed.

Units: millimetres, newtons, MPa. ``E_ABS_MPA`` is an assumed modulus for moulded ABS; set your own.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .assembly import Assembly
from .parts import PITCH_MM, PLATE_MM, G

E_ABS_MPA = 2300.0          # assumed, see the module docstring
NU_ABS = 0.35               # assumed


@dataclass
class VoxelModel:
    mesh: object                 # HexMesh
    element_brick: np.ndarray    # (m,) brick index of each element
    element_volume: float        # [mm^3]


def voxel_mesh(asm: Assembly, per_stud: int = 2) -> VoxelModel:
    """Hexahedral mesh of the occupied cells, ``per_stud`` elements across a stud pitch, one per plate height."""
    from pinneapple_simulation.numerical_solvers.solid_fem import HexMesh
    occ = {}
    for i, b in enumerate(asm.bricks):
        for c in b.cells():
            occ.setdefault(c, i)
    hx, hy, hz = PITCH_MM / per_stud, PITCH_MM / per_stud, PLATE_MM
    node_id: dict[tuple[int, int, int], int] = {}
    nodes, elems, owner = [], [], []

    def nid(ix, iy, iz):
        key = (ix, iy, iz)
        if key not in node_id:
            node_id[key] = len(nodes)
            nodes.append((ix * hx, iy * hy, iz * hz))
        return node_id[key]

    for (i, j, k), b in sorted(occ.items()):
        for a in range(per_stud):
            for c in range(per_stud):
                p, q = i * per_stud + a, j * per_stud + c
                corners = [(p, q, k), (p + 1, q, k), (p + 1, q + 1, k), (p, q + 1, k),
                           (p, q, k + 1), (p + 1, q, k + 1), (p + 1, q + 1, k + 1), (p, q + 1, k + 1)]
                elems.append([nid(*q) for q in corners])
                owner.append(b)
    nodes_a = np.array(nodes, float)
    lo, hi = nodes_a.min(0), nodes_a.max(0)
    mesh = HexMesh(nodes_a, np.array(elems, np.int64), (0, 0, 0), tuple((hi - lo).tolist()))
    return VoxelModel(mesh, np.array(owner), hx * hy * hz)


@dataclass
class FEAResult:
    fem: object                  # FEMResult (u [mm], stress [MPa], von_mises)
    model: VoxelModel
    brick_displacement: np.ndarray   # (n_bricks,) max |u| over the brick's nodes [mm]
    brick_von_mises: np.ndarray      # (n_bricks,) max nodal von Mises [MPa]
    max_displacement: float


def solve(asm: Assembly, loads: Sequence = (), *, E: float = E_ABS_MPA, nu: float = NU_ABS, per_stud: int = 2,
          gravity: bool = True, supports: str = "ground", fixed_nodes=None) -> FEAResult:
    """Static linear elasticity of the voxelised build under gravity and point ``loads`` (:class:`~.statics.Load`).

    ``supports``: ``"ground"`` clamps the bottom nodes over the ground regions; ``"simple"`` gives vertical support
    on those nodes with one corner pinned and another on a roller (a simply supported beam); or pass
    ``fixed_nodes`` = {node: dofs}."""
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM
    vm = voxel_mesh(asm, per_stud)
    mesh = vm.mesh
    fem = SolidFEM(mesh, E=E, nu=nu, incompatible=True)
    bottom = mesh.nodes_on(z=0.0)
    if asm.ground.regions is not None:
        xy = mesh.nodes[bottom, :2] / PITCH_MM
        keep = np.zeros(len(bottom), bool)
        for x0, x1, y0, y1 in asm.ground.regions:
            keep |= (xy[:, 0] >= x0 - 1e-9) & (xy[:, 0] <= x1 + 1e-9) & (xy[:, 1] >= y0 - 1e-9) & (xy[:, 1] <= y1 + 1e-9)
        bottom = bottom[keep]
    if fixed_nodes is not None:
        for n, dofs in fixed_nodes.items():
            fem.fix([n], dofs)
    elif supports == "ground":
        fem.fix(bottom)
    elif supports == "simple":
        fem.fix(bottom, (2,))
        P = mesh.nodes[bottom]
        a = bottom[np.lexsort((P[:, 1], P[:, 0]))[0]]              # min x, min y: pinned
        b = bottom[np.lexsort((P[:, 1], -P[:, 0]))[0]]             # max x, min y: roller along x
        fem.fix([a], (0, 1, 2))
        fem.fix([b], (1, 2))
    else:
        raise ValueError("supports must be 'ground' or 'simple'")
    if gravity:
        rho_g = np.array([asm.bricks[b].part.mass * G / sum(1 for e in vm.element_brick if e == b)
                          for b in range(len(asm.bricks))])        # weight per element of each brick [N]
        nodal = np.zeros(len(mesh.nodes))
        for e, b in enumerate(vm.element_brick):
            nodal[mesh.elements[e]] += rho_g[b] / 8.0
        for n in np.nonzero(nodal)[0]:
            fem.load_nodes([n], (0.0, 0.0, -nodal[n]))
    for ld in loads:
        d = np.linalg.norm(mesh.nodes - np.asarray(ld.point, float), axis=1)
        near = np.nonzero(d <= d.min() + 1e-6)[0]
        fem.load_nodes(near, tuple(np.asarray(ld.force, float)))
    res = fem.solve()
    un = np.linalg.norm(res.u, axis=1)
    vmn = res.von_mises
    nb = len(asm.bricks)
    bd, bv = np.zeros(nb), np.zeros(nb)
    for e, b in enumerate(vm.element_brick):
        ns = mesh.elements[e]
        bd[b] = max(bd[b], un[ns].max())
        bv[b] = max(bv[b], vmn[ns].max())
    return FEAResult(res, vm, bd, bv, float(un.max()))
