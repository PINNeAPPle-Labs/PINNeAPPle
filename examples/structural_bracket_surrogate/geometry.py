"""Generate a conforming, voxel-based L-bracket tetrahedral demonstration mesh.

Geometry is a simplified *solid* L profile, not a reproduction of the
source CAD. Coordinates are metres. Shared grid vertices are deduplicated.
"""
from __future__ import annotations
import numpy as np


def make_l_bracket(nx: int = 12, ny: int = 2, nz: int = 12,
                   length_m: float = 0.12, height_m: float = 0.12,
                   thickness_m: float = 0.02, depth_m: float = 0.02):
    if min(nx, ny, nz) < 2 or min(length_m, height_m, thickness_m, depth_m) <= 0:
        raise ValueError("Positive dimensions and at least two cells per axis required")
    xs = np.linspace(0, length_m, nx + 1)
    ys = np.linspace(0, depth_m, ny + 1)
    zs = np.linspace(0, height_m, nz + 1)
    vertices = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    def idx(i, j, k):
        return (k * (ny + 1) + j) * (nx + 1) + i
    local_tets = [(0,1,3,7),(0,3,2,7),(0,2,6,7),
                  (0,6,4,7),(0,4,5,7),(0,5,1,7)]
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                xc = (xs[i] + xs[i+1])/2
                zc = (zs[k] + zs[k+1])/2
                if not (xc <= thickness_m or zc <= thickness_m):
                    continue
                cube = [idx(i+di,j+dj,k+dk) for dk,dj,di in
                        [(0,0,0),(0,0,1),(0,1,0),(0,1,1),
                         (1,0,0),(1,0,1),(1,1,0),(1,1,1)]]
                elements.extend([[cube[a] for a in tet] for tet in local_tets])
    if not elements:
        raise ValueError("No solid cells; increase resolution or thickness")
    raw = np.asarray(elements, dtype=int)
    used, inverse = np.unique(raw, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 4)


def demo_case():
    nodes, tets = make_l_bracket()
    # Clamp free end of horizontal leg; apply distributed nodal force on
    # top of vertical leg. This is an illustrative boundary condition.
    fixed = np.flatnonzero(np.isclose(nodes[:, 0], 0.12) &
                           (nodes[:, 2] <= 0.02 + 1e-12))
    loaded = np.flatnonzero(np.isclose(nodes[:, 2], 0.12))
    if not len(fixed) or not len(loaded):
        raise ValueError("Boundary selection empty")
    force = np.zeros_like(nodes)
    force[loaded, 0] = 100.0 / len(loaded)
    return nodes, tets, fixed, force


if __name__ == "__main__":
    from fem_reference import solve_elasticity
    nodes, tets, fixed, forces = demo_case()
    result = solve_elasticity(nodes, tets, fixed, forces, 70e9, 0.33)
    print("Nodes:", len(nodes), "tetrahedra:", len(tets))
    print("Maximum von Mises (MPa):", result["von_mises_pa"].max()/1e6)
    print("Force-balance residual (N):", result["force_balance_n"])
