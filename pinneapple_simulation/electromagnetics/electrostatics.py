"""2-D electrostatics on P1 triangles: ``-div(eps grad phi) = rho`` with electrodes at fixed potentials.

It is the operator of :func:`.fem.solve_magnetostatic` with ``nu -> eps`` and ``J -> rho`` (the two problems are
the same Poisson equation with a piecewise-constant coefficient), so it inherits that solver's assembly and
constraints. Per unit length along z: the energy is ``W = (1/2) int eps |grad phi|^2 dA`` and the capacitance
between an electrode at ``V`` and grounded ones is ``C = 2 W / V^2`` (F/m), which converges at twice the order of
the potential's H1 error (second order on P1 for smooth fields).

Planar (``x, y``) geometry only; axisymmetric electrostatics would need the ``r`` weight.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .constants import EPS0
from .fem import Geometry, solve_magnetostatic
from .mesh import Mesh2D

__all__ = ["ElectrostaticSolution", "solve_electrostatic", "capacitance"]


@dataclass
class ElectrostaticSolution:
    phi: np.ndarray        # nodal potential (V)
    E: np.ndarray          # element field (V/m), E = -grad phi
    eps: np.ndarray        # element permittivity (F/m)
    energy: float          # (1/2) int eps |E|^2 dA, J/m
    geometry: Geometry

    @property
    def D(self) -> np.ndarray:
        return self.eps[:, None] * self.E


def solve_electrostatic(mesh: Mesh2D, eps_r, electrodes: Mapping[str, tuple[np.ndarray, float]] | None = None,
                        rho=None, dirichlet=None, dirichlet_values=None) -> ElectrostaticSolution:
    """Potential for element relative permittivities ``eps_r`` (scalar or (M,)), charge density ``rho``
    (C/m^3 per element) and electrodes ``{name: (node indices, potential)}`` (or raw ``dirichlet`` nodes and
    values). Nodes in no electrode carry the natural condition ``D.n = 0`` (a symmetry plane or an open side)."""
    eps = EPS0 * np.broadcast_to(np.asarray(eps_r, float), (mesh.n_elements,)).copy()
    if electrodes is not None:
        nodes = np.concatenate([np.asarray(n, np.int64) for n, _ in electrodes.values()])
        vals = np.concatenate([np.full(len(n), float(v)) for n, v in electrodes.values()])
        nodes, idx = np.unique(nodes, return_index=True)
        dirichlet, dirichlet_values = nodes, vals[idx]
    sol = solve_magnetostatic(mesh, eps, J=None if rho is None else np.asarray(rho, float), dirichlet=dirichlet,
                              dirichlet_values=dirichlet_values)
    g = sol.geometry
    grad_phi = np.column_stack([(sol.A[mesh.tris] * g.gx).sum(1), (sol.A[mesh.tris] * g.gy).sum(1)])
    E = -grad_phi
    energy = 0.5 * float(np.sum(g.area * eps * (E * E).sum(1)))
    return ElectrostaticSolution(sol.A, E, eps, energy, g)


def capacitance(sol: ElectrostaticSolution, V: float) -> float:
    """``C = 2 W / V^2`` (F/m) for one electrode at ``V`` and the others grounded, no free charge."""
    return 2.0 * sol.energy / (V * V)
