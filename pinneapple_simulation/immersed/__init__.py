"""Immersed (finite cell / CutFEM) analysis on raw geometry: no conforming mesh.

A part given as a signed distance (any SDF, CSG of SDFs), a voxel mask or an STL surface (generalised winding number)
is put in a uniform background grid of Q1 or Q2 elements; the cells it cuts are integrated by adaptive octree
quadrature compressed by moment fitting, small cuts are stabilised (ghost penalty or the finite cell method's
fictitious material, with the condition number reported), and boundary conditions on the curved surface are imposed
weakly: Nitsche's method for temperatures, surface quadrature for heat fluxes and convection.

    from pinneapple_simulation import immersed as im
    part = im.cylinder((0, 0, 0), 0.02, 0.1) - im.cylinder((0, 0, 0), 0.008, 0.12)      # or im.STLGeometry.from_file
    disc = im.Discretization(part, h=0.004, order=2)
    sol = im.solve_heat(disc, conductivity=200.0, source=0.0,
                        bcs=[im.Dirichlet(80.0, where=lambda X, n: np.hypot(X[:, 0], X[:, 1]) < 0.01),
                             im.Convection(h=25.0, T_inf=20.0)])
    sol(points), sol.balance, sol.info

Linear elasticity on the same discretisation: :func:`solve_elasticity` (Nitsche displacements, tractions, pressures
and total forces). Verification (manufactured solutions on curved domains, sphere, cylinder and Lame sphere in closed
form, conditioning of small cuts): :mod:`pinneapple_simulation.immersed.verification`; STL parts against conforming
FEM: :mod:`pinneapple_simulation.immersed.parts`.
"""
from .assembly import Discretization, condition_number, solve_linear
from .bcs import Convection, Dirichlet, Displacement, Force, HeatFlux, Pressure, Traction
from .elasticity import ElasticSolution, assemble_elasticity, solve_elasticity
from .geometry import (
                       Geometry,
                       SDFGeometry,
                       STLGeometry,
                       VoxelGeometry,
                       annulus,
                       ball,
                       box,
                       cylinder,
                       difference,
                       intersection,
                       read_stl,
                       torus,
                       union,
                       write_stl,
)
from .grid import BackgroundGrid, DofMap
from .heat import HeatSolution, assemble_heat, default_beta, default_ghost, solve_heat
from .quadrature import CutQuadrature, build_quadrature

__all__ = ["Discretization", "condition_number", "solve_linear", "Convection", "Dirichlet", "Displacement", "Force",
           "HeatFlux", "Pressure", "Traction", "ElasticSolution", "assemble_elasticity", "solve_elasticity",
           "Geometry", "SDFGeometry", "STLGeometry", "VoxelGeometry", "annulus", "ball", "box", "cylinder",
           "difference", "intersection", "read_stl", "torus", "union", "write_stl", "BackgroundGrid", "DofMap",
           "HeatSolution", "assemble_heat", "default_beta", "default_ghost", "solve_heat", "CutQuadrature",
           "build_quadrature"]
