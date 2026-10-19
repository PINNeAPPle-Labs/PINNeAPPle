"""Meshless and mesh-based solvers of the same 2-D problems, and the study that compares them.

* :mod:`.domains`   -- 2-D domains (square, disk, L-shape, annulus, plate with a hole, any polygon, an uploaded point
  cloud) with their signed distance and quasi-uniform node sets at a spacing ``h``.
* :mod:`.problems`  -- Poisson, Helmholtz and heat problems with Dirichlet conditions, from a manufactured solution
  (sympy expression) or from a source and a boundary value.
* :mod:`.methods`   -- Kansa (global RBF collocation), RBF-FD (polyharmonic splines + polynomials), finite differences
  on an embedded Cartesian grid (Shortley-Weller at the boundary), linear finite elements on the Delaunay mesh of
  the nodes, and a PINN on the same nodes.
* :mod:`.study`     -- refinement study: error, cost, condition number, noise amplification, observed order, and
  where each method wins.

    from pinneapple_simulation.meshless import Problem, domain, meshless_study
    res = meshless_study(Problem("poisson", exact="sin(pi*x)*sin(pi*y)"), domain("lshape"), h=[0.1, 0.05, 0.025])
"""
from .domains import DOMAINS, Domain2D, NodeSet, domain, point_cloud_domain, polygon_domain
from .methods import METHODS, Discretization, discretize, solve
from .problems import Problem
from .study import MeshlessStudy, meshless_study

__all__ = ["DOMAINS", "Domain2D", "NodeSet", "domain", "polygon_domain", "point_cloud_domain", "Problem",
           "METHODS", "Discretization", "discretize", "solve", "MeshlessStudy", "meshless_study"]
