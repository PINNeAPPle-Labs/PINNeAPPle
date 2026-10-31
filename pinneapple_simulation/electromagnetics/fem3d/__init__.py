"""3-D finite elements for electromagnetics: first-order Nedelec (Whitney edge) elements on tetrahedra.

* :mod:`.mesh` -- :class:`TetMesh` (edges, faces, boundary, point location), structured box meshes, gmsh input.
* :mod:`.nedelec` -- :class:`EdgeSpace` (element matrices, sources, evaluation), nonlinear magnetostatics by Newton
  (:func:`solve_magnetostatic`), time-harmonic eddy currents with the modified vector potential
  (:func:`solve_eddy`), tree gauge, discretely solenoidal sources, energy and co-energy.
* :mod:`.sphere` -- permeable sphere in a uniform field (closed form, gmsh mesh, errors).
* :mod:`.cases` -- closed-form cases on box meshes: manufactured magnetostatics, conducting slab (skin effect).
* :mod:`.getdp3d` -- GetDP as the external reference on the same meshes.
"""
from .mesh import TetMesh, box_mesh, from_gmsh, read_msh22
from .nedelec import MU0, NU0, EdgeSpace, solve_eddy, solve_magnetostatic

__all__ = ["TetMesh", "box_mesh", "from_gmsh", "read_msh22", "MU0", "NU0", "EdgeSpace", "solve_eddy",
           "solve_magnetostatic"]
