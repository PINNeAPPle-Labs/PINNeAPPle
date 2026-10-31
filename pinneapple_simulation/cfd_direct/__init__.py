"""CFD directly on polygon / STL geometry against body-fitted CFD: the cases, the runners and the campaign of
docs/research/cfd_direct_from_stl.md (lattice Boltzmann on the voxelised body, the immersed cut-cell flow solver,
and the Taylor-Hood reference on a gmsh mesh)."""
from .cases import (
    SCHAFER_TUREK,
    SchaferTurek,
    Sphere,
    clift_grace_weber,
    cylinder_polygon,
    schiller_naumann,
)

__all__ = ["SCHAFER_TUREK", "SchaferTurek", "Sphere", "clift_grace_weber", "cylinder_polygon", "schiller_naumann"]
