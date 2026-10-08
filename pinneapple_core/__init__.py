"""pinneapple_core: Field, Mesh, Domain and Geometry as shared primitives.

Physics AI needs tensors, geometry and topology in one object. These four are the
common vocabulary the solvers, models and datasets can build on:

- :class:`Domain`     the region (box or CSG shape): contains, sample interior/boundary
- :class:`Geometry`   a Domain with named boundaries and a mesher
- :class:`Mesh`       simplex mesh (1D/2D/3D) with exact P1 gradient/integral/interpolation
- :class:`Field`      values on a grid, mesh or point cloud
- :class:`FunctionField`  the same interface for a torch callable (PINN), by autograd

Operators (:mod:`pinneapple_core.operators`) give one function per derivative on every
representation: ``grad, div, curl, laplacian, jacobian, hessian, integrate, flux``.

Every field answers ``gradient()``, ``divergence()``, ``interpolate(x)`` and
``integrate()``, whatever it is discretised on.
"""
from .domain import Domain
from .field import Field, FunctionField, Grid, PointCloud
from .geometry import Geometry
from .mesh import Mesh
from .operators import curl, div, flux, grad, hessian, integrate, jacobian, laplacian

__all__ = ["Domain", "Geometry", "Mesh", "Field", "FunctionField", "Grid", "PointCloud",
           "grad", "div", "curl", "laplacian", "jacobian", "hessian", "integrate", "flux"]


def __getattr__(name):
    # torch-based submodules load on first use: ``pinneapple_core.func`` / ``.fem``
    if name in ("func", "fem", "backend", "module", "loss", "optim", "data", "transforms"):
        import importlib

        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name in _MODULE_EXPORTS:
        from . import module

        return getattr(module, name)
    if name in _OPTIM_EXPORTS:
        from . import optim

        return getattr(optim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_MODULE_EXPORTS = ("PhysicsModule", "Sequential", "SolverModule", "Hybrid")
_OPTIM_EXPORTS = ("PhysicsOptimizer",)
