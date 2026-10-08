"""Physics operators: ``grad, div, curl, laplacian, jacobian, hessian, integrate, flux``.

The ``torch.nn.functional`` of physics: one function per operator that works the same
on a continuous field (a torch callable, derivatives by autograd), on a grid, on a
simplex mesh and on a point cloud (a :class:`~pinneapple_core.Field`).

* On a :class:`Field` the operator takes no point argument and returns a ``Field``
  on the same support (``integrate`` and ``flux`` return numbers).
* On a callable or :class:`FunctionField` it takes the evaluation points ``x`` and
  returns a tensor that keeps the autograd graph.

Shape convention: ``(N,)`` scalar, ``(N, d)`` vector, ``jacobian`` is ``(N, c, d)``
with ``J[n, i, j] = d f_i / d x_j``, ``hessian`` is ``(N, d, d)``.

>>> import numpy as np
>>> from pinneapple_core import Field, laplacian
>>> x = np.linspace(0, 1, 41)
>>> u = Field.on_grid([x, x], lambda p: p[:, 0] ** 2 + 3 * p[:, 1] ** 2)
>>> round(float(laplacian(u).values[500]), 6)
8.0
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from .domain import Domain
from .field import Field, FunctionField, Grid, PointCloud
from .mesh import Mesh

Where = Union[None, str, Callable[[np.ndarray], np.ndarray]]

__all__ = ["grad", "div", "curl", "laplacian", "jacobian", "hessian", "integrate", "flux"]


# -- helpers -----------------------------------------------------------------
def _as_continuous(f):
    if isinstance(f, FunctionField):
        return f
    if callable(f):
        return FunctionField(f)
    return None


def _need_x(x, name: str):
    if x is None:
        raise ValueError(f"{name} of a continuous field needs the evaluation points x")
    return x


def _lib(a):
    """numpy-like module matching the array type (torch or numpy)."""
    if type(a).__module__.startswith("torch"):
        import torch

        return torch
    return np


def _check_field(f) -> Field:
    if not isinstance(f, Field):
        raise TypeError(f"expected a Field, a FunctionField or a callable, got {type(f).__name__}")
    return f


def _named(f: Field, op: str) -> Optional[str]:
    return None if f.name is None else f"{op}({f.name})"


# -- first-order operators ---------------------------------------------------
def grad(f, x=None):
    """Gradient: ``(N, d)`` for a scalar, ``(N, c, d)`` for ``c`` components."""
    cf = _as_continuous(f)
    return cf.gradient(_need_x(x, "grad")) if cf is not None else _check_field(f).gradient()


def div(f, x=None):
    """Divergence of a vector field with ``d`` components."""
    cf = _as_continuous(f)
    return cf.divergence(_need_x(x, "div")) if cf is not None else _check_field(f).divergence()


def jacobian(f, x=None):
    """Jacobian ``(N, c, d)``, ``J[n, i, j] = d f_i / d x_j``. A scalar has ``c = 1``."""
    cf = _as_continuous(f)
    if cf is not None:
        g = cf.gradient(_need_x(x, "jacobian"))
        return g[:, None, :] if g.ndim == 2 and cf(x).ndim == 1 else g
    fld = _check_field(f)
    g = fld.gradient()
    if fld.components == ():
        return Field(g.values[:, None, :], fld.support, _named(fld, "jac"))
    if len(fld.components) != 1:
        raise ValueError("jacobian needs a scalar or a vector field")
    return g


def curl(f, x=None):
    """Curl of a vector field: a vector in 3D (``(N, 3)``), a scalar in 2D (``(N,)``,
    the z component ``dv/dx - du/dy``)."""
    cf = _as_continuous(f)
    if cf is not None:
        j = cf.gradient(_need_x(x, "curl"))
        return _curl_from_jacobian(j)
    fld = _check_field(f)
    if fld.components != (fld.dim,):
        raise ValueError(f"curl needs a vector field with {fld.dim} components, got {fld.components}")
    return Field(_curl_from_jacobian(fld.gradient().values), fld.support, _named(fld, "curl"))


def _curl_from_jacobian(j):
    d = j.shape[-1]
    if j.ndim != 3 or j.shape[1] != d:
        raise ValueError("curl needs a vector field with as many components as dimensions")
    if d == 2:
        return j[:, 1, 0] - j[:, 0, 1]
    if d == 3:
        comps = [j[:, 2, 1] - j[:, 1, 2], j[:, 0, 2] - j[:, 2, 0], j[:, 1, 0] - j[:, 0, 1]]
        return np.stack(comps, axis=1) if _lib(j) is np else _lib(j).stack(comps, dim=1)
    raise ValueError("curl is defined for 2D and 3D fields")


# -- second-order operators --------------------------------------------------
def hessian(f, x=None):
    """Hessian ``(N, d, d)`` of a scalar field."""
    cf = _as_continuous(f)
    if cf is not None:
        x = _need_x(x, "hessian")
        if cf(x).ndim != 1:
            raise ValueError("hessian needs a scalar field")
        return cf.gradient_field().gradient(x)
    fld = _check_field(f)
    if fld.components != ():
        raise ValueError("hessian needs a scalar field")
    return Field(_second_derivatives(fld)[:, 0], fld.support, _named(fld, "hess"))


def laplacian(f, x=None):
    """Sum of second derivatives. A scalar gives a scalar, a vector field applies it
    to each component."""
    cf = _as_continuous(f)
    if cf is not None:
        x = _need_x(x, "laplacian")
        h = cf.gradient_field().gradient(x)  # (N, [c,] d, d)
        return _lib(h).diagonal(h, dim1=-2, dim2=-1).sum(-1) if _lib(h) is not np else np.trace(h, axis1=-2, axis2=-1)
    fld = _check_field(f)
    h = _second_derivatives(fld)  # (N, k, d, d)
    lap = np.trace(h, axis1=-2, axis2=-1).reshape(h.shape[0], *fld.components)
    return Field(lap, fld.support, _named(fld, "lap"))


def _second_derivatives(f: Field) -> np.ndarray:
    """``(N, k, d, d)`` second derivatives of the flattened components."""
    flat = f._flat()
    if hasattr(f.support, "hessian"):  # point clouds fit a local quadratic
        return f.support.hessian(flat)
    g = f.support.gradient(flat)  # (N, k, d)
    n, k, d = g.shape
    gg = f.support.gradient(g.reshape(n, k * d))  # (N, k*d, d)
    return gg.reshape(n, k, d, d)


# -- integrals ---------------------------------------------------------------
def integrate(f, **kwargs):
    """Integral over the field's support (trapezoid on a grid, exact P1 on a mesh,
    quadrature or Monte Carlo on a cloud). A continuous field takes ``n=`` samples
    (Monte Carlo over its Domain) or ``points=`` and ``weights=``."""
    cf = _as_continuous(f)
    if cf is not None:
        return cf.integrate(**kwargs)
    if kwargs:
        raise TypeError("integrate of a Field takes no extra arguments")
    return _check_field(f).integrate()


def flux(f, where: Where = None, *, domain: Optional[Domain] = None, n: int = 20_000, seed: int = 0):
    """Flux of a vector field out of the boundary, ``integral of f . n dS``.

    ``where`` restricts it to part of the boundary: a face name of a box (``"x_min"``,
    ``"x_max"``, ``"y_min"``, ...) or a predicate on boundary points. By the divergence
    theorem ``flux(f)`` equals ``integrate(div(f))``.

    * Grid: trapezoid over each face (face names only).
    * Mesh: exact for P1 fields over the boundary facets.
    * Point cloud, or a continuous field: Monte Carlo with ``n`` boundary samples
      of ``domain`` (a continuous field uses its own domain; pass ``domain=`` for a
      cloud). Statistical error ~ 1/sqrt(n).
    """
    cf = _as_continuous(f)
    if cf is not None:
        dom = domain or cf.domain
        if dom is None:
            raise ValueError("flux of a continuous field needs a Domain (FunctionField(fn, domain) or domain=)")
        return _mc_flux(lambda p: cf(p), dom, where, n, seed, torch_out=True)
    fld = _check_field(f)
    if fld.components != (fld.dim,):
        raise ValueError(f"flux needs a vector field with {fld.dim} components, got {fld.components}")
    s = fld.support
    if isinstance(s, Mesh):
        return _mesh_flux(s, fld.values, where)
    if isinstance(s, Grid):
        return _grid_flux(s, fld.values, where)
    if isinstance(s, PointCloud):
        dom = domain or s.domain
        if dom is None:
            raise ValueError("flux of a point cloud needs a Domain (Field.on_points(..., domain=) or domain=)")
        return _mc_flux(lambda p: fld.interpolate(p, fill="nearest"), dom, where, n, seed, torch_out=False)
    raise TypeError(f"flux is not implemented for {type(s).__name__}")


def _face_predicate(where: Where, lo: np.ndarray, hi: np.ndarray):
    if where is None:
        return None
    if callable(where):
        return lambda p: np.asarray(where(p), dtype=bool)
    if isinstance(where, str) and len(where) == 5 and where[0] in "xyz" and where[1:] in ("_min", "_max"):
        a = "xyz".index(where[0])
        if a >= lo.size:
            raise ValueError(f"face {where!r} does not exist in {lo.size}D")
        target = lo[a] if where.endswith("min") else hi[a]
        tol = 1e-9 * (hi[a] - lo[a])
        return lambda p: np.abs(p[:, a] - target) <= tol
    raise ValueError(f"where must be None, a face name like 'x_max', or a predicate; got {where!r}")


def _mesh_flux(mesh: Mesh, values: np.ndarray, where: Where) -> float:
    bg = mesh.boundary_geometry()
    mean_v = values[bg["facets"]].mean(axis=1)  # (K, d), exact for P1 over a facet
    contrib = (mean_v * bg["normals"]).sum(axis=1) * bg["measures"]
    pred = _face_predicate(where, mesh.points.min(axis=0), mesh.points.max(axis=0))
    if pred is not None:
        contrib = contrib * pred(bg["centroids"])
    return float(contrib.sum())


_trapz = getattr(np, "trapezoid", None) or np.trapz


def _grid_flux(grid: Grid, values: np.ndarray, where: Where) -> float:
    if callable(where):
        raise ValueError("flux on a grid takes face names ('x_min', 'x_max', ...) or None, not a predicate")
    v = values.reshape(*grid.shape, grid.dim)
    names = [f"{'xyz'[a]}_{side}" for a in range(grid.dim) for side in ("min", "max")]
    if where is not None:
        if where not in names:
            raise ValueError(f"unknown face {where!r}; have {names}")
        names = [where]
    total = 0.0
    for name in names:
        a = "xyz".index(name[0])
        sign, idx = (-1.0, 0) if name.endswith("min") else (1.0, -1)
        face = np.take(v[..., a], idx, axis=a)  # normal component on the face
        axes = [grid.axes[b] for b in range(grid.dim) if b != a]
        for ax in range(len(axes) - 1, -1, -1):
            face = _trapz(face, axes[ax], axis=ax)
        total += sign * float(face)
    return total


def _mc_flux(evaluate, domain: Domain, where: Where, n: int, seed: int, torch_out: bool):
    pts, nrm = domain.sample_boundary(n, seed=seed, return_normals=True)
    mask = np.ones(n, dtype=bool)
    pred = _face_predicate(where, domain.lo, domain.hi)
    if pred is not None:
        mask = pred(pts)
    vals = evaluate(pts)
    if torch_out:
        import torch

        nt = torch.as_tensor(nrm, dtype=vals.dtype, device=vals.device)
        mt = torch.as_tensor(mask, dtype=vals.dtype, device=vals.device)
        return domain.boundary_measure() * ((vals * nt).sum(dim=1) * mt).mean()
    return float(domain.boundary_measure() * np.mean((vals * nrm).sum(axis=1) * mask))
