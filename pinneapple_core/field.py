"""Field: values of a physical quantity on a grid, a mesh or a point cloud.

A :class:`Field` ties values to the space they live on, so the same four calls
work on every discretisation::

    f.gradient()         # derivative, as a Field on the same support
    f.divergence()       # trace of the gradient of a vector field
    f.interpolate(x)     # values at arbitrary points
    f.integrate()        # integral over the support

Values have shape ``(N, *components)``: ``(N,)`` for a scalar, ``(N, d)`` for a
vector, ``(N, c, d)`` for a gradient of a vector. :class:`FunctionField` is the
same interface for a callable (an analytic expression or a neural network) with
derivatives from autograd, which is what a PINN needs.

>>> import numpy as np
>>> from pinneapple_core import Field
>>> x = np.linspace(0, 1, 101)
>>> f = Field.on_grid([x], np.sin(np.pi * x))
>>> round(float(f.integrate()), 3)
0.637
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree

from .domain import Domain
from .mesh import Mesh, _as_points

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Supports other than Mesh (which implements the same three methods itself)
# ---------------------------------------------------------------------------
class Grid:
    """Tensor-product grid. ``axes`` are strictly increasing 1D coordinate arrays;
    points are ordered with the last axis varying fastest (``indexing="ij"``)."""

    def __init__(self, axes: Sequence[np.ndarray]) -> None:
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        if not self.axes or any(a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0) for a in self.axes):
            raise ValueError("each grid axis must be 1D, strictly increasing, with >= 2 points")
        self.shape = tuple(a.size for a in self.axes)

    @property
    def dim(self) -> int:
        return len(self.axes)

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    @property
    def points(self) -> np.ndarray:
        return np.stack([g.ravel() for g in np.meshgrid(*self.axes, indexing="ij")], axis=1)

    def gradient(self, values: np.ndarray) -> np.ndarray:
        v = values.reshape(*self.shape, -1)
        edge = 2 if min(self.shape) >= 3 else 1
        g = np.gradient(v, *self.axes, axis=tuple(range(self.dim)), edge_order=edge)
        if self.dim == 1:
            g = [g]
        return np.stack(g, axis=-1).reshape(self.n_points, v.shape[-1], self.dim)

    def interpolate(self, values: np.ndarray, x: np.ndarray, fill: str = "nan") -> np.ndarray:
        x = _as_points(x, self.dim)
        if fill == "nearest":
            x = np.clip(x, [a[0] for a in self.axes], [a[-1] for a in self.axes])
        elif fill != "nan":
            raise ValueError("fill must be 'nan' or 'nearest'")
        interp = RegularGridInterpolator(
            self.axes, values.reshape(*self.shape, -1), bounds_error=False, fill_value=np.nan
        )
        return interp(x)

    def integrate(self, values: np.ndarray) -> np.ndarray:
        v = values.reshape(*self.shape, -1)
        for ax in range(self.dim - 1, -1, -1):
            v = _trapezoid(v, self.axes[ax], axis=ax)
        return v


class PointCloud:
    """Scattered points. ``weights`` are quadrature weights (they sum to the
    measure of the region); without them, ``integrate`` falls back to Monte Carlo
    over ``domain`` and assumes the points are uniform samples of it."""

    def __init__(
        self,
        points: np.ndarray,
        *,
        domain: Optional[Domain] = None,
        weights: Optional[np.ndarray] = None,
        k_neighbors: Optional[int] = None,
    ) -> None:
        pts = np.asarray(points, dtype=np.float64)
        self.points = pts[:, None] if pts.ndim == 1 else pts
        if self.points.ndim != 2:
            raise ValueError("points must have shape (N, d)")
        self.domain = domain
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights is not None and self.weights.shape != (self.n_points,):
            raise ValueError("weights must have shape (N,)")
        d = self.dim
        self.k_neighbors = int(k_neighbors or max(2 * d + 3, 8))
        self._tree: Optional[cKDTree] = None

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """Local least-squares plane through the k nearest neighbours (exact for linear data)."""
        k = min(self.k_neighbors, self.n_points)
        if k < self.dim + 1:
            raise ValueError(f"need at least {self.dim + 1} points to estimate a gradient")
        _, nb = self.tree.query(self.points, k=k)
        dx = self.points[nb[:, 1:]] - self.points[:, None, :]  # (N, k-1, d)
        dist = np.linalg.norm(dx, axis=2)
        w = 1.0 / np.maximum(dist, 1e-12) ** 2
        dv = values[nb[:, 1:]] - values[:, None, :]  # (N, k-1, c)
        a = np.einsum("nk,nki,nkj->nij", w, dx, dx)
        a += 1e-12 * np.trace(a, axis1=1, axis2=2)[:, None, None] * np.eye(self.dim)
        b = np.einsum("nk,nki,nkc->nic", w, dx, dv)
        sol = np.linalg.solve(a, b)  # (N, d, c)
        return np.transpose(sol, (0, 2, 1))

    def interpolate(self, values: np.ndarray, x: np.ndarray, fill: str = "nan") -> np.ndarray:
        x = _as_points(x, self.dim)
        if fill not in ("nan", "nearest"):
            raise ValueError("fill must be 'nan' or 'nearest'")
        if self.dim == 1:
            order = np.argsort(self.points[:, 0])
            xs = self.points[order, 0]
            out = np.stack([np.interp(x[:, 0], xs, values[order, c], left=np.nan, right=np.nan)
                            for c in range(values.shape[1])], axis=1)
        else:
            out = LinearNDInterpolator(self.points, values, fill_value=np.nan)(x)
            out = out.reshape(x.shape[0], values.shape[1])
        if fill == "nearest":
            miss = np.isnan(out).any(axis=1)
            if miss.any():
                out[miss] = values[self.tree.query(x[miss])[1]]
        return out

    def integrate(self, values: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            return (values * self.weights[:, None]).sum(axis=0)
        if self.domain is None:
            raise ValueError("integrating a point cloud needs quadrature weights or a Domain (Monte Carlo)")
        return self.domain.volume() * values.mean(axis=0)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------
class Field:
    """Values on a :class:`Grid`, a :class:`~pinneapple_core.Mesh` or a :class:`PointCloud`."""

    def __init__(self, values: np.ndarray, support, name: Optional[str] = None) -> None:
        v = np.asarray(values, dtype=np.float64)
        if v.shape[:1] != (support.n_points,):
            if isinstance(support, Grid) and v.shape[: support.dim] == support.shape:
                v = v.reshape(support.n_points, *v.shape[support.dim:])  # accept grid-shaped arrays
            else:
                raise ValueError(f"expected {support.n_points} values on this support, got shape {v.shape}")
        self.values = v
        self.support = support
        self.name = name

    # -- constructors -------------------------------------------------------
    @classmethod
    def on_grid(cls, axes: Sequence[np.ndarray], values, name: Optional[str] = None) -> "Field":
        """Field on a tensor-product grid. ``values`` is an array (flat or grid-shaped)
        or a callable ``f(points) -> values``."""
        return cls._build(Grid(axes), values, name)

    @classmethod
    def on_mesh(cls, mesh: Mesh, values, name: Optional[str] = None) -> "Field":
        """Nodal (P1) field on a mesh. ``values`` is an array or a callable."""
        return cls._build(mesh, values, name)

    @classmethod
    def on_points(
        cls,
        points: np.ndarray,
        values,
        *,
        domain: Optional[Domain] = None,
        weights: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> "Field":
        """Field on scattered points. Pass ``weights`` or ``domain`` to enable ``integrate``."""
        return cls._build(PointCloud(points, domain=domain, weights=weights), values, name)

    @classmethod
    def _build(cls, support, values, name) -> "Field":
        if callable(values):
            values = values(support.points)
        return cls(values, support, name)

    # -- shape --------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.support.dim

    @property
    def components(self) -> tuple:
        return self.values.shape[1:]

    @property
    def points(self) -> np.ndarray:
        return self.support.points

    def _flat(self) -> np.ndarray:
        return self.values.reshape(self.values.shape[0], -1)

    def _like(self, values: np.ndarray, name: Optional[str] = None) -> "Field":
        return Field(values, self.support, name)

    # -- calculus -----------------------------------------------------------
    def gradient(self) -> "Field":
        """Derivative with respect to each coordinate: ``(N, d)`` for a scalar,
        ``(N, c, d)`` for a field with ``c`` components."""
        g = self.support.gradient(self._flat())  # (N, k, d)
        g = g.reshape(g.shape[0], *self.components, self.dim)
        return self._like(g, None if self.name is None else f"grad({self.name})")

    def divergence(self) -> "Field":
        """Divergence of a vector field with ``d`` components: a scalar Field."""
        if self.components != (self.dim,):
            raise ValueError(f"divergence needs a vector field with {self.dim} components, got {self.components}")
        g = self.gradient().values  # (N, d, d): g[n, i, j] = d u_i / d x_j
        return self._like(np.trace(g, axis1=1, axis2=2), None if self.name is None else f"div({self.name})")

    def interpolate(self, x: np.ndarray, fill: str = "nan") -> np.ndarray:
        """Values at points ``x`` of shape ``(Q, d)``; outside the support gives NaN
        (``fill="nan"``) or the nearest value (``fill="nearest"``)."""
        out = self.support.interpolate(self._flat(), x, fill)
        return out.reshape(out.shape[0], *self.components)

    def integrate(self) -> Union[float, np.ndarray]:
        """Integral over the support: trapezoid on a grid, exact P1 on a mesh,
        weighted sum or Monte Carlo on a point cloud. A scalar field gives a float."""
        out = self.support.integrate(self._flat())
        return float(out[0]) if self.components == () else out.reshape(self.components)

    def __repr__(self) -> str:
        return f"Field({type(self.support).__name__}, n={self.support.n_points}, components={self.components})"


# ---------------------------------------------------------------------------
# FunctionField: the same interface for a callable, derivatives by autograd
# ---------------------------------------------------------------------------
class FunctionField:
    """A field defined by a torch callable ``fn((N, d) tensor) -> (N,) or (N, c)``.

    All derivative methods take the evaluation points ``x`` and keep the graph, so
    results can go straight into a PINN loss. A trailing output dimension of 1 is
    treated as a scalar.
    """

    def __init__(self, fn: Callable, domain: Optional[Domain] = None, dim: Optional[int] = None) -> None:
        self.fn = fn
        self.domain = domain
        self._dim = dim if dim is not None else (domain.dim if domain is not None else None)

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @staticmethod
    def _tensor(x):
        import torch

        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x), dtype=torch.get_default_dtype())

    def __call__(self, x):
        y = self.fn(self._tensor(x))
        return y.squeeze(-1) if y.ndim > 1 and y.shape[-1] == 1 else y

    def interpolate(self, x):
        """Value at ``x`` (a function field is defined everywhere)."""
        return self(x)

    def gradient(self, x, create_graph: bool = True):
        """``(N, d)`` for a scalar field, ``(N, c, d)`` for ``c`` components."""
        import torch

        x = self._tensor(x)
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        y = self(x)
        if y.ndim == 1:
            (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
            return g
        flat = y.reshape(y.shape[0], -1)
        cols = [
            torch.autograd.grad(flat[:, j].sum(), x, create_graph=create_graph, retain_graph=True)[0]
            for j in range(flat.shape[1])
        ]
        return torch.stack(cols, dim=1).reshape(y.shape[0], *y.shape[1:], x.shape[1])

    def gradient_field(self) -> "FunctionField":
        """The gradient as another FunctionField, so derivatives compose
        (``u.gradient_field().divergence(x)`` is the Laplacian)."""
        return FunctionField(lambda x: self.gradient(x), self.domain, self._dim)

    def divergence(self, x, create_graph: bool = True):
        import torch

        x = self._tensor(x)
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        y = self(x)
        if y.ndim != 2 or y.shape[1] != x.shape[1]:
            raise ValueError("divergence needs a vector field with as many components as dimensions")
        total = 0.0
        for i in range(y.shape[1]):
            (g,) = torch.autograd.grad(y[:, i].sum(), x, create_graph=create_graph, retain_graph=True)
            total = total + g[:, i]
        return total

    def laplacian(self, x, create_graph: bool = True):
        """Sum of second derivatives of a scalar field."""
        return self.gradient_field().divergence(x, create_graph=create_graph)

    def integrate(self, n: int = 20_000, *, seed: int = 0, points=None, weights=None):
        """Integral over the domain. With ``points`` and ``weights`` it is a
        quadrature sum; otherwise Monte Carlo with ``n`` uniform samples."""
        import torch

        if points is not None:
            y = self(points)
            w = self._tensor(weights).to(y.dtype) if weights is not None else None
            if w is None:
                raise ValueError("pass weights together with points")
            return (y * w.reshape(-1, *([1] * (y.ndim - 1)))).sum(dim=0)
        if self.domain is None:
            raise ValueError("Monte Carlo integration needs a Domain")
        x = self._tensor(self.domain.sample_interior(n, seed=seed))
        return self.domain.volume() * self(x).mean(dim=0)

    def to_field(self, support, name: Optional[str] = None) -> Field:
        """Sample onto a Grid / Mesh / PointCloud as a numpy :class:`Field`."""
        import torch

        with torch.no_grad():
            y = self(support.points).cpu().numpy()
        return Field(y, support, name)
