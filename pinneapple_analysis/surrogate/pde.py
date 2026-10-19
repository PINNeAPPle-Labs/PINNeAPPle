"""PDE declarations for physics-informed surrogates: which residual a PINN candidate adds to its data loss.

A declaration names the equation, maps its coordinates to the field's coordinate names and its coefficients to
input columns (or constants)::

    {"type": "burgers", "coords": {"x": "x", "t": "t"}, "coefficients": {"nu": "nu"}}

Supported equations (``u`` the scalar field):

========== =============================== ==================
type        residual                        coords / coefficients
========== =============================== ==================
heat        u_t - alpha u_xx                x, t / alpha
burgers     u_t + u u_x - nu u_xx           x, t / nu
advection   u_t + c u_x                     x, t / c
poisson_1d  k u_xx + f                      x / k, f
laplace_2d  u_xx + u_yy                     x, y / -
========== =============================== ==================

The PINN candidate is only offered when a declaration is given: without a PDE there is no physics loss.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["PDE_TYPES", "PDEDeclaration", "parse_pde"]

import numpy as np

PDE_TYPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "heat": (("x", "t"), ("alpha",)),
    "burgers": (("x", "t"), ("nu",)),
    "advection": (("x", "t"), ("c",)),
    "poisson_1d": (("x",), ("k", "f")),
    "laplace_2d": (("x", "y"), ()),
}


@dataclass
class PDEDeclaration:
    type: str
    coord_index: dict[str, int]                 # equation coordinate -> column of the field's coordinates
    coefficients: dict[str, Any]                # coefficient -> input index (int) or constant (float)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "coord_index": dict(self.coord_index), "coefficients": dict(self.coefficients)}

    def residual(self, u, coords, params):
        """PDE residual at ``coords`` (n, n_coord_dims, requires_grad) for ``params`` (n, d) in physical units."""
        import torch

        def d(f, i):
            return torch.autograd.grad(f, coords, torch.ones_like(f), create_graph=True)[0][:, i:i + 1]

        def coef(name):
            c = self.coefficients[name]
            return params[:, c:c + 1] if isinstance(c, int) else torch.full_like(u, float(c))

        ix = self.coord_index
        if self.type in ("heat", "burgers", "advection"):
            u_x, u_t = d(u, ix["x"]), d(u, ix["t"])
            if self.type == "advection":
                return u_t + coef("c") * u_x
            u_xx = d(u_x, ix["x"])
            if self.type == "heat":
                return u_t - coef("alpha") * u_xx
            return u_t + u * u_x - coef("nu") * u_xx
        if self.type == "poisson_1d":
            u_x = d(u, ix["x"])
            return coef("k") * d(u_x, ix["x"]) + coef("f")
        if self.type == "laplace_2d":
            return d(d(u, ix["x"]), ix["x"]) + d(d(u, ix["y"]), ix["y"])
        raise ValueError(self.type)

    def fd_residual(self, fields: np.ndarray, coords: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Second-order finite-difference residual of fields ``(n, *grid)`` on a tensor-product grid (``coords``
        (n_points, n_dims) in ``ij`` order), RMS per sample over the interior points. Applied to the reference data
        it measures the data's own discretisation error, the floor a prediction is compared against."""
        fields = np.asarray(fields, float)
        shape = fields.shape[1:]
        C = np.asarray(coords, float).reshape(*shape, len(shape))
        axes = []
        for i in range(len(shape)):                     # coordinate values along axis i (others at index 0)
            sel = [0] * len(shape)
            sel[i] = slice(None)
            axes.append(C[(*sel, i)])
        out = np.empty(len(fields))
        ix = self.coord_index
        inner = tuple(slice(1, -1) for _ in shape)
        for k, (u, p) in enumerate(zip(fields, np.asarray(params, float), strict=True)):
            def d(f, name):
                i = ix[name]
                return np.gradient(f, axes[i], axis=i, edge_order=2)

            def coef(name, p=p):
                c = self.coefficients[name]
                return p[c] if isinstance(c, int) else float(c)
            if self.type in ("heat", "burgers", "advection"):
                u_x, u_t = d(u, "x"), d(u, "t")
                if self.type == "advection":
                    r = u_t + coef("c") * u_x
                elif self.type == "heat":
                    r = u_t - coef("alpha") * d(u_x, "x")
                else:
                    r = u_t + u * u_x - coef("nu") * d(u_x, "x")
            elif self.type == "poisson_1d":
                r = coef("k") * d(d(u, "x"), "x") + coef("f")
            else:
                r = d(d(u, "x"), "x") + d(d(u, "y"), "y")
            out[k] = float(np.sqrt(np.mean(r[inner] ** 2)))
        return out


def parse_pde(decl: Mapping[str, Any] | None, coord_names: list[str], input_names: list[str]) -> PDEDeclaration | None:
    """Validate a declaration against the data; ``None`` passes through."""
    if not decl:
        return None
    t = str(decl.get("type", "")).lower()
    if t not in PDE_TYPES:
        raise ValueError(f"unknown PDE type {t!r}; supported: {sorted(PDE_TYPES)}")
    need_c, need_k = PDE_TYPES[t]
    cmap = dict(decl.get("coords") or {c: c for c in need_c})
    idx = {}
    for c in need_c:
        name = cmap.get(c, c)
        if name not in coord_names:
            raise ValueError(f"PDE coordinate {c!r} -> {name!r} is not a field coordinate {coord_names}")
        idx[c] = coord_names.index(name)
    kmap = dict(decl.get("coefficients") or {})
    coefs: dict[str, Any] = {}
    for k in need_k:
        v = kmap.get(k, k)
        if isinstance(v, str):
            if v not in input_names:
                raise ValueError(f"PDE coefficient {k!r} -> {v!r} is neither an input column nor a number")
            coefs[k] = input_names.index(v)
        else:
            coefs[k] = float(v)
    return PDEDeclaration(t, idx, coefs)
