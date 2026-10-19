"""Density filter and Heaviside projection for density-based topology optimization.

Density filter (Bruns & Tortorelli 2001; Bourdin 2001): ``x_tilde = (H (v * x)) / (H v)`` with cone
weights ``H_ij = max(0, r_min - |c_i - c_j|)`` and element volumes ``v``. Works on any element
centroids (structured grids or the triangles of a rotor), built with a KD-tree.

Heaviside projection (Wang, Lazarov & Sigmund 2011)::

    x_bar = (tanh(beta eta) + tanh(beta (x_tilde - eta))) / (tanh(beta eta) + tanh(beta (1 - eta)))

``beta -> infinity`` gives a 0/1 design; ``beta`` is raised by continuation.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

__all__ = ["DensityFilter", "heaviside", "heaviside_derivative", "SymmetryMap"]


class DensityFilter:
    """Linear density filter ``x_tilde = W x`` with ``W = diag(1/(H v)) H diag(v)``."""

    def __init__(self, centroids: np.ndarray, r_min: float, volumes: np.ndarray | None = None):
        from scipy.spatial import cKDTree

        c = np.asarray(centroids, float)
        n = len(c)
        v = np.ones(n) if volumes is None else np.asarray(volumes, float)
        tree = cKDTree(c)
        pairs = tree.query_pairs(r_min, output_type="ndarray")
        i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        d = np.linalg.norm(c[i] - c[j], axis=1)
        w = np.maximum(0.0, r_min - d)
        H = sp.csr_matrix((w, (i, j)), shape=(n, n))
        Hv = H @ v
        self.W = (sp.diags(1.0 / Hv) @ H @ sp.diags(v)).tocsr()
        self.WT = self.W.T.tocsr()

    @classmethod
    def grid(cls, nelx: int, nely: int, r_min: float, h: float = 1.0) -> DensityFilter:
        """Filter on a structured ``nelx x nely`` grid, element ``e = i * nely + j``, size ``h``."""
        i, j = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        c = np.column_stack([(i.ravel() + 0.5) * h, (j.ravel() + 0.5) * h])
        return cls(c, r_min * h)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Chain rule: d/dx of a function of x_tilde with gradient ``g``."""
        return self.WT @ g


def heaviside(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    if beta <= 0:
        return x
    num = math.tanh(beta * eta) + np.tanh(beta * (x - eta))
    return num / (math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta)))


def heaviside_derivative(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    if beta <= 0:
        return np.ones_like(x)
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / (
        math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta)))


class SymmetryMap:
    """Element densities from fewer design variables: ``x_elem = S x_var`` (a 0/1 sparse matrix).

    ``rotational(centroids, n_fold)``: every element takes the variable of its image in the sector
    ``0 <= phi < 2 pi / n_fold`` (nearest centroid after rotation; exact on meshes with that symmetry).
    """

    def __init__(self, S: sp.csr_matrix):
        self.S = S.tocsr()
        self.ST = self.S.T.tocsr()

    @property
    def n_var(self) -> int:
        return self.S.shape[1]

    @classmethod
    def identity(cls, n: int) -> SymmetryMap:
        return cls(sp.identity(n, format="csr"))

    @classmethod
    def rotational(cls, centroids: np.ndarray, n_fold: int, tol: float | None = None) -> SymmetryMap:
        from scipy.spatial import cKDTree

        c = np.asarray(centroids, float)
        phi = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * math.pi)
        sector = 2 * math.pi / n_fold
        k = np.floor(phi / sector + 1e-9).astype(int) % n_fold
        base = np.where(k == 0)[0]
        ang = -k * sector
        rot = np.column_stack([c[:, 0] * np.cos(ang) - c[:, 1] * np.sin(ang),
                               c[:, 0] * np.sin(ang) + c[:, 1] * np.cos(ang)])
        tree = cKDTree(c[base])
        d, idx = tree.query(rot)
        if tol is not None and np.max(d) > tol:
            raise ValueError(f"mesh is not {n_fold}-fold symmetric (max mismatch {np.max(d):.3g})")
        S = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), idx)), shape=(len(c), len(base)))
        return cls(S)

    def __call__(self, x):
        return self.S @ x

    def backward(self, g):
        return self.ST @ g
