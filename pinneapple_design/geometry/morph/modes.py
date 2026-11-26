"""Shape modes: the low-dimensional design space a family of registered meshes already spans.

A family of meshes with the same topology — the AhmedML or DrivAerML variants, a set of shapes exported from the same
CAD with the same tessellation — is a cloud of points in ``3 n_vertices`` dimensions. Its principal components are the
shape modes: the first few coefficients reproduce most of the family, and moving along one of them produces a shape
between the ones measured. This is what a designer means when asking for "the space the real shapes live in", and it
is the cheapest of the three morphings: a design is a linear combination of precomputed fields.

The meshes must be *registered* — same vertex count, same faces, vertex ``i`` the same physical point in every
sample — which is what the FFD and RBF morphings produce from a single reference. Fitting on meshes that are merely
similar would mix the deformation with a correspondence error and the modes would be noise; ``fit`` refuses a sample
whose shape does not match rather than averaging it in.

The modes are signed; ``bounds`` gives the band each coefficient covered in the data, so a sampler stays inside the
family instead of extrapolating into shapes nobody simulated.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = ["ShapeModes"]


class ShapeModes:
    """PCA of a registered mesh family: ``mean + sum_k c_k mode_k``.

    ``energy`` (default 0.99) keeps the number of modes that explain that fraction of the variance, capped at
    ``max_modes``. ``fit`` takes the deformations, not the absolute meshes: a family is compared around its mean, and
    the mean is stored so the same object deforms the reference mesh.
    """

    def __init__(self, *, energy: float = 0.99, max_modes: int = 10):
        self.energy = float(energy)
        self.max_modes = int(max_modes)
        self.reference: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.modes: np.ndarray | None = None       # (n_modes, 3 n_vertices)
        self.std: np.ndarray | None = None         # standard deviation of each coefficient in the data
        self.n_parameters = 0

    def fit(self, meshes: Iterable[Any], reference: Any = None) -> ShapeModes:
        """Fit on registered meshes; the first one is the reference unless ``reference`` is given."""
        samples = [np.asarray(m, dtype=np.float64).reshape(-1, 3) for m in meshes]
        if len(samples) < 2:
            raise ValueError("ShapeModes: at least two registered meshes are needed")
        shape = samples[0].shape
        for s in samples:
            if s.shape != shape:
                raise ValueError(f"ShapeModes: meshes are not registered ({s.shape} against {shape}); "
                                 "every sample must have the same vertices and the same faces")
        X = np.stack([s.ravel() for s in samples])
        mean = X.mean(0)
        self.reference = samples[0].copy()
        if reference is not None:
            r = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
            if r.shape != shape:
                raise ValueError(f"ShapeModes: the reference has {r.shape} vertices, the family {shape}")
            self.reference = r.copy()
        U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
        var = S ** 2 / max(len(samples) - 1, 1)
        if not var.sum() > 0:
            raise ValueError("ShapeModes: the family has no variance — every sample is the same mesh")
        keep = int(np.searchsorted(np.cumsum(var) / var.sum(), self.energy) + 1)
        keep = max(1, min(keep, self.max_modes, len(Vt)))
        self.modes = Vt[:keep]
        self.std = np.sqrt(var[:keep])
        self.mean = mean - self.reference.ravel()        # stored as a displacement from the reference
        self.n_parameters = keep
        self.singular_values = S
        self.explained = var / var.sum()
        return self

    @property
    def names(self) -> list[str]:
        return [f"mode.{k}" for k in range(self.n_parameters)]

    def bounds(self, scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """``± scale`` standard deviations of each coefficient, the band the family itself covers."""
        mag = scale * np.asarray(self.std, dtype=float)
        return -mag, mag

    def deform(self, x: Any, vertices: Any = None) -> np.ndarray:
        """The mesh for coefficients ``x``; zeros give the family's mean shape, not the reference.

        The origin of a modal space is the middle of the family — that is the point the coefficients are measured
        from and the point a sampler should explore around. ``deform(project(V))`` is the identity on a registered
        mesh, and :meth:`residual` is how much of it the kept modes cannot express."""
        c = np.asarray(x, dtype=float).reshape(-1)
        if c.size != self.n_parameters:
            raise ValueError(f"ShapeModes: expected {self.n_parameters} coefficients, got {c.size}")
        base = self.reference if vertices is None else np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if base.shape != self.reference.shape:
            raise ValueError("ShapeModes: a mode is defined on the registered vertices, not on another mesh")
        return base + (self.mean + self.modes.T @ c).reshape(base.shape)

    def project(self, vertices: Any) -> np.ndarray:
        """The coefficients of a mesh in the mode basis: the inverse of :meth:`deform`, truncated to the kept modes.

        Fitting a new shape into the space is how a study compares a measured geometry with the family it came from:
        the residual of ``deform(project(V))`` against ``V`` is the part of the shape the modes cannot express."""
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if V.shape != self.reference.shape:
            raise ValueError("ShapeModes: a mode is defined on the registered vertices, not on another mesh")
        return self.modes @ ((V - self.reference).ravel() - self.mean)

    def residual(self, vertices: Any) -> float:
        """How much of a mesh the kept modes cannot express, relative to the family's own variation.

        The denominator is the distance from the family mean, not the size of the mesh: the question is what part of
        the *shape change* is left out, and a car measured against its own bounding box would look reconstructed when
        it is not."""
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        d = (V - self.reference).ravel() - self.mean
        back = self.modes.T @ (self.modes @ d)
        return float(np.linalg.norm(d - back) / max(np.linalg.norm(d), 1e-300))

    def modal_energy(self) -> list[float]:
        """The fraction of the family's variance each kept mode explains, largest first."""
        return [float(v) for v in self.explained[:self.n_parameters]]

    def __repr__(self) -> str:
        return f"ShapeModes({self.n_parameters} modes)"
