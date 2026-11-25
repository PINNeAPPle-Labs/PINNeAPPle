"""Free-form deformation: a lattice of control points around the body, and design variables that are their
displacements (Sederberg & Parry, "Free-Form Deformation of Solid Geometric Models", SIGGRAPH 1986).

The map is the tensor-product Bernstein polynomial of the lattice: a vertex at normalised coordinates ``(s, t, u)``
inside the lattice box moves by the Bernstein-weighted sum of the control-point displacements,

    d(s, t, u) = sum_i sum_j sum_k B_i^l(s) B_j^m(t) B_k^n(u) d_ijk

with ``B_i^l`` the Bernstein basis of degree ``l``. Two properties make this usable as a design space rather than a
toy: the weights are non-negative and sum to one, so a zero displacement vector returns the mesh **exactly** and a
bounded control-point displacement bounds every vertex displacement; and the map is smooth, so a lattice perturbation
moves a whole region instead of denting vertices.

What the deformation cost is worth knowing: the Bernstein weights depend on the reference mesh and the lattice, not on
the parameters, so they are computed once, and a new design is then one matrix product of shape
``(n_vertices, n_control) x (n_control, 3)``. On the road car of ``data/geometries`` (10 898 vertices, 27 control
points) that is 3.5 M flops per design — the measured time is in ``docs/geometry/morphing.md``.

Symmetry and keep-out are constraints on the control points, resolved once when the lattice is built: a mirror plane
identifies each control point with its image (one of the pair carries the parameters, the other follows with the
normal component flipped, and a control point sitting on the plane loses that component), and a keep-out box or a
frozen region removes the points inside it from the parameter vector. The parameter vector therefore never contains a
direction the geometry cannot take, which is what keeps a sampler from proposing a broken mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["FFD"]


def _bernstein(t: np.ndarray, degree: int) -> np.ndarray:
    """Bernstein basis of ``degree`` evaluated at ``t`` (n,) -> (n, degree + 1)."""
    t = np.clip(t, 0.0, 1.0)
    from math import comb

    out = np.empty((len(t), degree + 1))
    for i in range(degree + 1):
        out[:, i] = comb(degree, i) * t ** i * (1.0 - t) ** (degree - i)
    return out


@dataclass
class FFD:
    """A control lattice around a surface; ``deform`` maps a parameter vector to the deformed vertices.

    Built with :meth:`from_bounds` (or :meth:`fit`), which also fixes the parameter vector: ``n_parameters`` is
    ``3 x free control points``, minus the normal components pinned by the symmetry plane, and ``names`` gives each
    entry a readable name (``ffd.1.0.2.y``) so a study can report which control point the drag responded to.
    """

    lo: np.ndarray
    hi: np.ndarray
    control: np.ndarray                      # (l+1, m+1, n+1, 3) control-point positions
    reference: np.ndarray | None = None      # (n_vertices, 3) mesh the weights were built for
    symmetry: tuple[int, float] | None = None
    param_of: np.ndarray | None = None       # (n_control, 3) index into the parameter vector, -1 when fixed
    sign: np.ndarray | None = None           # (n_control, 3) apply to the parameter to get the displacement
    frozen: np.ndarray | None = None         # (n_control,) bool, kept at zero displacement
    _weights: np.ndarray | None = None       # (n_vertices, n_control) Bernstein weights
    _base: np.ndarray | None = None          # (n_vertices, 3) identity image of the reference mesh

    # ------------------------------------------------------------------------------------- construction
    @classmethod
    def fit(cls, vertices: Any, *, control: Sequence[int] = (3, 3, 3), padding: float = 0.05,
            symmetry: str | int | None = None, symmetry_plane: float | None = None,
            freeze: Iterable[Any] = (), freeze_boxes: Iterable[Sequence[Sequence[float]]] = (),
            ) -> FFD:
        """Build the lattice around ``vertices`` and compute the weights.

        ``control`` is the number of *cells* per axis (an ``(l, m, n)`` segmentation, so ``l + 1`` control points);
        ``padding`` grows the box by that fraction of the body size on every side, so the surface is strictly inside
        it. ``symmetry`` names the axis of the mirror plane (``"x"``, ``"y"``, ``"z"`` or 0/1/2) and
        ``symmetry_plane`` its coordinate (default: the centre of the body). ``freeze`` holds control-point indices or
        ``(i, j, k)`` tuples that may not move; ``freeze_boxes`` holds ``(lo, hi)`` boxes in world coordinates —
        every control point inside one is frozen. That is the keep-out: a mounting face, a region the solver may not
        see move.
        """
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if len(control) == 1:
            control = (control[0],) * 3
        lo, hi = V.min(0), V.max(0)
        size = np.maximum(hi - lo, 1e-12)
        lo, hi = lo - padding * size, hi + padding * size
        axes = [np.linspace(lo[a], hi[a], int(control[a]) + 1) for a in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        self = cls(lo=lo, hi=hi, control=grid, reference=V)
        if symmetry is not None:
            axis = symmetry if isinstance(symmetry, int) else "xyz".index(str(symmetry).lower())
            plane = float(0.5 * (V[:, axis].min() + V[:, axis].max()) if symmetry_plane is None
                          else symmetry_plane)
            self.symmetry = (axis, plane)
        self._resolve_parameters(freeze, freeze_boxes)
        self._build()
        return self

    # compatibility: the same object under the name the issue uses
    build = fit

    def _resolve_parameters(self, freeze: Iterable[Any], freeze_boxes: Iterable[Sequence[Sequence[float]]]) -> None:
        ctrl = self.control.reshape(-1, 3)
        nc = len(ctrl)
        self.frozen = np.zeros(nc, dtype=bool)
        for f in freeze:
            idx = np.ravel_multi_index(tuple(int(x) for x in f), self.control.shape[:3]) if (
                not isinstance(f, (int, np.integer))) else int(f)
            if not 0 <= idx < nc:
                raise ValueError(f"FFD: frozen control point {f!r} is outside the lattice ({nc} points)")
            self.frozen[idx] = True
        for box in freeze_boxes:
            blo, bhi = (np.asarray(b, dtype=float) for b in box)
            inside = np.all((ctrl >= blo) & (ctrl <= bhi), axis=1)
            self.frozen |= inside
        tol = 1e-9 * float(np.linalg.norm(self.hi - self.lo))
        if self.symmetry is not None:
            # a keep-out is symmetric too: freezing a control point freezes its image, or the body would lose the
            # symmetry the design space promises
            axis, plane = self.symmetry
            image = ctrl.copy()
            image[:, axis] = 2.0 * plane - ctrl[:, axis]
            mirror = np.array([int(np.argmin(np.linalg.norm(ctrl - q, axis=1))) for q in image])
            self.frozen |= self.frozen[mirror]
        if self.frozen.all():
            raise ValueError("FFD: every control point is frozen — the lattice has no parameters left")

        param_of = np.full((nc, 3), -1, dtype=np.int64)
        sign = np.ones((nc, 3))
        n = 0
        for i, p in enumerate(ctrl):
            if self.frozen[i]:
                continue
            if self.symmetry is not None:
                axis, plane = self.symmetry
                if p[axis] > plane + tol:
                    continue                       # the image of a free point on the low side: follows it
                sign[i, axis] = 1.0 if p[axis] < plane - tol else 0.0
            for d in range(3):
                if sign[i, d] == 0.0:
                    continue                       # on the mirror plane: the normal component is pinned
                param_of[i, d] = n
                n += 1
        if self.symmetry is not None:
            axis, plane = self.symmetry
            image = ctrl.copy()
            image[:, axis] = 2.0 * plane - ctrl[:, axis]
            for i, p in enumerate(ctrl):
                if self.frozen[i] or p[axis] <= plane + tol:
                    continue
                j = int(np.argmin(np.linalg.norm(ctrl - image[i], axis=1)))
                param_of[i] = param_of[j]
                sign[i] = sign[j].copy()
                sign[i, axis] = -sign[j, axis]
        self.param_of = param_of.reshape(self.control.shape[:3] + (3,))
        self.sign = sign.reshape(self.control.shape[:3] + (3,))
        self.n_parameters = int(n)
        if n == 0:
            raise ValueError("FFD: the constraints left no free parameter")

    def _build(self) -> None:
        V = self.reference
        s = (V - self.lo) / np.maximum(self.hi - self.lo, 1e-300)
        Bx = _bernstein(s[:, 0], self.control.shape[0] - 1)
        By = _bernstein(s[:, 1], self.control.shape[1] - 1)
        Bz = _bernstein(s[:, 2], self.control.shape[2] - 1)
        W = (Bx[:, :, None, None] * By[:, None, :, None] * Bz[:, None, None, :]).reshape(len(V), -1)
        self._weights = W
        self._base = W @ self.control.reshape(-1, 3)

    # ------------------------------------------------------------------------------------- parameters
    @property
    def names(self) -> list[str]:
        """One name per parameter, in order: ``ffd.i.j.k.<axis>``, the control point and the direction."""
        out: list[str] = [""] * self.n_parameters
        shape = self.param_of.shape[:3]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for d, ax in enumerate("xyz"):
                        p = int(self.param_of[i, j, k, d])
                        if p >= 0:
                            out[p] = f"ffd.{i}.{j}.{k}.{ax}"
        return out

    def full(self, x: Any) -> np.ndarray:
        """The full control-point displacement field for a parameter vector (or already-full field)."""
        x = np.asarray(x, dtype=float)
        nc = int(np.prod(self.control.shape[:3]))
        if x.shape == self.control.shape:
            return x.copy()
        x = x.reshape(-1)
        if x.size == nc * 3:
            return x.reshape(nc, 3)
        if x.size != self.n_parameters:
            raise ValueError(f"FFD: expected {self.n_parameters} parameters, got {x.size}")
        full = np.zeros((nc, 3))
        m = self.param_of.reshape(nc, 3)
        sel = m >= 0
        full[sel] = self.sign.reshape(nc, 3)[sel] * x[m[sel]]
        return full

    def deform(self, x: Any = 0.0, vertices: Any = None) -> np.ndarray:
        """The deformed vertices for the parameter vector ``x`` (zeros: the reference mesh, unchanged).

        ``vertices=`` applies the same deformation to another point set inside the lattice (a volume mesh, say),
        through :meth:`weights`."""
        full = self.full(x).reshape(-1, 3)
        if vertices is None:
            return self._base + self._weights @ full
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        return V + self.weights(V) @ full

    def weights(self, vertices: Any) -> np.ndarray:
        """Bernstein weights of another set of points, kept inside the lattice box (clamped)."""
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        s = (V - self.lo) / np.maximum(self.hi - self.lo, 1e-300)
        Bx = _bernstein(s[:, 0], self.control.shape[0] - 1)
        By = _bernstein(s[:, 1], self.control.shape[1] - 1)
        Bz = _bernstein(s[:, 2], self.control.shape[2] - 1)
        return (Bx[:, :, None, None] * By[:, None, :, None] * Bz[:, None, None, :]).reshape(len(V), -1)

    # ------------------------------------------------------------------------------------- design space
    def bounds(self, scale: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """A symmetric band on each parameter, ``scale`` times the lattice cell size (default ±15 % of a cell).

        The band is per axis, because a cell is not a cube: the x parameters of a long car may move further than the
        z ones without leaving the shape the mesh can take. The Bernstein weights sum to one, so no vertex moves
        further than the largest control-point displacement in the band."""
        cells = np.array(self.control.shape[:3], dtype=int) - 1
        cell = (self.hi - self.lo) / np.maximum(cells, 1)
        flat = self.param_of.reshape(-1, 3)
        mag = np.zeros(self.n_parameters)
        for idx in range(self.n_parameters):
            where = np.argwhere(flat == idx)[0]
            mag[idx] = scale * float(cell[int(where[1])])
        return -mag, mag

    def jacobian_free(self) -> np.ndarray:
        """Column of each parameter in the full field, as ``(n_control * 3,)`` indices — for adjoint-style reports."""
        flat = self.param_of.reshape(-1)
        return np.nonzero(flat >= 0)[0]

    def __repr__(self) -> str:
        c = "x".join(str(n + 1) for n in self.control.shape[:3])
        return (f"FFD({c} control points, {self.n_parameters} parameters"
                f"{', symmetric' if self.symmetry else ''})")
