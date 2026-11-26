"""RBF morphing: handlers on chosen surface points, and a radial-basis displacement field between them.

Where FFD moves a region through a lattice that knows nothing about the body, RBF morphing moves exactly the points the
engineer names — the trailing edge, the top of the roof, the lip of an inlet — and interpolates the displacement to
every other vertex, with the points that must not move held at zero. The field is

    u(x) = sum_i w_i phi(|x - h_i|) + a_0 + a_1 x + a_2 y + a_3 z

with the weights and the affine part solved from the handle values, one linear system per component. The polynomial
part is not decoration: without it the interpolant cannot reproduce a translation or a rotation, and a handle set that
is only moved rigidly would still bend the mesh. With it, a rigid motion of the handles reproduces the rigid motion
exactly, which is one of the tests.

The kernel is the thin-plate spline ``phi(r) = r^2 log r`` by default — the minimum-bending interpolant, which is the
reason a handful of handles can move a large smooth surface without the ripples a Gaussian leaves — and
``multiquadric``, ``gaussian`` and ``linear`` are available.

Cost: the interpolation matrix depends on the handle positions alone, so it is factorised once when the handles are
fitted; a new design is then one matrix product ``(n_vertices, 3 n_handles) x (3 n_handles,)``. With the road car and
ten handles that is 0.33 M flops, measured at the same order as one FFD evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RBFMorph"]

KERNELS = ("tps", "multiquadric", "gaussian", "linear")


def _kernel(name: str, r: np.ndarray, c: float) -> np.ndarray:
    if name == "tps":
        return np.where(r > 0, r * r * np.log(np.maximum(r, 1e-300)), 0.0)
    if name == "multiquadric":
        return np.sqrt(r * r + c * c)
    if name == "gaussian":
        return np.exp(-0.5 * (r / max(c, 1e-300)) ** 2)
    if name == "linear":
        return r
    raise ValueError(f"RBFMorph: unknown kernel {name!r} (one of {', '.join(KERNELS)})")


class RBFMorph:
    """Handles on a surface and the radial-basis field that follows them.

    Parameters
    ----------
    handles: the moving points, either indices into the reference vertices or explicit ``(m, 3)`` coordinates.
    fixed: points that must not move — indices or coordinates. They enter the interpolation as handles with zero
        displacement, which is what keeps a mounting face flat instead of letting the field shear it.
    kernel: ``"tps"`` (default), ``"multiquadric"``, ``"gaussian"`` or ``"linear"``.
    smoothing: Tikhonov term added to the diagonal. Zero (the default) interpolates the handles exactly; a small
        positive value trades that exactness for a smoother field when the handles are noisy.
    """

    def __init__(self, handles: Any, *, fixed: Any = (), kernel: str = "tps", smoothing: float = 0.0,
                 ridge: float | None = None):
        if kernel not in KERNELS:
            raise ValueError(f"RBFMorph: unknown kernel {kernel!r} (one of {', '.join(KERNELS)})")
        self.handles = np.asarray(handles, dtype=float).reshape(-1, 3) if not _is_index_list(handles) else handles
        self.fixed = fixed
        self.kernel = kernel
        self.smoothing = float(smoothing)
        self.ridge = ridge
        self.reference: np.ndarray | None = None
        self._eval: np.ndarray | None = None       # (n_vertices, n_moving, 3) displacement per parameter
        self._centres: np.ndarray | None = None

    # ------------------------------------------------------------------------------------- construction
    def fit(self, vertices: Any, handles: Any = None, *, fixed: Any = None) -> RBFMorph:
        """Resolve the handle positions against ``vertices`` and factorise the system once."""
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.reference = V
        h = self.handles if handles is None else handles
        self.handle_ids = np.asarray(h, dtype=np.int64) if _is_index_list(h) else np.arange(len(np.asarray(h)))
        self.handle_points = V[self.handle_ids] if _is_index_list(h) else np.asarray(h, dtype=float).reshape(-1, 3)
        f = self.fixed if fixed is None else fixed
        if _is_index_list(f) and len(np.asarray(f).reshape(-1)):
            self.fixed_ids = np.asarray(f, dtype=np.int64)
            self.fixed_points = V[self.fixed_ids]
        else:
            self.fixed_points = np.asarray(f, dtype=float).reshape(-1, 3) if np.size(f) else np.zeros((0, 3))
            self.fixed_ids = np.zeros(0, dtype=np.int64)
        if not len(self.handle_points):
            raise ValueError("RBFMorph: at least one handle is required")
        ctrl = np.vstack([self.handle_points, self.fixed_points])
        self.n_parameters = 3 * len(self.handle_points)
        self._centres = ctrl
        scale = float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 1.0
        c = float(np.median(_pdist(ctrl))) if len(ctrl) > 1 else scale
        self._c = max(c * 1e-6, 1e-9 * scale) if self.kernel in ("multiquadric", "gaussian") else 0.0
        K = _kernel(self.kernel, _pdist_mat(ctrl, ctrl), self._c)
        if self.smoothing:
            K = K + self.smoothing * np.eye(len(ctrl))
        P = np.hstack([np.ones((len(ctrl), 1)), ctrl])
        M = np.zeros((len(ctrl) + 4, len(ctrl) + 4))
        M[:len(ctrl), :len(ctrl)] = K
        M[:len(ctrl), len(ctrl):] = P
        M[len(ctrl):, :len(ctrl)] = P.T
        rhs = np.zeros((len(ctrl) + 4, 3 * len(ctrl)))
        for k in range(3 * len(ctrl)):
            rhs[k // 3, k] = 1.0                 # a unit handle displacement in one component
        rhs[len(ctrl):, :] = 0.0
        if len(ctrl) < 4:
            raise ValueError(
                f"RBFMorph: {len(ctrl)} handles/fixed points are not enough for a linear polynomial part in 3-D "
                "(4 are: the interpolation system is rank deficient below that, and the field would reproduce a "
                "rigid motion only on the plane through the handles). Add a handle, or pin one more point with "
                "fixed=.")
        if np.linalg.matrix_rank(M) < len(M):
            raise ValueError("RBFMorph: the handle system is rank deficient — the handles (and fixed points) are "
                             "coincident or coplanar; move one of them")
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        self._sol = sol
        Kx = _kernel(self.kernel, _pdist_mat(V, ctrl), self._c)
        Px = np.hstack([np.ones((len(V), 1)), V])
        E = np.hstack([Kx, Px]) @ sol                       # (n_vertices, 3 * n_ctrl)
        E = E[:, :self.n_parameters].reshape(len(V), -1, 3)  # the fixed handles carry zero displacement
        self._eval = E
        return self

    # ------------------------------------------------------------------------------------- parameters
    @property
    def names(self) -> list[str]:
        return [f"rbf.h{i}.{ax}" for i in range(len(self.handle_points)) for ax in "xyz"]

    def bounds(self, scale: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """A band of ``scale`` times the body size on every handle direction."""
        size = float(np.linalg.norm(self.reference.max(0) - self.reference.min(0))) if self.reference is not None else 1.0
        mag = np.full(self.n_parameters, scale * size)
        return -mag, mag

    def deform(self, x: Any, vertices: Any = None) -> np.ndarray:
        """The deformed vertices for handle displacements ``x`` (length ``3 n_handles``, zeros: unchanged).

        ``vertices=`` evaluates the same field at other points inside the influence region (a volume mesh, a probe
        cloud), by re-evaluating the kernel against the handle centres."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_parameters:
            raise ValueError(f"RBFMorph: expected {self.n_parameters} parameters, got {x.size}")
        if vertices is None:
            return self.reference + np.einsum("vic,ic->vc", self._eval, x.reshape(-1, 3))
        V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        full = np.zeros(3 * len(self._centres))
        full[:self.n_parameters] = x
        Kx = _kernel(self.kernel, _pdist_mat(V, self._centres), self._c)
        Px = np.hstack([np.ones((len(V), 1)), V])
        E = (np.hstack([Kx, Px]) @ self._sol).reshape(len(V), -1, 3)
        return V + np.einsum("vic,ic->vc", E, full.reshape(-1, 3))

    def displacement(self, x: Any) -> np.ndarray:
        """The displacement field alone, at the reference vertices."""
        return np.einsum("vic,ic->vc", self._eval, np.asarray(x, dtype=float).reshape(-1, 3))

    def __repr__(self) -> str:
        return f"RBFMorph({len(self.handle_points)} handles, {len(self.fixed_points)} fixed, {self.kernel})"


def _is_index_list(obj: Any) -> bool:
    a = np.asarray(obj)
    return a.ndim <= 1 and (a.size == 0 or np.issubdtype(a.dtype, np.integer))


def _pdist(a: np.ndarray) -> np.ndarray:
    if len(a) < 2:
        return np.zeros(0)
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
    return d[np.triu_indices(len(a), 1)]


def _pdist_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
