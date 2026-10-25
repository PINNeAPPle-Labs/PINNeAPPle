"""Stencil weights in any dimension for meshless discretisations: RBF-FD and GFDM (weighted least squares).

``stencil_weights(X, Q, idx, ops, method=...)`` returns, for every query point ``Q[i]`` and its stencil ``X[idx[i]]``,
the weights ``w`` such that ``(L u)(Q[i]) ~ sum_j w[i, j] u[idx[i, j]]`` for each operator in ``ops``:

* ``"eval"`` (interpolation), ``"d0"``, ``"d1"``, ``"d2"`` (first derivatives along each axis), ``"lap"`` (the
  Laplacian).

Methods:

``rbf_fd``
    Polyharmonic spline ``r^3`` plus all monomials up to ``degree`` in local coordinates centred at the query point
    and scaled by the stencil radius (Flyer, Fornberg, Bayona and Barnett, J. Comput. Phys. 321 (2016) 21; the 2-D
    Laplacian version is :func:`.methods.rbf_fd_weights`). Exact for the polynomials, interpolating in the stencil.
``gfdm``
    Generalised finite differences: a weighted least-squares fit of the monomials up to ``degree`` (Gaussian weights
    of the distance, scaled by the stencil radius), differentiated at the query point (Benito, Urena and Gavete,
    Appl. Math. Model. 25 (2001) 1039, doi:10.1016/S0307-904X(01)00029-4). Exact for the polynomials, not
    interpolating.

Both are exact for polynomials up to ``degree``: the consistency tests of ``tests/test_meshless_stencils.py`` check
it in 2-D and 3-D.
"""
from __future__ import annotations

from itertools import product

import numpy as np

OPS = ("eval", "d0", "d1", "d2", "lap")


def monomials(dim: int, degree: int) -> list[tuple[int, ...]]:
    """Exponent tuples of every monomial of total degree <= ``degree`` in ``dim`` variables (constant first)."""
    out = [e for e in product(range(degree + 1), repeat=dim) if sum(e) <= degree]
    return sorted(out, key=lambda e: (sum(e), [-x for x in e]))


def _poly_ops(mons, dim):
    """Values at the origin of each operator applied to each monomial."""
    res = {op: np.zeros(len(mons)) for op in OPS}
    for j, e in enumerate(mons):
        s = sum(e)
        if s == 0:
            res["eval"][j] = 1.0
        if s == 1:
            res[f"d{e.index(1)}"][j] = 1.0
        if s == 2 and max(e) == 2:
            res["lap"][j] = 2.0
    return res


def stencil_weights(X: np.ndarray, Q: np.ndarray, idx: np.ndarray, ops=("lap",), *, method: str = "rbf_fd",
                    degree: int = 2, batch: int = 2000) -> dict[str, np.ndarray]:
    """Weights ``{op: (q, n)}`` (see the module docstring). ``X``: (N, dim) nodes; ``Q``: (q, dim) query points;
    ``idx``: (q, n) stencil node indices."""
    X, Q = np.asarray(X, float), np.asarray(Q, float)
    q, n = idx.shape
    dim = X.shape[1]
    mons = monomials(dim, degree)
    m = len(mons)
    if n < m + (1 if method == "rbf_fd" else 0):
        raise ValueError(f"stencils of {n} nodes are too small for degree {degree} in {dim}-D ({m} monomials)")
    pv = _poly_ops(mons, dim)
    E = np.array(mons)                                                  # (m, dim)
    out = {op: np.empty((q, n)) for op in ops}
    for s in range(0, q, batch):
        st = idx[s:s + batch]
        b = len(st)
        Z = X[st] - Q[s:s + batch, None, :]                             # (b, n, dim)
        sc = np.sqrt((Z ** 2).sum(-1)).max(1) + 1e-300                  # (b,)
        Z = Z / sc[:, None, None]
        P = np.prod(Z[:, :, None, :] ** E[None, None, :, :], axis=-1)   # (b, n, m)
        if method == "rbf_fd":
            D = np.sqrt(((Z[:, :, None, :] - Z[:, None, :, :]) ** 2).sum(-1))
            A = np.zeros((b, n + m, n + m))
            A[:, :n, :n] = D ** 3
            A[:, :n, n:] = P
            A[:, n:, :n] = np.transpose(P, (0, 2, 1))
            r0 = np.sqrt((Z ** 2).sum(-1))                              # (b, n): |x_j - query|
            rhs = np.zeros((b, n + m, len(ops)))
            for c, op in enumerate(ops):
                if op == "eval":
                    rhs[:, :n, c] = r0 ** 3
                elif op == "lap":
                    rhs[:, :n, c] = 3.0 * (dim + 1) * r0               # Delta r^3 = 3 (dim + 1) r
                else:                                                   # d/dq_k of |x_j - q|^3 = -3 r (x_j - q)_k
                    k = int(op[1])
                    rhs[:, :n, c] = -3.0 * r0 * Z[:, :, k]
                rhs[:, n:, c] = pv[op]
            w = np.linalg.solve(A, rhs)[:, :n, :]                      # (b, n, nops)
        elif method == "gfdm":
            r = np.sqrt((Z ** 2).sum(-1))
            wt = np.exp(-(r / 0.6) ** 2)                                # (b, n)
            PtW = np.transpose(P, (0, 2, 1)) * wt[:, None, :]           # (b, m, n)
            G = PtW @ P                                                 # (b, m, m)
            coef = np.linalg.solve(G, PtW)                              # (b, m, n): monomial coefficients
            w = np.stack([np.einsum("m,bmn->bn", pv[op], coef) for op in ops], -1)
        else:
            raise ValueError(f"unknown method {method!r} (rbf_fd, gfdm)")
        for c, op in enumerate(ops):
            scale = 1.0 if op == "eval" else (sc if op != "lap" else sc ** 2)
            out[op][s:s + batch] = w[:, :, c] / (scale[:, None] if np.ndim(scale) else 1.0)
    return out
