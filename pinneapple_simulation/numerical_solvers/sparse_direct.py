"""One sparse direct solve: MKL PARDISO (pypardiso, when importable) or SciPy's SuperLU.

Both the body-fitted Taylor-Hood reference and the immersed flow solver use this, so a time comparison between them
is not a comparison of linear solvers. ``solver``: ``"auto"`` (PARDISO if available), ``"pardiso"`` or ``"superlu"``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["solve", "available"]


def available() -> bool:
    try:
        _mkl_path()
        import pypardiso  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - optional accelerator
        return False


def solve(A: sparse.spmatrix, b: np.ndarray, solver: str = "auto") -> np.ndarray:
    """x = A^-1 b for a general (non-symmetric, indefinite) sparse matrix."""
    if solver not in ("auto", "pardiso", "superlu"):
        raise ValueError("solver must be 'auto', 'pardiso' or 'superlu'")
    if solver != "superlu" and available():
        import pypardiso
        ps = pypardiso.PyPardisoSolver(mtype=11)                  # real non-symmetric
        try:
            return ps.solve(sparse.csr_matrix(A), np.asarray(b, float))
        finally:
            ps.free_memory(everything=True)
    if solver == "pardiso":
        raise ImportError("pip install pypardiso")
    from scipy.sparse.linalg import spsolve
    return spsolve(sparse.csc_matrix(A), np.asarray(b, float))


def _mkl_path() -> None:
    from pinneapple_simulation.electromagnetics.machines3d.spm import _mkl_path as f
    f()
