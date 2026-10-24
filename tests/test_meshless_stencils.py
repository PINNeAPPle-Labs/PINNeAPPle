"""n-D stencil weights (pinneapple_simulation.meshless.stencils): exact on polynomials up to the degree, in 2-D and
3-D, for RBF-FD and GFDM; the 2-D RBF-FD Laplacian equals the existing one of meshless.methods."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from pinneapple_simulation.meshless.methods import rbf_fd_weights
from pinneapple_simulation.meshless.stencils import stencil_weights


@pytest.mark.parametrize("dim,degree,n", [(2, 2, 13), (3, 2, 30), (3, 3, 45)])
@pytest.mark.parametrize("method", ["rbf_fd", "gfdm"])
def test_exact_on_polynomials(dim, degree, n, method):
    rng = np.random.default_rng(0)
    X = rng.random((800, dim))
    Q = 0.3 + 0.4 * rng.random((20, dim))
    _, idx = cKDTree(X).query(Q, k=n)
    W = stencil_weights(X, Q, idx, ("eval", "d0", "d1", "lap"), method=method, degree=degree)
    a = rng.normal(size=dim)
    B = rng.normal(size=(dim, dim))
    u = X @ a + np.einsum("ni,ij,nj->n", X, B, X) if degree >= 2 else X @ a
    exact = {"eval": Q @ a + np.einsum("ni,ij,nj->n", Q, B, Q), "d0": a[0] + Q @ (B[0] + B[:, 0]),
             "d1": a[1] + Q @ (B[1] + B[:, 1]), "lap": np.full(len(Q), 2 * np.trace(B))}
    for op, w in W.items():
        assert np.allclose((w * u[idx]).sum(1), exact[op], atol=1e-7), op


def test_matches_the_2d_laplacian_of_meshless_methods():
    rng = np.random.default_rng(1)
    X = rng.random((500, 2))
    Q = 0.3 + 0.4 * rng.random((10, 2))
    _, idx = cKDTree(X).query(Q, k=31)
    a = stencil_weights(X, Q, idx, ("lap",), degree=4)["lap"]
    assert np.allclose(a, rbf_fd_weights(X, Q, idx, "lap", degree=4), rtol=1e-7, atol=1e-6)
