"""Library code of the condensation falsification campaign: algebraic condensation is exact for symmetric,
non-symmetric and saddle-point operators and finds identical blocks by content; the reduced faces converge to the
exact answer; the MAC flow solver passes its analytic checks; stack material overrides and anisotropy reach the
operator and the cache key; the meshless hybrid and meshless condensation run."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pinneapple_condense import algebraic as AL
from pinneapple_condense import flow as FL
from pinneapple_condense import meshless2d as M
from pinneapple_condense.falsify import _primal_pressure, _scalar_case, _stokes_system


def _poisson(n, k):
    N = n * n
    idx = np.arange(N).reshape(n, n)
    rows, cols, vals = [], [], []
    for di, dj in ((1, 0), (0, 1)):
        a, b = idx[:n - di, :n - dj].ravel(), idx[di:, dj:].ravel()
        kk = 2 / (1 / k.ravel()[a] + 1 / k.ravel()[b])
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        vals += [kk, kk, -kk, -kk]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A = A + 1e-3 * sp.eye(N)
    x, y = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    return A.tocsr(), np.column_stack([x.ravel(), y.ravel()])


def test_algebraic_condensation_is_exact_and_reuses_identical_blocks():
    B = 8
    cell = np.ones((B, B))
    cell[3:5, 3:5] = 50.0
    A, X = _poisson(6 * B, np.tile(cell, (6, 6)))
    b = np.random.default_rng(0).random(A.shape[0])
    c = AL.Condensation(A, AL.partition(A, X, (B, B))).prepare()
    x, _ = c.solve(b)
    xr = spla.spsolve(A.tocsc(), b)
    assert np.abs(x - xr).max() / np.abs(xr).max() < 1e-11
    # interior, four edges, four corners (by content): 9 distinct of 36
    assert c.timings["blocks"] == 36 and c.timings["distinct_blocks"] == 9


def test_reduced_faces_converge_to_exact_with_all_modes():
    B = 8
    A, X = _poisson(4 * B, np.ones((4 * B, 4 * B)))
    P = AL.partition(A, X, (B, B))
    c = AL.Condensation(A, P).prepare()
    rng = np.random.default_rng(1)
    snaps = [spla.spsolve(A.tocsc(), rng.random(A.shape[0])) for _ in range(12)]
    b = rng.random(A.shape[0])
    xr = spla.spsolve(A.tocsc(), b)
    fc = AL.faces(P, A)
    errs = []
    for m in (1, 2, 4, B):
        x, _ = AL.Reduced(c, fc, AL.face_modes(snaps, fc, m)).prepare().solve(b)
        errs.append(np.linalg.norm(x - xr) / np.linalg.norm(xr))
    assert errs[-1] < 1e-10 and errs[0] > errs[2]


def test_stokes_and_advection_condense_exactly():
    g, A, b = _stokes_system((2, 2), 8, 4)
    X, F = g.coords()
    P = AL.partition(A, X, (1.0, 1.0), fieldid=F, primal=_primal_pressure(g, 8))
    x, _ = AL.Condensation(A, P).prepare().solve(b)
    xr = FL.solve(A, b)
    assert np.abs(x - xr).max() / np.abs(xr).max() < 1e-9
    u, v, _ = FL.split(g, xr)
    As, bs = _scalar_case(2, 8, 4, 100.0, u, v, g)
    Xs = FL.scalar_coords(g)
    xs, _ = AL.Condensation(As, AL.partition(As, Xs, (1.0, 1.0))).prepare().solve(bs)
    assert np.abs(xs - FL.solve(As, bs)).max() / np.abs(xs).max() < 1e-10


def test_flow_solver_poiseuille_and_nusselt():
    ny = 16
    h, nx = 1.0 / ny, 12 * ny
    g = FL.Grid(nx, ny, h)
    yc = (np.arange(ny) + 0.5) * h
    A, b = FL.assemble_oseen(g, 0.1, 6 * yc * (1 - yc))
    u, v, p = FL.split(g, FL.solve(A, b))
    dpdx = (p[3 * nx // 4, ny // 2] - p[nx // 4, ny // 2]) / (nx / 2 * h)
    assert dpdx == pytest.approx(-1.2, rel=0.01)
    alpha = 0.1
    As, bs = FL.assemble_scalar(g, u, v, np.full((nx, ny), alpha), np.ones((nx, ny)), np.zeros((nx, ny)))
    bs = bs.reshape(nx, ny)
    bs[:, 0] += h
    T = FL.solve(As, bs.ravel()).reshape(nx, ny)
    i = int(0.8 * nx)
    uc = 0.5 * (u[i] + u[i + 1])
    nu_ = 2.0 / (alpha * (T[i, 0] + 0.5 * h / alpha - (uc * T[i]).sum() / uc.sum()))
    assert nu_ == pytest.approx(5.385, rel=0.01)


def test_stack_overrides_reach_the_operator_and_the_cache_key():
    from dataclasses import replace

    from pinneapple_condense.package import reference_stack
    st = reference_stack(n=4)
    lo = replace(st, materials=(("underfill", 0.05, 1.8e6),))
    an = replace(st, anisotropy=(("tim", 10.0),))
    assert repr(st) != repr(lo) != repr(an)
    p0, p1, p2 = st.properties(), lo.properties(), an.properties()
    assert p1["k"].min() == pytest.approx(0.05) and p0["k"].min() == pytest.approx(0.5)
    tim = st.layer_slices()["tim"]
    assert np.allclose(p2["k_lat"][..., tim], 10 * p2["k"][..., tim])
    assert np.array_equal(p0["k_lat"], p0["k"])


def test_meshless_hybrid_and_meshless_condensation_run():
    A, b, X, kind, _ = M.hybrid_conduction((2, 2), 8, 0.5, 400.0, 1.0, 1e-3)
    T = spla.spsolve(A.tocsc(), b)
    assert np.isfinite(T).all() and (kind == 0).any()
    A, b, X, kind, _ = M.hybrid_conduction((4, 4), 8, 0.5, 400.0, 1.0, 1e-3)
    c = AL.Condensation(A.tocsr(), AL.partition(A, X, (1.0, 1.0))).prepare()
    x, _ = c.solve(b)
    assert np.abs(x - spla.spsolve(A.tocsc(), b)).max() / np.abs(x).max() < 1e-9
    assert c.timings["distinct_blocks"] < c.timings["blocks"]        # the four interior blocks are one
