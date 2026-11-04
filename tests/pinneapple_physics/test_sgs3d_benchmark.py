"""Tests for the 3-D a-priori subgrid benchmark (``multiscale.sgs3d``).

The benchmark's job is to measure closures, so the measurement itself has to be checked first: the spectral
derivatives against analytic ones, the filter against a field whose product stays inside the cut-off (where the
subfilter stress is exactly zero), the metrics against cases with a known answer (a perfect model, an inverted
one, a doubled one), and only then the closures and the end-to-end run.
"""
import numpy as np
import pytest

from pinneapple_physics.multiscale.sgs3d import (
    AprioriCase,
    apriori_case,
    apriori_metrics,
    closure_eddy_viscosity,
    default_closures,
    ideal_eddy_viscosity,
    learn_sgs_stress,
    modelled_stress,
    periodic_stencil,
    run_benchmark,
    sharp_filter,
    spectral_gradient,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid(n: int, length: float = 2 * np.pi):
    x = np.arange(n) * length / n
    return np.meshgrid(x, x, x, indexing="ij")


def _random_field(n: int, kmax: float, seed: int = 0) -> np.ndarray:
    """Divergence-free random field with modes ``|m_i| <= kmax`` (in mode indices)."""
    rng = np.random.default_rng(seed)
    m = np.fft.fftfreq(n) * n
    keep = (np.abs(m) <= kmax + 1e-9)
    mask = keep[:, None, None] & keep[None, :, None] & keep[None, None, :]
    k = np.stack(np.meshgrid(m, m, m, indexing="ij"))

    u = np.empty((3, n, n, n))
    for i in range(3):
        uh = (rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))) * mask
        u[i] = np.fft.ifftn(uh).real
    # project out the compressive part (in Fourier space), keeping the band
    uh = np.fft.fftn(u, axes=(-3, -2, -1))
    k2 = (k ** 2).sum(0)
    k2i = np.where(k2 > 0, 1.0 / np.maximum(k2, 1e-30), 0.0)
    uh = uh - k * (k * uh).sum(0) * k2i
    return np.fft.ifftn(uh, axes=(-3, -2, -1)).real


# ---------------------------------------------------------------------------
# spectral derivatives and the filter
# ---------------------------------------------------------------------------

def test_spectral_gradient_is_exact_on_an_analytic_periodic_field():
    n = 8
    X, Y, Z = _grid(n)
    u = np.stack([np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), np.sin(Z)])
    g = spectral_gradient(u)

    assert np.abs(g[0, 0] - np.cos(X) * np.cos(Y)).max() < 1e-12
    assert np.abs(g[0, 1] + np.sin(X) * np.sin(Y)).max() < 1e-12
    assert np.abs(g[1, 1] + np.cos(X) * np.cos(Y)).max() < 1e-12
    assert np.abs(g[2, 2] - np.cos(Z)).max() < 1e-12
    # second component of that field is independent of z, first of z: those derivatives are exactly 0
    assert np.abs(g[0, 2]).max() < 1e-12


def test_sharp_filter_keeps_the_modes_below_the_cutoff_and_drops_the_rest():
    n = 16
    X, Y, Z = _grid(n)
    low = np.sin(X)                      # mode 1
    high = np.sin(5 * X)                 # mode 5
    field = low + high

    filtered = sharp_filter(field, kc=3.0)
    assert np.abs(filtered - low).max() < 1e-12, "modes below the cut-off must pass untouched"
    assert np.abs(sharp_filter(field, kc=1.0) - low).max() < 1e-12

    dropped = field - filtered
    assert np.abs(dropped - high).max() < 1e-12, "modes above the cut-off must be removed exactly"


def test_subfilter_stress_is_zero_when_the_product_stays_inside_the_cutoff():
    """A field whose *product* still fits under the cut-off has nothing for the filter to remove, so the true
    subfilter stress must be exactly zero. This is the one case where the answer is known analytically, and it
    checks the whole chain: filter, product, restriction, and the aliasing-free padding."""
    n = 32
    u_fine = _random_field(n, kmax=2.5, seed=3)        # product modes <= 5 < kc = 16/3 on the coarse grid
    case = apriori_case(u_fine, n_coarse=16)

    scale = float(np.abs(case.u_bar).max()) ** 2
    assert scale > 1e-8, "the test field should be non-trivial"
    assert np.abs(case.tau).max() < 1e-10 * scale, np.abs(case.tau).max()
    assert np.abs(case.transfer()).max() < 1e-10 * scale


def test_uniform_flow_has_no_subfilter_stress():
    n = 16
    u_fine = np.broadcast_to(np.array([1.0, -0.5, 0.25])[:, None, None, None], (3, n, n, n)).copy()
    case = apriori_case(u_fine, n_coarse=8)
    assert np.abs(case.tau).max() < 1e-14
    assert np.abs(case.S_bar).max() < 1e-14


# ---------------------------------------------------------------------------
# the grid stencil used for the dynamic model's test filter
# ---------------------------------------------------------------------------

def test_periodic_stencil_is_normalised_includes_the_point_and_wraps_around():
    n = 4
    idx, w = periodic_stencil(n, radius=1, kernel="box")
    assert idx.shape == (n ** 3, 27)
    assert w.shape == (n ** 3, 27)
    assert np.allclose(w.sum(axis=1), 1.0)

    # the middle neighbour of a 3x3x3 box centred on a point is the point itself
    assert all(i in set(row.tolist()) for i, row in enumerate(idx)), \
        "each point must appear in its own stencil"

    # wrapping: the point at (0,0,0) in a periodic box has (-1,-1,-1) at (n-1, n-1, n-1)
    corner = ((n - 1) * n * n + (n - 1) * n + (n - 1))
    assert corner in set(idx[0].tolist())

    with pytest.raises(ValueError, match="radius"):
        periodic_stencil(n, radius=0)
    with pytest.raises(ValueError, match="kernel"):
        periodic_stencil(n, kernel="triangular")


# ---------------------------------------------------------------------------
# metrics: known answers
# ---------------------------------------------------------------------------

def test_apriori_metrics_recognise_a_perfect_inverted_and_doubled_model():
    n = 8
    case = apriori_case(_random_field(16, kmax=2, seed=1), n_coarse=n)

    perfect = apriori_metrics(case.tau, case)
    assert perfect["corr"] == pytest.approx(1.0, abs=1e-12)
    assert perfect["mag_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert perfect["transfer_corr"] == pytest.approx(1.0, abs=1e-12)

    inverted = apriori_metrics(-case.tau, case)
    assert inverted["corr"] == pytest.approx(-1.0, abs=1e-12)
    assert inverted["mag_ratio"] == pytest.approx(1.0, abs=1e-12)

    doubled = apriori_metrics(2.0 * case.tau, case)
    assert doubled["corr"] == pytest.approx(1.0, abs=1e-12)
    assert doubled["mag_ratio"] == pytest.approx(2.0, abs=1e-12)

    for key in ("corr_components", "nu_ideal_mean", "nu_ideal_negative_fraction", "backscatter_fraction"):
        assert key in perfect


def test_ideal_eddy_viscosity_recovers_a_known_coefficient():
    """If the true stress *is* an eddy-viscosity stress with a known coefficient, the least-squares viscosity has
    to give that coefficient back pointwise."""
    n = 8
    case = apriori_case(_random_field(16, kmax=2, seed=2), n_coarse=n)
    nu = 0.03
    synthetic = AprioriCase(
        u_bar=case.u_bar, S_bar=case.S_bar, g_bar=case.g_bar,
        tau=-2.0 * nu * case.S_bar, kc=case.kc, delta=case.delta, n_coarse=case.n_coarse,
    )
    assert np.allclose(ideal_eddy_viscosity(synthetic), nu, rtol=1e-10, atol=0)

    metrics = apriori_metrics(modelled_stress(np.full(case.S_bar.shape[2:], nu), case.S_bar), synthetic)
    assert metrics["corr"] == pytest.approx(1.0, abs=1e-10)
    assert metrics["mag_ratio"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# the closures, evaluated through the benchmark
# ---------------------------------------------------------------------------

def test_smagorinsky_viscosity_on_the_benchmark_matches_its_own_definition():
    """The benchmark feeds the closure the gradient and strain it computed spectrally; the value it gets back
    must still be ``(Cs Delta)^2 |S|`` computed from those same numbers -- i.e. the plumbing (flattening order,
    grid spacing, delta) did not quietly rescale anything."""
    n = 8
    case = apriori_case(_random_field(16, kmax=2, seed=4), n_coarse=n)
    smag = default_closures(case.delta, n)["Smagorinsky"]
    nu = closure_eddy_viscosity(smag, case)

    magS = np.sqrt(2.0 * (case.S_bar * case.S_bar).sum(axis=(0, 1)))
    assert np.allclose(nu, (smag.cs * case.delta) ** 2 * magS, rtol=1e-10, atol=0)


def test_wale_viscosity_vanishes_where_the_resolved_field_has_no_strain():
    """WALE is built to switch off in a strain-free flow; on the uniform field it must return exactly zero, while
    a strain-based closure on the same flow also returns zero -- and neither may produce NaN."""
    n = 8
    u_fine = np.broadcast_to(np.array([0.3, -0.2, 0.1])[:, None, None, None], (3, 16, 16, 16)).copy()
    case = apriori_case(u_fine, n_coarse=n)
    for name, closure in default_closures(case.delta, n).items():
        nu = closure_eddy_viscosity(closure, case)
        assert np.isfinite(nu).all(), f"{name} is not finite on a uniform flow"
        assert np.abs(nu).max() < 1e-20, f"{name} should be zero on a strain-free flow, got {np.abs(nu).max()}"


# ---------------------------------------------------------------------------
# the learned closure
# ---------------------------------------------------------------------------

def test_learned_closure_is_scored_on_the_snapshot_it_did_not_train_on():
    n = 8
    train = apriori_case(_random_field(16, kmax=n / 3, seed=5), n_coarse=n, label="train")
    test = apriori_case(_random_field(16, kmax=n / 3, seed=6), n_coarse=n, label="test")
    out = learn_sgs_stress(train, test, steps=50, hidden=8, seed=0)

    assert out["tau"].shape == test.tau.shape
    assert np.isfinite(out["tau"]).all()
    assert out["metrics"]["test_points"] == n ** 3
    assert out["metrics"]["train_points"] == n ** 3
    assert -1.0 <= out["metrics"]["corr"] <= 1.0
    # determinism: same seed, same closure
    again = learn_sgs_stress(train, test, steps=50, hidden=8, seed=0)
    assert np.array_equal(out["tau"], again["tau"])


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_run_benchmark_end_to_end_scores_every_closure_and_reports_its_settings():
    report = run_benchmark(n=16, re=1600.0, t_train=(0.1, 0.2), t_test=0.4, dt=0.01, learn_steps=20)

    assert set(report["table"]) == {"GradientModel", "Smagorinsky", "DynamicSmagorinsky", "WALE", "Learned"}
    for name, metrics in report["table"].items():
        for key in ("corr", "mag_ratio", "transfer_corr", "corr_components"):
            assert key in metrics, f"{name} is missing {key}"
        assert np.isfinite(metrics["corr"]), name
        assert np.isfinite(metrics["mag_ratio"]), name

    settings = report["settings"]
    assert settings["n_coarse"] == 8
    assert settings["kc"] == pytest.approx(8 / 3)
    assert settings["delta"] == pytest.approx(np.pi / (8 / 3))
    assert settings["t_train"] == [0.1, 0.2]
    assert len(report["case_train"]) == 2
    assert report["case_test"].label == "t=0.40"
    assert report["table"]["Learned"]["train_snapshots"] == 2


def test_learn_sgs_stress_accepts_several_training_snapshots():
    """The learned closure is trained on a *set* of snapshots, and the metrics say so."""
    n = 8
    train = [apriori_case(_random_field(16, kmax=2, seed=s), n_coarse=n) for s in (5, 6, 7)]
    test = apriori_case(_random_field(16, kmax=2, seed=9), n_coarse=n)
    out = learn_sgs_stress(train, test, steps=20, hidden=8, seed=0)
    assert out["metrics"]["train_snapshots"] == 3
    assert out["metrics"]["train_points"] == 3 * n ** 3
    assert out["tau"].shape == test.tau.shape


def test_run_benchmark_rejects_an_unknown_closure():
    with pytest.raises(ValueError, match="unknown closures"):
        run_benchmark(n=16, t_train=0.1, t_test=0.2, dt=0.01, closures=["Kolmogorov"], learn=False)
