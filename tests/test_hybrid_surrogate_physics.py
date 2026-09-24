"""Hybrid surrogate + explicit physics post-model, on a synthetic case with known physics.

Parameter x in [1, 3]; intermediate quantity = velocity V(x) = 2 + 3 sin(x) (what a surrogate learns from
"simulations"); derived quantity = wear rate K * V**n (the explicit physics post-model).
"""
import numpy as np
import pytest

from pinneapple_neural.workflows.hybrid_surrogate_physics import (
    CallablePostModel,
    GaussianProcessSurrogate,
    HybridSurrogatePhysics,
    PCASurrogate,
    SurrogatePrediction,
    TransformedSurrogate,
)


def velocity(x):
    return 2.0 + 3.0 * np.sin(x)


def power_law(K, n, name):
    return CallablePostModel(lambda inter, p: K * inter["velocity"] ** n, name=name, source=f"synthetic K*V^{n}")


@pytest.fixture(scope="module", params=["sklearn", "numpy"])
def fitted(request):
    if request.param == "sklearn":
        pytest.importorskip("sklearn")
    x = np.linspace(1.0, 3.0, 12)
    hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(use_sklearn=request.param == "sklearn"),
                                 power_law(1e-3, 2.6, "dnv_like"), param_names=["x"])
    hyb.fit({"x": x}, {"velocity": velocity(x)})
    return hyb


def test_prediction_matches_known_physics(fitted):
    xq = np.array([1.37, 2.11, 2.83])
    out = fitted.predict({"x": xq}, n_samples=4000)
    assert np.allclose(out.intermediates["velocity"], velocity(xq), rtol=1e-2)
    truth = 1e-3 * velocity(xq) ** 2.6
    assert np.allclose(out.derived_at_mean, truth, rtol=3e-2)
    assert out.uncertainty == "monte_carlo" and out.n_samples == 4000
    assert out.post_model == {"name": "dnv_like", "source": "synthetic K*V^2.6"}


def test_uncertainty_band_covers_truth_plausibly():
    """Noisy observations, few points: the MC band of K*V^n must cover the truth at a plausible rate."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    x = rng.uniform(1.0, 3.0, 10)
    v_obs = velocity(x) + rng.normal(0.0, 0.15, x.size)
    hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(), power_law(1e-3, 2.6, "dnv_like"), param_names=["x"])
    hyb.fit({"x": x}, {"velocity": v_obs})
    xq = np.linspace(1.05, 2.95, 60)
    out = hyb.predict({"x": xq}, n_samples=3000, interval=0.95, seed=1)
    truth = 1e-3 * velocity(xq) ** 2.6
    cover = np.mean((truth >= out.derived_lower) & (truth <= out.derived_upper))
    assert 0.7 <= cover <= 1.0, cover
    assert np.all(out.derived_upper > out.derived_lower)
    # nonlinear physics: the MC mean differs from the post-model at the surrogate mean, both are reported
    assert not np.allclose(out.derived_mean, out.derived_at_mean)


def test_swapping_the_post_model_does_not_retrain(fitted):
    xq = np.array([1.5, 2.5])
    a = fitted.predict({"x": xq}, n_samples=500, seed=0)
    other = fitted.with_post_model(power_law(2e-3, 2.41, "ecrc_like"))
    assert other.surrogate is fitted.surrogate                        # same fitted object, nothing refit
    b = other.predict({"x": xq}, n_samples=500, seed=0)
    assert np.array_equal(a.intermediates["velocity"], b.intermediates["velocity"])
    assert np.allclose(b.derived_at_mean, 2e-3 * a.intermediates["velocity"] ** 2.41)
    assert not np.allclose(a.derived_at_mean, b.derived_at_mean)
    assert fitted.post_model.name == "dnv_like"                       # the original is untouched


def test_state_roundtrip_with_another_post_model(fitted):
    import pickle

    state = pickle.loads(pickle.dumps(fitted.state()))
    back = HybridSurrogatePhysics.from_state(state, power_law(2e-3, 2.41, "ecrc_like"))
    xq = np.array([1.2, 2.2])
    v = fitted.predict({"x": xq}, n_samples=0).intermediates["velocity"]
    assert np.allclose(back.predict({"x": xq}, n_samples=0).derived_at_mean, 2e-3 * v ** 2.41)


def test_no_distribution_means_no_band():
    class PointOnly:
        def fit(self, X, Y):
            self.c = Y.mean(axis=0)
            return self

        def predict(self, X):
            return np.tile(self.c, (len(X), 1))

    hyb = HybridSurrogatePhysics(PointOnly(), power_law(1.0, 1.0, "id")).fit([[1.0], [2.0]], {"velocity": [1.0, 3.0]})
    out = hyb.predict([[1.5]], n_samples=100)
    assert out.uncertainty == "none" and out.derived_lower is None and out.derived_at_mean[0] == pytest.approx(2.0)


def test_vector_intermediates_with_pca_and_transform():
    """Histogram-like vector intermediate (positive, on a simplex) + scalar flux; derived = flux * <w, p>."""
    pytest.importorskip("sklearn")
    centers = np.linspace(0.05, 0.95, 10)

    def hist(x):
        p = np.exp(-0.5 * ((centers[None] - 0.3 - 0.2 * x[:, None]) / 0.12) ** 2)
        return p / p.sum(axis=1, keepdims=True)

    x = np.linspace(0.0, 2.0, 15)
    w = centers ** 2.6
    post = CallablePostModel(lambda i, p: i["flux"] * (i["pdf"] @ w), name="weighted", source="synthetic")
    pca = TransformedSurrogate(PCASurrogate(GaussianProcessSurrogate(), variance=0.999), np.sqrt,
                               lambda z: np.maximum(z, 0.0) ** 2, name="sqrt")
    # one surrogate for the flattened [pdf | flux] vector: flux is also positive, sqrt/square is fine
    hyb = HybridSurrogatePhysics(pca, post, param_names=["x"])
    flux = 1.0 + x
    hyb.fit({"x": x}, {"pdf": hist(x), "flux": flux})
    xq = np.array([0.33, 1.21])
    out = hyb.predict({"x": xq}, n_samples=500)
    truth = (1.0 + xq) * (hist(xq) @ w)
    assert out.intermediates["pdf"].shape == (2, 10) and out.intermediates["flux"].shape == (2,)
    assert np.allclose(out.derived_at_mean, truth, rtol=5e-2)
    assert pca.base.n_components >= 1 and pca.base.explained_variance >= 0.999


def test_one_surrogate_per_quantity():
    """PCA on sqrt(histogram) for the pdf, log-GP for the positive flux; derived = flux * <w, pdf>."""
    pytest.importorskip("sklearn")
    centers = np.linspace(0.05, 0.95, 10)
    w = centers ** 2.6

    def hist(x):
        p = np.exp(-0.5 * ((centers[None] - 0.3 - 0.2 * x[:, None]) / 0.12) ** 2)
        return p / p.sum(axis=1, keepdims=True)

    def square_normalized(z):
        q = np.maximum(z, 0.0) ** 2
        return q / q.sum(axis=-1, keepdims=True)

    x = np.linspace(0.0, 2.0, 15)
    post = CallablePostModel(lambda i, p: i["flux"] * (i["pdf"] @ w), name="weighted", source="synthetic")
    hyb = HybridSurrogatePhysics(
        {"pdf": TransformedSurrogate(PCASurrogate(GaussianProcessSurrogate(), n_components=3), np.sqrt,
                                     square_normalized),
         "flux": TransformedSurrogate(GaussianProcessSurrogate(), np.log, np.exp)},
        post, param_names=["x"])
    hyb.fit({"x": x}, {"pdf": hist(x), "flux": np.exp(0.5 * x)})
    xq = np.array([0.33, 1.21])
    out = hyb.predict({"x": xq}, n_samples=400)
    assert np.allclose(out.intermediates["pdf"].sum(axis=1), 1.0)
    assert np.allclose(out.derived_at_mean, np.exp(0.5 * xq) * (hist(xq) @ w), rtol=5e-2)
    assert out.uncertainty == "monte_carlo" and np.all(out.derived_lower <= out.derived_upper)
    swapped = hyb.with_post_model(CallablePostModel(lambda i, p: i["flux"], name="flux_only", source="synthetic"))
    assert np.allclose(swapped.predict({"x": xq}, n_samples=0).derived_at_mean, out.intermediates["flux"])
    with pytest.raises(KeyError):
        HybridSurrogatePhysics({"pdf": GaussianProcessSurrogate()}, post).fit({"x": x}, {"pdf": hist(x),
                                                                                           "flux": x})


def test_post_model_must_declare_name_and_source():
    with pytest.raises(ValueError):
        CallablePostModel(lambda i, p: i["v"], name="x", source="")

    class Anonymous:
        name, source = "", ""

        def __call__(self, i, p):
            return i["v"]

    with pytest.raises(ValueError):
        HybridSurrogatePhysics(GaussianProcessSurrogate(), Anonymous())


def test_predict_before_fit_and_shape_errors():
    hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(use_sklearn=False), power_law(1.0, 1.0, "id"))
    with pytest.raises(RuntimeError):
        hyb.predict([[1.0]])
    with pytest.raises(ValueError):
        hyb.fit([[1.0], [2.0]], {"velocity": [1.0, 2.0, 3.0]})


def test_surrogate_prediction_sampling():
    p = SurrogatePrediction(np.zeros((2, 3)), np.ones((2, 3)))
    s = p.sample(5000, np.random.default_rng(0))
    assert s.shape == (5000, 2, 3) and abs(s.std() - 1.0) < 0.05
    with pytest.raises(ValueError):
        SurrogatePrediction(np.zeros((1, 1))).sample(3, np.random.default_rng(0))
