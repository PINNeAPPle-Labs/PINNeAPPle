"""The common ``PhysicsModel`` interface over the five families (issue #155).

The contract is the same four calls for every family. These tests hold the parts of it that can be measured:

* every family that the library ships is wrapped by the factory, and an object outside the families is refused
  with a message that names them;
* ``fit`` reports the *kind* of fit it was (closed form for a ROM, gradient with a loss curve for the rest) and the
  gradient ones actually reduce the training loss — measured, on the epoch history;
* ``predict`` returns the same numbers the family's own forward returns (the adapter is a calling convention, not a
  reimplementation), with the shape the family documents;
* ``uncertainty`` either returns a spread with the reason recorded, or refuses and says what is missing. It is
  never a fabricated zero: a model with no dropout-able hidden layer is refused, which is the case of the FNO;
* ``export`` writes a file that reloads and reproduces the prediction (TorchScript), and a format that cannot be
  produced says so instead of writing a file that looks exported.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.architectures.graphnn.base import GraphBatch  # noqa: E402
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet  # noqa: E402
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet  # noqa: E402
from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FourierNeuralOperator  # noqa: E402
from pinneapple_neural.architectures.pinns.vanilla import VanillaPINN  # noqa: E402
from pinneapple_neural.architectures.rom.pod import POD  # noqa: E402
from pinneapple_neural.architectures.rom.rom_hybrid import ROMHybrid  # noqa: E402
from pinneapple_neural.physics_model import (  # noqa: E402
    FAMILIES,
    ExportNotAvailable,
    FitReport,
    GraphModel,
    HybridModel,
    NotAPhysicsModel,
    OperatorModel,
    PhysicsModel,
    PINNModel,
    Prediction,
    ROMModel,
    UncertaintyNotAvailable,
    as_physics_model,
    family_of,
)
from pinneapple_neural.workflows.hybrid_surrogate_physics import (  # noqa: E402
    CallablePostModel,
    GaussianProcessSurrogate,
    HybridSurrogatePhysics,
)


def _post_model():
    """A post-model with a name and a source, which the workflow requires (a physics model says what it is).

    It keeps the per-point shape of its inputs: ``rho`` arrives as ``(n, 1)`` and ``v`` as ``(n,)``, so the
    density is flattened before the product — otherwise the two broadcast into an ``(n, n)`` matrix.
    """
    return CallablePostModel(lambda intermediates, params: 0.5 * params["v"][:, None] ** 2 * intermediates["rho"],
                             name="dynamic pressure", source="Bernoulli: q = rho v^2 / 2")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ------------------------------------------------------------------------------------------- the five families
def _pinn():
    return VanillaPINN(2, 1, hidden=[16, 16])


def _fno():
    return FourierNeuralOperator(1, 1, width=8, modes=4, layers=1)


def _fno2d():
    return FNO2d(1, 1, width=8, modes1=4, modes2=4, layers=1)


def _deeponet():
    return DeepONet(branch_dim=6, trunk_dim=2, out_dim=1, hidden=8, modes=4)


def _mgn():
    return MeshGraphNet(node_in_dim=4, out_dim=2)


def _pod():
    return POD(r=3)


def _rom_hybrid():
    return ROMHybrid(field_dim=10, r=3, use_corrector=False)


def _hybrid():
    hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(use_sklearn=False), _post_model())
    rng = np.random.default_rng(0)
    params = {"v": rng.uniform(1.0, 3.0, 12)}
    rho = 0.4 * params["v"] + 0.05 * rng.standard_normal(12)
    return hyb.fit(params, {"rho": rho[:, None]})


# ---------------------------------------------------------------------------------------------- the factory
@pytest.mark.parametrize("build, expected", [
    (_pinn, "pinn"), (_fno, "operator"), (_fno2d, "operator"), (_deeponet, "operator"),
    (_mgn, "gnn"), (_pod, "rom"), (_rom_hybrid, "rom"), (_hybrid, "hybrid"),
])
def test_every_family_is_wrapped_by_the_factory(build, expected):
    model = build()
    assert family_of(model) == expected
    wrapped = as_physics_model(model)
    assert isinstance(wrapped, PhysicsModel) and wrapped.family == expected
    assert wrapped.model is model and wrapped.name and wrapped.n_params >= 0
    assert set(wrapped.capabilities()) == {"fit", "predict", "uncertainty", "export"}


def test_the_five_families_the_issue_names_are_the_ones_covered():
    assert FAMILIES == ("pinn", "operator", "gnn", "rom", "hybrid")


def test_an_object_outside_the_families_is_refused_and_named():
    with pytest.raises(NotAPhysicsModel, match="pinn, operator, gnn, rom, hybrid"):
        family_of(torch.nn.Linear(2, 1))
    with pytest.raises(NotAPhysicsModel):
        as_physics_model(object())


def test_describe_carries_the_same_fields_for_every_family():
    for build in (_pinn, _fno, _mgn, _pod):
        d = as_physics_model(build()).describe()
        assert set(d) == {"name", "family", "backend", "n_params", "class", "capabilities"}, d
        assert d["backend"] == "pytorch"


def test_the_adapter_class_matches_the_family():
    assert isinstance(as_physics_model(_pinn()), PINNModel)
    assert isinstance(as_physics_model(_fno()), OperatorModel)
    assert isinstance(as_physics_model(_mgn()), GraphModel)
    assert isinstance(as_physics_model(_pod()), ROMModel)
    assert isinstance(as_physics_model(_hybrid()), HybridModel)


# ------------------------------------------------------------------------------------------------ predict
def test_pinn_predict_returns_what_forward_returns():
    m = _pinn()
    wrapped = as_physics_model(m)
    x = torch.randn(7, 2)
    with torch.no_grad():
        expected = m(x).y
    got = wrapped.predict(x)
    assert isinstance(got, Prediction) and not got.has_uncertainty
    assert torch.allclose(got.mean, expected, atol=1e-6)


def test_operator_predict_matches_the_grid_forward():
    m = _fno2d().eval()
    u = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        expected = m(u).y
    assert torch.allclose(as_physics_model(m).predict(u).mean, expected, atol=1e-6)


def test_deeponet_binds_its_query_points_as_a_constant():
    """A point operator needs coordinates on every call; they are bound once, not passed through a special case."""
    m = _deeponet().eval()
    u, coords = torch.randn(3, 6), torch.rand(5, 2)
    with torch.no_grad():
        expected = m(u, coords).y
    wrapped = as_physics_model(m, coords=coords)
    assert torch.allclose(wrapped.predict(u).mean, expected, atol=1e-6)


def test_gnn_predict_takes_a_graph_batch_and_a_mapping():
    m = _mgn().eval()
    g = GraphBatch(x=torch.randn(2, 10, 4), edge_index=torch.randint(0, 10, (2, 20)))
    wrapped = as_physics_model(m)
    with torch.no_grad():
        expected = m(g).y
    assert torch.allclose(wrapped.predict(g).mean, expected, atol=1e-6)
    as_mapping = wrapped.predict({"x": g.x, "edge_index": g.edge_index})
    assert torch.allclose(as_mapping.mean, expected, atol=1e-6)


def test_a_graph_batch_without_edges_is_refused():
    with pytest.raises(KeyError, match="edge_index"):
        as_physics_model(_mgn().eval()).predict({"x": torch.randn(1, 5, 4)})


def test_rom_predict_reconstructs_the_snapshots_after_a_closed_form_fit():
    X = torch.randn(6, 8, 10)
    wrapped = as_physics_model(POD(r=10))
    report = wrapped.fit(X)
    assert report.method == "closed_form"
    y = wrapped.predict(X).mean
    assert y.shape == X.shape
    rel = float(torch.linalg.norm(y - X) / torch.linalg.norm(X))
    assert rel < 1e-4, f"full-rank POD should reconstruct exactly, relative error {rel:.3e}"


# ---------------------------------------------------------------------------------------------------- fit
def test_a_gradient_fit_reports_its_history_and_reduces_the_loss():
    """Measured on the epoch history: the adapter trains for real, it does not just record that it was called.

    The threshold is the ratio this problem reaches at these settings on a laptop (about 13x over 150 epochs, from
    a first-epoch loss near 0.5 to about 0.037), not a target: if a refactor stopped the loop from training, the
    ratio would sit at 1.
    """
    wrapped = as_physics_model(_pinn())
    x = torch.rand(128, 2)
    y = torch.sin(3 * x[:, :1])
    report = wrapped.fit({"x": x, "y": y}, epochs=150, lr=1e-2)
    assert isinstance(report, FitReport) and report.method == "gradient"
    assert report.epochs == 150 and len(report.history) == report.epochs
    ratio = report.improved()
    assert ratio is not None and ratio > 8.0, f"loss fell only {ratio}x: {report.summary()}"
    assert report.final_loss < 0.1 and report.n_params > 0 and report.seconds > 0


def test_a_gradient_fit_accepts_a_list_of_batches_and_an_xy_pair():
    wrapped = as_physics_model(_pinn())
    x, y = torch.rand(16, 2), torch.rand(16, 1)
    assert wrapped.fit([(x, y), {"x": x, "y": y}], epochs=5).epochs == 5


def test_the_default_loss_names_the_missing_target():
    with pytest.raises(KeyError, match="'y'"):
        as_physics_model(_pinn()).fit({"x": torch.rand(4, 2)}, epochs=1)


def test_an_empty_training_payload_is_refused():
    with pytest.raises(ValueError, match="empty"):
        as_physics_model(_pinn()).fit([], epochs=1)


def test_a_graph_fit_runs_and_reduces_its_loss():
    m = _mgn()
    g = {"x": torch.randn(2, 8, 4), "edge_index": torch.randint(0, 8, (2, 16)), "y": torch.randn(2, 8, 2)}
    report = as_physics_model(m).fit([g], epochs=40, lr=1e-2)
    assert report.method == "gradient" and report.epochs == 40
    first = report.history[0]["train_total"]
    assert report.final_loss < first, f"graph fit did not improve: {first} -> {report.final_loss}"


def test_a_rom_fit_is_closed_form_and_says_so():
    report = as_physics_model(POD(r=3)).fit(torch.randn(20, 10))
    assert report.method == "closed_form" and report.epochs == 0 and report.history == []
    assert report.final_loss is None and "closed-form" in report.summary()
    assert "20, 10" in report.notes, "the report says what it was fitted on"


def test_a_rom_fit_refuses_something_that_is_not_snapshots():
    with pytest.raises(TypeError, match="snapshot tensor"):
        as_physics_model(POD(r=3)).fit({"x": [[1.0, 2.0]]})


def test_a_hybrid_fit_takes_params_and_intermediates_and_never_trains_the_post_model():
    post_calls = []

    def counting(intermediates, params):
        post_calls.append(1)
        return intermediates["rho"] * params["v"]

    post = CallablePostModel(counting, name="rho v", source="momentum flux")
    hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(use_sklearn=False), post)
    rng = np.random.default_rng(1)
    params = {"v": rng.uniform(1.0, 2.0, 10)}
    data = {"params": params, "intermediates": {"rho": rng.uniform(0.1, 0.5, (10, 1))}}
    report = as_physics_model(hyb).fit(data)
    assert report.method == "closed_form" and report.n_params == 0
    assert post_calls == [], "the post-model is never trained"                 # the point of keeping it explicit
    with pytest.raises(TypeError, match="intermediates"):
        as_physics_model(hyb).fit({"params": params})


# ---------------------------------------------------------------------------------------------- uncertainty
def test_mc_dropout_on_a_pinn_gives_a_real_spread_around_the_prediction():
    """Measured: the spread is non-zero, and the deterministic prediction lies inside the band it reports. The
    sample mean is not the deterministic prediction (dropout is on for the samples, which is the method), so the
    honest comparison is coverage of the point estimate, not equality with it."""
    wrapped = as_physics_model(_pinn())
    x = torch.randn(5, 2)
    point = wrapped.predict(x).mean
    band = wrapped.uncertainty(x, method="mc_dropout", n_samples=200, seed=3)
    assert band.method == "mc_dropout" and band.has_uncertainty
    assert band.std.shape == point.shape and float(band.std.max()) > 0.0
    low, high = band.interval(0.95)
    assert bool(((low <= point) & (point <= high)).all()), "the deterministic prediction falls outside its own band"
    assert bool((low <= band.mean).all() and (band.mean <= high).all())


def test_a_model_with_no_dropout_able_layer_is_refused_with_the_measured_reason():
    """The FNO has no ``nn.Linear`` at all: MC dropout over it returns an exactly zero spread. Measured, then
    refused — reporting '0 ± 0' would be a fabricated uncertainty."""
    m = _fno2d().eval()
    wrapped = as_physics_model(m)
    hidden, dropouts = 0, 0
    for module in m.modules():
        if isinstance(module, torch.nn.Linear):
            hidden += 1
        if isinstance(module, torch.nn.Dropout):
            dropouts += 1
    assert hidden == 0 and dropouts == 0, "this test states the FNO's layer composition; re-measure if it changes"
    assert wrapped.capabilities()["uncertainty"].startswith("no")
    with pytest.raises(UncertaintyNotAvailable, match="0 hidden nn.Linear and 0 nn.Dropout"):
        wrapped.uncertainty(torch.randn(2, 1, 16, 16), n_samples=5)


def test_a_pinn_reports_uncertainty_as_available_in_capabilities():
    assert as_physics_model(_pinn()).capabilities()["uncertainty"] == "yes"


def test_ensemble_uncertainty_measures_the_disagreement_between_members():
    members = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = _pinn()
        with torch.no_grad():                                  # three different fits of the same architecture
            m.net[-1].bias.add_(float(seed))
        members.append(m)
    x = torch.randn(6, 2)
    wrapped = as_physics_model(members[0])
    band = wrapped.uncertainty(x, method="ensemble", members=members)
    assert band.method == "ensemble" and band.extras["members"] == 3
    singles = torch.stack([wrapped.predict(x).mean] + [as_physics_model(m).predict(x).mean for m in members[1:]])
    assert torch.allclose(band.mean, singles.mean(dim=0), atol=1e-6)
    assert torch.allclose(band.std, singles.std(dim=0, unbiased=True), atol=1e-6)
    assert float(band.std.max()) > 0.0


def test_ensemble_needs_at_least_two_members():
    with pytest.raises(UncertaintyNotAvailable, match="at least two"):
        as_physics_model(_pinn()).uncertainty(torch.randn(3, 2), method="ensemble", members=[_pinn()])


def test_a_rom_refuses_mc_dropout_and_explains_the_closed_form():
    wrapped = as_physics_model(POD(r=3))
    wrapped.fit(torch.randn(20, 10))
    assert wrapped.capabilities()["uncertainty"].startswith("no")
    with pytest.raises(UncertaintyNotAvailable, match="closed-form projection"):
        wrapped.uncertainty(torch.randn(4, 10), n_samples=5)


def test_an_unknown_uncertainty_method_is_refused():
    with pytest.raises(ValueError, match="unknown method"):
        as_physics_model(_pinn()).uncertainty(torch.randn(3, 2), method="magic")


def test_the_hybrid_band_is_the_one_pushed_through_the_post_model():
    hyb = _hybrid()
    wrapped = as_physics_model(hyb)
    params = {"v": np.array([1.5, 2.5])}
    point = wrapped.predict(params)
    band = wrapped.uncertainty(params, n_samples=400, seed=0)
    assert band.method == "monte_carlo" and band.has_uncertainty
    assert band.std.shape == point.mean.shape and band.extras["n_samples"] == 400
    # the plug-in value at the intermediate mean is not the mean of the derived quantity (post-model is nonlinear)
    assert not torch.allclose(band.mean, point.mean, atol=1e-6)
    assert wrapped.capabilities()["uncertainty"] == "yes"


def test_a_hybrid_without_a_predictive_distribution_says_so():
    class PointSurrogate:
        def fit(self, X, Y):
            self.mu = float(np.mean(Y))
            return self

        def predict(self, X):
            return np.full((len(X), 1), self.mu)

    post = CallablePostModel(lambda intermediates, params: intermediates["rho"] * 2.0,
                             name="twice the density", source="a made-up post-model, for the no-band case")
    hyb = HybridSurrogatePhysics(PointSurrogate(), post)
    rng = np.random.default_rng(2)
    params = {"v": rng.uniform(1, 2, 8)}
    hyb.fit(params, {"rho": rng.uniform(0.1, 0.3, (8, 1))})
    wrapped = as_physics_model(hyb)
    assert wrapped.capabilities()["uncertainty"].startswith("no")
    with pytest.raises(UncertaintyNotAvailable, match="point prediction"):
        wrapped.uncertainty(params, n_samples=10)


# ------------------------------------------------------------------------------------------------- export
def test_torchscript_export_reloads_and_reproduces_the_prediction(tmp_path):
    wrapped = as_physics_model(_pinn())
    x = torch.randn(4, 2)
    expected = wrapped.predict(x).mean
    path = wrapped.export(tmp_path / "pinn.pt", fmt="torchscript")
    assert path.is_file() and path.stat().st_size > 0
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        reloaded = torch.jit.load(str(path))
    assert torch.allclose(reloaded(x), expected, atol=1e-5)


def test_state_dict_export_carries_the_class_and_the_family(tmp_path):
    path = as_physics_model(_pod()).export(tmp_path / "pod.pt", fmt="state_dict")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["family"] == "rom" and saved["class"] == "POD" and saved["state_dict"]


def test_a_graph_model_is_exported_as_a_state_dict_not_as_a_fake_graph(tmp_path):
    wrapped = as_physics_model(_mgn())
    with pytest.raises(ExportNotAvailable, match="state_dict"):
        wrapped.export(tmp_path / "mgn.pt", fmt="torchscript")
    assert wrapped.export(tmp_path / "mgn.pt", fmt="state_dict").is_file()


def test_tracing_without_an_example_asks_for_one_instead_of_guessing(tmp_path):
    with pytest.raises(ExportNotAvailable, match="example="):
        as_physics_model(_fno2d()).export(tmp_path / "fno.pt", fmt="torchscript")
    x = torch.randn(1, 1, 16, 16)
    path = as_physics_model(_fno2d().eval()).export(tmp_path / "fno.pt", fmt="torchscript", example=x)
    assert path.is_file() and path.stat().st_size > 0


def test_an_unknown_export_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown fmt"):
        as_physics_model(_pinn()).export(tmp_path / "x.bin", fmt="pickle")


def test_a_hybrid_explains_that_its_persistence_is_not_a_torch_graph(tmp_path):
    with pytest.raises(ExportNotAvailable, match="state"):
        as_physics_model(_hybrid()).export(tmp_path / "h.pt")


# ---------------------------------------------------------------------------------------------- the records
def test_a_prediction_rejects_a_spread_of_the_wrong_shape():
    with pytest.raises(ValueError, match="std has shape"):
        Prediction(mean=torch.zeros(3, 1), std=torch.zeros(4, 1))


def test_a_confidence_interval_needs_the_spread_first():
    with pytest.raises(UncertaintyNotAvailable, match="point estimate"):
        Prediction(mean=torch.zeros(3, 1)).interval()


def test_the_interval_widens_with_the_level():
    band = Prediction(mean=torch.zeros(3, 1), std=torch.ones(3, 1))
    narrow, wide = band.interval(0.68), band.interval(0.99)
    assert float(wide[1].max()) > float(narrow[1].max()) > 0.0
    with pytest.raises(ValueError, match="level"):
        band.interval(1.5)


def test_the_fit_report_summarises_both_kinds_of_fit():
    gradient = FitReport(family="pinn", method="gradient", epochs=2, seconds=0.5, n_params=100,
                         history=[{"train_total": 1.0}, {"train_total": 0.25}])
    assert gradient.final_loss == 0.25 and gradient.improved() == 4.0
    assert "2 epochs" in gradient.summary() and "1 -> 0.25" in gradient.summary()
    closed = FitReport(family="rom", method="closed_form", epochs=0, seconds=0.1, n_params=0)
    assert closed.final_loss is None and closed.improved() is None
