"""pinneapple_arena.pointwise: the problems, reference data, losses, adapters and training protocol of the PINNeAPPle
Arena app. The adapter tests are the source repository's backend/tests/test_model_adapters.py, ported."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_arena import pointwise as pw  # noqa: E402
from pinneapple_arena.pointwise.adapters import ClosedFormAdapter, PointwiseAdapter  # noqa: E402
from pinneapple_arena.pointwise.tables import parse_numeric_table  # noqa: E402

IN_DIM, OUT_DIM, N = 3, 1, 6


# ---------------------------------------------------------------- ported: backend/tests/test_model_adapters.py

@pytest.mark.parametrize("arch_id", sorted(pw.POINTWISE_LAYOUTS))
def test_pointwise_adapter_is_a_pointwise_field_model(arch_id):
    model = pw.instantiate_model(arch_id, IN_DIM, OUT_DIM)
    assert isinstance(model, PointwiseAdapter)
    model.train()
    x = torch.rand(N, IN_DIM, requires_grad=True)
    y = pw.model_forward(model, x)
    assert y.shape == (N, OUT_DIM)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(y.sum(), params, allow_unused=True, retain_graph=True)
    assert any(g is not None for g in grads)
    g0 = torch.autograd.grad(y[0].sum(), x, retain_graph=True)[0]      # y[0] depends on x[0] only
    assert g0[0].abs().sum() > 0
    assert torch.all(g0[1:] == 0)


@pytest.mark.parametrize("arch_id", sorted(pw.CLOSED_FORM))
def test_closed_form_adapter_fits_and_predicts(arch_id):
    model = pw.instantiate_model(arch_id, IN_DIM, OUT_DIM)
    assert isinstance(model, ClosedFormAdapter)
    rng = np.random.default_rng(0)
    X = rng.random((80, IN_DIM)).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) ** 2).astype(np.float32)
    out = pw.train_closed_form_model(model, X[:60], y[:60], X[60:], y[60:])
    assert out["y_pred"].shape == (60, OUT_DIM)
    assert np.isfinite(out["best_val_loss"])


@pytest.mark.parametrize("arch_id", sorted(pw.NOT_APPLICABLE))
def test_trajectory_models_are_reported_not_applicable(arch_id):
    with pytest.raises(pw.ArchitectureNotApplicable, match="Not applicable"):
        pw.instantiate_model(arch_id, IN_DIM, OUT_DIM)


# ------------------------------------------------------------------------------------------------- new

def test_no_stand_in_for_an_architecture_the_registry_can_not_build():
    with pytest.raises(pw.ArchitectureUnavailable, match="pp.build_model"):
        pw.instantiate_model("lstm", 2, 1)            # needs sequence arguments a pointwise problem does not have
    with pytest.raises(pw.ArchitectureUnavailable):
        pw.instantiate_model("no_such_architecture", 2, 1)


def test_renamed_catalog_entries_build_from_their_registry_key():
    assert pw.registry_key("symm_transformer") == "noether_transformer"
    assert pw.registry_key("icosahedral_gnn") == "graphcast"
    status = pw.catalog_status()
    assert len(status) == len(pw.ARCHITECTURES_CATALOG) == 93
    assert all(status[a] for a in pw.REGISTRY_KEYS)   # all nine resolve in the registry


def test_a_model_with_the_wrong_output_shape_is_refused():
    class Broadcasts(torch.nn.Module):                # (N, d) -> (1, N, d): the source trained this on a broadcast loss
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return (x * self.w)[None]

    X = np.random.default_rng(0).random((20, 2)).astype(np.float32)
    y = X[:, :1].copy()
    with pytest.raises(ValueError, match="not a pointwise field model"):
        pw.train_torch_model(Broadcasts(), X, y, X, y, None, None, n_epochs=2, lr=1e-3, device="cpu")


@pytest.mark.parametrize("problem_id", sorted(pw.PROBLEMS_CATALOG))
def test_reference_data_matches_the_declared_dimensions(problem_id):
    d = pw.DatasetGenerator(problem_id, {}).generate(n_samples=300)
    assert d["coords"].shape[1] == pw.problem_in_dim(problem_id)
    assert d["solution"].shape[1] == pw.problem_out_dim(problem_id)
    assert np.isfinite(d["coords"]).all() and np.isfinite(d["solution"]).all()


def test_unknown_problem_has_no_synthetic_stand_in():
    with pytest.raises(KeyError):
        pw.DatasetGenerator("not_a_problem", {}).generate(100)


def test_collocation_points_lie_in_the_problem_domain():
    gen = pw.DatasetGenerator("burgers_1d", {})
    ref = gen.generate(2000)
    col = gen.get_collocation_data(n_col=500, n_bc=50, n_ic=50)["X_col"]
    lo, hi = ref["coords"].min(axis=0), ref["coords"].max(axis=0)
    assert (col >= lo - 1e-6).all() and (col <= hi + 1e-6).all()
    assert col[:, 0].min() < -0.5                      # x in [-1, 1], not [0, 1]


def test_burgers_residual_vanishes_on_an_exact_solution():
    """u = x / (1 + t) solves u_t + u u_x = nu u_xx exactly; a perturbed field does not."""
    loss_fn = pw.PhysicsLossGenerator("burgers_1d", {}).compile_losses()
    assert loss_fn is not None

    class Exact(torch.nn.Module):
        def forward(self, X):
            return X[:, :1] / (1.0 + X[:, 1:2])

    class Perturbed(torch.nn.Module):
        def forward(self, X):
            return X[:, :1] / (1.0 + X[:, 1:2]) + 0.1 * torch.sin(torch.pi * X[:, :1])

    X = torch.rand(200, 2) * torch.tensor([2.0, 1.0]) - torch.tensor([1.0, 0.0])
    r_exact, _ = loss_fn(Exact(), {"X": X})
    r_pert, _ = loss_fn(Perturbed(), {"X": X})
    assert float(r_exact) < 1e-10 < 1e-4 < float(r_pert)


def test_field_metrics_agree_with_the_library_metrics():
    from pinneapple_physics import metrics as M
    rng = np.random.default_rng(1)
    t = rng.normal(size=(50, 2))
    p = t + 0.1 * rng.normal(size=(50, 2))
    m = pw.field_metrics(p, t)
    assert m["l2_relative"] == pytest.approx(float(M.relative_l2(p.ravel(), t.ravel())))
    assert m["rmse"] == pytest.approx(float(np.sqrt(np.mean((p - t) ** 2))))
    assert 0.9 < m["r2"] <= 1.0
    with pytest.raises(ValueError):
        pw.field_metrics(p[:, :1], t)


def test_prediction_quality_flags_degenerate_outputs():
    t = np.linspace(0, 1, 100)
    assert pw.prediction_quality(np.zeros_like(t), t, {"r2": -1, "l2_relative": 1})["quality_level"] == "unreliable"
    good = pw.prediction_quality(t + 1e-3, t, pw.field_metrics(t + 1e-3, t))
    assert good["quality_level"] == "excellent" and not good["quality_flags"]


def _heat_run(n_epochs: int, n_train: int):
    gen = pw.DatasetGenerator("heat_equation_1d", {})
    ref = gen.generate(n_train + 400)
    X, y = ref["coords"], ref["solution"]
    col = gen.get_collocation_data(n_col=200, n_bc=50, n_ic=50)["X_col"]
    loss_fn = pw.PhysicsLossGenerator("heat_equation_1d", {}).compile_losses()
    torch.manual_seed(0)
    model = pw.instantiate_model("vanilla_pinn", X.shape[1], y.shape[1])
    out = pw.train_torch_model(model, X[:n_train], y[:n_train], X[n_train:n_train + 200],
                               y[n_train:n_train + 200], col, loss_fn, n_epochs=n_epochs, lr=3e-3, device="cpu")
    pred, _ = pw.predict(model, X[n_train + 200:])
    return out, pw.field_metrics(pred, y[n_train + 200:])


def test_train_torch_model_runs_the_residual():
    out, m = _heat_run(n_epochs=3, n_train=200)
    assert out["n_pde_evaluations"] == 3 and len(out["loss_history"]) == 3
    assert all(np.isfinite(v) for v in m.values() if v is not None)


@pytest.mark.skipif(not __import__("os").environ.get("PINNEAPPLE_SLOW"), reason="slow: set PINNEAPPLE_SLOW=1")
def test_train_torch_model_learns_a_field():
    _, m = _heat_run(n_epochs=300, n_train=1000)
    assert m["l2_relative"] < 0.5, m                 # 300 epochs: a rough approximation of the field


def test_leaderboard_aggregate_ranks_by_mean_relative_l2():
    exps = [
        {"id": "e1", "winner": "a", "problem_id": "p1", "model_results": [
            {"arch_id": "a", "status": "completed", "metrics": {"l2_relative": 0.1}, "training_time_s": 1.0},
            {"arch_id": "b", "status": "completed", "metrics": {"l2_relative": 0.4}, "training_time_s": 3.0}]},
        {"id": "e2", "winner": "b", "problem_id": "p2", "model_results": [
            {"arch_id": "a", "status": "completed", "metrics": {"l2_relative": 0.3}, "training_time_s": 1.0},
            {"arch_id": "b", "status": "completed", "metrics": {"l2_relative": 0.2}, "training_time_s": 3.0}]},
    ]
    lb = pw.aggregate_leaderboard(exps, families={"a": "PINN"})
    assert lb["total_experiments"] == 2 and lb["problems_covered"] == 2
    a, b = lb["entries"]
    assert (a["architecture_id"], a["rank"], a["avg_l2_error"], a["family"]) == ("a", 1, 0.2, "PINN")
    assert b["avg_l2_error"] == 0.3 and a["win_rate"] == 0.5
    assert a["scores"]["accuracy"] == 80.0 and a["scores"]["training_cost"] == 100.0
    assert pw.aggregate_leaderboard([])["entries"] == []


def test_parse_numeric_table_counts_skipped_rows():
    cols, arr, skipped = parse_numeric_table("d.csv", b"x;u\n0;1\n1;2\nbad;row\n2;3;4\n")
    assert cols == ["x", "u"] and arr.shape == (2, 2) and skipped == 2
    buf = __import__("io").BytesIO()
    np.savez(buf, x=np.arange(3.0), u=np.arange(3.0) ** 2)
    cols, arr, _ = parse_numeric_table("d.npz", buf.getvalue())
    assert cols == ["x", "u"] and arr[2, 1] == 4.0
