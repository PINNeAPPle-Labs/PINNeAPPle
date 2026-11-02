"""Regression tests for the eight bugs found while writing GETTING_STARTED.md (#939, #940, #941, #942, #945, #953,
#957, #958). Each test is the issue's own reproduction, with the expected value it states.

References: exact solutions with known derivatives (a harmonic polynomial has zero Laplacian, x^2 + y^2 has
Laplacian 4, sin(pi x) sin(pi y) has Laplacian -2 pi^2 times itself); the Cole-Hopf solution of viscous Burgers;
a reloaded model reproduces the saved one exactly.
"""
import contextlib
import io

import numpy as np
import pytest
import torch

import pinneapple as pp


class _Exact(torch.nn.Module):
    """A fixed function wrapped as a model (one dummy parameter so it has a device and a state)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, X):
        return self.fn(X) + 0 * self.p


def _y(out):
    return out if isinstance(out, torch.Tensor) else out.y


def _check(rep, cid):
    return next(c for c in rep.checks if c.id == cid)


# ------------------------------------------------------------------------------------------------ #939
def test_validate_accepts_a_problem_spec_and_measures_its_residual():
    lap = pp.get_preset("laplace_2d")
    harmonic = _Exact(lambda X: X[:, :1] ** 2 - X[:, 1:2] ** 2)
    rep = pp.validate(harmonic, {"pinn": lap})
    assert _check(rep, "pinn.residual").status == "pass" and _check(rep, "pinn.residual").value < 1e-6
    bowl = _Exact(lambda X: X[:, :1] ** 2 + X[:, 1:2] ** 2)
    r = _check(pp.validate(bowl, {"pinn": lap}), "pinn.residual")
    assert r.status == "fail" and r.value == pytest.approx(4.0, rel=1e-5)                    # Laplacian of the bowl

    def f(X, ctx=None):                                                                       # lap u = f
        X = np.asarray(X)
        return -2 * np.pi ** 2 * np.sin(np.pi * X[:, :1]) * np.sin(np.pi * X[:, 1:2])
    exact = _Exact(lambda X: torch.sin(np.pi * X[:, :1]) * torch.sin(np.pi * X[:, 1:2]))
    rep = pp.validate(exact, {"pinn": pp.get_preset("poisson_2d"), "ctx": {"source_fn": f}})
    assert _check(rep, "pinn.residual").status == "pass" and _check(rep, "pinn.residual").value < 1e-5
    with pytest.raises(ValueError, match="ProblemSpec"):
        pp.validate(exact, {"pinn": 42})


# ------------------------------------------------------------------------------------------------ #940
def test_arena_builds_modified_mlp():
    from pinneapple_arena.model_factory import ModelConfig, NetworkConfig, build_model
    m = build_model(ModelConfig(name="ModifiedMLP", type="modified_mlp", network=NetworkConfig(hidden=[32, 32, 32])),
                    in_dim=2, out_dim=1)
    assert type(m).__name__ == "ModifiedMLP" and _y(m(torch.rand(5, 2))).shape == (5, 1)
    m2 = build_model(ModelConfig(name="ModifiedMLP", type="modified_mlp"), in_dim=2, out_dim=1)   # no network block
    assert type(m2).__name__ == "ModifiedMLP"


def test_modified_mlp_takes_an_activation_by_name():
    from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
    assert _y(ModifiedMLP(in_dim=2, out_dim=1, hidden_dim=8, n_layers=2, activation="silu")(torch.rand(3, 2))).shape == (3, 1)
    with pytest.raises(ValueError, match="activation"):
        ModifiedMLP(in_dim=2, out_dim=1, activation="no-such-activation")


# ------------------------------------------------------------------------------------------------ #941
def test_a_saved_model_reloads_in_one_call(tmp_path):
    m = pp.build_model("modified_mlp", in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)
    path = tmp_path / "m.pt"
    m.save_checkpoint(str(path))
    back = pp.load_checkpoint(str(path))
    X = torch.rand(10, 2)
    assert type(back) is type(m) and torch.equal(_y(back(X)), _y(m(X)))                                # the same function
    old = torch.load(path, weights_only=False)
    assert old["model_kwargs"]["hidden_dim"] == 16 and old["registry_name"] == "modified_mlp"
    old.pop("model_kwargs")
    torch.save(old, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="kwargs|arguments"):
        pp.load_checkpoint(str(tmp_path / "old.pt"))                                         # pre-#941 checkpoint


# ------------------------------------------------------------------------------------------------ #942
def _cole_hopf(x, t, nu, n=200):
    """Viscous Burgers u_t + u u_x = nu u_xx on (-1, 1), u(x, 0) = -sin(pi x): Cole-Hopf by Gauss-Hermite."""
    z, w = np.polynomial.hermite.hermgauss(n)
    y = x[:, None] - np.sqrt(4 * nu * t) * z[None, :]
    e = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    return -(w * np.sin(np.pi * y) * e).sum(1) / (w * e).sum(1)


def test_the_burgers_reference_solution_matches_cole_hopf():
    sol = pp.solve("burgers_1d", "reference")
    nu = float(pp.get_preset("burgers_1d").pde.params.get("nu", 0.01 / np.pi))
    x = np.linspace(-0.95, 0.95, 77)
    for t in (0.25, 0.5, 1.0):
        u = np.asarray(sol.predict(np.c_[x, np.full_like(x, t)])).reshape(-1)
        ex = _cole_hopf(x, t, nu)
        err = np.linalg.norm(u - ex) / np.linalg.norm(ex)
        assert err < 0.05, (t, err)                                                             # was O(1) before


# ------------------------------------------------------------------------------------------------ #945
def test_quickstart_prints_a_short_summary():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pp.quickstart("burgers_1d")
    out = buf.getvalue()
    assert "Quickstart: burgers_1d" in out
    line = next(ln for ln in out.splitlines() if "Conditions" in ln)
    assert "ConditionSpec" not in line and " at 0x" not in line and "(initial)" in line


# ------------------------------------------------------------------------------------------------ #953
@pytest.mark.parametrize("given", [70, 70.0, "70 degC", "70degC", (343.15, "K"), (70, "degC")])
def test_a_bare_answer_is_read_in_the_unit_the_question_shows(given):
    from pinneapple_design.intent import formulate
    text = "size a heat sink for a 50 W chip in still air, keep it under 60 mm wide"
    asked = formulate(text)
    q = next(q for q in asked.questions if q.slot == "t_limit")
    assert q.unit in ("°C", "degC")                                                          # what the user sees
    spec = formulate(text, answers={"t_limit": given})
    assert spec.values["t_limit"].value == pytest.approx(343.15)                              # 70 degC, not 70 K


# ------------------------------------------------------------------------------------------------ #957
def test_every_catalogued_solver_can_be_run_and_says_how():
    import pinneapple_simulation.numerical_solvers as ns
    cat = ns.SolverCatalog()
    assert cat.get("fdm").cls.__name__ == "FDMSolver"
    for name in cat.list():
        d = cat.describe(name)
        assert d["entry_point"] in ("solve", "solve_from_spec", "forward") and d["example"].startswith("SolverCatalog")
        assert hasattr(ns.SolverRegistry.spec(name).cls, "solve")                                 # uniform entry point
    out = cat.build("fdm").solve(pp.get_preset("burgers_1d"))
    assert type(out).__name__ == "SolverOutput" and torch.isfinite(out.result).all()
    x = torch.sin(torch.linspace(0, 2 * np.pi, 64))
    assert type(cat.build("fft").solve(x)).__name__ == "SolverOutput"
    with pytest.raises(TypeError, match="constructor is FDMSolver"):
        cat.build("fdm", no_such_argument=1)


# ------------------------------------------------------------------------------------------------ #958
def test_fit_trains_without_a_validation_loader(tmp_path):
    from pinneapple_neural import TrainConfig, Trainer
    torch.manual_seed(0)
    x = torch.rand(64, 1)
    loader = [(x, 3 * x)]
    model = torch.nn.Linear(1, 1)

    def loss(m, y_hat, batch):
        return torch.mean((y_hat - batch["y"]) ** 2)
    cfg = TrainConfig(epochs=30, lr=0.05, log_dir=str(tmp_path), run_name="noval", device="cpu")
    out = Trainer(model, loss_fn=loss).fit(loader, cfg=cfg)
    assert out["validated"] is False and all(h["val_total"] is None for h in out["history"])
    assert out["history"][-1]["train_total"] < out["history"][0]["train_total"]               # it trained
    assert out["best_val"] == pytest.approx(min(h["train_total"] for h in out["history"]))   # selection on train
    out2 = Trainer(torch.nn.Linear(1, 1), loss_fn=loss).fit(loader, cfg)                     # positional cfg too
    assert out2["validated"] is False
    out3 = Trainer(torch.nn.Linear(1, 1), loss_fn=loss).fit(loader, loader, cfg)             # old call unchanged
    assert out3["validated"] and out3["history"][0]["val_total"] is not None
    assert Trainer.fit.__doc__ and "loss_fn(model, y_hat, batch)" in Trainer.fit.__doc__


def test_arena_builds_xtfc_without_the_default_activation():
    """#992: the Arena passed NetworkConfig's default activation ("tanh") to X-TFC, whose random Fourier features
    take none and refuse one, so every Arena run with an X-TFC fell back or failed."""
    import warnings

    from pinneapple_arena.model_factory import ModelConfig, build_model
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                                          # no fallback warning
        for t in ("xtfc", "extreme_tfc"):
            m = build_model(ModelConfig(name="X", type=t), in_dim=2, out_dim=1)
            assert type(m).__name__ == "XTFC"


def test_file_ref_takes_a_relative_path_under_an_absolute_root(tmp_path, monkeypatch):
    """#992: FileRef.of compared a relative path with an absolute root (a flow's run folder), so saving a dataset
    from a flow node raised ValueError; resolved on both sides they name the same file."""
    from pinneapple_data.contracts import FileRef
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.npz").write_bytes(b"abc")
    monkeypatch.chdir(tmp_path)
    assert FileRef.of("d/x.npz", root=str(tmp_path / "d")).path == "x.npz"
    assert FileRef.of(tmp_path / "d" / "x.npz", root="d").path == "x.npz"


def test_recorded_cfd_campaign_reads_the_committed_records(tmp_path):
    """#992: source="recorded" read _runs/cfd, which .gitignore keeps out of every checkout, so the Ahmed-body flow
    never ran; the committed records are results/cfd_runs."""
    from pinneapple_flow.nodes.body_study import _folders
    (tmp_path / "results" / "cfd_runs" / "results").mkdir(parents=True)
    assert _folders({"folder": str(tmp_path)})[1] == tmp_path / "results" / "cfd_runs"
    (tmp_path / "_runs" / "cfd" / "results").mkdir(parents=True)                         # a local campaign wins
    assert _folders({"folder": str(tmp_path)})[1] == tmp_path / "_runs" / "cfd"
