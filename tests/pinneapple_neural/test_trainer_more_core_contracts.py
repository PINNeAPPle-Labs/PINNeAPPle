"""Core coverage (#1129, toward 1.0) for the rest of the trainer: gray-box identification, multi-restart, two-phase
containers, the training advisor, data adapters, the adaptive sweep, causal weights and configuration objects.

Two numeric references:
* causal weights follow w_k = exp(-eps * sum_{j<k} L_j) exactly;
* gray-box training recovers an unknown term: u' = -u - g(u), with g(u) = u^2 hidden, from data of the exact
  solution u = 1 / (2 e^t - 1).
"""
import math

import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.trainer import adaptive_sweep as AS  # noqa: E402
from pinneapple_neural.trainer import advisor as AD  # noqa: E402
from pinneapple_neural.trainer import causal as CA  # noqa: E402
from pinneapple_neural.trainer import datamodule as DM  # noqa: E402
from pinneapple_neural.trainer import graybox as GB  # noqa: E402
from pinneapple_neural.trainer import loss_balancer as LB  # noqa: E402
from pinneapple_neural.trainer import multistart as MS  # noqa: E402
from pinneapple_neural.trainer import noether_trainer as NT  # noqa: E402
from pinneapple_neural.trainer import two_phase as TP  # noqa: E402


# ------------------------------------------------------------------------------------------------ gray box
def test_graybox_recovers_the_hidden_term_of_an_ode():
    torch.manual_seed(0)
    t = torch.linspace(0.0, 1.0, 64)[:, None]
    u = 1.0 / (2.0 * torch.exp(t) - 1.0)                     # exact solution of u' = -u - u^2, u(0) = 1
    mean, std = u.mean(0, keepdim=True), u.std(0, keepdim=True)

    def residual(model, x, learnable, known, nets):          # the model works in normalised units
        x = x.clone().requires_grad_(True)
        uu = model(x) * std + mean
        du = torch.autograd.grad(uu.sum(), x, create_graph=True)[0]
        return du + known["k"] * uu + nets["g"](uu)

    main = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, 1))
    g = GB.GrayBoxNet(1, 1, width=16, depth=2)
    cfg = GB.GrayBoxConfig(epochs_supervised=300, epochs_joint=1000, device="cpu", lr_graybox=3e-3)
    _, nets, params, hist = GB.train_graybox(main, {"g": g}, residual, t, u, lambda: torch.rand(128, 1),
                                             known_params={"k": 1.0}, config=cfg)
    assert params == {} and hist[-1]["loss_data"] < 1e-3
    uu = torch.linspace(0.25, 1.0, 20)[:, None]
    with torch.no_grad():
        rel = float((nets["g"](uu) - uu ** 2).norm() / (uu ** 2).norm())
    assert rel < 0.25                                         # measured 0.15 after 1300 CPU epochs


def test_distill_without_pysr_says_so():
    try:
        import pysr  # noqa: F401
        pytest.skip("pysr is installed")
    except ImportError:
        pass
    out = GB.distill_to_symbolic(GB.GrayBoxNet(), [(0.0, 1.0)])
    assert out["available"] is False and "pysr" in out["message"].lower()


# ------------------------------------------------------------------------------------------------ multi-restart
def test_multi_restart_keeps_the_best_score():
    seeds = []

    def train_fn(model, seed):
        seeds.append(seed)
        return {"loss": [float(seed % 7)]}, float((seed - 242) ** 2)            # best: seed 242 (restart 2)
    cfg = MS.MultiRestartConfig(n_restarts=4, seed_base=42, verbose=False, keep_all=True)
    res = MS.MultiRestartTrainer(lambda seed: torch.nn.Linear(1, 1), train_fn, config=cfg).fit()
    assert isinstance(res, MS.MultiRestartResult) and seeds == [42, 142, 242, 342]   # seed_base + 100 k
    assert res.score == 0.0 and res.best.seed == 242 and res.best.restart_idx == 2 and res.n_valid == 4
    assert len(res.all_results) == 4 and isinstance(res.model, torch.nn.Module) and res.history == {"loss": [4.0]}
    one = MS.RestartResult(model=torch.nn.Linear(1, 1), history={}, score=1.0, restart_idx=0, seed=0)
    assert not one.nan_detected


# ------------------------------------------------------------------------------------------------ two phase
def test_unnorm_model_and_two_phase_containers():
    base = torch.nn.Linear(2, 2)
    um = TP.UnnormModel(base, [1.0, -2.0], [10.0, 0.5])
    x = torch.randn(4, 2)
    assert torch.allclose(um(x), base(x) * torch.tensor([10.0, 0.5]) + torch.tensor([1.0, -2.0]))
    cfg = TP.TwoPhaseConfig(phase1_epochs=3, combined=True)
    assert cfg.phase1_epochs == 3 and cfg.physics_weight_start == 0.0 and cfg.physics_weight_end == 1.0
    h = TP.TwoPhaseHistory()
    assert h.phase1_loss == [] and h.phase2_pde_loss == []


# ------------------------------------------------------------------------------------------------ advisor
def test_advisor_flags_a_diverging_run_and_patches_the_config():
    adv = AD.TrainingAdvisor()
    short = adv.analyse("m", [1.0, 0.5], [], [], {}, 1e-3, 10, 32)
    assert isinstance(short, AD.DiagnosticReport) and short.signals == ["insufficient_history"]
    rising = [1.0 * 1.1 ** i for i in range(40)]
    rep = adv.analyse("m", rising, rising, [0.01] * 40, {"rel_l2": 0.9}, 1e-2, 40, 32)
    assert rep.signals and rep.suggestions
    assert [s.priority for s in rep.suggestions] == sorted(s.priority for s in rep.suggestions)   # best first
    patched = adv.apply_suggestions(rep.suggestions, {"lr": 1e-2}, max_priority=5)
    assert patched["_advisor_patches"] and patched != {"lr": 1e-2}
    s = AD.Suggestion("lower_lr", 1, "x", {"lr": 1e-4})
    assert adv.apply_suggestions([s], {"lr": 1e-2}) == {"lr": 1e-4, "_advisor_patches": ["lower_lr"]}


# ------------------------------------------------------------------------------------------------ data adapters
def test_adapters_turn_items_into_dict_batches():
    a = DM.ItemAdapter()
    assert a({"x": 1}) == {"x": 1} and a((1, 2)) == {"x": 1, "y": 2}
    with pytest.raises(TypeError):
        a(3)
    fa = DM.FnAdapter(lambda s: s * 2, lambda s: s + 1)
    ds = DM.AdaptedSequenceDataset([1, 2, 3], fa)
    assert len(ds) == 3 and ds[2] == {"x": 6, "y": 4, "meta": {}}


# ------------------------------------------------------------------------------------------------ sweep
def test_adaptive_sweep_finds_the_best_region():
    cfg = AS.AdaptiveSweepConfig(param_space={"x": ("float", (-2.0, 2.0)), "n": ("int", (1, 4))}, n_trials=40,
                                 seed=0)
    res = AS.run_adaptive_sweep(lambda x, n: (x - 0.5) ** 2 + (n - 3) ** 2, cfg)
    vals = [r["value"] for r in res if "value" in r]
    assert vals == sorted(vals) and res[0]["params"]["n"] == 3 and abs(res[0]["params"]["x"] - 0.5) < 0.3


# ------------------------------------------------------------------------------------------------ causal weights
def test_causal_weights_follow_their_definition():
    sch = CA.CausalWeightScheduler(n_time_chunks=4, epsilon=2.0)
    t = torch.tensor([0.1, 0.3, 0.6, 0.9])                                       # one point per chunk
    r = torch.tensor([1.0, 2.0, 0.5, 3.0])
    w = sch.compute_weights(t, r)
    L = r ** 2
    ref = torch.exp(-2.0 * torch.tensor([0.0, L[0], L[0] + L[1], L[0] + L[1] + L[2]]))   # exact
    assert torch.allclose(w, ref)
    loss = sch.weighted_loss(t, r)
    assert float(loss) == pytest.approx(float((ref * L).sum() / ref.sum()), rel=1e-6)


# ------------------------------------------------------------------------------------------------ configuration
def test_configuration_objects():
    nc = NT.NoetherTrainConfig()
    assert nc.epochs > 0 and nc.lr > 0
    res = NT.NoetherTrainResult(train_losses=[1.0, 0.5], val_losses=[1.2, 0.6], best_val_loss=0.6, best_epoch=1)
    assert "best val loss  : 0.600000  (epoch 1)" in res.summary() and "epochs trained : 2" in res.summary()
    ab = LB.AutoBalancerConfig()
    assert ab.method == "relobralo" and 0 < ab.plateau_rel_thresh < 1 and math.isfinite(ab.divergence_thresh)
