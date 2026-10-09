"""Adaptive ensembles of physics models (neural operators, numerical schemes, PINNs, GNNs) on cases with exact answers."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D  # noqa: E402
from pinneapple_physics.ensemble import (  # noqa: E402
    Expert,
    PhysicsEnsemble,
    fit_static_weights,
    from_callable,
    from_solution,
    from_torch,
)

AD = AdvectionDiffusion1D()


def fno_input(case):
    return torch.tensor(np.stack([case["u0"], np.full(AD.nx, case["nu"] * 50)])[None], dtype=torch.float32)


@pytest.fixture(scope="module")
def fno():
    """FNO trained only on the low-diffusion regime nu in [0.005, 0.02]."""
    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    cases = [AD.random_case(rng, rng.uniform(0.005, 0.02)) for _ in range(300)]
    X = torch.cat([fno_input(c) for c in cases])
    Y = torch.tensor(np.stack([AD.exact(c) for c in cases]), dtype=torch.float32)
    m = FourierNeuralOperator(2, AD.nt, width=32, modes=12, layers=4, use_grid=True)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    for _ in range(600):
        idx = torch.randint(0, len(cases), (64,))
        loss = ((m(X[idx]).y - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return m


def experts(fno_model):
    return [from_callable("upwind", lambda c: AD.finite_difference(c, "upwind")),
            from_callable("lax_wendroff_coarse", lambda c: AD.finite_difference(c, "lax_wendroff", coarsen=2)),
            from_callable("calibrated_nu_0.05", lambda c: AD.exact({**c, "nu": 0.05})),
            from_torch("fno", fno_model, fno_input, lambda y, c: y[0])]


@pytest.fixture(scope="module")
def stream():
    rng = np.random.default_rng(5)
    cases = [AD.random_case(rng, rng.uniform(0.005, 0.02)) for _ in range(50)] + \
            [AD.random_case(rng, rng.uniform(0.1, 0.3)) for _ in range(50)]
    return cases, [AD.exact(c) for c in cases]


def test_benchmark_schemes_have_the_expected_regime_dependent_errors():
    rng = np.random.default_rng(0)
    low, high = AD.random_case(rng, 0.01), AD.random_case(rng, 0.2)
    err = lambda f, c: np.linalg.norm(f - AD.exact(c)) / np.linalg.norm(AD.exact(c))  # noqa: E731
    assert err(AD.finite_difference(low, "lax_wendroff"), low) < err(AD.finite_difference(low, "upwind"), low)
    assert err(AD.finite_difference(high, "lax_wendroff", coarsen=2), high) < 1e-2
    assert AD.residual(AD.exact(high), high) < 1e-2 < AD.residual(AD.exact({**high, "nu": 0.05}), high)
    assert abs(AD.total(AD.exact(low)[-1]) - AD.total(low["u0"])) < 1e-10


def test_ensemble_switches_model_when_the_regime_changes(fno, stream):
    cases, refs = stream
    for mode in ("select", "combine"):
        run = PhysicsEnsemble(experts(fno), mode=mode).run(cases, refs)
        s = run.summary()
        best = s["best_single_in_hindsight"]["error"]
        # select: beats the best single model; combine: on par with it (a catastrophic out-of-distribution
        # expert still costs a little during the transition)
        assert s["ensemble"] < best if mode == "select" else s["ensemble"] < 1.05 * best
        lead_low = max(set(run.active[5:50]), key=run.active[5:50].count)
        lead_high = max(set(run.active[60:]), key=run.active[60:].count)
        assert lead_low == "fno" and lead_high == "lax_wendroff_coarse"
        first_switch = next(i for i, a in s["switches"] if i >= 50 and a == "lax_wendroff_coarse")
        assert first_switch <= 55                                  # adapts within a few cases
        assert abs(s["coverage"] - 0.9) < 0.08


def test_label_free_physics_residual_rejects_the_out_of_distribution_operator(fno, stream):
    cases, _ = stream
    run = PhysicsEnsemble(experts(fno), residual_fn=AD.residual, residual_weight=1.0).run(cases[50:], None)
    w = dict(zip(run.names, run.weights[-1], strict=True))
    assert w["fno"] < 0.01 and w["calibrated_nu_0.05"] < 0.01
    assert max(w, key=w.get) == "lax_wendroff_coarse"


def test_cost_aware_lazy_selection_saves_cost_and_keeps_accuracy(fno, stream):
    cases, refs = stream
    ex = experts(fno)
    ex.append(from_callable("spectral_solver", AD.exact, cost=50.0))   # exact but expensive
    for e in ex[:-1]:
        e.cost = 1.0 if e.name == "fno" else 5.0
    ens = PhysicsEnsemble(ex, mode="select", cost_weight=0.2, explore_every=10)
    errs, cost = [], 0.0
    for c, r in zip(cases, refs, strict=True):
        p = ens.predict(c, select_only_leader=True)
        cost += sum(p["costs"][n] for n in p["expert_predictions"])
        ens.update(c, r)
        errs.append(np.linalg.norm(p["prediction"] - r) / np.linalg.norm(r))
    run_all_cost = len(cases) * (1 + 5 * 3 + 50)
    assert cost < 0.3 * run_all_cost
    assert np.mean(errs[5:50]) < 0.02 and np.mean(errs[60:]) < 0.05


def test_residual_lookahead_switches_on_the_first_case_of_the_new_regime(fno, stream):
    """The residual of each expert on the current case (no reference needed) tilts the weights before predicting:
    the switch happens on the first case of the new regime instead of after the errors of earlier cases pile up."""
    cases, refs = stream
    base = PhysicsEnsemble(experts(fno), mode="select").run(cases, refs)
    ahead = PhysicsEnsemble(experts(fno), mode="select", residual_fn=AD.residual, residual_lookahead=2.0).run(cases, refs)
    assert base.active[50] != "lax_wendroff_coarse"                  # reacting to past errors lags
    assert ahead.active[50] == "lax_wendroff_coarse"                 # the FNO residual jumps on the first new case
    assert np.nanmean(ahead.ensemble_errors[50:55]) < 0.1 * np.nanmean(base.ensemble_errors[50:55])
    assert ahead.summary()["ensemble"] < 0.7 * base.summary()["ensemble"]
    assert ahead.active[5:50].count("fno") >= 40                     # FNO keeps its own regime
    # comparing residual levels across experts also switches at once, but ranks the schemes (that satisfy their
    # discrete equation) above the more accurate FNO in its own regime
    across = PhysicsEnsemble(experts(fno), mode="select", residual_fn=AD.residual, residual_lookahead=100.0,
                             lookahead_reference="experts").run(cases, refs)
    assert across.active[50] == "lax_wendroff_coarse" and across.active[5:50].count("fno") < 10
    with pytest.raises(ValueError):
        PhysicsEnsemble(experts(fno), residual_lookahead=1.0)


def test_no_look_ahead(fno, stream):
    cases, refs = stream
    refs2 = [r if i < 30 else r * 0 + 100.0 for i, r in enumerate(refs)]
    a = PhysicsEnsemble(experts(fno)).run(cases[:40], refs[:40])
    b = PhysicsEnsemble(experts(fno)).run(cases[:40], refs2[:40])
    np.testing.assert_array_equal(a.weights[:31], b.weights[:31])        # weights for case k use cases < k only
    assert not np.allclose(a.weights[32], b.weights[32])


def test_conservation_projection_and_failing_expert():
    rng = np.random.default_rng(3)
    case = AD.random_case(rng, 0.1)
    pw = np.full((AD.nt, AD.nx), 2 * np.pi / AD.nx) / AD.nt

    def broken(c):
        raise RuntimeError("solver crashed")

    ens = PhysicsEnsemble([from_callable("upwind", lambda c: AD.finite_difference(c, "upwind")),
                           from_callable("biased", lambda c: AD.exact(c) + 0.3), Expert("broken", broken)],
                          conserve=(pw, lambda c: AD.total(c["u0"]) * AD.nt * (1 / AD.nt) * 1.0))
    p = ens.predict(case)
    assert abs(float(np.sum(pw * p["prediction"])) - AD.total(case["u0"])) < 1e-9
    assert ens.failures["broken"] == 1 and "broken" not in p["expert_predictions"]
    ens.update(case, AD.exact(case))
    assert ens.weights()["broken"] < ens.weights()["upwind"]


def test_static_convex_weights_with_cross_validation():
    rng = np.random.default_rng(4)
    cases = [AD.random_case(rng, rng.uniform(0.05, 0.3)) for _ in range(30)]
    P = np.stack([[AD.finite_difference(c, "upwind"), AD.finite_difference(c, "lax_wendroff", coarsen=2),
                   AD.exact({**c, "nu": 0.05})] for c in cases])
    Y = np.stack([AD.exact(c) for c in cases])
    out = fit_static_weights(P, Y)
    assert np.all(out["weights"] >= 0) and abs(out["weights"].sum() - 1) < 1e-9
    best = out["best_single"]
    assert out["weights"][best] > 0.8                              # one scheme dominates: the weights find it
    assert out["cv_error"] <= min(out["single_errors"]) * 1.1     # and the cross-validated error stays close to it


def test_adapters_for_pinn_solution_and_graph_network():
    import pinneapple as pp
    from pinneapple_neural.architectures.graphnn.base import GraphBatch
    from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet

    sol = pp.solve("burgers_1d", "analytic", nu=0.01 / np.pi)
    X = np.column_stack([np.linspace(-1, 1, 50), np.full(50, 0.5)])
    e1 = from_solution("analytic", sol, points=lambda q: X, shape=lambda q: (50,))
    pinn = pp.solve("burgers_1d", "pinn", nu=0.01 / np.pi, epochs=20)
    e2 = from_solution("pinn", pinn, points=lambda q: X, shape=lambda q: (50,))
    assert e1.predict(None).shape == e2.predict(None).shape == (50,)

    n = AD.nx
    ring = torch.tensor([list(range(n)) + list(range(n)), [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]])
    gnn = MeshGraphNet(node_in_dim=2, out_dim=AD.nt, hidden_dim=16, n_message_passing=2)
    e3 = from_torch("gnn", gnn, lambda c: GraphBatch(x=fno_input(c).transpose(1, 2), edge_index=ring),
                    lambda y, c: y[0].T)
    case = AD.random_case(np.random.default_rng(0), 0.1)
    assert e3.predict(case).shape == (AD.nt, AD.nx)
