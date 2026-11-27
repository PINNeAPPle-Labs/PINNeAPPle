"""The model family of the black-hole weather forecast and its ensemble (issue #399, plan item 4).

Machinery only: a tiny grid, a couple of epochs and hand-made members whose errors are known by construction, so the
tests are seconds long and say nothing about how well the family forecasts the real flow (that is measured on the
simulated trajectories, see ``examples/black_hole_weather/family.py``).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_physics.blackhole import family as fam  # noqa: E402
from pinneapple_physics.blackhole.forecast import DensityCodec, MassEnvelope  # noqa: E402

H = W = 16                     # the U-Net of the paper pools four times
LO, HI = -6.0, 0.0


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def _codec() -> DensityCodec:
    return DensityCodec(LO, HI, H, 0)


def _volume() -> np.ndarray:
    return np.full((H, W), 1.0, np.float64)


def _ramp(frames: int = 48, rate: float = 0.01) -> np.ndarray:
    """A field that drifts linearly in time at exactly ``rate`` per frame, inside [0, 1] without touching either end,
    so the change a member has to predict is known by construction."""
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    base = 0.3 + 0.15 * np.sin(3.0 * x) * np.cos(2.0 * y)
    return np.stack([base + rate * t for t in range(frames)]).astype(np.float32)


def _trajectory(frames: int = 40, rate: float = 0.01, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A drifting field in normalised units: a smooth blob carried across the window, ``rate`` per frame."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    out = np.empty((frames, H, W), np.float32)
    for t in range(frames):
        blob = np.exp(-((x - 0.2 - rate * t) ** 2 + (y - 0.5) ** 2) / 0.08)
        out[t] = 0.2 + 0.5 * blob / max(blob.max(), 1e-9) + noise * rng.standard_normal((H, W))
    return np.clip(out, 0.0, 1.0)


class OffsetMember(torch.nn.Module):
    """A toy member: repeat the last input frame plus a constant offset. Its error against a trajectory that drifts
    at a known rate per frame is known by construction, which is what the aggregation test needs."""

    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """The block of the next frames: the last input frame repeated, plus the member's drift."""
        return x[:, -1:].expand(-1, x.shape[1], -1, -1) + self.delta + 0.0 * self.dummy


# ---------------------------------------------------------------------- the contract of the family
@pytest.mark.parametrize("name", fam.MEMBERS)
def test_every_member_obeys_the_forecast_contract_on_a_tiny_grid(name):
    model = fam.make_member(name)
    x = torch.rand(2, fam.K, H, W)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # the same call the skill and the ensemble make: iterative rollout from one block of frames
    from pinneapple_physics.blackhole.forecast import rollout
    r = rollout(model, x[0].numpy(), 2)
    assert r.shape == (2 * fam.K, H, W) and np.isfinite(r).all()


def test_untrained_sphere_backbone_is_exactly_persistence():
    """Its output layer is initialised to zero, so the residual branch starts at the identity: the untrained member
    returns the input block exactly, which is the sharpest statement that it predicts a *change*."""
    model = fam.make_member("sphere-backbone")
    x = torch.rand(2, fam.K, H, W)
    assert torch.equal(model(x), x)


def test_grid_pad_opens_r_and_replicates_theta():
    x = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W)
    p = fam.grid_pad(x, 1)
    assert p.shape == (1, 1, H + 2, W + 2)
    assert torch.equal(p[0, 0, 0], torch.zeros(W + 2))                    # r: zeros beyond the window
    assert torch.equal(p[0, 0, 1:-1, 0], x[0, 0, :, 0])                   # theta: the edge value, repeated
    assert torch.equal(p[0, 0, 1:-1, -1], x[0, 0, :, -1])


def test_make_member_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown member"):
        fam.make_member("transformer")


# ---------------------------------------------------------------------- the aggregation
def test_combine_is_the_convex_combination_and_the_identity_for_identical_fields():
    rng = np.random.default_rng(0)
    a, b = rng.random((3, H, W)), rng.random((3, H, W))
    assert np.array_equal(fam.combine([a, a], [0.5, 0.5]), a)              # identical members: exact identity
    assert np.allclose(fam.combine([a, a], [0.7, 0.3]), a)                 # ... and up to rounding otherwise
    assert np.allclose(fam.combine([a, b], [1.0, 1.0]), 0.5 * (a + b))      # weights are normalised
    shift = np.roll(a, 1, axis=2)
    assert np.allclose(fam.combine([a, shift], [0.25, 0.75]), 0.25 * a + 0.75 * shift)
    with pytest.raises(ValueError):
        fam.combine([a, b], [1.0])


def test_equal_weight_ensemble_returns_the_member_when_the_members_agree():
    members = {name: fam.make_member(name) for name in fam.MEMBERS}
    x = torch.rand(1, fam.K, H, W)
    twin = fam.EqualWeightEnsemble({"a": OffsetMember(0.0), "b": OffsetMember(0.0)})
    agree = x[:, -1:].expand(-1, fam.K, -1, -1)                            # what both members return
    assert torch.equal(twin(x), agree)                                     # both members agree: exact identity
    ens = fam.EqualWeightEnsemble(members)
    y = ens(x)
    assert torch.allclose(y, torch.stack([members[n](x) for n in fam.MEMBERS]).mean(0))


def test_aggregation_concentrates_on_the_member_with_the_known_best_error():
    """Three hand-made members around a trajectory that drifts 0.01 per frame: the deltas 0, 0.03 and 0.3 put the
    middle one closest to the truth over a block of five frames, and the weights must find it with no other hint."""
    codec, vol = _codec(), _volume()
    traj = _ramp(frames=120, rate=0.01)
    members = {"still": OffsetMember(0.0), "half": OffsetMember(0.03), "far": OffsetMember(0.3)}
    env = MassEnvelope(rate_max=1.0)
    ens = fam.adaptive_ensemble(members, codec, vol, 20.0, env, residual_lookahead=0.0, min_weight=0.0)
    queries = fam.forecast_queries(traj, blocks=2)
    refs = [traj[q["start"] + 1:q["start"] + 1 + 2 * fam.K] for q in queries]
    run = ens.run(queries, refs, keep_predictions=True)

    w = ens.weights()
    assert w["half"] == max(w.values()) and w["half"] > 0.6            # well above the uniform 1/3
    start_w = run.weights[2][ens.names.index("half")]
    assert w["half"] > start_w > 0.5                                   # and it grew as the cases arrived
    errors = {n: np.mean([np.abs(ens.experts[i].predict(q) - r).mean()
                          for q, r in zip(queries, refs, strict=True)]) for i, n in enumerate(ens.names)}
    assert errors["half"] == min(errors.values())                          # the known best is the measured best
    assert np.mean(run.ensemble_errors) <= errors["still"] + 1e-9          # and combining helps over the worst
    # the combination is exactly the aggregation of this module, with the weights the ensemble used
    for k, p in enumerate(run.predictions):
        names = ens.names
        assert np.allclose(p["prediction"],
                           fam.combine([p["expert_predictions"][n] for n in names],
                                       [run.weights[k][names.index(n)] for n in names]))


def test_the_horizon_orders_the_members_by_how_they_degrade():
    """A member whose error grows with lead has a shorter skill horizon against the same reference than one whose
    error stays put -- the property the trust-horizon figure and the docs rely on."""
    leads = 20.0 * (1 + np.arange(50))
    rng = np.random.default_rng(1)
    reference = np.full((30, 50), 0.02) + 1e-3 * rng.standard_normal((30, 50))
    flat = np.full((30, 50), 0.01) + 1e-3 * rng.standard_normal((30, 50))
    degrading = 0.01 * np.exp(leads / 400.0)[None] + 1e-3 * rng.standard_normal((30, 50))
    h_flat = fam.skill_against(flat, reference, leads, block=5).horizon
    h_bad = fam.skill_against(degrading, reference, leads, block=5).horizon
    assert h_flat > h_bad
    assert h_flat == float(leads[-1])                                      # better at every lead: the whole window
    res = fam.skill_against(degrading, reference, leads, block=5)
    assert res.diff[:, 2].min() < 0 < res.diff[:, 0].max()                 # worse somewhere, better somewhere


def test_mass_rate_residual_is_the_log_rate_of_an_exponential_drift():
    """The label-free tilt is |d ln M/dt|: on a field that decays exponentially it is the decay rate, exactly."""
    codec, vol = _codec(), _volume()
    rate = 0.05
    x0 = np.full((H, W), 0.5, np.float32)
    frames = np.stack([x0 + rate * t / (HI - LO) for t in range(6)])
    exact = np.log(10.0) * rate                                            # d ln M/dt of the exponential drift
    assert abs(fam.mass_rate_residual(frames, codec, vol, 1.0) - exact) < 1e-6   # float32 frames


# ---------------------------------------------------------------------- training
def test_train_member_keeps_the_contract_and_returns_a_history():
    traj = _trajectory(frames=24, rate=0.02, noise=0.002)
    from pinneapple_physics.blackhole.forecast import make_blocks
    X, Y = make_blocks(traj, fam.K)
    model = fam.make_member("sphere-backbone", widths=(8, 16), blocks=1, refine=8)
    hist = fam.train_member(model, X[:8], Y[:8], X[:4], Y[:4], fam.MemberTrainConfig(epochs=2, batch_size=4),
                            log=None)
    assert len(hist["train"]) == 2 and len(hist["val"]) == 2
    assert np.isfinite(hist["best_val"]) and hist["best_val"] <= max(hist["val"]) + 1e-9
    y = model(torch.from_numpy(X[:2]))
    assert y.shape == Y[:2].shape and torch.isfinite(y).all()


def test_forecast_queries_are_the_starts_of_the_protocol():
    traj = _trajectory(frames=40)
    qs = fam.forecast_queries(traj, blocks=2, start_every=5)
    assert [q["start"] for q in qs] == list(range(fam.K - 1, 40 - 2 * fam.K, 5))
    assert all(q["x0"].shape == (fam.K, H, W) for q in qs)


def test_trust_state_of_the_animation_bar():
    from pinneapple_physics.blackhole.figures import _trust_state
    assert _trust_state(0.9, 100.0, 50.0)[0] == "trusted"
    assert _trust_state(0.2, 100.0, 150.0)[0] == "fading"
    assert _trust_state(-0.1, 100.0, 400.0)[0] == "past the horizon"


def test_evaluate_skill_reports_the_curve_the_trust_bar_draws():
    """The animation's trust bar reads `acc_by_lead` and `acc_horizon_M` out of the skill report; both must exist and
    cover every lead the protocol scores."""
    from pinneapple_physics.blackhole.skill import evaluate_skill
    codec, vol = _codec(), _volume()
    train = [_trajectory(frames=40, rate=0.01, seed=0)]
    test = [_trajectory(frames=40, rate=0.01, seed=1)]
    members = {"sphere-backbone": fam.make_member("sphere-backbone", widths=(8, 16), blocks=1, refine=8)}
    rep = evaluate_skill({n: (m, codec) for n, m in members.items()}, train, train, test, frame_dt=20.0,
                         cell_volume=vol, blocks=2, start_every=2, dmd_ranks=(3,), log=lambda s: None)
    m, keys = rep["models"]["sphere-backbone"], ["persistence", "dmd", "climatology", "sphere-backbone"]
    assert len(m["acc_by_lead"]) == 2 * fam.K and np.all(np.isfinite(m["acc_by_lead"]))
    assert np.isfinite(m["acc_horizon_M"]) and 0.0 <= m["acc_horizon_M"] <= rep["leads_M"][-1]
    assert set(rep["mean_acc"]) == set(keys)
