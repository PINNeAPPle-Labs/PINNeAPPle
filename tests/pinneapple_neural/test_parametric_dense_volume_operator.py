"""Tests for pinneapple_neural.workflows.parametric_dense_volume_operator.

The real, load-bearing test here is ``test_generalizes_to_held_out_theta``
-- ROADMAP.md's own stated bar for this whole track ("does it actually
generalize across [the parameter], not just memorize [a few] cases") is
directly checked, quantitatively: a model trained on thetas {0.0, 1.0,
3.0} is queried at a held-out theta=1.5 it never saw a training case for,
and its prediction must be closer to the TRUE theta=1.5 dynamics than
either training boundary's own dynamics is -- i.e. the theta channel is
doing real conditioning work, not being ignored (which would make the
network fall back to some case-averaged blend independent of theta).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_neural.workflows.parametric_dense_volume_operator import (
    ParametricCase,
    ParametricDenseVolumeOperatorConfig,
    train_parametric_dense_volume_operator,
    load_parametric_dense_volume_operator,
    predict_single_step_parametric,
    rollout_parametric_dense_volume_operator,
)


def _make_sequence(T: int, D: int, H: int, W: int, theta: float, seed: int = 0) -> np.ndarray:
    """A deterministic sequence whose decay/nonlinearity strength is
    controlled by ``theta`` -- a stand-in for "Re_tau changes the flow's
    real dynamics", smooth and monotonic in theta so interpolation is a
    meaningful, checkable question (not chaos-sensitive)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, D, H, W)).astype(np.float32)
    decay = 0.97 - 0.03 * theta
    nl = 0.02 * (1.0 + theta)
    seq = [x[0]]
    for _ in range(T - 1):
        x = np.roll(x, shift=1, axis=-1) * decay + nl * np.tanh(x)
        seq.append(x[0])
    return np.stack(seq, axis=0)[:, None, :, :, :]  # (T, 1, D, H, W)


def _cases(thetas, T=25, D=6, H=6, W=6, seed=1):
    return [
        ParametricCase(name=f"theta_{t}", frames=_make_sequence(T, D, H, W, theta=t, seed=seed), theta=[t])
        for t in thetas
    ]


def test_train_and_predict_smoke():
    cases = _cases([1.0, 2.0, 3.0])
    cfg = ParametricDenseVolumeOperatorConfig(
        state_channels=1, theta_dim=1, theta_log_scale=False,
        width=4, modes=2, layers=2, epochs=6, batch_size=2,
        pushforward_warmup_epochs=2, pushforward_max_steps=2, pushforward_prob=0.5,
        log_every=2, device="cpu",
    )
    ckpt, history = train_parametric_dense_volume_operator(cases, cfg, seed=0)

    assert len(history) > 0
    assert all(np.isfinite(h["train_mse"]) for h in history)
    assert "theta_min" in ckpt.meta and "theta_span" in ckpt.meta

    net = load_parametric_dense_volume_operator(ckpt)
    assert net is not None

    frame0 = cases[0].frames[0]
    pred = predict_single_step_parametric(ckpt, frame0, theta=[1.0])
    assert pred.shape == frame0.shape
    assert np.isfinite(pred).all()

    rollout = rollout_parametric_dense_volume_operator(ckpt, frame0, theta=[1.0], n_steps=4)
    assert rollout.shape == (5, 1, 6, 6, 6)
    assert np.allclose(rollout[0], frame0)


def test_rejects_single_case():
    cases = _cases([1.0])
    cfg = ParametricDenseVolumeOperatorConfig(state_channels=1, theta_dim=1, epochs=1, device="cpu")
    with pytest.raises(ValueError):
        train_parametric_dense_volume_operator(cases, cfg, seed=0)


def test_generalizes_to_held_out_theta():
    """The real test: interpolating to an unseen theta beats not
    conditioning on theta at all."""
    torch.manual_seed(0)
    train_thetas = [0.0, 1.0, 3.0]
    held_out_theta = 1.5
    D, H, W = 8, 8, 8
    T = 40
    n_rollout_steps = 12

    train_cases = _cases(train_thetas, T=T, D=D, H=H, W=W, seed=7)
    cfg = ParametricDenseVolumeOperatorConfig(
        state_channels=1, theta_dim=1, theta_log_scale=False,
        width=8, modes=3, layers=2, epochs=80, batch_size=4,
        pushforward_warmup_epochs=15, pushforward_max_steps=4, pushforward_prob=0.6,
        log_every=80, device="cpu",
    )
    ckpt, _ = train_parametric_dense_volume_operator(train_cases, cfg, seed=42)

    # Ground truth at the held-out theta, started from ITS OWN real initial frame
    # (same seed as training cases so the initial condition distribution matches).
    true_seq = _make_sequence(n_rollout_steps + 1, D, H, W, theta=held_out_theta, seed=7)

    rollout_correct_theta = rollout_parametric_dense_volume_operator(
        ckpt, true_seq[0], theta=[held_out_theta], n_steps=n_rollout_steps,
    )
    rmse_correct = float(np.sqrt(np.mean((rollout_correct_theta[-1] - true_seq[-1]) ** 2)))

    # Same trained network, same starting frame, but fed a WRONG theta (a training
    # boundary) at every step -- if the network is actually using theta as real
    # conditioning (not ignoring it), this should be noticeably worse.
    rmse_wrong = []
    for wrong_theta in (0.0, 3.0):
        rollout_wrong = rollout_parametric_dense_volume_operator(
            ckpt, true_seq[0], theta=[wrong_theta], n_steps=n_rollout_steps,
        )
        rmse_wrong.append(float(np.sqrt(np.mean((rollout_wrong[-1] - true_seq[-1]) ** 2))))

    assert np.isfinite(rmse_correct)
    assert all(np.isfinite(r) for r in rmse_wrong)
    assert rmse_correct < min(rmse_wrong), (
        f"rollout at the correct (held-out, interpolated) theta={held_out_theta} was not more "
        f"accurate than the same model fed a wrong training-boundary theta: "
        f"rmse_correct={rmse_correct:.4g}, rmse_wrong={rmse_wrong}"
    )
