"""Tests for :class:`pinneapple_worldmodel.mega_model.PhysicsFoundationModel`.

The foundation model is the mega-model the 5-stage pipeline assembles in
stage 4, and it had no test of its own.  These tests pin down the parts of
its documented contract that can be checked structurally and cheaply:

* ``forward(state_t, descriptor)`` returns the *same grid* it was given, for
  1-D and 2-D grids, multi-field states, and odd (non power-of-two) grids —
  the multi-scale skip pools by 2 and must pad/crop back
  (:meth:`PhysicsFoundationModel._match_size`).
* the physics context is *not* ignored: changing the descriptor changes the
  prediction by far more than float32 noise, and the descriptor receives a
  non-zero gradient.  This is the check that would fail if the FiLM
  conditioning (or the descriptor encoder feeding it) were dropped.
* eval-mode forward is deterministic, ``rollout`` is the documented
  ``(B, n_steps, C, *grid)`` stack, and ``predict_with_uncertainty`` really
  produces a spread when dropout is on.
* the parameter budget grows with ``context_dim`` (the context encoder is a
  real path, not a pass-through), and ``freeze_backbone`` leaves exactly the
  LoRA adapters trainable.
* ``save`` / ``load`` round-trips bit-exactly.

All models here are deliberately tiny (width 8-16, depth 1-2, 4 Fourier
modes); the point is the contract, not accuracy.  Single-threaded on CPU:
the repo's tiny-network convention, and it keeps a busy runner out of the
way.
"""
from __future__ import annotations

import pytest
import torch

from pinneapple_worldmodel.mega_model import FoundationConfig, PhysicsFoundationModel

torch.set_num_threads(1)

DESCRIPTOR = {"scenario": "ns2d_cavity", "Re": 500.0, "alpha": 0.01}


def _config(**overrides) -> FoundationConfig:
    """A tiny but structurally complete foundation config."""
    base = dict(
        n_modes=4,
        width=8,
        depth=2,
        context_dim=8,
        n_heads=2,
        n_context_layers=1,
        lora_rank=2,
        dropout=0.0,
        device="cpu",
    )
    base.update(overrides)
    return FoundationConfig(**base)


def _model(grid_shape=(16, 16), n_fields: int = 1, **overrides) -> PhysicsFoundationModel:
    torch.manual_seed(0)
    return PhysicsFoundationModel(
        _config(**overrides), n_fields=n_fields, grid_shape=grid_shape
    ).eval()


# ---------------------------------------------------------------------------
# Forward: output grid == input grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "grid_shape, n_fields",
    [
        ((32,), 1),        # 1-D, even length (pool/upsample lines up exactly)
        ((15,), 1),        # 1-D, odd length (skip must pad back to 15)
        ((16, 16), 1),     # 2-D, the default grid of the built-in scenarios
        ((15, 17), 2),     # 2-D, odd in both directions, two fields
    ],
)
def test_forward_preserves_the_input_grid(grid_shape, n_fields):
    """The operator is a one-step map on fields: (B, C, *grid) → (B, C, *grid),
    including the odd-grid cases where the half-resolution skip has to be
    cropped/padded back."""
    model = _model(grid_shape, n_fields)
    state = torch.randn(2, n_fields, *grid_shape)

    out = model(state, DESCRIPTOR)

    assert out.shape == state.shape
    assert out.dtype == state.dtype
    assert torch.isfinite(out).all()


def test_forward_without_descriptor_runs_as_a_context_free_operator():
    """``descriptor=None`` is documented as "no physics context".  Note that
    it does *not* go through the descriptor encoder (the ``None`` branch feeds
    a zero vector straight to FiLM), so it is not the same thing as encoding a
    zero-scalar descriptor — it must simply run and keep the grid."""
    model = _model((16,), n_fields=1)
    state = torch.randn(2, 1, 16)

    out = model(state)

    assert out.shape == state.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out, model(state))


def test_forward_accepts_a_1d_descriptor_for_a_batch():
    """A single ``(D,)`` descriptor is broadcast over the batch (documented
    in ``forward``), so it must equal the same descriptor stacked B times."""
    model = _model((16,), n_fields=1)
    state = torch.randn(3, 1, 16)
    descriptor = torch.linspace(0.0, 1.0, 8)

    broadcast = model(state, descriptor)
    stacked = model(state, descriptor.unsqueeze(0).expand(3, -1))

    assert broadcast.shape == state.shape
    torch.testing.assert_close(broadcast, stacked)


def test_dict_and_tensor_descriptors_encode_the_same_scalar_vector():
    """Both documented input modes — a dict of scalars and a pre-encoded
    ``(B, D)`` tensor — feed the same scalar embedding.  The two paths differ
    only in how the (B, max_scalars) tensor is materialised (an expanded view
    vs. a padded copy), so they agree to float32 rounding, not bit-exactly."""
    model = _model((16,), n_fields=1)
    state = torch.randn(2, 1, 16)

    from_dict = model(state, {"Re": 100.0})
    from_tensor = model(state, torch.tensor([[100.0], [100.0]]))

    torch.testing.assert_close(from_dict, from_tensor, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# The physics context must actually reach the prediction
# ---------------------------------------------------------------------------

def test_different_descriptors_give_different_predictions():
    """Context conditioning (FiLM on every FNO block) is the whole point of a
    *physics* foundation model: two different physics descriptors over the
    same state must not produce the same next state.  For this configuration
    the two predictions differ by 7.6e-2 in relative norm, and the two
    *equivalent* descriptor paths above agree to ~1e-6, so the 1e-2 threshold
    is an order of magnitude below the real effect and far above the noise."""
    model = _model((32,), n_fields=1)
    state = torch.randn(2, 1, 32)
    generator = torch.Generator().manual_seed(1234)
    ctx_a = torch.randn(2, 8, generator=generator)
    ctx_b = torch.randn(2, 8, generator=generator) * 3.0

    out_a = model(state, ctx_a)
    out_b = model(state, ctx_b)
    relative_change = (out_a - out_b).norm().item() / out_a.norm().item()

    assert relative_change > 1e-2


def test_descriptor_receives_gradient():
    """A context that merely scales the output through a constant would still
    pass the test above; this one requires the descriptor to be on the graph
    the loss depends on (i.e. the context encoder is trained, not skipped)."""
    model = _model((16, 16), n_fields=1)
    state = torch.randn(2, 1, 16, 16)
    descriptor = torch.randn(2, 8, requires_grad=True)

    model(state, descriptor).sum().backward()

    assert descriptor.grad is not None
    assert torch.isfinite(descriptor.grad).all()
    assert descriptor.grad.abs().sum().item() > 0.0


def test_parameter_count_grows_with_the_context_path():
    """``context_dim`` sizes the descriptor encoder and every FiLM adapter;
    widening it must add parameters.  A model that ignored its context would
    have identical parameter counts."""
    narrow = PhysicsFoundationModel(_config(context_dim=8), n_fields=1, grid_shape=(16,))
    wide = PhysicsFoundationModel(_config(context_dim=32), n_fields=1, grid_shape=(16,))

    assert wide.parameter_count() > narrow.parameter_count()
    assert narrow.parameter_count() == sum(
        p.numel() for p in narrow.parameters() if p.requires_grad
    )


# ---------------------------------------------------------------------------
# Determinism, rollout, uncertainty
# ---------------------------------------------------------------------------

def test_eval_mode_is_deterministic():
    """Dropout is the only stochastic op; ``eval()`` must switch it off so the
    same (state, descriptor) gives the identical prediction."""
    model = _model((16, 16), n_fields=1, dropout=0.5)  # non-zero on purpose
    model.eval()
    state = torch.randn(2, 1, 16, 16)

    first = model(state, DESCRIPTOR)
    second = model(state, DESCRIPTOR)

    assert torch.equal(first, second)
    assert not model.training


def test_rollout_stacks_the_documented_number_of_steps():
    """``rollout(state_0, descriptor, n_steps)`` → (B, n_steps, C, *grid), and
    each step equals the corresponding chained ``forward`` call."""
    model = _model((16, 16), n_fields=1)
    state_0 = torch.randn(2, 1, 16, 16)

    states = model.rollout(state_0, DESCRIPTOR, n_steps=3)

    assert states.shape == (2, 3, 1, 16, 16)
    step_1 = model(state_0, DESCRIPTOR)
    torch.testing.assert_close(states[:, 0], step_1)
    torch.testing.assert_close(states[:, 1], model(step_1, DESCRIPTOR))


def test_mc_dropout_uncertainty_returns_a_mean_and_a_real_spread():
    """``predict_with_uncertainty`` documented output: (mean, std), both the
    shape of the state.  With dropout on, the samples must actually differ —
    otherwise the "uncertainty" is a constant zero and useless."""
    model = _model((32,), n_fields=1, dropout=0.5)
    model.eval()
    state = torch.randn(2, 1, 32)

    mean, std = model.predict_with_uncertainty(state, DESCRIPTOR, n_samples=5)

    assert mean.shape == state.shape
    assert std.shape == state.shape
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert std.mean().item() > 0.0
    assert not model.training  # left in eval mode for the caller


@pytest.mark.parametrize("grid_shape", [(4, 4, 4), (8, 8, 8)])
def test_three_dimensional_grids_are_rejected(grid_shape):
    """3-D grids are documented as not implemented: constructing the model
    must say so instead of failing later inside a convolution."""
    with pytest.raises(ValueError, match="1D and 2D"):
        PhysicsFoundationModel(_config(), n_fields=1, grid_shape=grid_shape)


# ---------------------------------------------------------------------------
# LoRA adapters and persistence
# ---------------------------------------------------------------------------

def test_freeze_backbone_leaves_only_the_lora_adapters_trainable():
    """The meta-learning contract: ``freeze_backbone`` keeps the LoRA A/B
    matrices (the per-task fast weights) and nothing else."""
    model = _model((16,), n_fields=1)
    total = model.parameter_count()
    assert len(list(model.lora_parameters())) > 0

    model.freeze_backbone()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "LoRA adapters should stay trainable"
    assert all(("lora_A" in name or "lora_B" in name) for name in trainable)
    assert model.parameter_count() < total
    assert model.parameter_count() == sum(p.numel() for p in model.lora_parameters())


def test_save_and_load_round_trip(tmp_path):
    """The pipeline saves ``mega_model.pt`` in stage 4 and the documented way
    to read it back is ``PhysicsFoundationModel.load``; the reloaded model
    must reproduce the saved model's predictions exactly (bit-for-bit: same
    weights, same graph, CPU)."""
    model = _model((16, 16), n_fields=2)
    path = tmp_path / "mega_model.pt"
    model.save(str(path))
    state = torch.randn(1, 2, 16, 16)
    expected = model(state, DESCRIPTOR)

    reloaded = PhysicsFoundationModel.load(str(path))

    assert reloaded.n_fields == model.n_fields == 2
    assert reloaded.grid_shape == model.grid_shape == (16, 16)
    assert torch.equal(reloaded(state, DESCRIPTOR), expected)
