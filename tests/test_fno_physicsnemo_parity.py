"""FNO-2D parity with PhysicsNeMo (#2529): the architecture options, the weight map and what the numbers mean.

The measurement itself is ``pinneapple_neural.physicsnemo_bridge.fno_compare``; these tests hold the claims made
around it: the new options leave the default architecture alone, the transplant refuses to compare two different
models, the advection-diffusion reference really is the exact propagator, and a missing PhysicsNeMo is reported as
"not compared" rather than as agreement.
"""
import numpy as np
import pytest
import torch

from pinneapple_neural.architectures.neural_operators.fno import FNO2d
from pinneapple_neural.physicsnemo_bridge import fno_compare as fc

HAS_PNEMO = fc.availability()["available"]


def _tiny(**kw):
    """Small enough to run in a second: 8 channels, 2 modes, 2 blocks, no padding."""
    arch = dict(width=8, modes=2, layers=2, padding=0, decoder_layer_size=4)
    arch.update(kw)
    return fc.FNOArch(**arch)


# --------------------------------------------------------------------------- defaults are untouched
def test_default_architecture_is_the_one_that_was_there():
    """The new arguments are opt-in: at their defaults the module tree and the state_dict names of ``FNO2d`` are
    the ones existing checkpoints were saved with."""
    m = FNO2d(1, 1, width=8, modes1=4, modes2=4, layers=2)
    assert sorted(m.state_dict()) == [
        "convs.0.weights1", "convs.0.weights2", "convs.1.weights1", "convs.1.weights2",
        "in_proj.bias", "in_proj.weight", "out_proj.bias", "out_proj.weight",
        "ws.0.bias", "ws.0.weight", "ws.1.bias", "ws.1.weight",
    ]
    assert isinstance(m.in_proj, torch.nn.Conv2d) and isinstance(m.out_proj, torch.nn.Conv2d)
    assert m.final_activation is True
    assert m(torch.randn(1, 1, 16, 16)).y.shape == (1, 1, 16, 16)


def test_default_spectral_init_is_a_scaled_complex_normal():
    m = FNO2d(1, 1, width=8, modes1=4, modes2=4, layers=1)
    w = m.convs[0].weights1
    assert w.is_complex() and w.shape == (8, 8, 4, 4)
    assert w.real.mean().item() == pytest.approx(0.0, abs=0.05)


def test_uniform_init_is_non_negative_and_scaled_like_li_et_al():
    """``spectral_init="uniform"`` draws on ``[0, scale)`` for the real and the imaginary part, which is the FNO
    paper's and PhysicsNeMo's initialisation; ``scale = 1/(in_c out_c)``."""
    m = FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1, spectral_init="uniform")
    w = m.convs[0].weights1
    scale = 1.0 / (8 * 8)
    assert w.real.min().item() >= 0.0 and w.imag.min().item() >= 0.0
    assert w.real.max().item() <= scale and w.imag.max().item() <= scale
    with pytest.raises(ValueError, match="normal"):
        FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1, spectral_init="gaussian")


def test_final_activation_false_leaves_the_last_block_linear():
    """With the spectral weights at zero and the bypass an identity, the model is ``out_proj(in_proj(u))`` when the
    last block is not activated and ``out_proj(gelu(in_proj(u)))`` when it is."""
    torch.manual_seed(0)
    u = torch.randn(2, 1, 6, 6)
    outs = {}
    for flag in (True, False):
        m = FNO2d(1, 1, width=4, modes1=1, modes2=1, layers=1, use_grid=False, final_activation=flag)
        with torch.no_grad():
            m.convs[0].weights1.zero_()
            m.convs[0].weights2.zero_()
            m.ws[0].weight.copy_(torch.eye(4)[:, :, None, None])
            m.ws[0].bias.zero_()
            stage = m.in_proj(u)
            outs[flag] = (m(u).y, m.out_proj(stage), m.out_proj(torch.nn.functional.gelu(stage)))
    assert torch.allclose(outs[True][0], outs[True][2])          # activated
    assert torch.allclose(outs[False][0], outs[False][1])        # linear


def test_lift_hidden_adds_the_two_step_lifting():
    plain = FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1)
    lifted = FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1, lift_hidden=4)
    assert isinstance(lifted.in_proj, torch.nn.Sequential)
    assert lifted.in_proj[0].out_channels == 4 and lifted.in_proj[2].out_channels == 8
    assert sum(p.numel() for p in lifted.parameters()) > sum(p.numel() for p in plain.parameters())


def test_decoder_hidden_replaces_the_projection_by_a_pointwise_mlp():
    m = FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1, decoder_hidden=4, decoder_activation="silu")
    assert isinstance(m.out_proj, torch.nn.Sequential) and isinstance(m.out_proj[1], torch.nn.SiLU)
    assert m.out_proj[0].out_channels == 4 and m.out_proj[2].out_channels == 1
    with pytest.raises(ValueError, match="activation"):
        FNO2d(1, 1, width=8, modes1=2, modes2=2, layers=1, decoder_hidden=4, decoder_activation="swish")


# --------------------------------------------------------------------------- the parity configuration
def test_the_parity_configuration_needs_no_physicsnemo():
    p = fc.pinneapple_fno2d(16, _tiny(), seed=0)
    assert p.module.net.final_activation is False
    assert isinstance(p.module.net.in_proj, torch.nn.Sequential)
    assert isinstance(p.module.net.out_proj, torch.nn.Sequential)
    shipped = fc.pinneapple_fno2d(16, _tiny(), seed=0, as_shipped=True)
    assert shipped.module.net.final_activation is True


def test_a_grid_too_coarse_for_the_requested_modes_is_refused():
    """Both implementations keep at most ``H // 2`` modes; comparing them on a grid where that bites would compare
    two truncations, so the builder refuses instead of reporting a number."""
    with pytest.raises(ValueError, match="do not fit"):
        fc.pinneapple_fno2d(8, _tiny(modes=12))
    with pytest.raises(ValueError, match="decoder_layers"):
        fc.FNOArch(width=8, modes=2, layers=2, padding=0, decoder_layer_size=4, decoder_layers=2)


def test_real_parameter_count_counts_a_complex_weight_twice():
    m = FNO2d(1, 1, width=4, modes1=1, modes2=1, layers=1, use_grid=False, final_activation=False)
    complex_numel = sum(p.numel() for p in m.parameters() if p.is_complex())
    assert complex_numel == 2 * 4 * 4 * 1 * 1                       # weights1 and weights2
    assert fc.real_parameter_count(m) == sum(p.numel() for p in m.parameters()) + complex_numel


def test_the_advection_diffusion_case_is_the_exact_propagator():
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

    case = fc.advdiff_case(n=32, n_samples=8, n_train=6, seed=3, n_modes=4, t_end=0.3)
    assert case.u0.shape == case.u1.shape == (8, 32, 32)
    assert len(case.train[0]) == 6 and len(case.test[0]) == 2
    direct = ob.advection_diffusion_2d(case.u0, case.velocity, case.kappa, case.t_end, 1)[-1]
    assert np.allclose(case.u1, direct, atol=0, rtol=0)             # the same call, not an approximation of it
    # the propagator conserves total mass exactly, and the band limit keeps it too
    assert abs(case.u1.mean() - case.u0.mean()) < 1e-12
    # and the input really is band-limited, so the error measured is the network's
    spectrum = np.abs(np.fft.fft2(case.u0[0]))
    outside = spectrum.copy()
    outside[1:5, :] = outside[-4:, :] = 0.0
    outside[:, 1:5] = outside[:, -4:] = 0.0
    assert outside.max() < 1e-9 * spectrum.max()


def test_score_is_zero_for_the_reference_and_blind_to_mass_alone():
    case = fc.advdiff_case(n=16, n_samples=4, n_train=2, seed=1, n_modes=3)
    perfect = fc.score(case.u1, case.u1)
    assert perfect["rel_l2_mean"] == 0.0 and perfect["mass_drift_mean"] == 0.0
    flat = np.full_like(case.u1, case.u1.mean())                    # right mass, no field
    marked = fc.score(flat, case.u1)
    assert marked["rel_l2_mean"] > 0.5 and marked["mass_drift_mean"] < 1e-12


def test_the_transfer_grid_carries_the_same_field_not_another_one():
    """Resolution transfer only means something if it is the same field on a finer grid: the inputs are band-limited,
    so the trigonometric interpolant is exact and the coarse nodes are a subset of the fine ones."""
    case = fc.advdiff_case(n=32, n_samples=6, n_train=4, seed=4, n_modes=4)
    fine, image = fc.transfer_pair(case, 64)
    assert fine.shape == image.shape == (2, 64, 64)
    coarse = np.stack([np.repeat(np.repeat(a, 2, 0), 2, 1) for a in case.test[0]])
    assert np.allclose(fine[:, ::2, ::2], coarse[:, ::2, ::2]), \
        "every other node of the fine grid must carry the coarse field's value"
    assert abs(image.mean() - fine.mean()) < 1e-12            # the exact propagator conserves mass there too
    assert np.array_equal(fc.transfer_pair(case, 32)[0], case.test[0])   # the same grid returns the case untouched


def test_predict_on_grid_follows_the_model_to_another_resolution():
    case = fc.advdiff_case(n=16, n_samples=8, n_train=6, seed=2, n_modes=3)
    pred = fc.pinneapple_fno2d(16, _tiny(), seed=0)
    pred.fit_normalisation(*case.train)
    on_grid = pred.predict(case.test[0])
    assert on_grid.shape == case.test[0].shape
    with pytest.raises(ValueError, match="grids"):
        pred.predict(np.ones((2, 24, 24)))                           # the wrapper refuses the wrong grid, on purpose
    transfer = fc.predict_on_grid(pred, np.ones((2, 24, 24)))
    assert transfer.shape == (2, 24, 24) and np.isfinite(transfer).all()


def test_a_trained_comparison_without_physicsnemo_reports_no_number(monkeypatch):
    monkeypatch.setattr(fc, "is_physicsnemo_available", lambda: False)
    case = fc.advdiff_case(n=8, n_samples=4, n_train=2, seed=0, n_modes=2)
    report = fc.compare_trained(case, seeds=(0,), steps=1)
    assert report.rows == {} and report.available["available"] is False
    assert "not installed" in report.available["reason"]


# --------------------------------------------------------------------------- with PhysicsNeMo installed
@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_the_weight_map_covers_every_parameter_of_the_parity_configuration():
    native = fc.physicsnemo_fno2d(16, _tiny(), seed=0)
    ours = fc.pinneapple_fno2d(16, _tiny(), seed=0)
    written = fc.transplant_fno2d_weights(native.physicsnemo_model, ours.module.net)
    assert sorted(written) == sorted(n for n, _ in ours.module.net.named_parameters())


@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_the_transplant_refuses_a_different_architecture():
    """Copying into the class's own defaults would leave half the target at its random initialisation; the map must
    say so instead of producing a number from a half-copied model."""
    native = fc.physicsnemo_fno2d(16, _tiny(), seed=0)
    shipped = fc.pinneapple_fno2d(16, _tiny(), seed=0, as_shipped=True)
    with pytest.raises(KeyError):
        fc.transplant_fno2d_weights(native.physicsnemo_model, shipped.module.net)


@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_same_weights_give_the_same_function():
    """The headline of the audit: at the float32 round-off of two FFT pipelines, the two implementations are the
    same function."""
    report = fc.fno2d_forward_parity(16, _tiny(), seed=0, grids=(16, 32))
    assert report.passed is True
    assert all(v <= fc.PARITY_TOLERANCE for v in report.relative_l2.values())
    assert report.parameters["pinneapple"] == report.parameters["physicsnemo"]
    assert set(report.relative_l2) == {"16", "32"}


@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_the_parity_number_does_not_depend_on_the_input_scale():
    """A relative measure, and it holds at both scales: the same weights on a field 100 times larger stay inside the
    tolerance (at this level the residual is float32 round-off, which is not itself scale-invariant, so what is
    checked is the tolerance, not the ratio)."""
    small = fc.fno2d_forward_parity(16, _tiny(), seed=0, grids=(16,), input_scale=1.0)
    large = fc.fno2d_forward_parity(16, _tiny(), seed=0, grids=(16,), input_scale=100.0)
    assert small.relative_l2["16"] <= fc.PARITY_TOLERANCE
    assert large.relative_l2["16"] <= fc.PARITY_TOLERANCE


@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_the_gradients_of_the_same_loss_agree():
    """The forward pass agreeing says the two models are the same function; the gradients agreeing says training them
    is the same computation. PhysicsNeMo freezes the two lifting biases, and the transplant mirrors that (14 of the
    16 parameters train on both sides)."""
    report = fc.fno2d_gradient_parity(16, _tiny(), seed=0)
    assert report.passed is True
    assert report.relative_l2 <= fc.PARITY_TOLERANCE
    assert report.relative_l2 < 1e-5 and report.worst_relative_l2 < 1e-4
    assert sum(_tiny_trainable(report)) == 14


def _tiny_trainable(report):
    return [v for v in report.trainable.values() if v]


@pytest.mark.skipif(not HAS_PNEMO, reason="physicsnemo is not installed")
def test_training_both_paths_from_the_same_weights_stays_on_the_same_trajectory():
    """After the same steps on the same samples with the same batch order, the two models are still the same model.

    The threshold is loose on purpose and that is a measured statement, not slack: Adam's step is
    ``lr * m / sqrt(v)``, which is scale free, so after a few steps a gradient component whose sign differs at the
    float32 round-off moves its weight by a full ``lr`` in the opposite direction. The trajectory agreement is
    therefore a weaker claim than the gradient parity above, and the benchmark reports it as such.
    """
    case = fc.advdiff_case(n=16, n_samples=12, n_train=10, seed=5, n_modes=3)
    report = fc.compare_trained(case, _tiny(), seeds=(0,), steps=3, batch_size=5, as_shipped=False)
    assert set(report.rows) == {"parity/pinneapple", "parity/physicsnemo", "parity/agreement"}
    agreement = report.rows["parity/agreement"]["relative_l2_between_models"]["mean"]
    assert agreement < 0.05, agreement
    ours = report.rows["parity/pinneapple"]["rel_l2_mean"]["mean"]
    theirs = report.rows["parity/physicsnemo"]["rel_l2_mean"]["mean"]
    assert ours == pytest.approx(theirs, rel=0.05)
