"""FNO-2D: PINNeAPPle's :class:`~pinneapple_neural.architectures.neural_operators.fno.FNO2d` against PhysicsNeMo's
``physicsnemo.models.fno.FNO``, on the same weights and on the same problem (#2529).

A comparison of two libraries by architecture name is not a parity measurement: two models both called "FNO" differ
in the lifting, in whether the last Fourier block is activated, in the decoder head and in the spectral weight
initialisation, so "FNO versus FNO, which is better" conflates the implementation with the architecture. This module
measures the two things separately.

**Do the two implement the same function?** :func:`transplant_fno2d_weights` copies a PhysicsNeMo FNO's weights into
an :class:`~pinneapple_neural.architectures.neural_operators.fno.FNO2d` built in the *parity configuration*
(:func:`pinneapple_fno2d`, :data:`PARITY_ARCH`: two-step lifting, no activation after the last Fourier block, a
pointwise MLP decoder, uniform spectral init), and :func:`fno2d_forward_parity` reports the relative L2 between the
two forward passes on the same input. Same weights and same function means one number near the float32 round-off;
a structural difference means one number near 1. Both are results.

**Do they reach the same number on the same problem?** :func:`compare_trained` trains both on the same samples with
the same optimiser steps, batch order and normalisation (:func:`pinneapple_analysis.cost.equal_budget.train_within_budget`)
and reports each side's error against the *exact* solution of the problem, plus their agreement. Two configurations
are run: the parity one (same weights, same function — the number should match) and each library's own defaults (the
question a user migrating actually asks).

The problem is the periodic advection-diffusion propagator on the unit square, ``u_t + c . grad u = kappa lap u``,
``u(0) = u0``, ``u(t_end)``; its solution for each Fourier mode is an exact multiplier
(``pinneapple_simulation.numerical_solvers.operator_benchmarks.advection_diffusion_2d``), so the reference is
analytical — no finite-volume solution, no downloaded dataset, no tolerance of its own — and total mass is conserved
exactly, which gives the physics QoI of the benchmark. The input fields are band-limited to ``n_modes`` so the exact
solution is representable by an FNO keeping ``12`` modes: the error measured is the network's, not the truncation's.

``physicsnemo`` is optional and this module is the only one in the bridge that needs it inside a measurement
function. Without it every function here still runs and the reports carry ``available: False`` with the reason —
"not installed" is never reported as "agrees".
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .adapters import is_physicsnemo_available
from .field_operators import (
    GridFieldPredictor,
    physicsnemo_fno,
    pinneapple_fno,
    train_grid_operator,
)

__all__ = [
    "PARITY_ARCH",
    "PARITY_TOLERANCE",
    "FNOArch",
    "ForwardParity",
    "AdvDiffCase",
    "TrainedComparison",
    "availability",
    "real_parameter_count",
    "physicsnemo_fno2d",
    "pinneapple_fno2d",
    "fno2d_weight_map",
    "transplant_fno2d_weights",
    "matched_pair",
    "fno2d_forward_parity",
    "GradientParity",
    "fno2d_gradient_parity",
    "advdiff_case",
    "transfer_pair",
    "predict_on_grid",
    "compare_trained",
]

PARITY_TOLERANCE = 1e-4
"""Relative L2 below which a transplanted PhysicsNeMo FNO and the PINNeAPPle FNO are called the same function.
Declared before the measurement: both compute the same arithmetic in float32 (two FFTs of the padded grid), and
``1e-4`` is about three orders of magnitude above the round-off the operations themselves carry (measured on an
Apple M-series CPU at 32x32, 64x64 and 96x96 grids: 5.9e-8, 6.8e-8, 7.7e-8 — see
``benchmarks/_out/fno_physicsnemo_parity.json``), so a structural difference is not hidden by the tolerance."""


@dataclass(frozen=True)
class FNOArch:
    """Hyper-parameters both implementations are built with.

    ``width`` is PhysicsNeMo's ``latent_channels`` and PINNeAPPle's ``width``; ``modes`` is its ``num_fno_modes``
    and this library's ``modes1 == modes2``; ``padding`` is the same domain padding in cells on both.
    ``decoder_layer_size`` is PhysicsNeMo's decoder width and this library's ``decoder_hidden``. PhysicsNeMo's
    lifting uses ``max(1, width // 2)`` channels in its hidden step, which is what :func:`pinneapple_fno2d` sets.

    The spectral convolution of either implementation keeps at most ``H // 2`` (respectively ``W // 2 + 1``) modes,
    so a parity run needs ``modes <= (n + 2 * padding) // 2``; :func:`physicsnemo_fno2d` refuses a coarser grid
    instead of silently comparing two different truncations.
    """

    width: int = 32
    modes: int = 12
    layers: int = 4
    padding: int = 8
    decoder_layer_size: int = 64
    decoder_layers: int = 1

    def __post_init__(self) -> None:
        if self.decoder_layers != 1:
            raise ValueError("the weight map covers a one-hidden-layer decoder (PhysicsNeMo's decoder_layers=1); "
                             f"got decoder_layers={self.decoder_layers}")
        for name in ("width", "modes", "layers", "decoder_layer_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if int(self.padding) < 0:
            raise ValueError(f"padding must be >= 0, got {self.padding}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


PARITY_ARCH = FNOArch()
"""The configuration of the benchmark's headline row: 32 channels, 12 modes, 4 blocks, 8 cells of padding, a
64-wide decoder — the FNO paper's reduced size, as in ``benchmarks/fno_padding_darcy.py``."""


def availability() -> dict[str, Any]:
    """Whether PhysicsNeMo can be used here, and the reason when it cannot (the shape every report carries)."""
    if is_physicsnemo_available():
        import physicsnemo

        return {"available": True, "provider": "physicsnemo", "version": getattr(physicsnemo, "__version__", ""),
                "reason": ""}
    return {"available": False, "provider": None, "version": "",
            "reason": "physicsnemo is not installed (pip install nvidia-physicsnemo): a parity measurement needs "
                      "both implementations, so no agreement is reported"}


def real_parameter_count(model: nn.Module) -> int:
    """Parameters counted as real numbers — a complex weight is two, which is how a complex ``cfloat`` tensor is
    stored and what PhysicsNeMo's ``(..., 2)`` real/imaginary layout makes explicit. Without it the two libraries'
    parameter counts differ by the number of spectral weights for no reason."""
    return int(sum(2 * p.numel() if p.is_complex() else p.numel() for p in model.parameters()))


def physicsnemo_fno2d(n: int, arch: FNOArch = PARITY_ARCH, *, seed: int | None = 0) -> GridFieldPredictor:
    """PhysicsNeMo's 2-D FNO on an ``n x n`` grid, coefficient -> solution, wrapped as a
    :class:`~pinneapple_neural.physicsnemo_bridge.field_operators.GridFieldPredictor` (so it trains, predicts and
    saves like the library's own). Raises the bridge's actionable ``ImportError`` when ``physicsnemo`` is missing."""
    _check_grid(n, arch)
    return physicsnemo_fno(n, latent_channels=arch.width, num_fno_layers=arch.layers, num_fno_modes=arch.modes,
                           decoder_layers=arch.decoder_layers, decoder_layer_size=arch.decoder_layer_size,
                           padding=arch.padding, seed=seed)


def pinneapple_fno2d(n: int, arch: FNOArch = PARITY_ARCH, *, seed: int | None = 0,
                     as_shipped: bool = False) -> GridFieldPredictor:
    """This library's ``FNO2d`` on the same grid.

    ``as_shipped=False`` (default) is the **parity configuration**: the physicsnemo architecture reproduced exactly
    (two-step lifting of ``width // 2`` channels, no activation after the last Fourier block, a pointwise
    ``SiLU(width -> decoder_layer_size -> 1)`` decoder, uniform spectral init), so a PhysicsNeMo state dict can be
    transplanted into it. ``as_shipped=True`` is the class's own defaults, which is what a user of this library gets
    when they ask for an FNO: single 1x1 lifting, GELU after every block, a linear projection.
    """
    if as_shipped:
        return pinneapple_fno(n, width=arch.width, modes=arch.modes, layers=arch.layers, padding=arch.padding,
                              seed=seed)
    _check_grid(n, arch)
    return pinneapple_fno(n, width=arch.width, modes=arch.modes, layers=arch.layers, padding=arch.padding,
                          seed=seed, lift_hidden=max(1, arch.width // 2), final_activation=False,
                          decoder_hidden=arch.decoder_layer_size, decoder_activation="silu",
                          spectral_init="uniform")


def _check_grid(n: int, arch: FNOArch) -> None:
    """Both spectral convolutions truncate to ``floor(H / 2)`` modes; a parity run on a grid too coarse for the
    requested modes would compare two different truncations, so refuse it here rather than report a number."""
    padded = int(n) + 2 * int(arch.padding)
    if arch.modes > padded // 2:
        raise ValueError(f"{arch.modes} modes do not fit a {n} x {n} grid with {arch.padding} cells of padding: the "
                         f"padded grid is {padded} wide and keeps at most {padded // 2} modes. Lower modes or "
                         f"padding, or use a finer grid.")


def fno2d_weight_map(source: nn.Module) -> list[tuple[str, str, str]]:
    """The correspondence between a PhysicsNeMo ``FNO``'s parameters and the parity-configuration ``FNO2d``'s, as
    ``(target, source, transform)`` with ``transform`` one of:

    ``"conv"``      same shape, a 1x1 convolution on both sides;
    ``"linear"``    PhysicsNeMo's ``nn.Linear`` ``(out, in)`` reshaped to a 1x1 convolution ``(out, in, 1, 1)``;
    ``"complex"``   PhysicsNeMo stores the spectral weights as a real/imaginary pair ``(..., 2)``, this library as
                    ``cfloat``.

    Both sides' spectral convolution keeps the same ``modes1 x modes2`` block with the same positive / negative
    corner convention, which is why the weights are interchangeable at all.
    """
    layers = len(source.spec_encoder.spconv_layers)
    out = [
        ("in_proj.0.weight", "spec_encoder.lift_network.0.conv.weight", "conv"),
        ("in_proj.0.bias", "spec_encoder.lift_network.0.conv.bias", "conv"),
        ("in_proj.2.weight", "spec_encoder.lift_network.2.conv.weight", "conv"),
        ("in_proj.2.bias", "spec_encoder.lift_network.2.conv.bias", "conv"),
    ]
    for i in range(layers):
        for key in ("weights1", "weights2"):
            out.append((f"convs.{i}.{key}", f"spec_encoder.spconv_layers.{i}.{key}", "complex"))
        out.append((f"ws.{i}.weight", f"spec_encoder.conv_layers.{i}.weight", "conv"))
        out.append((f"ws.{i}.bias", f"spec_encoder.conv_layers.{i}.bias", "conv"))
    out += [
        ("out_proj.0.weight", "decoder_net.layers.0.linear.weight", "linear"),
        ("out_proj.0.bias", "decoder_net.layers.0.linear.bias", "conv"),
        ("out_proj.2.weight", "decoder_net.final_layer.linear.weight", "linear"),
        ("out_proj.2.bias", "decoder_net.final_layer.linear.bias", "conv"),
    ]
    return out


def transplant_fno2d_weights(source: nn.Module, target: nn.Module) -> list[str]:
    """Copy a PhysicsNeMo ``FNO``'s weights into a parity-configuration ``FNO2d``, and its frozen/trainable split;
    returns the target names written.

    Every parameter of the target must be covered — a partial copy would leave part of the model at its own random
    initialisation and quietly turn an equivalence test into a similarity test — so a target parameter with no entry
    in :func:`fno2d_weight_map`, or a shape that does not agree, raises instead of being skipped.

    ``requires_grad`` is mirrored because PhysicsNeMo freezes the biases of its 2-D lifting layers
    (``Conv2dFCLayer.reset_parameters`` zeroes them and then sets ``requires_grad = False``, while its 1-D and 3-D
    counterparts leave the bias trainable): a model that trains them is not the same model, even when it starts from
    the same weights.
    """
    src, dst = dict(source.named_parameters()), dict(target.named_parameters())
    mapping = fno2d_weight_map(source)
    written: list[str] = []
    with torch.no_grad():
        for dest_name, src_name, transform in mapping:
            if dest_name not in dst:
                raise KeyError(f"the target FNO has no parameter {dest_name!r}: it is not in the parity "
                               f"configuration (see pinneapple_fno2d(as_shipped=False))")
            if src_name not in src:
                raise KeyError(f"the source model has no parameter {src_name!r}: is it a physicsnemo "
                               f"models.fno.FNO?")
            w = src[src_name].detach()
            if transform == "complex":
                if w.shape[-1] != 2:
                    raise ValueError(f"{src_name} is not stored as a real/imaginary pair: shape {tuple(w.shape)}")
                w = torch.view_as_complex(w.contiguous())
            elif transform == "linear":
                w = w[:, :, None, None]
            if tuple(w.shape) != tuple(dst[dest_name].shape):
                raise ValueError(f"{src_name} {tuple(w.shape)} does not fit {dest_name} "
                                 f"{tuple(dst[dest_name].shape)}: the two models are not the same architecture")
            dst[dest_name].copy_(w.to(dst[dest_name].dtype))
            dst[dest_name].requires_grad_(bool(src[src_name].requires_grad))
            written.append(dest_name)
    covered = set(written)
    missed = sorted(set(dst) - covered)
    if missed:
        raise ValueError(f"the weight map left {len(missed)} target parameter(s) at their own initialisation: "
                         f"{missed[:6]} (the target is not the parity configuration)")
    return written


def matched_pair(n: int, arch: FNOArch = PARITY_ARCH, *, seed: int | None = 0) -> dict[str, GridFieldPredictor]:
    """The two FNOs in the parity configuration, the PhysicsNeMo weights transplanted into the PINNeAPPle one.

    Both predictors then evaluate the same function on the same grid, so any difference in what they learn comes
    from the training or the autograd path, not from the architecture. Requires ``physicsnemo``."""
    native = physicsnemo_fno2d(n, arch, seed=seed)
    ours = pinneapple_fno2d(n, arch, seed=seed)
    transplant_fno2d_weights(native.physicsnemo_model, ours.module.net)
    return {"pinneapple": ours, "physicsnemo": native}


@dataclass
class ForwardParity:
    """What :func:`fno2d_forward_parity` measured: the two forward passes on the same input, per grid."""

    arch: dict[str, int]
    grids: list[int]
    relative_l2: dict[str, float]
    max_abs: dict[str, float]
    field_scale: dict[str, float]
    tolerance: float
    passed: bool | None
    parameters: dict[str, int]
    available: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def fno2d_forward_parity(n: int, arch: FNOArch = PARITY_ARCH, *, seed: int = 0, grids: Sequence[int] = (),
                         tolerance: float = PARITY_TOLERANCE, batch: int = 4,
                         input_scale: float = 1.0) -> ForwardParity:
    """Relative L2 between PhysicsNeMo's FNO and the PINNeAPPle ``FNO2d`` **with the same weights**, per grid.

    The weights are transplanted at ``seed`` (so the source is a reproducible model, not a network of zeros), and
    both are evaluated in ``eval`` mode on the same batch of uniform random fields of scale ``input_scale``. A
    resolution different from ``n`` is legitimate and is the point of the last row: the spectral layer keeps the same
    modes on a finer grid, so an implementation that truncated or cropped differently would show up here.

    ``passed`` is ``None`` — never ``True`` — when PhysicsNeMo is not installed: an absent comparison is not an
    agreement.
    """
    report = ForwardParity(arch=arch.to_dict(), grids=[], relative_l2={}, max_abs={}, field_scale={},
                           tolerance=float(tolerance), passed=None,
                           parameters={"pinneapple": 0, "physicsnemo": 0}, available=availability())
    if not report.available["available"]:
        return report
    pair = matched_pair(n, arch, seed=seed)
    ours, native = pair["pinneapple"], pair["physicsnemo"]
    report.parameters = {"pinneapple": real_parameter_count(ours.module.net),
                         "physicsnemo": real_parameter_count(native.module.net)}
    ours.module.eval()
    native.module.eval()
    gen = torch.Generator().manual_seed(int(seed))
    worst: dict[int, float] = {int(g): 0.0 for g in (grids or (n,))}
    for g in sorted(worst):
        x = input_scale * torch.randn(batch, int(g), int(g), generator=gen)   # (B, H, W), as the wrapper takes it
        a = native.module(x)
        b = ours.module(x)
        denom = float(a.norm()) or 1.0
        worst[g] = float((a - b).norm()) / denom
        report.max_abs[str(g)] = float((a - b).abs().max())
        report.field_scale[str(g)] = float(a.abs().max())
    report.grids = sorted(worst)
    report.relative_l2 = {str(g): v for g, v in worst.items()}
    report.passed = bool(all(v <= tolerance for v in worst.values()))
    return report


@dataclass
class GradientParity:
    """What :func:`fno2d_gradient_parity` measured: the two gradients of the same loss, per shared parameter."""

    arch: dict[str, int]
    relative_l2: float
    worst_parameter: str
    worst_relative_l2: float
    per_parameter: dict[str, float]
    trainable: dict[str, int]
    tolerance: float
    passed: bool | None
    available: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fno2d_gradient_parity(n: int, arch: FNOArch = PARITY_ARCH, *, seed: int = 0, batch: int = 4,
                          tolerance: float = PARITY_TOLERANCE) -> GradientParity:
    """Relative L2 between the **gradients** the two implementations compute for the same loss, per parameter.

    The stronger half of the parity claim: the forward pass agreeing says the two models are the same function, the
    gradients agreeing says training them is the same computation. Both are measured on one batch of uniform random
    fields with the FNO paper's loss (per-sample relative L2, mean over the batch), after the same transplant and
    with the same trainable set. Parameters either implementation freezes are left out of both sides.
    """
    report = GradientParity(arch=arch.to_dict(), relative_l2=float("nan"), worst_parameter="",
                            worst_relative_l2=float("nan"), per_parameter={}, trainable={}, tolerance=float(tolerance),
                            passed=None, available=availability())
    if not report.available["available"]:
        return report
    pair = matched_pair(n, arch, seed=seed)
    gen = torch.Generator().manual_seed(int(seed))
    x = torch.randn(batch, int(n), int(n), generator=gen)
    y = torch.randn(batch, int(n), int(n), generator=gen)
    native_net = pair["physicsnemo"].physicsnemo_model
    grads: dict[str, dict[str, torch.Tensor]] = {"pinneapple": {}, "physicsnemo": {}}
    for backend, model in (("pinneapple", pair["pinneapple"].module), ("physicsnemo", pair["physicsnemo"].module)):
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = (torch.linalg.vector_norm((out - y).flatten(1), dim=1)
                / torch.linalg.vector_norm(y.flatten(1), dim=1)).mean()
        loss.backward()
        grads[backend] = {k.removeprefix("net."): p.grad.detach().clone()
                          for k, p in model.named_parameters()
                          if p.grad is not None and p.requires_grad}
    num = den = 0.0
    for dest, src, transform in fno2d_weight_map(native_net):
        if dest not in grads["pinneapple"] or src not in grads["physicsnemo"]:
            continue                                        # frozen on one side (PhysicsNeMo's lifting biases)
        mine = grads["pinneapple"][dest]
        theirs = grads["physicsnemo"][src]
        if transform == "complex":
            theirs = torch.view_as_complex(theirs.contiguous())
        elif transform == "linear":
            theirs = theirs[:, :, None, None]
        mine = mine.to(torch.complex128 if mine.is_complex() else torch.float64)
        theirs = theirs.to(mine.dtype)
        diff, ref = float((mine - theirs).norm()), float(mine.norm())
        report.per_parameter[dest] = diff / ref if ref else 0.0
        num += diff * diff
        den += ref * ref
    report.trainable = {k: int(v.requires_grad) for k, v in pair["pinneapple"].module.net.named_parameters()}
    report.relative_l2 = (num / den) ** 0.5 if den else float("nan")
    if report.per_parameter:
        report.worst_parameter = max(report.per_parameter, key=report.per_parameter.get)
        report.worst_relative_l2 = report.per_parameter[report.worst_parameter]
    report.passed = bool(report.relative_l2 <= tolerance)
    return report


# --------------------------------------------------------------------------- the benchmark problem
@dataclass
class AdvDiffCase:
    """One draw of the advection-diffusion benchmark: the input fields, the exact solution, and the operator.

    ``u0`` and ``u1`` are dimensionless concentration fields on the periodic unit square, ``(n_samples, n, n)``, axis
    0 of the last two being ``x``. ``u1`` is the exact solution of ``u_t + c . grad u = kappa lap u`` from ``u0``
    after ``t_end``: the propagator is diagonal in Fourier space, so it is evaluated mode by mode
    (``operator_benchmarks.advection_diffusion_2d``) and carries no discretisation error of its own beyond
    float64 round-off. Total mass ``mean(u)`` is conserved by that operator, which is the QoI checked on the
    predictions.
    """

    n: int
    u0: np.ndarray
    u1: np.ndarray
    velocity: tuple[float, float]
    kappa: float
    t_end: float
    n_modes: int
    seed: int
    n_train: int

    @property
    def train(self) -> tuple[np.ndarray, np.ndarray]:
        return self.u0[: self.n_train], self.u1[: self.n_train]

    @property
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.u0[self.n_train:], self.u1[self.n_train:]

    def describe(self) -> dict[str, Any]:
        return {"n": self.n, "n_samples": int(len(self.u0)), "n_train": int(self.n_train),
                "n_test": int(len(self.u0) - self.n_train), "velocity": list(self.velocity),
                "kappa": self.kappa, "t_end": self.t_end, "n_modes_band": self.n_modes, "seed": self.seed,
                "units": "dimensionless concentration; velocity in box lengths per time unit; kappa in box "
                         "lengths squared per time unit; t_end in time units"}


def advdiff_case(n: int = 64, n_samples: int = 320, n_train: int = 256, *, seed: int = 7, n_modes: int = 6,
                 velocity: Sequence[float] = (0.7, 0.3), kappa: float = 0.01, t_end: float = 0.5) -> AdvDiffCase:
    """Draw ``n_samples`` fields and their exact advection-diffusion images (:class:`AdvDiffCase`).

    The inputs are Gaussian random fields (``operator_benchmarks.periodic_grf``) **band-limited to ``n_modes``**:
    every mode above ``n_modes`` in either direction is set to zero in Fourier space, and since the propagator is
    diagonal the exact solution is band-limited the same way. An FNO keeping ``n_modes`` or more modes can represent
    it, so the error the benchmark reports is the network's and the training's, not a truncation the architecture
    cannot avoid.
    """
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

    if n_modes > n // 2:
        raise ValueError(f"n_modes={n_modes} does not fit a {n} x {n} grid (at most {n // 2})")
    u0 = ob.periodic_grf(int(n_samples), (int(n), int(n)), alpha=2.0, tau=3.0, seed=seed)
    u0 = _band_limit(u0, int(n_modes))
    u1 = ob.advection_diffusion_2d(u0, tuple(float(v) for v in velocity), float(kappa), float(t_end), 1)[-1]
    return AdvDiffCase(n=int(n), u0=u0, u1=u1, velocity=(float(velocity[0]), float(velocity[1])),
                       kappa=float(kappa), t_end=float(t_end), n_modes=int(n_modes), seed=int(seed),
                       n_train=int(n_train))


def _band_limit(fields: np.ndarray, n_modes: int) -> np.ndarray:
    """Keep the Fourier modes ``|k_x| <= n_modes`` and ``|k_y| <= n_modes`` of each field (the propagator is
    diagonal, so the exact solution of a band-limited input is band-limited the same way)."""
    f = np.fft.fft2(fields, axes=(-2, -1))
    n = fields.shape[-1]
    k = np.fft.fftfreq(n, 1.0 / n)
    keep = (np.abs(k)[:, None] <= n_modes) & (np.abs(k)[None, :] <= n_modes)
    return np.real(np.fft.ifft2(f * keep, axes=(-2, -1)))


@torch.no_grad()
def predict_on_grid(pred: GridFieldPredictor, fields: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """The same trained model on a grid other than the one it was built for, in physical units.

    ``GridFieldPredictor.predict`` refuses a grid size different from the one the model was built at, on purpose (a
    model used on the wrong grid is usually a mistake). This is the documented way to make the resolution-transfer
    measurement: FNO's spectral layer keeps the same modes on a finer grid and regenerates its coordinate channels,
    so the model is defined there — what changes is how much of the field the extra cells resolve.
    """
    a = np.asarray(fields, dtype=np.float32)
    out = [pred.module(torch.from_numpy(a[i:i + batch_size])).numpy() for i in range(0, len(a), batch_size)]
    return np.concatenate(out).astype(np.float64)


@dataclass
class TrainedComparison:
    """What :func:`compare_trained` measured: each side's error against the exact solution and their agreement."""

    case: dict[str, Any]
    protocol: dict[str, Any]
    config: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    available: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_trained(case: AdvDiffCase | None = None, arch: FNOArch = PARITY_ARCH, *, seeds: Sequence[int] = (0,),
                    steps: int = 300, batch_size: int = 16, lr: float = 1e-3, threads: int | None = None,
                    transfer_grids: Sequence[int] = (), as_shipped: bool = True,
                    eval_at: Sequence[float] = ()) -> TrainedComparison:
    """Train both FNOs on the same samples for the same number of optimiser steps, score both against ``u1``.

    What is held equal, for both sides: the samples and their split, the input mean/std and output scale
    (``GridFieldPredictor.fit_normalisation``), the loss (per-sample relative L2, as in the FNO paper), Adam with the
    same learning rate and weight decay, the same cosine schedule over the same number of steps, the same batch order
    (seeded), and the number of CPU threads. What is reported per side: the relative L2 of the prediction against the
    **exact** solution on the held-out samples (mean, median, worst), the mass drift (the exact operator conserves
    total mass), and the wall time. Then the agreement between the two predictions.

    ``as_shipped`` adds the row in which each library uses its own default architecture (rather than the parity
    configuration) — the question a user migrating asks, answered with the same protocol. ``transfer_grids`` trains
    at ``case.n`` and additionally scores on a finer/coarser grid, using :func:`predict_on_grid`.

    ``eval_at`` records the held-out error at those fractions of the budget as well as at the end (evaluation time is
    not charged to the model, see ``train_within_budget``), so the report carries an error-versus-budget curve and
    not only the number at one arbitrary cut.
    """
    if not is_physicsnemo_available():
        return TrainedComparison(case=(case or advdiff_case()).describe(), protocol={}, config=arch.to_dict(),
                                 rows={}, available=availability())
    case = case or advdiff_case()
    a_tr, u_tr = case.train
    a_te, u_te = case.test
    transfers = {int(g): transfer_pair(case, int(g)) for g in transfer_grids}
    configs: dict[str, dict[str, bool]] = {"parity": {"pinneapple": False}}
    if as_shipped:
        configs["as_shipped"] = {"pinneapple": True}
    rows: dict[str, dict[str, Any]] = {}
    for config_name, flags in configs.items():
        per_seed: dict[str, dict[str, list[dict[str, Any]]]] = {
            backend: {"metrics": [], "predictions": []} for backend in ("pinneapple", "physicsnemo")}
        for seed in seeds:
            pair = {"pinneapple": pinneapple_fno2d(case.n, arch, seed=seed, as_shipped=flags["pinneapple"]),
                    "physicsnemo": physicsnemo_fno2d(case.n, arch, seed=seed)}
            if config_name == "parity":
                transplant_fno2d_weights(pair["physicsnemo"].physicsnemo_model, pair["pinneapple"].module.net)
            for backend, pred in pair.items():
                t0 = time.perf_counter()
                curve: list[float] = []

                def _evaluate() -> float:
                    curve.append(score(pred.predict(a_te), u_te)["rel_l2_mean"])
                    return curve[-1]

                train_grid_operator(pred, a_tr, u_tr, _steps(steps), batch_size=batch_size, lr=lr, seed=int(seed),
                                    threads=threads, evaluate=_evaluate, eval_at=eval_at)
                elapsed = time.perf_counter() - t0
                prediction = pred.predict(a_te)
                per_seed[backend]["predictions"].append(prediction)
                per_seed[backend]["metrics"].append({
                    "seed": int(seed),
                    "train_s": round(elapsed, 2),
                    "n_parameters": real_parameter_count(pred.module.net),
                    "curve": curve,
                    **score(prediction, u_te),
                    **{f"transfer_{g}": score(predict_on_grid(pred, fields), ref)["rel_l2_mean"]
                       for g, (fields, ref) in transfers.items()},
                })
        for backend, data in per_seed.items():
            rows[f"{config_name}/{backend}"] = {"config": {**flags, "parity_weights":
                                                           config_name == "parity"}, **_summarise(data["metrics"])}
        rows[f"{config_name}/agreement"] = {"config": {**flags, "parity_weights": config_name == "parity"},
                                            **_summarise([
                                                {"seed": int(s),
                                                 "relative_l2_between_models": _rel_l2(
                                                     per_seed["pinneapple"]["predictions"][i],
                                                     per_seed["physicsnemo"]["predictions"][i])}
                                                for i, s in enumerate(seeds)])}
    return TrainedComparison(case=case.describe(),
                             protocol={"steps": int(steps), "batch_size": int(batch_size), "lr": float(lr),
                                       "optimiser": "Adam, weight decay 1e-5, cosine schedule over the steps",
                                       "loss": "per-sample relative L2, mean over the batch",
                                       "eval_at": [float(f) for f in eval_at],
                                       "seeds": [int(s) for s in seeds], "threads": threads},
                             config=arch.to_dict(), rows=rows, available=availability())


def _steps(n: int):
    from pinneapple_analysis.cost.equal_budget import ComputeBudget

    return ComputeBudget(steps=int(n))


def transfer_pair(case: AdvDiffCase, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The held-out samples of ``case`` and their exact images on an ``n x n`` grid.

    On another grid it is the **same** field: the inputs are band-limited (``case.n_modes``), so the trigonometric
    interpolant of one grid evaluated at the nodes of the other is exact, not an approximation
    (``operator_benchmarks.FourierInterpolant``), and since the propagator is diagonal the exact image follows from
    applying it on the finer grid. Interpolating the *input* and recomputing the reference — rather than redrawing a
    field of the same family — is what makes the transfer measurement a statement about the operator on a new grid.
    """
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

    if int(n) == case.n:
        return case.test
    m = int(n)
    a_te, _ = case.test
    nodes = (np.arange(m) / m).astype(np.float64)
    gx, gy = np.meshgrid(nodes, nodes, indexing="ij")
    x = torch.tensor(gx.reshape(-1, 1), dtype=torch.float64)
    y = torch.tensor(gy.reshape(-1, 1), dtype=torch.float64)
    fine = np.stack([ob.FourierInterpolant(a, max_modes=case.n_modes, dtype=torch.float64)(x, y)
                     .reshape(m, m).numpy() for a in a_te])
    image = ob.advection_diffusion_2d(fine, case.velocity, case.kappa, case.t_end, 1)[-1]
    return fine, image


def _rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = float(np.linalg.norm(np.ravel(pred - ref)))
    den = float(np.linalg.norm(np.ravel(ref))) or 1.0
    return num / den


def score(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Relative L2 of each sample against the exact solution (mean, median, worst) and the mass drift.

    ``mass_drift`` is ``|mean(prediction) - mean(reference)| / mean(|reference|)`` per sample, averaged: the exact
    advection-diffusion propagator conserves total mass on the periodic square, so a model that gets the field
    roughly right and the mass wrong is a model whose conservation the field error alone does not show.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    per = [float(np.linalg.norm(p - r)) / (float(np.linalg.norm(r)) or 1.0) for p, r in zip(pred, ref, strict=True)]
    drift = [abs(float(p.mean()) - float(r.mean())) / ((float(np.abs(r).mean())) or 1.0)
             for p, r in zip(pred, ref, strict=True)]
    return {"rel_l2_mean": float(np.mean(per)), "rel_l2_median": float(np.median(per)),
            "rel_l2_worst": float(np.max(per)), "mass_drift_mean": float(np.mean(drift))}


def _summarise(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation over seeds, per measured quantity (the scatter is what a single seed hides)."""
    keys = [k for k, v in seed_rows[0].items() if isinstance(v, (int, float)) and k != "seed"]
    out: dict[str, Any] = {"seeds": [r["seed"] for r in seed_rows]}
    for k in keys:
        vals = np.array([r[k] for r in seed_rows], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                  "min": float(vals.min()), "max": float(vals.max())}
    out["per_seed"] = seed_rows
    return out
