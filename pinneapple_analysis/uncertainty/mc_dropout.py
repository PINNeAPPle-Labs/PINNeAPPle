"""Monte Carlo Dropout for epistemic uncertainty quantification.

References
----------
Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning*. ICML.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from pinneapple_analysis.uncertainty.core import UQResult

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def _set_dropout_training(model: nn.Module, mode: bool) -> None:
    """Set only Dropout-family submodules to train/eval mode, leaving every
    other module -- especially BatchNorm/LayerNorm/InstanceNorm, whose
    behavior and running statistics must not change during inference-time MC
    sampling -- exactly as it already is.

    Calling ``model.train()`` on the *whole* model (the previous approach)
    was a real bug: it also switches BatchNorm to batch-statistics mode, and
    since buffer updates are not blocked by ``torch.no_grad()``, repeatedly
    calling a wrapped model's forward pass for MC sampling would silently
    and permanently mutate that model's ``running_mean``/``running_var`` as
    a side effect of merely asking for an uncertainty estimate.
    """
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train(mode)


@dataclass
class MCDropoutConfig:
    """Configuration for Monte Carlo Dropout.

    Attributes
    ----------
    n_samples:
        Number of stochastic forward passes used to estimate the posterior
        predictive distribution.  Higher values yield lower variance estimates
        at the cost of more compute.
    dropout_p:
        Dropout probability applied to the input of every ``nn.Linear`` layer
        but the first (the hidden activations) during stochastic inference.
    seed:
        Optional integer RNG seed for reproducible sampling.  When ``None``
        the global PyTorch RNG state is used.
    """

    n_samples: int = 100
    dropout_p: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.dropout_p < 1.0:
            raise ValueError(
                f"dropout_p must be in (0, 1); got {self.dropout_p}."
            )
        _check_n_samples(self.n_samples)


def _check_n_samples(n: int) -> None:
    # the sample standard deviation needs two samples; one gives NaN
    if n < 2:
        raise ValueError(f"n_samples must be >= 2 to estimate a standard deviation; got {n}.")


@contextlib.contextmanager
def _sampling_state(model: Any, seed: int | None):
    """Seed a forked RNG (the caller's global RNG stream is left untouched) and restore every submodule's
    train/eval flag on exit, even when the forward pass raises."""
    modes = [(m, m.training) for m in model.modules()] if isinstance(model, nn.Module) else []
    with torch.random.fork_rng(enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        try:
            yield
        finally:
            for m, mode in modes:
                m.training = mode


class MCDropoutWrapper(nn.Module):
    """Wraps any ``nn.Module`` and adds Monte Carlo Dropout at inference time.

    A thin ``nn.Dropout`` pre-forward hook is registered on every ``nn.Linear``
    layer but the first, so dropout acts on the hidden activations (Gal &
    Ghahramani 2016, dropout before each weight layer). It is never applied to
    the network's output: dropping an output unit zeroes the prediction itself
    and turns the "uncertainty" into ``|mean| sqrt(p / (1 - p))``. The input
    coordinates are not dropped either. The wrapper owns the dropout modules;
    the wrapped model's parameters are **never** modified.

    Parameters
    ----------
    model:
        Any trained ``nn.Module``.  Its weights are frozen from the wrapper's
        perspective (no gradient bookkeeping is changed here).
    config:
        :class:`MCDropoutConfig` instance controlling dropout probability and
        number of samples.

    Examples
    --------
    >>> cfg = MCDropoutConfig(n_samples=200, dropout_p=0.15)
    >>> wrapper = MCDropoutWrapper(trained_model, cfg)
    >>> result = wrapper.predict_with_uncertainty(x, device=torch.device("cuda"))
    >>> lower, upper = result.confidence_interval(alpha=0.05)
    """

    def __init__(self, model: nn.Module, config: MCDropoutConfig) -> None:
        super().__init__()
        self.config = config
        self.model = model

        # Collect handles so we can remove them later if needed.
        self._hooks: List[Any] = []
        self._dropout_modules: List[nn.Dropout] = []
        self._dropout_enabled: bool = False

        self._register_dropout_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_dropout_hooks(self) -> None:
        """Attach a pre-forward Dropout hook to every Linear sub-module but the first."""
        p = self.config.dropout_p
        linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            warnings.warn(
                f"MCDropoutWrapper: the model has {len(linears)} nn.Linear layer(s), so there is no hidden "
                "activation to drop; the samples vary only through the model's own dropout layers.",
                stacklevel=3,
            )

        for module in linears[1:]:
            dropout = nn.Dropout(p=p)
            self._dropout_modules.append(dropout)

            def _hook(
                mod: nn.Module,
                inputs: tuple,
                _d: nn.Dropout = dropout,
            ) -> tuple | None:
                if self._dropout_enabled:
                    return (_d(inputs[0]),) + tuple(inputs[1:])
                return None

            self._hooks.append(module.register_forward_pre_hook(_hook))

    # ------------------------------------------------------------------
    # Dropout toggling
    # ------------------------------------------------------------------

    def enable_dropout(self) -> None:
        """Enable stochastic dropout on all hooked Linear layers."""
        self._dropout_enabled = True
        for d in self._dropout_modules:
            d.train()

    def disable_dropout(self) -> None:
        """Disable dropout — model behaves deterministically."""
        self._dropout_enabled = False
        for d in self._dropout_modules:
            d.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Standard deterministic forward pass (dropout disabled).

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.

        Returns
        -------
        Tensor
            Model output, same shape as the wrapped model's output.
        """
        self.disable_dropout()
        out = self.model(x)
        if hasattr(out, "y"):
            # Some architectures (e.g. SIREN, PINN-family models) return a
            # ``ModelOutput``/``PINNOutput`` dataclass wrapper instead of a
            # plain tensor. Unwrap it so downstream tensor ops work.
            out = out.y
        return out

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run *n_samples* stochastic forward passes and aggregate statistics.

        The wrapped model is kept in ``eval()`` mode for the whole call --
        so BatchNorm/LayerNorm/etc. use their frozen running statistics and
        are never mutated by this sampling loop -- while only Dropout-family
        submodules (both the hook-injected ones and any the original
        architecture already had) are individually switched to stochastic
        mode.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.
        n_samples:
            Number of Monte Carlo samples.  Overrides ``config.n_samples``
            when provided.
        device:
            Target device.  Defaults to ``x.device``.

        Returns
        -------
        UQResult
            ``mean`` and ``std`` computed over the *n_samples* predictions;
            ``samples`` contains the raw stacked tensor of shape
            ``(n_samples, N, ...)``.
        """
        n = n_samples if n_samples is not None else self.config.n_samples
        _check_n_samples(n)
        if device is None:
            device = x.device

        x = x.to(device)
        self.model.to(device)

        # Collect samples with dropout active. Keep the whole model in eval
        # mode (BatchNorm etc. stay deterministic and unmutated) and only
        # flip Dropout-family submodules to train mode; every module's
        # train/eval flag is restored afterwards.
        preds: List[Tensor] = []
        with _sampling_state(self.model, self.config.seed):
            self.model.eval()
            _set_dropout_training(self.model, True)
            self.enable_dropout()
            try:
                for _ in range(n):
                    out = self.model(x)
                    if hasattr(out, "y"):
                        # Unwrap ModelOutput/PINNOutput-style wrapper dataclasses.
                        out = out.y
                    # Detach to avoid accumulating the computation graph.
                    preds.append(out.detach())
            finally:
                self.disable_dropout()

        # Stack → (n_samples, N, ...)
        samples = torch.stack(preds, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)

        return UQResult(
            mean=mean,
            std=std,
            samples=samples,
            metadata={
                "method": "mc_dropout",
                "n_samples": n,
                "dropout_p": self.config.dropout_p,
            },
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the wrapped model.

        Call this if you need to use the wrapped model independently after
        uncertainty estimation.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self) -> None:  # pragma: no cover
        # Best-effort cleanup when the wrapper is garbage-collected.
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Standalone convenience class
# ---------------------------------------------------------------------------

class MCDropout:
    """Convenience class that applies MC Dropout to any callable model.

    Unlike :class:`MCDropoutWrapper`, this class accepts **any callable**
    (not just ``nn.Module``), making it suitable for wrapped models, partial
    functions, or ONNX runtimes that expose a ``__call__`` interface.

    Parameters
    ----------
    model:
        Any callable mapping a ``Tensor`` to a ``Tensor``.
    config:
        :class:`MCDropoutConfig` instance.  When ``None``, default config is
        used.

    Notes
    -----
    Because this class cannot inject hooks into an arbitrary callable, it
    relies on the model having dropout layers that respect ``model.train()``
    and ``model.eval()`` (i.e. the model must be an ``nn.Module``).  For
    non-module callables, the stochastic variation comes solely from any
    randomness already present in the callable.

    Examples
    --------
    >>> mc = MCDropout(model, MCDropoutConfig(n_samples=50))
    >>> result = mc.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        config: Optional[MCDropoutConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or MCDropoutConfig()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run stochastic forward passes and return uncertainty estimates.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.
        device:
            Target device.  Defaults to ``x.device``.

        Returns
        -------
        UQResult
            Aggregated mean, std, and raw samples.
        """
        if device is None:
            device = x.device if isinstance(x, Tensor) else torch.device("cpu")

        x = x.to(device)

        preds: List[Tensor] = []
        with _sampling_state(self.model, self.config.seed):
            # If the callable is an nn.Module, keep it in eval mode (so
            # BatchNorm/etc. stay deterministic and unmutated) and activate only
            # its Dropout-family submodules -- see `_set_dropout_training`'s
            # docstring for why blindly calling `.train()` on the whole model is
            # wrong here. Every module's train/eval flag is restored on exit.
            if isinstance(self.model, nn.Module):
                self.model.eval()
                _set_dropout_training(self.model, True)
                self.model.to(device)

            for _ in range(self.config.n_samples):
                out = self.model(x)
                if hasattr(out, "y"):
                    # Unwrap ModelOutput/PINNOutput-style wrapper dataclasses.
                    out = out.y
                preds.append(out.detach())

        samples = torch.stack(preds, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)

        return UQResult(
            mean=mean,
            std=std,
            samples=samples,
            metadata={
                "method": "mc_dropout",
                "n_samples": self.config.n_samples,
                "dropout_p": self.config.dropout_p,
            },
        )
