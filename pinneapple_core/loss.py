"""``pp.loss``: functional loss terms and a balancer selectable by name.

Terms are plain functions returning a scalar tensor::

    pde(residual)                     mean squared PDE residual (optionally causal)
    boundary(pred, target)            boundary / initial condition mismatch
    conservation(quantity, target)    a conserved quantity or a divergence-free constraint
    energy(e, mode=...)               energy conserved, or not increasing
    symmetry(model, x, transform)     invariance / equivariance under a transformation
    supervised(pred, target, kind)    data fit: mse, mae, huber, relative_l2
    inverse(pred, observed, params)   data misfit plus a prior on the unknown parameters

``combine(terms, weights)`` sums them. For weights that change during training build a
:class:`Balancer` with a strategy name::

    total = pp.loss.combine({"pde": pde(r), "bc": boundary(u_b, g)}, {"pde": 1.0, "bc": 10.0})

    balancer = pp.loss.Balancer("gradnorm", names=["pde", "bc"], model=net)
    total = balancer({"pde": pde(r), "bc": boundary(u_b, g)}, step=i)
    total.backward()

``Balancer.strategies()`` lists the names: the fixed and learned weights, GradNorm, NTK, ReLoBRaLo,
SoftAdapt, augmented Lagrangian, inverse-Dirichlet, learning-rate annealing, PCGrad, an automatic switcher,
and ``"curriculum"`` (weights ramped over the first steps). Causal training of time-dependent residuals is
``pde(residual, t=t, causal_epsilon=...)``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = ["pde", "boundary", "conservation", "energy", "symmetry", "supervised", "inverse", "causal_weights",
           "combine", "Balancer", "list_strategies"]


# -- terms -------------------------------------------------------------------
def causal_weights(residual: Tensor, t: Tensor, n_chunks: int = 32, epsilon: float = 1.0) -> Tensor:
    """Per-point causal weights ``exp(-epsilon * sum of the mean squared residual of earlier time chunks)``
    (Wang et al. 2022), detached. ``t`` must be normalised to [0, 1]."""
    from pinneapple_neural.trainer.causal import CausalWeightScheduler

    sched = CausalWeightScheduler(n_time_chunks=n_chunks, epsilon=epsilon)
    return sched.compute_weights(t.reshape(-1), residual.detach()).to(residual.device, residual.dtype)


def pde(residual: Tensor, *, weights: Optional[Tensor] = None, t: Optional[Tensor] = None,
        causal_epsilon: Optional[float] = None, n_chunks: int = 32) -> Tensor:
    """Mean squared residual. ``weights`` (one per point) gives a weighted mean ``sum(w r^2) / sum(w)``;
    ``t`` with ``causal_epsilon`` computes causal weights from the residual itself."""
    r2 = residual.pow(2)
    if r2.ndim > 1:
        r2 = r2.mean(dim=tuple(range(1, r2.ndim)))
    if causal_epsilon is not None:
        if t is None:
            raise ValueError("causal weighting needs the time coordinate t (normalised to [0, 1])")
        if weights is not None:
            raise ValueError("pass either weights or causal_epsilon, not both")
        weights = causal_weights(residual, t, n_chunks, causal_epsilon)
    if weights is None:
        return r2.mean()
    w = weights.detach().reshape(-1)
    return (w * r2).sum() / (w.sum() + 1e-12)


def boundary(pred: Tensor, target: Union[Tensor, float] = 0.0) -> Tensor:
    """Mean squared mismatch with the prescribed boundary or initial values."""
    return (pred - target).pow(2).mean()


def supervised(pred: Tensor, target: Tensor, kind: str = "mse", *, delta: float = 1.0) -> Tensor:
    """Data fit. ``kind``: ``mse``, ``mae``, ``huber`` (``delta``) or ``relative_l2`` (norm of the error over
    the norm of the target, over the whole batch)."""
    if kind == "mse":
        return (pred - target).pow(2).mean()
    if kind == "mae":
        return (pred - target).abs().mean()
    if kind == "huber":
        return torch.nn.functional.huber_loss(pred, target, delta=delta)
    if kind == "relative_l2":
        return (pred - target).norm() / target.norm().clamp_min(1e-12)
    raise ValueError(f"unknown kind {kind!r}; choose mse, mae, huber or relative_l2")


def conservation(quantity: Tensor, target: Union[Tensor, float] = 0.0, scale: Optional[float] = None) -> Tensor:
    """Squared deviation of a conserved ``quantity`` (total mass, a flux balance, a divergence) from ``target``,
    divided by ``scale`` when the quantity is not O(1)."""
    d = quantity - target
    if scale is not None:
        d = d / scale
    return d.pow(2).mean()


def energy(e: Tensor, reference: Union[Tensor, float, None] = None, mode: str = "conserve") -> Tensor:
    """Energy along time, ``e`` of shape ``(T,)`` or ``(T, ...)`` ordered in time.

    * ``conserve``: mean squared deviation from ``reference`` (default: the first value).
    * ``nonincrease``: penalises rises, ``mean(relu(e[k+1] - e[k])^2)`` (dissipative systems).
    * ``nondecrease``: penalises drops (growth, accumulation).
    """
    if mode == "conserve":
        ref = e[:1] if reference is None else reference
        return (e - ref).pow(2).mean()
    if mode in ("nonincrease", "nondecrease"):
        d = e[1:] - e[:-1]
        d = d if mode == "nonincrease" else -d
        return torch.relu(d).pow(2).mean()
    raise ValueError("mode must be 'conserve', 'nonincrease' or 'nondecrease'")


def symmetry(model: Callable[[Tensor], Tensor], x: Tensor, transform: Callable[[Tensor], Tensor],
             output_transform: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
    """Penalty for breaking a symmetry: ``mean((f(T x) - S f(x))^2)``. ``S`` is ``output_transform``
    (identity by default, which is invariance; a rotation of a vector output makes it equivariance)."""
    out = model(x)
    expected = out if output_transform is None else output_transform(out)
    return (model(transform(x)) - expected).pow(2).mean()


def inverse(pred: Tensor, observed: Tensor, params: Optional[Union[Tensor, Iterable[Tensor]]] = None,
            prior: Union[Tensor, float, None] = None, reg: float = 0.0) -> Tensor:
    """Misfit to the observations plus ``reg * mean((params - prior)^2)`` on the unknown parameters (prior 0 by default)."""
    loss = (pred - observed).pow(2).mean()
    if params is not None and reg:
        ps = [params] if isinstance(params, Tensor) else list(params)
        pr = 0.0 if prior is None else prior
        loss = loss + reg * sum((p - pr).pow(2).mean() for p in ps) / max(len(ps), 1)
    return loss


# -- combining ---------------------------------------------------------------
def combine(terms: Mapping[str, Tensor], weights: Optional[Mapping[str, float]] = None) -> Tensor:
    """Weighted sum of named terms (weight 1 where not given). Weights for terms that are absent raise
    ``KeyError``: a misspelt name must not silently drop a loss term."""
    weights = dict(weights or {})
    extra = set(weights) - set(terms)
    if extra:
        raise KeyError(f"weights given for terms that are not in the loss: {sorted(extra)}; terms: {sorted(terms)}")
    if not terms:
        raise ValueError("combine needs at least one term")
    total = None
    for name, term in terms.items():
        t = weights.get(name, 1.0) * term
        total = t if total is None else total + t
    return total


_MODEL_STRATEGIES = {"gradnorm", "ntk", "inverse_dirichlet", "lr_annealing", "pcgrad", "auto"}
_SCHEDULER_STRATEGIES = ("fixed", "self_adaptive", "gradnorm", "loss_ratio", "ntk", "relobralo", "softadapt",
                         "augmented_lagrangian", "inverse_dirichlet", "lr_annealing", "pcgrad", "joint_adaptive", "auto")


def list_strategies() -> List[str]:
    return sorted(_SCHEDULER_STRATEGIES + ("curriculum",))


class Balancer:
    """Loss weighting chosen by name. Call it with the dict of terms to get the weighted total.

    Args:
        strategy: one of :func:`list_strategies`.
        names: the term names, e.g. ``["pde", "bc", "ic"]``; calling with other names raises.
        model: the network, needed by the strategies that read gradients (``gradnorm``, ``ntk``,
            ``inverse_dirichlet``, ``lr_annealing``, ``pcgrad``, ``auto``).
        weights: starting (or fixed) weight per term, default 1.
        update_every: steps between weight updates for the strategies that update periodically.
        schedule / steps / shape: for ``"curriculum"``, ``{"bc": (start, end)}`` ramped over ``steps`` steps
            (``shape`` is ``linear`` or ``cosine``); terms not in the schedule keep ``weights``.
        options: forwarded to the strategy's configuration (``alpha``, ``lr``, ``clip_min``, ``clip_max``, ``tau``, ...).
    """

    def __init__(self, strategy: str = "fixed", names: Optional[Sequence[str]] = None, model: Optional[torch.nn.Module] = None,
                 weights: Optional[Mapping[str, float]] = None, update_every: int = 100, *,
                 schedule: Optional[Mapping[str, Tuple[float, float]]] = None, steps: int = 1000, shape: str = "linear",
                 **options: Any) -> None:
        if strategy not in list_strategies():
            raise ValueError(f"unknown strategy {strategy!r}; choose from {list_strategies()}")
        if not names:
            raise ValueError("names must list the loss terms, e.g. ['pde', 'bc']")
        if strategy in _MODEL_STRATEGIES and model is None:
            raise ValueError(f"strategy {strategy!r} reads gradients of the network: pass model=")
        self.strategy = strategy
        self.names = list(names)
        self.model = model
        base = {n: 1.0 for n in self.names}
        base.update(weights or {})
        unknown = set(base) - set(self.names)
        if unknown:
            raise KeyError(f"weights for unknown terms {sorted(unknown)}; names are {self.names}")
        self._base = base
        self._step = 0
        self._last: Dict[str, float] = dict(base)
        self._sched = None
        self._history: Dict[str, List[float]] = {n: [] for n in self.names}
        if strategy == "curriculum":
            self.schedule = {k: (float(a), float(b)) for k, (a, b) in (schedule or {}).items()}
            bad = set(self.schedule) - set(self.names)
            if bad:
                raise KeyError(f"schedule for unknown terms {sorted(bad)}; names are {self.names}")
            if shape not in ("linear", "cosine"):
                raise ValueError("shape must be 'linear' or 'cosine'")
            self.steps, self.shape = max(int(steps), 1), shape
        else:
            from pinneapple_neural.trainer.weight_scheduler import WeightScheduler, WeightSchedulerConfig

            fields = set(WeightSchedulerConfig.__dataclass_fields__)
            cfg = WeightSchedulerConfig(method=strategy, initial_weights=dict(base), update_every=update_every,
                                        **{k: v for k, v in options.items() if k in fields})
            for k, v in options.items():
                if k not in fields:
                    setattr(cfg, k, v)  # strategy-specific knobs the config reads with getattr (tau, rho, ...)
            self._sched = WeightScheduler(model, self.names, cfg)

    def _curriculum_weights(self, step: int) -> Dict[str, float]:
        frac = min(step / self.steps, 1.0)
        if self.shape == "cosine":
            frac = 0.5 * (1.0 - math.cos(math.pi * frac))
        w = dict(self._base)
        for k, (a, b) in self.schedule.items():
            w[k] = a + (b - a) * frac
        return w

    def __call__(self, terms: Mapping[str, Tensor], step: Optional[int] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None) -> Tensor:
        if set(terms) != set(self.names):
            raise KeyError(f"terms {sorted(terms)} do not match the balancer's names {sorted(self.names)}")
        s = self._step if step is None else step
        self._step = s + 1
        if self.strategy == "curriculum":
            self._last = self._curriculum_weights(s)
            for k, v in self._last.items():
                self._history[k].append(v)
            return combine(terms, self._last)
        total = self._sched.step(dict(terms), s, optimizer)
        return total

    @property
    def weights(self) -> Dict[str, float]:
        """The weights applied at the last call (the starting weights before any call)."""
        if self._sched is not None:
            current = {k: float(v) for k, v in self._sched.current_weights().items()}
            if current:  # some strategies expose no weight dict
                return current
        return dict(self._last)

    def history(self) -> Dict[str, List[float]]:
        """Weight applied to each term at every call (where the strategy records it)."""
        if self._sched is not None:
            return dict(self._sched.weight_history())
        return {k: list(v) for k, v in self._history.items()}

    def weight_parameters(self) -> List[torch.nn.Parameter]:
        """Learnable weights (``self_adaptive``): optimise them by gradient *ascent*, see the example. Empty otherwise."""
        if self._sched is not None and self._sched.has_weight_optimizer:
            return list(self._sched.weight_params())
        return []

    @staticmethod
    def strategies() -> List[str]:
        return list_strategies()

    def __repr__(self) -> str:
        return f"Balancer(strategy={self.strategy!r}, names={self.names})"
