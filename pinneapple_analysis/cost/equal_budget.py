"""Equal-budget comparison of methods: a compute budget, a training loop that stops at it, FLOPs, search, amortised cost.

Comparing a PINN with a neural operator or a graph network is only fair when every method gets the same compute.
This module provides the pieces a benchmark needs to enforce and report that, independent of the model family:

``ComputeBudget``
    a wall-clock, FLOP and/or step limit; training stops at whichever is reached first.
``count_flops``
    FLOPs of one call, counted with PyTorch's own per-operator formulas (``torch.utils.flop_counter``) plus FFTs.
    Unlike ``FlopCounterMode`` it works around ``torch.autograd.grad`` (PINN residuals), because it does not install
    module hooks.
``train_within_budget``
    runs ``step(progress)`` until the budget is spent; ``progress`` in [0, 1] drives budget-aware schedules
    (``cosine_lr``); optional evaluations pause the clock so they are not charged to the method.
``equal_budget_search``
    random search with the same number of trials (and, by the caller, the same budget per trial) for every method.
``AmortizedCost``
    data-generation, training and inference cost, per new query and for ``n`` queries; break-even against another.
``relative_l2``, ``time_per_query``, ``hardware_record``
    the metrics and the hardware line every equal-budget table needs.

What is counted as a FLOP: matrix products, convolutions and attention (PyTorch's formula registry, one
multiply-add = 2 FLOPs) and FFTs (``5 N log2 N`` for a complex transform of length ``N``, half that for a real
one). Elementwise operations (activations, residual arithmetic) are not counted, as in PyTorch's own counter;
they matter for small MLPs, which is why the wall-clock budget is the primary one and FLOPs are reported with it.
"""
from __future__ import annotations

import itertools
import math
import os
import platform
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pinneapple_core.errors import BadValue, ShapeMismatch

__all__ = [
    "ComputeBudget", "BudgetRun", "count_flops", "train_within_budget", "cosine_lr", "set_lr",
    "SearchTrial", "SearchResult", "equal_budget_search", "AmortizedCost", "relative_l2", "time_per_query",
    "hardware_record",
]


# --------------------------------------------------------------------------- budget
@dataclass(frozen=True)
class ComputeBudget:
    """A training budget: any combination of wall-clock seconds, FLOPs and optimiser steps (``None`` = no limit)."""
    wall_s: float | None = None
    flops: float | None = None
    steps: int | None = None

    def __post_init__(self):
        if self.wall_s is None and self.flops is None and self.steps is None:
            raise BadValue("a ComputeBudget with no limit does not bound anything",
                           fix="set at least one of wall_s=..., flops=..., steps=...")
        for name in ("wall_s", "flops", "steps"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise BadValue(f"{name} must be positive, got {v}",
                               fix=f"pass a positive {name}, or leave it as None when that quantity should not be "
                                   f"limited")

    def fraction(self, wall_s: float, flops: float, steps: int) -> float:
        """Fraction of the budget used: the largest over the limits that are set (1.0 = spent)."""
        used = []
        if self.wall_s is not None:
            used.append(wall_s / self.wall_s)
        if self.flops is not None:
            used.append(flops / self.flops)
        if self.steps is not None:
            used.append(steps / self.steps)
        return max(used)

    def scaled(self, factor: float) -> ComputeBudget:
        """The same budget times ``factor`` (for error-versus-budget curves)."""
        return ComputeBudget(None if self.wall_s is None else self.wall_s * factor,
                             None if self.flops is None else self.flops * factor,
                             None if self.steps is None else max(1, int(round(self.steps * factor))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetRun:
    """What a budgeted training run used and how it went."""
    steps: int
    wall_s: float
    flops: float
    flops_per_step: float
    stopped_by: str
    budget: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    eval_wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- FLOPs
def _fft_len(shape: Sequence[int], dims: Sequence[int]) -> tuple[int, int]:
    n = 1
    for d in dims:
        n *= int(shape[d])
    total = 1
    for s in shape:
        total *= int(s)
    return n, total // max(n, 1)


def _fft_flops(n: int, batches: int, real: bool) -> int:
    if n <= 1:
        return 0
    f = 5.0 * n * math.log2(n) * batches
    return int(f / 2 if real else f)


def _fft_r2c(x_shape, dim, *args, out_shape=None, **kwargs) -> int:
    n, b = _fft_len(x_shape, dim)
    return _fft_flops(n, b, real=True)


def _fft_c2r(x_shape, dim, *args, out_shape=None, **kwargs) -> int:
    n, b = _fft_len(out_shape, dim)
    return _fft_flops(n, b, real=True)


def _fft_c2c(x_shape, dim, *args, out_shape=None, **kwargs) -> int:
    n, b = _fft_len(x_shape, dim)
    return _fft_flops(n, b, real=False)


def _registry():
    import torch
    from torch.utils.flop_counter import flop_registry, shape_wrapper
    aten = torch.ops.aten
    reg = dict(flop_registry)
    reg[aten._fft_r2c] = shape_wrapper(_fft_r2c)
    reg[aten._fft_c2r] = shape_wrapper(_fft_c2r)
    reg[aten._fft_c2c] = shape_wrapper(_fft_c2c)
    return reg


def count_flops(fn: Callable[[], Any]) -> tuple[int, dict[str, int], Any]:
    """Run ``fn()`` once and count its FLOPs, forward and backward (including ``torch.autograd.grad``).

    Returns ``(total, per_operator, fn_result)``. See the module docstring for what counts as a FLOP.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    registry = _registry()
    counts: dict[Any, int] = defaultdict(int)

    class _Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            formula = registry.get(func._overloadpacket)
            if formula is not None:
                counts[func._overloadpacket] += int(formula(*args, **kwargs, out_val=out))
            return out

    with _Count():
        result = fn()
    per_op = {str(k): int(v) for k, v in counts.items()}
    return int(sum(counts.values())), per_op, result


# --------------------------------------------------------------------------- training under a budget
def cosine_lr(progress: float, lr0: float, final_fraction: float = 0.03) -> float:
    """Cosine decay from ``lr0`` to ``final_fraction * lr0`` over the budget (``progress`` in [0, 1])."""
    p = min(max(progress, 0.0), 1.0)
    return lr0 * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * p)))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def train_within_budget(step: Callable[[float], float | None], budget: ComputeBudget, *,
                        flops_per_step: float | None = None, count_first_step: bool = True,
                        evaluate: Callable[[], Any] | None = None,
                        eval_at: Sequence[float] = (),
                        log_every_s: float = 5.0,
                        checkpoint: Callable[[dict[str, Any]], None] | None = None,
                        checkpoint_every_s: float = 300.0,
                        resume: dict[str, Any] | None = None) -> BudgetRun:
    """Call ``step(progress)`` (one optimiser step; may return the loss) until ``budget`` is spent.

    ``progress`` is the fraction of the budget used before the step, so a schedule such as ``cosine_lr`` reaches its
    end exactly when the budget does, whatever the budget is.

    FLOPs: with ``flops_per_step`` given it is used; otherwise, with ``count_first_step``, the first step is run
    under ``count_flops`` (that step's FLOPs count towards the budget; its time does not, because counting slows it
    down). Steps must then cost the same FLOPs each (fixed batch shapes), which is the usual case.

    ``evaluate`` is called when progress first reaches each fraction in ``eval_at`` and at the end; the clock is
    paused while it runs, so evaluation is never charged to the method. Its results go into ``history``.

    Long runs: every ``checkpoint_every_s`` seconds of training, ``checkpoint(state)`` is called with the loop's
    accounting (steps, wall clock, FLOPs, history); save it with the model and optimiser. Passing that ``state`` back
    as ``resume`` continues the same budget where it stopped (the time spent saving is not charged).
    """
    history: list[dict[str, Any]] = []
    steps, flops, wall, eval_wall = 0, 0.0, 0.0, 0.0
    fps = float(flops_per_step) if flops_per_step is not None else 0.0
    if resume is not None:
        steps, flops, wall = int(resume["steps"]), float(resume["flops"]), float(resume["wall_s"])
        fps, eval_wall = float(resume["flops_per_step"]), float(resume.get("eval_wall_s", 0.0))
        history = list(resume.get("history", []))
        history.append({"step": steps, "progress": budget.fraction(wall, flops, steps), "wall_s": wall,
                        "flops": flops, "loss": None, "resumed": True})
    elif flops_per_step is None and count_first_step:
        n, _, loss = count_flops(lambda: step(0.0))
        fps = float(n)
        steps, flops = 1, fps
        history.append({"step": 1, "progress": budget.fraction(0.0, flops, 1), "wall_s": 0.0, "flops": flops,
                        "loss": _as_float(loss)})
    pending = sorted(float(f) for f in eval_at if budget.fraction(wall, flops, steps) < f < 1)
    last_log, last_ckpt = wall, wall
    stopped_by = "budget"
    loss = None
    while True:
        prog = budget.fraction(wall, flops, steps)
        if pending and prog >= pending[0]:
            pending.pop(0)
            if evaluate is not None:
                t0 = time.perf_counter()
                res = evaluate()
                eval_wall += time.perf_counter() - t0
                history.append({"step": steps, "progress": prog, "wall_s": wall, "flops": flops,
                                "loss": _as_float(loss), "eval": res})
        if prog >= 1.0:
            break
        _sync()
        t0 = time.perf_counter()
        try:
            loss = step(prog)
        except StopIteration:
            stopped_by = "converged"
            _sync()
            wall += time.perf_counter() - t0
            break
        _sync()
        wall += time.perf_counter() - t0
        steps += 1
        flops += fps
        if wall - last_log >= log_every_s:
            last_log = wall
            history.append({"step": steps, "progress": budget.fraction(wall, flops, steps), "wall_s": wall,
                            "flops": flops, "loss": _as_float(loss)})
        if checkpoint is not None and wall - last_ckpt >= checkpoint_every_s:
            last_ckpt = wall
            checkpoint({"steps": steps, "wall_s": wall, "flops": flops, "flops_per_step": fps,
                        "eval_wall_s": eval_wall, "history": list(history), "budget": budget.to_dict()})
    final = {"step": steps, "progress": budget.fraction(wall, flops, steps), "wall_s": wall, "flops": flops,
             "loss": _as_float(loss), "final": True}
    if evaluate is not None:
        t0 = time.perf_counter()
        final["eval"] = evaluate()
        eval_wall += time.perf_counter() - t0
    history.append(final)
    if stopped_by == "budget":
        used = {k: v for k, v in (("wall_s", wall / budget.wall_s if budget.wall_s else 0),
                                  ("flops", flops / budget.flops if budget.flops else 0),
                                  ("steps", steps / budget.steps if budget.steps else 0))}
        stopped_by = max(used, key=used.get)
    return BudgetRun(steps=steps, wall_s=wall, flops=flops, flops_per_step=fps, stopped_by=stopped_by,
                     budget=budget.to_dict(), history=history, eval_wall_s=eval_wall)


def _as_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


# --------------------------------------------------------------------------- hyper-parameter search
@dataclass
class SearchTrial:
    config: dict[str, Any]
    score: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    best_config: dict[str, Any]
    best_score: float
    trials: list[SearchTrial]
    n_trials: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"best_config": self.best_config, "best_score": self.best_score, "n_trials": self.n_trials,
                "seed": self.seed, "trials": [asdict(t) for t in self.trials]}


def equal_budget_search(space: Mapping[str, Sequence[Any]], objective: Callable[[dict[str, Any]], Any], *,
                        n_trials: int, seed: int = 0, minimize: bool = True) -> SearchResult:
    """Random search over the grid ``space`` with exactly ``n_trials`` distinct configurations.

    Give every method the same ``n_trials`` and the same budget per trial (inside ``objective``) and the search
    effort is equal. ``objective(config)`` returns a score, or ``(score, info)``; non-finite scores lose.
    If the grid has fewer than ``n_trials`` points, every point is tried (and ``n_trials`` reports that number).
    """
    keys = list(space)
    grid = list(itertools.product(*(list(space[k]) for k in keys)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))[:n_trials]
    trials: list[SearchTrial] = []
    for i in order:
        cfg = dict(zip(keys, grid[int(i)], strict=False))
        out = objective(cfg)
        score, info = (out if isinstance(out, tuple) else (out, {}))
        trials.append(SearchTrial(cfg, float(score), dict(info)))
    sign = 1.0 if minimize else -1.0
    best = min(trials, key=lambda t: sign * t.score if math.isfinite(t.score) else math.inf)
    return SearchResult(best.config, best.score, trials, len(trials), seed)


# --------------------------------------------------------------------------- amortised cost
@dataclass
class AmortizedCost:
    """Cost of answering new queries, in seconds (or any one unit used throughout).

    ``data_s``: generating the training data (simulations), paid once. ``train_s``: training, paid once.
    ``infer_s``: one new query. ``per_query_train_s``: training that every new query needs again (a PINN trained
    per instance has ``train_s = 0`` and ``per_query_train_s`` = its training time).
    """
    data_s: float = 0.0
    train_s: float = 0.0
    infer_s: float = 0.0
    per_query_train_s: float = 0.0

    def total(self, n_queries: int) -> float:
        return self.data_s + self.train_s + n_queries * (self.infer_s + self.per_query_train_s)

    def per_query(self, n_queries: int) -> float:
        if n_queries < 1:
            raise BadValue(f"per_query({n_queries}) has no meaning: the cost per query is total / n_queries",
                           fix="pass n_queries >= 1, or use total(n) when you only need the total")
        return self.total(n_queries) / n_queries

    def break_even(self, other: AmortizedCost) -> float | None:
        """Number of queries above which ``self`` is cheaper in total than ``other`` (None: never, 0: always)."""
        fixed = (self.data_s + self.train_s) - (other.data_s + other.train_s)
        marginal = (other.infer_s + other.per_query_train_s) - (self.infer_s + self.per_query_train_s)
        if fixed <= 0 and marginal >= 0:
            return 0.0
        if marginal <= 0:
            return None
        return fixed / marginal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- metrics, timing, hardware
def relative_l2(pred, true, *, sample_axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Per-sample relative L2 error ``||pred - true|| / ||true||`` over every axis except ``sample_axis``."""
    p = np.asarray(_to_numpy(pred), dtype=np.float64)
    t = np.asarray(_to_numpy(true), dtype=np.float64)
    if p.shape != t.shape:
        raise ShapeMismatch(f"relative_l2 compares point by point, but pred has shape {p.shape} and true {t.shape}",
                            fix="reshape or resample one to the other, or pass sample_axis= to say which axis "
                                "indexes the samples")
    p = np.moveaxis(p, sample_axis, 0).reshape(p.shape[sample_axis], -1)
    t = np.moveaxis(t, sample_axis, 0).reshape(t.shape[sample_axis], -1)
    return np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), eps)


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


def time_per_query(fn: Callable[[], Any], *, repeats: int = 20, warmup: int = 3, n_per_call: int = 1,
                   no_grad: bool = True) -> dict[str, float]:
    """Median wall time of ``fn()`` divided by ``n_per_call`` (instances answered per call), under ``no_grad``
    unless ``no_grad=False`` (a query that needs derivatives, e.g. a PINN's velocity from its stream function)."""
    import contextlib
    ctx = contextlib.nullcontext
    if no_grad:
        try:
            import torch
            ctx = torch.no_grad
        except Exception:  # pragma: no cover
            pass
    times = []
    with ctx():
        for _ in range(warmup):
            fn()
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            times.append(time.perf_counter() - t0)
    t = np.asarray(times) / max(1, n_per_call)
    return {"median_s": float(np.median(t)), "min_s": float(t.min()), "p90_s": float(np.percentile(t, 90)),
            "repeats": repeats, "n_per_call": n_per_call}


def hardware_record() -> dict[str, Any]:
    """CPU model, cores, memory, GPU, library versions and thread count: the line every budget table needs."""
    rec: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                           "logical_cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    rec["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        rec["cpu"] = platform.processor() or "unknown"
    try:
        with open("/proc/meminfo") as f:
            rec["mem_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import torch
        rec.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        rec["torch"] = None
    rec["numpy"] = np.__version__
    return rec
