"""Computational-cost validation per operation: measure, fit scaling, check budgets, catch regressions.

Three questions this answers about any operation (a solver step, a model forward pass, a residual
evaluation, an export):

1. **How much does it cost?** ``measure`` returns wall time (median of repeats, with spread), peak
   memory and, for PyTorch work, the exact FLOP count.
2. **How does the cost grow?** ``scaling_study`` runs the operation over problem sizes, fits the
   exponent ``k`` in ``cost ~ n^k`` with a bootstrap confidence interval, and compares it with the
   complexity the author *declared* (``"n log n"``, ``"n^2"``, ...).
3. **Did it get worse?** ``Budget`` turns limits into pass/fail, and ``CostLedger`` saves a baseline and
   flags regressions. FLOP counts are deterministic, so they gate strictly; wall time is noisy, so it gates
   loosely.

What is and is not measured (read before trusting a number):

- FLOPs come from ``torch.utils.flop_counter`` and only count PyTorch operators (matmul, conv,
  attention, ...). NumPy / SciPy / pure-Python work reports ``flops=None``, never a guess. Around
  ``torch.autograd.grad`` the flop counter cannot run, so the profiler's matmul/conv count is used and
  ``flops_source="profiler"`` marks it as a lower bound.
- Peak memory: CUDA via ``torch.cuda.max_memory_allocated``; on CPU the PyTorch profiler's allocation records
  (exact for CPU tensors) plus ``tracemalloc`` (exact for Python and NumPy, which reports to it), summed —
  ``mem_source="torch+trace"``. Neither is blind guessing and neither is a sampled lower bound; the limits of
  each method, and why the in-process RSS watermark cannot be reused, are measured in ``memory`` (issue #54).
  Sampled process RSS (``mem_source="rss"``, needs ``psutil``) is kept as a labelled lower bound for the cases
  the two methods cannot cover, for example native libraries.
- Time is wall clock on the machine that runs it; compare numbers only across runs on the same machine.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from pinneapple_core.errors import BadValue
from .memory import measure_peak


# --------------------------------------------------------------------------- records
@dataclass
class CostRecord:
    name: str
    n_runs: int
    time_s: float                      # median over repeats
    time_min_s: float
    time_iqr_s: float                  # interquartile range, a robust spread
    times_s: List[float] = field(default_factory=list)
    peak_mem_bytes: Optional[int] = None
    mem_source: str = ""
    mem_exact: bool = False            # True when the peak is a measured value, not a lower bound
    mem_components: dict[str, int] = field(default_factory=dict)   # allocator -> bytes it contributed
    flops: Optional[int] = None        # PyTorch operators only; None when not countable
    flops_source: str = ""             # "flop_counter" (exact) | "profiler" (matmul/conv only, lower bound)
    size: Optional[float] = None

    @property
    def relative_spread(self) -> float:
        return self.time_iqr_s / self.time_s if self.time_s > 0 else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CostRecord":
        return cls(**d)


# --------------------------------------------------------------------------- measuring
def _count_flops(fn: Callable[[], Any]):
    """``(flops, source)``. ``flop_counter`` is exact for the operators it knows; it cannot run around
    ``torch.autograd.grad`` (its module hooks fail there), so then ``equal_budget.count_flops`` is used: the same
    per-operator formulas without module hooks, plus FFTs. The profiler's count (matmul/conv only) is the last
    resort. A zero from the flop counter is not an answer -- it knows matmul, conv and attention and nothing else, so
    an FFT or an elementwise kernel counts as zero there while the hook-free dispatch counter sees them -- so a zero
    falls through to the next counter instead of ending the chain."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None, ""
    try:
        with FlopCounterMode(display=False) as counter:
            fn()
        total = int(counter.get_total_flops())
        if total > 0:
            return total, "flop_counter"
    except Exception:
        pass
    try:
        from .equal_budget import count_flops
        total = count_flops(fn)[0]
        if total > 0:
            return total, "dispatch_counter"
    except Exception:
        pass
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            fn()
        total = int(sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages()))
        return (total, "profiler") if total > 0 else (None, "")
    except Exception:
        return None, ""


def _measure_memory(fn: Callable[[], Any], method: str = "auto") -> tuple[int | None, str, bool, dict[str, int]]:
    """``(bytes, source, exact, components)`` for one call; ``method`` is passed to ``memory.measure_peak``."""
    peak = measure_peak(fn, method=method)
    return peak.bytes, peak.source, peak.exact, dict(peak.components)


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure(fn: Callable[..., Any], *args, repeats: int = 5, warmup: int = 1, name: Optional[str] = None,
            flops: bool = True, memory: bool = True, size: Optional[float] = None, mem_method: str = "auto",
            **kwargs) -> CostRecord:
    """Cost of one operation ``fn(*args, **kwargs)``.

    Time is the median of ``repeats`` timed runs after ``warmup`` untimed ones; FLOPs and peak memory are
    measured in separate runs so they do not perturb the timing. ``mem_method`` picks how the peak is measured
    (see ``memory.measure_peak``); the default is exact for tensors and for NumPy. When the peak cannot be
    measured it is ``None`` with ``mem_source="unknown"``, never a plausible number.
    """
    if repeats < 1:
        raise BadValue("measure() needs at least one timed run", fix="pass repeats=5 (the default) or any integer >= 1")
    call = lambda: fn(*args, **kwargs)  # noqa: E731
    for _ in range(warmup):
        call()
    times: List[float] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        call()
        _sync()
        times.append(time.perf_counter() - t0)
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    mem, src, exact, components = _measure_memory(call, mem_method) if memory else (None, "", False, {})
    n_flops, flops_src = _count_flops(call) if flops else (None, "")
    return CostRecord(name=name or getattr(fn, "__name__", "operation"), n_runs=repeats, time_s=float(med),
                      time_min_s=float(min(times)), time_iqr_s=float(q3 - q1), times_s=[float(t) for t in times],
                      peak_mem_bytes=mem, mem_source=src, mem_exact=exact, mem_components=components,
                      flops=n_flops, flops_source=flops_src, size=size)


# --------------------------------------------------------------------------- scaling
_COMPLEXITY = {
    "1": lambda n: np.ones_like(n, dtype=float),
    "log n": lambda n: np.log(n),
    "n": lambda n: n,
    "n log n": lambda n: n * np.log(n),
    "n^2": lambda n: n ** 2,
    "n^3": lambda n: n ** 3,
}


def _fit_exponent(sizes: np.ndarray, y: np.ndarray):
    x = np.log(sizes)
    ly = np.log(y)
    slope, intercept = np.polyfit(x, ly, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def expected_exponent(declared: Union[str, float], sizes: Sequence[float]) -> float:
    """Log-log slope of the declared complexity over the sizes actually used (``n log n`` is not 1.0)."""
    if isinstance(declared, (int, float)):
        return float(declared)
    key = declared.strip().lower().replace("**", "^").replace("o(", "").rstrip(")")
    if key not in _COMPLEXITY:
        raise BadValue(f"unknown complexity {declared!r}",
                       fix=f"declare one of {sorted(_COMPLEXITY)} or give the exponent as a number")
    n = np.asarray(sizes, dtype=float)
    if key == "1":
        return 0.0
    return _fit_exponent(n, _COMPLEXITY[key](n))[0]


@dataclass
class ScalingFit:
    quantity: str                      # "time" | "flops" | "memory"
    exponent: float
    ci: (float, float)
    r2: float
    coefficient: float                 # cost ~ coefficient * n^exponent
    declared: Optional[str] = None
    expected_exponent: Optional[float] = None
    verdict: str = ""                  # consistent | worse than declared | better than declared | undeclared

    def predict(self, n: float) -> float:
        return float(self.coefficient * n ** self.exponent)

    def max_size_within(self, budget: float) -> float:
        """Largest ``n`` whose predicted cost stays within ``budget`` (extrapolation: use with care)."""
        if self.exponent <= 0:
            return float("inf")
        return float((budget / self.coefficient) ** (1.0 / self.exponent))


@dataclass
class ScalingStudy:
    name: str
    sizes: List[float]
    records: List[CostRecord]
    fits: Dict[str, ScalingFit]

    def summary(self) -> str:
        lines = [f"scaling study: {self.name}"]
        for q, f in self.fits.items():
            decl = f" | declared {f.declared} (expected k={f.expected_exponent:.2f}) -> {f.verdict}" if f.declared else ""
            lines.append(f"  {q:<6} k = {f.exponent:.2f}  CI [{f.ci[0]:.2f}, {f.ci[1]:.2f}]  R2 = {f.r2:.3f}{decl}")
        return "\n".join(lines)


def _verdict(fit_ci, expected: float, tol: float) -> str:
    lo, hi = fit_ci
    if lo - tol > expected:
        return "worse than declared"
    if hi + tol < expected:
        return "better than declared"
    return "consistent"


def scaling_study(make_call: Callable[[int], Callable[[], Any]], sizes: Sequence[int], *, name: str = "operation",
                  repeats: int = 5, warmup: int = 1, declared: Optional[Union[str, float]] = None,
                  declared_flops: Optional[Union[str, float]] = None, tolerance: float = 0.15,
                  n_boot: int = 300, seed: int = 0, memory: bool = False, flops: bool = True) -> ScalingStudy:
    """Measure an operation over problem sizes and fit ``cost ~ n^k`` for time (and FLOPs when countable).

    ``make_call(n)`` builds the inputs for size ``n`` and returns a zero-argument callable; the setup is not
    timed. Use at least 4 sizes spanning a factor of 10 or more, and sizes large enough that the run takes well
    over a millisecond, otherwise fixed overheads flatten the fit (the R2 and the CI width say so).
    ``tolerance`` is the slack in the exponent when comparing with ``declared``. ``flops=False`` skips the FLOP
    count, which the counter's per-operator hooks make more expensive than the work for an operation that issues
    thousands of small operators; the record then carries ``flops=None`` with an empty source.
    """
    sizes = [int(s) for s in sizes]
    if len(sizes) < 3 or len(set(sizes)) != len(sizes):
        raise BadValue(
            f"a scaling fit needs at least 3 distinct sizes, got {len(set(sizes))} distinct of {len(sizes)}",
            fix="pass at least 3 distinct sizes spanning a factor of 10 or more, e.g. [10**3, 10**4, 10**5]")
    records = [measure(make_call(n), repeats=repeats, warmup=warmup, name=f"{name}[n={n}]", size=float(n),
                       memory=memory, flops=flops) for n in sizes]
    n_arr = np.array(sizes, dtype=float)
    fits: Dict[str, ScalingFit] = {}
    rng = np.random.default_rng(seed)

    def add(quantity: str, y: np.ndarray, boot_samples: Optional[List[np.ndarray]], decl):
        k, b, r2 = _fit_exponent(n_arr, y)
        ci = (k, k)
        if boot_samples is not None:
            ks = []
            for _ in range(n_boot):
                yb = np.array([np.median(rng.choice(s, size=len(s), replace=True)) for s in boot_samples])
                ks.append(_fit_exponent(n_arr, yb)[0])
            ci = (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))
        fit = ScalingFit(quantity, k, ci, r2, float(math.exp(b)))
        if decl is not None:
            fit.declared = str(decl)
            fit.expected_exponent = expected_exponent(decl, sizes)
            fit.verdict = _verdict(ci, fit.expected_exponent, tolerance)
        else:
            fit.verdict = "undeclared"
        fits[quantity] = fit

    add("time", np.array([r.time_s for r in records]), [np.array(r.times_s) for r in records], declared)
    if all(r.flops for r in records):
        add("flops", np.array([float(r.flops) for r in records]), None, declared_flops)
    if memory and all(r.peak_mem_bytes and r.peak_mem_bytes > 0 for r in records):
        add("memory", np.array([float(r.peak_mem_bytes) for r in records]), None, None)
    return ScalingStudy(name, [float(s) for s in sizes], records, fits)


# --------------------------------------------------------------------------- budgets
@dataclass
class Budget:
    """Limits for one operation. Any limit left as ``None`` is not checked."""
    max_time_s: Optional[float] = None
    max_mem_bytes: Optional[int] = None
    max_flops: Optional[int] = None
    max_time_exponent: Optional[float] = None   # checked against the CI's lower bound, so noise cannot fail it

    def check(self, result: Union[CostRecord, ScalingStudy]) -> List[str]:
        """Violations as readable strings; an empty list means the budget holds."""
        out: List[str] = []
        records = result.records if isinstance(result, ScalingStudy) else [result]
        for r in records:
            tag = f"{r.name}: "
            if self.max_time_s is not None and r.time_s > self.max_time_s:
                out.append(f"{tag}time {r.time_s:.4g}s exceeds {self.max_time_s:.4g}s")
            if self.max_mem_bytes is not None and r.peak_mem_bytes is not None and r.peak_mem_bytes > self.max_mem_bytes:
                out.append(f"{tag}memory {r.peak_mem_bytes} B exceeds {self.max_mem_bytes} B")
            if self.max_flops is not None and r.flops is not None and r.flops > self.max_flops:
                out.append(f"{tag}{r.flops} FLOPs exceeds {self.max_flops}")
        if self.max_time_exponent is not None and isinstance(result, ScalingStudy):
            f = result.fits["time"]
            if f.ci[0] > self.max_time_exponent:
                out.append(f"{result.name}: time exponent {f.exponent:.2f} (CI lower {f.ci[0]:.2f}) "
                           f"exceeds {self.max_time_exponent:.2f}")
        return out

    def validate(self, result: Union[CostRecord, ScalingStudy]) -> None:
        violations = self.check(result)
        if violations:
            raise BudgetExceeded(
                "; ".join(violations),
                fix="raise the limit in Budget(...) or shrink the problem; each violation above names the quantity "
                    "that went over, and CostLedger.compare() shows the same number as a regression ratio")


class BudgetExceeded(AssertionError):
    """The operation went over a ``Budget``. Carries ``fix``, the action that clears the violation."""

    def __init__(self, message: str, *, fix: str = ""):
        self.what = message
        self.fix = fix
        self.message = f"{message}\n  Fix: {fix}" if fix else message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


# --------------------------------------------------------------------------- ledger
@dataclass
class Regression:
    name: str
    quantity: str
    baseline: float
    current: float
    ratio: float

    def __str__(self) -> str:
        return f"{self.name}: {self.quantity} {self.baseline:.4g} -> {self.current:.4g} (x{self.ratio:.2f})"


class CostLedger:
    """Named operations with their costs; save a baseline, compare later runs against it."""

    def __init__(self):
        self.records: Dict[str, CostRecord] = {}
        self.environment = {"python": platform.python_version(), "machine": platform.machine(),
                            "system": platform.system(), "cpu_count": os.cpu_count()}

    def measure(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> CostRecord:
        rec = measure(fn, *args, name=name, **kwargs)
        self.records[name] = rec
        return rec

    def track(self, name: Optional[str] = None, *, repeats: int = 3, **measure_kwargs):
        """Decorator: each call measures the function (it runs ``warmup + repeats`` times plus one run each for
        FLOPs and memory, so use it on side-effect-free operations), stores the record and returns the result."""
        def deco(fn):
            label = name or fn.__name__

            def wrapper(*a, **kw):
                box: Dict[str, Any] = {}

                def run():
                    box["v"] = fn(*a, **kw)
                self.records[label] = measure(run, name=label, repeats=repeats, **measure_kwargs)
                return box["v"]
            wrapper.__name__ = fn.__name__
            return wrapper
        return deco

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump({"environment": self.environment, "records": {k: v.to_dict() for k, v in self.records.items()}},
                      f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "CostLedger":
        with open(path) as f:
            raw = json.load(f)
        ledger = cls()
        ledger.environment = raw.get("environment", {})
        ledger.records = {k: CostRecord.from_dict(v) for k, v in raw["records"].items()}
        return ledger

    def compare(self, baseline: "CostLedger", *, time_factor: float = 1.5, flops_factor: float = 1.05,
                mem_factor: float = 1.5) -> List[Regression]:
        """Operations that got worse than ``baseline``. FLOPs gate tightly (deterministic), time and memory
        loosely (noisy). Operations missing from either side are ignored."""
        out: List[Regression] = []
        for name, cur in self.records.items():
            base = baseline.records.get(name)
            if base is None:
                continue
            for quantity, b, c, factor in (("flops", base.flops, cur.flops, flops_factor),
                                           ("time_s", base.time_s, cur.time_s, time_factor),
                                           ("peak_mem_bytes", base.peak_mem_bytes, cur.peak_mem_bytes, mem_factor)):
                if b and c and c > b * factor:
                    out.append(Regression(name, quantity, float(b), float(c), c / b))
        return out
