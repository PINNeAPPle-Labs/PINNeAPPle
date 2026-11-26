"""Floating point and error analysis: where digits go, and whether the problem or the code took them (#1056).

Five things a user of a numerical library needs to be able to see, each of them *measured* here rather
than quoted from a table:

1. **The dtype's own parameters** -- machine epsilon (by the classical halving loop), unit in the last
   place, the largest finite value and the smallest subnormal, measured for float16, float32 and
   float64.
2. **Catastrophic cancellation** -- the quadratic formula, ``log1p``/``expm1`` and the one-pass
   variance, each in its naive and its stable form, with the error against a reference computed in
   ``decimal`` at 60 digits.
3. **Compensated summation** -- naive, pairwise, Kahan and Neumaier, with the error *growth in n*
   measured (it is a slope on a log-log plot: ~1/2 for naive, ~1 for the worst case, ~0 for a
   compensated sum until it hits the floor).
4. **Conditioning against backward error** -- the verdict that matters: is the error the problem's fault
   (``cond`` large, ``forward ~ cond * backward``) or the algorithm's (``forward >> cond * backward``)?
5. **Interval arithmetic** for guaranteed bounds, and the precision floor of a residual evaluated in
   mixed precision (the reason a PINN's float32 loss stops improving).

The limitation, once: a condition number is a local, first-order statement about the *problem*. It says
nothing about the algorithm, and it does not apply away from the point where it was measured. Every
verdict here is therefore paired with the measured forward and backward errors that produced it --
a number, not a label.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Any

import numpy as np

from .diagnostics import Diagnostic, explain

__all__ = [
    "ErrorBudget",
    "Interval",
    "MachineParameters",
    "PrecisionFloor",
    "QuadraticRoots",
    "SummationGrowth",
    "VarianceComparison",
    "digits_lost",
    "error_budget",
    "interval_quadratic_roots",
    "kahan_sum",
    "log1p_expm1_comparison",
    "machine_epsilon",
    "machine_parameters",
    "measured_condition_number",
    "naive_sum",
    "neumaier_sum",
    "pairwise_sum",
    "quadratic_roots",
    "quadratic_roots_reference",
    "relative_error",
    "residual_precision_floor",
    "summation_error_growth",
    "ulp",
    "variance_comparison",
    "variance_naive",
    "variance_welford",
]


def relative_error(computed: float, reference: float) -> float:
    """``|computed - reference| / |reference|``, with a zero reference handled as an absolute error.

    A relative error against zero is undefined; returning the absolute error there is the only honest
    option, and it is what every comparison in this module uses.
    """
    computed = float(computed)
    reference = float(reference)
    if reference == 0.0:
        return abs(computed)
    return abs(computed - reference) / abs(reference)


def digits_lost(computed: float, reference: float) -> float:
    """How many decimal digits the computation lost: ``-log10(relative_error)``.

    ``0`` means no digits lost (or better than the reference), ``8`` means the answer has 8 digits left
    of the ~16 that float64 carries. Negative values mean the computation was *more* accurate than the
    reference, which happens when the reference is the lower-precision one.
    """
    rel = relative_error(computed, reference)
    if rel == 0.0:
        return float("inf")
    return float(-math.log10(rel))


# --------------------------------------------------------------------------------------- dtype
@dataclass(frozen=True)
class MachineParameters:
    """The measured limits of one floating-point dtype."""

    dtype: str
    eps: float
    eps_half: float
    ulp_at_one: float
    max_finite: float
    min_subnormal: float
    decimal_digits: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "dtype": self.dtype,
            "eps": self.eps,
            "eps_half": self.eps_half,
            "ulp_at_one": self.ulp_at_one,
            "max_finite": self.max_finite,
            "min_subnormal": self.min_subnormal,
            "decimal_digits": self.decimal_digits,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def machine_epsilon(dtype: str | np.dtype = "float64") -> float:
    """Measure machine epsilon: the smallest ``eps`` with ``1 + eps != 1`` in ``dtype``.

    The halving loop is the definition, run in the dtype itself: start from 1 and keep halving while
    ``1 + eps`` still rounds up to something larger than 1. It halts because once ``eps`` is below half
    an ulp of 1 the sum rounds back to 1. Returns ``2**-52`` for float64, ``2**-23`` for float32 and
    ``2**-10`` for float16.
    """
    d = np.dtype(dtype)
    one = d.type(1.0)
    eps = d.type(1.0)
    while True:
        half = d.type(eps / d.type(2.0))
        if half == 0 or d.type(one + half) <= one:
            return float(eps)
        eps = half


def ulp(x: float, dtype: str | np.dtype = "float64") -> float:
    """The unit in the last place at ``x``: the distance to the next representable number.

    Equals ``eps * 2**floor(log2|x|)`` in the normal range, which is why a condition number multiplies
    the error of a *relative* perturbation by the magnitude of the quantity, not by eps alone.
    """
    d = np.dtype(dtype)
    xd = d.type(x)
    if not np.isfinite(float(xd)):
        return float("nan")
    nxt = np.nextafter(xd, d.type(np.inf))
    return float(abs(float(nxt) - float(xd)))


def machine_parameters(dtype: str | np.dtype = "float64") -> MachineParameters:
    """Measure the limits of ``dtype``: epsilon by halving, the subnormal floor by halving to zero.

    Epsilon and the smallest subnormal are measured, because both are the answer to a loop a user can
    read. The largest finite value is taken from ``np.finfo``: doubling up to it would need 2**24 steps
    in float32, which measures nothing that the table does not already guarantee. ``float16`` is included because a neural network's forward pass
    routinely runs there, and its epsilon is nine decimal digits coarser than float64's.
    """
    d = np.dtype(dtype)
    eps = machine_epsilon(d)

    with np.errstate(over="ignore", under="ignore"):
        big = float(np.finfo(d).max)      # a table value: doubling up to it would take 2**24 steps
        small = d.type(1.0)
        while d.type(small / d.type(2.0)) > d.type(0.0):
            small = d.type(small / d.type(2.0))

    digits = float(-math.log10(eps)) if eps > 0 else float("nan")
    diagnostics: list[Diagnostic] = []
    if digits < 15.0:  # float64 keeps 15.65 decimal digits; anything below is genuinely coarser
        diagnostics.append(
            explain("FLOAT.MIXED_PRECISION_RESIDUAL", dtype=str(d), decimal_digits=digits,
                    eps=eps)
        )
    return MachineParameters(
        dtype=str(d),
        eps=eps,
        eps_half=eps / 2.0,
        ulp_at_one=ulp(1.0, d),
        max_finite=float(big),
        min_subnormal=float(small),
        decimal_digits=digits,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- cancellation: quadratic
@dataclass(frozen=True)
class QuadraticRoots:
    """Both roots of ``a x^2 + b x + c = 0``, with the measured error of the formula that produced them.

    ``relative_errors`` is against the 50-digit ``decimal`` reference and is ``(nan, nan)`` when no
    reference was requested. Roots are returned in decreasing magnitude.
    """

    roots: tuple[float, float]
    discriminant: float
    method: str
    relative_errors: tuple[float, float] = (float("nan"), float("nan"))
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "roots": list(self.roots),
            "discriminant": self.discriminant,
            "method": self.method,
            "relative_errors": list(self.relative_errors),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def quadratic_roots_reference(
    a: float, b: float, c: float, *, digits: int = 50
) -> tuple[float, float]:
    """The two real roots computed in ``decimal`` at ``digits`` significant digits -- the reference.

    ``decimal`` is in the standard library, so this module measures cancellation without a dependency.
    Raises ``ValueError`` when the discriminant is negative (complex roots are out of scope here, and
    returning a NaN would hide the caller's mistake).
    """
    with localcontext() as ctx:  # the precision stays local to this call
        ctx.prec = digits
        da, db, dc = Decimal(a), Decimal(b), Decimal(c)
        disc = db * db - Decimal(4) * da * dc
        if disc < 0:
            raise ValueError(f"negative discriminant {float(disc):.6g}: the roots are complex")
        sq = disc.sqrt()
        two_a = Decimal(2) * da
        r1 = (-db + sq) / two_a
        r2 = (-db - sq) / two_a
    return tuple(sorted((float(r1), float(r2)), key=abs, reverse=True))  # type: ignore[return-value]


def quadratic_roots(
    a: float, b: float, c: float, *, method: str = "stable", reference: bool = True
) -> QuadraticRoots:
    """Roots of the quadratic, by the naive or the cancellation-free formula.

    The naive formula ``(-b +- sqrt(b^2 - 4ac)) / (2a)`` loses every digit of the root that subtracts
    two nearly equal numbers -- which is the *small* root whenever ``b^2 >> 4ac``. The stable form
    computes ``q = -(b + sign(b) sqrt(disc)) / 2`` first, then uses ``r1 = q/a`` and ``r2 = c/q``:
    Vieta's identity ``r1 r2 = c/a`` supplies the second root by multiplication instead of subtraction.

    Example where the naive formula is worthless: ``x^2 - 1e8 x + 1 = 0`` -- the small root is 1e-8, and
    ``-b + sqrt(b^2-4ac)`` cancels all of its leading digits.

    Parameters
    ----------
    method : ``"stable"`` (default) or ``"naive"``, for comparing the two.

    Raises ``ValueError`` for a negative discriminant or ``a == 0``.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    if a == 0.0:
        raise ValueError("a == 0: this is not a quadratic equation")
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError(f"negative discriminant {disc:.6g}: the roots are complex")
    sq = math.sqrt(disc)
    diagnostics: list[Diagnostic] = []

    if method == "naive":
        r1 = (-b + sq) / (2.0 * a)
        r2 = (-b - sq) / (2.0 * a)
    elif method == "stable":
        q = -0.5 * (b + math.copysign(sq, b))
        if q == 0.0:  # b == 0 and disc == 0: both roots are zero
            r1 = r2 = 0.0
        else:
            r1 = q / a
            r2 = c / q
    else:
        raise ValueError(f"method must be 'naive' or 'stable'; got {method!r}")

    roots = tuple(sorted((r1, r2), key=abs, reverse=True))
    errors: tuple[float, float] = (float("nan"), float("nan"))
    if reference:
        ref = quadratic_roots_reference(a, b, c)
        errors = (relative_error(roots[0], ref[0]), relative_error(roots[1], ref[1]))
        if errors[1] > 1e-8:
            diagnostics.append(
                explain("FLOAT.CANCELLATION", method=method, root=roots[1], reference=ref[1],
                        relative_error=errors[1], digits_lost=digits_lost(roots[1], ref[1]))
            )
            if digits_lost(roots[1], ref[1]) < 1.0:
                diagnostics.append(explain("FLOAT.NO_SIGNIFICANT_DIGITS", method=method,
                                           relative_error=errors[1]))
    return QuadraticRoots(
        roots=roots,
        discriminant=float(disc),
        method=method,
        relative_errors=errors,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- cancellation: log1p/expm1
def log1p_expm1_comparison(x: float) -> dict[str, float]:
    """Measure what ``log(1+x)`` and ``exp(x)-1`` lose against ``log1p``/``expm1`` near zero.

    ``log(1+x)`` first rounds ``1+x`` to the dtype, so for ``|x| < eps`` the addition throws the whole
    information away; ``exp(x)-1`` subtracts two numbers that both round to 1. Returns absolute errors
    for the naive and the stable form, both against the standard library's correctly rounded
    ``math.log1p``/``math.expm1``.

    The stable forms are not "better approximations" -- they are the same function computed without the
    intermediate that rounds.
    """
    x = float(x)
    ref_log = math.log1p(x)
    ref_exp = math.expm1(x)
    naive_log = math.log(1.0 + x)
    naive_exp = math.exp(x) - 1.0
    return {
        "x": x,
        "log1p_reference": ref_log,
        "log1p_naive": naive_log,
        "log1p_naive_error": abs(naive_log - ref_log),
        "log1p_stable_error": abs(math.log1p(x) - ref_log),
        "expm1_reference": ref_exp,
        "expm1_naive": naive_exp,
        "expm1_naive_error": abs(naive_exp - ref_exp),
        "expm1_stable_error": abs(math.expm1(x) - ref_exp),
    }


# --------------------------------------------------------------------------- one-pass variance
@dataclass(frozen=True)
class VarianceComparison:
    """The naive ``E[x^2] - E[x]^2`` against Welford's recurrence, both against an exact reference."""

    reference: float
    naive: float
    welford: float
    naive_relative_error: float
    welford_relative_error: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "reference": self.reference,
            "naive": self.naive,
            "welford": self.welford,
            "naive_relative_error": self.naive_relative_error,
            "welford_relative_error": self.welford_relative_error,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def variance_naive(values: Sequence[float] | np.ndarray) -> float:
    """``mean(x^2) - mean(x)^2`` -- the textbook shortcut, and the wrong one for large means.

    The two means are each of size ``mean^2``, and their difference is the variance. When the variance
    is small compared with the mean, the subtraction cancels all the significant digits of both terms:
    the result carries the rounding error of the *large* quantities.
    """
    x = np.asarray(values, dtype=np.float64)
    return float(np.mean(x * x) - np.mean(x) ** 2)


def variance_welford(values: Sequence[float] | np.ndarray) -> float:
    """Welford's one-pass recurrence: no subtraction of two large, nearly equal means.

    ``x -> x + (v - x)/n`` then ``M2 -> M2 + (v - x_new)(v - x_old)``. The second factor is the
    *residual* of each sample, so the accumulated quantity is of the size of the variance throughout.
    Returns the population variance (divide by ``n``), to match :func:`variance_naive`.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot compute a variance of an empty sample")
    mean = 0.0
    m2 = 0.0
    for k, v in enumerate(x, start=1):
        delta = float(v) - mean
        mean += delta / k
        m2 += delta * (float(v) - mean)
    return m2 / x.size


def variance_comparison(values: Sequence[float] | np.ndarray) -> VarianceComparison:
    """Both one-pass variances against ``math.fsum``-based exact arithmetic.

    The reference subtracts the exact mean (a correctly rounded ``fsum`` quotient) from each sample and
    sums the squares with ``fsum``, so it carries no cancellation of its own.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot compute a variance of an empty sample")
    exact_mean = math.fsum(float(v) for v in x) / x.size
    reference = math.fsum((float(v) - exact_mean) ** 2 for v in x) / x.size

    naive = variance_naive(x)
    welford = variance_welford(x)
    diagnostics: list[Diagnostic] = []
    naive_err = relative_error(naive, reference)
    if naive_err > 1e-10:
        diagnostics.append(
            explain("FLOAT.CANCELLATION", formula="mean(x^2) - mean(x)^2", mean=exact_mean,
                    variance=reference, relative_error=naive_err,
                    digits_lost=digits_lost(naive, reference))
        )
    return VarianceComparison(
        reference=float(reference),
        naive=float(naive),
        welford=float(welford),
        naive_relative_error=float(naive_err),
        welford_relative_error=float(relative_error(welford, reference)),
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- summation
def naive_sum(values: Sequence[float] | np.ndarray) -> float:
    """Left-to-right accumulation. Its error grows like ``n * eps * sum|x|``: the running total is
    rounded once per term, and each rounding is amplified by the size the total later reaches."""
    total = 0.0
    for v in values:
        total += float(v)
    return total


def kahan_sum(values: Sequence[float] | np.ndarray) -> float:
    """Kahan compensated summation: keep the lost low-order bits in a compensation term.

    ``y = v - c`` ; ``t = s + y`` ; ``c = (t - s) - y``. The compensation is subtracted from the next
    term, so the error no longer grows with ``n`` in the usual regime (its worst-case bound is
    ``2 eps sum|x|``, independent of ``n``). It assumes the compiler does not reassociate, which is true
    for CPython floats.
    """
    s = 0.0
    c = 0.0
    for v in values:
        y = float(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def neumaier_sum(values: Sequence[float] | np.ndarray) -> float:
    """Neumaier's variant of Kahan: also correct when a term is larger than the running total.

    Plain Kahan fails when ``|v| > |s|``, because the compensation ``(t - s) - y`` then evaluates to
    zero. Neumaier branches on the magnitudes and adds the compensation at the end, which makes the
    result as accurate as a doubly compensated sum for a wider range of inputs.
    """
    s = 0.0
    c = 0.0
    for v in values:
        v = float(v)
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


def pairwise_sum(values: Sequence[float] | np.ndarray) -> float:
    """Recursive bisection: ``O(log n)`` error growth instead of ``O(n)``, at no extra rounding cost.

    Each partial sum is short, so few roundings accumulate; the depth is ``log2 n`` and the error bound
    is ``eps log2(n) sum|x|``, which for a million terms is 20 roundings rather than a million.
    """
    x = np.asarray(values, dtype=np.float64).ravel()

    def rec(lo: int, hi: int) -> float:
        if hi - lo == 1:
            return float(x[lo])
        mid = (lo + hi) // 2
        return rec(lo, mid) + rec(mid, hi)

    if x.size == 0:
        return 0.0
    return rec(0, x.size)


@dataclass(frozen=True)
class SummationGrowth:
    """How each summation's error grows with the number of terms, with the measured exponent.

    ``sizes`` are the sample counts, and each ``*_error`` array is the relative error at that size
    against ``math.fsum``. ``growth_exponents`` fits ``error ~ n^q`` on the last half of the curve --
    ``q ~ 1/2`` for naive summation on random data (the random walk of roundings is the average case),
    ``q ~ 0`` for compensated or pairwise.
    """

    sizes: np.ndarray
    naive_error: np.ndarray
    pairwise_error: np.ndarray
    kahan_error: np.ndarray
    neumaier_error: np.ndarray
    growth_exponents: dict[str, float] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for the lab page."""
        return {
            "sizes": [int(v) for v in self.sizes],
            "naive_error": [float(v) for v in self.naive_error],
            "pairwise_error": [float(v) for v in self.pairwise_error],
            "kahan_error": [float(v) for v in self.kahan_error],
            "neumaier_error": [float(v) for v in self.neumaier_error],
            "growth_exponents": dict(self.growth_exponents),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def summation_error_growth(
    sizes: Sequence[int] = (10, 100, 1_000, 10_000, 100_000),
    *,
    seed: int = 0,
) -> SummationGrowth:
    """Measure the relative error of each summation method against ``fsum`` as ``n`` grows.

    The sample is uniform in ``[0, 1]``, so the partial sums grow like ``n`` and the *relative* error
    isolates the number of roundings instead of being confounded by the growth of the sum. The naive
    roundings add up like a random walk, which is why its exponent is about ``1/2`` and not ``1`` (``1``
    is the worst case, reached by data with one huge term and many tiny ones). ``fsum`` is exactly
    rounded, so it is a reference and not a competitor.

    A compensated method can report exactly zero error at these sizes; its exponent is then NaN, which
    means "at the floor of this measurement" and not "unknown".

    The exponent is fit on the upper half of the size range, where the asymptotic regime holds.
    """
    rng = np.random.default_rng(seed)
    biggest = int(max(sizes))
    data = rng.uniform(0.0, 1.0, size=biggest)
    rows = {name: [] for name in ("naive", "pairwise", "kahan", "neumaier")}
    for n in sizes:
        chunk = data[: int(n)]
        ref = math.fsum(float(v) for v in chunk)
        rows["naive"].append(relative_error(naive_sum(chunk), ref))
        rows["pairwise"].append(relative_error(pairwise_sum(chunk), ref))
        rows["kahan"].append(relative_error(kahan_sum(chunk), ref))
        rows["neumaier"].append(relative_error(neumaier_sum(chunk), ref))

    sizes_arr = np.asarray(sizes, dtype=np.float64)
    exponents: dict[str, float] = {}
    half = max(2, sizes_arr.size // 2)
    for name, values in rows.items():
        err = np.asarray(values, dtype=np.float64)
        positive = err > 0
        if positive.sum() >= 2:
            exponents[name] = float(np.polyfit(np.log(sizes_arr[positive][-half:]),
                                               np.log(err[positive][-half:]), 1)[0])
        else:
            exponents[name] = float("nan")

    diagnostics: list[Diagnostic] = []
    if exponents.get("naive", 0.0) > 0.2:
        diagnostics.append(
            explain("FLOAT.CANCELLATION", what="naive summation", growth_exponent=exponents["naive"],
                    largest_size=int(max(sizes)), final_error=rows["naive"][-1])
        )
    return SummationGrowth(
        sizes=sizes_arr,
        naive_error=np.asarray(rows["naive"]),
        pairwise_error=np.asarray(rows["pairwise"]),
        kahan_error=np.asarray(rows["kahan"]),
        neumaier_error=np.asarray(rows["neumaier"]),
        growth_exponents=exponents,
        diagnostics=tuple(diagnostics),
    )


# --------------------------------------------------------------------------- conditioning
@dataclass(frozen=True)
class ErrorBudget:
    """Forward and backward error against the condition number -- the verdict, with its evidence."""

    condition_number: float
    forward_error_rel: float
    backward_error_rel: float
    amplification_ratio: float
    digits_lost: float
    verdict: str
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def is_well_conditioned(self) -> bool:
        """Whether the condition number alone rules out rounding as the limiting factor."""
        return self.condition_number < 10.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "condition_number": self.condition_number,
            "forward_error_rel": self.forward_error_rel,
            "backward_error_rel": self.backward_error_rel,
            "amplification_ratio": self.amplification_ratio,
            "digits_lost": self.digits_lost,
            "verdict": self.verdict,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def error_budget(
    condition_number: float,
    forward_error_rel: float,
    backward_error_rel: float,
    *,
    tolerance: float = 10.0,
) -> ErrorBudget:
    """Decide whether an observed error is the problem's fault or the algorithm's.

    The first-order model is ``forward_error ~ cond * backward_error``: a backward-stable algorithm
    turns a perturbation of the input into at most a ``cond``-fold perturbation of the output. So:

    * ``forward <= tolerance * cond * backward`` -> the algorithm is doing its job; if the error is
      large it is because the *problem* amplifies. Verdict ``"problem"``.
    * ``forward > tolerance * cond * backward`` -> the algorithm amplifies more than the conditioning
      requires. Verdict ``"algorithm"``.
    * ``cond < 10`` -> verdict ``"well-conditioned"`` regardless, because the amplification is bounded
      by a small number and neither label is informative.

    ``tolerance`` is the factor by which the model may be off before the verdict flips (10 is generous
    on purpose: the model is first-order and the errors are measured).

    The verdict names what to change, which is the only actionable content: a bad condition number is
    fixed by reformulating the problem, a bad algorithm by replacing it.
    """
    cond = float(condition_number)
    forward = float(forward_error_rel)
    backward = float(backward_error_rel)
    if cond < 0 or forward < 0 or backward < 0 or not np.isfinite(cond):
        raise ValueError("condition number and errors must be finite and non-negative")

    amplification = float(forward / backward) if backward > 0 else float("inf")
    model = cond * backward
    diagnostics: list[Diagnostic] = []

    if cond < 10.0:
        verdict = "well-conditioned"
        diagnostics.append(explain("FLOAT.WELL_CONDITIONED", condition_number=cond,
                                   forward_error=forward, measured_digits_lost=0.0))
    elif forward <= tolerance * model:
        verdict = "problem"
        diagnostics.append(explain("FLOAT.PROBLEM_ILL_CONDITIONED", condition_number=cond,
                                   forward_error=forward, backward_error=backward,
                                   model_bound=model))
    else:
        verdict = "algorithm"
        diagnostics.append(explain("FLOAT.ALGORITHM_UNSTABLE", condition_number=cond,
                                   forward_error=forward, backward_error=backward,
                                   model_bound=model, ratio=forward / model if model > 0 else float("inf")))

    lost = float(-math.log10(forward)) if forward > 0 else float("inf")
    if lost < 1.0:
        diagnostics.append(explain("FLOAT.NO_SIGNIFICANT_DIGITS", condition_number=cond,
                                   forward_error=forward))
    return ErrorBudget(
        condition_number=cond,
        forward_error_rel=forward,
        backward_error_rel=backward,
        amplification_ratio=amplification,
        digits_lost=lost,
        verdict=verdict,
        diagnostics=tuple(diagnostics),
    )


def measured_condition_number(
    function: Callable[[float], float], x: float, *, relative_step: float = 1e-7
) -> float:
    """Measure ``cond = |df/f| / |dx/x|`` at ``x`` with a forward perturbation.

    This is the empirical definition, not a symbolic derivative: perturb the input by
    ``relative_step``, evaluate, and take the ratio of relative changes. It is one-sided, so for a
    function with a large second derivative use a small step (``relative_step ~ 1e-7`` in float64) --
    too small and the difference of two function values cancels, too large and the curvature moves the
    answer. Both limits are the same V-curve this package measures elsewhere.
    """
    x = float(x)
    if x == 0.0:
        raise ValueError("the relative condition number is undefined at x = 0")
    f0 = float(function(x))
    if f0 == 0.0:
        raise ValueError("the relative condition number is undefined where f(x) = 0")
    f1 = float(function(x * (1.0 + relative_step)))
    return float(abs((f1 - f0) / f0) / relative_step)


# --------------------------------------------------------------------------- interval arithmetic
@dataclass(frozen=True)
class Interval:
    """A closed interval with outward-rounded arithmetic -- a guaranteed enclosure.

    Every operation computes the exact real result of the four endpoint combinations and then widens by
    one ulp in each direction, so the returned interval provably contains the exact answer even though
    the arithmetic underneath is rounded. That is the difference between an error *estimate* (which can
    be wrong) and an error *bound* (which cannot, for the operations implemented here).

    Limitation: with float64 endpoints and one-ulp outward rounding, the width of the result is the
    accumulated rounding, which grows with the number of operations. An interval that is recomputed
    through a long chain eventually says only "it is somewhere in here", which is still true but
    useless -- interval arithmetic does not eliminate the conditioning of the problem, it makes the loss
    visible.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"interval with lo > hi: [{self.lo}, {self.hi}]")

    @classmethod
    def point(cls, x: float) -> Interval:
        """The degenerate interval ``[x, x]``."""
        return cls(float(x), float(x))

    @property
    def mid(self) -> float:
        """Midpoint (not necessarily representable inside the interval after rounding; it is)."""
        return 0.5 * (self.lo + self.hi)

    @property
    def width(self) -> float:
        """Absolute width."""
        return self.hi - self.lo

    @property
    def relative_width(self) -> float:
        """Width relative to the largest endpoint magnitude -- the number of digits still known."""
        scale = max(abs(self.lo), abs(self.hi))
        return float(self.width / scale) if scale > 0 else 0.0

    def contains(self, x: float) -> bool:
        """Whether ``x`` lies in the interval."""
        return self.lo <= x <= self.hi

    @staticmethod
    def _widen(lo: float, hi: float) -> Interval:
        return Interval(float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf)))

    def __add__(self, other: Interval) -> Interval:
        return Interval._widen(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: Interval) -> Interval:
        return Interval._widen(self.lo - other.hi, self.hi - other.lo)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: Interval) -> Interval:
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval._widen(min(products), max(products))

    def __truediv__(self, other: Interval) -> Interval:
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError(f"interval divisor {other} contains zero")
        quotients = (self.lo / other.lo, self.lo / other.hi, self.hi / other.lo, self.hi / other.hi)
        return Interval._widen(min(quotients), max(quotients))

    def sqrt(self) -> Interval:
        """Square root of an interval that must be non-negative: ``[sqrt(lo), sqrt(hi)]`` (monotone)."""
        if self.lo < 0.0:
            raise ValueError(f"sqrt of an interval with a negative lower bound: {self}")
        return Interval._widen(math.sqrt(self.lo), math.sqrt(self.hi))

    def __repr__(self) -> str:
        return f"Interval([{self.lo!r}, {self.hi!r}])"


def interval_quadratic_roots(a: float, b: float, c: float) -> tuple[Interval, Interval]:
    """Enclose both real roots of the quadratic using interval arithmetic on the *naive* formula.

    The point of doing it on the naive formula is that the enclosure for the small root comes out wide
    -- not because the arithmetic is wrong, but because the formula genuinely cannot determine that root
    to more than a few digits in double precision. The wide interval is the honest answer, and it is
    what a guaranteed bound looks like when the algorithm is unstable.

    Raises ``ValueError`` for a negative discriminant (no real enclosure exists).
    """
    ia, ib, ic = Interval.point(a), Interval.point(b), Interval.point(c)
    disc = ib * ib - Interval.point(4.0) * ia * ic
    if disc.lo < 0.0:
        raise ValueError(f"the discriminant enclosure {disc} crosses zero: no guaranteed real roots")
    sq = disc.sqrt()
    two_a = Interval.point(2.0) * ia
    r_plus = (-ib + sq) / two_a
    r_minus = (-ib - sq) / two_a
    return r_plus, r_minus


# --------------------------------------------------------------------------- precision floors
@dataclass(frozen=True)
class PrecisionFloor:
    """The residual floor a dtype imposes: the smallest residual a function can report there."""

    dtype: str
    eps: float
    mean_abs_residual: float
    max_abs_residual: float
    at_floor: bool
    digits_lost_vs_float64: float
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {
            "dtype": self.dtype,
            "eps": self.eps,
            "mean_abs_residual": self.mean_abs_residual,
            "max_abs_residual": self.max_abs_residual,
            "at_floor": self.at_floor,
            "digits_lost_vs_float64": self.digits_lost_vs_float64,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def residual_precision_floor(
    residual: Callable[[np.ndarray], np.ndarray],
    x: Sequence[float] | np.ndarray,
    *,
    dtype: str = "float32",
    tolerance_factor: float = 10.0,
) -> PrecisionFloor:
    """Measure the residual a function reports for a solution it should treat as exact.

    ``residual`` is evaluated in ``dtype`` on the points ``x``, all of which are points where the exact
    residual is zero (a manufactured-solution check: ``u = sin(x)`` in ``u'' + u = 0``). Whatever
    nonzero value comes back is the rounding floor of the dtype on this expression -- nothing about the
    model. A training loss built on that residual cannot go below it, which is why a float32 PINN
    plateauing at ``1e-7`` is arithmetic, not physics.

    ``at_floor`` is true when the measured residual is within ``tolerance_factor`` of
    ``eps * scale``, where ``scale`` is the magnitude of the terms being combined (taken as the largest
    residual over the dtypes measured, which cancels out of the comparison).

    The limitation: this measures the floor of *one* expression in *one* dtype. A different algebraic
    form of the same residual has a different floor, and rewriting the residual is often the fix.
    """
    xd = np.asarray(x, dtype=dtype)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r = np.asarray(residual(xd), dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("the residual returned no values")
    eps = machine_epsilon(dtype)
    mean_abs = float(np.mean(np.abs(r)))
    max_abs = float(np.max(np.abs(r)))
    scale = max(float(np.max(np.abs(np.asarray(x, dtype=np.float64)))), 1.0)
    at_floor = bool(max_abs <= tolerance_factor * eps * scale)

    diagnostics: list[Diagnostic] = []
    if dtype != "float64":
        diagnostics.append(
            explain("FLOAT.MIXED_PRECISION_RESIDUAL", dtype=dtype, eps=eps, mean_abs=mean_abs,
                    max_abs=max_abs, scale=scale)
        )
    return PrecisionFloor(
        dtype=dtype,
        eps=eps,
        mean_abs_residual=mean_abs,
        max_abs_residual=max_abs,
        at_floor=at_floor,
        digits_lost_vs_float64=float(-math.log10(machine_epsilon("float64")) + math.log10(eps)),
        diagnostics=tuple(diagnostics),
    )
