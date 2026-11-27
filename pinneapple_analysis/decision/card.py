"""The one-page card: five numbers and a verdict, which is what a reviewer asks a surrogate to hand over.

A surrogate that leaves the library without a card is not deployable, because nothing about it says whether the
design it recommends is the design the reference solver would recommend. The card puts the five decision-level
numbers on one page and states a verdict against thresholds that are written down, so the statement "the
surrogate is accurate enough" becomes falsifiable: it is accurate enough for this design set at this tolerance,
or it is not.

The five numbers
----------------
1. flip rate (fraction of design pairs whose order the surrogate reverses),
2. Spearman rho over the design set,
3. Kendall tau-b over the design set,
4. the fraction of the reference top-k that the surrogate also ranks in its top-k,
5. the relative regret: how much of the achievable improvement the surrogate's chosen design gives up, as a
   fraction of the spread of the reference values.

with the number of designs the card was computed on, the tolerance that produced the flip rate, and the verdict
``"decision-safe"``, ``"review"`` or ``"unsafe"``.

Usage
-----
>>> from pinneapple_analysis.decision import decision_card
>>> card = decision_card([1.0, 2.0, 5.0, 4.0, 3.0], [1.0, 2.0, 4.0, 5.0, 3.0])
>>> card.verdict
'review'
>>> card.flip_rate
0.1
>>> print(card.summary())
REVIEW: flip rate 10.0 %, Spearman 0.900, Kendall 0.800, top-1 kept 100 %, regret 0.0 % of the spread -- 5 designs, lower is better, margin 0 sd

``card.to_markdown()`` then gives the one page, ready to paste into a report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .metrics import FlipRate, RankPreservation, Regret, flip_rate, rank_preservation, regret

__all__ = ["DecisionThresholds", "DecisionCard", "decision_card"]

_VERDICTS = ("decision-safe", "review", "unsafe")


@dataclass(frozen=True)
class DecisionThresholds:
    """Where the verdict switches from safe to review to unsafe. **A convention, not a measured result.**

    Each gated number has two boundaries. ``*_safe`` is the value below (or above, for the correlations) which the
    card calls ``"decision-safe"``; ``*_review`` is the value past which it calls the surrogate ``"unsafe"``;
    between the two the verdict is ``"review"``. The defaults are the ones the issue states as the target for a
    sellable card -- a flip rate below 5 %, a rank correlation above 0.9, four fifths of the top-k kept and a
    regret below 5 % of the spread -- with the review band set at three times the safe value for the errors and a
    step below for the correlations so that a model that is clearly worse than the target is called unsafe and
    not review.

    These are an engineering choice, not something anyone measured: they say what a designer is willing to put in
    front of a customer, not what a given solver delivers. Change them by passing ``thresholds=`` to
    :func:`decision_card` -- the card prints the thresholds it used, so a card is always readable without the
    code that produced it, and a tighter threshold does not make a surrogate better. The Kendall tau is reported
    but not gated: it moves with Spearman and gating both would double-count the same disagreement.
    """

    flip_rate_safe: float = 0.05
    flip_rate_review: float = 0.15
    spearman_safe: float = 0.90
    spearman_review: float = 0.75
    top_k_safe: float = 0.80
    top_k_review: float = 0.60
    regret_safe: float = 0.05
    regret_review: float = 0.20

    def __post_init__(self):
        pairs = (("flip_rate", self.flip_rate_safe, self.flip_rate_review),
                 ("spearman", -self.spearman_safe, -self.spearman_review),
                 ("top_k", -self.top_k_safe, -self.top_k_review),
                 ("regret", self.regret_safe, self.regret_review))
        for name, safe, review in pairs:
            if not safe < review:
                raise ValueError(f"{name}: the safe boundary must be strictly inside the review boundary")

    def gate(self, metric: str) -> tuple[str, float, float, bool]:
        """``(name, safe, review, higher_is_better)`` for one of the four gated numbers."""
        table = {
            "flip_rate": ("Flip rate", self.flip_rate_safe, self.flip_rate_review, False),
            "spearman": ("Spearman rho", self.spearman_safe, self.spearman_review, True),
            "top_k": ("Top-k preserved", self.top_k_safe, self.top_k_review, True),
            "regret": ("Relative regret", self.regret_safe, self.regret_review, False),
        }
        return table[metric]


@dataclass(frozen=True)
class DecisionCard:
    """The one-page statement of a surrogate's decision-level quality. Built by :func:`decision_card`."""

    n: int
    flip_rate: float
    spearman: float
    kendall_tau: float
    top_k: int
    top_k_kept: float
    regret_relative: float
    regret_absolute: float
    verdict: str
    reasons: tuple[str, ...]
    margin: float
    tolerance: float
    lower_is_better: bool
    thresholds: DecisionThresholds
    inverted: tuple[int, ...]
    inverted_names: tuple[str, ...]
    choice_flip: bool
    best_truth: int
    best_prediction: int
    chosen_value: float
    best_value: float
    span: float
    n_pairs: int
    n_flipped_pairs: int

    # ------------------------------------------------------------------ views
    def numbers(self) -> dict[str, float]:
        """The five numbers of the card, in the order the markdown prints them."""
        return {
            "flip_rate": self.flip_rate,
            "spearman": self.spearman,
            "kendall_tau": self.kendall_tau,
            "top_k_kept": self.top_k_kept,
            "regret_relative": self.regret_relative,
        }

    def to_dict(self) -> dict[str, Any]:
        """The card as plain Python (dataclass, tuples and numpy values converted), ready for JSON."""
        data = asdict(self)
        data["names"] = list(self.inverted_names)
        return _jsonable(data)

    def summary(self) -> str:
        """One line: verdict, the five numbers and how many designs they were computed on."""
        direction = "lower is better" if self.lower_is_better else "higher is better"
        return (f"{self.verdict.upper()}: flip rate {100 * self.flip_rate:.1f} %, Spearman {self.spearman:.3f}, "
                f"Kendall {self.kendall_tau:.3f}, top-{self.top_k} kept {100 * self.top_k_kept:.0f} %, "
                f"regret {100 * self.regret_relative:.1f} % of the spread -- {self.n} designs, {direction}, "
                f"margin {self.margin:g} sd")

    # ------------------------------------------------------------------ the one page
    def to_markdown(self) -> str:
        """The card as a one-page markdown table, with the verdict, the thresholds and the scope statement."""
        rows = [
            ("1", "Flip rate", f"{100 * self.flip_rate:.1f} %",
             f"{self.n_flipped_pairs} of {self.n_pairs} comparable pairs",
             f"<= {100 * self.thresholds.flip_rate_safe:.0f} %"),
            ("2", "Spearman rho", f"{self.spearman:.3f}", "rank correlation over the set",
             f">= {self.thresholds.spearman_safe:.2f}"),
            ("3", "Kendall tau-b", f"{self.kendall_tau:.3f}", "rank correlation, tie corrected", "reported"),
            ("4", f"Top-{self.top_k} preserved", f"{100 * self.top_k_kept:.0f} %",
             f"{int(round(self.top_k_kept * self.top_k))} of {self.top_k} designs",
             f">= {100 * self.thresholds.top_k_safe:.0f} %"),
            ("5", "Relative regret", f"{100 * self.regret_relative:.1f} %",
             f"{self.regret_absolute:.4g} of a reference spread of {self.span:.4g}",
             f"<= {100 * self.thresholds.regret_safe:.0f} %"),
        ]
        lines = [
            "# Decision card -- surrogate against the reference solver",
            "",
            f"**Verdict: {self.verdict}**"
            + ("" if not self.reasons else f" ({self.reasons[0]})")
            + ".",
            "",
            "| # | Number | Value | Detail | Gate |",
            "|---|--------|-------|--------|------|",
        ]
        lines += [f"| {' | '.join(row)} |" for row in rows]
        lines += [""]
        direction = "minimised" if self.lower_is_better else "maximised"
        lines.append(f"- Design set: **{self.n} designs**, objective {direction}, "
                     f"{self.n_pairs} comparable pairs at margin {self.margin:g} sd "
                     f"(tolerance {self.tolerance:.3g} in objective units).")
        chosen = ("the same design as the reference" if not self.choice_flip
                  else f"a different design (index {self.best_prediction} against {self.best_truth})")
        lines.append(f"- Chosen design: {chosen}, at {self.chosen_value:.4g} against the reference best "
                     f"{self.best_value:.4g}.")
        if self.inverted:
            if self.inverted_names:
                names = ", ".join(self.inverted_names)
                lines.append(f"- Designs involved in a reversal ({len(self.inverted)}): {names}.")
            else:
                lines.append(f"- Designs involved in a reversal ({len(self.inverted)}): indices "
                             + ", ".join(str(i) for i in self.inverted) + ".")
        else:
            lines.append("- Designs involved in a reversal: none.")
        if self.reasons:
            lines.append("- Gates not met:")
            lines += [f"  - {reason}" for reason in self.reasons]
        else:
            lines.append("- Gates not met: none.")
        lines += [
            "",
            "## How to read this",
            "",
            "The thresholds in the last column are an engineering convention, not a measured result: "
            f"flip rate <= {100 * self.thresholds.flip_rate_safe:.0f} % (review up to "
            f"{100 * self.thresholds.flip_rate_review:.0f} %), Spearman >= {self.thresholds.spearman_safe:.2f} "
            f"(review down to {self.thresholds.spearman_review:.2f}), top-k >= "
            f"{100 * self.thresholds.top_k_safe:.0f} % (review down to {100 * self.thresholds.top_k_review:.0f} %), "
            f"relative regret <= {100 * self.thresholds.regret_safe:.0f} % (review up to "
            f"{100 * self.thresholds.regret_review:.0f} %). They belong to the customer: change them, and quote "
            "the flip rate with the tolerance that produced it.",
            "",
            "The card states the decision quality of this surrogate on this design set at this tolerance and "
            "nothing else. It is not a statement about a design outside the sampled space, about another "
            "objective, or about another design family. A surrogate without a card is not deployable.",
            "",
        ]
        return "\n".join(lines)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _level(value: float, safe: float, review: float, higher_is_better: bool) -> int:
    """0 = safe, 1 = review, 2 = unsafe. A number that could not be computed is unsafe, never safe."""
    if not np.isfinite(value):
        return 2
    if higher_is_better:
        if value >= safe:
            return 0
        return 1 if value >= review else 2
    if value <= safe:
        return 0
    return 1 if value <= review else 2


def _reason(label: str, value: float, safe: float, review: float, higher_is_better: bool,
            level: int, percent: bool = False) -> str:
    if not np.isfinite(value):
        return f"{label} could not be computed (constant vector): reported as not run"
    bound = safe if level == 1 else review
    verb = "below" if higher_is_better else "above"
    word = "safe gate" if level == 1 else "review gate"
    if percent:
        shown, bar = f"{100 * value:.1f} %", f"{100 * bound:.0f} %"
    else:
        shown, bar = f"{value:.3f}", f"{bound:.3f}"
    return f"{label} {shown} is {verb} the {word} of {bar}"


def decision_card(truth, prediction, *, names=None, lower_is_better: bool = True, margin: float = 0.0,
                  top_k: int | None = None,
                  thresholds: DecisionThresholds | None = None) -> DecisionCard:
    """Combine the decision metrics into the one-page card and its verdict.

    Parameters
    ----------
    truth, prediction
        One value per design from the reference solver and from the surrogate, on a test split disjoint from the
        fit split. Lower is better unless ``lower_is_better=False``.
    names
        Optional design names or case identifiers, one per design; the flipped designs are then printed by name
        in the card, which is how a reviewer finds the design to re-run.
    lower_is_better
        Direction of "better"; drives the chosen design, the top-k sets and the sign of the regret.
    margin
        The decision tolerance, as a fraction of the standard deviation of the reference vector; see
        :func:`pinneapple_analysis.decision.metrics.flip_rate`. It is printed on the card, because a flip rate
        without its tolerance is not a measurement.
    top_k
        Size of the top-k set kept by the surrogate; defaults to 10 % of the designs, at least one.
    thresholds
        The gates of the verdict, defaulting to :class:`DecisionThresholds`. **These thresholds are a convention
        and not a measured result**: they encode what a designer is willing to put in front of a customer, so they
        are printed on the card and passed in explicitly by anyone who wants a different bar. Tightening a
        threshold does not make a surrogate better and loosening one does not make the reference solver agree
        with it.

    Returns
    -------
    DecisionCard
        The five numbers (flip rate, Spearman, Kendall, top-k preserved, relative regret), the sample count, the
        verdict ``"decision-safe"`` / ``"review"`` / ``"unsafe"`` and the list of gates that were not met.

    Notes
    -----
    The verdict takes the worst of four gates -- flip rate, Spearman rho, top-k preserved and relative regret; the
    Kendall tau is reported but not gated. Every gate must be in the safe band for ``"decision-safe"``, and no
    gate may be in the unsafe band for ``"review"``. A number that could not be computed (Spearman on a constant
    vector) is treated as unsafe and reported as not run, never as a default value.
    """
    gates = DecisionThresholds() if thresholds is None else thresholds
    if names is not None:
        names = tuple(str(v) for v in names)
        if len(names) != len(np.asarray(truth).reshape(-1)):
            raise ValueError(f"names has {len(names)} entries against {len(np.asarray(truth).reshape(-1))} designs")

    fr: FlipRate = flip_rate(truth, prediction, margin=margin, lower_is_better=lower_is_better)
    rp: RankPreservation = rank_preservation(truth, prediction, top_k=top_k,
                                             lower_is_better=lower_is_better)
    rg: Regret = regret(truth, prediction, lower_is_better=lower_is_better)

    values = {"flip_rate": fr.flip_rate, "spearman": rp.spearman, "top_k": rp.top_k_kept,
              "regret": rg.relative}
    percent = {"flip_rate": True, "regret": True}
    levels, reasons = {}, []
    for key, value in values.items():
        label, safe, review, higher = gates.gate(key)
        level = _level(value, safe, review, higher)
        levels[key] = level
        if level:
            reasons.append(_reason(label, value, safe, review, higher, level, percent.get(key, False)))

    worst = max(levels.values())
    verdict = _VERDICTS[worst]
    inverted_names = tuple(names[i] for i in fr.inverted) if names is not None else ()

    return DecisionCard(
        n=int(rp.n),
        flip_rate=fr.flip_rate,
        spearman=rp.spearman,
        kendall_tau=rp.kendall_tau,
        top_k=rp.top_k,
        top_k_kept=rp.top_k_kept,
        regret_relative=rg.relative,
        regret_absolute=rg.absolute,
        verdict=verdict,
        reasons=tuple(reasons),
        margin=float(margin),
        tolerance=fr.tolerance,
        lower_is_better=bool(lower_is_better),
        thresholds=gates,
        inverted=fr.inverted,
        inverted_names=inverted_names,
        choice_flip=fr.choice_flip,
        best_truth=fr.best_truth,
        best_prediction=fr.best_prediction,
        chosen_value=rg.truth_at_chosen,
        best_value=rg.best_value,
        span=rg.span,
        n_pairs=fr.n_pairs,
        n_flipped_pairs=fr.n_flipped_pairs,
    )
