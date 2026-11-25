"""Which regime a problem is in, and what to change about it (#207, roadmap X23).

A stiff ODE, a boundary layer, a shock, an oscillation and a two-scale spectrum all fail for *different*
reasons, and each has a different remedy: an implicit solver, more points where the gradient is, a domain
split, Fourier features, a curriculum. The diagnosis below is a **rule table over measured numbers** --
not an agent, not a language model, not a guess. Every regime it reports comes with the number that
triggered it, and every strategy comes with the reason it is the remedy for that regime.

Two entry points, one rule set
------------------------------
``diagnose_ode``
    Samples the stiffness ratio and the Jacobian spectrum along a coarse implicit trajectory of
    ``y' = f(t, y)``. Detects ``stiff`` and ``oscillatory``.
``diagnose_field``
    Measures a field on a line: how concentrated the gradient is, how thick the thin layer is, where that
    layer sits, how many oscillations the field has, and the two-band structure of its spectrum. Detects
    ``boundary_layer``, ``shock``, ``high_frequency`` and ``multiscale``.

Both build a :class:`RegimeEvidence` and hand it to :func:`diagnose`, which is the rule table. That split
is deliberate: the numbers can be measured by someone else (an OpenFOAM slice, an experiment, a training
loss history) and still go through the same rules, and the rules can be inspected and contested without
touching the measurement.

The three worked cases
----------------------
The issue asks for a diagnosis report on three known cases; they are the tests in
``tests/test_numerics_regimes.py`` and the numbers are measured, not assumed:

============================  =================================  ===============================
case                          measurement                        regimes reported
============================  =================================  ===============================
Robertson kinetics            stiffness ratio ~4e8               ``stiff``
Blasius-like ``1-exp(-x/d)``  gradient x79, layer 0.5 % of L      ``boundary_layer``
``tanh((x-0.5)/d)``           gradient x46, layer 1 % of L,       ``shock``
                              interior, full jump
============================  =================================  ===============================

and the ones that must *not* fire: a cubic ``x^2(1-x)`` and an exponential with ``d = 0.3 L`` report
``smooth`` -- no strategy at all -- because a detector that fires on everything is not a detector.

Design rules this module follows
--------------------------------
* **One number per claim.** A regime is reported only when a named measure crosses a documented threshold,
  and the measure travels with the report. Thresholds are a :class:`Thresholds` value, so a caller can
  tighten them and see the report change.
* **The layer is measured, not assumed.** The thickness is the full width at half maximum of the gradient,
  the location is where the centre sits, and the jump is what the field did across the layer. A Gaussian
  bump has a thin interior layer but no jump, so it is not a shock.
* **Unresolved is said out loud.** Fewer than three grid points across a layer means the measured
  thickness is the grid, and the report carries ``REGIME.UNRESOLVED`` rather than a confident number.
* **No strategy out of nowhere.** Every strategy in the table is attached to the regime it answers, and
  a report only lists strategies for the regimes it actually found.

Not here yet
------------
The detectors are one-dimensional. A two-dimensional boundary layer (a wall-normal profile taken from a
slice) works, because the caller passes the profile, but anisotropy -- a layer that is thin in ``y`` and
thick in ``x`` -- is measured only along the line the caller hands over. Time-dependent problems are
covered through ``diagnose_ode`` and, when the caller has a field per time, by diagnosing the field at
which the jump is thinnest. Nothing here trains anything: the strategies are named and justified, the
choice of which to run is the caller's.

References
----------
* Hairer, E. and Wanner, G., *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic
  Problems*, 2nd ed., Springer, 1996 -- the stiffness ratio and why it sets the explicit step.
* Karniadakis, G.E. et al., "Physics-informed machine learning", *Nature Reviews Physics* 3, 2021 --
  the survey the adaptive-sampling / causal-training / XPINN strategy names come from.
* Wang, S., Teng, Y. and Perdikaris, P., "Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks", *SIAM J. Sci. Comput.* 43(5), 2021 -- time causality.
* Wang, S., Wang, H. and Perdikaris, P., "On the eigenvector bias of Fourier feature networks", *CMAME*
  384, 2021 -- the spectral bias the ``fourier_features`` strategy answers.
* Trefethen, L.N., *Spectral Methods in MATLAB*, SIAM, 2000 -- the two-band reading of a spectrum.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .diagnostics import Diagnostic, explain
from .ode import jacobian_spectrum, stiffness_report

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DiagnosisReport",
    "RegimeEvidence",
    "STRATEGIES",
    "Strategy",
    "Thresholds",
    "diagnose",
    "diagnose_field",
    "diagnose_ode",
    "strategies_for",
]

Regime = Literal[
    "stiff", "oscillatory", "boundary_layer", "shock", "sharp_feature", "high_frequency", "multiscale",
    "smooth",
]


@dataclass(frozen=True)
class Thresholds:
    """The bands every rule is written against, with the value each one was chosen from.

    ``stiffness_ratio``
        ``1e3``: from :func:`~.ode.stiffness_ratio` -- above it an explicit method's step is set by
        stability rather than by the accuracy of the solution.
    ``oscillation_index``
        ``0.5``: the largest ``|Im lambda| / |lambda|`` over the local spectrum. Above it the fastest
        eigenvalue makes an angle over 30 degrees with the real axis, so the solution turns before it
        decays.
    ``layer_thickness``
        ``0.02``: a layer thinner than 2 % of the domain. A uniform grid then spends fewer than one point
        per fifty on the region where the field is not flat.
    ``gradient_concentration``
        ``20``: ``max|du/dx| / (range / L)``. A smooth profile that fills the domain measures 2 to 8; a
        layer measures tens to hundreds.
    ``oscillation_count``
        ``4``: complete turning points of the field across the line. Below it the field is a layer or a
        bump; above it, points per wavelength is the limiting number.
    ``band_ratio`` and ``spectral_gap``
        ``10`` and ``1e-4``: the fine band peaks at least ten times higher in wavenumber than the coarse
        one, and the energy between them falls below ``1e-4`` of the peak. A smooth decay from ``k = 1``
        to the noise floor has no such dip, which is what separates "two scales" from "one sharp feature".
    ``resolved_points``
        ``3``: points across the measured layer below which the thickness is the grid and not the feature.
    ``jump_ratio``
        ``0.5``: how much the field changed from one end of the line to the other relative to its range.
        Near one is a plateau on each side -- a jump; near zero is a feature that returns to where it
        started, which is sharp but is not a shock.
    """

    stiffness_ratio: float = 1e3
    oscillation_index: float = 0.5
    layer_thickness: float = 2e-2
    gradient_concentration: float = 20.0
    oscillation_count: int = 4
    band_ratio: float = 10.0
    spectral_gap: float = 1e-4
    resolved_points: float = 3.0
    jump_ratio: float = 0.5


DEFAULT_THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------- strategies
@dataclass(frozen=True)
class Strategy:
    """A change to make, the regime that calls for it, and why it answers that regime."""

    code: str
    regime: Regime
    title: str
    action: str
    why: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form."""
        return {"code": self.code, "regime": self.regime, "title": self.title,
                "action": self.action, "why": self.why}

    def format(self) -> str:
        """``[stiff] Implicit time stepping -- ...``."""
        return f"[{self.regime}] {self.title}: {self.action}"


#: regime -> the strategies that answer it, in the order a run should try them. Data, not behaviour:
#: adding a strategy is adding a row, and the report only ever shows the rows for regimes it found.
STRATEGIES: dict[str, tuple[Strategy, ...]] = {
    "stiff": (
        Strategy(
            "implicit_solver", "stiff", "Implicit time stepping (BDF / Radau IIA)",
            "Integrate with an implicit, L-stable method (BDF of order 1-5, Radau IIA) instead of an "
            "explicit one.",
            "With a stiffness ratio of this size the explicit step is set by the fastest mode, which is "
            "not the mode anyone wants resolved. An L-stable method damps that mode instead of tracking "
            "it, so the step follows the solution.",
        ),
        Strategy(
            "adaptive_time_stepping", "stiff", "Step control on the local error",
            "Drive the step with an embedded error estimate and a minimum step, and report the steps "
            "rejected for stability separately from those rejected for accuracy.",
            "The ratio changes along the trajectory; a fixed step is either unstable where the problem is "
            "fast or wasteful where it is slow. Rejected steps are the measurement of where it changed.",
        ),
        Strategy(
            "time_marching", "stiff", "Split the time domain into windows",
            "Train or integrate one window at a time, carrying the state forward, instead of the whole "
            "horizon at once.",
            "The fast transient lives at the start and the slow manifold afterwards. One network or one "
            "step size cannot serve both, and windowing gives each window its own.",
        ),
    ),
    "oscillatory": (
        Strategy(
            "fourier_features", "oscillatory", "Sinusoidal features or activations",
            "Map the inputs through Fourier features (or use SIREN) so the network can represent a "
            "turning solution.",
            "The spectrum here has a large imaginary part, so the solution turns before it decays. A "
            "tanh/ReLU MLP is biased towards smooth monotone maps and fits a decay where the answer "
            "oscillates.",
        ),
        Strategy(
            "adaptive_sampling", "oscillatory", "Sample where the residual is largest",
            "Resample collocation points on the residual (or on the curvature) instead of uniformly in "
            "space and time.",
            "Uniform points place almost nothing on the steep part of a cycle, so the residual there "
            "never enters the loss.",
        ),
        Strategy(
            "causal_training", "oscillatory", "Time causality in the loss",
            "Weight the time slices so that an earlier slice has to be learned before a later one can "
            "contribute.",
            "An oscillatory solution is easy to fit late and hard to fit early; without a causal weight "
            "the optimiser settles on the late cycles and leaves the initial transient wrong.",
        ),
    ),
    "boundary_layer": (
        Strategy(
            "adaptive_sampling", "boundary_layer", "Concentrate points in the layer",
            "Resample the collocation points so the layer holds a fixed fraction of them -- by residual, "
            "by curvature, or by a known wall-normal stretching.",
            "The loss on a uniform grid is dominated by the flat part of the domain, where the residual "
            "is already small; the layer contributes a few terms out of thousands.",
        ),
        Strategy(
            "coordinate_scaling", "boundary_layer", "Scale the wall-normal coordinate",
            "Map the wall-normal coordinate so the layer has thickness O(1) -- a boundary-layer "
            "stretching, or a nondimensionalisation of the problem before training.",
            "A layer of thickness 1 % asks the network to resolve a feature a hundred times thinner than "
            "the domain in the same units as the domain. Rescaling puts the feature back at the scale "
            "the optimiser works at.",
        ),
        Strategy(
            "curriculum", "boundary_layer", "Ramp the problem to the layer",
            "Solve a sequence of problems whose layer thickness decreases: a larger diffusion or "
            "viscosity first, the target value last, warm-starting each stage.",
            "A thick layer is a problem the network can learn; its solution is a good initialisation for "
            "the thin one, and the failure mode of starting thin is no learning at all.",
        ),
        Strategy(
            "xpinn", "boundary_layer", "Decompose the domain at the layer edge",
            "Split the domain into an inner (layer) and an outer region with their own networks and "
            "interface conditions.",
            "Two networks remove the conflict between a region that needs fine resolution and one that "
            "needs none, instead of asking one network to be both.",
        ),
    ),
    "shock": (
        Strategy(
            "adaptive_sampling", "shock", "Refine around the jump",
            "Add points in a band around the measured jump, and re-measure the jump after each cycle.",
            "The residual of a smooth network is large in a band around a discontinuity and the L2 loss "
            "averages it away; points are the only thing that changes its weight.",
        ),
        Strategy(
            "causal_training", "shock", "Causal weighting in time",
            "For a shock that forms in time, weight the loss so an earlier time must be learned first.",
            "A jump that has already formed is sharp and locally wrong if the flow before it was never "
            "learned; causality removes the incentive to fit the late sharp state first.",
        ),
        Strategy(
            "viscosity_continuation", "shock", "Continue in viscosity",
            "Solve for a viscosity that smears the jump, then reduce it in stages towards the target, "
            "warm-starting each stage.",
            "A smeared jump is smooth and learnable; the target is not. Continuation walks the solution "
            "from one to the other instead of asking for the end point directly.",
        ),
        Strategy(
            "xpinn", "shock", "Split the domain at the jump",
            "Two networks either side of the jump, with the Rankine-Hugoniot relation imposed at the "
            "interface.",
            "A single smooth network has to represent a function whose derivative does not exist; a split "
            "puts the non-smoothness in the interface condition, which is a physics statement and not a "
            "fitting problem.",
        ),
    ),
    "sharp_feature": (
        Strategy(
            "adaptive_sampling", "sharp_feature", "Refine around the feature",
            "Add points where the gradient is, and re-measure the thickness after each cycle.",
            "The feature occupies a fraction of the domain that a uniform grid barely samples, so the "
            "residual that matters is a handful of terms out of thousands.",
        ),
        Strategy(
            "xpinn", "sharp_feature", "Split the domain at the feature",
            "A network for the feature and a network for the rest, with the value and the flux matched at "
            "the interface.",
            "One network asked to be sharp in one place and flat everywhere else spends its capacity on "
            "the compromise.",
        ),
    ),
    "high_frequency": (
        Strategy(
            "fourier_features", "high_frequency", "Fourier features at the measured wavenumber",
            "Feed the network with sin/cos of the measured dominant wavenumber (and its first harmonics) "
            "rather than the raw coordinate.",
            "A plain MLP is spectrally biased: it fits the envelope of an oscillation and misses the "
            "cycles, however long it trains. Features at the right wavenumber put the oscillation inside "
            "the model's reach.",
        ),
        Strategy(
            "adaptive_sampling", "high_frequency", "Resolve every wavelength",
            "Check the points per wavelength before training and resample until it is adequate; report "
            "the error per wavelength, not one global number.",
            "No model can learn a cycle it does not sample, and a global error hides the fact that the "
            "error lives in the oscillation.",
        ),
    ),
    "multiscale": (
        Strategy(
            "curriculum", "multiscale", "Learn the coarse scale first",
            "Filter the target, fit the coarse band, then add the fine band -- or weight the loss by "
            "wavenumber band.",
            "The two bands carry energy at very different levels; a joint loss is dominated by the coarse "
            "one, and the fine one stalls at the level of the coarse error.",
        ),
        Strategy(
            "xpinn", "multiscale", "One network per band",
            "Decompose the domain (or the field) so each network is responsible for one scale.",
            "A network that has to represent both scales allocates its capacity to the one that dominates "
            "the loss.",
        ),
        Strategy(
            "adaptive_sampling", "multiscale", "Sample the fine scale",
            "Place points according to the local wavenumber instead of uniformly.",
            "The fine band lives in a fraction of the domain, so a uniform grid under-resolves it while "
            "spending most points where the coarse band already fits.",
        ),
    ),
}


def strategies_for(regime: str) -> tuple[Strategy, ...]:
    """The strategies attached to ``regime`` (empty for ``smooth``, which needs none)."""
    return STRATEGIES.get(regime, ())


# --------------------------------------------------------------------------- evidence
@dataclass(frozen=True)
class RegimeEvidence:
    """The measured numbers the rules read. Every field is optional: an evidence from one source says
    nothing about the measures of the other, and a rule whose number is missing does not fire."""

    source: str
    stiffness_ratio: float | None = None
    oscillation_index: float | None = None
    gradient_concentration: float | None = None
    layer_thickness: float | None = None
    layer_location: str | None = None
    layer_points: float | None = None
    jump_ratio: float | None = None
    oscillation_count: int | None = None
    band_ratio: float | None = None
    spectral_gap: float | None = None
    samples: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form, with the unmeasured fields left out rather than written as ``None``."""
        out: dict[str, Any] = {"source": self.source}
        for key in ("stiffness_ratio", "oscillation_index", "gradient_concentration",
                    "layer_thickness", "layer_location", "layer_points", "jump_ratio",
                    "oscillation_count", "band_ratio", "spectral_gap", "samples"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(dict(self.extra))
        return out


@dataclass(frozen=True)
class DiagnosisReport:
    """Which regimes were found, the strategies that answer them, and the evidence behind both."""

    regimes: tuple[str, ...]
    strategies: tuple[Strategy, ...]
    evidence: RegimeEvidence
    diagnostics: tuple[Diagnostic, ...] = ()

    def strategies_for(self, regime: str) -> tuple[Strategy, ...]:
        """The strategies in this report that belong to ``regime``."""
        return tuple(s for s in self.strategies if s.regime == regime)

    def codes(self) -> tuple[str, ...]:
        """The strategy codes, in order."""
        return tuple(s.code for s in self.strategies)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form (what the Studio page and the Flow node receive)."""
        return {
            "regimes": list(self.regimes),
            "strategies": [s.to_dict() for s in self.strategies],
            "codes": list(self.codes()),
            "evidence": self.evidence.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def format(self) -> str:
        """A few lines for a CLI: the regimes, then one line per strategy."""
        head = "regimes: " + (", ".join(self.regimes) if self.regimes else "none")
        lines = [head]
        lines += [f"  - {s.title}: {s.action}" for s in self.strategies]
        lines += [f"  ! {d.format()}" for d in self.diagnostics if d.severity != "info"]
        return "\n".join(lines)


# --------------------------------------------------------------------------- field measurement
def _spectral_bands(energy: np.ndarray) -> tuple[int, int, float]:
    """``(k_dominant, k_highest, dip)`` of a one-sided power spectrum (DC already dropped).

    ``k_highest`` is the highest wavenumber still above ``1e-3`` of the peak, and ``dip`` is the smallest
    relative energy between the two -- near zero when the spectrum is two separated bands, of the order of
    ``1e-3`` when it decays smoothly from the peak to the floor, which is what a single sharp feature does.
    """
    if energy.size == 0 or not np.any(energy > 0):
        return 0, 0, 1.0
    relative = energy / float(np.max(energy))
    k_dominant = int(np.argmax(energy)) + 1
    above = np.where(relative >= 1e-3)[0]
    k_highest = int(above[-1]) + 1 if above.size else k_dominant
    dip = float(np.min(relative[k_dominant - 1:k_highest])) if k_highest > k_dominant else 1.0
    return k_dominant, k_highest, dip


def diagnose_field(
    values: Sequence[float] | np.ndarray,
    coordinates: Sequence[float] | np.ndarray | None = None,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> DiagnosisReport:
    """Diagnose a field sampled along a line: boundary layer, shock, oscillation or two scales.

    ``coordinates`` must be increasing; when omitted a uniform grid on ``[0, 1]`` is assumed. The layer
    is the full width at half maximum of ``|du/dx|``; its location is the centre of that width against the
    two ends; the jump is how much the field changed across the whole line relative to its range, which is
    what separates a shock (a jump, ~1) from a bump (no jump, ~0) even when both have a thin interior
    core.

        >>> import numpy as np
        >>> x = np.linspace(0.0, 1.0, 401)
        >>> report = diagnose_field(1.0 - np.exp(-x / 0.01), x)
        >>> report.regimes
        ('boundary_layer',)
    """
    u = np.asarray(values, dtype=np.float64).ravel()
    if u.size < 8:
        raise ValueError(f"a field needs at least 8 samples to have a gradient; got {u.size}")
    if not np.all(np.isfinite(u)):
        raise ValueError("the field contains non-finite values: the gradient would be undefined")
    if coordinates is None:
        x = np.linspace(0.0, 1.0, u.size)
    else:
        x = np.asarray(coordinates, dtype=np.float64).ravel()
        if x.size != u.size:
            raise ValueError(f"coordinates has {x.size} entries for {u.size} field values")
        if not np.all(np.diff(x) > 0):
            raise ValueError("coordinates must be strictly increasing")

    length = float(x[-1] - x[0])
    span = float(np.max(u) - np.min(u))
    gradient = np.gradient(u, x)
    peak = float(np.max(np.abs(gradient)))
    if span == 0.0 or peak == 0.0:
        evidence = RegimeEvidence(source="field", gradient_concentration=0.0, layer_thickness=1.0,
                                  layer_location="none", layer_points=float(u.size),
                                  jump_ratio=0.0, oscillation_count=0, band_ratio=1.0,
                                  spectral_gap=1.0, samples=int(u.size))
        return diagnose(evidence, thresholds=thresholds)

    typical = span / length
    concentration = peak / typical

    # the layer: full width at half maximum of the gradient, and what the field did across the whole line
    significant = np.abs(gradient) >= 1e-3 * peak
    above_half = np.where(np.abs(gradient) >= 0.5 * peak)[0]
    if above_half.size:
        width = float(x[above_half[-1]] - x[above_half[0]])
        centre = 0.5 * float(x[above_half[0]] + x[above_half[-1]])
    else:  # pragma: no cover - a peak is always at least half of itself
        width, centre = length, 0.5 * float(x[0] + x[-1])
    at_edge = min(centre - x[0], x[-1] - centre) < 0.15 * length
    jump = float(abs(u[-1] - u[0]) / span)
    turns = np.sign(gradient[significant])
    turns = turns[turns != 0]
    turning_points = int(np.sum(turns[1:] != turns[:-1])) // 2 if turns.size > 1 else 0

    energy = np.abs(np.fft.rfft(u - np.mean(u)))[1:] ** 2
    k_dominant, k_highest, dip = _spectral_bands(energy)
    band_ratio = k_highest / k_dominant if k_dominant else 1.0

    evidence = RegimeEvidence(
        source="field",
        gradient_concentration=concentration,
        layer_thickness=width / length,
        layer_location="boundary" if at_edge else "interior",
        layer_points=width / length * u.size,
        jump_ratio=jump,
        oscillation_count=turning_points,
        band_ratio=float(band_ratio),
        spectral_gap=dip,
        samples=int(u.size),
        extra={"dominant_wavenumber": k_dominant, "highest_wavenumber": k_highest},
    )
    return diagnose(evidence, thresholds=thresholds)


# --------------------------------------------------------------------------- ode measurement
def diagnose_ode(
    rhs: Callable[[float, np.ndarray], Any],
    y0: Sequence[float] | float,
    t_span: tuple[float, float],
    *,
    samples: int = 11,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> DiagnosisReport:
    """Diagnose ``y' = f(t, y)``: is it stiff, and does the local spectrum turn before it decays?

    The trajectory is the coarse implicit one :func:`~.ode.stiffness_report` already runs -- the stiffness
    is a property of the right-hand side, so the path only has to be in the right region, and sampling it
    before choosing a solver is exactly when the answer is useful.

        >>> from pinneapple_analysis.numerics.ode import robertson_rhs
        >>> report = diagnose_ode(robertson_rhs, [1.0, 0.0, 0.0], (0.0, 1e5))
        >>> report.regimes
        ('stiff',)
    """
    stiffness = stiffness_report(rhs, y0, t_span, samples=samples,
                                 threshold=thresholds.stiffness_ratio)
    times = np.asarray(stiffness["times"], dtype=np.float64)
    states = np.asarray(stiffness["states"], dtype=np.float64)
    oscillation = 0.0
    for t, state in zip(times, states, strict=False):
        if not np.all(np.isfinite(state)):
            continue
        spectrum = jacobian_spectrum(rhs, float(t), state)
        module = np.abs(spectrum)
        if module.size and float(np.max(module)) > 0:
            oscillation = max(oscillation, float(np.max(np.abs(np.imag(spectrum)) / np.max(module))))
    evidence = RegimeEvidence(
        source="ode",
        stiffness_ratio=float(stiffness["max_stiffness_ratio"]),
        oscillation_index=oscillation,
        samples=int(samples),
        extra={"times": [float(t) for t in times],
               "stiffness_ratios": [float(r) for r in stiffness["stiffness_ratios"]]},
    )
    return diagnose(evidence, thresholds=thresholds)


# --------------------------------------------------------------------------- the rules
def diagnose(
    evidence: RegimeEvidence, *, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> DiagnosisReport:
    """Apply the rule table to ``evidence`` and return the regimes, the strategies and the diagnostics.

    The rules are ordered, and the order is part of the answer. A thin interior core that also turns the
    field is a shock before it is an oscillation, because the jump is the feature that breaks the residual;
    a layer at a boundary is a boundary layer whatever its spectrum says. A field whose spectrum has two
    bands with a gap between them is ``multiscale`` before it is ``high_frequency``: many cycles with a
    spectral gap is a different problem from many cycles alone, and the gap is the more specific claim.
"""
    regimes: list[str] = []
    diagnostics: list[Diagnostic] = []

    if evidence.stiffness_ratio is not None and evidence.stiffness_ratio >= thresholds.stiffness_ratio:
        regimes.append("stiff")
        diagnostics.append(explain("ODE.STIFF", stiffness_ratio=evidence.stiffness_ratio,
                                   threshold=thresholds.stiffness_ratio))

    if evidence.oscillation_index is not None and evidence.oscillation_index >= thresholds.oscillation_index:
        regimes.append("oscillatory")
        diagnostics.append(explain("REGIME.OSCILLATORY", oscillation_index=evidence.oscillation_index,
                                   threshold=thresholds.oscillation_index))

    thin = (evidence.layer_thickness is not None
            and evidence.layer_thickness <= thresholds.layer_thickness
            and evidence.gradient_concentration is not None
            and evidence.gradient_concentration >= thresholds.gradient_concentration)
    if thin and evidence.layer_location == "boundary":
        regimes.append("boundary_layer")
        diagnostics.append(explain("REGIME.BOUNDARY_LAYER", layer_thickness=evidence.layer_thickness,
                                   gradient_concentration=evidence.gradient_concentration,
                                   threshold=thresholds.layer_thickness))
    elif thin and evidence.jump_ratio is not None and evidence.jump_ratio >= thresholds.jump_ratio:
        regimes.append("shock")
        diagnostics.append(explain("REGIME.SHOCK", layer_thickness=evidence.layer_thickness,
                                   gradient_concentration=evidence.gradient_concentration,
                                   jump_ratio=evidence.jump_ratio))
    elif thin:
        # a thin interior feature with no net change -- a bump, a spot, a hot streak. The residual has the
        # same shape as at a shock, but the physics does not: there is no Rankine-Hugoniot relation to
        # impose and no viscosity to continue in.
        regimes.append("sharp_feature")
        diagnostics.append(explain("REGIME.SHARP_FEATURE", layer_thickness=evidence.layer_thickness,
                                   gradient_concentration=evidence.gradient_concentration,
                                   jump_ratio=evidence.jump_ratio))
    elif (
        evidence.band_ratio is not None
        and evidence.spectral_gap is not None
        and evidence.band_ratio >= thresholds.band_ratio
        and evidence.spectral_gap <= thresholds.spectral_gap
    ):
        regimes.append("multiscale")
        diagnostics.append(explain("REGIME.MULTISCALE", band_ratio=evidence.band_ratio,
                                   spectral_gap=evidence.spectral_gap))
    elif (
        evidence.oscillation_count is not None
        and evidence.oscillation_count >= thresholds.oscillation_count
    ):
        regimes.append("high_frequency")
        diagnostics.append(explain("REGIME.HIGH_FREQUENCY", oscillation_count=evidence.oscillation_count,
                                   threshold=thresholds.oscillation_count))

    if (
        thin
        and evidence.layer_points is not None
        and evidence.layer_points < thresholds.resolved_points
    ):
        diagnostics.append(explain("REGIME.UNRESOLVED", layer_points=evidence.layer_points,
                                   samples=evidence.samples))

    if not regimes:
        regimes.append("smooth")
        diagnostics.append(explain("REGIME.SMOOTH", evidence=evidence.to_dict()))

    strategies: list[Strategy] = []
    seen: set[str] = set()
    for regime in regimes:
        for strategy in STRATEGIES.get(regime, ()):
            if strategy.code not in seen:
                seen.add(strategy.code)
                strategies.append(strategy)

    return DiagnosisReport(regimes=tuple(regimes), strategies=tuple(strategies), evidence=evidence,
                           diagnostics=tuple(diagnostics))
