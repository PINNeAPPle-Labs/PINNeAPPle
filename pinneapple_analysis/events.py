"""Generic event scoring: was the event there, how early, how strong, where, and how often did the model cry wolf.

The weather case studies score an event by how many days ahead it was well forecast
(``examples/weather_forecasting/run_use_cases.py``). The same questions apply to a thunderstorm, a motor failure, a
battery's knee, structural damage or a flood, and every one of them was answering them in its own script. This module
is that scorer once, for any field, any time axis and any kind of event.

An **event** is described (:class:`Event`) and **detected** on a single frame (:func:`detect`): a frame is the state
at one valid time, spatial axes after the first one. A forecast is then a set of frames indexed by the lead time it
was issued at, and :func:`event_score` lines them up against the truth at the same valid time. Nothing here assumes
what the field is: temperature on a plate, a residual of a motor, a water level, a wind field.

Four kinds (:data:`EVENT_KINDS`), and they are two different questions, not four spellings of one:

``threshold``, ``onset``
    A yes/no event. ``threshold`` is a level crossed somewhere in the region (a temperature, a pressure, a flood
    stage); ``onset`` is a change against the reference frame (the knee of a battery, damage appearing, a fault
    starting). These give a confusion matrix -- hits, misses, false alarms -- a critical success index and a lead
    time of detection. For an ensemble they give the probability of the event, its Brier score and the reliability
    curve (:func:`ensemble_event_score`).
``extremum``, ``track``
    The peak itself. A peak exists in every frame, so there is nothing to hit or miss: these are scored on *how far
    the forecast's peak is from the truth's* (``peak_error``) and, when coordinates are given, on *how far the peak
    moved* (``position_error``). ``track`` is the same with the location as the headline -- a moving feature.

What ``event_score`` reports, and what each number means:

``lead_time_ok``
    The longest lead at which the event was still scored correctly -- no miss and no false alarm at that lead -- and
    at every shorter lead too. It stops at the first lead that is wrong, so a lucky far lead does not count. ``-1``
    when the shortest lead is already wrong, and ``-1`` for a system that never announces the event at all (nothing
    detected is not a perfect forecast).
``peak_error`` / ``relative_peak_error``
    Mean absolute (and relative) difference between the forecast's event value and the truth's, over the cells where
    the event is present in both.
``position_error``
    Mean Euclidean distance between the forecast's event location and the truth's, in the units of ``coords``
    (grid indices when no ``coords`` is given). ``nan`` for a scalar field.
``detection_lead`` / ``timing_error``
    How many leads *ahead* of the truth the forecast first announced the event (positive is early, negative is
    late), and the mean absolute value of the same number. Only the inits where both detected it are counted.
``csi``
    ``hits / (hits + misses + false_alarms)`` -- the critical success index (threat score); ``nan`` when the
    denominator is zero.

``tolerance`` decides when a forecast event *is* the observed one: the value within ``value``, the lead within
``timing`` and the location within ``position``. A forecast event outside the tolerances is a false alarm and the
observed event is a miss -- the standard accounting, not a soft match.

Measured in ``tests/test_event_scoring.py`` (``2026-11-26``): Lorenz-96 with ``K = 8``, ``dt = 0.01``, 40 random
initial states, 30 leads, the event "the field's maximum above 3.0", tolerances ``value = 0.5``, ``timing = 2``.
Persistence (the frame never changes) against the same fields:

========================  =====  ======  ============  =====  =====  =============  ===========
forecast                  hits   misses  false alarms  CSI    POD    lead_time_ok   peak error
========================  =====  ======  ============  =====  =====  =============  ===========
persistence               470    780     670           0.245  0.376  0              0.957
the truth itself          1165   0       0             1.000  1.000  29             0.000
========================  =====  ======  ============  =====  =====  =============  ===========

The hits of persistence by lead run ``38, 38, 38, 37, 34, 29, 28, 23, 19, 18, 17, 17, 16, 15, 14, 14, 12, 11, 9, 7,
7, 6, 4, 3, 3, 3, 3, 3, 3, 1`` -- skill that decays with the horizon rather than stopping, which is why
``lead_time_ok`` (0) and the per-lead table say different things and both are reported. Scored as a moving feature,
persistence puts the maximum 0.954 in value and 1.045 grid cells away from the truth, against 0.000 and 0.000 for
the truth itself. Twenty perturbed members (``sigma = 0.25``) around persistence give a Brier score of 0.0145 with
reliability 0.0095, resolution 0.0232 and uncertainty 0.0283, on a base rate of 0.971.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "EVENT_KINDS",
    "Detection",
    "EnsembleEventScore",
    "Event",
    "EventScore",
    "LeadScore",
    "ReliabilityBin",
    "Tolerances",
    "detect",
    "ensemble_event_score",
    "event_score",
]

EVENT_KINDS = ("threshold", "extremum", "onset", "track")


@dataclass(frozen=True)
class Tolerances:
    """When a forecast event counts as the observed one.

    value : allowed ``|forecast value - truth value|``.
    timing : allowed ``|forecast lead - truth lead|``, in leads.
    position : allowed distance between the forecast and truth event locations, in the units of ``coords``.
    """
    value: float = 0.0
    timing: int = 0
    position: float = 0.0


@dataclass(frozen=True)
class Event:
    """One event, described so that the same scorer can judge any field.

    kind : one of :data:`EVENT_KINDS`.
    threshold : the level. For ``threshold`` it is the value crossed, for ``onset`` the change against the reference
        frame, for ``extremum``/``track`` ignored (the peak always occurs).
    above : ``True`` when the event is the value going up (a maximum crossed, a rise, a hot spot), ``False`` when it
        is going down (a minimum crossed, a drop, a cold spot).
    direction : which point of the region defines the value -- ``"max"`` or ``"min"``. Defaults to ``"max"`` when
        ``above`` and ``"min"`` otherwise.
    region : a boolean mask over the spatial axes (the same shape as one frame) restricting where the event can be;
        ``None`` means the whole field.
    window : ``(first_lead, last_lead)`` inclusive, restricting which leads are scored; ``None`` means all of them.
    reference : the lead used as the baseline frame for ``onset`` (default ``0``, the frame the forecast started
        from).
    tolerance : a :class:`Tolerances`.
    """
    kind: str = "threshold"
    threshold: float = 0.0
    above: bool = True
    direction: str | None = None
    region: Any = None
    window: tuple[int, int] | None = None
    reference: int = 0
    tolerance: Tolerances = field(default_factory=Tolerances)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}")
        d = self.direction if self.direction is not None else ("max" if self.above else "min")
        if d not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {d!r}")
        object.__setattr__(self, "direction", d)
        if self.window is not None:
            lo, hi = int(self.window[0]), int(self.window[1])
            if hi < lo:
                raise ValueError(f"window {self.window!r} ends before it starts")
            object.__setattr__(self, "window", (lo, hi))

    @property
    def confusion(self) -> bool:
        """Whether the kind is a yes/no event (a confusion matrix) or a peak comparison."""
        return self.kind in ("threshold", "onset")


@dataclass(frozen=True)
class Detection:
    """What :func:`detect` found in one frame.

    occurs : whether the event is in the frame (always ``True`` for ``extremum`` and ``track``).
    value : the event's value -- the level reached, the change against the reference frame, or the peak.
    position : the location of that value as a tuple, in ``coords`` units (grid indices without ``coords``).
    """
    occurs: bool
    value: float
    position: tuple[float, ...] | None = None


# ------------------------------------------------------------------ detection

def _mask(region: Any, shape: tuple[int, ...]) -> np.ndarray:
    if region is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(region, dtype=bool)
    if m.shape != shape:
        try:
            m = np.broadcast_to(m, shape).copy()
        except ValueError as exc:  # pragma: no cover - the message is the point
            raise ValueError(f"region of shape {m.shape} does not fit a frame of shape {shape}") from exc
    if not m.any():
        raise ValueError("the region keeps no point of the frame")
    return m


def _location(flat: int, shape: tuple[int, ...], coords) -> tuple[float, ...]:
    idx = np.unravel_index(flat, shape) if shape else ()
    if coords is None:
        return tuple(float(i) for i in idx)
    c = np.asarray(coords, dtype=float)[idx]
    return tuple(float(x) for x in np.atleast_1d(c))


def detect(frame, event: Event, reference=None, coords=None) -> Detection:
    """Detect ``event`` in one frame, with the location of its value.

    Parameters
    ----------
    frame : the field at one valid time; its shape is the spatial shape (a scalar frame has shape ``()``).
    event : the :class:`Event`.
    reference : the baseline frame, required for ``kind="onset"`` (the change is measured against it) and ignored
        otherwise.
    coords : optional positions of the grid points, shape ``frame.shape + (d,)``; without it the location is
        reported in grid indices.
    """
    a = np.asarray(frame, dtype=float)
    shape = a.shape
    m = _mask(event.region, shape)
    sel = a[m]
    if not np.all(np.isfinite(sel)):
        raise ValueError("the frame has non-finite values inside the region; mask them out with `region`")
    take_max = event.direction == "max"
    j = int(np.argmax(sel) if take_max else np.argmin(sel))
    value = float(sel[j])
    position = _location(int(np.flatnonzero(m.reshape(-1))[j]), shape, coords)

    if event.kind == "onset":
        if reference is None:
            raise ValueError("kind='onset' needs the reference frame: pass `reference=`")
        r = np.asarray(reference, dtype=float)
        if r.shape != shape:
            raise ValueError(f"the reference frame has shape {r.shape}, the frame {shape}")
        change = a - r
        csel = change[m]
        if not np.all(np.isfinite(csel)):
            raise ValueError("the change against the reference has non-finite values inside the region")
        k = int(np.argmax(csel) if take_max else np.argmin(csel))
        change_value = float(csel[k])
        return Detection(occurs=(change_value >= event.threshold if event.above
                                 else change_value <= event.threshold),
                         value=change_value,
                         position=_location(int(np.flatnonzero(m.reshape(-1))[k]), shape, coords))

    if event.kind == "threshold":
        occurs = value >= event.threshold if event.above else value <= event.threshold
        return Detection(occurs=bool(occurs), value=value, position=position)

    # extremum / track: the peak is in every frame
    return Detection(occurs=True, value=value, position=position)


# ------------------------------------------------------------------ scoring

@dataclass(frozen=True)
class LeadScore:
    """The cells at one lead: how many hits, misses and false alarms."""
    lead: int
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def ok(self) -> bool:
        """Nothing wrong at this lead -- no miss and no false alarm. ``lead_time_ok`` walks over this."""
        return self.misses == 0 and self.false_alarms == 0


@dataclass(frozen=True)
class EventScore:
    """The score of one deterministic forecast system on one event (see the module docstring for each field)."""
    event: Event
    n_cells: int
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    lead_time_ok: int
    peak_error: float
    relative_peak_error: float
    position_error: float
    detection_lead: float
    timing_error: float
    n_scored: int
    per_lead: tuple[LeadScore, ...]

    @property
    def csi(self) -> float:
        """Critical success index (threat score); ``nan`` when there is nothing to score."""
        den = self.hits + self.misses + self.false_alarms
        return float(self.hits / den) if den else float("nan")

    @property
    def probability_of_detection(self) -> float:
        den = self.hits + self.misses
        return float(self.hits / den) if den else float("nan")

    @property
    def false_alarm_ratio(self) -> float:
        den = self.hits + self.false_alarms
        return float(self.false_alarms / den) if den else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.event.kind,
            "n_cells": self.n_cells,
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
            "correct_negatives": self.correct_negatives,
            "csi": self.csi,
            "probability_of_detection": self.probability_of_detection,
            "false_alarm_ratio": self.false_alarm_ratio,
            "lead_time_ok": self.lead_time_ok,
            "peak_error": self.peak_error,
            "relative_peak_error": self.relative_peak_error,
            "position_error": self.position_error,
            "detection_lead": self.detection_lead,
            "timing_error": self.timing_error,
            "per_lead": [{"lead": r.lead, "hits": r.hits, "misses": r.misses, "false_alarms": r.false_alarms,
                          "correct_negatives": r.correct_negatives} for r in self.per_lead],
        }


def _leads(event: Event, n_lead: int) -> range:
    lo, hi = (0, n_lead - 1) if event.window is None else event.window
    lo, hi = max(0, lo), min(n_lead - 1, hi)
    return range(lo, hi + 1)


def _distance(p: tuple[float, ...] | None, q: tuple[float, ...] | None) -> float:
    if p is None or q is None or len(p) == 0:
        return float("nan")
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


def _matches(f: Detection, t: Detection | None, tol: Tolerances) -> bool:
    if t is None or not (f.occurs and t.occurs):
        return False
    if abs(f.value - t.value) > tol.value:
        return False
    d = _distance(f.position, t.position)
    return not (np.isfinite(d) and d > tol.position)


def event_score(forecasts, truth, event: Event, inits: Sequence[int] | None = None, coords=None) -> EventScore:
    """Score a deterministic forecast set against the truth on one event.

    Parameters
    ----------
    forecasts : array of shape ``(n_init, n_lead, *space)`` -- the forecast issued at each init, for every lead.
    truth : array of shape ``(n_truth, *space)`` -- the observed frames, indexed by truth step.
    event : the :class:`Event`.
    inits : the truth step each init starts from, length ``n_init`` (default ``0..n_init-1``).
    coords : optional grid positions, shape ``space + (d,)``, to report ``position_error`` in physical units.
    """
    F = np.asarray(forecasts, dtype=float)
    T = np.asarray(truth, dtype=float)
    if F.ndim < 2:
        raise ValueError("forecasts must be (n_init, n_lead, *space)")
    if T.ndim != F.ndim - 1:
        raise ValueError(f"truth of shape {T.shape} does not match frames of shape {F.shape[2:]}")
    n_init, n_lead = F.shape[0], F.shape[1]
    start = list(range(n_init)) if inits is None else [int(i) for i in inits]
    if len(start) != n_init:
        raise ValueError(f"{len(start)} inits given for {n_init} forecast sets")
    leads = _leads(event, n_lead)
    if len(leads) == 0:
        raise ValueError("the window keeps no lead of the forecast")

    per_lead = {i: LeadScore(i, 0, 0, 0, 0) for i in leads}
    peak_err: list[float] = []
    rel_err: list[float] = []
    pos_err: list[float] = []
    first_f: dict[int, int] = {}
    first_t: dict[int, int] = {}
    n_cells = 0

    for i in range(n_init):
        ref_f = ref_t = None
        if event.kind == "onset":
            ref_f = F[i, event.reference] if 0 <= event.reference < n_lead else None
            j = start[i] + event.reference
            ref_t = T[j] if 0 <= j < len(T) else None
            if ref_f is None or ref_t is None:
                raise ValueError(f"init {i} cannot reach the reference lead {event.reference}")
        detf: dict[int, Detection] = {}
        dett: dict[int, Detection] = {}
        for lead in leads:
            k = start[i] + lead
            if not 0 <= k < len(T):
                continue
            detf[lead] = detect(F[i, lead], event, reference=ref_f, coords=coords)
            dett[lead] = detect(T[k], event, reference=ref_t, coords=coords)
            n_cells += 1
        if not detf:
            continue
        for lead in leads:
            if lead not in detf:
                continue
            f, t = detf[lead], dett[lead]
            if event.confusion:
                # a forecast event is verified against the observed ones inside the timing tolerance, and an
                # observed event against the forecast ones; one outside every tolerance is a false alarm *and* the
                # observed event it should have been is a miss -- both errors are real
                near = [m for m in leads if m in dett and abs(m - lead) <= event.tolerance.timing]
                hit = f.occurs and any(_matches(f, dett[m], event.tolerance) for m in near)
                near_f = [m for m in leads if m in detf and abs(m - lead) <= event.tolerance.timing]
                caught = t.occurs and any(_matches(detf[m], t, event.tolerance) for m in near_f)
                r = per_lead[lead]
                per_lead[lead] = LeadScore(lead, r.hits + int(hit),
                                        r.misses + int(t.occurs and not caught),
                                        r.false_alarms + int(f.occurs and not hit),
                                        r.correct_negatives + int(not (f.occurs or t.occurs)))
            else:
                # the peak is in every frame, so there is nothing to hit or miss -- a lead is good when the peak is
                # within the value tolerance, and `lead_time_ok` reads the same way for all four kinds
                good = abs(f.value - t.value) <= event.tolerance.value
                r = per_lead[lead]
                per_lead[lead] = LeadScore(lead, r.hits + int(good), r.misses + int(not good),
                                        r.false_alarms, r.correct_negatives)
            if f.occurs and t.occurs:
                peak_err.append(abs(f.value - t.value))
                rel_err.append(abs(f.value - t.value) / max(abs(t.value), 1e-12))
                pos_err.append(_distance(f.position, t.position))
            if i not in first_f and f.occurs:
                first_f[i] = lead
            if i not in first_t and t.occurs:
                first_t[i] = lead

    hits = sum(r.hits for r in per_lead.values())
    misses = sum(r.misses for r in per_lead.values())
    fa = sum(r.false_alarms for r in per_lead.values())
    cn = sum(r.correct_negatives for r in per_lead.values())

    lead_ok = -1
    if hits > 0:
        for lead in leads:
            if per_lead[lead].ok:
                lead_ok = lead
            else:
                break

    timed = [(first_t[i] - first_f[i]) for i in first_f if i in first_t]
    det_lead = float(np.mean(timed)) if timed else float("nan")
    timing_err = float(np.mean(np.abs(timed))) if timed else float("nan")
    finite_pos = [d for d in pos_err if np.isfinite(d)]
    return EventScore(
        event=event,
        n_cells=n_cells,
        hits=hits,
        misses=misses,
        false_alarms=fa,
        correct_negatives=cn,
        lead_time_ok=lead_ok,
        peak_error=float(np.mean(peak_err)) if peak_err else float("nan"),
        relative_peak_error=float(np.mean(rel_err)) if rel_err else float("nan"),
        position_error=float(np.mean(finite_pos)) if finite_pos else float("nan"),
        detection_lead=det_lead,
        timing_error=timing_err,
        n_scored=len(peak_err),
        per_lead=tuple(per_lead[lead] for lead in leads),
    )


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of the reliability curve: the forecasts that said ``p`` close together, and what happened."""
    lower: float
    upper: float
    n: int
    mean_probability: float
    observed_frequency: float


@dataclass(frozen=True)
class EnsembleEventScore:
    """Brier score of an ensemble's event probability, with its reliability-resolution decomposition.

    ``brier = reliability - resolution + uncertainty`` (Murphy 1973), so a score can be read instead of only
    compared. ``reliability`` is how close the announced probabilities were to the observed frequencies (lower is
    better), ``resolution`` how much the forecast separated the events that happened from those that did not
    (higher is better) and ``uncertainty`` is the score of always announcing the base rate. The identity is exact
    when every distinct probability has its own bin (``n_bins`` large enough) and approximate otherwise: the spread
    of the probabilities *inside* a bin is what the bins cannot see.
    """
    n: int
    base_rate: float
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "base_rate": self.base_rate, "brier": self.brier, "reliability": self.reliability,
            "resolution": self.resolution, "uncertainty": self.uncertainty,
            "bins": [{"lower": b.lower, "upper": b.upper, "n": b.n, "mean_probability": b.mean_probability,
                      "observed_frequency": b.observed_frequency} for b in self.bins],
        }


def ensemble_event_score(members, truth, event: Event, inits: Sequence[int] | None = None, coords=None,
                         n_bins: int = 10) -> EnsembleEventScore:
    """Score the probability an ensemble gave to a yes/no event (``threshold`` or ``onset``).

    ``members`` has shape ``(n_init, n_member, n_lead, *space)``. The announced probability of a cell is the
    fraction of members in which the event is detected; the outcome is whether it is in the truth at that valid
    time.
    """
    if not event.confusion:
        raise ValueError(f"kind={event.kind!r} has no yes/no event to score; use event_score")
    E = np.asarray(members, dtype=float)
    T = np.asarray(truth, dtype=float)
    if E.ndim < 3:
        raise ValueError("members must be (n_init, n_member, n_lead, *space)")
    if T.ndim != E.ndim - 2:
        raise ValueError(f"truth of shape {T.shape} does not match frames of shape {E.shape[3:]}")
    n_init, n_mem, n_lead = E.shape[0], E.shape[1], E.shape[2]
    start = list(range(n_init)) if inits is None else [int(i) for i in inits]
    leads = _leads(event, n_lead)

    p_list: list[float] = []
    o_list: list[float] = []
    for i in range(n_init):
        ref_e = ref_t = None
        if event.kind == "onset":
            j_ref = start[i] + event.reference
            if not 0 <= j_ref < len(T) or not 0 <= event.reference < n_lead:
                raise ValueError(f"init {i} cannot reach the reference lead {event.reference}")
            ref_e = E[i, :, event.reference]
            ref_t = T[j_ref]
        for lead in leads:
            k = start[i] + lead
            if not 0 <= k < len(T):
                continue
            members_det = [detect(E[i, mm, lead], event, reference=None if ref_e is None else ref_e[mm], coords=coords)
                           for mm in range(n_mem)]
            p = float(np.mean([d.occurs for d in members_det]))
            o = float(detect(T[k], event, reference=ref_t, coords=coords).occurs)
            p_list.append(p)
            o_list.append(o)

    p_arr = np.asarray(p_list, dtype=float)
    o_arr = np.asarray(o_list, dtype=float)
    n = len(p_arr)
    if n == 0:
        raise ValueError("no cell could be scored: the window keeps none of the forecast")
    base = float(o_arr.mean())
    brier = float(np.mean((p_arr - o_arr) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, n_bins - 1)
    rel = res = 0.0
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        sel = which == b
        m = int(sel.sum())
        if not m:
            continue
        pb, ob = float(p_arr[sel].mean()), float(o_arr[sel].mean())
        rel += m / n * (pb - ob) ** 2
        res += m / n * (ob - base) ** 2
        bins.append(ReliabilityBin(float(edges[b]), float(edges[b + 1]), m, pb, ob))
    return EnsembleEventScore(n=n, base_rate=base, brier=brier, reliability=float(rel), resolution=float(res),
                              uncertainty=base * (1.0 - base), bins=tuple(bins))
