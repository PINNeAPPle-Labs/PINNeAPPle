"""pinneapple_analysis.trust.adapters — read the trust the library already computes into one :class:`Trust` (#345).

Nothing here recomputes a residual, an interval or a horizon: each adapter calls the module that already owns the
measurement and copies its number, with the source it came from.

===============  =============================================================================
adapter          source it reads
===============  =============================================================================
``from_ensemble``     :class:`pinneapple_physics.ensemble.PhysicsEnsemble` — ``predict`` (residuals, spread,
                      interval, suspect reading), ``update`` (the miss fraction of its intervals) and
                      :func:`pinneapple_physics.ensemble.residual_baseline` (the in-domain residual level)
``weather_horizon``   :func:`pinneapple_physics.weather.evaluate.horizons` (``useful_hours``, ``no_skill_hours``)
``rollout_horizon``   :func:`pinneapple_analysis.validation.rollout.divergence_time` (``trust_horizon``)
``calibration_uncertainty``  :class:`pinneapple_analysis.uncertainty.calibration.CalibrationMetrics`
                      (``coverage_at_level``, ``expected_calibration_error``)
``verification_measurement``  :class:`pinneapple_decision.verifiers.ThresholdVerifier` / ``VeriPhysicsVerifier``
                      (their ``Verification``: named checks, pass or fail)
===============  =============================================================================

:func:`trust` is the one call a user needs for the kind covered today (an ensemble or a surrogate that behaves like
one); the docstring says what it does not cover yet.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pinneapple_core.errors import NotSupported

from .record import (
    Horizon,
    Measurement,
    Provenance,
    Residual,
    Trust,
    TrustValue,
    Uncertainty,
    fingerprint_array,
    fingerprint_text,
    library_version,
    utc_now,
)

__all__ = ["from_ensemble", "weather_horizon", "rollout_horizon", "calibration_uncertainty",
           "verification_measurement", "domain_from_gate", "fingerprint_query", "trust"]

_ENSEMBLE_SOURCE = "pinneapple_physics.ensemble.PhysicsEnsemble"


def fingerprint_query(query: Any) -> tuple[str, str]:
    """A fingerprint of the case a prediction answers: the array itself when it is one, the parameters as sorted
    JSON when the query is a mapping of numbers, and nothing when neither is possible (rather than a hash of
    ``repr``, which is not a fingerprint of the data)."""
    digest, method = fingerprint_array(query)
    if digest:
        return digest, method
    if isinstance(query, Mapping):
        items = dict(query)
        try:
            text = __import__("json").dumps(items, sort_keys=True, default=float)
        except (TypeError, ValueError):
            return "", ""
        return fingerprint_text(text)[0], "sha256 of the query parameters as sorted JSON"
    return "", ""


# ------------------------------------------------------------------------------------------ ensemble
def _ensemble_residuals(ensemble: Any, out: Mapping[str, Any]) -> tuple[Residual, ...]:
    """The residual of each expert on this case, with the expert's own in-domain level as reference.

    ``PhysicsEnsemble`` evaluates them only when it was built with ``residual_lookahead`` (and not on a suspect
    reading), so a run without them yields no residual at all — the reason, not a zero.
    """
    raw = out.get("residuals")
    if raw is None:
        why = ("the reading was flagged suspect, so no residual was evaluated on it"
               if out.get("suspect") else
               "this ensemble was built without residual_lookahead, so no residual was evaluated for this case")
        return (Residual("pde", TrustValue.not_measured(why, source=_ENSEMBLE_SOURCE + ".predict"),
                         scope="whole field"),)
    baselines = dict(getattr(ensemble, "residual_baseline", {}) or {})
    out_res: list[Residual] = []
    for name, value in raw.items():
        v = float(value)
        if not np.isfinite(v):
            out_res.append(Residual(
                "pde", TrustValue.not_measured(f"expert {name!r} failed on this query, so it has no residual",
                                               source=_ENSEMBLE_SOURCE + ".predict"),
                scope="whole field", expert=str(name)))
            continue
        reference, reference_source = None, ""
        if name in baselines:
            reference = float(baselines[name])
            reference_source = ("pinneapple_physics.ensemble.residual_baseline: the median residual of this expert "
                                "on queries from its own domain")
        out_res.append(Residual(
            "pde",
            TrustValue.measured(v, unit="[-]", source=_ENSEMBLE_SOURCE + ".predict (residual_fn on this query)",
                                method="the physics residual the ensemble was configured with, for this expert on "
                                       "this case (relative form chosen by the residual_fn it was given)",
                                reference=reference, reference_source=reference_source),
            scope="whole field", expert=str(name)))
    return tuple(out_res)


def _ensemble_measurement(ensemble: Any, out: Mapping[str, Any]) -> Measurement:
    check = getattr(ensemble, "measurement_check", None)
    if check is None:
        return Measurement.not_measured(
            "this ensemble was built without measurement_check, so the readings it used were never checked",
            source=_ENSEMBLE_SOURCE)
    suspect = float(bool(out.get("suspect")))
    return Measurement(
        TrustValue.measured(suspect, unit="[-]", source=_ENSEMBLE_SOURCE + ".predict (measurement_check on this "
                                                                          "query)",
                            method="1 = the reading failed the model's own measurement check, 0 = it passed",
                            reference=0.0, reference_source="a sound reading is the one that did not fail the check"),
        source=_ENSEMBLE_SOURCE)


def _ensemble_uncertainty(ensemble: Any, out: Mapping[str, Any], info: Mapping[str, Any] | None) -> Uncertainty:
    lower, upper = np.asarray(out["lower"], float), np.asarray(out["upper"], float)
    half = float(np.mean((upper - lower) / 2.0))
    level = getattr(ensemble, "level", None)
    if np.isfinite(half):
        width = TrustValue.measured(
            half, source=_ENSEMBLE_SOURCE + ".predict (lower, upper)",
            method="mean half-width of the interval over the points of this case, in the units of the field")
    else:
        width = TrustValue.not_measured(
            "the interval the ensemble returned for this case is not finite: its conformal factor needs at least "
            "three scored cases before it exists, so there is no width to report (and none is written as 0)",
            source=_ENSEMBLE_SOURCE + ".predict")
    miss = (info or {}).get("miss_fraction")
    if miss is not None and np.isfinite(miss):
        coverage = TrustValue.measured(
            float(1.0 - miss), unit="[-]",
            source=_ENSEMBLE_SOURCE + ".update (miss_fraction)",
            method="fraction of the points that fell inside the interval on the case just scored",
            reference=float(level) if level else None,
            reference_source="the nominal level the ensemble states for its interval" if level else "")
    else:
        coverage = TrustValue.not_measured(
            "the ensemble adapts its interval online but reports coverage only for a case it has just scored with "
            "a reference; pass reference= to score this case, or measure it on held-out cases with "
            "CalibrationMetrics.coverage_at_level",
            source=_ENSEMBLE_SOURCE)
    return Uncertainty(level=None if not level else float(level), coverage=coverage, width=width,
                       ece=TrustValue.not_measured(
                           "no expected calibration error was computed for this case; it needs a held-out set of "
                           "predictions with their truth (CalibrationMetrics.expected_calibration_error)",
                           source="pinneapple_analysis.uncertainty.calibration.CalibrationMetrics"),
                       method="conformal scores kept online by the ensemble, widening the ensemble spread",
                       source=_ENSEMBLE_SOURCE + ".predict")


def domain_from_gate(gate: Any, query: Any, *, threshold: float = 1.0) -> tuple[TrustValue, TrustValue]:
    """(``in_domain``, ``domain_score``) from a :class:`pinneapple_analysis.trust.trust_gate.TrustGate`.

    ``domain_score`` is the fraction of query points whose chi-squared p-value against the training distribution
    reaches the gate's own ``ood_level`` (0.01: inside the 99 % region) — the same number the gate uses, not a
    second definition of it. ``in_domain`` is that fraction compared with ``threshold`` (1.0 by default: every
    point inside), and the threshold it was compared with is in the value's ``method``.
    """
    source = "pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values"
    try:
        p = np.asarray(gate.ood_p_values(query), float)
    except Exception as exc:                                   # the query is not points the gate can score
        why = (f"the case is not a set of coordinates the gate can score ({type(exc).__name__}: {exc}); fit it on "
               f"training points and pass them as the query to have an in-domain number")
        return (TrustValue.not_measured(why, source=source), TrustValue.not_measured(why, source=source))
    if p.size == 0 or not np.all(np.isfinite(p)):
        why = "the gate returned no finite p-value for this query"
        return (TrustValue.not_measured(why, source=source), TrustValue.not_measured(why, source=source))
    score = float(np.mean(p >= float(gate.ood_level)))
    domain_score = TrustValue.measured(
        score, unit="[-]", source=source,
        method=f"fraction of query points with chi2 p-value >= ood_level {gate.ood_level:g} under the fitted "
               f"training distribution")
    in_domain = TrustValue.measured(
        float(score >= threshold), unit="[-]", source=source,
        method=f"domain_score {score:.4g} >= {threshold:g}: every query point inside the "
               f"{100 * (1 - float(gate.ood_level)):.0f} % region of the training distribution")
    return in_domain, domain_score


def _provenance_ensemble(ensemble: Any, query: Any) -> Provenance:
    names = list(getattr(ensemble, "names", []) or [])
    kinds = [getattr(e, "kind", "") for e in getattr(ensemble, "experts", [])]
    text = "|".join(f"{n}:{k}:{getattr(e, 'cost', None)}"
                    for n, k, e in zip(names, kinds, getattr(ensemble, "experts", []), strict=False))
    model_fp, model_method = fingerprint_text(text)
    data_fp, data_method = fingerprint_query(query)
    return Provenance(
        kind="ensemble", model=f"PhysicsEnsemble({', '.join(names)})" if names else "PhysicsEnsemble",
        model_fingerprint=model_fp,
        model_fingerprint_method=(model_method + " of the expert names, kinds and declared costs") if model_method
        else "",
        data_fingerprint=data_fp, data_fingerprint_method=data_method,
        library_version=library_version(), created=utc_now(),
        inputs={k: v for k, v in (query.items() if isinstance(query, Mapping) else [])},
        notes=("the residual of an ensemble expert is a single scalar: this object says it was evaluated over the "
               "whole field and not where inside it the law is worst",))


def from_ensemble(ensemble: Any, query: Any = None, *, reference: Any = None, gate: Any = None,
                  domain_points: Any = None, problem: Any = None, horizon: Horizon | None = None,
                  requested_lead: float | None = None) -> Trust:
    """The Trust of one prediction of a :class:`pinneapple_physics.ensemble.PhysicsEnsemble`.

    Calls ``ensemble.predict(query)`` — a real prediction, which advances the ensemble's own state as any call
    does — and copies what the ensemble measured on that case. With ``reference`` it also calls
    ``ensemble.update(reference=...)``, the normal step of the ensemble's loop, which is the only place its interval
    coverage is measured (``miss_fraction``).

    What it cannot fill, and why it says so instead:

    * ``horizon`` — an ensemble does not measure a horizon. Build one with :func:`weather_horizon` or
      :func:`rollout_horizon` and pass it here, or leave it ``not_measured``;
    * ``residuals`` — only when the ensemble was built with ``residual_lookahead`` (and the reading is not suspect);
    * ``residual reference`` — only for the experts the ensemble declares in ``residual_baseline``; the others get
      their value with the note that it has no in-domain level to be judged against;
    * ``uncertainty.coverage`` and ``uncertainty.ece`` — only with a reference for this case (and ECE never here);
    * ``in_domain``/``domain_score`` — only with a ``gate`` (or ``problem.domain_bounds``, which the gate turns into
      a uniform box: coarser than a sample of the real training points and labelled as such). The gate scores
      coordinates, so when the query is a set of parameters rather than points, pass the coordinates of the case as
      ``domain_points``; without them the in-domain test is reported as not measured, never assumed.
    """
    out = ensemble.predict(query)
    info: Mapping[str, Any] | None = None
    if reference is not None:
        info = ensemble.update(query=query, reference=np.asarray(reference, float))
    if gate is None and problem is not None:
        bounds = dict(getattr(problem, "domain_bounds", {}) or {})
        if bounds and all(isinstance(v, Sequence) and len(v) == 2 for v in bounds.values()):
            from .trust_gate import TrustGate
            gate = TrustGate().fit_from_bounds({k: tuple(v) for k, v in bounds.items()})
    if gate is not None:
        in_domain, domain_score = domain_from_gate(gate, query if domain_points is None else domain_points)
    else:
        why = ("no TrustGate was given and the problem declares no domain_bounds, so the input was never tested "
               "against the training distribution")
        in_domain, domain_score = TrustValue.not_measured(why), TrustValue.not_measured(why)
    return Trust(
        kind="ensemble",
        in_domain=in_domain,
        domain_score=domain_score,
        horizon=(horizon if horizon is not None else Horizon.not_measured(
            "an ensemble does not measure a horizon: pass one from weather.evaluate.horizons "
            "(weather_horizon) or validation.rollout.divergence_time (rollout_horizon)")),
        residuals=_ensemble_residuals(ensemble, out),
        measurement=_ensemble_measurement(ensemble, out),
        uncertainty=_ensemble_uncertainty(ensemble, out, info),
        provenance=_provenance_ensemble(ensemble, query),
    )


# ------------------------------------------------------------------------------------------ horizons
def weather_horizon(horizons: Mapping[str, Any], *, requested_lead: float | None = None,
                    source: str = "pinneapple_physics.weather.evaluate.horizons") -> Horizon:
    """The horizon of a forecast from what :func:`pinneapple_physics.weather.evaluate.horizons` measured.

    ``horizons`` is its return value: ``useful_hours`` (the first lead at which the anomaly correlation falls below
    0.6) and ``no_skill_hours`` (the first lead at which the RMSE reaches climatology's). A lead that was never
    crossed inside the leads the forecast was scored on is ``inf`` there, and is recorded here as ``not_measured``
    with the range it stayed inside — the horizon is beyond the window, which is not the same number as "the last
    lead" and is not written as one.
    """
    unit = "h"

    def lead(key: str, what: str) -> TrustValue:
        if key not in horizons:
            return TrustValue.not_measured(f"the scored forecast reports no {key} (expected from {source})",
                                           source=source)
        v = float(horizons[key])
        if not np.isfinite(v):
            return TrustValue.not_measured(
                f"{what} was never crossed within the leads the forecast was scored on, so it is beyond them and "
                f"not measured here", source=source, method=f"first crossing of {key}")
        return TrustValue.measured(v, unit=unit, source=source,
                                   method=f"first lead at which {what} crosses its threshold, linearly "
                                          f"interpolated between leads")

    return Horizon(useful=lead("useful_hours", "the anomaly correlation 0.6"),
                   no_skill=lead("no_skill_hours", "the climatology RMSE"),
                   requested=(TrustValue.measured(float(requested_lead), unit=unit,
                                                  source="the lead this prediction is asked to cover",
                                                  method="given by the caller")
                              if requested_lead is not None else
                              TrustValue.not_measured("no lead was requested for this prediction",
                                                      source="pinneapple_analysis.trust.adapters.weather_horizon")),
                   unit=unit, source=source)


def rollout_horizon(divergence: Mapping[str, Any], *,
                    source: str = "pinneapple_analysis.validation.rollout.divergence_time") -> Horizon:
    """The horizon of an autoregressive rollout from
    :func:`pinneapple_analysis.validation.rollout.divergence_time`.

    That function reports ``trust_horizon`` (the earliest of the pointwise-error, spectral and invariant crossings)
    and ``horizon_end`` (the rollout length it could see). A rollout that never crosses has ``trust_horizon`` None:
    the horizon is then recorded as ``not_measured`` with the length of the rollout it was measured over.
    """
    unit = ""                                            # the caller's time unit: the divergence time uses dt
    end = divergence.get("horizon_end")
    th = divergence.get("trust_horizon")

    def value(v: Any, what: str, method: str) -> TrustValue:
        if v is None:
            return TrustValue.not_measured(
                f"{what} was not crossed within the rollout (length {end!r}); the horizon is beyond it and is not "
                f"measured here", source=source, method=method)
        return TrustValue.measured(float(v), source=source, method=method)

    return Horizon(
        useful=value(th, "no criterion crossing", "earliest of the error, spectral and invariant crossings"),
        no_skill=TrustValue.not_measured(
            f"divergence_time reports one trust horizon, not a no-skill lead of its own; the rollout length "
            f"({end!r}) is not the same number and was not replaced for it", source=source),
        requested=TrustValue.not_measured("no lead was requested for this prediction",
                                          source="pinneapple_analysis.trust.adapters.rollout_horizon"),
        unit=unit, source=source)


# ------------------------------------------------------------------------------------------ uncertainty
def calibration_uncertainty(y_pred: Any, y_true: Any, y_std: Any, *, level: float | None = None,
                            alpha: float | None = None,
                            source: str = "pinneapple_analysis.uncertainty.calibration.CalibrationMetrics") -> Uncertainty:
    """The interval of a model, with the coverage and the calibration error measured on held-out predictions.

    ``level`` is the nominal level the model claims (``alpha = 1 - level`` is accepted instead). Both numbers come
    from :class:`CalibrationMetrics`; the width is measured here from the interval itself (``2 * y_std`` mean) and
    says so.
    """
    from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics

    if level is None and alpha is not None:
        level = 1.0 - float(alpha)
    a = (1.0 - float(level)) if level is not None else 0.1
    coverage_value = CalibrationMetrics.coverage_at_level(y_pred, y_true, y_std, alpha=a)
    ece = CalibrationMetrics.expected_calibration_error(y_pred, y_true, y_std)
    width = float(np.mean(2.0 * np.asarray(y_std, dtype=float)))
    return Uncertainty(
        level=None if level is None else float(level),
        coverage=TrustValue.measured(
            float(coverage_value), unit="[-]", source=source + ".coverage_at_level",
            method=f"fraction of held-out points inside the {100 * (1 - a):.0f} % interval",
            reference=None if level is None else float(level),
            reference_source="the nominal level the model claims" if level is not None else ""),
        width=TrustValue.measured(width, source=source + ".coverage_at_level (y_std)",
                                  method="mean width 2 * y_std of the interval, in the units of the field"),
        ece=TrustValue.measured(float(ece), unit="[-]", source=source + ".expected_calibration_error",
                                method="area-weighted mean |empirical - nominal| coverage over 15 levels"),
        method="coverage and calibration error of the predictive distribution on held-out predictions",
        source=source)


# ------------------------------------------------------------------------------------------ verifier
def verification_measurement(verification: Any, *,
                             source: str = "pinneapple_decision.verifiers") -> Measurement:
    """The named checks a decision verifier ran, as measurements.

    ``Verification`` reports pass or fail per check and nothing else — no value behind it — so each check lands as a
    1/0 measurement whose ``source`` names the verifier that produced it, and ``suspect`` stays ``not_measured``:
    a threshold verifier does not look at sensor readings. A check that did not run is absent from ``checks``,
    exactly as it is absent from the ``Verification``, and never a pass.
    """
    checks = dict(getattr(verification, "checks", {}) or {})
    who = getattr(verification, "source", "") or "verifier"
    out = {name: TrustValue.measured(float(bool(ok)), unit="[-]",
                                     source=f"{source} ({who}).verify",
                                     method="1 = the check passed, 0 = it failed; the verifier reports the "
                                            "outcome, not the value behind it",
                                     reference=1.0, reference_source="a passing check")
           for name, ok in checks.items()}
    return Measurement(
        suspect=TrustValue.not_measured(
            "a decision verifier does not check the readings a prediction was built from; its checks are named "
            "below and in Measurement.checks", source=source),
        checks=out, source=f"{source} ({who})")


# ------------------------------------------------------------------------------------------ the entry point
def trust(model: Any, data: Any = None, problem: Any = None, **options: Any) -> Trust:
    """``pp.trust(model, data, problem)`` — the trust of one prediction, as a :class:`Trust` (#345).

    Parameters
    ----------
    model
        The model that made the prediction. Covered today: a
        :class:`pinneapple_physics.ensemble.PhysicsEnsemble` — and, because it is duck-typed, any surrogate that
        behaves like one (``predict(query)`` returning the ensemble's own dictionary: ``prediction``, ``spread``,
        ``lower``/``upper``, and ``residuals`` when it evaluates them). That is the kind that already carries the
        residual lookahead and the measurement check, so it is the one whose trust can be read instead of invented.
    data
        The case the prediction answers: the query passed to ``model.predict``. A mapping with ``"query"`` (and
        optionally ``"reference"``) is unpacked.
    problem
        Optional :class:`pinneapple_physics.PhysicalProblem`. Used for its ``domain_bounds`` (turned into a
        :class:`TrustGate` box, which is coarser than a sample of the training points and says so) and for its name
        in the provenance.
    **options
        ``reference``, ``gate``, ``domain_points``, ``horizon``, ``requested_lead`` — see :func:`from_ensemble`.

    Returns
    -------
    Trust
        With every field either measured (number, unit, source) or ``not_measured`` (with the reason).

    What it does **not** cover yet, and refuses with :class:`pinneapple_core.errors.NotSupported` instead of
    returning a half-empty object as if it were a measurement:

    * a **PINN** or a **neural operator** — their trust needs the residual sampled on collocation points (not the
      training loss), the boundary-condition error on the geometry, and a horizon from a rollout; none of that is
      read yet, and no adapter guesses it;
    * a **bare surrogate** (a ``torch.nn.Module`` or a callable) — wrap it in a ``PhysicsEnsemble`` with the
      reference solver as the second expert and it is covered;
    * the **consumers** (``twin3d``, the weather ``earth2_gif``, certificates) — they do not read :class:`Trust`
      yet; that is the next slice of #345.

    Examples
    --------
    >>> import numpy as np
    >>> import pinneapple as pp
    >>> from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
    >>> ens = PhysicsEnsemble({"fine": lambda q: q["u"] + q["d"], "coarse": lambda q: q["u"]},
    ...                       residual_fn=lambda p, q: float(abs(float(np.mean(p)) - q["u"])))
    >>> t = pp.trust(ens, {"u": 1.0, "d": 0.1})            # doctest: +SKIP
    >>> t.badge() in ("good", "fading", "no_skill")        # doctest: +SKIP
    True
    """
    if isinstance(data, Mapping) and "query" in data:
        reference = options.pop("reference", None)
        if reference is None:
            reference = data.get("reference")
        data = data["query"]
    if getattr(model, "experts", None) is not None and callable(getattr(model, "predict", None)):
        return from_ensemble(model, data, problem=problem, **options)
    raise NotSupported(
        f"pp.trust does not read the trust of a {type(model).__name__} yet",
        fix="pass a pinneapple_physics.ensemble.PhysicsEnsemble (or a surrogate with the same predict() contract); "
            "the PINN and neural-operator kinds need the residual sampled on collocation points and a rollout "
            "horizon, which #345 still lists as open")
