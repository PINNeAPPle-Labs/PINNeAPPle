"""pinneapple_analysis.trust.forecast_skill — the ensemble measured against each of its experts alone, and the
weather producer of the trust record (#348, #345).

Issue #348 asks two things of a demo: that the docs say how to add a pre-trained model (they do, see
``docs/guides/pretrained_experts.md``) and that **the demo reports the ensemble's skill against each model alone**.
That report did not exist: the guide told the reader to measure it. This module is the measurement, offline and
reproducible.

Why an analytic reference
-------------------------
A real WeatherBench2 subset and the pre-trained weights are not in the repository, so the truth here is the analytic
solution of the equation the weather domain checks are written for — zonal advection of a tracer on the ERA5 grid the
library uses (32 latitudes from 87.1875 to -87.1875 by 6.5625, 64 longitudes by 5.625 degrees):

    q(lat, lon, t) = q0(lat, lon - omega t),   u(lat) = omega a cos(lat),   v = 0

which advects the field exactly, at 7.5 degrees of longitude every 6 h and 38.5 m/s at the equator. The truth is then
known by construction, the error of each forecaster is its own, and the ranking is a ranking of the forecasts and not
of the reference. Every number in the report comes from a forecaster here reading a store built here: nothing is
transcribed from a published table.

The four models and the cheap baseline
--------------------------------------
==========================  ======  ==================================================================
``persistence``             cost 0  the initial state, held: the cheap baseline a forecast has to beat
``climatology``             cost 0  the zonal mean of the reference field, held: the second cheap baseline
``upwind-coarse``           cost 0.2  first-order upwind advection, one substep per 6 h: diffusive and cheap
``upwind-fine``             cost 1.0  the same scheme with eight substeps per 6 h: sharper and dearer
==========================  ======  ==================================================================

:func:`ensemble_skill` runs :class:`pinneapple_physics.ensemble.PhysicsEnsemble` over the same cases as the four
alone, with the residual lookahead of :func:`pinneapple_physics.domain_checks.advection_residual` as its label-free
score, and reports the relative L2 and the anomaly correlation by lead for each of the five, plus the cases in which
**one expert alone beats the ensemble** — which is the honest half of the answer and is reported, not smoothed over.

>>> from pinneapple_analysis.trust.forecast_skill import ensemble_skill          # doctest: +SKIP
>>> report = ensemble_skill()                                                   # doctest: +SKIP
>>> report["verdict"]                                                           # doctest: +SKIP
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pinneapple_physics.ensemble import PhysicsEnsemble, Expert, from_callable, relative_l2, residual_baseline
from pinneapple_physics.weather.evaluate import acc, horizons, rmse

__all__ = ["LAT", "LON", "STEPS", "DT_H", "OMEGA_DEG", "N_INIT", "reference_field", "truth", "initial_state",
           "experts", "climatology_forecast", "ensemble_skill", "weather_trust"]

#: The grid: the ERA5 32 x 64 lattice ``pinneapple_physics.weather`` is written against.
LAT = np.linspace(87.1875, -87.1875, 32)
LON = np.arange(64) * 5.625
#: Leads of 6 h, and how many of them a case asks for (5 days).
DT_H = 6.0
STEPS = 20
#: How far the field travels eastward per 6-h step, in degrees of longitude, and the matching equatorial speed.
OMEGA_DEG = 7.5
_A_EARTH = 6_371_000.0
#: Initialisations, one every 12 days of the reference field's own cycle: the cases every forecaster is scored on.
N_INIT = 12


# ------------------------------------------------------------------------------------------ the reference case
def _q0(P: np.ndarray, L: np.ndarray) -> np.ndarray:
    """``q0`` on arrays of latitude and longitude in degrees: a smooth tracer of unit amplitude."""
    return (0.6 * np.cos(np.radians(P)) * np.cos(np.radians(2.0 * L))
            + 0.4 * np.sin(np.radians(2.0 * P)) * np.sin(np.radians(3.0 * L)))


def initial_state(lat: np.ndarray | None = None, lon: np.ndarray | None = None) -> np.ndarray:
    """``q0`` on the grid, or on the axes given: the field every case starts from."""
    lat = LAT if lat is None else np.asarray(lat, float)
    lon = LON if lon is None else np.asarray(lon, float)
    P, L = np.meshgrid(lat, lon, indexing="ij")
    return _q0(P, L)


def reference_field() -> np.ndarray:
    """``q0`` on the grid: the field the climatology and the initial states are built from."""
    return initial_state()


def omega_deg(lat: np.ndarray | None = None) -> np.ndarray:
    """How far eastward the flow carries the tracer per 6-h step, in degrees of longitude: ``(lat,)``.

    It is not uniform: ``OMEGA_DEG * (1 + 0.5 sin lat)``, a jet that is fastest in the southern hemisphere. A uniform
    rotation would be trivial for any scheme -- every application is the same shift by the same fraction of a cell --
    and the deformation is what a forecast actually has to get right, so the tracer is carried by a shear flow and
    the exact solution below stretches with it.
    """
    lat = LAT if lat is None else np.asarray(lat, float)
    return OMEGA_DEG * (1.0 + 0.5 * np.sin(np.radians(lat)))


def truth(i0: int, steps: int = STEPS) -> np.ndarray:
    """The exact solution of the advection for ``steps`` 6-h steps from initialisation ``i0``.

    ``(steps + 1, lat, lon)``: the analytic field advected by the zonal wind :func:`zonal_wind`, which for a purely
    zonal flow is ``q0(lat, lon - omega(lat) t)`` exactly -- the shear stretches it and the scheme has to follow.
    """
    shift = (i0 * 13.0) % 360.0                              # each initialisation starts at another place in the cycle
    P, L = np.meshgrid(LAT, LON, indexing="ij")
    om = omega_deg()[:, None]
    return np.stack([_q0(P, L - shift - k * om) for k in range(steps + 1)])


def zonal_wind(lat: np.ndarray | None = None, lon: np.ndarray | None = None) -> np.ndarray:
    """The wind that carries :func:`truth` eastward: ``u = omega a cos(lat)`` in m/s, ``(lat, lon)``."""
    lat = LAT if lat is None else np.asarray(lat, float)
    lon = LON if lon is None else np.asarray(lon, float)
    omega = np.radians(omega_deg(lat)) / (DT_H * 3600.0)      # rad/s
    u = omega * _A_EARTH * np.cos(np.radians(lat))
    return np.repeat(u[:, None], len(lon), axis=1)


def climatology_forecast(steps: int = STEPS) -> np.ndarray:
    """The second cheap baseline: the zonal mean of :func:`reference_field`, held for every lead.

    A field advected around a latitude circle has, averaged over a full revolution, exactly its zonal mean; over the
    twelve initialisations of this testbed the average is that mean, so the baseline is the analytic one and not an
    estimate of it.
    """
    clim = np.mean(reference_field(), axis=1, keepdims=True) * np.ones((1, 1, len(LON)))
    return np.repeat(clim, steps + 1, axis=0)


# ------------------------------------------------------------------------------------------ the numerical experts
def _shift(f: np.ndarray, cells: np.ndarray, order: int) -> np.ndarray:
    """Shift a ``(lat, lon)`` field eastward by ``cells`` cells of longitude per latitude, periodically.

    ``order = 1`` interpolates the departure point linearly and ``order = 3`` with a Catmull-Rom cubic: the two
    schemes every semi-Lagrangian model is built on, cheap and dear, diffusive and sharp. The scheme is stable at any
    Courant number, which is why it is semi-Lagrangian and not upwind: at 7.5 degrees per step on a 5.625 degree grid
    the Courant number is 4/3, and a first-order upwind scheme is unusable there.
    """
    n = f.shape[1]
    idx = np.arange(n)[None, :] - np.asarray(cells, float).reshape(-1, 1)
    i0 = np.floor(idx).astype(int)
    frac = idx - i0

    def at(offset: int) -> np.ndarray:
        return np.take_along_axis(f, (i0 + offset) % n, axis=1)

    if order == 1:
        return at(0) * (1.0 - frac) + at(1) * frac
    p0, p1, p2, p3 = at(-1), at(0), at(1), at(2)
    return 0.5 * (2.0 * p1 + (-p0 + p2) * frac + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * frac ** 2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * frac ** 3)


def _advect(state: np.ndarray, steps: int, order: int) -> np.ndarray:
    """Advect ``state`` for ``steps`` 6-h steps with the semi-Lagrangian scheme of the given interpolation order."""
    cells = omega_deg() * len(LON) / 360.0                   # cells of longitude per 6-h step, per latitude
    out = np.empty((steps + 1, *state.shape))
    out[0] = state
    f = state.copy()
    for k in range(1, steps + 1):
        f = _shift(f, cells, order)
        out[k] = f
    return out


def _state_of(i0: int) -> np.ndarray:
    return truth(i0, 0)[0]


def _forecast(query: Mapping[str, Any], order: int) -> np.ndarray:
    return _advect(_state_of(int(query["i0"])), int(query["steps"]), order)


def experts() -> list[Expert]:
    """The forecasters that are ensemble members, at the costs the table declares.

    ``persistence`` is the cheap baseline every forecast has to beat, and the two semi-Lagrangian schemes are the
    models. Climatology is not an ensemble member: it is the reference the anomaly correlation is taken against (see
    :func:`climatology_forecast`), and averaging it into a forecast would be averaging in something no model would
    ever emit.
    """
    def persistence(q):
        return np.repeat(_state_of(int(q["i0"]))[None], int(q["steps"]) + 1, axis=0)

    return [
        from_callable("persistence", persistence, cost=0.0, kind="baseline"),
        from_callable("semi-lagrangian-linear", lambda q: _forecast(q, 1), cost=0.2, kind="numerical"),
        from_callable("semi-lagrangian-cubic", lambda q: _forecast(q, 3), cost=1.0, kind="numerical"),
    ]


def baselines() -> dict[str, Any]:
    """The cheap baselines a forecast is compared with, scored the same way and never members of the ensemble."""
    clim = climatology_forecast()

    def persistence(q):
        return np.repeat(_state_of(int(q["i0"]))[None], int(q["steps"]) + 1, axis=0)

    def climatology(q):
        return clim[: int(q["steps"]) + 1]

    return {"persistence": persistence, "climatology": climatology}


# ------------------------------------------------------------------------------------------ the measurement
def _queries(n: int = N_INIT, steps: int = STEPS) -> list[dict[str, Any]]:
    return [{"i0": i, "steps": steps} for i in range(n)]


def _advection_score(prediction: np.ndarray, query: Mapping[str, Any]) -> float:
    """The ensemble's label-free residual: the part of the change the wind does not explain
    (:func:`pinneapple_physics.domain_checks.advection_residual`)."""
    from pinneapple_physics.domain_checks import advection_residual

    p = np.asarray(prediction, float)
    if p.ndim != 3 or p.shape[0] < 2:
        return float("inf")
    u, _v = zonal_wind(), np.zeros_like(zonal_wind())
    return float(advection_residual(p, u, _v, DT_H, LAT, LON))


def _l2_by_lead(pred: Sequence[np.ndarray], ref: Sequence[np.ndarray]) -> np.ndarray:
    """Relative L2 of each forecast against its truth, by lead (mean over the cases)."""
    k = len(ref[0])
    out = np.zeros(k)
    for p, r in zip(pred, ref, strict=True):
        out += np.array([relative_l2(np.asarray(p[j], float), np.asarray(r[j], float)) for j in range(k)])
    return out / max(len(ref), 1)


def _acc_by_lead(pred: Sequence[np.ndarray], ref: Sequence[np.ndarray], steps: int = STEPS) -> np.ndarray:
    """Anomaly correlation against the climatology, by lead, area-weighted (the same ``acc`` the skill curves use)."""
    clim = climatology_forecast(steps)[0]
    w = np.cos(np.radians(LAT))
    w = w / w.mean()
    out = np.zeros(steps + 1)
    for p, r in zip(pred, ref, strict=True):
        out += np.asarray(acc(np.asarray(p, float), np.asarray(r, float), clim, w), float)
    return out / max(len(ref), 1)


def _rmse_by_lead(pred: Sequence[np.ndarray], ref: Sequence[np.ndarray], steps: int = STEPS) -> np.ndarray:
    w = np.cos(np.radians(LAT))
    w = w / w.mean()
    out = np.zeros(steps + 1)
    for p, r in zip(pred, ref, strict=True):
        out += np.asarray(rmse(np.asarray(p, float), np.asarray(r, float), w), float)
    return out / max(len(ref), 1)


def ensemble_skill(*, n_init: int = N_INIT, steps: int = STEPS) -> dict[str, Any]:
    """The ensemble against each of its experts alone, on the same cases, against the analytic truth.

    Returns the report the guide prints: the relative L2 and the anomaly correlation by lead for every forecaster and
    for the ensemble, the skill horizon of each (``weather.evaluate.horizons``), the cost per case, and the honest
    half -- for how many cases and leads a **single expert** was closer to the truth than the ensemble, and which one.
    """
    qs = _queries(n_init, steps)
    refs = [truth(q["i0"], steps) for q in qs]
    exps = experts()
    names = [e.name for e in exps]
    alone = {e.name: [e.predict(q) for q in qs] for e in exps}
    cheap = {name: [fn(q) for q in qs] for name, fn in baselines().items() if name not in alone}

    base = residual_baseline(exps, {e.name: qs for e in exps}, _advection_score)
    ens = PhysicsEnsemble(exps, mode="combine", residual_fn=_advection_score, residual_lookahead=1.0,
                          residual_baseline=base, cost_weight=0.0, min_weight=0.0, interval_level=0.9)
    run = ens.run(qs, refs, keep_predictions=True)
    ens_pred = [p["prediction"] for p in run.predictions]

    # the same ensemble with cost in its objective: the skill is the one above, the price is not
    cheap_ens = PhysicsEnsemble(experts(), mode="combine", residual_fn=_advection_score, residual_lookahead=1.0,
                                residual_baseline=base, cost_weight=0.1, min_weight=0.0)
    cheap_run = cheap_ens.run(qs, refs, keep_predictions=True)
    cheap_pred = [p["prediction"] for p in cheap_run.predictions]

    leads_h = np.arange(steps + 1) * DT_H
    records: dict[str, Any] = {}
    for name in names:
        records[name] = _record(alone[name], refs, leads_h, steps)
    for name, pred in cheap.items():
        records[name] = _record(pred, refs, leads_h, steps)
    records["ensemble"] = _record(ens_pred, refs, leads_h, steps)

    # the honest half: on which cases and leads is one expert alone better than the combination?
    better: dict[str, int] = {}
    beats_ensemble = 0
    for k, (p_ens, r) in enumerate(zip(ens_pred, refs, strict=True)):
        for j in range(steps + 1):
            e_ens = relative_l2(np.asarray(p_ens[j], float), np.asarray(r[j], float))
            for name in names:
                e = relative_l2(np.asarray(alone[name][k][j], float), np.asarray(r[j], float))
                if e < e_ens:
                    better[name] = better.get(name, 0) + 1
                    beats_ensemble += 1
                    break
    total = len(refs) * (steps + 1)
    best_overall = min(names, key=lambda n: records[n]["mean_l2"])
    verdict = ("the ensemble beats every expert alone" if records["ensemble"]["mean_l2"] < records[best_overall]["mean_l2"]
               else f"{best_overall} alone beats the ensemble")

    # the same question on the cases after the weights have learned: an online ensemble starts at equal weights, so
    # the overall mean charges it for a warm-up that a forecaster alone never pays
    half = len(refs) // 2
    late = {name: float(np.mean(_l2_by_lead(alone[name][half:], refs[half:]))) for name in names}
    late["ensemble"] = float(np.mean(_l2_by_lead(ens_pred[half:], refs[half:])))
    best_late = min(names, key=lambda n: late[n])
    late_verdict = ("the ensemble beats every expert alone" if late["ensemble"] < late[best_late]
                    else f"{best_late} alone beats the ensemble")
    return {
        "schema": "pinneapple.forecast_skill/1",
        "reference": {
            "kind": "analytic",
            "equation": "q(lat, lon, t) = q0(lat, lon - omega(lat) t) with u = omega a cos(lat), v = 0",
            "why": ("a WeatherBench2 subset and the pre-trained weights are not in the repository; the truth is the "
                    "analytic solution of the advection the weather domain checks are written for, on the ERA5 "
                    "32 x 64 grid, so the error of each forecaster is its own"),
            "grid": {"lat": len(LAT), "lon": len(LON)}, "steps": steps, "dt_h": DT_H,
            "omega_deg_per_step": {"min": float(omega_deg().min()), "max": float(omega_deg().max())},
            "u_equator_m_s": float(zonal_wind()[len(LAT) // 2, 0]),
            "cases": len(qs)},
        "records": records,
        "ensemble": {"weights_final": {k: float(v) for k, v in ens.weights().items()},
                     "residual_baseline": {k: float(v) for k, v in base.items()},
                     "mean_cost_per_case": float(run.summary()["mean_cost_per_case"]),
                     "licences": run.summary().get("licences") or {}},
        "honest": {"leads_where_an_expert_beats_the_ensemble": beats_ensemble, "leads_total": total,
                   "fraction": beats_ensemble / total, "by_expert": better, "verdict": verdict,
                   "second_half": {"mean_l2": late, "verdict": late_verdict,
                                   "note": (f"the cases from {half} on, after the fixed-share weights have learned: "
                                            f"the same comparison without the warm-up charge")}},
        "with_cost": {"skill_only": {"mean_l2": float(np.mean(_l2_by_lead(ens_pred, refs))),
                                     "mean_cost_per_case": float(run.summary()["mean_cost_per_case"]),
                                     "weights_final": {k: float(v) for k, v in ens.weights().items()}},
                      "cost_weight_0.1": {"mean_l2": float(np.mean(_l2_by_lead(cheap_pred, refs))),
                                          "mean_cost_per_case": float(cheap_run.summary()["mean_cost_per_case"]),
                                          "weights_final": {k: float(v) for k, v in cheap_ens.weights().items()}},
                      "note": ("the same ensemble with cost_weight = 0.1 next to the one that ignores cost: skill "
                               "alone is not the only objective a combination is asked to win, and the two numbers "
                               "are what the trade is made of")},
    }


def _climatology_rmse(refs: Sequence[np.ndarray], steps: int = STEPS) -> np.ndarray:
    """The climatology's own RMSE by lead: the denominator a forecast's RMSE has to reach to have no skill."""
    clim = climatology_forecast(steps)[: steps + 1]
    return _rmse_by_lead([clim] * len(refs), refs, steps)


def _record(pred: Sequence[np.ndarray], refs: Sequence[np.ndarray], leads_h: np.ndarray, steps: int) -> dict:
    """One forecaster's skill: the two curves, their horizons, and the value at a few leads."""
    l2 = _l2_by_lead(pred, refs)
    a = _acc_by_lead(pred, refs, steps)
    r = _rmse_by_lead(pred, refs, steps)
    h = horizons(leads_h, a, r, _climatology_rmse(refs, steps))
    return {"mean_l2": float(np.mean(l2)),
            "l2_by_lead": [float(x) for x in l2],
            "acc_by_lead": [float(x) for x in a],
            "useful_hours": _finite(h.get("useful_hours")),
            "no_skill_hours": _finite(h.get("no_skill_hours")),
            "l2_at": {f"{leads_h[j]:g}h": float(l2[j]) for j in (0, len(leads_h) // 2, len(leads_h) - 1)}}


def _finite(x: Any) -> float | None:
    """A horizon that was crossed is a number; one that was not is ``None`` (never ``inf`` written as one)."""
    v = float(x) if x is not None else float("nan")
    return v if np.isfinite(v) else None


# ------------------------------------------------------------------------------------------ the weather record
def weather_trust(*, kind: str = "weather", lead_h: float = 48.0) -> Any:
    """The :class:`pinneapple_analysis.trust.record.Trust` of one forecast, filled by the adapters that exist.

    It is the *same* record the CFD surrogate emits (:func:`pinneapple_analysis.trust.producers.
    cfd_surrogate_trust`) and the same one the ensemble emits (:func:`pinneapple_analysis.trust.producers.
    ensemble_trust`), with ``kind`` saying which of the three it is: the keys, the gates and the sources do not
    change, because the adapters do not.

    The case is the last initialisation of the testbed, scored with its truth; the horizon comes from
    :func:`pinneapple_physics.weather.evaluate.horizons` on the forecast that was scored, and the envelope is the lead
    the forecast was validated for.

    Two details keep the two comparisons honest. The in-domain residual level of each expert is measured on the
    **other** initialisations (the last one is held out), so the residual of this case is compared with a level it did
    not contribute to. And the horizon is measured over the whole window the testbed scores (five days), not over the
    requested lead: a forecast that is still useful at 48 h has no crossing inside 48 h, and reporting the horizon as
    "not measured" because the window ended before the crossing would be a fact about the window, not the forecast.
    """
    from .adapters import from_ensemble, weather_horizon
    from .trust_gate import TrustGate

    qs = _queries()
    refs = [truth(q["i0"], STEPS) for q in qs]
    exps = experts()
    held_out = qs[-1]
    base = residual_baseline(exps, {e.name: qs[:-1] for e in exps}, _advection_score)
    ens = PhysicsEnsemble(exps, mode="combine", residual_fn=_advection_score, residual_lookahead=1.0,
                          residual_baseline=base, min_weight=0.0)
    run = ens.run(qs, refs, keep_predictions=True)
    preds = [p["prediction"] for p in run.predictions]
    leads_h = np.arange(STEPS + 1) * DT_H
    a = _acc_by_lead(preds, refs, STEPS)
    r = _rmse_by_lead(preds, refs, STEPS)
    clim_rmse = _climatology_rmse(refs, STEPS)
    h = horizons(leads_h, a, r, clim_rmse)

    gate = TrustGate().fit_from_bounds({"lead_h": (0.0, float(STEPS * DT_H))})
    return from_ensemble(ens, held_out, reference=refs[-1], gate=gate, domain_points=np.array([[lead_h]]),
                         horizon=weather_horizon(h, requested_lead=lead_h), kind=kind)
