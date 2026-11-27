"""The sea state with a horizon: a measured spectrum advanced an hour at a time.

Two mechanisms move a spectrum at a fixed point, and both are in here:

* **the wind sea.** The local wind keeps pumping energy into the bands it can hold. A band slower than
  ``ratio * U`` (``c_p/U`` of 1.14 is a fully developed Pierson-Moskowitz sea) is the wind sea, and it relaxes toward
  the energy a fully developed sea of the wind's speed carries, on a growth timescale of a few hours. A sea that is
  *younger* than the wind allows keeps growing with the wind held constant -- that is the part of the forecast that
  beats carrying the last observation forward;
* **the swell.** Everything above that ratio was made somewhere else and is arriving or leaving. It decays, and it
  decays *faster at higher frequency*, because whitecapping removes the short waves first: the measured signature is a
  falling ``Hs`` with a *rising* ``Tp`` as the long-period part survives, which is the one thing persistence cannot say.

The group velocity is the third piece, and it is not a decay: :func:`arrival_hours` turns a distance into the lead time
at which a band can have been touched by anything that happened there, and :func:`inspected_radius` is the distance the
peak of a measured spectrum travelled during the forecast. A 14 s swell covers 39 km an hour; a 5 s sea covers 8 km.

The two coefficients (:attr:`SeaStateModel.swell_decay_hours`, :attr:`SeaStateModel.windsea_growth_hours`) are the only
things fitted anywhere in this module, and :func:`calibrate` fits them on one slice of a record so that another slice
can be scored. Nothing else is tuned to the measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import spectrum as sp

__all__ = ["SeaStateModel", "Forecast", "forecast", "calibrate", "skill", "arrival_hours", "inspected_radius",
           "separation_frequency", "split_swell"]

G = sp.G

#: Below this wind a sea has nothing being fed into it and the wind-sea branch is not entered at all.
CALM_WIND = 1.0


def separation_frequency(wind_speed, ratio: float = 1.2) -> float:
    """``f_sep = g / (2 pi ratio U)``: the band at which ``c_p = ratio U``.

    A fully developed Pierson-Moskowitz sea has its peak at ``c_p/U = 1.14``, so ``ratio = 1.2`` puts the separation
    just above the peak of the sea the wind itself would make -- the wind sea is what the wind is still feeding, the
    swell is what is passing through.
    """
    wind = np.maximum(np.asarray(wind_speed, dtype=float), 1e-6)
    return G / (2.0 * math.pi * ratio * wind)


def arrival_hours(distance_km, period_s, depth: float | None = None) -> np.ndarray:
    """The group travel time over a distance: ``t = L / c_g(T)``, in hours."""
    speed = sp.group_velocity(period_s, depth)
    return np.asarray(distance_km, dtype=float) * 1000.0 / speed / 3600.0


def inspected_radius(hours, period_s, depth: float | None = None) -> np.ndarray:
    """How far the energy arriving now travelled during ``hours``: ``c_g(T) t``, in kilometres."""
    return sp.group_velocity(period_s, depth) * np.asarray(hours, dtype=float) * 3600.0 / 1000.0


def split_at(freq, density, f_sep: float) -> tuple[np.ndarray, np.ndarray]:
    """``(swell, wind_sea)``: the spectrum cut at a frequency, the wind sea above it."""
    freq = np.asarray(freq, dtype=float)
    density = np.asarray(density, dtype=float)
    wind = freq > f_sep
    return np.where(wind, 0.0, density), np.where(wind, density, 0.0)


def split_swell(freq, density, wind_speed, ratio: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """``(swell, wind_sea)``, cut at :func:`separation_frequency` for that wind."""
    return split_at(freq, density, float(separation_frequency(wind_speed, ratio)))


@dataclass(frozen=True)
class SeaStateModel:
    """The evolution of a spectrum at a fixed point, with two calibrated timescales."""

    swell_decay_hours: float = 36.0       # e-folding of swell energy at the separation frequency
    windsea_growth_hours: float = 6.0     # how fast a wind sea relaxes toward a fully developed one (see closure)
    ratio: float = 1.2                    # c_p/U above which a band is swell
    windsea_growth: bool = False          # whether a wind sea below a fully developed one is allowed to grow

    def __post_init__(self) -> None:
        if self.swell_decay_hours <= 0 or self.windsea_growth_hours <= 0:
            raise ValueError("the two timescales must be positive")
        if self.ratio <= 1.0:
            raise ValueError(f"ratio must be above 1 (a wind sea peaks at 1.14), got {self.ratio}")

    def targets(self, freq, wind_speed: float) -> tuple[float, np.ndarray, float]:
        """``(f_sep, target, energy)``: the separation band and the fully developed spectrum the wind sea heals toward.

        Computed once per analysis and then reused for every lead time -- it does not depend on the horizon, and
        rebuilding it inside the lead loop is most of the cost of a forecast.
        """
        freq = np.asarray(freq, dtype=float)
        wind_speed = float(wind_speed)
        if not np.isfinite(wind_speed) or wind_speed < 0:
            raise ValueError(f"the wind speed has to be a measured number, got {wind_speed}")
        if wind_speed < CALM_WIND:
            return float("inf"), np.zeros_like(freq), 0.0
        f_sep = float(separation_frequency(wind_speed, self.ratio))
        target = np.where(freq > f_sep, sp.fully_developed(wind_speed).density_hz(freq), 0.0)
        return f_sep, target, float(np.trapezoid(target, freq))

    def advance(self, freq, density, wind_speed: float, hours: float) -> np.ndarray:
        """The spectrum ``hours`` after the analysis.

        Swell: ``S(f) exp(-t / tau(f))`` with ``tau(f) = tau_swell * (f_sep / f)``, so the long-period end outlives the
        short-period end and the peak walks down in frequency.

        Wind sea: **carried**, not grown, because the wind is carried too. Holding a measured wind is not a forecast
        that it will keep blowing, so a wind sea that is below what that wind could make is left where it is, and only
        one above it decays toward what the wind still supports. ``windsea_growth=True`` turns on the other closure --
        relaxation toward the fully developed sea -- and it is there to be measured against: on a fetch-limited record
        it inflates the height, because the sea that a measured 10 m wind can make at a buoy 20 miles offshore is a
        fifth to a half of the fully developed one.
        """
        f_sep, target, e_target = self.targets(freq, wind_speed)
        return self._advance(np.asarray(freq, dtype=float), np.asarray(density, dtype=float), f_sep, target, e_target,
                             float(hours))

    def _advance(self, freq, density, f_sep: float, target, e_target: float, hours: float) -> np.ndarray:
        density = np.nan_to_num(density, nan=0.0)
        if hours <= 0:
            return density.copy()
        if not np.isfinite(f_sep):
            # no wind, no wind sea: everything present is swell and nothing is being fed into it. It still ages at the
            # rate the bands age -- the short ones first -- so the separation is taken at the peak of what is there
            swell, wind = density.copy(), np.zeros_like(density)
            f_sep = float(sp.peak_frequency(freq, density))
            if not np.isfinite(f_sep) or f_sep <= 0:
                return density * math.exp(-hours / self.swell_decay_hours)
        else:
            swell, wind = split_at(freq, density, f_sep)
        tau = self.swell_decay_hours * (f_sep / np.maximum(freq, 1e-6))
        swell = swell * np.exp(-hours / tau)
        if e_target > 0:
            e_now = float(np.trapezoid(wind, freq))
            e_next = e_target + (e_now - e_target) * math.exp(-hours / self.windsea_growth_hours)
            if not self.windsea_growth:
                # the wind is held where it was measured, and holding a wind is not a forecast that it will keep
                # blowing: a wind sea below what this wind could hold is *carried*, and only one above it can decay,
                # because that one no longer has the wind it needs
                e_next = min(e_now, e_next) if e_now <= e_target else e_next
            wind = target * (e_next / e_target) if e_next > 0 else wind
        return swell + wind

    def forecast(self, freq, density, wind_speed: float, *, leads=None) -> Forecast:
        """Advance a measured spectrum over a set of lead times and report Hs and Tp at each one."""
        leads = np.arange(1, 25, dtype=float) if leads is None else np.atleast_1d(np.asarray(leads, dtype=float))
        freq = np.asarray(freq, dtype=float)
        f_sep, target, e_target = self.targets(freq, wind_speed)
        density = np.asarray(density, dtype=float)
        spectra = np.array([self._advance(freq, density, f_sep, target, e_target, float(t)) for t in leads])
        hs = np.array([sp.hs_from_spectrum(freq, row) for row in spectra])
        tp = np.array([sp.peak_period(freq, row) for row in spectra])
        return Forecast(leads_hours=leads, hs=hs, tp=tp, spectra=spectra, freq=freq)


@dataclass(frozen=True)
class Forecast:
    """Hs and Tp at a set of lead times, with the spectra they were read off."""

    leads_hours: np.ndarray
    hs: np.ndarray
    tp: np.ndarray
    spectra: np.ndarray
    freq: np.ndarray
    extra: dict = field(default_factory=dict)

    def at(self, hours: float) -> dict[str, float]:
        """The forecast at one lead time, linearly interpolated between the leads it holds."""
        return {"hs": float(np.interp(hours, self.leads_hours, self.hs)),
                "tp": float(np.interp(hours, self.leads_hours, self.tp))}


def forecast(model: SeaStateModel, freq, density, wind_speed: float, *, leads=None) -> Forecast:
    """Convenience wrapper: :meth:`SeaStateModel.forecast`."""
    return model.forecast(freq, density, wind_speed, leads=leads)


def skill(predicted, observed, reference) -> dict[str, float]:
    """RMSE, bias, correlation and the skill score against a reference forecast, for one field.

    ``skill = 1 - MSE(model) / MSE(reference)``: positive means the model beat the reference, zero means it added
    nothing, negative means the reference was better. The reference is persistence unless somebody says otherwise.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ok = np.isfinite(predicted) & np.isfinite(observed) & np.isfinite(reference)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "rmse": float("nan"), "bias": float("nan"), "correlation": float("nan"),
                "skill": float("nan"), "rmse_reference": float("nan")}
    p, o, r = predicted[ok], observed[ok], reference[ok]
    err = p - o
    mse, mse_ref = float(np.mean(err ** 2)), float(np.mean((r - o) ** 2))
    corr = float(np.corrcoef(p, o)[0, 1]) if np.std(p) > 0 and np.std(o) > 0 else float("nan")
    return {"n": int(ok.sum()), "rmse": math.sqrt(mse), "bias": float(np.mean(err)), "correlation": corr,
            "skill": 1.0 - mse / mse_ref if mse_ref > 0 else float("nan"), "rmse_reference": math.sqrt(mse_ref)}


def calibrate(freq, density, wind_speed, observed_hs, observed_tp, leads, *,
              grid_swell=(12.0, 24.0, 36.0, 48.0, 72.0), grid_growth=(2.0, 4.0, 6.0, 10.0, 16.0),
              weights=(1.0, 0.05)) -> tuple[SeaStateModel, dict[str, float]]:
    """Pick the two timescales on one slice of a record, by minimising the Hs error (with Tp as a small tie-breaker).

    ``freq``, ``density``, ``wind_speed`` and the two observations are all ``(n_analysis, ...)`` sets of *analysis
    times* with the matching verification; ``leads`` are the lead times. The Hs error is in metres and the Tp error in
    seconds, so the Tp weight is small by construction rather than by accident.
    """
    leads = np.atleast_1d(np.asarray(leads, dtype=float))
    best: tuple[float, SeaStateModel] | None = None
    scores: dict[str, float] = {}
    for swell in grid_swell:
        for growth in grid_growth:
            model = SeaStateModel(swell_decay_hours=float(swell), windsea_growth_hours=float(growth))
            err_hs, err_tp = [], []
            for i in range(len(observed_hs)):
                out = model.forecast(freq[i], density[i], wind_speed[i], leads=leads)
                err_hs.append((out.hs - observed_hs[i]) ** 2)
                err_tp.append((out.tp - observed_tp[i]) ** 2)
            cost = float(weights[0] * np.mean(err_hs) + weights[1] * np.mean(err_tp))
            scores[f"swell={swell:g},growth={growth:g}"] = cost
            if best is None or cost < best[0]:
                best = (cost, model)
    if best is None:
        raise ValueError("the calibration grid is empty")
    return best[1], {"cost": best[0], **scores}


def evolution(model: SeaStateModel, freq, density, wind_speed: float, days: float = 3.0) -> dict[str, np.ndarray]:
    """The forecast carried out to ``days`` with a fine step -- what the figures plot, and where the decay shows."""
    step = 0.25
    leads = np.arange(step, days * 24.0 + step, step)
    out = model.forecast(freq, density, wind_speed, leads=leads)
    hs = np.array([sp.hs_from_spectrum(freq, row) for row in out.spectra])
    return {"hours": leads, "hs": hs, "tp": out.tp, "spectra": out.spectra,
            "peak_period": 1.0 / np.array([sp.peak_frequency(freq, row) for row in out.spectra])}


def replaced(model: SeaStateModel, **kwargs) -> SeaStateModel:
    """A copy of the model with one field changed -- the way a sensitivity sweep is written."""
    return replace(model, **kwargs)
