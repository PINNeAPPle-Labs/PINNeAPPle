"""Rollout validation of autoregressive surrogates (#701): invariant drift, energy spectrum over the horizon,
divergence time ("trust horizon") and long-run statistics without a reference trajectory.

An autoregressive model can look accurate for a few steps and then drift, lose energy, blur or blow up. Given a
rollout ``pred`` of shape ``(T, ...)`` (time first, then any leading axes, then ``ndim`` spatial axes) and,
optionally, the reference trajectory ``ref`` from the same initial state:

``invariant_series`` / ``invariant_drift``
    Conserved or slowly varying quantities (mass, energy, enstrophy, momentum, or any callable) along the rollout:
    their drift rate (least-squares slope relative to the initial value) and, with a reference, the largest relative
    deviation from the reference's own series (decaying turbulence loses energy; the model must lose it at the same
    rate).
``spectral_drift``
    The high-wavenumber energy ratio (``validation.spectral``) at every horizon, against the reference at that time
    or against the training statistics (a reference spectrum). Spectral blurring is a falling ratio.
``divergence_time``
    The first horizon where the pointwise error, the spectral bias or an invariant deviation crosses its tolerance;
    the trust horizon is the earliest of them.
``long_run_statistics``
    For chaotic systems, where pointwise error is meaningless after a few Lyapunov times: the histogram of the field
    (Wasserstein-1 distance), the temporal autocorrelation (decorrelation time) and the time-averaged spectrum,
    compared with a long reference run, relative to the distance between two halves of the reference itself (the
    sampling noise floor).
``validate_rollout``
    All of the above as ``rollout.*`` checks for ``pp.validate`` and the trust card.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spectral import energy_spectrum

__all__ = ["BUILTIN_INVARIANTS", "invariant_series", "invariant_drift", "spectral_drift", "divergence_time",
           "long_run_statistics", "RolloutReport", "validate_rollout", "rollout"]


def _space(x: np.ndarray, ndim: int) -> tuple[int, ...]:
    return tuple(range(x.ndim - ndim, x.ndim))


BUILTIN_INVARIANTS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "mass": lambda u, nd: u.mean(axis=_space(u, nd)),
    "energy": lambda u, nd: 0.5 * (u ** 2).mean(axis=_space(u, nd)),
}
#: the scale a deviation is measured against: a zero-mean field has zero mass, so mass is measured against mean |u|
_SCALES: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "mass": lambda u, nd: np.abs(u).mean(axis=_space(u, nd)),
}


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Running mean over ``w`` frames (trailing, so a horizon only uses the frames up to it)."""
    if w <= 1:
        return x
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0, axis=0), axis=0)
    out = np.empty_like(np.asarray(x, float))
    for t in range(len(x)):
        a = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[a]) / (t + 1 - a)
    return out


def rollout(step: Callable[[np.ndarray], np.ndarray], u0: np.ndarray, n_steps: int) -> np.ndarray:
    """``(n_steps + 1, ...)`` states of an autoregressive model ``step(u) -> u_next`` from ``u0``; stops early
    (and returns what it has) when the state stops being finite."""
    out = [np.asarray(u0, float)]
    for _ in range(n_steps):
        nxt = np.asarray(step(out[-1]), float)
        out.append(nxt)
        if not np.all(np.isfinite(nxt)):
            break
    return np.stack(out)


def invariant_series(traj: np.ndarray, ndim: int, invariants: Sequence[str] | Mapping[str, Callable] = ("mass",
                                                                                                         "energy")
                     ) -> dict[str, np.ndarray]:
    """``{name: (T,) series}`` (leading axes averaged)."""
    fns = {k: BUILTIN_INVARIANTS[k] for k in invariants} if not isinstance(invariants, Mapping) else dict(invariants)
    out = {}
    for k, f in fns.items():
        v = np.asarray(f(np.asarray(traj, float), ndim), float)
        out[k] = v.reshape(len(traj), -1).mean(axis=1)
    return out


def _scale(series: np.ndarray, ref: np.ndarray | None = None) -> float:
    base = ref if ref is not None else series
    return max(float(np.max(np.abs(base))), 1e-300)


def invariant_drift(pred: np.ndarray, ndim: int, ref: np.ndarray | None = None, *, dt: float = 1.0,
                    invariants: Sequence[str] | Mapping[str, Callable] = ("mass", "energy"), window: int = 1
                    ) -> dict[str, dict]:
    """Per invariant: ``rate`` (slope / scale, per unit time), ``max_drift`` (largest |I(t) - I(0)| / scale) and,
    with ``ref``, ``max_deviation`` (largest |I_pred(t) - I_ref(t)| / scale, both averaged over ``window`` frames).
    The scale is the invariant's largest magnitude (for ``mass``: the mean of ``|u|``)."""
    sp = invariant_series(pred, ndim, invariants)
    sr = invariant_series(ref[: len(pred)], ndim, invariants) if ref is not None else {}
    base = ref if ref is not None else pred
    out = {}
    t = dt * np.arange(len(pred))
    for k, s in sp.items():
        builtin = k in _SCALES and (not isinstance(invariants, Mapping) or invariants[k] is BUILTIN_INVARIANTS[k])
        sc = float(np.nanmax(np.abs(_SCALES[k](np.asarray(base, float), ndim)))) if builtin \
            else _scale(s, sr.get(k))                         # a user function named like a built-in: its own scale
        sc = max(sc, 1e-300)
        fin = np.isfinite(s)
        rate = float(np.polyfit(t[fin], s[fin], 1)[0] / sc) if fin.sum() > 1 else float("nan")
        d = {"series": s, "rate": rate, "max_drift": float(np.nanmax(np.abs(s - s[0])) / sc)}
        if k in sr:
            dev = np.abs(_smooth(s, window) - _smooth(sr[k][: len(s)], window)) / sc
            d.update(ref_series=sr[k], deviation=dev, max_deviation=float(np.nanmax(dev)))
        out[k] = d
    return out


def _band_energy(u: np.ndarray, ndim: int, periodic: bool, band: tuple[int, int]) -> float:
    _, E = energy_spectrum(u, ndim, periodic=periodic)
    return float(E[band[0]:band[1] + 1].sum())


def spectral_drift(pred: np.ndarray, ndim: int, ref: np.ndarray | None = None, *, ref_spectrum: np.ndarray | None
                   = None, periodic: bool = True, window: int = 1, floor: float = 1e-6) -> dict[str, np.ndarray]:
    """High-wavenumber energy bias ``E_pred / E_ref - 1`` at every horizon, the band being the top third of the
    shells the reference resolves (energy above ``floor`` times its peak: amplitudes below 0.1 % of the peak are
    round-off for a float32 model, not content); against ``ref[t]`` (same initial state) or a fixed reference spectrum (e.g. the
    mean spectrum of the training data). Energies are averaged over ``window`` frames first, so the fluctuations of a
    chaotic state do not count as bias. Returns ``hf_bias`` (``(T,)``) and the ``band``."""
    T = len(pred)
    R = ref_spectrum if ref is None else energy_spectrum(ref[0], ndim, periodic=periodic)[1]
    if R is None:
        raise ValueError("give ref or ref_spectrum")
    top = int(np.nonzero(R[1:] > floor * R[1:].max())[0].max()) + 1
    band = (max(1, int(round(2 * top / 3))), top)
    ep = np.full(T, np.nan)
    er = np.full(T, np.nan)
    for t in range(T):
        if not np.all(np.isfinite(pred[t])):
            break
        ep[t] = _band_energy(pred[t], ndim, periodic, band)
        er[t] = _band_energy(ref[t], ndim, periodic, band) if ref is not None and t < len(ref) else \
            float(R[band[0]:band[1] + 1].sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = _smooth(ep, window) / np.maximum(_smooth(er, window), 1e-300) - 1
    return {"hf_bias": bias, "band": band}


def _first(mask: np.ndarray) -> int | None:
    k = np.nonzero(mask)[0]
    return int(k[0]) if len(k) else None


def divergence_time(pred: np.ndarray, ref: np.ndarray, ndim: int, *, dt: float = 1.0, err_tol: float = 0.2,
                    spec_tol: float = 0.25, inv_tol: float = 0.05, periodic: bool = True,
                    invariants: Sequence[str] | Mapping[str, Callable] = ("mass", "energy"), window: int = 10
                    ) -> dict[str, Any]:
    """First horizon (in time units, ``dt`` per frame) where each criterion crosses its tolerance, and the trust
    horizon (the earliest). ``None`` = never crossed within the rollout. A non-finite state crosses everything.
    The spectral and invariant criteria use ``window``-frame running means (statistics, not instantaneous values)."""
    T = min(len(pred), len(ref))
    p, r = pred[:T], ref[:T]
    ax = _space(p, ndim)
    num = np.sqrt(np.nansum((p - r) ** 2, axis=ax).reshape(T, -1).sum(1))
    den = np.sqrt((r ** 2).sum(axis=ax).reshape(T, -1).sum(1))
    err = np.where(np.isfinite(num), num / np.maximum(den, 1e-300), np.inf)
    bad = ~np.all(np.isfinite(p.reshape(T, -1)), axis=1)
    err[bad] = np.inf
    spec = np.zeros(T) if ndim == 0 else \
        np.abs(spectral_drift(p, ndim, r, periodic=periodic, window=window)["hf_bias"])       # ndim 0: a state vector
    spec[bad | ~np.isfinite(spec)] = np.inf
    inv = invariant_drift(p, ndim, r, dt=dt, invariants=invariants, window=window)
    dev = np.max(np.stack([d["deviation"] for d in inv.values()]), axis=0)
    dev = np.where(np.isfinite(dev), dev, np.inf)
    steps = {"error": _first(err > err_tol), "spectrum": _first(spec > spec_tol), "invariants": _first(dev > inv_tol)}
    times = {k: (None if v is None else v * dt) for k, v in steps.items()}
    crossed = [v for v in times.values() if v is not None]
    return {"times": times, "trust_horizon": min(crossed) if crossed else None, "horizon_end": (T - 1) * dt,
            "error": err, "spectrum_bias": spec, "invariant_deviation": dev,
            "tolerances": {"error": err_tol, "spectrum": spec_tol, "invariants": inv_tol}}


def _wasserstein1(a: np.ndarray, b: np.ndarray, n: int = 512) -> float:
    q = np.linspace(0.0, 1.0, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _decorrelation_time(traj: np.ndarray, ndim: int, dt: float) -> float:
    """Time for the spatially averaged temporal autocorrelation to drop below 1/e."""
    x = traj.reshape(len(traj), -1)
    x = x - x.mean(0)
    c0 = float((x * x).sum())
    for lag in range(1, len(x) // 2):
        c = float((x[:-lag] * x[lag:]).sum()) / c0 * len(x) / (len(x) - lag)
        if c < np.exp(-1):
            return lag * dt
    return float("nan")


def long_run_statistics(pred: np.ndarray, ref: np.ndarray, ndim: int, *, dt: float = 1.0, burn_in: int = 0,
                        periodic: bool = True) -> dict[str, Any]:
    """Compare invariant measures of a long model run with a long reference run (no common initial state needed).

    ``w1`` (Wasserstein-1 of the value histograms), ``tau`` (decorrelation times) and ``spectrum`` (largest |log
    ratio| of the time-averaged energy in the low, mid and high thirds of the resolved shells), each also for the two
    halves of ``ref`` -- the distance sampling alone produces. A model is consistent when its distances are within a
    small multiple of that floor (default 2)."""
    p, r = np.asarray(pred[burn_in:], float), np.asarray(ref[burn_in:], float)
    h = len(r) // 2
    r1, r2 = r[:h], r[h:]

    def spec(x):
        return energy_spectrum(x, ndim, periodic=periodic)[1]

    Er, Ep, E1, E2 = spec(r), spec(p), spec(r1), spec(r2)
    top = int(np.nonzero(Er[1:] > 1e-8 * Er[1:].max())[0].max()) + 1
    edges = [1, max(2, round(top / 3)), max(3, round(2 * top / 3)), top + 1]       # low / mid / high thirds

    def logr(a, b):                     # largest |log ratio| of the band energies
        return float(max(abs(np.log(max(a[lo:hi].sum(), 1e-300) / max(b[lo:hi].sum(), 1e-300)))
                         for lo, hi in zip(edges[:-1], edges[1:], strict=True)))

    out = {
        "w1": _wasserstein1(p, r), "w1_floor": _wasserstein1(r1, r2),
        "tau": _decorrelation_time(p, ndim, dt), "tau_ref": _decorrelation_time(r, ndim, dt),
        "tau_floor": abs(_decorrelation_time(r1, ndim, dt) - _decorrelation_time(r2, ndim, dt)),
        "spectrum": logr(Ep, Er), "spectrum_floor": logr(E1, E2), "scale": float(np.std(r)),
    }
    return out


@dataclass
class RolloutReport:
    checks: list[dict[str, Any]]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c["status"] != "fail" for c in self.checks)

    def __getitem__(self, cid: str) -> dict[str, Any]:
        return next(c for c in self.checks if c["id"] == cid)


def validate_rollout(pred: np.ndarray, ndim: int, ref: np.ndarray | None = None, *, dt: float = 1.0,
                     conserved: Sequence[str] = (), invariants: Sequence[str] | Mapping[str, Callable] = ("mass",
                                                                                                           "energy"),
                     long_ref: np.ndarray | None = None, burn_in: int = 0, periodic: bool = True,
                     window: int = 10, chaotic: bool = False,
                     tolerances: Mapping[str, float] | None = None) -> RolloutReport:
    """``rollout.*`` checks of a model rollout (status pass / fail / not-applicable with evidence).

    ``ref``: the reference trajectory from the same initial state; ``conserved``: invariants that must stay constant
    (checked without a reference; built-in names or keys of ``invariants``, a mapping ``name -> f(traj, ndim)``);
    ``ndim = 0``: the state is a vector (particles, a ROM's coefficients), with no spatial spectrum; ``long_ref``: a long reference run for the statistics of a chaotic system (the
    model's run ``pred`` must be long too); ``chaotic``: after the pointwise error crosses its tolerance the two
    trajectories are decorrelated, so a later spectral or invariant crossing compares two different states of the
    attractor and is reported, not failed (``rollout.statistics`` judges that regime). Tolerances: ``error``, ``spectrum``, ``invariants`` (divergence time),
    ``drift`` (conserved invariants, relative), ``floor`` (statistics: allowed multiple of the noise floor)."""
    tol = {"error": 0.2, "spectrum": 0.25, "invariants": 0.05, "drift": 1e-3, "floor": 2.0, **(tolerances or {})}
    C: list[dict[str, Any]] = []
    ev: dict[str, Any] = {}
    finite = bool(np.all(np.isfinite(pred)))
    C.append({"id": "rollout.finite", "status": "pass" if finite else "fail",
              "summary": "every state finite" if finite else
              f"the rollout blows up at frame {int(np.argmax(~np.all(np.isfinite(pred.reshape(len(pred), -1)), 1)))}"})
    for name in conserved:
        custom = dict(invariants) if isinstance(invariants, Mapping) else {}
        fn = {name: custom[name]} if name in custom else [name]          # a user function wins over a built-in name
        d = invariant_drift(pred, ndim, None, dt=dt, invariants=fn)[name]
        ok = d["max_drift"] <= tol["drift"]
        C.append({"id": f"rollout.invariant.{name}", "status": "pass" if ok else "fail",
                  "summary": f"{name} drifts by {d['max_drift']:.3g} of its scale over the rollout (rate "
                             f"{d['rate']:.3g} per unit time)", "value": d["max_drift"], "threshold": tol["drift"]})
    if ref is not None:
        dv = divergence_time(pred, ref, ndim, dt=dt, err_tol=tol["error"], spec_tol=tol["spectrum"],
                             inv_tol=tol["invariants"], periodic=periodic, invariants=invariants, window=window)
        ev["divergence"] = dv
        th = dv["trust_horizon"]
        C.append({"id": "rollout.horizon", "status": "pass" if th is None else "warn",
                  "summary": ("no criterion crossed within the rollout" if th is None else
                              f"trust horizon {th:.4g} (first crossing: " + ", ".join(
                                  f"{k} {v:.4g}" for k, v in dv["times"].items() if v is not None) + ")")
                  + f"; rollout length {dv['horizon_end']:.4g}", "value": th, "threshold": None})
        ts = dv["times"]

        def judged(key):                        # a crossing that counts against the model
            t = ts[key]
            if t is None:
                return False
            return not (chaotic and ts["error"] is not None and t > ts["error"])
        early = ts["spectrum"] is not None and (ts["error"] is None or ts["spectrum"] < ts["error"])
        after = "" if ts["error"] is None else f" (after decorrelation at t = {ts['error']:.4g}: judged by statistics)"
        if ndim == 0:
            C.append({"id": "rollout.spectrum", "status": "not-applicable",
                      "summary": "a state vector (ndim = 0): no spatial spectrum"})
        else:
            C.append({"id": "rollout.spectrum", "status": "fail" if judged("spectrum") else "pass",
                      "summary": ("small-scale energy within tolerance over the whole rollout" if ts["spectrum"] is None
                                  else f"spectral bias above {tol['spectrum']:g} from t = {ts['spectrum']:.4g}"
                                  + (" (before the pointwise error crosses its tolerance)" if early else
                                     after if not judged("spectrum") else "")),
                      "value": ts["spectrum"], "threshold": tol["spectrum"]})
        C.append({"id": "rollout.invariants", "status": "fail" if judged("invariants") else "pass",
                  "summary": ("invariants follow the reference" if ts["invariants"] is None else
                              f"an invariant deviates by more than {tol['invariants']:g} from t = "
                              f"{ts['invariants']:.4g}" + ("" if judged("invariants") else after)),
                  "value": ts["invariants"], "threshold": tol["invariants"]})
    if long_ref is not None:
        st = long_run_statistics(pred, long_ref, ndim, dt=dt, burn_in=burn_in, periodic=periodic)
        ev["statistics"] = st
        f = tol["floor"]
        ok_h = st["w1"] <= f * max(st["w1_floor"], 1e-3 * st["scale"])
        ok_s = st["spectrum"] <= f * max(st["spectrum_floor"], 0.05)
        ok_t = np.isfinite(st["tau"]) and abs(st["tau"] - st["tau_ref"]) <= f * max(st["tau_floor"],
                                                                                     0.1 * st["tau_ref"])
        C.append({"id": "rollout.statistics", "status": "pass" if (ok_h and ok_s and ok_t) else "fail",
                  "summary": f"histogram W1 {st['w1']:.3g} (floor {st['w1_floor']:.3g}), spectrum |log ratio| "
                             f"{st['spectrum']:.3g} (floor {st['spectrum_floor']:.3g}), decorrelation time "
                             f"{st['tau']:.3g} vs {st['tau_ref']:.3g}", "value": st["w1"], "threshold": None})
    return RolloutReport(C, ev)
