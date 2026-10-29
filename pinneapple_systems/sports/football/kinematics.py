"""Player kinematics from tracking: smoothing with uncertainty, physical plausibility, metabolic power and
short-horizon trajectory prediction with and without physical limits.

* :func:`kalman_smooth` -- constant-acceleration (white-jerk) Kalman filter with a Rauch-Tung-Striebel smoother per
  coordinate, vectorised over players; positions, velocities and accelerations with their standard deviations.
  Missing samples (NaN) are bridged by the model.
* :func:`savgol_derivatives`, :func:`finite_differences` -- the usual data-only alternatives.
* :func:`plausibility` -- the share of samples above human limits (speed, acceleration).
* :func:`metabolic_power` -- the equivalent-slope model: accelerated running on flat ground costs as much as
  running uphill at constant speed on the slope ``ES = a / g`` with the body weight scaled by
  ``EM = sqrt(ES^2 + 1)`` (di Prampero, P. E. et al. (2005) Sprint running: a new energetic approach, J. Exp.
  Biol. 208, 2809-2816; Osgnach, C. et al. (2010) Energy cost and metabolic power in elite soccer: a new match
  analysis approach, Med. Sci. Sports Exerc. 42, 170-178, doi:10.1249/MSS.0b013e3181ae5cfd). The cost of running
  on a slope ``i`` is the polynomial of Minetti, A. E. et al. (2002) Energy cost of walking and running at extreme
  uphill and downhill slopes, J. Appl. Physiol. 93, 1039-1046:
  ``C(i) = 155.4 i^5 - 30.4 i^4 - 43.3 i^3 + 46.3 i^2 + 19.5 i + 3.6`` J/(kg m), valid for ``|i| <= 0.45``.
  Osgnach et al. multiply the cost by a terrain factor for grass; it is the ``terrain`` argument here (default 1:
  treadmill-equivalent values; take the factor from the paper before comparing with published match loads).
* :func:`predict_constant_velocity`, :func:`predict_limited`, :class:`MLPredictor` -- trajectory prediction
  baselines; :func:`reachable_radius` bounds any prediction by the player's acceleration and speed limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["V_MAX", "A_MAX", "kalman_smooth", "savgol_derivatives", "finite_differences", "plausibility",
           "metabolic_power", "minetti_cost", "reachable_radius", "predict_constant_velocity", "predict_limited",
           "project_reachable", "history_features", "fit_noise", "MLPredictor", "displacement_errors"]

#: physical limits used for plausibility and constraints. 12.5 m/s is above the fastest recorded human sprint
#: speeds (about 12.3 m/s peak in the 2009 100 m world record); 10 m/s^2 is above the horizontal accelerations
#: reported in elite sprint starts. Values are deliberately generous: they flag errors, not effort.
V_MAX = 12.5
A_MAX = 10.0
G = 9.81


# ------------------------------------------------------------------------------------------------------ smoothing
def kalman_smooth(xy: np.ndarray, dt: float, *, sigma_pos: float = 0.1, jerk: float = 30.0,
                  causal: bool = False, max_gap: float = 1.0) -> dict[str, np.ndarray]:
    """RTS-smoothed position, velocity and acceleration of tracks ``xy`` of shape ``(T, ..., 2)``.

    The state per coordinate is ``(p, v, a)`` with white jerk of spectral density ``jerk**2`` (m^2/s^5) and
    measurement noise ``sigma_pos`` (m). Returns ``pos, vel, acc`` (same shape as ``xy``) and ``pos_std, vel_std,
    acc_std`` and the innovation log-likelihood ``loglik`` (for :func:`fit_noise`). NaN measurements are skipped (prediction only). The smoother is linear and exact for this model, so
    its standard deviations are the posterior uncertainty given the stated noise levels. ``causal=True`` returns
    the forward filter only (each estimate uses past samples only, as a real-time or a fair predictive use needs).
    """
    xy = np.asarray(xy, float)
    T = xy.shape[0]
    flat = xy.reshape(T, -1)                                   # each column one coordinate of one track
    m = flat.shape[1]
    F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    q = jerk ** 2
    Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6], [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                      [dt ** 3 / 6, dt ** 2 / 2, dt]])
    R = sigma_pos ** 2
    xf = np.zeros((T, m, 3))
    Pf = np.zeros((T, m, 3, 3))
    xp = np.zeros((T, m, 3))
    Pp = np.zeros((T, m, 3, 3))
    loglik = np.zeros(m)
    gap = max(int(round(max_gap / dt)), 1)
    P0 = np.diag([R, 25.0, 25.0])
    x = np.zeros((m, 3))
    P = np.broadcast_to(P0, (m, 3, 3)).copy()
    last = np.full(m, -10 ** 9)
    for k in range(T):
        live = (k - last) <= gap                      # tracks measured recently: predict; others wait (frozen)
        if k > 0 and live.any():
            x[live] = x[live] @ F.T
            P[live] = F @ P[live] @ F.T + Q
        xp[k], Pp[k] = x, P
        z = flat[k]
        ok = np.isfinite(z)
        fresh = ok & ~live                            # first sample, or back after a long gap: restart the state
        if fresh.any():
            x[fresh] = 0.0
            x[fresh, 0] = z[fresh]
            P[fresh] = P0
        upd = ok & live
        if upd.any():
            S = P[upd, 0, 0] + R
            K = P[upd, :, 0] / S[:, None]
            innov = z[upd] - x[upd, 0]
            loglik[upd] += -0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)
            x[upd] = x[upd] + K * innov[:, None]
            P[upd] = P[upd] - K[:, :, None] * P[upd, 0, :][:, None, :]
        last[ok] = k
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    Ps = Pf.copy()
    for k in range(T - 2, -1, -1) if not causal else ():
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + np.einsum("mij,mj->mi", C, xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ np.transpose(C, (0, 2, 1))
    shape = xy.shape
    out = {}
    for i, name in enumerate(("pos", "vel", "acc")):
        out[name] = xs[:, :, i].reshape(shape)
        out[name + "_std"] = np.sqrt(np.clip(Ps[:, :, i, i], 0, None)).reshape(shape)
    out["loglik"] = float(loglik.sum())
    # samples farther than max_gap from a measurement (absent players, long dropouts) are NaN
    meas = np.isfinite(flat)
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(meas, idx, -10 ** 9), axis=0)
    nxt = np.minimum.accumulate(np.where(meas, idx, 10 ** 9)[::-1], axis=0)[::-1]
    far = ((idx - prev) > gap) | (((nxt - idx) > gap) if not causal else False)
    for k, v in out.items():
        if k != "loglik":
            v.reshape(T, -1)[far] = np.nan
    return out


def fit_noise(xy: np.ndarray, dt: float, sigmas=(0.01, 0.02, 0.05, 0.1, 0.2), jerks=(5.0, 10.0, 20.0, 40.0, 80.0)):
    """Measurement noise and jerk density by maximum marginal likelihood of :func:`kalman_smooth` (grid search on
    the innovations): the data, not a guess, set how much the smoother trusts the samples."""
    table = {(s, j): kalman_smooth(xy, dt, sigma_pos=s, jerk=j, causal=True)["loglik"] for s in sigmas for j in jerks}
    best = max(table, key=table.get)
    return {"sigma_pos": best[0], "jerk": best[1], "table": {f"{k[0]}_{k[1]}": v for k, v in table.items()}}


def finite_differences(xy: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Central differences of raw positions: ``vel``, ``acc`` (the data-only baseline with no smoothing)."""
    xy = np.asarray(xy, float)
    vel = np.gradient(xy, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return {"pos": xy, "vel": vel, "acc": acc}


def savgol_derivatives(xy: np.ndarray, dt: float, window: float = 0.5, order: int = 2) -> dict[str, np.ndarray]:
    """Savitzky-Golay smoothed position, velocity and acceleration (``window`` in seconds; NaN filled linearly)."""
    from scipy.signal import savgol_filter
    xy = np.asarray(xy, float)
    filled = xy.copy()
    T = len(xy)
    flat = filled.reshape(T, -1)
    idx = np.arange(T)
    for j in range(flat.shape[1]):
        ok = np.isfinite(flat[:, j])
        if ok.sum() >= 2 and not ok.all():
            flat[:, j] = np.interp(idx, idx[ok], flat[ok, j])
        elif ok.sum() < 2:
            flat[:, j] = 0.0
    w = max(int(round(window / dt)) | 1, order + 2 | 1)
    out = {"pos": savgol_filter(filled, w, order, axis=0),
           "vel": savgol_filter(filled, w, order, deriv=1, delta=dt, axis=0),
           "acc": savgol_filter(filled, w, max(order, 2), deriv=2, delta=dt, axis=0)}
    nan = ~np.isfinite(xy)
    for v in out.values():
        v[nan] = np.nan
    return out


def plausibility(vel: np.ndarray, acc: np.ndarray, *, v_max: float = V_MAX, a_max: float = A_MAX) -> dict[str, float]:
    """Share of finite samples whose speed exceeds ``v_max`` or whose acceleration magnitude exceeds ``a_max``."""
    sp = np.linalg.norm(vel, axis=-1)
    ac = np.linalg.norm(acc, axis=-1)
    ok = np.isfinite(sp) & np.isfinite(ac)
    return {"speed_violation": float(np.mean(sp[ok] > v_max)), "accel_violation": float(np.mean(ac[ok] > a_max)),
            "speed_p99": float(np.percentile(sp[ok], 99)), "accel_p99": float(np.percentile(ac[ok], 99))}


# -------------------------------------------------------------------------------------------------- metabolic power
def minetti_cost(i):
    """Energy cost of running on slope ``i`` (Minetti et al. 2002), J/(kg m); ``i`` clipped to [-0.45, 0.45]."""
    i = np.clip(np.asarray(i, float), -0.45, 0.45)
    return 155.4 * i ** 5 - 30.4 * i ** 4 - 43.3 * i ** 3 + 46.3 * i ** 2 + 19.5 * i + 3.6


def metabolic_power(speed, accel_fwd, *, terrain: float = 1.0) -> np.ndarray:
    """Metabolic power in W/kg from speed (m/s) and forward acceleration (m/s^2), equivalent-slope model.

    ``ES = a / g``, ``EM = sqrt(ES^2 + 1)``, ``C = minetti_cost(ES) * EM * terrain``, ``P = C v``.
    """
    v = np.asarray(speed, float)
    es = np.asarray(accel_fwd, float) / G
    em = np.sqrt(es ** 2 + 1)
    return minetti_cost(es) * em * terrain * v


def forward_acceleration(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Component of the acceleration along the velocity (tangential), the input of :func:`metabolic_power`."""
    sp = np.linalg.norm(vel, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sp > 1e-6, np.sum(vel * acc, axis=-1) / sp, 0.0)


__all__.append("forward_acceleration")


# ---------------------------------------------------------------------------------------------------- prediction
def reachable_radius(speed0, horizon: float, *, v_max: float = V_MAX, a_max: float = A_MAX):
    """Largest distance a player moving at ``speed0`` can cover in ``horizon`` s (accelerate at ``a_max`` up to
    ``v_max``): the physical bound on any displacement prediction."""
    s0 = np.minimum(np.asarray(speed0, float), v_max)
    t_up = (v_max - s0) / a_max
    return np.where(horizon <= t_up, s0 * horizon + 0.5 * a_max * horizon ** 2,
                    s0 * t_up + 0.5 * a_max * t_up ** 2 + v_max * (horizon - t_up))


def predict_constant_velocity(pos, vel, horizon: float):
    """Position after ``horizon`` s at constant velocity."""
    return np.asarray(pos) + np.asarray(vel) * horizon


def predict_limited(pos, vel, acc, horizon: float, *, v_max: float = V_MAX, a_max: float = A_MAX,
                    tau: float = 0.5, dt: float = 0.04):
    """Physics prediction: the current acceleration (capped at ``a_max``) decays with time constant ``tau`` and the
    speed is capped at ``v_max`` (players cannot keep accelerating; a simple first-order motor model)."""
    p = np.array(pos, float)
    v = np.array(vel, float)
    a = np.array(acc, float)
    an = np.linalg.norm(a, axis=-1, keepdims=True)
    a = np.where(an > a_max, a * a_max / np.maximum(an, 1e-9), a)
    t = 0.0
    while t < horizon - 1e-9:
        h = min(dt, horizon - t)
        v = v + a * h
        sp = np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.where(sp > v_max, v * v_max / np.maximum(sp, 1e-9), v)
        p = p + v * h
        a = a * np.exp(-h / tau)
        t += h
    return p


def project_reachable(pos, vel, pred, horizon: float, *, v_max: float = V_MAX, a_max: float = A_MAX):
    """Project predictions onto the reachable set: the disc around the coasting point ``pos + vel * horizon`` of
    radius ``a_max horizon^2 / 2`` intersected with the disc of radius :func:`reachable_radius` around ``pos``."""
    pos, vel, pred = (np.asarray(a, float) for a in (pos, vel, pred))
    c = pos + vel * horizon
    r1 = 0.5 * a_max * horizon ** 2
    d = pred - c
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    out = np.where(n > r1, c + d * r1 / np.maximum(n, 1e-9), pred)
    r2 = reachable_radius(np.linalg.norm(vel, axis=-1), horizon, v_max=v_max, a_max=a_max)[..., None]
    d = out - pos
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > r2, pos + d * r2 / np.maximum(n, 1e-9), out)


def history_features(pos_hist: np.ndarray, ball_rel: np.ndarray | None = None) -> np.ndarray:
    """Features for the ML predictor: past positions relative to the current one (``(n, k, 2)`` -> ``(n, 2k)``),
    optionally the ball position relative to the player."""
    rel = pos_hist - pos_hist[:, -1:, :]
    f = [rel.reshape(len(rel), -1), pos_hist[:, -1, :]]
    if ball_rel is not None:
        f.append(ball_rel)
    return np.concatenate(f, axis=1)


@dataclass
class MLPredictor:
    """Data-only displacement predictor: gradient-boosted trees (scikit-learn) on :func:`history_features`.

    ``residual_of="cv"`` learns the correction to the constant-velocity prediction instead of the raw displacement
    (a physics prior the trees only correct); ``project=True`` projects the output onto the reachable set.
    """

    residual_of: str | None = None
    project: bool = False
    max_iter: int = 200
    seed: int = 0
    _models: Any = None

    def _base(self, pos, vel, horizon):
        if self.residual_of == "cv":
            return predict_constant_velocity(pos, vel, horizon) - pos
        return np.zeros_like(pos)

    def fit(self, X, pos, vel, target_pos, horizon: float):
        from sklearn.ensemble import HistGradientBoostingRegressor
        y = target_pos - pos - self._base(pos, vel, horizon)
        self._models = [HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=self.seed).fit(X, y[:, k])
                        for k in range(2)]
        self._h = horizon
        return self

    def predict(self, X, pos, vel):
        d = np.stack([m.predict(X) for m in self._models], axis=1)
        pred = pos + self._base(pos, vel, self._h) + d
        if self.project:
            pred = project_reachable(pos, vel, pred, self._h)
        return pred


def displacement_errors(pred, true, pos=None, vel=None, horizon: float | None = None) -> dict[str, float]:
    """Mean and 90th-percentile final displacement error (m); with ``pos, vel, horizon`` also the share of
    predictions outside the physically reachable set."""
    e = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    out = {"fde_mean": float(np.mean(e)), "fde_p90": float(np.percentile(e, 90)), "n": int(e.size)}
    if pos is not None:
        proj = project_reachable(pos, vel, pred, horizon)
        out["unreachable"] = float(np.mean(np.linalg.norm(proj - pred, axis=-1) > 1e-6))
    return out
