"""Player-kinematics studies: smoothing with known truth, processing uncertainty of metabolic power on real
tracking, and trajectory prediction with and without physical limits at equal data.

* :func:`synthetic_smoothing` -- simulated runs (bounded speed and acceleration) observed with noise: error of
  speed, acceleration and metabolic power per method (raw differences, Savitzky-Golay, Kalman-RTS) and the
  coverage of the Kalman 95 % intervals.
* :func:`match_load` -- one Metrica match: noise level by maximum likelihood, plausibility per method, metabolic
  power per player per method (the spread is the processing uncertainty).
* :func:`prediction_samples` / :func:`prediction_study` -- position of every outfield player 1 s or 2 s ahead
  from his past second: constant velocity, decaying-acceleration physics, gradient-boosted trees on the history,
  trees on the residual of constant velocity, the same projected onto the reachable set; trained on one match and
  tested on the other with growing training sets.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import kinematics as K

__all__ = ["synthetic_runs", "synthetic_smoothing", "match_load", "prediction_samples", "prediction_study"]


def synthetic_runs(n: int = 20, T: float = 60.0, fps: float = 25.0, *, seed: int = 0, v_cap: float = 9.0,
                   a_cap: float = 6.0, tau: float = 1.0, sigma_a: float = 3.0):
    """True positions, velocities and accelerations of ``n`` players: Ornstein-Uhlenbeck acceleration (time
    constant ``tau``, stationary std ``sigma_a``) clipped at ``a_cap``, speed clipped at ``v_cap``."""
    rng = np.random.default_rng(seed)
    dt = 1 / fps
    N = int(T * fps)
    p = np.zeros((N, n, 2))
    v = np.zeros((N, n, 2))
    a = np.zeros((N, n, 2))
    ai = np.zeros((n, 2))
    vi = rng.normal(0, 1.5, (n, 2))
    pi_ = rng.uniform(-30, 30, (n, 2))
    for k in range(N):
        ai = ai - ai / tau * dt + sigma_a * np.sqrt(2 * dt / tau) * rng.normal(size=(n, 2))
        an = np.linalg.norm(ai, axis=1, keepdims=True)
        ai = np.where(an > a_cap, ai * a_cap / an, ai)
        vn = vi + ai * dt
        sp = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = np.where(sp > v_cap, vn * v_cap / sp, vn)
        acc = (vn - vi) / dt
        pi_ = pi_ + 0.5 * (vi + vn) * dt
        vi = vn
        p[k], v[k], a[k] = pi_, vi, acc
    return p, v, a


def _mp(vel, acc):
    return K.metabolic_power(np.linalg.norm(vel, axis=-1), K.forward_acceleration(vel, acc))


def synthetic_smoothing(noise=(0.05, 0.2), fps=(25.0, 10.0), *, n: int = 20, seed: int = 0) -> list[dict[str, Any]]:
    """RMSE of speed, acceleration and metabolic power per smoothing method against the known truth."""
    rows = []
    for f in fps:
        p, v, a = synthetic_runs(n, fps=f, seed=seed)
        dt = 1 / f
        mp_true = _mp(v, a)
        for s in noise:
            rng = np.random.default_rng(seed + 1)
            obs = p + rng.normal(0, s, p.shape)
            fitted = K.fit_noise(obs, dt, sigmas=(s / 2, s, 2 * s), jerks=(5.0, 10.0, 20.0, 40.0))
            methods = {"raw_differences": K.finite_differences(obs, dt),
                       "savitzky_golay_0.5s": K.savgol_derivatives(obs, dt, 0.5),
                       "savitzky_golay_1.0s": K.savgol_derivatives(obs, dt, 1.0),
                       "kalman_rts": K.kalman_smooth(obs, dt, sigma_pos=fitted["sigma_pos"], jerk=fitted["jerk"])}
            for name, r in methods.items():
                cut = slice(int(f), -int(f))                     # drop 1 s at both ends (edge effects)
                sp_err = np.linalg.norm(r["vel"], axis=-1)[cut] - np.linalg.norm(v, axis=-1)[cut]
                acc_err = np.linalg.norm(r["acc"] - a, axis=-1)[cut]
                mp = _mp(r["vel"], r["acc"])[cut]
                row = {"fps": f, "noise_m": s, "method": name, "speed_rmse": float(np.sqrt(np.mean(sp_err ** 2))),
                       "accel_rmse": float(np.sqrt(np.mean(acc_err ** 2))),
                       "mp_rmse": float(np.sqrt(np.mean((mp - mp_true[cut]) ** 2))),
                       "mp_mean_bias": float(np.mean(mp) - np.mean(mp_true[cut])),
                       "mp_true_mean": float(np.mean(mp_true[cut])), **K.plausibility(r["vel"], r["acc"])}
                if name == "kalman_rts":
                    z = np.abs(r["vel"] - v)[cut] / np.maximum(r["vel_std"][cut], 1e-9)
                    row["vel_coverage95"] = float(np.mean(z <= 1.96))
                    row["sigma_fitted"], row["jerk_fitted"] = fitted["sigma_pos"], fitted["jerk"]
                rows.append(row)
    return rows


def match_load(match, *, team: str = "Home", max_frames: int | None = None) -> dict[str, Any]:
    """Metabolic power of one team of a Metrica match per player and per method, and plausibility per method
    (the first ``max_frames`` frames only, if given). ``gated_mean_mp_per_player`` drops the samples above the
    human limits of :mod:`.kinematics` (tracking jumps), which otherwise dominate a match mean."""
    P = match.players[team][:max_frames]
    dt = 1 / match.fps
    fit = K.fit_noise(P[:15000], dt, sigmas=(0.002, 0.005, 0.01, 0.02), jerks=(10.0, 20.0, 40.0, 80.0))
    methods = {"raw_differences": K.finite_differences(P, dt), "savitzky_golay_0.5s": K.savgol_derivatives(P, dt, 0.5),
               "kalman_rts": K.kalman_smooth(P, dt, sigma_pos=fit["sigma_pos"], jerk=fit["jerk"])}
    out: dict[str, Any] = {"noise_fit": {k: fit[k] for k in ("sigma_pos", "jerk")}, "methods": {}}
    for name, r in methods.items():
        mp = _mp(r["vel"], r["acc"])
        sp = np.linalg.norm(r["vel"], axis=-1)
        ac = np.linalg.norm(r["acc"], axis=-1)
        plausible = (sp <= K.V_MAX) & (ac <= K.A_MAX)
        gated = np.where(plausible, mp, np.nan)                # physically impossible samples are tracking errors
        okg = np.isfinite(gated)
        on = np.isfinite(mp)
        enough = min(25 * 600, len(P) // 2)
        mean_mp = np.array([np.nanmean(mp[on[:, j], j]) if on[:, j].sum() > enough else np.nan
                            for j in range(P.shape[1])])
        med_mp = np.array([np.nanmedian(mp[on[:, j], j]) if on[:, j].sum() > enough else np.nan
                           for j in range(P.shape[1])])
        hs = np.array([np.nansum((sp[:, j] > 5.5) * sp[:, j]) * dt for j in range(P.shape[1])])
        gated_mp = np.array([np.nanmean(gated[okg[:, j], j]) if okg[:, j].sum() > enough else np.nan
                             for j in range(P.shape[1])])
        out["methods"][name] = {"mean_mp_per_player": mean_mp.tolist(), "median_mp_per_player": med_mp.tolist(),
                                "gated_mean_mp_per_player": gated_mp.tolist(),
                                "implausible_share": float(1 - np.sum(okg) / max(np.sum(on), 1)),
                                "hs_distance_per_player": hs.tolist(),
                                **K.plausibility(r["vel"], r["acc"]),
                                "mp_above_20_share": float(np.nanmean(mp[on] > 20))}
    return out


# ----------------------------------------------------------------------------------------------- prediction
def prediction_samples(match, horizon: float, *, every: float = 1.0, history: float = 1.0) -> dict[str, np.ndarray]:
    """Samples ``(features, pos, vel, acc, target)`` for every outfield player every ``every`` s with a complete
    past ``history`` and a tracked position ``horizon`` s later. States are causal Kalman estimates."""
    dt = 1 / match.fps
    H = int(round(horizon / dt))
    k = int(round(history / dt))
    step = int(round(every / dt))
    lag = np.arange(-k, 1, max(k // 5, 1))
    rows = {"X": [], "pos": [], "vel": [], "acc": [], "target": []}
    for team, P in match.players.items():
        st = K.kalman_smooth(P, dt, sigma_pos=0.01, jerk=40.0, causal=True)
        for j in range(P.shape[1]):
            if j == match.gk[team]:
                continue
            for f in range(k, len(P) - H, step):
                hist = P[f + lag, j]
                if not np.all(np.isfinite(hist)) or not np.all(np.isfinite(P[f + H, j])) or not np.isfinite(match.ball[f, 0]):
                    continue
                if match.period[f] != match.period[f + H]:
                    continue
                rows["X"].append(np.concatenate([(hist - hist[-1]).ravel(), hist[-1], match.ball[f] - hist[-1],
                                                 st["vel"][f, j], st["acc"][f, j]]))
                rows["pos"].append(P[f, j])
                rows["vel"].append(st["vel"][f, j])
                rows["acc"].append(st["acc"][f, j])
                rows["target"].append(P[f + H, j])
    return {kk: np.asarray(v, float) for kk, v in rows.items()}


def prediction_study(train: dict, test: dict, horizon: float, *, sizes=(100, 300, 1000, 3000, None),
                     seeds=(0, 1, 2)) -> dict[str, Any]:
    """Final displacement error on ``test`` of each predictor trained on random subsets of ``train``."""
    rows = []
    phys = {"constant_velocity": K.predict_constant_velocity(test["pos"], test["vel"], horizon),
            "physics_limited": K.predict_limited(test["pos"], test["vel"], test["acc"], horizon)}
    for name, pr in phys.items():
        rows.append({"model": name, "n_train": 0, "seed": 0,
                     **K.displacement_errors(pr, test["target"], test["pos"], test["vel"], horizon)})
    for n in sizes:
        for s in seeds if n is not None else seeds[:1]:
            rng = np.random.default_rng(100 + s)
            idx = np.arange(len(train["X"])) if n is None else rng.choice(len(train["X"]), n, replace=False)
            for name, kw in (("gbm", {}), ("gbm_residual_cv", {"residual_of": "cv"}),
                             ("gbm_residual_cv_projected", {"residual_of": "cv", "project": True})):
                m = K.MLPredictor(seed=s, **kw).fit(train["X"][idx], train["pos"][idx], train["vel"][idx],
                                                    train["target"][idx], horizon)
                pr = m.predict(test["X"], test["pos"], test["vel"])
                rows.append({"model": name, "n_train": int(len(idx)), "seed": s,
                             **K.displacement_errors(pr, test["target"], test["pos"], test["vel"], horizon)})
    return {"horizon_s": horizon, "n_test": int(len(test["X"])), "rows": rows}
