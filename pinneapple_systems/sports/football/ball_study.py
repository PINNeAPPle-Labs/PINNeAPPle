"""Ball-flight studies: does flight physics help on real tracking data, and when is the inverse problem posed?

* :func:`synthetic_recovery` -- the inverse solver on simulated flights with known drag, spin and gravity
  (noise like broadcast tracking): the method check before any real data.
* :func:`identifiability` -- what 3-D and 2-D (no ball height) tracking can recover (Cramer-Rao bounds).
* :func:`gravity_audit` -- fit gravity freely to every aerial flight of a data set: a physically consistent ball
  height gives ``g ~ 9.81 m/s^2``; anything else means the height channel is not usable for aerodynamics.
* :func:`holdout_prediction` -- observe the first part of each flight, predict the rest: polynomial extrapolation
  (data only), vacuum and drag ballistics (physics), gradient-boosted trees trained on other matches (ML), and a
  hybrid (trees with the physics prediction as a feature). Train and test flights come from different matches.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import ball as B

__all__ = ["synthetic_flights", "synthetic_recovery", "identifiability", "gravity_audit", "holdout_prediction",
           "flights_from_skillcorner", "summarise"]


def synthetic_flights(n: int = 40, *, noise: float = 0.2, fps: float = 10.0, seed: int = 0,
                      g: float = B.G) -> list[tuple[B.Flight, dict]]:
    """Simulated kicks (speed 15-30 m/s, elevation 15-40 deg, drag-crisis C_D, random spin) observed at ``fps``
    with Gaussian noise ``noise`` (m) on each coordinate. Returns ``(Flight, truth)`` pairs."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sp = rng.uniform(15, 30)
        el = np.radians(rng.uniform(15, 40))
        az = rng.uniform(0, 2 * np.pi)
        v0 = sp * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        cd = float(B.cd_crisis(B.BALL.reynolds(sp)))
        cl = rng.uniform(0.0, 0.25)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Bm = B.BALL.k(cl) * axis
        T = 2 * v0[2] / g * 0.9                       # roughly to landing
        t = np.arange(0, T, 1 / fps)
        P = B.simulate([0, 0, 0.11], v0, t, cd=cd, magnus=Bm, g=g)
        keep = P[:, 2] > 0
        t, P = t[keep], P[keep]
        obs = P + rng.normal(0, noise, P.shape)
        out.append((B.Flight(f"syn{i}", t, obs, np.ones(len(t))), {"v0": v0, "cd": cd, "magnus": Bm, "g": g,
                                                                    "true": P}))
    return out


def synthetic_recovery(n: int = 40, *, noise: float = 0.2, seed: int = 0) -> dict[str, Any]:
    """Fit ``drag`` and ``drag_free_g`` to :func:`synthetic_flights`: error of launch speed, C_D and g, and the
    coverage of the Laplace 95 % intervals (should be close to 0.95 when the model is right)."""
    rows = []
    for fl, tr in synthetic_flights(n, noise=noise, seed=seed):
        f = B.fit_flight(fl.t, fl.xyz, "drag_magnus", sigma=noise, loss="linear")
        fg = B.fit_flight(fl.t, fl.xyz, "drag_free_g", sigma=noise, loss="linear")
        rows.append({"speed_err": f.speed0 - float(np.linalg.norm(tr["v0"])), "cd_err": f.cd - tr["cd"],
                     "cd_cover": abs(f.cd - tr["cd"]) <= 1.96 * f.std("cd"), "g_err": fg.g - tr["g"],
                     "g_cover": abs(fg.g - tr["g"]) <= 1.96 * fg.std("g"), "cd_std": f.std("cd"),
                     "g_std": fg.std("g")})
    a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    return {"n": n, "noise_m": noise,
            "speed_mae": float(np.mean(np.abs(a["speed_err"]))), "cd_mae": float(np.mean(np.abs(a["cd_err"]))),
            "cd_coverage95": float(np.mean(a["cd_cover"])), "g_mae": float(np.mean(np.abs(a["g_err"]))),
            "g_coverage95": float(np.mean(a["g_cover"])), "g_median_std": float(np.median(a["g_std"])),
            "cd_median_std": float(np.median(a["cd_std"]))}


def identifiability(speed: float = 22.0, elevation_deg: float = 25.0, *, fps: float = 25.0, sigma: float = 0.2,
                    duration: float = 1.6) -> dict[str, Any]:
    """Cramer-Rao standard deviations for one lofted pass seen in 3-D versus 2-D (ball height not observed)."""
    el = np.radians(elevation_deg)
    v0 = speed * np.array([np.cos(el), 0.0, np.sin(el)])
    t = np.arange(0, duration, 1 / fps)
    out = {}
    for name, obs in (("3d", (0, 1, 2)), ("2d", (0, 1))):
        r = B.fisher_identifiability([0, 0, 0.11], v0, t, cd=0.25, magnus=(0, B.BALL.k(0.2), 0), sigma=sigma,
                                     observe=obs)
        out[name] = r
    return out


def flights_from_skillcorner(match_ids, root=None, **kw) -> list[B.Flight]:
    """Aerial flights of SkillCorner matches (:func:`ball.extract_flights` on :func:`data.skillcorner_ball`)."""
    from . import data
    fl = []
    for m in match_ids:
        fl += B.extract_flights(data.skillcorner_ball(m, root), match_id=m, **kw)
    return fl


def _audit_one(f: B.Flight):
    c = np.polyfit(f.t, f.xyz[:, 2], 2)
    fit = B.fit_flight(f.t, f.xyz, "drag_free_g")
    return {"g_vacuum": float(-2 * c[0]), "g_drag": fit.g, "cd": fit.cd, "speed0": fit.speed0, "rmse": fit.rmse,
            "ok": bool(fit.success), "apex": f.apex, "duration": f.duration}


def gravity_audit(flights: list[B.Flight], *, max_rmse: float = 0.6, workers: int = 1) -> dict[str, Any]:
    """Gravity fitted freely to each flight (vacuum: quadratic in time; drag: ODE with free ``g``).

    Flights whose drag fit leaves a position RMSE above ``max_rmse`` m (deflections, tracking switches) are dropped.
    Returns the per-flight values, their median with a bootstrap 95 % interval, and the median per launch-speed
    band (backspin lift grows with the speed squared; a deficit at low speed cannot be lift).
    """
    rows = [r for r in _map(_audit_one, flights, workers) if r["ok"] and r["rmse"] <= max_rmse]
    col = {k: np.array([r[k] for r in rows], float) for k in ("g_vacuum", "g_drag", "cd", "speed0", "apex",
                                                               "duration")} if rows else {}
    rng = np.random.default_rng(0)

    def ci(x):
        if len(x) == 0:
            return [np.nan, np.nan]
        bs = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out = {"n_flights": len(flights), "n_used": len(rows), **{k: v.tolist() for k, v in col.items()}}
    for k in ("g_vacuum", "g_drag"):
        x = col.get(k, np.array([]))
        out[k + "_median"] = float(np.median(x)) if len(x) else float("nan")
        out[k + "_ci95"] = ci(x)
    bands = []
    for lo, hi in ((0, 12), (12, 18), (18, 24), (24, 45)):
        sel = (col["speed0"] >= lo) & (col["speed0"] < hi) if rows else np.array([], bool)
        if sel.sum() >= 5:
            bands.append({"speed_band": [lo, hi], "n": int(sel.sum()), "g_drag_median": float(np.median(col["g_drag"][sel])),
                          "g_drag_ci95": ci(col["g_drag"][sel])})
    out["by_speed"] = bands
    return out


# --------------------------------------------------------------------------------------------- held-out prediction
def _cut(f: B.Flight, frac: float):
    k = max(4, int(round(frac * len(f.t))))
    return k


def _ml_features(t_obs, P_obs, h):
    tt = t_obs - t_obs[-1]
    c = [np.polyfit(tt, P_obs[:, k], 2) for k in range(3)]       # value, velocity, acceleration at the cut
    vel = [ci[1] for ci in c]
    acc = [2 * ci[0] for ci in c]
    return [h, P_obs[-1, 2], *vel, *acc, float(np.hypot(vel[0], vel[1])), t_obs[-1] - t_obs[0]]


def _physics_predict(args):
    to, Po, tn, model = args
    try:
        fit = B.fit_flight(to, Po, model, cd_prior=(0.25, 0.15) if "drag" in model else None)
        return fit.predict(tn)
    except Exception:
        return B.kinematic_extrapolation(to, Po, tn)


def _map(fn, tasks, workers: int):
    if workers <= 1:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, tasks, chunksize=8))


def _cuts(flights, fracs):
    return [(fi, frac, _cut(f, frac)) for fi, f in enumerate(flights) for frac in fracs
            if _cut(f, frac) < len(f.t) - 1]


def _rows(flights, cuts, phys=None):
    X, Y, meta = [], [], []
    for ci, (fi, _frac, k) in enumerate(cuts):
        f = flights[fi]
        to, Po = f.t[:k], f.xyz[:k]
        for j in range(k, len(f.t)):
            x = _ml_features(to, Po, f.t[j] - to[-1])
            if phys is not None:
                x += list(phys[ci][j - k] - Po[-1])
            X.append(x)
            Y.append(f.xyz[j] - Po[-1])
            meta.append((fi, j))
    return np.array(X), np.array(Y), meta


def holdout_prediction(train: list[B.Flight], test: list[B.Flight], *, frac: float = 0.5,
                       train_fracs=(0.3, 0.5, 0.7), seed: int = 0, workers: int = 1) -> dict[str, Any]:
    """Observe the first ``frac`` of each test flight, predict the remaining positions; errors per model.

    Models: ``poly2`` (quadratic per axis; data only, no training), ``vacuum`` and ``drag`` (ballistics with
    g = 9.81 m/s^2; ``drag`` with a weak prior C_D = 0.25 +- 0.15), ``drag_free_g`` (gravity fitted too), ``ml``
    (gradient-boosted trees on the observed-part features, trained on ``train`` flights cut at ``train_fracs``),
    ``hybrid`` (the same trees with the ``drag_free_g`` prediction as extra features). Metrics: RMSE of the
    horizontal and vertical position over the predicted part and the horizontal error at the last tracked sample
    (landing). Physics fits run on ``workers`` processes.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    test_cuts = _cuts(test, (frac,))
    train_cuts = _cuts(train, train_fracs)
    truth = [test[fi].xyz[k:] for fi, _, k in test_cuts]
    preds: dict[str, list[np.ndarray]] = {
        "poly2": [B.kinematic_extrapolation(test[fi].t[:k], test[fi].xyz[:k], test[fi].t[k:]) for fi, _, k in test_cuts]}
    tasks = [(test[fi].t[:k], test[fi].xyz[:k], test[fi].t[k:], m) for m in ("vacuum", "drag", "drag_free_g")
             for fi, _, k in test_cuts]
    tasks += [(train[fi].t[:k], train[fi].xyz[:k], train[fi].t[k:], "drag_free_g") for fi, _, k in train_cuts]
    res = _map(_physics_predict, tasks, workers)
    nt = len(test_cuts)
    for i, m in enumerate(("vacuum", "drag", "drag_free_g")):
        preds[m] = res[i * nt:(i + 1) * nt]
    phys_test, phys_train = preds["drag_free_g"], res[3 * nt:]
    for name, (ptr, pte) in (("ml", (None, None)), ("hybrid", (phys_train, phys_test))):
        Xtr, Ytr, _ = _rows(train, train_cuts, ptr)
        Xte, _, meta = _rows(test, test_cuts, pte)
        models = [HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, Ytr[:, c]) for c in range(3)]
        D = np.stack([m.predict(Xte) for m in models], 1)
        by: dict[int, list] = {}
        for (fi, _j), d in zip(meta, D, strict=True):
            by.setdefault(fi, []).append(d)
        preds[name] = [np.array(by[fi]) + test[fi].xyz[k - 1] for fi, _, k in test_cuts]
    out: dict[str, Any] = {"n_test_flights": nt, "n_train_flights": len(train), "frac_observed": frac}
    for m, P in preds.items():
        eh = np.concatenate([np.linalg.norm(p[:, :2] - y[:, :2], axis=1) for p, y in zip(P, truth, strict=True)])
        ez = np.concatenate([np.abs(p[:, 2] - y[:, 2]) for p, y in zip(P, truth, strict=True)])
        land = np.array([np.linalg.norm(p[-1, :2] - y[-1, :2]) for p, y in zip(P, truth, strict=True)])
        out[m] = {"rmse_xy": float(np.sqrt(np.mean(eh ** 2))), "rmse_z": float(np.sqrt(np.mean(ez ** 2))),
                  "landing_err_median": float(np.median(land)), "landing_err_mean": float(np.mean(land)),
                  "landing_errors": land.tolist()}
    return out


def summarise(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows ``model, rmse_xy, rmse_z, landing_err_median`` of :func:`holdout_prediction`, best landing first."""
    rows = [{"model": m, **{k: v for k, v in r.items() if k != "landing_errors"}}
            for m, r in res.items() if isinstance(r, dict)]
    return sorted(rows, key=lambda r: r["landing_err_median"])
