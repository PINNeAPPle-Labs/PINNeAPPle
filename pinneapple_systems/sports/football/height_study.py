"""Repair of the broadcast ball height: the checks of #911 on real tracking data.

:func:`height_correction` is the study behind ``examples/sports/06_height_correction.py``: it fits the monotone map
of :func:`pinneapple_systems.sports.football.ball.fit_height_map` on a training split of matches and answers, on
held-out matches, what the issue asks -- the free-gravity fit of the corrected height, the landing and apex errors
against the uncorrected feed, and where the Cramer-Rao bound lets a drag coefficient or a spin be reported.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import ball as B

__all__ = ["height_correction", "height_band", "landing_and_apex", "identifiability_summary"]

#: residual altitude the ball has when the feed reports the flight as over, metres (the ball's own radius)
_GROUND = 0.11


def height_band(fit: B.HeightFit, flights, *, groups=None, n_boot: int = 200, grid=None, seed: int = 0,
                n_nodes: int = 12, anchor_weight: float = 0.1, max_rmse: float = 0.6) -> dict[str, Any]:
    """Bootstrap band of the fitted map, resampling **matches** (``groups``), not flights.

    The band is that of the pooled monotone half-step conditional on the per-flight fits of the joint fit: each
    resample pools the pairs of the matches drawn with replacement and reruns the update. It therefore measures how
    much matches disagree about the map, which is the question of whether the correction generalises, and not the
    (smaller) uncertainty of the per-flight parameters.
    """
    flights = list(flights)
    pairs = B._profile_pairs(flights, fit.fits, fit.height, max_rmse)
    if grid is None:
        top = float(np.nanmax(np.concatenate([f.xyz[:, 2] for f in flights])))
        grid = np.linspace(0.0, top, 40)
    zm, zf, w, u, grp = (np.asarray(a, float) for a in pairs)
    lab = np.asarray([flights[int(i)].match_id for i in grp]
                     if groups is None else [groups[int(i)] for i in grp], dtype=object)
    keep = np.isfinite(zm) & np.isfinite(zf) & (w > 0)
    zm, zf, w, u, lab = zm[keep], zf[keep], w[keep], u[keep], lab[keep]
    uniq = list(dict.fromkeys(lab.tolist()))
    if len(uniq) < 3 or len(zm) < 50:
        return {"grid": grid.tolist(), "lo": fit.height(grid).tolist(), "hi": fit.height(grid).tolist(),
                "median": fit.height(grid).tolist(), "n_boot": 0, "n_groups": len(uniq)}
    rng = np.random.default_rng(seed)
    idx = {g: np.where(lab == g)[0] for g in uniq}
    curves = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[i]] for i in pick])
        hm = B._update_height("spline" if fit.kind != "scale" else "scale",
                              (zm[sel], zf[sel], w[sel], u[sel]), n_nodes, anchor_weight)
        curves.append(hm(grid))
    c = np.array(curves)
    return {"grid": np.asarray(grid).tolist(), "lo": np.nanpercentile(c, 2.5, axis=0).tolist(),
            "hi": np.nanpercentile(c, 97.5, axis=0).tolist(), "median": np.nanmedian(c, axis=0).tolist(),
            "n_boot": n_boot, "n_groups": len(uniq)}


def _fit_jobs_pred(job):
    """One held-out prediction for :func:`landing_and_apex` (module level for the process pool)."""
    t, xyz, kw = job
    kw = dict(kw)
    t_pred = np.asarray(kw.pop("t_pred"), float)
    try:
        fit = B.fit_flight(t, xyz, "drag", **kw)
        return fit.predict(t_pred) if fit.success else None
    except Exception:
        return None


def landing_and_apex(flights, height: B.HeightMap | None = None, *, frac: float = 0.5, g: float = B.G,
                     min_samples: int = 5, fit_kw: dict[str, Any] | None = None,
                     workers: int = 1) -> dict[str, Any]:
    """Observe the first ``frac`` of each flight and predict the rest: horizontal landing error and apex error.

    The landing point is the last tracked position's ``(x, y)``, which the height channel does not touch, so the
    comparison against the uncorrected feed is fair. The apex is compared in **feed** units (``height.to_feed`` of the
    predicted apex against the feed's own apex), the only unit in which both models are observed -- a caveat, since
    the map itself was fitted on that relation.
    """
    height = height or B.HeightMap.identity()
    kept, jobs = [], []
    for f in flights:
        if len(f.t) < min_samples + 3:
            continue
        k = max(4, int(round(frac * len(f.t))))
        if k >= len(f.t) - 1:
            continue
        cf = B.correct_flight(f, height)
        kept.append(f)
        jobs.append((cf.t[:k], cf.xyz[:k], {**(fit_kw or {}), "g": g, "t_pred": f.t}))
    land, apex = [], []
    for f, fit in zip(kept, B._pmap(_fit_jobs_pred, jobs, workers), strict=True):
        if fit is None or not np.all(np.isfinite(fit)):
            continue
        land.append(float(np.linalg.norm(fit[-1, :2] - f.xyz[-1, :2])))
        obs_apex = float(np.nanmax(f.xyz[:, 2]))
        j = int(np.argmax(fit[:, 2]))
        apex.append(float(abs(float(height.to_feed(fit[j, 2], f.xyz[j, 0], f.xyz[j, 1])) - obs_apex)))
    land, apex = np.asarray(land, float), np.asarray(apex, float)
    if len(land) == 0:
        return {"n": 0}
    return {"n": len(land), "landing_err_median": float(np.median(land)), "landing_err_mean": float(np.mean(land)),
            "apex_err_median": float(np.median(apex)), "apex_err_mean": float(np.mean(apex)),
            "landing_errors": land, "apex_errors": apex}


def identifiability_summary(flights, height: B.HeightMap | None = None, *, g: float = B.G, workers: int = 1,
                            frac: float = 0.5, fit_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit the drag-Magnus model to each flight and report ``C_D`` and the spin **only where Cramer-Rao allows it**.

    The fit sees the whole flight with gravity fixed. ``C_D`` is reported where its standard error is at most
    :data:`B.CR_CD_MAX` and the Magnus coefficient where every component is at most :data:`B.CR_CL_MAX`.
    """
    height = height or B.HeightMap.identity()
    jobs = []
    for f in flights:
        cf = B.correct_flight(f, height)
        k = max(5, int(round(frac * len(cf.t))))
        jobs.append((cf.t[:k], cf.xyz[:k], "drag_magnus", g, {"cd_prior": (0.25, 0.2), **(fit_kw or {})}))
    fits = B._pmap(B._fit_jobs, jobs, workers)
    rows = [B.identifiability_of(f) for f in fits]
    ok = [r for r in rows if r["cd_identified"] or r["cl_identified"]]
    cd = np.array([r["cd"] for r in rows if r["cd_identified"]], float)
    cl = np.array([r["cl"] for r in rows if r["cl_identified"]], float)
    out = {"n": len(rows), "n_ok": len(ok),
           "cd_fraction": float(np.mean([r["cd_identified"] for r in rows])),
           "cl_fraction": float(np.mean([r["cl_identified"] for r in rows])),
           "cd_median_identified": float(np.median(cd)) if len(cd) else float("nan"),
           "cd_ci95_identified": ([float(np.percentile(cd, 2.5)), float(np.percentile(cd, 97.5))] if len(cd)
                                  else [float("nan")] * 2),
           "cl_median_identified": float(np.median(cl)) if len(cl) else float("nan"),
           "cd_std_median": float(np.median([r["cd_std"] for r in rows if np.isfinite(r["cd_std"])]))
           if np.any(np.isfinite([r["cd_std"] for r in rows])) else float("nan")}
    return out


def _g_audit(flights, height: B.HeightMap | None, *, max_rmse: float = 0.6, workers: int = 1,
             fit_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import ball_study as S
    if height is not None:
        scale = float(np.nanmedian(np.concatenate([height.local_scale(f.xyz[:, 2]) for f in flights])))
        flights = [B.correct_flight(f, height) for f in flights]
        max_rmse = max_rmse * max(scale, 1e-6)
    a = S.gravity_audit(flights, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
    g = np.asarray(a["g_drag"], float)
    return {"n_used": a["n_used"], "g_median": a["g_drag_median"], "g_ci95": a["g_drag_ci95"],
            "g_hz": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))] if len(g) else [float("nan")] * 2,
            "values": g}


def _paired(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = 2000) -> dict[str, float]:
    """Paired bootstrap of the difference of two per-flight error samples (median, 95 % interval)."""
    rng = np.random.default_rng(seed)
    k = min(len(a), len(b))
    a, b = np.asarray(a, float)[:k], np.asarray(b, float)[:k]
    d = [float(np.median(rng.choice(a, k) - rng.choice(b, k))) for _ in range(n)]
    return {"diff_median": float(np.median(a) - np.median(b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def height_correction(train, test, *, kinds=("scale", "camera", "spline"), rounds: int = 2, workers: int = 1,
                      max_rmse: float = 0.6, n_boot: int = 200, known_n: int = 30, seed: int = 0,
                      fit_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole study of #911 on a train/test split of matches; returns the JSON-serialisable result.

    For every map kind: the joint fit on the training matches, its rounds, the map and its band, then on the held-out
    matches the four checks of the issue. The uncorrected feed is the baseline of every comparison.
    """
    train, test = list(train), list(test)
    out: dict[str, Any] = {"n_train_flights": len(train), "n_test_flights": len(test), "kinds": list(kinds)}
    before_test = _g_audit(test, None, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
    before_train = _g_audit(train, None, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
    out["before"] = {"test": {k: v for k, v in before_test.items() if k != "values"},
                     "train": {k: v for k, v in before_train.items() if k != "values"},
                     "test_hist": np.histogram(before_test["values"], bins=np.linspace(0, 16, 33))[0].tolist(),
                     "train_hist": np.histogram(before_train["values"], bins=np.linspace(0, 16, 33))[0].tolist()}
    base_land = landing_and_apex(test, None, fit_kw=fit_kw)
    out["baseline_prediction"] = {k: v for k, v in base_land.items() if not k.endswith("errors")}
    out["kinds_result"] = {}
    for kind in kinds:
        hf = B.fit_height_map(train, kind=kind, rounds=rounds, workers=workers, max_rmse=max_rmse, fit_kw=fit_kw)
        band = height_band(hf, train, n_boot=n_boot, seed=seed, max_rmse=max_rmse)
        gmap = np.asarray(band["grid"], float)
        r: dict[str, Any] = {"map": hf.height.as_dict(), "history": hf.history, "n_used_train": hf.n_used,
                             "residual_rms_feed": hf.residual(), "band": band,
                             "curve": {"grid": gmap.tolist(), "values": np.asarray(hf.height(gmap), float).tolist()}}
        g_test = _g_audit(test, hf.height, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
        g_train = _g_audit(train, hf.height, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
        r["g_test"] = {k: v for k, v in g_test.items() if k != "values"}
        r["g_train"] = {k: v for k, v in g_train.items() if k != "values"}
        r["hist_test"] = np.histogram(g_test["values"], bins=np.linspace(0, 16, 33))[0].tolist()
        r["g_test_by_match"] = _per_match_g(test, hf.height, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
        pred = landing_and_apex(test, hf.height, fit_kw=fit_kw)
        r["prediction"] = {k: v for k, v in pred.items() if not k.endswith("errors")}
        for key in ("landing", "apex"):
            if base_land.get(key + "_errors") is not None and pred.get(key + "_errors") is not None:
                r[f"{key}_paired"] = _paired(pred[key + "_errors"], base_land[key + "_errors"], seed=seed)
        r["identifiability"] = identifiability_summary(test, hf.height, workers=workers, fit_kw=fit_kw)
        out["kinds_result"][kind] = r
    out["identifiability_before"] = identifiability_summary(test, None, workers=workers, fit_kw=fit_kw)
    out["known_answer"] = {k: B.height_recovery_test(n=known_n, kind=k, seed=seed, workers=workers, rounds=1,
                                                     fit_kw=fit_kw) for k in kinds}
    return out


def _per_match_g(test, height: B.HeightMap, *, max_rmse: float, workers: int,
                 fit_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Held-out gravity per match: does the map learned on one set of matches hold in each other match?"""
    out = {}
    for m in dict.fromkeys(f.match_id for f in test):
        sel = [f for f in test if f.match_id == m]
        if len(sel) < 10:
            continue
        a = _g_audit(sel, height, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
        out[str(m)] = {"n": a["n_used"], "g_median": a["g_median"], "g_ci95": a["g_ci95"]}
    return out
