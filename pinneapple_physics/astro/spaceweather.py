"""Space weather for observers: a Kp forecast with calibrated uncertainty and the probability of seeing the aurora.

* Data: observed 3-hourly Kp and daily F10.7 (bundled: the year 2000, near solar maximum; live: NOAA SWPC JSON).
* Models, all scored on held-out months (time-ordered split, no look-ahead):
  - persistence (Kp now) and 27-day recurrence (the Sun's synodic rotation brings back the same coronal holes);
  - a quantile neural network (pinball loss, q = 0.1 / 0.5 / 0.9) on the recent Kp history, the 27-day recurrence
    and F10.7, with its interval conformalised (CQR, Romano et al. 2019) on a calibration block for honest coverage.
* Aurora: the equatorward edge of the visible auroral oval moves ~2 deg of geomagnetic latitude per Kp unit
  (NOAA SWPC rule of thumb: 66.5 deg at Kp 0 ... 48 deg at Kp 9); the observer's geomagnetic latitude comes from the
  IGRF-13 (2020) dipole. P(aurora overhead) = P(Kp >= Kp needed); "low on the poleward horizon" allows ~6 deg more.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "spaceweather", "kp_f107_2000.csv")
SWPC_KP = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
STEPS_PER_DAY = 8
RECUR = 27 * STEPS_PER_DAY
QUANTILES = (0.1, 0.5, 0.9)

# IGRF-13 degree-1 coefficients for 2020.0 (nT): the geomagnetic dipole
G10, G11, H11 = -29404.8, -1450.9, 4652.5


def load_history():
    rows = [ln.strip().split(",") for ln in open(DATA) if ln[0].isdigit()]
    kp = np.array([[int(x) / 10 for x in r[1:9]] for r in rows]).ravel()
    f107 = np.repeat(np.array([float(r[9]) for r in rows]), STEPS_PER_DAY)
    days = [r[0] for r in rows]
    return {"kp": kp, "f107": f107, "days": days}


def geomagnetic_latitude(lat_deg, lon_deg):
    """Dipole (centred, IGRF-13 2020) geomagnetic latitude."""
    b0 = math.sqrt(G10 ** 2 + G11 ** 2 + H11 ** 2)
    pole_lat = math.degrees(math.asin(-G10 / b0))                    # ~80.65 N
    pole_lon = math.degrees(math.atan2(-H11, -G11))                   # ~-72.68 E
    la, lo, pla, plo = map(math.radians, (lat_deg, lon_deg, pole_lat, pole_lon))
    s = math.sin(la) * math.sin(pla) + math.cos(la) * math.cos(pla) * math.cos(lo - plo)
    return math.degrees(math.asin(max(-1, min(1, s)))), pole_lat, pole_lon


def oval_boundary(kp):
    """Equatorward geomagnetic latitude of the visible aurora for a Kp (NOAA SWPC table: 66.5 deg at Kp 0 to
    48.1 deg at Kp 9, ~2.04 deg per Kp)."""
    return 66.5 - 2.044 * np.asarray(kp, dtype=float)


def kp_needed(maglat_deg, horizon_margin_deg=0.0):
    return float(np.clip((66.5 - (abs(maglat_deg) + horizon_margin_deg)) / 2.044, 0, 9))


def _features(kp, f107, t):
    """Features for the forecast issued at index t (after Kp[t] is known)."""
    last = kp[t - 7: t + 1]
    rec = kp[t - RECUR + 1: t - RECUR + 9] if t - RECUR + 9 <= len(kp) and t - RECUR + 1 >= 0 else np.full(8, kp[t])
    return np.concatenate([last, [last.mean(), last.max(), kp[max(0, t - 23): t + 1].mean()], rec,
                           [f107[t] / 100.0]])


def build_xy(kp, f107, horizon):
    idx = np.arange(RECUR, len(kp) - horizon)
    X = np.array([_features(kp, f107, t) for t in idx])
    y = kp[idx + horizon]
    return idx, X, y


def _pinball(pred, y, qs):
    import torch
    loss = 0
    for k, q in enumerate(qs):
        e = y - pred[:, k]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss


def train_forecaster(horizon: int = 1, seed: int = 0, epochs: int = 300):
    """Quantile network for Kp ``horizon`` 3-hour steps ahead, conformalised; evaluated on the last two months."""
    import torch
    torch.manual_seed(seed)
    h = load_history()
    kp, f107 = h["kp"], h["f107"]
    idx, X, y = build_xy(kp, f107, horizon)
    n_days = len(kp) // STEPS_PER_DAY
    day = idx // STEPS_PER_DAY
    train = day < int(n_days * 8 / 12)
    cal = (day >= int(n_days * 8 / 12)) & (day < int(n_days * 10 / 12))
    test = day >= int(n_days * 10 / 12)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
    Z = torch.tensor((X - mu) / sd)
    base = kp[idx]                                               # persistence: the model learns the change only
    Y = torch.tensor(y - base)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                              torch.nn.Tanh(), torch.nn.Linear(32, len(QUANTILES))).double()
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)                           # start exactly at persistence
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
    tr = torch.tensor(np.nonzero(train)[0])
    for _ in range(epochs):
        loss = _pinball(net(Z[tr]), Y[tr], QUANTILES)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = net(Z).numpy() + base[:, None]
    P.sort(1)                                                    # no quantile crossing
    # model selection on the calibration block (never on the test months): keep the network's median only if it
    # beats persistence there
    use_net = float(np.mean(np.abs(P[cal, 1] - y[cal]))) < float(np.mean(np.abs(base[cal] - y[cal])))
    if not use_net:
        P[:, 1] = base
        P.sort(1)
    # conformalised quantile regression on the calibration block (coverage target 80 %)
    s = np.maximum(P[cal, 0] - y[cal], y[cal] - P[cal, 2])
    n = len(s)
    qhat = float(np.sort(s)[min(n - 1, math.ceil((n + 1) * 0.8) - 1)])
    lo, hi = np.clip(P[:, 0] - qhat, 0, 9), np.clip(P[:, 2] + qhat, 0, 9)
    med = np.clip(P[:, 1], 0, 9)
    persistence = kp[idx]
    recurrence = kp[idx + horizon - RECUR]
    mae = lambda a: float(np.mean(np.abs(a[test] - y[test])))  # noqa: E731
    cover = float(np.mean((y[test] >= lo[test]) & (y[test] <= hi[test])))
    storms = y[test] >= 5
    hit = float(np.mean(hi[test][storms] >= 5)) if storms.any() else None
    checks = [
        {"id": "kp_mae", "args": {"hours": horizon * 3},
         "name": f"Median forecast MAE vs persistence (test: Nov-Dec, {horizon * 3} h ahead)", "value": mae(med),
         "threshold": mae(persistence), "unit": "Kp", "status": "pass" if mae(med) <= mae(persistence) else "fail"},
        {"id": "kp_coverage", "name": "80 % interval coverage on the test months", "value": cover, "threshold": 0.75, "unit": "fraction",
         "status": "pass" if cover >= 0.75 else "warn", "higher_is_better": True},
    ]
    if hit is not None:
        checks.append({"id": "kp_storms", "name": "Storm periods (Kp >= 5) inside the upper bound", "value": hit, "threshold": 0.5,
                       "unit": "fraction", "status": "pass" if hit >= 0.5 else "warn", "higher_is_better": True})
    model = {"net": net, "mu": mu, "sd": sd, "qhat": qhat, "horizon": horizon, "use_net_median": use_net}
    report = {
        "horizon_h": horizon * 3, "n_train": int(train.sum()), "n_cal": int(cal.sum()), "n_test": int(test.sum()),
        "mae_model": mae(med), "mae_persistence": mae(persistence), "mae_recurrence": mae(recurrence),
        "mae_climatology": float(np.mean(np.abs(np.median(y[train]) - y[test]))),
        "coverage_80": cover, "storm_hit_rate": hit, "conformal_qhat": qhat, "checks": checks,
        "median_source": "network" if use_net else "persistence (network median lost on the calibration block)",
        "verdict": {"status": "trustworthy" if all(c["status"] == "pass" for c in checks) else
                    "use_with_caution" if not any(c["status"] == "fail" for c in checks) else "not_trustworthy"},
        "test_series": {"t_index": idx[test].tolist(), "observed": y[test].tolist(), "median": med[test].tolist(),
                        "lo": lo[test].tolist(), "hi": hi[test].tolist(), "persistence": persistence[test].tolist()},
        "method": "quantile MLP (pinball loss) for the change from persistence, on the last 24 h of Kp, the 27-day recurrence and F10.7; split-"
                  "conformal widening on Sep-Oct; scored on Nov-Dec 2000 against observed Kp (GFZ)",
    }
    return model, report


def forecast(model, kp_hist, f107_now):
    """Forecast (lo, median, hi) for Kp ``horizon`` steps after the last value of ``kp_hist`` (needs >= 27 days)."""
    import torch
    kp_hist = np.asarray(kp_hist, dtype=float)
    f = np.full(len(kp_hist), f107_now)
    x = (_features(kp_hist, f, len(kp_hist) - 1) - model["mu"]) / model["sd"]
    with torch.no_grad():
        p = model["net"](torch.tensor(x[None])).numpy()[0] + kp_hist[-1]
    if not model.get("use_net_median", True):
        p[1] = kp_hist[-1]
    p = np.sort(p)
    return (float(np.clip(p[0] - model["qhat"], 0, 9)), float(np.clip(p[1], 0, 9)),
            float(np.clip(p[2] + model["qhat"], 0, 9)))


def aurora_probability(lo, med, hi, maglat_deg):
    """P(Kp >= needed) from the forecast quantiles (piecewise-linear CDF through q10 / q50 / q90, clipped 0..9),
    overhead and low on the poleward horizon (~6 deg more reach)."""
    qs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    vals = np.array([0.0, lo, med, hi, 9.0])
    vals = np.maximum.accumulate(vals)

    def p_ge(k):
        return float(1 - np.interp(k, vals, qs))
    k_over, k_hor = kp_needed(maglat_deg), kp_needed(maglat_deg, 6.0)
    return {"kp_needed_overhead": k_over, "kp_needed_horizon": k_hor,
            "p_overhead": p_ge(k_over) if k_over < 9 else 0.0, "p_horizon": p_ge(k_hor) if k_hor < 9 else 0.0}


def fetch_live_kp(timeout=10.0):
    """Last ~7 days of 3-hourly planetary Kp from NOAA SWPC (JSON). Returns (iso times, kp) or raises."""
    req = urllib.request.Request(SWPC_KP, headers={"User-Agent": "PINNeAPPle-observatory/1.0"})
    rows = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    rows = [r for r in rows[1:] if r[1] not in (None, "")]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])
