"""BATADAL adapter for the physics-residual FDI benchmark: real SCADA data from the C-Town water network.

BATADAL (Battle of the Attack Detection Algorithms; Taormina et al., J. Water Resour. Plann. Manage. 144(8), 2018)
publishes hourly SCADA data of a simulated 429-pipe network with 7 tanks, 11 pumps and a flow-control valve: tank
levels (m), pump and valve flows (L/s) and status, and junction pressures (m). Files used here, all from
https://www.batadal.net/data.html:

* ``BATADAL_dataset03.csv``: one attack-free year (8761 h), the trusted data a detector is fitted and calibrated on;
* ``BATADAL_dataset04.csv``: 4177 h with attacks, labelled only partly. ``ATT_FLAG`` is 1 in five labelled attack
  episodes and ``-999`` elsewhere, which means *unlabelled*, not clean: the organisers say undiscovered attacks remain;
* ``CTOWN.INP``: the EPANET model, from which the tank diameters and the pump and valve connections below come.

The competition's test dataset has no labels in the file and its attack times are not published on the site, so the
official competition score cannot be reproduced from public files. This module therefore scores on dataset 04's
labelled episodes, and measures the false-alarm rate on attack-free data held out from dataset 03.

**Licence.** The BATADAL site states "This work is licensed under a Creative Commons Attribution 4.0 International
License" (CC BY 4.0) and its footer also reads "(c) BATADAL team. All rights reserved". The data are therefore not
committed here: :func:`fetch` downloads them to a cache (``$PINNEAPPLE_BATADAL_CACHE``, default
``~/.cache/pinneapple/batadal``) and checks their SHA-256. Cite Taormina et al. (2018) when you use them.

**Grey-box twin.** Tank balances (:class:`TankBalance`) and pump head curves (:class:`PumpLaw`), constants identified
on the attack-free year. Demand is not measured, so each balance carries a demand-by-hour-of-day profile that is
identified together with its flow coefficients; the flows that may enter a balance come from the INP topology
(:data:`TANK_FLOWS`), not from a search over every flow.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import math
import os
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cps import Data, FlowLaw, PhysicsResidualDetector, episode_metrics, window_auc

__all__ = ["FILES", "TANK_FLOWS", "SENSORS", "cache_dir", "fetch", "load", "episodes", "TankBalance", "PumpLaw",
           "ctown_detector", "ttd_score", "classification_score", "batadal_score", "run_benchmark"]

BASE_URL = "https://www.batadal.net/data/"
FILES = {  # name: (SHA-256, bytes), as downloaded on 2026-10-15
    "BATADAL_dataset03.csv": ("8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b", 2940389),
    "BATADAL_dataset04.csv": ("4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b", 1225392),
    "CTOWN.INP": ("9198b0bb43f551b29bfb550e206ff95f1df220118fc0068c7f130cec78d58d9e", 116999),
}
SAMPLES_PER_DAY = 24
FLOW_TO_M3_PER_H = 3.6                                                  # 1 L/s = 3.6 m^3/h

#: Tank diameters (m) from ``CTOWN.INP`` ``[TANKS]``; the floor area is pi d^2 / 4.
TANK_DIAMETER = {"T1": 31.3, "T2": 20.78, "T3": 13.73, "T4": 11.64, "T5": 11.89, "T6": 8.33, "T7": 7.14}
TANK_AREA = {k: math.pi * d * d / 4 for k, d in TANK_DIAMETER.items()}

#: The flows that can enter each tank's balance, ``(inflows, outflows)``, read off the INP: a pump fills the tank
#: nearest its discharge node and drains the tank nearest its suction node, and valve V2 moves water from T1 to T2.
#: PU10 and PU11 discharge nearest T6 in the pipe graph, but T7's level follows them in the data and T6's does not, so
#: both tanks list them and the identification decides. Pumps that rarely run in the attack-free year (PU3, PU5, PU6,
#: PU9, PU11) have almost no samples to identify a coefficient from, which the fit tolerates (their columns are ~0).
TANK_FLOWS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "T1": (("F_PU1", "F_PU2", "F_PU3"), ("F_V2", "F_PU8", "F_PU9", "F_PU10", "F_PU11")),
    "T2": (("F_V2",), ("F_PU4", "F_PU5", "F_PU6", "F_PU7")),
    "T3": (("F_PU4", "F_PU5"), ()),
    "T4": (("F_PU6", "F_PU7"), ()),
    "T5": (("F_PU8", "F_PU9"), ()),
    "T6": (("F_PU10", "F_PU11"), ()),
    "T7": (("F_PU10", "F_PU11"), ()),
}
#: Every measured quantity except the 0/1 pump status: the columns the data-driven baselines read.
SENSORS = (tuple(f"L_T{i}" for i in range(1, 8)) + tuple(f"F_PU{i}" for i in range(1, 12)) + ("F_V2",)
           + ("P_J280", "P_J269", "P_J300", "P_J256", "P_J289", "P_J415", "P_J302", "P_J306", "P_J307", "P_J317",
              "P_J14", "P_J422"))


def cache_dir() -> Path:
    return Path(os.environ.get("PINNEAPPLE_BATADAL_CACHE") or Path.home() / ".cache" / "pinneapple" / "batadal")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dest: str | Path | None = None, names: Sequence[str] | None = None) -> Path:
    """Download the BATADAL files that are not in ``dest`` yet and verify their SHA-256; returns the directory.
    Nothing is downloaded at import time and nothing is committed to the repository."""
    dest = Path(dest) if dest is not None else cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name in names or FILES:
        target = dest / name
        digest, _size = FILES[name]
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(target)
        if _sha256(target) != digest:
            target.unlink()
            raise ValueError(f"{name}: SHA-256 differs from the recorded one, file removed; the host may have changed it")
    return dest


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Read a BATADAL csv into float columns by name, plus ``hour`` (hour of day) and ``att_flag`` (1 attack, 0 clean,
    -999 unlabelled; all 0 in dataset 03)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    body = [r for r in rows[1:] if r]
    out: dict[str, np.ndarray] = {}
    for j, h in enumerate(head):
        if h == "DATETIME":
            continue
        out[h] = np.array([float(r[j]) for r in body])
    stamps = [datetime.datetime.strptime(r[0].strip(), "%d/%m/%y %H") for r in body]
    out["hour"] = np.array([t.hour for t in stamps], dtype=float)
    out["att_flag"] = out.pop("ATT_FLAG")
    return out


def episodes(flag: np.ndarray) -> list[tuple[int, int]]:
    """Runs of ``flag == 1`` as ``(start, end)`` with ``end`` exclusive."""
    on = np.asarray(flag) == 1
    edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(int)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


@dataclass
class TankBalance:
    """Storage balance of one tank at an hourly step, with unmeasured demand.

    ``area * (level[t] - level[t-1])`` (m^3) is explained by the flows at ``t-1`` (m^3/h, from L/s) with identified
    coefficients, plus a demand-by-hour-of-day profile (24 constants, the unmeasured consumption and leakage as the
    attack-free year averaged it). The residual is what is left: a level sensor rewritten, or a pump or valve flow
    that disagrees with what the tank saw. ``coef_`` has one coefficient per flow (about +1 for an inflow, about -1
    for an outflow when the tank sees all of it) then the 24 hourly terms. Causal: the residual at ``t`` needs
    readings up to ``t`` only."""
    level: str
    inflows: Sequence[str]
    outflows: Sequence[str] = ()
    area: float = 1.0
    name: str = ""
    coef_: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name or f"balance[{self.level}]"

    @property
    def flows(self) -> tuple[str, ...]:
        return (*self.inflows, *self.outflows)

    def _design(self, D: Data) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(D[self.level], dtype=float)
        dv = np.full_like(h, np.nan)
        dv[1:] = self.area * np.diff(h)                                 # volume change over the last hour, m^3
        n = len(h)
        X = np.zeros((n, len(self.flows) + 24))
        for j, k in enumerate(self.flows):
            X[1:, j] = FLOW_TO_M3_PER_H * np.asarray(D[k], dtype=float)[:-1]
        hours = np.asarray(D["hour"], dtype=int)[:-1] % 24
        X[np.arange(1, n), len(self.flows) + hours] = 1.0
        return X, dv

    def fit(self, D: Data) -> TankBalance:
        X, dv = self._design(D)
        ok = np.isfinite(dv)
        self.coef_ = np.linalg.lstsq(X[ok], dv[ok], rcond=None)[0]
        return self

    def residual(self, D: Data) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit() first")
        X, dv = self._design(D)
        return dv - X @ self.coef_


@dataclass
class PumpLaw:
    """Head curve of a pump while it runs: ``q^2 = a + b * (discharge - suction pressure)`` (:class:`FlowLaw` with
    the status column as a mask). The residual is NaN while the pump is off, so a CUSUM leaves it untouched; fitting
    uses the running hours only."""
    flow: str
    status: str
    up: str
    down: str
    name: str = ""
    _law: FlowLaw = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name or f"law[{self.flow}]"
        self._law = FlowLaw(self.flow, self.up, self.down, exponent=0.5)

    def _masked(self, D: Data) -> dict[str, np.ndarray]:
        on = np.asarray(D[self.status], dtype=float) > 0.5
        return {self.flow: np.where(on, np.asarray(D[self.flow], dtype=float), np.nan),
                self.up: D[self.up], self.down: D[self.down]}

    @property
    def coef(self) -> tuple[float, float] | None:
        return self._law.coef

    def fit(self, D: Data) -> PumpLaw:
        self._law.fit(self._masked(D))
        return self

    def residual(self, D: Data) -> np.ndarray:
        return self._law.residual(self._masked(D))


def ctown_detector(k: float = 0.5, h: float = 8.0) -> PhysicsResidualDetector:
    """The grey-box twin of C-Town: seven tank balances and the head curves of the two pumps that have pressure
    sensors at both ends and run in the attack-free year (PU2 and PU10; PU5, PU6 and PU9 have sensors but are off
    nearly all year, so their curves cannot be identified)."""
    comps: list[Any] = [TankBalance(f"L_{t}", ins, outs, area=TANK_AREA[t], name=f"balance[{t}]")
                        for t, (ins, outs) in TANK_FLOWS.items()]
    comps += [PumpLaw("F_PU2", "S_PU2", "P_J269", "P_J280", name="law[PU2]"),
              PumpLaw("F_PU10", "S_PU10", "P_J317", "P_J307", name="law[PU10]")]
    return PhysicsResidualDetector(components=comps, k=k, h=h)


def ttd_score(alarms: Sequence[int], eps: Sequence[tuple[int, int]]) -> float:
    """Time-to-detection score ``S_TTD = 1 - mean_i(TTD_i / dT_i)``: ``dT_i`` the length of attack ``i`` and ``TTD_i``
    the delay of the first alarm inside it, or ``dT_i`` when it is never detected (1: every attack caught at once,
    0: none caught). Written from the BATADAL rules as summarised in the literature; the official rules document
    could not be read when this was written, so check the normalisation against it before quoting a number."""
    a = np.asarray(sorted(alarms), dtype=int)
    ratios = []
    for s, e in eps:
        hit = a[(a >= s) & (a < e)]
        ratios.append(((hit[0] - s) if len(hit) else (e - s)) / (e - s))
    return float(1.0 - np.mean(ratios))


def _flags(alarms: Sequence[int], n: int, hold: int) -> np.ndarray:
    f = np.zeros(n, dtype=bool)
    for t in alarms:
        f[t:t + hold] = True
    return f


def classification_score(alarms: Sequence[int], eps: Sequence[tuple[int, int]], n: int, hold: int = 12,
                         clean_alarms: Sequence[int] | None = None, n_clean: int | None = None,
                         grace: int = 6) -> float:
    """``S_CLF = (TPR + TNR) / 2`` per hour, with a hold: each alarm flags the next ``hold`` hours as attack
    (``hold`` is a choice of this module: alarms are points in time, the competition scored a flag per hour).

    TPR is over the hours of the labelled attacks ``eps`` in a trace of ``n`` hours. TNR is over the attack-free
    trace of ``n_clean`` hours with alarms ``clean_alarms`` when given. Without it, TNR is taken over the hours of the
    same trace outside the episodes and their ``grace`` hours, which treats every unlabelled hour as clean: dataset 04
    has no hour labelled clean and may hold undiscovered attacks, so this is a *floor* on the TNR."""
    f = _flags(alarms, n, hold)
    hours = np.concatenate([np.arange(s, e) for s, e in eps]) if eps else np.array([], dtype=int)
    tpr = float(f[hours].mean()) if len(hours) else float("nan")
    if clean_alarms is not None:
        tnr = 1.0 - float(_flags(clean_alarms, int(n_clean), hold).mean())
    else:
        outside = np.ones(n, dtype=bool)
        for s, e in eps:
            outside[s:min(e + grace, n)] = False
        tnr = 1.0 - float(f[outside].mean())
    return 0.5 * (tpr + tnr)


def batadal_score(alarms_attack: Sequence[int], alarms_clean: Sequence[int], eps: Sequence[tuple[int, int]],
                  n_attack: int, n_clean: int, hold: int = 12, grace: int = 6) -> dict[str, float]:
    """``S = (S_TTD + S_CLF) / 2``, the equal-weight blend the competition used, with its parts, twice: ``S`` takes
    the TNR from a held-out attack-free trace; ``S_floor`` from the unlabelled hours of the attacked trace, which
    exposes a detector that alarms constantly on data unlike its training year. Quote them together."""
    ttd = ttd_score(alarms_attack, eps)
    clf = classification_score(alarms_attack, eps, n_attack, hold, alarms_clean, n_clean, grace)
    floor = classification_score(alarms_attack, eps, n_attack, hold, grace=grace)
    return {"S": 0.5 * (ttd + clf), "S_TTD": ttd, "S_CLF": clf, "S_floor": 0.5 * (ttd + floor), "S_CLF_floor": floor}


def run_benchmark(data_dir: str | Path | None = None, false_alarms_per_day: float = 0.25, grace: int = 6,
                  detectors: Mapping[str, Any] | None = None, hold: int = 12) -> dict[str, Any]:
    """Fit every detector on the first half of the attack-free year, calibrate its threshold to
    ``false_alarms_per_day`` on the next quarter, then report:

    * ``clean_false_alarms_per_day``: alarms on the last quarter, attack-free and used for neither (the false-alarm
      rate to trust);
    * on dataset 04: detection per labelled episode, median delay (hours), the episode AUC, ``S_TTD`` / ``S_CLF`` /
      ``S`` and the conservative ``S_floor`` (:func:`batadal_score`), and ``alarms_outside_per_day``: alarms
      outside the labelled episodes (and their ``grace`` hours), an *upper bound* on false alarms because the
      unlabelled rows may hold undiscovered attacks.

    The three baselines of :mod:`~pinneapple_security.cps_bench` run on the same data and alarm rule."""
    from .cps_bench import ARResidualDetector, IsolationForestDetector, PCAResidualDetector
    d = fetch(data_dir, ["BATADAL_dataset03.csv", "BATADAL_dataset04.csv"])
    clean, att = load(d / "BATADAL_dataset03.csv"), load(d / "BATADAL_dataset04.csv")
    n = len(clean["hour"])
    cut = lambda D, a, b: {k: v[a:b] for k, v in D.items()}               # noqa: E731
    fit, calib, test = cut(clean, 0, n // 2), cut(clean, n // 2, 3 * n // 4), cut(clean, 3 * n // 4, n)
    eps = episodes(att["att_flag"])
    if detectors is None:
        detectors = {"physics": ctown_detector(),
                     "ar_twin": ARResidualDetector(sensors=SENSORS),
                     "pca": PCAResidualDetector(sensors=SENSORS)}
        try:
            import sklearn  # noqa: F401
            detectors["iforest"] = IsolationForestDetector(sensors=SENSORS)
        except ImportError:
            pass
    out: dict[str, Any] = {"config": {"false_alarms_per_day": false_alarms_per_day, "grace": grace, "hold": hold,
                                      "fit_hours": len(fit["hour"]), "calibration_hours": len(calib["hour"]),
                                      "clean_test_hours": len(test["hour"]), "attack_hours": len(att["hour"]),
                                      "episodes": eps}, "results": {}}
    for name, det in detectors.items():
        det.fit(fit)
        det.calibrate(calib, false_alarms_per_day, samples_per_day=SAMPLES_PER_DAY)
        res_clean = det.detect(test)
        res_att = det.detect(att)
        m = episode_metrics(res_att["alarms"], eps, len(att["hour"]), samples_per_day=SAMPLES_PER_DAY, grace=grace)
        row = {"threshold": det.h,
               "clean_false_alarms_per_day": len(res_clean["alarms"]) / (len(test["hour"]) / SAMPLES_PER_DAY),
               "detected": m["detected"], "episodes": m["episodes"], "median_delay_h": m["median_delay"],
               "delays_h": m["delays"], "alarms_outside_per_day": m["false_alarms_per_day"],
               "auc": window_auc(det.score(att), eps, det.score(test))}
        row.update(batadal_score(res_att["alarms"], res_clean["alarms"], eps, len(att["hour"]), len(test["hour"]),
                                 hold, grace))
        out["results"][name] = row
    return out
