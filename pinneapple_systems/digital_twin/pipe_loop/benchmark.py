"""Controlled test cases with a known answer, and the confusion matrix of the diagnosis.

``make_cases`` draws scenarios of four kinds (seeded): healthy operation, a partial obstruction in one pipe (process
changed), a drifting or noisy sensor (sensor wrong), the plant driven outside the twin's training envelope (model
outside its domain). Fault onset, size, location and the operating history are random per case. The diagnoser is
calibrated on separate healthy runs (other seeds), then every case is diagnosed at its end; the matrix counts true
kind (rows) against diagnosed kind (columns), and the location / sensor / input is scored separately.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .diagnose import Diagnoser, Surrogate
from .plant import SENSORS, SIGMA, Loop, Scenario, run_plant

KINDS = ("normal", "process", "sensor", "domain")


def make_cases(n_per_kind: int = 20, seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    out: list[Scenario] = []
    for i in range(n_per_kind):
        base = 10_000 + 100 * i + seed * 7
        onset = float(rng.uniform(900.0, 2400.0))
        out.append(Scenario(f"normal_{i}", "normal", base))
        out.append(Scenario(f"process_{i}", "process", base + 1, onset_s=onset, target=str(rng.choice(["A", "B", "C"])),
                            magnitude=float(rng.uniform(1.0, 12.0))))
        j = int(rng.integers(len(SENSORS)))
        if rng.random() < 0.6:
            mag = float(rng.choice([-1, 1]) * rng.uniform(2.0, 12.0) * SIGMA[j])
            out.append(Scenario(f"sensor_{i}", "sensor", base + 2, onset_s=onset, target=SENSORS[j], magnitude=mag,
                                mode="drift"))
        else:
            out.append(Scenario(f"sensor_{i}", "sensor", base + 2, onset_s=onset, target=SENSORS[j],
                                magnitude=float(rng.uniform(2.0, 8.0)), mode="noise"))
        tgt, lo, hi = [("n", 1.03, 1.20), ("u_C", 0.12, 0.33), ("T_in", 38.0, 60.0),
                       ("Q_heat_kW", 950.0, 1300.0)][int(rng.integers(4))]
        out.append(Scenario(f"domain_{i}", "domain", base + 3, onset_s=onset, target=tgt,
                            magnitude=float(rng.uniform(lo, hi))))
    return out


def _diag_case(args):
    diag, sc = args
    t0 = time.perf_counter()
    run = run_plant(diag.loop, sc)
    d = diag.diagnose(run)
    t = run["t"]
    early = [k for k, ti in zip(d["kinds"], t, strict=True) if ti < sc.onset_s and k != "normal"] \
        if sc.kind != "normal" else [k for k in d["kinds"] if k != "normal"]
    after = [ti for k, ti in zip(d["kinds"], t, strict=True) if ti >= sc.onset_s and k != "normal"]
    return {"name": sc.name, "truth": sc.truth(), "kind": d["kind"], "target": d.get("target", ""),
            "mode": d.get("mode", ""), "size": d.get("size"), "size_std": d.get("size_std"),
            "action": d["action"], "first_alarm_s": d["first_alarm_s"],
            "false_alarm_steps": len(early),
            "delay_s": (after[0] - sc.onset_s) if (after and sc.kind != "normal") else None,
            "statistics_final": d["statistics_final"], "newton_max_residual_m": run["newton_max_residual_m"],
            "seconds": time.perf_counter() - t0}


def run_benchmark(loop: Loop | None = None, *, n_per_kind: int = 20, n_calibration: int = 12, seed: int = 0,
                  n_train: int = 120, workers: int = 4, log=None) -> dict[str, Any]:
    """Fit the twin's fast model, calibrate the diagnoser on healthy runs, diagnose every controlled case and
    return the confusion matrix, isolation scores, false alarms, delays and timings."""
    loop = loop or Loop()
    t0 = time.perf_counter()
    sur = Surrogate(n_train=n_train, seed=seed).fit(loop)
    t_fit = time.perf_counter() - t0
    diag = Diagnoser(loop, sur)
    t1 = time.perf_counter()
    cal = [run_plant(loop, Scenario(f"cal_{i}", "normal", 900_000 + i)) for i in range(n_calibration)]
    diag.calibrate(cal)
    t_cal = time.perf_counter() - t1
    if log:
        log(f"fast model fitted in {t_fit:.1f} s, thresholds {diag.thresholds} from {n_calibration} healthy runs")
    cases = make_cases(n_per_kind, seed)
    jobs = [(diag, sc) for sc in cases]
    t2 = time.perf_counter()
    if workers > 1:
        import multiprocessing as mp

        from pinneapple_design.thermal.board_cooling.simulate import _one_thread
        with mp.get_context("fork").Pool(workers, initializer=_one_thread) as pool:
            rows = pool.map(_diag_case, jobs)
    else:
        rows = [_diag_case(j) for j in jobs]
    t_cases = time.perf_counter() - t2
    M = np.zeros((len(KINDS), len(KINDS)), int)
    for r in rows:
        M[KINDS.index(r["truth"]["kind"]), KINDS.index(r["kind"])] += 1
    iso = {}
    for k in ("process", "sensor", "domain"):
        rk = [r for r in rows if r["truth"]["kind"] == k and r["kind"] == k]
        ok = [r for r in rk if r["target"] == r["truth"]["target"]]
        iso[k] = {"correct_kind": len(rk), "correct_target": len(ok)}
    sens_mode = [r for r in rows if r["truth"]["kind"] == "sensor" and r["kind"] == "sensor"
                 and r["target"] == r["truth"]["target"]]
    iso["sensor"]["correct_mode"] = sum(r["mode"] == r["truth"]["mode"] for r in sens_mode)
    proc = [r for r in rows if r["truth"]["kind"] == "process" and r["kind"] == "process"
            and r["target"] == r["truth"]["target"]]
    k_err = [abs(r["size"] - r["truth"]["magnitude"]) for r in proc]
    k_in = [abs(r["size"] - r["truth"]["magnitude"]) <= 2 * r["size_std"] for r in proc]
    delays = {k: [r["delay_s"] for r in rows if r["truth"]["kind"] == k and r["delay_s"] is not None]
              for k in KINDS[1:]}
    n_cases = len(rows)
    return {"kinds": list(KINDS), "confusion": M.tolist(), "accuracy": float(np.trace(M) / n_cases),
            "n_cases": n_cases, "isolation": iso,
            "kalman_k_abs_error": {"mean": float(np.mean(k_err)) if k_err else None,
                                   "max": float(np.max(k_err)) if k_err else None,
                                   "inside_2std": int(sum(k_in)), "n": len(k_in)},
            "cases_with_false_alarm": sum(r["false_alarm_steps"] > 0 for r in rows),
            "false_alarm_steps": int(sum(r["false_alarm_steps"] for r in rows)),
            "delay_s_median": {k: (float(np.median(v)) if v else None) for k, v in delays.items()},
            "delay_s_max": {k: (float(np.max(v)) if v else None) for k, v in delays.items()},
            "thresholds": diag.thresholds, "n_calibration": n_calibration,
            "newton_max_residual_m": float(max(r["newton_max_residual_m"] for r in rows)),
            "seconds": {"fit_fast_model": t_fit, "calibration": t_cal, "cases": t_cases,
                        "per_case": float(np.median([r["seconds"] for r in rows]))},
            "rows": rows, "diagnoser": diag}
