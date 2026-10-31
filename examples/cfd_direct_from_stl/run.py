"""The campaign of docs/research/cfd_direct_from_stl.md: every method, case, Reynolds number and rung of the
pre-registered ladders, one fresh subprocess each, one at a time (timed runs never share the machine), with a
memory watchdog (14 GB) and a wall-time limit (4 h). Resumable: a finished run is not repeated.

    python examples/cfd_direct_from_stl/run.py run              # all runs (hours)
    python examples/cfd_direct_from_stl/run.py run --only C1    # one case
    python examples/cfd_direct_from_stl/run.py report           # verdicts, tables and figures from results/

A rung that exceeds the memory or the time is recorded with the reason, and the finer rungs of that ladder are not
run (each needs more of both).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_simulation.cfd_direct.runners import LADDERS  # noqa: E402

MEM_LIMIT_MB = 14_000
TIME_LIMIT_S = 4 * 3600
RE = {"C1": [None], "C2": [20, 50, 100], "C3": [100, 200, 400]}
METHODS = ("reference", "immersed", "lbm")


def key(method: str, case: str, re, n: int) -> str:
    return f"{method}_{case}{'' if re is None else f'_Re{re}'}_n{n}"


def _rss_mb(pid: int) -> float:
    """Resident memory of a process and its children (from /proc)."""
    total = 0.0
    try:
        out = subprocess.run(["ps", "-o", "rss=", "--ppid", str(pid), "-p", str(pid)], capture_output=True, text=True)
        total = sum(float(x) for x in out.stdout.split()) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return total


def run_one(method: str, case: str, re, n: int) -> dict:
    k = key(method, case, re, n)
    out = RESULTS / f"{k}.json"
    work = RESULTS / "work" / k
    work.mkdir(parents=True, exist_ok=True)
    kw = {} if re is None else {"Re": re}
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    cmd = [sys.executable, "-m", "pinneapple_simulation.cfd_direct.runners", method, case, str(n), "--out", str(out),
           "--work", str(work), "--kw", json.dumps(kw)]
    t0 = time.time()
    load0 = os.getloadavg()
    log = open(work / "log.txt", "w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    peak, why = 0.0, None
    while p.poll() is None:
        time.sleep(2)
        peak = max(peak, _rss_mb(p.pid))
        if peak > MEM_LIMIT_MB:
            why = f"stopped: memory above {MEM_LIMIT_MB} MB"
        elif time.time() - t0 > TIME_LIMIT_S:
            why = f"stopped: wall time above {TIME_LIMIT_S} s"
        if why:
            p.kill()
            p.wait()
            break
    log.close()
    if why is None and p.returncode == 0 and out.is_file():
        rec = json.loads(out.read_text())
        rec.update(ok=True, watchdog_peak_mb=peak, wall_s=time.time() - t0)
    else:
        tail = (work / "log.txt").read_text()[-1500:]
        rec = {"method": method, "case": case, "n": n, "Re": re, "ok": False, "why": why or f"error: {tail[-400:]}",
               "watchdog_peak_mb": peak, "wall_s": time.time() - t0, "load_before": list(load0)}
    out.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return rec


def run(only: str | None, methods: tuple[str, ...]) -> None:
    RESULTS.mkdir(exist_ok=True)
    for case in ("C1", "C3", "C2"):
        if only and case != only:
            continue
        for re in RE[case]:
            for method in methods:
                for n in LADDERS[method][case]:
                    k = key(method, case, re, n)
                    p = RESULTS / f"{k}.json"
                    if p.is_file():
                        rec = json.loads(p.read_text())
                        if not rec.get("ok", True) and "stopped" in rec.get("why", ""):
                            break                              # finer rungs need more memory and time
                        continue
                    print(time.strftime("%H:%M:%S"), k, flush=True)
                    rec = run_one(method, case, re, n)
                    msg = rec.get("why") if not rec["ok"] else {q: rec.get(q) for q in
                                                                 ("Cd", "Cl", "dp", "x1", "err_Cd", "err_x1")
                                                                 if q in rec}
                    print("   ", round(rec["wall_s"], 1), "s", msg, flush=True)
                    if not rec["ok"] and "stopped" in rec["why"]:
                        break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--only")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    if a.what == "run":
        run(a.only, tuple(a.methods.split(",")))
    else:
        from report import report
        report(RESULTS, HERE / "figures")


if __name__ == "__main__":
    main()
