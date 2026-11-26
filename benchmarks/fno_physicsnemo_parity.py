"""FNO-2D: PINNeAPPle against PhysicsNeMo, same weights and same problem (#2529).

Writes ``benchmarks/_out/fno_physicsnemo_parity.json``: the environment, the forward-parity measurement (same weights
transplanted from PhysicsNeMo into this library's ``FNO2d``, per grid), and the trained comparison against the exact
solution of the periodic advection-diffusion propagator — once in the parity configuration and once with each
library's own default architecture.

Everything measured here is library code (``pinneapple_neural.physicsnemo_bridge.fno_compare``); this file only runs
it and writes the record. Tolerances were declared before the run (``fno_compare.PARITY_TOLERANCE`` and, for the
trained rows, agreement between the two parity rows to within the reported seed-to-seed scatter).

    python -m benchmarks.fno_physicsnemo_parity            (from the repo root; about 20 min on 4 CPU cores)
    python -m benchmarks.fno_physicsnemo_parity --quick    (a minute: the forward parity only)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import torch

from pinneapple_neural.physicsnemo_bridge import fno_compare as fc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "fno_physicsnemo_parity.json")
GRIDS = (32, 64, 96)
SEEDS = (0, 1, 2)
CASE = dict(n=64, n_samples=256, n_train=192, seed=7, n_modes=6)
STEPS, BATCH, THREADS = 200, 16, 4
TRANSFER = (96,)
EVAL_AT = (0.25, 0.5, 0.75)


def environment() -> dict:
    import numpy
    import scipy

    env = {"pinneapple": _version("pinneapple"), "torch": torch.__version__, "numpy": numpy.__version__,
           "scipy": scipy.__version__, "python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "cpu_count": os.cpu_count(), "threads": THREADS,
           "precision": "float32", "device": "cpu", **fc.availability()}
    try:
        env["cpu"] = platform.processor() or "unknown"
    except Exception:                                    # noqa: BLE001 — cosmetic only
        env["cpu"] = "unknown"
    return env


def _version(name: str) -> str:
    try:
        return __import__(name).__version__
    except Exception:                                    # noqa: BLE001 — a missing package is not a failure here
        return "not importable"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="the forward parity only (about a minute)")
    ap.add_argument("--out", default=OUT, help="where the JSON goes")
    args = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    report: dict = {"schema": "pinneapple.physicsnemo_fno_parity/1",
                    "issue": 2529, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "environment": environment(),
                    "protocol": {"grids": list(GRIDS), "seeds": list(SEEDS), "case": CASE,
                                 "steps": STEPS, "batch_size": BATCH, "threads": THREADS,
                                 "transfer_grids": list(TRANSFER), "eval_at": list(EVAL_AT),
                                 "arch": fc.PARITY_ARCH.to_dict(),
                                 "tolerance": fc.PARITY_TOLERANCE,
                                 "problem": "u_t + c.grad u = kappa lap u on the periodic unit square, u0 -> u(t_end),"
                                            " exact Fourier-multiplier reference"}}
    if not report["environment"]["available"]:
        report["forward_parity"] = fc.fno2d_forward_parity(GRIDS[0], grids=GRIDS).to_dict()
        report["gradient_parity"] = fc.fno2d_gradient_parity(GRIDS[0]).to_dict()
        report["trained"] = None
        _write(args.out, report)
        print("physicsnemo is not installed: nothing to compare against (see available.reason)")
        return 0

    t0 = time.time()
    report["forward_parity"] = fc.fno2d_forward_parity(GRIDS[0], grids=GRIDS, seed=0).to_dict()
    report["gradient_parity"] = fc.fno2d_gradient_parity(GRIDS[0], seed=0).to_dict()
    print("forward parity:", {k: f"{v:.2e}" for k, v in report["forward_parity"]["relative_l2"].items()},
          f"({time.time() - t0:.1f}s)", flush=True)
    print("gradient parity:", f"{report['gradient_parity']['relative_l2']:.2e}", flush=True)
    if args.quick:
        _write(args.out, report)
        return 0

    case = fc.advdiff_case(**CASE)
    t0 = time.time()
    trained = fc.compare_trained(case, seeds=SEEDS, steps=STEPS, batch_size=BATCH, threads=THREADS,
                                 transfer_grids=TRANSFER, as_shipped=True, eval_at=EVAL_AT)
    report["trained"] = trained.to_dict()
    report["case"] = case.describe()
    report["timing_s"] = round(time.time() - t0, 1)
    for row, data in trained.rows.items():
        mean = data.get("rel_l2_mean", {}).get("mean")
        print(f"{row:28s} rel_l2 {mean:.4f}" if mean is not None else f"{row:28s}", flush=True)
    _write(args.out, report)
    return 0


def _write(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("written", path)


if __name__ == "__main__":
    raise SystemExit(main())
