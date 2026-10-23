"""Antipodal Vivaldi antenna, 2-4 GHz, 10 dBi target: limits, an open EMerge model, checks, surrogates, design loop.

    python examples/use_cases/antennas/vivaldi_2to4ghz.py --quick            # 3 designs x 2 frequencies (minutes)
    python examples/use_cases/antennas/vivaldi_2to4ghz.py                    # 16 designs x 5 frequencies + fine runs
    python examples/use_cases/antennas/vivaldi_2to4ghz.py --package <dir>    # reproduce an external study package

The work is done by pinneapple_simulation.electromagnetics.antenna.study; this script picks the mode and records the
experiment. EMerge (``pip install emerge``) runs one frequency per process with SuperLU; without it only the limits
are computed. See docs/guides/antenna_design.md.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

from pinneapple_lab import experiment  # noqa: E402
from pinneapple_simulation.electromagnetics.antenna import study  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {
    "quick": dict(freqs=(2.0, 3.5), n_designs=3, n_fine=1, al_rounds=1, al_per_round=1, budget=200, timeout=1800),
    "full": dict(freqs=(2.0, 2.5, 3.0, 3.5, 4.0), n_designs=16, n_fine=4, al_rounds=2, al_per_round=2, budget=800,
                 timeout=3600),
}


def record(ex, s):
    """Metrics and checks of the summary in the experiment manifest (measured values only)."""
    runs = s.get("runs_total") or s.get("rerun", {}).get("runs")
    if isinstance(runs, dict):
        ex.log_metric("emerge_processes", runs["processes"], "", "EMerge processes, one frequency each")
        if runs["wall_s_median_ok"] is not None:
            ex.log_metric("emerge_wall_s_median", runs["wall_s_median_ok"], "s", "median wall time per process")
            ex.log_metric("emerge_max_rss_mb", runs["max_rss_mb"], "MB", "peak RSS of a child (os.wait4)")
    for target, v in s.get("lodo", {}).items():
        for model, e in v["rmse"].items():
            ex.log_metric(f"lodo_rmse_{target}_{model}", e, "dB", f"leave-one-design-out over {v['n']} rows")
    chk = s.get("checks_coarse") or s.get("checks")
    if chk:
        ex.check("efficiency_le_1_every_row", "efficiency_le_1" not in chk["failed"],
                 detail=json.dumps(chk["failed"]), kind="physics")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="3 designs x 2 frequencies")
    ap.add_argument("--package", help="folder of an external study package (results table + CLI script)")
    ap.add_argument("--rerun", default=None, help="package design tag to re-run ('none' to skip; default: first)")
    ap.add_argument("--rerun-freqs", type=int, default=1, help="how many of its frequencies to re-run")
    ap.add_argument("--interpolated", action="store_true", help="the package S11 comes from a vector fit")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per EMerge process")
    ap.add_argument("--out", help="output folder (default: outputs/<mode> next to this script)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mode = "package" if a.package else ("quick" if a.quick else "full")
    out = a.out or os.path.join(HERE, "outputs", mode)
    with experiment("vivaldi_2to4ghz", title="Antipodal Vivaldi 2-4 GHz, 10 dBi target", domain="electromagnetics",
                    methods=["FEM (EMerge)", "end-fire limits", "Gaussian process", "NSGA-II", "active learning"],
                    params={"mode": mode}, out_dir=out, register=False) as ex:
        if a.package:
            s = study.package_study(a.package, out, rerun=a.rerun, rerun_freqs=a.rerun_freqs,
                                    interpolated=a.interpolated, timeout=a.timeout or 3600)
        else:
            cfg = dict(MODES[mode], timeout=a.timeout or MODES[mode]["timeout"])
            s = study.open_study(out, seed=a.seed, **cfg)
        print(json.dumps({k: s[k] for k in s if k in ("limits", "checks", "checks_coarse", "lodo", "runs_total",
                                                      "multi_fidelity", "active_learning", "wall_s")},
                         indent=1, default=float)[:6000])
        record(ex, s)


if __name__ == "__main__":
    main()
