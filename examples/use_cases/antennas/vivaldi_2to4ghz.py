"""Antipodal Vivaldi antenna, 2-4 GHz, 10 dBi target: limits, an open EMerge model, checks, surrogates, design loop.

    python examples/use_cases/antennas/vivaldi_2to4ghz.py --quick            # 3 designs x 2 frequencies (minutes)
    python examples/use_cases/antennas/vivaldi_2to4ghz.py                    # 16 designs x 5 frequencies + fine runs
    python examples/use_cases/antennas/vivaldi_2to4ghz.py --package <dir>    # reproduce an external study package
    python examples/use_cases/antennas/vivaldi_2to4ghz.py --3d               # 3-D of the base and the best design

``--3d`` (also run at the end of the full mode) re-solves the base design (``AVAParams()`` defaults) and the best
design of ``results/summary.json`` with their near field and writes results/3d/: glTF / USD per frequency, two
self-contained viewers (frequency sweep; one period of E_x), Blender renders and GIF / MP4.

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


RESULTS = os.path.join(HERE, "results")


def run_3d(results_dir, a):
    """The base design and the best design of the study in ``results_dir``, in 3-D (results_dir/3d)."""
    designs = {"base": {}}
    p = os.path.join(results_dir, "summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            best = json.load(fh).get("best", {}).get("design")
        if best:
            designs["best"] = {k: v for k, v in best.items() if k != "res"}
    return study.render_3d(designs, os.path.join(results_dir, "3d"), render=not a.no_render)


def record_3d(ex, s):
    for name, r in s.items():
        if r["balloon_vs_table_max_abs_dB"] is not None:
            ex.check(f"{name}_balloon_matches_table", r["balloon_vs_table_max_abs_dB"] < 1e-6,
                     value=r["balloon_vs_table_max_abs_dB"], max=1e-6, kind="consistency",
                     detail="peak directivity of the 3-D balloon vs D_peak_dBi of the table")
        ex.log_metric(f"{name}_3d_emerge_wall_s_median", r["runs"]["wall_s_median_ok"] or 0.0, "s",
                      "median EMerge process time with the near field")
        if "render_s" in r:
            ex.log_metric(f"{name}_blender_render_s", r["render_s"], "s", "all Blender stills and animations")


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
    ap.add_argument("--3d", dest="three_d", action="store_true", help="only the 3-D outputs (results/3d)")
    ap.add_argument("--no-render", action="store_true", help="3-D without the Blender renders")
    a = ap.parse_args()
    mode = "package" if a.package else ("3d" if a.three_d else ("quick" if a.quick else "full"))
    out = a.out or os.path.join(HERE, "outputs", mode)
    if a.three_d:                                   # the 3-D outputs sit next to the study they come from
        out = os.path.join(a.out or RESULTS, "3d")
    with experiment("vivaldi_2to4ghz", title="Antipodal Vivaldi 2-4 GHz, 10 dBi target", domain="electromagnetics",
                    methods=["FEM (EMerge)", "end-fire limits", "Gaussian process", "NSGA-II", "active learning"],
                    params={"mode": mode}, out_dir=out, register=False) as ex:
        if a.three_d:
            s = run_3d(os.path.dirname(out), a)
            record_3d(ex, s)
            return
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
        if mode == "full":
            record_3d(ex, run_3d(out, a))


if __name__ == "__main__":
    main()
