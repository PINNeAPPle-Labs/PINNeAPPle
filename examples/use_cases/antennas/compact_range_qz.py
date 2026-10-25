"""Compact range quiet zone by physical optics: validation, feed gain x reflector size (x rolled edge) sweep, a
surrogate with held-out error and the trade-off map. Generic example parameters (3 GHz, F = 1.5 m, a 1.2 m square
section with a superellipse junction, a 0.6 m quiet zone at 3 m).

    python examples/use_cases/antennas/compact_range_qz.py --quick     # 3 x 3 designs, coarser mesh (minutes)
    python examples/use_cases/antennas/compact_range_qz.py             # 7 x 6 designs x 2 rolled edges

Every number comes from pinneapple_simulation.electromagnetics.reflector; see docs/guides/compact_range.md.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

from pinneapple_lab import experiment  # noqa: E402
from pinneapple_simulation.electromagnetics.reflector import CompactRange, evaluate  # noqa: E402
from pinneapple_simulation.electromagnetics.reflector.study import (  # noqa: E402
    qz_figure,
    qz_study,
    validate,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "outputs", "compact_range_quick" if a.quick else "compact_range")
    os.makedirs(out, exist_ok=True)
    base = CompactRange()                                       # the generic example
    if a.quick:
        base = replace(base, mesh_per_wavelength=6.0, qz_step_per_wavelength=4.0)
        gains, sides, rolls = [10.0, 14.0, 18.0], [1.0, 1.2, 1.4], [base.reflector.roll]
    else:
        gains, sides, rolls = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], [0.9, 1.05, 1.2, 1.35, 1.5, 1.65], \
            [0.0, base.reflector.roll]
    with experiment("compact_range_qz", title="Compact range quiet zone: feed gain x reflector size",
                    domain="electromagnetics", methods=["physical optics", "Gaussian process"],
                    params={"quick": a.quick, "design": base.to_dict()}, out_dir=out, register=False) as ex:
        val = validate(base, mesh_densities=(6, 8) if a.quick else (8, 12, 16),
                       rolls_wl=(3.0,) if a.quick else (3.0, 6.0))
        ref = evaluate(base)
        qz_figure(base, os.path.join(out, "qz_reference.png"))
        st = qz_study(base, gains, sides, rolls, out, mark=(base.feed_gain_dbi, 2 * base.reflector.half_x,
                                                           "reference design"))
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump({"validation": val, "reference": ref, "study": st["summary"]}, fh, indent=1, default=float)
        ex.log_metric("po_disk_max_rel_error", val["disk"][-1]["max_rel_error"], "",
                      f"PO disk on axis vs closed form, lambda/{val['disk'][-1]['per_wavelength']:g} mesh")
        ex.log_metric("po_vs_go_rms_dB", val["go"][-1]["rms_dB"], "dB", "smooth PO QZ amplitude vs GO taper")
        ex.log_metric("reference_ripple_pp_dB", ref["ripple_pp_dB"], "dB", "PO, reference design")
        ex.log_metric("reference_usable_diameter_m", ref["usable_diameter"], "m", "within +-0.5 dB / +-5 deg")
        ex.log_metric("evaluation_s_median", st["summary"]["seconds_median"], "s", "PO mesh + QZ field per design")
        ex.check("po_disk_closed_form", val["disk"][-1]["max_rel_error"] < 0.01, value=val["disk"][-1]["max_rel_error"],
                 max=0.01, kind="reference")


if __name__ == "__main__":
    main()
