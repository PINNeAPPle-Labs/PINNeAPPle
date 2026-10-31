"""Compact-range design at 2 GHz: quiet-zone performance against feed gain, reflector size and focal length, a GP
surrogate with uncertainty, design maps, the Pareto front of QZ size against reflector size and the smallest
reflector meeting a spec, verified by physical optics. A generic range (no particular installation): F/D about 1,
a 3-wavelength rolled edge, a 1 m quiet zone at 3 m in front of the focal plane.

    python examples/use_cases/antennas/compact_range_2ghz.py --quick      # 3 x 3 designs, coarse mesh (minutes)
    python examples/use_cases/antennas/compact_range_2ghz.py              # 5 gains x 7 sizes x 3 focal lengths
    python examples/use_cases/antennas/compact_range_2ghz.py --template ~/my_range.json   # a config to edit
    python examples/use_cases/antennas/compact_range_2ghz.py --private ~/my_range.json --out ~/my_range_results

``run_private(config_path, out_dir)`` runs any geometry JSON (the template's keys) and refuses a configuration or an
output folder inside this repository: client geometry and results never land in a commit. Every number comes from
pinneapple_simulation.electromagnetics.reflector.design; see docs/guides/compact_range.md.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

from pinneapple_lab import experiment  # noqa: E402
from pinneapple_simulation.electromagnetics.reflector.design import (  # noqa: E402
    PO_LIMITS,
    from_config,
    generic_2ghz,
    outside,
    run_design_study,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

GENERIC = generic_2ghz()          # 2 GHz, F = 4 m, 3-lambda rolled edge; spec 1 m within +-0.5 dB / +-5 deg, -25 dB
QUICK = generic_2ghz(quick=True)


def study(cfg: dict, out: str, *, compare_887: bool = False, register: bool = False) -> dict:
    """The design study of ``cfg`` into ``out``, recorded as a Lab experiment."""
    c = from_config(cfg)
    with experiment("compact_range_2ghz", title="Compact range: QZ vs feed gain and reflector size (2 GHz)",
                    domain="electromagnetics", methods=["physical optics", "Gaussian process"],
                    params={k: v for k, v in cfg.items() if k != "name"}, out_dir=out, register=register) as ex:
        st = run_design_study(c["base"], c["spec"], c["grid"]["gains"], c["grid"]["sides"], out,
                              freqs_hz=c["grid"]["freqs_hz"], focals=c["grid"]["focals"], compare_887=compare_887,
                              checks=c["checks"], densities=c["densities"], workers=c["workers"], **c["optimise"])
        s = st["summary"]
        if compare_887:
            ex.log_metric("agreement_887_max_abs_diff", s["agreement_887"]["max_abs_diff_vs_887"], "",
                          "the #887 reference design solved again by qz_report against #887's numbers")
        opt = s["optimiser"]
        ex.log_metric("designs", s["n_designs"], "", "PO solves in the sweep (both polarisations each)")
        ex.log_metric("seconds_median", s["seconds_median"], "s", "PO mesh + QZ field, V and H, per design")
        u = s["surrogate"]["usable_diameter"]
        ex.log_metric("gp_usable_rmse_m", u["rmse"], "m", "5-fold held-out RMSE of the GP, usable QZ diameter")
        ex.log_metric("gp_usable_coverage95", u["coverage95"], "", "held-out fraction inside the GP 95% interval")
        if "power_balance" in s:
            ex.log_metric("power_ratio", s["power_balance"]["ratio"], "",
                          "power through the QZ plane / feed power intercepted by the reflector (PO)")
        if opt["found"]:
            ex.log_metric("smallest_side_m", opt["design"]["side"], "m", "optimiser, verified by PO")
            ex.log_metric("smallest_side_gain_dBi", opt["design"]["feed_gain_dBi"], "dBi", "its feed gain")
        ex.check("optimiser_verified_by_po", bool(opt["found"]), kind="reference")
    return st


def run_private(config_path: str, out_dir: str | None = None) -> dict:
    """Run a user-supplied geometry JSON (``--template`` writes one to edit) from outside the repository and write
    its results outside it (default: ``<config folder>/<config name>_results``). Raises ``ValueError`` when the
    configuration or the output folder is inside the repository."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not outside(config_path, REPO):
        raise ValueError(f"{config_path} is inside the repository; keep private geometry outside it")
    out = os.path.abspath(os.path.expanduser(out_dir)) if out_dir else \
        os.path.splitext(config_path)[0] + "_results"
    if not outside(out, REPO):
        raise ValueError(f"{out} is inside the repository; private results go outside it")
    with open(config_path) as fh:
        cfg = json.load(fh)
    os.makedirs(out, exist_ok=True)
    return study(cfg, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--private", metavar="CONFIG.json", help="run a geometry JSON kept outside the repository")
    ap.add_argument("--template", metavar="PATH.json", help="write the generic configuration to edit, then exit")
    a = ap.parse_args()
    if a.template:
        with open(a.template, "w") as fh:
            json.dump(GENERIC, fh, indent=1)
        print(f"wrote {a.template}; edit it and run --private {a.template}")
        return
    if a.private:
        st = run_private(a.private, a.out)
    else:
        out = a.out or os.path.join(HERE, "outputs" if a.quick else "results",
                                    "compact_range_2ghz_quick" if a.quick else "compact_range_2ghz")
        st = study(QUICK if a.quick else GENERIC, out, compare_887=not a.quick)
    o = st["summary"]["optimiser"]
    print("smallest reflector:", o["design"] if o["found"] else "none verified", "|", PO_LIMITS)


if __name__ == "__main__":
    main()
