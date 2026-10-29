"""Potential-based surrogate against a direct multi-output surrogate on CO2 near its critical point (issue #928).

Three configurations, each over five seeds, every model with the same data, loss, optimiser and number of steps:

* ``all_properties``: the training samples carry p, s, u, c_v, c_p or w (mixed data);
* ``without_cp``: the same without c_p samples (c_p is still scored: the potential predicts it from the other
  properties, the direct model's c_p output is never trained);
* ``all_properties_3x``: the first configuration with three times the number of steps.

Writes ``results/<config>.json`` (raw runs and the summary with intervals). Needs CoolProp. About 1.5 h on one CPU;
``--smoke`` runs one seed with a few steps.

    python examples/potential_co2/01_run_study.py [--smoke] [--only all_properties]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinneapple_core.jsonable import to_jsonable
from pinneapple_physics.potential import co2_study as cs

HERE = Path(__file__).resolve().parent

CONFIGS = {
    "all_properties": {"props": cs.PROPERTIES, "epochs": 3000},
    "without_cp": {"props": ("p", "s", "u", "cv", "w"), "epochs": 3000},
    "all_properties_3x": {"props": cs.PROPERTIES, "epochs": 9000},
}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    out_dir = HERE / ("results_smoke" if a.smoke else "results")
    out_dir.mkdir(exist_ok=True)
    grids = cs.default_grids(7 if a.smoke else 21)
    done = {}
    for name, cfg in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        kw = dict(n_per_prop=200, hidden=32, layers=3, lr=1e-3, **cfg)
        seeds = range(5)
        if a.smoke:
            kw.update(n_per_prop=20, epochs=20)
            seeds = range(1)
        res = cs.run_study(seeds, grids=grids, **kw)
        (out_dir / f"{name}.json").write_text(json.dumps(to_jsonable(res), indent=1))
        s = res["summary"]["models"]
        print(name, {m: round(v["interp"]["nrmse_mean"]["mean"], 4) for m, v in s.items()}, flush=True)
        done[name] = res
    return done


if __name__ == "__main__":
    main()
