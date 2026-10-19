"""Run the public benchmarks shown on arena.pinneapple.org and store their results (results/<id>.json, .md).

Every number the public page shows comes from these files, written by an actual run: rerun this script on the
deployment hardware to refresh them (python run_public_benchmarks.py [--budget 10] [--seeds 3] [ids...]).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinneapple_analysis.surrogate import examples as E
from pinneapple_arena.regimes import run_regime_arena
from pinneapple_core.jsonable import to_jsonable

OUT = Path(__file__).parent / "results"
BENCHMARKS = {
    "airfoil_rans": {"title": "Airfoil Cl and Cd from OpenFOAM RANS (CST shapes, Re 4e6)", "data": E.airfoil_rans,
                     "baselines": {}},
    "burgers_fields": {"title": "Viscous Burgers fields u(x, t) across viscosity", "data": E.burgers_fields,
                       "baselines": {"fdm_201": E.burgers_fdm_baseline(nx=201)}},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--budget", type=float, default=10.0, help="wall-clock seconds per training run")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    for bid in a.ids:
        b = BENCHMARKS[bid]
        res = run_regime_arena(b["data"](), budget=a.budget, seeds=a.seeds, baselines=b["baselines"],
                               log=lambda s, bid=bid: print(bid, s, flush=True))
        doc = {"id": bid, "title": b["title"], **to_jsonable(res.to_dict())}
        (OUT / f"{bid}.json").write_text(json.dumps(doc, indent=1))
        (OUT / f"{bid}.md").write_text(res.to_markdown(b["title"]))
        print(bid, "done in", round(res.wall_s), "s", flush=True)


if __name__ == "__main__":
    main()
