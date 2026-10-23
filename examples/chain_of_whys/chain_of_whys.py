"""Chain of whys: four projects, four scales, the same reasoning (pinneapple_reason).

1. roof           flat tiles -> pitch (rain through the laps) -> ventilated ceiling (condensation on a clear night)
                  -> radiant barrier (warm ceiling on a sunny day)
2. footbridge     girders sized for strength -> deeper girders (deflection) -> lateral bracing (lateral-torsional
                  buckling) -> tuned mass damper (resonance with footfall)
3. power_board    MOSFET on a bare board -> heat sink (Joule heating) -> forced air
4. pumped_main    pump above the sump -> lower pump (cavitation) -> slow valve closure (water hammer)

Each case writes ``_out/<case>/`` (report.html, report.md, chain.json, trust.json, figures/). Every number in the
reports comes from the failure modes of the library; the inputs are the premises listed in each report.

Run: ``python examples/chain_of_whys/chain_of_whys.py [--case roof] [--lang en]``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_reason import cases, run_case, trust_report, write_report

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(cases.CASES), action="append")
    ap.add_argument("--lang", choices=("pt", "en"), default="pt")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()
    with pinneapple_lab.experiment(
            "examples/chain_of_whys", title="Chain of whys",
            description="Engineering reasoning from physics: naive concept -> computed failure mode -> countermeasure "
                        "-> check again, for a roof, a footbridge, a power board and a pumped main.",
            domain="design and geometry", methods=["closed-form checks", "FD beam", "3-D board FV", "MOC"],
            modules=["pinneapple_reason"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/chain_of_whys.md", "script": "examples/chain_of_whys/chain_of_whys.py"}) as ex:
        for name in a.case or list(cases.CASES):
            print(f"\n=== {name}")
            chain = run_case(name, progress=lambda m: print("  ", m))
            paths = write_report(chain, os.path.join(a.out, name), lang=a.lang)
            rep = trust_report(chain)
            print(chain.explain(a.lang))
            ex.log_metric(f"{name}_versions", len(chain.steps), "versions", "versions until every requirement passes")
            ex.log_metric(f"{name}_evaluations", chain.meta["evaluations"], "checks",
                          "failure-mode evaluations the engine ran")
            ex.check(f"{name}: every requirement passes on the final version", chain.resolved, kind="physics",
                     detail=" -> ".join(f"{m}:{c}" for _, m, c in chain.path()))
            for c in rep.checks:
                if c.name.startswith(("solver vs closed form", "known answer")):
                    ex.check(f"{name}: {c.name}", c.status == "supports", kind="verification", detail=c.source[:120])
            ex.check(f"{name}: trust report decision", rep.decisao == "APPROVED", detail=f"{rep.decisao} "
                     f"{rep.motivos}", kind="sanity")
            for k, p in paths.items():
                if k.startswith("fig_") or k in ("md", "chain", "trust"):
                    ex.add_file(str(p))


if __name__ == "__main__":
    main()
