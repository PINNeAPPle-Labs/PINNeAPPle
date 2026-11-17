"""The Digital Twin Library: nine systems, each one closing its own conservation laws.

FPSO and subsea production, a blast furnace, a chemical synthesis loop, a reusable booster, a lunar
base, a gas pipeline network, a compression station, a green hydrogen plant and a seawater
desalination train. Each is a reference model: SI parameters with declared ranges, a
machine-readable result that carries two independent routes to every conservation law, invariants
that must hold for any parameter set, and its own list of what it does not verify.

Run: ``python examples/digital_twin_library/01_reference_systems.py [--smoke] [--out DIR]``
Outputs in ``examples/digital_twin_library/_out/``.
"""
from __future__ import annotations

import argparse
import os

import pinneapple_lab
from pinneapple_systems.digital_twin.library import TWINS, catalogue, compare, run

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with pinneapple_lab.experiment(
            "examples/digital_twin_library", title="Digital Twin Library: nine reference systems",
            description="FPSO and subsea production, blast furnace, chemical synthesis loop, reusable "
                        "booster, lunar base, gas pipeline network, compression station, green hydrogen "
                        "plant and desalination train, each a 0D/1D "
                        "reference model with its conservation laws computed twice by independent routes "
                        "and its own stated limits.",
            domain="digital twins", methods=["material balance", "stoichiometry", "thermodynamic equilibrium",
                                             "point-mass flight", "radiative balance",
                                             "isothermal compressible flow", "polytropic compression",
                                             "electrolysis stoichiometry", "osmotic pressure"],
            modules=["pinneapple_systems.digital_twin.library"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/digital_twin_library.md",
                   "script": "examples/digital_twin_library/01_reference_systems.py"}) as ex:
        print(compare())
        for entry in catalogue():
            print(f"  {entry.slug:<18s} {entry.fidelity:<7s} {entry.status}")

        slugs = sorted(TWINS) if not a.smoke else sorted(TWINS)[:2]
        worst = 0.0
        for slug in slugs:
            result = run(slug)
            print()
            print(result.report())
            # The runs go to _out/ (git-ignored): the manifest carries their numbers, not the files.
            result.save(os.path.join(a.out, f"{slug}.json"))
            for balance in result.balances:
                worst = max(worst, abs(balance.residual))
            ex.check(f"{slug}: every conservation law closes", result.balanced,
                     detail="; ".join(f"{b.name} {b.residual:+.1e}" for b in result.balances), kind="physics")
            ex.check(f"{slug}: every invariant holds", result.invariants_hold,
                     detail=", ".join(c.name for c in result.checks), kind="physics")
        ex.log_metric("worst_balance_residual", worst, "relative", f"{len(slugs)} systems, all their balances")
        ex.log_metric("systems_in_the_catalogue", float(len(TWINS)), "systems",
                      "every system registered in the library")
        print(f"\nworst balance residual over the library: {worst:.2e}")


if __name__ == "__main__":
    main()
