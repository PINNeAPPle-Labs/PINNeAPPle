"""Text to simulation: four requests through the same compiler, ladder, decision and replay.

    python examples/intent_to_study/intent_to_study.py

Every number printed here comes from a closed form, a solver call, or a value the contract records with its source.
The script prints, for each request, the compiled contract, the ladder that ran, the checks, and the quantity of
interest with the decision that says whether it can be acted on. It then replays the first run from its files alone,
to show that the record is enough.
"""
from __future__ import annotations

import json
import os

import pinneapple as pp
from pinneapple_intent import run_study, replay

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")

CASES = [
    ("heat-wall", "a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC outside with h = 12 W/m2K",
     "simulation"),
    ("pipe-laminar", "a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s", "simulation"),
    ("beam", "a cantilever beam 0.5 m long with E = 210 GPa, 20 mm deep, carrying 500 N at the tip", "simulation"),
    ("heat-wall-surrogate",
     "a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC outside with h = 12 W/m2K",
     "physics_ai"),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name, text, mode in CASES:
        print(f"\n=== {name} ({mode})")
        print(f"    {text}")
        contract = pp.compile_study(text, mode=mode)
        if not hasattr(contract, "status"):
            print(f"    refused: {contract}")
            summary.append({"case": name, "outcome": "out_of_scope", "reason": str(contract)})
            continue
        print(f"    contract {contract.id}: governed by {contract.equations[0]}")
        for row in contract.table():
            print(f"      {row['name']:>14} = {row['value']:<16} [{row['provenance']}] {row['source']}")
        run = run_study(contract, out_dir=os.path.join(OUT, name), progress=lambda m: print(f"      · {m}"))
        print(f"    -> {run.summary()}")
        for item in run.checks:
            mark = "pass" if item["passed"] else "FAIL"
            print(f"       [{mark}] {item['name']} -> {item.get('value')} (limit {item.get('limit')})")
        summary.append({"case": name, "contract": contract.id, "mode": mode, "decision": run.decision,
                        "qoi": run.outputs()["qoi"], "files": run.files})

    # the replay: the same run, rebuilt from contract.json, run.json and nothing else
    first = CASES[0][0]
    result = replay(os.path.join(OUT, first), out_dir=os.path.join(OUT, f"{first}-replay"))
    print(f"\n=== replay of {first}: {result.summary()}")

    summary_path = os.path.join(OUT, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"cases": summary, "replay_matches": result.matches}, handle, indent=2)
    print(f"\nwritten: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
