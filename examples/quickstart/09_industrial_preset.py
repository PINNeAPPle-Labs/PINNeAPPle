"""09 — Run a ready-made industrial case.

`pp.presets` holds industrial cases with a data sheet, sourced defaults, a solver,
references with measured error, a surrogate, inversion and optimisation. Nine
lines get you a pressure-surge answer with its trust decision.

Run from the repository root:

    python examples/quickstart/09_industrial_preset.py
"""
from __future__ import annotations

import pinneapple as pp

preset = pp.presets.load("water_hammer_valve_closure")
card = preset.card()
print("case    :", card["title"])
print("sector  :", card["sector"])

# Slow closure: the surge stays inside the pipe rating.
result = preset.run(t_close=10.0)
print()
print(result)

# Fast closure: too fast for the line — the decision says so.
fast = preset.run(t_close=2.0)
print()
print("t_close=2.0 s ->", fast.decision["verdict"], "| trust:", fast.trust.decisao)

# The case also validates itself against classical theory.
print("\npreset.validate():", preset.validate().get("all_pass"))
