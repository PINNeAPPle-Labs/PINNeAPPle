"""01 — First look: what is installed, and what is already solved for you.

Run from the repository root:

    python examples/quickstart/01_first_look.py

Everything printed here comes from the installed package, so the numbers are
always current for your version.
"""
from __future__ import annotations

import pinneapple as pp

print(f"pinneapple {pp.__version__}")

# --- What is installed, and which optional dependencies are missing -------
pp.info()

# --- Problem presets, model architectures, and things already trained -----
print(f"{len(pp.list_presets())} problem presets, e.g. {pp.list_presets()[:4]}")
print(f"{len(pp.list_models())} model architectures, e.g. {pp.list_models()[:4]}")
print(f"pre-trained models in the zoo   : {pp.zoo.list()}")
print(f"industrial cases ready to run   : {pp.presets.names()[:6]}")

# --- Inspect one problem before training anything ------------------------
spec = pp.get_preset("burgers_1d", nu=0.01)
print("\nburgers_1d")
print("  coordinates :", spec.coords)
print("  fields      :", spec.fields)
print("  domain      :", spec.domain_bounds)
print("  PDE         :", spec.pde.kind, spec.pde.params)
print("  conditions  :", [c.name for c in spec.conditions])
