"""07 — Benchmark architectures on the same problem.

The Arena trains several registered models on one problem with the same budget,
scores them against the same reference, and ranks them.

Run from the repository root:

    python examples/quickstart/07_benchmark_models.py
"""
from __future__ import annotations

import pinneapple_arena as ar

config = ar.ArenaConfig.from_dict({
    "problem": {"name": "burgers_1d", "params": {"nu": 0.01}},
    "models": [
        {"name": "VanillaPINN", "type": "vanilla_pinn", "training": {"epochs": 600, "lr": 1e-3}},
        {"name": "SIREN", "type": "siren", "training": {"epochs": 600, "lr": 1e-3}},
        {"name": "HashGridMLP", "type": "hash_grid_mlp", "training": {"epochs": 600, "lr": 1e-3}},
    ],
    "output": {"dir": "runs/quickstart_arena", "prefix": "guide"},
})

arena = ar.Arena(config).run()
arena.compare()

# The same numbers are available as data, one dict per model:
print()
for run in arena.results:
    m = run["metrics"]
    print(f"{run['name']:<14} L2 {m['L2_u']:.3e}   rel {m['rel_u']:.3e}   Linf {m['Linf_u']:.3e}")
