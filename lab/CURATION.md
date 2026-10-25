# PINNeAPPle Lab curation

Generated 2026-10-25 12:31 UTC. Tiers: **A flagship** (product / paper / marketing once reviewed), **B solid** (demos, use cases, datasets), **C plausible** (partly checked), **D exploratory** (not validated), **E known problems** (a failing check), **F not usable** (crashed, diverged, failed physics or contradicts the reference). Gates, not averages: see `pinneapple_lab/curation.py`.

| experiment | tier | best score | runs (A/B/C/D/E/F) | usable | product | paper | marketing | data | reviewed |
|---|---|---|---|---|---|---|---|---|---|
| `manifest` | **B** solid | 80.0 | 0/1/0/0/0/0 | 100 % | - | - | - | - | no |

## Trust card: six questions per item

Data and geometry (was the physical problem represented correctly?), model (can it represent the dynamics?), physical constraints (equations and boundary conditions respected?), benchmark (against a solver or reference data?), uncertainty (where may it not be reliable?), engineering decision (adequate for the intended use?). ✓ answered by a passing check, ~ partial evidence, ? open, ✗ a failing check.

| item | tier | data & geometry | model | physics | benchmark | uncertainty | decision |
|---|---|---|---|---|---|---|---|
| `manifest/examples/freecad_openfoam` | B | ✓ | ✓ | ? | ✓ | ? | ~ |

## What each experiment needs next

- `manifest` (B): show it beats a baseline or satisfies a physical law it was not trained on; add figures (a movie, a render or a 3-D view for communication); document it (docs page, references)
