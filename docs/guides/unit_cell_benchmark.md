# Periodic unit cell: an FNO over frequency × scan, and an equal-budget benchmark against Bayesian optimisation

An infinite phased array (a tightly coupled dipole array, TCDA) or a frequency-selective surface is simulated as
one **unit cell** with periodic boundaries carrying the phase of the scan angle and a Floquet port. The quantity a
wideband array designer watches is the **active reflection coefficient** at the element port over frequency and scan
angle. This guide runs a campaign of cells in EMerge, trains a Fourier neural operator (FNO) from the cell
parameters to the whole (frequency × scan) map, measures its error on cells it never saw, and compares, at the same
number of solver calls, two ways of minimising the worst-case active reflection: an FNO plus an optimiser (verified
by the solver), and Bayesian optimisation calling the solver directly.

Everything lives in `pinneapple_simulation.electromagnetics.periodic`; the example
`examples/use_cases/antennas/unit_cell_benchmark.py` only calls it.

```bash
pip install emerge
python examples/use_cases/antennas/unit_cell_benchmark.py --quick                 # 3 cells, budget 6 (~6 min)
python examples/use_cases/antennas/unit_cell_benchmark.py --cells 64 --budget 24  # the pilot below
```

## The cell (`tcda.TCDACell`, generic)

A square cell of side `a` (22 mm by default); the ground plane at z = 0 (the bottom face of the cell); a dipole
strip of width `w` at height `h`, fed across a gap `s` by a lumped port (100 Ω); its arms stop `g/2` from the cell
edge, so neighbouring dipoles couple capacitively across `g`; an optional dielectric superstrate (`er_s`, `t_s`) at
`d_s` above the dipole; air up to 0.35 λ above the stack and a Floquet port. `check()` refuses a cell that lets a
grating lobe in at the highest frequency and scan (`a ≥ λ/(1 + sin θ)`). The default ranges of the campaign
(`benchmark.CELL_RANGES`) vary `h`, `w`, `g`, `t_s` and `er_s`.

EMerge details that matter: the periodic boundary conditions exist only after `mw._initialize_bc_data()`, so the scan
angle is set after it; one mesh then serves every scan angle of a cell (the angle changes only the boundary phases
and the port), and solving the angles in sequence on one model gives the same S11 as solving each alone (test).
SuperLU is used.

## Measured

PILOT_RESULTS

## What it would take to run 1,000 cells

COST_PROJECTION

## API

```python
from pinneapple_simulation.electromagnetics.periodic import benchmark as bm
from pinneapple_simulation.electromagnetics.periodic.tcda import TCDACell, simulate

simulate(TCDACell(h=9), [3, 4.5, 6], [(0, 0), (30, 0)])           # rows of active reflection
table, sim = bm.campaign(64, freqs, scans, "runs/")               # DOE of cells, one process per cell
X, Y, tags = bm.to_arrays(table, list(bm.CELL_RANGES), freqs, scans)
predict, info = bm.fit_fno(X[:48], Y[:48], wall_s=60)            # FNO over the (f, scan) map
bm.heldout(predict, X[48:], Y[48:])                               # error on unseen cells
bm.equal_budget(bm.solver_from(sim, names, freqs, scans), bm.CELL_RANGES, budget=24)
```
