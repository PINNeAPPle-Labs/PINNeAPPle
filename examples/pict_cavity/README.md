# Lid-driven cavity against Ghia et al. (1982)

The cavity every incompressible solver is asked for: a square box, three still walls, a lid moving at `U = 1`, and the
published centreline profiles of Ghia, Ghia and Shin (*J. Comput. Phys.* **48** (1982) 387–411) to compare with.

PINNeAPPle runs it here with the **local engine** of `pinneapple_simulation.external_solvers.pict` — a
vorticity–streamfunction solver in numpy/scipy — so this example needs **no GPU**. The same call with `--engine pict`
runs PICT (multi-block PISO on CUDA) when it is installed, and says what is missing when it is not; the output always
names the engine that produced the numbers.

```bash
python examples/pict_cavity/run.py            # 64x64 at Re=100 and 128x128 at Re=1000: ~23 s on one core
python examples/pict_cavity/run.py --smoke    # 32x32 at Re=100: a tenth of a second (CI)
python examples/pict_cavity/run.py --engine pict --re 100 1000   # on a machine with PICT and an NVIDIA GPU
```

## What is measured

| Metric | Re = 100 (64²) | Re = 1000 (128²) |
|---|---|---|
| rms of `u(x = 0.5)` against Ghia Table I | 1.4e-03 | 8.1e-03 |
| rms of `v(y = 0.5)` against Ghia Table II | 2.8e-03 | 5.3e-03 |
| largest discrete `du/dx + dv/dy` (normalised) | 3.6e-15 | 7.1e-15 |
| speed on the three closed walls | 0 | 0 |
| wall clock (one CPU core) | 0.7 s | 22 s |

`cavity_profiles.png` puts the computed centrelines next to the published points. The numbers above are the ones
recorded in `experiment.json` by `pinneapple_lab.experiment`; a figure saved inside the `with` block is attached to the
run automatically.

## What this example is not

It is not PICT and does not stand in for it: the local engine solves a uniform square grid with the lid at the `+y`
wall, reports no pressure field (it solves vorticity and streamfunction), and marches to steady state rather than
offering an accurate transient. PICT on a GPU is tracked in the repository issues; the Ghia comparison that PICT would
have to reproduce is the one implemented here.
