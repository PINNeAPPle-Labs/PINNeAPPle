# Surface-pressure surrogate benchmark (CarBench protocol)

Geometry point cloud in, surface kinematic pressure out, over a parametric car family. The full description, the
mapping to the published benchmark and the limits of the recreation are in
[`docs/guides/crash_and_surface_benchmarks.md`](../../docs/guides/crash_and_surface_benchmarks.md).

```bash
# a minute: 4 designs at a token resolution, 2 epochs
python examples/carbench_surface_pressure/run.py --smoke

# the campaign this folder ships results for: 40 designs, 44 cells per car length, 3000 lattice steps
python examples/carbench_surface_pressure/run.py --designs 40 --parallel 7 --cells-per-length 44 --steps 3000
```

Outputs in `_out/` (not committed — regenerate them):

| file | what it is |
|---|---|
| `surface_pressure.npz` | the field dataset: one geometry per sample, its wall-face points and their pressure |
| `coefficients.csv` | design vector, Cd, Cl, frontal and wetted area per design |
| `leaderboard.md` / `.json` | the model ranking with bootstrap intervals and the cost columns |
| `summary.json`, `failures.json` | the campaign manifest, and every design that could not be run |
| `experiment.json` | the Lab manifest of the run (its metrics carry how they were measured) |

The labels come from this library's lattice-Boltzmann tunnel, not from a RANS code: the absolute error levels are
those of a coarse LBM label and are not comparable with the published table.
