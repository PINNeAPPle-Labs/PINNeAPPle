# Which geometry surrogate, and what it costs

The families that dominate the recent industrial benchmarks — Transolver, GINO, the PointNet/RegDGCNN baselines,
MeshGraphNet — all take a discretised shape and return a field on it. They were each written with their own input
convention, so a study that wants to compare them ends up comparing the plumbing. Roadmap item **P1**
([#293](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/293)) asks for one interface and at least one of the
families trained on real data; this page is the interface, the table of when to use which, and the measured numbers.

```python
import numpy as np
from pinneapple_analysis.surrogate import (GeometrySample, samples_from_panel, panel_integrals,
                                           build_geometry_surrogate)

samples = samples_from_panel(shapes, alphas=(0.0, 4.0, 8.0))          # one sample per body and condition
model = build_geometry_surrogate("transolver", fields=("cp",), hidden=96, depth=4, slices=16,
                                 integrate=panel_integrals)
model.fit(samples, epochs=60, batch=8)                                # returns the model
pred = model.predict(points, normals, conditions, elements)           # fields + integrals, at any point count
report = model.evaluate(held_out)                                     # relative L2, integral error, ranking R²
```

## The interface

| | |
|---|---|
| `GeometrySample` | `points` (N, 3), `fields` per point, `normals`, `elements` (panel length, thickness), `conditions` (angle of attack, Re), `integrals` (Cl, Cd) |
| `.fit(samples, epochs=, batch=, lr=, n_points=)` | trains any family; every epoch subsamples each sample, which is how a variable point count is handled and how the model avoids memorising one discretisation |
| `.predict(points, normals, conditions, elements)` | the field at the points handed in, and the integrals — no fixed grid, no fixed count |
| `.evaluate(held_out)` | relative L2 per field, absolute integral error, and the R² of the designs ranked by the first integral |
| `.save(dir)` / `.load(dir)` | weights plus the whole input convention, so a reload predicts the same numbers |

Three decisions carry it:

- **A sample is a point set, not a grid**, so the same weights answer for a body with a different panel count, a
  decimated surface or a point cloud from a scan.
- **The integrals are integrated from the predicted field** by an `integrate` callable the case supplies, not by a
  second head. A predicted drag therefore cannot disagree with the pressure it came from: if the field is wrong, the
  integral is wrong in the same direction. On the panel family, `panel_integrals` is the solver's own pressure
  integration, and the test checks that it reproduces the solver's Cl exactly on the solver's own field.
- **A family that is not here says so, with the reason.** `available_families()` is data a program can read; it never
  returns a stand-in under the name of a published model.

## Which one, and when

| family | in the library | use it when | measured here |
|---|---|---|---|
| `transolver` | yes — wrapped, not reimplemented (`neural_operators.transolver`) | you need a **field** on a large surface (10⁴–10⁵ points): the cost is linear in the point count because attention runs on learned slices, not on points | L2 0.154, ranking R² 0.11 |
| `pointnet` | yes — `architectures.point_surrogates` | you need **integrated quantities** and the smallest training budget; it gives up on the field before it gives up on the ranking | L2 0.241, ranking R² 0.60 |
| `gino` | **open** | a latent **regular grid** is acceptable and you have `torch-scatter`/`torch-geometric`: a graph encoder, an FNO on the grid, a graph decoder. The interface already carries what it needs (arbitrary points, normals, variable count), so a backend implements three methods and nothing else changes | not trained |
| `meshgraphnet` | yes, but **not wrapped here** (`architectures.graphnn`, with #236's world-space edges) | the mesh is **fixed** — one topology, many solutions. It is the aero app's baseline; putting it behind this interface means re-triangulating every shape to a common connectivity, which would quietly turn "geometry-agnostic" into "one topology" | not trained |

The catalogue in code (`available_families()`, also written into the benchmark JSON) carries the same reasons, so the
page and the program cannot drift apart.

## The measured case

`benchmarks/geometry_surrogate_interface.py` trains both families on a real solver: bodies of the CST family the
library's aerodynamic work uses, labelled by the Hess-Smith panel method (`pinneapple_design.aero.panel`), which is the
inviscid reference the library already treats as ground truth. Each body gives the `cp` field on its panel mid-points
and `cl`/`cd` from the solver's own pressure integration; three angles of attack per body make the conditions
non-trivial. Training shapes are a Latin hypercube inside the documented ranges; held out are shapes from the same box
that no model saw.

```bash
python -m benchmarks.geometry_surrogate_interface            # the protocol, a few minutes on one CPU
python -m benchmarks.geometry_surrogate_interface --quick     # under a minute, not a benchmark
```

108 training samples (36 bodies × 3 angles of attack) and 24 held-out ones, 120 panels per body, two training seeds
each, on one laptop CPU:

| | parameters | training | relative L2 of `cp` (mean / worst) | \|ΔCl\| (mean / worst) | ranking R² | prediction |
|---|---|---|---|---|---|---|
| `transolver` | 294 241 | 157 s (153–161) | **0.154** / 0.345 | **0.0396** / 0.0988 | **0.11** (0.08–0.15) | 5–8 ms |
| `pointnet` | 121 825 | 44 s (43–47) | 0.241 / 0.551 | 0.078 / 0.304 | **0.60** (0.55–0.65) | 1 ms |

Two things to read out of that table, and the second is the interesting one:

- **Transolver gives the better field and the better lift coefficient** at half the error of the baseline, for four
  times the training. That is the expected ordering and it is measured, not quoted.
- **It ranks the designs *worse*.** The R² of the designs ordered by lift is 0.11 for Transolver and 0.60 for the
  baseline: the model with the more accurate field does not preserve the ordering of the bodies. At this data size
  the attention is spending its capacity on the field and the pooled baseline on the one number the ordering needs.
  A study that only needs "which of these five is best" is not served by the model that wins the field, and a
  benchmark that reported only the L2 would have hidden it.

The published numbers are in `benchmarks/_out/geometry_surrogate_interface.json`, next to the protocol, the parameters,
the training time and the peak memory of every row — the file, not this page, is the record. The training times are
from a laptop CPU shared with other work, so treat them as an order of magnitude, not as a hardware rating.

## Two properties that change how you use it

- **These models pool over the point set.** Transolver's slices are weighted means of the points in the set and the
  PointNet baseline max-pools it, so the field at a point is a function of the *whole* set it arrives with. Predicting
  on a decimated surface is a different query, not a cheaper one; the `chunk` argument of `predict` exists to keep a
  very large set from exhausting memory, and it changes the answer. A test guards the property so it cannot become a
  surprise in a study.
- **The conditions are per sample, not per point.** The angle of attack reaches every point of a body as a channel.
  Two bodies at two angles are two samples; the model has no notion of a sweep, which is the honest reading of what
  the families do.

## What is not here

- **GINO**, with the reason above. The interface was written so that it is a backend, not a rewrite.
- The **car benchmarks** (AhmedML #241, a DrivAerNet++ subset #282): the numbers here are the interface's, on the
  small case, not a leaderboard. The families' published strengths are on those datasets, and the surface fields they
  need are not in the repository.
- **MeshGraphNet as a row of the comparison.** It stays the aero app's baseline; wrapping it badly would make the
  comparison dishonest, and the note in `FAMILIES` says exactly that.
- **Training-time and memory across seeds**: the benchmark reports the spread between seeds, but two seeds are two
  points, not a distribution.
