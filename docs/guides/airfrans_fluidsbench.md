# AirfRANS → FluidsBench: a submission campaign you can run again

`pinneapple_simulation.pipelines.airfrans_benchmark` takes the public AirfRANS release, fits a coordinate network, and
drives the FluidsBench participant workflow until a validated result package exists. It is the same experiment every
time: the dataset and the scoring support are fixed by the benchmark, the model and the budget are recorded, and the
metric values come from the benchmark's own evaluator.

```bash
python -m pinneapple_simulation.pipelines.airfrans_benchmark all spec.json work/airfrans-full
```

```json
{
 "dataset_root": "…/airfrans/Dataset",
 "support_root": "…/support/airfrans",
 "checkout": "…/fluidsbench-submission",
 "store_dir": "…/store",
 "submission_id": "pp-coordinate-net-full-v1",
 "split_id": "full",
 "training": {"train_cases": 640, "epochs": 30, "physics_weight": 0.0}
}
```

## The four stages

| stage | what it does | what it needs |
|---|---|---|
| `prepare` | extracts every case of the release into the feature store, one `.npz` per case, `parallel=4` by default | the pinned AirfRANS/VTK stack (`airfrans==0.1.5.1`, `pyvista==0.48.4`, `vtk==9.6.2`) |
| `train` | fits the network on one split's training cases and writes `model.pt` + `training.json` | a store |
| `predict` | runs it over the official test cases and packs `<entry>/predictions/<case>.npz` | a store, weights |
| `submit` | writes the entry and the package configuration, then runs the benchmark's `validate-entry` → `evaluate-entry` → `package` → `verify-package` → `validate_submission.py`, and finally re-checks the package and recomputes the domain metrics | the three paths, and the participant interpreter for the benchmark's tools |

Every stage is resumable: an existing store, run directory or entry is reused. `store_dir` can point at a shared store,
because the extraction is the expensive stage and every campaign on the same release reads the same arrays.

## What the model is, exactly

Eleven input channels per point: `x/c`, `y/c`, the AirfRANS signed distance divided by the chord and its signed
logarithm, the wall normals, the surface flag, the unit freestream vector (the case's angle of attack) and `log|U∞|`.
Three outputs, scaled by the freestream: `u/U∞`, `v/U∞`, `p/U∞²`. A point-wise MLP of four SiLU layers (256 wide by
default), fitted with AdamW on sampled points — a quarter of every sample comes from the wall and the boundary-layer
band, because that is where the wall-derived metrics are read.

`physics_weight > 0` adds two differentiable residuals: no penetration on the airfoil wall, and Bernoulli's relation in
the outer domain. Measured on the full split, that trade is small and in one direction: field errors up by about 1 %
relative, wall shear down 12 %, `|Cd R²|` down 32 %, `overall_score` 28.49 → 28.55.

## The wall-aware variant, and why it exists

The benchmark never reads a submitted wall shear or a submitted force coefficient: it re-integrates both from the
submitted velocity and pressure, through the point-based velocity gradient at the surface nodes. Measured on the plain
coordinate network, the wall velocity — which must be exactly zero — came out at 3.7 % of `U∞`, the first mesh cell of
the release sits at `y⁺ ≈ 0.2`, the reconstructed viscous drag came out 5.6× the reference value, and 73 % of the
pressure-drag error sat ahead of 10 % chord. No amount of capacity fixes that; the field near the wall is wrong by
construction.

`WallAwareNet` addresses it with three additions:

- **an exact no-slip wall layer**, `U = A(d)·U_outer + (1−A(d))·s·d` with `A(d) = 1 − exp(−(d/λ)²)` and `λ` the
  viscous-sublayer height. The wall value is zero by construction, and the first derivative at the wall *is* the slope
  the evaluator reconstructs. Blending the true field into this layer with `du_t/dd = +τ_w/ν` reproduces the evaluator's
  wall shear to 0.85–1.7 % on two cases;
- **a separate wall network**: the slope is predicted by its own small MLP over the raw point channels, not by a head on
  the shared trunk. Measured three ways on the wall slope — linear probe 74 %, standalone MLP 60 %, head on the trunk
  330 % — the trunk smooths away exactly the near-wall detail the readout needs;
- **a per-case shape descriptor**: the airfoil's upper and lower surfaces on 32 chord bins, 64 channels. A point-wise
  network cannot tell two airfoils apart at the same location, and that ambiguity is worth 75.7 % → 42.1 % on a
  nearest-neighbour probe of the wall slope.

`WallSpec` / `train_wall` fit it with three objectives: the data term, a robust relative squared error on the wall
slope, and the squared error of the drag and lift reconstructed from the predicted curve through
`wall.curve_coefficients` — the evaluator's own polyline quadrature, reproduced to five decimals against
`Simulation.force_coefficient` and asserted in `tests/test_airfrans_benchmark.py`.

Two properties of the *objective* had to be measured rather than assumed, because both look like a capacity problem
and are not:

- **the gradient clip has to be per group.** `clip_grad_norm_` over every parameter measures ~140 at initialisation —
  the force term dominates the trunk — and its 0.007 scale factor landed on the slope head too, training it at 1/140 of
  its learning rate. The head then could not fit its own training cases (356 % relative error on the training set) while
  the same readout trained alone reaches 6 %. `clip_wall_gradients` clips the trunk and the readout against separate
  norms;
- **the relative error has to be flattened on the median, not the RMS.** The wall slope has rms 26 500 and median
  6 500, the top 1 % of nodes carrying 39 % of its energy. A floor on the RMS (or none) turns the term into a
  tail-weighted MSE: the training error falls to 3–13 % while the held-out error *rises* to 970–1 240 % and the
  predicted magnitude runs to 3.6× the truth. Five reductions on one fixed slice, 60-case held-out set: the median floor
  is the only one that keeps improving, ending at 145 % against 970 %.

Neither of those fixes moved the wall shear on the holdout, and neither did removing the force term or detaching the
readout from the data term. What did move it is taking the magnitude *out of the network*: `slope_prior=True` has the
readout predict a log-correction to `flat_plate_slope`, the turbulent flat-plate skin friction `Cf = 0.0576 Re_x^{-1/5}`
at the predicted edge speed. The untrained law already sits at 168 % relative error with a magnitude ratio of 1.08, and
on the 20-case `full` holdout the trained model reaches **38.5 %** wall shear against 43.5 % for the entry without it,
with `Cd R²` moving from −5.03 to −1.88. Two unit traps are worth remembering: the freestream channels travel as a unit
direction and `x_over_chord` is dimensionless, so the law has to be fed metres per second and metres (without the inlet
factor it comes out a thousand times short and training diverges), and standardising the readout on wall nodes sends a
far-field row to a huge z-score, which the exponent then turns into an inf — both the z and the correction are clamped.

**What the wall shear actually measures, found the hard way.** Splitting the budgets — freeze the readout and fit the
field, then freeze the field and refit the readout — gives the best near-wall numbers of the campaign (wall shear
43.5 % → 30.9 %, `Cd R²` −5.03 → −0.97 on the 20-case holdout) but costs field accuracy (velocity 4.31 % → 5.80 %).
The informative part is the pattern: the wall shear is a property of the *pair* (field, readout), not of the readout.
Freezing the readout while the field trains keeps the near-wall layer consistent with the law by construction and is
what buys the gain; refitting the readout afterwards changes almost nothing (30.9 → 29.6), and refitting it on a trunk
that was trained with a *co-adapting* readout makes it much worse (38.5 % → 75.4 %), because the readout then disagrees
with the field it sits in. Chasing the readout's own slope error is therefore the wrong lever, which is worth knowing
before spending another run on it.

On the `full` split this takes `overall_score` from 29.44 to **63.20** (field components 16.09 %/758.6 %/12.15 %/34.41 %
→ 2.93 %/78.7 %/4.49 %/3.66 %, Cl R² 0.988 → 0.9995, profile R² 0.782 → 0.968, `field_score` 0 → 58.03).

## What the campaign guarantees, and what it does not

- The package is written and validated by the benchmark's own tooling; the campaign only drives it.
- It checks the result twice: `pinneapple_arena.fluidsbench.validate_package` (digest chain, every relative L2 against its
  sufficient statistics, the aggregate against the macro-average, `overall_score` against its composite) and an
  independent recomputation of the six domain metrics from the same predictions — agreement at 1e-10.
- It never claims a split's result is an extrapolation test unless the model was fitted in that split's own regime: the
  methodology record states which `*_train` key the cases came from.

## Related

- [`fluidsbench_datasets.md`](fluidsbench_datasets.md) — what each of the other open FluidsBench datasets costs, and
  what this machine can carry.
- [`fluidsbench.md`](fluidsbench.md) — the adapter: the contract, the support chain, the reductions, the trust decision.
- [`examples/fluidsbench_airfrans`](../../examples/fluidsbench_airfrans/README.md) — the same path rehearsed on an exact
  reference (potential flow around a cylinder), with no dataset download.
- `pinneapple-engineering-api` exposes this pipeline as the `engineering.fluidsbench_campaign` job capability
  (`POST /v1/fluidsbench/jobs`).
