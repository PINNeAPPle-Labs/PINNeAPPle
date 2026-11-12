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

## What the campaign guarantees, and what it does not

- The package is written and validated by the benchmark's own tooling; the campaign only drives it.
- It checks the result twice: `pinneapple_arena.fluidsbench.validate_package` (digest chain, every relative L2 against its
  sufficient statistics, the aggregate against the macro-average, `overall_score` against its composite) and an
  independent recomputation of the six domain metrics from the same predictions — agreement at 1e-10.
- It never claims a split's result is an extrapolation test unless the model was fitted in that split's own regime: the
  methodology record states which `*_train` key the cases came from.

## Related

- [`fluidsbench.md`](fluidsbench.md) — the adapter: the contract, the support chain, the reductions, the trust decision.
- [`examples/fluidsbench_airfrans`](../../examples/fluidsbench_airfrans/README.md) — the same path rehearsed on an exact
  reference (potential flow around a cylinder), with no dataset download.
- `pinneapple-engineering-api` exposes this pipeline as the `engineering.fluidsbench_campaign` job capability
  (`POST /v1/fluidsbench/jobs`).
