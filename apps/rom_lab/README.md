# Reduced Order Model Lab

**Cheaper models of your simulations, with the error on held-out data in sight.** Day 19 of the PINNeAPPle apps
program; live at <https://rom.pinneapple.org> (port 8102).

**Pain.** High-fidelity simulations are too slow for interactive exploration, optimisation or control. A reduced
model replays them in microseconds, but how much error does it cost, at which rank, and does it forecast or only
interpolate? This lab answers that on snapshots the model never saw.

| You give it | You get |
|---|---|
| Field snapshots of one mesh: an OpenFOAM case with several time directories (zip), a series of VTU / VTK / CalculiX `.frd` / other meshio results, NPZ / NPY / HDF5 arrays (`snapshots` or one array per field, `times` or `params` + `param_names`, `points`), or a CSV / Parquet table with one row per snapshot (`t`, `param_*` columns, or a `DatasetSpec` sidecar whose inputs are the parameters) | Every method at several ranks, scored on **held-out** snapshots: mean / max relative L2 error, per-field errors, the POD **projection floor**, fit time, **inference time per snapshot**, **speed-up** over the full-order model (when its time per snapshot is known), the singular values and energy, error vs rank, error vs speed with the Pareto front, error per held-out snapshot, and the reconstruction of any held-out snapshot (truth, ROM, difference) |

**Methods** (library: `pinneapple_neural.architectures.rom`, study in `rom.study`): POD + RBF and POD + GPR
(`ParametricPOD`), exact DMD (`DynamicModeDecomposition`), quadratic Operator Inference (`OperatorInference`, its
regularisation chosen inside the training window), the baselines every ROM must beat (persistence for a forecast,
the nearest snapshot for unseen designs), and a neural surrogate trained at equal wall-clock budget
(`pinneapple_analysis.cost.train_within_budget`): parameters to field, or a residual stepper of the POD
coefficients. Held-out data: the last times (forecast), interleaved times, or random designs. Several fields are
scaled by their training RMS; parameters can be regressed in log scale; an amplitude / shape split helps responses
that span orders of magnitude.

**The held-out design on its body (3-D).** When the snapshots carry a mesh (a VTU / VTK / FRD series, or arrays
with `cells_<type>` connectivity, optional `snapshot_points` per snapshot and `node_sets` in `meta`), the result page
shows the selected held-out snapshot on its own body: the full-order field, the ROM prediction and the pointwise error
|ROM − FEM| (switch between them; full-order and ROM share one colour scale), on the shape deformed by the FEM or the
ROM displacement with a scale factor. The dataset's node sets are marked (the cantilever's clamp and loaded tip), the
error hotspots (connected nodes in the top 10 % of the error) are listed and can be zoomed to, and a short text says
where the error concentrates: each set grown by 10 % of the body length is a zone, with its share of the squared error
against its share of the nodes, next to the study's error for that design and the study mean. All of it is computed
from the stored snapshot and the model's prediction at its inputs (`pinneapple_neural.architectures.rom.views`,
`pinneapple_analysis.validation.error_map`, `pp.viz.field_view`). Datasets without connectivity keep the 2-D map.

![POD + GPR on a held-out cantilever: |ROM − FEM| of the von Mises stress on the deformed body](../../docs/assets/apps/d19-rom-beam-3d-error.jpg)

**Outputs and integrations.** Each model downloads as a portable `.rom.npz` (arrays + JSON manifest, no pickle:
`ReducedModel.from_bytes(...).predict(times=... | params=...)`), as **ONNX** (POD + RBF, nearest, the parametric neural
model, DMD) with the `spec` the model gateway publishes (`ModelGateway.publish(name, "x.onnx", kind="onnx",
spec=...)`), and as a **`ModelRef`** contract (held-out metrics with the split named, training range, SHA-256 of the
file). The study is a **`StudyResult`** (table, best model, checks with their values, limits). `POST /api/run` runs an
exported ROM and says for each query whether it is inside the training range. Inputs come from OpenFOAM, VTK,
CalculiX, HDF5 and the other apps' `DatasetSpec` tables.

## Validation (`tests/test_rom_study.py`, `tests/test_app_rom_lab.py`)

| Case | Result |
|---|---|
| Two travelling waves (an exact rank-4 linear system), forecast of the last 20 % | DMD at rank 8 (uncentred: the mean is a mode) and OpInf at rank 4 below 1e-3; rank 2 fails; POD + RBF in time does not extrapolate; persistence > 50 % |
| Every model | its held-out error is never below the POD projection error at its rank |
| Export | `.rom.npz` predicts exactly what the fitted model does; ONNX agrees to < 1e-4 relative |
| 3-D view | the error map of a held-out cantilever reproduces the relative L2 and the max of the 2-D reconstruction; the deformed shape is the stored displacement in metres; the zones' shares of the squared error add up to 1 |
| Cylinder wake, PINNeAPPle LBM at Re 100 (example, 30 forecast snapshots) | DMD rank 16: 0.81 % (max 1.4 %), OpInf rank 16: 1.0 %, neural stepper (3 s): 3.8 %, persistence: 136 % |
| Cantilever, solid FEM, 100 designs (example, 20 unseen designs, log H and W) | POD + GPR rank 8: 4.4 %, POD + RBF: 17 %, neural (3 s): 16 %, nearest design: 39 % |

The examples are produced by `examples/make_examples.py` from the library's own solvers; the full-order times stored
with them (0.68 s per wake snapshot interval, 0.72 s per FEM design) were measured when they were generated. The
cantilever's mesh (hexahedra, each design's node coordinates, the clamp and loaded-tip node sets) was added with
`make_examples.py --mesh-only`, which first solves two designs again and checks that they match the stored fields.

## Run

```bash
pip install -e . -r apps/rom_lab/requirements.txt          # from the repository root
cd apps/rom_lab && uvicorn rom_lab.api:app --port 8102
```

Environment: `ROM_USER` / `ROM_PASSWORD` (HTTP Basic login), `ROM_MAX_JOBS` (studies at once, default 2),
`ROM_MAX_MB` (default 200), `ROM_MAX_VALUES` (values per snapshot, default 400,000), `ROM_MAX_BUDGET_S` (largest
neural budget, default 30 s), `ROM_DATA_DIR`, `ROM_ADMIN_TOKEN`, `ROM_REQUIRE_API_KEY`, `ROM_FREE_STUDIES` /
`ROM_FREE_EXPORTS` / `ROM_FREE_QUERIES` (free use per caller and month: 30 / 100 / 20,000). Docker:
`docker build -f apps/rom_lab/Dockerfile -t rom-lab .`; deployment with the other apps in [`apps/deploy`](../deploy)
(service `rom`).

## API

`POST /api/studies` (multipart `files`, `options` JSON: methods, ranks, budget_s, split, test_fraction, tolerance,
log_params, amplitude_split, hf_seconds) queues a study; `GET /api/jobs/{id}` gives progress and the result;
`GET /api/jobs/{id}/files/...`, `/onnx/{model}`, `/gateway/{model}`, `/reconstruction`, `/view3d` (the 3-D view of a
held-out snapshot: fields, colour ranges, markers, error maps and summaries) and `/view3d.glb`; `POST /api/run`;
`POST /api/examples/{wake|beam}`. Interactive documentation at `/docs`.

## Limits

- Snapshots must share one mesh (same points or cells, same order): nothing is interpolated between meshes.
- Time steppers need a uniform time step and predict from the latest training snapshot before the requested time;
  POD + RBF in time interpolates and does not forecast. Nothing is validated outside the training window or box.
- Errors are relative L2 on this dataset's held-out snapshots (time series: relative to the fluctuation about the
  training mean). A small error does not guarantee conservation or boundary values: no physics check is run on the
  reduced fields (the `ModelRef` says `UNCHECKED`); check exported predictions in the Physics Validation Lab.
- Speed-ups compare the measured inference time on this server with the full-order time you give (or that was
  measured with the example); they do not include building the snapshots.
- Linear in the POD basis: transport-dominated or strongly nonlinear problems need many modes. GPR and OpInf have no
  ONNX graph (use the `.rom.npz`); the amplitude split has no ONNX graph either.
- The upload is kept with its study in the server's data volume (for the reconstructions); studies belong to their
  caller (login, API key or address).

Monetization: building, exporting and running reduced models: studies, exported models and ROM queries per month,
with private projects and API keys for teams.
