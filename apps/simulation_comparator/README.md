# Simulation Comparator

**How far is this result from the reference, field by field, and where.**

Day 10 of the PINNeAPPle 30-app program. Comparing results is everyday work in CAE: a finer mesh, another turbulence
model, a new solver version, a measurement campaign, and now surrogate models, PINNs, FNO and DeepONet predictions.
It usually takes a ParaView session or a script per pair of formats. Upload a reference and a candidate:

| You give it | You get |
|---|---|
| Two results in any supported format: OpenFOAM case (latest or chosen time), VTK/VTU, CalculiX `.frd`, Gmsh, other meshio formats, CSV/Parquet/HDF5/NPZ point tables (sensors, benchmark tables, AI predictions) | **Global error** (MAE, RMSE, max, normalised by each field's range) and **per-field error**: relative L2, MAE, RMSE, max and where, p99, bias, NRMSE, R², per vector component; PASS/FAIL against a tolerance; the **spatial error map** on the reference geometry (or a line / 2D map for point data), parity plot, error distribution, the largest differences |

Modes: simulation vs simulation, vs experiment, vs AI, AI vs experiment, vs analytical or benchmark data. The
candidate is interpolated onto the reference locations: identical locations are compared directly, otherwise linear
(Delaunay) interpolation inside the candidate, nearest outside (counted). OpenFOAM boundary values (fixed values,
noSlip, written wall values) join the interpolation, so points on walls and inlets compare correctly. Fields are
matched by name, by alias (U ~ velocity, p ~ pressure, T ~ temperature, DISP ~ displacement) and by component
(`Ux` ↔ `U[x]`, `DISP_z` ↔ `DISP[z]`), or by an explicit mapping.

## Validation (`tests/test_simulation_comparator.py`)

| Comparison | Result |
|---|---|
| Lid-driven cavity Re = 100 (icoFoam) vs **Ghia, Ghia & Shin (1982)**, u on the vertical centreline | relative L2 **1.26 %** on 20×20 cells, **0.19 %** on 40×40: the expected convergence towards the benchmark |
| CalculiX cantilever (C3D8I, NLGEOM) vs **Euler-Bernoulli** w(x) = F x² (3L − x) / 6EI | **0.71 %** (tip 7.57 vs 7.62 mm: shear deformation and geometric stiffening) |
| PINN temperature map (PINNeAPPle, day 7) vs independent finite volumes | 0.39 %, max 0.67 °C, at the edge farthest from the sensors |
| pitzDaily (simpleFoam, k-ε) 12,225 vs 48,900 cells | U 3.2 %, k 4.6 %, p 8.3 %; the error map puts the coarse mesh's error in the shear layer and along the upper wall |
| pitzDaily k-ε vs k-ω SST, same mesh | U 4.5 %, k 13 %, νt 26 %: same locations, the differences are the models |
| Interpolation alone, analytic field between the two pitzDaily meshes | relative L2 0.04 %, NRMSE 0.09 % |

All results were produced for this app (OpenFOAM v1912, CalculiX 2.21) and are in [`examples/`](examples).

## Run

```bash
pip install -e . -r apps/simulation_comparator/requirements.txt          # from the repository root
cd apps/simulation_comparator && uvicorn simulation_comparator.api:app --port 8089
```

Environment: `CMP_USER` / `CMP_PASSWORD` (HTTP Basic login), `CMP_MAX_MB` (default 300), `CMP_MAX_POINTS`
(default 2,000,000), `CMP_MAX_HEAVY` (default 2). Docker: `docker build -f apps/simulation_comparator/Dockerfile -t
simulation-comparator .`; deployment with the other apps in [`apps/deploy`](../deploy) (service `compare`).

## API and library

`POST /api/compare` (multipart `reference` and `candidate` files; `mode`, `mapping` JSON, `method`, `tolerance` %,
`reference_time`, `candidate_time`). Library: `from pinneapple_data.cae.compare import compare`.
