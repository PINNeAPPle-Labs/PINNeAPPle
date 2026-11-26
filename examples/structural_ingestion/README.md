# Structural ingestion — one explicit-FEA run as one object, and the normalisation to a training set

An explicit structural solver does not produce a dataset; it produces a *run*. This example takes three coupon
impacts, each with its own geometry, wraps every one in `pinneapple_data.cae.StructuralRun`, reads each back through
the two paths #1500 names, and normalises the campaign into a training set.

```
python examples/structural_ingestion/run.py            # about 25 s on an M-series Mac
python examples/structural_ingestion/run.py --smoke    # two designs, smaller meshes
```

## What it measures

| step | what is compared | number from the run |
|---|---|---|
| `CrashResult` → `StructuralRun` | frames, time axis, parts with their elements, unit per field, provenance, eroded/contact/fixed masks | 3 runs, 18 frames, 975 nodes, 576 cells, 6 parts |
| the run as a legacy-VTK time series (what `anim_to_vtk` writes) and back | every field | worst relative discrepancy 3.1e-10 |
| the staged OpenRadioss layout (`frame_000.vtk` + `disp_XXX.txt`/`time_XXX.txt`) and back | displacement | relative discrepancy 1.7e-07 |
| decimation to a 120-node budget | displacement of every dropped node against its nearest kept node | relative L2 0.13 |
| the Zarr round trip | `StructuralRun.checksum()` | identical at float64, 4.1e-08 at float32 |

The solver here is **this library's own explicit crash solver**, not OpenRadioss and not LS-DYNA. The VTK series and
the staged case layout are written by the library from that solver's output, in the format each source writes; the
LS-DYNA ASCII path is exercised by `tests/test_structural_run.py` against fixtures built from the documented card and
print layouts. `docs/guides/structural_ingestion.md` carries that as a table row per format, with what was run and what
was not.

## What a run is

`StructuralRun` holds `frames` of one `Mesh` sharing one connectivity, `times` in the run's time unit, `parts`
(id → name and element indices), `units` per field, `provenance` (solver, version, commit, data hash, mesh, case), and
the masks a surrogate needs: `eroded` `(F, m)`, `contact_force` `(F, 3)` and `fixed_nodes` `(n,)`. Nodal quantities
live in `Mesh.point_data` (`displacement`, `velocity`), elemental ones in `Mesh.cell_data` (`von_mises`,
`plastic_strain`), and `Mesh.points` carries the deformed geometry of that frame, so a frame renders directly.

`to_units` moves a run between `mm-tonne-s` and SI through `physical_units`; `to_zarr`/`from_zarr` round trip it;
`checksum` digests the arrays and the metadata, and excludes the keys that describe the container rather than the run
(`stored_as`), so a float32 copy carries the same digest.

## What the normalisation is for

`build_training_set` does the five things between a run and a dataset, and reports every one:

- **dedup** — a repeated state is dropped and counted;
- **decimation** — elements are selected by a farthest-point spread over their centroids until their corners fit the
  node budget, and the error is measured, not assumed: every dropped node's displacement against its nearest kept
  node. `keep_cells=False` asks for a point cloud instead, and says the elemental fields are gone;
- **units** — every run is brought to SI;
- **non-dimensionalisation** — per field, divided by one reference taken over the whole set, with the reference kept
  (geometry shares the displacement's length scale, so a frame stays `undeformed + displacement`);
- **the split holds out a geometry family** — frames of one impact share a mesh and a design point, so splitting them
  at random leaks the answer across the split.
