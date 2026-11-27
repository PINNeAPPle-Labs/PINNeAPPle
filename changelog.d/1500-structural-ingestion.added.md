- A structural run as one object: `pinneapple_data.cae.StructuralRun` holds the frames of one mesh, the time axis, the
  named parts with the element indices each one owns, a unit per field, a provenance record and the masks a surrogate
  needs (eroded elements, contact force, fixed nodes). `StructuralRun.from_crash_result` is the one-call conversion of
  `numerical_solvers.crash_fem.CrashResult`; `from_vtk_series` reads a legacy-VTK time series (the format
  `anim_to_vtk` writes), `from_openradioss` reads a staged OpenRadioss case directory (`frame_000.vtk` plus the
  `disp_XXX.txt`/`time_XXX.txt` reduction) and `from_lsdyna` reads the ASCII half of an LS-DYNA run — the keyword deck
  and `nodout` — with the quantities that route cannot carry stated in `info["unsupported"]` rather than left as an
  empty array. `to_units` moves a run between `mm-tonne-s` and SI through `physical_units`, `to_zarr`/`from_zarr` round
  trip it (bit-identical at float64, documented at float32) and `checksum` digests the arrays and the metadata.
- `pinneapple_data.cae.build_training_set`: the normalisation between a run and a training set — dedup of repeated
  states, node decimation with the error it introduces measured against the nearest kept node, unit conversion,
  per-field non-dimensionalisation with the scales kept, the part hierarchy preserved, and a split by geometry family
  so that frames of one impact never land on both sides.
- `pinneapple_data.cae.read_vtk_series`/`write_vtk_series` and `read_lsdyna_directory`/`read_nodout`/`read_keyword_deck`:
  the readers behind those constructors, no `meshio` needed. `docs/guides/structural_ingestion.md` states the format
  coverage as a table — read, cross-checked, not supported — and the two-path agreement that was measured is printed
  by `tests/test_structural_run.py` (#1500).
