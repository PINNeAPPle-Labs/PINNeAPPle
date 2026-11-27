- **The core API snapshot catches up with the structural CAE ingestion**: `api/core.json` did not carry the 77 names
  that `pinneapple_data.cae` gained in #1500/#2494 -- `StructuralRun`, `TrainingSet`, `build_training_set`,
  `read_vtk_series`, `write_vtk_frame`, the OpenRadioss (`lsdyna`) readers, the `vtk_series` and `structural` modules
  -- nor the two signature changes that came with them (`pinneapple_data.cae.frd.stress` gained `until=None`, and
  `Trainer.__init__` gained `constraints=` from #221), so `tests/test_deprecation_and_api.py` failed on main. Written
  with `python -m pinneapple_stability.api --write`, which is the command the test names; added: 77, removed: 0,
  changed: 2.
