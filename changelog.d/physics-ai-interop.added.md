- **Universal physics data schema and the first external dataset adapters** (`pinneapple_data.schema`,
  `pinneapple_data.datasets.pdebench`, `pinneapple_data.datasets.the_well`). One pydantic manifest
  (`pinneapple.physics-data/1`) describes a physics dataset -- geometry, mesh axes with units, fields with their
  node/cell location, the governing physics, provenance with the licence and the SHA-256 of every input, and the
  checks that ran -- and `PhysicsDataBundle` saves it as a directory (`bundle.json` + `arrays.npz`) whose digest is
  re-checked on load, so an edited file no longer matches. `run_gates` decides seven checks arithmetically
  (integrity, units, location, frame, provenance, conservation, reference) and can seal each verdict into the
  VeriPhysics evidence ledger; a gate that cannot be asked reads "not run", never "passed".
- **PDEBench and The Well read without downloading them**: `HTTPRangeFile` gives h5py a seekable file over HTTP, so
  one sample of a 1.3-88 GB PDEBench file costs megabytes (measured on the real archive in the tests, behind
  `PINNEAPPLE_NETWORK_TESTS=1`). The PDEBench reader handles both layouts the archive publishes (root `tensor`
  arrays and per-sample groups), canonicalises `Vx`/`Vy` into one velocity field, and aligns a time coordinate that
  is one step longer than its data -- the mismatch the first run on real data caught.
- **`pinneapple_data.datasets.catalog`**: the twelve public datasets of the Physics AI mapping with domain,
  priority, format, access, licence (marked verified only where it was checked) and whether this library can
  actually read it -- `implemented()` returns two, and the rest name the work.
- **`pinneapple_arena.bundle_bench`**: an equal-budget PINN vs FNO comparison on a bundle (same split, optimiser
  budget and seed; held-out time steps; relative L2), gated by the same checks, with a markdown report and an
  optional ledger record per model.
- Example `examples/physics_data_schema/01_pdebench_bundle.py` runs the path end to end offline.
