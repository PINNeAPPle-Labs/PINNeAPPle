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
- **CAE ⇄ bundle** (`pinneapple_data.schema.cae`, INT-004): a solver's neutral `Mesh` -- mixed element blocks and
  OpenFOAM's face-based polyhedral meshes included -- becomes a bundle and goes back, with `export_bundle` writing
  VTU, HDF5, NPZ, Parquet, CSV, JSON or the `.zip` bundle. The licence is asked for, never defaulted; a unit the
  reader can only call "model units" becomes `unknown` and fails the `units` gate naming the field.
- **Model-side gates** (`pinneapple_veriphysics.surrogate_gates`, INT-005): accuracy against a reference,
  conservation drift, Dirichlet satisfaction on the boundary, and the PDE residual of advection/burgers/diffusion/
  wave by finite differences, normalised by the size of the operator's own terms -- with the measured truncation
  floor (1.3e-3 on a 200x400 grid) behind the default tolerance. Gates that cannot be asked read "not run".
- **Operator datasets, graphs, UQ, visualisation, serving** (INT-008 … INT-011, INT-013): `datasets.operator`
  (initial condition -> trajectory, split by trajectory, with a leakage check), `pinneapple_data.graph`
  (`grid_graph`/`mesh_graph` with a per-element edge table and an optional `to_torch_geometric`),
  `pinneapple_arena.bundle_uq` (seed ensembles reporting a disagreement, not a posterior; Latin-hypercube design
  search with rank-correlation sensitivity), `pinneapple_data.schema.view` (PyVista datasets, VTU, `.pvd` series),
  and `pinneapple_arena.serving_bench` (PyTorch latency, ONNX export, `onnxruntime` comparison with the agreement
  between backends -- an FNO's `fft_rfft` does not lower to ONNX at opset 17, and the report says so).
- **PhysicsNeMo bridge** (`pinneapple_data.schema.physicsnemo`, INT-014): a bundle as the dictionary of named
  tensors its datapipes yield, plus the inverse; the `Datapipe` subclass is built only when the package is
  installed, and its absence is an `ImportError` that names what to install.
- **An FNO reaches ONNX after all** (`pinneapple_neural.architectures.neural_operators.fno_export`,
  `pinneapple_arena.serving_bench`): a spectral model keeps complex tensors, which neither PyTorch exporter lowers --
  the TorchScript one refuses `aten::fft_rfft`, the graph one refuses the complex-valued input once the spectrum is
  multiplied. `RealSpectralConv1d` joins the 2-D equivalent, so `to_exportable` rewrites either FFT as dense
  cosine/sine transforms with the same weights, and `benchmark_model` now tries TorchScript, then the graph exporter,
  then that rewrite, saying which one produced the graph. Measured: the copy agrees with the model to 1.5e-08 (1-D)
  and 3.0e-08 (2-D) in PyTorch and to 2.98e-08 through onnxruntime, where a 1-D FNO serves at 0.144 ms against
  0.190 ms in PyTorch.
