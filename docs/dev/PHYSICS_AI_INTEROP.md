# Open-source tools and datasets for Physics AI: what is integrated, what is not

This page is the status of the mapping "Mapeamento Open Source e Datasets para Physics AI" (13/11/2026). It records
what this repository can do **today**, with the test or the command that shows it, and what is still missing. A row
that says "not integrated" is a decision left open on purpose, not an omission: the catalogue names the format and
the homepage so the next step is a choice rather than a discovery.

Every claim below is checkable in this checkout:

```bash
pytest tests/pinneapple_data tests/test_arena_bundle_bench.py tests/test_arena_bundle_uq.py \
       tests/test_arena_serving_bench.py -q
pytest tests/pinneapple_veriphysics/test_surrogate_gates.py -q   # the model-side gates of section 6
PINNEAPPLE_NETWORK_TESTS=1 pytest tests/pinneapple_data/test_pdebench.py -q -k "real_file or large_file"
python examples/physics_data_schema/01_pdebench_bundle.py
```

## 1. The universal data contract (INT-001) — done

`pinneapple_data.schema` is one pydantic model, `pinneapple.physics-data/1`, for a physics dataset whoever produced
it, and `PhysicsDataBundle` is that model plus the arrays it references, saved as a directory (`bundle.json` +
`arrays.npz`) whose digest is recomputed and checked on load.

| Section | What it carries | What it refuses |
|---|---|---|
| `geometry` | kind, dimension, the geometry's own length unit, input files with SHA-256, regions, bounding box | a unit this library cannot parse; a bounding box of the wrong length |
| `mesh` | structured grid with named axes (coordinates + unit each) or nodes/cells + cell types, physical groups, quality | a grid with no axes, duplicate axis names, an unstructured mesh without nodes/cells |
| `physics` | `governing` as text with its symbols defined, boundary and initial conditions, parameters, materials, dimensionless numbers | a unit that does not parse |
| `fields` | per variable: unit, `nodal`/`cell`, components, the axes it indexes, dtype and shape | a field whose shape is not the shape of the axes it names |
| `provenance` | source, kind, adapter, solver, config, input files with SHA-256, **licence (required)**, citation, url | an empty source or an empty licence |
| `verification` | per check: id, tool, value, tolerance, unit, verdict; `passed=None` = "not run" | a verdict without a number |

Interop: `bundle.to_xarray()` returns a `Dataset` with the units in the attributes; `pack()`/`unpack()` ship one as
a `.zip`; `BundleManifest.model_json_schema()` gives the JSON Schema other languages and the REST API use.

Two decisions worth knowing:

* **`nodal` vs `cell` is not cosmetic.** A cell-centred field compared with a nodal reference is a half-cell shift,
  which shows up as a first-order error. The schema forces every field to declare which it is, and the `location`
  gate checks the shape against the mesh (both cell conventions that occur in the wild are accepted; a shape that
  fits neither is the error).
* **The licence is a required field.** A bundle whose licence is `unknown` fails the `provenance` gate, because that
  is exactly the bundle that cannot be redistributed.

## 2. Datasets (INT-002, INT-003) — five integrated, the rest catalogued

`pinneapple_data.datasets.catalog` holds the twelve public datasets of the mapping with domain, priority, formats,
access route, licence and — the field that matters — whether this library can read it:

```python
from pinneapple_data.datasets.catalog import entries, implemented, missing
[e.id for e in implemented()]      # ['cfdbench', 'flowbench', 'pdebench', 'spe10', 'the_well']
[e.id for e in missing("P0")]      # ['nasa_tmr']
```

`license_verified` is `True` only where the licence was read on the distribution page and checked: PDEBench's data
(CC BY 4.0, on the DaRUS record), CFDBench's (Apache-2.0 on the HuggingFace dataset card, which is what covers the
data) and FlowBench's (CC-BY-NC-4.0, on the card and in the paper — verified is **not** a commercial clearance). The
Well's package is BSD-3-Clause but its *data* terms are per scenario, so the adapter refuses to build a bundle without
an explicit `license=` and says where to look; SPE10 states no terms at all, so its reader requires the licence as an
argument.

### SPE10, CFDBench and FlowBench: three layouts, each checked against the real file

* **SPE10** — `spe_perm.dat` gives six values per line, three permeability components per cell, and no index at all.
  The reader's claim that ``i`` varies fastest is checked against the model's own structure: read that way the mean
  permeability per layer jumps by a factor of two at layer 36, where the Tarbert formation ends; read the other way
  the layers are statistically indistinguishable. Which of the three values is *kx*, *ky* or *kz* cannot be recovered
  from the numbers (the three columns have the same median, mean and spread), so the bundle names them in the SPE
  convention and says in its metadata that the order is a convention. The axes are cell centres in feet, top at
  12 000 ft, from the dataset's description.
* **CFDBench** — one directory per case with `u.npy` and `v.npy` shaped `(time, 64, 64)`, a `mask.npy` where the case
  has one, and `case.json` with the case's parameters (measured on the published `tube.zip`: `(107, 64, 64)` and
  `{height, width, vel_in, density, viscosity}`). The archive states **no units**, so every field is `unknown` and the
  `units` gate says so until the caller passes `units={...}`; the coordinates come from `case.json`'s extent when it
  has one and are the cell index otherwise, and the bundle's metadata says which.
* **FlowBench** — the LDC families store `X`/`Y` tensors shaped `(n_samples, n_channels, height, width)` whose channel
  lists are Table 12 of the paper, and an FPO sample stores `(n_time, height, width, 3)` with `u, v, p` **last**
  (measured: `(242, 256, 1024, 3)` for a 1.4 GB sample, `(1000, 3, 128, 128)` and `(1000, 4, 128, 128)` for the
  128 x 128 LDC pair). The reader names channels by their count and refuses a count the paper does not document,
  because the 128 x 128 file has the 3-channel input of one family and the 4-channel output of another.
* **NASA TMR** — no reader, and the reason is in the catalogue: on 2026-11-13 the legacy host answered every path
  with the same HTML page and the nasa.gov landing page the old URLs redirect to is the generic site shell, so there
  is no file whose layout could be checked. The entry says that instead of pointing at a link that looks alive.

A finding from turning on the three readers: the `units` gate **crashed** on an axis declared `unknown` (it parsed the
unit before deciding the axis had none). It now reports it as a problem, which is what an imported case with no
recorded length unit deserves.

### Reading a 1.3 GB file to get 65 kB of it

The files are 1.3-88 GB (PDEBench) and 2.8-21 GB per file (The Well). `pinneapple_data.datasets.remote_hdf5`
gives h5py a seekable file over HTTP (`driver="fileobj"`), so HDF5's own metadata B-trees decide what crosses the
network:

```python
from pinneapple_data.datasets import pdebench
bundle = pdebench.load_bundle(pdebench.darus_url(133150))     # Sod6.hdf5, 4.9 MB, read remotely
```

Measured (2026-11-13) by
`tests/pinneapple_data/test_pdebench.py::test_reads_one_sample_of_a_large_file_without_downloading_it`, behind
`PINNEAPPLE_NETWORK_TESTS=1`: **68 kB over 10 range requests, in 7 s, for one sample of the 1310.7 MB 2D Darcy flow
file**. The test asserts the cost stays under 4 MiB and 200 requests, so a regression that starts pulling whole
datasets fails instead of quietly getting slower. Files up to 64 MiB are fetched once and served from memory.

### What the first run on real data caught

`Sod6.hdf5` (201 snapshots, 1024 points, `density`/`pressure`/`Vx`) carries **202** time coordinates. Every reader
that trusts the coordinate silently mislabels the trajectory by one step. The adapter aligns the shorter of the two,
records the discrepancy in `metadata["pdebench"]["notes"]` and leaves a `time_alignment` check on the bundle. That
mismatch is the argument for the schema: it is a property of the data, not of any one model.

The PDEBench adapter handles both layouts the archive publishes: root datasets (`tensor`, `density`, `Vx`, ...) and
one group per sample (`0000/`, `0001/`, ...). Which leading axis is *samples* and which is *time* is not written in
the file, so the reader decides it arithmetically from the coordinate lengths, keeps the decision in
`handle.axis_flags`, and reads it the same way for every field.

## 3. Verification gates (INT-005) — the data half

`pinneapple_data.schema.run_gates(bundle)` decides seven checks with a number and a tolerance:

| Gate | Question | Decided on |
|---|---|---|
| `integrity` | do the references resolve and does the digest match? | SHA-256 over the manifest and every array |
| `units` | is every declaration present and dimensionally consistent? | 0 problems; spatial axes must share a dimension |
| `location` | does every field's shape fit its node/cell declaration? | count of misfits |
| `frame` | are the axes strictly monotonic and uniformly spaced? | `max|dx-mean| / mean`, tolerance 1e-5 **or the dtype's own rounding, whichever is larger** |
| `provenance` | source, licence, citation and input hashes? | count of missing pieces |
| `conservation` | does the integral of a conserved field drift? | `max_t|I(t)-I(0)| / max|I|` |
| `reference` | how far is the field from a reference array? | relative L2 |

The `frame` tolerance is measured, not guessed, and it follows the storage precision rather than a constant:
PDEBench stores coordinates as float32, and rounding a coordinate to float32 moves each neighbour difference by
~2·eps·|x|, which is 1.2e-4 of the spacing for `Sod6.hdf5`'s x on [-1, 1] (measured deviation: 1.1e-5). A fixed 1e-5
— the value the first version used — failed the real file for a reason that is the file format, not the geometry.
The gate now applies `max(1e-5, 2·eps·max|x| / mean(spacing))` for the axis's own dtype and says which of the two
it used. A float64 axis is still held to 1e-5; a genuinely stretched or shuffled axis fails in either dtype
(`tests/pinneapple_data/test_frame_gate_precision.py`).

A gate that cannot be asked — no time axis for conservation, no reference array, a sign-changing field whose
integral is zero to rounding — reports **"not run"**, and `gate.passed` is `None`. `GateReport.record(ledger)` seals
only the gates that ran into the VeriPhysics evidence ledger, so a report cannot say "passed" without the record.

Not done: the model-side gates of the mapping (residuals, boundary-condition satisfaction for a trained surrogate)
live in `pinneapple_veriphysics` and `pinneapple check`; this module gates the *data*.

## 4. Benchmark harness (INT-007) — done, reduced budget

`pinneapple_arena.bundle_bench` trains a PINN and an FNO on the same bundle, with the same split, optimiser budget
and seed, and scores both with relative L2 on held-out time steps:

```python
from pinneapple_arena.bundle_bench import BenchConfig, run_benchmark
report = run_benchmark(bundle, BenchConfig(field="density", epochs=200))
print(report.to_markdown())
```

Measured on the real archive (`python -m benchmarks.pinn_vs_fno_pdebench`, Sod6.hdf5 read over HTTP, 150 epochs,
201 snapshots, last 20 % held out): FNO **4.02e-3** vs PINN **1.48e-2** relative L2, 390 s and 70 s respectively, all
gates passed. `benchmarks/_out/pinn_vs_fno_pdebench-sod-0000.{md,json}` holds the report.

The protocol is a surrogate's real question — the first 80 % of the record in, the last 20 % scored — not a random
shuffle a smooth field would pass by interpolation. Three limits are written into the report: the PINN is
data-supervised unless a `residual=` callable is supplied (the library cannot turn `physics.governing` prose into an
equation), the budget is CPU-sized, and the result compares the two families on this data, not against published
numbers. The winner's field then goes through the same gates against the reference, so the report carries the
verification with the ranking.

## 5. CAE ⇄ bundle (INT-004) — a solver's result leaves as data

`pinneapple_data.schema.cae` closes the loop the CAE readers left open. `bundle_from_mesh(mesh, source=, license=)`
turns the neutral `pinneapple_data.cae.model.Mesh` that OpenFOAM, CalculiX, Abaqus, VTU and Gmsh readers already
produce into a bundle; `bundle_to_mesh` goes back; `export_bundle` writes VTU, HDF5, NPZ, Parquet, CSV, JSON or the
`.zip` bundle with the exporters that already exist.

What the reader had and the bundle now carries: the **licence** (asked for, never defaulted to `unknown`), the unit
per field, the point/cell split, the named patches and ELSETs as `physical_groups`, and the input files with their
SHA-256.

Two mesh shapes are kept as they are, because flattening either one loses the case:

* **mixed-cell meshes** — a volume of hexahedra beside a surface of triangles is the normal CalculiX result, so
  `Mesh` in the schema holds a list of `Block`s, each with its own element type, and `n_cells` counts only the
  non-boundary blocks (counting them made a hexahedron + two triangles report three volume cells);
* **polyhedral OpenFOAM meshes** — faces, owner/neighbour and patches survive as `Polyhedral`, not as an invented
  connectivity; the cavity case round-trips at 400 cells with its `movingWall`/`fixedWalls` patches and kinematic
  pressure (`p`, m²/s²) intact.

A unit the reader can only report as "model units (length)" (CalculiX without a deck-level unit) becomes the schema's
`unknown` sentinel and **fails the `units` gate with a message naming the field** — a verdict a reviewer needs,
rather than a conversion nobody declared. `bundle_from_mesh(units={"DISP": "mm"})` passes the override.

## 6. Model-side gates (INT-005, second half) — a surrogate asked the same questions

`pinneapple_veriphysics.surrogate_gates.run_surrogate_gates(predicted, ...)` takes a trained model's output and
decides four gates with a number each:

| Gate | Question | Decided on |
|---|---|---|
| `accuracy` | how far is it from the reference? | relative L2 against a tolerance (default 5e-2) |
| `conservation` | does the integral drift against the reference? | `max|I-I_ref| / max|I_ref|`, 1e-2 |
| `boundary` | does it satisfy the declared Dirichlet values on `x_low`/`x_high`? | max deviation from the boundary value, 5e-2 |
| `residual` | does it satisfy the PDE? | the residual of `advection`/`burgers`/`diffusion`/`wave` by finite differences, **normalised by the size of the operator's own terms**, 1e-1 |

The residual normalisation is the point: a raw residual of 1e-2 means nothing until it is divided by the terms the
equation is made of. Measured on a 200×400 grid with an exact solution, the finite-difference truncation alone
leaves a residual of 1.3e-3, which is why the default tolerance is 1e-1 and not 1e-6 — a tolerance tighter than the
discretisation's own error would fail correct models. Every gate that cannot be asked (no reference, no equation, a
model that returns only a field) reports **not run**, and `record(ledger, inputs_sha256=...)` seals what did run into
the VeriPhysics evidence ledger.

## 7. Operators, graphs, UQ, visualisation, serving (INT-008 … INT-011, INT-013)

* **Operator datasets (INT-008)** — `pinneapple_data.datasets.operator` turns bundles into
  `(initial condition, trajectory, parameters)` with `split()` **by trajectory**, which is the split that matters for
  an operator: splitting by time step leaks the answer into the training set. `leakage()` proves the three sides are
  disjoint; `channels()` adds one constant channel per parameter, which is how an FNO sees a varying coefficient.
  `from_pdebench(path, params=("Nu",))` builds one from the archive file directly.
* **Geometry → graph (INT-009)** — `pinneapple_data.graph.grid_graph` gives a structured bundle its nearest
  neighbours along each axis at one step; `mesh_graph` gives an unstructured one the unique edges of its element
  connectivity. The edges come from a per-element table (`hexahedron` 12, `tetra` 6, …), not from a closed ring:
  the first version connected `i` to `i+1` modulo the element and silently missed a hexahedron's four vertical
  edges, which a graph network would have had to rediscover. `to_torch_geometric` is the only function that imports
  PyG, so the graph is usable without it.
* **UQ and design search (INT-010)** — `pinneapple_arena.bundle_uq.seed_ensemble` trains the same architecture on
  the same split with different seeds and reports the held-out relative L2 per seed, the agreement between them and
  the pointwise spread divided by the field's own standard deviation. The report says what it is: a **disagreement
  between seeds, not a posterior**. `design_optimize` samples an input box by Latin hypercube, evaluates an
  objective and ranks the inputs by rank correlation — a rank statistic, so it measures monotonicity, not magnitude,
  and the note says so.
  The Latin hypercube had the bug the test now pins: one permutation shared by every column lines the strata up and
  makes the inputs correlated by construction, so an input the objective ignores still came back correlated. Each
  column gets its own permutation (`test_an_input_the_objective_ignores_has_no_rank_correlation`).
* **Visualisation (INT-011)** — `pinneapple_data.schema.view.to_pyvista` gives a `StructuredGrid` for a grid and an
  `UnstructuredGrid` for a mesh, with nodal fields as point data and cell fields as cell data; `save_vtu` writes one
  snapshot, `save_series` writes the `.pvd` + per-step `.vtu` pair ParaView opens as an animation. Without PyVista
  the unstructured path still writes a VTU through the CAE exporter — the fallback is tested with `available()`
  forced to false, because a fallback nobody runs does not work. A structured grid cannot write `.vtu` (only `.vts`),
  so it is cast; and because the schema's cell fields carry one value per axis entry and not one per cell, such a
  field lands as point data **with a note in the file** instead of being misindexed.
* **Serving (INT-013)** — `pinneapple_arena.serving_bench.benchmark_model` times eager PyTorch (p50/p95, warm-up,
  throughput), exports ONNX and measures `onnxruntime` on the same input, reporting the max absolute difference
  between the two backends as a serving smoke check. It tries **three exports in order**, because they do not cover
  the same models: the TorchScript exporter; the graph (dynamo) exporter, which needs `onnxscript`; and, when both
  refuse, the model rewritten in real arithmetic by
  `pinneapple_neural.architectures.neural_operators.fno_export.to_exportable`.
  The spectral case is why: an FNO's convolution keeps **complex tensors**, which the TorchScript exporter never
  lowers (`aten::fft_rfft`) and the graph exporter lowers only while the output is real
  (`No decompositions registered for the complex-valued input` otherwise). The rewrite replaces the FFT with dense
  cosine/sine transforms -- the same weights, the same function to **1.5e-08** (1-D) and **3.0e-08** (2-D) in
  PyTorch, and to 2.98e-08 through onnxruntime, measured on a small 1-D FNO at **0.190 ms in PyTorch vs 0.144 ms
  served**. `ServingReport.exporter` says which attempt produced the graph, and the report says what that means: the
  graph is the real-arithmetic copy, not the FFT one, and its input size is fixed.
  A pointwise surrogate (a PINN MLP) takes the first attempt and serves normally. The `onnxruntime` leg is not
  measured on a machine without it, and the report says that too.

## 8. PhysicsNeMo bridge (INT-014) — the dictionary, not the package

PhysicsNeMo's datapipes yield a plain dictionary of named tensors (`yield {"permeability": ..., "darcy": ...}`,
read from its own `Darcy2D` datapipe), so `pinneapple_data.schema.physicsnemo.to_dict(bundle)` returns exactly that,
one key per field plus `pos` when coordinates are wanted, and `from_dict` goes back. `BundleDictDataset` is the
iterable a loop consumes (torch tensors when torch is installed).

`to_physicsnemo_dataset` returns a real `Datapipe` subclass **when the package is installed** and raises an
`ImportError` naming `nvidia-physicsnemo` when it is not. PhysicsNeMo is a multi-gigabyte CUDA-first dependency; this
repository converts the data and tests the conversion, and does not claim to have run a training loop it did not run.

## 9. Engineering API (INT-006, INT-012) — catalogue, schema, capability records, telemetry

In `pinneapple-engineering-api` (branch `feat/physics-data-interop`):

* `GET .../physics-data/external-catalog` — the catalogue, filtered by `priority`, `domain`, `readable`, with the
  library identity it was read from;
* `GET .../physics-data/schema` — the JSON Schema of a bundle and the gates with their default tolerances;
* **one capability record per catalogued dataset** (INT-006), merged into `registry.capabilities()`: status is
  `tested` where an adapter exists and `discovered` otherwise, `executable` is always `false` and the evidence names
  the adapter path. A catalogue entry never becomes `validated` because a record exists;
* `POST .../physics-data/telemetry/{twin_id}` and `GET` on the same path (INT-012) — a batch of sensor readings of a
  digital twin becomes a bundle (the union of the timestamps as the time axis, one field per sensor, **missing
  readings as NaN**), is published to immutable storage and registered as a dataset version through the code that
  already does that. Two digests come back and mean different things: `digest` is what is stored (SHA-256 of the
  `.zip`), `bundle_digest` is the content digest of the numbers. A reading is provenance `kind="experiment"`, not
  `simulation`. Two sensors that disagree about their unit are a 422, a repeated `(sensor, timestamp)` is a 422, and
  the same batch sent twice returns the first version instead of a conflict. An API pinned to a PINNeAPPle older than
  the schema answers `503` with the reason.

Three behaviours are tested on purpose: an older pin answers `available: false` **with the reason** (an empty list
would read as "there are no datasets"), an unverified licence is passed through unchanged, and the telemetry route
refuses rather than storing a bundle it cannot describe.

## 10. Native execution backends (INT-015) — one C ABI, C today

Python stays the scientific interface, PINNeAPPle the orchestration layer, and compiled code an execution backend
behind a checked boundary. A backend implements four functions of
`pinneapple_core/native/include/pinneapple_native.h` — `pp_abi_version`, `pp_build`, `pp_ops`, `pp_call` — and this
package calls it with `ctypes`. That header is the whole integration surface, so C, C++ (`extern "C"`), Rust
(`extern "C"`), Fortran (`iso_c_binding`) and Zig are all first-class without a bridge per pair of languages, which
is the combinatorial explosion the alternative has.

```python
from pinneapple_core.native import NativeKernels, native_status

native_status()          # {'enabled': True, 'compiler': '/usr/bin/cc', 'available': True,
                         #  'library': '~/Library/Caches/pinneapple/native/<hash>/libpinneapple_native.dylib', ...}
k = NativeKernels()
k.backend                # 'native:libpinneapple_native.dylib', or 'reference:numpy' with the reason
k.jacobi_2d(u0, f, steps=2000, omega=2 / 3)
```

**The NumPy implementation is the definition, not the fallback.** `laplacian_2d_reference`, `heat_step_2d_reference`
and `jacobi_2d_reference` are what the kernel *means*; the C version exists to be faster at the same arithmetic, and
the tests compare them element by element. A machine without a compiler, or a run with `PINNEAPPLE_NATIVE=0`, gets the
reference **and the sentence that says why** — never a wrong number, and never a result attributed to a backend that
did not run.

| Operation | What it computes | Boundary |
|---|---|---|
| `laplacian_2d(u; nx, ny, h)` | five-point Laplacian, one pass | zero Dirichlet, the input's boundary ring is never read |
| `heat_step_2d(u; nx, ny, alpha, dt, steps)` | `steps` explicit steps of `u_t = alpha * laplacian(u)` on the unit square | zero Dirichlet; the scheme's limit is enforced, not blown up |
| `jacobi_2d(u0, f; nx, ny, h, steps, omega)` | `steps` damped-Jacobi sweeps for `-laplacian(u) = f` | zero Dirichlet; `omega` in `(0, 1]` |

Measured on this machine (`python -m benchmarks.native_backend`, best of 7, Apple clang 21 / arm64, NumPy 2.4.6,
Python 3.11.16) — `benchmarks/_out/native_backend.{md,json}` holds the run and every sample behind it:

| grid | operation | C (ms) | NumPy (ms) | speedup | max abs difference |
|---|---|---|---|---|---|
| 129 × 129 | `laplacian_2d` | 0.035 | 0.105 | 3.0× | 0.0 |
| 129 × 129 | `heat_step_2d` (50 steps) | 0.979 | 8.169 | 8.3× | 3.3e-16 |
| 129 × 129 | `jacobi_2d` (200 sweeps) | 4.854 | 160.539 | 33.1× | 6.2e-17 |
| 513 × 513 | `laplacian_2d` | 0.198 | 1.453 | 7.3× | 0.0 |
| 513 × 513 | `heat_step_2d` (50 steps) | 14.841 | 597.507 | 40.3× | 4.4e-16 |
| 513 × 513 | `jacobi_2d` (200 sweeps) | 402.921 | 3057.864 | 7.6× | 3.9e-18 |

**Read the agreement column, not the speedup column.** The agreement is a property of the code: 0.0 for the Laplacian
(the same expression, in the same order, so the two backends produce identical bits) and machine epsilon elsewhere. The
speedup is a property of this machine at this moment, and it moved by factors of two to three between the runs
recorded in the report — the C column repeated to within tens of percent while NumPy's own time swung further, because
other work was running on the machine. The number worth keeping is the shape of it: **single-digit speedup on the
one-pass kernel, one to two orders of magnitude on the iterated ones**, and the Jacobi sweep falls from 33× to 7.6×
between the two grids because at 513 × 513 each sweep touches three 2 MB arrays and the kernel stops being
compute-bound. No amount of C changes that.

**Refusals, each one a way a native layer goes wrong:**

* a library reporting another `pp_abi_version` is **refused by name** before any call (`AbiVersionMismatch`) — a
  struct-layout mismatch is not something to discover from a wrong number;
* a library that kept the operation names and changed the signatures is refused too, because `pp_ops` is compared
  against the loader's specs, not merely checked for the presence of the names. A test compiles a fake backend that
  renames the signatures and asserts the refusal;
* nothing is allocated across the boundary: the caller owns the output buffer and passes its capacity, so there is no
  `free` whose allocator might differ;
* a non-contiguous or `float32` array is made contiguous `float64` **before** crossing — passing a strided buffer to C
  is the classic silent corruption, and a test feeds a Fortran-ordered array through the kernel.

Two findings from the first run, both kept because they are the reason the checks exist. Damped Jacobi with
`omega = 1.9` **overflowed to `inf`** in 2000 sweeps on a 33 × 33 grid: the iteration matrix has eigenvalues
`1 - omega(1 - mu)` with `mu` up to `cos(pi*h)`, so above 1 a modulus exceeds 1. The range in the contract is now
`(0, 1]` with that reason in the message, instead of a number the caller has to diagnose. And the manufactured solution
`u = sin(pi x) sin(pi y)` caught a **sign error** in the Jacobi right-hand side immediately — the result came back as
`-u`, an error of 2.0 where the exact field is 1.0, which no per-element comparison between C and NumPy could have
found because both implementations had it. After the fix both backends land on the stencil's own truncation error,
`pi^2 h^2 / 12 = 8.03e-4` on a 33 × 33 grid, measured **8.04e-4**: the answer is right, not merely self-consistent.

## 11. Backlog status

| ID | Item | Status |
|---|---|---|
| INT-001 | Universal physics data schema | **done** — `pinneapple_data.schema` |
| INT-002 | Dataset registry + PDEBench | **done** — `datasets.catalog`, `datasets.pdebench` |
| INT-003 | The Well adapter | **done** — `datasets.the_well` (layout reproduced in tests; licence per scenario) |
| INT-004 | CAE universal exporter | **done** — `schema.cae` (mixed-cell and polyhedral meshes, OpenFOAM and CalculiX cases) |
| INT-005 | VeriPhysics integration gates | **done** — `schema.gates` (data) + `pinneapple_veriphysics.surrogate_gates` (model) |
| INT-006 | Capability registry sync | **done** — one record per catalogued dataset, never above `tested` |
| INT-007 | Benchmark harness | **done** — `pinneapple_arena.bundle_bench` |
| INT-008 | NeuralOperator adapter | **done for the data side** — `datasets.operator`; the FNO itself is the arena's |
| INT-009 | Geometry → graph pipeline | **done** — `pinneapple_data.graph`; FlowBench, read here, turned out to be structured grids (its "triangulated domains" were a wrong note in the catalogue, now corrected), so the graph adapters cover everything the archives publish |
| INT-010 | UQ + design optimisation | **done** — `bundle_uq.seed_ensemble`, `bundle_uq.design_optimize` |
| INT-011 | Scientific visualisation adapters | **done** — `schema.view` (PyVista, VTU, PVD) |
| INT-012 | Digital twin telemetry bridge | **done** — telemetry routes in the API, bundles through the same storage path |
| INT-013 | ONNX serving benchmarks | **done** — three export attempts; an FNO's complex tensors are served from the real-arithmetic copy of its spectral layers (agreement 3e-08) |
| INT-014 | PhysicsNeMo training bridge | **done for the data contract** — `schema.physicsnemo`; the package itself is not installed here |
| INT-015 | Native execution backends | **done for C** — `pinneapple_core.native`: a versioned C ABI, ctypes loader, three kernels in C and NumPy, agreement at machine epsilon, 3-93x measured; Rust and Fortran speak the same header and are not built here |

What is *not* done, said plainly: no pretrained PhysicsNeMo model runs in this repository, the ONNX runtime leg is
measured only where `onnxruntime` is installed, an FNO is served from the real-arithmetic copy of its spectral layers
rather than from its own FFT graph (same outputs, fixed input size), and none of the imported archives is
redistributed — the adapters read them where they live.

## 11. Limitations of this page

Licences were verified only where `license_verified` says so. The Well's reader is exercised on a file written to
the layout its own loader reads, not on a downloaded multi-gigabyte scenario. The benchmark numbers are a reduced
CPU budget. The `onnxruntime` comparison was measured in a throwaway environment, not in this checkout's default
interpreter. No adapter here re-implements a solver: they read what the archive published.

The native layer (section 10) is measured for **one** backend on **one** machine: the C kernels, built by this
machine's Apple clang, single-threaded and CPU-only. A Rust or Fortran backend speaks the same header and would be
loaded by the same ctypes code, but neither is built in this repository, so nothing is claimed for them — and there is
no speedup claim on a machine whose compiler differs from the one recorded in the report. The three kernels are the
same five-point stencil, not a survey of what a native layer can accelerate: sampling, mesh operations and linear
algebra are the next candidates, and each one moves the same way.

## References

- PDEBench: Takamoto et al. (2022), NeurIPS Datasets and Benchmarks, arXiv:2210.07182; data DOI
  [10.18419/darus-2986](https://doi.org/10.18419/darus-2986) (CC BY 4.0).
- The Well: Ohana et al. (2024), NeurIPS Datasets and Benchmarks; code BSD-3-Clause; data per scenario at
  <https://huggingface.co/polymathic-ai>.
- Sod (1978), *A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws*,
  J. Comput. Phys. 27(1), 1-31 — the case `Sod6.hdf5` solves.
- Li et al. (2021), *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR — the family used
  in the harness.
- HDF5 file driver `fileobj`: <https://docs.h5py.org/en/stable/high/file.html>.
- Krumbiegel et al. (2024), *PhysicsNeMo* (formerly Modulus), NVIDIA — the datapipe contract the bridge follows:
  <https://github.com/NVIDIA/physicsnemo>.
- PyVista and the VTK XML formats (`UnstructuredGrid`, `.vtu`, `.pvd`): <https://docs.pyvista.org>.
- ONNX export of PyTorch models and the opset operator list: <https://pytorch.org/docs/stable/onnx.html>.
- PyTorch Geometric `Data`, the object `to_torch_geometric` produces: <https://pytorch-geometric.readthedocs.io>.
