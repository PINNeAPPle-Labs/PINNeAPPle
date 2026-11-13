# Open-source tools and datasets for Physics AI: what is integrated, what is not

This page is the status of the mapping "Mapeamento Open Source e Datasets para Physics AI" (13/11/2026). It records
what this repository can do **today**, with the test or the command that shows it, and what is still missing. A row
that says "not integrated" is a decision left open on purpose, not an omission: the catalogue names the format and
the homepage so the next step is a choice rather than a discovery.

Every claim below is checkable in this checkout:

```bash
pytest tests/pinneapple_data/test_schema_bundle.py tests/pinneapple_data/test_dataset_catalog.py \
       tests/pinneapple_data/test_pdebench.py tests/pinneapple_data/test_the_well.py \
       tests/test_arena_bundle_bench.py -q
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

## 2. Datasets (INT-002, INT-003) — two integrated, the rest catalogued

`pinneapple_data.datasets.catalog` holds the twelve public datasets of the mapping with domain, priority, formats,
access route, licence and — the field that matters — whether this library can read it:

```python
from pinneapple_data.datasets.catalog import entries, implemented, missing
[e.id for e in implemented()]      # ['pdebench', 'the_well']
[e.id for e in missing("P0")]      # ['cfdbench', 'spe10', 'nasa_tmr']
```

`license_verified` is `True` only where the licence was read on the distribution page and checked: PDEBench's data
(CC BY 4.0, on the DaRUS record). The Well's package is BSD-3-Clause but its *data* terms are per scenario, so the
adapter refuses to build a bundle without an explicit `license=` and says where to look.

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
| `frame` | are the axes strictly monotonic and uniformly spaced? | `max|dx-mean| / mean`, tolerance 1e-5 |
| `provenance` | source, licence, citation and input hashes? | count of missing pieces |
| `conservation` | does the integral of a conserved field drift? | `max_t|I(t)-I(0)| / max|I|` |
| `reference` | how far is the field from a reference array? | relative L2 |

The 1e-5 on `frame` is measured, not guessed: PDEBench stores coordinates as float32, and a 64-point linspace in
float32 already differs from perfectly uniform by ~2e-6. A gate that fired on every real dataset would be noise.

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

The protocol is a surrogate's real question — the first 80 % of the record in, the last 20 % scored — not a random
shuffle a smooth field would pass by interpolation. Three limits are written into the report: the PINN is
data-supervised unless a `residual=` callable is supplied (the library cannot turn `physics.governing` prose into an
equation), the budget is CPU-sized, and the result compares the two families on this data, not against published
numbers. The winner's field then goes through the same gates against the reference, so the report carries the
verification with the ranking.

## 5. Engineering API (INT-006) — catalogue and schema served

In `pinneapple-engineering-api` (branch `feat/physics-data-interop`):

* `GET /v1/platform/workspaces/{workspace_id}/physics-data/external-catalog` — the catalogue, filtered by
  `priority`, `domain`, `readable`, with the library identity it was read from;
* `GET /v1/platform/workspaces/{workspace_id}/physics-data/schema` — the JSON Schema of a bundle and the gates
  with their default tolerances.

Two behaviours are tested on purpose: an older pin answers `available: false` **with the reason** (an empty list
would read as "there are no datasets"), and an unverified licence is passed through unchanged.

## 6. Backlog status

| ID | Item | Status |
|---|---|---|
| INT-001 | Universal physics data schema | **done** — `pinneapple_data.schema` |
| INT-002 | Dataset registry + PDEBench | **done** — `datasets.catalog`, `datasets.pdebench` |
| INT-003 | The Well adapter | **done** — `datasets.the_well` (layout reproduced in tests; licence per scenario) |
| INT-004 | CAE universal exporter | not started — `pinneapple_data.cae.io` (OpenFOAM/FRD/INP readers) is the natural source of bundles |
| INT-005 | VeriPhysics integration gates | **data gates done**; the model-side gates remain in `pinneapple_veriphysics` |
| INT-006 | Capability registry sync | **partly** — catalogue + schema endpoints; no registry entry per dataset yet |
| INT-007 | Benchmark harness | **done** — `pinneapple_arena.bundle_bench` |
| INT-008 | NeuralOperator adapter | not started — the bundle is the input contract it would consume |
| INT-009 | Geometry → graph pipeline | not started — FlowBench (triangulated domains) is the dataset it needs |
| INT-010 | UQ + design optimisation | not started |
| INT-011 | Scientific visualisation adapters | not started — `to_xarray()` is the first half |
| INT-012 | Digital twin telemetry bridge | not started |
| INT-013 | ONNX serving benchmarks | not started |
| INT-014 | PhysicsNeMo training bridge | not started |

The sequence the mapping recommends (INT-001 → INT-002 → INT-007 → INT-005 → INT-006) is the one that is done; the
next honest step is INT-004, because every solver in this repository already produces fields that cannot yet leave
it as a bundle.

## 7. Limitations of this page

Licences were verified only where `license_verified` says so. The Well's reader is exercised on a file written to
the layout its own loader reads, not on a downloaded multi-gigabyte scenario. The benchmark numbers are a reduced
CPU budget. No adapter here re-implements a solver: they read what the archive published.

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
