# Structural ingestion: a solver run as one object, and the normalisation that makes it a training set

A structural solver does not hand back a dataset. It hands back a *run*: a mesh, a sequence of instants, parts, fields
with units, and the circumstances of the computation. This page is the contract for turning that into
`pinneapple_data.cae.StructuralRun`, and then into a training set — and it is equally a statement of which formats were
**actually exercised**, because a reader of a proprietary result file that has only ever been tested against its own
assumption of the layout is not a reader.

## The contract, in one sentence

A solver's structural output becomes frames of one `Mesh` sharing one connectivity, a time axis in the run's time
unit, named parts holding the element indices each one owns, a unit per field, a provenance record, and the masks a
surrogate needs — eroded elements, the contact force, the fixed nodes.

| member | type | what it is |
|---|---|---|
| `frames` | `list[Mesh]` | the deformed geometry and the fields of each instant; nodal fields in `point_data`, elemental in `cell_data` |
| `times` | `(F,)` float64 | the instants, in `units["time"]` |
| `parts` | `list[Part]` | id, name, **element indices** (the same indexing as `cell_data`), kind, shell thickness |
| `units` | `dict[str, str]` | the unit of every field, never guessed — a solver's numbers are only meaningful with the system its deck was written in |
| `provenance` | `dict` | solver, library version, commit, data hash, mesh, case; `pinneapple_data.provenance.build_provenance` fills it and `require` validates it |
| `eroded` | `(F, m)` bool | the element is deleted at that frame |
| `contact_force` | `(F, 3)` | the force on the body |
| `fixed_nodes` | `(n,)` bool | the nodes the deck held (a mask, or the index list the crash solver takes) |
| `dropped`, `info` | `dict` | what a source could not carry or a step removed |

Constraints the constructor enforces: every frame has the same point and cell counts, `times` has one entry per frame,
and the masks have the shape of the run. A run whose frames disagree is refused — frames of different meshes are
different runs.

## Format coverage

**Read** means the reader exists and produces a `StructuralRun`. **Cross-checked** means the same run was read through
two paths and the object compared, with the discrepancy printed. Nothing in this table claims more than what was run.

| format | what it gives | status | how it was checked |
|---|---|---|---|
| legacy VTK ASCII time series (`UNSTRUCTURED_GRID`, `POLYDATA`; points, cells, cell types, `POINT_DATA`/`CELL_DATA` scalars, vectors and fields, `TIME`, `NODE_ID`/`PART_ID`) | everything the file carries: geometry, nodal and elemental fields, part ids, times | **read** via `read_vtk_series` / `StructuralRun.from_vtk_series`, **cross-checked** | a real run of this library's explicit crash solver was written as a series and read back: worst relative discrepancy over every field **3.1e-10** (`examples/structural_ingestion`, `tests/test_structural_run.py`) |
| OpenRadioss staged case (`frames/frame_000.vtk`, `frames/disp_XXX.txt`, `frames/time_XXX.txt` — what `external_solvers.openradioss.runner` leaves on disk) | geometry and parts from the full frames, displacement of every step from the reduced dumps, elemental fields only on the steps that kept a full frame (`info["fields_partial"]` says which) | **read** via `StructuralRun.from_openradioss`, **cross-checked** | the same run laid out as that directory and read back: displacement relative discrepancy **1.7e-07**, times to 1e-09 |
| OpenRadioss native ANIM (the binary `A001`-style files, before `anim_to_vtk`) | — | **not supported** | writing a native ANIM reader needs the vendor's binary layout checked against a real conversion; no OpenRadioss build is available on the machines this was written on, and an unexercised parser of an unpublished binary layout is exactly what this page refuses to call support. The VTK route above is the one that is exercised |
| LS-DYNA keyword deck (`*NODE`, `*ELEMENT_SHELL`/`*ELEMENT_SOLID`, `*PART`, `*SECTION_SHELL`/`*SECTION_SOLID`, `*CONTROL_TERMINATION`) | geometry, connectivity, part names, shell thickness, the termination time | **read** via `read_keyword_deck`, **layout-tested only** | parsed against fixtures built from the documented card layout (free format, comma separated, and the 8/16-character field widths); no LS-DYNA run was made, so this row claims the card layout, not a vendor file |
| LS-DYNA `nodout` (ASCII) | the time axis, nodal displacement and velocity | **read** via `read_nodout` / `StructuralRun.from_lsdyna`, **layout-tested only** | columns are taken from the labels the file prints (`x-disp`, `y-velocity`, …) with the documented order as the fallback; fixtures only, no vendor run |
| LS-DYNA `d3plot` (binary result database) | node coordinates and displacement, connectivity and part ids, effective stress and plastic strain, the eroded state, state times, the header unit system | **not supported** | it is a proprietary binary database and there is no LS-DYNA licence here, so a reader would be tested against its own assumption of the layout and nothing else. #1500 forbids claiming it from layout tests alone; it needs a golden fixture from a licensed run, and until then the ASCII half above is what exists |
| LS-DYNA `glstat` (energies) | kinetic, internal, hourglass and sliding energy histories | **not supported** | the ASCII layout differs between releases and no run was available to fix the columns against. `StructuralRun.from_lsdyna` records this in `info["unsupported"]` so a run built from that route cannot be mistaken for a complete one |
| LS-DYNA `elout` (element stress and plastic strain) | the element fields a stress surrogate needs | **not supported** | same reason; a run from the ASCII route therefore carries displacement and velocity and no stress |
| Zarr (this library's own store) | the whole run | **read and written** via `to_zarr`/`from_zarr`, **cross-checked** | bit-identical at `float64` (`checksum()` matches); at `dtype="float32"` the displacement differs by 4.1e-08 and the store says so in `info["stored_as"]` |

A run built from a partial source is not padded: a field the source did not carry is *absent* from `fields`, and a
frame that did not carry a field the run does have is `nan` — never a zero that would train as a measurement.

## Reading a run

```python
from pinneapple_data.cae import StructuralRun

run = StructuralRun.from_crash_result(result, part_names={1: "front", 2: "rear"}, units="SI", case="coupon_07")
run = StructuralRun.from_vtk_series("case/frames", units="mm-tonne-s")
run = StructuralRun.from_openradioss("runs/case_07", units="mm-tonne-s")
run = StructuralRun.from_lsdyna("runs/case_07", units="mm-tonne-s")     # deck + nodout; no stress

run = run.to_units("SI")                     # through pinneapple_data.physical_units — no unit arithmetic here
run.checksum()                               # sha256 over the arrays and the metadata
run.to_zarr("case_07.zarr")                  # StructuralRun.from_zarr("case_07.zarr")
run.to_vtk_series("out/", precision="single")
print(run.summary())                         # frames, elements, parts, fields, units, provenance, what was dropped
```

Units are never inferred. A consistent system is chosen from `UNIT_SYSTEMS` (`SI`, `m-kg-s`, `mm-tonne-s`,
`mm-kg-ms`) or given field by field, and `to_units` is the only path between systems. A run written in `mm-tonne-s`
and the same run in `m-kg-s` are the same object after one call — `tests/test_structural_run.py` checks exactly that.

## From a run to a training set

`pinneapple_data.cae.build_training_set` performs the five steps between a run and a dataset, and reports each one in
`TrainingSet.dropped` and `TrainingSet.scales`:

1. **dedup** — a state identical to the one before it is dropped, and counted;
2. **decimation** — elements are selected by a farthest-point spread over their centroids until their corners fit the
   node budget, so the kept mesh is a part of the body and not a scattering of nodes no element can be built on. The
   error is *measured*: every dropped node's displacement against its nearest kept node, as a relative L2 per field.
   `keep_cells=False` asks for a point cloud instead, and the log says the elemental fields were dropped;
3. **units** — every run is brought to one system;
4. **non-dimensionalisation** — per field, divided by one reference taken over the set, with the reference kept in
   `scales`. Geometry shares the displacement's length scale, so `frame.points == undeformed + displacement` survives;
5. **the split holds out a geometry family, not a frame** — `TrainingSet.split` names whole families, and the test
   asserts that no family appears on two sides. Frames of one impact share a mesh, a material and a design point;
   splitting them at random leaks the answer across the split and reports a score that means nothing.

The part hierarchy survives all five: each normalised run keeps its parts with the element indices remapped to the
thinned mesh.

## What is exercised, and where

```
PYTHONPATH=$PWD python3.11 -m pytest tests/test_structural_run.py -q
PYTHONPATH=$PWD python3.11 examples/structural_ingestion/run.py
```

The test file runs a real impact of the library's own explicit crash solver (63 nodes, 24 hexahedra, six frames) and
checks the contract, the two-path agreement, the unit equivalence, the Zarr round trip, the normalisation and the
LS-DYNA ASCII readers. The example runs a three-design campaign at coupon scale and prints the discrepancy of each
path. Neither of them runs OpenRadioss or LS-DYNA: the numbers they quote belong to this library's solver and to the
formats' own storage precision.

## Not done yet, and named

- the native OpenRadioss ANIM reader and the cross-check of `anim_to_vtk` against it on a real case (needs an
  OpenRadioss build);
- the LS-DYNA `d3plot` reader and its check against the `nodout`/`elout`/`glstat` companions of one run (needs a
  licensed run to produce the pair);
- `glstat` and `elout`;
- the node-by-node ingestion of a campaign into a single sharded Zarr store (the per-run round trip exists;
  `pinneapple_data.zarr_shards` is where the campaign-scale store lives).

The issue #1500 stays open on those four.
