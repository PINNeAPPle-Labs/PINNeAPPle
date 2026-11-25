# The physics-AI workflow, stage by stage: open tools and public datasets

This page is the companion of `PHYSICS_AI_INTEROP.md`. That one is organised by integration (INT-001 … INT-015) and
answers *"what does this repository do today?"*. This one is organised by **stage of the workflow** and answers the
question a person building a pipeline asks first: **at this step, which open tools exist, which public datasets
feed it, and how far does this repository already get?**

The map is not a document first. It is `pinneapple_catalog.stages`, and this page is generated from it, so a row here
and a row in the code cannot disagree. Every claim below is checkable:

```bash
pytest tests/pinneapple_catalog -q
python -c "from pinneapple_catalog import stages; print(stages.coverage()['totals'])"
python -c "from pinneapple_catalog import stages; print(stages.to_json('inverse')['counts'])"
```

## The thirteen stages are the Flow palette's

`STAGE_IDS` is the palette's `SECTION_IDS`, and a test compares the two lists. That is deliberate: the palette is
what a user sees in Flow Studio, so a stage that exists in the map and not on the canvas would be a catalogue of
something nobody can build. The last section, `library`, is the palette's drawer of generated `auto.*` calls — it is
carried so the lists match exactly, it is marked `kind="internal"`, and it asks no question and holds no tools.

## How a tool is described, and what the description refuses to claim

| Field | What it says | How it was obtained |
|---|---|---|
| `license` | an SPDX id, or the licence's own name where no id exists | read from the artefact that governs the **code**: the repository's `LICENSE` through the GitHub licence API, or the distribution metadata on PyPI for a package that ships none |
| `license_verified` | whether that read happened **here** | `True` for the reads above; the ten rows that were `False` until 2026-11-25 are listed at the end of this page with what their text says |
| `integration` | `wired` / `named` / `candidate` | `wired` means a module in this repository calls it, and a test resolves that module's path; `named` means the repository mentions it (a recommender, a lazy export, a docstring) with no path that would run it; `candidate` means nothing here touches it |
| `also_in` | the `TOOL_CATALOG` id, where the same tool has a page there | so the two catalogues reference each other instead of disagreeing |

Two things the licence column is *not*. A verified licence is **not** a commercial clearance: GPL-3.0, AGPL-3.0 and
ODbL are read correctly and are still unusable in a closed product, which is why the id is carried rather than a
boolean. And an unread licence is **not** a licence that forbids use — it is one this checkout did not open, and the
row says so rather than guessing.

The datasets side is a filter, not a copy: `CatalogEntry` in `pinneapple_data.datasets.catalog` gained the `stages` it
feeds, so `catalog.entries(stage="inverse")` and `stages.datasets("inverse")` are the same list by construction. A
test asserts it, and `stages.unassigned_datasets()` names any catalogued dataset that no stage claims.

## What the map looks like

| | |
|---|---|
| Stages | 13 (12 of them workflow, 1 the palette's drawer) |
| Tools | 111 — **35 wired**, 10 named, 66 candidate |
| Licences read here | 101 of 111 |
| Public datasets | 13, every one tagged with at least one stage |

Read the `wired` column, not the tool count. A stage with twenty candidates and no wiring is a stage where the work
has not started, and the map says so in the same breath as it names the tools.

| Stage | Tools | Wired | Datasets |
|---|---|---|---|
| problem | 8 | 2 | 3 |
| equations | 9 | 5 | 0 |
| solvers | 27 | 13 | 1 |
| physics_ai | 15 | 6 | 8 |
| surrogates | 10 | 2 | 4 |
| data | 17 | 8 | 12 |
| inverse | 11 | 1 | 2 |
| validation | 11 | 3 | 6 |
| optimization | 14 | 3 | 0 |
| twin | 15 | 5 | 3 |
| viz | 13 | 4 | 0 |
| export | 15 | 6 | 0 |
| library | 0 | 0 | 0 |

## What the shape says

Four things are worth saying plainly, because they are the reason to build the map at all:

* **The workflow is a T at the solvers.** 27 tools at the classical-solver stage, 13 of them wired, against 11 and
  **1** at the inverse stage and 10 with 2 at the ROM stage. The repository is strong where the physics is known and
  thin where it has to be inferred from data — which is exactly the stage the trust layer and the design loop depend
  on.
* **The inverse stage has one wired tool, and it is not a sampler.** PySINDy recovers an equation from data; it does
  not produce a posterior. PyMC, emcee, dynesty and UltraNest are catalogued and unused, so the "uncertainty" this
  library reports is still a seed-ensemble disagreement (`bundle_uq.seed_ensemble` says so itself). Closing that gap
  is the shortest path from "a number" to "an uncertainty".
* **Four stages have no datasets at all** — equations, optimization, viz, export. For viz and export that is honest
  (they consume a result rather than a public archive); for optimization it is a real hole: comparing a design loop
  needs the *reference optimum* of a case, and no catalogued dataset carries one.
* **The copyleft list is short and specific**: OpenFOAM (GPL-3.0), FEniCSx and UFL (LGPL-3.0), Firedrake
  (LGPL-3.0-or-later), preCICE (LGPL-3.0), CasADi, RBniCS, dolfin-adjoint and OpenTURNS (LGPL-3.0), OpenRadioss
  (AGPL-3.0-or-later), jax-fem and REBOUND (GPL-3.0-or-later), DEAP (LGPL-3.0), UltraNest (GPL-3.0), and — new to
  this page — **AirfRANS under ODbL-1.0**, the one *dataset* whose licence is share-alike. Outside the copyleft
  list and harder to live with, **FluidX3D forbids commercial use outright**.

### problem — Problem & geometry

*What is being solved, on what shape, and did the engineer's sentence become a machine-readable problem (domain, units, boundary patches, material)?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| CadQuery | Apache-2.0 | named | problem |
| FreeCAD | LGPL-2.1 | wired | problem |
| Gmsh | GPL-2.0-or-later (with the Gmsh linking exception) | named | problem, solvers |
| Open3D | MIT | candidate | viz, problem |
| OpenCASCADE (OCCT) | LGPL-2.1 | candidate | problem, export |
| OpenFOAM | GPL-3.0 | wired | problem, solvers, validation |
| SALOME | LGPL-2.1 | candidate | problem |
| trimesh | MIT | named | problem, viz |

Datasets: nasa_tmr, airfrans, flowbench

Gap: CAD arrives as a file, not as a parameterised intent: the STEP/BREP edit that a design study needs requires an external kernel, and just two of them (FreeCAD, CadQuery) are reached from here.

### equations — Equations

*Which governing equations, which conditions, and are they written as something a solver or a residual can consume — or only as prose?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| Cantera | BSD-3-Clause | wired | equations, solvers |
| CoolProp | MIT | wired | equations |
| FEniCSx (DOLFINx) | LGPL-3.0 | wired | equations, solvers |
| Firedrake | LGPL-3.0-or-later | candidate | equations, solvers |
| Modelica Standard Library | BSD-3-Clause | candidate | equations, twin |
| OpenModelica | OSMC-PL-1.8 | wired | equations, twin |
| SUNDIALS | BSD-3-Clause | candidate | equations, solvers |
| SymPy | BSD-3-Clause | wired | equations |
| UFL (FEniCS form language) | LGPL-3.0 | candidate | equations, solvers |

Gap: `physics.governing` is text with its symbols defined; nothing here turns that prose into a symbolic form without a human, so SymPy/UFL are catalogued and only named.

### solvers — Classical solvers

*What produces the reference numbers that the surrogate will be scored against?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| Gmsh | GPL-2.0-or-later (with the Gmsh linking exception) | named | problem, solvers |
| OpenFOAM | GPL-3.0 | wired | problem, solvers, validation |
| Cantera | BSD-3-Clause | wired | equations, solvers |
| FEniCSx (DOLFINx) | LGPL-3.0 | wired | equations, solvers |
| Firedrake | LGPL-3.0-or-later | candidate | equations, solvers |
| SUNDIALS | BSD-3-Clause | candidate | equations, solvers |
| UFL (FEniCS form language) | LGPL-3.0 | candidate | equations, solvers |
| CalculiX | GPL-2.0 | wired | solvers, validation |
| Diffrax | Apache-2.0 | candidate | physics_ai, solvers |
| Elmer | GPL-2.0 | candidate | solvers |
| FluidX3D | custom, non-commercial (no SPDX id) | wired | solvers |
| Genesis | Apache-2.0 | wired | solvers, twin |
| JAX-CFD | Apache-2.0 | wired | solvers, physics_ai |
| Kratos Multiphysics | BSD-4-Clause (Kratos licence, per application) | candidate | solvers |
| MuJoCo | Apache-2.0 | wired | solvers, twin |
| NVIDIA Warp | Apache-2.0 | named | solvers, physics_ai |
| Newton Physics | Apache-2.0 | wired | solvers, physics_ai |
| OpenRadioss | AGPL-3.0-or-later | wired | solvers |
| Project Chrono | BSD-3-Clause | candidate | solvers |
| REBOUND | GPL-3.0-or-later | wired | solvers |
| SU2 | LGPL-2.1 | named | solvers, optimization |
| SciPy | BSD-3-Clause | wired | optimization, solvers, data |
| Taichi | Apache-2.0 | wired | solvers, physics_ai |
| deal.II | LGPL-2.1-or-later | candidate | solvers, validation |
| jax-fem | GPL-3.0 | candidate | solvers, inverse |
| preCICE | LGPL-3.0 | candidate | solvers |
| Φ-Flow | MIT | candidate | solvers, physics_ai |

Datasets: nasa_tmr

Gap: The bridges are per product (CFD, structural), not one backend contract: a new code is a new adapter. Multi-solver coupling (preCICE) is catalogued and unused.

### physics_ai — Physics AI

*Which model family (PINN, neural operator, graph network, differentiable solver) and which package actually trains it?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| Diffrax | Apache-2.0 | candidate | physics_ai, solvers |
| JAX-CFD | Apache-2.0 | wired | solvers, physics_ai |
| NVIDIA Warp | Apache-2.0 | named | solvers, physics_ai |
| Newton Physics | Apache-2.0 | wired | solvers, physics_ai |
| Taichi | Apache-2.0 | wired | solvers, physics_ai |
| Φ-Flow | MIT | candidate | solvers, physics_ai |
| DGL | Apache-2.0 | candidate | physics_ai |
| DeepXDE | LGPL-2.1 | named | physics_ai |
| Equinox | Apache-2.0 | candidate | physics_ai |
| PINA | MIT | candidate | physics_ai |
| PhysicsNeMo (Modulus) | Apache-2.0 | wired | physics_ai, surrogates |
| PySINDy | MIT | wired | physics_ai, inverse |
| PyTorch Geometric | MIT | wired | physics_ai |
| neuraloperator | MIT | named | physics_ai, surrogates |
| torchdiffeq | MIT | candidate | physics_ai |

Datasets: cfdbench, pdebench, spe10, the_well, airfrans, deepmind_mgn, flowbench, pdearena

Gap: No common training interface: the user picks the architecture, and each family's own package is a different dependency tree. The GPU wave solvers (Warp, Taichi) are wired for one demo path each.

### surrogates — Surrogates & ROMs

*Which cheap model stands in for the solver once the data exists, and how is it chosen?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| PhysicsNeMo (Modulus) | Apache-2.0 | wired | physics_ai, surrogates |
| neuraloperator | MIT | named | physics_ai, surrogates |
| EZyRB | MIT | candidate | surrogates |
| GPy | BSD-3-Clause | candidate | surrogates |
| GPyTorch | MIT | candidate | surrogates, validation |
| OpenTURNS | LGPL-3.0 | named | surrogates, validation, inverse |
| RBniCS | LGPL-3.0 | candidate | surrogates |
| SMT | BSD-3-Clause | candidate | surrogates, optimization |
| pyMOR | BSD-2-Clause AND BSD-3-Clause | candidate | surrogates |
| scikit-learn | BSD-3-Clause | wired | surrogates, data |

Datasets: cfdbench, pdebench, the_well, airfrans

Gap: Projection-based ROM (pyMOR, RBniCS, EZyRB) is catalogued and not used; the ROMs here are the library's own POD/DMD.

### data — Data

*Where does the training data come from, in what format, and is the licence one a product may ship?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| SciPy | BSD-3-Clause | wired | optimization, solvers, data |
| scikit-learn | BSD-3-Clause | wired | surrogates, data |
| ADIOS2 | Apache-2.0 | candidate | data |
| Apache Airflow | Apache-2.0 | candidate | twin, data |
| Apache Arrow / Parquet | Apache-2.0 | wired | data |
| CGNS | Zlib-style (CGNS licence, derived from zlib/libpng) | candidate | export, data |
| Chaospy | MIT | candidate | data, validation |
| DVC | Apache-2.0 | candidate | data |
| MLflow | Apache-2.0 | candidate | data, validation |
| Pint | BSD-3-Clause | wired | validation, data |
| Prefect | Apache-2.0 | candidate | twin, data |
| SALib | MIT | candidate | data, validation |
| VTK | BSD-3-Clause | wired | viz, export, data |
| Zarr | MIT | wired | data |
| h5py / HDF5 | BSD-3-Clause | wired | data |
| pyDOE2 | BSD-3-Clause | named | data |
| xarray | Apache-2.0 | wired | data |

Datasets: cfdbench, pdebench, spe10, the_well, airfrans, deepmind_mgn, flowbench, jhtdb, norne, pdearena, era5, nrel_wind

Gap: The catalogue names twenty-odd public sources and reads six; the readers are per archive layout, and each new archive is a new reader.

### inverse — Inverse & calibration

*Which parameters, materials or source terms are unknown, and which inference buys them from measurements?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| jax-fem | GPL-3.0 | candidate | solvers, inverse |
| PySINDy | MIT | wired | physics_ai, inverse |
| OpenTURNS | LGPL-3.0 | named | surrogates, validation, inverse |
| ArviZ | Apache-2.0 | candidate | validation, inverse |
| PyMC | Apache-2.0 | candidate | inverse, validation |
| Pyro | Apache-2.0 | candidate | twin, inverse |
| UltraNest | GPL-3.0 | candidate | inverse |
| dolfin-adjoint | LGPL-3.0 | candidate | inverse, optimization |
| dynesty | MIT | candidate | inverse |
| emcee | MIT | candidate | inverse |
| lmfit | BSD-3-Clause | candidate | inverse |

Datasets: spe10, norne

Gap: The library calibrates with its own optimisers; the Bayesian samplers (PyMC, emcee, dynesty) are catalogued and named, not wired, so no posterior comes out of a PINNeAPPle run.

### validation — Validation & UQ

*How far is the prediction from a reference, and what is the number that says so?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| OpenFOAM | GPL-3.0 | wired | problem, solvers, validation |
| CalculiX | GPL-2.0 | wired | solvers, validation |
| deal.II | LGPL-2.1-or-later | candidate | solvers, validation |
| GPyTorch | MIT | candidate | surrogates, validation |
| OpenTURNS | LGPL-3.0 | named | surrogates, validation, inverse |
| Chaospy | MIT | candidate | data, validation |
| MLflow | Apache-2.0 | candidate | data, validation |
| Pint | BSD-3-Clause | wired | validation, data |
| SALib | MIT | candidate | data, validation |
| ArviZ | Apache-2.0 | candidate | validation, inverse |
| PyMC | Apache-2.0 | candidate | inverse, validation |

Datasets: cfdbench, nasa_tmr, pdebench, spe10, jhtdb, pdearena

Gap: Data-side and model-side gates exist and report a number; the UQ behind them is a seed-ensemble disagreement, not a posterior, and no catalogued sampler is wired to make it one.

### optimization — Optimization & control

*Given a trustworthy surrogate, which design should be built?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| SU2 | LGPL-2.1 | named | solvers, optimization |
| SciPy | BSD-3-Clause | wired | optimization, solvers, data |
| SMT | BSD-3-Clause | candidate | surrogates, optimization |
| dolfin-adjoint | LGPL-3.0 | candidate | inverse, optimization |
| CVXPY | Apache-2.0 | candidate | optimization |
| CasADi | LGPL-3.0 | wired | optimization, twin |
| DEAP | LGPL-3.0 | candidate | optimization |
| HiGHS | MIT | candidate | optimization |
| Ipopt | EPL-2.0 | candidate | optimization |
| NLopt | MIT | candidate | optimization |
| Nevergrad | MIT | candidate | optimization |
| Pyomo | BSD-3-Clause | candidate | optimization |
| pycma | BSD-3-Clause | candidate | optimization |
| pymoo | Apache-2.0 | wired | optimization |

Gap: Bayesian optimisation is present but not connected to the trust layer: a candidate can be ranked without the gate ever deciding whether the surrogate may be believed there.

### twin — Digital twin

*Which live signals update the model, and what does the model say before the next measurement?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| Modelica Standard Library | BSD-3-Clause | candidate | equations, twin |
| OpenModelica | OSMC-PL-1.8 | wired | equations, twin |
| Genesis | Apache-2.0 | wired | solvers, twin |
| MuJoCo | Apache-2.0 | wired | solvers, twin |
| Apache Airflow | Apache-2.0 | candidate | twin, data |
| Prefect | Apache-2.0 | candidate | twin, data |
| Pyro | Apache-2.0 | candidate | twin, inverse |
| CasADi | LGPL-3.0 | wired | optimization, twin |
| Apache Kafka | Apache-2.0 | candidate | twin |
| Eclipse Mosquitto (MQTT) | EPL-2.0 OR EDL-1.0 | candidate | twin |
| FMPy | BSD-2-Clause | wired | twin |
| FilterPy | MIT | candidate | twin |
| InfluxDB (open-source core) | Apache-2.0 | candidate | twin |
| NVIDIA Triton Inference Server | BSD-3-Clause | candidate | export, twin |
| pykalman | BSD-3-Clause | candidate | twin |

Datasets: norne, era5, nrel_wind

Gap: Streaming is HTTP polling and the telemetry bundle; no message broker, no FMU co-simulation in a twin loop (FMPy is wired only for the onboarding doctor).

### viz — Visualization

*Can an engineer see the field, the mesh and the difference, in 3-D and as an animation?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| Open3D | MIT | candidate | viz, problem |
| trimesh | MIT | named | problem, viz |
| VTK | BSD-3-Clause | wired | viz, export, data |
| Blender | GPL-2.0-or-later | wired | viz, export |
| F3D | BSD-3-Clause | candidate | viz |
| Matplotlib | Matplotlib (PSF-based, no SPDX id) | wired | viz, export |
| OpenUSD | TOST-1.0 (Tomorrow Open Source Technology License) | named | export, viz |
| ParaView | BSD-3-Clause | candidate | viz |
| Plotly | MIT | candidate | viz |
| PyVista | MIT | wired | viz, export |
| napari | BSD-3-Clause | candidate | viz |
| vedo | MIT | candidate | viz |
| yt | BSD-3-Clause | candidate | viz |

Gap: PyVista is wired for the schema's bundles; the CAE readers' own results still go through per-product plot code.

### export — Export & report

*How does the result leave — as a portable model, a mesh, or a document a reviewer signs?*

| Tool | Licence | State | Serves |
|---|---|---|---|
| OpenCASCADE (OCCT) | LGPL-2.1 | candidate | problem, export |
| CGNS | Zlib-style (CGNS licence, derived from zlib/libpng) | candidate | export, data |
| VTK | BSD-3-Clause | wired | viz, export, data |
| NVIDIA Triton Inference Server | BSD-3-Clause | candidate | export, twin |
| Blender | GPL-2.0-or-later | wired | viz, export |
| Matplotlib | Matplotlib (PSF-based, no SPDX id) | wired | viz, export |
| OpenUSD | TOST-1.0 (Tomorrow Open Source Technology License) | named | export, viz |
| PyVista | MIT | wired | viz, export |
| ONNX | Apache-2.0 | wired | export |
| ONNX Runtime | MIT | wired | export |
| OpenVINO | Apache-2.0 | candidate | export |
| Pandoc | GPL-2.0 | candidate | export |
| WeasyPrint | BSD-3-Clause | candidate | export |
| XlsxWriter | BSD-2-Clause | candidate | export |
| python-docx | MIT | candidate | export |

Gap: ONNX is exported and timed with three fallbacks; the rest of the list (OpenUSD, CGNS, Triton) is catalogued and not wired.

### library — Library calls (advanced)

Gap: Not a workflow stage: the palette's drawer of generated `auto.*` calls, kept in the list so the stage ids match the palette exactly.


## The map in the planner: a proposal now says which stages it spans

A table nobody queries is a document. The map is joined to the flow planner in `pinneapple_flow/stage_fit.py`:
`FlowBuilder.proposal()` annotates every proposal with `stage_map`, and the annotation travels inside the flow
(`flow.meta.stage_map`), so the Studio, the API and a saved flow all see the same thing.

```python
from pinneapple_flow.stage_fit import fit
fit([{"id": "calibrate", "type": "inverse.calibrate"}])
# {'stages': [{'id': 'inverse', 'wired': 1, 'wired_tools': ['pysindy'], 'datasets': [...],
#              'readable_datasets': [...], 'gap': 'The library calibrates with its own optimisers; ...'}],
#  'thin': [{'id': 'inverse', 'wired': 1, 'gap': '...'}], 'available': True, ...}
```

`thin` is deliberately **not** "zero wired tools". It is "at most one", because a stage where a single external tool
is wired offers no choice at all: the planned flow runs that step on PINNeAPPle's own code whatever the user wanted.
The stage's `gap` sentence goes with it, so the finding is actionable rather than a score.

A thin stage is also recorded as a decision (`stage.coverage`) with the rule's own citation status, so the reason
travels with the flow instead of living in a returned dict that a saved flow would lose. The rules gate
(`tests/test_flow_decisions.py`) requires every rule to cite a reference or to say why it has none; this one says it
is a property of this repository's map, not a literature rule.

What it looks like on a real plan — `plan_heuristic("identify the conductivity of a rod from measurements")` spans
six stages and flags one:

| Stage | Wired tools there | Verdict |
|---|---|---|
| problem | 2 | fine |
| equations | 5 | fine |
| solvers | 13 | fine |
| data | 8 | fine |
| **inverse** | **1** (PySINDy) | **thin — recorded as a decision** |
| export | 6 | fine |

The join degrades instead of failing: without `pinneapple_catalog` installed, `fit()` still reports the stages the
flow spans and says in `reason` why no coverage came back. An empty map would read as "nothing is available
anywhere", which is a different and wrong statement.

## The ten licences this page had not read, and what they say

These ten rows carried a licence nobody here had opened. They were read on 2026-11-25, each one at the place named
in its `license_source`, and `license_verified` is now `True` on all of them, so
`stages.coverage()['totals']['license_unverified']` is `0`.

| Tool | The licence, as its own text says it | What it implies for a paid study and for distribution |
|---|---|---|
| Gmsh | GPL-2.0-or-later, plus the Gmsh exception | commercial use fine; the GPL reaches a modified Gmsh, and the exception keeps it away from the caller that links it |
| SALOME | LGPL-2.1 | commercial use fine; a modified SALOME stays LGPL, and linking does not reach the caller |
| FluidX3D | custom: research, education and personal use, "Commercial use is not allowed" | **not usable in delivery**: nothing may be sold, nor a paid service whose value derives from it; no military use, no AI training on the source |
| Genesis | Apache-2.0 | commercial use and redistribution fine, notice kept and changes stated |
| Newton Physics | Apache-2.0 | the same |
| OpenRadioss | AGPL-3.0-or-later, (C) 2026 SISW Siemens | the strictest here: a modified version offered as a network service must publish its source; Siemens withdrew the open distribution in 2026-10 |
| REBOUND | GPL-3.0-or-later | shipped inside a product, the product becomes GPL: usable in a study, a decision before it is a dependency |
| CGNS | the CGNS licence, borrowed from zlib/libpng and superseding the LGPL | permissive with no copyleft: shippable, notice kept and changes marked |
| Blender | GPL-2.0-or-later | the GPL covers the application and the add-ons inside it, not the image it renders |
| Matplotlib | the MDT's own PSF-based agreement, no SPDX id | permissive and not BSD: keep the agreement and the MDT notice, and summarise the changes in a derivative |

Two of the reads changed the row rather than confirming it. **REBOUND's own header says "version 3 or any later
version"**, so `GPL-3.0` became `GPL-3.0-or-later`. **Matplotlib's text is an agreement between the Matplotlib
Development Team and a licensee**, not a BSD-family licence, and the row that called it "BSD-style" now quotes the
obligation it actually carries.

FluidX3D is the one entry whose licence forbids what this repository sells, so `is_open` is `False` for it: a
custom licence that bans commercial use is source-available, not open-source.

```python
from pinneapple_catalog import stages
[(t.id, t.license) for t in stages.tools() if not t.license_verified]   # []
[t.id for t in stages.tools() if not t.is_open]                         # ['fluidx3d']
```

## Decisions left open, each one a row and not an omission

1. **Agents are not a stage.** The palette has no section for the engineering agent, so the map has none either —
   the intent compiler and the advisor live under `problem` and `optimization` for now. If the workflow gains an
   agent step, it belongs in `SECTIONS` first and in this map second, which is the order the test enforces.
2. **`named` is not `wired`, and ten tools sit there.** They are named by a recommender or an optional-dependency
   table, with no path that runs them. Promoting one means writing the adapter, not editing this map.
3. **The map does not rank.** It says which tools serve a stage and how far this repository goes with each; it does
   not say which of two candidates is better for a given problem. That ranking exists for solvers in
   `tool_recommendation.recommend_tools`, and it is deliberately problem-scoped, not stage-scoped.
4. **Dataset stage tags are a judgement, not a measurement.** A dataset is tagged with the stages it *feeds*
   (`pdebench` feeds physics-AI training, data, surrogates and validation). Two people could tag the same archive
   differently; the tag lives in one place so the disagreement is a one-line edit rather than thirteen.

## References

The licences above were read through the GitHub licence API and the PyPI JSON API; the homepages are the projects'
own. Where a tool's *scope* is at stake rather than its licence, the entry defers to
`pinneapple_analysis.verification.tool_recommendation.TOOL_CATALOG`, which cites each tool's documentation.

- GitHub licence API: <https://docs.github.com/en/rest/licenses>
- PyPI JSON API (`info.license`, `info.license_expression`): <https://docs.pypi.org/api/json/>
- SPDX licence list, the ids used above: <https://spdx.org/licenses/>
- ODbL-1.0 (AirfRANS): <https://opendatacommons.org/licenses/odbl/1-0/>
- The companion page, organised by integration: `docs/dev/PHYSICS_AI_INTEROP.md`
