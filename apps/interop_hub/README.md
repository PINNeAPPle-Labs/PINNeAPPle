# Simulation Interoperability Hub

**Upload your simulation result. Get it in a neutral engineering representation.**

Day 12 of the PINNeAPPle 30-app program. Moving results between CAE tools usually takes a script per pair of formats,
and units and provenance get lost on the way. Every result uploaded here becomes the same physical dataset, and from
there any format:

```
OpenFOAM · CalculiX · VTK/VTU · Gmsh · STL · CSV/HDF5/NPZ
        ↓
pinneapple.physical_dataset/1: geometry · mesh · coordinates · fields (quantity, unit) · metadata
        ↓
VTK (.vtu) · HDF5 · Parquet · CSV · NPZ · JSON · PINNeAPPle dataset (UPD Zarr)
```

```json
{
  "schema": "pinneapple.physical_dataset/1",
  "geometry": {"dimension": 3, "length_unit": "m", "bounding_box": {...}},
  "mesh": {"kind": "polyhedral", "cells": 12225, "points": 25012, "element_types": {"hexahedron": 12225}},
  "coordinates": {"location": "cell", "count": 12225, "unit": "m"},
  "fields": {
    "U": {"quantity": "velocity", "unit": "m/s", "location": "cell", "components": ["x", "y", "z"]},
    "p": {"quantity": "kinematic pressure (p/ρ)", "unit": "m2/s2", "location": "cell"}
  },
  "metadata": {"solver": "OpenFOAM simpleFoam", "time": 282, "unit_system": "SI", "files": [...sha256...]}
}
```

- **Units**: OpenFOAM is SI (incompressible `p` is kinematic: give ρ to get Pa); CalculiX/Abaqus have no units, so
  the deck's unit system (SI or N-mm-t-s) labels the fields and converts them to SI on request.
- **Meshes**: OpenFOAM polyhedral cells are rebuilt as VTK hexahedra, wedges, tetrahedra and pyramids from their faces;
  any other cell is written as a VTK polyhedron. HDF5 also keeps the raw polyMesh (faces, owner, neighbour, patches).
- **PINNeAPPle dataset**: a `PhysicalSample` written with the library's own UPD Zarr store
  (`pinneapple_data.serialization.save_zarr`), read back with `load_zarr` (fixed in this change: it called a
  method the store did not have). Units, quantities and the source are in the provenance.

## Validation (`tests/test_interop_hub.py`)

- OpenFOAM pitzDaily (12,225 cells) → VTU, reread with **VTK (PyVista)** and meshio: same cells, total volume
  1.4516e-5 m³ and smallest cell 1.6902e-10 m³ as checkMesh reports, field values identical in VTK order.
- A non-standard cell (a cube with one face split in two triangles) → VTK polyhedron: VTK reads 1 cell, type 42,
  volume exactly 1.
- Parquet, CSV, NPZ, HDF5, JSON and the PINNeAPPle dataset reread and compared value by value; units carried in each
  format's own metadata.
- CalculiX N-mm-t-s → SI: displacements × 1e-3, stresses × 1e6, geometry in metres; kinematic pressure × ρ = Pa.

## Run

```bash
pip install -e . -r apps/interop_hub/requirements.txt          # from the repository root
cd apps/interop_hub && uvicorn interop_hub.api:app --port 8091
```

Environment: `IOP_USER` / `IOP_PASSWORD` (HTTP Basic login), `IOP_MAX_MB` (default 500), `IOP_MAX_HEAVY` (default 2),
`IOP_CACHE_MIN` (minutes a converted upload stays available for exports, default 30). Docker:
`docker build -f apps/interop_hub/Dockerfile -t interop-hub .`; deployment with the other apps in
[`apps/deploy`](../deploy) (service `interop`).

## API

`POST /api/convert/{format}` (one call), or `POST /api/inspect` then `GET /api/export/{id}/{format}`; formats `vtu`,
`hdf5`, `parquet`, `csv`, `npz`, `json`, `pinneapple`. Options: `time`, `unit_system`, `to_si`, `rho`, `location`.
Library: `from pinneapple_data.cae.dataset import build_dataset, export`.
