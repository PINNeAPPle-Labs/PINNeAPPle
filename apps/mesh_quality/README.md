# Mesh Quality

**Where your mesh is bad, how bad, and what to do about it, before the solver finds out.**

Day 9 of the PINNeAPPle 30-app program. A bad mesh is usually discovered by the solver: a run that diverges after an
hour, or results that quietly depend on a few distorted cells. This service reads the mesh and reports its health in
seconds:

| You give it | You get |
|---|---|
| OpenFOAM case or `polyMesh` (zip or folder), Gmsh `.msh`, VTK `.vtk`/`.vtu`, STL, Abaqus/CalculiX `.inp`, CalculiX `.frd`, other meshio formats | PASS / WARNING / FAIL; every check with why it matters and how to fix it; statistics and histograms per metric; **problem regions** (clusters of bad cells) with their position, worst value, element number and nearest boundary; a 3D view coloured by any metric with the regions labelled; the worst elements; JSON report and CSV checklist |

## Metrics

| Family | Metrics | Thresholds |
|---|---|---|
| Finite volume (any 3D mesh) | non-orthogonality, face skewness, cell aspect ratio, cell volume, open cells, face pyramids | OpenFOAM `checkMesh`: 70°, 4, 1000, > 0 |
| Finite element | scaled Jacobian (min over corners), element skewness (volume-based for triangles/tetrahedra, equiangle otherwise), edge ratio | 0.2 / ≤ 0 inverted; 0.9 / 0.98; 20 |
| Connectivity | disconnected regions, parts touching only through a node or an edge | |
| Surface (STL) | open edges (holes), non-manifold edges, inconsistent normals, zero-area and duplicate faces, separate shells, enclosed volume | |

The finite-volume metrics are computed with OpenFOAM's own formulas (face centres and areas by triangle decomposition,
cell centres and volumes by pyramid decomposition, `faceSkewness` / `boundaryFaceSkewness`, `cellClosedness`).
Finite-element meshes are converted to faces first, so a Gmsh or VTK mesh is checked exactly as OpenFOAM would after
conversion. Element node-order conventions (VTK vs Gmsh/Abaqus wedges) are detected per block; only truly inverted
elements are flagged.

## Validation (`tests/test_cae_mesh_quality.py`)

| Mesh | Reference (OpenFOAM v1912 checkMesh) | This service |
|---|---|---|
| pitzDaily, 12,225 hexahedra | aspect ratio 8.1407, non-orthogonality 5.95045° / avg 1.63034°, skewness 0.260575 | same digits |
| Gmsh bracket, 8,828 tetrahedra (`.msh` read directly; checkMesh after `gmshToFoam`) | 69.027° / 21.8577°, skewness 0.816741, aspect ratio 7.91293, volume 1.11077e-4 | same digits |
| Sheared channel (blockMesh) | "Max skewness = 6.29531, 22 highly skew faces", "Failed 1 mesh checks" | 6.29531, 22 faces, in two regions next to the walls |

Plus: the scaled Jacobian is 1 for the ideal tetrahedron, hexahedron, wedge and pyramid in both node-order
conventions; the closed STL of the bracket encloses exactly the tetrahedral mesh volume; the examples with an inverted
element, a stray part, a part hinged at one node and a damaged STL are all caught.

## Run

```bash
pip install -e . -r apps/mesh_quality/requirements.txt          # from the repository root
cd apps/mesh_quality && uvicorn mesh_quality.api:app --port 8088
```

Environment: `MQA_USER` / `MQA_PASSWORD` (HTTP Basic login), `MQA_MAX_MB` (upload, default 200), `MQA_MAX_CELLS`
(default 3,000,000), `MQA_MAX_HEAVY` (concurrent checks per worker, default 2). Docker:
`docker build -f apps/mesh_quality/Dockerfile -t mesh-quality .`; deployment with the other apps in
[`apps/deploy`](../deploy) (service `mesh`).

## API

`POST /api/check` (multipart `files`, optional `target` = auto | cfd | fea | surface) returns the report: `status`,
`checks`, `metrics` (worst, mean, p50, p99, count beyond the limit, histogram), `regions`, `worst_cells`, `summary`,
`usage.elements` and the 3D view payload. Python: `from pinneapple_data.cae import read_any, mesh_report`.
