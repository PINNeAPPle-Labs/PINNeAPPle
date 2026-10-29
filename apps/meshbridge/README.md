# MeshBridge

**Upload a CFD / FEM / CAD mesh file; get its points and fields as NPZ, or a repaired triangle surface.**

Imported from PINNeAPPle-apps (`cfd_format_converter`). The reading, repair and writing are
`pinneapple_simulation.external_solvers.cfd_formats.convert` (`read_point_bundle`, `read_surface_mesh`,
`convert_surface`); this folder is the HTTP layer. Related apps: `apps/interop_hub` converts whole simulation
results (OpenFOAM, CalculiX, VTK, Gmsh, ...) to a neutral dataset with units, and `apps/mesh_quality` grades volume
meshes; MeshBridge covers the legacy readers (CGNS, Exodus II, Fluent ASCII mesh, Abaqus `.inp`) and surface
repair, which those apps do not.

| Input | `output=npz` (coords + fields) | surface (`stl` / `obj` / `ply`, `repair`, `simplify_target_faces`) |
|---|---|---|
| `.cgns`, `.exo` / `.e`, `.msh` (Fluent ASCII) | yes | no: the readers return no connectivity |
| `.inp` (Abaqus) | yes | triangle shells as they are; tetrahedra reduced to their boundary faces |
| `.stl`, `.obj`, `.ply`; `.step` / `.stp` (needs cadquery) | no | yes |

## Run

```bash
pip install -e . -r apps/meshbridge/requirements.txt
cd apps/meshbridge && uvicorn meshbridge.api:app --port 8141        # http://localhost:8141/docs
curl -F file=@part.stl -F output=stl -F repair=true http://localhost:8141/v1/convert -o fixed.stl
```

Logins `MBR_USER` / `MBR_PASSWORD_HASH`, `MBR_UPLOAD_MAX_MB` (200), `MBR_FREE_CONVERSIONS` (200 per month); deployed
as `meshbridge` in `apps/deploy`.

## Tests

`pytest tests/test_app_meshbridge.py tests/test_cfd_formats_convert.py` (4 + 6 pass). The source app's
`examples/repair_case.py` (an icosphere with 15 duplicate faces, 10 degenerate faces and a flipped patch) is a test:
after `/v1/convert?repair=true` the surface has 320 faces, no duplicates or degenerates and consistent winding.

## Fixed while porting

- Abaqus connectivity used node ids as row indices (wrong for any deck whose ids do not start at 0).
- A tetrahedral `.inp` became one face per tetrahedron instead of the mesh boundary.
- Fluent was listed as a surface format but had no surface reader; STEP output was a stub that always failed: both
  removed.
- STL is read with coincident vertices merged; without that, duplicate faces and flipped patches are invisible to
  the repair.

## Known limitations

- The CGNS, Exodus and Fluent readers follow the published specs and were not checked against files from the
  real writers (`cfd_formats` package docstring). Exodus reads classic NetCDF only.
- Repair is trimesh's (duplicates, degenerates, normals, optional hole filling): no self-intersection or
  non-manifold repair. Simplification needs `fast-simplification` or `open3d`.
