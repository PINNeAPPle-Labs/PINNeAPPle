- `pinneapple_design.geometry.brep.bake` bakes exact STEP bodies to plain meshes (#1118):
  - STL, OBJ, glTF and USD files, with body names, optional per-vertex colours and units;
  - `to_meshdata`, a bridge to `MeshData`;
  - a gmsh volume mesh through `generate_mesh`;
  - `mesh_quality`: watertightness, self-intersections, angles, aspect ratio, and the chordal, volume and area error
    against the exact B-rep.
