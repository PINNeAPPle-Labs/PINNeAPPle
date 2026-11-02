- **Behavior:** two faults in STEP tessellation, both found by `mesh_quality`, are fixed. Fast meshes of such plates
  change by edge flips only; the vertices stay the same.
  - The fast mesher no longer leaves zero-area triangles where hole tops are collinear.
  - The kernel mesher no longer labels a long thin triangle with the face next to its centroid.
