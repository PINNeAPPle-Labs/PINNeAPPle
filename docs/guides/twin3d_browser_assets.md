# Browser-ready digital twins: geometry LOD (initial implementation)

`pinneapple_twin3d.lod.build_visual_lods` creates separate **visual-only** glTF LOD
files after `Scene.export`. The original `geometry.glb`, `fields.bin`,
per-vertex physics data, sensor streams and simulation meshes are unchanged.
Field-backed parts retain their original topology at every LOD.

```python
from pinneapple_twin3d import Scene
from pinneapple_twin3d.lod import build_visual_lods

scene = Scene("factory")
scene.add_part("machine", vertices, faces)
scene.export("out/factory")
report = build_visual_lods(scene, "out/factory", ratios=(0.5, 0.2))
print(report["levels"])
```

Install the optional real quadric decimator with `pip install "pinneapple[geom_simplify]"`.
If unavailable, simplification raises an actionable ImportError rather than silently
producing an unoptimized file. Ratios are targets, not compression guarantees.
A bounding-box guard prevents obvious scale/extent changes; it is **not** a
Hausdorff or engineering-accuracy certificate.

**Current boundary:** the bundled viewer still renders the authoritative
`geometry.glb`. The generated `visual_lods` metadata is for compatible consumers
and future viewer integration. No Draco, Meshopt, KTX2, texture packing, automatic
camera-distance switching, draw-call deduplication or asset-library ingestion is
claimed in this initial change. Those require a separate verified pipeline.

A production implementation must measure both silhouette and per-vertex field
errors, support per-part switching, test VRAM / bandwidth / frame times on
real devices, record source provenance and asset licenses, and never treat a
rendering LOD as a physics simulation mesh.
