`pinneapple_simulation.pipelines.ahmedml.surface`: the reader and the reductions for the half of the AhmedML contract
that fits on a laptop. A case's boundary is an XML PolyData whose `pMean` and `wallShearStressMean` are CellData on
1.1–1.24 M polygons, with the published face areas shipped as their own `.npy` — 83 MB per case against 5.6 GB for the
volume. `read_surface` recovers polygon centroids, outward normals from the winding and both fields, and its fan
triangulation reproduces the released areas to the last digit (1.6843 m² on `run_4`); `surface_field_metrics` computes
both published reductions for either field — face-area weighted and equal-entity — with the spec's vector rule (one
entity weight times the squared vector magnitude) for the wall shear; `aggregate_metrics` is the declared macro average
over geometries. The composite's surface half is 25 % of the score; the other half needs the per-case volume, and the
contract requires complete entity coverage of both, which the module states rather than works around.
