`docs/guides/fluidsbench_datasets.md` now carries the per-case numbers of the four open 3-D datasets, read from the
`case-support.json` records in the checkout instead of estimated from the guides, and the DrivAerNet++ contract was
verified by fetching one case: a VTK legacy surface with PointData `p`, 511 063 points, 474 686 polygons, five connected
components, 32.35 m² of dual area, dynamic head 450 m²/s². The support archive's SHA-256 matches the manifest's. The
support carries the truth, the area weights and the profile indices but **no coordinates**, so scoring that split costs
30.5 GB of geometry plus 5.49 GB of support for its 1 154 designs — which is what makes it out of reach on a 24 GB disk,
not the 210 GB of public release.
