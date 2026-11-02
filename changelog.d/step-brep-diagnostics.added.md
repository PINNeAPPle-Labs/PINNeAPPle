`pinneapple_design.geometry.brep.diagnostics` checks the surface quality of exact STEP bodies (#1117). It covers:
- exact principal, Gaussian and mean curvature from the kernel at mesh vertices;
- discrete curvature from the mesh alone;
- G0/G1/G2 classification of every edge, with its gap, normal angle and curvature jump;
- zebra stripes and highlight lines;
- `refine` for dense evaluation and a PNG `render`.
