# CAD-to-FEM benchmark pipeline

The primary benchmark path now uses **CadQuery B-rep STEP solids**, not the voxelized L-bracket demonstration.

## Reproducible commands

Install the optional CAD stack (prefer a clean environment with matching OCCT/Gmsh libraries):

```bash
python -m pip install numpy scipy torch pytest cadquery gmsh
cd examples/structural_bracket_surrogate
python -m pytest -q test_baseline.py test_dataset.py test_cad_brackets.py test_surface_loads.py
python cad_brackets.py --output-dir artifacts/cad
python run_cad_fem.py --family G --output-dir artifacts/gusset --size-mm 4
```

Each run exports STEP, mesh NPZ, FEM NPZ, and JSON manifest. Units are millimetres for CAD and metres for FEM. The tetrahedral FEM solver uses first-order elements and reports elementwise stresses.

## Loading assumptions

The demonstration fixes exterior mesh faces on the minimum-z plane and applies a *constant vector traction* to the maximum-z plane. Loads are assembled by consistent triangle integration: each boundary face contributes its area times traction divided by three to each node. This is better posed than equal nodal forces, but **is not a reconstruction of the paper's fixture or bolt-bearing contact**. Holes are present in CAD but bolts, friction, contact and pretension are not modeled.

## Scientific limitations

- The source presentation lacks exact dimensions, load magnitudes, mesh convergence and full training protocol.
- Geometries are original parametric approximations of the four bracket families.
- CAD validity tests, Gmsh execution, FEM mesh convergence and independent solver comparison remain required.
- Sharp corners and hole edges may yield high mesh sensitivity.
- A single FEM case is not a training dataset; geometric and load sweeps, branch/trunk feature builders and grouped splits remain to be implemented.
- No model training, accuracy, latency, or VeriPhysics certification is claimed.

## Next validation gates

1. Run the complete CAD/STEP/Gmsh pipeline in an environment with the optional dependencies.
2. Inspect each family for intended solid topology, hole location and non-intersecting features.
3. Refine the mesh at least three times, compare displacements and integral stress measures away from singularities.
4. Validate FEM against a trusted solver with identical boundary conditions.
5. Add named CAD physical surface groups for bolts, bearing loads and actual fixtures.
6. Generate parametric DOE cases and leak-free geometry-family splits.
7. Train DeepONet, quantify error and uncertainty, then connect VeriPhysics and Engineering API.

Tracking: https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1723
