# Running the first structural bracket FEM example

This example is a **simplified demonstration**, not a reproduction of the original CAD, geometry splits, or published error values.

## Environment

Python 3.10+ with NumPy, SciPy, PyTorch, and pytest.

```bash
python -m pip install numpy scipy torch pytest
cd examples/structural_bracket_surrogate
python -m pytest -q test_baseline.py test_dataset.py
python generate_dataset.py --output artifacts/l_bracket_demo.npz
python deeponet.py
```

## Dataset contract

The NPZ file includes nodes (metres), tetrahedral connectivity, fixed node IDs, nodal forces (newtons), displacements (metres), element stress vectors [xx, yy, zz, xy, yz, xz] (pascals), element von Mises (pascals), reactions (newtons), and JSON provenance.

**Do not use this one-case NPZ to train and claim generalization.** The Branch feature schema requires distinct simulation cases, a fixed ordering of geometry/material/load attributes, and training-only normalization. The Trunk feature schema requires geometric distance features for each spatial point. Those builders are not yet implemented.

## Validation gates before benchmark results

1. Run smoke tests on the target environment and record versions.
2. Compare the FEM implementation with an analytical patch test and an independent FEM solver.
3. Perform mesh refinement and verify reaction equilibrium and convergence away from singularities.
4. Replace nodal point-force approximations with consistent face-traction integration.
5. Build independent cases and group-based train/validation/test partitions.
6. Train DeepONet and compare to FEM with shared field color limits, hotspot errors, and confidence gates.

## Important limits

- Linear tetrahedra give elementwise constant stress; do not confuse element stress with nodal stress.
- Geometric cutouts are voxelized and may exhibit stair-step boundaries.
- The demo uses fixed-end and distributed nodal loading selected by coordinates; it is not the original publication's load case.
- No results or latency claims from the original work have been reproduced.
