# HexaMSH experimental bridge

Install upstream HexaMSH separately in Python 3.12 and make its executable available.
Install PINNeAPPle with the geom extra. This bridge does not bundle upstream code.
HexaMSH uses Gmsh and its GUI; unattended operation is not claimed.

```python
from pinneapple_design.fitmesh.hexamsh import generate_hex_mesh, import_hex_mesh
mesh, report = generate_hex_mesh("part.step", "verified.msh")
# Or import an existing upstream MSH:
# mesh, report = import_hex_mesh("exported.msh", "verified.msh")
```

Save exactly one MSH inside the temporary CAD directory before closing the GUI.
The native MSH is copied without topology conversion after the existing CAE FEA
quality gate passes. This is not a solver validation or proof of positive Jacobian
everywhere. Review physical boundary groups and units before simulation.
Hex8 only; simplex core Mesh is not compatible. No automatic refinement, FNO grid
mapping, or headless generation is claimed. Upstream BSD-3-Clause and Gmsh GPL
terms must be reviewed before redistribution.

Local tests: python -m pytest tests/test_hexamsh_adapter.py
