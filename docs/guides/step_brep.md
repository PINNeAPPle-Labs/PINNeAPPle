# STEP files as exact B-rep: import, meshing, surface quality, bake

`pinneapple_design.geometry.brep` keeps a STEP part **exact** for as long as possible:
1. read every body at its real size;
2. mesh it at a chosen quality;
3. check its surfaces;
4. bake it to a plain mesh for 3-D printing, CFD/FEM or PINN boundary sampling.

The plan is in issue #1114. This guide grows with each part.

The exact kernel is **SMLib** (NVIDIA, Apache-2.0), used through **smlibpy** (MIT):

```bash
pip install "pinneapple[brep]"
```

Without it, `pinneapple_design.geometry.io.step.step_to_mesh` still turns a STEP file into a mesh with gmsh, but
not into exact bodies.

The workflow is the one the Blender add-on [BlenderSTEP](https://github.com/cekuhnen/BlenderSTEP) shows inside
Blender: import, mesh, verify, bake. BlenderSTEP is GPL-3.0, so PINNeAPPle implements the workflow on its own
and copies no code from it.

## 1. Import (#1115)

```python
from pinneapple_design.geometry.brep import read_step

bodies = read_step("bracket.step")          # one CadBody per solid or shell, in metres
b = bodies[0]
b.volume_m3, b.area_m2, b.size_m             # exact, from the kernel
b.face_counts()                              # {'plane': 6, 'cylinder': 4}
b.faces[6].params                            # {'radius_m': 0.004, 'axis_origin_m': [...], 'axis_direction': [...]}
b.summary()                                  # JSON-ready
```

What `read_step` does:
- It reads AP203, AP214 and AP242 B-rep data.
- It applies assembly placements and keeps body names.
- It heals invalid bodies; valid ones are left alone.
- The kernel works in millimetres, and `read_step` converts everything to metres.

Each face carries its surface type and parameters:

| Surface | Parameters |
|---|---|
| plane | origin and normal |
| cylinder, sphere | axis and radius |
| cone | axis, reference radius and half-angle |
| torus | axis, major and minor radii |

Free-form faces (B-spline / NURBS) keep the kernel's type name.

`write_step(shapes_or_bodies, path, unit="mm")` writes exact STEP AP214. The tests use it to generate their parts,
so no third-party CAD file is stored.

**Checked** (`tests/test_brep_step.py`; parts written to STEP and read back):

| Part | Check |
|---|---|
| Plate 80 × 50 × 6 mm with four Ø 8 mm holes | volume and area against the closed form within 1e-9; 6 planes and 4 cylinders; hole radius and axis exact |
| Sphere, frustum, torus | volume and area against the closed forms within 1e-6; cone half-angle atan((R − r)/h); torus radii |
| Block with four vertical fillets, r = 5 mm | V = (ab − (4 − π)r²)h and the matching area, within 1e-9 |
| The same part written in mm and in inch | the same body in metres, within 1e-9 |
| Two named bodies in one file | names and volumes kept |
| A loft from a square to a circle | reported with a free-form face type |
