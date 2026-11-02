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

## 2. Meshing (#1116)

```python
from pinneapple_design.geometry.brep import tessellate

m = tessellate(b, quality="normal", mesher="auto")     # TessMesh in metres
m.vertices, m.triangles, m.face_id                       # a body face index per triangle
m.is_watertight(), m.euler_characteristic(), m.report()
```

**Quality presets.** Each sets a chordal deviation, as a fraction of the body's bounding-box diagonal D, and the
largest angle between neighbouring normals on curved faces:

| Preset | Chordal deviation | Angle |
|---|---|---|
| draft | 2e-3 D | 30° |
| normal | 5e-4 D | 15° |
| fine | 1e-4 D | 7.5° |

**Meshers:**
- **`fast`** handles bodies made of planes, cylinders, cones and spheres whose curved faces are bands between
  coaxial full circles, or whole spheres. Typical cases are plates with holes, bosses, tubes and turned parts.
  - Every edge is sampled once and shared by the two faces on its sides, so the mesh is watertight by
    construction.
  - A circle of radius r gets n = ⌈π / arccos(1 − tol/r)⌉ segments (and at least 360°/angle), so its sagitta
    r(1 − cos(π/n)) stays within the tolerance.
  - Planar faces, holes included, are triangulated with earcut. Bands are strips between their two rings.
- **`kernel`** uses the exact kernel's tessellation of the whole body and works for any surface. Face IDs come from
  the per-face meshes at the same tolerance.
- **`auto`** uses `fast` when every face qualifies, and otherwise `kernel` for the whole body: mixing the two
  would leave cracks where their edge samplings differ. `fallback_faces` and `reason` say why.

**Measured** on a 200 × 100 × 5 mm plate with 50 holes of Ø 6 mm. Every mesh is watertight, with Euler
characteristic −98 = 2 − 2·50:

| Quality | Mesher | Triangles | Volume error | Time |
|---|---|---|---|---|
| draft | fast | 2 664 | 3.3e-3 | 0.13 s |
| draft | kernel | 3 412 | 1.9e-3 | 0.60 s |
| normal | fast | 5 064 | 8.5e-4 | 0.23 s |
| normal | kernel | 6 612 | 4.9e-4 | 1.25 s |
| fine | fast | 9 864 | 2.1e-4 | 0.25 s |
| fine | kernel | 13 012 | 1.2e-4 | 2.30 s |

**Checked** in `tests/test_brep_tessellate.py`:
- the sagitta and angle bounds of `circle_segments`;
- the chord error of every hole edge within the preset tolerance, for all three presets;
- watertightness and genus for both meshers;
- volume convergence with quality on a tube;
- cone and sphere through the fast mesher, with the sphere's vertices exactly on the sphere;
- the fallback to the kernel on a torus, with its reason;
- kernel face IDs whose areas add up to the exact face areas.

## 3. Surface quality (#1117)

```python
from pinneapple_design.geometry.brep import diagnostics as D

m = D.refine(tessellate(b, "normal"))          # denser mesh, same face ids (planes have corner vertices only)
c = D.exact_curvature(b, m)                    # k1, k2, K [1/m^2], H [1/m], outward normals, from the exact surface
d = D.discrete_curvature(m)                    # K and H from the triangles alone
edges = D.edge_continuity(b)                   # per edge: level G0/G1/G2, gap, angle, curvature jump
z = D.zebra(m, c["normal"], n_stripes=16)
h = D.highlight_lines(m, c["normal"])
D.render(m, c["H"], "mean_curvature.png", colorbar="H [1/m]")
```

What each one measures:
- **Exact curvature.** The kernel evaluates the surface derivatives of the face each vertex lies on. Curvature is
  positive where the surface bends away from its outward normal (a ball seen from outside) and negative in a bore.
- **Discrete curvature.** The angle deficit gives K and the cotangent Laplacian gives H. Where it departs from the
  exact value, the mesh is too coarse for the shape. Sharp edges show up as large values, as expected of a discrete
  estimate there.
- **Edge continuity.**
  - G0: the faces meet at an angle.
  - G1: they are tangent.
  - G2: they are curvature continuous too.
  - The tolerances are the kernel's defaults: 0.5° and a 1 % relative curvature jump.
- **Zebra.** It reflects a cylinder of black and white stripes around the part, seen from an eye point. The stripes
  run smoothly across G2 joins, kink at G1 and break at G0.
- **Highlight lines** (Beier & Chen, 1994). They are lit where the line along a surface normal passes near one of a
  set of parallel straight light lines, so they bend where the surface does.
- **`render`.** It draws any per-vertex value to a PNG: an orthographic view, back faces culled, triangles painted
  back to front.

**Checked** in `tests/test_brep_diagnostics.py`:

| Case | Check |
|---|---|
| Sphere, R = 10 mm | exact K = 1/R² and H = 1/R within 1e-9; outward normals; discrete K and H within 1 % |
| Cylinder, r = 5 mm | H = 1/(2r), K = 0 on the side; zero on the caps |
| Torus, 20 / 5 mm | K and H against the closed forms at every vertex within 1e-6; K < 0 inside, > 0 outside |
| Block, sharp / filleted r = 5 mm / `continuity=2` | 12 G0 / 16 G0 + 8 G1 / 16 G0 + 8 G2; G1 jump in curvature 1/r = 200 1/m |
| `refine` | watertight, same Euler characteristic and face areas, edges below the limit |
| Highlight lines on a plane parallel to the light plane | straight: the value depends on one coordinate only |
| Zebra on a plane seen from far away | one uniform value; on a sphere, both stripe colours |
