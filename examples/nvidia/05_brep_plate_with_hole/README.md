# 05 Exact geometry for a stress analysis: PINNeAPPle + NVIDIA USD BRep

**What.** A 150 x 60 x 5 mm 6061-T6 strap with a 20 mm hole carries 10 MPa of tension. NVIDIA USD BRep (the SMLib
NURBS kernel, `pip install usd-brep`) builds the exact solid (box minus cylinder) from a recipe and tessellates it;
PINNeAPPle measures the meshes against the exact solid, solves plane stress on the B-rep face meshes and checks
the stress concentration against Heywood's closed form.

**Why.** A stress result is only as good as the geometry it was computed on. The exact B-rep is the reference for
the mesh (chord error, area, volume, the hole radius resolved); the closed form is the reference for the stress.

| Step | Who | Ran here |
|---|---|---|
| part recipe; exact volume, area, mass, face curvature; tessellations at 1, 0.1 and 0.01 mm chord, and at 4, 2, 1 and 0.5 mm element size | USD BRep 1.0.1 (CPython 3.12 worker) | yes |
| chord error measured against the exact surfaces; volume / area against the analytic ones; radius resolved | PINNeAPPle (`pinneapple_tools.nvidia.brep`) | yes |
| plane-stress FEM (constant-strain triangles) on each B-rep face mesh, patch test, force balance, Richardson, Heywood | PINNeAPPle (`plane_stress_fem`) | yes |

```bash
python3.12 -m venv /opt/brep && /opt/brep/bin/pip install usd-brep numpy     # the usd-brep wheel is cp312 only
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/05_brep_plate_with_hole/run.py    # ~10 s
```

## Results (this run: 4-core Xeon 2.1 GHz)

- Exact solid: volume equals the analytic one to 6e-11, area to 2e-8, smallest radius of curvature 10.000 mm (the
  hole). Mass 117.3 g.
- Tessellation: asked 1, 0.1, 0.01 mm chord -> measured largest deviation 0.18, 0.047, 0.0030 mm; at 0.01 mm the mesh
  volume and area are within 1.5e-5 and 1.1e-5 of the exact ones.
- FEM on the B-rep meshes (1 208 to 70 388 triangles): patch test passes to 4e-12 on every mesh; net-section Kt =
  1.894, 2.093, 2.220, 2.276 at 4, 2, 1, 0.5 mm; observed order 1.18; Richardson **2.320** against Heywood
  2 + (1 - d/W)^3 = **2.296** (1.0 %).
- **Decision: APPROVED.**

Figures in `results/`: `architecture.png`, `brep_part.png`, `convergence.png` (tessellation vs exact; Kt vs element
size), `stress.png` (sigma_xx on the finest B-rep mesh), `checks.png`.
