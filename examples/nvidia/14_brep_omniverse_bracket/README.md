# 14 Exact CAD into Omniverse: PINNeAPPle + NVIDIA USD BRep + Omniverse

**What.** An L bracket (6 mm 6061-T6 plate, 5 mm inner fillet, two 9 mm bolt holes, a 12 mm hole in the web) is a
recipe of B-rep operations. USD BRep builds it, exports its **native B-rep USD** (`BrepArray` prims) and tessellates
it. PINNeAPPle reads the exact curvature at every mesh vertex, casts rays for the wall thickness, checks everything,
and writes an OpenUSD stage for Omniverse that references the native B-rep and carries the mesh with
`curvature`, `wall_thickness`, `brep_face` primvars and the trust decision.

**Why.** A mesh in a scene forgets the design: radii become facets, walls become triangle pairs. Keeping the exact
B-rep next to a mesh annotated from it lets Omniverse show the design intent, and PINNeAPPle checks that the two agree.

| Step | Who | Ran here |
|---|---|---|
| exact solid, mass properties, tessellation, native B-rep USD export | USD BRep 1.0.1 | yes |
| exact curvature per vertex (closest point on the B-rep surface), wall thickness (ray casting), checks | PINNeAPPle | yes |
| OpenUSD stage, UsdValidation, read-back previews | OpenUSD (`usd-core`) | yes |
| RTX render in Omniverse Kit (`results/kit_render.py`) | Omniverse | **no** (needs an RTX GPU) |

```bash
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/14_brep_omniverse_bracket/run.py    # ~30 s
```

## Results (this run: 4-core Xeon 2.1 GHz)

- Exact volume matches the closed form (plates, holes, fillet) to 1.6e-9; mass 105.4 g.
- Radii read from the B-rep at the mesh vertices: **4.5, 5.0, 6.0 mm**, the bolt holes, fillet and web hole exactly.
- Thinnest wall by ray casting: 6.000 mm (the plate). Mesh: 17 500 triangles, largest chord error 7.1 um, volume
  within 1.0e-4 and area within 2.9e-5 of the exact solid; 1.5 mm elements resolve the 4.5 mm radius.
- **Decision: APPROVED.** The stage that references the B-rep validates cleanly.
- Finding: the native USD BRep layer has no `defaultPrim`, `metersPerUnit` or `upAxis`; the validators flag it. The
  stage written here sets them and references the B-rep prim by path, so a consumer gets units and orientation.

Figures in `results/`: `architecture.png`, `usd_curvature.png`, `usd_thickness.png` (both read back from the USD
files), `checks.png`.
