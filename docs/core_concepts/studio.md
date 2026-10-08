# 3D studio and external flow

Two layers turn any geometry and its physics into pictures people understand: `pp.viz`, which renders, exports and
browses, and `pp.cfd`, which runs OpenFOAM around any body. They came out of the Aircraft Design Optimizer
(`apps/aero_optimizer`) and work the same way for a car, a building, a heat sink or a beam.

```python
import pinneapple as pp

scene = pp.viz.Scene.from_file("result.frd")          # any result pinneapple_data.cae reads, or an STL/OBJ
scene.save("result.glb")                              # glTF with the fields; .usda for Omniverse; .stl
pp.viz.render(scene, "stress.jpg", field="STRESS")    # Blender Cycles, jet scale, colour bar
pp.viz.web_viewer(scene, "viewer/")                   # interactive page (python -m http.server -d viewer)
```

![Ahmed body, skin pressure](../assets/studio/ahmed-cp.jpg)
![Ahmed body, wake vortices](../assets/studio/ahmed-wake-vorticity.jpg)
![CalculiX cantilever, von Mises stress](../assets/studio/cantilever-von-mises.jpg)

## Scene

A `Scene` holds surfaces (vertices, triangles, a material and per-vertex fields), polylines with values
(streamlines) and slice planes (a value grid on a rectangle, NaN where there is no fluid).

| Build it from | Call |
|---|---|
| STL (binary/ASCII, one surface per solid), OBJ | `Scene.from_file("part.stl")` |
| glTF/GLB (node transforms applied; float attributes `_NAME` become fields) | `Scene.from_file("model.glb")`, `axes="aircraft"` for files written in aircraft axes |
| VTK PolyData `.vtp` (needs `pip install vtk`) | `Scene.from_file("surface.vtp")`: point and cell data |
| OpenFOAM case or zip, CalculiX `.frd`, Abaqus `.inp`, Gmsh, VTK | `Scene.from_file(path)`: boundary surface with the cell or point fields; vectors become magnitudes, stress tensors von Mises |
| A `pinneapple_data.cae` mesh you already have | `Scene.from_mesh(mesh)` |
| Arrays | `Scene.from_arrays(vertices, faces)` |
| Parts of `pinneapple_design.aero` | `Scene.from_parts(parts)` (aircraft axes) |

Data known at other points (CFD wall faces, FEA nodes, sensors) goes onto the surface with
`scene.map_field("cp", points, values, mirror_y=True)`, using inverse distance over the nearest points. `mirror_y`
handles half models. `scene.labels["cp"] = "Pressure coefficient Cp"` sets the colour-bar title.

## Renders

`pp.viz.render(scene, out, field=..., lines=..., slice=..., view="iso", background="studio")` needs
`pip install bpy` (Blender as a Python module, CPU Cycles).

- **Field colours:** CFD colours use the jet scale, blue for low and red for high. The colour bar, with five ticks,
  is burnt into the image.
- **Views:** `iso`, `iso_back`, `iso_low`, `front`, `back`, `side`, `side_left`, `top` or `bottom`. The names assume
  x is the object's length (the flow direction) and z is up. A direction vector also works.
- **Backgrounds:** `background="studio"` gives a light gradient with true colours; `"sky"` gives a physical sky
  with filmic tone mapping, for the realistic materials.

## Browser viewer

`pp.viz.web_viewer(scene, folder)` writes `index.html`, `viewer.js`, `scene.glb`, `scene.json` and a copy of
three.js, so no internet is needed. `pp.viz.serve(scene)` does the same in a temporary folder and serves it. The page
has these modes:
- **Realistic:** the materials;
- **one mode per field:** jet scale with a colour bar;
- **Streamlines:** coloured by their values;
- **one mode per slice plane.**

## External flow: `pp.cfd.ExternalFlow`

```python
flow = pp.cfd.ExternalFlow({"ahmed": pp.bodies.ahmed_body(25)}, speed=40.0, ground=0.0, half_model=True,
                           resolution="medium")
res = flow.solve("cases/ahmed", procs=4)      # write, snappyHexMesh, simpleFoam, read
res.coefficients                              # CD, CL, CS, pressure/friction split, per body, newtons
scene = res.to_scene()                        # skin Cp and Cf, streamlines by |U|/U∞, mid-plane and wake slices
```

**Setup**
- Steady incompressible RANS: simpleFoam, k-ω SST, wall functions.
- Bodies: a dict of `(vertices, faces)`, an STL path or a `Scene`.
- The domain and background cells scale with the bodies' length. `resolution` sets the surface refinement levels:
  `coarse`, `medium` or `fine`.
- `alpha` and `beta` tilt the flow.
- `half_model` adds a symmetry plane at y = 0.
- `ground` adds a road (`True`: at the lowest point of the bodies; a number: its height), moving with the flow unless
  `ground_moving=False`.

**Results**
- Forces: pressure plus wall-function shear from the log law at the first cell.
- Coefficients use `ref_area`; the default is the frontal area of the bounding box.
- A `FlowResult` can be saved with `res.save("result.npz")`.

**Validation, Ahmed body, 25° slant, 40 m/s, half model, moving road**

| Mesh | Cells | CD | Time (2 cores) |
|---|---|---|---|
| coarse | 0.15 M | 0.321 | 4 min |
| medium | 0.20 M | 0.298 | 10 min |
| wind tunnel (Ahmed et al. 1984) | | ≈ 0.285 | |

Needs OpenFOAM (v1912 and later tested). Set `FOAM_BASHRC` if its bashrc is not at
`/usr/share/openfoam/etc/bashrc`. `pp.cfd.openfoam_available()` checks it. The aircraft case
(`pinneapple_design.aero.case3d`) is built on the same pieces: `write_flow_fields`, `snappy_dict`, `run_case` and
`CaseFields`, which provides wall forces, streamline tracing and plane sampling for any case.

## Bodies

`pp.bodies.ahmed_body(slant_deg)`, `sphere`, `cylinder` and `box` return closed, outward-facing triangle meshes in
metres, with x along the flow and z up.

## Examples

- `examples/studio/01_any_result_in_3d.py`: a CalculiX result as glTF, USD, viewer and render.
- `examples/studio/02_ahmed_body_cfd.py`: from geometry to pictures, through OpenFOAM.
- `examples/studio/03_airliner_with_cfd.py`: the aero app's airliner with its OpenFOAM skin pressure.
