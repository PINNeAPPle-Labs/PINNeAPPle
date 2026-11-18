- Renders from the Blender binary without `bpy`: `pp.viz.render_headless(scene, out, **kw)` and
  `pp.viz.blender_executable()`, which looks in `$BLENDER`, `PATH` and the usual install locations. The scene travels
  as a `.glb` and keeps its `scene.materials` and labels, so a material only the scene knows about renders as written
  (a name read back from a `.glb` is no longer replaced by `grey`). `render`/`render_headless` take `exposure` (EV),
  and translucent materials use Blender 4.2's `surface_render_method` on top of the older `blend_method`.
- 3-D annotation geometry: `pp.viz.sphere_surface`, `pp.viz.rod_surface` and `pp.viz.disc_surface` for a source, a
  beam axis, an aperture or a measurement plane, plus the `magma` colour map in `pp.viz.magma` / `pp.viz.CMAPS`
  (also in the browser viewer).
- 2-D compact range: `Range2D.qz_line()` and `planar2d.near_field_map()`, which puts the PO density and the exact
  surface current on one `(y, z)` grid and returns the quiet-zone cut and where the two solvers part company.
