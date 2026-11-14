- Unreal Engine as a third 3-D backend next to the Blender render and the browser viewer:
  `pp.viz.export_unreal(scene, "ue/out", field="STRESS")` writes a folder Unreal imports -- `geometry.glb` with the
  field baked into `COLOR_0` vertex colours (the attribute Unreal's importer reads; the `_NAME` attributes
  three.js uses are kept alongside), `scene.usda` for the USD Importer plugin, `fields.bin` with the raw values and
  the offsets in `manifest.json`, and `PinneAPPleImport.py`, which runs inside the editor to import the geometry,
  build the vertex-colour material, place the actors and (`cinematic=True`) light the stage and frame a cine camera.
  The export side is tested; the editor script needs a real Unreal install and logs each step as it goes.
- A scroll-driven cinematic page for a client-facing hero: `pp.viz.cinematic_page(scene, "site/demo", field="cp")`
  writes `index.html` with the geometry and its field, a camera that walks around it as the page scrolls, one
  caption per beat, a colour bar with the field's range, and the vendored three.js travelling with it (no CDN, no
  server, opens from `file://`). It honours `prefers-reduced-motion` and falls back to a readable page without
  WebGL. `css` and `head` carry a site's own fonts and colours, which is how `demo/` on the PINNeAPPle Labs site is
  built from the same call.
