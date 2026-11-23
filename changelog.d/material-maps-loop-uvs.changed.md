- Texture coordinates are per face corner instead of per vertex, and a bake now stores the layout it used:
  `Surface.uv` holds one pair per corner (a per-vertex array is still accepted and expanded), `scene.box_uvs` gives
  every face its own square instead of averaging at the shared vertices, and `bake_materials` unwraps with Blender's
  smart projection by default (`uv="smart"`), packed per material. That is what a closed or strongly curved body
  needed — box projection folds its islands onto each other, so the map overlapped itself. The layout comes back from
  a headless bake onto the caller's surfaces, the glTF reader now reads `TEXCOORD_0` back (with the v flip), and the
  exporter splits vertices at the seams because glTF indexes one UV per vertex; a baked map therefore lands in the
  same place in the render, in the .glb and in the browser.
