- Texture maps reach every consumer: a material entry may name `"base_color_map"`, `"roughness_map"` and
  `"normal_map"`, and the Blender renderer reads them (with the box-projected UVs of `scene.box_uvs`, which need no
  unwrap step), the glTF exporter embeds them in the GLB as `images`/`textures` with `TEXCOORD_0` (so the browser
  viewer gets them with no side-car files) and the USD exporter writes the base colour as a `UsdUVTexture` over
  `primvars:st`. The flat numbers of the entry stay as the fallback and for anything that cannot read a map.
- `pp.viz.bake_materials(scene, "maps/")` turns the procedural shaders of the material library into those maps: base
  colour, roughness and normal, taken from Cycles' DIFFUSE + GLOSSY and ROUGHNESS and NORMAL passes — not from one
  node of the tree, because a shader like `painted_metal` is two BSDFs mixed and a diffuse bake of a metal returns
  black, which would lose the bare metal the paint wears through to. It returns the entries ready to merge into
  `scene.materials`; `bake_materials_headless` does the same through the Blender binary where `bpy` is not importable.
  Known limitation, written in the doc: box projection folds the UV islands of a closed or strongly curved body onto
  each other, so a bbox-shaped part bakes cleanly and a curved one shows seams — per-loop UVs and a real unwrap are
  the next step.
