- The material library grew an industrial set and, with it, a way to say what a flat table cannot:
  `pp.viz.MATERIALS` now has `brushed_steel`, `stainless_steel`, `galvanized_steel`, `cast_iron`, `painted_steel`,
  `oxidized_steel`, `rubber` and `industrial_glass`, and an entry may carry `"shader": "<name>"`, which Blender runs
  at render time (`pinneapple_tools.visualization.studio.pbr`): paint that chips through to bare metal where it wears,
  brushed stainless with streaks along the object's x and a roughness that runs from polished to dull, the spangle of
  hot-dip zinc, the pits of a sand casting, rust over bare metal in patches, concrete aggregate, rubber. The detail is
  driven by object coordinates, so it has a physical scale in metres and stays put when the body moves. Everything
  else in the pipeline still sees the flat numbers the entry declares — glTF, USD and the browser viewer export
  exactly what they exported before, so the shader improves the render without changing what a downstream tool gets.
  `concrete` gains its shader (the same material, with aggregate); the recipes are the ones the industrial pump-station
  example (`examples/use_cases/physics_data_factory/blender_factory_render.py`) proved out, rewritten against the
  table's parameters so any colour gets the same treatment, and `pp.viz.SHADERS` lists them for projects that want to
  register their own.
