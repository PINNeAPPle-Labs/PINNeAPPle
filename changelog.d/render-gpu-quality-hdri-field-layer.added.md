- `pp.viz.render` / `render_headless` now render on the GPU: `device="auto"` (the new default) takes the first
  Cycles backend the machine exposes — Metal on Apple silicon, OptiX or CUDA on NVIDIA, HIP on AMD, oneAPI on Intel —
  and falls back to the CPU when there is none, so the same script runs on a laptop and on a headless node;
  `device="cpu"`, `"gpu"` or a backend name force the choice. On an M5, the centrifugal impeller of the geometry
  library (13 744 triangles, 1600x900, 96 samples, denoised, dark stage with a floor) takes 62.3 s on the CPU and
  25.2 s on Metal, and the A320 airliner (34 560 triangles, 64 samples) 75.9 s against 23.5 s; only the first Metal
  render on a machine pays the shader compilation. The same `device` argument reaches `render_animation` and
  `particles.render_particle_frames`.
- Three render presets: `quality="preview"` (24 samples, 960x540), `"production"` (96, 1600x900 — the default, and
  the numbers the renderer always used) and `"cinematic"` (512, 3840x2160, 16 bounces); an explicit `samples=` or
  `size=` still wins, and `pp.viz.QUALITY` lists the table.
- HDRI environments: `background="hdri:<name-or-path>"` loads an environment image (`$PINNEAPPLE_HDRI_DIR`, default
  `data/hdri`, or a path, or a name registered in `pp.viz.HDRIS`) as both the light and what the metals reflect,
  with `hdri_strength` and `hdri_rotation`; `pp.viz.hdri_path(name)` says which file a name resolves to and, when
  there is none, where it looked.
- Scientific fields can now be a layer over the material instead of a repaint: `field_alpha=0.6` in `render()` mixes
  the colour map over each surface's own `MATERIALS` entry, so a bronze or steel part keeps its colour, roughness and
  highlights under a translucent temperature (or pressure, or stress) layer. `field=` alone still replaces the
  material, which is the reading a figure that only carries data wants.
