- `pp.viz.render` gained the photography side of a figure. `camera="product"` (85 mm on the third line, f/8),
  `"hero"` (35 mm from slightly below, f/2.8), `"documentary"` (50 mm, everything in focus, centred), `"head_on"`
  (100 mm down the main axis) and `"wide"` (24 mm) fill `lens`, `view`, `distance`, `fstop` and `composition` only
  where they are not given; `composition="thirds_left"` (or right/high/low) moves the subject off centre with a lens
  shift, so the framing the auto-fit computed still holds; `fstop` opens the aperture focused on the subject.
- Finishing after the render, in the new `pinneapple_tools.visualization.studio.post`: `look="low" … "high"`,
  `"very_high"` or `"punchy"` (an S-curve on the display values, plus a little saturation for punchy) and
  `post="bloom"`, `"vignette"` or `"bloom+vignette"`. They act on the written PNG/JPEG — not on an EXR, which is the
  input to a grade — and they live in Python rather than in the scene compositor because the compositor and the AgX
  contrast looks moved between Blender 4 and 5: a figure should not depend on which one is installed. In a headless
  render they run in the calling process, where Pillow exists.
- `passes=True` with an `.exr` output writes the beauty as a 32-bit OpenEXR and, next to it in `<out>_passes/`, a
  multilayer EXR with the depth, normal, mist, diffuse, glossy, emission and transmission passes, through File Output
  nodes of the compositor. `film_transparent=True` drops the world behind the geometry and keeps an alpha.
