- **Behavior:** a render can now be measurable. `pp.viz.render` / `render_headless` take `units=` (metres per scene
  coordinate unit: `"mm"`, `"cm"`, `"m"`, `"in"`, `"ft"` or a number, default `scene.unit` = 1 m), and with it
  `backdrop="cyclorama"` (a seamless infinity cove sized from the subject), `backdrop="floor"`, `grid=True` (a metric
  grid on that floor, squares about a tenth of the subject, snapped to 1/2/5 × 10ⁿ; `grid="25 mm"` for an explicit
  spacing), `scale_ref="auto"` (the largest of a 1.75 m mannequin, a 300 mm rule, a sheet of A4 or a 25.4 mm coin that
  does not dwarf the subject, built at its real size and framed with it), `scale_bar=True` (a metric bar burnt into
  the picture at the image scale of the subject's plane) and `dimensions=True` (the three extents, the grid square and
  the prop, top-left). The new `pp.viz.scale` module is the pure-numpy half: the unit table, the round-value ladder,
  and the mesh builders for the cove, the grid and the props.
- **Fixed:** `floor=True` was honoured only on `background="dark"` and silently dropped for every other background —
  including the HDRI cases the render benchmark itself ships — so those figures showed a part floating in a room with
  no ground, no contact shadow and nothing to measure it against. A floor is now built for whatever background is
  asked for, textured in the scene's real metres.
- The render benchmark's three reference cases now carry the metric set (cove, grid, scale reference, bar and
  extents), and `data/benchmark/baseline.json` was re-recorded for them with the reason on the record. The heatsink
  case moves from `camera="hero"` to `"product"`: a set has a floor, and the hero preset looks from slightly below
  it, so it would be raised to the floor line with a warning on every run — a measuring shot wants the 85 mm.
