- A visual benchmark for the renderer: `python -m pinneapple_tools.visualization.studio.benchmark --out provas/`
  renders three reference cases — the centrifugal impeller in cast bronze on a machine-shop HDRI, the pin-fin heat
  sink in brushed aluminium under the studio light and the A320 with the material library in the warehouse — and
  writes a record per case with the time, the device, the picture's mean luminance and spread, how much of the frame
  the subject covers and the edge energy. `--baseline` (or `pp.viz.benchmark.compare`) prints the deltas of a later
  run. The numbers are the gate, not a pass/fail: they catch a render that went black, blew out, lost its micro-detail
  or stopped showing the subject, without the brittleness of a golden-image comparison across GPUs, drivers and
  Blender versions. `pp.viz.benchmark.CASES` lists the scenes; each is a geometry from the library with the materials,
  HDRI and camera preset a figure of it would use.
