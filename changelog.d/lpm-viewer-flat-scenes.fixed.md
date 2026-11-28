- `benchmarks/lpm_figures.py`, `pinneapple_tools/visualization/studio/{scene.py,web/viewer.js}`: the 3-D viewer of a
  **flat family** is no longer blank. A 2-D family comes as a sheet, and two things made the airfoil and the exchanger
  draw an empty stage with a working colour bar: Delaunay returns each triangle in an arbitrary orientation and the
  viewer culls back faces, so the bodies faced away from the camera and were not drawn at all; and the default 3/4 view
  puts a sheet edge-on, where it is a line. Every triangle is now turned to the surface normal that points up, the row
  of samples is turned a quarter so it crosses the screen instead of standing on end, and a scene can name the preset
  the viewer opens on (`Scene.view`, default `iso`; a whole-scene flat test picks `top`). Clicking **Realistic** on
  those families showed the same empty stage for the same reason and now shows the four grey bodies. Eight new
  assertions across two tests pin the face direction, the opening preset, the 3-D case that must keep its oblique view,
  and the shelves still not overlapping. The figure cache also invalidates on a change to the drawing code, which is
  why the fixed viewers were not being rebuilt before.
