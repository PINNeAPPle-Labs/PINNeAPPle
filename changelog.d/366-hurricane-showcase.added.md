- **The hurricane showcase's media** (`pinneapple_physics.weather.cyclones`, issue #366): the figures and the video
  the deliverable asks for, all of them drawn from the run's own arrays. `track_vs_observed` puts the official HURDAT2
  track and the ones each model actually followed on one map, with the map drawn at its own degrees and the median
  error of each line in the legend; `field_vs_grid` puts the same instant on the 0.25-degree analysis and on the
  5.625-degree archive side by side, one cell each, with isobars at the round pressures -- a closed ring around the
  core on one panel and an open line on the other, which is the measurement the caption states; `confidence_by_lead`
  is the confidence bar along the lead: green while the forecast is inside the track and intensity thresholds with the
  storm still found, amber past one of them, red where the grid lost it. `stages_video` is the 60-to-90-second video:
  the analysis crossing the globe, each forecast lead by lead and the coarse grid losing the core, every frame
  captioned with the instant and the stage and carrying the note the issue asks for -- a tropical cyclone's core is
  smaller than a global model's grid, which is why the chain has finer stages to hand over to. The video is written
  with `pinneapple_design.bricks.visuals.write_mp4`, so it needs `imageio[ffmpeg]` or the `ffmpeg` program and nothing
  else. `examples/weather_forecasting/hurricane_ian.py` grows the figures, the GIF cut and the video, and `--figures`
  publishes them next to the example instead of into the ignored `_out`.
