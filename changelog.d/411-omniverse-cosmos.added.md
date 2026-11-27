- `pinneapple_tools.nvidia.livelink` (#411, the Omniverse half): a forecast as an OpenUSD stage with the time samples in
  it — parts with transient fields, a pose and a scale per step, and sensors with their envelope — written, read back
  and played without Omniverse Kit, since `pxr` alone is enough for that path. The trust record of the forecast rides in
  the layer metadata as `pinneapple_trust`, flattened to JSON because a stage has no dict property. `track.check()`
  refuses a field whose number of steps disagrees with the times, before the stage is written. The Kit half
  (`KitLiveLink`, `kit_livelink_script`, `open_live_layer`) raises with `pinneapple[omniverse]` named when Kit is absent.
- `pinneapple_tools.nvidia.cosmos` (#411, the world-model half): the physical-plausibility judge for generated clips —
  rigid-body impulse/acceleration from the tracked object and mass/momentum/divergence from optical flow, limits
  calibrated on clean clips at a stated quantile, and a labelled set with violations injected at known frames. Measured
  on this machine, one seed: detection 0,727, false alarm 0,0, attribution 0,25, lag 1 frame; the horizon (three seeds)
  is 26,5 of the 48 frames. `BenchmarkReport.horizon()` returns a `pinneapple_analysis.trust.record.Horizon`, so the lead
  a generated clip buys is consumed by the same trust machinery as any other forecast. The Cosmos clients themselves stay
  behind their extra; the clips are rendered from a stated generator here.
