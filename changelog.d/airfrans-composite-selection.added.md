### Added

- `pinneapple_simulation.pipelines.airfrans_benchmark.train.composite_proxy`: the benchmark's composite on the terms
  that are measurable during training, and `WallSpec.select="composite"` to make `best.pt` follow it. Selecting on the
  validation wall slope instead costs about 1.5 points, because the slope's best epoch sits past the term's cap.
- `pinneapple_simulation.pipelines.airfrans_benchmark.boundary_layer`: Thwaites, Head and Michel as an integral
  boundary-layer method driven by an edge velocity — validated against Blasius to 1.2 % and the flat-plate law to 6 %.
