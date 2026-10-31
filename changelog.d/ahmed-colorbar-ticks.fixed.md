- Colour-bar ticks (`studio.colormap.ticks`, `studio-core.js`): a tick within rounding of zero prints as zero (no
  `-4.44e-16` on a symmetric range), and Python prints a zero tick as the browser does, so the JS parity test passes.
