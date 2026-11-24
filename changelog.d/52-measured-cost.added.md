- `pinneapple_catalog.costs` — the measured cost table (#52, roadmap D1). One probe per catalog method, run for real:
  `pinneapple_catalog/method_costs.json` carries, per method, the probe that ran, what the size axis means, the per-size
  median wall time with its interquartile spread, the peak memory with the way it was sampled (`rss` with psutil,
  `tracemalloc` without), the FLOP count where a PyTorch path makes it countable (`flop_counter`, exact) or `null`
  where it does not, and the log-log scaling exponent with a bootstrap confidence interval and a verdict against the
  complexity the probe declares. The machine (CPU, cores, platform, Python, torch) is recorded with the numbers, and
  `compare_with_baseline` refuses to compare two tables measured on different machines: the same code on two CPUs is
  two measurements, not a regression.
- Coverage is a value, not a footnote: `costs.coverage()` returns one row per solver and training method —
  `measured`, or `unmeasured` **with the reason** (an external solver that is not installed, a GPU kernel, a method
  whose entry point needs a case only the user can build). `python scripts/measure_method_costs.py --check` fails when
  a method is missing from the table entirely, so a new solver cannot enter the catalog without either a probe or a
  stated reason. Measured so far: eleven methods across five families (spectral, FFT, beam BVP, Bekker-Wong,
  Mie, multilayer optics, Biot-Savart, stable fluids, shallow water, ETDRK4, the discrete-time PINN); the other
  sixty-two carry their reason in the file.
