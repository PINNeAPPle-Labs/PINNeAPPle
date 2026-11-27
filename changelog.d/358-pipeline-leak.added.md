- Leak detection on a buried pipeline (#358): `pinneapple_physics.pipeline_leak` solves the transient of a crude line
  by the method of characteristics from its steady state, with a pump curve upstream, a valve downstream and a hole as
  an internal boundary condition solved in closed form (`simulate` reproduces `H_L = H_0 - (B/2) C_v sqrt(H_L - H_amb)`
  to `1.3e-12`, and the flow each side of the hole takes `Q_L/2` to `2.1e-12`). Two transmitters are sampled, quantised
  and made noisy, with `Fault` (`offset`, `drift`, `noise`, `stuck`) injected per channel, and `detect` reads the
  residual against the nominal transient to say **leak**, **instrument** or **operation** -- a hole is seen by both
  ends and lowers the head there, a broken transmitter by one, and a valve the model was not told about moves both but
  leaves the mass balance unchanged. The leak is placed by the difference of the two arrival times and its rate taken
  from the settled imbalance; `measurement_check` is the mirror of `detect` for `PhysicsEnsemble(measurement_check=...)`.
  `examples/pipeline_leak` measures the numbers rather than claiming them: a 10 mm hole at 9.8 km of a 20 km line is
  found 8.30 s after it opens, against the 8.17 s the wave needs to reach the transmitter, placed within 4.5 m; over
  the sweep the 72 positions whose quieter front stood 2 sigma above the noise have a median error of 9.65 m and a
  worst of 89.4 m, and the 17 that did not are withheld rather than guessed at (their guesses would have run to 2.5 km).
  Six instrument faults out of six are called an instrument and there is no alarm in 23.3 h of normal operation --
  0.73/h at a decision interval of 8 sigma. Guide: `docs/guides/pipeline_leak.md`.
