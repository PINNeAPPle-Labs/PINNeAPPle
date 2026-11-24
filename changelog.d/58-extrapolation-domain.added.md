- Every benchmark case now states **where it holds**, and the error outside that envelope is measured, not estimated
  (#58, roadmap D7). `pinneapple_analysis.benchmark.validity` carries the two halves: a `ValidityDomain` (one range per
  variable, plus the source of those ranges — the case's own generator, a sample of it, or the training split — and
  `contains()`/`distance()`, the last one normalized so "how far outside" is comparable across variables and cases);
  and `study()`, which walks an axis outward from the edge of the envelope, evaluating the case's model against its
  reference at every step. The distance-0 step is the in-envelope control, so every later number is a measured error
  next to a measured baseline; `valid_until()` reads off the point where the error crosses the case's tolerance,
  refined between the two measured steps around the crossing, and says "not within the sampled range" rather than
  inventing an extrapolation. A `TrustGate` can be passed to record its OOD score next to each measured error.
- In the report, that is an **extrapolation column** in every case's table — the out-of-envelope error over the
  in-envelope one, with whether it stays inside the factor the case allows — plus the envelope printed under the
  problem's heading and stored in `summary.json` (`validity_domain`, `validity_text`). All five built-in cases
  declare one: the airfoil states the angle bins and the CST box its sections are drawn from, inverse heat states the
  power and convection-coefficient ranges, and the field-operator cases state the input value range measured on their
  training split (or on a sample of their own generator, before any data is prepared).
