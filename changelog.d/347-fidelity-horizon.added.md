- The trust horizon as a condition of the adaptive-fidelity router (#347): `pinneapple_analysis.fidelity.serve` reads a
  `Horizon` (`horizon=`, one for the run or one per request) or the one a `Trust` passed as `trust` already carries,
  and sends a request to the solver with the `past_horizon` code when the lead it asks to cover is past the first lead
  at which the model measured itself to stop being trustworthy (`useful`, or `no_skill` when only that one was
  measured) — whatever its verdict, its domain and its interval say. A horizon that was given but measured nothing is
  not a pass: it escalates with `horizon_not_measured` unless `FidelityPolicy(unknown_horizon="surrogate")` says
  otherwise. Measured on the 400-request rig of `tests/test_fidelity.py`: the 237 requests the verdict, the domain and
  the tolerance accept are served by the surrogate at a 120 h useful lead with a 48 h request and none of them at a
  24 h one (0 answers beyond the tolerance either way, cost ratio 0.413 against 1.0).
