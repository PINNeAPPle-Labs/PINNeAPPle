- **`pinneapple_analysis.events`: one scorer for any event, on any field** (#351): `Event` describes the event
  (`threshold`, `extremum`, `onset`, `track`) with a region, a lead window and `Tolerances`; `detect` finds it in one
  frame with its value and location; `event_score` scores a deterministic forecast set against the truth -- hits,
  misses, false alarms, critical success index, `lead_time_ok`, and the peak, position and timing errors -- and
  `ensemble_event_score` gives the probability of the event, its Brier score and the reliability-resolution
  decomposition. Measured on Lorenz-96 (K = 8, 40 random initial states, 30 leads, event "maximum above 3.0"):
  persistence scores CSI 0.245 and `lead_time_ok` 0, the truth itself 1.000 and 29, and the hits by lead fall from
  38 to 1 -- the decay the per-lead table shows and the single number does not. As a moving feature persistence puts
  the maximum 0.954 away in value and 1.045 grid cells away in position; 20 perturbed members give Brier 0.0145.
