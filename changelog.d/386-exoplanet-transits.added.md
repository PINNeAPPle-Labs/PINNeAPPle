- `pinneapple_physics.astro.transits` (#386): a transit model with limb darkening (`transit_model`, `occultation_area`,
  `flux_of_separation`), the geometry that ties it to a star (`a_over_rstar`, `transit_durations`, `stellar_density`,
  `impact_parameter`), stellar variability (`stellar_variability`, a Matérn-3/2 GP) and the instrument systematics the
  issue names — momentum dumps and scattered light — plus a labelled generator (`synthetic_light_curve`) for six kinds:
  planet, alternating eclipsing binary, secondary eclipse, grazing eclipse, blend with a centroid shift, and a systematic
  sinusoid. `analyze` fits and vets, returning a `MeasurementCheck` whose verdict names the checks that fired
  (`odd_even`, `secondary`, `v_shape`, `shape_inconsistent`, `centroid_shift`). `benchmark` scores three pipelines on the
  same labelled set with Wilson intervals: measured here, BLS alone recovers 3/3 planets with a **0,800 false-positive
  rate**, the transit fit does not change that, and the vetting takes it to **0,000** with the recovery intact; AUC 0,889.
  `phase_fold_gif` draws the folded light curve with the fitted transit. `load_light_curve` names the MAST product and
  fails cleanly — no observed light curve is available offline, and the rates are synthetic, which the page says.
