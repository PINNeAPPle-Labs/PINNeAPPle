- `benchmarks/lpm_recipes.json`: `pp_airfoil` goes from 400 to **900 bodies** (3 600 designs, ~10.8 h for the fit on one
  machine). The campaign's first full run measured 0.710 held-out against a 0.974 constant-field baseline, and the
  distribution is the reason: the median design is 0.62 but p90 is 3.13 and **150 of 400 held-out bodies sit above 1.0**,
  which is worse than predicting the mean field. The loss curve over the same run had already flattened — the last 10 %
  of the 600 epochs moved it 0.6 % — so the lever is coverage of the design space, not epochs.
