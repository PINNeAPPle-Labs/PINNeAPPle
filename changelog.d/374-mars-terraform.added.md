- `pinneapple_physics.terraform` is a zonal energy-balance model of Mars with the CO2 cycle coupled to it and run for
  centuries: the insolation is integrated over Mars' own orbit (eccentricity 0.0934, obliquity 25.19 deg, and the
  areocentric longitude of perihelion **derived** from the pole and the J2000 elements to `250.99 deg`), the surface
  is regolith or CO2 frost, the greenhouse saturates, and the transport conserves the global mean to roundoff.
- the CO2 thermostat, which is the point: the frost point is Clausius-Clapeyron through the triple point (216.58 K,
  5.185 bar) and it reproduces the dry-ice sublimation point to 6 % and the winter cap temperature at 6.36 mbar to
  0.4 K. Its slope is 7.4 K per e-fold of pressure, so **a greenhouse warms the surface without warming the cap** --
  from 6.36 to 20 mbar the frost point outruns the greenhouse and the ice stays. Melting it takes energy aimed at
  the cap, and that is what the scenario does.
- the reservoirs are Jakosky & Edwards 2018, page by page, each carrying the quote it came from: the polar cap
  9 mbar, the regolith 40 mbar, the clathrate 150 mbar as their extreme upper limit, the near-surface carbonates
  12 mbar, and the '~1 bar' of deep carbonates their figure 3 calls inaccessible. The present atmosphere plus the
  polar cap is the paper's "less than 15 mbar total", which is where the model's own caps end up.
- the scenario -- 3 W/m2 of engineered greenhouse and 0.06 of dust on the caps, ramped over 150 years and held for
  350 -- takes the pressure from 6.05 to 12.80 mbar, the global mean from 210.96 to 217.26 K, and the frosted area
  from 100 % to 25 % of the cap region, and then all three stop: the CO2 that is left has nowhere to go.
- the ensemble sweeps the five things nobody has measured -- frost albedo, poleward diffusivity, regolith reservoir,
  greenhouse strength and the strength of the engineering itself -- and its spread on the global mean reaches the
  declared 2 K after **44 years**. Past that century the model cannot tell its own prediction from its own
  uncertainty. It never reaches 6 K, a tenth of the ~60 K liquid water needs: the model is unsure how much Mars
  would warm, never whether the attempt works.
- the verdict, measured rather than asserted: every accessible reservoir at its published upper bound (67.4 mbar)
  warms Mars by 11.0 K of the 60 it needs, and the only inventory that reaches the melting point is the 1.22 bar
  that includes the deep carbonates -- the reservoir the paper rules out. The model's mean temperature at the present
  6.36 mbar is 214.29 K against the fact sheet's 214 K, and nothing in it was fitted to that point.
- `examples/mars_terraform` writes a rotating globe with the caps retreating and a confidence bar that empties
  exactly at the trust horizon, the four curves behind it, and the JSON; its topography is **synthetic and says so**
  in every frame, because the repository has no MOLA data and this package does not invent a map of Mars.
- three honest negatives, all in the guide: the annual-mean model puts the polar band at 128.5 K where the observed
  annual mean is nearer 150 K, so the poleward diffusivity has to be calibrated weak (0.004 W/m2/K) to hold a cap at
  all; the model is bistable, so a cap that has melted does not come back; and the biggest reservoir, 40 mbar of
  adsorbed CO2, is released on a 10^4-year time scale, so over five centuries it contributes 5 % of itself.
