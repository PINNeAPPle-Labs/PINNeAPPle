- **Sensor placement as a combinatorial problem, with a budget and a spacing** (#352): `Layout`,
  `random_sensor_placement`, `maximin_sensor_placement`, `evolutionary_sensor_placement` and `layout_curves` in
  `pinneapple_analysis.data_assimilation.sensor_placement`; the greedy walk gains the two criteria the issue asks
  for -- the variance of a quantity of interest (`qoi=`) and a price per measurement with a budget (`cost=`,
  `budget=`) -- and the evolutionary search runs on `EvolutionaryDesignOptimizer` with a hard minimum distance
  between sensors, which the walk cannot express. Measured on a plate and on a slope, criterion against 1/2/4/8
  sensors at the default search settings: on the plate the walk goes 0.647 -> 0.0006 against 0.0233 for the
  evolutionary search and 0.1906 for the median random layout; on the slope the search beats the walk at two and four
  sensors (0.2368 against 0.2913, 0.0872 against 0.0965) and loses at eight (0.0244 against 0.0193), so the greedy
  guarantee is a bound and not optimality. At eight sensors on the plate the walk's closest pair is 0.063 apart; held
  to 0.15 the spaced search reaches a closest pair of 0.173 at a criterion of 0.104, against 0.374 and 0.268 for the
  even spread.
