- **Sensor placement as a combinatorial problem, with a budget and a spacing** (#352): `Layout`,
  `random_sensor_placement`, `maximin_sensor_placement`, `evolutionary_sensor_placement` and `layout_curves` in
  `pinneapple_analysis.data_assimilation.sensor_placement`; the greedy walk gains the two criteria the issue asks
  for -- the variance of a quantity of interest (`qoi=`) and a price per measurement with a budget (`cost=`,
  `budget=`) -- and the evolutionary search runs on `EvolutionaryDesignOptimizer` with a hard minimum distance
  between sensors, which the walk cannot express. Measured on a plate and on a slope, criterion against 1/2/4/8
  sensors: on the plate the walk goes 0.647 -> 0.0006 against 0.2134 for the sensor lottery; on the slope the search
  beats it at two and four sensors (0.2528 against 0.2913, 0.0907 against 0.0965), so the greedy guarantee is a
  bound and not optimality. At eight sensors on the plate the walk's closest pair is 0.063 apart; held to 0.15 the
  spaced search reaches 0.173 and 0.104, against 0.374 and 0.268 for the even spread.
