# Twin benchmark: heated_channel (heated_channel)

Protocol: 200 readings at dt = 1 s, 80 members, seed 0, plant at {'H': 42.0}, sensor sigma [0.15, 0.15], limits warn/stop 244.9/309.1.

| section | measurement | value | verdict |
|---|---|---|---|
| state error | RMSE of T2 | 0.03909 | pass |
| state error | RMSE of persistence (the baseline to beat) | 27.07 | — |
| state error | mean NIS against its band [1.72, 2.3] | 2.17 | pass |
| state error | H: estimate 42.009 against the plant's 42 | 0.02% | — |
| state error | NEES check / forecast coverage | pass / pass | — |
| latency | per reading (mean / p95 / max) | 0.22 / 0.43 / 1.30 ms | — |
| latency | readings over the 1000 ms data period | 0.0% | pass |
| latency | one recalibration (inverse workflow) | 3.79 s against a 50.0 s window | pass |
| noise | plant readings at 0.5x the sigma the twin was told: RMSE 0.02692 (0.69x), mean NIS 0.577 | outside the band | — |
| noise | plant readings at 1x the sigma the twin was told: RMSE 0.03909 (1.00x), mean NIS 2.17 | consistent | — |
| noise | plant readings at 2x the sigma the twin was told: RMSE 0.0564 (1.44x), mean NIS 8.37 | outside the band | — |
| noise | plant readings at 4x the sigma the twin was told: RMSE 0.1481 (3.79x), mean NIS 33.8 | outside the band | — |
| sensor failure | dropout: RMSE 0.04092 (1.05x the healthy run), finite state True, alarmed False | — |
| sensor failure | outage: RMSE 0.04181 (1.07x the healthy run), finite state True, alarmed False | — |
| sensor failure | stuck: RMSE 3.205 (82.00x the healthy run), finite state True, alarmed True after 2 readings (anomaly) | — |
| sensor failure | bias: RMSE 0.752 (19.24x the healthy run), finite state True, alarmed True after 0 readings (anomaly) | — |
| anomaly | spikes detected | 15/18 (83%), delay 0 readings, 0.00 false alarms per 100 readings | review |
| RUL | not measurable on this case | the indicator has no usable trend at any checkpoint (R^2 0.06 to 0.10, against the 0.7 the prognostics module calls reliable): a smoothed residual does not degrade in a straight line while the filter follows the wear, so a linear extrapolation of it predicts nothing. The end of life the record shows is at reading 525, of 1180. | not_measurable |
| decision | actions agreeing with the truth's | 100.0% of 200 readings | pass |
| decision | cost: twin 0, never act 3500, perfect 0 | 0 per reading | — |
| decision | limits warn/stop 70.11/91.61 | the 60th and 85th percentile of the true outlet over the record (the exercise needs all three action classes to occur) | — |

Verdicts: state_error pass, latency pass, noise pass, sensor_failure pass, anomaly review, rul not_measurable, decision pass.
Measured in 25.77 s (quick sizes) on a synthetic plant.
