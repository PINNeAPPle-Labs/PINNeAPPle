# Adaptive forecasting: switching and combining models online

`pinneapple_systems.time_series.AdaptiveForecaster` keeps a pool of forecasting models (experts) and decides, at
every step, how much to trust each one, using **only the errors they made on data they had not seen**. When the
series changes regime (seasonal, then trending, then a random walk), the weights move to the model that fits the new
regime within a few steps.

```python
from pinneapple_systems.time_series import AdaptiveForecaster, LagRegressorExpert, default_experts
from sklearn.ensemble import RandomForestRegressor

experts = default_experts(season_length=12)                     # naive, drift, SES x3, Holt, Holt-Winters, Theta, AR...
experts["forest"] = LagRegressorExpert(RandomForestRegressor(100), n_lags=12, refit_every=50)

af = AdaptiveForecaster(experts, horizon=6)                      # mode="select" to switch instead of combining
run = af.run(y, start=48)                                         # backtest, one origin at a time
run.errors(h=1)                                                   # ensemble vs every expert vs best-in-hindsight
run.coverage(h=1), run.switches()                                 # interval coverage, model changes over time

af.update(new_value)                                              # online use
af.forecast()                                                     # forecast, interval, weights, active model
```

## Why it does not overfit

* **Prequential**: every weight update uses the error of a forecast issued *before* its target was observed; the
  experts are refitted only on past data. A test checks it: changing the data after time t leaves every forecast
  issued up to t unchanged.
* **No hyper-parameter fitted on the evaluation period**: smoothing constants are separate experts (selected online),
  and the learning and switching rates of the aggregation are themselves chosen online by AdaHedge.
* **Guarantees, not tuning**: Fixed-Share has a regret bound against the best *sequence* of experts with a limited
  number of switches (Herbster & Warmuth 1998); AdaHedge has a regret bound against the best (eta, alpha) without any
  parameter (de Rooij et al. 2014). The ensemble cannot be much worse than the best model, and it can be better when
  the best model changes over time.

## How it works

1. Each expert issues forecasts for horizons 1..H at every origin.
2. When the target arrives, each expert's error (absolute or squared, divided by the mean absolute one-step change of
   the series so far, clipped) updates, for that horizon, a grid of Fixed-Share aggregators with different learning
   rates eta and switching rates alpha.
3. AdaHedge weights the aggregators; the forecast is the weighted average ("combine") or the forecast of the expert
   with the largest effective weight ("select").
4. Adaptive conformal inference (Gibbs & Candes 2021) gives intervals whose coverage is corrected after every miss.

Failed experts (exceptions, non-finite forecasts) are left out of that step and scored with the worst loss.

## Results (tests/test_adaptive_forecasting.py)

| Series | Ensemble MAE (h = 1) | Best single model in hindsight |
|---|---|---|
| seasonal -> trend -> random walk | 0.735 | 0.929 (AR) |
| seasonal only | 0.380 | 0.378 (AR) |

The "best single model in hindsight" is not available online: it is the model you would have picked had you known the
whole series. 90 % intervals cover 89 % of the outcomes on the regime-switching series.
