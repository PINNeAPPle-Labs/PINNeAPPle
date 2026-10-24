Give a neural network pure white noise and score it against persistence: it shows +0.33 RMSE skill. Most "my model beats the baseline" claims on noisy series are this trap.

So we built a protocol in PINNeAPPle that tries hard not to fool itself. `predictability_report` runs first: AR fit, entropies, Lyapunov estimate and surrogate tests (shuffle, IAAFT) say whether there is anything to predict before any model is fitted. Then `forecast_lab` runs a model zoo (AR/ARIMA and other baselines, GBM, MLP, LSTM, TCN, transformer, fitted ODEs, EnKF, hybrids) under one rule set: the last 20 % of each series is fixed as the test set before tuning and touched once; tuning uses purged walk-forward CV; every model is compared with the best baseline chosen on validation, with Diebold-Mariano tests, Holm-adjusted.

Measured on 16 series (`examples/near_random_forecasting`):
- That white-noise LSTM: zero skill against the best baseline, at every horizon.
- Lorenz-63 with observation noise of 0.1 × std: every model family keeps significant skill out to 1.4 Lyapunov times.
- A 546k-parameter MLP without early stopping: train RMSE 0.166, test 0.350. A 64x2 MLP with early stopping on a purged validation block: 0.245 and 0.248.

What did not work, as committed: on the linear stochastic series (incl. the double-well), white noise, a random walk and USD/EUR returns, nothing beat the baseline. On a real series, 205 days of YouTube US trending data, the transformer was closer than exponential smoothing on 68 % of the 34 test days, and still no model is significant at any horizon. With 34 days the test has little power; the guide says so.

Guide, code and every number: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/near_random_forecasting.md

#TimeSeries #Forecasting #MachineLearning #Statistics #ChaosTheory #ScientificMachineLearning #OpenSource

---
## Media
1. `media/near-random-forecasting/01_forecast_cone.png` — One GBM forecaster, horizons 1-48: the Lorenz-63 cone widens, the white-noise cone is wide from h = 1; 90 % band coverage 89.0 % / 89.4 %.
2. `media/near-random-forecasting/02_predictability_report.png` — Step 1, before any model: nearest-neighbour error vs shuffled and IAAFT surrogates for Lorenz-63, Ornstein-Uhlenbeck, white noise and USD/EUR.
3. `media/near-random-forecasting/03_persistence_trap.png` — White noise and USD/EUR: +0.3 skill against persistence, zero against the best baseline chosen on validation.
4. `media/near-random-forecasting/04_skill_vs_best_baseline.png` — Skill vs the best baseline with Diebold-Mariano/Holm significance: real on Lorenz-63 and Lorenz-96, none on Ornstein-Uhlenbeck, one year on sunspots.
5. `media/near-random-forecasting/05_memorise_vs_learn.png` — Double-well: an over-parametrised MLP memorises the training noise (0.166 -> 0.350); early stopping on a purged block does not (0.245 -> 0.248).
6. `media/near-random-forecasting/06_youtube_trending.png` — YouTube US trending, 34 untouched test days, day by day: model vs best baseline; nothing significant.
7. `media/near-random-forecasting/07_landscape.png` — Predictability landscape: skill vs best baseline per series and family; where to find the guide.

## Sources
- +0.33 RMSE skill of the white-noise LSTM vs persistence (h = 1) -> examples/near_random_forecasting/results/tables.md "## scores", row "White noise | lstm" (0.33 0.28 0.31 0.28 0.26)
- Zero skill vs best baseline, every horizon (white noise, also USD/EUR, random walk) -> tables.md "## skill_horizon" rows White noise / Random walk / USD/EUR ("none"); "## zoo" last column "none"; docs/guides/near_random_forecasting.md line 266 ("Noise controls and USD/EUR returns: nothing beats the baseline") and line 617 ("would have credited the noise controls with +0.3")
- Last 20 % fixed as test set; purged walk-forward CV; best baseline chosen on validation; one-sided DM with HLN correction, Holm-adjusted -> docs/guides/near_random_forecasting.md "## The protocol" (lines 43-74)
- 16 series -> tables.md "## predictability" (16 rows); results/cases/*.json (16 files)
- Lorenz-63 noise 0.1 × std: last horizon with significant skill 1.43 LT for all four families -> tables.md "## skill_horizon" row "Lorenz-63, noise 0.1 std" (32 (1.43 LT) for data, data+physics, physical, hybrid); guide line 251 ("1.4 LT for every family at noise 0.1"); noise definition guide line 94
- (slide 4) Lorenz-96 EnKF 2.7 LT -> tables.md "## skill_horizon" row Lorenz-96 (enkf 32 (2.68 LT)); guide line 256. Sunspots one year ahead -> same table row Sunspots (".../ 1"); guide lines 268-270
- Overfit MLP 546 312 params, RMSE train 0.166 / test 0.350; 64×2 + early stopping 0.245 / 0.248 -> guide "Memorise vs learn" table (lines 591-592); results/visuals.json key `memorise_vs_learn`; figure figures/visuals/memorise_vs_learn.png panel titles
- Linear stochastic series, double well, white noise, random walk, USD/EUR: nothing beat the baseline -> tables.md "## skill_horizon" ("none" as last h with skill for OU, oscillator, double-well, white noise, random walk, USD/EUR); guide lines 257-267
- YouTube: 205 days -> results/cases/youtube_us_log_views.json `n` = 205; 34 test days -> `zoo.protocol.n_test_origins` = 34; guide line 527 (2018-05-06 to 2018-06-08)
- YouTube 68 % of 34 days (transformer vs exp_smoothing, h = 1); 38 % (LSTM vs AR) -> results/visuals.json `surprise_calendar[0].fraction_model_closer` = 0.6765, `[1]` = 0.3824; guide line 602
- YouTube: no model significant at any horizon -> tables.md "## zoo" YouTube rows, last column "none"; guide lines 534-535; "little power" -> guide lines 535-536 and 660-661
- (slide 1) 90 % band coverage 89.0 % (Lorenz-63) and 89.4 % (white noise) -> guide line 558; results/visuals.json key `cones`
- (slide 2) surrogate p-values Lorenz-63 0.01/0.01, OU IAAFT 0.76, white noise 0.81/0.76, USD/EUR 0.01/0.01 -> tables.md "## predictability" (p shuffle, p IAAFT columns)
