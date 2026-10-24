How far ahead can you predict El Niño, and does physics, a neural net or a PINN buy you more lead time? We put 30 models through one protocol to find out.

What we built: `pinneapple_systems.time_series.enso`, a library with recharge and delayed oscillators, an intermediate coupled ocean-atmosphere model (equatorial shallow-water ocean + Gill atmosphere), PINNs, a LIM, an SST-map CNN and an EnKF. One script, `examples/enso_forecasting/run_study.py`, scores all of them on public data (OISST, World Ocean Database, ERA5) with the `forecast_lab` protocol: 132 monthly test origins 2012-2022, leads 1-24 months.

Measured on the test period:
- Lead 6 anomaly correlation: CNN 0.70, LIM 0.67, seasonal recharge oscillator 0.63, ensemble 0.58, against 0.36 for persistence and 0.42 for ARIMA.
- Spring barrier: for June-August targets at lead 6, persistence drops to 0.06; the seasonal recharge oscillator keeps 0.64.
- The shallow-water ocean propagates a Kelvin wave at 2.689 m/s against 2.7 m/s in theory, converging at second order.

The limits are part of the result. Against climatology (Diebold-Mariano, Holm-corrected), every model is significantly better only at leads 1-2. Beyond about 9 months no model keeps a correlation above 0.5. The PINN that identifies the recharge oscillator fails on observations (lead-3 correlation -0.69). The coupled intermediate model is too damped (Niño-3.4 std 0.12 K vs 0.87 K observed), and its EnKF ensemble covers only 17 % of targets with its 95 % interval at lead 6.

Every number is read from committed JSON; every downloaded source file is pinned by SHA-256.

Guide and code: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/enso_forecasting.md

#ElNino #ENSO #ClimateScience #ScientificMachineLearning #PINN #Forecasting #Oceanography #OpenSource

---
## Media
1. `media/enso-30-models/01_el_nino_1997_98.png` — The 1997-98 El Niño from observations: SST on top, the 20 °C isotherm below, January vs December 1997.
2. `media/enso-30-models/02_skill_vs_lead.png` — Anomaly correlation vs lead for all 30 models; best four highlighted against persistence and ARIMA.
3. `media/enso-30-models/03_all_models_lead6.png` — All 30 models ranked at lead 6, coloured by family; significance vs climatology only at leads 1-2.
4. `media/enso-30-models/04_spring_barrier.png` — Skill by target month and lead: persistence collapses for summer targets, the seasonal recharge oscillator does not.
5. `media/enso-30-models/05_solver_checks.png` — Kelvin wave speed, Gill (1980) response and recharge-oscillator period checked against theory.
6. `media/enso-30-models/06_honest_failures.png` — What did not work: PINN identification, a too-damped coupled model, an under-dispersed EnKF.

## Sources
- 30 models -> examples/enso_forecasting/results/study.json, `zoo.models` (30 keys); results/tables.md "## zoo" (30 rows)
- 132 test origins, 2012-2022, leads 1-24 -> results/study.json `zoo.protocol.n_test_origins` = 132, `zoo.protocol.horizons`; docs/guides/enso_forecasting.md "The protocol"
- Lead-6 ACC CNN 0.70, LIM 0.67, seasonal RO 0.63, ensemble 0.58, persistence 0.36, ARIMA 0.42 -> results/study.json `enso_scores.<model>.acc[5]` (0.6992, 0.6712, 0.6284, 0.578, 0.3612, 0.4201); results/tables.md "## zoo" column ACC h=6; guide line 34
- Significant vs climatology only at leads 1-2 -> results/tables.md "## skill_vs_climatology" last column; guide lines 36-37
- No model above ACC 0.5 beyond ~9 months -> guide "Beyond about 9 months" (line 43); tables.md ACC h=9 max 0.50 (cnn), h=12 max 0.30
- Spring barrier: persistence 0.06, seasonal RO 0.64 (lead 6, Jun-Aug targets, CV 1985-2009) -> results/tables.md "## spring_barrier"; results/spring_barrier.json; guide line 45
- Kelvin wave 2.689 vs 2.7 m/s, second order -> results/solver_checks.json `kelvin.speed_measured` = 2.68867, `kelvin.speed_theory` = 2.7, `kelvin_convergence.observed_order` = [1.93, 1.98]
- (slide 5) Gill rel. L2 5.2e-5 -> solver_checks.json `gill.rows[3].rel_l2_u` = 5.1905e-05; RO period rel. error ~3e-6 -> `recharge_oscillator.rows[*].period_rel_error`
- PINN RO lead-3 ACC -0.69 -> study.json `enso_scores.pinn_ro.acc[2]` = -0.6866; tables.md zoo row pinn_ro; (slide 6) F2 ≈ 100 vs 1.3 -> guide line 49
- IM std 0.12 K vs 0.87 K observed -> guide lines 55-56 and 211-212 (from study.json `model_info.im_nudged`); (slide 6) tilt r = 0.85 -> guide line 55
- EnKF 17 % inside 95 % interval at lead 6 -> guide lines 56-57 and 292
- (slide 6) CRPS skill vs climatology < 0.07 beyond lead 9 -> guide line 44; tables.md "## families"
- SHA-256 pinning -> results/manifests/*.json.gz; guide "Data" section
