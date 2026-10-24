# The digital twin that knows when it is wrong

**Trust decision: APPROVED.** Every number here was measured in this run.

## Mission

> Acompanhe a linha de água quente (bomba, aquecedor, cotovelo, junção em T, ramais B e C) com o gêmeo digital. Diga quando a previsão deixar de ser confiável e por quê: o processo mudou, um sensor está errado ou o modelo saiu do domínio. Mostre a evidência, peça nova simulação quando precisar e meça quantas vezes você acerta a causa em casos controlados.

The loop: pump -> heater -> pipe A with a 90° elbow -> T junction -> branch B (valve, tank at 6 m) and branch C (valve, tank at 3 m). Ten sensors (4 pressures, 3 flow meters, 3 temperatures); five inputs (n, u_B, u_C, T_in, Q_heat_kW). Schematic in the 3-D scene, lengths not to scale (A 60 m, B 40 m, C 35 m).

## Plan and evidence

Planner: `template`.

| # | Tool | Why | Result | Time [s] |
|---|---|---|---|---|
| 1 | `read_mission` | what to watch and what to answer | n_sensors=10, n_inputs=5, hypotheses=normal / process / sensor / domain | 0.0 |
| 2 | `build_twin` | a fast model, measured before it is trusted | holdout_max_over_sigma=0.7828, holdout_median_over_sigma=0.02781, coverage_2std=0.9, n_train=120 | 3.2 |
| 3 | `calibrate` | alarm thresholds from healthy operation, not guessed | n_runs=12, threshold_shift=55.76, threshold_spread=5.039, threshold_model=43.13 | 18.2 |
| 4 | `monitor` | three events and a healthy hour | correct=4, n=4 | 5.8 |
| 5 | `request_simulation` | where the model is out of its domain, ask the physics | n=1, physics_consistent=True, old_error_over_sigma=3.583, new_error_over_sigma=0.07776, after=normal | 7.1 |
| 6 | `benchmark` | how often the cause is right | accuracy=0.9625, n_cases=80, cases_with_false_alarm=1 | 50.2 |
| 7 | `trust` | can the twin be trusted, and where | decision=APPROVED | 0.0 |

## The fast model

POD + Gaussian process on 120 reference solves of the operating envelope; on 40 held-out points the largest error is 0.78 sensor sigmas (median 0.028), +-2 std covers 90 %.

## Three events and a healthy hour

- **obstruction_B** (truth: process B, onset 25 min): diagnosed **process B**, first alarm 26.2 min. Kalman: extra K = 6.18 +- 0.16 (true 6). Next: pipe B: extra loss coefficient 6.18 +- 0.16 (Kalman); inspect it for a partial obstruction.
- **drift_Q_B** (truth: sensor Q_B, onset 25 min): diagnosed **sensor Q_B**, first alarm 27.0 min. Estimated drift: 2.26 (true 2.4 m3/h). Next: Q_B: bias -- check / recalibrate it; meanwhile estimate it from the other sensors.
- **overspeed** (truth: domain n, onset 25 min): diagnosed **domain n**, first alarm 26.0 min. Next: run the reference simulation at n = 1.15 and add it to the model; until then trust the physics residuals, not the fast model.
- **healthy** (truth: normal , onset 30 min): diagnosed **normal **, first alarm -. Next: none: physics, model and domain agree.

## Asking the physics

For **overspeed** the twin ran the reference at the new operating point: readings vs reference |z| <= 1.67 (the plant still obeys the physics), fast model error 3.6 sigma before, 0.08 sigma after adding 40 reference solves in the widened envelope; diagnosis after: **normal**.

## Controlled cases

80 cases (20 per kind), random onset (15-40 min), size and location; thresholds from 12 separate healthy runs. Obstructions: extra K 1-12 in a random pipe; sensors: bias drift 2-12 sigma or noise x2-8 on a random sensor; domain: pump speed 1.03-1.20, valve C 0.12-0.33, inlet 38-60 °C or heater 950-1300 kW (the small end of each range is meant to be hard).

| true \ diagnosed | normal | process | sensor | domain |
|---|---|---|---|---|
| normal | 20 | 0 | 0 | 0 |
| process | 0 | 20 | 0 | 0 |
| sensor | 3 | 0 | 17 | 0 |
| domain | 0 | 0 | 0 | 20 |

Cause right: **96.2 %**. Location right (pipe / sensor / input): process 20/20, sensor 17/17, domain 20/20; sensor fault mode (drift / noise) right 16/17. Kalman extra-K error: mean 0.0523, max 0.205, true value inside +-2 std in 18/20. Cases with an alarm before the fault (or any alarm when healthy): 1. Median detection delay after onset [s]: process 71, sensor 160, domain 181. Time: fast model 2.7 s, calibration 17.7 s, cases 29.7 s (1.45 s each).

Misdiagnosed cases:

- sensor_1: truth sensor Q_B (2.02) -> normal 
- sensor_8: truth sensor T_B (3.38) -> normal 
- sensor_16: truth sensor T_B (2.17) -> normal 

## Trust report

**Decisão:** APPROVED  
**Score:** 1.00  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| fast_model_holdout | supports | não | 0.782762 | 3 | 40 held-out operating points, max |error| / sensor sigma |
| fast_model_coverage_2std | supports | não | 0.9 | 0.9 | held-out errors inside +-2 std |
| cause_accuracy | supports | sim | 0.9625 | 0.9 | 80 controlled cases, diagnosed kind = true kind |
| location_accuracy | supports | não | 1 | 0.9 | right pipe / sensor / input among correctly classified faults |
| healthy_runs_stay_normal | supports | sim | 1 | 1 | healthy cases diagnosed normal at the end |
| false_alarm_cases | supports | não | 0.0125 | 0.1 | cases with any alarm before the fault (or at all, when healthy) |
| kalman_size_inside_2std | supports | não | 0.9 | 0.8 | obstructions whose true extra K is inside the filter's +-2 std |
| showcase_correct | supports | não | 4 | 4 | obstruction B, drifting Q_B, overspeed, healthy |
| reference_newton_residual | supports | não | 1.77636e-14 | 1e-08 | plant network solve, max head residual [m] |
| out_of_domain_physics_consistent | supports | não | 1.67403 | 4 | reference at the new operating point vs readings (30-step mean, |z|) |
| extended_model_inside_noise | supports | não | 0.0777605 | 3 | fast model after adding the reference solves, max |error| / sigma |


## Files

- control_room_obstruction_B: `figures/control_room_obstruction_B.png`
- control_room_drift_Q_B: `figures/control_room_drift_Q_B.png`
- control_room_overspeed: `figures/control_room_overspeed.png`
- control_room_healthy: `figures/control_room_healthy.png`
- confusion: `figures/confusion_matrix.png`
- request_simulation: `figures/request_simulation.png`
- control_room_after_new_simulation: `figures/control_room_after_new_simulation.png`
- impact_gif_2d: `figures/impact_overspeed.gif`
- viewer: `3d/twin_viewer.html`
- glb: `3d/loop_twin.glb`
- usda: `3d/loop_twin.usda`
- impact_gif_3d: `3d/obstruction/obstruction.gif`
- impact_mp4_3d: `3d/obstruction/obstruction.mp4`
- still_png: `3d/obstruction_last_frame.png`

## Model scope

- The plant is simulated (quasi-steady network: pump affinity curve, Darcy-Weisbach / Colebrook pipes, minor losses, valve law, heater and branch heat loss; noise of typical instrument classes). Faults and their sizes are injected, so every answer is known; a real plant adds unmodelled effects.
- The quasi-steady model holds for valve moves slower than ten acoustic round trips (`plant.quasi_steady_valid`); faster moves need the water-hammer solver (`pinneapple_simulation.numerical_solvers.water_hammer_moc`), not exercised here.
- The elbow streamlines in the 3-D scene are the 1/7 power-law profile, not a CFD solution.

## Environment

```
{
 "python": "3.13.17",
 "platform": "Linux-6.18.44-fc-v153-x86_64-with-glibc2.39",
 "numpy": "2.5.3",
 "scipy": "1.18.1",
 "cpus": 4,
 "cpu": "Intel(R) Xeon(R) Processor @ 2.10GHz",
 "scikit-learn": "1.9.1",
 "blender": "5.2.3 LTS"
}
```
