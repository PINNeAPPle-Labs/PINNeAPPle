# The self-optimizing engineer: cooling the SoC

**Decision: APPROVED** (trust report below). Every number here was measured in this run.

## Mission

> Placa de controle 160 x 100 mm com um SoC de 40 W superaquecendo. Temperatura de junção máxima 90 °C, ar a 35 °C dentro da caixa. Cabe um dissipador de até 70 x 70 x 30 mm sobre o SoC e as ventoinhas podem gastar no máximo 0.02 W de potência de ar.

| Quantity | Value | Source |
|---|---|---|
| SoC power | 40 W | 40 W |
| Junction limit | 90 °C | junção máxima 90 °C |
| Air in the enclosure | 35 °C | ar a 35 °C |
| Heat-sink envelope | 70 x 70 x 30 mm | 70 x 70 x 30 mm |
| Fan air power budget | 0.02 W | ventoinhas podem gastar no máximo 0.02 W |
| Board | 160 x 100 mm, 4 layers | 160 x 100 mm |

Other parts (their own junction limits): VRM1 2 W (QFN-32 5x5, 125 °C), VRM2 2 W (QFN-32 5x5, 125 °C), DDR 1.5 W (BGA 17x17, 95 °C).

## Plan and evidence

Planner: `template`.

| # | Tool | Why | Result | Time [s] |
|---|---|---|---|---|
| 1 | `read_mission` | turn the request into numbers with their source | hot=SoC, power_w=40, tj_limit_c=90, t_ambient_c=35, envelope_mm=70 x 70 x 30, fan_power_budget_w=0.02, from_text=6 | 0.0 |
| 2 | `baseline` | how hot is it today | tj_hot_c=323.2, board_max_c=293.9, over_limit_k=233.2, seconds_solve=1.285 | 1.3 |
| 3 | `screen_natural` | is a passive heat sink enough | tj_hot_c=145.1, tj_hot_pessimistic_c=156.9, feasible=False | 47.4 |
| 4 | `sample` | reference solves to learn from | n=64, median_seconds=2.56, max_energy_balance=8.441e-11, all_converged=True | 49.9 |
| 5 | `learn` | fast approximation, measured on unseen designs, more solves until it is good enough | accepted=True, n_solves=224, tj_mae_k=0.1965, tj_max_abs_k=2.203, map_rmse_k=0.2032, coverage_2std=0.9861, nearest_tj_mae_k=5.165, speedup=1.384e+04 | 185.2 |
| 6 | `optimize` | search on the surrogate, confirm on physics | confirmed=True, rounds=1, mass_g=29.63, fan_power_w=0.01921, tj_hot_pessimistic_ref_c=88.53, max_abs_error_k=0.1751 | 22.2 |
| 7 | `verify` | independent finer solve and trust report | decision_str=APPROVED, tj_hot_fine_pessimistic_c=88.53, grid_max_dtj_k=1.467 | 6.7 |

## What happened

1. **Today**: bare board in still air, SoC junction **323.2 °C**, +233.2 K over the limit.
2. **Passive cooling is not enough**: the largest heat sink the envelope allows, in still air, gives 145.1 °C nominal / 156.9 °C pessimistic (best of 4 fin gaps).
3. **Learn**: 224 reference solves (median 2.56 s each, board energy balance <= 8.4e-11); surrogate measured on 45 held-out designs: junction |error| mean **0.20 K**, max **2.20 K**; board map RMSE **0.20 K**, max 1.51 K; +-2 std covers 99 % of the errors; nearest-snapshot baseline mean 5.17 K. Prediction 0.16 ms vs reference 2.16 s per design (13838x).
   Learning curve (hold-out per number of solves): 64 solves: max |error| 24.54 K, coverage 88 %; 96 solves: max |error| 3.50 K, coverage 89 %; 128 solves: max |error| 5.25 K, coverage 85 %; 160 solves: max |error| 5.56 K, coverage 81 %; 192 solves: max |error| 4.82 K, coverage 95 %; 224 solves: max |error| 2.20 K, coverage 99 %.
4. **Optimise and confirm**: 1 round(s) of surrogate search + reference check, confirmed.
   - round 1 (224 snapshots): 29.6 g, fan air 19.2 mW; SoC pessimistic predicted 88.35 +- 1.45 °C, reference 88.53 °C; max |error| over parts 0.18 K; meets the limits.
5. **Verify independently** (board grid 101 x 63 x 7, heat-sink base grid 56^2): SoC **83.76 °C** nominal, **88.53 °C** pessimistic (limit 90); largest change from the training grid 1.47 K.

## The design

| Parameter | Value |
|---|---|
| Base | 37.8 x 35.3 x 2.03 mm |
| Fins | 15 x 0.60 mm, 26.0 mm tall, gap 2.05 mm |
| Total height | 28.0 mm |
| Air speed | 1.99 m/s |
| Material | al6063 |
| Mass | 29.6 g |
| Fan air power (dp x flow) | 19.21 mW (9.9 Pa) |
| Heat up through the sink | 33.8 of 40 W |

| Part | Tj nominal [°C] | Tj pessimistic [°C] | Limit [°C] |
|---|---|---|---|
| SoC | 83.76 | 88.53 | 90 |
| VRM1 | 97.64 | 101.61 | 125 |
| VRM2 | 97.64 | 101.61 | 125 |
| DDR | 78.60 | 83.56 | 95 |

## Trust report

**Decisão:** APPROVED  
**Score:** 1.00  ·  **Cobertura:** 100%

| Checagem | Status | Crítica | Valor | Limite | Fonte |
|---|---|---|---|---|---|
| limit_tj_SoC | supports | sim | 88.525 | 90 | fine coupled solve, pessimistic cooling |
| limit_tj_VRM1 | supports | sim | 101.613 | 125 | fine coupled solve, pessimistic cooling |
| limit_tj_VRM2 | supports | sim | 101.613 | 125 | fine coupled solve, pessimistic cooling |
| limit_tj_DDR | supports | sim | 83.5641 | 95 | fine coupled solve, pessimistic cooling |
| fan_power_budget | supports | sim | 0.0192113 | 0.02 | closed form (dp x flow) |
| envelope | supports | sim | 28.0377 | 30 | geometry |
| energy_balance_board | supports | não | 1.38518e-11 | 1e-06 | pcb.solve: heat out / heat in - 1 |
| coupling_converged | supports | não | — | — | fixed point board <-> heat sink |
| heat_sink_energy_balance | supports | não | 8.42643e-13 | — | heatsink engine |
| heat_sink_spreading_methods_agree | supports | não | 0.00183079 | — | heatsink engine |
| heat_sink_grid_convergence | supports | não | 1.41299e-05 | — | heatsink engine |
| heat_sink_correlation_in_range | supports | não | — | — | heatsink engine |
| heat_sink_temperature_above_ambient | supports | não | 31.5746 | — | heatsink engine |
| grid_convergence_tj | supports | não | 1.46683 | 2 | train grid vs verify grid (max |dTj| over parts, nominal and pessimistic) |
| surrogate_within_band | supports | não | 1.54583 | 1.95732 | surrogate vs fine solve, 2 std + grid difference |
| surrogate_map_max_error | supports | não | 3.23132 | 3.46683 | board-top map, surrogate vs fine solve [K] |
| holdout_tj_max_error | supports | não | 2.20306 | 3 | 45 unseen designs [K] |
| holdout_beats_nearest | supports | não | 0.196499 | 5.16534 | mean |error| vs nearest-snapshot baseline [K] |
| holdout_coverage_2std | supports | não | 0.986111 | 0.9 | fraction of unseen errors inside +-2 std |
| inside_design_space | supports | não | — | — | box the snapshots were drawn from |
| nearest_snapshot_distance | supports | não | 0.550319 | 0.602076 | scaled distance to the nearest training design vs the held-out designs |


## Figures and 3-D

- maps: `figures/maps_before_after.png`
- evolution: `figures/tmax_evolution.png`
- holdout: `figures/surrogate_holdout.png`
- impact_gif_2d: `figures/impact_2d.gif`
- glb: `3d/verified_design.glb`
- usda: `3d/verified_design.usda`
- viewer: `3d/thermal_viewer.html`
- bare_png: `3d/bare_board.png`
- verified_png: `3d/verified_board.png`
- impact_gif_3d: `3d/impact/impact.gif`
- impact_mp4_3d: `3d/impact/impact.mp4`

## Model scope

- Reference = coupled 3-D board finite volumes (`pinneapple_physics.pcb`) and plate-fin heat-sink model (`pinneapple_design.thermal.heatsink`); both are engineering models with published correlations, not CFD. Pessimistic = board surface coefficients x 0.8, heat-sink coefficient x 0.85 (each engine's stated band).
- Air in the enclosure at a uniform temperature; air heating along the board and bypass around the sink are not modelled (the heat-sink model assumes ducted flow). The air paths of the 3-D scene are an energy balance per strip, for illustration, not a flow solution.
- Package thermal data: typical JEDEC values of `pcb_thermal.PACKAGES`; use datasheet values for sign-off.
- No conjugate CFD (OpenFOAM) was run here: it is not installed in this environment.

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
