# Convection coefficient of a fin from five thermocouples (inverse PINN)

**Question.** A pin fin sits on an 80 °C wall in 25 °C air. How strong is the convection, i.e. what is the heat transfer
coefficient *h*? Textbook correlations for a real installation (obstructions, nearby walls, a fan somewhere) can easily be
off by 20–30 %, and *h* drives the heat the fin removes.

**Data.** Five thermocouples along the fin, ±0.5 °C noise. Nothing else.

**Method.** A physics-informed neural network (PINNeAPPle `PINNFactory`, symbolic residuals compiled to PyTorch)
learns the temperature field *T(x)* and *h* together. The loss holds the fin equation, the base temperature, the
convective tip condition (where *h* appears too) and the five readings; *h* is a trainable scalar that starts 4× off
(100 W/m²K). The network never sees the answer.

```
θ'' − m² θ = 0,   m² = 4h/(kD),   θ = (T − T∞)/(T_b − T∞)
θ(0) = 1                        base at 80 °C
−k θ'(L) = h θ(L)               convective tip
```

![result](results/fin_result_card.png)

## Results (10 independent noise draws, `results/summary.json`)

Stainless-steel pin fin, k = 16 W/m·K, Ø 5 mm × 50 mm, true h = 25 W/m²K (used only to generate the readings).

| Quantity | PINN | Check |
|---|---|---|
| h (W/m²K) | **24.8 ± 0.4** (24.2 to 25.3) | true 25.0 |
| Heat dissipated, Q = −kA dT/dx at the base (W) | **0.578 ± 0.005** | analytic 0.579 |
| Worst error on the temperature field, all draws (°C) | **0.74** | thermocouple noise is 0.5 |
| h by least-squares fit of the analytic profile (W/m²K) | 24.8 ± 0.3 | needs the formula |

The PINN is as accurate as fitting the textbook formula to the same readings, which is the best one can do with this
data, but it does not need a formula: change the geometry, add a heat source or make it 2D and the same script
applies, where no closed form exists. The fin was chosen because it has one, so every number above can be checked.

![h during training](results/fin_h_convergence.png)

## Run

```bash
python examples/use_cases/fin_convection_inverse/fin_convection_inverse.py   # about 5 minutes on a laptop CPU
python examples/use_cases/fin_convection_inverse/plots.py                    # figures again from summary.json
```

## What to change for your case

* `K, D, L, T_BASE, T_AIR` and `SENSORS_MM`: your fin and where the thermocouples are; `t_meas` are your readings.
* `NOISE_C`: your sensor accuracy (it only matters for the synthetic readings and the error bars).
* For a fin without an analytic solution, keep `identify()` and drop `theta_exact`, `q_exact` and the least-squares
  reference, which are here only to check the result.

Limitations: steady state, one-dimensional conduction (Biot number of the fin cross-section ≪ 1), uniform h along the
fin, and synthetic readings generated from the same model the PINN uses. Real measurements add model error
(radiation, h varying along the fin, thermocouple contact), which this example does not include.
