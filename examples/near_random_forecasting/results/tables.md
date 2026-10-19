## predictability

| series | AR order (R² 1-step) | perm. entropy | sample entropy | Rosenstein λ (true λ) | NPE / std | p shuffle | p IAAFT | p TRA IAAFT | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Lorenz-63, noise 0.01 std | 14 (0.99) | 0.43 | 0.44 | 0.89 (0.893) | 0.13 | 0.01 | 0.01 | 0.53 | nonlinear structure |
| Lorenz-63, noise 0.1 std | 13 (0.95) | 0.67 | 0.72 | 0.73 (0.893) | 0.21 | 0.01 | 0.01 | 0.75 | nonlinear structure |
| Lorenz-63, noise 0.3 std | 14 (0.80) | 0.91 | 1.38 | 0.64 (0.893) | 0.42 | 0.01 | 0.01 | 0.97 | nonlinear structure |
| Mackey-Glass, noise 0.01 std | 20 (0.99) | 0.59 | 0.62 | 0.01 (0.005) | 0.03 | 0.01 | 0.01 | 0.04 | nonlinear structure |
| Mackey-Glass, noise 0.1 std | 20 (0.94) | 0.64 | 0.86 | 0.01 (0.005) | 0.14 | 0.01 | 0.01 | 0.09 | nonlinear structure |
| Mackey-Glass, noise 0.3 std | 19 (0.81) | 0.83 | 1.50 | 0.01 (0.005) | 0.39 | 0.01 | 0.01 | 0.39 | nonlinear structure |
| Lorenz-96 (K=40), noise 0.1 std | 8 (0.95) | 0.74 | 0.82 | 0.76 (1.677) | 0.30 | 0.01 | 0.01 | 0.01 | nonlinear structure |
| Ornstein-Uhlenbeck | 2 (0.80) | 0.94 | 1.41 | 0.14 | 0.50 | 0.01 | 0.76 | 0.84 | linear structure only |
| Noise-driven damped oscillator | 14 (0.97) | 0.68 | 0.87 | 0.17 | 0.24 | 0.01 | 0.86 | 0.26 | linear structure only |
| Double-well (regime switching) | 6 (0.93) | 0.95 | 0.91 | 0.20 | 0.30 | 0.01 | 0.08 | 0.89 | linear structure only |
| White noise | 0 (0.00) | 1.00 | 2.20 | 0.00 | 1.05 | 0.81 | 0.76 | 0.88 | no structure found |
| Random walk | 0 (0.00) | 1.00 | 2.20 | -0.00 | 1.05 | 0.71 | 0.67 | 0.98 | no structure found (tested on differences) |
| USD/EUR daily log-returns | 0 (0.00) | 1.00 | 1.97 | 0.00 | 0.90 | 0.01 | 0.01 | 0.15 | nonlinear structure |
| Sunspots (yearly) | 9 (0.86) | 0.60 | 0.84 | 0.03 | 0.60 | 0.01 | 0.84 | 0.01 | nonlinear structure |
| YouTube US trending: log10 total daily views | 10 (0.97) | 0.74 | 0.53 | 0.02 | 1.78 | 1.00 | 1.00 | 0.03 | nonlinear structure |
| YouTube US trending: Entertainment slots per day | 1 (0.80) | 0.85 | 1.04 | 0.01 | 0.81 | 0.01 | 0.45 | 0.37 | linear structure only |

## skill_horizon

| series | horizons | data: best (val) / last h with skill | data+physics: best (val) / last h with skill | physical: best (val) / last h with skill | hybrid: best (val) / last h with skill |
|---|---|---|---|---|---|
| Lorenz-63, noise 0.01 std | 1, 2, 4, 8, 16, 32, 64, 128 | gbm: 128 (5.72 LT) / 32 (1.43 LT) | gbm_physics_features: 128 (5.72 LT) / 32 (1.43 LT) | enkf: 128 (5.72 LT) / 64 (2.86 LT) | ode_fitted+gbm: 128 (5.72 LT) / 64 (2.86 LT) |
| Lorenz-63, noise 0.1 std | 1, 2, 4, 8, 16, 32, 64, 128 | transformer: 128 (5.72 LT) / 32 (1.43 LT) | gbm_physics_features: 128 (5.72 LT) / 32 (1.43 LT) | enkf: 128 (5.72 LT) / 32 (1.43 LT) | ode_fitted+gbm: 128 (5.72 LT) / 32 (1.43 LT) |
| Lorenz-63, noise 0.3 std | 1, 2, 4, 8, 16, 32, 64, 128 | lstm: 128 (5.72 LT) / 32 (1.43 LT) | mlp_physics_loss: 128 (5.72 LT) / 32 (1.43 LT) | enkf: 128 (5.72 LT) / 32 (1.43 LT) | ode_missing_xz+gbm: 128 (5.72 LT) / 16 (0.71 LT) |
| Mackey-Glass, noise 0.01 std | 1, 2, 4, 8, 16, 32, 64, 128, 256 | tcn: 256 (3.76 LT) / 128 (1.88 LT) | gbm_physics_features: 256 (3.76 LT) / 128 (1.88 LT) | dde_fitted: 256 (3.76 LT) / 64 (0.94 LT) | dde_fitted+gbm: 256 (3.76 LT) / 128 (1.88 LT) |
| Mackey-Glass, noise 0.1 std | 1, 2, 4, 8, 16, 32, 64, 128, 256 | tcn: 256 (3.76 LT) / 64 (0.94 LT) | gbm_physics_features: 256 (3.76 LT) / 128 (1.88 LT) | dde_fitted: 256 (3.76 LT) / 32 (0.47 LT) | dde_fitted+gbm: 256 (3.76 LT) / 128 (1.88 LT) |
| Mackey-Glass, noise 0.3 std | 1, 2, 4, 8, 16, 32, 64, 128, 256 | transformer: 256 (3.76 LT) / 32 (0.47 LT) | gbm_physics_features: 256 (3.76 LT) / 32 (0.47 LT) | dde_fitted: 256 (3.76 LT) / 16 (0.24 LT) | dde_fitted+gbm: 256 (3.76 LT) / 32 (0.47 LT) |
| Lorenz-96 (K=40), noise 0.1 std | 1, 2, 4, 8, 16, 32, 64 | gbm: 64 (5.37 LT) / 8 (0.67 LT) | gbm_physics_features: 64 (5.37 LT) / 8 (0.67 LT) | enkf: 64 (5.37 LT) / 32 (2.68 LT) | ode_fitted+gbm: 64 (5.37 LT) / 16 (1.34 LT) |
| Ornstein-Uhlenbeck | 1, 2, 4, 8, 16, 32 | lstm: 32 / none | gbm_physics_features: 32 / none | ou_kalman: 32 / none | ou_kalman+gbm: 32 / none |
| Noise-driven damped oscillator | 1, 2, 4, 8, 16, 32, 64 | mlp: 64 / none | gbm_physics_features: 64 / none | oscillator_kalman: 64 / none | oscillator_kalman+gbm: 64 / none |
| Double-well (regime switching) | 1, 2, 4, 8, 16, 32, 64, 128 | transformer: 16 / none | gbm_physics_features: 16 / none | sde_poly3: 8 / none | sde_poly1_nn_closure: 16 / none |
| White noise | 1, 2, 4, 8, 16 | lstm: 16 / none | – | – | – |
| Random walk | 1, 2, 4, 8, 16 | transformer: none / none | – | – | – |
| USD/EUR daily log-returns | 1, 2, 5, 10, 20 | mlp: 20 / none | – | – | – |
| Sunspots (yearly) | 1, 2, 3, 5, 8, 11 | mlp: 8 / none | gbm_physics_features: 8 / 1 | oscillator_kalman: 8 / none | oscillator_kalman+gbm: 8 / 1 |
| YouTube US trending: log10 total daily views | 1, 2, 3, 5, 7 | transformer: none / none | – | – | – |
| YouTube US trending: Entertainment slots per day | 1, 2, 3, 5, 7 | lstm: none / none | – | – | – |

## scores

| series | model | family | RMSE skill vs persistence per h | CRPS skill vs climatology per h | coverage of the central 80 % and 95 % intervals (mean over h) | PIT deviation (pooled) | params | final fit [s] |
|---|---|---|---|---|---|---|---|---|
| Lorenz-63, noise 0.01 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.73 0.49 0.10 -0.25 -0.23 -0.26 -0.32 -0.38 | 0.78 (80 %), 0.94 (95 %) | 0.05 | 0 | 0.0 |
| Lorenz-63, noise 0.01 std | climatology | baseline | -2.74 -0.95 -0.12 0.19 0.25 0.24 0.26 0.28 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.74 (80 %), 0.97 (95 %) | 0.13 | 2 | 0.0 |
| Lorenz-63, noise 0.01 std | ar | baseline | 0.71 0.56 0.32 0.23 0.25 0.25 0.26 0.28 | 0.92 0.78 0.40 0.06 0.01 0.00 -0.00 -0.00 | 0.74 (80 %), 0.94 (95 %) | 0.12 | 16 | 0.4 |
| Lorenz-63, noise 0.01 std | phase_climatology | baseline | -2.98 -1.08 -0.21 0.13 0.20 0.20 0.21 0.23 | -0.06 -0.06 -0.07 -0.07 -0.06 -0.05 -0.06 -0.06 | 0.79 (80 %), 0.96 (95 %) | 0.11 | 668 | 0.6 |
| Lorenz-63, noise 0.01 std | gbm | data | 0.93 0.96 0.96 0.95 0.81 0.66 0.27 0.23 | 0.98 0.98 0.97 0.94 0.78 0.59 0.04 -0.06 | 0.83 (80 %), 0.95 (95 %) | 0.21 | 69600 | 3.5 |
| Lorenz-63, noise 0.01 std | gbm_physics_features | data+physics | 0.94 0.96 0.96 0.94 0.78 0.60 0.27 0.25 | 0.99 0.98 0.96 0.93 0.75 0.55 0.04 -0.04 | 0.85 (80 %), 0.95 (95 %) | 0.23 | 69600 | 3.1 |
| Lorenz-63, noise 0.01 std | enkf | physical | 0.96 0.98 0.99 0.98 0.96 0.88 0.74 0.29 | 0.99 0.99 0.99 0.98 0.97 0.95 0.82 0.16 | 0.57 (80 %), 0.70 (95 %) | 0.39 | 9 | 0.1 |
| Lorenz-63, noise 0.01 std | ode_fitted+gbm | hybrid | 0.95 0.98 0.98 0.97 0.89 0.80 0.51 0.25 | 0.99 0.99 0.98 0.97 0.85 0.76 0.42 -0.03 | 0.87 (80 %), 0.96 (95 %) | 0.43 | 10403 | 1.3 |
| Lorenz-63, noise 0.1 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.70 0.47 0.10 -0.25 -0.24 -0.27 -0.33 -0.38 | 0.78 (80 %), 0.94 (95 %) | 0.05 | 0 | 0.0 |
| Lorenz-63, noise 0.1 std | climatology | baseline | -2.31 -0.90 -0.11 0.19 0.25 0.24 0.26 0.27 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.76 (80 %), 0.97 (95 %) | 0.14 | 2 | 0.0 |
| Lorenz-63, noise 0.1 std | ar | baseline | 0.24 0.26 0.19 0.22 0.25 0.25 0.26 0.27 | 0.77 0.61 0.28 0.04 0.01 0.00 -0.00 -0.00 | 0.75 (80 %), 0.95 (95 %) | 0.11 | 15 | 0.9 |
| Lorenz-63, noise 0.1 std | phase_climatology | baseline | -2.51 -1.01 -0.19 0.14 0.20 0.20 0.22 0.23 | -0.05 -0.05 -0.05 -0.06 -0.05 -0.05 -0.06 -0.05 | 0.79 (80 %), 0.96 (95 %) | 0.12 | 669 | 0.8 |
| Lorenz-63, noise 0.1 std | transformer | data | 0.62 0.78 0.84 0.78 0.70 0.52 0.22 0.25 | 0.89 0.88 0.86 0.76 0.66 0.40 -0.05 -0.03 | 0.82 (80 %), 0.95 (95 %) | 0.11 | 18056 | 22.8 |
| Lorenz-63, noise 0.1 std | gbm_physics_features | data+physics | 0.61 0.77 0.84 0.78 0.64 0.42 0.23 0.25 | 0.88 0.88 0.86 0.76 0.58 0.28 -0.03 -0.04 | 0.82 (80 %), 0.95 (95 %) | 0.13 | 69600 | 3.2 |
| Lorenz-63, noise 0.1 std | enkf | physical | 0.65 0.80 0.87 0.84 0.74 0.62 0.31 0.27 | 0.88 0.88 0.88 0.84 0.78 0.65 0.21 0.05 | 0.57 (80 %), 0.71 (95 %) | 0.39 | 9 | 0.1 |
| Lorenz-63, noise 0.1 std | ode_fitted+gbm | hybrid | 0.61 0.78 0.84 0.77 0.64 0.45 0.25 0.25 | 0.88 0.88 0.86 0.76 0.60 0.32 -0.00 -0.04 | 0.81 (80 %), 0.95 (95 %) | 0.18 | 31203 | 2.7 |
| Lorenz-63, noise 0.3 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.53 0.37 0.05 -0.27 -0.26 -0.29 -0.35 -0.37 | 0.79 (80 %), 0.95 (95 %) | 0.05 | 0 | 0.0 |
| Lorenz-63, noise 0.3 std | climatology | baseline | -1.10 -0.59 -0.06 0.20 0.25 0.25 0.27 0.27 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.76 (80 %), 0.97 (95 %) | 0.15 | 2 | 0.0 |
| Lorenz-63, noise 0.3 std | ar | baseline | 0.05 0.09 0.14 0.22 0.26 0.25 0.27 0.27 | 0.55 0.43 0.19 0.03 0.01 0.00 0.00 -0.00 | 0.76 (80 %), 0.95 (95 %) | 0.12 | 15 | 0.1 |
| Lorenz-63, noise 0.3 std | phase_climatology | baseline | -1.22 -0.69 -0.13 0.15 0.20 0.20 0.22 0.23 | -0.05 -0.06 -0.06 -0.05 -0.06 -0.05 -0.06 -0.04 | 0.78 (80 %), 0.95 (95 %) | 0.17 | 670 | 1.6 |
| Lorenz-63, noise 0.3 std | lstm | data | 0.36 0.53 0.65 0.62 0.56 0.34 0.26 0.24 | 0.70 0.70 0.68 0.55 0.44 0.14 -0.01 -0.03 | 0.79 (80 %), 0.94 (95 %) | 0.05 | 18312 | 8.8 |
| Lorenz-63, noise 0.3 std | mlp_physics_loss | data+physics | 0.33 0.50 0.61 0.50 0.46 0.33 0.26 0.25 | 0.68 0.69 0.64 0.41 0.32 0.12 -0.01 -0.03 | 0.78 (80 %), 0.94 (95 %) | 0.09 | 67712 | 6.5 |
| Lorenz-63, noise 0.3 std | enkf | physical | 0.36 0.52 0.64 0.58 0.54 0.34 0.27 0.25 | 0.65 0.66 0.64 0.55 0.46 0.21 0.06 -0.00 | 0.55 (80 %), 0.70 (95 %) | 0.44 | 9 | 0.1 |
| Lorenz-63, noise 0.3 std | ode_missing_xz+gbm | hybrid | 0.33 0.49 0.60 0.51 0.50 0.30 0.26 0.25 | 0.68 0.68 0.63 0.44 0.36 0.09 -0.00 -0.03 | 0.80 (80 %), 0.95 (95 %) | 0.06 | 31203 | 2.9 |
| Mackey-Glass, noise 0.01 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.57 0.18 -0.41 -0.92 0.22 0.27 -0.15 -0.67 -0.64 | 0.77 (80 %), 0.97 (95 %) | 0.10 | 0 | 0.0 |
| Mackey-Glass, noise 0.01 std | climatology | baseline | -1.32 -0.22 0.29 0.47 -0.25 -0.32 0.14 0.40 0.39 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.79 (80 %), 0.96 (95 %) | 0.30 | 2 | 0.0 |
| Mackey-Glass, noise 0.01 std | ar | baseline | 0.74 0.64 0.65 0.71 0.12 0.18 0.33 0.43 0.39 | 0.89 0.71 0.52 0.46 0.31 0.39 0.22 0.06 0.00 | 0.79 (80 %), 0.96 (95 %) | 0.17 | 21 | 0.7 |
| Mackey-Glass, noise 0.01 std | phase_climatology | baseline | -0.20 0.34 0.60 0.71 0.29 0.28 0.52 0.60 0.61 | 0.49 0.47 0.45 0.45 0.44 0.46 0.46 0.33 0.37 | 0.79 (80 %), 0.94 (95 %) | 0.17 | 18 | 0.0 |
| Mackey-Glass, noise 0.01 std | tcn | data | 0.94 0.94 0.95 0.95 0.94 0.90 0.87 0.68 0.61 | 0.97 0.95 0.93 0.91 0.95 0.93 0.85 0.49 0.39 | 0.74 (80 %), 0.90 (95 %) | 0.16 | 29065 | 57.6 |
| Mackey-Glass, noise 0.01 std | gbm_physics_features | data+physics | 0.94 0.94 0.96 0.97 0.88 0.80 0.73 0.68 0.60 | 0.98 0.95 0.95 0.95 0.91 0.85 0.70 0.51 0.38 | 0.79 (80 %), 0.92 (95 %) | 0.08 | 78300 | 4.3 |
| Mackey-Glass, noise 0.01 std | dde_fitted | physical | 0.96 0.98 0.99 0.99 0.95 0.92 0.87 0.67 0.37 | 0.98 0.98 0.98 0.97 0.96 0.94 0.86 0.51 -0.00 | 0.82 (80 %), 0.94 (95 %) | 0.15 | 2 | 0.9 |
| Mackey-Glass, noise 0.01 std | dde_fitted+gbm | hybrid | 0.97 0.98 0.99 0.99 0.97 0.96 0.94 0.79 0.61 | 0.99 0.99 0.99 0.98 0.98 0.97 0.93 0.68 0.39 | 0.79 (80 %), 0.91 (95 %) | 0.07 | 78302 | 10.4 |
| Mackey-Glass, noise 0.1 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.54 0.18 -0.41 -0.92 0.22 0.26 -0.15 -0.67 -0.64 | 0.78 (80 %), 0.97 (95 %) | 0.09 | 0 | 0.0 |
| Mackey-Glass, noise 0.1 std | climatology | baseline | -1.18 -0.22 0.29 0.47 -0.25 -0.30 0.14 0.40 0.39 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.80 (80 %), 0.96 (95 %) | 0.29 | 2 | 0.0 |
| Mackey-Glass, noise 0.1 std | ar | baseline | 0.47 0.54 0.64 0.70 0.13 0.19 0.33 0.43 0.39 | 0.76 0.63 0.50 0.44 0.31 0.39 0.22 0.06 0.01 | 0.79 (80 %), 0.96 (95 %) | 0.15 | 22 | 0.0 |
| Mackey-Glass, noise 0.1 std | phase_climatology | baseline | -0.14 0.34 0.60 0.71 0.29 0.28 0.52 0.60 0.60 | 0.49 0.47 0.45 0.45 0.44 0.45 0.45 0.32 0.37 | 0.79 (80 %), 0.95 (95 %) | 0.12 | 18 | 0.0 |
| Mackey-Glass, noise 0.1 std | tcn | data | 0.73 0.85 0.89 0.90 0.80 0.75 0.72 0.63 0.60 | 0.88 0.88 0.85 0.82 0.85 0.81 0.68 0.41 0.38 | 0.79 (80 %), 0.92 (95 %) | 0.05 | 29065 | 44.3 |
| Mackey-Glass, noise 0.1 std | gbm_physics_features | data+physics | 0.71 0.83 0.89 0.92 0.73 0.61 0.60 0.63 0.59 | 0.87 0.86 0.85 0.85 0.79 0.71 0.55 0.42 0.36 | 0.81 (80 %), 0.95 (95 %) | 0.08 | 35100 | 4.0 |
| Mackey-Glass, noise 0.1 std | dde_fitted | physical | 0.71 0.84 0.91 0.92 0.76 0.66 0.57 0.54 0.33 | 0.87 0.87 0.87 0.85 0.82 0.76 0.54 0.27 -0.07 | 0.82 (80 %), 0.94 (95 %) | 0.14 | 2 | 1.1 |
| Mackey-Glass, noise 0.1 std | dde_fitted+gbm | hybrid | 0.72 0.85 0.91 0.93 0.78 0.71 0.65 0.64 0.58 | 0.87 0.88 0.88 0.86 0.83 0.79 0.62 0.43 0.35 | 0.81 (80 %), 0.94 (95 %) | 0.10 | 35102 | 9.9 |
| Mackey-Glass, noise 0.3 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.41 0.13 -0.41 -0.89 0.17 0.19 -0.15 -0.65 -0.64 | 0.79 (80 %), 0.97 (95 %) | 0.07 | 0 | 0.0 |
| Mackey-Glass, noise 0.3 std | climatology | baseline | -0.69 -0.16 0.29 0.47 -0.18 -0.21 0.14 0.39 0.39 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.80 (80 %), 0.96 (95 %) | 0.26 | 2 | 0.0 |
| Mackey-Glass, noise 0.3 std | ar | baseline | 0.25 0.42 0.59 0.66 0.15 0.20 0.31 0.42 0.39 | 0.57 0.50 0.43 0.37 0.29 0.35 0.20 0.06 0.01 | 0.81 (80 %), 0.96 (95 %) | 0.12 | 21 | 0.4 |
| Mackey-Glass, noise 0.3 std | phase_climatology | baseline | 0.03 0.33 0.57 0.68 0.29 0.28 0.48 0.58 0.58 | 0.43 0.42 0.41 0.41 0.40 0.41 0.41 0.30 0.33 | 0.81 (80 %), 0.96 (95 %) | 0.06 | 18 | 0.0 |
| Mackey-Glass, noise 0.3 std | transformer | data | 0.42 0.61 0.76 0.81 0.55 0.44 0.50 0.56 0.55 | 0.67 0.67 0.67 0.65 0.63 0.54 0.43 0.29 0.30 | 0.78 (80 %), 0.93 (95 %) | 0.08 | 18537 | 43.7 |
| Mackey-Glass, noise 0.3 std | gbm_physics_features | data+physics | 0.40 0.59 0.74 0.79 0.44 0.36 0.49 0.58 0.56 | 0.65 0.65 0.64 0.62 0.55 0.48 0.42 0.33 0.31 | 0.80 (80 %), 0.95 (95 %) | 0.07 | 35100 | 4.0 |
| Mackey-Glass, noise 0.3 std | dde_fitted | physical | 0.35 0.57 0.73 0.77 0.38 0.21 0.23 0.41 0.31 | 0.62 0.64 0.63 0.58 0.50 0.38 0.14 0.05 -0.09 | 0.83 (80 %), 0.95 (95 %) | 0.10 | 2 | 1.0 |
| Mackey-Glass, noise 0.3 std | dde_fitted+gbm | hybrid | 0.42 0.59 0.75 0.80 0.48 0.38 0.48 0.58 0.55 | 0.66 0.66 0.65 0.63 0.58 0.50 0.41 0.33 0.29 | 0.81 (80 %), 0.95 (95 %) | 0.06 | 35102 | 9.5 |
| Lorenz-96 (K=40), noise 0.1 std | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.72 0.49 0.11 -0.31 -0.48 -0.32 -0.43 | 0.82 (80 %), 0.95 (95 %) | 0.14 | 0 | 0.0 |
| Lorenz-96 (K=40), noise 0.1 std | climatology | baseline | -2.44 -0.89 -0.08 0.25 0.33 0.25 0.31 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.80 (80 %), 0.95 (95 %) | 0.14 | 2 | 0.0 |
| Lorenz-96 (K=40), noise 0.1 std | ar | baseline | 0.25 0.30 0.25 0.27 0.33 0.25 0.31 | 0.79 0.64 0.32 0.03 0.01 0.00 -0.00 | 0.80 (80 %), 0.95 (95 %) | 0.06 | 11 | 1.0 |
| Lorenz-96 (K=40), noise 0.1 std | phase_climatology | baseline | -2.19 -0.84 -0.12 0.18 0.25 0.14 0.23 | 0.09 0.04 -0.02 -0.08 -0.10 -0.12 -0.12 | 0.82 (80 %), 0.95 (95 %) | 0.10 | 33 | 0.0 |
| Lorenz-96 (K=40), noise 0.1 std | gbm | data | 0.40 0.52 0.49 0.43 0.35 0.24 0.26 | 0.83 0.76 0.55 0.26 0.05 -0.00 -0.06 | 0.79 (80 %), 0.93 (95 %) | 0.05 | 20300 | 16.2 |
| Lorenz-96 (K=40), noise 0.1 std | gbm_physics_features | data+physics | 0.52 0.67 0.62 0.47 0.37 0.24 0.26 | 0.86 0.83 0.66 0.31 0.08 -0.00 -0.07 | 0.80 (80 %), 0.94 (95 %) | 0.06 | 60900 | 14.2 |
| Lorenz-96 (K=40), noise 0.1 std | enkf | physical | 0.62 0.79 0.87 0.87 0.73 0.44 0.35 | 0.87 0.88 0.87 0.84 0.72 0.36 0.08 | 0.53 (80 %), 0.69 (95 %) | 0.46 | 81 | 0.6 |
| Lorenz-96 (K=40), noise 0.1 std | ode_fitted+gbm | hybrid | 0.53 0.74 0.81 0.80 0.63 0.31 0.27 | 0.86 0.86 0.83 0.75 0.51 0.12 -0.06 | 0.82 (80 %), 0.95 (95 %) | 0.10 | 9101 | 7.3 |
| Ornstein-Uhlenbeck | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 | 0.56 0.44 0.24 0.03 -0.21 -0.45 | 0.76 (80 %), 0.92 (95 %) | 0.10 | 0 | 0.0 |
| Ornstein-Uhlenbeck | climatology | baseline | -1.25 -0.78 -0.31 -0.03 0.17 0.29 | 0.00 0.00 0.00 0.00 0.00 0.00 | 0.78 (80 %), 0.93 (95 %) | 0.13 | 2 | 0.0 |
| Ornstein-Uhlenbeck | ar | baseline | 0.03 0.04 0.07 0.12 0.20 0.29 | 0.57 0.46 0.29 0.14 0.03 -0.01 | 0.77 (80 %), 0.93 (95 %) | 0.11 | 4 | 0.2 |
| Ornstein-Uhlenbeck | phase_climatology | baseline | -1.22 -0.76 -0.32 -0.06 0.13 0.25 | 0.01 0.00 -0.01 -0.03 -0.06 -0.08 | 0.74 (80 %), 0.94 (95 %) | 0.16 | 820 | 0.6 |
| Ornstein-Uhlenbeck | lstm | data | 0.02 0.04 0.07 0.12 0.20 0.29 | 0.57 0.46 0.29 0.14 0.02 -0.02 | 0.76 (80 %), 0.93 (95 %) | 0.11 | 17670 | 3.6 |
| Ornstein-Uhlenbeck | gbm_physics_features | data+physics | 0.03 0.04 0.07 0.12 0.19 0.28 | 0.57 0.46 0.29 0.15 0.02 -0.03 | 0.77 (80 %), 0.93 (95 %) | 0.09 | 7800 | 1.2 |
| Ornstein-Uhlenbeck | ou_kalman | physical | 0.03 0.04 0.07 0.12 0.20 0.29 | 0.57 0.46 0.29 0.14 0.03 -0.00 | 0.80 (80 %), 0.95 (95 %) | 0.09 | 4 | 0.6 |
| Ornstein-Uhlenbeck | ou_kalman+gbm | hybrid | 0.03 0.04 0.08 0.12 0.19 0.28 | 0.57 0.46 0.30 0.14 0.02 -0.03 | 0.78 (80 %), 0.93 (95 %) | 0.09 | 7804 | 1.2 |
| Noise-driven damped oscillator | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.70 0.48 0.03 -0.65 -0.76 -0.47 -0.60 | 0.84 (80 %), 0.97 (95 %) | 0.10 | 0 | 0.0 |
| Noise-driven damped oscillator | climatology | baseline | -2.35 -0.92 -0.03 0.39 0.43 0.32 0.37 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.81 (80 %), 0.96 (95 %) | 0.08 | 2 | 0.0 |
| Noise-driven damped oscillator | ar | baseline | 0.42 0.56 0.62 0.66 0.63 0.41 0.37 | 0.83 0.77 0.63 0.45 0.35 0.13 0.01 | 0.82 (80 %), 0.96 (95 %) | 0.06 | 12 | 0.1 |
| Noise-driven damped oscillator | phase_climatology | baseline | -1.71 -0.60 0.09 0.45 0.50 0.31 0.34 | 0.19 0.16 0.12 0.09 0.12 -0.02 -0.04 | 0.85 (80 %), 0.98 (95 %) | 0.12 | 27 | 0.0 |
| Noise-driven damped oscillator | mlp | data | 0.41 0.55 0.61 0.65 0.62 0.39 0.35 | 0.82 0.77 0.62 0.43 0.34 0.10 -0.04 | 0.80 (80 %), 0.96 (95 %) | 0.08 | 19591 | 1.5 |
| Noise-driven damped oscillator | gbm_physics_features | data+physics | 0.40 0.54 0.61 0.66 0.63 0.40 0.37 | 0.82 0.76 0.62 0.45 0.34 0.13 0.00 | 0.83 (80 %), 0.97 (95 %) | 0.07 | 9100 | 2.8 |
| Noise-driven damped oscillator | oscillator_kalman | physical | 0.42 0.56 0.62 0.66 0.63 0.41 0.37 | 0.83 0.77 0.63 0.45 0.35 0.13 0.01 | 0.80 (80 %), 0.95 (95 %) | 0.05 | 5 | 2.4 |
| Noise-driven damped oscillator | oscillator_kalman+gbm | hybrid | 0.42 0.56 0.62 0.67 0.63 0.41 0.36 | 0.83 0.77 0.63 0.45 0.35 0.13 -0.01 | 0.82 (80 %), 0.96 (95 %) | 0.05 | 9105 | 2.5 |
| Double-well (regime switching) | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.73 0.67 0.55 0.43 0.28 0.06 -0.18 -0.47 | 0.80 (80 %), 0.95 (95 %) | 0.06 | 0 | 0.0 |
| Double-well (regime switching) | climatology | baseline | -2.57 -1.82 -1.09 -0.65 -0.28 0.00 0.18 0.34 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.73 (80 %), 1.00 (95 %) | 0.45 | 2 | 0.0 |
| Double-well (regime switching) | ar | baseline | 0.03 0.02 0.04 0.05 0.08 0.14 0.21 0.33 | 0.74 0.67 0.56 0.46 0.33 0.16 0.04 0.00 | 0.80 (80 %), 0.96 (95 %) | 0.13 | 6 | 0.2 |
| Double-well (regime switching) | phase_climatology | baseline | -2.60 -1.89 -1.17 -0.73 -0.36 -0.08 0.12 0.26 | 0.01 -0.01 -0.02 -0.03 -0.04 -0.06 -0.07 -0.10 | 0.79 (80 %), 0.99 (95 %) | 0.44 | 490 | 0.8 |
| Double-well (regime switching) | transformer | data | 0.04 0.03 0.05 0.07 0.10 0.14 0.21 0.31 | 0.74 0.68 0.57 0.47 0.34 0.17 0.05 -0.03 | 0.78 (80 %), 0.95 (95 %) | 0.17 | 2936 | 8.5 |
| Double-well (regime switching) | gbm_physics_features | data+physics | 0.04 0.04 0.06 0.08 0.09 0.13 0.19 0.32 | 0.74 0.68 0.58 0.48 0.34 0.17 0.02 -0.01 | 0.81 (80 %), 0.95 (95 %) | 0.11 | 10400 | 0.5 |
| Double-well (regime switching) | sde_poly3 | physical | 0.04 0.05 0.06 0.07 0.09 0.13 0.21 0.34 | 0.75 0.69 0.59 0.50 0.37 0.19 0.06 0.03 | 0.77 (80 %), 0.94 (95 %) | 0.07 | 5 | 0.0 |
| Double-well (regime switching) | sde_poly1_nn_closure | hybrid | 0.04 0.04 0.06 0.08 0.09 0.13 0.20 0.33 | 0.75 0.68 0.59 0.50 0.37 0.19 0.05 0.01 | 0.77 (80 %), 0.94 (95 %) | 0.08 | 324 | 2.9 |
| White noise | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 | -0.50 -0.38 -0.46 -0.40 -0.36 | 0.79 (80 %), 0.95 (95 %) | 0.04 | 0 | 0.0 |
| White noise | climatology | baseline | 0.34 0.28 0.31 0.28 0.26 | 0.00 0.00 0.00 0.00 0.00 | 0.80 (80 %), 0.94 (95 %) | 0.07 | 2 | 0.0 |
| White noise | ar | baseline | 0.34 0.28 0.31 0.28 0.26 | 0.00 0.00 0.00 0.00 0.00 | 0.80 (80 %), 0.94 (95 %) | 0.07 | 2 | 0.7 |
| White noise | phase_climatology | baseline | 0.33 0.27 0.31 0.28 0.25 | -0.01 -0.01 -0.01 -0.01 -0.02 | 0.79 (80 %), 0.94 (95 %) | 0.05 | 46 | 0.0 |
| White noise | lstm | data | 0.33 0.28 0.31 0.28 0.26 | -0.00 -0.00 0.00 -0.00 0.00 | 0.80 (80 %), 0.94 (95 %) | 0.08 | 4709 | 3.3 |
| Random walk | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 | 0.96 0.95 0.92 0.89 0.84 | 0.77 (80 %), 0.93 (95 %) | 0.09 | 0 | 0.0 |
| Random walk | climatology | baseline | -25.31 -17.02 -11.81 -8.09 -5.11 | 0.00 0.00 0.00 0.00 0.00 | 0.97 (80 %), 1.00 (95 %) | 1.31 | 2 | 0.0 |
| Random walk | ar | baseline | -0.00 -0.00 -0.00 -0.00 -0.01 | 0.96 0.95 0.92 0.89 0.84 | 0.76 (80 %), 0.93 (95 %) | 0.10 | 3 | 0.1 |
| Random walk | phase_climatology | baseline | -12.65 -8.47 -5.80 -3.95 -2.40 | 0.49 0.49 0.48 0.47 0.45 | 0.93 (80 %), 0.98 (95 %) | 0.39 | 846 | 0.6 |
| Random walk | transformer | data | -0.00 -0.00 -0.00 -0.00 0.00 | 0.96 0.95 0.92 0.89 0.84 | 0.76 (80 %), 0.93 (95 %) | 0.10 | 17893 | 5.4 |
| USD/EUR daily log-returns | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 | -0.36 -0.36 -0.37 -0.40 -0.38 | 0.76 (80 %), 0.90 (95 %) | 0.10 | 0 | 0.0 |
| USD/EUR daily log-returns | climatology | baseline | 0.29 0.28 0.29 0.31 0.30 | 0.00 0.00 0.00 0.00 0.00 | 0.91 (80 %), 0.98 (95 %) | 0.42 | 2 | 0.0 |
| USD/EUR daily log-returns | ar | baseline | 0.29 0.28 0.29 0.31 0.30 | 0.04 0.04 0.04 0.04 0.04 | 0.77 (80 %), 0.90 (95 %) | 0.13 | 2 | 0.2 |
| USD/EUR daily log-returns | phase_climatology | baseline | 0.26 0.25 0.27 0.27 0.27 | 0.01 -0.00 0.01 -0.02 -0.01 | 0.77 (80 %), 0.90 (95 %) | 0.11 | 5 | 0.0 |
| USD/EUR daily log-returns | mlp | data | 0.29 0.28 0.29 0.30 0.29 | 0.04 0.04 0.05 0.04 0.04 | 0.77 (80 %), 0.90 (95 %) | 0.12 | 18309 | 1.4 |
| Sunspots (yearly) | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 0.00 | 0.44 -0.05 -0.49 -0.86 -0.26 0.40 | 0.46 (80 %), 0.70 (95 %) | 0.68 | 0 | 0.0 |
| Sunspots (yearly) | climatology | baseline | -0.79 0.01 0.27 0.40 0.16 -0.65 | 0.00 0.00 0.00 0.00 0.00 0.00 | 0.58 (80 %), 0.77 (95 %) | 0.75 | 2 | 0.0 |
| Sunspots (yearly) | ar | baseline | 0.35 0.48 0.55 0.61 0.47 -0.07 | 0.65 0.52 0.43 0.38 0.41 0.40 | 0.64 (80 %), 0.81 (95 %) | 0.50 | 20 | 0.0 |
| Sunspots (yearly) | phase_climatology | baseline | -0.17 0.30 0.46 0.56 0.40 -0.20 | 0.27 0.23 0.22 0.22 0.24 0.24 | 0.40 (80 %), 0.63 (95 %) | 0.81 | 12 | 0.0 |
| Sunspots (yearly) | mlp | data | 0.39 0.45 0.50 0.51 0.28 -0.06 | 0.67 0.48 0.35 0.20 0.09 0.38 | 0.58 (80 %), 0.72 (95 %) | 0.56 | 18438 | 0.6 |
| Sunspots (yearly) | gbm_physics_features | data+physics | 0.48 0.48 0.50 0.54 0.28 0.02 | 0.72 0.53 0.35 0.22 0.11 0.44 | 0.61 (80 %), 0.77 (95 %) | 0.37 | 7386 | 0.8 |
| Sunspots (yearly) | oscillator_kalman | physical | 0.31 0.36 0.38 0.44 0.24 -0.48 | 0.62 0.41 0.18 0.07 0.12 0.12 | 0.62 (80 %), 0.81 (95 %) | 0.52 | 5 | 1.1 |
| Sunspots (yearly) | oscillator_kalman+gbm | hybrid | 0.47 0.52 0.54 0.57 0.35 -0.08 | 0.72 0.58 0.44 0.33 0.22 0.37 | 0.68 (80 %), 0.81 (95 %) | 0.43 | 7529 | 1.2 |
| YouTube US trending: log10 total daily views | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 | 0.95 0.92 0.92 0.92 0.91 | 0.83 (80 %), 0.95 (95 %) | 0.25 | 0 | 0.0 |
| YouTube US trending: log10 total daily views | climatology | baseline | -10.04 -7.58 -7.23 -7.45 -7.09 | 0.00 0.00 0.00 0.00 0.00 | 0.00 (80 %), 0.00 (95 %) | 1.80 | 2 | 0.0 |
| YouTube US trending: log10 total daily views | ar | baseline | -0.48 -0.87 -1.35 -2.47 -3.18 | 0.91 0.84 0.80 0.71 0.63 | 0.50 (80 %), 0.92 (95 %) | 1.16 | 14 | 0.0 |
| YouTube US trending: log10 total daily views | phase_climatology | baseline | -0.47 -0.27 -0.28 -0.31 -0.28 | 0.91 0.90 0.89 0.88 0.88 | 0.98 (80 %), 1.00 (95 %) | 0.82 | 19 | 0.0 |
| YouTube US trending: log10 total daily views | transformer | data | -0.00 0.02 -0.12 -0.43 0.02 | 0.95 0.93 0.91 0.88 0.92 | 0.73 (80 %), 0.88 (95 %) | 0.32 | 17637 | 0.1 |
| YouTube US trending: Entertainment slots per day | persistence | baseline | 0.00 0.00 0.00 0.00 0.00 | 0.76 0.66 0.61 0.49 0.46 | 0.84 (80 %), 0.95 (95 %) | 0.41 | 0 | 0.0 |
| YouTube US trending: Entertainment slots per day | climatology | baseline | -2.50 -1.50 -1.31 -0.85 -0.84 | 0.00 0.00 0.00 0.00 0.00 | 0.49 (80 %), 0.69 (95 %) | 1.19 | 2 | 0.0 |
| YouTube US trending: Entertainment slots per day | ar | baseline | -0.01 -0.01 -0.05 -0.06 -0.15 | 0.76 0.67 0.60 0.48 0.42 | 0.76 (80 %), 0.89 (95 %) | 0.46 | 3 | 0.0 |
| YouTube US trending: Entertainment slots per day | phase_climatology | baseline | -1.71 -0.95 -0.89 -0.63 -0.88 | 0.31 0.29 0.25 0.18 0.01 | 0.86 (80 %), 0.97 (95 %) | 0.54 | 24 | 0.0 |
| YouTube US trending: Entertainment slots per day | lstm | data | -0.13 -0.12 -0.17 -0.13 -0.08 | 0.73 0.62 0.54 0.42 0.44 | 0.71 (80 %), 0.89 (95 %) | 0.54 | 17605 | 0.7 |

## zoo

| series | model | family | params | tuning configs | tuning [s] | final fit [s] | test RMSE skill (mean over h) | significant vs best baseline (Holm) at h |
|---|---|---|---|---|---|---|---|---|
| Lorenz-63, noise 0.01 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Lorenz-63, noise 0.01 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.32 | none |
| Lorenz-63, noise 0.01 std | ar | baseline | 16 | 1 | 0.0 | 0.4 | 0.36 | none |
| Lorenz-63, noise 0.01 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | -0.00 | none |
| Lorenz-63, noise 0.01 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.93 | none |
| Lorenz-63, noise 0.01 std | phase_climatology | baseline | 668 | 2 | 6.3 | 0.6 | -0.41 | none |
| Lorenz-63, noise 0.01 std | arima | baseline | 4 | 1 | 0.0 | 6.9 | 0.22 | none |
| Lorenz-63, noise 0.01 std | gbm | data | 69600 | 4 | 46.1 | 3.5 | 0.72 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | mlp | data | 20744 | 4 | 34.0 | 2.9 | 0.66 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | lstm | data | 18312 | 4 | 102.2 | 13.6 | 0.64 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | tcn | data | 16616 | 4 | 351.6 | 19.4 | 0.68 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | transformer | data | 18056 | 4 | 175.8 | 22.5 | 0.69 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | gbm_physics_features | data+physics | 69600 | 4 | 25.0 | 3.1 | 0.71 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | mlp_physics_loss | data+physics | 67712 | 4 | 65.3 | 4.2 | 0.63 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | ode_fitted | physical | 3 | 1 | 0.0 | 0.1 | 0.76 | 1, 2, 4, 8, 16, 32, 64 |
| Lorenz-63, noise 0.01 std | ode_missing_xz | physical | 3 | 1 | 0.0 | 0.1 | 0.34 | none |
| Lorenz-63, noise 0.01 std | sindy | physical | 24 | 4 | 5.5 | 0.3 | 0.50 | 1, 2, 4, 8 |
| Lorenz-63, noise 0.01 std | enkf | physical | 9 | 1 | 0.0 | 0.1 | 0.85 | 1, 2, 4, 8, 16, 32, 64 |
| Lorenz-63, noise 0.01 std | ode_fitted+gbm | hybrid | 10403 | 4 | 30.4 | 1.3 | 0.79 | 1, 2, 4, 8, 16, 32, 64 |
| Lorenz-63, noise 0.01 std | ode_missing_xz+gbm | hybrid | 69603 | 4 | 29.6 | 3.8 | 0.71 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.01 std | ude_missing_xz | hybrid | 4614 | 4 | 575.5 | 57.6 | 0.72 | 1, 2, 4, 8, 16, 32, 64 |
| Lorenz-63, noise 0.1 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Lorenz-63, noise 0.1 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.26 | none |
| Lorenz-63, noise 0.1 std | ar | baseline | 15 | 1 | 0.0 | 0.9 | 0.24 | none |
| Lorenz-63, noise 0.1 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.1 | -0.00 | none |
| Lorenz-63, noise 0.1 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.85 | none |
| Lorenz-63, noise 0.1 std | phase_climatology | baseline | 669 | 2 | 6.6 | 0.8 | -0.34 | none |
| Lorenz-63, noise 0.1 std | arima | baseline | 5 | 1 | 0.0 | 6.4 | 0.23 | none |
| Lorenz-63, noise 0.1 std | gbm | data | 69600 | 4 | 45.5 | 3.5 | 0.58 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | mlp | data | 20744 | 4 | 33.8 | 3.1 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | lstm | data | 18312 | 4 | 103.6 | 14.5 | 0.58 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | tcn | data | 16616 | 4 | 342.7 | 19.3 | 0.58 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | transformer | data | 18056 | 4 | 180.6 | 22.8 | 0.59 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | gbm_physics_features | data+physics | 69600 | 4 | 24.7 | 3.2 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | mlp_physics_loss | data+physics | 67712 | 4 | 65.6 | 6.8 | 0.54 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | ode_fitted | physical | 3 | 1 | 0.0 | 0.1 | 0.47 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.1 std | ode_missing_xz | physical | 3 | 1 | 0.0 | 0.1 | 0.32 | 1, 2, 4 |
| Lorenz-63, noise 0.1 std | sindy | physical | 28 | 4 | 5.6 | 0.2 | 0.39 | 1, 2, 4, 8 |
| Lorenz-63, noise 0.1 std | enkf | physical | 9 | 1 | 0.0 | 0.1 | 0.64 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | ode_fitted+gbm | hybrid | 31203 | 4 | 31.7 | 2.7 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | ode_missing_xz+gbm | hybrid | 31203 | 4 | 29.6 | 3.2 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.1 std | ude_missing_xz | hybrid | 1286 | 4 | 528.2 | 48.9 | 0.45 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.3 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Lorenz-63, noise 0.3 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.06 | none |
| Lorenz-63, noise 0.3 std | ar | baseline | 15 | 1 | 0.0 | 0.1 | 0.19 | none |
| Lorenz-63, noise 0.3 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| Lorenz-63, noise 0.3 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.58 | none |
| Lorenz-63, noise 0.3 std | phase_climatology | baseline | 670 | 2 | 3.7 | 1.6 | -0.13 | none |
| Lorenz-63, noise 0.3 std | arima | baseline | 5 | 1 | 0.0 | 8.2 | 0.18 | none |
| Lorenz-63, noise 0.3 std | gbm | data | 69600 | 4 | 46.4 | 5.6 | 0.41 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.3 std | mlp | data | 20744 | 4 | 32.7 | 2.9 | 0.42 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | lstm | data | 18312 | 4 | 100.2 | 8.8 | 0.45 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | tcn | data | 16616 | 4 | 234.9 | 15.2 | 0.42 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | transformer | data | 2968 | 4 | 185.9 | 15.8 | 0.41 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | gbm_physics_features | data+physics | 31200 | 4 | 26.7 | 2.8 | 0.40 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.3 std | mlp_physics_loss | data+physics | 67712 | 4 | 65.4 | 6.5 | 0.40 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | ode_fitted | physical | 3 | 1 | 0.0 | 0.1 | 0.25 | 1, 2, 4, 8 |
| Lorenz-63, noise 0.3 std | ode_missing_xz | physical | 3 | 1 | 0.0 | 0.1 | 0.26 | 1, 2, 4 |
| Lorenz-63, noise 0.3 std | sindy | physical | 24 | 4 | 5.8 | 0.3 | 0.19 | 1, 2, 4, 8 |
| Lorenz-63, noise 0.3 std | enkf | physical | 9 | 1 | 0.0 | 0.1 | 0.44 | 1, 2, 4, 8, 16, 32 |
| Lorenz-63, noise 0.3 std | ode_fitted+gbm | hybrid | 31203 | 4 | 31.4 | 3.4 | 0.40 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.3 std | ode_missing_xz+gbm | hybrid | 31203 | 4 | 32.1 | 2.9 | 0.40 | 1, 2, 4, 8, 16 |
| Lorenz-63, noise 0.3 std | ude_missing_xz | hybrid | 4614 | 4 | 484.0 | 55.3 | 0.22 | 1, 2, 4, 8 |
| Mackey-Glass, noise 0.01 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Mackey-Glass, noise 0.01 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.05 | none |
| Mackey-Glass, noise 0.01 std | ar | baseline | 21 | 1 | 0.0 | 0.7 | 0.47 | none |
| Mackey-Glass, noise 0.01 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.1 | -0.00 | none |
| Mackey-Glass, noise 0.01 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.15 | none |
| Mackey-Glass, noise 0.01 std | phase_climatology | baseline | 18 | 2 | 0.1 | 0.0 | 0.42 | none |
| Mackey-Glass, noise 0.01 std | arima | baseline | 5 | 1 | 0.0 | 6.3 | 0.31 | none |
| Mackey-Glass, noise 0.01 std | gbm | data | 78300 | 4 | 22.4 | 3.3 | 0.87 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.01 std | mlp | data | 1897 | 4 | 28.0 | 2.6 | 0.79 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.01 std | lstm | data | 17865 | 4 | 81.5 | 8.3 | 0.81 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.01 std | tcn | data | 29065 | 4 | 313.1 | 57.6 | 0.86 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.01 std | transformer | data | 18537 | 4 | 221.1 | 37.3 | 0.83 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.01 std | gbm_physics_features | data+physics | 78300 | 4 | 42.1 | 4.3 | 0.83 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.01 std | dde_fitted | physical | 2 | 1 | 0.0 | 0.9 | 0.85 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.01 std | dde_wrong_delay | physical | 2 | 1 | 0.0 | 0.7 | 0.27 | 4, 8 |
| Mackey-Glass, noise 0.01 std | dde_fitted+gbm | hybrid | 78302 | 4 | 106.1 | 10.4 | 0.91 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.01 std | dde_wrong_delay+gbm | hybrid | 78302 | 4 | 92.3 | 10.3 | 0.86 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.1 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Mackey-Glass, noise 0.1 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.03 | none |
| Mackey-Glass, noise 0.1 std | ar | baseline | 22 | 1 | 0.0 | 0.0 | 0.42 | none |
| Mackey-Glass, noise 0.1 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | -0.00 | none |
| Mackey-Glass, noise 0.1 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.15 | none |
| Mackey-Glass, noise 0.1 std | phase_climatology | baseline | 18 | 2 | 0.1 | 0.0 | 0.42 | none |
| Mackey-Glass, noise 0.1 std | arima | baseline | 5 | 1 | 0.0 | 9.3 | 0.38 | none |
| Mackey-Glass, noise 0.1 std | gbm | data | 78300 | 4 | 27.8 | 3.9 | 0.75 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.1 std | mlp | data | 1897 | 4 | 31.0 | 2.9 | 0.72 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.1 std | lstm | data | 17865 | 4 | 74.5 | 8.7 | 0.74 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.1 std | tcn | data | 29065 | 4 | 328.2 | 44.3 | 0.76 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.1 std | transformer | data | 18537 | 4 | 213.0 | 41.4 | 0.75 | 1, 2, 4, 8, 16, 32, 64 |
| Mackey-Glass, noise 0.1 std | gbm_physics_features | data+physics | 35100 | 4 | 42.6 | 4.0 | 0.72 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.1 std | dde_fitted | physical | 2 | 1 | 0.0 | 1.1 | 0.69 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.1 std | dde_wrong_delay | physical | 2 | 1 | 0.0 | 0.8 | 0.26 | 2, 4, 8 |
| Mackey-Glass, noise 0.1 std | dde_fitted+gbm | hybrid | 35102 | 4 | 108.0 | 9.9 | 0.75 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.1 std | dde_wrong_delay+gbm | hybrid | 35102 | 4 | 97.6 | 9.0 | 0.74 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mackey-Glass, noise 0.3 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Mackey-Glass, noise 0.3 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | 0.05 | none |
| Mackey-Glass, noise 0.3 std | ar | baseline | 21 | 1 | 0.0 | 0.4 | 0.38 | none |
| Mackey-Glass, noise 0.3 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.1 | 0.00 | none |
| Mackey-Glass, noise 0.3 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.17 | none |
| Mackey-Glass, noise 0.3 std | phase_climatology | baseline | 18 | 2 | 0.1 | 0.0 | 0.42 | none |
| Mackey-Glass, noise 0.3 std | arima | baseline | 5 | 1 | 0.0 | 4.9 | 0.31 | none |
| Mackey-Glass, noise 0.3 std | gbm | data | 78300 | 4 | 26.6 | 3.8 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | mlp | data | 1897 | 4 | 31.9 | 2.7 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | lstm | data | 4841 | 4 | 77.4 | 7.5 | 0.59 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | tcn | data | 16393 | 4 | 235.3 | 24.0 | 0.57 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | transformer | data | 18537 | 4 | 218.6 | 43.7 | 0.58 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | gbm_physics_features | data+physics | 35100 | 4 | 42.9 | 4.0 | 0.55 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | dde_fitted | physical | 2 | 1 | 0.0 | 1.0 | 0.44 | 1, 2, 4, 8, 16 |
| Mackey-Glass, noise 0.3 std | dde_wrong_delay | physical | 2 | 1 | 0.0 | 0.7 | 0.21 | 4 |
| Mackey-Glass, noise 0.3 std | dde_fitted+gbm | hybrid | 35102 | 4 | 102.4 | 9.5 | 0.56 | 1, 2, 4, 8, 16, 32 |
| Mackey-Glass, noise 0.3 std | dde_wrong_delay+gbm | hybrid | 35102 | 4 | 90.7 | 8.6 | 0.56 | 1, 2, 4, 8, 16, 32, 128 |
| Lorenz-96 (K=40), noise 0.1 std | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Lorenz-96 (K=40), noise 0.1 std | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.33 | none |
| Lorenz-96 (K=40), noise 0.1 std | ar | baseline | 11 | 1 | 0.0 | 1.0 | 0.28 | none |
| Lorenz-96 (K=40), noise 0.1 std | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.1 | -0.00 | none |
| Lorenz-96 (K=40), noise 0.1 std | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.77 | none |
| Lorenz-96 (K=40), noise 0.1 std | phase_climatology | baseline | 33 | 2 | 0.1 | 0.0 | -0.34 | none |
| Lorenz-96 (K=40), noise 0.1 std | arima | baseline | 5 | 1 | 0.0 | 4.0 | 0.26 | none |
| Lorenz-96 (K=40), noise 0.1 std | gbm | data | 20300 | 4 | 242.7 | 16.2 | 0.39 | 1, 2, 4, 8 |
| Lorenz-96 (K=40), noise 0.1 std | mlp | data | 38023 | 4 | 13.2 | 1.7 | 0.31 | 1, 2, 4 |
| Lorenz-96 (K=40), noise 0.1 std | lstm | data | 27719 | 4 | 20.4 | 1.7 | 0.21 | none |
| Lorenz-96 (K=40), noise 0.1 std | tcn | data | 10087 | 4 | 73.9 | 8.3 | 0.18 | none |
| Lorenz-96 (K=40), noise 0.1 std | transformer | data | 10407 | 4 | 49.8 | 4.1 | 0.30 | 2, 4 |
| Lorenz-96 (K=40), noise 0.1 std | gbm_physics_features | data+physics | 60900 | 4 | 120.1 | 14.2 | 0.45 | 1, 2, 4, 8 |
| Lorenz-96 (K=40), noise 0.1 std | mlp_physics_loss | data+physics | 744448 | 4 | 144.0 | 27.4 | 0.24 | 8 |
| Lorenz-96 (K=40), noise 0.1 std | ode_fitted | physical | 1 | 1 | 0.0 | 0.4 | 0.54 | 1, 2, 4, 8, 16 |
| Lorenz-96 (K=40), noise 0.1 std | sindy | physical | 257 | 4 | 278.9 | 11.3 | 0.53 | 1, 2, 4, 8, 16 |
| Lorenz-96 (K=40), noise 0.1 std | enkf | physical | 81 | 1 | 0.0 | 0.6 | 0.67 | 1, 2, 4, 8, 16, 32 |
| Lorenz-96 (K=40), noise 0.1 std | ode_fitted+gbm | hybrid | 9101 | 4 | 106.1 | 7.3 | 0.58 | 1, 2, 4, 8, 16 |
| Ornstein-Uhlenbeck | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Ornstein-Uhlenbeck | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.32 | none |
| Ornstein-Uhlenbeck | ar | baseline | 4 | 1 | 0.0 | 0.2 | 0.13 | none |
| Ornstein-Uhlenbeck | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| Ornstein-Uhlenbeck | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.57 | none |
| Ornstein-Uhlenbeck | phase_climatology | baseline | 820 | 2 | 0.7 | 0.6 | -0.33 | none |
| Ornstein-Uhlenbeck | arima | baseline | 4 | 1 | 0.0 | 5.3 | 0.13 | none |
| Ornstein-Uhlenbeck | gbm | data | 17400 | 4 | 20.1 | 0.9 | 0.12 | none |
| Ornstein-Uhlenbeck | mlp | data | 1798 | 4 | 13.4 | 1.4 | 0.12 | none |
| Ornstein-Uhlenbeck | lstm | data | 17670 | 4 | 46.1 | 3.6 | 0.12 | none |
| Ornstein-Uhlenbeck | tcn | data | 16294 | 4 | 105.5 | 6.3 | 0.11 | none |
| Ornstein-Uhlenbeck | transformer | data | 17926 | 4 | 81.9 | 7.7 | 0.12 | none |
| Ornstein-Uhlenbeck | gbm_physics_features | data+physics | 7800 | 4 | 17.6 | 1.2 | 0.12 | none |
| Ornstein-Uhlenbeck | ou_kalman | physical | 4 | 1 | 0.0 | 0.6 | 0.13 | none |
| Ornstein-Uhlenbeck | ou_kalman+gbm | hybrid | 7804 | 4 | 20.0 | 1.2 | 0.12 | none |
| Noise-driven damped oscillator | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Noise-driven damped oscillator | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.26 | none |
| Noise-driven damped oscillator | ar | baseline | 12 | 1 | 0.0 | 0.1 | 0.52 | none |
| Noise-driven damped oscillator | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | -0.00 | none |
| Noise-driven damped oscillator | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.15 | none |
| Noise-driven damped oscillator | phase_climatology | baseline | 27 | 2 | 0.1 | 0.0 | -0.09 | none |
| Noise-driven damped oscillator | arima | baseline | 5 | 1 | 0.0 | 9.2 | 0.50 | none |
| Noise-driven damped oscillator | gbm | data | 60900 | 4 | 32.4 | 3.0 | 0.51 | none |
| Noise-driven damped oscillator | mlp | data | 19591 | 4 | 21.3 | 1.5 | 0.51 | none |
| Noise-driven damped oscillator | lstm | data | 4775 | 4 | 50.2 | 3.6 | 0.51 | none |
| Noise-driven damped oscillator | tcn | data | 28999 | 4 | 173.7 | 29.8 | 0.50 | none |
| Noise-driven damped oscillator | transformer | data | 9927 | 4 | 110.4 | 15.8 | 0.51 | none |
| Noise-driven damped oscillator | gbm_physics_features | data+physics | 9100 | 4 | 50.2 | 2.8 | 0.52 | none |
| Noise-driven damped oscillator | oscillator_kalman | physical | 5 | 1 | 0.0 | 2.4 | 0.52 | none |
| Noise-driven damped oscillator | ou_kalman | physical | 4 | 1 | 0.0 | 1.1 | 0.16 | none |
| Noise-driven damped oscillator | oscillator_kalman+gbm | hybrid | 9105 | 4 | 51.5 | 2.5 | 0.52 | none |
| Noise-driven damped oscillator | ou_kalman+gbm | hybrid | 27304 | 4 | 27.7 | 2.7 | 0.51 | none |
| Double-well (regime switching) | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Double-well (regime switching) | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.74 | none |
| Double-well (regime switching) | ar | baseline | 6 | 1 | 0.0 | 0.2 | 0.11 | none |
| Double-well (regime switching) | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.01 | none |
| Double-well (regime switching) | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -1.17 | none |
| Double-well (regime switching) | phase_climatology | baseline | 490 | 2 | 3.1 | 0.8 | -0.81 | none |
| Double-well (regime switching) | arima | baseline | 5 | 1 | 0.0 | 4.7 | 0.10 | none |
| Double-well (regime switching) | gbm | data | 23200 | 4 | 23.9 | 1.1 | 0.12 | none |
| Double-well (regime switching) | mlp | data | 18696 | 4 | 14.5 | 1.9 | 0.12 | 1 |
| Double-well (regime switching) | lstm | data | 17800 | 4 | 57.6 | 5.6 | 0.11 | 2 |
| Double-well (regime switching) | tcn | data | 29032 | 4 | 127.4 | 17.3 | 0.10 | none |
| Double-well (regime switching) | transformer | data | 2936 | 4 | 79.4 | 8.5 | 0.12 | none |
| Double-well (regime switching) | gbm_physics_features | data+physics | 10400 | 4 | 18.0 | 0.5 | 0.12 | none |
| Double-well (regime switching) | sde_poly3 | physical | 5 | 1 | 0.0 | 0.0 | 0.12 | 2 |
| Double-well (regime switching) | sde_poly1 | physical | 3 | 1 | 0.0 | 0.0 | 0.10 | none |
| Double-well (regime switching) | ou_kalman | physical | 4 | 1 | 0.0 | 0.6 | 0.11 | none |
| Double-well (regime switching) | sde_poly1_nn_closure | hybrid | 324 | 4 | 201.4 | 2.9 | 0.12 | none |
| Double-well (regime switching) | sde_poly3+gbm | hybrid | 10405 | 4 | 599.3 | 56.0 | 0.12 | none |
| White noise | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| White noise | climatology | baseline | 2 | 1 | 0.0 | 0.0 | 0.29 | none |
| White noise | ar | baseline | 2 | 1 | 0.0 | 0.7 | 0.29 | none |
| White noise | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.29 | none |
| White noise | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.01 | none |
| White noise | phase_climatology | baseline | 46 | 2 | 0.0 | 0.0 | 0.29 | none |
| White noise | arima | baseline | 2 | 1 | 0.0 | 6.8 | 0.29 | none |
| White noise | gbm | data | 14500 | 4 | 20.0 | 0.9 | 0.29 | none |
| White noise | mlp | data | 18309 | 4 | 13.6 | 1.0 | 0.29 | none |
| White noise | lstm | data | 4709 | 4 | 59.3 | 3.3 | 0.29 | none |
| White noise | tcn | data | 28933 | 4 | 129.7 | 19.2 | 0.28 | none |
| White noise | transformer | data | 17893 | 4 | 91.5 | 8.9 | 0.29 | none |
| Random walk | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Random walk | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -13.47 | none |
| Random walk | ar | baseline | 3 | 1 | 0.0 | 0.1 | -0.00 | none |
| Random walk | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| Random walk | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -12.29 | none |
| Random walk | phase_climatology | baseline | 846 | 2 | 1.3 | 0.6 | -6.65 | none |
| Random walk | arima | baseline | 1 | 1 | 0.0 | 4.3 | -0.00 | none |
| Random walk | gbm | data | 14500 | 4 | 14.6 | 0.6 | -0.01 | none |
| Random walk | mlp | data | 1765 | 4 | 10.0 | 0.5 | -0.00 | none |
| Random walk | lstm | data | 4709 | 4 | 27.7 | 3.0 | -0.00 | none |
| Random walk | tcn | data | 16261 | 4 | 115.6 | 7.2 | -0.00 | none |
| Random walk | transformer | data | 17893 | 4 | 49.9 | 5.4 | -0.00 | none |
| USD/EUR daily log-returns | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| USD/EUR daily log-returns | climatology | baseline | 2 | 1 | 0.0 | 0.0 | 0.29 | none |
| USD/EUR daily log-returns | ar | baseline | 2 | 1 | 0.0 | 0.2 | 0.29 | none |
| USD/EUR daily log-returns | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.1 | 0.29 | none |
| USD/EUR daily log-returns | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| USD/EUR daily log-returns | phase_climatology | baseline | 5 | 2 | 0.0 | 0.0 | 0.26 | none |
| USD/EUR daily log-returns | arima | baseline | 2 | 1 | 0.0 | 3.7 | 0.29 | none |
| USD/EUR daily log-returns | gbm | data | 14500 | 4 | 17.9 | 1.0 | 0.29 | none |
| USD/EUR daily log-returns | mlp | data | 18309 | 4 | 16.1 | 1.4 | 0.29 | none |
| USD/EUR daily log-returns | lstm | data | 4709 | 4 | 56.0 | 4.8 | 0.29 | none |
| USD/EUR daily log-returns | tcn | data | 16261 | 4 | 140.0 | 9.4 | 0.28 | none |
| USD/EUR daily log-returns | transformer | data | 9861 | 4 | 98.5 | 7.0 | 0.29 | none |
| Sunspots (yearly) | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| Sunspots (yearly) | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -0.10 | none |
| Sunspots (yearly) | ar | baseline | 20 | 1 | 0.0 | 0.0 | 0.40 | none |
| Sunspots (yearly) | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | -0.00 | none |
| Sunspots (yearly) | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | 0.30 | none |
| Sunspots (yearly) | phase_climatology | baseline | 12 | 2 | 0.0 | 0.0 | 0.23 | none |
| Sunspots (yearly) | arima | baseline | 4 | 1 | 0.0 | 2.4 | 0.32 | none |
| Sunspots (yearly) | gbm | data | 7458 | 4 | 3.8 | 0.2 | 0.34 | none |
| Sunspots (yearly) | mlp | data | 18438 | 4 | 2.7 | 0.6 | 0.35 | none |
| Sunspots (yearly) | lstm | data | 17670 | 4 | 4.5 | 0.2 | 0.28 | none |
| Sunspots (yearly) | tcn | data | 4310 | 4 | 9.7 | 1.4 | 0.33 | none |
| Sunspots (yearly) | transformer | data | 17670 | 4 | 4.2 | 1.0 | 0.38 | none |
| Sunspots (yearly) | gbm_physics_features | data+physics | 7386 | 4 | 17.5 | 0.8 | 0.38 | 1 |
| Sunspots (yearly) | oscillator_kalman | physical | 5 | 1 | 0.0 | 1.1 | 0.21 | none |
| Sunspots (yearly) | oscillator_kalman+gbm | hybrid | 7529 | 4 | 13.6 | 1.2 | 0.39 | 1 |
| YouTube US trending: log10 total daily views | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| YouTube US trending: log10 total daily views | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -7.88 | none |
| YouTube US trending: log10 total daily views | ar | baseline | 14 | 1 | 0.0 | 0.0 | -1.67 | none |
| YouTube US trending: log10 total daily views | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| YouTube US trending: log10 total daily views | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -0.58 | none |
| YouTube US trending: log10 total daily views | phase_climatology | baseline | 19 | 2 | 0.0 | 0.0 | -0.32 | none |
| YouTube US trending: log10 total daily views | arima | baseline | 4 | 1 | 0.0 | 6.8 | -2.40 | none |
| YouTube US trending: log10 total daily views | gbm | data | 3848 | 4 | 4.7 | 0.3 | -0.81 | none |
| YouTube US trending: log10 total daily views | mlp | data | 1701 | 4 | 14.1 | 0.0 | -0.18 | none |
| YouTube US trending: log10 total daily views | lstm | data | 17605 | 4 | 1.5 | 0.1 | -0.07 | none |
| YouTube US trending: log10 total daily views | tcn | data | 16261 | 4 | 4.3 | 0.2 | -0.79 | none |
| YouTube US trending: log10 total daily views | transformer | data | 17637 | 4 | 2.1 | 0.1 | -0.10 | none |
| YouTube US trending: Entertainment slots per day | persistence | baseline | 0 | 1 | 0.0 | 0.0 | 0.00 | none |
| YouTube US trending: Entertainment slots per day | climatology | baseline | 2 | 1 | 0.0 | 0.0 | -1.40 | none |
| YouTube US trending: Entertainment slots per day | ar | baseline | 3 | 1 | 0.0 | 0.0 | -0.06 | none |
| YouTube US trending: Entertainment slots per day | exp_smoothing | baseline | 1 | 1 | 0.0 | 0.0 | 0.00 | none |
| YouTube US trending: Entertainment slots per day | seasonal_naive | baseline | 1 | 1 | 0.0 | 0.0 | -1.84 | none |
| YouTube US trending: Entertainment slots per day | phase_climatology | baseline | 24 | 2 | 0.0 | 0.0 | -1.01 | none |
| YouTube US trending: Entertainment slots per day | arima | baseline | 3 | 1 | 0.0 | 6.4 | -0.01 | none |
| YouTube US trending: Entertainment slots per day | gbm | data | 3670 | 4 | 5.4 | 0.2 | -0.18 | none |
| YouTube US trending: Entertainment slots per day | mlp | data | 18181 | 4 | 14.0 | 0.4 | -0.14 | none |
| YouTube US trending: Entertainment slots per day | lstm | data | 17605 | 4 | 1.1 | 0.7 | -0.12 | none |
| YouTube US trending: Entertainment slots per day | tcn | data | 28933 | 4 | 2.9 | 1.0 | -0.17 | none |
| YouTube US trending: Entertainment slots per day | transformer | data | 2597 | 4 | 2.4 | 0.5 | -0.07 | none |

## surrogate_trained

| series | GBM trained on real data | on IAAFT surrogate | on shuffled surrogate | real − IAAFT |
|---|---|---|---|---|
| Lorenz-63, noise 0.01 std | 0.67 | 0.24 | -0.34 | 0.42 |
| Lorenz-63, noise 0.1 std | 0.51 | 0.20 | -0.30 | 0.31 |
| Lorenz-63, noise 0.3 std | 0.33 | 0.15 | -0.09 | 0.17 |
| Mackey-Glass, noise 0.01 std | 0.85 | 0.51 | -0.07 | 0.34 |
| Mackey-Glass, noise 0.1 std | 0.73 | 0.49 | -0.06 | 0.25 |
| Mackey-Glass, noise 0.3 std | 0.56 | 0.43 | 0.03 | 0.12 |
| Lorenz-96 (K=40), noise 0.1 std | 0.27 | 0.24 | -0.33 | 0.03 |
| Ornstein-Uhlenbeck | 0.11 | 0.11 | -0.31 | 0.00 |
| Noise-driven damped oscillator | 0.51 | 0.50 | -0.29 | 0.01 |
| Double-well (regime switching) | 0.11 | 0.07 | -0.83 | 0.04 |
| White noise | 0.28 | 0.28 | 0.28 | -0.00 |
| Random walk | -0.03 | -0.04 | -14.95 | 0.01 |
| USD/EUR daily log-returns | 0.28 | 0.28 | 0.28 | 0.01 |
| Sunspots (yearly) | 0.39 | 0.34 | 0.02 | 0.05 |
| YouTube US trending: log10 total daily views | -1.17 | -1.45 | -4.95 | 0.28 |
| YouTube US trending: Entertainment slots per day | -0.06 | -0.18 | -1.14 | 0.12 |

## overfitting

| series | variant | params | train | val | test | test − train | CV: shuffled k-fold | CV: walk-forward, no purge | CV: purged walk-forward | CV: purged k-fold + embargo |
|---|---|---|---|---|---|---|---|---|---|---|
| Double-well (regime switching) | overfit: MLP 512x3, 32 lags, no early stop | 546312 | 0.587 | 1.132 | 1.159 | 0.572 | 0.990 | 1.207 | 1.205 | 1.239 |
| Double-well (regime switching) | MLP 512x3 + early stopping | 546312 | 0.872 | 0.837 | 0.908 | 0.036 |  |  |  |  |
| Double-well (regime switching) | MLP 512x3 + early stop + wd 1e-2 + dropout 0.2 | 546312 | 0.912 | 0.843 | 0.909 | -0.003 |  |  |  |  |
| Double-well (regime switching) | MLP 16x1 + early stopping | 408 | 0.896 | 0.850 | 0.893 | -0.003 |  |  |  |  |
| Double-well (regime switching) | MLP 64x2 + early stopping | 5768 | 0.894 | 0.843 | 0.894 | 0.000 | 0.878 | 0.897 | 0.896 | 0.911 |
| Double-well (regime switching) | MLP 64x2 + early stopping, 5-member ensemble | 28840 | 0.889 | 0.842 | 0.889 | 0.000 |  |  |  |  |
| Double-well (regime switching) | overfit: GBM 255 leaves, 1000 rounds, leaf>=1 | 4072000 | 0.000 | 0.929 | 0.933 | 0.933 | 0.932 | 0.964 | 0.966 | 0.988 |
| Double-well (regime switching) | GBM 7 leaves, 100 rounds, leaf>=50, l2=1 | 10400 | 0.853 | 0.862 | 0.879 | 0.026 | 0.882 | 0.900 | 0.896 | 0.913 |
| Lorenz-63, noise 0.1 std | overfit: MLP 512x3, 32 lags, no early stop | 579080 | 0.265 | 0.429 | 0.476 | 0.212 | 0.438 | 0.467 | 0.468 | 0.511 |
| Lorenz-63, noise 0.1 std | MLP 512x3 + early stopping | 579080 | 0.369 | 0.383 | 0.423 | 0.054 |  |  |  |  |
| Lorenz-63, noise 0.1 std | MLP 512x3 + early stop + wd 1e-2 + dropout 0.2 | 579080 | 0.548 | 0.533 | 0.554 | 0.006 |  |  |  |  |
| Lorenz-63, noise 0.1 std | MLP 16x1 + early stopping | 920 | 0.498 | 0.478 | 0.504 | 0.005 |  |  |  |  |
| Lorenz-63, noise 0.1 std | MLP 64x2 + early stopping | 7816 | 0.410 | 0.395 | 0.426 | 0.017 | 0.401 | 0.409 | 0.414 | 0.432 |
| Lorenz-63, noise 0.1 std | MLP 64x2 + early stopping, 5-member ensemble | 39080 | 0.393 | 0.385 | 0.408 | 0.016 |  |  |  |  |
| Lorenz-63, noise 0.1 std | overfit: GBM 255 leaves, 1000 rounds, leaf>=1 | 4072000 | 0.000 | 0.439 | 0.438 | 0.438 | 0.457 | 0.450 | 0.450 | 0.469 |
| Lorenz-63, noise 0.1 std | GBM 7 leaves, 100 rounds, leaf>=50, l2=1 | 10400 | 0.434 | 0.461 | 0.465 | 0.031 | 0.465 | 0.467 | 0.469 | 0.472 |
| Lorenz-63, noise 0.1 std | MLP 128x2, 4 lags + early stop (no physics) | 67712 | 0.442 | 0.427 | 0.446 | 0.004 |  |  |  |  |
| Lorenz-63, noise 0.1 std | MLP 128x2, 4 lags + early stop + physics loss 0.1 | 67712 | 0.444 | 0.428 | 0.447 | 0.003 |  |  |  |  |
| USD/EUR daily log-returns | overfit: MLP 512x3, 32 lags, no early stop | 544773 | 0.516 | 0.965 | 0.919 | 0.403 | 0.862 | 0.919 | 0.921 | 0.870 |
| USD/EUR daily log-returns | MLP 512x3 + early stopping | 544773 | 0.689 | 0.737 | 0.723 | 0.034 |  |  |  |  |
| USD/EUR daily log-returns | MLP 512x3 + early stop + wd 1e-2 + dropout 0.2 | 544773 | 0.694 | 0.720 | 0.708 | 0.014 |  |  |  |  |
| USD/EUR daily log-returns | MLP 16x1 + early stopping | 357 | 0.694 | 0.725 | 0.711 | 0.017 |  |  |  |  |
| USD/EUR daily log-returns | MLP 64x2 + early stopping | 5573 | 0.695 | 0.724 | 0.710 | 0.015 | 0.714 | 0.714 | 0.715 | 0.713 |
| USD/EUR daily log-returns | MLP 64x2 + early stopping, 5-member ensemble | 27865 | 0.691 | 0.721 | 0.708 | 0.017 |  |  |  |  |
| USD/EUR daily log-returns | overfit: GBM 255 leaves, 1000 rounds, leaf>=1 | 2545000 | 0.000 | 0.796 | 0.756 | 0.756 | 0.779 | 0.776 | 0.776 | 0.773 |
| USD/EUR daily log-returns | GBM 7 leaves, 100 rounds, leaf>=50, l2=1 | 6500 | 0.684 | 0.728 | 0.713 | 0.028 | 0.720 | 0.717 | 0.718 | 0.721 |
