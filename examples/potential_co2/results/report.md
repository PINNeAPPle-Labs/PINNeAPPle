# CO2 near the critical point: potential-based against direct surrogates

Data: CoolProp HEOS (Span & Wagner 1996) for CO2. Training region T 305-335 K, rho 100-800 kg/m3; extrapolation: T 335-365 K (hotter) and rho 800-950 kg/m3 (denser). Each training sample carries one property. Same data, loss, optimiser, steps, width (32) and depth (3) for every model; 5 seeds; intervals are 95 % Student-t over seeds. Error = RMS error / standard deviation of the property on the held-out grid of the training region (21 x 21 points per region).

## all six properties, 3000 steps

200 samples per training property (p, s, u, cv, cp, w), 3000 Adam steps.

| model | parameters | training s / seed | error, training region | error, hotter | error, denser | stable share, hotter |
|---|---|---|---|---|---|---|
| direct multi-output | 2406 | 21 | 0.0995 [0.0722, 0.127] | 0.249 [0.196, 0.302] | 0.661 [0.615, 0.706] | 1 [1, 1] |
| potential (MLP) | 2244 | 85 | 1.61 [1.6, 1.62] | 1.97 [1.91, 2.02] | 3.2 [3.19, 3.21] | 0.327 [0.288, 0.366] |
| potential (ICNN) | 6597 | 213 | 1.24 [0.792, 1.69] | 3.24 [1.43, 5.05] | 2.69 [2.27, 3.1] | 0.698 [0.185, 1.21] |

Per property, training region (mean [95 % interval]):

| model | p | s | u | cv | cp | w |
|---|---|---|---|---|---|---|
| direct multi-output | 0.038 [0.026, 0.05] | 0.024 [0.022, 0.027] | 0.014 [0.011, 0.016] | 0.11 [0.08, 0.14] | 0.39 [0.24, 0.54] | 0.024 [0.021, 0.027] |
| potential (MLP) | 2.8 [2.8, 2.8] | 0.99 [0.98, 1] | 0.97 [0.97, 0.97] | 0.29 [0.22, 0.36] | 1.1 [1.1, 1.1] | 3.5 [3.5, 3.6] |
| potential (ICNN) | 0.82 [0.5, 1.1] | 0.9 [0.024, 1.8] | 1.2 [0.038, 2.4] | 1.1 [0.51, 1.6] | 1 [0.77, 1.3] | 2.4 [1.6, 3.2] |

Paired against the direct surrogate (log10 error ratio, mean of the six properties; < 0: the potential is more accurate):

| model | interp | hotter | denser |
|---|---|---|---|
| potential (MLP) | 1.22 [1.09, 1.35]: direct more accurate | 0.903 [0.812, 0.993]: direct more accurate | 0.685 [0.654, 0.717]: direct more accurate |
| potential (ICNN) | 1.09 [0.873, 1.31]: direct more accurate | 1.09 [0.758, 1.41]: direct more accurate | 0.607 [0.542, 0.672]: direct more accurate |

Thermodynamic identities, held-out grid of the training region (relative violation, 95th percentile, mean over seeds; finite differences of the predictions, step 1e-4):

| identity | direct multi-output | potential (MLP) | potential (ICNN) |
|---|---|---|---|
| Maxwell relation (Helmholtz) | 0.41 | 2.8e-05 | 1.1e-06 |
| c_v from the entropy | 0.57 | 1.5e-06 | 6.7e-06 |
| energy equation | 0.46 | 2.3e-05 | 1e-06 |
| density dependence of c_v | 11 | 3.1e-06 | 2.1e-05 |
| Mayer relation | 2.4 | 6.4e-05 | 4.9e-06 |
| speed of sound | 0.88 | 5.4 | 9.9e-06 |

## no c_p in training, 3000 steps

200 samples per training property (p, s, u, cv, w), 3000 Adam steps.

| model | parameters | training s / seed | error, training region | error, hotter | error, denser | stable share, hotter |
|---|---|---|---|---|---|---|
| direct multi-output | 2406 | 20 | 0.221 [0.214, 0.228] | 0.241 [0.21, 0.273] | 0.663 [0.609, 0.717] | 1 [1, 1] |
| potential (MLP) | 2244 | 73 | 1.57 [-0.886, 4.02] | 2.59 [2.02, 3.16] | 0.889 [0.651, 1.13] | 0.688 [0.558, 0.818] |
| potential (ICNN) | 6597 | 182 | 3.49 [0.716, 6.27] | 5.59 [3.23, 7.94] | 2.37 [2.11, 2.63] | 0.671 [0.42, 0.922] |

Per property, training region (mean [95 % interval]):

| model | p | s | u | cv | cp | w |
|---|---|---|---|---|---|---|
| direct multi-output | 0.035 [0.024, 0.046] | 0.023 [0.021, 0.024] | 0.012 [0.0077, 0.017] | 0.14 [0.094, 0.18] | 1.1 [1.1, 1.1] | 0.02 [0.016, 0.023] |
| potential (MLP) | 0.23 [0.14, 0.32] | 0.11 [0.066, 0.16] | 0.082 [0.054, 0.11] | 0.36 [0.22, 0.5] | 8.4 [-6.4, 23] | 0.21 [0.099, 0.31] |
| potential (ICNN) | 0.71 [0.47, 0.95] | 0.31 [0.18, 0.45] | 0.36 [0.22, 0.5] | 0.82 [0.5, 1.1] | 17 [0.88, 32] | 2.1 [1.5, 2.8] |

Paired against the direct surrogate (log10 error ratio, mean of the six properties; < 0: the potential is more accurate):

| model | interp | hotter | denser |
|---|---|---|---|
| potential (MLP) | 0.654 [0.141, 1.17]: direct more accurate | 1.03 [0.962, 1.09]: direct more accurate | 0.119 [0.0254, 0.213]: direct more accurate |
| potential (ICNN) | 1.09 [0.638, 1.55]: direct more accurate | 1.35 [1.13, 1.56]: direct more accurate | 0.551 [0.478, 0.625]: direct more accurate |

Thermodynamic identities, held-out grid of the training region (relative violation, 95th percentile, mean over seeds; finite differences of the predictions, step 1e-4):

| identity | direct multi-output | potential (MLP) | potential (ICNN) |
|---|---|---|---|
| Maxwell relation (Helmholtz) | 0.41 | 3.7e-07 | 5.4e-07 |
| c_v from the entropy | 0.52 | 2.4e-06 | 6.5e-06 |
| energy equation | 0.41 | 4.5e-07 | 7.1e-07 |
| density dependence of c_v | 9.1 | 5.9e-06 | 9e-06 |
| Mayer relation | 9.2 | 7e-06 | 1.9e-05 |
| speed of sound | 0.83 | 5.1e-07 | 1.7e-06 |

## all six properties, 9000 steps

200 samples per training property (p, s, u, cv, cp, w), 9000 Adam steps.

| model | parameters | training s / seed | error, training region | error, hotter | error, denser | stable share, hotter |
|---|---|---|---|---|---|---|
| direct multi-output | 2406 | 57 | 0.0378 [-0.0374, 0.113] | 0.22 [-0.174, 0.614] | 0.581 [0.356, 0.807] | 1 [1, 1] |
| potential (MLP) | 2244 | 250 | 1.51 [-1.08, 4.1] | 2.6 [-4.34, 9.55] | 3.05 [0.995, 5.1] | 0.342 [-0.551, 1.24] |
| potential (ICNN) | 6597 | 636 | 1.3 [-7.03, 9.62] | 4.83 [-24.8, 34.5] | 2.62 [0.0347, 5.2] | 0.509 [-2.88, 3.89] |

Per property, training region (mean [95 % interval]):

| model | p | s | u | cv | cp | w |
|---|---|---|---|---|---|---|
| direct multi-output | 0.024 [-0.017, 0.065] | 0.011 [-0.018, 0.04] | 0.0066 [-0.016, 0.029] | 0.032 [-0.06, 0.12] | 0.14 [-0.28, 0.56] | 0.012 [-0.019, 0.042] |
| potential (MLP) | 1.8 [-8.7, 12] | 0.85 [-0.67, 2.4] | 0.99 [0.89, 1.1] | 0.91 [-1.8, 3.6] | 1.1 [1, 1.1] | 3.4 [2.4, 4.4] |
| potential (ICNN) | 0.51 [-4.7, 5.8] | 0.56 [-6.1, 7.2] | 0.75 [-7.9, 9.4] | 3.6 [-13, 21] | 0.97 [0.18, 1.8] | 1.4 [-12, 15] |

Paired against the direct surrogate (log10 error ratio, mean of the six properties; < 0: the potential is more accurate):

| model | interp | hotter | denser |
|---|---|---|---|
| potential (MLP) | 1.6 [-0.0199, 3.23]: no significant difference | 1.07 [0.676, 1.46]: direct more accurate | 0.719 [0.595, 0.844]: direct more accurate |
| potential (ICNN) | 1.48 [-0.721, 3.68]: no significant difference | 1.29 [-2.4, 4.98]: no significant difference | 0.653 [0.0548, 1.25]: direct more accurate |

Thermodynamic identities, held-out grid of the training region (relative violation, 95th percentile, mean over seeds; finite differences of the predictions, step 1e-4):

| identity | direct multi-output | potential (MLP) | potential (ICNN) |
|---|---|---|---|
| Maxwell relation (Helmholtz) | 0.27 | 2e-05 | 2.3e-06 |
| c_v from the entropy | 0.38 | 7.2e-06 | 1.6e-05 |
| energy equation | 0.27 | 1.8e-05 | 2e-06 |
| density dependence of c_v | 14 | 5.9e-06 | 4.7e-05 |
| Mayer relation | 1.5 | 0.00012 | 4.8e-06 |
| speed of sound | 0.48 | 10 | 1.7e-05 |

## Not measured

- Two-phase states and the subcritical region near the saturation line (the data are supercritical).
- Other fluids, noisy data, other widths or depths, other optimisers.
- Wall-clock at equal time: the budget is equal in steps; training seconds are reported (single thread, three studies running at once on four cores).
