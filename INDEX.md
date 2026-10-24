# PINNeAPPle app series, days 13-30: tests, videos and LinkedIn posts

Each app was tested (its pytest files plus a real end-to-end run in the UI, with the downloads checked over HTTP), then recorded
at 1440x900 with English captions. Every number in a caption or post was read from that run. Files:
`videos/<day>_<app>.mp4`, `posts/<day>_<app>.md`, `frames/<day>_<app>.jpg` (one keyframe).

| Day | App | Tests | Video | Result shown |
|---|---|---|---|---|
| 13 | simulation_campaign | 41 passed / 0 failed / 1 skipped (no ccx); bug in "Plan the campaign" found and fixed in PR #873 (+2 tests) | 86.2 s | 24 infill cases on 436 OpenFOAM runs: fill distance 0.730→0.688, discrepancy 0.011→0.009; 2 near-duplicate pairs and 18 out-of-bounds runs flagged; 4/4 checks pass |
| 14 | surrogate_studio | 32 passed / 0 failed | 79.7 s | MLP chosen of 4 methods at equal budget: held-out 2.58 %, unseen airfoils 2.96 % (ridge 8.29 %, GBT 10.4 %), 81.6 µs/query; trust APPROVED (shift/held-out 1.15 ≤ 3); alpha 21° flagged OOD (range −2.0..13.4°) |
| 15 | uncertainty_lab | 16 passed / 0 failed | 86.8 s | interval coverage 91.0 % on 900 held-out designs (95 % CI 89.0–92.7 %); 5 of 11 designs routed to the high-fidelity solver; the only interval miss had been flagged and routed; P(R ≤ 0.4 K/W) 66.6 % → 54.5 % with model error |
| 16 | sensitivity_lab | 20 passed / 0 failed | 87.6 s | n_fins controls thermal resistance: total Sobol index 0.54 (95 % CI 0.50–0.59); t_ambient_c and power_w can be frozen (upper bound 1.5e-4); re-run check variance ratio 1.000; Ishigami reference max error 0.0027, 6/6 exact indices inside the CI |
| 17 | engineering_optimizer | 3 passed / 0 failed | 88.9 s | NSGA-II, 600 surrogate calls: 371 feasible, 6 on the Pareto front; best mass 82.4 g vs 486 g baseline (−83 %); 5/5 candidates confirmed by the 3-D FVM engine, largest fast-vs-FVM gap 1.45 %; active 85 °C limit at 84.42 °C (FVM 84.39); 9/9 checks |
