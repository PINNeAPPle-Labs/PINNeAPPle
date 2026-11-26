# Native backend vs the NumPy reference

Machine: macOS-26.5.2-arm64-arm-64bit · arm · Python 3.11.16 · NumPy 2.4.6

Library: `/Users/yanbarros/Library/Caches/pinneapple/native/347ab857064d8dcf/libpinneapple_native.dylib`
Build: `cc -O3 -fPIC -shared -std=c11 -fno-fast-math -Wall -Wextra (darwin/arm64)`

| grid | operation | C median (ms) | NumPy median (ms) | speedup (median of paired ratios) | ratio range | max abs difference |
|---|---|---|---|---|---|---|
| 129x129 | `laplacian_2d` | 0.034 | 0.104 | 3.1× | 2.5–3.5× | 0.0e+00 |
| 129x129 | `heat_step_2d` | 1.224 | 56.257 | 15.7× | 1.4–99.3× | 3.3e-16 |
| 129x129 | `jacobi_2d` | 12.500 | 189.303 | 14.8× | 0.7–54.5× | 6.2e-17 |
| 513x513 | `laplacian_2d` | 0.267 | 1.447 | 5.8× | 0.2–48.0× | 0.0e+00 |
| 513x513 | `heat_step_2d` | 101.733 | 724.641 | 7.6× | 4.6–20.3× | 4.4e-16 |
| 513x513 | `jacobi_2d` | 404.707 | 2242.726 | 5.6× | 4.0–8.2× | 3.9e-18 |

11 pairs per row, the two backends timed back to back and the order alternated, on the kernels as written — the heat and Jacobi rows carry 50 and 200 steps respectively. The speedup is the median of the per-pair ratios and the range is the smallest and largest of those same ratios: on a host that is also running other work the range is wide for both backends, and a paired ratio is the only comparison that survives it. The difference column is the two backends' agreement, so a fast row that disagreed would be visible rather than reported as a win.

