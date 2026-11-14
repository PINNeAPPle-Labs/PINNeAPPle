# Native backend vs the NumPy reference

Machine: macOS-26.5.2-arm64-arm-64bit · arm · Python 3.11.16 · NumPy 2.4.6

Library: `/Users/yanbarros/Library/Caches/pinneapple/native/347ab857064d8dcf/libpinneapple_native.dylib`
Build: `cc -O3 -fPIC -shared -std=c11 -fno-fast-math -Wall -Wextra (darwin/arm64)`

| grid | operation | C best (ms) | NumPy best (ms) | speedup | C spread | NumPy spread | max abs difference |
|---|---|---|---|---|---|---|---|
| 129x129 | `laplacian_2d` | 0.035 | 0.105 | 3.0× | 1.28× | 1.65× | 0.0e+00 |
| 129x129 | `heat_step_2d` | 0.979 | 8.169 | 8.3× | 9.89× | 12.54× | 3.3e-16 |
| 129x129 | `jacobi_2d` | 4.854 | 160.539 | 33.1× | 19.07× | 3.02× | 6.2e-17 |
| 513x513 | `laplacian_2d` | 0.198 | 1.453 | 7.3× | 8.74× | 30.89× | 0.0e+00 |
| 513x513 | `heat_step_2d` | 14.841 | 597.507 | 40.3× | 13.38× | 2.30× | 4.4e-16 |
| 513x513 | `jacobi_2d` | 402.921 | 3057.864 | 7.6× | 1.73× | 1.38× | 3.9e-18 |

Each cell is the best of 7 runs on the kernels as written — not a microbenchmark of the loop: the heat and Jacobi rows carry 50 and 200 steps respectively, and the speedup compares those two bests. The spread columns are (slowest run / fastest run) for each backend, and they are large on a loaded machine -- for both sides, not only NumPy's -- so a single speedup number here would claim a precision the measurement does not have. Quote the C column and the order of magnitude of the speedup; rerun on an idle machine for anything tighter. The difference column is the two backends' agreement, so a fast row that disagreed would be visible rather than reported as a win.

