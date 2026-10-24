A simulation library should be able to say which parts of science it covers, and how far it can be trusted in each. We started that map in PINNeAPPle, beginning with electromagnetism.

What is in PINNeAPPle now (pp.science, PRs #882 to #884):
- an atlas of 20 areas and 77 subareas. 13 areas are physics and 7 are other sciences (chemistry, materials, biology, neuroscience, ecology and epidemiology, Earth and climate, applied mathematics). Each subarea is marked available, partial or planned, and points to the library code that already covers it. A test fails if any pointer names code that does not exist;
- one way to describe an experiment: area; level (academic or research); kind (theory, or practice on a virtual bench with noisy instruments); dimension (0-D, 1-D, 2-D, 3-D, or 3-D plus time);
- electromagnetism first: 20 experiments, from Coulomb and Gauss to a magnet falling through a loop, waveguides, cavities and Mie scattering. Each result comes with a trust report and a Lab seal.

Measured, from the committed results:
- 89 comparisons against cited references; all 89 are within tolerance.
- Faraday's law in 3-D plus time: peak EMF 5.25 mV against 5.27 mV from Maxwell's coaxial-loop formula (0.40 %). The EMF changes sign at 127.73 ms; free fall predicts 127.73 ms.
- Capacitor edge effect: 0.39 % from Palmer's formula for strips, 0.82 % from Kirchhoff's for discs.
- A virtual bench measures mu0 = (1.2517 ± 0.0129) × 10⁻⁶ N/A² (k = 2). The CODATA value lies inside that interval. When the whole bench was repeated 300 times, the interval held the true value 96.7 % of the time.
- 3-D FDTD of a dipole: power error 1.50 %, 0.68 % and 0.33 % at 10, 14 and 20 cells per wavelength, an observed order of 2.65.

Limits are stated, not hidden. With a dielectric sphere next to the dipole there is no closed form, so the study reports first-order convergence (1.05) and a 2.8 % grid-convergence index instead of a pass against a reference. The surrogate's ±2σ band covers 80 % of held-out runs, exactly at the threshold it declares.

Guide and code: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/science.md

#Physics #Electromagnetics #ComputationalPhysics #ScientificComputing #FDTD #Verification #UncertaintyQuantification #OpenSource

---
## Media

1. `media/science-by-area/01_atlas.png`: the atlas: subareas per area, available / partial / planned (from `pinneapple_science/data/atlas.json`).
2. `media/science-by-area/02_faraday_3d_t.png`: a magnet falling through a loop. One frame of |B| in 3-D plus time, and the flux and EMF against Maxwell's formula.
3. `media/science-by-area/03_capacitor_edge.png`: fringing of 2-D strips against Palmer, and 3-D discs against Kirchhoff, with Kirchhoff's error shrinking as a/d grows.
4. `media/science-by-area/04_bench_mu0.png`: the mu0 bench (fit, uncertainty budget, coverage over 300 benches) and the RC-discharge bench.
5. `media/science-by-area/05_fdtd_3d_t.png`: FDTD in 3-D plus time: convergence, far field, and E_z·r on three planes.
6. `media/science-by-area/06_research.png`: permittivity recovered from a noisy reflection spectrum, and the surrogate's hold-out error.
7. `media/science-by-area/science_em_3d_t.mp4` (15 s, 1080x1350): title card; the falling magnet (|B| and the EMF in time); the FDTD field of a dipole beside a dielectric sphere on three planes; end card.

`media/science-by-area/make_media.py` rebuilds every image and the video from the committed files.

## Sources

All paths are in PINNeAPPle `main` at b5387e86, after PRs #882, #883 and #884 were merged. E/ = `examples/science/electromagnetism/`.

- 20 areas, 77 subareas, 13 physics + 7 other sciences: `pinneapple_science/data/atlas.json` (`areas`: 20 entries, 13 with `branch` = "physics"; 77 `subareas`).
- A test fails on a bad pointer: `tests/test_science.py::test_atlas_points_only_to_code_that_exists` and `::test_atlas_validation_catches_a_bad_pointer` (`pp.science.validate()` returns []).
- 20 experiments: `pinneapple_science.names()` (20 names, all `em.*`); folders under E/.
- 89 comparisons, 89 within tolerance: every `comparisons[*].passed` in E/*/*/result.json (89 entries, all true); catalogue `examples/science/README.md`.
- 5.25 mV vs 5.27 mV, 0.40 %: E/faraday_induction/3D_plus_t/result.json, comparison "peak EMF": `computed` = 0.0052490, `reference` = 0.0052699, `rel_error` = 0.00398.
- 127.73 ms vs 127.73 ms: same file, "time of the EMF sign change": `computed` = 0.127727, `reference` = 0.127732.
- 0.39 % from Palmer: E/parallel_plate_capacitor/2D/result.json "C per unit length (extrapolated)" `rel_error` = 0.003928.
- 0.82 % from Kirchhoff: E/parallel_plate_capacitor/3D/result.json "C, axisymmetric (extrapolated)" `rel_error` = 0.008167.
- mu0 = (1.2517 ± 0.0129) × 10⁻⁶, CODATA inside the interval: E/bench_mu0_solenoid/3D/result.json `values.mu0_measured` = 1.251689e-06, `values.expanded_uncertainty_k2` = 1.2948e-08, comparison `passed` = true (|mu0 − 1.25663706127e-6| ≤ U).
- 96.7 % of 300 benches: same file, `values.coverage_k2` = 0.96667; `params.monte_carlo` = 300.
- 1.50 %, 0.68 %, 0.33 %, order 2.65: E/research_fdtd_3d/3D_plus_t/result.json `values.power_error_10cpw` = 0.015002, `power_error_14cpw` = 0.006783, `power_error_20cpw` = 0.003288, `observed_order_vacuum` = 2.647.
- Sphere case, order 1.05 and GCI 2.8 %: same file, `values.observed_order_configured` = 1.050, `values.gci_fine_configured` = 0.02842.
- Surrogate ±2σ covers 80 %, threshold 80 %: E/research_surrogate_hall/2D/result.json `values.coverage_2sigma` = 0.8; check "+-2 sigma coverage of the hold-out at least 80 %".
- Carousel only: 11 / 53 / 13 subareas (atlas.json statuses); fringing +16.8 % (E/parallel_plate_capacitor/2D/result.json `values.fringe_excess` = 0.16820); Kirchhoff +6.2 %, +2.3 %, +0.8 % (E/parallel_plate_capacitor/3D/result.json `data.scaling.deviation_from_kirchhoff`); eps_r = 3.6998 ± 0.0017 (E/research_inverse_permittivity/1D/result.json `values.eps_r_recovered`); 2u coverage 97 %, 90 % before the bias correction (same file, `values.coverage_2u_eps` = 0.97, `values.coverage_2u_eps_uncorrected` = 0.9, over `params.repeats` = 100); Monte Carlo vs GUM std 2.5 % (E/research_uq_waveguide/2D/result.json, comparison "TE10: Monte Carlo std vs first-order GUM" `rel_error` = 0.0252); hold-out worst 0.079 % (E/research_surrogate_hall/2D/result.json `values.holdout_max_rel_error` = 0.00079); dipole sign change and EMF values as above.
