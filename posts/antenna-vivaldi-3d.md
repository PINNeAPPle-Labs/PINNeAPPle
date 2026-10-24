A Vivaldi radiates end-fire, along the board. So its gain is bounded by its length, not its area: 10 dBi at 2 GHz needs L ≥ 206 mm (Hansen–Woodyard), before any simulation.

We built an open toolkit around that fact in PINNeAPPle (pinneapple_simulation.electromagnetics.antenna): an open antipodal Vivaldi model for 2–4 GHz, solved in EMerge (open-source FEM), with end-fire limits, physics checks, surrogates with measured errors, NSGA-II and active learning. Then we put the antenna in 3-D from the solved fields: gain balloon, near-field cuts, a one-file HTML viewer and Blender Cycles renders.

Measured in the committed run:
- 125 EMerge processes, all ok. Peak directivity of the best design rises from 7.09 to 11.27 dBi across 2 → 4 GHz, and the balloon peak matches the table to 0.0 dB.
- Coarse vs fine mesh: 0.62 dB RMSE on realized gain, and a per-frequency bias correction does not reduce it.
- The best design reaches 5.37 dBi realized gain at 2 GHz. The 10 dBi goal at 2 GHz is not met, and the guide says so.

The checks report what fails. Over 80 coarse rows, 525 checks support and 35 contradict, and all 35 are power_closure. The pattern power and the Poynting flux differ by 7.2 % (median), up to 34.6 %, so those rows go to REVIEW and gains carry about 0.5 dB of uncertainty. The best design's trust card is also REVIEW, because the GP's leave-one-design-out error (1.83 dB) is above the 1.5 dB limit.

The model is synthetic and open, and one command re-runs everything.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/antenna_design.md
Example: https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/examples/use_cases/antennas

#Antennas #RFEngineering #ComputationalElectromagnetics #FEM #SimulationVerification #Python #OpenSource

---
## Media
1. `media/antenna-vivaldi-3d/antenna-vivaldi-3d.mp4`: a 50 s video (1080x1080, H.264). It covers the title, the Blender beauty render, the end-fire length limit plot, the gain balloon at 3 GHz, the 2 → 4 GHz sweep, the near-field phase at 3 GHz, the HTML viewer with |E| cuts, the check results (power_closure REVIEW, coarse-vs-fine) and an end card with the repo URL.

## Sources
- 206 mm (10 dBi at 2 GHz, Hansen–Woodyard): examples/use_cases/antennas/results/summary.json -> limits.min_length_hw_mm["2 GHz"] = 205.90; docs/guides/antenna_design.md §1 table
- 125 EMerge processes, all ok: examples/use_cases/antennas/results/summary.json -> runs_total.processes, runs_total.status.ok
- 7.09 → 11.27 dBi (best design, D peak 2 → 4 GHz): examples/use_cases/antennas/results/3d/summary_3d.json -> best.balloon[*].D_peak_dBi (7.0859, 11.2745)
- 0.0 dB balloon vs table: examples/use_cases/antennas/results/3d/summary_3d.json -> best.balloon_vs_table_max_abs_dB
- 0.62 dB coarse-vs-fine RMSE, realized gain; bias-corrected 0.62: examples/use_cases/antennas/results/summary.json -> multi_fidelity.Greal_bore_dBi.rmse_coarse_vs_fine (0.6161), rmse_corrected_lofo (0.6227)
- 5.37 dBi realized gain at 2 GHz (best design): examples/use_cases/antennas/results/summary.json -> best.simulated[0].Greal_bore_dBi (5.3677); also 3d/summary_3d.json best.rows[0]
- 80 rows, 525 support, 35 contradict, all power_closure, REVIEW 35: examples/use_cases/antennas/results/summary.json -> checks_coarse
- 7.2 % median, 34.6 % max power gap; ≈0.5 dB uncertainty: docs/guides/antenna_design.md §7 "Checks" (lines 144–151)
- GP LODO 1.83 dB > 1.5 dB, trust card REVIEW: examples/use_cases/antennas/results/summary.json -> lodo.Greal_bore_dBi.rmse.gp (1.8325); best.card.decisao, best.card.checagens[surrogate_error_known].limit
- 4-core container: docs/guides/antenna_design.md §7 line 124
