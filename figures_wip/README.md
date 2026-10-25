# Ahmed-body study: every image produced so far (work in progress)

Snapshot taken 2026-10-25 20:58 UTC from the session on `claude/ahmed-body-end-to-end`. Labels:

- **final**: measured on the final runs;
- **interim**: measured, but before the L3 mesh and the extra runs (validation on L2, acquisition, random, Pareto
  picks) entered the analysis; the numbers will move when the final analysis runs;
- **preview**: a render test at reduced quality (16 Cycles samples, 960 x 540 / 1280 x 720, few frames).

Nothing here is final yet: the L3 mesh run finished (CD 0.2640 on 1.40 M cells, 107 min), but the analysis has
not been re-run with it; the L2 validation sweep and the 19 extra runs are in progress.

| File | Label | What it shows |
|---|---|---|
| `figures/validation_cd.png` | interim | CFD (L1 campaign mesh) against Ahmed et al. (1984), six slant angles; the L2 sweep will be added |
| `figures/grid_convergence_cd.png` | interim | baseline CD on L0, L1, L2 (oscillatory); L3 is measured but not yet in the figure |
| `figures/parity_CD.png`, `figures/parity_CL.png` | interim | every scalar family on the test and extrapolation cases, 54-run dataset |
| `figures/coverage_cd.png` | interim | coverage of the cross-conformal 90 % intervals of CD per family |
| `figures/coverage_cp.png` | interim | pointwise coverage of the Cp field families (re-run with the new field-scalar check) |
| `figures/sobol_cd.png` | interim | Sobol indices of CD on the selected surrogate, training region |
| `figures/posterior_slant_25.png`, `figures/posterior_slant_30.png` | interim | MDN and Metropolis posteriors of the slant from the measured CD at 25 and 30 deg (deployment surrogate on 54 runs) |
| `twin/twin_app_slant40.png` | interim | the twin page served by `apps/ahmed_twin` (verify button enabled), sliders at 40 deg / 50 mm / 60 m/s |
| `twin/twin_static_slant40.png` | interim | the static copy of the page (verify button disabled) |
| `renders/cp_streamlines_front.jpg`, `renders/cp_streamlines_rear.jpg` | preview | Blender Cycles, dark stage: Cp on the 25 deg validation run with streamlines coloured by speed |
| `renders/mid_plane_speed.jpg`, `renders/wake_vorticity.jpg` | preview | the symmetry-plane speed and the wake vorticity slices of the same run |
| `film/ahmed_body.mp4` (1.8 MB, 65 s, English narration), `film/ahmed_body.gif` | preview | the film cut at quick settings: title card, CFD orbit (12 frames), slices, surrogate sweep (9 frames), end card |
| `film/frame_1s.png`, `film/frame_30s.png`, `film/frame_45s.png`, `film/frame_52s.png` | preview | frames of that film |
| `film/caption_test.png` | preview | the caption / title-card style test |

Not included: raw CFD data (run records, fields and flow scenes stay in the study folder and in PR 2).
