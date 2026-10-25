# Handoff: the Ahmed-body end-to-end study

State at hand-off (2026-10-25, branch `claude/ahmed-body-end-to-end`). Every number below was measured in this
folder's runs; the JSON in `results/` holds them. Nothing here is a target or an estimate unless it says so.

## The case and why

The Ahmed body (Ahmed, Ramm & Faltin, SAE 840300, 1984): a public, fully specified bluff body (1.044 x 0.389 x
0.288 m, 0.1 m front radii, 0.222 m rear slant) with a measured drag curve against the slant angle at 60 m/s
(`pinneapple_autolab.domains.ahmed.AHMED_1984_CD`, transcribed earlier in `examples/qualitative/03_ahmed_slant.py`;
**not re-verified against the paper here**: kbwiki.ercoftac.org, cfd.mace.manchester.ac.uk, cfd-online.com,
zenodo.org and arxiv.org were blocked by the proxy). It gives real measurements for the inverse step (a heat sink
would have given correlations) and the library already had a generic OpenFOAM external-flow pipeline.

## Where things are

| What | Where |
|---|---|
| Generic stages (DOE + resumable costed runs, GCI, dataset, benchmark + UQ + physics checks + selection rule, Sobol / NSGA-II / acquisition, twin page, figures, report) | `pinneapple_autolab/study/` |
| Ahmed simulator, CAD, checks | `pinneapple_autolab/domains/ahmed.py` (`AhmedAero`, `ahmed_cad`, `import_body`, `qa_checks`, `prediction_checks`, `cp_from_flow`) |
| Ahmed stages (one call each) | `pinneapple_autolab/domains/ahmed_study.py` |
| Amortised inverse (MDN) | `pinneapple_analysis/inverse_problems/amortized.py` |
| Library changes | `ExternalFlow` (prism layers, `cells_per_length`, y+), `pp.viz.render` (`background="dark"`, `cmap`, `floor`), `ahmed_body` (`n_body`, `n_slant`, `n_corner`, `side`, `slant_length`) |
| Flow Studio | nodes `pinneapple_flow/nodes/body_study.py` (`study.*`), flow `pinneapple_flow/flows/ahmed_body_end_to_end.json` |
| App | `apps/ahmed_twin` (serves the twin page, queues verifying CFD) |
| Tests | `tests/test_autolab_study.py` (fast; `PINNEAPPLE_SLOW=1` adds one OpenFOAM run) |
| Docs | `docs/guides/ahmed_body_end_to_end.md` (structure; no numbers yet), AGENTS.md row, CHANGELOG |
| Run records (58) and Cp fields | `results/cfd_runs/{results,fields}` (committed copy of `_runs/cfd`); `environment.json` = the machine |
| Flow scenes (skin Cp/Cf, streamlines, slices) | `_runs/cfd/cases/<id>.flow.npz` in the old container only (185 MB, not committed); `results/cfd_runs/cases/val_25.0.flow.npz` is committed for the renders |
| Results JSON + figures | `results/*.json`, `results/figures/*.png`, `results/dataset/ahmed_body.{npz,json}` |

The scripts read `_runs/cfd` when it exists, else `results/cfd_runs`. To resume CFD into the committed records:
`mkdir -p _runs && cp -r results/cfd_runs _runs/cfd` (then `01_cfd_campaign.py` skips every finished id).

## Environment

OpenFOAM v1912 (Ubuntu `openfoam` package, `FOAM_BASHRC=/usr/share/openfoam/etc/bashrc`), gmsh 4.12 and Blender 4.0
(apt), `pip install -e ".[dev]" scikit-learn sqlalchemy fastapi uvicorn bpy usd-core` (torch from PyPI:
download.pytorch.org is blocked). 4 cores: one L1 run with 800 iterations takes 4.5-5 min.

## Done, with the measurements

**Geometry.** Parametric body; field template of 1751 points on the y >= 0 half, the same material points for every
variant (`field_vertices`). CAD: OpenCASCADE B-rep through gmsh (`results/cad/ahmed.geo`, `.step`; the STL is
regenerated), imported back by `import_body`: closed, symmetric to 1.0 mm (tessellation chord), volume 0.10993 m3
vs 0.10959 parametric (the CAD's long edges are sharp), named patches (`results/cad_import.json`). CFD on the CAD
body at the baseline: CD 0.3004 vs 0.2973 parametric (+1.0 %), checks REVIEW (residuals).

**Mesh.** Without prism layers the median first-cell y+ was 540-720 at 60 m/s (outside the wall-function range);
5 relative layers give median 96 (p05 50, p95 248), 96 % of faces in 30-300 (baseline, L1). Family with constant
ratio sqrt(2), baseline 25 deg / 50 mm / 60 m/s, 1000 iterations: CD 0.2890 (L0, 84 k cells), 0.2966 (L1, 207 k),
0.2707 (L2, 534 k). **Oscillatory**: the GCI does not apply (`results/mesh_study.json` says so); the campaign mesh L1
differs from L2 by 9.5 %. L3 (about 1.4 M cells) was started and stopped for the hand-off: not run.
800 iterations reproduce the 1000-iteration window mean within 0.24 % (CD), 1 % (CL) on L1.

**Validation (L1, 60 m/s, 50 mm)** vs Ahmed 1984: 0 deg 0.269 vs 0.250 (+7.7 %), 12.5 deg 0.256 vs 0.230 (+11.2 %),
25 deg 0.297 vs 0.285 (+4.3 %), 30 deg 0.338 vs 0.378 (-10.6 %), 35 deg 0.384 vs 0.260 (+47.5 %), 40 deg 0.301 vs
0.250 (+20.2 %). The CFD drag peak is at 35 deg, not 30 deg: at 35 deg the slant flow stays attached (slant pressure
drag 0.284, base 0.017), the known k-omega SST failure at the critical angle. `results/figures/validation_cd.png`.

**DOE.** 48 maximin LHS designs (slant 0-40 deg, clearance 30-80 mm, speed 20-60 m/s): discrepancy 0.0039 vs 0.0141
for random designs (median of 20), minimum distance 0.111 vs 0.049. 48/48 ok.

**Run checks** (58 runs): residual > 1e-3 in 49 (largest final initial residual 3e-3-1e-2, typically Uz / Uy),
force stationarity (window halves differ > 1 %) in 14 (mostly slant >= 30 deg), mass balance > 1e-3 in 4 (L0 and
three others); trust decisions 7 APPROVED, 47 REVIEW, 4 REJECT. Limits were not widened.

**Cost** (`cost_table`): 58 runs, 0 failed, 4.96 h wall, 19.9 CPU-h (doe 15.4, validation 1.8, mesh + CAD 2.6).

**Dataset** `results/dataset/ahmed_body.{npz,json}` (DatasetSpec, SHA-256, run ids): 54 cases (DOE + validation),
split train 28 / val 6 / test 10 / extrapolation 10 (slant > 34 deg). Validation runs <= 34 deg are in test.

**Surrogates** (`results/benchmark.json`; 5-fold CV on train + val, cross-conformal 90 % intervals; test /
extrapolation measured once after selection):

| target | chosen (rule) | CV rel. RMSE | test RMSE | test cov. 90 % | extrap. RMSE | extrap. cov. |
|---|---|---|---|---|---|---|
| CD | mlp_ensemble | 0.23 (gp 0.34, quadratic 0.36) | 0.0058 | 0.90 | 0.055 | 0.70 |
| CL | gp | 0.10 (quadratic 0.12, mlp 0.18) | 0.020 | 0.90 | 0.184 | 0.70 |
| CD_pressure | mlp_ensemble | 0.34 | 0.010 | 0.80 | 0.062 | 0.90 |
| CD_friction | mlp_ensemble | 0.44 (quadratic fails the Re-trend check: 0.88 < 0.90) | 0.0010 | 0.80 | 0.0011 | 1.00 |
| Cp field | **none** | pod_gp 0.11, deeponet 0.12, pod_rbf 0.13, pod_mlp 0.13 (rel. L2) | pod_gp rel. L2 0.099 | 0.89 | 0.20 | 0.83 |

No Cp family is eligible: all fail `cp_integral_matches_CD_pressure` (pod_gp 0.055 vs tolerance 0.052, deeponet
0.213). The tolerance is the CFD's own template gap (p95 0.042 + 0.01, `results/template_integral.json`). See
"Open decisions".

**Also written** by `02_analysis.py`: `sobol.json` + `figures/sobol_cd.png`, `pareto_plan.json` (CD vs CL at 40 m/s
with CD half width <= 0.02, three picks), `next_runs.json` (8 acquisition designs, 8 random, 3 Pareto picks),
`figures/{grid_convergence_cd,validation_cd,parity_CD,parity_CL,coverage_cd,coverage_cp}.png`.

**Tested on synthetic records only** (all stages run end to end, `tests/test_autolab_study.py` + a smoke run):
inverse (`ahmed_study.inverse`), twin page (`ahmed_study.twin`; checked in Chromium: sliders, morphing body,
diverging Cp, band chart, trust warning, no console errors), visuals (one dark-stage still checked by eye).

## What remains, per stage, with the commands

Run from the repository root; every CFD stage is resumable.

1. **Mesh convergence.** L3 and the validation sweep on L2 (separates mesh and model error):
   `python examples/end_to_end/ahmed_body/01_cfd_campaign.py mesh_fine validation_fine` (about 1.5 h + 2 h).
   Then `02_analysis.py` refreshes `mesh_study.json` / `validation.json` (4-level triplets).
2. **Open decision on Cp eligibility** (below), then re-run `02_analysis.py` (about 10 min).
3. **Extra CFD (extras).** `python examples/end_to_end/ahmed_body/03_more_cfd.py run` (19 runs, about 1.5 h), then
   `03_more_cfd.py evaluate` (error drop: acquired vs random; Pareto picks predicted vs CFD).
4. **Inverse on the wind-tunnel data.** `python examples/end_to_end/ahmed_body/04_inverse.py` (synthetic checks,
   held-out CFD cases, then the six measurements with leave-one-out discrepancy; MDN vs least squares vs
   Metropolis). Expect the 30 / 35 deg measurements to sit where the CFD curve is wrong: report it, the
   `measurement_within_model_reach` flag is there for that.
5. **Twin.** `python examples/end_to_end/ahmed_body/05_twin.py` (static page in `results/twin`) and the app
   `cd apps/ahmed_twin && uvicorn ahmed_twin.api:app --port 8171` (check the CSP of `apps/_shared/appkit.install`
   lets the page's inline module script run; untested in the app). Needs a selected Cp family.
6. **Visuals.** `python examples/end_to_end/ahmed_body/06_visuals.py` (stills, mp4 + gif, three.js viewer, USD).
   Uses `results/cfd_runs/cases/val_25.0.flow.npz`. Full quality is about 1-2 h of Cycles on 4 cores.
7. **Report + manifest.** `07_report.py` is not written: build the sections from `results/*.json` with
   `pinneapple_autolab.study.report.write_report` + `evidence_package`, wrap in `pinneapple_lab.experiment(...)`
   (`register=False`) so `experiment.json` lists the metrics with units and how they were measured; fill the
   numbers into `docs/guides/ahmed_body_end_to_end.md` and a `README.md` here.
8. **PRs.** Split as asked: library first, then example + app, then visuals.

## Open decisions and known issues

- **Cp eligibility.** The integral check compares the field's pressure drag with the *selected CD_pressure model*,
  so its gap includes that model's own error (test RMSE 0.010). Options: compare with the CFD-derived value on the
  test cases instead; or widen by the scalar model's conformal half width (justify it, do not just widen); or derive
  CD_pressure from the field. Decide and document it before re-running.
- **Mesh.** Oscillatory convergence on L0-L2; L1 vs L2 differ by 9.5 % at the baseline. If L3 does not settle it,
  the campaign mesh choice (cost) must be stated as the main numerical uncertainty.
- **Residuals.** 49 of 58 runs end above 1e-3 (steady RANS of a bluff body with separation); force windows are
  averaged over iterations 400-800 and their spread is in each record (`meta.iteration_std`).
- **Cp vs scalars.** Scalars are window means; Cp is the final iteration's skin (re-sampled from the flow scene by
  `cp_from_flow`, so the template can change without re-running CFD).
- **References.** Ahmed 1984 values were not re-checked online (hosts blocked); no Lienhart & Becker profiles.
- **Pre-existing test failures** unrelated to this branch: `test_studio_core_js_matches_python`, three
  `test_blender_bridge.py` tests with the apt Blender, `test_flow_runtime.py` needs `sqlalchemy`.
