# The Ahmed body, end to end

Geometry, mesh study, wind-tunnel validation, a design of experiments in OpenFOAM, a dataset with a held-out region,
surrogate families at equal budget with calibrated uncertainty and physics checks, Sobol indices, a Pareto front with
its CFD checks, uncertainty-driven acquisition, a digital twin, an inverse problem on the published drag measurements,
renders and a short film, and an evidence report. Every stage is one library call
(`pinneapple_autolab.domains.ahmed_study` on `pinneapple_autolab.study`); these scripts only call them. Guide:
[docs/guides/ahmed_body_end_to_end.md](../../../docs/guides/ahmed_body_end_to_end.md).

Every number on this page was measured by these scripts; the JSON in `results/` holds them, `results/report.html`
shows them with their figures and `results/evidence.zip` hashes the sources. What was not measured is said.

## Run it

```bash
python examples/end_to_end/ahmed_body/01_cfd_campaign.py       # CAD, mesh family, validation, DOE (OpenFOAM, resumable)
python examples/end_to_end/ahmed_body/01_cfd_campaign.py mesh_fine validation_fine   # L3 and the L2 validation sweep
python examples/end_to_end/ahmed_body/02_analysis.py           # GCI, validation, dataset, surrogates, Sobol, Pareto plan
python examples/end_to_end/ahmed_body/03_more_cfd.py run       # acquisition, random and Pareto-check runs
python examples/end_to_end/ahmed_body/03_more_cfd.py evaluate  # what the extra runs bought
python examples/end_to_end/ahmed_body/04_inverse.py            # slant from the measured drag, with its posterior
python examples/end_to_end/ahmed_body/05_twin.py               # static twin page in results/twin
python examples/end_to_end/ahmed_body/06_visuals.py            # Blender Cycles stills, the narrated film, viewer, USD
python examples/end_to_end/ahmed_body/07_report.py             # costs, report.html, evidence.zip, experiment.json
cd apps/ahmed_twin && uvicorn ahmed_twin.api:app --port 8171   # the twin with its "verify with CFD" button
```

Without `_runs/cfd` the scripts read the committed run records in `results/cfd_runs`. Environment used: OpenFOAM v1912
(Ubuntu package), gmsh 4.12, Blender (the `bpy` wheel), eSpeak NG and ffmpeg, 4 cores.

## The case

Ahmed, Ramm & Faltin (1984): 1.044 x 0.389 x 0.288 m, 0.1 m front radii, 0.222 m rear slant, stilts left out. Half
model, moving road, steady RANS k-omega SST with wall functions (snappyHexMesh with 5 prism layers). Design variables:
slant 0-40 deg, ground clearance 30-80 mm, speed 20-60 m/s.

## Results

**CAD.** OpenCASCADE B-rep through gmsh, imported back from its STL: closed, symmetric to 1.0 mm, volume 0.10993 m3
against 0.10959 parametric. CFD on the CAD body at the baseline: CD 0.3004 against 0.2966 parametric (+1.3 %).

**Mesh convergence** (baseline 25 deg, 50 mm, 60 m/s, 1000 iterations; constant refinement ratio sqrt(2)):

| mesh | cells | CD | run checks |
|---|---|---|---|
| L0 | 84 223 | 0.2890 | REJECT (mass balance) |
| L1 (campaign) | 207 080 | 0.2966 | REVIEW (residuals) |
| L2 | 534 472 | 0.2707 | REVIEW (residuals) |
| L3 | 1 398 073 | 0.2640 | REVIEW (residuals) |

L0-L1-L2 converges oscillatorily. L1-L2-L3 converges monotonically: observed order 4.26, Richardson value 0.2617, GCI on
L3 1.1 %. The observed order is above the scheme's formal order (2), so the GCI is a lower bound on the numerical
uncertainty, not a sharp estimate. The campaign mesh L1 is 13.3 % above the Richardson value. It was kept for cost
(7.5 min against 19 min for L2 and 107 min for L3 on 4 cores); that 13 % is the main numerical uncertainty of every
surrogate trained on the campaign.

**Validation** against Ahmed et al. (1984), 60 m/s:

| slant deg | measured | L1 | L2 |
|---|---|---|---|
| 0 | 0.250 | 0.2693 (+7.7 %) | 0.2547 (+1.9 %) |
| 12.5 | 0.230 | 0.2557 (+11.2 %) | 0.2399 (+4.3 %) |
| 25 | 0.285 | 0.2973 (+4.3 %) | 0.2715 (-4.7 %) |
| 30 | 0.378 | 0.3378 (-10.6 %) | 0.3184 (-15.8 %) |
| 35 | 0.260 | 0.3835 (+47.5 %) | 0.4079 (+56.9 %) |
| 40 | 0.250 | 0.3005 (+20.2 %) | 0.3958 (+58.3 %) |

Below the critical angle the finer mesh moves the CFD towards the measurements. At 30 deg and above it does not:
steady k-omega SST keeps the slant flow attached past 30 deg, so the drag peak sits at 35-40 deg instead of 30 deg.
That is a model error, not a mesh error, and the mesh refinement makes it larger.

**DOE.** 48 maximin Latin-hypercube designs: centred L2 discrepancy 0.0039 against 0.0141 for random designs of the
same size (median of 20). 48 of 48 runs succeeded.

**Run checks** (every run): checkMesh, non-orthogonality, skewness, final residuals <= 1e-3, force stationarity
<= 1 %, mass balance <= 1e-3, y+ in the wall-function range. Most runs end with a final residual above 1e-3 (steady
RANS of a bluff body with separation), so most decisions are REVIEW. The limits were not widened.

**Dataset** (`results/dataset/ahmed_body.{npz,json}`, a `DatasetSpec` with the SHA-256 and every run id): 54 cases,
train 28 / val 6 / test 10 / extrapolation 10 (every case with slant > 34 deg).

**Surrogates** (5-fold CV on train + val, cross-conformal 90 % intervals; test and extrapolation measured once,
after the selection; selected = *):

| target | family | CV rel. error | test error | test cov. | extrap. error | extrap. cov. | physics |
|---|---|---|---|---|---|---|---|
| CD | GP | 0.343 | 0.0042 | 1.0 | 0.052 | 0.6 | pass |
| CD | MLP ensemble * | 0.230 | 0.0058 | 0.9 | 0.055 | 0.7 | pass |
| CD | quadratic | 0.358 | 0.0063 | 1.0 | 0.055 | 0.4 | pass |
| CL | GP * | 0.105 | 0.020 | 0.9 | 0.184 | 0.7 | pass |
| CL | MLP ensemble | 0.178 | 0.041 | 0.9 | 0.173 | 0.8 | pass |
| CL | quadratic | 0.120 | 0.022 | 0.9 | 0.212 | 0.2 | pass |
| CD_friction | quadratic | 0.348 | 0.0010 | 0.9 | 0.0009 | 1.0 | fails "falls with Re" |
| Cp | POD-GP * | 0.113 | 0.099 | 0.89 | 0.200 | 0.83 | pass |
| Cp | POD-MLP | 0.131 | 0.119 | 0.85 | 0.262 | 0.89 | pass |
| Cp | POD-RBF | 0.129 | 0.115 | 0.92 | 0.304 | 0.68 | pass |
| Cp | DeepONet | 0.124 | 0.115 | 0.89 | 0.286 | 0.97 | fails the Cp integral |

Errors: RMSE for scalars, mean relative L2 for Cp; coverage of Cp is pointwise. Inside the training region the
intervals cover at about their nominal 90 %. Beyond it (slant > 34 deg) they under-cover for CD and CL (0.4-0.8):
the extrapolation split is where the models are not to be trusted, and the twin says so.

**The best-model rule** (`pinneapple_autolab.study.benchmark.SELECTION_RULE`): only families that pass every physics
check are eligible; the lowest cross-validated error wins; families within 5 % of it are tied and the narrower
interval wins; the test and extrapolation splits never enter the choice.

**The Cp check and its tolerance.** The field and the scalar model must agree: the pressure drag integrated from the
predicted Cp against the predicted CD_pressure. Its p95 over 128 Sobol points, divided by the quadrature sum of the
CFD's own template gap and the scalar model's calibrated half width, must be <= 1. The template gap is the CFD's Cp
integrated on the 1751 template points against the CFD's CD_pressure: p95 0.0415 and maximum 0.069 over the 54 cases.
Ratios measured: POD-GP 0.56, POD-MLP 0.64, POD-RBF 0.51, DeepONet 1.007. An earlier DeepONet fit gave 1.044; it
trains to a wall-clock budget, so its fit varies run to run. The decision, the rejected alternatives and the old
formulation (under which no family was eligible) are in the guide and in the report.

**Sobol indices** on the selected surrogates over the training region (total index, slant / clearance / speed):
CD 0.96 / 0.05 / 0.01, CL 0.94 / 0.06 / 0.00, CD_friction 0.09 / 0.04 / 0.93.

**Inverse problem.** The posterior of the slant from a drag measurement comes from a mixture density network,
amortised over 40 000 draws of the CD surrogate. Its noise model is the surrogate's calibrated uncertainty plus the
CFD-vs-wind-tunnel discrepancy (sigma_d 0.059, RMS over the validation sweep). Synthetic checks: 90 % credible
intervals cover 91 % of 300 surrogate draws (slant). The slant of 19 of 20 held-out CFD cases lies inside its 90 % HDI.
On the six measurements of Ahmed et al. (1984) the measured slant lies inside the 90 % HDI for 4 of 6. The HDIs span
most of 0-35 deg, because the discrepancy is as large as the whole drag range. 35 and 40 deg fall outside: the CFD
curve puts the drag peak there, so a measured 0.25-0.26 maps to low slants. Metropolis on the same
likelihood agrees with the MDN medians within 2 deg. Multi-start least squares finds several roots per measurement
(for example 0 and 23.5 deg at 25 deg), the non-uniqueness the posterior shows. Lienhart & Becker (2003, 40 m/s, 25 and 35 deg): their drag
values could not be obtained from this environment, so that set is listed with its conditions and was not inverted.

**Twin.** The page shows sliders, the Cp field on the body (mean or 90 % half width), CD and CL with their band, and
a warning beyond 0.25 (unit cube) from the nearest CFD run. It was checked in Chromium: no console errors, and the
warning appears at 8 deg, 80 mm, 60 m/s (distance 0.45). Served by `apps/ahmed_twin`, the "verify with CFD" button
ran an OpenFOAM case at 20 deg, 45 mm, 40 m/s in 9.7 min and reported it against the band. The page interpolates a
grid of model evaluations. Its maximum error against the models on random points is 0.013 for CD, 0.033 for CL and
0.58 for Cp (at the slant edge, where Cp changes fastest with the slant), so the page's Cp is a preview and the
models are the reference.

**Visuals** (`results/visuals`, `results/visuals.json`). Four Blender Cycles stills of the 25 deg validation run on a
dark product stage (64 samples, 1600 x 900): Cp with streamlines from the front and from the rear, speed on the
symmetry plane, streamwise vorticity in the wake. A 65 s film in English (1280 x 720, 12 fps, eSpeak NG narration):
a title card, an orbit of the CFD run, the two slices, the slant swept 0-40 deg through the Cp surrogate (each frame
labelled "no CFD in this shot", with its mean 90 % half width), and an end card with the test errors. Also a three.js
viewer and an OpenUSD stage carrying the run's trust decision (validated by `pxr.UsdValidation`, no problems). The
full render took 64 min on 4 cores, shared with the CFD.

**Extra CFD** (acquisition, random, Pareto checks): see `results/more_cfd.json` and the report. Any of the 19 runs
not finished when the report was written are listed there as not run.

**Cost** (`results/costs.json`): CFD wall time x MPI processes per purpose, plus the recorded seconds of the other
stages. See the report for the table; the total so far is 37 CPU-h.

## What was not measured, and the limits

- Steady RANS k-omega SST with wall functions cannot reproduce the drag peak at the critical angle; see the
  validation table.
- The campaign mesh carries a 13 % numerical error at the baseline (against Richardson). The GCI uses an observed
  order above the formal one.
- Most runs fail the 1e-3 residual limit (REVIEW). Their outputs are window means, with the spread recorded per run.
- Ahmed et al. (1984) values: 12.5 deg, 35 deg and the Reynolds number were confirmed by secondary sources, and 25 deg
  is widely quoted. 0, 30 and 40 deg are the repository's transcription (the SAE paper was not reachable). Lienhart &
  Becker drag values: not obtained.
- The surrogates are valid in the DOE box only; the extrapolation split shows how they fail beyond it.
