# Black-hole weather: forecasting accretion flows with a trust horizon

`pinneapple_physics.blackhole` reproduces and extends **Duarte, Nemmen & Navarro (2022)**, *Black hole weather
forecasting with deep learning: a pilot study* (MNRAS 512, 5848; [arXiv:2102.06242](https://arxiv.org/abs/2102.06242);
code, MIT: [black-hole-group/DL_BH_fluids](https://github.com/black-hole-group/DL_BH_fluids)). The original work trained
a U-Net on hydrodynamic simulations of the hot, radiatively inefficient gas around a black hole and forecast the flow
about 10⁴ times faster than the fluid solver. This module adds what the weather work in PINNeAPPle asks of any
forecast: **for how long can it be trusted, and can we tell without the truth?** Issue
[#399](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/399).

## Pieces

| module | what it does |
|---|---|
| `hydro` | `AccretionFlow`: axisymmetric viscous hydrodynamics around a Schwarzschild black hole (Paczyński-Wiita potential, α-viscosity "SS" or "ST", equilibrium torus), finite volume (HLL, MUSCL, SSP-RK2) on a log-r grid; `bondi_pw`: exact transonic Bondi accretion for validation |
| `forecast` | `DuarteUNet` (layer-for-layer port of the original), `DensityCodec`, `make_blocks`, `train_forecaster`, `rollout`, `lead_time_scores`, `trust_horizon`, `MassEnvelope` |
| `family` | the model family (`DuarteUNet`, `FourierForecaster`, `SphereBackbone`), the two aggregations (`EqualWeightEnsemble`, `adaptive_ensemble`) and `evaluate_family`, which scores all of them with `skill.evaluate_skill` |
| `twin` | `accretion_scene`: the flow revolved in 3D as a cutaway for PINNeAPPle-Twin3D, simulation and forecast side by side, trust signals as sensors |
| `capture` | `capture_twin`: records the Twin3D viewer (headless Chromium) to frames for GIFs |

## The simulations

The paper's training data are PLUTO runs (Almeida & Nemmen 2020) that are not reachable from every environment, so
`examples/black_hole_weather/simulate.py` regenerates the same kind of flow with the library's own solver:

* units G = M = c = 1; potential Φ = −1/(r − 2), which places the innermost stable orbit at r = 6;
* adiabatic gas (γ = 5/3) without cooling, total energy evolved so viscous heat stays in the gas;
* viscosity only in the azimuthal stresses T<sub>rφ</sub>, T<sub>θφ</sub> (Stone, Pringle & Begelman 1999), with
  ν = α c<sub>s</sub>²/Ω<sub>K</sub> ("SS") or ν = α r<sup>1/2</sup> ("ST");
* initial equilibrium torus with l(R) = l<sub>c</sub>(R/R<sub>c</sub>)<sup>a</sup> (pressure maximum at R<sub>c</sub> = 20,
  inner edge 12) in a cold atmosphere; r from 4 to 400 GM/c², 128 × 64 cells;
* runs named like the paper's, `PL{a}{SS|ST}{α}`.

**Validation** (`tests/test_blackhole_weather.py`):

* mass and angular momentum close to round-off (10⁻¹⁶) once the boundary fluxes are counted;
* a uniform pressure at rest stays at rest (exact geometric source terms);
* the inviscid torus holds its equilibrium;
* exact Bondi accretion in the Paczyński-Wiita potential stays steady: accretion rate within 0.5 % and density
  within 0.2 % (median) over 2000 GM/c³.

As in other black-hole codes, the tenuous atmosphere has floors and a speed / temperature ceiling (`v_max`, `t_cap`);
`AccretionFlow.capped` counts the cells where they act (≈1 % of the cells, next to the hole).

## The forecaster

Five consecutive density snapshots go in as channels, the next five come out; the density is log-normalised and the
outer atmosphere and polar cells are cropped, as in the original. The loss is `MAE + α·MAE(pixels with y > 0.5)` (the
paper's multi-simulation loss; its one-simulation loss uses pixel boxes of its own grid, which do not transfer).

## Trust

* **Trust horizon.** Iterative rollouts from many start times; per lead time, the error of the forecast against
  persistence and against the time-mean flow, and the anomaly correlation (ACC) with the simulation. The horizon is
  where the ACC drops below 0.6 (the weather convention).
* **Mass check, no truth needed.** The gas in the forecast window can only change by what crosses its edges, and the
  training simulations show how fast that ever happens (`MassEnvelope`). A rollout whose mass moves faster is creating
  or destroying gas — the "artificial mass injection" the original authors identify as the cause of drift — and the
  first lead where that happens is a trust limit available in operation.

## 3D digital twin

```python
from pinneapple_physics.blackhole.twin import accretion_scene
sc = accretion_scene(grid, {"simulation": logrho_true, "forecast": logrho_pred}, times, r_view=60,
                     sensors=[{"id": "forecast error", "panel": "forecast", "series": err, "envelope": (0, err_max)}])
sc.export("out/bh_twin")            # pinneapple_twin3d.serve("out/bh_twin") -> browser
```

The axisymmetric flow is revolved into a 3/4 cutaway (two meridional planes and the equatorial sector) around a black
sphere at the event horizon. The viewer accepts `?field=log10_density&cmap=inferno&view=1,0.75,-1&range=-6,0` to open
on that view.

## Interstellar view (general-relativistic ray tracing)

`raytrace.Camera.trace` integrates the null geodesics of the Schwarzschild metric once per camera: each ray stays in
a plane through the hole, and u = 1/r obeys u'' = 3u² − u, solved by RK4 with arc-length-limited steps.
`raytrace.render` then integrates the emission of any (r, θ) density / temperature / velocity frame along those rays.
It includes:

- the gravitational and Doppler redshift (I ∝ g⁴);
- optional grey absorption;
- a colour temperature that follows g;
- a procedural star field that is lensed with the flow;
- a film-like bloom.

The tests check that the shadow edge sits at the critical impact parameter b = 3√3 M and that a thin disc shows its
far side lensed above the shadow, as in *Interstellar* (James et al. 2015).

### Orbiting, non-axisymmetric discs (#1091)

Every sample of a trace also stores its azimuth φ, so the emission no longer has to be axisymmetric:

- `render` takes fields of shape (nr, nθ, nφ), periodic in φ, or a thin-disc `modulation` of shape (nr, nφ) that
  multiplies the density.
- `orbiting_clumps(r, t)` builds such a modulation: hot spots and a streaky background carried round at
  `keplerian_omega(r)` = √(M/r³), the exact circular-orbit rate in Schwarzschild coordinate time. The inner disc laps
  the outer one, so spots shear into spirals and flare on the approaching side (g⁴).
- `camera_azimuth`: the metric is static and axisymmetric, so moving the camera by α in azimuth is the same trace with
  the flow turned by −α. The stars turn with the camera. An orbit costs no new trace.
- `render_series(paths, rho, T, v, frames)` renders many frames of one trace in which only the modulation and the
  camera azimuth change. It gathers the axisymmetric fields once, so each frame costs one lookup and two sums.
- `RayPaths.select(mask)` drops samples that can never emit, for example outside a thin disc. This keeps large
  frames in memory.

Light-travel-time differences across the disc are neglected ("fast light").

The tests check that:

- a φ-constant field renders like the axisymmetric one;
- the camera orbit equals the flow turned back;
- a spot returns after 2π/Ω(r) while an outer ring does not;
- a spot is more than twice as bright on the approaching side, and about equally bright with beaming off;
- `render_series` matches `render` frame by frame.

The twin also renders in Blender: `Scene.render_blender` / `render_blender_frames` (Cycles, `pip install bpy`).

## Resolution study, zoo dataset and retrained forecaster

The reference was checked on four grids (90 × 45 to 256 × 128, ratio √2, three seeds each, the same seeded
perturbation on every grid). The tools are in the library:

- `convergence.resolution_study`: conservative remap, Richardson order and GCI through
  `pinneapple_analysis.verification`, and statistical convergence;
- `dataset`: the zoo dataset `blackhole_riaf_density_256`, whole trajectories per split;
- `forecast.train_on_trajectories`: rollout loss, input noise, mass penalties, resumable;
- `skill.evaluate_skill`: persistence, climatology and DMD baselines, moving-block bootstrap intervals, physics
  checks;
- `figures`.

![resolution study](../assets/blackhole/convergence_256.png)

- **The means converge** within their 95 % bands: the accretion rate is 2.18 ± 0.58 on 181 × 90 and 1.97 ± 0.63 on
  256 × 128, and the mass in the window and the density slope also agree.
- **The profiles are not in the asymptotic range.** From 181 to 256 they change by 4-11 %, against 0.5-2.5 % between
  the coarser grids, and l/l_K rises 3.8 % beyond its band. The likely cause is that the speed and temperature
  ceilings act on more cells as the resolution grows. These changes are the stated uncertainty band of the data.
- **The forecaster is not shipped.** Retrained on 256 × 128 trajectories with replicate padding, a rollout loss and a
  mass penalty, it beats persistence with 95 % confidence from 60 to 2000 GM/c³ (skill 0.27 at 2000 M) and beats
  DMD. But it breaks the mass envelope of the training runs in 14-54 % of the frames.

![skill against lead time](../assets/blackhole/skill_vs_lead_256.png)

Details and every number: `examples/black_hole_weather/README.md`, section "Zoo version".

## The model family and its ensemble (#399, item 4)

Item 4 of the issue asks for the paper's U-Net, a Fourier/spherical neural operator and the SphereUNet-style backbone
of `pinneapple_physics.weather`, combined in `PhysicsEnsemble`, and for the measured answer to whether that
combination beats the best single model over the horizon. `pinneapple_physics.blackhole.family` is that comparison.
Every member obeys the contract of `forecast.rollout` — five consecutive normalised log-density frames in, the next
five out, same shape — so all of them are rolled out and scored by the same `skill.evaluate_skill` as the numbers
above, and their horizons are the same quantity.

### The three members

| member | what it is | adaptation stated, not hidden |
|---|---|---|
| `duarte-unet` | `DuarteUNet` (the paper's U-Net, 5×5 convolutions, filters 16, residual) | none: the port documented above |
| `fourier-operator` | `FourierForecaster`: `FNO2d` of `pinneapple_neural` (width 32, four spectral layers, 12 × 8 low modes, two cells of zero padding on every axis) mapping the 5 frames to their change, added to the last input frame | the pixel grid is (r, θ), not periodic; the padding keeps the two open radial edges from wrapping into each other through the transform |
| `sphere-backbone` | `SphereBackbone`: the `SphereUNet` of `pinneapple_physics.weather` — residual group-norm blocks, the patch-unshuffle embedding (off here: `patch = 1`, the window is small enough to keep the U-Net at full resolution), zero-initialised head, so an untrained backbone is exactly persistence | its `sphere_pad` (periodic in longitude, wrapping across the poles) is replaced by `grid_pad` (zeros in r, replication in θ): the accretion grid is one meridional half-plane with an open outer edge and reflecting poles, so the globe's padding is the wrong stencil here; the state is the block of frames and the "previous state" is the same block shifted by one frame |

Why a Fourier operator at all: the axisymmetric structure and the spiral waves of the flow live in a few low modes of
the plane, which is exactly what the spectral convolution carries and what a 5×5 kernel has to build out of depth.
The three are trained with the paper's loss (`MAE + 8·MAE` of the high-density pixels), Adam with gradient clipping,
on the same blocks; only the learning rate differs (2·10⁻⁴ for the deep U-Net, 10⁻³ for the two shallower members,
for the reason `forecast.TrainConfig` gives).

### The two aggregations

* `equal` — `EqualWeightEnsemble`: the arithmetic mean of the three fields. It needs no label, and it is an exact
  identity on identical members (`combine([P, P], [½, ½]) == P`).
* `adaptive` — `adaptive_ensemble`: `PhysicsEnsemble(mode="combine")` over the same three. Weights come from the
  fixed-share / AdaHedge grid and are updated **prequentially**: the forecast of case k is combined with the weights
  the labels of the cases before k produced, never its own. Before each case the weights are tilted by
  `residual_lookahead` on a signal that needs no truth — the mass rate of each member's own rollout, `|d ln M/dt|` of
  the window in units of the training bound, the same idea as `MassEnvelope` above. `min_weight = 0`, so the
  aggregation stays a convex combination and `combine` reproduces the ensemble's field exactly (a test checks it).

The honest reading of the two: `equal` is "combine them and see"; `adaptive` also uses the truth of *earlier* cases,
which an operational forecast has from verification against the analysis. Neither sees the truth of its own case.

### The measurement

`family.evaluate_family` runs every member and both aggregations over the same start times of the held-out trajectory
(10-block rollouts, 1000 GM/c³, one every 5 frames, moving-block bootstrap with blocks of 1000 GM/c³), with
persistence, climatology and DMD as baselines, and the best single model chosen on the **validation** trajectory —
never on the test one. `figures.plot_family` is the figure, `figures.forecast_gif` the animation.

The run below is this machine, `runs/bh_data`: train on `bh_n128_s0`, validation on `bh_n128_s4`, test on
`bh_n128_s5` — 22 start times, 50 lead times from 20 to 1000 GM/c³, window 96 × 48, `frame_dt` 20, DMD at the rank
its validation MAE chose (20). The skill score is against persistence at the 1000 GM/c³ lead, with the moving-block
bootstrap interval.

| model | MAE at 20 M | MAE at 1000 M | mean MAE | skill at 1000 M | skill horizon | ACC horizon |
|---|---|---|---|---|---|---|
| `duarte-unet` | 0,00292 | 0,0642 | 0,0319 | −0,224 [−0,301; −0,178] | 0 M | 400 M |
| `fourier-operator` | 0,00122 | 0,0244 | 0,0152 | **+0,534 [+0,460; +0,619]** | 1000 M | 640 M |
| `sphere-backbone` | 0,00251 | 0,0424 | 0,0183 | +0,191 [+0,185; +0,231] | 0 M | 580 M |
| `ensemble-equal` | 0,00161 | 0,0323 | 0,0167 | +0,491 [+0,435; +0,550] | 1000 M | 680 M |
| `ensemble-adaptive` | 0,00131 | 0,0256 | 0,0142 | **+0,511 [+0,410; +0,627]** | 1000 M | — |
| persistence | 0,00184 | 0,0525 | 0,0316 | 0 | — | — |

The answer to item 4 is not the comfortable one, and it is split in two:

- **the plain mean does not beat the best single model.** `ensemble-equal` is better than `fourier-operator` at
  **3 of the 50 leads** — averaging three members when one of them is far ahead and one of them is `duarte-unet`,
  which ends the horizon at −0,224, mostly drags the best one down. Its mean MAE (0,0167) is worse than the single
  Fourier operator's (0,0152);
- **the adaptive aggregation does**, at **43 of the 50 leads**, with the best mean MAE of the whole table (0,0142).
  It gets there by weight, not by averaging: the final weights are `fourier-operator` 0,959, `sphere-backbone` 0,031,
  `duarte-unet` 0,011, and they are assigned **prequentially** — the forecast of a case uses only the labels of the
  cases before it.

Two honest qualifications on top of that table:

- **DMD is not beaten anywhere.** `skill_horizon_vs_dmd_M` is 0 M for every member and both aggregations: the linear
  model at rank 20 has a validation MAE of 0,0213, and beyond the first lead no network here stays ahead of it. The
  ensemble comparison above is against the best *single network*, which is the question the issue asks, not against
  the best *forecast* available.
- **the deep member is the one that fails.** `duarte-unet` has 13,6 M parameters against 0,79 M for the Fourier
  operator and is the only member with negative skill at the horizon; its validation MAE (0,0277) never came near the
  Fourier operator's (0,0151). At filters 16 and 12 epochs it is under-trained rather than refuted, and that is what
  the GPU budget below would buy.


### What a GPU would be needed for

The comparison is honest about its size. It is **not** a training campaign:

* the data are three runs of the library's own solver at 128 × 64, 3200 GM/c³ each, not the paper's PLUTO runs
  (256 × 192 crop) and not the zoo dataset (256 × 128, 8000 GM/c³, six trajectories);
* each member is trained once, with no hyper-parameter search, in the budget the number of epochs above describes;
  the paper's U-Net is at filters 16 instead of 32, and the weather backbone's default widths (128-384) are cut to
  16-48;
* a GPU would buy the paper's resolution and filters, several training runs per member (the spread between them is
  the part of the comparison this cannot measure), and the multi-simulation training the paper's own loss is written
  for.

What the measurement does support is the question item 4 asks: on the same data, the same contract and the same
scoring, does combining the three beat the best one alone, with intervals.


## Results of the first version

The port of the paper's U-Net does not beat persistence at the 20 GM/c³ frame spacing used here. It loses or gains
4-6 % of the gas per frame, which the mass check flags at the first frame. The residual variant (predicting the
change) beats persistence over the whole 1200 GM/c³ test rollout: 45 % lower error in distribution and 21 % lower
for an unseen α. Numbers, figures and GIFs are in `examples/black_hole_weather/README.md`.
