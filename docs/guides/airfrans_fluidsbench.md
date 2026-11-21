# AirfRANS → FluidsBench: a submission campaign you can run again

`pinneapple_simulation.pipelines.airfrans_benchmark` takes the public AirfRANS release, fits a coordinate network, and
drives the FluidsBench participant workflow until a validated result package exists. It is the same experiment every
time: the dataset and the scoring support are fixed by the benchmark, the model and the budget are recorded, and the
metric values come from the benchmark's own evaluator.

```bash
python -m pinneapple_simulation.pipelines.airfrans_benchmark all spec.json work/airfrans-full
```

```json
{
 "dataset_root": "…/airfrans/Dataset",
 "support_root": "…/support/airfrans",
 "checkout": "…/fluidsbench-submission",
 "store_dir": "…/store",
 "submission_id": "pp-coordinate-net-full-v1",
 "split_id": "full",
 "training": {"train_cases": 640, "epochs": 30, "physics_weight": 0.0}
}
```

## The four stages

| stage | what it does | what it needs |
|---|---|---|
| `prepare` | extracts every case of the release into the feature store, one `.npz` per case, `parallel=4` by default | the pinned AirfRANS/VTK stack (`airfrans==0.1.5.1`, `pyvista==0.48.4`, `vtk==9.6.2`) |
| `train` | fits the network on one split's training cases and writes `model.pt` + `training.json` | a store |
| `predict` | runs it over the official test cases and packs `<entry>/predictions/<case>.npz` | a store, weights |
| `submit` | writes the entry and the package configuration, then runs the benchmark's `validate-entry` → `evaluate-entry` → `package` → `verify-package` → `validate_submission.py`, and finally re-checks the package and recomputes the domain metrics | the three paths, and the participant interpreter for the benchmark's tools |

Every stage is resumable: an existing store, run directory or entry is reused. `store_dir` can point at a shared store,
because the extraction is the expensive stage and every campaign on the same release reads the same arrays.

## What the model is, exactly

Eleven input channels per point: `x/c`, `y/c`, the AirfRANS signed distance divided by the chord and its signed
logarithm, the wall normals, the surface flag, the unit freestream vector (the case's angle of attack) and `log|U∞|`.
Three outputs, scaled by the freestream: `u/U∞`, `v/U∞`, `p/U∞²`. A point-wise MLP of four SiLU layers (256 wide by
default), fitted with AdamW on sampled points — a quarter of every sample comes from the wall and the boundary-layer
band, because that is where the wall-derived metrics are read.

`physics_weight > 0` adds two differentiable residuals: no penetration on the airfoil wall, and Bernoulli's relation in
the outer domain. Measured on the full split, that trade is small and in one direction: field errors up by about 1 %
relative, wall shear down 12 %, `|Cd R²|` down 32 %, `overall_score` 28.49 → 28.55.

## The wall-aware variant, and why it exists

The benchmark never reads a submitted wall shear or a submitted force coefficient: it re-integrates both from the
submitted velocity and pressure, through the point-based velocity gradient at the surface nodes. Measured on the plain
coordinate network, the wall velocity — which must be exactly zero — came out at 3.7 % of `U∞`, the first mesh cell of
the release sits at `y⁺ ≈ 0.2`, the reconstructed viscous drag came out 5.6× the reference value, and 73 % of the
pressure-drag error sat ahead of 10 % chord. No amount of capacity fixes that; the field near the wall is wrong by
construction.

`WallAwareNet` addresses it with three additions:

- **an exact no-slip wall layer**, `U = A(d)·U_outer + (1−A(d))·s·d` with `A(d) = 1 − exp(−(d/λ)²)` and `λ` the
  viscous-sublayer height. The wall value is zero by construction, and the first derivative at the wall *is* the slope
  the evaluator reconstructs. Blending the true field into this layer with `du_t/dd = +τ_w/ν` reproduces the evaluator's
  wall shear to 0.85–1.7 % on two cases;
- **a separate wall network**: the slope is predicted by its own small MLP over the raw point channels, not by a head on
  the shared trunk. Measured three ways on the wall slope — linear probe 74 %, standalone MLP 60 %, head on the trunk
  330 % — the trunk smooths away exactly the near-wall detail the readout needs;
- **a per-case shape descriptor**: the airfoil's upper and lower surfaces on 32 chord bins, 64 channels. A point-wise
  network cannot tell two airfoils apart at the same location, and that ambiguity is worth 75.7 % → 42.1 % on a
  nearest-neighbour probe of the wall slope.

`WallSpec` / `train_wall` fit it with three objectives: the data term, a robust relative squared error on the wall
slope, and the squared error of the drag and lift reconstructed from the predicted curve through
`wall.curve_coefficients` — the evaluator's own polyline quadrature, reproduced to five decimals against
`Simulation.force_coefficient` and asserted in `tests/test_airfrans_benchmark.py`.

Two properties of the *objective* had to be measured rather than assumed, because both look like a capacity problem
and are not:

- **the gradient clip has to be per group.** `clip_grad_norm_` over every parameter measures ~140 at initialisation —
  the force term dominates the trunk — and its 0.007 scale factor landed on the slope head too, training it at 1/140 of
  its learning rate. The head then could not fit its own training cases (356 % relative error on the training set) while
  the same readout trained alone reaches 6 %. `clip_wall_gradients` clips the trunk and the readout against separate
  norms;
- **the relative error has to be flattened on the median, not the RMS.** The wall slope has rms 26 500 and median
  6 500, the top 1 % of nodes carrying 39 % of its energy. A floor on the RMS (or none) turns the term into a
  tail-weighted MSE: the training error falls to 3–13 % while the held-out error *rises* to 970–1 240 % and the
  predicted magnitude runs to 3.6× the truth. Five reductions on one fixed slice, 60-case held-out set: the median floor
  is the only one that keeps improving, ending at 145 % against 970 %.

Neither of those fixes moved the wall shear on the holdout, and neither did removing the force term or detaching the
readout from the data term. What did move it is taking the magnitude *out of the network*: `slope_prior=True` has the
readout predict a log-correction to `flat_plate_slope`, the turbulent flat-plate skin friction `Cf = 0.0576 Re_x^{-1/5}`
at the predicted edge speed. The untrained law already sits at 168 % relative error with a magnitude ratio of 1.08, and
on the 20-case `full` holdout the trained model reaches **38.5 %** wall shear against 43.5 % for the entry without it,
with `Cd R²` moving from −5.03 to −1.88. Two unit traps are worth remembering: the freestream channels travel as a unit
direction and `x_over_chord` is dimensionless, so the law has to be fed metres per second and metres (without the inlet
factor it comes out a thousand times short and training diverges), and standardising the readout on wall nodes sends a
far-field row to a huge z-score, which the exponent then turns into an inf — both the z and the correction are clamped.

**What the wall shear actually measures, found the hard way.** Splitting the budgets — freeze the readout and fit the
field, then freeze the field and refit the readout — gives the best near-wall numbers of the campaign (wall shear
43.5 % → 30.9 %, `Cd R²` −5.03 → −0.97 on the 20-case holdout) but costs field accuracy (velocity 4.31 % → 5.80 %).
The informative part is the pattern: the wall shear is a property of the *pair* (field, readout), not of the readout.
Freezing the readout while the field trains keeps the near-wall layer consistent with the law by construction and is
what buys the gain; refitting the readout afterwards changes almost nothing (30.9 → 29.6), and refitting it on a trunk
that was trained with a *co-adapting* readout makes it much worse (38.5 % → 75.4 %), because the readout then disagrees
with the field it sits in. Chasing the readout's own slope error is therefore the wrong lever, which is worth knowing
before spending another run on it.

On the `full` split this takes `overall_score` from 29.44 to **63.20** (field components 16.09 %/758.6 %/12.15 %/34.41 %
→ 2.93 %/78.7 %/4.49 %/3.66 %, Cl R² 0.988 → 0.9995, profile R² 0.782 → 0.968, `field_score` 0 → 58.03).

**Where the last twelve points came from.** Splitting the budgets — the field trained with the readout held at the law,
then the readout refitted once on that frozen trunk — is what carries the rest: 63.20 → 65.26 when the field gets 28
epochs instead of 14, → 70.07 when the readout is refitted on the frozen trunk, → 72.50 when the refit's loss is tilted
towards the leading edge, → **75.44** when the tilt is the square of the chord distance (`1 + 2(1 − x/c)²`), which
concentrates it on the first percent of chord where the slope the evaluator integrates varies fastest. The tuning of
that readout is closed by measurement: a cubic weight (29.9 % wall shear) and 24 epochs (26.8 %) are both worse than the
quadratic at 16 on the same holdout. The same recipe refitted from scratch in the AoA regime gives **67.24** there (from
55.01), with `Cd R²` +0.702, the best drag of the campaign in any split.

**The residual wall-shear error is a shape error, not a magnitude one** — measured, because it decides whether the next
run should calibrate or should redesign. The per-case distribution of the packaged entries is uniform (on `full`: mean
36.95 %, median 36.45 %, p90 49.2 %, and the ten worst cases carry 8.3 % of the sum; on AoA: mean 48.71 %, median
48.83 %, the ten worst 8.0 %), so it is not a tail. Weighting every case by its own least-squares scale — the best any
calibration could do per case — takes the mean per-case error from 32.67 % to only **28.13 %** on `full` and from
32.80 % to 23.79 % on AoA, and a single constant shared by all cases makes it *worse* than doing nothing (33.05 % and
31.34 %). The magnitude is barely biased on `full` (mean scale 1.025) and drifts case to case (σ 0.20–0.28). So the
traction is roughly the right size and the wrong shape, which is why no scale, no per-case normalisation and no
re-weighting of the readout's loss has moved it: what is left is how the shear is distributed along the surface.
`campaign/diag_wall_scale.py` reproduces the measurement on any checkpoint and case list.

**And the field does not set it either.** Doubling the near-wall sampling (`wall_fraction` 0.25 → 0.50, nothing else
changed, same split, same seed, same two stages) makes the field 2.5× worse — velocity 3.13 % → 7.96 %, `Cd R²`
+0.449 → −0.437 — and the wall shear moves by 1.3 points (22.05 % → 23.38 %). A field that is far worse near the wall
leaves the wall-shear metric almost where it was, so the metric is set by the readout stage rather than by the resolved
field. Together with the scale measurement this closes the two cheap families of ideas at once: it is not calibration,
and it is not near-wall resolution. What remains is a structural change to the readout itself.

**And a chord curve splits the two metrics apart rather than solving both.** The truth's wall slope *is* well
represented by two short curves per case: binning the true slope into 33 cells per surface reproduces it to **12.1 %**
(65 cells: 7.3 %, 16 cells: 17.2 %), so the shape is capturable by a smooth function of chord position. Projecting the
trained readout onto that class — estimated on the wall nodes, evaluated everywhere, linearly interpolated between bin
centres — takes the holdout wall shear from 22.05 % to **78.7 %** while the drag R² goes from +0.449 to **+0.875**. The
curve carries the level the polyline force integral wants and destroys the node-wise structure the evaluator's
least-squares gradient reads, which is the sharpest evidence in this campaign that the two metrics read different
things: `Cd` is a chord-scale integral, the wall shear is a local derivative. `curve_projection` and
`campaign/eval_wall_curve.py` reproduce it.

**Where the field error lives, and the prior that answers it.** Nineteen of twenty points of this campaign's gap to the
first entry of the leaderboard sit in two components: the domain velocity (3.94 % against 0.42 %) and the wall shear
(36.95 % against 1.22 %). Measured against the distance to the surface, the velocity error is not spread out — 47 % of
the release's nodes are within 0.01 chords of the wall and they carry **89.5 % of the error energy** (5.79 % relative L2
there against 1.61 % in the far field), and the far field carries 3.3 % of it. The leader's own `submission.json` says
what it does about that: a *potential-flow (panel-method) prior as an input*. The reason it works is that the outer
problem has a closed form — the edge velocity, the pressure and the far field are inviscid — so a network given that
field only has to learn the viscous correction, which is what the boundary layer is.

`:mod:`.panel` is that prior, written and validated before it is used. It is the constant-strength source plus
constant-strength vortex method (Hess–Smith): one unknown source strength per panel, one vortex strength for the body,
no penetration at every panel midpoint. Its checks are against closed forms, in the repository's test suite:

- the exact circular-cylinder solution to **0.17 % at 60 panels and 0.046 % at 240**, in the same measure the
  repository's `examples/fluidsbench_airfrans` uses, and **identically for either contour orientation** — the solve
  normalises the direction, and a contour walked the other way is the same body;
- Kutta–Joukowski: the lift integrated from the solved pressure is `2Γ/(U·c)` to 0.02 over unity chord;
- on a NACA 0012 the surface pressure peaks at `Cp = −0.41` at zero incidence and `−5.4` at ten degrees, against the
  textbook `−0.45` and `−5.5`, and the no-penetration residual on the collocation points is 3·10⁻⁴ of the freestream
  (the trailing-edge corner, where the potential field is genuinely singular, is the only exception);
- the wall the prior shares with the target is exact: `|v·n|` at the collocation points is 8·10⁻¹³ of the freestream,
  because the circulation enters the *solve* rather than being added to the field afterwards.

Four things about it were measured rather than assumed, and three of them were the intuitive answer's opposite:

- **The circulation must be inside the solve.** Adding the vortex sheet to the sources' field afterwards leaves the wall
  open by an amount of the order of the freestream (measured: eight times it), because the sheet induces a normal
  velocity of its own that nothing cancels. It is a one-line change to the right-hand side and it is the difference
  between a prior and noise.
- **The self-influence has a side.** A collocation point lies exactly on its own panel, where the induced normal
  velocity is ±½ depending on which side the point is approached from. The analytic influence coefficients in this
  module's own frame therefore carry a sign, and hard-coding the diagonal as +½ makes the *solve* and the *evaluation*
  disagree by a whole panel strength. With the diagonal taken from the coefficients — the fluid side, pinned to −0
  before the sign of a rounding residue can decide it — the residual above falls from 4·`U∞` to 8·10⁻¹³.
- **Uniform panels beat panels clustered at the edges.** The cosine spacing that resolves a leading-edge singularity is
  the obvious choice and the worse one: on 125 cases of the release the lift error is **0.035 median uniform against
  0.041 clustered**, and the worst case is **0.25 against 3.45** — the clustered contour shortens its leading-edge
  panels below the release's own node spacing and the solve has less to work with, not more.
- **The contour is the whole problem, and the field is not the symptom when it goes wrong.** Two constructions that
  look right — ordering the cloud by the angle around the chord's midpoint, and ordering each surface by chord
  position — cross the contour over itself on the more distorted sections, and the field they solve is off by five
  orders of magnitude while still being finite, smooth and plausible. What catches it is the algebra: with the wall
  condition imposed and the self-term consistent, `|v·n|` at the collocation points is `8·10⁻¹³` of the freestream.
  Against the release's own coefficients the prior's lift correlates at **0.996** over 167 cases, with a median error of
  0.041 on a range of 2.4, and the 1 000 sections' no-penetration systems are conditioned at 18 (median) and 1.7·10⁴
  (worst) — a plain solve, exactly.

The prior is three extra input channels — the inviscid velocity in units of the freestream and the kinematic pressure in
units of its dynamic head, which is exactly the scaling the targets use — so a network that starts from it predicts a
correction in its own output units. `point_features(..., panel=True)` appends them, `CoordinateNet(..., panel=True)` and
`WallAwareNet(..., panel=True)` take them, and `TrainingSpec.panel` / `WallSpec.panel` carry the flag into the recorded
methodology, the feature cache key and the checkpoint, so a model trained with the prior cannot be loaded without it.
The velocity is clipped at 4.5 times the freestream: the release's own flow never exceeds 4.2 (its lowest pressure is
−8.7 U²), and the discretised corner at the leading edge reaches 28 times it on 0.2 % of the nodes.

**And on this pipeline it does not help — which is the measurement that decides whether the two points of gap above
are worth more work.** Two arms, same split, same seed, same budget (200 cases, Fourier and shape channels, scored on
the 20-case holdout):

| budget | arm | velocity | pressure | p surface | wall shear | `Cd R²` |
|---|---|---|---|---|---|---|
| 16 epochs, 4 096 pts | without | 12.87 % | 10.95 % | 5.55 % | 82.7 % | −8.65 |
| 16 epochs, 4 096 pts | with | 12.45 % | 24.50 % | 22.40 % | 60.9 % | −4.24 |
| 28 + 16 epochs, 6 144 pts | without | 7.66 % | 4.75 % | 5.26 % | 127.3 % | −3.08 |
| 28 + 16 epochs, 6 144 pts | with | 8.68 % | 27.57 % | 11.77 % | 46.1 % | −1.78 |
| 28 epochs, no refit | without | 8.02 % | 4.75 % | 5.26 % | 38.2 % | −1.89 |
| 28 epochs, no refit | with | 9.21 % | 27.57 % | 11.77 % | 46.6 % | −1.44 |

The wall, the drag and the lift improve with the prior; the *field* gets worse, and the pressure worst of all. Dropping
the prior's pressure channel does not fix it (26.0 % against 10.9 % at the short budget), so it is not that one channel.
The reason is measurable and it is the opposite of the leaderboard's situation: **the prior is a worse answer than the
network already gives**. Its absolute error is 4.4 % of the freestream in the far field and 0.72 U inside one percent of
chord, against 1.61 % and 5.79 % for this pipeline's own field — an inviscid field two to three times further from the
truth than the prediction it is offered to. For an entry at 0.42 % velocity it is a starting point; for one at
1.6–5.8 % it is a second thing to learn around. The gap to the leader is therefore not the prior, and closing it means
a field whose error is already below the prior's — which is the architecture question, not the feature question.

The flag is off by default, the four measurements above are the argument for leaving it that way, and `--panel` /
`--no-panel-pressure` in the campaign reproduce either arm.

**A holdout that can predict a score.** Twenty cases did not: on the AoA split the 20-case holdout said 22.5 % wall
shear where the official test measured 48.7 %, and on the scarce split it said 8.02 % velocity where the official said
8.73 % — both optimistic, both in the same direction. `campaign/make_holdout60.py` draws 61 cases from the far half of
the same shuffled pool — so every training slice up to 400 cases stays disjoint from it — stratified on inlet velocity
and incidence, three levels of each. It matches the release's own composition (34 % of its cases above 8° against 31 %,
33 % above 70 m/s against 37 %) and, more importantly, it *corrected a conclusion*: on 20 cases the arm with the panel
prior looked better on the wall shear (46.1 % against 127.3 %); on 61 cases the plain field wins (44.9 % against 62.1 %).
That is what a holdout is for, and it is why every row below is quoted on the larger one.

**Selecting the checkpoint by the score, not by a capped term.** `best.pt` used to follow the validation wall slope,
and the wall shear is past its cap on every entry that works — so the criterion optimises a term that scores zero. On
the 61-case holdout the two criteria do not pick the same epoch, and the first pair measured looks exactly like the
prediction: at the 200-case field stage the slope-selected checkpoint gives 32.2 % wall shear against 44.9 % for the
final epoch, and that 12.7-point gain is worth **nothing** (both past the 20 % cap) while the field it gives up (9.18 %
velocity against 8.57 %, 5.26 % pressure against 4.47 %) is worth **1.5 points**, so the composite proxy prefers the
final epoch by 1.46.

`composite_proxy` is the benchmark's composite on the six terms that can be measured during training, and
`WallSpec.select="composite"` makes `best.pt` follow it. The second pair, at the 340-case readout stage, is the caution
that belongs next to the first: there the slope selects an epoch with a *better* drag (`Cd R²` +0.083 against −0.39) and
a worse wall shear (79.3 % against 70.7 %), and on the holdout that pair scores 39.3 against 38.1 — the slope wins, and
the composite-selected checkpoint merely ties the final epoch (38.08 against 38.05). With twenty validation cases the
`Cd R²` of the proxy is itself noisy, so the composite did not see the epoch the holdout likes. The rule that transfers
is the narrower one, and it is still worth stating: **optimising a term that is already past its cap is negative-sum** —
what the criterion should do is not visible from the validation slope alone, and the composite is a better-motivated
default than the slope, not a demonstrated improvement.

**An integral boundary layer for the wall, and the first measurement of it.** What a flat-plate law cannot express is
why the traction varies along the chord: the pressure gradient thins and thickens the layer and separates it on the aft
surface. `boundary_layer.py` carries the classical answer — Thwaites' laminar closure, Head's turbulent one with
Ludwig–Tillmann's skin friction, Michel's criterion for transition — and it is validated against the forms that precede
it: on a constant edge velocity Thwaites reproduces Blasius to **1.2 %** and Head reproduces the `0.0576 Re_x^{-1/5}`
law the campaign has been using to **6 %**, with the shape factor staying in its physical band (1.40–1.49) instead of
running to its bounds, which is the failure mode that takes the skin friction down by a factor of three.

Driven along the whole chord by the panel solve's edge velocity — geometry and freestream alone, no truth — it reaches
**109.7 %** on the release's node-wise wall shear, worse than the flat-plate law it would replace, while correlating at
0.58 with it per chord bin and a median 30 % error there. The reading is the campaign's oldest finding applied to a new
tool: the node-wise metric is dominated by the leading edge, and that is exactly where an integral method is weakest.
The next form to try is therefore not the full-field prior but a *shape correction* — the law's level multiplied by the
integral method's variation along the chord — which keeps what the law already gets right and adds only the part it
cannot say.

**The shape correction, and what one measurement says about it.** `panel_shape_correction` builds that channel: the
panel solve supplies the edge velocity along the section, the integral boundary layer turns it into `τ(x)`, and the
channel is the integral model's variation over the flat-plate law's at the same edge speed. Three things had to be
measured before it was worth a training run, and the first two were bugs the measurement found.

The panel's edge velocity travels in units of the freestream, and the integral method needs metres per second: fed as it
was, every case ran at a Reynolds number some thirty times too low, which moves transition and thickens the layer.
Separately, the channel's side lookup read the raw per-point block's columns 4:6 — `(n_y, surface)`, not the wall
normals — and the release only fills the normals on the surface nodes themselves, so every off-wall node fell through
to the default and the channel was a constant one over the whole field. Both are fixed; the side now comes from the
node's nearest contour point, which is unambiguous where the channel acts and inert where it does not.

With those fixed, the three measurements: the law's shape error along the chord, at the edge speed the flow actually
has, is **rms log 0.401** on 24 training cases and 13,178 nodes — a typical 40 % bias that varies along the chord. The
integral model's variation correlates with that error at **0.568**, so the direction is right. And one constant fitted
on those cases, `log R = α · log c`, removes **17.7 %** of it at **α = 0.322** (0.401 → 0.330): the model's variation is
about three times too large in amplitude, which is what the gain means. Applied raw, without the gain and without
dividing out the per-surface median, the channel does the opposite of help — 0.296 → 0.924 rms on held-out cases — which
is the integral method's own level leaking into a law whose level was already right.

So the channel now carries only the variation, damped by the fitted constant, and `CORRECTION_GAIN` records where the
constant came from. Whether it earns its place in the network is a question only training answers: 17.7 % of a 40 %
shape error is a small quantity to hand a readout whose own wall-shear error is several times larger, and the honest
expectation is a marginal gain, not the one the wall shear needs.

## What the campaign guarantees, and what it does not

- The package is written and validated by the benchmark's own tooling; the campaign only drives it.
- It checks the result twice: `pinneapple_arena.fluidsbench.validate_package` (digest chain, every relative L2 against its
  sufficient statistics, the aggregate against the macro-average, `overall_score` against its composite) and an
  independent recomputation of the six domain metrics from the same predictions — agreement at 1e-10.
- It never claims a split's result is an extrapolation test unless the model was fitted in that split's own regime: the
  methodology record states which `*_train` key the cases came from.

## Related

- [`fluidsbench_datasets.md`](fluidsbench_datasets.md) — what each of the other open FluidsBench datasets costs, and
  what this machine can carry.
- [`fluidsbench.md`](fluidsbench.md) — the adapter: the contract, the support chain, the reductions, the trust decision.
- [`examples/fluidsbench_airfrans`](../../examples/fluidsbench_airfrans/README.md) — the same path rehearsed on an exact
  reference (potential flow around a cylinder), with no dataset download.
- `pinneapple-engineering-api` exposes this pipeline as the `engineering.fluidsbench_campaign` job capability
  (`POST /v1/fluidsbench/jobs`).
- `pinneapple_simulation/pipelines/airfrans_benchmark/panel.py` — the inviscid prior, and the four measurements above.
- `pinneapple_simulation/pipelines/airfrans_benchmark/boundary_layer.py` — the integral method, and what it does to the
  wall shear on the release.
