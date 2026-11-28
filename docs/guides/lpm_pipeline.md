# Large Physics Model pipelines: one model per design space

An LPM is *one trained model for a whole design space*. `pinneapple_neural.lpm` holds the model; this pipeline turns a
parametric family of bodies into the dataset it is trained on, fits it, and reports what the fit measured.
`pinneapple_neural.lpm_pipeline` is the pipeline, `pinneapple_neural.lpm_families` the five families that ship with it.

```python
from pinneapple_neural.lpm_pipeline import run, families

summary = run(families()["pp_fin"], "out/pp_fin", n_geometries=80, conditions_per_geometry=3, epochs=400)
print(summary["held_out_relative_l2"], summary["ood_probes_flagged"])
```

or from the command line:

```bash
python -m pinneapple_neural.lpm_pipeline pp_nozzle --geometries 80 --conditions 3 --epochs 400
python -m pinneapple_neural.lpm_pipeline pp_nozzle --dataset-only --geometries 2000   # data, no fit
```

## The five stages

| Stage | What it does | Where it is written |
|---|---|---|
| draw | samples a geometry, then `conditions` operating points for that same body | in memory (`DesignSpec`) |
| build | surface point cloud + normals (the geometry the model sees), normalised conditioning vector, target field | in memory (`lpm.Design`) |
| write | one `.npz` per design, plus a manifest with the reference, the bounds and every SHA-256 | `dataset/` |
| fit | split **by geometry**, batched design-space training, held-out measurement | `model/model.pt` |
| card | what this run measured and what it did not | `model/card.json`, `summary.json` |

## The five families

| Name | Vertical | Field | Reference |
|---|---|---|---|
| `pp_fin` | electronics cooling: extruded-fin heat sink | fin and base temperature (°C) | 1-D fin theory on the `plate_fin_heatsink` resistance network (Incropera; Elenbaas; Teertstra) |
| `pp_airfoil` | UAV, wind and compressor sections | surface Cp | Joukowski conformal map with the Kutta condition — exact potential flow (Milne-Thomson; Anderson) |
| `pp_nozzle` | propulsion: converging-diverging nozzle | wall Mach and p/p_chamber | quasi-1-D isentropic flow (Sutton & Biblarz; NACA 1135) |
| `pp_exchanger` | process and HVAC: shell-and-tube exchanger | dimensionless shell and tube temperatures | Kern shell side, Gnielinski tube side, ε-NTU from the 1-2 energy equations |
| `pp_bracket` | structural: L-bracket under a tip load | displacement (m) and von Mises (Pa) | immersed finite-element elasticity (`pinneapple_simulation.immersed`), solver-backed |

Each family declares its `reference` — the model behind the target, its citation and the assumptions it does **not**
include — and every card repeats it. A held-out number from this pipeline is therefore a statement about the fit to
that reference. It is not a validation of the model for a real part, and the card says so in `trust.not_measured`.

Two of the five predict dimensional fields (the heat sink in °C, the bracket in m and Pa); the other three predict
dimensionless ones (Cp, Mach and p/p0, θ) because that is how their physics scales. `pp_exchanger` reconstructs the
physical temperatures as `T_cold_in + θ (T_hot_in − T_cold_in)`, and `pp_nozzle` as `(p/p0) × p_chamber`.

## Fitting data this repository did not produce

The five families above compute their own targets, so a good fit proves the *model* can learn a physics this
repository implements — not that it can learn physics as it is actually solved. `pinneapple_neural.lpm_real_data`
closes that gap with a **published campaign**: `pp_airfoil_rans` fits the `airfoil_rans_cst_re4e6_fields` dataset, 436
steady 2-D RANS runs (OpenFOAM `simpleFoam`, k-omega SST, y+ < 1) over 100 Latin-hypercube CST shapes at four
incidences plus four NACA sections at nine, on a shared 228 x 88 O-grid, cc-by-4.0.

Nothing is drawn here: `specs()` returns the campaign's runs exactly as recorded, and `published_split()` hands back
the 15 % of shapes its authors held out before publishing, so a held-out number means what it means in the campaign.
The geometry is the **wall ring** of the O-grid, the field is surface Cp from the solved kinematic pressure, and the
condition is the incidence.

```python
from pinneapple_neural.lpm_real_data import data_gates
print(data_gates())                              # what passed the gates and what was refused
python -m pinneapple_neural.lpm_pipeline pp_airfoil_rans --dataset-only --out out/rans
```

Three gates stand between the data and the fit, and a run that fails one is not written:

| Gate | What it catches |
|---|---|
| the run's own status | the campaign flags 32 runs unsteady; a final-iteration snapshot of an oscillation is not a steady answer |
| the wall pressure must integrate to the campaign's published Cl | the ring, the normals or the sign was read wrong — this is the only check of the extraction itself |
| surface Cp inside what a RANS section can produce | a mis-indexed grid, before it is trained on |

The campaign's own verification — grid study on NACA 2412 (Cl 0.6614/0.6625/0.6624 at 6840/20064/45144 cells, Cd GCI
1.9 %), XFOIL 6.99 (Cl −2.3 %, Cd +4.5 %), a 0.110/deg lift slope against thin-airfoil theory, and NASA TMR NACA 0012
SST (Cd 0.00812 vs 0.0081) — is quoted in the reference and repeated in every card. It is the campaign's, not this
pipeline's. What this family still does not do: it does not validate the RANS itself, and it does not turn a 2-D
section into a 3-D wing.

### What it measured

372 runs pass the gates (100 shapes, 314 train / 58 test by the campaign's own split, no refusal), and the wall
pressure integrates to the campaign's published Cl to a median of 0.0033 in coefficient, worst 0.0159 — the residual
is the viscous part of the reported coefficient, which a pressure integral does not contain.

| Run | Held-out relative L2 | p90 | Cl from the predicted Cp |
|---|---|---|---|
| 96 wide, 150 epochs, 32 designs/batch | 0.216 | 0.321 | 9.3 % median, r = 0.979 |
| 128 wide, 700 epochs, 48 designs/batch, 48-code, 8 frequencies, 3 attention layers | **0.057** | 0.125 | **2.8 % median, p90 6.8 %, r = 0.998** |

The constant-field baseline on the same 58 designs is 0.959, so the model explains what a mean field cannot. The
second row is `train(dataset, out, epochs=700, batch_designs=48, width=128, code=48, n_freq=8, n_heads=6,
attn_layers=3, lr=1.5e-3)`; it takes 19 minutes on one MPS device. The first row is the same data at the defaults,
and it is a useful comparison: on this space the limit was training budget, not the data — the error fell by 3.8x
without a single extra solver run. Pointwise Cp error is now flat along the chord (0.05 of the local rms from the
leading edge to 70 % chord); what remains sits near the trailing edge, where Cp is small and the relative error is
large while the absolute error is not.

## A corpus nobody here solved: predicting on SHIFT

The RANS campaign is published by an academic group and this pipeline reads it; Luminary's **SHIFT** datasets are
published by a company that trains its own LPMs, and they are used here for the opposite purpose. Not one design read
from them may enter a fit: the moment a model has seen those bodies, what they say about them is not outside
anymore. `pinneapple_neural.lpm_stages.shift` reads them off disk for one job — asking a model fitted here what it
says about geometry it never saw, and scoring the answer against a solver that owed us nothing.

Fifteen repositories live under `luminary-shift`; thirteen are `cc-by-nc-4.0` and two (`Submarine`, `CCA`) are
`license:other`; every full repository is `gated: manual` and every `-sample` twin is `gated: auto`. Nothing is
downloaded here: the data is on disk after somebody accepted the terms, and the module's job starts there.

| Fact | Value |
|---|---|
| the case directory | the one holding `merged_surfaces.vtp` — seven datasets name it six ways (`sample_NNNNNN` at the root or under a version directory, `RUN_N`, a case hash, `yaris_xNNNNN`, `variant_NNNN`) |
| the surface file | `merged_surfaces.vtp`, 231 MB for one WING case, read with `read_surface` so only the requested arrays are decoded |
| the field | `CpMeanTrim`, else `Cp`, else the error lists the arrays the file carries |
| the conditions | `params.json`, read whole and flattened (`velocity.u`), with a `condition_map` saying which key feeds which family parameter |
| the licence | travels with the number: `require_validation_usage` accepts `validation-only` and nothing else, and a card built on it carries the repository and the URL |

```python
from pinneapple_neural.lpm_stages import shift
cases = shift.samples("~/data/luminary-shift/Wing-sample", limit=2)     # walks for merged_surfaces.vtp
out = shift.report(model, family, shift.designs("...", family, {"alpha_deg": "alpha_deg"}, limit=2),
                   train_designs=train, dataset_name="Wing-sample")
out["outside"]["licence"], out["held_out"]["relative_l2"]
```

A case is subsampled to a deterministic, evenly spaced subset (2 048 points by default) because the model's
neighbourhood features are a distance matrix from the query to the body: a raw 200 000-node surface is a
200 000² matrix. The subset size goes into the design's label, so a report says which one produced the number.

`python -m benchmarks.lpm_compare` puts the answer beside the constant field, on the same gates, for a run of this
campaign or for a SHIFT tree:

```
corpus: pp_nozzle held-out split (400 designs, split by geometry)
model                   held-out     p90   worst  baseline  beaten  gates
lpm                        0.044   0.063   0.180     0.666       0  acc FAIL / base ok
tolerance: 0.02 (the family's own)
```

That is the whole reading of the campaign in one line: the model beats the mean field on all 400 held-out designs and
misses the tolerance its own family declared (0.02). The `held-out` column is the **global** relative L2 — the number
the cards report — while `p90` and `worst` are the per-design spread, and the two are not interchangeable: on the
airfoil family the mean of the per-design errors is 1.31 where the global number is 0.71, both measured on the same
400 designs. PhysicsNeMo's DoMINO, GeoTransolver and X-MeshGraphNet appear in the same run as **not compared**, with
the reason: their inputs are a signed-distance grid and a mesh bundle built from an Ashton case, not a point-cloud
design, and `pinneapple_data.ashton.physicsnemo_models` measures them there.

## Three dimensions

The five families above are 2-D: a section, a meridian, a plate. `pinneapple_neural.lpm_families_3d` keeps the same
pipeline and answers the same five problems in three dimensions, where an industrial surrogate is actually asked for
one.

| Name | Problem | Field | Reference |
|---|---|---|---|
| `pp_wing3d` | swept, tapered, twisted wing at a compressible free stream | surface Cp (960 points) | 3-D vortex lattice on the camber surface, strip-coupled to the 2-D Hess-Smith section solution, Prandtl-Glauert |
| `pp_fin3d` | extruded-fin heat sink under a chip of known power | fin and base temperature (950 points) | 3-D conduction with a Robin boundary, immersed finite-cell method |
| `pp_bracket3d` | L-bracket under a tip load | displacement and von Mises (192 points) | 3-D linear elasticity on conforming quadratic tetrahedra (gmsh P2) |
| `pp_nozzle3d` | scarfed (vectoring) nozzle | wall Mach and p/p0 (128 points) | sector-wise quasi-1-D isentropic expansion on a surface of revolution |
| `pp_exchanger3d` | tube sheet of a shell-and-tube exchanger | temperature (194 points) | 3-D conduction with a Robin condition on every surface, immersed finite-cell method |

```python
from pinneapple_neural.lpm_families_3d import families_3d
from pinneapple_neural.lpm_pipeline import run

summary = run(families_3d()["pp_wing3d"], "out/pp_wing3d", n_geometries=80, conditions_per_geometry=3)
```

Each is checked against a property of the physics rather than against itself: the wing's composed pressure must
integrate, section by section, to the load the lattice asked for, its lift must obey Prandtl's lifting-line law as the
aspect ratio grows, a symmetric section at zero incidence must carry no load, and Cp at Mach must be Cp at zero Mach
divided by β. The heat sink must balance its energy and run cooler when the convection coefficient rises. The bracket
must be linear in the load and unchanged when the mesh is refined.

The nozzle and the exchanger name the assumption that carries them, because it is the one a reader has to weigh.
`pp_nozzle3d` expands each meridional sector as its own quasi-1-D streamtube: the sectors do not exchange mass,
momentum or heat. The real three-dimensional flow does, and the non-uniformity that communication leaves behind is
exactly the loss a scarf produces — so the family's exit Mach is the sector-wise answer, not the flow's.
`pp_exchanger3d` takes both fluids as boundary conditions (film coefficients and inlets) and solves only the metal: it
is the temperature a thermal-stress analysis would take as input, not the stress.

One thing none of them claims: there is no 3-D compressible Navier-Stokes here. `pp_wing3d` is the linearised
compressible lifting surface, so a transonic case is a subsonic field stretched by 1/β and a shock is not in it.

A note on the bracket, because it is the kind of thing a card cannot say for itself: the immersed finite-cell grid was
the first attempt, the one the heat sink uses, and it was rejected here. On a thin plate in bending at order one,
halving the grid moved the tip displacement by 55 %. Conforming P2 tetrahedra on the same part move by 0.9 % and then
0.3 %; that is the path the repository already verifies against CalculiX, and it is the one the family uses.

## Running the campaign

`benchmarks/lpm_campaign.py` runs every design space end to end: one dataset, one fit, one card, in the order that
leaves usable models if the night gets cut short.

```bash
./benchmarks/run_lpm_campaign.sh                       # plan, then start it detached, output in ~/lpm-campaign
./benchmarks/run_lpm_campaign.sh --list                # the menu: what each design space predicts and how big it is
./benchmarks/run_lpm_campaign.sh --status              # what already exists, and what each model scored
./benchmarks/run_lpm_campaign.sh --plan                # the plan and the cost estimate, nothing starts
./benchmarks/run_lpm_campaign.sh --stage dataset       # the data only
./benchmarks/run_lpm_campaign.sh --stage train         # the fits only
./benchmarks/run_lpm_campaign.sh --families pp_wing3d --geometries 800 --epochs 900   # one family, bigger
```

The sizes are **data**, in `benchmarks/lpm_recipes.json`: `full` is the campaign, `smoke` is the wiring check. Change
the file to change the campaign, or override it for the families of one run with `--geometries`, `--conditions`,
`--epochs` and `--width` — which is what you want when a family turns out to be the one worth scaling.

`--status` is the one to run before and after a night: it reads each `manifest.json` and `card.json` under `--out` and
prints, per family, the designs written, the held-out error, the constant-field baseline and the OOD probes. A model at
or above its baseline has not learnt the design space, and no amount of watching it train changes that.

or the module directly, if you would rather keep the log where you choose:

```bash
python -m benchmarks.lpm_campaign --plan                     # what would run, and the estimate
python -m benchmarks.lpm_campaign --scale smoke              # the whole wiring, about four minutes
python -m benchmarks.lpm_campaign --detach ~/lpm-campaign.log --out ~/lpm-data
```

It **resumes**: a dataset whose manifest is there is not regenerated and a model whose card is there is not refitted,
so a run that dies at the seventh family restarts at the seventh family. `--force` is the only way to change that. It
also writes `failures.json` with the traceback of any family that raised and carries on with the rest, and exits 1 at
the end so a scheduler sees it. `--detach` re-runs the command with `os.setsid()`, which is what actually protects a
long fit: `nohup` only ignores SIGHUP, and what kills it is the process group going away.

The full recipe, as estimated from the measured cost of one reference evaluation and of the 3-D wing's fit:

| Family | Bodies × conditions | Dataset | Fit | Note |
|---|---|---|---|---|
| `pp_fin`, `pp_airfoil`, `pp_nozzle`, `pp_exchanger` | 400–600 × 3-4 | under a minute | 3.6–5.4 h each | closed-form or cheap FEM references |
| `pp_bracket` | 200 × 2 | 24 s | 1.2 h | immersed FEM, 2-D |
| `pp_airfoil_rans` | the campaign's 372 runs | 19 s (a 500 MB download the first time) | 1.3 h | no design space of ours: its authors' |
| `pp_wing3d` | 400 × 3 | 72 s | 3.6 h | a lattice and a section solve per design |
| `pp_fin3d` | 300 × 2 | 16 min | 1.8 h | 3-D immersed heat, ~1.6 s per design |
| `pp_nozzle3d` | 400 × 4 | 32 s | 3.6 h | closed form per sector, milliseconds |
| `pp_exchanger3d` | 250 × 2 | 6 min | 1.5 h | 3-D immersed heat, ~0.7 s per design |
| `pp_bracket3d` | 150 × 1 | 38 min | 0.5 h | P2 tet mesh and direct solve, ~15 s per design |

About 28 hours in all on one machine, and the two 3-D solids are half of the wall clock even though they are the two
with the fewest designs. Put `--out` outside the repository: the datasets are the bulk of it.

What to check first when the campaign lands: `campaign.json` holds the held-out error, the constant-field baseline and
the OOD probe count per family. A family whose held-out error is close to its baseline has not learnt the design space,
and the fix is more bodies, not more epochs — the 2-D families needed 60 to 240 bodies to reach 0.06-0.17 and the 3-D
wing is at 0.51 with 24, which is a smoke test, not a result.

## One design space at a time

The campaign runs its families **in sequence, in one process, on one device**: nothing is parallelised, because the
machine that runs a campaign is usually the machine somebody is working on. A lock (`<out>/.campaign.lock`, holding the
pid) makes that true across processes too — a second campaign pointed at the same `--out` refuses to start rather than
refitting what the first is fitting.

Two flags exist for a machine that has to stay usable:

```bash
./benchmarks/run_lpm_campaign.sh --threads 4    # cap the torch/BLAS pools (default: half the cores; 0 leaves them)
./benchmarks/run_lpm_campaign.sh --max-hours 6  # finish the family in flight, then stop; re-run to go on
```

`--max-hours` stops **between** families, never inside one: a dataset that is half written has no manifest and would be
thrown away, which is exactly the work a night is meant to keep. A stopped run writes what it has to `campaign.json`,
says so in `runner.json`, and exits 0.

## When it dies

A long campaign on a desk machine gets killed — a lid closed at the wrong moment, a memory spike, something the
unified log does not record. Two things make that cost minutes instead of a family.

**The fit checkpoints.** While `train(..., resume=True)` runs — what the campaign always passes — `fit` rewrites
`model/fit.pt` every fifteen minutes and after the last epoch, with the weights, the optimizer and scheduler state and
the loss history. The next start continues from the epoch after the checkpoint instead of from one, so the cosine
schedule and the Adam moments are the ones the run had. The file is deleted when the fit finishes, and a checkpoint
whose recipe (designs, epochs, batch, learning rate, seed) does not match the call is ignored and replaced, so a
`--force` with a different recipe cannot inherit an old run's history. A half-written dataset is a different case: a
dataset without a manifest is still thrown away, which is why `--max-hours` stops between families.

**Something restarts it.** Three launchd jobs, each of which can be submitted from a shell without a plist:

```bash
launchctl submit -l com.pinneapple.lpm -- /bin/bash -lc \
  "cd ~/Documents/GitHub/lpm-five && exec python3 -m benchmarks.lpm_campaign --out ~/lpm-campaign"
launchctl submit -l com.pinneapple.lpm-watch -- /bin/bash ~/lpm-campaign-watch.sh   # restarts the campaign, then the 900-body airfoil refit
launchctl submit -l com.pinneapple.lpm-guard -- /bin/bash ~/lpm-campaign-guard.sh   # restarts the two above
```

`launchctl list | grep pinneapple` is the status. The guard exits once `~/lpm-campaign/.finished` exists, which the
watcher writes after the last refit. The guard checks for a running process before submitting, because a `launchctl
submit` job leaves the table when its process exits but a process can also outlive its entry, and a second campaign
against the same `--out` would only fight over the lock.

## Following it on one screen

A fit is four hours with nothing on stdout, so a screen that read `card.json` would show every family as *pending*
until the moment it was done. The pipeline writes progress instead — `progress.json` in `dataset/` and in `model/`, at
the interval `--progress-every` asks for (five seconds by default), with the designs or epochs finished, the rate and
the time still needed. `benchmarks/lpm_monitor.py` reads those files plus `runner.json`, `campaign.json` and
`failures.json`, and serves them:

```bash
python -m benchmarks.lpm_monitor --out ~/lpm-campaign            # http://127.0.0.1:8130/
python -m benchmarks.lpm_monitor --out ~/lpm-campaign --state    # the same state as JSON, and exit
python -m benchmarks.lpm_monitor --out ~/lpm-campaign --detach ~/lpm-campaign/panel.log   # outlives the shell
./benchmarks/run_lpm_campaign.sh --watch                         # start the campaign and open the panel
```

One row per design space: what it predicts and against which reference, the stage it is in (draw → build → write →
fit → card), the progress of that stage with its ETA, the held-out error and the constant-field baseline once the card
exists, and when it was written. A family whose held-out error is at or above its baseline is marked as a fit that did
not learn the space — not as a run that failed, because it ran.

Two things the panel deliberately will not do. It does not import the families to ask how a run is going: it reads
files, so what it shows is what a run that died left behind. And it does not draw a progress file older than three
minutes on a running stage as live — it marks it stale, because a fit that stopped writing is a fit that stopped.

### The figures and the 3-D viewer

Clicking a finished design space opens its figures. They are drawn by `benchmarks/lpm_figures.py` from
`model/view.json` — the file the fit writes next to the card — using this repository's own plotting
(`pinneapple_tools.visualization`), so a figure the campaign shows and a figure a report carries come from one theme.

| Figure | What it answers |
|---|---|
| `loss` | did the fit converge, or did it stop early with the curve still falling? log scale over the sampled epochs |
| `design` | where the designs are in the design space, on its two widest parameters, with the held-out bodies in a second colour — the split is by geometry, so a separated dot is a body the model never saw |
| `geometry` | the bodies behind the samples, drawn from the point cloud the model is given, one panel per sample |
| `prediction` | the reference (solid) against the model (dashed) along the ordered query points, one row per field and one column per body, each with its own held-out relative L2 |
| `viewer` | the **Studio's own 3-D page** (`pinneapple_tools.visualization.studio.web.web_viewer`, `single_file`), served in the panel above the figures: drag to orbit, scroll to zoom, switch between the fields on its top bar |

The viewer needs a skin, and the families hand the model a point cloud, so `lpm_figures.scene` triangulates each body
in its own best-fit plane. That is the least inventive surface the points support — it makes a plate, a wing or a
bracket the thin body it is, and it fills a concave silhouette, so a nozzle's interior reads as solid. Its caption says
"the point cloud, triangulated in the best-fit plane" rather than "the body". The samples stand side by side: they are
four different bodies of the same family and, drawn at their own coordinates, they would sit on top of each other. The
field colours the skin only when the field's values were computed on the body's own points (`view.json` records that as
`at_points`), and a view written before the flag existed is decided from the data instead of guessed.

The samples are chosen, not cherry-picked: the best, the median and the worst held-out body by relative L2, plus one
training body for contrast. A picture of one held-out body is a picture of that body — it is not a statement about the
design space, and the caption says so.

A family fitted before `view.json` existed has no figures; the panel says that instead of drawing an empty chart. An
old figure is never served against a newer fit: the PNG is rebuilt when the view is newer than it.

## Three rules the pipeline enforces

**The split is by geometry, never by design.** A design space runs several operating points on the same body. Splitting
by design would let the model see a body during training and be scored on that same body at another operating point,
which reads as generalisation without being it. `_split_by_geometry` holds out whole bodies.

**The point cloud is the geometry, and everything else is conditioned.** A taller fin, a thicker section or a wider
nozzle changes the points, so the geometry code sees it. Parameters the outline cannot carry — a tube count, a
shell-to-length ratio — go into `geometry_features` and join the conditioning vector next to the operating point. A
family that leaves out a parameter its field depends on gives the model two bodies it cannot tell apart, and the fit
pays for it: `pp_exchanger` declares three such features for exactly that reason.

**A design the reference cannot represent is refused, not written.** `FieldFamily.validate` returns the reason;
`draw` re-draws the operating point and `build` raises rather than writing a number the reference never justified
(a forced-convection draw at 1e-6 m/s is numerically finite and physically meaningless).

Field scaling (`field_scale`) brings every field to order one before the loss sees it, so a family predicting a
displacement in metres and a stress in pascal does not train on the stress alone. Errors are always reported in the
family's physical units, with the scale undone.

## What a run measures

`summary.json` and `model/card.json` carry, for that run only:

- `held_out.relative_l2` and `relative_l2_p90` on bodies held out by geometry;
- `held_out.constant_field_baseline_relative_l2` — predicting the training-average field, the bar any surrogate must
  clear;
- `ensemble_spread` — the spread between bagged heads, **not** a confidence level;
- `ood` — the library's geometry-code rule on the family's probes outside the family and on the held-out designs,
  reported as measured: a probe the model does not flag is not thereby inside its validity domain;
- `trust.decision` — always `not_verified_until_independent_review`, with the list of what was not measured.

A family also declares `tolerance`, the held-out relative L2 its own gate asks for. The gate is reported with the
measured value, never used to hide it.

## Adding a family

Subclass `FieldFamily` and implement two methods:

```python
from pinneapple_neural.lpm_pipeline import DesignSpec, FieldFamily, Reference

class MyFamily(FieldFamily):
    name, title, domain = "pp_mine", "...", "..."
    field_names, field_units = ("field_a",), ("Pa",)
    geometry_params = {"length_m": (0.1, 1.0)}          # how the body is drawn
    operating_params = {"load_n": (1e3, 1e4)}           # how it is run
    characteristic_length_m = 1.0                        # the geometry is divided by this
    reference = Reference("...", "...citation...", ("assumption it does not include",))

    def points_and_normals(self, spec: DesignSpec):      # (P, dim) points and outward normals, metres
        ...
    def field(self, spec: DesignSpec):                   # (P, len(field_names)), from the reference
        ...
```

Then add it to `families()`. If the reference has a validity range, override `validate`; if the body has geometry the
outline cannot show, declare `geometry_features` and `geometry_feature_values`. The tests in
`tests/test_lpm_pipeline.py` are the template: every family is checked against an identity that does not come from the
family itself (lift against ρUΓ, the area-Mach round trip, ε-NTU against its closed form, the fin equation, linearity
and mesh convergence of the FEM).

## Limits

- The five references are screening models: potential flow has no boundary layer, quasi-1-D flow has no shocks or
  separation, fin theory is 1-D, the exchanger holds UA constant with fixed water properties, and the bracket is
  linear 2-D elasticity. Each family lists its own assumptions.
- The models are only as good as their data. A design space of a few dozen bodies is a smoke test; the held-out error
  falls with the number of bodies, which is what generating the full datasets is for.
- Nothing here validates a model against a solver run or a test. That is an independent review, and the card's
  `trust.not_measured` says so.
