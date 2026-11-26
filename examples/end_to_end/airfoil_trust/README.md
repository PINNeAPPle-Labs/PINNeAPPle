# Trust beyond accuracy

A complete run of the PINNeAPPle verification pipeline on one problem, built so that every claim it
makes can be traced to a number a script produced. The surrogate is the vehicle, not the subject: an
accurate model is the starting point, and the study is about the narrower question a decision
actually needs -- *where, and for what, is this prediction supported?*

The problem is a NACA four-digit family in steady incompressible flow: angle of attack -4 to 8
degrees, freestream speed 10 to 30 m/s, thickness 8 to 16 % of chord, unit chord, sea-level air,
`simpleFoam` with k-omega SST and wall functions. Camber is zero for the first model and is opened
later as a *new parameter dimension*, with its own campaign, because that is what opening a
dimension costs.

Every number on this page was measured by these scripts; the JSON in `results/` holds them and
`results/dossier.md` is the report they add up to. What was not measured is said, at the end.

The headline is the one this example exists for: **lift comes back with a relative L2 error of 4.84
% over the sixteen designs held out by design, which is inside the 5 % tolerance the study declared
before it looked -- and the dossier still comes back FAILED**, because 4 of the sixteen are above
that tolerance on their own, because the field the scalar is integrated from is 14.8 % away, because
the uncertainty band covers 43 % of the errors it is supposed to bound, and because at the centre of
its own design box the model has no evidence behind it. The average passes. The claim does not
survive being looked at point by point.

## Run it

```bash
python examples/end_to_end/airfoil_trust/run_all.py                  # all seven stages, resumable
python examples/end_to_end/airfoil_trust/01_reference_campaign.py mesh doe1 --workers 5 python
examples/end_to_end/airfoil_trust/02_analysis.py              # grid convergence, POD, the first
surrogate python examples/end_to_end/airfoil_trust/03_physics_validation.py    # the model gates,
and input sensitivity python examples/end_to_end/airfoil_trust/04_validated_domain.py      # where
the evidence supports a prediction python examples/end_to_end/airfoil_trust/05_camber_round.py
--workers 5   # 68 cambered designs python examples/end_to_end/airfoil_trust/06_revalidation.py
# the second model, and evidence arriving late python examples/end_to_end/airfoil_trust/07_report.py
# the dossier, the figures, experiment.json
```

Environment used: OpenFOAM v2606 (ESI) native on macOS, gmsh 4.15, Python 3.11 with numpy, torch and
matplotlib, 10 cores. The two campaigns are the expensive stages (64 and 68 solves); everything
after them is minutes. Records, fields and cases go to `_runs/` and are not committed.

## The film

`make_video.py` renders the two cuts written in `VIDEO_SCRIPT.md` out of the same files the report is built from: the
frames are drawn from `results/`, and the narration is synthesised locally with macOS `say`, so no cut can say
anything the study did not measure. Each frame is regenerated on every run, and the per-scene narrations are written
next to the output as `scene01.wav` ... so a human read can replace the synthetic voice without touching the script.

```bash
python examples/end_to_end/airfoil_trust/make_video.py --cut long     # 9 min 54 s, 11 scenes and a closing card
python examples/end_to_end/airfoil_trust/make_video.py --cut short    # 90 s, for a feed
```

The MP4 is not committed here: twenty megabytes of video in a git history is not a review artefact. The script, the
figures and the numbers that go into it are.

## The case

| | |
|---|---|
| geometry | NACA four-digit, thickness 8-16 % of chord, camber 0 (round 1) and 0.003-0.02 (round 2), 0.4 chord camber position |
| domain | 12 chords around the section, one cell thick in z (0.01 chord), `empty` front and back |
| flow | steady incompressible RANS, k-omega SST with wall functions, nu 1.5e-5 m2/s, rho 1.225 kg/m3 |
| Reynolds | 0.67e6 to 2.0e6 over the design box |
| solver | `simpleFoam`, 1000 iterations, GAMG pressure with a Gauss-Seidel smoother, `smoothSolver` for the rest, force coefficients averaged over the second half |
| mesh | quad-dominant plane extruded to hexahedra, 8-layer boundary layer, first cell 0.23 % chord |

The solution is sampled on a **fixed lattice**: 41 chordwise stations, each dragged out along the
surface normal at 7 fixed offsets, plus the wall point -- 656 samples per design, upper then lower
surface. That is what makes a field surrogate across geometries possible at all: every design has
the same samples, in the same order, and each one is a material point of *that* section.

The lattice is also where the first real bug of this study lived, and it is worth naming because
nothing else would have caught it: with the wall normals pointing the wrong way on one surface, 427
of the 656 samples landed inside the solid, OpenFOAM returned `-1e300` for two thirds of the probes,
and every quantity of interest -- force coefficients, lift, drag -- stayed perfectly healthy. A
solver reports the points it cannot find as a number like any other. The run now rejects a probe
that comes back above 1e100, and a ray-casting test pins the sampling to the geometry.

## Results

### 1. The reference is a measurement, not the truth

Four meshes, every spacing -- including the boundary layer -- scaled by the same factor 1.45:

| mesh | cells | CL | CD | avg non-orthogonality | max skewness | checkMesh | time |
|---|---|---|---|---|---|---|---|
| L0 | 6 427 | 0.4522 | 0.00550 | 6.8 | 1.35 | OK | 64 s |
| L1 | 8 603 | 0.4431 | 0.01056 | 6.6 | 1.40 | OK | 90 s |
| L2 (campaign) | 11 540 | 0.4467 | 0.01108 | 6.7 | 1.29 | OK | 232 s |
| L3 | 16 117 | 0.4423 | 0.01204 | 6.7 | 1.77 | OK | not timed |

The refinement ratio is the one the case declares -- the first cell height, set a factor 1.45 apart
between consecutive levels -- and not one inferred from cell counts: the family is one cell thick in
z, so neither
`cells**(-1/3)` nor `cells**(-1/2)` is its refinement ratio, and both would misreport it.

**Lift** moves by 1.0 % between the two finest meshes, and it has moved in both directions as the mesh was
refined. The family is converged in the only sense that matters for this study -- the remaining
movement is smaller than the tolerance -- but it oscillates at the level of its own discretisation
error, so no observed order and no grid-convergence index is claimed for CL, and the study reports
the change between the two finest meshes instead.

**Drag** converges monotonically across all four levels, and the two triplets disagree about how: observed orders of
6.06 for the coarse triplet and 1.69 for the fine one, an asymptotic ratio of 0.32 in the fine
triplet -- so the family is *not* in the asymptotic range on drag. The finest-triplet grid-
convergence index is 11.8 %, and it is an indicator of the reference's own noise rather than a
converged uncertainty. Either number is larger than the difference the surrogate could be judged on,
and the consequence is recorded rather than smoothed over: **drag is a diagnostic here, not a
validated quantity**, and the drag comparison in the dossier is made against that index instead of
against the 5 % lift tolerance, because a tighter bar would compare the model with the reference's
own noise.

Convergence is recorded, not assumed: on the production mesh the pressure residual reaches 1.9e-08
and the global continuity error is 5.0e-12. The force coefficients are averaged over the second half
of the iterations only, and a run whose average has not settled is rejected rather than recorded.

### 2. The surrogate

Sixty designs from a maximin Latin hypercube, split by whole designs: 32 train, 12 calibration, 16
held out. Fields are reduced with 8 POD modes taken from the training designs only (98.87 % of the
field energy, 4.65 % held-out reconstruction from the basis alone), and a five-member MLP ensemble
predicts the quantities of interest and the POD coefficients together.

| | CL | CD |
|---|---|---|
| RMSE on the 16 held-out designs | 0.0213 | 0.00093 |
| relative L2 | 4.84 % | 8.13 % |

Field reconstruction, relative L2 over u, v and cp: 14.8 % on average and 69.4 % in the worst case.
Where it is wrong is not uniform: the error lives in the first 15 % of the chord, where the
predicted suction at the nose overshoots the reference by up to 0.85 in Cp; past x/c = 0.2 it falls
to about 0.2 and keeps its sign, so the average hides a model that is close almost everywhere and
wrong where the flow turns.

### 3. Accuracy is a number, and the number has a shape

4.84 % on lift is inside the declared tolerance, so the accuracy check passes. That is exactly the
point at which a normal report stops, and why this one does not:

- The average is over sixteen designs, and 4 of those sixteen are above the same 5 % bar on their own. The worst is
  va008 at 42.9 % -- the ensemble flagged it too, with the largest spread of the sixteen, which is the one
  place where the disagreement bar did the job it is hired for.
- The quantity of interest is an integral of a field, and the field is much worse than the integral: 14.8 %
  relative L2 against a 4.65 % floor that the POD basis imposes on its own. A model whose scalar is right and whose
  field is wrong is a model that has learned the average, not the flow.
- The error has a place, and a mean cannot say where: it is not a uniform haze over the section, it is the
  leading-edge shape above.

The dossier records the first as `accuracy_CL` (passed) and the second as `field_accuracy` (failed).
Both are in the same table, which is the only way the pair means anything.

### 4. Is it a flow, or only a fit?

Three of the checks below are the verification pipeline's own gates, run through
`run_surrogate_gates` on a bundle built over the lattice, so the gate reads the boundary condition
from the case and not from this page:

| gate | surrogate | reference | judged as |
|---|---|---|---|
| `accuracy` (relative L2 of Cp) | 0.2711 | — | 5 % |
| `boundary` (no-slip at the wall) | 0.0214 | 0.0218 | no worse than 1.5× the reference |
| `residual` (continuity, normalised) | 1.025 | 1.015 | no worse than 2× the reference |

Two of the three gates are read against the reference and not against an absolute: at the height the
sample is taken from, the solver itself sits at 0.0261 of the freestream, because a wall-function
solution has no sample on the wall, and its own continuity residual is 1.015. An absolute bar there
would fail the reference first. What is asked instead is that the prediction be no worse than the
solver that produced its training data -- which it is, on both counts: the wall condition by 0.0214
against 0.0218, the continuity residual by 1.025 against 1.015.

The accuracy gate on the *field* fails and is not hidden: the worst held-out design is 0.70, and the
error is not spread along the chord. It has the leading-edge shape above -- the shape the single
number 14.8 % cannot show, and the reason the next stages exist.

Two more checks the gates do not cover: freestream decay at the outermost ring, 0.084 against the
reference's 0.079; and the lift obtained by integrating the predicted wall pressure against the lift
the same network predicts, 0.32 apart on average -- a field and a scalar output that disagree are
two predictions, not one.

**Robustness.** Each input is nudged by 1 % of its own range on every held-out design; the movement of CL is compared
with the ensemble spread. Worst input: 0.72. A model that swings more than its members disagree has
memorised points between its data, not learned the input.

### 5. Uncertainty, and what it may not claim

The ensemble spread is a disagreement bar. Its coverage was measured on the twelve calibration
designs, pooled over the five predicted quantities: one standard deviation contains 43 % of their
errors, two contain 71 %, and the nominal values are 68 % and 95 %. The band under-covers, so it is
a disagreement bar and nothing more.

The dossier records it as `calibrated = False`, next to the method, and the check that would have
let the spread be called an interval fails.

### 6. Where the prediction is supported

Each design the model was not trained on -- 28 of them, the sixteen held out and the twelve used for
calibration -- contributes one evidence record: the error against its own independent simulation,
the two physical checks as ratios to the reference's own departure, and the ensemble spread. 20 of
the 28 pass all four criteria. The domain answers four questions that a marginal-range check cannot
tell apart:

| query | α | speed | thickness | answer | why |
|---|---|---|---|---|---|
| midpoint of the two closest references | 4.1 | 23.9 | 0.129 | EMPIRICALLY SUPPORTED | `local_checks_passed` |
| the centre of the box | 2.0 | 20.0 | 0.12 | UNSUPPORTED | `insufficient_local_reference` |
| outside the box | 4.0 | 20.0 | 0.18 | UNSUPPORTED | `outside_training_bounds` |
| a parameter the model does not have | 4.0 | 20.0 | 0.12 + camber | UNSUPPORTED | `parameter_schema_changed` |

The second query is the one this whole example is about. Every marginal range is satisfied as well
as it can be -- that angle of attack was simulated, that thickness was simulated, that speed was
simulated -- and the answer is UNSUPPORTED, because fewer than two independent references sit inside
the neighbourhood radius. *Inside the box is not the same as covered by the evidence.*

How thin the evidence is, measured rather than implied: over 200 candidates drawn uniformly from the
box, 0.5 % are supported and 99.5 % have no reference near enough to support a claim. That is not a
defect of the model. It is a measurement of how much evidence the campaign bought, and of how much
of a three-variable box sixty simulations can actually cover.

The neighbourhood radius is a policy choice and not a measurement, so the study reports the curve
instead of defending one point on it -- a claim that only survives at one radius is a claim about
the radius. Supported share of the same 200 candidates, by radius in the normalised design space:
0.10 → 0 %, 0.15 → 0 %, 0.20 → 0 %, 0.25 → 3 %, 0.30 → 14 %, 0.40 → 27 %. At the radius the policy
declares (0.206, the median distance between two training designs) the answer is the one above; a
study that wanted to claim more of the box would have to run more simulations, not pick a larger
radius, and the report shows the whole curve so that the choice is visible.

### 7. Opening a parameter

Camber is not a value outside a range -- it is a dimension the first model does not have, and a
query carrying it comes back `parameter_schema_changed`, not `outside_training_bounds`. Sixty-eight
cambered designs (32 train, 12 calibration, 24 held out) buy it, and the second model is a
*different* model: 4.13 % on lift and 5.50 % on drag over designs it has never seen.

### 8. Evidence arriving after the model is frozen

A model version outlives the evidence it was born with. `ValidatedDomain.extend` adds references to
the domain of the same weights -- nothing retrained -- and the campaign ran in two waves, so both
states are real: after 40 references, and after 61.

Two measurements, because they answer different questions and only one of them flatters the method:

- **Uniform over the box.** Of the same 400 candidates, 2 were supported before the late batch arrived and 6 after.
  The evidence grew by a half and the supported volume grew by four candidates, which is what a bracketing rule in
  four dimensions does when references are this sparse.
- **Built where the evidence lands.** For each late reference, the geometry halfway between it and the nearest
  reference the domain already had: 21 candidates, which are exactly the ones that had one neighbour and now have
  two. 7 of them went from UNSUPPORTED to EMPIRICALLY_SUPPORTED, each naming the reference that changed it. The other
  12 stayed unsupported because a reference near them fails the declared tolerances -- a reference arriving is not the
  same as evidence arriving -- and 2 because the model was never trained near them.

Nothing is forced, and that is the property being demonstrated: where the new evidence does not
reach, the answer does not move.

## The dossier

`results/dossier.md` is the report: the declared use and its tolerance, set before the answer was looked at; the
status and its reason; every check with the file that holds its number; one entry per independent
reference; the applicability at the condition the study was asked about; the uncertainty with its
measured coverage; the maturity on both axes -- level 4 of 5 on each, with what the fifth still
needs; the limitations written by whoever ran the study; and the provenance, down to the SHA-256 of
the code and of the 132 reference runs, so that reproducing it is a file and not a promise.

Every verdict in it comes from the evidence ledger: each check that ran is sealed against the SHA-256 of the file
its number came from, and the dossier reads its verdicts back out of the ledger rather than out of this script. A
check with no intact record cannot be reported as a pass -- it reads NOT RUN, which is what `reproducibility_rerun`
does here, and what any check would do if its evidence file changed under it. `results/ledger.json` is that record.

**Status: FAILED (technical)** -- 3 of 11 checks failed (field_accuracy, applicability_domain_measured, uncertainty_calibration_coverage) and 4 of the 16 held-out designs are above the 5 % lift tolerance.

That is the result of the study, and it is the one it should have. The declared use was to rank
candidate sections for a first design pass at 5 % on lift. The model meets that bar on average (4.84
%), and the verification pipeline declines to certify the model for that use anyway, naming each
reason: four designs above the bar, a field three times its own reconstruction floor, an uncertainty
band that covers less than its nominal share, and a design box whose centre has no evidence behind
it. Every one of those is a number in this repository, and every one of them would have been
invisible in the accuracy figure that a report usually opens with.

## What was not measured

These are named rather than implied, because a pipeline that hides its gaps is the thing this
example argues against:

- **Adaptive sampling.** `prioritize()` returns the unsupported candidates ranked; it does not choose, and no loop
  feeds them back into a campaign. Choosing is engineering judgement here.
- **Conformal prediction.** The intervals are an ensemble spread with a measured coverage, not distribution-free
  intervals. `calibration_metrics` reports coverage and ECE; it does not produce a bounded guarantee.
- **Geometry evidence authentication.** The checks are sealed in the ledger, but the evidence records the domain
  reads are plain dataclasses: they are not signed against a geometry hash, so nothing stops an evidence record being
  rewritten between the campaign and the assessment.
- **Closed-loop revalidation.** A new campaign does not trigger a re-assessment automatically.
- **Inference blocking.** The domain answers questions; it does not sit in front of a prediction and refuse it.
- **Experimental data.** Nothing here is compared with a wind tunnel. The reference kind is "numerical reference",
  and the profile says so.
- **Robustness beyond inputs.** No perturbation of the turbulence model, the mesh family or the Reynolds number; only
  the design inputs were nudged.
- **The wall time of the finest mesh.** L3 was re-run on its own after the campaign process died without leaving a
  record, and that run was not timed. The table says so instead of inventing a number; the coefficients from it are
  the solver's and are used like the others.

## References

- Celik, I.B., Ghia, U., Roache, P.J. (2008). Procedure for estimation and reporting of uncertainty due to
  discretization in CFD applications. *Journal of Fluids Engineering* 130, 078001.
- Roache, P.J. (1994). Perspective: a method for uniform reporting of grid refinement studies. *Journal of Fluids
  Engineering* 116, 405-413.
- Abbott, I.H., von Doenhoff, A.E. (1959). *Theory of Wing Sections*. Dover. (NACA four-digit geometry.)
