# Trust Beyond Accuracy -- video script

Two cuts of the same story: a technical demo of **8 to 10 minutes** and a **90 to 120 second** cut for a general
audience. Every number in this script is produced by the scripts in this folder and reproduced by `run_all.py`;
nothing is estimated for the camera.

**Naming rule for all public material.** The pipeline is called *the PINNeAPPle verification pipeline*. The suite the
modules live in is never named in narration, on screen or in the captions; the modules are referred to by what they do
("the evidence domain", "the dossier").

**Voice.** There is none. The narration below is burnt in as subtitles, one cue at a time, in the band under each
frame: the reader controls the pace of a silent film in a way a viewer of a narrated one cannot, and a page of numbers
reads better than it listens. The text is written, not dictated -- `4.84 %`, not "four point eight four percent";
`make_video.py` converts it back to words if a spoken cut is ever wanted again (`--narration voice`).

**Sound.** An ambient bed, synthesised by `make_video.py` itself: four slow chords in A minor, soft attacks, no
percussion, mixed around -29 dB so it never competes with the words. No licensed music, nothing downloaded.

**Pace.** The technical cut is 1732 words, read at 190 words per minute -- the pace of a reader who also has a figure
to look at, and about nine and a half minutes with the cues. If it needs to be shorter, cut the lattice anecdote in
Scene 03 (60 words) and the radius paragraph in Scene 08 (45 words) before cutting any number: the numbers are the
argument.

---

## The 8-10 minute technical demo

### Scene 01 -- The question (0:00 - 0:45)

**ON SCREEN.** Black. The question fades in, centred, in the same type as the figures:

> How do you know when to trust a Physics AI model?

Then, underneath, three smaller lines appear one at a time:
`a validation error is an average`, `a design is not an average`, `an average has no shape`.

**NARRATION.**

> A surrogate can be accurate and still be wrong about the design you care about. Its error is an
> average over cases; your design is one point, and nothing in that number says whether your point
> was near the evidence. That is the question the PINNeAPPle verification pipeline answers.
>
> This is one complete run of it on a family of airfoils: the reference simulations, the model, the checks it passes,
> the checks it fails, and the report that says what the model may be used for. It ends with the model refused -- and
> the refusal is the demonstration.

**LOWER THIRD.** `NACA four-digit family - steady incompressible RANS - OpenFOAM`

---

### Scene 02 -- The reference is not the truth either (0:45 - 2:00)

**ON SCREEN.** The mesh family, four levels, isosurface of cell size. Then the grid-convergence plot: CL and CD
against cell count, with the Richardson extrapolation as a dashed line and the GCI band shaded.

Command on screen: `python examples/end_to_end/airfoil_trust/01_reference_campaign.py mesh`

**NARRATION.**

> Everything here is measured against a reference, and the first thing the pipeline does is refuse
> to take it on faith.
>
> The same section is solved on four meshes, the finest at 16 117 cells, every spacing refined by
> the same factor. CheckMesh reports 6.8 degrees of average non-orthogonality, a maximum skewness of
> 1.77, no failed check. Convergence is recorded, not assumed: pressure residual 1.9e-08, global
> continuity 5.0e-12, and a run whose coefficient average has not settled is rejected.
>
> Then the discretisation study. Lift moves by 1.0 % between the two finest meshes, in both
> directions: converged, but not in the asymptotic range, so no observed order is claimed for it.
>
> Drag is the uncomfortable case. The two triplets disagree about the order -- 6.06 and 1.69 -- and
> the finest index is 11.8 %, an indicator rather than a converged uncertainty. Either is larger
> than any difference the surrogate could be judged on, so drag becomes a diagnostic and lift the
> validated quantity.

**CAPTION.** `results/mesh_study.json - Richardson extrapolation and GCI (Celik et al. 2008)`

---

### Scene 03 -- The first surrogate (2:00 - 3:00)

**ON SCREEN.** The design space, three axes, with 60 points appearing in three colours: 32 training, 12 calibration,
16 held out. Then the POD spectrum, then the training loss of the five ensemble members.

**NARRATION.**

> Sixty designs from a maximin Latin hypercube: angle of attack, speed, thickness, camber held at
> zero. The split is by whole designs, never by points -- two samples of one simulation share a
> geometry and a mesh, and a point-level split would leak.
>
> The fields come back on a fixed lattice, so one geometry's output lines up with the next. That
> lattice is also where this study nearly lied to itself: with the wall normals inverted, two thirds
> of the probes landed inside the solid, the solver returned -1e300, and every force coefficient stayed healthy. The run now rejects a probe it cannot place.
>
> Eight POD modes hold 98.87 % of the field energy and reconstruct an unseen field to 4.65 % before
> any network is involved. Five networks then train with different initialisations. Their
> disagreement is not a confidence interval -- we will come back to that -- but it is the first
> honest signal.

---

### Scene 04 -- Accuracy, and where it is wrong (3:00 - 4:05)

**ON SCREEN.** The parity plot for CL with error bars, the worst held-out design highlighted. Then the surface-pressure
comparison for that design and the error trace below it.

**NARRATION.**

> On the sixteen designs held out by design, lift comes back with a relative L2 error of 4.84 %, inside the 5 %
> tolerance this study declared before it looked at the answer. Drag, 8.13 %.
>
> Those are the numbers a normal report stops at. This one does not, because an average hides a
> shape. 4 of the sixteen are above that same bar on their own; the worst is va008 at 42.9 %, and
> the ensemble flagged it with the largest spread of the sixteen.
>
> Here is that case's surface pressure against its reference, and the error along the chord. The
> model is not uniformly mediocre: it is wrong in the first 15 % of the chord, where the
> predicted suction at the nose overshoots by eight tenths of a Cp, and past a fifth of the chord
> the deviation falls to two tenths and keeps its sign.
>
> A mean of 5 % and a leading edge that far off are the same model. Only one of those facts
> tells you what to do next.

---

### Scene 05 -- Is it a flow, or only a fit? (4:05 - 5:00)

**ON SCREEN.** The gate table: wall condition, continuity, field accuracy, freestream decay, pressure integral. Each
with a surrogate value and a reference value side by side.

Command: `python examples/end_to_end/airfoil_trust/03_physics_validation.py`

**NARRATION.**

> Accuracy asks whether the model reproduces what it was shown. It does not ask whether the
> prediction is a flow, and a model can pass the first question while failing the second.
>
> Wall condition: 0.0257 of the freestream, against the reference's 0.0261. Continuity: 0.995, against the
> reference's own 1.026 -- the surrogate is not smoother than the solver behind its data. Freestream decay at the
> outermost ring: 0.084 against 0.079. And the check that catches a field and a number that do not belong together:
> integrate the predicted wall pressure, compare with the lift the same network predicts -- 0.32 apart.
>
> Two of these are read as ratios, because at the sample height the solver itself departs from the
> wall condition by 2.5 %: an absolute bar would fail the reference first.
>
> And one fails. The field the scalar is integrated from is 14.8 % away in relative L2, against a
> 4.65 % floor the eight-mode basis imposes on its own. The lift is inside tolerance and the field
> it comes from is not.

---

### Scene 06 -- Robustness, and the check that did not run (5:00 - 5:30)

**ON SCREEN.** The check table with a column of statuses. Most read passed. One reads **NOT RUN**, in a different
colour, and stays on screen.

**NARRATION.**

> Robustness was measured, not promised: every input of every held-out design is nudged by 1 % of its range,
> and the movement of lift is compared with the ensemble spread. The worst input moves the prediction by 0.72 of what
> the members disagree by. The model is not memorising points between its data.
>
> Reproducibility was not measured. No independent re-execution of the campaign was run in this round.
>
> It appears in the report as NOT RUN. Not as a pass, not as a neutral grey, not as an omission. A check that did not
> run has failed the only thing it can be asked: it did not run.

---

### Scene 07 -- Uncertainty, and what it is allowed to claim (5:30 - 6:10)

**ON SCREEN.** Histogram of the error divided by the ensemble spread, with the one-sigma and two-sigma lines. Then the
coverage curve against the nominal one.

**NARRATION.**

> The five members disagree by some amount on every prediction, and it is tempting to call that amount a confidence
> interval.
>
> It is not one until its coverage is measured. On the twelve calibration designs, one standard deviation contains
> 43 % of the true errors; two contain 71 %. Nominal would be sixty-eight and ninety-five.
>
> So the spread under-covers, the check fails, and the dossier records the method next to the words "not calibrated".
> A disagreement bar detects instability. It does not bound the error, and the pipeline never presents it as if it
> did.

---

### Scene 08 -- Where the prediction is supported (6:10 - 7:10)

**ON SCREEN.** The design space at fixed speed, coloured by status: dark where there is no evidence, green where there
is. Independent references plotted as points. Three queries highlighted in turn, each with its verdict appearing
beneath.

Command: `python examples/end_to_end/airfoil_trust/04_validated_domain.py`

**NARRATION.**

> Every design the model was not trained on carries a record: the error against its own simulation,
> the physical checks, the ensemble spread. 28 designs, 20 of which pass every criterion. Those
> records are the evidence, and the evidence has a shape.
>
> Query one: a section nobody simulated, the midpoint of the two closest references. Both within the
> radius, both passing the declared tolerances: EMPIRICALLY SUPPORTED.
>
> Query two is the centre of the design box. Every marginal range is satisfied -- that angle was
> simulated, that speed was, that thickness was -- and the answer is UNSUPPORTED, because fewer than
> two independent references sit near it. Inside the box is not covered by the evidence.
>
> Query three, a thickness just outside the box: UNSUPPORTED, for the other reason.
>
> And the size of it: of two hundred candidates from the box, 99.5 % have nothing behind them. Not a
> defect of the model -- a measurement of what sixty simulations bought.
>
> The radius that defines "near enough" is a policy choice, so the study reports the whole curve:
> 0.30 gives 14 %, 0.40 gives 27 %. More of the box is bought with more simulations, not a larger
> radius.

---

### Scene 09 -- A new parameter is a new model (7:10 - 7:45)

**ON SCREEN.** Camber appears as a fourth axis. The first model is greyed out. A second campaign runs: 68 designs,
camber strictly inside the new range.

Command: `python examples/end_to_end/airfoil_trust/05_camber_round.py`

**NARRATION.**

> Camber was not in the first model's schema -- not a value outside a range, but a dimension the
> model does not have. Ask for a cambered section and the pipeline does not extrapolate: it answers
> UNSUPPORTED, reason recorded as a schema change.
>
> So the dimension is opened the only way it can be: a second campaign, sixty-eight cambered designs, a third held
> out, and a second model that replaces the first rather than pretending to extend it. 4.13 % relative L2 on lift, on
> designs it has never seen.

---

### Scene 10 -- What to simulate next (7:45 - 8:10)

**ON SCREEN.** A ranked list of geometries with no evidence behind them, each with the reason it is unsupported.

**NARRATION.**

> The domain also answers the inverse question: of a set of candidates, which have nothing behind
> them? They come back ranked, each with its reason -- outside the bounds, no reference nearby, or a
> parameter the model lacks.
>
> This is a list, not a decision. Choosing what to simulate is engineering judgement, and the pipeline does not
> pretend to automate it here. It makes the gap impossible to overlook.

---

### Scene 11 -- The domain grows without the model moving (8:10 - 8:50)

**ON SCREEN.** The same model version, the same four-dimensional domain, before and after evidence is added. Some
points flip from dark to green. A counter: `answers changed: 7`.

Command: `python examples/end_to_end/airfoil_trust/06_revalidation.py`

**NARRATION.**

> A model version outlives the evidence it was born with. New simulations arrive and are added to the domain of the
> same weights -- same version, nothing retrained. 40 references, and then 61.
>
> Now the honest part. Over four hundred candidates from the box, 2 were supported before the late
> batch and 6 after: the evidence grew by half and the supported volume grew by four designs.
>
> So the question is asked where the new evidence lands. Take the 21 geometries halfway between a
> late reference and its nearest earlier one -- exactly those that had one neighbour and now have
> two. 7 became supported, each naming the reference that changed it. 12 did not: a reference
> arrived near them and failed the tolerances, which is not evidence arriving. 2 more because the
> model was never trained there.
>
> Nothing is forced. Where the new evidence does not reach, the answer does not move.

---

### Scene 12 -- The report (8:50 - 9:40)

**ON SCREEN.** `results/dossier.md`, scrolling section by section: declared use, findings, checks, references,
applicability, uncertainty, limitations, maturity, provenance, revision history. Ends on the status line and the list
of what has to happen next.

Command: `python examples/end_to_end/airfoil_trust/07_report.py`

**NARRATION.**

> Everything above becomes one document. It opens with the use the model was declared for and the
> tolerance that use accepts, set before the answer was looked at, and then reports the status.
>
> FAILED (technical).
>
> Not because the model is bad. Because the check on the average passes and 4 designs do not;
> because the field is three times its own reconstruction floor; because the uncertainty band covers
> less than it claims; because the centre of the design box has no evidence behind it. Each verdict
> carries the check that produced it and the file where the number lives; the limitations are part
> of the result.
>
> Every verdict in it is backed by a sealed ledger entry against the hash of the file the number came
> from, which is why the check that did not run reads NOT RUN and not "assumed fine".
>
> And it holds what is missing: a check that did not run, every finding with the steps that would close
> it, and the one item no script can tick -- a second engineer's review.
>
> Trust is not a score. It is a map with edges on it. A pipeline that says yes to everything is not
> answering the question; this one says no, in writing, with the reasons numbered.

**CLOSING CARD.**

> A model that tells you where its predictions are supported.
>
> `python examples/end_to_end/airfoil_trust/run_all.py`

---

## The 90-120 second cut

Target: 105 seconds, ~250 words, no terminal commands on screen. Same voice.

**0:00 - 0:12 -- The question.**

> A surrogate model can be accurate and still be wrong about your design. Its error is an average. Your design is one
> point.

**0:12 - 0:32 -- The reference is measured, not assumed.**

> So the first thing this pipeline does is measure its own reference. Four meshes, a convergence study: lift settles
> to 1.0 % between the two finest, drag carries an 11.8 % index at its finest. The simulation is a measurement with an
> error bar, not the truth -- and that index is why this study validates lift and only reports drag.

**0:32 - 0:52 -- The model, and what accuracy hides.**

> Sixty simulations train a surrogate over a family of airfoils. On designs held out by design, lift is right to
> 4.84 % -- inside the 5 % this study declared. And four of those sixteen designs are outside it anyway,
> with the error sitting at the leading edge, where an average cannot see it.

**0:52 - 1:12 -- The checks a number cannot make.**

> Then the checks an error cannot make. Does the prediction satisfy no-slip, continuity, the freestream? Is the lift
> it predicts consistent with the pressure field it predicts? Is the uncertainty band as wide as it claims? Two of
> them fail. One check did not run, and the report says so.

**1:12 - 1:38 -- Where it holds.**

> Finally the question that matters: for this particular design, is there evidence? Not "is the angle of attack in
> range" -- is there an independent simulation near enough to support the claim? At the centre of its own design box,
> the answer is no. 99.5 % of the sampled designs have nothing behind them, and the pipeline says which ones.

**1:38 - 1:48 -- Close.**

> Trust is not a score. It is a map with edges on it.
>
> A model that tells you where its predictions are supported.

---

## Production notes

- **Figures.** `results/figures/mesh_convergence.png`, `accuracy.png`, `fields.png`, `uncertainty.png`,
  `domain_map.png`. All are generated by `07_report.py` from the files the earlier stages wrote; none is drawn by
  hand.
- **The domain animation.** `domain_map.png` is the static frame. For motion, evaluate `ValidatedDomain.assess` on a
  grid at two speeds (10 and 30 m/s) and interpolate between the two frames. What moves is how little is green.
- **The NOT RUN moment (Scene 06)** is the one place where the edit should hold. Two full seconds of a status column
  with one non-green entry, silence under the narration.
- **Scene 12 must not soften the status.** The film argues that a pipeline which says yes to everything is answering
  nothing. Cutting the FAILED line would cut the demonstration.
- **Numbers.** Every figure in this script has a source file in `results/`. If a number changes, re-run `07_report.py`
  and update the script from `results/dossier.md`, never from memory.
- **What is not in this video.** No adaptive sampling loop, no conformal intervals, no closed-loop retraining. Those
  are named as future work in the example README; showing them as if they existed would be the exact failure the
  video is arguing against.
