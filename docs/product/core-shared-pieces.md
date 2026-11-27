# The shared pieces: what makes the use cases one product

Issue #343 is an umbrella, not a delivery: the application areas are what users see, and what makes them one product
is a set of pieces in the library core that they all share. Eleven things were listed — eight numbered sub-items and
three evolutionary methods — and each one has its own issue.

This page is the state of all eleven, with the evidence for each. A piece is **delivered** only when there is a
module on `main`, a test that proves it, and the pull request that put it there; a piece with a module and no test is
**not** delivered, and a piece whose halves exist but are not composed says so.

Regenerate the evidence with:

```bash
PYTHONPATH=$PWD python -m pytest tests/test_event_scoring.py tests/test_sensor_placement.py \
    tests/pinneapple_analysis/test_trust_record.py tests/pinneapple_analysis/test_trust_three_kinds.py \
    tests/test_trust_consumers.py tests/test_pretrained_experts.py tests/test_fidelity.py \
    tests/test_embedded_export.py tests/test_arena_battle.py tests/pinneapple_analysis/test_gp_symbolic.py \
    tests/pinneapple_design/test_quality_diversity.py tests/test_data_assimilation_conservation.py \
    tests/test_state_estimation.py tests/test_trust_benchmark_suite.py tests/test_trust_survivors.py \
    tests/pinneapple_analysis/test_neural_assimilation.py -q
```

## The eleven

| # | piece | issue | module on `main` | proof | PR | state |
|---|-------|-------|------------------|-------|----|-------|
| 1 | Trust layer as a first-class object (`pp.trust`) | #345 | `pinneapple_analysis/trust/` — `record`, `adapters`, `certificate`, `corrections`, `drift`, `trust_gate`; exported as `pp.trust` in `pinneapple/__init__.py` | `tests/pinneapple_analysis/test_trust_record.py`, 32 tests; `tests/pinneapple_analysis/test_trust_three_kinds.py`, 14, and `tests/test_trust_consumers.py`, 7 — the three kinds of prediction emitting one record, and the consumers reading it | #2533, #2566 | **delivered** |
| 2 | Data assimilation with neural models (EnKF, 4D-Var) | #346 | `pinneapple_analysis/data_assimilation/var4d.py` (4D-Var through autograd, Lorenz-96 twin experiment, gradient check) and `pinneapple_analysis/state_estimation/kalman.py` (EKF, EnKF) | `tests/test_data_assimilation_conservation.py`, 8 tests; `tests/test_state_estimation.py`, 6 tests | `tests/pinneapple_analysis/test_neural_assimilation.py`, 10 — one assimilation object over both kernels, and a model trained here inside the filters | #240 (4D-Var), #1255 (EnKF), #2564 (the neural half) | **delivered — PR #2564** |
| 3 | Adaptive fidelity driven by the trust horizon | #347 | `pinneapple_analysis/fidelity.py` — `FidelityPolicy`, `serve`, `audit` | `tests/test_fidelity.py`, 11 tests: 0 answers served by the surrogate outside the tolerance, against 7 with the domain switched off | #2357, #2559 (the horizon decides the route) | **delivered** |
| 4 | "Bring your foundation model, PINNeAPPle verifies it" | #348 | `pinneapple_physics/pretrained/` (registry with licence, four adapters), `pinneapple_physics/domain_checks`, `Expert` in `pinneapple_physics/ensemble.py` | `tests/test_pretrained_experts.py`, 14 tests, plus the 2020 ensemble demo next to each model alone | #2545, #2565 | **delivered** |
| 5 | Trust benchmark suite with a public leaderboard | #349 | `pinneapple_tools/benchmark_suite/trust.py` — the four failure kinds behind one `run_suite(candidates)` — and `scripts/trust_benchmark.py --run|--docs|--check` | `tests/test_trust_benchmark_suite.py`, 29 tests | #2561 | **delivered** |
| 6 | Embedded execution for drones and robots (ONNX) | #350 | `pinneapple_physics/embedded.py` — `EmbeddedSpec`, `export`, `EmbeddedRuntime` | `tests/test_embedded_export.py`, 16 tests | #2548 | **delivered** |
| 7 | Generic event scoring | #351 | `pinneapple_analysis/events.py` — `Event`, `Tolerances`, `detect`, `event_score`, `ensemble_event_score` | `tests/test_event_scoring.py`, 21 tests | this PR (the code is here, not on `main`) | **delivered by this PR** |
| 8 | Experiment design and sensor placement | #352 | `pinneapple_data/doe/` (designs, coverage, campaigns) and `pinneapple_analysis/data_assimilation/sensor_placement.py` (`Layout`, the four strategies, `layout_curves`) | `tests/pinneapple_data/test_doe.py`, 23 tests; `tests/test_sensor_placement.py`, 14 tests | #767 (doe), #679 (GP placement), this PR (search, budget, spacing — the code is here, not on `main`) | **delivered by this PR** |
| E1 | Genetic-programming symbolic regression | #353 | `pinneapple_analysis/inverse_problems/gp_symbolic.py` | `tests/pinneapple_analysis/test_gp_symbolic.py`, 34 tests | #2535 | **delivered** |
| E2 | Quality-diversity search (MAP-Elites) | #354 | `pinneapple_design/design_optimizer/quality_diversity.py` | `tests/pinneapple_design/test_quality_diversity.py`, 25 tests | #2539 | **delivered** |
| E3 | Trust-aware surrogate-assisted NSGA-II | #297 | `pinneapple_design/design_optimizer/trust_survivors.py` — trusted, contradicted or indeterminate against a per-objective tolerance — over `pinneapple_analysis/surrogate_loop.py` | `tests/test_trust_survivors.py`, 19 tests | #2429, #2562 | **delivered** |

Eight of the eleven are on `main` and proved by their own test. Two more — the generic event scorer (#351) and the
sensor search with a budget and a spacing constraint (#352) — are delivered **by this pull request**, which is why the
column says so instead of naming a merged one. The eleventh, the neural half of the data assimilation (#346), is
written and measured in PR #2564. Each of the three is said in the state it is, rather than counted as if it were
already in the tree.

## What was delivered in this pass

### 7. Generic event scoring (#351)

`pinneapple_analysis/events.py`. The weather case studies each scored an event their own way; this is that scorer
once, for any field and any time axis. `Event` names the kind (`threshold`, `extremum`, `onset`, `track`), the region
it can happen in, the lead window that is scored and the `Tolerances` (value, timing, position) that decide when a
forecast event *is* the observed one. Two of the kinds are a yes/no question — a level crossed, a change against a
reference frame — and get a confusion matrix, a critical success index, `lead_time_ok`, and the peak, position and
timing errors; the other two are the peak itself and get the peak and position errors without a confusion matrix.
`ensemble_event_score` gives the announced probability, the Brier score and its reliability-resolution decomposition.

The measurement, in the test and in the module docstring: Lorenz-96 (`K = 8`, `dt = 0.01`, 40 random initial states,
30 leads, event "the field's maximum above 3.0", tolerances 0.5 in value and 2 in leads), persistence against the
same fields —

| forecast | hits | misses | false alarms | CSI | POD | `lead_time_ok` | peak error |
|----------|------|--------|--------------|-----|-----|----------------|------------|
| persistence | 470 | 780 | 670 | 0.245 | 0.376 | 0 | 0.957 |
| the truth itself | 1165 | 0 | 0 | 1.000 | 1.000 | 29 | 0.000 |

and the hits of persistence by lead fall `38, 38, 38, 37, 34, 29, ..., 3, 1` — skill that decays with the horizon
rather than stopping, which is why both the single number and the per-lead table are reported. Scored as a moving
feature, persistence puts the maximum 0.954 away in value and 1.045 grid cells away in position; 20 perturbed
members around it give a Brier score of 0.0145 (reliability 0.0095, resolution 0.0232, uncertainty 0.0283, the
decomposition leaving 2e-4 inside the bin).

### 8. Experiment design and sensor placement (#352) — the half that was missing

The design of experiments already existed (`pinneapple_data/doe`: Latin hypercube, maximin, fractional factorial,
infill, coverage, campaign plans; 23 tests) and so did greedy sensor placement on a Gaussian process (#679). What was
missing is the combinatorial half the issue describes: the criterion that matters (the variance of a quantity of
interest, not only the weighted pointwise variance), a price per measurement with a budget, and a genetic algorithm
for the layouts a greedy walk cannot express.

`pinneapple_analysis/data_assimilation/sensor_placement.py` now has `Layout`, `random_sensor_placement`,
`maximin_sensor_placement`, `evolutionary_sensor_placement` (on `EvolutionaryDesignOptimizer`, with a hard minimum
distance between sensors and a budget) and `layout_curves`, and `greedy_sensor_placement` gained `qoi=`, `cost=` and
`budget=`.

Two cases, criterion against 1/2/4/8 sensors, all at the default search settings (`population_size = 30`,
`n_generations = 25`, `seed = 0`). A 32 x 20 grid on a 1 x 0.6 plate with an isotropic Matern
(`l = 0.18`, `nu = 5/2`), scored on the variance of the mean temperature over the hot corner (prior 0.647):

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| greedy | 0.0539 | 0.0341 | 0.0089 | 0.0006 |
| evolutionary | 0.2130 | 0.1492 | 0.1018 | 0.0233 |
| uniform (maximin) | 0.6470 | 0.3799 | 0.3778 | 0.2681 |
| random (median of 20 seeds) | 0.6034 | 0.4087 | 0.2550 | 0.1906 |

and 80 stations down a 3 m slope with a long Matern (`l = 0.35`, `nu = 3/2`), scored on the weighted pointwise
variance with the weights on the toe (prior 1.000):

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| greedy | 0.4629 | 0.2913 | 0.0965 | 0.0193 |
| evolutionary | 0.4629 | 0.2368 | 0.0872 | 0.0244 |
| uniform (maximin) | 1.0000 | 0.9443 | 0.3542 | 0.1020 |
| random (median of 20 seeds) | 0.7449 | 0.6175 | 0.3774 | 0.1551 |

The lottery gets one draw, not a median, so it is reported as one: over twenty seeds a single random layout on the
plate runs from 0.0916 to 0.6470 at one sensor and from 0.0264 to 0.6283 at eight -- wide enough that comparing one
draw against the walk says nothing. Its luckiest draw at two sensors, 0.0605, is six times better than the median
(0.4087) and still behind the walk (0.0341); at eight sensors no draw comes within a factor of forty of it. The
median is the honest baseline and it is what is quoted.

The greedy walk wins everywhere on the plate, by two to three orders of magnitude against every other layout, and is
beaten on the slope at two and four sensors (0.2913 against 0.2368, 0.0965 against 0.0872) and only there: at eight
the walk is back ahead (0.0193 against 0.0244). Its guarantee is a bound, not optimality. The reason to search anyway
is the constraint the walk cannot see — at eight sensors on the plate the walk's closest pair is 0.063 apart; held to
0.15, the spaced search reaches a closest pair of 0.173 at a criterion of 0.104, against 0.374 and 0.268 for the even
spread.

## The four criteria that were missing, now delivered

The pass that wrote this note delivered items 7 and 8. The four criteria this epic was waiting on were built
separately, each in its own issue, each with the test that asserts its own "Done when" -- and they are reviewed here
by where the evidence lives, not by the claim:

**5. Trust benchmark suite with a public leaderboard (#349) — PR #2561.** `pinneapple_tools/benchmark_suite/trust.py`
composes the four failure kinds behind one protocol (`run_suite(candidates)`), injects them with a ground truth and
ranks candidates on a declared, recomputable rule. Measured: on the four-model field, gp ranks 1 and gbt 4 on every
kind; out of domain gp errs 0.1246 against gbt's 1.9440 (15.6x); the hidden regime change is caught in 4.5 to 6.5
readings while the inputs stay inside the KS box. `scripts/trust_benchmark.py --run|--docs|--check` writes and guards
the record.

**2. Data assimilation with neural models (#346) — PR #2564.** One assimilation object over both kernels, and a model
trained in this repository used inside the filters. Measured: the neural analysis beats no assimilation in both
kernels, and an undertrained surrogate makes the analysis worse -- the negative case is asserted too, which is what
makes the positive one mean something.

**3. Adaptive fidelity driven by the trust horizon (#347) — PR #2559.** `pinneapple_analysis/fidelity.py` now reads
`pp.trust`'s `Horizon` (37 mentions where `grep horizon` used to be empty), and two tests hold the two halves: a short
horizon sends every otherwise acceptable request to the solver, and it escalates only the requests asking for a lead
it cannot cover.

**E3. Trust-aware surrogate-assisted NSGA-II (#297) — PR #2562.** The decision is written and implemented in
`pinneapple_design/design_optimizer/trust_survivors.py`: a candidate is trusted, contradicted or indeterminate against
a per-objective tolerance, and an interval that straddles the limit is undetermined rather than counted as a pass.
Measured against plain NSGA-II on the same problem and budget.

What is left is not code. This page closes the umbrella: eight of the eleven are on `main` with the test that proves
each, two land with it here, and the last one was waiting on PR #2564 when this page was written. #345 (the `Trust`
object the horizon and the certificates read from) landed in #2566 and #348 (bring your own model) in #2565, both cited
above.
