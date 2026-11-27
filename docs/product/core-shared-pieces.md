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
    tests/pinneapple_analysis/test_trust_record.py tests/test_pretrained_experts.py tests/test_fidelity.py \
    tests/test_embedded_export.py tests/test_arena_battle.py tests/pinneapple_analysis/test_gp_symbolic.py \
    tests/pinneapple_design/test_quality_diversity.py tests/test_data_assimilation_conservation.py \
    tests/test_state_estimation.py -q
```

## The eleven

| # | piece | issue | module on `main` | proof | PR | state |
|---|-------|-------|------------------|-------|----|-------|
| 1 | Trust layer as a first-class object (`pp.trust`) | #345 | `pinneapple_analysis/trust/` — `record`, `adapters`, `certificate`, `corrections`, `drift`, `trust_gate`; exported as `pp.trust` in `pinneapple/__init__.py` | `tests/pinneapple_analysis/test_trust_record.py`, 32 tests | #2533 | **delivered** |
| 2 | Data assimilation with neural models (EnKF, 4D-Var) | #346 | `pinneapple_analysis/data_assimilation/var4d.py` (4D-Var through autograd, Lorenz-96 twin experiment, gradient check) and `pinneapple_analysis/state_estimation/kalman.py` (EKF, EnKF) | `tests/test_data_assimilation_conservation.py`, 8 tests; `tests/test_state_estimation.py`, 6 tests | #240 (4D-Var), #1255 (EnKF) | **partial** |
| 3 | Adaptive fidelity driven by the trust horizon | #347 | `pinneapple_analysis/fidelity.py` — `FidelityPolicy`, `serve`, `audit` | `tests/test_fidelity.py`, 11 tests: 0 answers served by the surrogate outside the tolerance, against 7 with the domain switched off | #2357 | **partial** |
| 4 | "Bring your foundation model, PINNeAPPle verifies it" | #348 | `pinneapple_physics/pretrained/` (registry with licence, four adapters), `pinneapple_physics/domain_checks`, `Expert` in `pinneapple_physics/ensemble.py` | `tests/test_pretrained_experts.py`, 14 tests | #2545 | **delivered** |
| 5 | Trust benchmark suite with a public leaderboard | #349 | — no single suite | — | — | **missing** |
| 6 | Embedded execution for drones and robots (ONNX) | #350 | `pinneapple_physics/embedded.py` — `EmbeddedSpec`, `export`, `EmbeddedRuntime` | `tests/test_embedded_export.py`, 16 tests | #2548 | **delivered** |
| 7 | Generic event scoring | #351 | `pinneapple_analysis/events.py` — `Event`, `Tolerances`, `detect`, `event_score`, `ensemble_event_score` | `tests/test_event_scoring.py`, 21 tests | this PR | **delivered** |
| 8 | Experiment design and sensor placement | #352 | `pinneapple_data/doe/` (designs, coverage, campaigns) and `pinneapple_analysis/data_assimilation/sensor_placement.py` (`Layout`, the four strategies, `layout_curves`) | `tests/pinneapple_data/test_doe.py`, 23 tests; `tests/test_sensor_placement.py`, 14 tests | #767 (doe), #679 (GP placement), this PR (search, budget, spacing) | **delivered** |
| E1 | Genetic-programming symbolic regression | #353 | `pinneapple_analysis/inverse_problems/gp_symbolic.py` | `tests/pinneapple_analysis/test_gp_symbolic.py`, 34 tests | #2535 | **delivered** |
| E2 | Quality-diversity search (MAP-Elites) | #354 | `pinneapple_design/design_optimizer/quality_diversity.py` | `tests/pinneapple_design/test_quality_diversity.py`, 25 tests | #2539 | **delivered** |
| E3 | Trust-aware surrogate-assisted NSGA-II | #297 | `pinneapple_analysis/surrogate_loop.py` (the budget-and-verification loop) | — the issue's acceptance is not met | #2429 | **open by decision** |

Eight of the eleven are delivered and proved. Three are not, and they are not the same kind of not.

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
members around it give a Brier score of 0.0145 (reliability 0.0096, resolution 0.0224, uncertainty 0.0283).

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

Two cases, criterion against 1/2/4/8 sensors. A 32 x 20 grid on a 1 x 0.6 plate with an isotropic Matern
(`l = 0.18`, `nu = 5/2`), scored on the variance of the mean temperature over the hot corner (prior 0.647):

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| greedy | 0.0539 | 0.0341 | 0.0089 | 0.0006 |
| evolutionary | 0.2130 | 0.0471 | 0.0527 | 0.0324 |
| uniform (maximin) | 0.6470 | 0.3799 | 0.3778 | 0.2681 |
| random | 0.5427 | 0.4147 | 0.3171 | 0.2134 |

and 80 stations down a 3 m slope with a long Matern (`l = 0.35`, `nu = 3/2`), weighted towards the toe (prior 1.000):

| k | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| greedy | 0.4629 | 0.2913 | 0.0965 | 0.0193 |
| evolutionary | 0.4629 | 0.2528 | 0.0907 | 0.0342 |
| uniform (maximin) | 1.0000 | 0.9443 | 0.3542 | 0.1020 |
| random | 0.7102 | 0.5439 | 0.4091 | 0.1822 |

The greedy walk wins on the plate by three orders of magnitude and is beaten on the slope at two and four sensors
(0.2913 against 0.2528, 0.0965 against 0.0907): its guarantee is a bound, not optimality. The reason to search anyway
is the constraint the walk cannot see — at eight sensors on the plate the walk's closest pair is 0.063 apart; held to
0.15, the spaced search reaches 0.173 and 0.104, against 0.374 and 0.268 for the even spread.

## What is missing

**5. Trust benchmark suite with a public leaderboard (#349).** Nothing here is one suite. The pieces exist
separately: the data regimes including distribution shift (`pinneapple_arena/regimes.py`), the red-team/blue-team
loop against the checks (`pinneapple_analysis/validation/arena`, #707), the drift monitor on served traffic
(`pinneapple_analysis/trust/drift.py`, #2354) and the battle scoreboard of who leaves the validated domain first
(`pinneapple_arena/battle.py`, #2536). What the issue asks for — hidden regime changes, sensor faults, slow drift and
out-of-domain cases behind one protocol, with a leaderboard anyone can be placed on — is not built: there is no
module that composes them and no test that runs the four failure kinds through the same API. The scoreboard of #2536
is the closest thing and it ranks models on one stream of conditions, not on a suite of injected faults.

**2. Data assimilation with neural models (#346) — partial.** 4D-Var through autograd is delivered and measured on a
Lorenz-96 twin experiment, and the EnKF is delivered and measured on linear systems. The neural half is not proved:
no test runs either filter on a model trained here, and the variational and ensemble paths are two APIs, not one
assimilation object. Closing it needs a test where the observation operator or the forward model is a neural network.

**3. Adaptive fidelity driven by the trust horizon (#347) — partial.** The router is delivered and measured
(`tests/test_fidelity.py`: over 400 requests against a surrogate fitted on a sub-box, every answer it served stayed
inside the tolerance, while the same surrogate with the domain switched off served seven outside it). But it
escalates on the trust *verdict*, the applicability domain and the interval width; it reads no horizon
(`grep horizon pinneapple_analysis/fidelity.py` is empty), and `pp.trust`'s `Horizon` — the weather and rollout
horizons of #345 — is not one of its conditions. The issue's wording is not met.

**E3. Trust-aware surrogate-assisted NSGA-II (#297) — open by decision.** The verification-budget loop exists
(`pinneapple_analysis/surrogate_loop.py`, #2429) and the issue remains open on its own acceptance. It is not a gap in
this epic to be filled by writing code here: an umbrella closes when its children close, and this child has not
decided what "trust-aware" means for the survivor selection yet.

The umbrella #343 stays open for the same reason. What closes it is #349 built, #346 proved with a neural model, and
#347 wired to the horizon — or a decision that the piece is not worth its cost, written on the issue.
