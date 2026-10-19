# Red team / blue team for the validation checks

The [check power](check_power.md) matrix measures the checks against a fixed list of defects, which only shows that
they catch the defects someone already thought of. The arena (`pinneapple_analysis.validation.arena`, issue #707)
adds an adversary. A red team searches for wrong outputs that pass every check (escapes) and for correct outputs
that fail (false alarms). A blue team then calibrates the checks with a stated false-alarm rate and strengthens them
until every stored escape is caught. Escapes and false alarms are stored as regression cases, and CI replays them.

```python
import pinneapple as pp

rep = pp.validate.arena("darcy_2d", budget="small", alpha=0.05)      # = arena.run_arena(...)
print(rep.to_markdown())        # per round: escapes, false alarms, mutation score, ROC/AUC, min. detectable magnitude
assert rep.passed, rep.failures()
rep.save_cases("cases/darcy_2d.json")
```

```bash
pinneapple check --arena-cases                    # replay the cases shipped with the library
pinneapple check --arena-cases cases/*.json       # or your own; exit 1 on a regression
python examples/validation_arena/run_arena.py     # full size on Darcy 2-D and Burgers 1-D
```

## Red team

- **Mutation operators.** These are the fault library of #697 (bias, scale, smoothing, noise, wrong boundary values,
  sign flips, out-of-range use, stale models) applied at every magnitude. The *mutation score* is the share of the
  mutants that matter (true relative error ≥ 10 %) which the defence rejects. This is the test-adequacy measure of
  mutation testing for deep learning (DeepCrime, Humbatova et al. 2021; MuFF, arXiv:2501.09846), with the checks as
  the tests. A mutant that matters and passes is an escape.
- **Search-based evasion.** Each generator is a parametrised family of outputs. The search maximises the true error
  against the reference solver while every check stays under its threshold, using differential evolution (SciPy),
  CMA-ES (written from Hansen's tutorial, arXiv:1604.00772) or finite-difference L-BFGS-B (`method=`).
- **Physics-consistent attacks.** These outputs solve a nearby problem exactly, so the residual stays small:
  - Darcy: a smooth change of `log a` (`coefficient_field`) or of `f` (`source_field`);
  - Burgers: another viscosity (`viscosity`), the exact initial condition followed by the evolution of a smoothed one
    (`hard_ic_evolution`, what a hard-constrained PINN with spectral bias returns), or rescaled time (`time_warp`).
- **False-alarm search.** The search runs over correct cases from the verified solvers, with coefficient contrast,
  feature size, amplitude, viscosity and solver resolution varied inside the training envelope, and maximises the
  largest margin of the defence. The result is the most suspicious-looking correct case, which counts as a false
  alarm when it is rejected.

## Blue team

- **Calibrated combined test.** The clean calibration cases are split in two halves, A and B. Each check's scale
  `s_j` is its split-conformal quantile on A (`pinneapple_analysis.uncertainty.conformal.conformal_quantile`), with a
  floor of 1 % of its own tolerance. The combined statistic is `T = max_j value_j / s_j`, and its threshold is the
  split-conformal quantile of `T` on B. The probability that a new clean case is rejected by *any* check is then at
  most `alpha_cal` (Vovk et al. 2005; Angelopoulos & Bates, arXiv:2107.07511). This is the family-wise rate, so false
  alarms do not grow with the number of checks.
  - `alpha_cal = safety × alpha`, with `safety = 0.5` by default, so that the rate measured on a finite held-out set
    stays under the stated `alpha`.
  - The guarantee needs `ceil((n_B + 1)(1 - alpha_cal)) ≤ n_B`, and `Defence.alpha_guaranteed` reports whether that
    holds.
- **Hardening.** Stored false alarms are correct cases, so the threshold is raised just above them; this can only
  lower the false-alarm rate. While a stored escape still passes, the blue team adds the check from
  `check_library(problem)` that rejects the most remaining escapes after recalibration. The log records what was
  added and why.
- **Regression cases.** Every escape and false alarm is saved with its case seed, variation, generator and
  parameters, together with the hardened thresholds. `pinneapple check --arena-cases` regenerates each case and
  requires escapes to be rejected and false alarms to pass.

## Results (seed 0, `alpha = 0.01`, 2 rounds, full budget)

Full budget: 400 clean calibration cases, 300 held-out clean cases, 3 base cases × 160 evaluations per generator, and
4 random fields × 80 evaluations for the false-alarm search. Regenerate with
`python examples/validation_arena/run_arena.py --rounds 2`.

| | Darcy 2-D | Burgers 1-D |
|---|---|---|
| round-0 defence (hand-set tolerances) | FD residual, boundary condition, bounds, training envelope | FD residual, initial condition, bounds, training envelope |
| escapes found at round 0 (true error ≥ 10 %, every check passes) | 15 | 16 |
| worst escape | `coefficient_field`: 87 % true error | `time_warp` (s = 0.5): 50 % true error |
| mutation score, before → after | 67.3 % → 100 % (419 mutants) | 75.5 % → 100 % (351 mutants) |
| held-out false-alarm rate, hand tolerances → combined test | 1.33 % → 0.00 % (0.67 % after round 1) | 0.00 % → 1.00 % |
| worst false alarm (most suspicious correct case) | margin −0.11, accepted | margin −0.03, accepted |
| check added by the blue team | conservative residual on the solver stencil | one-step re-integration residual |
| stored escapes rejected after blue | 15/15 | 16/16 |
| escapes ≥ 10 % found at round 1 against the hardened defence | 0 (largest accepted error 4.9 %) | 0 |

How to read these results:

- Round 0 shows that the default checks have real blind spots. A Darcy output solved for the wrong permeability
  field, or a Burgers solution at the wrong time, has a small finite-difference residual and the right boundary or
  initial condition. Every check accepts it, yet it is wrong by 50-87 %.
- The strengthened checks close these gaps because they use the solver's own discretisation.
- No correct case was rejected by the hand-set tolerances in the search. The held-out set did contain 4 correct Darcy
  cases (out of 300) that the hand-set tolerances flag.
- The calibrated test states its rate:
  - Darcy stays under 1 % with a margin.
  - Burgers is at the bound (3 of 300). Its one-step residual is calibrated on correct cases solved at finer
    resolution, so the tail is real.

The report also lists, for every check and fault, the detection rate, the false-alarm rate, the ROC AUC over fault
magnitude and the minimal detectable magnitude, before and after the round.

## External models

`audit_model(problem, predict)` runs any model through the hardened defence shipped with the library and compares
each output with the reference solver. `predict(inputs, params)` returns the field on the problem's grid (Darcy
`a → u` on 32 × 32 cells, Burgers `u0 → u(t, x)`). The result lists the escapes (wrong outputs the defence accepts) and
the accurate outputs it rejects. This is the hook for the FNO and MeshGraphNet models of the PhysicsNeMo trust
experiment: wrap their inference as `predict` (PhysicsNeMo is not a dependency of PINNeAPPle, so this is not run in
CI).

## Scope and limits

- Two problems with verified solvers. A check is only as strong as the discretisation it shares with the reference:
  the conservative residual flags correct solutions computed on a different grid at the level of the discretisation
  error, which is why it is calibrated rather than set to round-off.
- The evasion search is local to its budget. "No escape found" means none within the stated evaluations, not none at
  all; the report says which.
- The data-level arena (datasets rather than single outputs) is measured by the fault-injection matrix with the
  dataset checks; the conformal calibration covers the case-level checks.
