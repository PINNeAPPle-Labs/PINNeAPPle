---
title: "Water hammer with a slow closure peaks at 0.70 of Michaud, not at it — pin a reference for the slow regime"
labels: [enhancement, solvers]
---

**Who has it:** `pinneapple_simulation.numerical_solvers.water_hammer_moc` and the checks that call it
(`pinneapple_systems.process_components.water_hammer`, the pipe presets, `pinneapple_reason`'s MOC-vs-Joukowsky rule).

**What happens:** the module is verified against Joukowsky for an instantaneous closure, and that is exact — the
experiment that motivated this issue measured the surge at 108.1953 m against `aΔV/g` = 108.1953 m, a relative error of
3e-14. But the slow regime, `t_c > 2L/a`, has no reference of its own: the docstring offers Michaud
(`2LΔV/(g t_c)`) and stops there, and the module's own `michaud` field is returned as if it were the answer.

It is not the answer. Running the MOC with a linear closure law (`closure_exponent = 1`) on a frictionless pipe of
1200 m at `a` = 1000 m/s (`ΔV` = 1.061 m/s, `2L/a` = 2.4 s) gives, as a fraction of Michaud:

| `t_c` / `(2L/a)` | 0.5 | 1 | 2 | 4 | 8 | 12 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| MOC peak / Michaud | 1.000 | 1.000 | 0.828 | 0.755 | 0.721 | 0.710 | 0.702 | 0.695 | 0.692 |

The ratio is 1 exactly where the module claims it, and then settles near **0.70**, never returning to it — so either
Michaud is an upper bound that the module's own solver contradicts, or the linear closure law is not the one Michaud
was derived for. Today nobody in the repository knows which, and a check can only bracket the answer between the
rigid-column estimate (`LΔV/(g t_c)`) and Michaud.

**Proposal:** settle the slow regime. Either

- derive or cite the closed form for a linear (not equal-percentage) closure and check against it, the way Joukowsky
  is checked; or
- if Michaud is the right reference and the 0.70 is a property of the linear law, say so in the docstring, record the
  ratio as a metric, and make the returned `michaud` field clearly an upper bound rather than a prediction.

**Acceptance (rule D1 — independent reference):** a check for `t_c > 2L/a` against a named closed-form reference, with
the tolerance justified by the derivation rather than fitted to the numbers above; and the `michaud` field either
renamed or documented as a bound where it is returned.

**Origin:** reproducing the "blank machine + agent + open-source toolchain" experiment, 2026-11-14 — the slow-closure
case was the only one of the twenty that had to be checked as a range instead of a value.
