# The validation loop: a use case, its evidence, its maturity, and the person who signs it

A trust score answers one question: did the checks that ran hold? It does not answer the four an engineer actually
has to answer before a prediction goes into a decision:

1. Is the model operating inside the domain the evidence covers?
2. Which evidence supports or contradicts this prediction?
3. What does a specialist have to verify before it is used?
4. Which conditions were actually validated -- and which remain unknown?

`pinneapple_veriphysics` now carries that process. The unit is not a score but a **validation case**: a model version,
a reproducible problem, the checks that ran, the reference comparisons, the measured domain, the uncertainty, an
expert decision with its scope, and an append-only history of all of it.

| Module | What it adds |
|---|---|
| `usecase` | `ValidationProfile`: the declared use -- decision, decision quantity, tolerance, required reference, consequence of being wrong |
| `findings` | every signal as a `ValidationFinding` with four answers: what happened, why it matters, what is still unknown, what to do next (deterministic, no LLM) |
| `maturity` | `assess_maturity`: two independent axes -- scientific maturity and industrial readiness -- five verifiable levels each, with the gaps and how to close them |
| `validation_case` | `ValidationCase` (append-only revisions), `ExpertReview`, `ValidationRegistry` (evidence registry + improvement tasks) |
| `dossier` | `ValidationDossier`: the case as text, Markdown or a self-contained HTML document a reviewer signs |

## 1. Declare what success means, before running anything

```python
from pinneapple_veriphysics.usecase import PROFILE_PRESETS

profile = PROFILE_PRESETS["industrial_screening"].with_changes(
    primary_metric="relative error of the peak wall temperature",
    target_error_pct=5.0,                       # the use accepts 5 % on this quantity
    decision="screen candidate heat-sink designs before a full CFD run",
    critical_quantities=("peak wall temperature",),   # the region the decision depends on
    consequence="medium",
)
print(profile.describe())    # what this profile requires, and what it does not claim
print(profile.problems())    # [] when the declaration is complete
```

The tolerance is a **requirement**, not a measurement, and `describe()` says so: the number in `OBJECTIVES` is an
illustrative starting point read from common practice, never a norm. 0.5 % can be right for one benchmark quantity and
useless for another; 5-15 % is not automatically acceptable in industry -- it depends on the decision, the margin and
what happens if the number is wrong.

## 2. Build the case from evidence that already exists

Everything here converts objects the other modules produced; nothing re-runs a check or invents a number.

```python
from pinneapple_veriphysics.validation_case import (
    CheckVerdict, ModelVersion, ProblemRef, RunProvenance, ReferenceComparison, UncertaintyEvidence, ValidationCase,
    ApplicabilityFacts,
)

case = ValidationCase.open(profile=profile,
                           model=ModelVersion("pinn_heat2d", "v3", checkpoint_sha256="..."),
                           problem=ProblemRef("steady heat conduction, unit square", pde="poisson"),
                           requested_use="screening at 95 C", actor="analyst")

case = case.with_checks([                              # from a trust card, a lab report or a ledger:
    CheckVerdict("PDE residual", "verification", True, 3e-3, 1e-2),
    CheckVerdict("boundary conditions", "physics", True, 1e-7, 1e-5),
])                                                     # checks_from_trust_card / checks_from_lab_report / checks_from_ledger
case = case.with_reference(ReferenceComparison("ref01", "analytic solution", "peak wall temperature", 3.2,
                                               source="manufactured exact solution", artefact_sha256="..."))
case = case.with_applicability(ApplicabilityFacts(checked=True, requested_condition="screening at 95 C",
                                                  outside_variables=("T_inlet",)))   # or applicability_facts_from_map(...)
case = case.with_uncertainty(UncertaintyEvidence("ensemble spread", spread_pct=1.7, limitation="uncalibrated"))
case = case.with_provenance(RunProvenance(tool="pinneapple check", tool_version="1.2", data_sha256="...",
                                       code_sha256="...", seeds=(0, 1, 2), tolerances=(("residual", 1e-2),)))
```

`applicability_facts_from_map(amap.to_dict(), {"T_inlet": 95.0})` turns an
`applicability.build_applicability_map(...)` result into the extrapolation verdict: each queried variable outside its
verified range lands in `outside_variables`, widened only by a margin the study actually measured.

## 3. Read the findings, not a score

```python
a = case.assess()
print(a.status, "--", a.reason)       # "review required" here: the evidence holds, no expert has decided yet
for f in a.findings:
    print(f.render())
```

Every finding answers the same four questions and cites the evidence that fired it -- for a run like the one above:

```
VP-002 [WARNING] Condition of use outside the verified domain
  What happened: requested condition 'screening at 95 C' is outside the verified range in: T_inlet.
  Why it matters: The condition the use asks for falls outside the domain the evidence covers. Being outside the
    domain does not make the prediction wrong, and being inside it does not make it right; it makes it unsupported.
  What is still unknown: The behaviour of the model in this condition is unknown ...
  Recommended next steps:
    1. Run a reference computation or an experiment at this condition and compare the fields ...
```

The status vocabulary is fixed and each state means something different:

| Status | Meaning |
|---|---|
| `failed` | a mandatory check failed, a comparison exceeded the declared tolerance, or an expert refused the use |
| `unvalidated` | the evidence the declared use requires has not been produced (nothing ran, a required check or an independent reference is missing) |
| `review required` | evidence exists but carries flags (extrapolation, uncalibrated uncertainty, an incomplete requirement), or the expert approval that authorizes use is missing |
| `validated` | the requirements hold **and** an authorized expert approved the use inside a documented scope |

A case with nothing recorded is `unvalidated`; it never becomes `validated` because nothing was detected.

## 4. Maturity on two axes, with the gaps that matter

```python
print(a.maturity.render())
print(a.maturity.level("scientific"), a.maturity.level("industrial"))
for task in a.maturity.improvement_actions():
    print(task["priority"], task["dimension"], "L%d" % task["level"], task["action"])
```

A model can be scientifically mature and industrially unready (excellent evidence, no monitoring or fallback), or the
reverse (an approved operational scope resting on thin evidence). The two axes are never averaged. A level is reached
only when that level *and every level below it* are met, so a good metric cannot lift a case past a missing reference,
a missing reproducibility record or a missing review.

The five levels are this package's own starting scale, not an academic or industrial standard.

## 5. Record the expert decision -- as a revision, never an edit

```python
from pinneapple_veriphysics.validation_case import ExpertReview

case = case.with_review(ExpertReview.record(
    "eng. Silva", "approved with conditions",
    "error inside tolerance in the verified envelope; the 95 C condition still needs a reference run",
    role="reviewer", scope={"uses": ["screening at 85 C"]}))
print(case.assess().status)        # validated -- for that use; a different requested_use does not inherit it
```

Every change appends a sealed revision (`case.history`), and `case.history_problems()` reports a snapshot edited after
sealing. A reviewer cannot overwrite a technical failure: a failed check keeps the case `failed` whatever the review
says, and the review is recorded as what it is.

## 6. File the dossier

```python
from pinneapple_veriphysics.dossier import ValidationDossier

d = ValidationDossier.from_case(case)
d.write("validation_case.html")            # also "md" and "txt"
print(d.render_text())
```

The dossier carries the declared use, the criteria, the findings, every check (including the ones that did not run),
the reference comparisons with their measured error against the tolerance, the applicability envelope, the uncertainty
with its limitations, the expert decisions, the maturity with the open gaps, the provenance needed to reproduce it and
the revision history. It prints `DISCLAIMER` on itself: **not a certification** -- a documented engineering judgement
for a declared scope.

## 7. Keep the registry, and turn gaps into work

```python
from pinneapple_veriphysics.validation_case import ValidationRegistry

reg = ValidationRegistry()
reg.add(case)
reg.save("validation_registry.json")
print(reg.by_status("review required"))
print(reg.tasks())            # open maturity requirements, failed findings and evidence a reviewer asked for
```

`reg.evidence_rows()` is the auditable index: which evidence exists, for which model version, under which verdict.

## Where this fits

`validation_lab` runs the checks; `applicability` and `robustness` measure where the evidence holds; `ledger` proves
which checks ran; `decision` and `scope` say what a run supports. This loop sits on top and adds the three things
those modules deliberately do not do: the *declared use* the evidence is judged against, the *two-axis maturity* of
that evidence, and the *human decision with its scope and history*. Re-training, re-calibration and new reference runs
come back through the same door: as a new revision, with the previous one still in the file.
