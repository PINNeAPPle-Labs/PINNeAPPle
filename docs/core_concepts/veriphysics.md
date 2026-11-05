# Verification and evidence (`pinneapple_veriphysics`)

A recommendation is only as good as the evidence the reader can check. `pinneapple_veriphysics` is the layer that turns what the
analysis modules (`pinneapple_analysis.verification`) already computed into something an engineer can review and sign off:

| Module | What it gives you |
|---|---|
| `decision` | `DecisionRecord`: what is recommended, the trust score and its **coverage** (checks that did not run are absent, never faked), the per-check confidence breakdown, alternatives, and what to try if you disagree |
| `evidence_report` | the same record as a PDF (`render_evidence_report_pdf`; needs the optional `reportlab`) |
| `applicability` | the **applicability map**: an evidence chain, an 8-item checklist and the envelope of variables that was actually tested |
| `robustness` | measured robustness studies: does the verdict survive perturbing the problem? |
| `recommend` | `formulate_and_recommend`: formulate a problem and recommend a solver family and architecture from the verification catalog, citing sources |
| `execution_log` | a collector for what each step did, shown next to the result |
| `validation_lab` | `pinneapple check` grouped into numerical verification, validation against data and physical consistency, with an evidence-ledger entry per outcome |
| `ledger` | the evidence ledger: a check reads "passed" only with an intact record that it ran (inputs SHA-256, times, sealed entry) |
| `scope` | `ScopeOfUse`: which uses a result serves, which it does not and why |
| `usecase` | `ValidationProfile`: the declared use the evidence is judged against — decision, decision quantity, tolerance, required reference, consequence of being wrong |
| `findings` | each signal as a `Finding` with four answers: what happened, why it matters, what is still unknown, what to do next (deterministic, no LLM) |
| `maturity` | `assess_maturity`: two independent axes — scientific maturity and industrial readiness — five verifiable levels each, with the gaps and how to close them |
| `validation_case` | `ValidationCase` with append-only revisions, `ExpertReview`, and `ValidationRegistry` (evidence registry plus improvement tasks) |
| `dossier` | `ValidationDossier`: the case as text, Markdown or a self-contained HTML document a reviewer signs |

```python
from pinneapple_veriphysics.decision import DecisionRecord
from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf

record = DecisionRecord(...)                       # built from a ProvenanceRecord, a PhysicsConfidenceScore, a ToolRecommendation
open("evidence.pdf", "wb").write(render_evidence_report_pdf(record))
```

## The validation loop

The per-run modules answer "did the checks hold". The validation loop answers the question an engineer signs for: *for
which use, on which evidence, decided by whom, and what remains unknown?*

```python
from pinneapple_veriphysics.dossier import ValidationDossier
from pinneapple_veriphysics.usecase import PROFILE_PRESETS
from pinneapple_veriphysics.validation_case import CheckRecord, ModelRef, ProblemRef, ValidationCase

case = ValidationCase.open(profile=PROFILE_PRESETS["industrial_screening"], model=ModelRef("pinn", "v3", ...),
                           problem=ProblemRef("steady heat 2-D", pde="poisson"), requested_use="screening at 95 C")
case = case.with_checks([CheckRecord("PDE residual", "verification", True, 3e-3, 1e-2)])
case = case.with_review(ExpertReview.record("eng. Silva", "approved with conditions", "...", scope={"uses": [...]}))

print(case.assess().status)          # validated | review required | failed | unvalidated, with the reason
print(ValidationDossier.from_case(case).render_text())
```

Absence of alerts is never a pass: a case with nothing recorded is `unvalidated`, a check that did not run is missing
evidence, and an approval is a documented judgement for a declared scope — not a proof of validity. See the
[guide](../guides/validation_loop.md).

The design rule is the one of every `pinneapple_analysis.verification` module: this package **never computes a number or a verdict of its own**; it only
renders ones that exist. A check that was not run shows up as "not run", and the score says how much of it was covered.

Not here: the job queue, the HTTP API, billing and the web app. They belong to the Veriphysics product, which imports this package.
