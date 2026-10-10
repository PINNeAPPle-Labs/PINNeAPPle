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

```python
from pinneapple_veriphysics.decision import DecisionRecord
from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf

record = DecisionRecord(...)                       # built from a ProvenanceRecord, a PhysicsConfidenceScore, a ToolRecommendation
open("evidence.pdf", "wb").write(render_evidence_report_pdf(record))
```

The design rule is the one of every `pinneapple_analysis.verification` module: this package **never computes a number or a verdict of its own**; it only
renders ones that exist. A check that was not run shows up as "not run", and the score says how much of it was covered.

Not here: the job queue, the HTTP API, billing and the web app. They belong to the Veriphysics product, which imports this package.
