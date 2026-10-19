# pp.validate: one call, one validation report

Other libraries train a Physics AI model; `pp.validate` says when to trust it. One call picks the checks that apply
to the model and the problem, runs them, and returns a report with per-check evidence, a credibility level per factor
and a deploy decision. A check that cannot apply is listed as *not applicable* with the reason, never skipped
silently.

```python
import pinneapple as pp

report = pp.validate(model, "darcy_2d", data={"train": {"a": a_tr, "u": u_tr}, "test": {"a": a_te, "u": u_te}},
                     query=a_new, level="standard")
print(report)                     # verdict, credibility per factor, one line per check
report.to_json("validation.json") # validation-report/v1
report.render("sarif")            # also "junit", "github", "html", "text"
report.to_trust_report()          # the trust card's APPROVED / REVIEW / REJECT

pp.validate(pp.zoo.load("heatsink_forced_mlp"), query=inputs)   # a zoo model: its card is the problem
pp.validate("fno.onnx", "darcy_2d", data=..., query=...)          # an ONNX file, run with onnxruntime
```

Models: a zoo model or its name, a `torch.nn.Module` (plain PyTorch, PhysicsNeMo), an `.onnx` file or any callable.
Problems: a zoo model's own card (`problem=None`), `"darcy_2d"` (field operators `a -> u` on the cell-centred grid
of `operator_benchmarks.darcy_2d`, the discretisation of the zoo dataset `darcy_2d_li2021`), or a small spec
`{"valid_range": {...}, "outputs": [...], "relations": [...]}` for a tabular callable. `pp.validate.<name>` still gives
the names of `pinneapple_analysis.validation` (`pp.validate.PhysicsValidator`).

To export an FNO to ONNX: `pinneapple_neural.architectures.neural_operators.fno_export.export_onnx(model, example,
path)` rewrites its spectral convolutions in real arithmetic (the exporters of PyTorch do not handle complex
weights), checks onnxruntime against PyTorch and returns the difference (below 1.2e-7 in the tests).

## Levels

| Level | Adds |
|---|---|
| `quick` | applicability (validity range, training envelope), guaranteed residual bound, flux balance, reference error, the zoo gate, catalog status |
| `standard` | spectral fidelity (#702), metamorphic relations (#698), the data audit (#700), the calibrated error bound (#699) |
| `thorough` | robustness to 1 % input noise against the solver's own sensitivity; fault injection (#697) where it applies |

Rollout checks (#701, `problem={"rollout": {...}}`) run at every level; their long-run statistics from `standard` on ([guide](rollout_validation.md)).

## Credibility

Checks are grouped by factor, modelled on NASA-STD-7009A and ASME V&V 40 (this library's reading of them, not a
certification): code verification, solution verification, validation, input pedigree, uncertainty, robustness,
applicability. Levels: 0 no evidence, 1 a critical check failed, 2 passes with warnings, 3 passes quantitative
checks, 4 passes against an independent reference or a guaranteed bound. The statuses, factors and check IDs are
defined once in `pinneapple_analysis.validation.schema`; a report refuses an ID that is not there.

The decision is `fail` (REJECT) when a critical check fails, `warn` (REVIEW) when a non-critical one fails or warns,
`pass` (APPROVED) otherwise; the verdict names the reasons, e.g. `fail: out of validity range`.

## What it found

From `tests/test_pp_validate.py` (Darcy 32², an FNO trained in plain PyTorch on 400 pairs, mean test error 4.2 %):

- **in distribution** (`thorough`): decision `warn`. Passing: training envelope, guaranteed energy-norm bound (median
  0.41, max 0.51: loose, about ten times the actual error), relative L2 4.2 %, reflection and transpose symmetry
  (worst 5 %), the data audit, the calibrated 90 % bound (mean coverage 90.2 % over 200 splits), 1 % input noise (2 x
  the solver's sensitivity). Flagged for review: the boundary flux does not balance the source (21 % imbalance), and
  the high-wavenumber energy is 12 x the reference's (small-scale noise).
- **queries from the contrast shift** a in {1, 20}, no ground truth: `fail: out of validity range` (amplitude 9.5,
  trained on 4.4-4.5); the residual agrees (guaranteed bound 4.4, flux imbalance 4.9).
- the same FNO **as ONNX** gives the same statuses; values agree to 1e-4.
- **`heatsink_forced_mlp`**: `pass`; the zoo gate's checks (card metrics, acceptance limits, the model's own physics
  checks), the metamorphic relations and the stored audit card of its dataset all pass, and the decision equals
  `pp.zoo.validate`'s (APPROVED). An air speed of 20 m/s (range 0.5-6) gives `fail: out of validity range`.

There is no Darcy model in the zoo, so the zoo case uses the forced-convection heat sink.

## The report as an artifact

`pinneapple_analysis.validation.report_artifact` (also `pp.validate.diff`, `pp.validate.badge`, ...):

| What | How |
|---|---|
| Versioned schema | `REPORT_SCHEMA` (`validation-report/v1`, JSON Schema 2020-12); `check_report(report)` lists what is wrong, including check IDs outside the vocabulary of `schema.CHECKS` |
| Provenance | `provenance.model_hash` (SHA-256 of the weights or the ONNX file), `data_hash`, `library_version`, `physics` (problem, tolerances, seed); zoo reports also carry the card's validity range |
| Diff between versions | `pp.validate.diff(old, new)`: checks lost (passed before, not now), gained, metrics moved beyond 10 %, validity-envelope changes; `.regressed` |
| Model card | `model_card_section(report)`: verdict, credibility table, checks, model hash |
| Badge | `badge(report)` (SVG, "physics-validated: 16/16 checks"), `shields_endpoint(report)` (shields.io endpoint JSON) |
| Signature | `sign_report(report, key)` / `verify_report(envelope, key)` with `pinneapple_security.signing` (#644) |

The zoo stores one report per model (`pinneapple_hub/zoo/validation_reports/`, `scripts/zoo/validation_reports.py
build`); each Hub card carries its section. The publishing workflow runs `scripts/zoo/validation_reports.py check`
before any upload: a model whose fresh report loses a check its stored report passed is not published. In the tests a
heat-sink "v1.1" with a sign error in the resistance loses the zoo gate's checks and is blocked.

Stored reports today: `heatsink_forced_mlp`, `heatsink_natural_mlp`, `rover_contact_grc1_viper` pass;
`airfoil_cst_rans_mlp` is `warn` because its training split has no recorded `Cm` (documented missing values).

## Limits

- Built-in physics: Darcy (and steady conduction, the same operator) for fields; tabular models get applicability,
  reference and metamorphic checks only. Other PDEs: run `pp.trust_card` on a grid, or add a problem here.
- The training envelope uses four scalar features of the input field (amplitude, spectral centroid, min, max); a
  shift that keeps them (a rearrangement of the same values) is not detected by it, only by the residual.
- Fault injection (#697) and rollout checks (#701) are reported as not applicable until those modules land.
