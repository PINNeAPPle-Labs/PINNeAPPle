# Validation dossier — NACA four-digit surrogate: an evidence-based validation study

**Case** `VC-52ddc41c9f` · **model** `naca-family-surrogate v1` · **requested use** rank candidate sections of a family for a first design pass

**Status: FAILED** (technical) — mandatory criterion not met: VP-001 (A mandatory check failed); VP-002 (A mandatory check failed); VP-003 (A mandatory check failed); VP-004 (Error above the declared tolerance); VP-005 (Error above the declared tolerance); VP-006 (Error above the declared tolerance); VP-007 (Error above the declared tolerance)

> Not a certification. This dossier records what was checked, what was compared, what was decided, by whom, with which evidence and inside which limits. A status of 'validated' is a documented engineering judgement for a declared scope, not a proof of validity, and it does not extend beyond that scope or survive a change of model version, inputs or intended use.

## Declared use

```
Validation profile -- airfoil preliminary screening
Objective: Industrial application (industrial); consequence of a wrong decision: medium
Primary metric: CL
Acceptance tolerance: 5 % (declared by the caller; 10 % is the illustrative starting point of 'industrial')
Reference evidence: numerical reference
Critical quantities: CL
Critical regions: none declared
Decision it feeds: screen candidate sections of a family before committing a design cycle

Industrial acceptance depends on the decision the number feeds, the consequence of being wrong and the design margin: single-digit to low-teens percent is a common starting range, never a rule. A high-consequence use needs its tolerance argued, not inherited from here.
```

## Findings

### VP-001 — A mandatory check failed `FAIL`

- **What happened:** field_accuracy: value 0.148198 against a threshold of 0.0929061 -- relative L2 over u, v and cp on the lattice, mean over the held-out designs: 14.8%. The 8-mode basis alone reconstructs the same fields to 4.6%, so the bar is twice the floor the model's own representation imposes -- a scalar can be right while the field it integrates is not
- **Why it matters:** A check that the configured criteria treated as mandatory failed for this result, so the result cannot be used as evidence for the declared use.
- **What is still unknown:** A failed check establishes that the property does not hold for the evaluated points; it does not by itself identify whether the cause is the model, the discretisation of the check, or the inputs.
- **Recommended next steps:**
  1. Read the recorded value and threshold of the check and reproduce it, so the failure is not a harness artefact.
  2. Fix the cause; do not relax the threshold to make it pass.
  3. Record the new run as a new revision: the failure stays in the history.
- **Evidence:** `field_accuracy`

### VP-002 — A mandatory check failed `FAIL`

- **What happened:** applicability_domain_measured: -- evidence domain queried at {'alpha_deg': 2.0, 'speed': 20.0, 'thickness': 0.12}: UNSUPPORTED (insufficient_local_reference)
- **Why it matters:** A check that the configured criteria treated as mandatory failed for this result, so the result cannot be used as evidence for the declared use.
- **What is still unknown:** A failed check establishes that the property does not hold for the evaluated points; it does not by itself identify whether the cause is the model, the discretisation of the check, or the inputs.
- **Recommended next steps:**
  1. Read the recorded value and threshold of the check and reproduce it, so the failure is not a harness artefact.
  2. Fix the cause; do not relax the threshold to make it pass.
  3. Record the new run as a new revision: the failure stays in the history.
- **Evidence:** `applicability_domain_measured`

### VP-003 — A mandatory check failed `FAIL`

- **What happened:** uncertainty_calibration_coverage: value 0.433333 against a threshold of 0.68 -- the raw ensemble spread covers 43% of held-out errors at one sigma (narrower than nominal: the bar under-covers)
- **Why it matters:** A check that the configured criteria treated as mandatory failed for this result, so the result cannot be used as evidence for the declared use.
- **What is still unknown:** A failed check establishes that the property does not hold for the evaluated points; it does not by itself identify whether the cause is the model, the discretisation of the check, or the inputs.
- **Recommended next steps:**
  1. Read the recorded value and threshold of the check and reproduce it, so the failure is not a harness artefact.
  2. Fix the cause; do not relax the threshold to make it pass.
  3. Record the new run as a new revision: the failure stays in the history.
- **Evidence:** `uncertainty_calibration_coverage`

### VP-004 — Error above the declared tolerance `FAIL`

- **What happened:** CL against numerical reference 'cfd-va001': 16.6671 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error of the primary quantity is larger than the tolerance the declared use accepts, so this result does not satisfy the criterion that was set for it.
- **What is still unknown:** This comparison does not tell whether the excess comes from the surrogate, from the reference itself (resolution, convergence, measurement), or from both. It also says nothing about the region the decision depends on: the figure reported is the one in the record.
- **Recommended next steps:**
  1. Check the convergence of the reference before touching the model: a reference that is not converged cannot condemn or exonerate a surrogate.
  2. Break the error down by region and by quantity, and look at the one the decision actually uses.
  3. Re-run the comparison with an independent second reference if one is available.
- **Evidence:** `cfd-va001`

### VP-005 — Error above the declared tolerance `FAIL`

- **What happened:** CL against numerical reference 'cfd-va008': 42.9314 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error of the primary quantity is larger than the tolerance the declared use accepts, so this result does not satisfy the criterion that was set for it.
- **What is still unknown:** This comparison does not tell whether the excess comes from the surrogate, from the reference itself (resolution, convergence, measurement), or from both. It also says nothing about the region the decision depends on: the figure reported is the one in the record.
- **Recommended next steps:**
  1. Check the convergence of the reference before touching the model: a reference that is not converged cannot condemn or exonerate a surrogate.
  2. Break the error down by region and by quantity, and look at the one the decision actually uses.
  3. Re-run the comparison with an independent second reference if one is available.
- **Evidence:** `cfd-va008`

### VP-006 — Error above the declared tolerance `FAIL`

- **What happened:** CL against numerical reference 'cfd-va012': 9.97206 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error of the primary quantity is larger than the tolerance the declared use accepts, so this result does not satisfy the criterion that was set for it.
- **What is still unknown:** This comparison does not tell whether the excess comes from the surrogate, from the reference itself (resolution, convergence, measurement), or from both. It also says nothing about the region the decision depends on: the figure reported is the one in the record.
- **Recommended next steps:**
  1. Check the convergence of the reference before touching the model: a reference that is not converged cannot condemn or exonerate a surrogate.
  2. Break the error down by region and by quantity, and look at the one the decision actually uses.
  3. Re-run the comparison with an independent second reference if one is available.
- **Evidence:** `cfd-va012`

### VP-007 — Error above the declared tolerance `FAIL`

- **What happened:** CL against numerical reference 'cfd-va015': 8.98408 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error of the primary quantity is larger than the tolerance the declared use accepts, so this result does not satisfy the criterion that was set for it.
- **What is still unknown:** This comparison does not tell whether the excess comes from the surrogate, from the reference itself (resolution, convergence, measurement), or from both. It also says nothing about the region the decision depends on: the figure reported is the one in the record.
- **Recommended next steps:**
  1. Check the convergence of the reference before touching the model: a reference that is not converged cannot condemn or exonerate a surrogate.
  2. Break the error down by region and by quantity, and look at the one the decision actually uses.
  3. Re-run the comparison with an independent second reference if one is available.
- **Evidence:** `cfd-va015`

### VP-008 — Uncertainty estimated but not calibrated `WARNING`

- **What happened:** method 'ensemble spread over five members on the POD coefficients': spread 2.81906 % of field scale; not calibrated against reference data (a disagreement bar detects instability; it is a confidence interval only once its coverage is measured, and here one sigma covers 43% of the held-out errors (two sigma, 65%))
- **Why it matters:** The case carries an uncertainty estimate (for instance the disagreement between members of an ensemble) that was never checked against reference data. It is a disagreement bar, not a confidence interval.
- **What is still unknown:** Whether the bars cover the true error at the stated confidence is unknown, so they cannot be used as a safety margin.
- **Recommended next steps:**
  1. Compare the estimate against reference data and report coverage and calibration error.
  2. Until then, use the estimate only to detect that the model is unstable, not to bound the error.
- **Evidence:** `results/analysis_v1.json`

### VP-009 — Expert review pending `NOT RUN`

- **What happened:** no expert decision is recorded; automated checks alone do not authorize use.
- **Why it matters:** The automated checks do not authorize use on their own. This case has no expert decision recorded, so it cannot be reported as validated whatever the checks say.
- **What is still unknown:** Whether the evidence is sufficient for the intended use, and inside which conditions, is a judgement this case has not received yet.
- **Recommended next steps:**
  1. Have a qualified engineer review the evidence, the findings and the limitations, and record a decision with its justification and the conditions it applies to.
  2. If the reviewer needs more evidence, record that request: it becomes a task, not a change of status.

### VP-010 — Automated checks recorded `PASS`

- **What happened:** 10 check(s) produced a verdict: 7 passed, 3 failed; 1 recorded check(s) did not run.
- **Why it matters:** The case carries verdicts of automated checks, each traced to the entry that produced it.
- **What is still unknown:** Which checks are missing is more important than how many ran: the coverage of the declared criteria is reported separately, and a check that did not run is not a pass.
- **Recommended next steps:**
  1. Read the not-run findings next to this one; they are the actual gaps.
- **Evidence:** `mesh_convergence_CL`, `gate_continuity_residual`, `boundary_condition_no_slip`, `freestream_decay`, `pressure_integral_consistency`, `accuracy_CL`, `field_accuracy`, `applicability_domain_measured`, `uncertainty_calibration_coverage`, `robustness_input_sensitivity`

### VP-011 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va000': 4.037 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va000`

### VP-012 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va002': 4.32206 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va002`

### VP-013 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va003': 0.373412 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va003`

### VP-014 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va004': 2.39604 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va004`

### VP-015 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va005': 4.97068 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va005`

### VP-016 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va006': 3.64626 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va006`

### VP-017 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va007': 3.12075 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va007`

### VP-018 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va009': 4.61182 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va009`

### VP-019 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va010': 0.150443 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va010`

### VP-020 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va011': 1.31871 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va011`

### VP-021 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va013': 1.67155 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va013`

### VP-022 — Error within the declared tolerance `PASS`

- **What happened:** CL against numerical reference 'cfd-va014': 2.29398 % error, tolerance 5 % (declared for this use), source: examples/end_to_end/airfoil_trust/_runs/cfd
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-va014`

### VP-023 — Error within the declared tolerance `PASS`

- **What happened:** CD against numerical reference 'cfd-held-out-CD': 8.13157 % error, tolerance 11.8213 % (declared for this use), source: results/analysis_v1.json
- **Why it matters:** The measured error is inside the tolerance configured for the declared use, on the quantities and region this comparison covered.
- **What is still unknown:** Agreement on the sampled points does not establish agreement outside them, and a mean or global figure can hide a region where the sign of the error changes. Passing here is a precondition of validation, not validation.
- **Recommended next steps:**
  1. Confirm the comparison covers the critical quantities and regions declared in the profile.
  2. Add a case at the edge of the intended operating range if none was run.
- **Evidence:** `cfd-held-out-CD`

## Checks recorded

| Check | Group | Status | Value | Limit |
|---|---|---|---|---|
| mesh_convergence_CL | verification | passed | 0.00986217 | 0.05 |
| gate_continuity_residual | verification | passed | 1.00983 | 2 |
| boundary_condition_no_slip | verification | passed | 0.987172 | 1.5 |
| freestream_decay | physics | passed | 1.05869 | 1.5 |
| pressure_integral_consistency | physics | passed | 1.01602 | 1.5 |
| accuracy_CL | validation | passed | 0.0483586 | 0.05 |
| field_accuracy | validation | failed | 0.148198 | 0.0929061 |
| applicability_domain_measured | applicability | failed |  |  |
| uncertainty_calibration_coverage | uncertainty | failed | 0.433333 | 0.68 |
| robustness_input_sensitivity | robustness | passed | 0.717847 | 1 |
| reproducibility_rerun | reproducibility | not run |  |  |

## Reference evidence

- `cfd-va000` numerical reference: CL = **4.037 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va001` numerical reference: CL = **16.6671 %** against 5 % (above); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va002` numerical reference: CL = **4.32206 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va003` numerical reference: CL = **0.373412 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va004` numerical reference: CL = **2.39604 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va005` numerical reference: CL = **4.97068 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va006` numerical reference: CL = **3.64626 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va007` numerical reference: CL = **3.12075 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va008` numerical reference: CL = **42.9314 %** against 5 % (above); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va009` numerical reference: CL = **4.61182 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va010` numerical reference: CL = **0.150443 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va011` numerical reference: CL = **1.31871 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va012` numerical reference: CL = **9.97206 %** against 5 % (above); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va013` numerical reference: CL = **1.67155 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va014` numerical reference: CL = **2.29398 %** against 5 % (inside); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-va015` numerical reference: CL = **8.98408 %** against 5 % (above); examples/end_to_end/airfoil_trust/_runs/cfd
- `cfd-held-out-CD` numerical reference: CD = **8.13157 %** against 11.8213 % (inside); results/analysis_v1.json

## Applicability

- requested condition: alpha_deg=2, speed=20, thickness=0.12
- verified ranges: alpha_deg in [-4, 8], speed in [10, 30], thickness in [0.08, 0.16]
- outside the verified range: none
- note: the evidence domain answers UNSUPPORTED at the requested condition (insufficient_local_reference)
- note: 199 of 200 sampled candidates have no reference within the neighbourhood radius
- note: camber is not a parameter of this model's schema: a query carrying it is outside the model, not merely outside the evidence

## Uncertainty

- method: ensemble spread over five members on the POD coefficients; NOT calibrated (a disagreement bar); spread 2.81906 % of field scale; coverage 0.43 vs target 0.68
- limitation: a disagreement bar detects instability; it is a confidence interval only once its coverage is measured, and here one sigma covers 43% of the held-out errors (two sigma, 65%)

## Limitations

- the surrogate is a function of three variables; camber is outside its schema and its evidence does not transfer
- the evidence covers the sampled design box only, and only where an independent reference sits within the neighbourhood radius
- steady RANS with wall functions: separated or transitional regimes are outside what the reference itself resolves, so no amount of surrogate accuracy extends the claim there
- the campaign is numerical throughout: no experimental data enters this case
- drag is not a validated quantity here: the reference's own grid-convergence index on CD is 12 % at the finest mesh, larger than any difference the surrogate could be judged on

## Maturity

```
Scientific maturity: level 4 (Generalisation characterised)
  To reach level 5 (Rigorously evidenced):
    - A second, distinct expert reviewed the evidence. -> have another qualified reviewer record a decision
Industrial maturity: level 4 (Operationally assessed)
  To reach level 5 (Ready for controlled operation):
    - An authorised expert approved the use, with justification and scope. -> record an expert decision
    - Monitoring, the fallback procedure and change control are documented. -> document what is monitored, the fallback and the change-control rule
    - The approved scope is documented with the conditions it applies to. -> record the conditions and quantities the approval covers
```

## Expert decisions

- none recorded

## RunProvenance

- tool: PINNeAPPle autolab 0.8.0+local; python 3.11.16
- inputs sha256: 8bb3f83be3859c8efe3f5595edb1dd35a2a7dcc1b8a4032e26469e9174291b35
- code sha256: 44541f6fa7036e6dba59fdc31a6786da700b5dfe174ac7c039cd0e1a95554398
- seeds: [30, 7, 11, 23]
- tolerances: {'lift_tolerance': 0.05, 'reference_tolerance': 0.05, 'wall_ratio': 1.5, 'continuity_ratio': 2.0, 'uncertainty_quantile': 0.9}
- compute: {'cores': '10', 'solver': 'simpleFoam', 'iterations': '1000', 'meshes': '4', 'reference_runs': '132'}

## Revision history

- rev 1 2026-11-26T18:19:43.281Z `unvalidated` case opened — sha256 `ef58d824e3cf`
- rev 2 2026-11-26T18:19:43.281Z `failed` checks recorded — sha256 `78138d1faa15`
- rev 3 2026-11-26T18:19:43.282Z `failed` evidence added: cfd-va000 — sha256 `4b8bc83ec9d1`
- rev 4 2026-11-26T18:19:43.283Z `failed` evidence added: cfd-va001 — sha256 `9646a76f31ab`
- rev 5 2026-11-26T18:19:43.284Z `failed` evidence added: cfd-va002 — sha256 `9289051d01b6`
- rev 6 2026-11-26T18:19:43.284Z `failed` evidence added: cfd-va003 — sha256 `8e84cbad26cf`
- rev 7 2026-11-26T18:19:43.285Z `failed` evidence added: cfd-va004 — sha256 `70312e66025c`
- rev 8 2026-11-26T18:19:43.286Z `failed` evidence added: cfd-va005 — sha256 `cec4bbfa8455`
- rev 9 2026-11-26T18:19:43.287Z `failed` evidence added: cfd-va006 — sha256 `a2be1ddd03b2`
- rev 10 2026-11-26T18:19:43.288Z `failed` evidence added: cfd-va007 — sha256 `e1b7a8108de0`
- rev 11 2026-11-26T18:19:43.288Z `failed` evidence added: cfd-va008 — sha256 `d23693d20304`
- rev 12 2026-11-26T18:19:43.289Z `failed` evidence added: cfd-va009 — sha256 `cfb4f3b92be6`
- rev 13 2026-11-26T18:19:43.290Z `failed` evidence added: cfd-va010 — sha256 `13a2b5d029c3`
- rev 14 2026-11-26T18:19:43.290Z `failed` evidence added: cfd-va011 — sha256 `43d71d5919ac`
- rev 15 2026-11-26T18:19:43.291Z `failed` evidence added: cfd-va012 — sha256 `ac90234455b3`
- rev 16 2026-11-26T18:19:43.292Z `failed` evidence added: cfd-va013 — sha256 `5537252bd04d`
- rev 17 2026-11-26T18:19:43.293Z `failed` evidence added: cfd-va014 — sha256 `bad2cad8db77`
- rev 18 2026-11-26T18:19:43.294Z `failed` evidence added: cfd-va015 — sha256 `a364fa103828`
- rev 19 2026-11-26T18:19:43.294Z `failed` evidence added: cfd-held-out-CD — sha256 `a0b8a9788e1a`
- rev 20 2026-11-26T18:19:43.295Z `failed` uncertainty recorded — sha256 `328504165a21`
- rev 21 2026-11-26T18:19:43.296Z `failed` applicability recorded — sha256 `b008dd8da72d`
- rev 22 2026-11-26T18:19:43.383Z `failed` provenance recorded — sha256 `85aec123b178`
- rev 23 2026-11-26T18:19:43.384Z `failed` limitations recorded — sha256 `f9c1e4b5a18c`

## What to do next

- (low) [industrial L5] record an expert decision
- (low) [industrial L5] document what is monitored, the fallback and the change-control rule
- (low) [industrial L5] record the conditions and quantities the approval covers
- (low) [scientific L5] have another qualified reviewer record a decision