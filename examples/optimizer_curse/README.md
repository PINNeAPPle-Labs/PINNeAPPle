# PINNeAPPle physics-validated design: synthetic pilot

Run from the PINNeAPPle repository root:

    python examples/optimizer_curse/run.py --seeds 5 --budget 512 --out /tmp/optimizer-curse

This is a manufactured analytic objective with an RBF surrogate, NOT CFD, a NACA 2412 experiment, or wind-tunnel validation. It does NOT reproduce reported results of 43%, 88%, or +37%.

The baseline and screened arms use the same surrogate and candidate budget. The screened arm uses the reusable DesignVerificationHarness and existing SHA-256 sealed EvidenceLedger. Neither arm consults the reference during selection. Independent reference adjudication occurs only after the selected candidate lists freeze.

Generated: results.json, REPORT.md, confirmation.png, evidence/seed_*.json.

Interpretation: the uncertainty multiplier is a heuristic, not a guaranteed 95% uncertainty interval. The ledger is tamper-evident locally, not an externally immutable audit store. Screening alone never validates physical correctness.

Physical-validation roadmap: parameterized NACA geometry, real OpenFOAM CFD campaign with mesh independence, external wind-tunnel comparison if conditions match, PINN/FNO/DeepONet benchmark under consistent train/calibration/test splits, reference-budget-matched randomized gates and no-OOD/no-escalation ablations, blind adjudication, CIs across seeds and expert engineering approval.
