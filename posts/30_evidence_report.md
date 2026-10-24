Day 30 of 30: Engineering Evidence Report

At the end of a study, the write-up says "all checks passed". Which checks ran, on which inputs, and when? Usually nobody can tell.

The Engineering Evidence Report turns a study's outputs (Validation Lab reports, evidence ledgers, metrics tables, StudyResult and model files) into a report (HTML, PDF) and an evidence package (zip). It has one rule, enforced in code: a test is reported as passed or failed only when an evidence record shows that it ran, with the SHA-256 of its inputs, its outputs and its start and finish time. A source that only claims a pass is printed as "not run".

In the video, the example study runs on the server: FEM baseline vs a small PINN on the same Poisson problem, plus a temperature field through the Validation Lab. Result: 23 tests passed and 5 failed, each with a record; 1 not run, a planted grid-convergence claim with no record behind it. Relative L2 error is 7.54e-3 for FEM vs 5.64e-2 for the PINN, 7.5x worse, and the PINN fails its 4 boundary-condition checks and its reference check. The Validation Lab passes 12 of 12.

The PDF and the package download from the page. The package holds the report, the ledger, the figures, the 5 original files unchanged and a manifest: 13 of 13 SHA-256 hashes matched on download.

This closes the 30-day series with the app that records which checks actually ran.

Try it: https://evidence.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/evidence_report)

#Simulation #Verification #Reproducibility #ScientificComputing #PINN #FiniteElements #EngineeringQuality
