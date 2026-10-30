# X-TFC (Schiassi et al.): reproduction

**Source:** arXiv:2005.10632v1, the preprint. The journal version (Neurocomputing 457, 2021) was not read,
because doi.org is refused by this session's network.

**Order of work:**

1. Targets were quoted and frozen in [TARGETS.md](TARGETS.md) before any run.
2. The baseline amendment was frozen before any baseline run.
3. Every number below was measured by `run_xtfc.py` and `run_baselines.py`.

The full tables are in [RESULTS.md](RESULTS.md) and [BASELINES.md](BASELINES.md); the raw data are in
`results.json` and `baselines.json`.

## Status

The score is the median, over seeds, of the maximum absolute test error. The status rule is
`pinneapple_analysis.reproduction.status`.

| Target | Paper | Ours: median [min, max] | Seeds | Status |
|---|---|---|---|---|
| ODE 1 (Fig. 3) | 6.66e-16 | 4.4e-16 [0, 8.4e-14] | 1000 | reproduced |
| ODE 2 y1 (Fig. 5) | 5.54e-10 | 0.91 [1.2e-7, 3.1] | 1000 | not reproduced |
| ODE 2 y2 (Fig. 6) | 1.58e-9 | 2.6 [3.5e-7, 8.7] | 1000 | not reproduced |
| PDE 1 (Table 4) | 5.05e-13 | 1.4e-13 [7.8e-14, 2.0e-13] | 10 | reproduced |
| PDE 2 (Table 5) | 7.58e-12 | 7.1e-12 [2.6e-12, 4.5e-11] | 10 | reproduced |
| PDE 3 (Table 6) | 8.98e-11 | 2.3e-10 [1.4e-11, 2.5e-7] | 10 | reproduced |
| PDE 4, 1D heat | 5.79e-7 | 6.5e-7 [3.8e-7, 1.1e-6] | 10 | reproduced |
| PDE 5, 2D heat | 4.03e-4 | 7.3e-4 [5.1e-4, 1.2e-3] | 10 | reproduced (factor 1.8) |
| PDE 6, non-linear 2D + t | 1.66e-8 | 5.5e-6 [1.9e-8, 5.9e-4] | 10 | not reproduced |
| PDE 7, non-linear 3D + t | 2.98e-5 | 2.4e-5 [3.9e-6, 8.0e-5] | 10 | reproduced |

**Result: 7 of 10 targets reproduced, at full scale.** Every case runs at the paper's settings on CPU.

![targets vs ours](targets_vs_ours.png)

### What did not reproduce, and what we know about why

- **ODE 2.** The protocol fixes Gauss-Newton from beta = 0 with the ELM fed physical coordinates, since the
  paper states no input scaling. Under that protocol the iteration diverges for most seeds.
  - A **post-hoc** variant maps t onto [-1, 1] before the ELM, which is the usual TFC practice. It reaches a
    median of 3.5e-9 (y1) and 1.0e-8 (y2), against the paper's 5.5e-10 and 1.6e-9.
  - That variant was chosen after seeing the failure, so it does not change the status. It is reported in
    RESULTS.md as "post-hoc".
- **PDE 6.** The error depends on the seed: from 1.9e-8 to 5.9e-4.
  - It is not an iteration limit. A post-hoc check gives the same errors with 15, 30 and 60 Gauss-Newton
    iterations.
  - The paper's single run (1.66e-8) sits at the good end of our spread: 3 of 10 seeds are at or below 5.2e-8.
    One possible explanation is that the paper reports one favourable run, but that cannot be checked.

### Declared secondary variants (frozen in TARGETS.md)

- **Chebyshev-Gauss-Lobatto training points** instead of a uniform grid.
  - Better on P3 (1.0e-11), P4 (3.8e-7) and P5 (3.4e-4).
  - Worse on P6 (1.5e-4) and P7 (4.2e-5).
- **ODE 1 with n = L = 50:** 6.7e-16.

## Physics checks of every run

- **Constraints.** These are exact by construction, and the measured error is rounding.
  - The measured maximum is 4e-9 or less on every case except P5, which reaches 3.2e-7. The P5 check
    ("constraints hold", threshold 1e-8) fails in `experiment.json`.
  - Cause, checked: the least-squares output weights of P5 reach |beta| ≈ 1e9 (seed 0). The constrained
    expression is a sum of differences of g, so it loses digits to cancellation. With beta = 1, the same
    expression meets the constraints to 3e-15.
- **Fresh-point residual.** It was evaluated on the 10 000 test points and is recorded per run in
  `results.json` and as checks in `experiment.json`.

## Fair comparison at the declared budgets (TARGETS.md, Amendment 1)

All methods are scored on the same 10 000 test points.

| Case | X-TFC | ELM-only (constraints as rows) | PINN, pp.solve, 5000 Adam epochs | FEM P1, 30x30 nodes |
|---|---|---|---|---|
| P1 | 1.4e-13 | 1.0e-9 | 2.3e-3 | 7.7e-4 |
| P2 | 7.1e-12 | 2.3e-8 | 6.3e-3 | refused (Neumann) |
| P3 | 2.3e-10 | 7.5e-8 | - | - |
| P4 | 6.5e-7 | 3.9e-5 | 5.3e-3 | refused (time-dependent) |
| P5 | 7.3e-4 | 4.9e-2 | - | - |
| P6 | 5.5e-6 | 1.2e-1 | - | - |
| P7 | 2.4e-5 | 1.0 | - | - |

**Findings:**

- On every case, X-TFC beats the same ELM without the constrained expression (same seeds, neurons and points).
  The margin is 2 to 5 orders of magnitude.
- The PINN baseline is far behind at this budget: 120-360 s per seed, against 0.03-0.6 s for X-TFC.
  - This is a statement about `pp.solve("pinn")` at 5 000 Adam epochs with resampled points.
  - It is not a statement about the best possible PINN: there is no L-BFGS and no tuning.

**Limits of the timings:**

- The X-TFC time is the solver's own total.
- The ELM-only time also includes evaluation on the test points and the constraint check.
- The PINN time is the full `pp.solve` call.

## Run it

```bash
python examples/reproductions/xtfc_schiassi2021/run_xtfc.py        # ~9 min on 4 CPU threads
python examples/reproductions/xtfc_schiassi2021/run_baselines.py   # ~35 min (the PINN runs dominate)
```
