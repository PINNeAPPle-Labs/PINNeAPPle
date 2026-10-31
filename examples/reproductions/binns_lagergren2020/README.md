# BINNs (Lagergren et al. 2020): measured status

Targets: [TARGETS.md](TARGETS.md) (committed in 84b32bfe before any run). Scored with
`pinneapple_analysis.reproduction.status` (no spread reported by the paper: factor-2 band).

## T1, T3, T4: classical FKPP and Generalized Porous-FKPP (`classical.py`)

Each model fitted by L-BFGS-B from a fixed grid of neutral starting points (no paper value used as a start), the
paper's bounds, the paper's solver (201 points, method of lines) and GLS error (gamma = 0.2).

| Model | Density | GLS ours | GLS paper (Table 2) | Parameters ours / paper (um^2/hr, 1/hr) | Status |
|---|---|---|---|---|---|
| FKPP | 10,000 | 787.29 | 786.80 | D 309.8 / 309.7, r 0.0436 / 0.0437 | reproduced |
| FKPP | 12,000 | 557.96 | 557.28 | D 254.0 / 253.8, r 0.0438 / 0.0438 | reproduced |
| FKPP | 14,000 | 617.07 | 616.76 | D 681.4 / 681.8, r 0.0483 / 0.0483 | reproduced |
| FKPP | 16,000 | 619.36 | 619.12 | D 540.8 / 540.9, r 0.0490 / 0.0490 | reproduced |
| FKPP | 18,000 | 685.90 | 685.17 | D 735.4 / 735.7, r 0.0540 / 0.0540 | reproduced |
| FKPP | 20,000 | 964.85 | 964.19 | D 978.3 / 978.5, r 0.0649 / 0.0649 | reproduced |
| Porous-FKPP | 10,000 | 681.66 | 681.18 | D 1847.2 / 1851.8, m 0.968 / 0.970, r 0.0434 / 0.0435 | reproduced |
| Porous-FKPP | 12,000 | 540.96 | 540.29 | D 464.7 / 465.0, m 0.300 / 0.300, r 0.0436 / 0.0436 | reproduced |
| Porous-FKPP | 14,000 to 20,000 | | | | not done (run in progress) |

Our GLS errors are 0.03 % to 0.1 % above the paper's, the parameters within 0.3 %: the data handling, the solver
and the GLS definition match the paper.

## Not done in this PR

- T1 row "BINN-guided model", T2 (ordering), T5 (its parameters), T6 (shape of the learned terms): the BINN training
  (`binn.py`, `ReactionDiffusionBINN`) and the delay-model fits have not been run at a measurable scale yet. The code
  and a smoke test are in; the numbers will come in a follow-up PR, labelled reduced-scale if fewer than the paper's
  20 retrainings per data set are run.
