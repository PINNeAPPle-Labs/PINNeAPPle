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
| Porous-FKPP | 14,000 | 418.74 | 418.57 | D 3127.7 / 2993.4, m 1.019 / 0.992, r 0.0480 / 0.0481 | reproduced |
| Porous-FKPP | 16,000 | 533.65 | 566.89 | D 1370.7 / 2371.0, m 0.573 / 0.988, r 0.0484 / 0.0484 | reproduced (lower GLS, other minimum) |
| Porous-FKPP | 18,000 | 674.45 | 744.44 | D 1052.6 / 2377.8, m 0.237 / 0.827, r 0.0536 / 0.0526 | partial (D, m not reproduced) |
| Porous-FKPP | 20,000 | 926.58 | 928.38 | D 1779.1 / 2017.1, m 0.500 / 0.624, r 0.0640 / 0.0639 | reproduced |

FKPP and Porous-FKPP at 10,000 to 14,000 and 20,000: GLS within 0.2 % of the paper's, D and m within 12 % and 20 % at
20,000 and within 5 % elsewhere. At 16,000 and 18,000 our multi-start fit ends 5.9 % and 9.4 % below the paper's GLS
at a different (D, m): the paper's own parameters, put through our solver, give 567.61 and 744.68 (0.1 % from its
Table 2), so the solver and GLS agree with the paper and the paper's single fit stopped higher in the D-m valley.
Status per number with `status()` (factor-2 band); at 18,000 D and m fall outside it.

## Not done in this PR

- T1 row "BINN-guided model", T2 (ordering), T5 (its parameters), T6 (shape of the learned terms): the BINN training
  (`binn.py`, `ReactionDiffusionBINN`) and the delay-model fits have not been run at a measurable scale yet. The code
  and a smoke test are in; the numbers will come in a follow-up PR, labelled reduced-scale if fewer than the paper's
  20 retrainings per data set are run.
