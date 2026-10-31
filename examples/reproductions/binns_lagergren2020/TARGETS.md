# BINNs (Lagergren, Nardini, Baker, Simpson, Flores 2020): frozen targets

Frozen before any run. Source read: arXiv:2005.13073v1 ("Biologically-informed neural networks guide mechanistic
modeling from sparse experimental data", preprint dated May 28, 2020), downloaded from arxiv.org. The PLOS Comput
Biol version (journals.plos.org) and doi.org are refused by this session's network, so the page numbers below are
the arXiv preprint's ("n/34" footers), not the journal's.

Data: `Data/cell_density_profiles.npy` from github.com/jlager/BINNs (master @ 41d4b8f3), the scratch-assay data of
Jin et al. 2016 (J Theor Biol 390:136-145), downloaded at run time and not committed (the repository has no
licence file). Loader settings taken from the paper's Methods (p. 19): x in mm, t in days, u in cells/mm^2, the
left-most profile removed.

## Setting (quoted)

- GLS objective, Eq (6), p. 7: `L_GLS = 1/(MN) sum ((u_MLP(x_i,t_j) - u_ij) / |u_MLP(x_i,t_j)|^gamma)^2` "with
  proportionality constant gamma = 0.2". For the mechanistic models the model solution replaces u_MLP.
- Forward solver, Eqs (17)-(20), p. 22: method of lines, "x_i = i dx, t_j = j dt for i = 0,...,200 and
  j = 0,...,1,000", flux average of Eq (20), no-flux boundaries by ghost points, explicit RK4.
- Mechanistic fits, p. 22: "L-BFGS-B ... with default tolerance values ... gamma = 0.2"; "the carrying capacity was
  fixed at K = 1.7 x 10^3 cells/mm2" (p. 13); bounds "D_min = 0 mm2/day, D_max = 0.096 mm2/day, m_min = 0,
  m_max = 4, r_min = 0 /day, and r_max = 2.4 1/day ... beta0 and beta1 were bounded by [-10, 10]".
- Models: classical FKPP Eq (13); Generalized Porous-FKPP Eq (14); BINN-guided delay-reaction-diffusion Eq (11)
  with D = D0 + D (u/K)^m, G = r (1 - u/K), T = 1/(1 + exp(-(beta1 t + beta0))) (Eqs 12a-c), p. 12.

## Targets

T1. Table 2, p. 14, "Generalized least squares (GLS) errors" (verbatim):

| Model | 10,000 | 12,000 | 14,000 | 16,000 | 18,000 | 20,000 |
|---|---|---|---|---|---|---|
| classical FKPP | 786.80 | 557.28 | 616.76 | 619.12 | 685.17 | 964.19 |
| Porous-FKPP | 681.18 | 540.29 | 418.57 | 566.89 | 744.44 | 928.38 |
| BINN-guided model | 557.01 | 317.18 | 410.79 | 393.15 | 307.74 | 386.52 |

T2. Ordering claimed (p. 14): "Eq (11) with Eqs (12a)-(12c) fit each data set more accurately than the classical
FKPP or Generalized Porous-FKPP models." (lowest GLS in every column).

T3. Tab S1, p. 31, classical FKPP parameters (verbatim): D (um^2/hr) 309.7, 253.8, 681.8, 540.9, 735.7, 978.5;
r (1/hr) 0.0437, 0.0438, 0.0483, 0.0490, 0.0540, 0.0649.

T4. Tab S2, p. 31, Generalized Porous-FKPP parameters (verbatim): D (um^2/hr) 1851.8, 465.0, 2993.4, 2371.0,
2377.8, 2017.1; m 0.9704, 0.3001, 0.9923, 0.9879, 0.8265, 0.6235; r (1/hr) 0.0435, 0.0436, 0.0481, 0.0484,
0.0526, 0.0639.

T5. Table 1, p. 14, BINN-guided model parameters (verbatim):

| Parameter | 10,000 | 12,000 | 14,000 | 16,000 | 18,000 | 20,000 |
|---|---|---|---|---|---|---|
| D0 (um^2/hr) | 95.7 | 353.3 | 482.1 | 604.3 | 804.0 | 675.8 |
| D (um^2/hr) | 3987.1 | 3166.4 | 3775.0 | 3773.8 | 2201.8 | 1954.9 |
| m | 1.5976 | 3.4708 | 1.9060 | 3.5173 | 3.2204 | 0.9876 |
| r (1/hr) | 0.0525 | 0.0714 | 0.0742 | 0.0798 | 0.0772 | 0.0951 |
| beta0 | -1.0292 | -3.3013 | -3.1953 | -2.9660 | -1.2695 | -4.0651 |
| beta1 (1/hr) | 0.2110 | 0.2293 | 0.2761 | 0.2180 | 0.1509 | 0.4166 |

T6. Qualitative, p. 12 (Fig 7): "the learned diffusivities ... are non-zero when u = 0 and appear increasing and
concave up ... the learned growth terms are approximately linear ... and the learned delay terms all exhibit
sigmoidal dynamics." Scored on our own learned D_MLP, G_MLP, T_MLP (BINN training through
`pinneapple_analysis.inverse_problems.missing_term`).

## Scoring

Numbers T1, T3, T4, T5 with `pinneapple_analysis.reproduction.status` (no spread reported: factor 2 band); T2 and
T6 as orderings / shape checks. T1-T4 do not involve a neural network (classical fits of the paper's models to the
paper's data with the paper's solver); they test the data handling and the GLS definition. The BINN itself (20
retrainings per data set, early stopping 5,000 epochs) is run at whatever scale fits this machine and labelled
reduced-scale if smaller.
