# SBINN on the ultradian glucose-insulin model: frozen targets

Frozen before any run.

Requested paper: Yazdani, Lu, Raissi, Karniadakis, "Systems biology informed deep learning for inferring
parameters and hidden dynamics", PLoS Comput Biol 16(11):e1007575, 2020. **Not accessible from this session**:
journals.plos.org, doi.org, pmc.ncbi.nlm.nih.gov and biorxiv.org are refused by the network policy, and the paper
is not on arXiv. Its own numbers are therefore not targets here; status for it: not attempted (paper not
accessible).

Paper read instead: Daneker, Zhang, Karniadakis, Lu, "Systems Biology: Identifiability analysis and parameter
identification via systems-biology informed neural networks", arXiv:2202.01723v1 (3 Feb 2022), the same authors'
workflow chapter on the same ultradian model with the same SBINN, downloaded from arxiv.org. Code referenced by it:
github.com/lu-group/sbinn (Apache-2.0, main @ 2d2639a8); nothing is copied from it.

## Setting (quoted, arXiv:2202.01723v1)

- Model: Eqs (1)-(2), p. 2-3, six states (Ip, Ii, G, h1, h2, h3), nominal parameters in Table 1, p. 4.
- Data, p. 3: "solving the system from time t = 0 to t = 1800 min with the initial conditions x(0) = [12.0 (uU/ml)
  4.0 (uU/ml) 110.0 (mg/dl) 0.0 0.0 0.0] and three nutrition events (tj, mj) = [(300, 60) (650, 40) (1100, 50)]
  (min, g) pairs"; Fig 2 caption: "360 measurements on glucose level (G) only are randomly sampled in the time
  window of 0 - 1800 minutes".
- Fixed: Vp, Vi, Vg (p. 8: "all the parameters in Table 1 except Vp, Vi and Vg"); search ranges: Table 1.
- Training, Fig 12, p. 12: Adam lr 1e-3, "10,000 epochs ... then for 600,000"; aux loss on all states at t = 0 and
  t = 1800 (p. 11).

## Targets

T1. Table 1, p. 4, column "Inferred Value" (verbatim), nominal value in brackets:
E 0.201 (0.2), tp 5.99 (6), ti 101.20 (100), td 11.98 (12), k 0.00833 (0.0083), Rm 208.62 (209), a1 6.59 (6.6),
C1 301.26 (300), C2 37.65 (144), C4 78.76 (80), C5 25.94 (26), Ub 71.33 (72), U0 0.0406C3 (4), Um 0.890C3 (90),
Rg 179.86 (180), alpha 7.54 (7.5), beta 1.783 (1.772).
Scored as the relative error of our inferred value to the nominal value, against the paper's relative error
(e.g. E: |0.201 - 0.2|/0.2 = 0.5 %), with `reproduction.status`. C2 is a target in itself: the paper's value is
off by 74 % and the chapter explains it as practical non-identifiability (T3).

T2. Section 3.2, p. 6, Eq (7): three scaling invariances (Rm with Ip(0), Ii(0); alpha with C5; U0, Um with C3) and
"the ODE system of Eq. (6) is structurally locally identifiable when Rm, C3, C5, Vp, Vi and Vg are fixed";
Table 2, p. 7, row "Given G" with Vp, Vi, Vg fixed: all of E tp ti td C1 C2 Ub U0 Um Rg alpha beta identifiable.
Scored with a local numerical test (Fisher-information rank, `pinneapple_analysis.inverse_problems.identifiability`)
on Eqs (1)-(2) with the initial conditions known, which is a different (weaker) test than the chapter's symbolic one
on Eq (6); the comparison reports where they agree and where they differ.

T3. Section 5, p. 14 and Fig 17: with 1 % noise on G, "There is only one eigenvalue close to 0, and the associated
eigenvector ... the value for the C2 component is dominant and all other components are approximately zero."

T4. Section 4.3, p. 12 / Fig 13: forecast of a meal at "tj = 2000 min with carbohydrate intake of mj = 100 g" with
the inferred parameters, "with high accuracy" (no number given: scored as the relative L2 error of G on
[1800, 3000] min, reported without a status since there is no target value).

## Scoring

T1 with `pinneapple_analysis.reproduction.status` per parameter; a run shorter than 610,000 Adam epochs is
labelled reduced-scale.
