# SBINN on the ultradian glucose-insulin model: measured status

Targets: [TARGETS.md](TARGETS.md) (committed in 84b32bfe before any run).

| Target | Status |
|---|---|
| Yazdani et al. 2020 (PLoS Comput Biol) | not attempted: paper not accessible (journals.plos.org, doi.org, PMC, bioRxiv refused) |
| T2 (structural identifiability, Eq 7, Table 2) | reproduced in its local numerical form, see `examples/biology/identifiability_ultradian.py` and `docs/guides/bio_informed.md` |
| T3 (Fisher null eigenvector = C2) | reproduced: eigenvalue ratio 9.7e-16, C2 component 0.9996, all others below 0.022 |
| T1 (17 inferred parameters, 10,000 + 600,000 Adam epochs) | not done: full-scale runs in progress (checkpointed; 70,000 epochs reached); results in a follow-up PR |
| T4 (forecast of the meal at 2,000 min) | not done: computed at the end of the T1 runs |

A 25,000-epoch pilot (4 % of the paper's training) did not converge in either loss scaling: relative parameter errors
up to 80 % (median 20 % with scaled losses, 68 % with the raw-unit weights of the authors' code), and G from the ODE
integrated with the inferred parameters 0.27 / 0.26 (relative L2) from the data. Pilot numbers are not reproduction
results.
