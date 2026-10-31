# SBINN on the ultradian glucose-insulin model: measured status

Targets: [TARGETS.md](TARGETS.md) (committed in 84b32bfe before any run).

| Target | Status |
|---|---|
| Yazdani et al. 2020 (PLoS Comput Biol) | not reproduced: paper not accessible (journals.plos.org, doi.org, PMC, bioRxiv refused) |
| T2 (structural identifiability, Eq 7, Table 2) | reproduced in its local numerical form, see `examples/biology/identifiability_ultradian.py` and `docs/guides/bio_informed.md` |
| T3 (Fisher null eigenvector = C2) | reproduced: eigenvalue ratio 9.7e-16, C2 component 0.9996, all others below 0.022 |
| T1 (17 inferred parameters, 10,000 + 600,000 Adam epochs) | reduced-scale (100k of 610k epochs): far from the paper, see below |
| T4 (forecast of the meal at 2,000 min) | reduced-scale (100k of 610k epochs): relative L2 of G on [1800, 3000] min 0.123 (scaled) and 0.105 (raw); the paper gives no number |

A 25,000-epoch pilot (4 % of the paper's training) did not converge in either loss scaling: relative parameter errors
up to 80 % (median 20 % with scaled losses, 68 % with the raw-unit weights of the authors' code), and G from the ODE
integrated with the inferred parameters 0.27 / 0.26 (relative L2) from the data. Pilot numbers are not reproduction
results.

## T1 at reduced scale (100k of 610k epochs)

Two runs, seed 0, the paper's data and settings, 10,000 data-only epochs, then 90,000 epochs with every loss term
(Adam, lr 1e-3, float32, one CPU thread):
- "scaled": every loss term scaled by the state magnitudes and the time span;
- "raw": physical units with the weights of the authors' code (ODE 1, glucose equation 1e-2, data 1e-2).

Both start from the middle of the search ranges, which is the nominal value for most parameters, as in the paper.
The status is "reduced-scale" for every row: 100k epochs is 16 % of the paper's training, so this is not a full
reproduction. Even ignoring that label, no parameter is within the factor-2 band of the paper's error: ours are
7 % to 80 % off, the paper's 0.1 % to 1.6 % (C2 aside).

| Parameter | True | Paper (Table 1) | Paper rel. error | Ours, scaled losses | rel. error | Ours, raw-unit weights | rel. error | Status (scaled / raw) |
|---|---|---|---|---|---|---|---|---|
| E | 0.2 | 0.201 | 0.50 % | 0.1219 | 39.1 % | 0.1249 | 37.6 % | reduced-scale / reduced-scale |
| tp | 6 | 5.99 | 0.17 % | 7.321 | 22.0 % | 7.999 | 33.3 % | reduced-scale / reduced-scale |
| ti | 100 | 101.2 | 1.20 % | 92.73 | 7.3 % | 129.9 | 29.9 % | reduced-scale / reduced-scale |
| td | 12 | 11.98 | 0.17 % | 16.67 | 38.9 % | 16.67 | 38.9 % | reduced-scale / reduced-scale |
| k | 0.008333 | 0.00833 | 0.36 % | 0.001667 | 80.0 % | 0.003245 | 61.1 % | reduced-scale / reduced-scale |
| Rm | 209 | 208.6 | 0.18 % | 179.7 | 14.0 % | 46.31 | 77.8 % | reduced-scale / reduced-scale |
| a1 | 6.6 | 6.59 | 0.15 % | 7.599 | 15.1 % | 3.911 | 40.7 % | reduced-scale / reduced-scale |
| C1 | 300 | 301.3 | 0.42 % | 376.1 | 25.4 % | 540 | 80.0 % | reduced-scale / reduced-scale |
| C2 | 144 | 37.65 | 73.85 % | 239.1 | 66.0 % | 51.73 | 64.1 % | reduced-scale / reduced-scale |
| C4 | 80 | 78.76 | 1.55 % | 17.94 | 77.6 % | 36.59 | 54.3 % | reduced-scale / reduced-scale |
| C5 | 26 | 25.94 | 0.23 % | 6.904 | 73.4 % | 18.2 | 30.0 % | reduced-scale / reduced-scale |
| Ub | 72 | 71.33 | 0.93 % | 27.91 | 61.2 % | 29.14 | 59.5 % | reduced-scale / reduced-scale |
| U0 | 4 | relative to C3 (not comparable) | - | 2.6 | 35.0 % | 0.8368 | 79.1 % | reduced-scale / reduced-scale |
| Um | 90 | relative to C3 (not comparable) | - | 159.8 | 77.6 % | 124.6 | 38.4 % | reduced-scale / reduced-scale |
| Rg | 180 | 179.9 | 0.08 % | 286.7 | 59.3 % | 318 | 76.7 % | reduced-scale / reduced-scale |
| alpha | 7.5 | 7.54 | 0.53 % | 6.338 | 15.5 % | 13.5 | 80.0 % | reduced-scale / reduced-scale |
| beta | 1.772 | 1.783 | 0.62 % | 1.088 | 38.6 % | 1.371 | 22.6 % | reduced-scale / reduced-scale |

scaled: G from the ODE with the inferred parameters vs the data, relative L2 0.159; meal forecast on [1800, 3000] min, relative L2 0.123; 100000 epochs, 0.0 h

raw: G from the ODE with the inferred parameters vs the data, relative L2 0.104; meal forecast on [1800, 3000] min, relative L2 0.105; 100000 epochs, 0.1 h


What this shows and does not show: at 16 % of the paper's training both loss scalings move the parameters away
from the values they started at, towards the bounds (k, C1, alpha, Rg at or near a bound), while the ODE fitted
with them reproduces G only to 10 % to 16 %. Whether the paper's 600,000 epochs reach its Table 1 values with this
implementation is not tested here.
