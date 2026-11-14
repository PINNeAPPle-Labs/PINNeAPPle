# Lamb et al. (2026), ice crystal growth: frozen targets

Frozen before the neural-ODE runs of `run.py`. The two parameterised baselines were integrated once while the data
loader was being built (they are deterministic: one `solve_ivp`/RK4 pass each, no fitting), so the numbers below
are quoted from the paper and were not chosen after seeing ours.

Paper: Lamb, K. D., Harrington, J. Y., Moyle, A. M., Pokrifka, G. F., Clouser, B. W., Ebert, V., Möhler, O., &
Saathoff, H., "Discovering how ice crystals grow using neural ODEs and symbolic regression",
*Science Advances* **12**(39), eaee6174 (25 Sep 2026), doi:10.1126/sciadv.aee6174. Read from the preprint
arXiv:2510.17935v1 (20 Oct 2025, 21 pages, 4 figures + SI), which carries the same equations, tables and numbers.
Code the paper points at: `github.com/kdlamb/IceNODE` (read for the data layout and the published constants; no
code is copied from it — it carries no licence).

## Setting (quoted from arXiv:2510.17935v1)

- Growth law, Eq. 7: `dm/dt = 4 pi C (S_i - 1) G(r, T, S_i, alpha)` with `C = r` (sphere).
- `G`, Eq. 8: `G = [ R T / (e_sat,i(T) Dv* Mw) + (Ls / (ka T)) (Ls Mw / (R T) - 1) ]^-1`.
- Modified diffusivity, Eq. 9: `Dv* = Dv / ( r/(r + Delta_v) + r/(alpha w) )`, `w = sqrt(8 R T / (pi Mw))`.
- Deposition coefficient, Eq. 10 (Nelson & Baker 1996): `alpha = (s/scrit)^m tanh(scrit/s)`, `s = S_i - 1`;
  the paper's synthetic data use `m = 1`.
- `scrit`, Eq. 11: `9.6066e-5 |Tc|^1.9171`, `Tc` in Celsius (Harrington et al. 2019).
- Continuum limit, Eq. 12: `Gc = G(alpha -> 1)`.
- Constants: Table S1 (rho_i 910, Mw 18e-3, R 8.3144521, Ls 2.837e6, lambda_a 8e-8, Delta_v 1.3 lambda_a),
  `e_sat,i` Eq. 13 (Murphy & Koop 2005), `Dv` Eq. 14 (Pruppacher & Klett), `ka` Eq. 15.
- Networks: 3 linear layers x 50, ReLU after the first two, sigmoid output (so `0 <= alpha <= 1` and
  `0 < G/Gc <= 2`); inputs `(S_i, T)` for the strongly-constrained model, `(S_i, T, m)` for the weakly-constrained.
- Training: MSE (Eq. 5) over all series at once, AdamW lr 0.01, 500 epochs, cosine decay, `torchdiffeq` RK4
  (fixed step) or DOP853 (adaptive). Data interpolated to 1 Hz, first 500 s used to fit.
- Data: 290 mass-ratio time series, 205-240 K, 1.0 < S_i < 1.8, initial radii 6-26 um.

## Data actually available

290 = 4 "cold" high-supersaturation series + 18 (Pokrifka et al. 2018, doi:10.26208/z7bf-nq20) + 268 (Pokrifka et
al. 2023, doi:10.26208/htw5-q166). The 4 cold series were reprocessed by the authors from raw lower-plate voltage
files that are not in either archive; they are present *as processed mass ratios* inside the 268 (the four
(T, S_i, r0) triples match conditions rows 188, 227, 115 and 72). We therefore fit **286 series**, 98.6 % of the
paper's sample, and `bel.load(cold_dir=...)` adds the 4 back if the raw files are supplied.

## Targets

**T1. Table 1** — MSE loss (sum over all series) at `t < 500 s` and `t < 1000 s`, and the count of individual
experiments on which each model performs best:

| model | MSE (t<500 s) | MSE (t<1000 s) | best (t<500 s) | best (t<1000 s) |
|---|---|---|---|---|
| no surface kinetics (`G = Gc`) | 41396 | 222750 | 62 | 55 |
| Nelson & Baker 1996 | 40751 | 220930 | 29 | 31 |
| strongly-constrained NODE | 30714 | 203645 | 61 | 62 |
| weakly-constrained NODE | 16495 | 184723 | 138 | 142 |

Scored with `pinneapple_analysis.reproduction.status` against our numbers, and the *ordering* of the four models is
checked separately (it is the paper's actual claim).

**T2. Figure 2 / Figure S4** — trained on synthetic series whose kinetics are the Nelson & Baker alpha, the
strongly-constrained NODE recovers that alpha. The paper shows it graphically (panels a-c, residuals d-f) and quotes
no number; we report the median and mean of `|alpha_NN/alpha_NB - 1|` over the 286 conditions with no status.

**T3. Figure 3b / Table S2, Eq. 6** — the weakly-constrained network's `G` is distilled into a closed form. The
paper's own discovered equation (its Table S2, equation 8, printed as Eq. 6) is
`G = a0 Gc^a1 (a2 + a3/m)^-1 + a4` with `a0 = 688.267`, `a1 = 1.3153`, `a2 = 0.85601`, `a3 = 2.6606e-12`,
`a4 = 0.1123e-9`. Two targets: (a) our symbolic search reproduces the trained network's `G` (reported as relative
RMSE against the network, no paper number), and (b) the paper's Eq. 6, evaluated on our trained network's own
samples, should sit within a factor 2 of that error — it is the equation the paper claims for this data.

**T4. Figure 4, AIDA** — **not attempted.** It needs the IsoCloud archive (zenodo.org/records/7986868) *and* a bin
microphysics parcel model constrained to the observed ice number concentration, itself taken from Lamb et al. 2023;
that is a second study, not a scoring run of this one. No status is claimed for it and nothing here is evidence
about it.

**T5. The paper's qualitative claims**, scored from the same runs: (i) weak beats strong beats both baselines on the
summed loss; (ii) every expression discovered for `G` depends on `Gc` and on `m`, and only the most complex ones on
`S_i` or `T` (i.e. most of the variance is crystal size); (iii) the strongly-constrained single-alpha model is too
restrictive to fit the majority of the experiments.

## Scoring

Per metric: "reproduced" when our value is inside the paper's reported spread or, with a single reported number,
within a factor 2; "partially reproduced" when the ordering holds but the number is outside that band; "not
reproduced" otherwise. Runs with fewer than 500 epochs, or on fewer than the paper's series, are labelled
reduced-scale and never upgraded. A different RK4 step than the paper's (unstated) is not a reduced-scale label, but
`run.py --dt` records it.
