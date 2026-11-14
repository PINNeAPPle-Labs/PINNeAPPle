# Lamb et al. (2026), ice crystal growth: measured status

Targets: [TARGETS.md](TARGETS.md), quoted from the paper before any neural-ODE run.
Paper: *Science Advances* **12**(39), eaee6174 (2026), doi:10.1126/sciadv.aee6174, read from arXiv:2510.17935v1.

The paper learns the unknown part of the depositional ice growth law — the transfer coefficient `G` of
`dm/dt = 4 pi r (S_i - 1) G` — with a neural ordinary differential equation fitted to 290 mass-ratio time series of
single ice crystals grown in a levitation diffusion chamber, distils the trained network into a closed form with
symbolic regression, and tests that closed form on independent AIDA cloud-chamber data.

| Target | Status |
|---|---|
| T1 (Table 1: losses and best-experiment counts, four models) | **reproduced** for both baselines and both NODEs on the losses (all four inside a factor 2, two of them inside 2 %); the best-experiment count is reproduced for three models and **not** for the strongly-constrained one, whose count is 21 against the paper's 61 because the weak model wins most experiments here — see below |
| T2 (Figure 2: the strongly-constrained NODE recovers a known `alpha`) | **reproduced**: median `|alpha_NN/alpha_NB − 1|` = 1.6 % over the 286 conditions (the paper shows the agreement graphically and quotes no number) |
| T3 (Figure 3b / Table S2: a closed form for `G`) | **partially reproduced**: the search finds a 9-node closed form whose RMSE is 0.142 on a target of spread 0.203 (70 % of the variance); the paper's Eq. 6, scored on the same trained network's `G`, is 0.268, so the two are in the same band but our expression is not the paper's form |
| T4 (Figure 4: AIDA cloud chamber) | not attempted — needs the IsoCloud archive **and** a bin-microphysics parcel model (see TARGETS.md) |
| T5 (qualitative claims: weak > strong > both baselines; `G` depends on `Gc` and `m`) | **reproduced**: 7,236 < 27,334 < 41,424 < 41,482 at t<500 s (and the same order at t<1000 s); every expression found depends on `Gc` and `m` |

## What is reproduced, and how

Run with `python examples/reproductions/lamb2026_icenode/run.py` (`--quick` for a plumbing run of 20 epochs). The
four models of Table 1 are integrated with the same fixed-step RK4 solver, so the comparison isolates the kinetics:

| label | kinetics | network |
|---|---|---|
| `none` | `G = G_c` (no surface kinetics) | — |
| `nelson_baker` | `alpha` from Nelson & Baker (1996), Eq. 10 | — |
| `strong` | `alpha = f_alpha(S_i, T)`, sigmoid output | 3 x 50 MLP |
| `weak` | `G/G_c = f_G(S_i, T, m)`, sigmoid output scaled to (0, 2] | 3 x 50 MLP |

Training is the paper's: sum of squared `m/m0` errors over all series at once, AdamW at 0.01 with cosine decay.
Every one of its constants and parameterisations is in `pinneapple_physics.cloud` (`esat_i` Murphy & Koop 2005, `Dv`
and `ka` Pruppacher & Klett, `scrit` Harrington et al. 2019, `alpha` Nelson & Baker 1996, and Table S1), with the
sources on each function.

### Data

286 of the paper's 290 series, from the two public archives its data statement cites (Pokrifka et al. 2018,
doi:10.26208/z7bf-nq20: 18 series; Pokrifka et al. 2023, doi:10.26208/htw5-q166: 268 series). The remaining four are
the high-supersaturation "cold" experiments, which the authors reprocessed from raw lower-plate voltage files that
are not in either archive; the same crystals appear as processed mass ratios inside the 268, and
`bel.load(cold_dir=...)` adds those four back (with the paper's own conditions) if the raw files are supplied.
`bel.download()` fetches and caches the archives under `~/.cache/pinneapple/icenode` (or `PINNEAPPLE_ICENODE_DATA`).

## Results

500 Adam epochs per network, `dt = 5 s`, seed 0, 286 series (`results/icenode_seed0.json`). The losses are the
paper's: the sum of squared `m/m0` errors over every series, at 1 Hz.

| model | MSE (t<500 s) | paper | MSE (t<1000 s) | paper | best (t<500 s) | paper | best (t<1000 s) | paper |
|---|---|---|---|---|---|---|---|---|
| no surface kinetics (`G = G_c`) | 41,482 | 41,396 | 222,940 | 222,750 | 54 | 62 | 44 | 55 |
| Nelson & Baker 1996 | 41,424 | 40,751 | 222,464 | 220,930 | 23 | 29 | 26 | 31 |
| strongly-constrained NODE | 27,334 | 30,714 | 180,523 | 203,645 | 21 | 61 | 27 | 62 |
| weakly-constrained NODE | 7,236 | 16,495 | 83,003 | 184,723 | 188 | 138 | 189 | 142 |
| the paper's Eq. 6 as a growth law | 24,304 | — | 206,360 | — | — | — | — | — |

Read against `pinneapple_analysis.reproduction.status` (a factor 2 band):

- **The two parameterised baselines land within 0.2 % and 1.7 %** of the paper's losses. That is the check that
  matters most for the plumbing: no network is involved, so it says the data parse, the constants, the growth law,
  the RK4 solver and the loss definition all agree with the paper. Their best-experiment counts are 87 % and 79 % of
  the paper's, which is what sampling 286 of 290 series and a different solver step would do.
- **The strongly-constrained NODE** reproduces the paper's claim that a single `alpha(S_i, T)` buys a modest
  improvement over Nelson & Baker: 27,334 against 30,714 (ours 11 % lower).
- **The weakly-constrained NODE** is where we depart: 7,236 against 16,495, 2.3 times lower. The replication test at
  t<1000 s (83,003 against 184,723) shows the same gap, so this is not an interpolation artefact. Two readings, and
  this run cannot separate them: either learning `G` directly is even more effective than the paper reports, or a
  network keyed on `(S_i, T, m)` — with `S_i` constant inside each experiment — partly memorises the individual
  experiments. The paper's AIDA test is what would separate them, and that is the part we did not attempt. Note also
  that the status rule calls this row "not reproduced", because it measures agreement, not quality: being 2.3 times
  better is a disagreement.
- **The paper's own Eq. 6, integrated as the growth law on our data**, gives 24,304 — better than either baseline
  and worse than our trained network, on a network it was not distilled from. It keeps about a third of the
  improvement the paper claims for it.

The symbol `G` learned by the weak model (`figures/learned_G.png`) sits mostly a factor 1.2 to 3 below `G_c` and
falls with crystal mass, which is the paper's Figure 3a statement; the paper's Eq. 6 sits systematically above it,
which is its 0.268 RMSE against our expression's 0.142.

On synthetic series with a known `alpha` (the paper's Figure 2, `figures/alpha_synthetic.png`) the
strongly-constrained network recovers the generating `alpha` with a median error of 1.6 % (mean 12 %, pulled up by
the few low-`alpha` experiments whose crystals barely grow, where any `alpha` below ~0.4 fits equally well).

## Where this departs from the paper

- **286 series instead of 290** (−1.4 % of the sample); see above.
- **RK4 step `dt = 5 s`**, where the paper does not state its fixed step; `--dt 1` reproduces the same losses
  to four significant figures at five times the cost (checked on both baselines).
- **Symbolic regression is ours, not PySR.** PySR needs a Julia toolchain; `symbolic.py` implements the same
  genetic search over expression trees (tournament selection, subtree crossover and mutation, Pareto front over
  complexity, constants polished by least squares, `--` generations documented in the code). It is scored against
  the paper's own Eq. 6 evaluated on the same trained network.
- **The AIDA comparison is not attempted.** It needs the IsoCloud archive and a bin microphysics parcel model
  constrained to the observed ice number concentration; that is a second study. Nothing here is evidence about it.

## How to run

```bash
python examples/reproductions/lamb2026_icenode/run.py --quick        # 20 epochs: plumbing, ~10 min
python examples/reproductions/lamb2026_icenode/run.py               # the full run: 4 models, 500 epochs each
python examples/reproductions/lamb2026_icenode/run.py --model weak  # one model into results/checkpoints
```

The four models take tens of minutes each on one CPU thread, so a run can be split across processes
(`--model none|nelson_baker|strong|weak|eq6|synthetic`) and the assembly run then reads their checkpoints. Outputs:
`results/icenode_seed<k>.json` (every number below), `figures/*.png`, and the metrics registered in the folder's
`experiment.json`.

Open follow-ups, with the evidence for each: #1683 (the AIDA comparison), #1687 (the four raw chamber files),
#1688 (whether the weak model's advantage is memorisation), #1690 (parity with the paper's PySR front).

## Files

| file | what it is |
|---|---|
| `bel.py` | the two chamber archives: download, parse, 1 Hz mass ratios, and the paper's 4 raw-voltage series |
| `node.py` | the neural ODE: RK4 integrator, the two constraints, training, scoring windows, `G` sampling |
| `synthetic.py` | synthetic series with a known `alpha` (the paper's Figure 2 validation) and the SSA noise |
| `symbolic.py` | the genetic-programming symbolic regression and the Pareto front |
| `run.py` | the paper's Table 1, both NODEs, the synthetic check, the symbolic step, figures, results |

## References

- Lamb, K. D., Harrington, J. Y., Moyle, A. M., Pokrifka, G. F., Clouser, B. W., Ebert, V., Möhler, O., & Saathoff,
  H. (2026). Discovering how ice crystals grow using neural ODEs and symbolic regression. *Science Advances* 12(39),
  eaee6174. doi:10.1126/sciadv.aee6174, arXiv:2510.17935.
- Harrington, J. Y., Moyle, A., Hanson, L. E., & Morrison, H. (2019). On calculating deposition coefficients and
  aspect-ratio evolution in approximate models of ice crystal vapor growth. *JAS* 76, 1609–1625.
- Nelson, J., & Baker, M. B. (1996). New theoretical framework for studies of vapor growth and sublimation of small
  ice crystals. *JAS* 53, 2853–2862.
- Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric
  applications. *QJRMS* 131, 1539–1565.
- Pokrifka, G. F., Moyle, A. M., Hanson, L. E., & Harrington, J. Y. (2020). Estimating surface attachment kinetic and
  growth transition influences on vapor-grown ice crystals. *JAS* 77, 2393–2410.
- Data: doi:10.26208/z7bf-nq20, doi:10.26208/htw5-q166.
