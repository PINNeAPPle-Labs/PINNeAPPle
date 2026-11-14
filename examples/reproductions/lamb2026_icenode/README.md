# Lamb et al. (2026), ice crystal growth: measured status

Targets: [TARGETS.md](TARGETS.md), quoted from the paper before any neural-ODE run.
Paper: *Science Advances* **12**(39), eaee6174 (2026), doi:10.1126/sciadv.aee6174, read from arXiv:2510.17935v1.

The paper learns the unknown part of the depositional ice growth law — the transfer coefficient `G` of
`dm/dt = 4 pi r (S_i - 1) G` — with a neural ordinary differential equation fitted to 290 mass-ratio time series of
single ice crystals grown in a levitation diffusion chamber, distils the trained network into a closed form with
symbolic regression, and tests that closed form on independent AIDA cloud-chamber data.

| Target | Status |
|---|---|
| T1 (Table 1: losses and best-experiment counts, four models) | {{T1_STATUS}} |
| T2 (Figure 2: the strongly-constrained NODE recovers a known `alpha`) | {{T2_STATUS}} |
| T3 (Figure 3b / Table S2: a closed form for `G`) | {{T3_STATUS}} |
| T4 (Figure 4: AIDA cloud chamber) | not attempted — needs the IsoCloud archive **and** a bin-microphysics parcel model (see TARGETS.md) |
| T5 (qualitative claims: weak > strong > both baselines; `G` depends on `Gc` and `m`) | {{T5_STATUS}} |

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

{{RESULTS_TABLE}}

## Where this departs from the paper

- **286 series instead of 290** (−1.4 % of the sample); see above.
- **RK4 step `dt = {{DT}} s`**, where the paper does not state its fixed step; `--dt 1` reproduces the same losses
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
