# Schiassi et al. (2020), X-TFC for epidemiological parameter discovery: reproduction

E. Schiassi, A. D'Ambrosio, M. De Florio, R. Furfaro, F. Curti, "Physics-Informed Extreme Theory of Functional
Connections Applied to Data-Driven Parameters Discovery of Epidemiological Compartmental Models",
arXiv:2008.05554v1 (2020).

## How the targets were set

- **Source:** the arXiv PDF (v1). No code is linked.
- **Frozen before any run:** [TARGETS.md](TARGETS.md). It lists every model setting, the 3 tables of targets and
  7 ambiguities with our fixed choices.
- **Count of scored targets:** TARGETS.md says 33, but its tables list 36 parameter entries (8 SIR + 12 SEIR + 16
  SEIRS). The 33 was a counting slip; no target value changes. All 36 are scored. R0 is reported, not scored.
- **Produced by `run_epi.py`** (2.6 min on 4 CPU cores):
  - [RESULTS.md](RESULTS.md), with every row;
  - every run, in `all_runs.json`;
  - [experiment.json](experiment.json).

## What is implemented

- `pinneapple_simulation.numerical_solvers.xtfc_inverse.solve_xtfc_inverse`, the paper's §2:
  - each compartment is a TFC constrained expression, so its initial value holds exactly;
  - the data losses and the ODE residuals are stacked, unweighted;
  - the ELM output weights of every field and the unknown rates are updated together by Gauss-Newton;
  - it stops when the loss RMS is below 1e-12 or rises (eq. 9).
- `pinneapple_physics.paper_problems_epi`: the SIR, SEIR and SEIRS models at the paper's settings, the noisy data
  sets (`synthetic_data`) and the shooting baseline (`fit_shooting`).
- `xtfc_general.activation_derivative` gains `arctan`, the paper's activation.

## Result

**Primary, the frozen protocol: 14 of 36 targets reproduced.**

| | Clean data (δ = 0) | δ = 0.1 | δ ≥ 1 |
|---|---|---|---|
| SIR (2 rates) | 2 of 2 | 0 of 2 | 0 of 4 |
| SEIR (3 rates) | 3 of 3 | 1 of 3 | 0 of 6 |
| SEIRS (4 rates) | 4 of 4 | 4 of 4 | 0 of 8 |

- **Clean data.** All 9 targets of 0 are met. The errors are 3e-12 for SIR and 4e-9 to 4e-8 for SEIR and SEIRS,
  against the < 1e-6 threshold.
- **SEIR and SEIRS at δ = 0.1.** 5 of 7 are within a factor of 2 of the paper. The SEIR γ and σ errors are 3 to 5
  times the paper's.
- **SIR with noise.** Not reproduced. Our median error is about 9 times the paper's at every δ, for both rates;
  the paper's values sit below our lowest seed at δ = 0.1 and 1.
  - The ratio is the same at δ = 0.1, 1 and 5, and the errors grow linearly with δ, as in the paper.
  - So the gap looks like a difference in the noise or the data, not in the solver. The paper's noise formula
    is printed as `unif[−1, −1]` (ambiguity 3), and its tables come from one unstated realisation.
  - The R0 errors, which are not scored, agree: 3.3e-4 against 3.2e-4 at δ = 0.1, and 3.3e-3 against 3.2e-3 at
    δ = 1.
- **SEIR and SEIRS at δ ≥ 1.** Not reproduced under the frozen protocol. Half or more of the seeds end with rates
  that are off by 0.1 to 2·10³.
  - The cause is the start, not the method. Our start (ambiguity 5) is a data-only fit followed by one linear
    step for the rates. With noisy data, that fit's derivatives are poor, the start lands in another basin, and
    eq. 9 stops Gauss-Newton there.
  - The paper does not state its start.

### Post-hoc variants (not the frozen protocol)

They are run on the same seeds and data. They are not scored as the reproduction.

- **Multi-start:** the same start, plus 8 random starts with each rate drawn from U(0, 2). The fit with the lowest
  final loss is kept. It needs no knowledge of the true values.
  - **23 of 36 reproduced.**
  - It lowers the median final loss at δ = 3 from 7.3 to 1.19 (SEIR) and from 157 to 1.19 (SEIRS). The median
    rate errors match the oracle start's to within 2%.
  - Under multi-start, the targets left are SIR with noise (unchanged) and SEIR γ and σ, which are 3 to 5 times
    the paper's. SEIRS β at δ = 3 is also left: 0.067 against 0.017.
- **Oracle start:** Gauss-Newton from the true rates. This is a diagnostic only. It shows that the X-TFC loss has
  its minimum near the truth at every δ.

## Baseline at equal data: single shooting

`scipy.optimize.least_squares` fits the rates on the integrated model, from the same start as the primary
X-TFC fit.

- **Bounds.** The rates are bounded to [0, 10] and the start is clipped into [1e-3, 10]. From the unbounded start,
  a negative rate makes the model blow up and the integrator stall.
- **Accuracy.**
  - SIR: shooting is 2.5 to 3 times more accurate than X-TFC at every δ (for example, β: 1.5e-3 against 3.9e-3
    at δ = 1).
  - SEIR and SEIRS: shooting is within a factor of 2.5 of X-TFC multi-start, either way. It does not depend on the
    start the way primary X-TFC does.
- **Clean data.** Shooting reaches 2e-13 to 7e-12. X-TFC reaches 3e-12 on SIR and 3e-9 to 4e-8 on SEIR and SEIRS.
- **Time.** The median time per run is similar: 0.05 to 0.13 s for X-TFC (primary) and 0.05 to 1.1 s for shooting.
  Multi-start X-TFC takes 0.45 to 1.5 s.
- **Takeaway.** For these 3- and 4-compartment systems, X-TFC brings no accuracy gain over classical shooting. Its
  what it offers is structural: one linear least-squares solve per step and no integrator in the loop.

![errors vs noise](errors_vs_noise.png)

## Reproduce

```bash
python examples/reproductions/schiassi_epi2020/run_epi.py           # full protocol, writes the files above
python examples/reproductions/schiassi_epi2020/run_epi.py --smoke   # 2 seeds, writes *_smoke files only
```
