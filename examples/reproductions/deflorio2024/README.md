# De Florio, Schiassi, Calabrò, Furfaro (2024): reproduction

M. De Florio et al., "Physics-Informed Neural Networks for 2nd order ODEs with sharp gradients", *J. Comput.
Appl. Math.* 436 (2024) 115396.

## How the targets were set

- **Source of the targets:** PDF text that the owner supplied on 2026-10-25. Only the numbers are quoted; the
  article is copyrighted and its text is not in this repository.
- **Undefined in the paper, taken from arXiv:**
  - the weight sampling, from ref. [9] (Calabrò, Fabiani, Siettos, arXiv:2012.05871);
  - ε_T and ε_G, from ref. [14] (Mishra & Molinaro, arXiv:2006.16144).
- **Frozen before any run:** [TARGETS.md](TARGETS.md) and its amendment.
- **Produced by `run_deflorio.py`:**
  - the tables, in [RESULTS.md](RESULTS.md);
  - every run, in `all_runs.json`.

## Primary targets: PIELM and X-TFC at full scale

Full scale means 10 000 points and the paper's neuron counts. The score is the median of max|u* − u| over the
seeds.

| ν | Neurons | PIELM: paper / ours | Status | X-TFC: paper / ours | Status |
|---|---|---|---|---|---|
| 1 | 100 | 6.1e-15 / 4.0e-15 | reproduced | 3.9e-16 / 3.3e-16 | reproduced |
| 0.1 | 100 | 3.2e-13 / 4.7e-13 | reproduced | 1.5e-13 / 1.7e-13 | reproduced |
| 0.01 | 1000 | 2.2e-8 / **3.3e-12** | partially reproduced | 1.7e-12 / 2.4e-12 | reproduced |
| 0.001 | 10000 | 4.9e-5 / **3.1e-11** | partially reproduced | 2.8e-12 / 2.1e-11 | reproduced |
| 0.0001 | 10000 | 0.5 / 0.5 | reproduced | 2.9e-3 / **4.4e-4** | partially reproduced |

- There are 10 seeds for ν ≥ 0.01 and 3 for ν ≤ 0.001.
- Result: **7 of 10 reproduced, 3 partially reproduced, and none not reproduced.**
- All three partial entries are cases where **our error is lower than the paper's**. The status rule puts them
  outside the band, but the ordering the paper claims still holds (X-TFC ≤ PIELM).
- **PIELM at ν = 0.01 and 0.001.**
  - The paper reports 2e-8 and 5e-5. In its tables PIELM's max, mean and boundary errors are almost the same
    number, so the error appears to come from the boundary rows.
  - With unweighted boundary rows and `gelsy` least squares, our PIELM stays at 1e-11 to 1e-12, close to X-TFC.
  - So we do not reproduce PIELM's degradation. Its cause is not stated in the paper.
- **X-TFC at ν = 1e-4:** ours is 6.6× more accurate (4.4e-4 against 2.9e-3).
- **ε_T** (eq. 3.8 of ref. [14]): ours is 2 to 4 orders of magnitude below the paper's for ν from 1 to 1e-3.
  For example, X-TFC at ν = 0.01 has 1.2e-9 here against 1.23e-6 in the paper.
  - At ν = 1e-4 we get 0.71 for PIELM (the paper has 1e0) and 6.7 for X-TFC (the paper has 4.87e2).
  - The paper says it follows [14] but does not state its quadrature weights. If it summed the squared residuals
    instead of averaging them, its ε_T would come out larger by a factor of up to √10000 = 100.
  - ε_T is reported, not scored.

![max error vs nu](max_error_vs_nu.png)

## Secondary targets: PINN and Deep-TFC, reduced scale

These runs are smaller than the paper's (declared in TARGETS.md item 7 and the amendment):

| | Ours | Paper |
|---|---|---|
| Width × depth, ν ≤ 0.01 | 50 × 10 | 200-1000 × 10 |
| Width × depth, ν ≥ 0.1 | 20 × 10 | 20 × 10 |
| Training | 2000 Adam steps + 500 L-BFGS steps | 15 000 epochs |

They can only be "reduced-scale"; what we compare is the qualitative pattern.

| ν | PINN learns? paper / ours | Deep-TFC learns? paper / ours |
|---|---|---|
| 1 | yes / yes (L-BFGS: 1e-7 / 2.9e-6) | yes / yes (1.6e-7 / 4.5e-6) |
| 0.1 | yes / yes (1.5e-5 / 3.3e-5) | yes / yes (1.1e-4 / 2.3e-4) |
| 0.01 | no / no (0.5 / 0.5) | yes / yes (6.2e-5 / 7.4e-4) |
| 0.001 | no / no | no / no (0.96 / 0.99) |
| 0.0001 | no / no | no / no |

**The paper's pattern holds:**

- PINN first fails at ν = 0.01.
- Deep-TFC first fails at ν = 0.001.
- Only the ELM-based methods survive below that.

Our PINN and Deep-TFC errors, where they learn, are 10-30× above the paper's, which is consistent with 15×
fewer epochs.

## Physics checks

- Boundary values: X-TFC's are exact (0) by construction. PIELM's boundary errors are recorded per run.
- ε_T, the residual on the training points, is recorded per run, as is ε_G.

## Baselines at equal budget

Same points, neurons and solver. The score is the median max error.

| ν | X-TFC, Gaussian, [9] weights | X-TFC, tanh, unif(−1, 1) | Gated X-TFC (same basis count, gate search) |
|---|---|---|---|
| 1 | 3.3e-16 | 1.6e-14 | 3.9e-5 |
| 0.1 | 1.7e-13 | 1.1e-7 | 7.8e-5 |
| 0.01 | 2.4e-12 | 0.62 | 1.1e-3 |
| 0.001 | 2.1e-11 | 0.62 | not run (10 000 basis functions) |
| 0.0001 | 4.4e-4 | 0.61 | not run |

- The [9] sampling is what makes X-TFC work at small ν. Its slopes scale with the neuron count, up to |α| ≈ 1000
  at 10⁴ neurons. With unif(−1, 1) slopes, the same X-TFC fails from ν = 0.01 on.
- The Gated X-TFC basis of arXiv:2510.01039 (deterministic Gaussian RBFs) is far less accurate here than the
  randomly sampled Gaussian ELM.

## Run it

```bash
python examples/reproductions/deflorio2024/run_deflorio.py --part elm        # ~30 min on 4 CPU threads
python examples/reproductions/deflorio2024/run_deflorio.py --part dnn        # ~2.5 h (resumable)
python examples/reproductions/deflorio2024/run_deflorio.py --part baselines
python examples/reproductions/deflorio2024/run_deflorio.py --part report
```
