# X-TFC for epidemiological parameter discovery (Schiassi et al.): frozen targets

Written and committed **before any run**. Every number is quoted from the PDF named below, with its page and table.

## Source

- E. Schiassi, A. D'Ambrosio, M. De Florio, R. Furfaro, F. Curti, "Physics-Informed Extreme Theory of Functional
  Connections Applied to Data-Driven Parameters Discovery of Epidemiological Compartmental Models",
  arXiv:2008.05554v1 (12 Aug 2020), 14 pages.
- Downloaded from `https://arxiv.org/pdf/2008.05554` on 2026-10-30.
- No code is linked. The authors used MATLAB: `ode113` for the data, and the MATLAB Deep Learning Toolbox for
  automatic differentiation.

## Method as the paper states it (§2, p. 2-4; §4, p. 8-9)

- **Representation.** Each compartment is a constrained expression with an initial-value constraint:
  X = (h − h0)ᵀ ξ_X + X0 (eqs. 17, 21, 25).
  - h is a single-layer ELM, with input weights and biases drawn once and frozen.
  - The activation is ArcTan, and the weights are drawn from "a uniform random distribution ranging within
    [-10,10]" (p. 9).
- **Losses** (eqs. 4-6, 18, 22, 26).
  - Data losses: L_data = X̃ − X at the data points, one per compartment.
  - DE losses: the ODE residuals.
  - Both are stacked into one vector, unweighted.
- **Unknowns.** Ξ = [ξ, λ]: the output weights of every compartment, plus the unknown parameters.
- **Solve.** Iterative least squares: Ξ_{k+1} = Ξ_k − (JᵀJ)⁻¹ Jᵀ L (eqs. 7-8).
- **Stopping rule** (eq. 9). Stop when L2[L(Ξ_k)] < ε, or when L2[L(Ξ_{k+1})] > L2[L(Ξ_k)].
- **Data.**
  - Clean data (f̃_orig) come from integrating the model with `ode113`.
  - Perturbed data: f̃_pert = f̃_orig + δ unif[−1, −1] (eq. 28). The printed interval is a typo for [−1, 1]; see
    ambiguity 3.
- **Score.** The absolute error between the true and estimated parameters, plus R0 = β/γ.

## Models and settings (§3, p. 5-8; §4, p. 9-12)

### SIR (eq. 16, §4.1 p. 9)

- **Equations.**
  - S' = μN − βSI/N − μS
  - I' = βSI/N − γI − μI
  - R' = γI − μR
  - with N = S + I + R.
- **Values.**
  - μ = 0.1 (known), β = 1/2 and γ = 1/3 (unknown).
  - Initial state: S0 = 100, I0 = 5, R0 = 0.
  - "analysis time: 15 days".
- **Run settings** (Table 1 caption): n = 100 points, L = 50 neurons.

### SEIR (eq. 20, §4.2 p. 9-11)

- **Equations.**
  - S' = μ(N − S) − βSI/N − νS
  - E' = βSI/N − (μ + σ)E
  - I' = σE − γI − μI
  - R' = γI − μR + νS
- **Values.**
  - μ = ν = 0.5 (known); β = 0.3, γ = 0.6 and σ = 0.9 (unknown).
  - Initial state: S0 = 70, E0 = 30, I0 = 10, R0 = 0.
  - 15 days.
- **Run settings:** n = 100, L = 80.

### SEIRS (eq. 24, §4.3 p. 11-12)

- **Equations.**
  - S' = μN − βSI/N + ζR − νS
  - E' = βSI/N − σE − νE
  - I' = σE − γI − μI
  - R' = γI − νR − ζR
- **Values.**
  - μ = 0.5 (known); β = 0.3, γ = 0.6, σ = 0.9 and ζ = 0.5 (unknown).
  - Initial state: S0 = 70, E0 = 30, I0 = 10, R0 = 0.
  - 15 days.
- **Run settings:** n = 100, L = 80.
- ν is not given (see ambiguity 1).

## Targets: the absolute error of each estimated parameter (Tables 1-3)

The R0 columns are reported next to the targets, but they are not scored separately.

**Table 1, SIR (p. 9)**

| δ | \|err β\| | \|err γ\| | \|err R0\| |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0.1 | 4.2e−5 | 4.2e−5 | 3.2e−4 |
| 1 | 4.4e−4 | 4.2e−4 | 3.2e−3 |
| 5 | 2.2e−3 | 2.1e−3 | 1.56e−2 |

**Table 2, SEIR (p. 11)**

| δ | \|err β\| | \|err γ\| | \|err σ\| | \|err R0\| |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 0.1 | 2.9e−3 | 3.7e−4 | 4.9e−4 | 4.5e−3 |
| 1 | 2.9e−2 | 3.8e−3 | 4.8e−3 | 4.5e−2 |
| 3 | 8.7e−2 | 1.2e−2 | 1.4e−2 | 1.4e−1 |

**Table 3, SEIRS (p. 12)**

| δ | \|err β\| | \|err γ\| | \|err σ\| | \|err ζ\| | \|err R0\| |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0.1 | 1.1e−3 | 2.0e−3 | 2.6e−3 | 2.8e−3 | 1.7e−4 |
| 1 | 9.3e−3 | 1.8e−2 | 2.5e−2 | 2.5e−2 | 2.1e−4 |
| 3 | 1.7e−2 | 4.7e−2 | 6.8e−2 | 5.8e−2 | 9.1e−3 |

That makes 33 scored targets: 6 for SIR, 12 for SEIR and 15 for SEIRS. The iteration counts and CPU times are
reported, not scored.

**Text claims** (p. 9-12):

- SIR: "the absolute errors on the parameters result to have at least 2 digits of accuracy".
- SEIR: "at least one digit of accuracy".
- SEIRS: "at least two digits of accuracy".

## Ambiguities and our fixed choices

1. **SEIRS ν** is used in the equations but not given.
   - We take ν = μ = 0.5. The text sets the rates "equal ... to simulate a constant number of population", and
     with ν = μ, N' = 0.
   - ν is known, not estimated.
2. **Time and points.**
   - t ∈ [0, 15], with n = 100 points uniform on [0, 15], endpoints included.
   - The data and the DE losses use the same points.
   - Each compartment is observed at every point.
3. **Noise.**
   - We read eq. (28) as δ · U(−1, 1), independent for each point and each compartment.
   - For δ > 0 we run **10 noise seeds** per level; the paper's tables come from a single unstated
     realisation. The score is the median over seeds of |err|.
   - δ = 0 is deterministic.
4. **ELM inputs.**
   - The paper does not say how t enters the network. TFC practice maps t onto [−1, 1], and we use that mapping
     (`map_to_unit=True`). Our X-TFC reproduction showed unif(−10, 10) weights fail on raw times of order 10.
   - Weights and biases ~ U(−10, 10), from 10 ELM seeds per configuration, paired with the noise seeds.
5. **Initial guess** (not stated). It has two steps:
   - ξ comes from a data-only least-squares fit of each compartment.
   - The parameters come from a linear least-squares fit of the DE losses at that fit. The DE losses are linear
     in the parameters.
6. **Stopping.** Eq. (9) is applied with ε = 1e−12 on the RMS of L, and at most 200 iterations.
7. **"Exact" data.** We use `scipy.integrate.solve_ivp` with method `DOP853`, rtol 1e−13 and atol 1e−13,
   instead of MATLAB's ode113.
   - The clean-data targets are "0". An error of zero can only be met to the accuracy of the data and the
     least-squares solve.

## Status rule (`pinneapple_analysis.reproduction.status`, PR #891)

The quantity scored is the median over seeds of |err| for each parameter at each δ.

- **Paper value 0:** "reproduced" means our value is < 1e−6.
- **Paper value below 1e−6:** "reproduced" means within one order of magnitude.
- **Otherwise:** "reproduced" means within a factor of 2.
- `ordering_holds` is None, because these targets have no ordering claim.

A noisy target is therefore "reproduced" only if our median error is within a factor of 2 of a single
realisation in the paper. That is a strict test, since one realisation is noisy itself. We also report our
min-max over seeds.

## Baseline at equal budget (after the reproduction is scored)

- The same data, with the parameters fitted by `scipy.optimize.least_squares` on the ODE solution: a classical
  shooting fit with `solve_ivp` inside the loop.
- Started from the same initial guess as item 5, with the same seeds.
- We report |err| and wall time.
