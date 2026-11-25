- **Measured cost now covers 29 of the 73 executable catalog methods** (#52, roadmap D1): twelve entries that carried
  only a blocked reason in the first pass are measured on this machine with the canonical case that was missing —
  the Kansa RBF solve (S7), the axisymmetric eddy-current Helmholtz solve (S14), one molecular-dynamics step (S18),
  the P1 magnetostatics and electrostatics solves (S29, S32), the Jiles-Atherton major loop (S36), the Drude/Lorentz
  pole sums (S37), the Gauss magnetic check over a 3-D box (S38), 20 Yee FDTD steps (S40), Brownian first-passage
  Monte Carlo (S42), DMD of a 32-state trajectory (S46) and the large-physics-model Fourier encoding (T22). Each
  fits its exponent over at least three sizes and records the peak memory with the way it was sampled.
- Four probe attempts are recorded as failures instead of numbers: a fixed-step LBM (S5), an immersed-boundary solve
  (S12), SINDy (T14) and the self-scaled quasi-Newton step (T20) each measured the operator overhead — 30 ms flat from
  32 to 128 LBM cells, 3.5 ms flat from 2k to 32k SINDy samples — so no exponent was published for them; the reasons
  in `method_costs.json` say so, with the sizes.
- `instrument_overhead` joins the closed set of blockers: the SPH step runs and scales (0.14 s to 2.5 s from 128 to
  1536 particles), but one 128-particle step costs 40 s under the torch profiler the memory figure comes from, so the
  number is missing because of the instrument and the table says exactly that.
