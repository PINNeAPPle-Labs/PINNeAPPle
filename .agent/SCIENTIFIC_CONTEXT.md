# SCIENTIFIC_CONTEXT — PINNeAPPle, validation batch 4

Scientific/mathematical background for the work on this branch. Numbers quoted are measured
results recorded in test docstrings and this session's investigation. See the other `.agent/`
package (main checkout) for the project's earlier scientific history (SSBroyden benchmarks, LBM
Strouhal numbers, the Open-Meteo climate-twin notebook, batch-3 fixes) — not repeated here.

## 1. Validation philosophy (unchanged from earlier sessions, restated for a fresh reader)

A catalog item is `validated` only when a test plugs an **independent exact/closed-form/published
solution** into the compiled residual (or solver) and gets ~round-off, AND a deliberately-wrong
solution gives a clearly nonzero residual. `tests/test_manufactured_solutions*.py` files use
float64 throughout so "zero" genuinely means round-off (~1e-20 to 1e-35), not "small".

## 2. Exact solutions derived this session (batch 4)

**Buckley–Leverett (E15), `tests/test_manufactured_solutions_batch4.py`:** a stratified reservoir
with `p = -Gx`, `Sw = S(y)` (any layering). Because the total flux `q = k*lambda_t(S(y))*G` doesn't
vary along x, both the saturation-transport and pressure equations hold exactly for ANY `S(y)`,
including a wiggly one — a stronger test than a constant-saturation solution. Corey relperm
`krw=Sw^2, kro=(1-Sw)^2` (the compiler's own documented default).

**Neo-Hookean (E21):** the compiler's own docstring already stated the correct constitutive law
(`P = mu(F - F^-T) + K*ln(J)*F^-T`, compressible Neo-Hookean, plane strain, K an independent bulk
penalty) but implemented the WRONG equilibrium equation. Collocation points are REFERENCE
coordinates X (undeformed configuration). The correct total-Lagrangian equilibrium is
`Div_X P + b0 = 0`. The old code instead built the Cauchy stress `sigma = P F^T / J` and
differentiated THAT with respect to X — but the Piola identity says `Div_X P = J div_x sigma`,
where `div_x` is w.r.t. CURRENT (deformed) coordinates, so differentiating sigma w.r.t. X does not
equal either side of that identity in general. Fixed by manufacturing a genuinely non-homogeneous
displacement field `u = (0.2xy + 0.1y^2, 0.15x^2 - 0.1xy)`, computing `P(X)` and `b0 = -Div_X P` by
autograd (ground truth, independent of the compiler), and checking the compiler's residual against
it. **Measured: residual 0.028 (old, Cauchy-in-reference-coords) vs 4.5e-32 (new, Div_X P) against
the identical manufactured field** — this is a ~6%-of-|b0|^2-magnitude error at finite strain, not
a rounding issue, i.e. it would have been visibly wrong in any real hyperelastic simulation.

**Thermoelasticity (E24):** `T = x` is harmonic (satisfies the decoupled heat equation); the
displacement field `ux = alpha(x^2-y^2)/2, uy = alpha*x*y` gives strain `eps = alpha*T*I` exactly
(pure thermal expansion, no mechanical strain), so the thermal and mechanical strains cancel and
`sigma = 0` everywhere — a "free expansion" check that exercises the plane-stress reduction
`lambda* = 2*lambda*mu/(lambda+2mu)` and the thermal-stress term.

**Biot poroelasticity (E26):** Terzaghi 1D consolidation mode, `p = exp(-ct) sin(x)`,
`u = -alpha exp(-ct) cos(x)/(lambda+2mu)`, with the decay rate `c = (k/mu_f) / (1/M +
alpha^2/(lambda+2mu))` derived by substituting into both the momentum and flow equations
simultaneously (the rate is not free — it's the one value that satisfies both PDEs at once).

**Heston (E32):** two independent exact solutions for two different claims, both checked against
the SAME compiled residual (`heston_pde_2d`) to exercise different derivative terms:
- Claim paying `S*v`: `V = S(a + b*v)`, `b = exp((rho*sigma_v - kappa)*tau)`,
  `a = kappa*theta*(b-1)/(rho*sigma_v - kappa)` — exercises the mixed `d2V/dS dv` term.
- Claim paying `v^2` (the CIR process's second moment): `V = exp(-r*tau)(c0 + c1*v + c2*v^2)` with
  `c2 = exp(-2*kappa*tau)`, `c1 = (sigma^2+2*kappa*theta)(exp(-kappa*tau)-exp(-2*kappa*tau))/kappa`,
  `c0 = theta*(sigma^2+2*kappa*theta)*[(1-exp(-kappa*tau)) - (1-exp(-2*kappa*tau))/2]/kappa` —
  exercises the `0.5*sigma_v^2*v*d2V/dv2` term.

## 3. Solver-level exact solutions (`tests/test_solver_exact_solutions_batch4.py`)

**FEM (Q1 elements):** `u = x^2+y^2` is reproduced at the nodes up to float32 round-off (the
Laplacian of a quadratic is exactly captured by piecewise-bilinear elements with a constant RHS);
the harmonic `u = sin(pi x) sinh(pi y)/sinh(pi)` converges at measured ratio 3.99, 4.00 when halving
h (2nd order, as expected for Q1 with a smooth solution).

**Kansa RBF collocation, the multiquadric Laplacian bug:** for `phi(r) = sqrt(1+eps^2 r^2)` in `d`
dimensions, the correct radial Laplacian (`Delta_x phi(|x-c|)` evaluated AT x, i.e. `phi'' +
(d-1)phi'/r`, the standard radial-Laplacian formula in d dimensions) is
`Delta phi = eps^2 (d + (d-1) eps^2 r^2) / phi^3`. **The code had `(d-2)` instead of `(d-1)`** —
verified by an independent autograd check
(`tests/test_solver_exact_solutions_batch4.py::test_multiquadric_laplacian_matches_autograd`,
computing the same Laplacian via `torch.autograd.grad` twice and comparing to the closed form,
`atol=1e-12`). Measured max error on `u=x^2+y^2`, n=8 -> n=16: **6.6e-4 -> 1.8e-5 (old formula:
stuck around 0.06, not converging at all)**. The other two kernels (gaussian, imq) were already
correct and converge similarly (1.5e-2->3.7e-5, 6.3e-3->1.8e-4).

**Axisymmetric eddy current:** manufactured `A = sin(k(r-a)) sin(mz)` (vanishes on the domain
boundary by construction), with `J_source` back-solved from the exact operator
`L[A] = A_rr + A_r/r - A/r^2 + A_zz` so that `L[A] - i*omega*mu*sigma*A = -mu*J_source` holds
exactly. Convergence ratio > 3.8 at each of two halvings (2nd order FDM, as expected).

**Compressor similarity map:** affinity laws (`phi = Q/(N D^3)` constant implies `psi` constant and
head scales as `N^2`) verified in closed form; `required_speed_for_head` checked against the
closed-form inversion of `H = n_stages * psi(phi) * U^2` at fixed Q.

## 4. Immersed boundary "channel" mode: mass conservation (fixed, #627)

`immersed_boundary_fdm.py` "channel" mode used a divergence-penalty relaxation for the pressure
step and lost mass: pipe R=0.5, L=3, Re=10, 60x22x22 grid, 1500 steps, flow rate 0.85 / 0.53 / 0.39
of the inlet value at x = 0.5 / 1.5 / 2.5. A cell-centred central-difference projection does not
help, because div of a central grad is a stride-2 Laplacian that decouples odd and even grid points
(the 2026-10-02 attempts in `docs/dev/AUDIT_REPORT.md`). The fix projects on cell faces
(`_MACProjection`): compact 7-point Laplacian over the open faces, `p = 0` on the outlet cells,
factorised once. Same case now: flow rate 1.0 at every section, face divergence 6e-15. The
cell-centred `divergence_rms` stays O(1) because of the cells next to the masked wall; check
`face_divergence_rms` instead. Still open: the 1.5-cell wall band shrinks the effective radius
(u_max/u_mean 1.84 against the Hagen-Poiseuille 2 on that grid), and `solve_ibm_external_flow`
keeps its Jacobi projection, not re-checked for mass conservation.

## 5. Key references for this batch

Kansa 1990 (RBF collocation / the "Kansa method"); Biot 1941 and Terzaghi (poroelastic
consolidation); Heston 1993; Boley & Weiner (thermoelastic stresses, plane-stress reduction);
Boyce 2003 / ASME PTC 10 (compressor similarity groups); standard finite-strain continuum mechanics
(the Piola identity relating `Div_X P` and `div_x sigma`, e.g. Bonet & Wood, *Nonlinear Continuum
Mechanics for Finite Element Analysis*).
