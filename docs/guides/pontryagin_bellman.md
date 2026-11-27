# Pontryagin and Bellman networks: optimal control, and knowing when to stop trusting it

`pinneapple_systems.control.optimal` gets an optimal control out of a differential equation instead of out of a
shooting loop around a simulator. Two ways of doing it, both built on the X-TFC machinery already in
`pinneapple_simulation.numerical_solvers.xtfc_ivp` (fixed random ELM basis, Chebyshev-Gauss-Lobatto collocation,
analytic basis derivatives):

- **PoNN** — the Pontryagin Minimum Principle turns the problem into a two-point boundary value problem in the
  state `x(t)` and the costate `lambda(t)`. The boundary conditions are carried by a TFC *constrained
  expression*, so they hold identically for every coefficient vector and never as a penalised loss; for the
  problems below the collocation residual is affine in the coefficients, so **one least-squares solve** is the
  whole solution.
- **BeNN** — the Hamilton-Jacobi-Bellman residual is imposed on a value network `V(t, x)` whose terminal
  condition `V(tf, x) = 1/2 (x - xf)' Qf (x - xf)` is again a TFC constrained expression (the basis is projected
  onto functions vanishing at `tf`). The policy falls out of the value gradient, `u = -R^-1 B' grad V`, so —
  unlike PoNN — it acts on states it was never solved for. That is what makes it a feedback law, and what makes
  it worth checking.

Every number on this page came from a run of this code on this machine; the assertions that pin them are in
`tests/test_control_optimal.py`.

## 1. The problem

```python
import numpy as np
from pinneapple_systems.control.optimal import LQProblem

problem = LQProblem(
    A=[[0.0, 1.0], [0.0, 0.0]], B=[[0.0], [1.0]],      # xdot = A x + B u + c
    Q=np.eye(2), R=[[1.0]],                             # running cost 1/2 (x'Qx + u'Ru)
    x0=[1.0, 0.0], t_span=(0.0, 10.0),
)
```

`LQProblem` is a control-affine system `xdot = A x + B u + c` with a quadratic cost and, optionally, a terminal
cost `1/2 (x(tf) - xf)' Qf (x(tf) - xf)` or a *hard* terminal constraint (`terminal_mask`). `A` is kept exactly as
given — symmetrising it would turn `pdot = v, vdot = u` into a different system and every closed form below would
then agree with a wrong answer. The half in the cost is the convention that makes the BeNN policy read
`u = -R^-1 B' grad V`; the same `P` solves the Riccati equation either way, so `steady_state_lqr` reports
`1/2 x0' P x0` while `pinneapple_systems.control.tuning.lqr` reports `x0' P x0`.

## 2. The LQR regulator, against the algebraic Riccati equation

`steady_state_lqr` is the closed form of the infinite-horizon problem, and `riccati_solution` integrates the
finite-horizon one backwards from `P(tf) = Qf`. Both are checked against references that share no code with the
networks: `scipy.linalg.solve_continuous_are` for the stationary case, the backward ODE integration for the finite
one, and a direct transcription (RK4 march + SLSQP) as a third opinion.

Double integrator, `Q = I`, `R = 1`, `x0 = [1, 0]`:

| quantity | reference | this code | gap |
|---|---|---|---|
| `P` (stationary) | `scipy.linalg.solve_continuous_are` | `[[1.73205, 1], [1, 1.73205]]` | 1.8e-16 absolute |
| `K = R^-1 B' P` | `solve(R, B'P_are)` | `[1.00000, 1.73205]` | 0.0 absolute |
| cost to go, infinite horizon | `1/2 x0' P x0` | `0.866025` | — |
| cost, finite horizon `T = 5 s` | backward Riccati `0.865425` | PoNN `0.865515` | 1.0e-4 relative |
| collocation residual, PoNN | 0 | `4.5e-8` RMS | — |
| boundary conditions, PoNN | 0 | `0.0` | exact by construction |
| cost, finite horizon `T = 10 s` | backward Riccati `0.866025` | BeNN `0.925829` | +6.9 % |
| policy gain at `t0` | `K(t0) = [1.00000, 1.73205]` | BeNN `[0.99834, 1.71662]` | 0.17 % / 0.89 % |
| cost, finite horizon `T = 5 s` | backward Riccati `0.865425` | transcription, 24 knots `0.869538` | +0.48 % |

Read that table as it is: **PoNN reproduces the Riccati answer to 1e-4 and BeNN reproduces the gain to 1 %, but
the BeNN value is 7 % off.** The features carry the gradient better than they carry the value, and the value is
what the policy is integrated from.

```python
from pinneapple_systems.control.optimal import pontryagin, bellman, riccati_solution, steady_state_lqr

solved = pontryagin(problem)            # one least-squares solve, no training loop
solved.cost, solved.residual_rms, solved.boundary_error
solved.state_at([0.0, 5.0, 10.0])       # the constrained expression, re-evaluated anywhere
solved.control_at([0.0])                # u = -R^-1 B' lambda(t)

net = bellman(problem)                  # Gauss-Newton on the HJB residual, analytic Jacobian
net.policy(np.array([0.0]), np.array([[1.0, 0.0]]))     # the closed-loop policy, shape (N, m)
net.cost_to_go()                        # V(t0, x0)
```

## 3. The 1-D intercept, and a hard terminal constraint

With both states pinned at `tf` (`terminal_mask`) the costate endpoint is *free* — it is the multiplier of that
constraint. Imposing `lambda(tf) = Qf (x(tf) - xf) = 0` there instead is a different, unsolvable problem, and it
shows up as a residual that will not come down. With the free endpoint, the X-TFC solve lands on the closed form:

Minimum-energy transfer `p0 = 0`, `v0 = 10`, `pf = 100`, `vf = 0`, `T = 10 s`:

| quantity | closed form `min_energy_transfer` | PoNN | gap |
|---|---|---|---|
| `u(0)` | `+2.0` | `+2.00000` | < 1e-5 |
| `u(T)` | `-4.0` | `-4.00000` | < 1e-5 |
| cost `1/2 int u^2 dt` | `20.0` | `20.000750` | 3.8e-5 relative |
| `x(tf)` | `[100, 0]` | `[100, 0]` | 0.0 |
| collocation residual | 0 | `5.1e-8` RMS | — |
| transcription, 24 knots | `20.0` | `20.059` | +0.29 % |

```python
from pinneapple_systems.control.optimal import LQProblem, pontryagin, min_energy_transfer

transfer = LQProblem(A=[[0.0, 1.0], [0.0, 0.0]], B=[[0.0], [1.0]], Q=np.zeros((2, 2)), R=[[1.0]],
                     x0=[0.0, 10.0], t_span=(0.0, 10.0), xf=[100.0, 0.0],
                     terminal_mask=[True, True])
pontryagin(transfer).summary()          # cost, residual, boundary error, features
min_energy_transfer(0.0, 10.0, 100.0, 10.0, vf=0.0).summary()
```

The minimum-*fuel* 1-D landing is closed form in the other sense: the switching function is linear in time, so
the extremal is *coast then full thrust*, and touchdown at rest gives the burn time in closed form. Lunar values
`h0 = 120 m`, `v0 = -15 m/s`, `g = 1.62`, `a_max = 6.48`:

| quantity | value |
|---|---|
| flight time | 8.3998 s |
| coast, then burn | 3.9850 s, then 4.4148 s |
| fuel `int a dt` | 28.6077 m/s |
| mean thrust | 3.4058 m/s² |
| touchdown state | `h = 2.1e-14 m`, `v = 3.6e-15 m/s` |

## 4. Trust: the residual along the trajectory, and the hand-over to MPC

Once a policy exists, the question is not whether it is optimal but whether it is still inside the region it was
fitted on. `BeNN.hjb_residual` answers exactly that: it is zero for the true value function, and along the
trajectory the policy is *actually flying* it grows as the state leaves the box the value was fitted on.
`landing_demo` puts a lunar landing through it:

```python
from pinneapple_systems.control.optimal import landing_demo

out = landing_demo()
print("\n".join(out["table"]))
```

Measured on this machine:

```
min-fuel (bang-bang): flight 8.40 s, coast 3.99 s, burn 4.41 s, fuel 28.61 m/s
HJB residual, nominal flight: rms 2.130e+02, relative rms 4.094e-01
HJB residual, after the gust: rms 9.263e+03, relative rms 1.781e+01
threshold 4.094e+00 (10x the nominal relative rms); gust of -20.0 m/s at 3.78 s; hand-over at 4.00 s
touchdown, policy alone:  h +36.417 m, v -1.530 m/s
touchdown, with hand-over: h +2.588 m, v -2.577 m/s
cost to go at the start: Riccati 74.38, network 1341.98
```

Three things to take from those lines:

- the residual **orders** the two flights the right way — 0.409 relative RMS inside the fitted corridor against
  17.81 after a -20 m/s gust takes the vehicle out of it, a factor of 43. The claim is about the fit, not about
  optimality: the trajectory that stayed in the box is the one with the small residual.
- the hand-over **fires on the crossing** and not before: the threshold sits at ten times the nominal residual
  (4.094), the gust arrives at 3.78 s, the switch happens on the next step, 4.00 s. `residual_report` reports
  the crossing time and whether there was one.
- the hand-over **is what lands it**: the learned policy alone gives up at +36.42 m and 1.53 m/s, the MPC takes
  it to +2.59 m. The MPC step re-solves the Pontryagin problem from the current state over the shrinking horizon,
  which for these linear-quadratic problems is the exact optimal feedback (`test_mpc_step_is_the_exact_feedback_of_the_linear_problem`
  checks it against `u = -R^-1 B' (P x + s)` from the Riccati reference: `-0.450257` from the MPC against
  `-0.450258` from Riccati).
- the network's cost to go is 18× the Riccati value (1341.98 against 74.38). That is the same weakness the LQR
  table shows, larger: **the value is not accurate, the gradient is, and the residual ordering does not depend on
  either being accurate.**

## 5. Scope, stated plainly

Implemented and measured against closed-form answers: **linear-quadratic problems with an affine drift
(`xdot = A x + B u + c`) and a quadratic running cost.** That covers the three case *classes* this module claims —
the LQR regulator, the energy-optimal 1-D intercept, and the 1-D planetary landing.

Not implemented, and no table from these papers is reproduced:

- **The 3-D non-linear intercept** of D'Ambrosio, Schiassi, Curti and Furfaro (*PoNN for optimal intercept
  problems*, Mathematics 9, 2021) and the closed-loop guidance case of Furfaro, D'Ambrosio, Schiassi et al.
  (*Physics-informed neural networks for closed-loop guidance and control*, AIAA 2022). Their dynamics are
  non-linear in the state (`mu/r^3` gravity, rotating frames). The collocation residual is then no longer affine
  in the coefficients, so the single least-squares solve that makes PoNN exact here is replaced by a Gauss-Newton
  iteration whose convergence is a research question, not a test. `pontryagin(..., linear=False)` runs that
  iteration; it is not exercised against any published case here, and it should not be read as reproducing one.
- **CasADi/IPOPT.** The papers' comparisons are against a direct transcription solved with IPOPT. Neither is a
  dependency of this library and neither is installed on the machine this was developed on, so the independent
  reference used throughout is `direct_transcription` — RK4 march plus SLSQP on the discretised problem. It is
  an honest third opinion (0.48 % and 0.29 % from the closed forms at 24 knots) but it is not IPOPT.
- **The stationary (infinite-horizon) value network.** It existed and was removed, and the reason is measured:
  the stationary HJB residual is solved to machine zero by *every* symmetric root of the algebraic Riccati
  equation. Fitted on the double integrator, a residual-only value network converged to the anti-stabilising root
  at residual `3.8e-16`, with gain `[-0.62, +0.62]` and cost to go `-0.69` against the Riccati `[1.00, 1.73]` and
  `+0.87`. A terminal condition removes the ambiguity, so only the finite horizon is offered. The infinite-horizon
  answer keeps its closed form in `steady_state_lqr`.
- **Accuracy of the BeNN value.** The measured gaps are in the table above (+6.9 % on the regulator, 18× on the
  landing). The feature set is the limit: projecting the exact Riccati value onto the features and measuring the
  best possible fit gives 3.7 % relative error on the landing corridor with `Qf = diag(0.5, 5)`, and 25 % with
  `Qf = 2000 I` — which is why `landing_demo` uses a modest terminal cost. Two knobs are exposed and documented
  because they change the answer measurably: `weighted=True` divides each collocation residual by the size of the
  Hamiltonian terms there (it is what keeps the landing fit finite, and it costs the regulator accuracy — 6.9 %
  above Riccati unweighted against 27.6 % below it weighted), and the number of time-to-go powers.

## 6. API

| object | what it is |
|---|---|
| `LQProblem` | `xdot = A x + B u + c`, cost, optional terminal cost or hard terminal constraint |
| `riccati_solution(problem)` | the finite-horizon closed form: `P(t)`, `s(t)`, `q`, `K(t)`, cost, by backward ODE integration |
| `steady_state_lqr(problem)` | the infinite-horizon closed form, reusing `pinneapple_systems.control.tuning.lqr` |
| `pontryagin(problem)` | the PoNN solve: `state`, `costate`, `control`, `cost`, `gain`, `residual_rms`, `boundary_error`, `state_at`/`costate_at`/`control_at` |
| `bellman(problem)` | the BeNN solve: a value network with `value`, `grad_value`, `policy`, `hjb_residual`, `hjb_jacobian`, `cost_to_go`, `gain` |
| `residual_report(net, trajectory)` | RMS and max residual along a flown trajectory, and the time it crossed a threshold |
| `mpc_control(problem, t, x)` | one shrinking-horizon step: re-solve the Pontryagin problem from the current state |
| `min_energy_transfer`, `min_fuel_landing` | the closed forms, with their cost, fuel and switching times |
| `direct_transcription(problem)` | the independent discretised reference (RK4 + SLSQP) |
| `landing_demo()` | the residual-triggered hand-over on the 1-D lunar landing, with its table |

```python
import numpy as np
from pinneapple_systems.control.optimal import bellman, residual_report, mpc_control

net = bellman(problem, box=np.array([[-2.0, 4.0], [-3.0, 3.0]]))
report = residual_report(net, {"t": times, "state": states}, threshold=2.0)
if not report["within"]:
    control = mpc_control(problem, report["crossed_at"], state_now)["control"]
```

See also: [control engineering](control_engineering.md) for the linear analysis, loops and tuning this sits next
to, and [design studies](design_studies.md) for what to do once a control law is a free parameter of a design.
