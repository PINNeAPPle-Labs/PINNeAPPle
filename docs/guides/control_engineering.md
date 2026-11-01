# Control engineering: analysis, loop architecture, tuning

`pinneapple_systems.control` works on any dynamic model `dx/dt = f(x, u, d, t)` written as a Python function.
`pinneapple_simulation.naval` builds two naval cases on it: submarine depth-keeping and ship initial stability.
Everything here runs in simulation. There is no path to real equipment.

## 1. Is the operating point stable?

```python
import numpy as np
from pinneapple_systems.control import equilibrium, linearize, stability, controllability, loop_margins

f = lambda x, u, d, t: np.array([x[1], -4.0 * x[0] - 0.8 * x[1] + u[0]])
eq = equilibrium(f, x0=[0.1, 0.0], u0=[0.0])              # trim, with the residual
lin = linearize(f, eq["x"], [0.0])                        # A, B, E and a step-size error estimate
st = stability(lin["A"])                                  # eigenvalues, growth rates, doubling times
controllability(lin["A"], lin["B"])["controllable"]
loop_margins(([1.0], [1, 3, 2, 0]), delay=0.1)            # gain / phase / delay margins, peak sensitivity Ms
```

`stability` reports each mode's growth rate, time constant, damping ratio and, when unstable, its **doubling
time**. It also accepts a discrete-time matrix (`dt=`), whose eigenvalues it maps to continuous time.

Checks in `tests/test_control_engineering.py`:
- The mass–spring–damper eigenvalues and step overshoot match the closed forms.
- The margins of `1/(s(s+1)(s+2))` are GM = 6 at ω = √2, and PM from the gain crossover.
- A delay lowers the phase margin by ωc·τ.

## 2. What does the controller command?

Between the controller and the plant sits an `Actuator`:
- range limits;
- a rate limit;
- a first-order lag;
- a dead band;
- a transport delay;
- optionally **integrating**: the command is the rate of the applied value, as a pump flow is to the mass in a tank.

```python
from pinneapple_systems.control import Plant, Actuator, PID, simulate_loop, loop_eigenvalues

plant = Plant(rhs=lambda x, u, d, t: np.array([u + d, x[0]]), x0=[0, 0], output=1, rate=lambda x: x[0])
pd = PID(1.0, 0.0, 1.4, derivative="measurement")
loop_eigenvalues(plant, pd, Actuator(), x_eq=[0, 0], u_eq=0, reference=0, dt=0.01)["stable"]                 # True
loop_eigenvalues(plant, pd, Actuator(integrating=True), x_eq=[0, 0], u_eq=0, reference=0, dt=0.01)["stable"]  # False
```

`loop_eigenvalues` differentiates the complete one-step map of the loop: the plant, the actuator states, the
controller states and the integrator. The integrating actuator turns `s² + 1.4 s + 1` into `s³ + 1.4 s + 1`. The
`s²` term is missing, so by the Routh criterion no gains make that loop stable.
`compare_architectures` runs several (controller, actuator) choices on the same plant and disturbance. It reports
both the linear verdict and the simulated excursion.

The controllers:
- **`PID`:** derivative on the measurement or on a measured rate, a derivative filter, setpoint weighting,
  anti-windup (conditional or back-calculation), feed-forward and a bumpless start.
- **`Cascade`.**
- **`StateFeedback`.**

## 3. Tuning and robustness

| Need | Call |
|---|---|
| FOPDT model from a step test | `fit_fopdt(t, y, du)` |
| Rules | `simc_pid(K, tau, theta)`, `imc_pid(..., kind="pi"/"pid")`, `ziegler_nichols(ku=, tu=)` or `(K=, tau=, theta=)`, `ultimate_gain(L)` |
| Poles of `m y'' + c y' + k y = b u`, also for `k < 0` | `place_pid_second_order(m, c, k, b, wn, zeta, p)` |
| LQR | `lqr(A, B, Q, R)`, `dlqr(Ad, Bd, Q, R)` |
| Minimise IAE / ITAE / energy under constraints | `tune_by_optimization(simulate, initial, bounds, constraints={"overshoot": 10})` |
| Stability under uncertain parameters | `robustness(run, {"kh": (lo, hi), ...}, n=64)` |

## 4. Naval: submarine depth-keeping

A compressible hull in compressible seawater has buoyancy `B(z) = ρ0(1 + κw z)·V0(1 − κh z)·g`. With `κh > κw`,
`dB/dz < 0`, so the depth equilibrium is unstable. The trim tank, driven by a flow-limited pump, has to be adjusted
all the time.

```
python examples/control/submarine_depth.py
```

The parameters are illustrative (`pinneapple_simulation.naval.ILLUSTRATIVE`, not a real boat). Measured by the
script, with a 1500 kg extra mass at t = 60 s and a 0.25 s step:

| Case | Linear closed loop | Depth | Tank at the end |
|---|---|---|---|
| No control | doubling time 146.8 s | reaches the 600 m limit at t = 642 s | 20,000 kg |
| PID → target tank mass (pump as a rate-limited servo) | stable, max Re −0.0099 1/s | 99.77–102.39 m | 18,500 kg |
| Same PID → pump flow | **unstable**, max Re +0.0091 1/s (doubling 76 s) | surfaces at t = 591 s | — |
| Gains placed on the linear model (ωn = 8× growth rate, ζ = 0.8) | stable, max Re −0.0094 1/s | 100.00–101.97 m | 18,500 kg |

Two more results from the same script:
- **Optimisation from the example's gains:** under a cap of 1.2× the pump travel, IAE drops from 360 to 213 m·s.
- **Robustness:** with the hull compressibility ±30 % and the drag /3 to ×3, 32 of 32 samples stay stable. The
  95 % quantile of the excursion is 2.40 m.

Why the flow command fails: with the derivative on the vertical speed, the target-mass loop has the polynomial
`m s³ + g Kd s² + (k + g Kp) s + g Ki`. Commanding the flow adds a pole at the origin:
`m s⁴ + c s³ + (k + g Kd) s² + g Kp s + g Ki`. The drag is quadratic, so `c = 0` at rest and the `s³` coefficient
vanishes. The lesson is that the variable the controller commands matters as much as its gains.

## 5. Naval: ship initial stability

`pinneapple_simulation.naval.hydrostatics` provides:
- box-barge hydrostatics: `KB = T/2`, `BM = B²/12T`;
- `GM = KB + BM − KG − FSC`, with the free-surface correction of slack tanks;
- the righting arm, small-angle and wall-sided;
- the roll period `2π k / √(g GM)`.

`initial_stability` checks the corrected GM against the 0.15 m criterion and explains when a slack tank makes the
upright position unstable.

## 6. Inverted pendulum on a cart

`pinneapple_systems.control.cartpole.cartpole_study()` runs the cart-pole case. Measured by
`examples/control/cartpole_swing_up.py`:

- **Linearisation at upright:** a real pole at +4.43 1/s, equal to √(g/l). An error doubles every 156 ms. Without
  control, 1° becomes 90° in 1.19 s.
- **LQR:** gains from the Riccati equation (`lqr`) put all four closed-loop poles in the left half-plane.
- **Energy swing-up** from 3° off hanging, with the cart limited to 3 m/s²: the LQR takes over at 4.32 s, and the
  pendulum stays within 2° from 6.74 s.
- **Push at t = 12 s:** peak 15.9°, back within 2° in 1.19 s.

The Flow node is `control.cartpole`, and the flow is `pinneapple_flow/flows/examples/control/cartpole.json`.

## Flow nodes

- `control.linear_analysis`
- `control.pid_tuning`
- `naval.submarine_depth`
- `naval.ship_stability`
- `control.cartpole`

The flow is `pinneapple_flow/flows/examples/control/submarine_depth.json`.
