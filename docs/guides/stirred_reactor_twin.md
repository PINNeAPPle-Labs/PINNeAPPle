# Stirred exothermic reactor: a digital-twin benchmark case

`pinneapple_systems.digital_twin.stirred_reactor` is a benchmark case for a digital twin: a plant
with a known answer, three perturbations, and a results table whose columns are **state error**,
**latency** and **alarm detection**. It came from issue #179, which cites a
[LinkedIn post](https://pt.linkedin.com/posts/gabrielhrozo_engenharia-industria40-digitaltwin-activity-7510809002918916096-JPf0)
describing a two-dimensional Python/Numba CFD model of an exothermic reactor with a three-blade
impeller and five cooling tubes, and claiming about 75 % conversion at 48 °C with agitation against
20 % and a 75 °C hotspot without it. Those numbers are the author's and were not verified anywhere
here. They are used as a *reference to reproduce or contradict*, and every number in this guide was
measured by this repository.

```python
from pinneapple_systems.digital_twin.stirred_reactor import run_benchmark, results_table

bench = run_benchmark()          # 11 controlled cases, about a minute
print(results_table(bench))
```

The example that runs it and writes the figures is
`examples/reactor_twin/reactor_twin_benchmark.py`; the outputs land in
`examples/reactor_twin/_out/`.

## What is solved, and why it is not the post's case

The post's model is a 2-D field solved with an immersed boundary. This machine does not run that,
and the case does not pretend to. What runs here is a **continuous stirred tank**: three states,
five tubes, an impeller that only touches the heat transfer.

```
d c_A/dt = (F/V)(c_Af - c_A) - r                    r = k_f(T) c_A - k_b(T) c_B
d c_B/dt = (F/V)(0    - c_B) + r                    k_b = k_f / K(T),  K = exp(-dH/RT + dS/R)
d T  /dt = (F/V)(T_f - T) + (-dH) r / (rho cp) - Q_cool/(rho cp V) - UA_w (T - T_amb)/(rho cp V)
```

Agitation enters through exactly one path, and it is the physical one: the tank-side film
coefficient of an agitated vessel, `Nu = C Re^(2/3) Pr^(1/3)` with `Re = rho N D^2 / mu`, floored at
natural convection when the impeller stops. The overall coefficient is the two films in series and
each of the five tubes passes its duty by effectiveness-NTU against the coolant inlet:

```
1/U = 1/h_tank(N) + 1/h_coolant          eps = 1 - exp(-UA_tube / (m_dot cp_c))
```

`eps`-NTU is exact for a tube whose shell side is one well-mixed temperature, which is what a
well-stirred tank is, and it gives the right limit when a tube is isolated (`UA -> 0`, `Q -> 0`)
without a singularity. The impeller carries three blades and the tank five tubes because the post
says so; everything else — 3 m³, 1.2 m³/h of feed, 2.5 h of residence time, 1.6 m² of tube area,
5 kg/s of coolant — is an illustrative industrial sizing, stated in `Reactor` and not taken from
anyone.

**The thermodynamics is read off the post's two numbers**, and that is the whole debt of this case
to the post. Two operating points read as two equilibrium conversions of one reversible step
determine `dH` and `dS`:

```python
from pinneapple_systems.digital_twin.stirred_reactor import thermo_from_pair
thermo_from_pair(48.0, 0.75, 75.0, 0.20)      # (-85557.0, -257.3) J/mol and J/(mol K)
```

An exothermic step of `-85.6 kJ/mol` that loses `-257 J/(mol K)` of entropy is the size of a
gas-phase association — industrial methanol synthesis sits near `-90.6 kJ/mol` and `-220 J/(mol K)`
— and an order of magnitude more entropy than a liquid-phase isomerisation. That is the first thing
the pair says. It also means **reproducing the agitated point is arithmetic, not evidence**, and the
report says so on the line where the number appears.

## The protocol

Eleven controlled cases in four families, five hours each at a 30 s sample period, a perturbation
at 1.5 h, alarm at 56 °C and interlock at 65 °C. Seeded, so a run is reproducible:

| family | what changes | cases |
|---|---|---|
| `baseline` | nothing: slow operator moves inside the normal box | 2 |
| `noise` | the sensor class: 0.2 K / 1.0 K / 3.0 K RTD with its analyser | 3 |
| `cooling` | a tube is isolated at 1.5 h; the impeller trips; both | 3 |
| `margin` | the coolant ramps so the stationary margin to the alarm goes 12→8, 8→2 or 5→-3 K | 3 |

The reference is the same model integrated with a stiff solver at `rtol = 1e-10`, sampled where the
twin is sampled. The twin sees only the noisy readings and the four instrumented inputs.

## The results table

Measured by `run_benchmark()` on the machine that ran the benchmark (11 cases, 19 s of wall clock on an idle run, 56 s when the machine was loaded — the latency column is the one that moves):

| family | cases | state error T rms [K] | conv. rms [pp] | open loop [K] | RTD alone [K] | latency [ms/sample] | % of the 30 s cycle | crossed / detected / false / missed | median lead [min] |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2 | 0.06 | 0.15 | 0.01 | 1.03 | 1.1 | 0.004 % | 0 / 0 / 0 / 0 | — |
| noise | 3 | 0.09 | 0.12 | 0.01 | 1.02 | 1.0 | 0.003 % | 0 / 0 / 0 / 0 | — |
| cooling | 3 | 0.24 | 0.28 | 1.51 | 0.94 | 0.8 | 0.003 % | 2 / 2 / 0 / 0 | 85 |
| margin | 3 | 0.05 | 0.14 | 0.01 | 0.97 | 0.8 | 0.003 % | 1 / 1 / 0 / 0 | 69 |
| **overall** | **11** | **0.07** | **0.14** | **0.01** | **1.00** | **1.0** | **0.003 %** | **3 / 3 / 0 / 0** | **82** |

Three columns, three readings of them:

- **state error.** The twin tracks the plant to 0.07 K rms while the instrument it is given is worth
  1.0 K — the Kalman filter is worth twenty times the sensor. The informative row is `cooling`,
  where the twin's model is *not* the plant: the open-loop model (same equations, no assimilation)
  drifts to 1.51 K while the filtered twin holds 0.24 K, because the latent `phi` absorbs the
  cooling it was never told about (0.788 estimated against 0.788 true for the isolated tube; 0.671
  against 0.581 when the impeller also trips, where one `phi` cannot describe both a lost tube and
  a lost film coefficient).
- **latency.** One twin sample — Kalman predict and update plus the 90-minute forecast that decides
  the alarm — costs 1.0 ms median, 0.003 % of the 30 s cycle, on a shared machine whose p95 reached
  41 ms under load (12 ms on the committed run). Over the eleven cases the twin's compute was 14 s
  for 55 h of modelled operation, a real-time factor of 7·10⁻⁵. **The twin is not faster than this 0-D reference** and
  the report says so: a 3-state stiff integration is cheap. What a twin buys is that it runs on the
  sample it is given, not that it beats a model it could also just run.
- **alarm detection.** Three cases drove the plant across the 56 °C alarm — the impeller trip, the
  trip together with a lost tube, and the margin case whose coolant ramp crosses it. All three were
  detected, none was missed, and the four families produced no false alarm in the eight cases where
  the plant never crossed. The lead is large (69 to 88 minutes) because the alarm is not a
  threshold on the reading: the twin rolls its model forward an hour and a half from the estimate,
  with the current inputs held, and alarms when the forecast reaches the setpoint.

## What the post's numbers come out as

| | agitated | impeller stopped |
|---|---|---|
| the post | 48 °C, 75 % | 75 °C, 20 % |
| this model, same feed and tubes | 47.8 °C, 74.7 % | 59.1 °C, 50.5 % |
| cooling conductance | 1 475 W/K | 149 W/K |

The agitated point is where the case is defined to operate, so its agreement is arithmetic. The
impeller-off point is a prediction of the same tank and it **contradicts** the post: stopping the
impeller drops the cooling conductance tenfold and takes the bulk to 59 °C with 50 % conversion,
not to 75 °C with 20 %. It also says what the pair would take: to sit at 75 °C with the impeller
stopped, this tank needs its feed at **73.3 °C** — at which point its well-mixed conversion is
**20.0 %**, the post's number, because the thermodynamics was pinned there. Read as two
energy-balance points of one 40 °C feed, the pair is not reachable: 30 points of conversion at 16 K
hotter, on the same feed, is a transport limitation and a well-mixed model cannot carry one.

## What is not measured, and why

`not_measurable()` returns these, and the report prints them:

- **the 2-D field with an immersed-boundary impeller** — hours per operating point and not the same
  code; it is not run and no number from it is quoted as if it had been.
- **a spatial hotspot** — a 0-D model has one temperature. The post's 75 °C hotspot is a local
  maximum in an unmixed zone; here the bulk is what there is.
- **a conversion limited by mixing** — the well-mixed assumption removes the mechanism by
  construction. If the post's 20 % is a segregation or bypass limit, no 0-D model reproduces it.
- **the post's numbers as an independent measurement** — the thermodynamics is derived from them.

## Why the reference is the reference

Three checks, computed by `reference_trust()` on every benchmark run:

- **two routes to the stationary point.** The algebraic stationary solve gives 47.8424 °C; starting
  the transient 15 K away and marching 16 h gives 47.8422 °C — 0.18 mK apart.
- **tolerance convergence.** The same controlled run at `rtol = 1e-10` and at `rtol = 1e-12`
  differs by 5.5·10⁻⁸ K over the whole history.
- **the first law.** Over a case with a tube isolated and the impeller stopped, 226 MJ of reaction
  heat closes against the sensible change plus what the coolant and the wall removed to 1.9·10⁻³
  relative — the trapezoidal rule on the 30 s grid, not the model.

## Where the twin is not enough

The forecast holds the utilities where they are, because a twin has no way to know the operator
will open a valve tomorrow. A coolant that keeps drifting is therefore caught later than a forecast
that knew the ramp: the margin case that crosses has 69 minutes of lead, less than the 85 the
cooling cases get from a step the twin *does* see coming (a stopped drive is instrumented). That
asymmetry is the honest statement of what a predictive alarm is worth here, and it is measured, not
assumed.
