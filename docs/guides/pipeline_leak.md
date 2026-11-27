# A leak in a crude-oil pipeline: when it is heard, where it is, and whether it was ever there

`pinneapple_physics.pipeline_leak` solves the transient of a buried line and then answers the question an operator
actually asks: *was that a hole, or is transmitter A broken?* The two questions are answered by physics, not by a
classifier — **a hole moves both ends of the line and lowers the head there; a broken transmitter moves one** — and
every number below was measured on the example, not asserted.

The line is 20 km of 400 mm steel carrying 850 kg/m³ crude at 0.25 m³/s. Korteweg gives the pressure-wave speed,
1 140.31 m/s, so a wave crosses the line in 17.5 s and comes back in 35.1 s. Colebrook gives `f = 0.0200`, which over
20 km at 1.99 m/s is 202.01 m of head — 10 m per kilometre, which is what a real main of this size costs.

```python
from pinneapple_physics.pipeline_leak import Pipeline, Leak, Sensors, simulate

pipe = Pipeline(length=20000.0, diameter=0.4)          # the defaults are the line above
sensors = Sensors(upstream=480.0, downstream=480.0)    # 480 m inside each end, on the line
run = simulate(pipe, n_reaches=500, t_end=200.0, leak=Leak(distance=9800.0, area=7.854e-5, t_open=20.0),
               monitor=[480.0, 19520.0])
```

## The transient, and the two things that can be checked about it

The method of characteristics on the characteristic grid, `dx = a dt`, with Darcy-Weisbach friction in the
compatibility equations. The hole is an **internal boundary condition**, not a source term: the flow is discontinuous
across the node and continuity there reads `Q_up = Q_down + Q_L`, with `Q_L = C_d A_h sqrt(2 g (H - H_amb))`.
Substituting the two characteristics leaves one quadratic in `sqrt(H_L - H_amb)`, so the node is solved in closed form
and nothing is iterated:

```
H_L = H_0 - (B/2) C_v sqrt(H_L - H_amb),    H_0 = (C_P + C_M)/2,    C_v = C_d A_h sqrt(2 g)
```

The first step after the hole opens is exact, and it is what the test checks:

| quantity | computed | closed form | relative error |
| --- | --- | --- | --- |
| head at the hole, 10 mm at 9.8 km | `-1.261087 m` | `-1.261087 m` | `1.3e-12` |
| flow the upstream side takes | `+0.00136286 m³/s` | `Q_L/2` | `2.1e-12` |
| flow the downstream side takes | `-0.00136286 m³/s` | `-Q_L/2` | `2.1e-12` |
| front width at 8 s | `18 160 m` | `2 a t = 18 245 m` | `0.5 %` |

The front speed comes out 1 120, 1 130, 1 133 and 1 135 m/s at 2, 4, 6 and 8 s, against the Korteweg 1 140 m/s: the
grid measures the width between the outermost node that has moved, so it is short by one cell until the front is far
enough from the hole to forget it.

Then there is the check that the friction term is what shrinks the wave and not an indexing slip: with `f = 0` the
transmitter receives **exactly** the head the hole dropped, 1.6038 m, to round-off. With this line's friction it
receives 0.838 m. **A third of the wave is gone over 9.3 km**, and that is steady friction alone — `R q²` per cell
turns against the flow change for as long as the front travels. A real line adds unsteady friction and, in a lined
pipe, viscoelasticity on top, and loses more.

`figures/transient.png` is the space-time map of the same event: two fronts leaving the hole at `±a`, the reflections
at the two ends, and, on the right, the same two places as the transmitters record them.

## The transmitters, and the four ways they lie

`Sensors` says what the acquisition does — 0.15 m of head noise (0.04 % of a 400 m range, a good transmitter), 12-bit
quantisation (a 0.098 m step), a 20 Hz sample — and `Fault` says what is wrong with one channel:
`offset`, `drift`, `noise`, `stuck`. Faults are applied *after* `t_start`, on top of the noise, which is the way an
instrument fails: the process is what it is and the reading is not.

The separation is four pieces of physics, and `detect` reports all four so that the verdict can be argued with:

* **one side or both.** A hole pushes a rarefaction in both directions, so within `(x_b - x_a)/a` of the first
  departure the other transmitter must see it too — and in the same direction: a hole can only lower the head. The
  leak test is one-sided because the physics is; the instrument test is two-sided.
* **the arrival difference.** `x_L = (x_a + x_b - a (t_b - t_a))/2`, with the arrival timed at `-3σ` and interpolated
  between the two samples that bracket it.
* **the mass balance.** `Q_in - Q_out` is the leak and nothing else, but only once the line has forgotten the event,
  which is why it confirms rather than triggers.
* **a flat channel.** A transmitter that has not moved by one quantisation step for five seconds has latched.

Six fault cases on records of normal operation, six noise draws each, all called `instrument`:

| fault | called instrument |
| --- | --- |
| bias `+3 m` on A | 6/6 |
| bias `-3 m` on A | 6/6 |
| drift `-20 mm/s` on A | 6/6 |
| noise `×8` on A | 6/6 |
| latched at 40 s, A | 6/6 |
| latched at 40 s, B | 6/6 |

`figures/residuals.png` is the argument in four panels: the hole moves both ends down; a bias moves one up; a drift
walks one away; a latch freezes one while the other keeps reporting. Nothing in the picture needs a training set.

A latched channel is worth a note. In perfectly steady operation the residual against the model is a constant and the
change detector has nothing to see, so a latch is caught by the *variance* of the channel and not by the residual.
That is a sensor diagnostic, not a leak detector, and the module says so instead of pretending the residual did it.

## Time: the floor is the flight, not the detector

| hole | position | flight to the near transmitter | measured delay | median error |
| --- | --- | --- | --- | --- |
| 10 mm | 3.0 km | 2.21 s | 2.30 s | 2.6 m |
| 10 mm | 9.8 km | 8.17 s | 8.30 s | 4.5 m |
| 10 mm | 15.5 km | 3.53 s | 3.65 s | 40.6 m |

Across the whole sweep — eight hole sizes, three positions, six noise draws — the median delay is 0.14 s above the
flight time. The detector is not what makes a leak late; the line is. A hole in the middle cannot be reported in less
than 8.2 s by anything that listens at the two ends, and a hole at 3 km cannot be reported in less than 2.2 s.

### Where the position is worth having, and where it is not

The position is a different story, and it does not have one number. Two fronts of one leak do **not** arrive equally
loud: the one that crossed the longer part of the line is shorter — 3.4 % per kilometre, measured — and it is quieter
still at 15.5 km than at 3.0 km because the hole there sits at 103 m of head against 230 m and so passes less oil to
begin with. When the quieter front falls into the noise, the change point the detector times is no longer the arrival,
and a position built from two of those is worth kilometres. So the localiser reads the height of the smaller front and,
below **2σ** at its own transmitter, **keeps the leak and the rate and withholds the place** (`front_sigma`,
`position_reliable`).

That is not a claim of accuracy, it is a refusal, and both sides of it are measured. Over 144 case/seed combinations at
a decision interval of 12σ:

| | count | median \|error\| | worst \|error\| |
| --- | --- | --- | --- |
| positions **accepted** | 72 | **9.65 m** | **89.4 m** |
| positions **withheld** — what the guess would have said | 17 | 33.4 m | **2 545.9 m** |

The gate buys the 2.5 km tail and pays for it with sixteen answers that would have been inside 72 m. It is a 2σ rule,
not a filter: it cannot tell a 1.9σ front from a 1.2σ one, and it rejects both.

By band, over the same 144 combinations:

| hole at | farther transmitter | accepted | median \|error\| | worst \|error\| | withheld | its median / worst |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0 km | 16.5 km | 25 of 48 | 7.5 m | 67.1 m | 5 | 43.9 m / 72.1 m |
| 9.8 km | 9.7 km | 24 of 48 | **3.7 m** | 53.4 m | 7 | 33.4 m / 142.1 m |
| 15.5 km | 15.0 km | 23 of 48 | 16.3 m | 89.4 m | 5 | 26.5 m / 2 545.9 m |

The middle of the line is the best place to be placed, and 15.5 km the worst — not because the leg is longest there
(3.0 km is longer) but because the hole is at the lowest head and the leak therefore the smallest. Both halves of that
are in the fronts of one 10 mm hole: at 3.0 km it reads 9.21σ at the transmitter 2.5 km away and 4.76σ at the one
16.5 km away; at 15.5 km it reads 5.71σ at 4.0 km and **3.62σ at 15.0 km** — quieter over a leg 1.5 km *shorter*,
because the hole there sits at 103 m of head instead of 230 m.

The remaining floor is the acquisition: one sample of the arrival is worth `a Δt_s / 2 = 28.5 m`, and interpolating
inside it is what buys the rest. A hole **outside** the two transmitters sends one front past both of them, and no time
difference can place it — `outside_transmitters` is set and the position withheld for the same reason.

## False alarms: what the model has to know

Sixty records of 1 400 s each (23.3 hours) of normal operation, including a 10 % valve closure and reopening every
170 s:

| decision interval | alarms | per hour |
| --- | --- | --- |
| 8σ | 17 | 0.73 |
| 12σ | 0 | 0.00 |
| 16σ | 0 | 0.00 |
| 20σ | 0 | 0.00 |

Zero in 23 hours is not zero, and the honest way to read it is as an upper bound: the CUSUM's run length at 12σ on
white noise is of order 10⁸ samples, so no simulation of a reasonable length will show one.

Now the same 23.3 hours judged against a model that was **never told about the manoeuvres**: 60 of 60 records alarm,
35 of them as `operation` and **25 as a leak** — 2.6 alarms per hour, a third of them wrong about what happened. An
unrecorded valve closure moves both ends; an unrecorded reopening sends a rarefaction down the line that is
indistinguishable from a hole by pressure alone. What refuses it is the mass balance — and there the balance is
contaminated too, because a manoeuvre the model does not know is a storage transient in `Q_in - Q_out` that the
threshold was set to ignore. **The model's knowledge of the boundary conditions is the whole ballgame**, and this is
the number that says so.

## The curve: what a quieter detector costs

`figures/tradeoff.png` and `figures/localisation.png`. The smallest hole the wave finds is set by the transducer
noise, not by the decision interval: a hole drops the head by `a Q_L/(2 g A)` at the hole, less what the travel takes,
and what reaches the transmitter must clear that transmitter's noise, so

```
Q_min = 2 g A σ / (a (1 - attenuation))
```

with the measured 30 % attenuation that is `5.60e-4 m³/s` — 0.22 % of the flow, a hole about **4.6 mm** across. The
measured floor is between 4 and 6 mm, and the two sizes say different things. At 8σ a **4 mm** hole is found on all
18 case/seed combinations and placed on four, the best of them 6.2 km off: the balance hears it, the wave does not
hear it well enough at both ends to time it. A **6 mm** hole is found at all three positions at every decision interval
in the sweep, and placed at all three at 8σ. The amplitude argument is optimistic by about a factor 1.3 because the
leak also has to clear the CUSUM's slack at *both* ends and the far one sees less.

Raising the decision interval from 8σ to 12σ costs the smallest holes and buys the quiet: at 8σ a 3 mm hole is found
nine times in eighteen and the false alarms are 0.73/h; at 12σ it is found once in eighteen and there is no false
alarm in 23 hours. That is the trade, and both ends of it are on the plot.

The balance deserves its own line. `Q_in - Q_out` averages down to about `4e-5 m³/s` over the 45 s the line needs to
settle — a thirteenth of the pressure path's floor — but it knows nothing until `2L/a = 35 s` after the event and it
cannot say where. The wave is the fast, blunt sensor; the balance is the slow, sharp one. A system wants both, which
is what `detect` is.

## Using it as a measurement check

`measurement_check` is the mirror of `detect`, with the shape `PhysicsEnsemble` expects:

```python
from pinneapple_physics.pipeline_leak import detect, expected, measurement_check

reference = expected(nominal_run, sensors)          # the intact line, noiseless, on the same time base
verdict = detect(observation, reference, pipe, sensors)
verdict["source"]                                   # "leak" | "instrument" | "operation" | "none"
verdict["leak_distance"], verdict["leak_rate"]      # metres along the line, m³/s
verdict["evidence"]                                 # the arrival times, the gap, the balance, the latch

ensemble = PhysicsEnsemble(experts, measurement_check=lambda obs: measurement_check(obs, reference, pipe, sensors))
```

## What this is not, and the numbers that say so

* **Steady friction only.** A third of the wave is lost over 9.3 km here, and that is friction alone; a real line
  loses more, and the smallest leak it gives up is larger than 4.6 mm.
* **The boundary conditions are assumed known.** With them, zero false alarms in 23 hours. Without them, 2.6 an hour
  and a third of those called a leak. The detector re-zeroes on the first 20 s of the same record, which absorbs a
  static calibration error but nothing that moves during it.
* **No reference data.** No pipeline leak data set was used or is claimed: every case is generated by this solver, and
  the only external anchors are the analytic ones — Korteweg for the wave speed, Colebrook for the friction,
  Joukowsky for the amplitude, the orifice law for the hole. There is no comparison with a recorded event.
* **No growing crack, no column separation, no gas.** The hole is circular, fixed and discharges to a constant
  ambient head; the fluid is one liquid at one temperature.
* **The position is only as good as the quieter front, and the quieter front is often not good.** The median is
  9.65 m over the 72 positions the localiser accepted, but the worst accepted one is 89.4 m and 17 more were withheld
  altogether — their guesses would have run to 2.5 km. The 2σ gate is a refusal, not an accuracy claim, and it rejects
  some answers that would have been fine. A transmitter further inside the pipe would hear the clipped front longer
  than `2 x_a / a = 0.84 s` and place it better; a station boundary is where it is.

## Running it

```
python examples/pipeline_leak/pipeline_leak.py            # the campaign, about 25 s
python examples/pipeline_leak/pipeline_leak.py --quick    # three seeds and 20 records, about 14 s
```

The results go to `examples/pipeline_leak/results/pipeline_leak.json`, the four figures to
`examples/pipeline_leak/figures/`.
