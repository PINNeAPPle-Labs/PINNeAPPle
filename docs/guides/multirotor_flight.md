# A drone that says what is wrong with it

A multirotor's dynamics change during a flight: a payload is picked up, a propeller is chipped, a motor degrades, the
pack sags, a gust arrives. Its sensors also fail, and the two are not the same problem — a payload is a change of the
*vehicle*, a GPS glitch is a lie in the *sensor*, and telling them apart in real time is what keeps the aircraft flying.

This guide is the whole chain for a 1.5 kg quad-X: the rotors solved from blade element momentum theory, the airframe
integrated, eight faults injected where they physically act (two of them flown a second time at the edge of what the
vehicle can feel), a Newton-Euler residual measured against the IMU and the pack, a diagnosis that reads the *direction*
of the residual, and three detectors compared on equal terms. Then the three questions that come before and after that:
what the vehicle actually *is*, identified in flight rather than declared; what the nominal model is *missing*, learned
from flights and measured on flights it never saw; and how many seconds the answer can be believed. The first part is
produced by [`examples/multirotor_flight/flight_safety.py`](../../examples/multirotor_flight/flight_safety.py), the last
three by
[`examples/multirotor_flight/expert_identification.py`](../../examples/multirotor_flight/expert_identification.py).

```python
from pinneapple_physics.multirotor.sim import fly
from pinneapple_physics.multirotor.faults import Fault
from pinneapple_physics.multirotor.detect import calibrate, tune, ThresholdDetector, diagnose

healthy = [fly(duration=20.0, seed=s) for s in (0, 1, 2)]                    # calibration
calibration = calibrate(healthy)
flight = fly((Fault("payload", 8.0, 0.40),), duration=20.0, seed=3)          # 0.4 kg picked up at t = 8 s

detector = tune(ThresholdDetector, calibration, healthy)                     # no false alarm allowed here
detection = detector.run(flight)
detection.time, detection.channels                                           # 8.00, ('force_z',)
diagnose(flight, detection, calibration)["label"]                            # 'payload'
```

## The vehicle

| what | value | how it was fixed |
| --- | --- | --- |
| mass, inertia | 1.5 kg, (0.011, 0.011, 0.021) kg m² | a 10-inch quad |
| rotor | 10x4.7-class, BEMT, 2 blades | figure of merit **0.65**, inside the published 0.5-0.8 band |
| thrust law | `T ~ omega^2` | exponent **2.000**, the propeller law every bench measurement obeys |
| hover | 6 126 rpm, **164 W** at the pack | 9.1 g/W, which is what a quad of this class does |
| battery | 4S 5 Ah, 40 mΩ | `V = V_oc(SoC) - I R`, and the pack reports its own `I²R` loss |
| closed loop | hover 1 mm, 4 m circle 18 cm, climb 15 cm | the faults are measured on a *flying* vehicle |

The rotor is not a fitted thrust curve. It is BEMT per annulus with Prandtl tip loss, and the checks that say it is a
real rotor are that its figure of merit lands where a propeller's does and that doubling the speed quadruples the thrust
to three decimals. A chipped tip, in that model, costs thrust and **buys** efficiency: losing 40 % of the span takes
61 % of the thrust and raises the figure of merit from 0.650 to 0.787, because the tip is the part Prandtl's induced
loss punishes hardest. The obvious story — a damaged prop works harder — is the wrong one, and it is the BEMT that says
so.

## The residual, and why the direction is the diagnosis

`ResidualModel` runs Newton-Euler on the **nominal** vehicle — the mass, the rotor table, the drag — using only what a
flight controller has: the commanded rotor speeds, the accelerometer, the gyro, the pack telemetry. The controller is
never told about the fault, so it re-trims, and the residual is what is left over.

The signature of each fault: the mean residual from 2 s after the injection to the end of the flight, less the mean before it, in units of the fault-free spread of each channel.

| fault | `f_x` | `f_y` | `f_z` | `t_x` | `t_y` | `t_z` | `P` | `gps` | `hdg` | `vib` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| payload +0.40 kg | 0 | 0 | **-43** | 0 | 0 | 0 | **-6** | 0 | 0 | 0 |
| payload +0.10 kg | 0 | 0 | **-11** | 0 | 0 | 0 | -1 | 0 | 0 | 0 |
| motor 2 at 70 % | 0 | 0 | **-17** | **-8** | **-8** | 0 | -2 | 0 | 0 | 0 |
| blade chipped (12 % of span) | 0 | 0 | **-9** | **-4** | **-4** | 0 | -1 | 0 | 0 | 0 |
| pack resistance x3 | 0 | 0 | 0 | 0 | 0 | 0 | **-12** | 0 | 0 | 0 |
| gust 10 m/s | **-18** | -2 | **-11** | 0 | 0 | 0 | -1 | 0 | 0 | 0 |
| gust 5 m/s | **-5** | 0 | -1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GPS glitch 3 m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| magnetometer 25 deg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| IMU vibration 1.2 m/s^2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **46** |

Two things are worth reading twice.

* **The payload is exactly the mass ratio.** In a hover the vertical channel is `g (m_true/m_nominal − 1)`: 0.4 kg on
  1.5 kg is 2.61 m/s², and the measured −2.60 m/s² is that number to the noise — which is why the test asserts it
  against `g Δm/m` and not against a constant. This is a *checkable* prediction, not a fitted one — it is why the test asserts it against `g Δm/m`.
* **A failing rotor is not a force, it is a torque.** The planar force channels stay at zero because the controller
  re-trims the vehicle level; what it cannot hide is the couple it has to fight. The residual `(−22.8, −22.8)` sits on
  the right-rear arm and on no other, which is how the motor is named: thrust on an arm `(x, y)` makes
  `tau = (−yT, xT)`, so a deficit points along `(tau_y, −tau_x)`.

The `gps` and `heading` rows are near zero because they are *means over the whole flight*: the checks are transient,
and their residual decays once the dead-reckoned position (or the gyro's heading) has been trimmed onto the wrong
answer. What detects a GPS glitch is the jump, not the average — which is why the detectors read the stream and not the
table.

The three measurement checks are separated on purpose. A GPS glitch must not be diagnosed as a change of the vehicle,
so the fix is compared with dead reckoning; the magnetometer is compared with the integrated gyro; the accelerometer's
high-frequency content is compared with the declared noise. A sensor fault then shows up in exactly one channel and
leaves the dynamics quiet — which is the test the diagnosis uses before it believes anything else.

## The detectors, compared on equal terms

Three detectors on the same residual stream: a **fixed threshold** on each channel (the baseline everybody starts with),
a **Kalman** filter per channel whose *innovation* is the statistic (the classic model-based detector), and a
**CUSUM** for the small persistent shift. Comparing them by giving each a different sensitivity would compare three
different false-alarm rates and call the difference a result, so each one is tuned the same way: **the most sensitive
setting that raises no alarm on the calibration flights**, and then measured on flights it has never seen.

Two things had to be built before that comparison meant anything, and both were found by a test failing:

* **the vibration check has to be a high-pass, not a spread.** Measured as the standard deviation of the accelerometer's
  magnitude, it fired on a payload — a 2.6 m/s² step is, to a standard deviation, indistinguishable from vibration. As
  the second difference `x[k] − 2x[k−1] + x[k−2]` it is blind to a step and keeps a 90 Hz oscillation: the healthy
  flight reads 0.009, a payload reads 0.009, the loose-IMU case reads 3.32.
* **the CUSUM is the one that does not transfer, and the table says so.** A CUSUM accumulates, so the *allowance* it
  subtracts is the whole design. At the textbook one sigma this residual fills the accumulator on a healthy flight — the
  vibration channel alone reaches 49.6 sigma, because it is rectified: half its samples sit at zero and the rest arrive
  in bursts. Setting the allowance at each channel's own 95th percentile of the calibration data makes those three
  flights quiet, and the detector then tunes to 40 sigma — the top of the grid — and *still* alarms on two of the three
  healthy flights it has never seen. Its delays below are negative for that reason: it fires before the fault exists.
  The allowance is a property of the residual's tails, and the tails move between flights; the threshold and the Kalman
  filter, which do not look at the tails, both run with **zero** false alarms on the same unseen flights.

| fault | ThresholdDetector | KalmanDetector | CusumDetector |
| --- | --- | --- | --- |
| *false alarms, healthy flights* | 0 % | 0 % | 67 % |
| *sensitivity needed* | 8$\sigma$ | 6$\sigma$ | 40$\sigma$ |
| payload +0.40 kg | 0.00 s | 0.00 s | -1.21 s |
| payload +0.10 kg | 0.00 s | 0.00 s | -1.21 s |
| motor 2 at 70 % | 0.00 s | 0.00 s | -1.21 s |
| blade chipped (12 % of span) | 0.07 s | 0.00 s (33 %) | -1.21 s |
| pack resistance x3 | 0.00 s | 0.00 s | -1.21 s |
| gust 10 m/s | 0.00 s | 0.00 s | -1.21 s |
| gust 5 m/s | missed | missed | -1.20 s |
| GPS glitch 3 m | 0.00 s | 0.00 s | -1.21 s |
| magnetometer 25 deg | 0.00 s | 0.00 s | -1.21 s |
| IMU vibration 1.2 m/s^2 | 0.00 s | 0.00 s | -1.21 s |

And what each one called it:

| fault | ThresholdDetector | KalmanDetector | CusumDetector |
| --- | --- | --- | --- |
| payload +0.40 kg | payload | payload | payload |
| payload +0.10 kg | payload | payload | payload |
| motor 2 at 70 % | motor 2 low efficiency or blade chipped | motor 2 low efficiency or blade chipped | motor 2 low efficiency or blade chipped |
| blade chipped (12 % of span) | motor 2 low efficiency or blade chipped | no fault detected | motor 2 low efficiency or blade chipped |
| pack resistance x3 | battery sag | battery sag | battery sag |
| gust 10 m/s | gust | gust | gust |
| gust 5 m/s | no fault detected | no fault detected | gust |
| GPS glitch 3 m | GPS glitch | GPS glitch | GPS glitch |
| magnetometer 25 deg | magnetometer interference | magnetometer interference | magnetometer interference |
| IMU vibration 1.2 m/s^2 | IMU vibration | IMU vibration | IMU vibration |

## Who the vehicle actually is

Everything above compares the flight with a *declared* vehicle. `OnlineExpert` identifies the vehicle instead, in
flight, from what a flight controller already has — the rotor speeds it commanded, the accelerometer, the gyro and the
GPS velocity — by recursive least squares on the same 6-DoF model, written so that it is **linear in the parameters**:

| what is estimated | from | linear because |
| --- | --- | --- |
| mass, `mu = m_nom/m` | `f = (-T z + drag)/m`, with `T` from BEMT at the reported rotor speed | the thrust is a known number, so the force equation divides by an unknown mass |
| drag area, `CdA_i/m` | the same equation, quadratic in the body velocity | `-0.5 rho CdA v v` is linear in `CdA` |
| inertia, `I_x, I_y, I_z` | `I d(omega)/dt = tau - omega x I omega` | linear in the diagonal inertias once the gyroscopic term is taken at the current estimate |

Every flight below starts from a declaration **two times too heavy, twice as hard to turn and with four times the drag
area** it should have, and is flown at 200 Hz for twenty seconds:

| flight | what it excites | mass error | inertia error, worst axis | `CdA_x` error | model `sigma` |
| --- | --- | --- | --- | --- | --- |
| hover | nothing | **+0.01 %** | -69 % | +100 % | 0.267 m/s² |
| circle, 2.5 m/s | airspeed | +0.20 % | -15.3 % | +1.1 % | 0.091 m/s² |
| weave | rates and airspeed | +0.36 % | **-3.4 %** | **-5.0 %** | 0.041 m/s² |

Three things are worth reading twice.

* **A hover identifies the mass on its first sample.** The accelerometer measures `T/m` and the ESC reports `T`, so the
  answer is one division, not a convergence: it goes from 3.0 kg to 1.5001 kg on the first reading and stays. What
  takes *seconds* is everything else, because everything else needs the vehicle to move.
* **The inertia is identified only by a maneuver, and the covariance is what says so.** In a hover the torque equation
  is `0 = 0`: nothing rotates, so nothing says what it costs to rotate, and the estimate wanders 69 % away with a
  ±63 % uncertainty attached to it. The `weave` — a figure-eight with an oscillating heading, added to `Trajectory` for
  this — puts a torque and an angular acceleration on all three axes, and the same estimate lands 3.4 % from the truth
  with ±0.8 %. The uncertainty is not a decoration: it is the honest answer to "what did this flight teach you".
* **The drag area is identified along the axes the flight flies along.** The circle is aligned with its velocity, so it
  identifies `CdA_x` to 1.1 % and knows nothing about `CdA_y` (−29 %); the weave sweeps both, and both come out within
  5 %. A hover does not move at all, so its drag estimate is still the `+100 %` it was declared with -- which is what a
  prior is.

The estimate is filtered at the motor's own lag first, on **both** sides of the equation: the rotor speed the ESC
reports is a command, the propeller turns that command through a first-order lag, and filtering the accelerometer by the
same `1/(1 + tau s)` cancels it to first order. Without it the thrust the expert divides by is one lag behind the
specific force it is compared with, and the mass — which is that division — pays for it.

## What the model is missing

Two of the residual's causes are not faults at all. A steady wind and the ground effect near the floor are physics the
declared vehicle does not contain, and a detector cannot tell either from a payload, so `ResidualCorrection` models them
— and the interesting part is that they are **not the same kind of unknown**.

**The ground effect is a property of the vehicle**, so it is fitted once and transferred. The basis is two decaying
terms on the thrust per unit mass (`exp(-h/0.25 m)`, `exp(-h/0.5 m)`) rather than the `h0 = 0.35 m` the airframe
declares, so the fit is handed a basis and not the answer. It is taught on three flights that bob through the skirt,
and measured on flights it has never seen:

| force residual, rms, m/s² | before | after | |
| --- | --- | --- | --- |
| unseen amplitude, 0.22 m | 0.483 | **0.184** | −62 % |
| unseen amplitude, 0.15 m | 0.428 | **0.248** | −42 % |
| a calm flight at 5 m | 0.104 | 0.104 | unchanged |

The two coefficients it recovers are `+0.444` at 0.25 m and `-0.159` at 0.5 m, summing to **0.285** where the simulator
used `k = 0.15`. That is 90 % high and it is reported as it is: the true curve is a single exponential at 0.35 m and the
two-term basis is not orthogonal to it, so the total is the number to read and the split is not.

**The wind is a property of the day, not of the vehicle**, and on a vehicle with no airspeed sensor it is not
transferable at all: two wind speeds are two different constant forces, and one coefficient fitted across both describes
neither. That is measured rather than assumed — a single constant world force fitted on the gusts makes a flight with no
wind at all **6.7x worse** and a 3 m/s gust 4.0x worse, because it carries the average of the conditions it was taught
on into every flight. So the wind is a **state**, estimated causally per flight by the same recursive least squares —
three parameters, the rotation as the regressor, only samples before the one it is applied at:

| force residual, rms, m/s² | before | wind as a coefficient | wind as a state |
| --- | --- | --- | --- |
| 6 m/s, another seed | 0.442 | 0.285 (0.64x) | **0.107 (0.24x)** |
| 10 m/s, another seed | 1.307 | 0.554 (0.42x) | **0.187 (0.14x)** |
| **unseen 8 m/s** | 0.801 | 0.192 (0.24x) | **0.121 (0.15x)** |
| unseen 3 m/s | 0.147 | 0.593 (4.0x) | **0.104 (0.71x)** |
| calm, no wind at all | 0.105 | 0.701 (6.7x) | **0.104 (1.00x)** |

The coefficient happens to be *better* than the state on the two gusts it was fitted on and on the 8 m/s one, because
those are the conditions it interpolates between; it is worse everywhere else, and it is worse by a factor, not by a
per cent. The state is never worse than doing nothing and takes 85 % of the residual away from a wind speed it was never
told about.

## How long the model answers for

Everything above assumes the active model is still describing the vehicle. With the expert online, that assumption has
a number attached to it, in seconds:

**`N = (threshold − level − sigma) / rate`**

* the **level** is the force residual the model is carrying right now, in m/s²;
* the **rate** is how fast that level is drifting, from a least-squares slope over the last second (m/s² per second);
* the **threshold** is the level at which a fault is declared — here the signature of the smallest fault the campaign
  carries, a 0.10 kg payload, `g · 0.10 / 1.5 = 0.65 m/s²`;
* the **`sigma`** is the model's own uncertainty, propagated from the parameter covariance through the regressor at a
  declared envelope (5 m/s of airspeed). It is the term that makes the answer depend on *how* the model was identified
  instead of only on what the residual happens to be doing.

A payload ramping in over six seconds, measured as it grows:

| N, seconds | +0.5 s | +1.0 s | +1.5 s | +2.0 s | +4.0 s |
| --- | --- | --- | --- | --- | --- |
| payload +0.15 kg | 60 (cap) | 4.0 | 1.7 | 0.7 | 0.0 |
| payload +0.30 kg | 9.0 | 0.7 | 0.0 | 0.0 | 0.0 |
| payload +0.60 kg | 1.9 | 0.0 | 0.0 | 0.0 | 0.0 |

It shortens as the fault grows, and it shortens *with* the fault: half a second after the onset the three sizes are
already ordered 60 / 9.0 / 1.9 s, and the smallest of them is down to 4.0 s a second later and to 1.7 s a second after
that. Zero is not a failure of
the measure — it is the model saying it has expired, which is what the residual's being past the threshold means.

The same flight judged by the model a *hover* taught and the model a *maneuver* taught:

| model identified on | `sigma` | N |
| --- | --- | --- |
| a hover | 0.267 m/s² | **1.7 s** |
| a weave | 0.041 m/s² | **3.8 s** |

Both are looking at the same residual. The one that was taught by a flight that never rotated does not know what the
inertia is, says so in its covariance, and therefore promises less — which is the whole point of quoting a number
instead of a colour.

## Four honest negatives

1. **A degraded motor and a chipped blade are not told apart by these channels.** In theory they differ: a chip leaves
   more torque per newton than a motor that is simply weak. In the flight, the power-per-newton ratio is 1.96 for the
   motor and 2.02 for the blade — inside the noise, because the controller re-trims both into the same signature ("more
   collective thrust than the nominal model explains, plus a torque asymmetry"). The diagnosis reports them together and
   names the motor; claiming more would be claiming a discrimination the numbers do not support.
2. **A 4 m/s gust is a 3-sigma event on a 1.5 kg quad.** The drag on a 0.03 m² CdA at 4 m/s is 0.29 N against a 14.7 N
   weight: the vehicle barely notices, so the residual barely notices. The gust in the campaign is 10 m/s for that
   reason, and the table above keeps the 4 m/s case to show where the floor is.
3. **The drag area along the thrust axis is not identified by any flight this vehicle can fly.** At 2 m/s the vertical
   drag is 0.16 m/s², at 5 m/s it is 1.0 m/s², and the thrust model's own lag error at the same speeds is 0.10 m/s² —
   the same order. Every flight here returns a vertical `CdA` between −0.15 and +0.15 m² against a true 0.08, and the
   fitted uncertainty is *small* while being wrong, which is the worst combination and is reported as such. The two
   horizontal axes, which are what a gust pushes against, come out at 0.0285 and 0.0299 against 0.030.
4. **The mass uncertainty the filter reports is optimistic.** It follows from the declared accelerometer noise, and the
   model mismatch — the motor lag, the BEMT's own error — is not in that number: it claims ±0.09 g on a mass it gets
   right to 5 g. The error is small because the equation is good, not because the covariance was told the truth.


## What this is not

No EKF: the residual is evaluated on the sensors as they come, and the estimate error is a limitation, not a modelling
choice — with a filter in front, the payload channel would shrink and so would every delay. Drag is dropped from the
predicted force (it is second order at hover speeds, and first order in a 10 m/s gust). The torque channel differentiates
the raw gyro, which is why its fault-free spread is 2.8 rad/s² — a filtered rate would buy back most of that.

See also: [detecting false data injection with physics residuals](physics_residual_attack_detection.md) for the same
idea on a process plant, where the channels are mass balances and flow laws instead of Newton-Euler.
