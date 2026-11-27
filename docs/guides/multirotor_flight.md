# A drone that says what is wrong with it

A multirotor's dynamics change during a flight: a payload is picked up, a propeller is chipped, a motor degrades, the
pack sags, a gust arrives. Its sensors also fail, and the two are not the same problem — a payload is a change of the
*vehicle*, a GPS glitch is a lie in the *sensor*, and telling them apart in real time is what keeps the aircraft flying.

This guide is the whole chain for a 1.5 kg quad-X: the rotors solved from blade element momentum theory, the airframe
integrated, eight faults injected where they physically act (two of them flown a second time at the edge of what the
vehicle can feel), a Newton-Euler residual measured against the IMU and the pack, a diagnosis that reads the *direction*
of the residual, and three detectors compared on equal terms. Every number
below was produced by [`examples/multirotor_flight/flight_safety.py`](../../examples/multirotor_flight/flight_safety.py).

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

## Two honest negatives

1. **A degraded motor and a chipped blade are not told apart by these channels.** In theory they differ: a chip leaves
   more torque per newton than a motor that is simply weak. In the flight, the power-per-newton ratio is 1.96 for the
   motor and 2.02 for the blade — inside the noise, because the controller re-trims both into the same signature ("more
   collective thrust than the nominal model explains, plus a torque asymmetry"). The diagnosis reports them together and
   names the motor; claiming more would be claiming a discrimination the numbers do not support.
2. **A 4 m/s gust is a 3-sigma event on a 1.5 kg quad.** The drag on a 0.03 m² CdA at 4 m/s is 0.29 N against a 14.7 N
   weight: the vehicle barely notices, so the residual barely notices. The gust in the campaign is 10 m/s for that
   reason, and the table above keeps the 4 m/s case to show where the floor is.

## What this is not

No EKF: the residual is evaluated on the sensors as they come, and the estimate error is a limitation, not a modelling
choice — with a filter in front, the payload channel would shrink and so would every delay. Drag is dropped from the
predicted force (it is second order at hover speeds, and first order in a 10 m/s gust). The torque channel differentiates
the raw gyro, which is why its fault-free spread is 2.8 rad/s² — a filtered rate would buy back most of that.

See also: [detecting false data injection with physics residuals](physics_residual_attack_detection.md) for the same
idea on a process plant, where the channels are mass balances and flow laws instead of Newton-Euler.
