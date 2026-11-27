- **`pinneapple_physics.multirotor`: a drone that says what is wrong with it** (#335): a 1.5 kg quad-X whose four
  10x4.7-class rotors are solved with **BEMT** (figure of merit 0.65, in the published 0.5-0.8 band, and `T ~ omega^2`
  to 2 %), a pack that sags with `V = V_oc(SoC) - I R` and reports its own `I^2 R` loss, drag, ground effect, motor lag
  and a cascaded controller that holds a hover to 1 mm and a 4 m circle to 18 cm. Eight faults are injected where they
  physically act -- payload, degraded motor, chipped blade, gust, GPS glitch, magnetometer interference, IMU vibration,
  battery sag (the chip measured from BEMT: losing 40 % of the span takes 61 % of the thrust and *raises* the figure of
  merit from 0.650 to 0.787, because the tip is the part the induced loss punishes hardest) -- and `ResidualModel` measures **Newton-Euler on the nominal vehicle** against the IMU, the gyro and the
  pack telemetry, using only what a flight controller has. The diagnosis reads the *direction* of the residual: a
  payload is `g (m_true/m_nom - 1)` along body `-z` (0.4 kg on 1.5 kg measured at -2.44 m/s^2, 41 sigma), a failing
  rotor points along its own arm through the torque channel (-22.8, -22.8 for the right rear), and a GPS glitch moves
  the fix and nothing else. Three detectors -- fixed threshold, Kalman innovation gate and CUSUM -- are each tuned to
  the most sensitive setting with **no false alarm on the calibration flights** and then measured on flights they never
  saw, which is what makes their delays comparable. Two honest negatives are recorded: a degraded motor and a chipped
  blade are not told apart by these channels (the controller re-trims both into the same signature; the
  power-per-newton ratio that separates them in theory is 1.96 against 2.02, inside the noise), and a 4 m/s gust is
  only a 3-sigma event on a 1.5 kg quad.
- **`pinneapple_physics.multirotor.estimate`: who the vehicle is, learned in flight, and how long the answer holds**
  (#335): the residual of the previous bullet compares the flight with a *declared* vehicle; this one identifies the
  vehicle. `OnlineExpert` runs recursive least squares on the same 6-DoF model, written linear in the parameters -- the
  mass ratio and the drag-area ratios from `f = (-T z + drag)/m`, the three diagonal inertias from
  `I d(omega)/dt = tau - omega x I omega` -- on nothing but the commanded rotor speeds, the IMU and the GPS velocity,
  with both sides filtered at the motor's own lag so the commanded speed and the flown thrust are the same number.
  From a declaration two times too heavy it recovers the mass **to 0.00 % on a hover, 0.19 % on a circle and 0.35 % on
  a maneuver**, and what the flight *cannot* identify it says so: after a hover the yaw inertia is 69 % wrong with a
  +-54 % uncertainty, after the maneuver it is 3.4 % wrong with +-0.8 %. The drag area along the axis the vehicle flies
  (0.0298 against 0.030) is identified by a maneuver and the one along the thrust axis is not -- its specific force at
  the speeds this quad flies is 0.16 m/s^2, the same order as the thrust model's own lag error. `ResidualCorrection`
  learns the two things the nominal model is missing and measures them on flights it never saw: the **ground effect**
  is a property of the vehicle (a two-term height basis fitted through the 35 cm skirt, which cuts a held-out flight's
  force residual from 0.48 to 0.18 m/s^2 and leaves a flight at 5 m untouched), and the **wind** is a property of the
  day -- a constant world force fitted across flights makes a calm flight **6.7x worse**, so it is a state estimated
  causally per flight instead, which takes an unseen 8 m/s gust from 0.80 to 0.12 m/s^2. `confidence_horizon` turns all
  of that into the number the issue asked for, in seconds: *this model answers for N seconds before its residual
  crosses the fault threshold*, built from the level the model is carrying, the rate it is drifting at and its own
  parameter uncertainty as a margin -- and it shortens as the fault grows (60 / 9.0 / 1.9 s half a second after the
  onset of a 0.15 / 0.30 / 0.60 kg ramp, 1.7 s one second later, 0 s once the model has expired) and as the excitation
  gets poorer (1.7 s judged by a model a hover taught against 3.8 s by one a maneuver taught). `diagnose` now carries
  it beside the verdict, and `Fault(ramp_s=...)` gives a progressive fault to measure it on.
