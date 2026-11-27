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
