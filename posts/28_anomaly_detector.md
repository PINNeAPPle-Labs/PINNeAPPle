Day 28 of 30: Physics Anomaly Detector

Plant historians are full of alarms nobody trusts. A threshold on one sensor fires at every set-point change and misses a slow leak, and a neural anomaly score does not say why. Engineers already know the physics: what enters a tank leaves it or accumulates, for mass and for energy.

The Physics Anomaly Detector writes those balances as formulas over your sensor columns and runs them next to robust statistics, AR + CUSUM and the residuals of a physical reference model. Every detector is calibrated only on a window you declare normal.

In the run shown in the video (a heated tank, 9.5 h, six labelled faults), it found 6 of 6 faults with 0 false alarms per hour. The leak was flagged at its start: mass balance broken by -0.39 L/s, 5.8 % of the inflow, 7.2 sigma, over the sensors h, q_in and q_out. The heater loss broke the energy balance by -26.7 kW while the mass balance stayed intact.

On held-out benchmark runs, the data-only detectors found 0.0 % (statistics) and 50.0 % (AR + CUSUM, 0.18 false alarms/h) of the faults. The balances found 83.3 % and the model residuals 100.0 %, both with no false alarms. The app also states what the balances miss: a drain-valve fault, where the measured outflow keeps the physics consistent. Events export as CSV and as a StudyResult.

Try it: https://anomaly.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/anomaly_detector)

#AnomalyDetection #ProcessEngineering #PhysicsInformed #ConditionMonitoring #IndustrialAI #FaultDetection #ScientificML
