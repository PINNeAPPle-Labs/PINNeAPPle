Day 20 of 30: Digital Twin Runtime

A model gets validated once, in a report. Then the plant changes: a valve wears, a sensor drifts, a leak opens. Nobody runs the model against the live signals again, so nobody notices when its predictions stop matching the plant.

The Digital Twin Runtime runs a validated model next to operational data (MQTT, REST or a recorded table). An ensemble Kalman filter assimilates every reading, estimates hidden parameters such as an unaccounted outflow, forecasts with 5-95 % bands and raises alerts. Every 30 readings a drift check compares the model's own one-step predictions with the measurements.

In the video, two hours of a heated tank (721 readings) with a leak that opens at 60 min. The lumped mass and energy model estimates the leak at 0.503 L/s (± 0.1) against a true 0.5 L/s and raises an alert at 62.3 min. 93 % of level readings fall inside the 95 % band, and an open-loop revalidation gives a level RMSE of 4.6 mm against a 9 mm limit. Passed.

What sets it apart is the second run: the same record with a reduced-order surrogate that has no leak term. The twin does not hide it. Level band coverage drops to 0 %, the decision is "retrain", and revalidation fails at 141 mm against 9 mm. That is the signal to stop trusting the model.

Results leave as a StudyResult with each check, plus a state endpoint for other systems.

Try it: https://twin.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/twin_runtime)

#DigitalTwin #KalmanFilter #ReducedOrderModels #ProcessEngineering #ModelValidation #ScientificML #Python
