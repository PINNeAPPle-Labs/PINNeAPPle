Day 29 of 30: Physics-Based Control Lab

Most process loops still run on PID gains tuned years ago, or on on/off rules. They react to the error after it appears and know nothing about the constraint just ahead. The mass and energy balances that would let a controller plan ahead are already known; they are just not used.

The Control Lab puts model predictive control (linear, and nonlinear with CasADi/IPOPT) next to the controllers that run today, on the same simulated heated tank, with the same sensor noise and the same unmeasured disturbances: a 20 % demand increase and colder inlet water.

In the run shown in the video, the nonlinear MPC brought the temperature tracking error (IAE) from 1191 K·s with PID to 255 K·s, 4.7 times lower, with zero constraint violations for all five controllers. The cost is shown too: heater travel goes from 1313 to 1957 kW. An MPC built on a reduced-order surrogate, which has no outflow input, tracked the level worse (IAE h 80 vs 21 m·s), and the table says so instead of hiding it.

Every study ends with recorded checks (8/8 passed: constraints held, optimiser converged at every step), a StudyResult and the trajectories as CSV. The app runs in simulation and recommendation mode only: there is no command path to a controller, PLC or DCS. A person decides.

Try it: https://control.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/control_lab)

#ProcessControl #MPC #ModelPredictiveControl #PID #PhysicsBasedModeling #ControlEngineering #ScientificML
