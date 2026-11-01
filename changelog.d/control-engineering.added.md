- Control engineering for any dynamic model: `pinneapple_systems.control` provides
  - linear analysis: trim, linearisation, eigenvalues with doubling times, controllability, gain/phase/delay margins, step metrics;
  - actuators: range, rate, lag, dead band, delay, integrating;
  - controllers: PID with derivative on measurement or rate, filter and back-calculation anti-windup; cascade; state feedback;
  - a closed-loop simulator, and the linearised complete closed loop to compare what the controller commands;
  - tuning: FOPDT fit, SIMC, IMC, Ziegler–Nichols, pole placement for unstable plants, LQR, optimisation under constraints, Monte Carlo robustness.

  `pinneapple_simulation.naval` adds submarine depth-keeping and ship initial stability:
  - depth-keeping: hydrostatic instability, trim tank, target-mass vs pump-flow command;
  - initial stability: GM, free-surface correction, righting arm, roll period.

  Also: Flow nodes `control.linear_analysis`, `control.pid_tuning`, `naval.submarine_depth` and `naval.ship_stability`; `examples/control/submarine_depth.py`; and the guide `docs/guides/control_engineering.md` (#1040).
