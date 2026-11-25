- `pinneapple_systems.digital_twin.stirred_reactor`: the stirred exothermic reactor of #179 as a digital-twin benchmark case.
  A continuous stirred tank with a reversible exothermic step, a three-blade impeller and five cooling tubes — the
  well-mixed reduction of the third-party two-dimensional model the issue cites, with the equations it solves written
  in the module and the difference to that case recorded — a lumped fast model with an extended Kalman filter that
  carries the cooling it was not told about, and a predictive alarm with its safety margin. Eleven controlled cases in
  three perturbations (sensor noise at three instrument classes, a failed cooling tube and a stopped impeller, and the
  margin against a 56 °C alarm) produce a results table of state error, latency and alarm detection: 0.07 K rms
  against a reference whose stationary point two independent routes agree on to 0.2 mK, 1.4 ms per sample (0.005 % of
  the 30 s cycle), and 3 of 3 crossings detected with no false alarm. The post's own unagitated point does not
  reproduce: the same tank with the impeller stopped settles at 59 °C and 50 %, against its 75 °C and 20 %, and the
  feed temperature it would need is measured and reported. `examples/reactor_twin` runs it and writes the report,
  the figures and the JSON; `tests/test_stirred_reactor_twin.py` is the gate.
