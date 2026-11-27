- **The drone itself in the graph** (#350, on top of #2548): `pinneapple_physics.embedded_flight` exports the
  multirotor of #335 -- one RK4 step of the 6-DoF dynamics over the BEMT rotor table, the seven Newton-Euler channels
  and the two decisions -- as one straight-line ONNX file with seven fixed inputs and thirteen outputs. Parity against
  `Multirotor.step` and `ResidualModel` over 400-step flights is 1e-7 relative; one step costs 0.45 ms (2.2 kHz) on one
  ARM core of a Mac, and no board was measured because there is none on this machine.
- `export_probe` measures where `torch.onnx` cannot follow the numpy model: the table lookup by index is refused at
  opset 17, and the BEMT rotor and a Python `if` on a tensor export *and freeze* (60 % of the thrust, 10.1 % of the
  lift), which is worse than a refusal.
- `pinneapple_physics.embedded_bridge`: the flight-stack contract as a 69-byte frame -- active model, trust, trust
  horizon in control steps, suspect sensor and channel, `off_physics` -- with a CRC, four refused-frame errors, a
  publish/subscribe double that carries the bytes, and the `rclpy` adapter marked as untested (no ROS 2 here).
- `embedded.ExpertModule`/`export_expert`/`parity_check` export an expert of `PhysicsEnsemble` with its residual and
  measurement check, and measure the gap against the expert and against the module's own float32/float64 floor. An
  expert with no torch path is refused: tracing a numpy `predict` freezes the numbers the tracer saw.
- `EmbeddedRuntime.run`/`benchmark` take a feed keyed by input name, so one runtime serves a graph with two inputs and
  one with seven; `embedded.quantize` wraps dynamic int8 quantisation -- which on the flight graph finds nothing to
  quantise (no matrices), makes the file 42 % larger and leaves the numbers identical.
