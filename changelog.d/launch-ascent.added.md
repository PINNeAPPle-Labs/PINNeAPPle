- Ascent to orbit as a first-class simulation: `pinneapple_simulation.launch` flies a launch vehicle from a pad to a
  target orbit in 3-DOF through the US Standard Atmosphere 1976, with a pitch schedule, a coast to apogee and a
  circularisation burn, a velocity bookkeeping that closes on the Tsiolkovsky budget, a design closure that sizes the
  stages from the losses the flight measures, and a verification suite (atmosphere against the published tables, pad
  rotation, elements against an independent solver, one revolution of the inserted orbit, the checks of the flight).
  Displays: a globe with the ascent, a chase view of the launcher and a mission-control page. Worked case:
  `examples/rocket_launch` (200 kg from Tanegashima to a 500 km near-circular orbit), guide
  `docs/guides/launch_ascent.md`.
