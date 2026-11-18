- **Behavior:** the two thermal flagship examples are now conjugate and multi-source. `examples/nvidia/07` solves a
  microchannel cold plate coupled to its coolant (Shah & London / Hausen `h`, Muzychka-Yovanovich `dp`, the coolant's
  own energy balance closed against the heat the slots take), learns it with a three-channel PhysicsNeMo FNO
  (conductance, slot-coupling and source maps), and answers the engineering question with a second surrogate plus a
  Monte-Carlo of the plant. `examples/nvidia/10` rates every heat-sink candidate twice -- the closed-form network and
  a 3-D finite-volume solve of the base with two chips on it -- reports the gap between the two as a check, adds
  radiation to the network, and searches seven design variables instead of three.
- `pinneapple_physics.darcy_checks` checks the reacting problem `-div(a grad u) + s u = f`: `sink=` on the residual,
  the two certified bounds, the flux balance and the estimated error maps. The certificates carry over unchanged
  (a non-negative sink only stiffens the operator), and `exact=True` replaces the constant-coefficient relaxation --
  vacuous when the sink carries most of the energy -- with the discrete error itself, from one preconditioned
  conjugate-gradient solve per sample.
- `operator_benchmarks.darcy_reaction_2d` / `darcy_reaction_matrix`: the same finite volumes as `darcy_2d` with the
  reaction term, which is the equation a conjugate cold plate satisfies.
- `physicsnemo_fno` / `pinneapple_fno` take `in_channels` (several input fields per sample, one output field);
  single-channel models keep their scalar normalisation buffers, so existing checkpoints still load.
- `pinneapple_tools.nvidia.coldplate`: a liquid-cooled cold plate with microchannels -- the fractional slot geometry,
  the coolant state and its limits, the coupled solve and the engineering checks.
- `pinneapple_tools.nvidia.heatsink.evaluate_full`: a design rated by the closed-form network, by a 3-D solve of the
  base with several chips, and by the exact USD BRep solid, with the network-to-3-D gap reported.
- `pinneapple_design.thermal.heatsink.field.solve_base_field` takes `sources=[...]`, so one base solve resolves
  several chips instead of one.
- `plate_fin_heatsink.evaluate` takes `emissivity` / `sky_fraction` / `t_surround_c` and reports
  `h_rad_w_m2k` and `radiation_share`; the default stays the ducted case with radiation neglected.
