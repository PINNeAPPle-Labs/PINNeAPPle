- Output units for `pinneapple_systems`. In `process_components`: the real-gas state (Pa, K, J/kg, J/(kg K), m/s, Pa s), heat
  exchanger, polytropic path, compressor similarity map, valve flow, pipe pressure profile, curved path, Miner damage,
  torsional stick-slip and water-hammer records, and the scalar helpers (section properties, flows, head, speed,
  Joukowsky rise, wave period, acid-base fractions). Units come from the field names, their comments or the docs; values
  in the units of the caller's inputs (the beam's moment and stress, the buckling load, the transient's time axis) stay
  empty. In `time_series`: the gridded data's latitude and longitude (deg), D20 depths (m), the standardised SOI,
  ACC, ENSO categories and interval coverage (`-`), the equatorial wind-forcing projections (m/s2) and the Hilbert
  phase (rad); the synthetic series state no unit and stay empty.
