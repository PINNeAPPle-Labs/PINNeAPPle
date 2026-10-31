- Output units for `pinneapple_systems.process_components`: the real-gas state (Pa, K, J/kg, J/(kg K), m/s, Pa s), heat
  exchanger, polytropic path, compressor similarity map, valve flow, pipe pressure profile, curved path, Miner damage,
  torsional stick-slip and water-hammer records, and the scalar helpers (section properties, flows, head, speed,
  Joukowsky rise, wave period, acid-base fractions). Units come from the field names, their comments or the docs; values
  in the units of the caller's inputs (the beam's moment and stress, the buckling load, the transient's time axis) stay
  empty.
