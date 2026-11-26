- **Cost report: joules and emissions per operation, with the declared method printed under the table**
  (roadmap K20, #119). `pinneapple_analysis.cost` gains `estimate_energy`, which prices an operation by a named
  method -- FLOPs times a declared pJ/FLOP from `ENERGY_PER_FLOP` (one sourced entry: 45 nm CMOS arithmetic,
  Horowitz, ISSCC 2014), a declared power draw times the measured time, or a RAPL counter delta measured on Linux --
  and `cost_report`, which writes time, FLOPs and joules per operation with the method, the source and the hardware
  the number assumed. A row no method can price says why instead of showing a guess, and emissions appear only
  against a grid intensity the caller declares with its source.
