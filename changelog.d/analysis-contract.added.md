- **The analysis contract** (`pinneapple_core.analysis`, `pp.analysis`, `pp.analyses`, `pp.run_analysis`,
  `pp.analysis_card`, `pp.check_analysis`): a calculation declares the unit, the range of validity and the name in
  a sentence of every input, the outputs with units, and one `describe` template that words its finding in English
  or Portuguese with the numbers in the reader's format. A call never raises at the caller: a missing value, a NaN,
  a value outside the model or an exception inside it comes back as a ``Result`` with a status (`ok`, `warning`,
  `bad`, `not_applicable`, `insufficient_data`, `error`), a severity and a reason a person can act on.
  `check_analysis` is the kit that proves it for every registered analysis (missing values, NaN, ±inf, text,
  out-of-range, empty arrays), and `pp.register(..., kind="analysis")` puts an analysis in the same registry
  protocol as every other extensible object. The library ships four, wrapping the closed-form models with no
  physics added: `pipe.friction_factor`, `pipe.darcy_pressure_drop`, `acoustics.helmholtz_frequency` and
  `thermal.fin_array_base_temperature`.
