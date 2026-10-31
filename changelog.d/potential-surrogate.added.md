- Potential-based surrogates in `pp.potential` (issue #928): `surrogate.PotentialSurrogate`, a network for the
  Helmholtz energy whose properties are its derivatives, with an input-convex option in the specific volume
  (`PICNN`), and `surrogate.DirectSurrogate`, one output per property. Both are trained on mixed data, where each
  sample carries one property, by the same `surrogate.fit`. The measured CO2 study is in `co2_study` and
  `examples/potential_co2` (CoolProp Span-Wagner data near the critical point, 3 configurations, 5 or 2 seeds).
  - The direct surrogate is more accurate in every configuration and region where the paired interval excludes zero.
    In the training region with all six properties, its error is 0.10 against 1.61 (MLP potential) and 1.24 (ICNN).
  - The potential surrogates satisfy the Maxwell relation to about 1e-5. The direct surrogate violates it by about
    0.4, and by 2 to 11 for the second-derivative identities.
  - With c_p in the mixed data, the MLP potential collapses to near-zero pressure.

  Also adds the Flow node `potential.co2_surrogates` and the example flow `potential_co2/surrogate_study`.
