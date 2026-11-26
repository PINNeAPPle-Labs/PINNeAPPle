- The six friction-contact cases are the library's (#1507): `pinneapple_simulation.external_solvers.calculix.contact`
  holds the structured hexahedral blocks and CalculiX's brick face numbering (`mesh`), the deck writer PrePoMax opens
  unchanged (`deck`), the closed-form verification — `|F_t| = mu*N`, Hertz line contact, `T/N = min(tan(theta), mu)`,
  two-material Lamé, the Coulomb cone and the pressure integral (`verify`), the six cases with their parameters,
  assumptions and references (`cases`), and the run (`run`), whose results are read by `pinneapple_data.cae` and not by
  a parser of its own. `run_case` returns the numbers as read and the checks; a report, a route or an evidence record
  is written from those and cannot change what the checks saw, and without a `ccx` binary the deck is still written
  and the result says nothing was solved. The cases the system already trusts as capabilities are the same six —
  sliding block, double shear, bolt preload, Hertz, self-locking incline and shrink fit — built from the same
  `Case`/`Built` objects, and `get`/`build`/`catalog` take a wider case book so a caller can add its own problems
  without copying the readers.
