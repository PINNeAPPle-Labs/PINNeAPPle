- `pinneapple_analysis.dynamics.dimensional`: dimensional analysis and Buckingham pi in exact rational arithmetic,
  guide `docs/guides/dynamics_dimensional.md`.
  - `Dimension` (any base system, from unit strings through `pinneapple_data.physical_units`), `PhysicalQuantity`,
    `DimensionalMatrix` (exact rank, integer null space) and `check_homogeneity`.
  - `BuckinghamPiAnalyzer`: groups with repeating variables, recognition of named groups (Re, Pe, Fo, Pr, Ra, Nu,
    ...), `express` to write a known group in the basis, characteristic scales with their ambiguity, the scaling
    law implied by a single group. Isolated dimensions and targets that cannot be formed are reported as
    insufficient quantities.
  - `ScalingLaw`: power laws fitted in log space with t intervals, or exact exponents with a fitted constant.
  - `TrajectoryNormalizer` (dimensional or empirical, labelled), `CurveCollapseAnalyzer` (collapse measure against
    the raw curves and a permutation control, degenerate and by-construction flags, exponent fit and stability)
    and `SimilarityMetrics`.
  - Benchmarks: `python -m pinneapple_analysis.dynamics.benchmarks dimensional`.
