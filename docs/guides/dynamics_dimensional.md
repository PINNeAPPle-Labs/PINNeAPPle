# Dimensional analysis, Buckingham pi and curve collapse

`pinneapple_analysis.dynamics.dimensional` finds the dimensionless groups of a problem, checks that the quantities
given are enough to form them, and derives the scaling laws that dimensions alone determine. It also measures whether
a family of curves collapses onto one after rescaling, against controls that can expose a collapse that is not real.

All the algebra is exact rational arithmetic (`fractions.Fraction`), so group exponents have no round-off.

Results follow the toolkit's [shared contract](dynamics_toolkit.md). They carry:

- the assumptions, each marked checked or unchecked;
- diagnostics;
- the conditions under which the method does not apply.

## Two different things: dimensional analysis and empirical normalisation

**Dimensional analysis** uses only the dimensions of the quantities. Take `n` quantities over `k` base dimensions,
and let `D` (`k × n`) be the matrix of exponents. A product `Π = ∏ q_i^α_i` is dimensionless exactly when `D α = 0`.
The groups are a basis of that null space: there are `n − rank(D)` of them (Buckingham 1914).

**Empirical normalisation** uses the data: dividing by the maximum, z-scoring, or min-max scaling a curve and its
time span. Its result is dimensionless, but it is **not** a pi group. Two curves that obey different laws can look
alike after it.

The module keeps the two apart:

- `TrajectoryNormalizer.dimensional(...)` builds its scales from quantity products and checks their dimensions.
  It raises `DimensionError` if a "time scale" is not a time.
- `TrajectoryNormalizer.empirical(...)` marks every trajectory it produces as
  `source["normalization"]["kind"] == "empirical"`, with the note *"not a pi representation"*.

## What it does

### Analyzing a problem

| Task | Call | Gives |
|---|---|---|
| Find the groups | `BuckinghamPiAnalyzer(quantities, target=..., repeating=...).analyze()` | The `n − rank(D)` groups with integer exponents, named when they match a known group (Re, Pe, Fo, Pr, Ra, Gr, Nu, Bi, Ma, Fr, We, St, Sc). Diagnostics for isolated dimensions and insufficient quantities. |
| Check a known group | `analyzer.express(DimensionlessGroup({...}))` | Exact exponents writing it as a product of the groups found, or `None` if it is not one. |
| Find characteristic scales | `analyzer.characteristic_scales(Dimension.of(T=1))` | One product with that dimension, plus the groups that make it ambiguous. For example, `L/U` and `L²/α` differ by Pe. |
| Get the implied scaling law | `analyzer.scaling_law()` | When only the target's group exists, the exponents are exact and only the constant is unknown. Fit the constant with `.fit_constant(y, values)`. |
| Fit a power law freely | `ScalingLaw.fit(y, {"x1": ..., ...})` | Least squares in log space, with t intervals per exponent and `R²`. |
| Check an equation | `check_homogeneity({"term": Dimension, ...})` | Whether every term has the same dimension. |

### Normalising and comparing curves

| Task | Call | Gives |
|---|---|---|
| Normalise trajectories | `TrajectoryNormalizer.dimensional(...)` or `.empirical(...)`, then `.apply(tr)` and `.invert(tr)` | A labelled, invertible rescaling |
| Measure a collapse | `CurveCollapseAnalyzer(curves).analyze(x_scales, y_scales, origin=...)` | The collapse measure, the raw measure, and a permutation p-value, plus degenerate and by-construction flags |
| Find the best collapse exponents | `.fit_exponents(P)` and `.sensitivity(P, p, q)` | The best `x / P^p`, `y / P^q`, and the curvature at that optimum (stability) |
| Compare two systems | `SimilarityMetrics.groups(groups, a, b)` and `.curves(a, b)` | Log-ratio of each group, and the distance between two curves |

**Units and systems.** `Dimension` accepts any base names, so a force-length-time system works as well as SI.
`Dimension.from_unit("W/m2/K")` and `PhysicalQuantity.from_unit(name, value, unit)` read units with
`pinneapple_data.physical_units`, which converts values to SI. A unit with an offset (°C, °F) is refused for absolute
values.

`DimensionlessGroup.sympy()` returns the group as a SymPy expression.

## Foundations and implementation

### How the groups are built

With a `target`, the groups use repeating variables (Rayleigh's method):

1. Choose `r = rank(D)` dimensionally independent quantities, not including the target.
2. Make one group for each remaining quantity `q_j`, as `q_j ∏ R^a`, solving `D_R a = −D_j` exactly.
3. Scale the result to coprime integers.

The target therefore appears in exactly one group, which is listed first. Without a `target`, the groups are the
integer null-space basis of the reduced row echelon form.

Either way, the groups are unique only up to products and powers of each other. `express` shows that two bases agree.

### When the quantities are insufficient

| Situation | Status | What it means |
|---|---|---|
| A base dimension carried by a single quantity | `isolated_dimension:<base>`, warn | That quantity cannot enter any group, so either it is irrelevant (like the mass of a pendulum) or a quantity carrying that dimension is missing. |
| The target's dimension cannot be built from the other quantities | failure | The heat flux `q` from `k` and `L` alone needs a temperature difference. The result is not applicable. |
| `rank(D) = n` | failure | No group exists. |

### The collapse measure

The measure is the mean across-member variance on the common support, divided by the variance of the master curve:

```text
C = mean_x Var_k[y_k(x)] / Var_x[mean_k y_k(x)]
```

`C = 0` means the curves are identical, and `C` of order 1 or more means there is no collapse. It is unchanged by a
common rescaling of `y`.

Three controls go with it:

1. **The raw measure.** The collapse is compared with the curves before any rescaling.
2. **A permutation control.** The same scale factors are assigned to the wrong members. If the permuted scalings
   collapse as well, the scaling explains nothing. The p-value cannot go below `1/m!` for `m` members: about 0.04
   for 4 members and 0.008 for 5.
3. **`origin=`.** It says where the scales came from:
   - `"dimensional"`: built from quantities, independent of these curves.
   - `"empirical"`: taken from the curves themselves. This raises the `collapse_by_construction` warning.
   - `"fitted"`: exponents fitted on these same curves. This raises a warning that the p-value is optimistic, and
     that the result should be validated on held-out members.

A rescaling that flattens the master curve is reported as **degenerate**, and the result is not applicable, rather
than as success.

## Examples

**When it works.** Drag on a sphere:

```python
from pinneapple_analysis.dynamics.dimensional import BuckinghamPiAnalyzer, PhysicalQuantity as Q

qs = [Q.from_unit("F", None, "N"), Q.from_unit("rho", None, "kg/m3"), Q.from_unit("U", None, "m/s"),
      Q.from_unit("d", None, "m"), Q.from_unit("mu", None, "Pa s")]
r = BuckinghamPiAnalyzer(qs, target="F").analyze(aliases={"d": "L"})
print(r.estimate["groups"])   # ['Pi_1 = F / (rho U^2 d^2)', '1/Reynolds = mu / (rho U d)']
```

Dimensions say only that `F / (ρ U² d²) = f(Re)`. They do not give `f`. In the creeping-flow limit, where `ρ` drops
out, the analysis gives `F = C μ U d` exactly, and the data fix `C = 3π`; see `stokes_constant` in the benchmarks.

**When it tells you the list is wrong.** The pendulum, with its mass:

```python
r = BuckinghamPiAnalyzer([Q.of("T", T=1), Q.of("L", L=1), Q.of("g", L=1, T=-2), Q.of("m", M=1)], target="T").analyze()
print(r.estimate["groups"])          # ['Pi_1 = T^2 g / L']  ->  T = C sqrt(L/g)
print(r.diagnostic("isolated_dimension:M"))   # warn: m cannot enter any group
```

Remove `g` from the list and the target can no longer be formed. The result then reports insufficient quantities
instead of returning a group.

**When a collapse is not evidence.** Take five curves that follow five different laws (an exponential, a line, a
logistic, a cosine and a cubic). After min-max and span normalisation they reach a collapse measure of 0.44. That is
the same value as the raw curves of the same five laws (0.44), and the result carries the `collapse_by_construction`
warning. A true dimensional collapse is many orders of magnitude tighter: 1.6e-5 for Newton cooling, 4e-9 for the
conduction profiles below.

## Assumptions, failures, limits

**Always unchecked.** The list of quantities is complete and relevant. Buckingham's theorem rewrites the relation
among the quantities given, and cannot tell you which quantities matter. Dimensional constants (`g`, a material
constant) must be listed when they matter.

**Not handled.**
- Vector or directional dimensions (Huntley's `L_x`, `L_y`). You can declare them as separate base names, but the
  module does not reason about them.
- Angles, which are treated as dimensionless.
- Mixing unit systems in one matrix. Any base names are accepted, but nothing is converted between systems.

**Collapse.**
- The measure is computed on the common support only, by linear interpolation of each curve.
- A good collapse supports a scaling. It does not explain it.

**Cost.**
- Exact elimination is `O(k² n)` in fractions, which is negligible for physical problems (`n` below about 20).
- The automatic choice of repeating variables tries combinations in order, `O(C(n, r))`.
- Collapse with `P` permutations costs `O(P · m · n_grid)`.

## Benchmarks

To reproduce the table:

```bash
python -m pinneapple_analysis.dynamics.benchmarks dimensional
```

It uses `seed=0` and 999 permutations. The values below were measured on 2026-10-31.

| Case | Quantity | Reference | Computed | Error | Pass |
|---|---|---|---|---|---|
| reynolds | Re | 7.485e+04 (`compute_dimensionless_numbers`) | 7.485e+04 | 0 | yes |
| peclet | Pe | 5.357e+05 (`compute_dimensionless_numbers`) | 5.357e+05 | 1.2e-10 | yes |
| prandtl | Pr | 7.143 (`compute_dimensionless_numbers`) | 7.143 | 0 | yes |
| fourier | Fo | 0.00168 | 0.00168 | 0 | yes |
| rayleigh_in_span | Ra is a product of the groups found | yes | yes: Ra = Π₁ Π₂^½ Π₃^½ | – | yes |
| pendulum_constant | C in T = C √(L/g), 40 points, 1 % noise | 2π = 6.2832 | 6.2887 ± 0.0109 | 0.0055 (z = 0.50) | yes |
| pendulum_free_exponent_L | exponent of L, free fit | 0.5 | 0.4998 ± 0.0024 | −0.0002 (z = 0.07) | yes |
| stokes_constant | C in F = C μ U d | 3π | 9.4248 | 1.8e-15 | yes |
| cooling_collapse | collapse under t/τ, T/ΔT | 0 | 1.6e-05 | 1.6e-05 | yes (raw 11.2, permuted median 50.3, p = 0.007) |
| conduction_collapse | collapse under x/√(αt) | 0 | 4.0e-09 | 4.0e-09 | yes (raw 1.38, p = 0.036: 4 members, floor 1/24) |
| fitted_exponent_x | exponent of τ in t/τ | 1 | 1.0005 | 4.7e-04 | yes |
| fitted_exponent_y | exponent of τ in ΔT = τ^½ | 0.5 | 0.4994 | −6.2e-04 | yes |
| empirical_false_collapse | 5 different laws after min-max and span | 0.4387 (raw) | 0.4429 | – | demonstration, flagged `collapse_by_construction` |

What the rows show, and what they do not:

- **Group values.** The Re, Pe and Pr values agree with an existing implementation of the same textbook formulas.
  That shows the groups are the right ones, not that the formulas were validated independently.
- **Collapse residuals.** These come from linear interpolation of irregularly sampled curves, 81 points per cooling
  curve.
- **The pendulum.** It uses the small-angle period. Large amplitudes add the amplitude as a group, which this case
  does not test.
