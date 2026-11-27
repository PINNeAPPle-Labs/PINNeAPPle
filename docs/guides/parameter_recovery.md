# A recovered parameter and the interval it was measured to deserve

Roadmap item #77. A calibrated parameter is a number *and* an interval, and the interval is a claim: "the truth is
inside this range 95 % of the time". The claim is testable, so here it is tested, on an experiment whose truth is
known.

```python
from pinneapple_analysis.inverse_problems import ParameterSet, coverage_study, get_preset, recover

built = get_preset("cooling_h").build()            # 450 J/K part cooling in air, ±0.3 °C thermocouple
params = ParameterSet(built["parameters"])
result = recover(built["model"], params, built["observations"])
print(result)

# h = 10.9618 [9.21424, 13.0409] (95%)
# T0 = 93.9946 [93.682, 94.3072] (95%)
# T_air = 22.2012 [12.5222, 31.8802] (95%)
```

The data of that preset were generated at `h = 11`, `T0 = 94`, `T_air = 23`, so `result.contains(truth)` is a check
that does not need a statistician: all three intervals contain the value they should.

## The interval

`recover` fits the free parameters (weighted least squares in the scaled coordinates `z = ln θ` or `θ / scale`, with
the optional prior), then takes the Fisher information at the optimum, `Jᵀ Σ⁻¹ J` plus the prior precision, and maps
its inverse back through `Parameter.from_z`. `level` is the two-sided coverage of the interval; the noise is the one
the observations declare, so a dataset that understates its own error gets a narrow, wrong interval and no warning
from here — that is what `coverage_study` is for.

`interval="profile"` swaps the linearised interval for the profile-likelihood one (each parameter fixed on a grid, the
others re-fitted, the `Δχ² = 3.84` crossing): slower, and the honest choice when the model is visibly nonlinear over a
standard deviation. It is a 95 % interval by construction, so `level` other than 0.95 is refused rather than silently
rescaled.

## What the data cannot recover

A parameter the measurements do not separate from another does not get a confident number. The null space of the
Fisher information is read direction by direction, each direction is described by the parameters that carry it, and the
interval of every parameter in it is **infinite**:

```
h = 26.926 [−∞, +∞] (95%)        warning: moving h up and C up together leaves the predictions
C = 1105.35 [−∞, +∞] (95%)       unchanged: the data fix h/C, not h and C separately
T0 = 93.9946 [93.682, 94.3072]   → the intervals are the data's way of saying so
```

That is the `cooling_h_C` preset, where the heat capacity is unknown as well: the curve depends on `h` and `C` only
through `hA/C`, the fit is good (`χ² = 19.6` on 31 points), and the two parameters are still not known. The parameters
the data do see keep their finite intervals in the same report.

## The coverage

`coverage_study(case, n=200, level=0.95)` repeats the experiment at the known truth with fresh noise and counts how
often the interval contains it. The reading of each count is against the Wilson interval of the coverage itself, so 95 %
out of 20 runs is not reported as "calibrated" — 20 runs cannot tell 95 % from 85 %.

| parameter | coverage | 95 % CI of the coverage | median width | reading |
|---|---|---|---|---|
| `h` | 94.0 % | [90 %, 97 %] | 3.86 W/m²K | ok |
| `T0` | 94.0 % | [90 %, 97 %] | 0.63 °C | ok |
| `T_air` | 95.0 % | [91 %, 97 %] | 19.5 °C | ok |

Two hundred repetitions of a three-parameter fit take about 14 s on a laptop: the cooling curve is a closed form, and
the interval costs one Jacobian, not a chain.

The profile-likelihood interval, on the same case (30 repetitions — it re-fits a grid per parameter):

| parameter | coverage | 95 % CI of the coverage | median width | reading |
|---|---|---|---|---|
| `h` | 93.3 % | [79 %, 98 %] | 3.84 W/m²K | ok |
| `T0` | 96.7 % | [83 %, 99 %] | 0.63 °C | ok |
| `T_air` | 96.7 % | [83 %, 99 %] | 19.9 °C | ok |

### What the coverage study caught

The first profile run reported **56 % coverage for `T0`** while the linearised interval on the same runs was 94 % — the
profile interval was 0.34 °C wide where the data supported 0.63 °C. The cause was not the profile: it was the
interpolation of the `Δχ²` crossing. `T0` is a linear parameter, scaled by its nominal 94 °C, so its standard deviation
in `z` is 0.002 while the grid had a floor of 0.05 in `z` on each side; interpolating the chord between two grid points
across a convex profile reports a crossing too far in, by a factor of three in this case. The crossing is now the root
of the parabola through the three nearest points, which is exact for a Gaussian profile, and the two intervals agree to
three digits. The fix is in `profile_likelihood` itself, so it reaches `analyze_identifiability` and every interval it
reports, not only this flow.

### When the model is wrong

The same case with a 2 °C offset added to every measurement — a thermocouple the model has no term for (100
repetitions):

| parameter | coverage | 95 % CI of the coverage | median width | reading |
|---|---|---|---|---|
| `h` | 93.0 % | [86 %, 97 %] | 3.86 W/m²K | ok |
| `T0` | 0.0 % | [0 %, 4 %] | 0.62 °C | **under** |
| `T_air` | 86.0 % | [78 %, 91 %] | 19.8 °C | ok |

The bias is absorbed by `T0`, the parameter that shifts the whole curve, and its interval stops containing the truth
immediately while the parameters that set the *shape* keep their coverage. A coverage study does not merely say "the
answer is off"; it says which parameter is carrying the error the model cannot represent.

## What it refuses

A level outside `(0, 1)`, an interval kind it does not know, `interval="profile"` with a level other than 0.95, a
coverage study whose case did not set `Parameter.truth` (there would be nothing to check against), and observations
without measured values. A silent fallback would make the number this page is about meaningless.

Reproduce every table with:

```bash
python scripts/identifiability_recovery.py                 # --n, --profile-n, --level, --seed, --markdown
```

The script exits non-zero when the well-specified case is not calibrated: a study whose job is to detect a
miscalibrated interval should not report success when it finds one.
