# Checking a surrogate against its model, and a twin page for one quantity

Two generic tools for any model with a closed form or a solver behind it. The worked case is an ideal isothermal
plug-flow tube with a first-order reaction (`pinneapple_physics.closed_form.ideal_reactors`):

| function | answer |
|---|---|
| `pfr_conversion(D, L, k, Q)` | `X = 1 - exp(-k tau)`, `tau = V / Q`, `V = pi D^2 L / 4` |
| `pfr_conversion_ktau(k, tau)` | the same from `k` and `tau` |
| `cstr_conversion_ktau(k, tau)` | the stirred tank, `X = k tau / (1 + k tau)` |

Each declares the input box it is valid for (`x_ranges`). The module-level `X_OUTPUTS` declares the outputs of every tool here with their units, and the Studio catalog reads it.

## `verify_against_model`: one point, four checks

```python
from pinneapple_autolab.study.verify import verify_against_model

out = verify_against_model({"k": 0.02, "tau": 120.0}, 0.95,
                           "pinneapple_physics.closed_form.ideal_reactors:pfr_conversion_ktau", rtol=0.01)
out["status"], out["model_value"], out["rel_error"]     # 'fail', 0.909282, 0.0448
```

The checks are `model_runs`, `inside_model_ranges`, `units_agree` and `agreement`. Each one has the status
pass, fail, inconclusive or not_run.

The answer is `pass` only when all of these hold:

- the model ran;
- the point is inside the model's declared ranges;
- the units agree;
- `|surrogate - model| <= rtol |model|`.

Three cases give `inconclusive`, never `pass`, because the check did not happen:

- the model raises, or returns no value;
- the point is outside the declared ranges;
- the units differ.

`rtol` has no default.

## `surrogate_twin`: a page that only shows model evaluations

```python
from pinneapple_autolab.study.scalar_twin import surrogate_twin

out = surrogate_twin("pinneapple_physics.closed_form.ideal_reactors:pfr_conversion_ktau",
                     [{"name": "k", "unit": "1/s", "low": 0.0, "high": 0.1},
                      {"name": "tau", "unit": "s", "low": 0.0, "high": 200.0}],
                     {"name": "X", "unit": "-"}, "twin_out", samples_used=training_points)
```

It writes `twin_out/index.html` and `twin_out/manifest.json`.

**Values.** The model is evaluated on a grid of every input before the page is written. The page shows the table
entry at the grid node the sliders select. Nothing is interpolated or computed in the browser.

**Sliders.** They move by grid index, so they cannot leave `[low, high]`.

**Extrapolation.** A node outside the convex hull of `samples_used` is marked "extrapolation", even when it is
inside the box. With no training set, every node is marked.

**Band.** The band uses one of three methods, and the page names the method it used:

- `none`: labelled "no uncertainty estimate";
- `surrogate_std`: plus or minus the `{quantity}_std` the model returns;
- `holdout_error`: plus or minus the RMS error the caller measured on held-out data.

**Manifest.** It records:

- the variables and their ranges and grids;
- the model id;
- the SHA-256 of the training set;
- the band method;
- the library commit.

For the same model and grid, the manifest changes only when a training point changes, and every such change shows.

**Offline.** The page is self-contained and works offline. Its Content-Security-Policy blocks every network
request. User-supplied names are escaped in the embedded data and shown as text.

Tests: `tests/test_reactor_verify_twin.py`.
