# Bring your own foundation model: wrap it, check it, and know where it stops

Nobody will train a better global model than the large pre-trained ones. The value is in knowing **when** to trust
them: which case, which region, which lead. This guide puts one such model next to the models trained here, under the
same residual, measurement and horizon checks — and adds the checks the domain itself makes, which need no reference
field.

## 1. The registry knows the open models and their licence

```python
from pinneapple_physics.pretrained import known_models, spec, licence_report

known_models()          # ('depth-anything-v2-large', 'graphcast', 'ifs-hres', 'neuralgcm', 'pangu-weather',
                        #  'pinneapple-weather', 'prithvi-100m', 'sam2.1')
m = spec("pangu-weather")
m.source, m.licence.spdx, bool(m.licence.verified)
# ('gs://weatherbench2/datasets/pangu/...', 'NOASSERTION', False)  -- nobody read Pangu's licence at the source, and
# the registry says so instead of guessing. 'pinneapple-weather', 'prithvi-100m' and 'sam2.1' come back Apache-2.0,
# verified; 'depth-anything-v2-large' comes back CC-BY-NC-4.0 with commercial_use=False.
```

Every entry carries the source, the grid and variables it was trained on, and the licence — with the date and place it
was read, or an explicit `NOASSERTION`. A model this repository does not know can still be wrapped, if you bring the
licence record yourself: the point is that the record travels with the prediction.

## 2. An adapter loads it and maps the query

| adapter | when |
|---|---|
| `WeatherBench2Adapter(store, "IFS HRES")` | the forecasts published in WeatherBench2, read through the same `PublishedForecasts` the skill curves use |
| `TorchCheckpointAdapter(store, "ck.pt")` | a `torch` checkpoint saved with the `{"state_dict", "config"}` layout of `pinneapple_physics.weather.train` |
| `ArrayArchiveAdapter("wb2_subset.npz")` | a local `.npz` of fields — the offline path, and how a subset of an archive is scored on a machine with no network |
| `HuggingFaceAdapter("facebook/sam2.1-hiera-large", to_input=..., from_output=...)` | weights from the Hub, or a local path with `local_path=` |

An adapter that cannot load raises `PretrainedUnavailable` with what is missing and how to point it at a local copy.
It never returns a prediction it did not make; `PhysicsEnsemble` counts that expert as failed for the case and the
others carry it.

## 3. Wrap it as an expert of the ensemble

```python
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.pretrained import from_pretrained_expert

expert = from_pretrained_expert("ifs-hres", WeatherBench2Adapter(store, "IFS HRES"), lazy=True)
ens = PhysicsEnsemble([expert, from_callable("persistence", persistence, cost=0.0)])
run = ens.run(queries, truth)
run.summary()["licences"]        # {'ifs-hres': 'NOASSERTION', ...} -- the licence travels with the score
```

The expert's `cost` may be a function of the query: a forecast whose cost grows with its lead time is priced per case,
and `EnsembleRun.summary()["mean_cost_per_case"]` is the **weighted mean** (the weights sum to one), not the sum.

## 4. The checks the domain makes, with no reference

A pre-trained model is quiet outside its training distribution, and there is no truth to compare against there. These
three ask what the domain demands of any field; each is a `ValidityCheck`, and `as_prediction_check` turns it into the
`(score, reason)` callable `PhysicsEnsemble(prediction_checks=...)` uses to tilt the weights.

| check | the question | measured |
|---|---|---|
| `advection_residual` | is the 6-hour change one the wind can explain? | **0.0028** for a field that is exactly advected (the scheme's truncation error), **1.000** for the same field sitting still under that wind, 0.974 for noise |
| `global_drift` | does the area-weighted total wander? | **0.000** for a constant total, **0.040** spreads per day for a 5 % step over 6 h, doubling with the step |
| `hand` + `flood_mask_score` | is the water where the terrain drains? | hand 0 at the outlet and 10 m uphill on a 10 m slope; a mask inside the depth scores **1.000**, one that wets ground above it **0.306**, one that wets the outlet without its donors **0.800** |
| `temporal_residual` | does the frame change as one moving scene? | flow residual **5e-9** for frames translated by two pixels, **0.974** with 7.5 % of pixels teleporting for noise |

Two things to read carefully. A mask that wets **nothing** scores 1 — admissible and uninformative, which is why the
score is documented as such and never read as a pass. And `advection_residual` divides by the larger of the time
change and the advection: a tracer that does not move under a wind that should carry it scores 1, from the other side
of the same failure.

## 5. Adding a new pre-trained model

1. Read its licence at the source and add the entry to `REGISTRY`: `name`, `family`, `source`, the grid and variables,
   and `Licence(spdx, url, commercial_use, verified)` with the date and place you read it. `NOASSERTION` is allowed
   and is reported as unverified — `from_pretrained_expert(..., require_verified=True)` will refuse it.
2. Pick the adapter, or write one: `predict(query)`, `cost(query)`, `cost_method`, `open()`. The contract is the
   common output shape and units of the ensemble, nothing else.
3. Add the domain check the domain needs, if the three here do not cover it: a `ValidityCheck` with a value, a
   tolerance and a sentence, and a test that shows the field that satisfies the structure scoring near zero and the
   field that breaks it scoring near one.
4. Measure it: an ensemble of the new model, the models already here and a cheap baseline on the same cases, with each
   model's skill alone next to the ensemble's.

## 6. What this does not do yet

The demo the issue asks for — IFS HRES, Pangu-Weather and the PINNeAPPle weather model over 2020, with the weights
choosing per region and season and the residual flagging where each leaves its domain — needs the archive subset
downloaded and a trained checkpoint; it is the next slice. What exists here is the mechanism, the licence record, the
three checks and the measurement of the checks themselves.
