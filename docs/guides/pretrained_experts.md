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

That recipe is executed in a test for every entry the registry already has
(`test_the_recipe_of_the_guide_holds_for_every_model_in_the_registry`): a registry entry that no adapter can wrap, or
one whose licence stops travelling with the prediction, fails there.

## 6. The demo: three models over 2020, per region and season

`examples/weather_forecasting/pretrained_ensemble.py` puts IFS HRES, Pangu-Weather and the PINNeAPPle weather model
(phase 1) side by side as experts of one `PhysicsEnsemble`, with the two weather checks as `prediction_checks` — so
the mixture is not chosen by the error alone: a model whose `z500` change its own 500 hPa wind cannot explain, or whose
dry static energy drifts, is taken out of that case and the others carry it.

```bash
python examples/weather_forecasting/pretrained_ensemble.py wx/era5 \
    examples/weather_forecasting/checkpoints/phase1_step15000_fp16.pt --out wx/pretrained --every-days 30
```

The cases are the 00 UTC initialisations of 2020 every 30 days (11 of them), each carried to +240 h; the cells are four
latitude bands × four seasons (16). Regions and seasons are learned separately and in chronological order, so the error
reported for a case is the one the ensemble made *before* seeing its truth. The first run reads the two published
archives over HTTPS (about 100 s per case) and caches them as `.npz`; a rerun is offline and takes seconds.

**Measured 2026-11-27** (11 cases, 16 cells, `z500` relative L2 over leads 6 h to +240 h, cosine-latitude weights;
the run is in `examples/weather_forecasting/data/pretrained_ensemble_2020.json`):

| | ensemble | IFS HRES | Pangu-Weather | PINNeAPPle |
|---|---|---|---|---|
| relative L2, +6 h to +240 h | **0.0079** | 0.0086 | 0.0081 | 0.0152 |
| RMSE at day 5 (m²/s²) | 235 | **207** | 212 | 541 |
| learned weight | — | 0.248 | 0.249 | 0.252 |
| share of the combination | — | 0.385 | 0.347 | 0.267 |
| cases a check took it out of | — | **0 %** | 5.7 % | **18.8 %** |
| advection residual (median) | — | 0.954 | 0.953 | 0.956 |
| dry static energy drift, spreads/day | — | 0.0022 | 0.0028 | 0.0042 |

Four things are worth reading carefully, and they are the honest part of the demo:

- **The ensemble has the lowest error of the four over the window** (0.0079 against 0.0081 of the best single model,
  and 0.0152 of the model trained here) **and the worst day-5 point of the two published ones** (235 against 207 of IFS
  HRES). It beats the best single model in **7 of the 16 cells** and on **33 % of the cases**: a mixture is a bet, not
  a free improvement, and both numbers belong in the table.
- **The learned weights stay near the fixed-share floor** (0.25 each with three experts; the extremes are 0.17 for
  Pangu and 0.33 for the PINNeAPPle model in the Arctic DJF cell). What moves the mixture is not the error weight but
  **the checks**: the share of the combination is 0.385 / 0.347 / 0.267 while the weights are 0.248 / 0.249 / 0.252,
  because the model a check took out of a case gets no share of it.
- **The residual flags each model where it leaves its domain, and the pattern differs by model.** IFS HRES is never
  taken out. Pangu-Weather is, only in the tropics (25 % of the MAM cases, 33 % of JJA and of SON). The PINNeAPPle
  model is taken out in **100 % of the Arctic DJF cases, 75 % of MAM, 67 % of SON, 33 % of JJA** and in 25 % of the
  tropics MAM ones — 18.8 % of all cells, and it is also the model with the worst day-5 RMSE (541 against 207). The
  sentence the check gives is the evidence: in Arctic DJF its dry static energy drifts 0.0134 spreads per day against
  the 0.0117 the analysis's own level put the boundary at, while IFS HRES drifts 0.0044.
- **The advection check does not separate them, and the demo says so.** Every model, and the analysis itself, lands
  between 0.84 and 1.02: `z500` is not a materially advected field, so the residual is a floor all three meet and the
  check never fires. It is reported anyway, because a check that is *reported and flat* is information; a check tuned
  until it fires would only be a threshold in disguise.

The tolerance each check is read against is measured on the same cases, not declared: it is twice the level the
analysis itself scores, which — with `as_prediction_check` mapping a residual equal to the tolerance to a score of 0
and the ensemble's threshold sitting at 0.5 — takes a model out when it is **twice as far from the dynamics as the
analysis** (`calibrate` in the script prints the boundary it used for every region: 1.68 to 2.04 for advection, 0.005
to 0.012 spreads per day for the energy).

Two notes on running it. `pretrained_ensemble_2020.json` records the wall time of the run that wrote it (5.5 s here,
because the 11 cases were already in the `.npz` cache); the first run reads two published archives per case over HTTPS
and took **748 s** for the 11 cases. And the checkpoint is the same phase-1 one the rest of the example uses — the
rollout phases 2, 4 and 8 have not run, which is what the 541 m²/s² at day 5 is made of.

## 7. What this does not do yet

- **GraphCast and NeuralGCM** are in the registry with their licence and envelope but not in the demo: GraphCast is
  published on the 1440 x 721 grid and needs a regridder to 64 x 32 first. NeuralGCM is on the right grid and is one
  `PUBLISHED` entry away.
- **Prithvi, SAM 2 and Depth-Anything** have adapters and licence records, and the flood and video checks exist, but
  no demo runs them: the perception side needs the HLS and video inputs, which are not in the repository.
- **One year, eleven cases.** The weights move little in eleven cases per cell (the fixed-share floor dominates); a
  longer archive would show the seasonal signal the current table only hints at.
- **No conservation in the combination itself**: the ensemble combines the fields, it does not project them onto a
  conserved total (`PhysicsEnsemble(conserve=...)` does that and the demo does not use it).

