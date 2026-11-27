# Penmanshiel wind farm: physics-informed power and analytical wakes against five years of SCADA

A demonstration of what PINNeAPPle does with real wind-farm data, and — just as
importantly — what it refuses to claim.

The case is [Penmanshiel wind farm](https://doi.org/10.5281/zenodo.16807304)
(Scottish Borders, 14 × Senvion MM82, 28.7 MW, CC-BY-4.0): 10-minute SCADA for
every machine, with the operator's own availability, downtime and curtailment
columns. Two models are built and judged:

1. **One turbine's power curve** — a physics-informed model with hard physical
   constraints, against the IEC binned curve, a logistic fit, a gradient-boosted
   baseline and the operator's own potential-power estimate, on a **later**
   window that no model saw.
2. **The farm's wakes** — the Jensen/PARK and Gaussian (Bastankhah & Porté-Agel)
   models on the published layout, compared sector by sector with the efficiency
   the farm actually achieved.

## Run it

```bash
# 1. data (about 1.2 GB for 2016-2017; the whole record is ~7.5 GB)
mkdir -p ~/pn-wind-data/raw && cd ~/pn-wind-data/raw
for f in Penmanshiel_WT_static.csv Penmanshiel_WT_dataSignalMapping.xlsx \
         Penmanshiel_SCADA_2016_WT01-10_3107.zip Penmanshiel_SCADA_2016_WT11-15_3107.zip \
         Penmanshiel_SCADA_2017_WT01-10_3114.zip Penmanshiel_SCADA_2017_WT11-15_3115.zip; do
  curl -sSL -O "https://zenodo.org/records/16807304/files/$f?download=1"
done

# 2. the POC
python examples/wind_farm_penmanshiel/run_poc.py \
    --raw ~/pn-wind-data/raw \
    --cache ~/pn-wind-data/cache \
    --out  examples/wind_farm_penmanshiel/outputs
```

The first run parses ~300-column CSVs (about a minute per turbine-year); after
that the pickle cache makes it a few seconds. `--quick` runs the single-turbine
half only.

Outputs land in `outputs/`: `results.json` (every number with its window and
filter), `figures/*.png`, `applicability_map.txt`, `twin/` (the `pinneapple_twin3d`
scene plus its web viewer) and `wind_farm_demo.mp4` (the same scene rendered as a
film — needs `ffmpeg`; `--no-video` skips it, `--no-twin` skips both).

Open the 3D twin with:

```python
from pinneapple_twin3d import serve
serve("examples/wind_farm_penmanshiel/outputs/twin")   # http://localhost:8765/
```

`presentation.html` is the short meeting deck built from those numbers; open it
in a browser next to `outputs/figures/`.

## The notebooks

Two notebooks are the same pipeline, read top to bottom, with the outputs
committed so they can be read without the 1.2 GB download:

| Notebook | What it walks through |
|---|---|
| `notebooks/01_turbine_power_model.ipynb` | one turbine: the evidence-based operating mask, the temporal split, five curves fitted on the training window, their test-window scores, the four physical guardrail checks, the conformal interval and the VeriPhysics applicability map |
| `notebooks/02_wake_models.ipynb` | the wake shapes, the farm polar, the SCADA-observed efficiency, the decay constant fitted on validation and scored on test, and one frame of the 3D twin |

They load the parsed pickle cache when one exists, so a second run takes under a
minute; without it they pay the one-off CSV parse. `PENMANSHIEL_RAW` and
`PENMANSHIEL_CACHE` override the data locations.

## What the code is

| Step | Library call |
|---|---|
| Load and quality-check SCADA | `pinneapple_systems.wind.load_farm`, `operating_mask`, `qc_report` |
| Density correction, IEC 61400-12-1 | `add_density_correction`, `density_corrected_speed` |
| Reference / binned / logistic curves | `ReferencePowerCurve`, `fit_binned_curve`, `fit_logistic_curve` |
| Physics-informed model | `PhysicsInformedPowerModel` (monotone hinge, hard cap and gate) |
| Conventional baseline | `GradientBoostedPowerModel` |
| Scoring | `point_metrics`, `per_bin_error`, `conformal_interval` |
| Physics checks | `physical_guardrails` |
| Domain of validity | `domain_report`, `power_applicability_map` |
| Wakes | `jensen_deficit`, `gaussian_deficit`, `farm_power`, `direction_polar` |
| Farm vs SCADA | `farm_efficiency_observed`, `compare_with_model` |
| Surrogate and curtailment | `fit_surrogate`, `curtailment_study` |

The example itself only orchestrates: every physical or numerical step is a
documented call into `pinneapple_systems.wind`.

## What this does and does not establish

- The power models are scored on a **temporal holdout** with no shuffled split,
  no lagged target and every scaler fitted on the training window.
- The physics-informed model's constraints (non-negative, non-decreasing, capped
  at rated, zero outside cut-in/cut-out, zero at zero wind) hold **by
  construction** and are re-measured on the fitted curve rather than asserted.
  The Betz bound and the region-2 cubic law are penalties, so they are measured
  and reported even when they fail.
- The wake models are **analytical engineering models, not CFD**. They reproduce
  the direction dependence and the order of magnitude of the loss; they do not
  resolve the flow. Every output names the model and its constants.
- The SCADA comparison is **not** an independent measurement of wake loss: the
  free-stream reference is a group of upwind turbines whose own inflow is
  inferred from metered power, and the observed efficiency also absorbs
  curtailment, icing and real per-turbine performance differences. It is an
  upper bound on the loss and a test of the loss's *direction dependence*.
- The curtailment study is a **model estimate** under an explicit assumption
  (reducing C_T stands in for yaw or derating; the wake is not deflected). No
  measured steering response exists in this dataset.

The full list of gaps is in the guide: [docs/guides/wind_power.md](../../docs/guides/wind_power.md).
