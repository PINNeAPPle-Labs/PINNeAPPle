# The Omniverse live link and the Cosmos plausibility benchmark (#411)

Two ways the NVIDIA stack meets the library, and what each of them actually needs: a PINNeAPPle forecast driving an
Omniverse stage, and a judge of physical plausibility for generated video. **Neither Omniverse Kit nor a Cosmos model is
installed in this environment**, and there is no network, so this page separates what was measured here from what needs
the extra installed.

| part | needs | measured here |
|---|---|---|
| live link: writing the stage, reading it back, playing it | `usd-core` (`pxr`) | **yes** |
| live link: streaming into a running Kit session | `pinneapple[omniverse]` (Kit) | no — the extra is named by the code |
| Cosmos plausibility benchmark on the labelled clip set | this repository only | **yes** |
| Cosmos: generating the clips themselves | a Cosmos endpoint and an API key | no — the clips are generated here |

## 1. The live link: a forecast becomes a USD stage with time samples

`pinneapple_tools.nvidia.livelink` takes a `ForecastTrack` — parts (geometry, fields over time, a pose and a scale per
step) and sensors (position, series, envelope) — and writes a stage whose attributes carry the time samples. Nothing in
that path needs Kit: `pxr` is enough, which is why it can be tested here.

```python
from pinneapple_tools.nvidia import livelink

track = livelink.ForecastTrack(
    title="heated channel", times=[0.0, 0.5, 1.0],
    parts=[livelink.ForecastPart(name="wall", vertices=v, faces=f,
                                 fields={"T": temperature_over_time}, field_units={"T": "K"})],
    sensors=[livelink.SensorTrack(name="tc1", position=(0.0, 0.0, 0.2), series=series, unit="K")],
    trust={"status": "measured", "horizon": 12.0, "unit": "frames"},
)
path = livelink.write_forecast_stage(track, "forecast.usda")
livelink.play_stage(path)                      # reads the stage back and prints each step
```

Measured on this machine, with a four-vertex body, two transient fields, a pose and a scale per step, and one sensor:
`track.check()` returns no problems, the stage is written as a **4.147-byte** `.usda`, reading it back gives the same
**5** time steps, and `play_stage` walks them — `body at (0, 0, 0) scale 1 … body at (0.4, 0, 0) scale 1.4`, with the
sensor going from `300 K` to `400 K`. The trust record travels in the layer metadata as the flattened JSON string
`pinneapple_trust`, because a USD stage has no dict property and a string survives every USD tool.

`track.check()` is the gate that matters: a field with a different number of steps from the times would produce a stage
that is silently wrong, so a `(3, 4)` field against 5 steps is refused with `expected (5, 4)`.

The Kit half — `KitLiveLink`, `kit_livelink_script`, `open_live_layer` — raises with
`pip install "pinneapple[omniverse]"` named, and `kit_available()` answers `False` here.

## 2. Cosmos: where a generated clip stops being physical

`pinneapple_tools.nvidia.cosmos` scores a generated clip against conservation, from the tracked object or from optical
flow: rigid-body impulse and acceleration for a bouncing body, mass, momentum and divergence for a channel flow. The
limits are **calibrated** on clean clips at a stated quantile and then applied to the rest.

The labelled set is generated here, not downloaded: clean clips from a stated physical model, and violated clips that
carry one named violation at a known frame — `reverse`, `teleport`, `mass_gain`, `gravity_off` for the rigid body;
`mass_appears`, `momentum_gain`, `compress` for the flow.

Measured, one seed (15 clips: 4 clean, 11 violated):

| quantity | measured | what it means |
|---|---|---|
| detection rate | **0,727** (8 of 11) | a calibrated limit was broken at all |
| false alarm rate | **0,0** (0 of 4) | no clean clip was flagged |
| attribution rate | **0,25** | the check that fired was the one the violation breaks |
| lag | 1 frame (min 0, max 1) | the violation is caught one frame after it happens |
| horizon | **26,5** of 48 frames | with three seeds, the median frame at which the clips stopped being physical |

The honest reading is in that table and is not flattering: **detection is 73% and attribution is 25%**. The benchmark
finds that something is wrong far more often than it says *what* is wrong, and the horizon is 26,5 frames only because
the violations are injected around the middle of a 48-frame clip — half of the clip is spent before anything is wrong.
Both are reported by `BenchmarkReport.summary()` and `attribution_rate()`, and the tests pin them as they are measured
rather than as they should be.

The calibration on three seeds gives the limits `acceleration 0,585`, `area 0,118`, `impulse 6,58`, `mass 0,559`,
`divergence 0,677`, `momentum 2,92` — the numbers a consumer has to accept to use the judge.

`BenchmarkReport.horizon()` returns a `pinneapple_analysis.trust.record.Horizon`, so the lead a generated clip buys is
consumed by the same trust machinery as any other forecast: the useful lead against the no-skill one (the whole clip).

## What is not here

- **A Cosmos endpoint.** `cosmos` carries the clients for `/v1/infer` and Cosmos Reason behind their extra; here the
  clips are rendered from the stated generator, and the HTTP path is exercised by the local fake server that
  `pinneapple_tools.nvidia.testing` already provides.
- **A real video.** The frames are synthetic renderings of a disc and of channel tracers, which is what makes the
  residuals measurable with a known answer — a real Cosmos clip would need the endpoint above.
- **Isaac Sim / Isaac Lab.** `pinneapple_tools.nvidia.isaac` builds `UsdPhysics` articulations and their checks; the
  runner needs Isaac installed, and the sim-to-real switching story of #411 stays behind that extra.
