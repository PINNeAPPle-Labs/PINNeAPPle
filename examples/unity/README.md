# PINNeAPPle in Unity

Three pipelines that put PINNeAPPle's physics inside a Unity project, and take a Unity scene back out as OpenUSD.
They run on a machine **without Unity and without a GPU**: the Unity side is a folder layout plus a C# package,
and the parts of it that carry data are compiled and run by the repository's own tests.

```
01_engineering_twin_to_unity.py     PINNeAPPle -> Unity      a twin folder a Unity project imports as it is
02_unity_to_pinneapple_to_openusd.py Unity -> PINNeAPPle -> OpenUSD   a scene gets its physics, leaves as a USD layer
03_live_bridge.py                   Unity <-> PINNeAPPle     the runtime asks for a field while it plays
04_surrogate_turntable.py           surrogate -> twin        a network trained and checked here drives the field
```

```bash
python examples/unity/01_engineering_twin_to_unity.py
python examples/unity/02_unity_to_pinneapple_to_openusd.py
python examples/unity/03_live_bridge.py
```

Each writes only into `examples/unity/_out/` (ignored by git).

## 1. A twin in Unity

A heat-exchanger spool and its cooling fin. Two fields, from two closed-form solutions in the library, each carried
by the checks that decided it:

| part | physics | field | checks |
|---|---|---|---|
| spool | transient 1-D conduction (Carslaw & Jaeger slab, `pinneapple_physics.closed_form.heat`) | `temperature` [C] | heat-equation residual, Dirichlet walls, maximum principle, the analytic bound on the approach to steady |
| fin | pin-fin excess temperature (Incropera Table 3.4 case A) | `temperature` [C] | base at the base temperature, cooler tip, integrated convection = closed-form heat dissipated, values inside the range |

The export is `Assets/TwinData` (the same `scene.json` + `geometry.glb` + `fields.bin` the web viewer reads) plus
`Assets/PinNeAPPleTwin` (the C# runtime: loader, field animator, legend, sensors, bridge client, editor menu). Drop
the `Assets` folder into a project, add the **PINNeAPPle Twin View** component, press Play.

## 2. Unity -> PINNeAPPle -> OpenUSD

A plate with a cantilevered bracket, as a team would model it in Unity. Unity writes `geometry.glb` (Y-up) and
`pinneapple_link.json` — parts, units, sensors, `UsdPhysics`, and **the jobs PINNeAPPle should run**. That link is the
whole contract of the lane. PINNeAPPle reads it (Z-up), runs the jobs, and writes back:

* `scene.usda` — parts as `Mesh` with `PhysicsCollisionAPI`, a `PhysicsMaterialAPI` contact material, a
  `PhysicsScene` with gravity, both fields as time-sampled `primvars:<field>`, and the provenance (title, source,
  units, the real times) in the layer metadata: opens in usdview, Omniverse, Isaac Sim;
* a twin folder (`scene.json` + glb + fields.bin), and a Unity project folder written back from it
  (`_out/bracket_back_in_unity`), so the scene the team modelled reopens with the physics on it — the loop closes.

`emulate_unity_export` writes exactly the two files the editor menu item
`PINNeAPPle > Export Selection as Scene Link` writes, so the pipeline is demonstrable and testable without Unity.

## 3. The live bridge

`python -m pinneapple_unity serve` (or `pinneapple_unity.serve_bridge`) starts an HTTP bridge with three verbs —
`/v1/health`, `/v1/run` (one job on one part), `/v1/scene` (a whole twin) — speaking `pinneapple-unity/1`. The Unity
component `PinneappleBridgeClient` sends a request from a coroutine and gets the field **plus its checks plus a
decision** (`APPROVED` / `REJECT` / `REVIEW`). The demo shows a bracket that passes at 40 N and is refused at 1000 N
(300 MPa against a 250 MPa yield), which is the point: a game can decline to draw a field the physics does not
support.

`POST /v1/scene` also takes `save` (and `usd: true`): the bridge writes the computed twin as a Unity project folder
and answers with its path and the problems `verify()` found, so a running client can produce something the project
imports.

The bridge binds `127.0.0.1` by default (it runs physics on demand), warms the library up at start-up so the first
request is milliseconds, and takes `--token` (header `X-Pinneapple-Token`) when it has to be reachable.

## 4. A surrogate drives the twin

`04_surrogate_turntable.py` trains a small MLP (2 → 64 → 64 → 64 → 1) on the closed-form slab solution with the
library's own `Trainer`, then asks the trust layer whether the network's field is a field anyone should look at:

| | |
|---|---|
| inputs | `x/L` along the spool and `log t` (log-uniform in the window, because diffusion is fast at the start) |
| fit | 6,000 samples, 1,500 epochs, Adam at 5e-4 with gradient clipping; a seed that misses the target is retried, printed, not hidden |
| check | `pp.trust_card(..., pde="heat", dirichlet=...)` on a 41 × 25 grid, plus the relative L2 against the closed form |
| result | relative L2 **0.17 %**, heat-equation residual **0.031** (tolerance 0.05) → `APPROVED` |
| output | `_out/surrogate.pt` (the weights Unity would take through `export_onnx` + Sentis) and the turntable clip |

The clip is the point: the twin's six baked time steps are gone, and the surface is coloured **at 30 frames per
second straight from the surrogate**, camera orbiting 360°. That is the same role the network plays inside Unity --
a field it can evaluate at any instant, with the decision that allowed it to be shown.

## Recorded

Three short MP4s of the fields the demos compute, rendered from the exported twins by `make_movies.py`
(matplotlib + ffmpeg, no Unity, no GPU) — the same geometry, the same values and the same colour range the runtime
shows:

| file | what it shows |
|---|---|
| [`media/01_spool_temperature.mp4`](media/01_spool_temperature.mp4) | the spool heating from the 180 °C flange, the front advancing towards the 30 °C end, `TC-01`/`TC-02` on their vertices |
| [`media/02_bracket_temperature.mp4`](media/02_bracket_temperature.mp4) | the plate from the Unity-authored scene warming from 20 °C |
| [`media/03_bracket_bending_stress.mp4`](media/03_bracket_bending_stress.mp4) | the cantilever: 12 MPa at the root falling linearly to zero at the free tip |
| [`media/04_surrogate_turntable.mp4`](media/04_surrogate_turntable.mp4) | 8 s turntable of the spool, the temperature evaluated continuously by the surrogate (not from the baked steps) |

```bash
python examples/unity/make_movies.py            # rebuilds the twins if needed, then movies 01-03
python examples/unity/04_surrogate_turntable.py # trains and checks the surrogate, then records movie 04
```

What these are **not**: a capture of Unity's Play mode. That needs a licensed editor and the Unity Recorder package
(*Package Manager → Recorder → Movie*, 30 fps) with `PINNeAPPle Twin View` playing the steps.

## What is verified, and what needs the engine

| | |
|---|---|
| the data layer (JSON, GLB read/write, manifest, fields, colormap, and the runtime field the live lane applies) | compiled with `dotnet` and run against an exported twin: `tests/test_unity_csharp.py`, `dotnet run --project pinneapple_unity/unity/tests -- <folder>` |
| the file/link/live lanes | `tests/test_unity_integration.py` (export, verify, ingest, USD, jobs, bridge, HTTP + token) |
| the MonoBehaviours, the editor menu and the two shaders | written against the public Unity API; **Unity compiles them, not this repository's CI** — a mistake there is a compile error in the console, not a silent wrong number |

The wire format, the axis convention and the endpoints are documented in `docs/guides/unity_integration.md` and in
`pinneapple_unity/protocol.py`.
