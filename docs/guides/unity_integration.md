# PINNeAPPle in Unity

Unity is where a twin gets looked at: the people who own the pipe, the bracket or the heat sink rotate it, cut it,
and judge it with their eyes. PINNeAPPle is where the field comes from and where it is checked. This guide is the
seam between the two, and it is deliberately thin: three files on disk, one JSON link, one HTTP verb.

```
   PINNeAPPle twin  ──scene.json + geometry.glb + fields.bin──▶  Unity project (Assets/TwinData + C# runtime)
   Unity scene      ──geometry.glb + pinneapple_link.json────▶  jobs, checks, OpenUSD layer
   Unity runtime    ──HTTP /v1/run, /v1/scene───────────────▶  PINNeAPPle computes, checks, answers
```

## Quick start

```python
from pinneapple_twin3d.demo import pipe_bend_scene
from pinneapple_unity import export_for_unity

exp = export_for_unity(pipe_bend_scene(), "out/bend_unity")
assert exp.verify() == []            # GLB, field offsets and the C# package, re-read and checked
```

Copy `out/bend_unity/Assets` into a Unity project, add **PINNeAPPle Twin View** to an empty GameObject, press Play.
`exp.summary()` says what went in; `exp.problems` says what did not add up.

## The three lanes

### File lane: a twin folder Unity imports

`export_for_unity(scene, folder)` writes a Unity project root:

| path | what it is |
|---|---|
| `Assets/TwinData/scene.json` | format `pinneapple-twin3d/1`: parts, fields, time steps, sensors, units, provenance |
| `Assets/TwinData/geometry.glb` | one mesh per part (glTF 2.0 binary) |
| `Assets/TwinData/fields.bin` | float32 field values, rigid motions and per-step visibility, offsets in the manifest |
| `Assets/TwinData/pinneapple_unity.json` | up axis, the rotation Unity applies, units, counts |
| `Assets/PinNeAPPleTwin/` | the C# runtime, two field shaders (Built-in + URP), the editor menu |
| `README.md` | the four steps in Unity |

These are the same three data files `pinneapple_twin3d.Scene.export` writes for the web viewer: one layout, several
front ends (web, Unity, Blender, Omniverse). `read_unity_twin(folder)` and `scene_from_twin(folder)` read them back
in Python, which is how a twin is reopened for more steps or more fields.

**Axis convention.** `pinneapple_twin3d` scenes and OpenUSD are Z-up; Unity and glTF are Y-up. The conversion is a
-90° turn about X, `(x, y, z) -> (x, z, -y)`. It is applied on export metadata (the runtime rotates the root), on
ingest, and recorded in `pinneapple_unity.json` / `pinneapple_link.json` (`up_axis`), so the hand-off carries the
convention instead of assuming it.

### Link lane: a scene authored in Unity becomes physics + OpenUSD

Unity writes `geometry.glb` (Y-up) and `pinneapple_link.json` — in the editor with
`PINNeAPPle > Export Selection as Scene Link`, or from Python with `emulate_unity_export` (the same two files, so the
lane runs and is tested without Unity installed):

```json
{"protocol": "pinneapple-unity/1",
 "title": "Bracket: plate + cantilever",
 "geometry": "scene.glb", "length_unit": "m", "time_unit": "s", "up_axis": "y",
 "times": [0, 60, 300, 900, 1800, 3600],
 "parts": [{"name": "plate", "node": "Plate", "group": "structure", "color": [0.72, 0.75, 0.8], "axis": "auto"}],
 "sensors": [{"id": "TC-01", "unit": "C", "position": [0.1, 0.0, 0.003], "label": "TC-01 plate"}],
 "jobs": [{"job": "slab_transient", "part": "plate",
           "params": {"alpha": 1e-5, "t_left": 120, "t_right": 20, "t0": 20, "length": 0.2, "sensor": "TC-01"}}],
 "physics": {"gravity": [0, -9.81, 0],
             "parts": {"plate": {"rigid_body": false, "collider": true,
                                  "material": {"static_friction": 0.4, "dynamic_friction": 0.3}}}}}
```

```python
from pinneapple_unity import unity_to_usd
out = unity_to_usd("out/bracket_unity", "out/bracket_twin")     # -> out/bracket_twin/scene.usda
print(out["decision"], [(r["job"], r["decision"]) for r in out["reports"]])
```

`from_unity` reads the glTF (node transforms applied) into a Z-up `Scene`, `apply_link` runs each job and attaches its
field and checks, and `scene.export(..., usd=True)` writes the twin plus an OpenUSD layer with:

* every field as a time-sampled `primvars:<name>` with its unit in `customData`;
* `UsdPhysics`: `PhysicsScene` with gravity, `PhysicsCollisionAPI` / `PhysicsMeshCollisionAPI` on the parts,
  `PhysicsMaterialAPI` contact materials bound through `material:binding:physics`, rigid bodies and kinematic replay
  where the link says so;
* provenance in the layer metadata: title, source, units and the real time values.

A job that raises is reported (`decision: FAILED`, `error`), never swallowed.

### Live lane: the runtime asks while it plays

```bash
python -m pinneapple_unity serve --port 8766          # --token SECRET requires X-Pinneapple-Token
```

| verb | body | answer |
|---|---|---|
| `GET /v1/health` | — | `ok`, protocol, package version, jobs, requests, warnings |
| `GET /v1/jobs` | — | every registered job and what it computes |
| `POST /v1/run` | `{job, part, vertices | folder, times, params, encoding}` | the field(s), the checks, the decision |
| `POST /v1/scene` | `{folder, jobs?, save?, usd?}` or `{scene, jobs, save?}` | every field of every part, the reports, the sensors, and — with `save` — the Unity folder it wrote |

Field payloads are float32 in base64 by default (`{"steps", "count", "unit", "encoding": "base64", "data"}`), or plain
JSON numbers with `"encoding": "json"`. The Unity component `PinneappleBridgeClient` sends the same bodies from a
coroutine, parses the same answers (`PinneappleBridgeClient.Parse`), and `RunAndApply()` shows the field it asked for:
`PinneappleTwinView.ApplyField(part, name, steps, count, values, unit)` adds it to the loaded twin in memory and
switches the legend to it, so the live loop closes without touching the files on disk.

`save` closes the loop the other way: the bridge writes the computed twin as a Unity project folder
(`export_for_unity`) and answers `{"twin": {"folder", "manifest", "usd", "problems", "summary"}}`. It follows the
same `allow_folders` switch as `folder`.

The bridge binds `127.0.0.1` because it runs physics on demand: opening it to a network is a decision, not a default.
`warm_up()` pays the library import (torch) at start-up, so the first request is milliseconds instead of tens of
seconds; `allow_folders=False` refuses requests that name a path on the machine.

## The jobs

| job | field | checks |
|---|---|---|
| `slab_transient` | `temperature` [C] — transient 1-D conduction (Carslaw & Jaeger) | heat-equation residual and Dirichlet walls (trust card), maximum principle on the exported field, the first-Fourier-mode bound on the approach to steady |
| `cantilever_stress` | `bending_stress` [MPa], optional `fatigue_life_cycles` | root stress against `F L / W` recomputed from statics, the `1 - x/L` moment line, the stress-free tip, the yield decision, Basquin life against a target |
| `fin_temperature` | `temperature` [C] — pin-fin profile (Incropera Table 3.4 case A) | base at the base temperature, cooler tip, integrated convection equals the closed-form heat dissipated, values inside the range |

Parameters are SI and every one of them is a parameter, not a material claim: `alpha` defaults to 1e-5 m²/s (the order
of a steel), `k` to 200 W/mK. `axis` (`x`, `y`, `z` or `auto` = the part's longest side) and `flip` (the root or hot
wall at the high end) decide how the vertices are mapped onto the profile.

A job is three lines — a callable `JobRequest -> JobResult`:

```python
from pinneapple_unity import JobRequest, JobResult, register_job

def my_job(req: JobRequest) -> JobResult:
    values = ...                      # (steps, vertices) from the library
    return JobResult(req.part, req.job, {"my_field": (values, "mm")},
                     checks=[{"name": "conservation", "passed": True, "kind": "physics",
                              "value": 0.0, "threshold": 1e-6, "detail": "", "critical": True}])

register_job("my_job", my_job, "what it computes")
```

`decision` is `APPROVED` (every check ran and passed), `REJECT` (a critical check failed) or `REVIEW` (something did
not run). That decision is the reason to route physics through PINNeAPPle at runtime instead of trusting a number.

## OpenUSD, natively: Unity's USD Toolkit

Since Unity 6.4 there is a first-party way to move stages in and out of the editor and of built players:
[`com.unity.usd-toolkit`](https://github.com/Unity-Technologies/com.unity.usd-toolkit) wraps Pixar's OpenUSD 26.05
through P/Invoke. It reads and writes the same `.usd/.usda/.usdc/.usdz` files Isaac Sim and Omniverse use, turns
`UsdPhysics` into rigidbodies, colliders and `ArticulationBody` chains, keeps references, payloads and variants
through a round trip, plays `UsdSkel`/point-cache animation, and ships the Live Sync samples. That is the target the
`scene.usda` PINNeAPPle writes was built for, so the two lanes meet on one file:

```
PINNeAPPle twin  --scene.usda (time-sampled primvars + UsdPhysics + provenance)-->  Unity USD Toolkit import
Unity scene      --UsdExporter.ExportGameObjectWithResult-->  stage  --from_usd-->  jobs, checks  -->  .usda
```

Installing it in a project (Unity 6.4+; the package is experimental, pin a tag):

```bash
python -m pinneapple_unity.usd manifest ~/PinneappleUnityTest      # writes the line below into Packages/manifest.json
# "com.unity.usd-toolkit": "https://github.com/Unity-Technologies/com.unity.usd-toolkit.git#v0.8.1-exp.1"
```

Then the component **PINNeAPPle → USD Stage** (`PinneappleUsdStage.cs`) imports `Assets/TwinData/scene.usda` in
Play mode or in the editor (`PINNeAPPle > USD > Preview / Import Selected Stage`), with *Import Physics* on to get
the colliders and rigid bodies the stage declares, and exports any hierarchy back out
(`PINNeAPPle > USD > Export Selection As USD`). The file compiles with or without the package: the runtime asmdef
declares a `versionDefines` entry for `com.unity.usd-toolkit`, so without it the component tells you what to install
instead of breaking the build.

What the import will find is printed before anything is built — the same summary on both sides of the seam:

```bash
python -m pinneapple_unity.usd summary out/bracket_twin/scene.usda
# #usda 1.0   up axis Z   1.0 m per unit   frames 0.0..5.0
# prims: Material x1, Mesh x2, PhysicsScene x1, Scope x1, Sphere x1, Xform x1
# apiSchemas: MaterialBindingAPI x2, PhysicsCollisionAPI x2, PhysicsMassAPI x1, PhysicsMaterialAPI x1,
#             PhysicsMeshCollisionAPI x2, PhysicsRigidBodyAPI x1, PhysicsScene x1
# time-sampled: primvars:bending_stress, primvars:temperature
# primvars: primvars:displayColor, primvars:temperature [C], primvars:bending_stress [MPa]
# provenance: {"pinneapple_format": "pinneapple-twin3d/1", ...}
```

And it is checked against OpenUSD itself, not only against this repository's reader: `pinneapple_unity.usd.validate`
opens the stage with `usd-core` (the library the toolkit wraps) and lists dead rels, meshes without points and time
samples outside the layer's range — our exported layers return an empty list. `pinneapple_unity.usd.from_usd` is the
inverse: a stage authored in Unity or Isaac Sim becomes a `pinneapple_twin3d.Scene` (meshes, time-sampled primvars as
fields, `UsdPhysics` as colliders, the stage's units and up axis), so the jobs run on someone else's stage and the
result goes back out as USD.

Both `validate` and `from_usd` need `pip install usd-core`; the text summary does not.

## A surrogate in the loop

The same seam that carries a closed-form field carries a **surrogate**: a network trained here, checked here, and
shipped to the runtime (`export_onnx` → Sentis / Inference Engine) so Unity can evaluate the field at any instant
instead of playing back baked steps. `examples/unity/04_surrogate_turntable.py` is the reference:

```python
model, predict, out = train_surrogate(samples=6000, epochs=1500)   # pinneapple_neural.Trainer, log-uniform time
card, rel = check_surrogate(predict)                              # trust card on pde="heat" + relative L2
assert rel < 0.02 and card.decision == "APPROVED"                 # a fit nobody checked is not a surrogate
field = predict(vertices[:, 0], t)                                # what the twin colours, at any t
```

What the checks say about this one: relative L2 **0.17 %** against the closed form, heat-equation residual **0.031**
against a 0.05 tolerance — the card approved it. Two things the demo makes explicit rather than hiding:

* the training window starts after the initial-condition jump (`t = 0` is discontinuous at both walls); a network
  asked to fit that singularity spends its capacity on something no user ever sees, and the card would measure the
  jump, not the physics;
* a seed that misses the target is retried and printed. Fits that diverge after converging are a fact of Adam at a
  fixed rate; a demo that silently keeps the bad one would be lying about what is on screen.

Ship it to Unity with `pinneapple_tools.model_export.export_onnx(model, "surrogate.onnx", example_input)`: the
`.onnx` drops into `Assets/`, Sentis builds a `Worker` from it, and the script feeds it the same `(x, log t)`
inputs the demo used -- one network, the same numbers, on the GPU the player already has.

## In Unity

| | |
|---|---|
| `PinneappleTwinView` | loads the folder, builds the parts, animates the field over the steps (`field`, `colormap`, `secondsPerStep`, `colorize`), replays motions and visibility, marks the sensors, draws the legend in IMGUI; `ApplyField(...)` shows a field computed at runtime, `SetField`, `ShowStep`, `Load` |
| `PinneappleBridgeClient` | `Run(response => ...)` in a coroutine: sends geometry + times, gets fields + checks + decision; `RunAndApply(...)` also shows the field |
| `PINNeAPPle > Check Twin Data` | re-reads `Assets/TwinData` and logs every inconsistency |
| `PINNeAPPle > Export Selection as Scene Link` | writes `Assets/TwinLink/scene.glb` + `pinneapple_link.json` for the selection |
| `PINNeAPPle > Copy Bridge Command` | puts the bridge command on the clipboard |

The runtime picks `PINNeAPPle/Field URP` when it exists, then `PINNeAPPle/Field` (Built-in), and falls back to an
unlit material without the field with a warning — it never fails silently on a shader.

## Checking the Unity side without opening the editor

The Hub ships a headless CLI, and the editor compiles the project in batch mode -- the two commands that turn
"it compiled on my machine" into something a CI runner can do:

```bash
"/Applications/Unity Hub.app/Contents/MacOS/Unity Hub" -- --headless editors --releases     # what is available
"/Applications/Unity Hub.app/Contents/MacOS/Unity Hub" -- --headless install \
    --version 6000.0.85f2 --architecture arm64                                              # one editor, ~5 GB

UNITY=/Applications/Unity/Hub/Editor/6000.0.85f2/Unity.app/Contents/MacOS/Unity
"$UNITY" -batchmode -quit -nographics -projectPath <project> \
         -executeMethod PinNeAPPle.Unity.EditorTools.PinneappleMenu.CheckTwinData \
         -logFile /tmp/unity-check.log ; echo "exit=$?"
grep -nE "error CS|PINNeAPPle:" /tmp/unity-check.log
```

`exit=0` plus a `PINNeAPPle: ... every file checks out` line means the scripts compiled **and** the editor menu read
the twin. A compile error shows up as `error CS` with file and line, and makes the exit code non-zero.

## What the editor verified (Unity 6000.6.8f1, macOS)

Run headless on this checkout's output, with the project created by `-createProject` and the twin copied into
`Assets/`:

```bash
UNITY="/Applications/Unity/Hub/Editor/6000.6.8f1/Unity.app/Contents/MacOS/Unity"
"$UNITY" -batchmode -quit -nographics -projectPath ~/PinneappleUnityTest \
         -executeMethod PinNeAPPle.Unity.EditorTools.PinneappleMenu.CheckTwinData -logFile /tmp/u3.log
# exit=0, 0 error CS
# PINNeAPPle: Heat-exchanger spool with a cooling fin: 2 part(s), 2 field(s), 6 step(s) -- every file checks out
```

and with `com.unity.usd-toolkit#v0.8.1-exp.1` in `Packages/manifest.json`:

```bash
"$UNITY" ... -executeMethod PinNeAPPle.Unity.EditorTools.PinneappleMenu.UsdStageCheck -logFile /tmp/u6.log
# exit=0, 0 error CS
# PINNeAPPle USD: scene.usda -- 4 mesh(es), 4288 triangle(s), up axis Z
```

Three things the real compiler settled, which the documentation alone would have got wrong:

* `UsdImportOptions.VariantSelections` is a `List<UsdVariantSelection>` (`primPath, variantSet, variant`), not a
  `string[]`; `UsdExporter.ExportGameObjectWithResult` takes a **GameObject**, not a `Transform`.
* our assembly declaration must reference the toolkit's assembly by name (`"references": ["Unity.USDToolkit"]`) --
  the package's `autoReferenced` only reaches `Assembly-CSharp`, not another asmdef.
* Unity 6 does **not** define `UNITY_2023_1_OR_NEWER`; the version guard for `FindFirstObjectByType` needs
  `|| UNITY_6000_0_OR_NEWER`.

Removing the toolkit from the manifest and compiling again gives 0 errors too (the `#if PINNEAPPLE_USD_TOOLKIT`
guard and the `versionDefines` entry keep the unreferenced assembly harmless), so the package works in a project
with and without it. What is still not exercised here: Play mode, the camera/lighting look, and the actual
`ImportAsync`/`ExportGameObjectWithResult` calls (only `GetPreviewInfo` was run headless).

## What is checked, and what is not

| | how |
|---|---|
| the C# data layer (`PinNeappleJson`, `PinNeappleGlb`, `PinNeappleTwinData`, `PinNeappleColormap`) | no `UnityEngine` types; compiled with C# 9 and run against a real exported twin by `tests/test_unity_csharp.py` (`dotnet run --project pinneapple_unity/unity/tests -- <folder>`), including a GLB read → write → read round trip |
| the file, link and live lanes | `tests/test_unity_integration.py`: export/verify, ingest with the axis conversion, USD content, the three jobs and their rejections, the bridge over HTTP with a token |
| the MonoBehaviours, the editor menu, the two shaders | compiled by Unity, not by this repository's CI. A mistake there is a console compile error, not a wrong number — keep those files small, as they are |

## Notes and limits

* The Unity side is 2021.3 LTS or newer (C# 9, assembly definitions, `UnityWebRequest`); no package dependency, no
  URP requirement beyond the shader variant.
* Per-vertex colouring is CPU work when a step changes: the view writes into **one `Color[]` per part, reused
  across steps** (a step allocates nothing) and uploads it with `Mesh.SetColors`. What remains is O(V) per changed
  step, comfortable to ~10^5 vertices; a million-vertex mesh wants a `ComputeBuffer` with all `(step, vertex)` values
  and a shader that samples it, not a texture (a texture caps at 16384 px per side, which a big mesh exceeds).
* Two ways to get a computed field into the view: in memory with `RunAndApply` (nothing written, gone when you stop
  Play), or as files with the bridge's `save` / `unity_to_usd` and a reimport (`load`). There is no live edit inside
  the editor itself: the editor menu writes the link, the runtime or a script writes the twin.
* Reference frames, units and provenance travel with the files (`pinneapple_unity.json`, link metadata, USD layer
  metadata) so a twin can be read a year later without the script that wrote it.
