# PINNeAPPle Twin for Unity

The Unity half of the PINNeAPPle integration: a twin exported from Python (`pinneapple_unity.export_for_unity`)
opens in Unity as it is — parts, a per-vertex field that plays over the scene's time steps, rigid motions, sensors
with their series and alarm envelope — and, while playing, the runtime can ask a PINNeAPPle bridge for a new field
and get it back with its physics checks.

## Drop it in

1. `Assets/PinNeAPPleTwin` — this folder: the runtime, the two field shaders, the editor menu.
2. `Assets/TwinData` — the data (`scene.json`, `geometry.glb`, `fields.bin`, `pinneapple_unity.json`), written by
   `pinneapple_unity.export_for_unity(scene, "out/my_twin")`.

Both come from one export call: point Unity at `out/my_twin` (or copy its `Assets` into your project). Built-in,
URP and HDRP all work: `PINNeAPPle/Field` is the Built-in shader, `PINNeAPPle/Field URP` the URP one, and the runtime
picks whichever exists (falling back to an unlit material without the field if neither is found — the console says so).

## Use it

Add **PINNeAPPle Twin View** (`PINNeAPPle/Twin View`) to an empty GameObject, set *Twin Folder* (`Assets/TwinData`),
press Play. The component exposes:

| | |
|---|---|
| `twinFolder` | the data folder, relative to the project root or absolute |
| `field` | which field to colour (`""` = the first); `SetField("temperature")` at runtime |
| `colormap` | `turbo` (default) or `viridis`, with the legend, the unit and the range on screen |
| `play`, `secondsPerStep` | walk the twin's steps; `ShowStep(k)` to drive it yourself |
| `colorize` | off = parts in their own colours, geometry only |

Add **PINNeAPPle Bridge Client** (`PINNeAPPle/Bridge Client`) to the same object to bring physics in live:

```
python -m pinneapple_unity serve --port 8766          # the bridge, on the machine that runs Python
```

then in Play mode

```csharp
StartCoroutine(client.RunAndApply(r => Debug.Log(r.Decision)));   // ask, then show it on the twin
```

`RunAndApply` sends the part's geometry and the twin's times, gets back the field (steps × vertices) with its checks
and applies it through `PinneappleTwinView.ApplyField(part, name, steps, count, values, unit)` -- the legend switches
to the new field and its range. Use `Run(...)` instead when you want the data without showing it. The request runs in
a coroutine: the frame never waits, and nothing is written to disk: a computed field lives in memory for the session.
`r.Checks` carries the same residual / boundary-condition / design checks the library reports, and `r.Decision` is
the verdict on them.

## Native USD (optional, Unity 6.4+)

With Unity's [USD Toolkit](https://github.com/Unity-Technologies/com.unity.usd-toolkit) installed, the component
**PINNeAPPle → USD Stage** (`PinneappleUsdStage`) imports `Assets/TwinData/scene.usda` through `UsdImporter`
(materials, textures, animation and — with *Import Physics* on — rigid bodies, colliders and `ArticulationBody`
chains from the stage's `UsdPhysics`), and exports a hierarchy back to a stage with `UsdExporter`
(`PreserveHierarchy`, composition preserved). Without the package the component logs the install line and the rest of
this package keeps working: the asmdef only defines `PINNEAPPLE_USD_TOOLKIT` when the toolkit is present.

Add it to a project from Python: `python -m pinneapple_unity.usd manifest <project>`. The runtime assembly
references `Unity.USDToolkit` by name, so the toolkit's types are compile-checked (against 0.8.1 here); without the
package the `#if`-guarded code is skipped and the assembly still compiles. `PINNeAPPle > USD > Check
Assets/TwinData stage` prints what the importer sees, and works headless with `-executeMethod`.

## Menu

| Item | What it does |
|---|---|
| `PINNeAPPle > Check Twin Data` | re-reads `Assets/TwinData` and reports every inconsistency in the console |
| `PINNeAPPle > Open Twin Data Folder` | opens the data folder |
| `PINNeAPPle > Export Selection as Scene Link` | writes `Assets/TwinLink/scene.glb` + `pinneapple_link.json` for the selection, with a job template — the Unity → PINNeAPPle → OpenUSD lane (`python -m pinneapple_unity ingest Assets/TwinLink --out out/twin`) |
| `PINNeAPPle > Copy Bridge Command` | puts the bridge command on the clipboard |
| `PINNeAPPle > USD > Preview / Import Selected Stage` | `UsdImporter.GetPreviewInfo` and `ImportAsync` on the stage of the selected object |
| `PINNeAPPle > USD > Export Selection As USD` | `UsdExporter.ExportGameObjectWithResult` with the hierarchy preserved |

## What is checked here, and what is not

The data layer (`PinNeappleJson.cs`, `PinNeappleGlb.cs`, `PinNeappleTwinData.cs`, `PinNeappleColormap.cs`) has no
Unity types and is compiled and run against a real exported twin outside Unity:

```
dotnet run --project pinneapple_unity/unity/tests -- out/my_twin
```

That build is what `tests/test_unity_csharp.py` runs. The three MonoBehaviour/editor files (`PinneappleTwinView`,
`PinneappleBridgeClient`, `PinneappleMenu`) and the two shaders need the engine: they are written against the public
Unity API and kept small, but they are **compiled by Unity, not by this repository's CI** — if something is off, it
will be a compile error in the console, not a silent wrong number.

## Wire format

`pinneapple-unity/1` — see `pinneapple_unity/protocol.py` and the guide `docs/guides/unity_integration.md`.
