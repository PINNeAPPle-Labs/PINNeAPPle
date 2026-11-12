# PINNeAPPle <-> Unity

PINNeAPPle's physics into a Unity project, and a Unity scene back out as OpenUSD. Three lanes, one protocol
(`pinneapple-unity/1`), no Unity needed to test the data:

| lane | API | what moves |
|---|---|---|
| file | `export_for_unity(scene, folder)`, `verify_unity_folder`, `read_unity_twin`, `scene_from_twin` | a twin folder a Unity project imports as it is (`Assets/TwinData` + the C# runtime `Assets/PinNeAPPleTwin`) |
| link | `from_unity(folder)`, `apply_link(scene, link)`, `unity_to_usd(folder, out)`, `emulate_unity_export` | a Unity scene (`geometry.glb` + `pinneapple_link.json`) -> checked fields -> twin + OpenUSD layer |
| live | `serve_bridge()` / `python -m pinneapple_unity serve`, `Bridge.post` | HTTP `/v1/run` and `/v1/scene`: a field, its checks and a decision, while Unity plays; `save` writes the computed twin back as a Unity folder |

```python
from pinneapple_twin3d.demo import pipe_bend_scene
from pinneapple_unity import export_for_unity, serve_bridge, unity_to_usd

exp = export_for_unity(pipe_bend_scene(), "out/bend_unity")   # drag out/bend_unity/Assets into Unity
assert exp.verify() == []

out = unity_to_usd("out/bracket_unity", "out/bracket_twin")   # Unity folder -> twin + out/bracket_twin/scene.usda
print(out["decision"], [(r["job"], r["decision"]) for r in out["reports"]])

srv = serve_bridge(port=8766, block=False)                    # the runtime calls srv.url from a coroutine
```

Jobs (`slab_transient`, `cantilever_stress`, `fin_temperature`) come with their checks; `register_job` adds more.
The Unity component `PinneappleBridgeClient` is the client side of the live lane (`Run` returns the field with its
checks, `RunAndApply` also shows it through `PinneappleTwinView.ApplyField`), `PinneappleTwinView` the viewer.

* guide: [`docs/guides/unity_integration.md`](../docs/guides/unity_integration.md)
* examples: [`examples/unity/`](../examples/unity/README.md)
* Unity package (C# runtime, shaders, editor menu): [`unity/Assets/PinNeAPPleTwin`](unity/Assets/PinNeAPPleTwin/README.md)
* axis convention: twins and OpenUSD are Z-up, Unity and glTF are Y-up; the rotation `(x, y, z) -> (x, z, -y)` is
  applied on export, on ingest and by the runtime, and recorded in the manifests.
