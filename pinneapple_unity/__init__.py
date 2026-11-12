"""PINNeAPPle <-> Unity: 3-D visualisation in the engine, physics from the library, OpenUSD on the way out.

Three lanes, one protocol (``pinneapple-unity/1``, see :mod:`pinneapple_unity.protocol`):

**File lane -- a PINNeAPPle twin in Unity.** :func:`export_for_unity` writes a folder whose ``Assets`` a Unity
project takes as it is: the same ``scene.json`` + ``geometry.glb`` + ``fields.bin`` the web viewer reads, plus the
C# runtime (``Assets/PinNeAPPleTwin``) that builds the parts, colours the field per vertex over time, plays rigid
motions and shows the sensors::

    from pinneapple_twin3d.demo import pipe_bend_scene
    from pinneapple_unity import export_for_unity
    exp = export_for_unity(pipe_bend_scene(), "out/bend_unity")
    assert exp.verify() == []           # the GLB, the field offsets and the package, re-read and checked

**Link lane -- a scene authored in Unity becomes physics + OpenUSD.** Unity writes a glTF and a
``pinneapple_link.json`` (parts, units, sensors, ``UsdPhysics``, and the *jobs* to run). :func:`from_unity` reads it,
:func:`apply_link` runs the jobs (with their checks), and the scene exports as OpenUSD with time-sampled fields::

    from pinneapple_unity import unity_to_usd
    out = unity_to_usd("out/gripper_unity", "out/gripper_twin")     # -> out/gripper_twin/scene.usda
    print(out["decision"], [r["job"] for r in out["reports"]])

**Live lane -- the runtime asks PINNeAPPle while it plays.** :func:`serve_bridge` starts an HTTP bridge
(``python -m pinneapple_unity serve``); the Unity component ``PinneappleBridgeClient`` posts a part and gets the
field back with the checks, without leaving Play mode::

    from pinneapple_unity import serve_bridge
    srv = serve_bridge(port=8766, block=False)
    srv.post("/v1/run", {"job": "slab_transient", "vertices": [[0, 0, 0], [0.2, 0, 0]], "times": [0, 900]})

Jobs are the extension point: :func:`register_job` a ``JobRequest -> JobResult`` callable and it is available to
the link, the bridge and the demos. :func:`list_jobs` says what is registered.

Axis note: twins and OpenUSD are Z-up, Unity and glTF are Y-up; the conversion is one rotation
(``(x, y, z) -> (x, z, -y)``) applied on export, on ingest, and by the C# runtime (recorded in
``pinneapple_unity.json`` so the hand-off carries the convention with it).
"""
from __future__ import annotations

__version__ = "1.0.0"          # of the pinneapple-unity protocol package

from .bridge import Bridge, serve_bridge  # noqa: E402
from .export import (  # noqa: E402
    UnityExport,
    export_for_unity,
    read_unity_twin,
    scene_from_twin,
    verify_unity_folder,
)
from .ingest import (  # noqa: E402
    apply_link,
    emulate_unity_export,
    from_unity,
    read_link,
    unity_to_usd,
)
from .jobs import (  # noqa: E402
    JOBS,
    JobRequest,
    JobResult,
    list_jobs,
    register_job,
    run_job,
    unregister_job,
)
from .protocol import (  # noqa: E402
    LINK_FILE,
    PROTOCOL,
    UNITY_FILE,
    Z_UP_TO_Y_UP,
    Link,
    LinkJob,
    LinkPart,
    decode_field,
    encode_field,
    y_up_to_z_up,
    z_up_to_y_up,
)


def __getattr__(name: str):
    """``pinneapple_unity.usd`` (the native OpenUSD lane) loads on first use, so ``python -m pinneapple_unity.usd``
    does not import it twice."""
    if name == "usd":
        import importlib

        return importlib.import_module(f"{__name__}.usd")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # file lane
    "export_for_unity", "UnityExport", "verify_unity_folder", "read_unity_twin", "scene_from_twin",
    # link lane
    "from_unity", "apply_link", "unity_to_usd", "emulate_unity_export", "read_link",
    "Link", "LinkPart", "LinkJob", "LINK_FILE", "UNITY_FILE",
    # live lane
    "serve_bridge", "Bridge",
    # jobs
    "JOBS", "JobRequest", "JobResult", "register_job", "unregister_job", "list_jobs", "run_job",
    # native OpenUSD
    "usd",
    # wire format
    "PROTOCOL", "encode_field", "decode_field", "z_up_to_y_up", "y_up_to_z_up", "Z_UP_TO_Y_UP",
    "__version__",
]
