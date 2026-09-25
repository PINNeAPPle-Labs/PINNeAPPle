"""3D digital-twin visualization: build a scene in Python, open it in the bundled web viewer.

    from pinneapple_twin3d import Scene, serve
    sc = Scene("CFD-04 bend", times=[0, 1, 2], time_unit="year")
    sc.add_trimesh("bend", trimesh.load("bend.stl"), group="pipe")
    sc.add_field("bend", "wear_depth", wear_TxV, unit="mm")
    sc.add_sensor("PT-101", (0.1, 0.0, 0.2), unit="bar", series=[4.1, 4.0, 3.8], envelope=(3.5, 5.0))
    sc.export("out/twin")          # scene.json + geometry.glb + fields.bin + viewer
    serve("out/twin")              # http://localhost:8765/  (add ?live=ws://... for live sensors)

Decision D1 in docs/dev/PEDIDOS_2026-09-24.md: the viewer lives in the library so that
PINNeAPPle-CFD (3D wear over time), PINNeAPPle-apps and pinneapple_systems.digital_twin all
reuse it.
"""
from .scene import Part, Scene, Sensor
from .serve import serve

__all__ = ["Part", "Scene", "Sensor", "serve"]
