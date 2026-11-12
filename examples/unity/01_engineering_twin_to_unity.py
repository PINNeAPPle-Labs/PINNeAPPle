"""PINNeAPPle -> Unity: an engineering twin, with fields the library computed and checked, in a Unity-ready folder.

    python examples/unity/01_engineering_twin_to_unity.py

A heat-exchanger spool with a cooling fin. Two fields, both from closed-form solutions in the library and both
carried by their checks:

    spool  transient 1-D conduction   (Carslaw & Jaeger slab)   -> temperature [C]
    fin    pin-fin excess temperature (Incropera 3.4 case A)    -> temperature [C]

The export is a folder a Unity project imports as it is -- the data (`Assets/TwinData`) plus the C# runtime
(`Assets/PinNeAPPleTwin`) that builds the parts, colours the field per vertex over the six time steps, plays the
sensors and draws the legend. Nothing about the twin changes on the way into the engine: same scene.json, same
geometry.glb, same fields.bin the web viewer reads.

Prints the check table, the folder that was written, what `verify()` found, and the four steps in Unity.
"""
from __future__ import annotations

import os

import numpy as np

from pinneapple_twin3d import Scene
from pinneapple_twin3d.primitives import sweep
from pinneapple_unity import JobRequest, export_for_unity, run_job

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out", "spool_unity")
TIMES = [0.0, 30.0, 120.0, 600.0, 1800.0, 3600.0]          # seconds of heat-up


def _circle(radius: float, n: int = 24):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)


def _square(half: float):
    return np.array([(-half, -half), (half, -half), (half, half), (-half, half)], dtype=float)


def _line(start, length: float, n: int = 41):
    s = np.linspace(0.0, length, n)
    return np.stack([start[0] + s, np.full(n, start[1]), np.full(n, start[2])], axis=1)


def build_twin() -> Scene:
    """The spool (a swept tube) and its fin, with the two sensors."""
    scene = Scene("Heat-exchanger spool with a cooling fin", length_unit="m", times=TIMES, time_unit="s",
                  source="examples/unity/01_engineering_twin_to_unity.py")
    v, f = sweep(_circle(0.03), _line((0.0, 0.0, 0.0), 0.4))
    scene.add_part("spool", v, f, group="piping", color=(0.72, 0.75, 0.80))
    v, f = sweep(_square(0.002), _line((-0.06, 0.0, 0.0), 0.06))     # the fin runs from the hot flange outwards
    scene.add_part("fin", v, f, group="cooling", color=(0.85, 0.76, 0.55))
    scene.add_sensor("TC-01", (0.2, 0.0, 0.03), label="TC-01 spool, mid length", unit="C")
    scene.add_sensor("TC-02", (-0.06, 0.0, 0.0), label="TC-02 fin tip", unit="C")
    return scene


def main() -> None:
    scene = build_twin()
    reports = []

    # 1. the hot flange is at x = 0 (u = 0 of the closed form); 180 C in, 30 C at the cold end
    spool = next(p for p in scene.parts if p.name == "spool")
    res = run_job(JobRequest("slab_transient", "spool", spool.vertices, times=TIMES,
                             params={"alpha": 4e-6, "t_left": 180.0, "t_right": 30.0, "t0": 30.0,
                                     "axis": "x", "length": 0.4, "sensor": "TC-01"}))
    for name, (values, unit) in res.fields.items():
        scene.add_field("spool", name, values, unit=unit)
    reports.append(res)

    # 2. the fin hangs off that flange: its base is at the high end of its axis, so flip the parametrization
    fin = next(p for p in scene.parts if p.name == "fin")
    res = run_job(JobRequest("fin_temperature", "fin", fin.vertices, times=TIMES,
                             params={"h": 60.0, "k": 205.0, "D": 0.004, "t_base": 180.0, "t_inf": 25.0,
                                     "axis": "x", "flip": True}))
    for name, (values, unit) in res.fields.items():
        scene.add_field("fin", name, values, unit=unit)
    reports.append(res)

    # 3. the sensors read the fields at the vertex nearest to them (the job names the sensor, nothing is guessed)
    for sensor_id, report in (("TC-01", reports[0]), ("TC-02", reports[1])):
        sensor = next(s for s in scene.sensors if s.id == sensor_id)
        part = next(p for p in scene.parts if p.name == report.part)
        k = int(np.argmin(np.linalg.norm(part.vertices - np.asarray(sensor.position, float), axis=1)))
        sensor.series = np.asarray(report.fields["temperature"][0][:, k], float)
        sensor.unit = "C"

    print("fields computed by the library, each with the checks that decided them")
    for report in reports:
        print(f"\n{report.job} on {report.part}: {report.decision}")
        for c in report.checks:
            mark = {True: "PASS", False: "FAIL", None: "n/a "}[c["passed"]]
            value = "" if c["value"] is None else f"  value {c['value']:.4g}"
            print(f"   {mark}  {c['name']}{value}")
        for note in report.notes:
            print(f"        {note}")

    # 4. the Unity folder
    exp = export_for_unity(scene, OUT, usd=True)          # the stage travels with the twin
    print(f"\nexported: {exp.summary()}")
    print("verify():", exp.problems or "every file checks out (glb, field offsets, package)")
    print("\nfiles:")
    for root, _dirs, files in os.walk(OUT):
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(root, name), OUT)
            print(f"   {rel}")

    print("""
In Unity
  1. copy  examples/unity/_out/spool_unity/Assets  into a Unity project (2021.3 LTS or newer)
  2. add an empty GameObject with the component  PINNeAPPle > Twin View
  3. Twin Folder = Assets/TwinData, press Play: six time steps, the spool cooling from the hot flange,
     the fin profile, both sensors with their series
  4. PINNeAPPle > Check Twin Data re-reads the folder in the editor and reports anything inconsistent

Without Unity installed, the same three data files can be re-read here:
  python -m pinneapple_unity verify examples/unity/_out/spool_unity
  dotnet run --project pinneapple_unity/unity/tests -- examples/unity/_out/spool_unity     (C# data layer)""")


if __name__ == "__main__":
    main()
