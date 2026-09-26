"""Demo scene: a 90-degree pipe bend with erosion wear growing over time, plus two sensors.

Mirrors the PINNeAPPle-CFD v1 case (bend with sand, wear concentrated on the outer wall of the
bend). The wear field here is a smooth *illustrative* pattern, not a CFD result: use it to try the
viewer, then replace it with the fields of a real run.

    from pinneapple_twin3d.demo import pipe_bend_scene
    pipe_bend_scene().export("out/bend"); serve("out/bend")
"""
from __future__ import annotations

import numpy as np

from .scene import Scene


def _bend(R=0.3, r=0.05, n_arc=80, n_theta=48, angle=np.pi / 2):
    phi = np.linspace(0, angle, n_arc)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    P, T = np.meshgrid(phi, th, indexing="ij")
    rho = R + r * np.cos(T)  # distance from the bend axis; cos(T)=1 is the outer wall (extrados)
    v = np.stack([rho * np.cos(P), rho * np.sin(P), r * np.sin(T)], -1).reshape(-1, 3)
    faces = []
    for i in range(n_arc - 1):
        for j in range(n_theta):
            a, b = i * n_theta + j, i * n_theta + (j + 1) % n_theta
            faces += [[a, b, a + n_theta], [b, b + n_theta, a + n_theta]]
    return v, np.array(faces), P.ravel(), T.ravel()


def _straight(p0, direction, r=0.05, length=0.3, n_len=20, n_theta=48):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(d, e1)
    s = np.linspace(0, length, n_len)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    S, T = np.meshgrid(s, th, indexing="ij")
    v = (np.asarray(p0)[None, :] + S.reshape(-1, 1) * d
         + r * (np.cos(T).reshape(-1, 1) * e2 + np.sin(T).reshape(-1, 1) * e1))
    faces = []
    for i in range(n_len - 1):
        for j in range(n_theta):
            a, b = i * n_theta + j, i * n_theta + (j + 1) % n_theta
            faces += [[a, b, a + n_theta], [b, b + n_theta, a + n_theta]]
    return v, np.array(faces)


def pipe_bend_scene(years=(0, 1, 2, 3, 4, 5)) -> Scene:
    times = np.asarray(years, float)
    sc = Scene("Demo: 90° bend with sand erosion", length_unit="m", times=times, time_unit="year",
               source="pinneapple_twin3d.demo (illustrative field, not a CFD result)")
    v, f, P, T = _bend()
    sc.add_part("bend", v, f, group="piping", color=(0.72, 0.74, 0.78))
    # wear hot spot on the extrados, ~2/3 along the bend (where the particles hit), growing with time
    hot = np.exp(-((P - 0.62 * np.pi / 2) / 0.28) ** 2) * np.clip(np.cos(T), 0, None) ** 3
    sc.add_field("bend", "wear_depth", np.outer(0.35 * times, hot), unit="mm")
    sc.add_field("bend", "wall_shear", (4 + 6 * hot)[None, :].repeat(len(times), 0), unit="Pa")
    vi, fi = _straight([0.3, 0.0, 0.0], [0, -1, 0])
    vo, fo = _straight([0.0, 0.3, 0.0], [-1, 0, 0])
    sc.add_part("inlet", vi, fi, group="piping", color=(0.62, 0.66, 0.72))
    sc.add_part("outlet", vo, fo, group="piping", color=(0.62, 0.66, 0.72))
    sc.add_sensor("UT-01", (0.3 * np.cos(0.97) + 0.05 * np.cos(0.97), 0.35 * np.sin(0.97), 0.0),
                  label="UT-01 thickness loss", unit="mm", quantity="wear",
                  series=0.35 * times * 0.93, envelope=(0.0, 1.2))
    sc.add_sensor("PT-101", (0.3, -0.3, 0.06), label="PT-101 inlet", unit="bar",
                  series=[6.1, 6.0, 6.0, 5.9, 5.9, 5.8], envelope=(5.5, 7.0))
    return sc
