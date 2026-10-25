"""pinneapple_design.thermal.meshes: closed shells, payload and STL layout."""
from __future__ import annotations

import struct
from collections import Counter

import numpy as np
import pytest

from pinneapple_design.thermal.meshes import (
    mesh_payload,
    plate_fin_mesh,
    ring_mesh,
    stl_bytes,
    vented_rotor_mesh,
)


def _closed(V, F):
    """Every edge of the welded mesh is shared by exactly two triangles."""
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    Fw = inv.reshape(-1)[F]
    edges = Counter(tuple(sorted(e)) for f in Fw for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))
    return all(c == 2 for c in edges.values())


def _volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_ring_is_closed_and_outward():
    V, F = ring_mesh(10.0, 20.0, 0.0, 5.0, n=256)
    assert _closed(V, F)
    polygon = 256 / 2 * np.sin(2 * np.pi / 256)          # area of the inscribed 256-gon of unit radius
    assert _volume(V, F) == pytest.approx(polygon * (20**2 - 10**2) * 5.0, rel=1e-9)


def test_parts_and_stl():
    V, F = plate_fin_mesh(80, 80, 3, 20, 1, 25)
    assert len(F) == 12 * 21 and V[:, 2].max() == 28 and V[:, 0].max() == 80
    V2, F2 = vented_rotor_mesh(120.0, 60.0, 6.0, 8.0, 36, 4.0)
    assert V2[:, 2].max() == 20.0
    p = mesh_payload(V, F)
    assert len(p["positions"]) == V.size and len(p["indices"]) == F.size
    data = stl_bytes(V, F)
    n = struct.unpack("<I", data[80:84])[0]
    assert n == len(F) and len(data) == 84 + 50 * n
