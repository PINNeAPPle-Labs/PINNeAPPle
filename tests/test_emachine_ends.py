"""End effects: filament inductance against closed forms, end-winding inductance, and the drive with it."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.statics import (
    filament_inductance,
    loop_self_inductance,
    mutual_inductance_coaxial_loops,
)


def circle(R: float, z: float = 0.0, n: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n + 1)
    return np.column_stack([R * np.cos(t), R * np.sin(t), np.full_like(t, z)])


def test_loop_self_inductance_is_rosa():
    R, a = 0.05, 1e-3
    L = filament_inductance([circle(R, n=192)], a, pieces_per_metre=2000.0)[0, 0]
    assert L == pytest.approx(loop_self_inductance(R, a, internal=True), rel=2e-3)


def test_mutual_of_coaxial_loops_converges_to_maxwell():
    R1, R2, d = 0.05, 0.04, 0.02
    ref = mutual_inductance_coaxial_loops(R1, R2, d)
    errs = []
    for n in (24, 48, 96):
        M = filament_inductance([circle(R1, 0.0, n), circle(R2, d, n)], 1e-4, pieces_per_metre=1.0)
        errs.append(abs(M[0, 1] - ref) / ref)
    assert errs[-1] < 1e-3
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0  # polygon chords: second order


def test_iron_mirror_adds_the_image_loop():
    R, a, h = 0.05, 1e-3, 0.01
    P = circle(R, h, 192)
    L_air = filament_inductance([P], a, pieces_per_metre=2000.0)[0, 0]
    L_mir = filament_inductance([P], a, pieces_per_metre=2000.0, mirror_z=0.0)[0, 0]
    assert L_mir - L_air == pytest.approx(mutual_inductance_coaxial_loops(R, R, 2 * h), rel=2e-3)


def test_end_winding_inductance_of_the_prius():
    from pinneapple_simulation.electromagnetics.machine import belt_winding
    from pinneapple_simulation.electromagnetics.machines import prius_2004
    from pinneapple_simulation.electromagnetics.machines.endwinding import end_winding_inductance

    out = end_winding_inductance(prius_2004(), belt_winding(48, 4), 9.0, 6, 0.019366, pieces_per_metre=100.0)
    assert out["coils"] == 24
    for k in ("air", "iron_face"):
        assert out[k]["L_aa"] > 0 and 0 < out[k]["L_end_dq"] < 1e-3


def test_drive_end_winding_is_an_equal_series_inductance_on_both_axes():
    """L_end adds to L_d and L_q alike, so the torque 1.5 p (psi_m i_q + (L_d - L_q) i_d i_q) is unchanged and only the
    voltage limit sees it: the drive equals a linear machine with both inductances raised by L_end."""
    from pinneapple_simulation.electromagnetics.machines.drive import Drive, LinearDQ, v_max_svpwm

    Le = 0.15e-3
    kw = dict(R=0.0, I_max=250.0, V_max=v_max_svpwm(500.0), n_d=121)
    d1 = Drive(LinearDQ(L_d=0.6e-3, L_q=2.4e-3, psi_m=0.2, pole_pairs=4), **kw, L_end=Le)
    d2 = Drive(LinearDQ(L_d=0.6e-3 + Le, L_q=2.4e-3 + Le, psi_m=0.2, pole_pairs=4), **kw)
    for rpm in (500, 3000, 6000):
        w = rpm * 2 * math.pi / 60
        assert d1.max_torque(w)["T"] == pytest.approx(d2.max_torque(w)["T"], rel=1e-9)
