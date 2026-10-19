"""Step skew, tolerance propagation and 3-D export of the IPM machine (``machines.robustness``, ``machines.export``)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pinneapple_analysis.uncertainty.tolerance import propagate
from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
from pinneapple_simulation.electromagnetics.machines.export import solution_to_vtu, write_vtu
from pinneapple_simulation.electromagnetics.machines.robustness import (
    skewed_field_3d,
    skewed_no_load,
    skewed_torque,
    step_offsets,
    step_skew_factor,
    tolerance_model,
)


@pytest.fixture(scope="module")
def machine():
    return IPMMachine(prius_2004(), n_gap=576)


def test_step_skew_factor_closed_form():
    assert step_offsets(3, 0.1).tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert step_skew_factor(1, 2, 0.5, 1.0) == pytest.approx(0.0, abs=1e-15)  # half-period two-step skew
    assert step_skew_factor(2, 2, 0.5, 1.0) == pytest.approx(1.0)
    assert step_skew_factor(1, 3, 0.1, 1.0) == pytest.approx(abs(math.sin(0.3 * math.pi) / (3 * math.sin(0.1 * math.pi))))


def test_multislice_cogging_obeys_the_skew_factor(machine):
    """Each segment is a shifted copy: the harmonics of the skewed cogging are those of the straight rotor times the
    step-skew factor (exactly, up to the solver tolerance), when the offsets are multiples of the band step."""
    lam, N = machine.cogging_period, 12
    th = machine.positions(N, period=lam)
    c0 = np.array([machine.torque(machine.solve(t)) for t in th])
    s = 4 * machine.mm.band_step  # a third of the cogging period
    sk = skewed_no_load(machine, th, 3, s)["cogging"]
    F0, Fs = np.fft.rfft(c0) / N, np.fft.rfft(sk) / N
    h = np.arange(len(F0))
    x = h * np.pi * s / lam
    with np.errstate(all="ignore"):
        fac = np.where(np.abs(np.sin(x)) < 1e-12, np.cos(3 * x) / np.cos(x), np.sin(3 * x) / (3 * np.sin(x)))
    assert np.abs(Fs - F0 * fac).max() < 1e-6 * np.abs(F0[1:]).max()
    assert abs(Fs[1]) < 1e-6 * abs(F0[1])  # the fundamental cogging harmonic is gone


def test_skew_lowers_ripple_and_keeps_mean_torque(machine):
    th = machine.positions(12)
    straight = skewed_torque(machine, -160.0, 190.0, th, 1, 0.0)
    skewed = skewed_torque(machine, -160.0, 190.0, th, 2, machine.cogging_period / 2)
    assert np.ptp(skewed) < np.ptp(straight)
    assert abs(skewed.mean() - straight.mean()) / straight.mean() < 0.03


def test_tolerance_propagation_through_the_fea():
    fn = tolerance_model(prius_2004(), n_gap=288, n_positions=3)
    nominal = [1.0, 0.0065, 0.0189, 0.00075]
    res = propagate(fn, nominal, [0.01, 0.05e-3, 0.05e-3, 0.02e-3], n=8, seed=1, limits=(None, [np.inf, 1.0]))
    assert fn.calls == 8 and np.all(np.isfinite(res.outputs))
    T0 = fn(np.array([nominal]))[0, 0]
    assert abs(res.mean[0] - T0) < 4 * res.std[0] + 1.0


def test_vtu_export(tmp_path, machine):
    ms = machine.solve(0.0)
    p = solution_to_vtu(tmp_path / "m.vtu", ms)
    root = ET.parse(p).getroot()
    piece = root.find("UnstructuredGrid/Piece")
    assert int(piece.get("NumberOfCells")) == ms.mesh.n_elements
    f3 = skewed_field_3d(machine, 0.0, (0.0, 0.0, 0.0), 2, machine.cogging_period / 2)
    p3 = write_vtu(tmp_path / "m3.vtu", f3["points"], f3["wedges"], {"Bmag": f3["Bmag"]})
    assert int(ET.parse(p3).getroot().find("UnstructuredGrid/Piece").get("NumberOfCells")) == 2 * ms.mesh.n_elements
    assert f3["points"][:, 2].max() == pytest.approx(machine.length)


def test_skew_verification_case():
    from pinneapple_simulation.electromagnetics.machines import verification as mv

    r = mv.run(["Step skew: multi-slice cogging vs skew factor"], quick=True)[0]
    assert r.passed, r.row() + " " + r.detail
