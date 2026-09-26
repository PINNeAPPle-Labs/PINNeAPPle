"""CalculiX bridge: .inp / .frd round trips, virtual ccx (drop-in surrogate) and a real ccx run."""
import json
import os

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.calculix import (
    OutOfEnvelope, cantilever_hex_mesh, ccx_backend, predict_to_frd, read_frd, read_inp, run_ccx, write_frd,
    write_inp,
)
from pinneapple_simulation.external_solvers.calculix.virtual_ccx import main as virtual_ccx_main

E, NU, L, B, H, P = 210e9, 0.3, 1.0, 0.1, 0.1, 1000.0
I_ = B * H ** 3 / 12
G = E / (2 * (1 + NU))


def timoshenko_tip():
    return P * L ** 3 / (3 * E * I_) + P * L / (5 / 6 * G * B * H)


def _beam(tmp_path, element_type="C3D20R"):
    m = cantilever_hex_mesh(L, B, H, nx=20, ny=2, nz=2, element_type=element_type)
    tip = m.nsets["TIP"]
    m.boundary = [("FIXED", 1, 3, 0.0)]
    m.cloads = [(str(n), 3, -P / len(tip)) for n in tip]
    path = write_inp(m, str(tmp_path / "beam.inp"))
    return m, path


class BeamTheory:
    """Analytic 'surrogate': Timoshenko deflection w(x) = P x^2 (3L - x) / (6EI) + P x / (k G A)."""
    __name__ = "BeamTheory"
    envelope = {"bbox": [[0, 0, 0], [L, B, H]], "E": [200e9, 220e9], "load": [0, 2000]}

    def __call__(self, model, coords):
        x = coords[:, 0]
        w = P * x ** 2 * (3 * L - x) / (6 * E * I_) + P * x / (5 / 6 * G * B * H)
        return {"DISP": np.stack([0 * x, 0 * x, -w], 1)}


def test_inp_round_trip_and_mesh_sets(tmp_path):
    m, path = _beam(tmp_path)
    r = read_inp(path)
    assert (len(r.nodes), len(r.elements), r.element_type) == (len(m.nodes), len(m.elements), "C3D20R")
    assert all(len(c) == 20 for c in r.elements.values())
    assert r.E == E and r.nu == NU and len(r.cloads) == len(m.nsets["TIP"])
    assert sorted(r.nsets["FIXED"]) == sorted(m.nsets["FIXED"])


def test_frd_write_read_round_trip(tmp_path):
    ids = np.array([1, 2, 5])
    xyz = np.array([[0, 0, 0], [1.5, -2.0, 3e-4], [7, 8, 9]], float)
    disp = np.array([[1e-3, -2e-3, 3e-3], [0, 0, 0], [-1.25e-5, 4e5, 1]], float)
    stress = np.arange(18, dtype=float).reshape(3, 6) * 1e6
    p = write_frd(str(tmp_path / "x.frd"), ids, xyz, {1: (1, 2, 5, 1, 2, 5, 1, 2)}, "C3D8",
                  {"DISP": disp, "STRESS": stress})
    r = read_frd(p)
    assert r.node_ids.tolist() == [1, 2, 5]
    np.testing.assert_allclose(r.coords, xyz, rtol=1e-5)
    np.testing.assert_allclose(r.field("DISP"), disp, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(r.field("STRESS"), stress, rtol=1e-5)


def test_virtual_ccx_writes_a_readable_frd_and_guards_its_envelope(tmp_path):
    m, path = _beam(tmp_path)
    r = read_frd(predict_to_frd(path, BeamTheory()))
    tip = np.isin(r.node_ids, m.nsets["TIP"])
    assert r.field("DISP")[tip, 2].mean() == pytest.approx(-timoshenko_tip(), rel=1e-5)
    m.E = 70e9  # aluminium: outside the model's declared envelope
    write_inp(m, path)
    with pytest.raises(OutOfEnvelope, match="E=7e"):
        predict_to_frd(path, BeamTheory())


def test_virtual_ccx_cli_refuses_without_predictor_and_logs(tmp_path, monkeypatch):
    _, path = _beam(tmp_path)
    monkeypatch.delenv("PINNEAPPLE_CCX_PREDICTOR", raising=False)
    monkeypatch.delenv("PINNEAPPLE_CCX_FALLBACK", raising=False)
    assert virtual_ccx_main(["-i", str(tmp_path / "beam")]) == 2
    monkeypatch.setitem(__import__("sys").modules, "beam_theory_mod", type("M", (), {"predictor": BeamTheory()}))
    assert virtual_ccx_main(["-i", str(tmp_path / "beam"), "--predictor", "beam_theory_mod:predictor"]) == 0
    log = json.load(open(tmp_path / "beam.pinneapple.json"))
    assert log["mode"] == "surrogate" and os.path.exists(tmp_path / "beam.frd")


@pytest.mark.skipif(ccx_backend() is None, reason="CalculiX not available (no ccx, no Docker)")
def test_real_ccx_cantilever_matches_timoshenko_and_the_virtual_ccx(tmp_path):
    m, path = _beam(tmp_path)
    run = run_ccx(str(tmp_path), "beam")
    assert run.returncode == 0, run.stdout[-500:]
    real = read_frd(run.frd)
    tip = np.isin(real.node_ids, m.nsets["TIP"])
    uz_tip = real.field("DISP")[tip, 2].mean()
    assert uz_tip == pytest.approx(-timoshenko_tip(), rel=0.02)  # measured: -0.94 %
    virt = read_frd(predict_to_frd(path, BeamTheory(), str(tmp_path / "virtual.frd")))
    assert virt.node_ids.tolist() == real.node_ids.tolist()  # same mesh, same order: post-processors see no difference
    uz_r, uz_v = real.field("DISP")[:, 2], virt.field("DISP")[:, 2]
    assert np.abs(uz_r - uz_v).max() < 0.03 * abs(uz_tip)
