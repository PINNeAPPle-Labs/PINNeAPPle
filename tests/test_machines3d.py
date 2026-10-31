"""3-D surface-PM machine on edge elements (``pinneapple_simulation.electromagnetics.machines3d``): conforming
extrusion, symmetry of the no-load torque, axial invariance of a straight stack, the multi-slice composition and
agreement with the 2-D code. Small models (72 angular steps), each test a few seconds."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.machines3d import SPM3D, extrude, step_skew
from pinneapple_simulation.electromagnetics.machines3d.skew import slice_offsets
from pinneapple_simulation.electromagnetics.materials import LinearMaterial


@pytest.fixture(scope="module")
def machine():
    return SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=72)


@pytest.fixture(scope="module")
def one_layer(machine):
    return SPM3D(machine, n_theta=72, stack_layers=1)


def test_extrusion_is_conforming():
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(0)
    pts = rng.random((40, 2))
    d = Delaunay(pts)
    for rank in (None, rng.permutation(len(pts))):
        m, tri, lay = extrude(pts, d.simplices, np.linspace(0, 1, 4), rank=rank)
        cnt = m._face_topology[2]
        assert cnt.max() == 2
        assert (cnt == 1).sum() == 2 * len(d.simplices) + 6 * len(d.convex_hull)
        assert m.volumes.sum() == pytest.approx(_area(pts, d.simplices), rel=1e-12)   # height 1


def _area(p, t):
    a = p[t]
    return 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                        - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])).sum()


def test_cogging_is_odd_and_matches_the_2d_code(machine, one_layer):
    step = one_layer.step
    th = np.arange(-3, 4) * step
    c3 = one_layer.cogging(th)
    peak = np.abs(c3).max()
    assert abs(c3[3]) < 1e-6 * peak                          # aligned rotor: no torque (no mesh handedness)
    assert np.allclose(c3[:3], -c3[4:][::-1], atol=1e-6 * peak)
    c2 = np.array([machine.torque(machine.solve(x)) for x in th])
    assert np.max(np.abs(c3 - c2)) < 0.05 * np.abs(c2).max()


def test_straight_stack_is_axially_invariant(machine, one_layer):
    # the layered stack has in-plane unknowns the one-layer model lacks: 3 % apart at 72 steps (measured), 0.3 %
    # at the 360 steps of the skew study
    m3 = SPM3D(machine, n_theta=72, stack_layers=3)
    th = one_layer.step
    assert m3.torque(m3.solve(th)) == pytest.approx(one_layer.torque(one_layer.solve(th)), rel=0.05)
    assert m3.torque(m3.solve(0.0)) == pytest.approx(0.0, abs=1e-8)


def test_multislice_is_the_mean_of_the_slices(machine, one_layer):
    m3 = SPM3D(machine, n_theta=72, stack_layers=2)
    total = 2 * m3.step
    off = slice_offsets(m3, "step2", total)
    assert np.allclose(off, [-m3.step, m3.step])
    th = [one_layer.step]
    ms = m3.multislice(th, off, slice_model=one_layer)
    ref = 0.5 * sum(one_layer.torque(one_layer.solve(th[0] + d)) for d in off)
    assert ms[0] == pytest.approx(ref, rel=1e-6)                 # CG tolerance 1e-10 on each solve
    sk = m3.cogging(th, step_skew(total, 2, machine.length))   # the 3-D model with the same segments
    assert np.isfinite(sk[0]) and abs(sk[0]) < 2 * abs(ref) + 1e-3


def test_cogging_period_needs_an_even_number_of_steps(machine, one_layer):
    from pinneapple_simulation.electromagnetics.machines3d.skew import skew_study

    with pytest.raises(ValueError, match="even number of steps"):      # 72 steps: a slot pitch of 3 steps
        skew_study(one_layer, [0.0], kinds=("step2",))
    m = SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=144)       # 6 steps per slot pitch
    s = SPM3D(m, n_theta=144, stack_layers=1)
    a, b = s.cogging([s.step, 7 * s.step])
    assert a == pytest.approx(b, rel=1e-6)                               # the discrete machine repeats every period


def test_skew_study_resumes_from_its_checkpoint(tmp_path):
    from pinneapple_simulation.electromagnetics.machines3d.skew import skew_study

    m = SPMMachine(iron=LinearMaterial(mu_r=1000.0), n_theta=144)
    s = SPM3D(m, n_theta=144, stack_layers=2)
    ck = tmp_path / "rows.jsonl"
    th = [s.step, 2 * s.step]
    a = skew_study(s, [0.0, 2 * s.step], kinds=("step2",), thetas=th, checkpoint=ck)
    assert len(ck.read_text().splitlines()) == 2
    msgs = []
    b = skew_study(s, [0.0, 2 * s.step], kinds=("step2",), thetas=th, checkpoint=ck, progress=msgs.append)
    assert all("checkpoint" in x for x in msgs) and len(msgs) == 2
    assert [r["pkpk_3d"] for r in a["rows"]] == [r["pkpk_3d"] for r in b["rows"]]


def test_search_coil_links_the_2d_flux_and_end_air_lowers_it(machine):
    from pinneapple_simulation.electromagnetics.machines3d.endeffects import (
        end_effect_study,
        search_coil_linkage,
    )

    m3 = SPM3D(machine, n_theta=72, stack_layers=2)
    s = m3.solve(0.0)
    lam = search_coil_linkage(m3, s)
    # Stokes: the loop integral is the flux of B through the coil's cylindrical surface (midpoint rule in phi, z)
    r = 0.5 * (2 * machine.r_bore + machine.slot_depth)
    tau = np.pi / machine.pole_pairs
    n = 2000                                         # 0.25 % from the loop integral (2.4 % at 200)
    phi = (np.arange(n) + 0.5) / n * tau - tau / 2
    z = (np.arange(20) + 0.5) / 20 * machine.length - machine.length / 2
    P, Z = np.meshgrid(phi, z)
    pts = np.column_stack([r * np.cos(P.ravel()), r * np.sin(P.ravel()), Z.ravel()])
    B = s.B[m3.mesh.locate(pts)]
    Br = B[:, 0] * np.cos(P.ravel()) + B[:, 1] * np.sin(P.ravel())
    flux = Br.sum() * (r * tau / n) * (machine.length / 20)
    assert lam == pytest.approx(flux, rel=0.01)
    row = end_effect_study(machine, (0.5,), end_length=0.02, end_layers=2, n_theta=72)["rows"][0]
    assert 0.98 < row["k_o0mm"] < 1.0                # 0.9945 measured: end leakage, a fraction of a percent
