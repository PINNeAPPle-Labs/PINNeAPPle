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
    n = 2000                                         # sampling B: 0.25 % from the exact loop integral
    phi = (np.arange(n) + 0.5) / n * tau - tau / 2
    z = (np.arange(20) + 0.5) / 20 * machine.length - machine.length / 2
    P, Z = np.meshgrid(phi, z)
    pts = np.column_stack([r * np.cos(P.ravel()), r * np.sin(P.ravel()), Z.ravel()])
    B = s.B[m3.mesh.locate(pts)]
    Br = B[:, 0] * np.cos(P.ravel()) + B[:, 1] * np.sin(P.ravel())
    flux = Br.sum() * (r * tau / n) * (machine.length / 20)
    assert lam == pytest.approx(flux, rel=0.005)
    row = end_effect_study(machine, (0.5,), end_length=0.02, end_layers=2, n_theta=72)["rows"][0]
    assert 0.98 < row["k_o0mm"] < 1.0                # end leakage: a fraction of a percent


def test_rectangle_loss_series_limits():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import (
        loss_factor,
        plate_loss,
        strip_loss,
    )

    assert loss_factor(1.0, 1000.0) == pytest.approx(1.0, abs=2e-3)          # long strip: the 2-D value
    assert plate_loss(0.02, 0.01, 0.004, 6.7e5, 1e3, 0.05) == pytest.approx(
        plate_loss(0.01, 0.02, 0.004, 6.7e5, 1e3, 0.05), rel=1e-12)       # symmetric in its sides
    assert plate_loss(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) < strip_loss(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_magnet_block_loss_matches_the_closed_form():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import segmentation_study

    r = segmentation_study(0.02, 0.02, 0.004, 6.7e5, 0.01, [(1, 1), (1, 2)], [100.0], h=2e-3)
    for row in r["rows"]:                       # 1.4 % and 4.0 % at 2 mm, 0.4 % and 1.0 % at 1 mm (measured)
        assert row["ratio_3d_closed_form"] == pytest.approx(1.0, abs=0.05)
    two = r["rows"][1]
    assert two["ratio_3d_2d"] < 0.2                # two axial pieces keep 17 % of the 2-D loss


@pytest.mark.parametrize("topology", ["single", "double_rotor"])
def test_afpm_mesh_conforms_and_aligned_torque_vanishes(topology):
    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, AFPMMachine

    mod = AFPM3D(AFPMMachine(topology=topology), n_theta=192, dr=0.02, dz=0.008, air_r=0.01, air_z=0.008)
    assert mod.mesh._face_topology[2].max() == 2                       # the seam is closed and conforming
    th = np.array([-2, 0, 2]) * mod.step
    c = mod.cogging(th)
    assert abs(c[1]) < 1e-8 * abs(c[2])                                  # no mesh handedness
    assert c[0] == pytest.approx(-c[2], rel=1e-8)                        # odd about the aligned rotor
    W = [mod.solve(x).coenergy for x in (0.0, 4 * mod.step)]
    assert np.sign(W[1] - W[0]) == np.sign(c[2])                         # torque = + d(co-energy)/d(theta)


def test_afpm_quasi3d_slices_tile_the_annulus():
    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, AFPMMachine

    mod = AFPM3D(AFPMMachine(), n_theta=192, dr=0.01, dz=0.008, air_r=0.01, air_z=0.008)
    q = mod.quasi3d_models(2)
    active = mod.rs[(mod.rs >= 0.05 - 1e-12) & (mod.rs <= 0.09 + 1e-12)]
    assert np.allclose(np.concatenate([q[0].rs, q[1].rs[1:]]), active)    # same radial nodes as the 3-D grid
    s3 = mod.solve(0.0)
    lq = sum(x.tooth_linkage(x.solve(0.0)) for x in q)
    l3 = mod.tooth_linkage(s3)
    assert lq == pytest.approx(l3, rel=0.1) and np.sign(lq) == np.sign(l3)


# known answers with literal inputs (the Studio smoke-runs these tools with the same inputs)
def test_square_piece_keeps_the_single_series_share_of_the_strip_loss():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import loss_factor

    n = np.arange(1, 400, 2)
    single = 1 - 192 / np.pi ** 5 * np.sum(np.tanh(n * np.pi / 2) / n ** 5)    # the one-sum form, a = b
    assert loss_factor(1.0, 1.0) == pytest.approx(single, rel=1e-6)          # 0.42173


def test_strip_loss_is_the_textbook_formula():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import strip_loss

    w = 2 * np.pi * 1000.0
    expected = 0.004 * 6.7e5 * w ** 2 * 0.05 ** 2 * 0.02 ** 3 * 0.05 / 24      # h sigma w^2 B^2 a^3 b / 24
    assert strip_loss(0.02, 0.05, 0.004, 6.7e5, 1000.0, 0.05) == pytest.approx(expected, rel=1e-12)   # 4.408 W


def test_plate_loss_of_a_square_is_its_share_of_the_strip():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import plate_loss, strip_loss

    p = plate_loss(0.02, 0.02, 0.004, 6.7e5, 1000.0, 0.05)
    assert p == pytest.approx(0.4217310 * strip_loss(0.02, 0.02, 0.004, 6.7e5, 1000.0, 0.05), rel=1e-6)


def test_segmented_block_magnet_volume():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import segmented_block

    m = segmented_block(0.02, 0.02, 0.004, 1, 2, 0.0002, 0.002)
    assert m.volumes[m.mask("magnet")].sum() == pytest.approx(0.02 * (0.02 - 0.0002) * 0.004, rel=1e-9)


def test_segmentation_study_coarse_block():
    from pinneapple_simulation.electromagnetics.machines3d.magnet_loss import segmentation_study

    r = segmentation_study(0.02, 0.02, 0.004, 6.7e5, 0.01, [(1, 1)], [100.0], h=0.004)
    assert r["rows"][0]["ratio_3d_closed_form"] == pytest.approx(1.0, abs=0.08)   # 1.051 at 4 mm (measured)


def test_afpm_coarse_grid_builds():
    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D

    model = AFPM3D(n_theta=192, dr=0.02, dz=0.008, air_r=0.01, air_z=0.008)
    assert len(model.mesh.tets) == 50688 and model.mesh._face_topology[2].max() == 2


def test_designer_grid_steps_known_answers():
    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPMMachine
    from pinneapple_simulation.electromagnetics.machines3d.afpm_design import grid_for

    assert grid_for(AFPMMachine(slots=12, poles=8))["n_theta"] == 192      # 8 even steps per 15 deg period
    assert grid_for(AFPMMachine(slots=12, poles=10))["n_theta"] == 480     # period 6 deg
    with pytest.raises(ValueError, match="576 steps"):
        grid_for(AFPMMachine(slots=9, poles=8))                             # period 5 deg


def test_quasi3d_slices_share_the_radial_nodes_of_the_annulus():
    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPMMachine, quasi3d_slices

    q = quasi3d_slices(AFPMMachine(poles=8, slots=12), 2, n_theta=192, dr=0.02, dz=0.008, air_z=0.008)
    assert [len(x.rs) for x in q] == [2, 2] and q[0].rs[0] == 0.05 and q[1].rs[-1] == 0.09
    assert q[0].rs[-1] == q[1].rs[0] == pytest.approx(0.07)


def test_fraction_edges_give_the_magnet_arc_and_slot_opening_asked_for():
    import numpy as np

    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, IRON, AFPMMachine

    kw = {"n_theta": 192, "dr": 0.02, "dz": 0.008, "air_r": 0.01, "air_z": 0.008}
    for arc, sf in ((0.78, 0.43), (0.8, 0.5), (0.62, 0.68)):      # between grid steps, and on them
        m = AFPMMachine(magnet_arc=arc, slot_fraction=sf)
        cell, frac = AFPM3D(m, **kw), AFPM3D(m, edges="fraction", **kw)
        width = np.diff(frac.th)
        pol = frac._magnet_base[0]
        assert (np.abs(pol) * width).sum() == pytest.approx(arc * 2 * np.pi)          # the arc asked for
        assert np.abs(cell._magnet_base[0]).sum() == 2 * round(arc * 12) * 8          # whole cells, two at a time
        assert (pol * cell._magnet_base[0] >= 0).all()                                # same polarity per pole
        i, j, k = frac.cell
        t = np.flatnonzero((frac.zkind[k] == "teeth") & (frac._static == IRON))[0]
        ring = (i == i[t]) & (k == k[t]) & (frac._static == IRON)                    # one ring of tooth iron
        assert width[np.unique(j[ring])].sum() == pytest.approx((1 - sf) * 2 * np.pi)  # the slot opening asked for
    m = AFPMMachine(magnet_arc=5 / 6, slot_fraction=0.5)            # edges on the uniform grid: the same model
    a, b = AFPM3D(m, **kw), AFPM3D(m, edges="fraction", **kw)
    assert np.allclose(a.th, b.th) and np.array_equal(a._magnet_base[0], b._magnet_base[0])
    assert np.array_equal(a._static, b._static)
    with pytest.raises(ValueError, match="edges"):
        AFPM3D(m, edges="nearest", **kw)


def test_designer_outputs_move_smoothly_with_the_magnet_arc_and_slot_opening():
    from pinneapple_simulation.electromagnetics.machines3d.afpm_design import (
        estimate,
        grid_for,
        machine_from,
    )

    g = {**grid_for(machine_from({})), "dr": 0.02, "dz": 0.008}          # the designer's edges, a cheaper grid
    # magnet arc: whole-cell edges (even-step angles) jumped here, cogging 0.693, 10.779, 10.779 N m on this grid;
    # fractional edges with 16 angles per period: 7.078, 7.628, 8.080
    # slot opening: a tooth edge as a share of iron jumped at 0.5 (tooth linkage -5 % between 0.495 and 0.5 on the
    # designer's grid); with the tooth edges on node planes it is linear
    for key, xs in (("magnet_arc", (0.79, 0.795, 0.80)), ("slot_fraction", (0.495, 0.5, 0.505))):
        r = [estimate({key: x}, grid=g) for x in xs]
        for out in ("cogging_pkpk_Nm", "tooth_linkage_Wb"):
            c = [abs(x[out]) for x in r]
            assert (c[0] < c[1] < c[2]) or (c[0] > c[1] > c[2]), (key, out, c)
            assert abs(c[1] - 0.5 * (c[0] + c[2])) < 0.05 * abs(c[2] - c[0]) + 1e-3 * c[1], (key, out, c)


def test_fraction_edges_turn_the_rotor_by_any_angle():
    import numpy as np

    from pinneapple_simulation.electromagnetics.machines3d.afpm import AFPM3D, AFPMMachine

    model = AFPM3D(AFPMMachine(magnet_arc=0.78), n_theta=192, dr=0.02, dz=0.008, air_r=0.01, air_z=0.008,
                   edges="fraction")
    base = model._polarity(0.0)
    assert np.allclose(model._polarity(4 * model.step), np.roll(base, 4))    # whole steps: the pattern moves
    for frac in (0.25, 0.5, 1.3):                                             # inside cells: the magnet is kept
        p = model._polarity(frac * model.step)
        assert np.abs(p).sum() == pytest.approx(np.abs(base).sum())
        assert (p * np.roll(base, round(frac)) >= -1e-12).all()             # no cell changes polarity
    _, Br = model.materials(0.5 * model.step)
    _, Br0 = model.materials(0.0)
    assert not np.allclose(Br, Br0)                                           # half a step is not rounded away
