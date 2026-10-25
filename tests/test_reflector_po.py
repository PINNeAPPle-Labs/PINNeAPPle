"""Physical optics, compact-range quiet zones and the reflector study: closed forms, feed normalisation, metrics on
known fields, geometry, surrogates and maps."""
import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.reflector import analytic, geometry, po, qz
from pinneapple_simulation.electromagnetics.reflector.compact_range import (
    CompactRange,
    evaluate,
    go_check,
    scaled,
)

F_HZ = 3e9
K = po.wavenumber(F_HZ)
LAM = po.C0 / F_HZ


def test_po_disk_matches_the_closed_form_and_converges():
    z = np.linspace(2 * LAM, 30 * LAM, 40)
    Ea = analytic.disk_on_axis(z, 2 * LAM, K)
    errs = []
    for m in (8, 16):
        d = geometry.flat_disk(2 * LAM, LAM / m)
        _, H = po.plane_wave(d.centres, K, (0, 0, -1), (1, 0, 0))
        E = po.radiate(d.centres, d.areas, po.po_currents(d.normals, d.centres, H), np.c_[0 * z, 0 * z, z], K)
        errs.append(np.max(np.abs(E[:, 0] - Ea)) / np.max(np.abs(Ea)))
        assert np.abs(E[:, 1:]).max() < 1e-10                        # symmetry: no y or z field on the axis
    assert errs[0] < 6e-3 and errs[1] < errs[0] / 2.5              # ~h^2 convergence
    # the same closed form, simplified: E_x = -E0 [e^{-jkz} - (1 + z^2/R_a^2 + j a^2/(k R_a^3)) e^{-jkR_a} / 2]
    for a in (2 * LAM, 50 * LAM):
        Ra = np.sqrt(z ** 2 + a ** 2)
        simple = -(np.exp(-1j * K * z) - 0.5 * (1 + z ** 2 / Ra ** 2 + 1j * a ** 2 / (K * Ra ** 3))
                   * np.exp(-1j * K * Ra))
        assert np.allclose(analytic.disk_on_axis(z, a, K), simple, atol=1e-12)


@pytest.mark.parametrize("gain", [8.0, 12.9, 20.0])
def test_cosq_feed_radiates_one_watt_and_has_its_gain(gain):
    q = po.q_from_gain(gain)
    assert po.gain_from_q(q) == pytest.approx(gain)
    th = np.linspace(0, np.pi, 721)
    ph = np.linspace(-np.pi, np.pi, 361)
    T, P = np.meshgrid(th, ph)
    r = 50.0
    pts = r * np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    E, H = po.cosq_feed(pts, K, position=(0, 0, 0), axis=(0, 0, 1), q=q)
    S = 0.5 * np.real(np.cross(E, np.conj(H)))
    Sr = np.einsum("ij,ij->i", S, pts / r).reshape(T.shape)
    prad = np.trapezoid(np.trapezoid(Sr * np.sin(T), th, axis=1), ph) * r ** 2
    assert prad == pytest.approx(1.0, rel=2e-3)
    U = Sr * r ** 2
    assert 10 * np.log10(4 * np.pi * U.max()) == pytest.approx(gain, abs=0.01)
    # Ludwig-3: co-polar (y at boresight) everywhere in the forward hemisphere, no x component on the axes
    on_axis = np.abs(pts[:, 0]) < 1e-9
    assert np.abs(E[on_axis & (pts[:, 2] > 0), 0]).max() < 1e-12


def test_qz_metrics_on_known_fields():
    xy = qz.disk_grid(1.2, 0.01)
    r2 = (xy ** 2).sum(1)
    E = 10 ** ((-2.0 * r2 / 0.36) / 20)                     # a quadratic taper of 2 dB at r = 0.6
    m = qz.qz_metrics(xy, E, 1.2)
    assert m["taper_dB"] == pytest.approx(2.0, abs=0.01) and m["ripple_pp_dB"] < 1e-9
    assert m["phase_pp_deg"] == pytest.approx(0, abs=1e-9)
    # +-0.5 dB (1 dB peak to peak) is reached at r = 0.6 sqrt(1/2)
    assert m["usable_diameter"] == pytest.approx(2 * 0.6 * np.sqrt(0.5), abs=0.03)
    ripple = E * np.exp(1j * np.deg2rad(8) * np.cos(2 * np.pi * xy[:, 0] / 0.3)) * \
        10 ** (0.3 * np.cos(2 * np.pi * xy[:, 1] / 0.25) / 20)
    m2 = qz.qz_metrics(xy, ripple, 1.2)
    assert m2["ripple_pp_dB"] > 0.4 and m2["phase_pp_deg"] == pytest.approx(16, abs=0.5)


def test_offset_reflector_geometry_and_rolled_edge():
    r = geometry.OffsetReflector(focal=1.5, half_x=0.5, half_y=0.5, centre_y=0.8, n=6, roll=0.15, end_angle=-90)
    m = geometry.mesh_reflector(r, LAM / 6)
    par = ~m.roll
    V = m.centres[par]
    assert np.allclose(V[:, 2], (V[:, 0] ** 2 + V[:, 1] ** 2) / (4 * 1.5), atol=2e-3)       # on the paraboloid
    assert np.all(np.einsum("ij,ij->i", m.normals[par], r.focus - V) > 0)                   # concave side to F
    assert m.vertices[:, 2].min() < V[:, 2].min()                                          # the roll goes back
    sharp = geometry.mesh_reflector(geometry.OffsetReflector(1.5, 0.5, 0.5, 0.8, n=40), LAM / 6)
    proj = sharp.area * np.mean(sharp.normals[:, 2])
    assert proj == pytest.approx(1.0, rel=0.03)                                             # ~1 m x 1 m square
    with pytest.raises(ValueError):
        geometry.OffsetReflector(1.5, -0.5, 0.5, 0.8)


def test_compact_range_go_and_scaling():
    cr = CompactRange(mesh_per_wavelength=6.0, qz_step_per_wavelength=3.0, qz_diameter=0.3, eval_factor=1.2)
    m = evaluate(cr)
    assert m["n_points"] > 50 and m["feed_q"] == pytest.approx(po.q_from_gain(12.0))
    assert -40 < m["xpol_dB"] < -10
    s = scaled(cr, side=1.0)
    assert s.reflector.centre_y - s.reflector.half_y == pytest.approx(cr.reflector.centre_y - cr.reflector.half_y)
    g = go_check(scaled(cr, roll=6 * LAM))
    assert g["rms_dB"] < 0.6                     # with a large rolled edge the smooth PO amplitude follows GO


def test_surrogates_and_tradeoff_map(tmp_path):
    from pinneapple_simulation.electromagnetics.antenna.runner import ResultTable
    from pinneapple_simulation.electromagnetics.reflector.study import fit_surrogates, tradeoff_map
    rows = []
    for g in (8, 11, 14, 17, 20):
        for s in (0.8, 1.0, 1.2, 1.4, 1.6):
            rows.append({"tag": f"g{g}s{s}", "feed_gain_dBi": g, "side": s, "roll": 0.1, "f_GHz": 3.0,
                         "ripple_pp_dB": 3 - 0.1 * g + 0.5 * np.sin(4 * s), "amp_pp_dB": 2 - 0.05 * g,
                         "phase_pp_deg": 20 - 0.5 * g, "taper_dB": 1.0, "usable_diameter": 0.1 + 0.02 * g * s})
    t = ResultTable(rows)
    ms = fit_surrogates(t)[0.1]
    assert ms["ripple_pp_dB"][1] < 0.1 and np.isfinite(ms["ripple_pp_dB"][3])
    p = tradeoff_map(t, ms, str(tmp_path / "map.png"), roll=0.1, qz_diameter=0.5, mark=(12, 1.2, "ref"))
    assert (tmp_path / "map.png").stat().st_size > 20_000 and p.endswith("map.png")
