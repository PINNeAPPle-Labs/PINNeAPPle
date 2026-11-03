"""Computational lithography (#1180): imaging against closed forms, Abbe = Hopkins, metrics, ILT re-verified."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_physics.lithography.imaging import (
    Optics,
    Source,
    aerial_image,
    aerial_image_socs,
    resolution_limit,
    socs_kernels,
)
from pinneapple_physics.lithography.metrics import blur, cd_along, edge_length, epe, printed


def _grating(N, p, duty=0.5):
    x = np.arange(N) * p / N
    m = (np.mod(x, p) < duty * p).astype(float)
    return m, np.tile(m[:, None], (1, N))


def test_a_coherent_grating_keeps_exactly_its_passing_orders():
    N, p, o = 128, 300.0, Optics(193.0, 0.93)                 # lambda/NA = 207.5 < p < 2 lambda/NA: orders 0, +-1
    m, mask = _grating(N, p)
    img = aerial_image(mask, p, o, Source("coherent")).numpy()[:, 0]
    c = np.fft.fft(m) / N
    th = 2 * np.pi * np.arange(N) / N
    ref = np.abs(c[0] + c[1] * np.exp(1j * th) + c[-1] * np.exp(-1j * th)) ** 2
    assert np.abs(img - ref).max() < 1e-12


def test_clear_mask_is_one_and_no_modulation_below_the_resolution_limit():
    o = Optics(193.0, 0.93)
    clear = aerial_image(np.ones((64, 64)), 500.0, o, Source("annular", 0.9, 0.6)).numpy()
    assert np.abs(clear - 1).max() < 1e-12
    pmin = resolution_limit(o, 0.5)
    for f, modulated in ((0.98, False), (1.05, True)):
        _, mask = _grating(128, f * pmin)
        img = aerial_image(mask, f * pmin, o, Source("conventional", 0.5, n=15)).numpy()
        assert bool(np.ptp(img) > 1e-3) == modulated


def _clip(N=64, seed=0):
    rng = np.random.default_rng(seed)
    m = np.ones((N, N))
    for _ in range(6):
        i, j = rng.integers(0, N - 12, 2)
        a, b = rng.integers(3, 12, 2)
        m[i:i + a, j:j + b] = 0
    return m


def test_hopkins_socs_equals_abbe_and_defocus_is_symmetric():
    o = Optics(193.0, 1.35, defocus=40.0, medium_index=1.44)
    src = Source("annular", 0.9, 0.6, n=5)
    m = _clip()
    Ia = aerial_image(m, 1024.0, o, src)
    S = socs_kernels(64, 1024.0, o, src)
    assert float((Ia - aerial_image_socs(m, S)).abs().max()) < 1e-12
    S12 = socs_kernels(64, 1024.0, o, src, max_kernels=12)
    assert S12["kept_energy"] < 1 and float((Ia - aerial_image_socs(m, S12)).abs().max()) < 0.05
    minus = aerial_image(m, 1024.0, Optics(193.0, 1.35, -40.0, 1.44), src)
    assert float((Ia - minus).abs().max()) < 1e-12
    assert float((Ia - aerial_image(m, 1024.0, Optics(193.0, 1.35), src)).abs().max()) > 1e-3


def test_metrics_cd_epe_and_blur():
    N, L = 128, 1024.0
    img = torch.ones((N, N), dtype=torch.float64)
    img[40:56, :] = 0.0                                            # a 128 nm dark line
    assert cd_along(img.numpy(), L, 0.5, row=10) == pytest.approx(128.0, abs=8.0)
    t = np.zeros((N, N), bool)
    t[40:56, 20:100] = True
    assert edge_length(t, L) == pytest.approx(2 * (16 + 80) * 8.0)
    shifted = np.roll(t, 1, axis=0)                              # one-pixel (8 nm) shift along x
    assert epe(shifted, t, L) == pytest.approx(2 * 80 * 64 / (2 * (16 + 80) * 8.0))
    assert float(blur(img, L, 0.0).sum()) == float(img.sum())
    assert float(blur(img, L, 10.0).sum()) == pytest.approx(float(img.sum()))     # diffusion keeps the dose
    p = printed(img, L, threshold=0.5)
    assert p[45, 10] and not p[10, 10]


def test_ilt_beats_the_uncorrected_mask_on_the_exact_model():
    from pinneapple_physics.lithography.design import LithoSpec, clip_lines_and_elbow, correct_mask
    spec = LithoSpec(clip_lines_and_elbow(128, 1024.0), 1024.0, Optics(193.0, 1.35, medium_index=1.44),
                     Source("annular", 0.9, 0.6, n=4), focus=(0.0,), doses=(1.0,), epe_tol=2.0, min_feature_nm=0.0)
    r = correct_mask(spec, steps=150, lr=0.6, kernel_energy=0.97)
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["litho.improves"] == st["litho.epe_nominal"] == st["model.kernels_vs_abbe"] == "pass"
    assert r["epe_nm"]["corrected"] < 0.1 * r["epe_nm"]["uncorrected"]      # measured: 10.3 nm -> 0.22 nm
    assert r["recommendation"]["recommended"][0] == "none"       # one clip: the exact model is cheap enough
    assert r["surrogate"] is None


def test_the_learned_operator_is_checked_before_use():
    from pinneapple_physics.lithography.surrogate import operator_imager, train_operator
    o, src = Optics(193.0, 1.35), Source("annular", 0.9, 0.6, n=4)
    S = socs_kernels(32, 512.0, o, src, energy=0.99)
    r = train_operator(S, 512.0, 32, n_train=128, n_test=16, epochs=80, modes=12, width=20,
                       socs_fine=socs_kernels(64, 512.0, o, src, energy=0.99))
    assert r["held_out_rel_l2"] < 0.05 and r["fine_rel_l2"] < 0.1
    img = operator_imager(r["model"])(torch.ones((32, 32), dtype=torch.float64), 0.0)
    assert img.shape == (32, 32)
    with pytest.raises(ValueError, match="best focus"):
        operator_imager(r["model"])(torch.ones((32, 32)), 50.0)


def test_flow_node_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "litho.opc" in REGISTRY.types()
    assert match("OPC de uma máscara de litografia com janela de processo")[0][1].id == "litho.opc"
