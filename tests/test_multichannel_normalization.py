"""Lesson from a crash-surrogate project (Ben Chang, 2026, OpenRadioss + GeoTransolver): training
displacement, stress and strain jointly *without per-channel normalisation* degrades the small-scale
channel (displacement) the longer one trains. StandardScaler must give each channel its own scale,
including for mesh fields shaped (samples, nodes, channels)."""
import torch

from pinneapple_data.transforms import StandardScaler


def _fields(n=64, nodes=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    disp_m = 1e-3 * torch.randn(n, nodes, 1, generator=g)  # metres
    stress_pa = 2e8 + 5e7 * torch.randn(n, nodes, 1, generator=g)  # pascals
    strain = 0.02 * torch.rand(n, nodes, 1, generator=g)
    return torch.cat([disp_m, stress_pa, strain], dim=2)


def test_per_channel_scaling_of_mesh_fields():
    y = _fields()
    sc = StandardScaler().fit(y, dim=(0, 1))  # reduce over samples and nodes, keep channels
    z = sc.transform(y)
    assert sc.std_.shape == (1, 1, 3)
    torch.testing.assert_close(z.mean(dim=(0, 1)), torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(z.std(dim=(0, 1)), torch.ones(3), atol=1e-3, rtol=0)
    torch.testing.assert_close(sc.inverse(z), y, rtol=1e-5, atol=1e-8)


def test_a_single_global_scale_would_bury_the_displacement_channel():
    y = _fields()
    global_std = y.std()
    # with one scale for all channels, displacement becomes ~1e-11: invisible to an MSE loss
    assert (y[..., 0] / global_std).abs().max() < 1e-9
    per_channel = StandardScaler().fit(y, dim=(0, 1)).transform(y)
    assert per_channel[..., 0].abs().max() > 1.0
