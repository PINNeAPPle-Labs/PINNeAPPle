"""Synthetic chamber data with a known growth law, the way Lamb et al. (2026) validate their method.

The paper builds 290 synthetic mass-ratio time series: the observed ``S_i``, ``T`` and ``m0`` of every experiment,
the Nelson & Baker (1996) deposition coefficient of Eq. 10 as the *known* kinetics, Eq. 7 integrated as the truth,
and measurement noise taken from the trailing eigenmodes of a singular spectrum analysis (SSA) of the measured
series (its Supplementary Figure S3). The strongly-constrained neural ODE is then trained on the synthetic series
and its learned ``alpha`` is compared with the one that generated them (its Figure 2), which is what makes the
real-data result believable: the method recovers a known function before it is trusted on an unknown one.

    syn = make(real, seed=0)
    syn.mass_ratio        # noisy synthetic m/m0, same shape as the measured series
    syn.clean             # noise-free truth
    syn.alpha             # the alpha(S_i, T) that generated it
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinneapple_physics.cloud import DepositionalIceGrowth, alpha_nelson_baker, solve_mass_ratio

__all__ = ["SyntheticData", "ssa_noise", "make"]


@dataclass
class SyntheticData:
    mass_ratio: np.ndarray
    clean: np.ndarray
    time: np.ndarray
    Si: np.ndarray
    T: np.ndarray
    P: np.ndarray
    r0: np.ndarray
    m0: np.ndarray
    alpha: np.ndarray
    noise: np.ndarray

    def __len__(self):
        return int(self.Si.size)


def ssa_noise(series: np.ndarray, window: int = 60, keep: float = 0.5) -> np.ndarray:
    """The trailing-eigenmode residual of a singular spectrum analysis, i.e. its real measurement noise.

    Builds the lagged trajectory matrix of a 1 Hz series, decomposes it, reconstructs the leading ``keep`` fraction
    of the modes (the signal) and returns ``series - signal``, the noise realisation. The paper takes the trailing
    eigenmodes as the measurement uncertainty on ``m/m0``; this returns exactly that residual.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    window = min(window, n // 3)
    if window < 4:                                                  # too short to decompose
        return np.zeros_like(x)
    k = n - window + 1
    traj = np.lib.stride_tricks.sliding_window_view(x, window)
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    n_keep = max(1, int(round(keep * s.size)))
    recon = (u[:, :n_keep] * s[:n_keep]) @ vt[:n_keep]
    rec = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(k):
        rec[i:i + window] += recon[i]
        cnt[i:i + window] += 1
    signal = rec / np.maximum(cnt, 1)
    return x - signal


def make(real, *, seed: int = 0, noisy: bool = True, keep: float = 0.5) -> SyntheticData:
    """Build the synthetic data set from the measured conditions of ``real`` (a ``bel.ChamberData``)."""
    rng = np.random.default_rng(seed)
    n = len(real)
    n_t = real.mass_ratio.shape[1]
    clean = np.full((n, n_t), np.nan)
    noisy_out = np.full((n, n_t), np.nan)
    alpha = alpha_nelson_baker(real.Si, real.T)
    for i in range(n):
        model = DepositionalIceGrowth(T=real.T[i], p=real.P[i], Si=real.Si[i], kinetics="nelson_baker")
        length = int(real.length()[i])
        t = np.arange(length, dtype=float)
        ratio = solve_mass_ratio(model, t, real.m0[i], method="LSODA")
        clean[i, :length] = ratio
        if noisy:
            # the paper's noise is the measurement residual at the same amplitude as the measured series; the
            # scale comes from the real series' own SSA residual and its shape from the synthetic one
            sigma = float(np.std(ssa_noise(real.mass_ratio[i], keep=keep))) if length > 12 else 0.0
            if sigma > 0:
                shape = ssa_noise(ratio, keep=keep)
                shape = shape / (np.std(shape) or 1.0)
                noisy_out[i, :length] = ratio + sigma * shape
            else:
                noisy_out[i, :length] = ratio
        else:
            noisy_out[i, :length] = ratio
    del rng
    return SyntheticData(mass_ratio=noisy_out, clean=clean, time=np.where(np.isfinite(real.time), real.time, np.nan),
                         Si=real.Si, T=real.T, P=real.P, r0=real.r0, m0=real.m0, alpha=alpha,
                         noise=noisy_out - clean)
