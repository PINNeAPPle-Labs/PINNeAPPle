"""Synthetic wear campaigns: explicit hot spots + growth law + repairs. Always labelled SYNTHETIC.

Use it to demonstrate a twin before real thickness data exists. It is *not* a physical model: the hot
spots encode where the process is known to attack the lining (slag line, jet impact, ...), the amplitudes
are illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .model import SYNTHETIC, VesselSpec, WearDataset


@dataclass(frozen=True)
class HotSpot:
    zone: str
    s: float  # position along the profile, 0..1 (row direction)
    theta_deg: Optional[float]  # None = all around (axisymmetric band)
    ds: float  # width along the profile (fraction)
    dtheta_deg: float = 40.0
    amp: float = 0.5  # extra consumed fraction of usable thickness at the end of the campaign


def _angdiff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def base_pattern(spec: VesselSpec, hotspots: Sequence[HotSpot], base: float) -> Dict[str, np.ndarray]:
    out = {}
    for z in spec.zones:
        N, M = z.shape
        s = ((np.arange(N) + 0.5) / N)[:, None]
        th = (((np.arange(M) + 0.5) / M) * 2 * np.pi)[None, :]
        f = np.full((N, M), base)
        for h in hotspots:
            if h.zone != z.name:
                continue
            g = np.exp(-0.5 * ((s - h.s) / h.ds) ** 2)
            if h.theta_deg is not None:
                g = g * np.exp(-0.5 * (_angdiff(th, np.radians(h.theta_deg)) / np.radians(h.dtheta_deg)) ** 2)
            f = f + h.amp * g
        out[z.name] = f
    return out


def synthetic_campaign(spec: VesselSpec, hotspots: Sequence[HotSpot], *, n_readings: int = 41,
                       x_max: float = 120.0, growth_power: float = 1.0, base: float = 0.2,
                       noise: float = 0.01, seed: int = 0,
                       repairs: Sequence[float] = (), repair_zones: Sequence[str] = (),
                       repair_gain: float = 0.4) -> WearDataset:
    """``noise`` is the sigma of additive measurement noise as a fraction of the usable thickness.
    Wear grows as ``(x/x_max)**growth_power`` towards ``pattern * usable``; at each ``repairs`` x the
    wear of ``repair_zones`` drops by ``repair_gain`` of its value (gunning), then resumes."""
    if n_readings < 2:
        raise ValueError("n_readings must be >= 2")
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, x_max, n_readings)
    pat = base_pattern(spec, hotspots, base)
    g = (x / x_max) ** growth_power
    wear = {z.name: np.zeros((n_readings,) + z.shape) for z in spec.zones}
    cur = {z.name: np.zeros(z.shape) for z in spec.zones}
    done = set()
    for k in range(1, n_readings):
        for z in spec.zones:
            cur[z.name] = cur[z.name] + pat[z.name] * z.usable * (g[k] - g[k - 1])
        for r in repairs:
            if r not in done and x[k - 1] < r <= x[k]:
                done.add(r)
                for zn in repair_zones:
                    cur[zn] = cur[zn] * (1 - repair_gain)
        for z in spec.zones:
            # additive instrument noise (sigma = noise * usable), independent of the wear level
            wear[z.name][k] = np.maximum(cur[z.name] + noise * z.usable * rng.standard_normal(z.shape), 0.0)
    return WearDataset(spec, x, wear, origin=SYNTHETIC,
                       note="SYNTHETIC campaign (hot spots + growth law). Not operational data.")
