"""Electromechanical co-energy ``W'(i, theta)`` as a potential: torque, flux linkages and inductances from one map.

At constant currents the torque on the rotor is ``T = dW'/dtheta`` and the flux linkage of phase ``k`` is
``psi_k = dW'/di_k``; the differential inductances are ``L_jk = d2W'/di_j di_k`` (symmetric by construction) and
the reciprocity ``dpsi_k/dtheta = dT/di_k`` is the Maxwell relation of the co-energy.

A finite-element solver gives the co-energy at chosen points. :func:`local_polynomial` fits a polynomial potential to
such samples (least squares in scaled variables); :func:`machine_coenergy` samples a machine of
``pinneapple_simulation.electromagnetics`` (any object with ``coenergy(theta, currents)``, e.g. ``SPMMachine`` or
``IPMMachine``) around an operating point -- ``2 n + 1`` rotor angles at the operating currents, nine current
perturbations at the operating angle -- and returns the local co-energy potential. With ``n = 2`` and angles on the
moving-band grid the torque is the same fourth-order virtual-work estimate as ``machine.torque_coenergy``; the map
adds the flux linkages and inductances from the same potential. Cross terms between angle and current are not
sampled, so the map does not carry ``dpsi/dtheta``.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .core import Potential, Response

__all__ = ["poly_terms", "local_polynomial", "CoenergyMap", "machine_coenergy"]


def poly_terms(names: Sequence[str], caps: Mapping[str, int], total: int) -> list[tuple[int, ...]]:
    """Exponent tuples with ``exponent[v] <= caps[v]`` and total degree ``<= total``."""
    rng = [range(caps.get(v, total) + 1) for v in names]
    return [e for e in itertools.product(*rng) if sum(e) <= total]


def local_polynomial(samples: Mapping[str, np.ndarray], values: np.ndarray, center: Mapping[str, float],
                     scale: Mapping[str, float], terms: Sequence[tuple[int, ...]], *, name: str = "Phi",
                     unit: str = "", units: Mapping[str, str] | None = None, **kwargs) -> tuple[Potential, float]:
    """Fit ``Phi = sum_k c_k prod_v ((x_v - center_v) / scale_v)^e_kv`` to ``values`` at ``samples`` by least
    squares. Returns the potential and the RMS fit residual (zero when the samples determine the terms exactly)."""
    names = list(center)
    X = np.column_stack([(np.asarray(samples[v], float) - center[v]) / scale[v] for v in names])
    E = np.asarray(terms, int)
    B = np.prod(X[:, None, :] ** E[None, :, :], axis=2)
    y = np.asarray(values, float)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    rms = float(np.sqrt(np.mean((B @ c - y) ** 2)))
    ct = torch.tensor(c, dtype=torch.float64)
    cen = [float(center[v]) for v in names]
    sc = [float(scale[v]) for v in names]
    emax = E.max(axis=0)

    def fn(**x):
        # powers by repeated products (x ** 0 would give NaN derivatives at x = 0)
        pw = []
        for i, v in enumerate(names):
            u = (x[v] - cen[i]) / sc[i]
            p = [torch.ones_like(u)]
            for _ in range(int(emax[i])):
                p.append(p[-1] * u)
            pw.append(p)
        out = 0.0
        for k, e in enumerate(E):
            term = ct[k]
            for i, ei in enumerate(e):
                if ei:
                    term = term * pw[i][ei]
            out = out + term
        return out + torch.zeros_like(x[names[0]])

    units = dict(units or {})
    return Potential(fn, {v: units.get(v, "") for v in names}, name=name, unit=unit, **kwargs), rms


@dataclass
class CoenergyMap:
    """A local co-energy potential of a machine and its samples."""

    potential: Potential
    theta: float
    currents: tuple[float, ...]
    samples: dict[str, np.ndarray]
    values: np.ndarray
    fit_rms: float

    def at_operating_point(self) -> dict[str, np.ndarray]:
        """Torque, flux linkages and inductance matrix at the sampled operating point."""
        from .core import tree

        pt = {"theta": self.theta, **{f"i{k}": c for k, c in enumerate(self.currents)}}
        t = tree(self.potential, 2, at=pt, symbolic=False)
        n = len(self.currents)
        return {"torque": float(t["T"]), "psi": np.array([float(t[f"psi{k}"]) for k in range(n)]),
                "L": np.array([[float(t[(f"i{j}", f"i{k}")]) for k in range(n)] for j in range(n)])}


def machine_coenergy(machine, theta: float, currents: Sequence[float], *, n: int = 2, h: float | None = None,
                     di: float | None = None) -> CoenergyMap:
    """Sample ``machine.coenergy`` around ``(theta, currents)`` and fit the local co-energy potential
    ``W'(i0, i1, i2, theta)`` (J): polynomial of degree ``2 n`` in the angle, quadratic in the currents.

    ``h``: angle step (default: the machine's moving-band step, ``2 pi / n_theta`` or ``mm.band_step``);
    ``di``: current step (default 5 % of the largest current, at least 0.5 A)."""
    if h is None:
        h = getattr(getattr(machine, "mm", None), "band_step", None) or 2 * math.pi / machine.n_theta
    cur = np.asarray(currents, float)
    m = len(cur)
    di = di if di is not None else max(0.05 * float(np.abs(cur).max()), 0.5)
    pts = [(theta + k * h, cur) for k in range(-n, n + 1)]
    for a in range(m):
        for s in (1, -1):
            e = np.zeros(m)
            e[a] = s * di
            pts.append((theta, cur + e))
    for a, b in itertools.combinations(range(m), 2):
        e = np.zeros(m)
        e[a] = e[b] = di
        pts.append((theta, cur + e))
    W = np.array([machine.coenergy(t, tuple(c)) for t, c in pts])
    names = [f"i{k}" for k in range(m)] + ["theta"]
    samples = {f"i{k}": np.array([c[k] for _, c in pts]) for k in range(m)}
    samples["theta"] = np.array([t for t, _ in pts])
    center = {**{f"i{k}": cur[k] for k in range(m)}, "theta": theta}
    scale = {**{f"i{k}": di for k in range(m)}, "theta": h}
    caps = {**{f"i{k}": 2 for k in range(m)}, "theta": 2 * n}
    terms = [e for e in poly_terms(names, caps, 2 * n)
             if (e[-1] == 0 and sum(e[:-1]) <= 2) or sum(e[:-1]) == 0]
    resp = {f"L{j}{k}": Response(lambda d, x, j=j, k=k: d[(f"i{min(j, k)}", f"i{max(j, k)}")], 2, "H",
                                 f"differential inductance d2W'/di{j}di{k}")
            for j in range(m) for k in range(j, m)}
    pot, rms = local_polynomial(
        samples, W, center, scale, terms, name="W'", unit="J",
        units={**{f"i{k}": "A" for k in range(m)}, "theta": "rad"},
        names={**{(f"i{k}",): (f"psi{k}", 1, "Wb", f"flux linkage of phase {k}") for k in range(m)},
               ("theta",): ("T", 1, "N m", "torque at constant currents")},
        responses=resp,
        relation_names={(f"i{k}", "theta"): "electromechanical reciprocity" for k in range(m)})
    return CoenergyMap(pot, float(theta), tuple(float(c) for c in cur), samples, W, rms)
