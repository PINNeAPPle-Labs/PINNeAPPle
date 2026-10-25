"""Effective-conductivity (Keff) homogenisation: the baseline every condensation must beat.

Each layer with inclusions (bumps, TSVs) becomes a homogeneous anisotropic layer:

* through the layer (the inclusions are straight prisms): the parallel (Voigt) rule ``k_z = phi k_i + (1-phi) k_m``,
  exact for straight columns in one-dimensional heat flow;
* in plane: the Maxwell Garnett / Hashin-Shtrikman lower bound for aligned cylinders (Hashin and Shtrikman,
  J. Appl. Phys. 33 (1962) 3125), ``k_xy = k_m ((k_i + k_m) + phi (k_i - k_m)) / ((k_i + k_m) - phi (k_i - k_m))``.

The homogenised package is solved on a coarse lateral grid (``cells`` per block side, default 1) with the same
vertical rows, contact resistances and cooling. Its junction temperatures are block averages by construction:
whatever happens inside a block (the cooler junction above a bump, a power hotspot smaller than the pitch, the share
of heat taken by each bump) is erased, and :func:`compare` measures by how much.
"""
from __future__ import annotations

import time

import numpy as np

from . import fv
from .package import BASE, MATERIALS, Package, Stack, Variant


def effective_layer(k_m: float, k_i: float, phi: float) -> tuple[float, float]:
    """(k_xy, k_z) of a layer with a volume fraction ``phi`` of aligned prisms of conductivity ``k_i``."""
    kz = phi * k_i + (1 - phi) * k_m
    kxy = k_m * ((k_i + k_m) + phi * (k_i - k_m)) / ((k_i + k_m) - phi * (k_i - k_m))
    return kxy, kz


def effective_properties(stack: Stack, variant: Variant = BASE) -> dict[str, np.ndarray]:
    """Per z row of one block: ``kxy``, ``kz``, ``rho_c`` (volume average), ``dz`` and contact ``R``."""
    p = stack.properties(variant)
    kxy, kz, rc = np.empty(stack.nz), np.empty(stack.nz), np.empty(stack.nz)
    incl_k = dict(variant.inclusion_k)
    for L, sl in zip(stack.layers, stack.layer_slices().values(), strict=False):
        km, cm = stack.material(L.matrix)
        phi = float(p["masks"][L.name].mean())
        if phi > 0:
            ki, ci = stack.material(L.inclusion)
            a, b = effective_layer(km, ki * incl_k.get(L.name, 1.0), phi)
            kxy[sl], kz[sl], rc[sl] = a, b, phi * ci + (1 - phi) * cm
        else:
            kxy[sl], kz[sl], rc[sl] = km, km, cm
    return {"kxy": kxy, "kz": kz, "rho_c": rc, "dz": p["dz"], "R": p["R"]}


def homogenized_system(pkg: Package, cells: int = 1, transient: bool = False):
    """The homogenised package on ``cells x cells`` lateral cells per block (``n`` must be divisible by it)."""
    st = pkg.stack
    nbx, nby = pkg.shape
    eff = [effective_properties(st, v) for v in pkg.variants]
    lay = np.repeat(np.repeat(pkg.layout, cells, 0), cells, 1)
    kxy = np.stack([e["kxy"] for e in eff])[lay]
    kz = np.stack([e["kz"] for e in eff])[lay]
    rc = np.stack([e["rho_c"] for e in eff])[lay]
    R = np.stack([e["R"] for e in eff])[lay]
    return fv.assemble((kxy, kxy, kz), st.dz(), R, st.pitch / cells, st.h_top, st.h_bottom,
                       rho_c=rc if transient else None)


def coarse_power(pkg: Package, cells: int = 1, power: np.ndarray | None = None) -> np.ndarray:
    p = pkg.power if power is None else power
    f = pkg.stack.n // cells
    nx, ny = p.shape
    return p.reshape(nx // f, f, ny // f, f).sum((1, 3))


def solve_homogenized(pkg: Package, cells: int = 1, power: np.ndarray | None = None):
    """Coarse temperature field ``(nbx * cells, nby * cells, nz)`` and the solve info."""
    t0 = time.perf_counter()
    sysm = homogenized_system(pkg, cells)
    f = np.zeros(sysm.shape)
    f[:, :, pkg.stack.active_k] = coarse_power(pkg, cells, power)
    T, info = fv.solve(sysm.K, f.ravel())
    info.seconds = time.perf_counter() - t0
    return T.reshape(sysm.shape), sysm, info


def homogenized_observables(pkg: Package, T: np.ndarray, sysm, cells: int = 1) -> dict[str, np.ndarray]:
    """Per block: mean junction temperature ``Tj_mean``, the die/TIM heat ``q_tim``, the heat down the microbump
    layer ``q_ubump_layer`` and the share a bump would take by the parallel rule (``q_ubump`` estimate)."""
    st = pkg.stack
    nbx, nby = pkg.shape
    sl = st.layer_slices()

    def per_block(a):
        return a.reshape(nbx, cells, nby, cells).sum((1, 3))
    tj = T[:, :, st.active_k].reshape(nbx, cells, nby, cells).mean((1, 3))
    f = sl["die"].stop - 1
    q_tim = per_block(sysm.Gz[:, :, f] * (T[:, :, f] - T[:, :, f + 1]))
    s = sl["ubump"]
    f = s.start + (s.stop - s.start) // 2 - 1
    q_layer = per_block(sysm.Gz[:, :, f] * (T[:, :, f] - T[:, :, f + 1]))
    L = [x for x in st.layers if x.name == "ubump"][0]
    km, ki = st.material(L.matrix)[0], st.material(L.inclusion)[0]
    phi = L.side ** 2
    share = phi * ki / (phi * ki + (1 - phi) * km)
    q_top = per_block(sysm.g_top * T[:, :, -1])
    q_bot = per_block(sysm.g_bottom * T[:, :, 0])
    return {"Tj_mean": tj, "Tmax": float(T[:, :, st.active_k].max()), "q_tim": q_tim, "q_ubump_layer": q_layer,
            "q_ubump": share * q_layer, "q_top": q_top, "q_bottom": q_bot}


def compare(pkg: Package, ref: dict[str, np.ndarray], hom: dict[str, np.ndarray]) -> dict[str, float]:
    """What homogenisation erases, against fine-scale reference observables (:func:`.substructure.
    reference_observables`): the peak junction temperature, block mean and block peak junction temperatures, the
    intra-block junction spread the coarse model cannot represent, and the interface heats."""
    tj = ref["Tj"]
    mean, peak = tj.mean(-1), tj.max(-1)
    out = {"Tmax_ref": float(tj.max()), "Tmax_hom": hom["Tmax"]}
    out["err_Tmax"] = hom["Tmax"] - out["Tmax_ref"]
    out["err_Tj_block_mean_max"] = float(np.abs(hom["Tj_mean"] - mean).max())
    out["err_Tj_block_peak_max"] = float(np.abs(hom["Tj_mean"] - peak).max())
    out["intra_block_spread_max"] = float((peak - tj.min(-1)).max())
    for name in ("q_tim", "q_ubump", "q_top", "q_bottom"):
        out[f"err_{name}"] = float(np.abs(hom[name] - ref[name]).sum() / max(np.abs(ref[name]).sum(), 1e-300))
    out["energy_balance"] = float((hom["q_top"].sum() + hom["q_bottom"].sum()) / pkg.power.sum() - 1)
    return out
