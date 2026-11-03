"""Scalar imaging of a periodic mask cell under partially coherent illumination (Abbe and Hopkins), in wafer units.

The mask is a transmission ``m(x, y)`` sampled on an ``N x N`` grid over a period ``L`` (nm). With the Fourier
coefficients ``M(f)`` of the cell, Abbe's method sums the coherent images of every source point ``s`` (a direction
of illumination, in units of ``NA / lambda`` as the partial-coherence factor sigma):

    I(x) = sum_s w_s | sum_f M(f) P(f + s NA/lambda) exp(2 pi i f.x) |^2 ,      sum_s w_s = 1

with the pupil ``P(f) = 1`` for ``|f| <= NA / lambda`` (zero outside) and the defocus phase
``exp(2 pi i z (sqrt(1/lambda^2 - |f|^2) - 1/lambda))`` (non-paraxial, in the medium of the image). A clear mask
images to ``I = 1``. Hopkins's form gathers the same sum into the transmission cross coefficient
``TCC(f1, f2) = sum_s w_s P(f1 + s) P*(f2 + s)``; its eigen-decomposition (sum of coherent systems, SOCS) keeps the
kernels that carry the image and is exact when all are kept.

Born & Wolf, *Principles of Optics*, 7th ed., sec. 10.5 (partially coherent imaging); Hopkins, *Proc. R. Soc. A* 217
(1953) 408; Mack, *Fundamental Principles of Optical Lithography* (2007), ch. 2-3; Cobb, PhD thesis, UC Berkeley
(1998) (SOCS). Scalar theory: polarization and the mask's 3-D topography are not modelled; at NA above about 0.7
the vector (high-NA) effects reduce the image contrast and are the next term to add.

Everything is a ``torch`` float64 computation, so the image is differentiable in the mask (inverse lithography).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Optics:
    wavelength: float = 193.0        # nm (ArF)
    NA: float = 1.35                 # numerical aperture (water immersion)
    defocus: float = 0.0             # nm, at the wafer
    medium_index: float = 1.0        # refractive index where the defocus is measured (1.44 water, image side)


@dataclass(frozen=True)
class Source:
    """Illumination: ``shape`` "conventional" (disk of ``sigma_out``), "annular" (``sigma_in``..``sigma_out``),
    "dipole_x" / "dipole_y" (two poles of radius ``pole`` at ``center``), or "coherent" (one on-axis point);
    sampled on a square grid of ``n`` points per sigma radius."""
    shape: str = "conventional"
    sigma_out: float = 0.7
    sigma_in: float = 0.0
    center: float = 0.7
    pole: float = 0.2
    n: int = 9

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """``(s, w)``: source points (k, 2) in sigma units and their weights (sum 1)."""
        if self.shape == "coherent":
            return np.zeros((1, 2)), np.ones(1)
        g = np.linspace(-1.0, 1.0, 2 * self.n + 1)
        X, Y = np.meshgrid(g, g, indexing="ij")
        r = np.hypot(X, Y)
        if self.shape == "conventional":
            keep = r <= self.sigma_out + 1e-12
        elif self.shape == "annular":
            keep = (r <= self.sigma_out + 1e-12) & (r >= self.sigma_in - 1e-12)
        elif self.shape in ("dipole_x", "dipole_y"):
            cx, cy = (self.center, 0.0) if self.shape == "dipole_x" else (0.0, self.center)
            keep = (np.hypot(X - cx, Y - cy) <= self.pole + 1e-12) | (np.hypot(X + cx, Y + cy) <= self.pole + 1e-12)
        else:
            raise ValueError(f"unknown source shape {self.shape!r}")
        s = np.stack([X[keep], Y[keep]], 1)
        if not len(s):
            raise ValueError("the source has no point: refine n")
        return s, np.full(len(s), 1.0 / len(s))


def frequencies(N: int, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    f = torch.fft.fftfreq(N, d=L / N, dtype=torch.float64)          # cycles / nm
    return torch.meshgrid(f, f, indexing="ij")


def pupil(FX: torch.Tensor, FY: torch.Tensor, optics: Optics) -> torch.Tensor:
    cut = optics.NA / optics.wavelength
    f2 = FX ** 2 + FY ** 2
    inside = (f2 <= cut ** 2 * (1 + 1e-12)).to(torch.complex128)
    if optics.defocus == 0:
        return inside
    k = optics.medium_index / optics.wavelength
    kz = torch.sqrt(torch.clamp(k ** 2 - f2, min=0.0))
    return inside * torch.exp(2j * math.pi * optics.defocus * (kz - k))


def _spectrum(mask: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(mask.to(torch.complex128)) / mask.numel()


def aerial_image(mask, L: float, optics: Optics | None = None, source: Source | None = None) -> torch.Tensor:
    """Abbe image of the mask cell (``(N, N)`` array or tensor of transmissions, complex allowed), clear = 1."""
    optics, source = optics or Optics(), source or Source()
    m = torch.as_tensor(mask)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("the mask is a square (N, N) cell")
    N = m.shape[0]
    M = _spectrum(m)
    FX, FY = frequencies(N, L)
    s, w = source.points()
    a = optics.NA / optics.wavelength
    img = torch.zeros((N, N), dtype=torch.float64)
    for (sx, sy), wk in zip(s, w, strict=True):
        P = pupil(FX + sx * a, FY + sy * a, optics)
        E = torch.fft.ifft2(M * P) * N * N
        img = img + float(wk) * (E.real ** 2 + E.imag ** 2)
    return img


def socs_kernels(N: int, L: float, optics: Optics | None = None, source: Source | None = None, *, energy: float = 1.0,
                 max_kernels: int | None = None) -> dict:
    """Hopkins TCC over the frequencies that can reach the pupil, eigen-decomposed (SOCS): ``{"kernels" (k, N, N)
    complex in frequency, "weights" (k,), "kept_energy", "n_freq"}``. ``energy=1`` keeps every non-zero kernel
    (exact); less trades accuracy for speed."""
    optics, source = optics or Optics(), source or Source()
    FX, FY = frequencies(N, L)
    s, w = source.points()
    a = optics.NA / optics.wavelength
    smax = float(np.hypot(s[:, 0], s[:, 1]).max())
    sel = (FX ** 2 + FY ** 2) <= (a * (1 + smax)) ** 2 * (1 + 1e-9)
    idx = torch.nonzero(sel.flatten()).flatten()
    fx, fy = FX.flatten()[idx], FY.flatten()[idx]
    A = torch.stack([pupil(fx + sx * a, fy + sy * a, optics) * math.sqrt(wk) for (sx, sy), wk in
                     zip(s, w, strict=True)], 0)                         # (sources, freq)
    tcc = A.T @ A.conj()                                                 # (freq, freq), Hermitian PSD
    vals, vecs = torch.linalg.eigh(tcc)
    order = torch.argsort(vals, descending=True)
    vals, vecs = vals[order].clamp(min=0.0), vecs[:, order]
    total = float(vals.sum())
    cum = torch.cumsum(vals, 0) / max(total, 1e-300)
    k = int((cum < energy - 1e-12).sum()) + 1 if energy < 1 else int((vals > 1e-12 * vals[0]).sum())
    if max_kernels:
        k = min(k, int(max_kernels))
    ker = torch.zeros((k, N * N), dtype=torch.complex128)
    ker[:, idx] = vecs[:, :k].T
    return {"kernels": ker.reshape(k, N, N), "weights": vals[:k], "kept_energy": float(cum[k - 1]),
            "n_freq": int(len(idx))}


def aerial_image_socs(mask, socs: dict) -> torch.Tensor:
    """The image from the SOCS kernels (Hopkins): ``sum_k lambda_k |IFFT(M phi_k)|^2``."""
    m = torch.as_tensor(mask)
    N = m.shape[0]
    M = _spectrum(m)
    E = torch.fft.ifft2(M[None] * socs["kernels"], dim=(-2, -1)) * N * N
    return (socs["weights"][:, None, None] * (E.real ** 2 + E.imag ** 2)).sum(0)


def resolution_limit(optics: Optics, sigma: float) -> float:
    """Smallest pitch with any modulation, ``lambda / (NA (1 + sigma))`` (nm): below it no first order reaches
    the pupil from any source point."""
    return optics.wavelength / (optics.NA * (1 + sigma))
