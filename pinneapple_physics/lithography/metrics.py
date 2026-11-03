"""Resist and printed-pattern metrics: threshold resist with acid diffusion, CD, NILS, EPE, process window.

* **Resist**: the aerial image blurred by a Gaussian of length ``diffusion`` (nm, acid diffusion), then a constant
  threshold at the dose: a pixel clears (positive resist) where ``dose * I_blur >= threshold`` (Mack 2007, ch. 9:
  the threshold model is the simplest resist model; it ignores development kinetics).
* **CD** of a line along a cut: the width between the two threshold crossings, linearly interpolated.
* **NILS** at an edge: ``CD * |d ln I / dx|`` (normalized image log-slope).
* **EPE** of a 2-D pattern: the area where printed and target differ divided by the target's edge length, a mean
  edge-placement error in nm (pixel-exact for Manhattan patterns).
* **Process window**: the printed pattern over a focus x dose grid; the depth of focus is the focus range over
  which the EPE stays within a tolerance at the nominal dose, the exposure latitude the dose range at best focus.
"""
from __future__ import annotations

import numpy as np
import torch


def blur(img: torch.Tensor, L: float, diffusion: float) -> torch.Tensor:
    if diffusion <= 0:
        return img
    N = img.shape[0]
    f = torch.fft.fftfreq(N, d=L / N, dtype=torch.float64)
    FX, FY = torch.meshgrid(f, f, indexing="ij")
    G = torch.exp(-2 * (np.pi * diffusion) ** 2 * (FX ** 2 + FY ** 2))     # FT of a unit-area Gaussian
    return torch.fft.ifft2(torch.fft.fft2(img) * G).real


def printed(img: torch.Tensor, L: float, *, threshold: float = 0.3, dose: float = 1.0, diffusion: float = 0.0,
            tone: str = "dark") -> np.ndarray:
    """The printed pattern (bool): ``tone="dark"`` keeps resist where the dose is below threshold (dark features
    on a clear field print as resist lines); ``"clear"`` the opposite (holes, trenches)."""
    J = blur(img, L, diffusion).numpy() * dose
    return J < threshold if tone == "dark" else J >= threshold


def cd_along(img: np.ndarray, L: float, threshold: float, row: int | None = None, axis: int = 0) -> float:
    """Width (nm) of the dark region (below threshold) that contains the cut's minimum, on one line of pixels."""
    a = np.asarray(img, float)
    line = a[:, row if row is not None else a.shape[1] // 2] if axis == 0 else a[row if row is not None else
                                                                                a.shape[0] // 2]
    N = len(line)
    dx = L / N
    k = int(np.argmin(line))
    if line[k] >= threshold:
        return 0.0
    if (line < threshold).all():
        return float(L)

    def cross(step):
        i = k
        while line[i % N] < threshold:
            i += step
        j = i - step                                         # last inside
        a0, a1 = line[j % N], line[i % N]
        return j + step * (threshold - a0) / (a1 - a0)
    return float((cross(1) - cross(-1)) * dx)


def nils(img: np.ndarray, L: float, threshold: float, row: int | None = None) -> float:
    line = np.asarray(img, float)[:, row if row is not None else np.asarray(img).shape[1] // 2]
    N = len(line)
    dx = L / N
    cd = cd_along(img, L, threshold, row)
    slope = np.abs(np.gradient(np.log(np.maximum(line, 1e-12)), dx))
    edge = np.argmin(np.abs(line - threshold))
    return float(cd * slope[edge])


def edge_length(target: np.ndarray, L: float) -> float:
    t = np.asarray(target, bool)
    dx = L / t.shape[0]
    return float((np.sum(t != np.roll(t, 1, 0)) + np.sum(t != np.roll(t, 1, 1))) * dx)


def epe(pattern: np.ndarray, target: np.ndarray, L: float) -> float:
    """Mean edge-placement error (nm): mismatch area / target edge length."""
    dx = L / target.shape[0]
    return float(np.sum(np.asarray(pattern, bool) != np.asarray(target, bool)) * dx * dx / edge_length(target, L))


def process_window(image_at, target: np.ndarray, L: float, *, focus: list[float], doses: list[float],
                   threshold: float, diffusion: float, tone: str, epe_tol: float) -> dict:
    """EPE over focus x dose (``image_at(defocus) -> I``); DOF at the nominal dose and EL at best focus."""
    table = np.zeros((len(focus), len(doses)))
    for i, z in enumerate(focus):
        img = image_at(z)
        for j, d in enumerate(doses):
            table[i, j] = epe(printed(img, L, threshold=threshold, dose=d, diffusion=diffusion, tone=tone), target, L)
    jn = int(np.argmin(np.abs(np.array(doses) - 1.0)))
    ok_f = [z for z, e in zip(focus, table[:, jn], strict=True) if e <= epe_tol]
    ib = int(np.argmin(table[:, jn]))
    ok_d = [d for d, e in zip(doses, table[ib], strict=True) if e <= epe_tol]
    return {"focus": list(focus), "doses": list(doses), "epe": table.tolist(),
            "dof_nm": float(max(ok_f) - min(ok_f)) if len(ok_f) > 1 else 0.0,
            "exposure_latitude": float((max(ok_d) - min(ok_d)) / 1.0) if len(ok_d) > 1 else 0.0,
            "best_focus": float(focus[ib])}
