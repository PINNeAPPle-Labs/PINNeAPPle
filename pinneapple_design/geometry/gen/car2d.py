"""Parametric 2-D car side profile (an Ahmed-type road-vehicle body, no wheels) and its rasterisation.

The profile is built in units of the car length (x from the nose, 0..1; y from the road) from six design
parameters, then rounded with Chaikin corner cutting so the outline has no sharp corners except where the
parameters put them (base, slant edge).

    from pinneapple_design.geometry.gen.car2d import CarProfile
    car = CarProfile(slant_deg=25, windshield_deg=35, hood=0.62, clearance=0.08, diffuser_deg=4, nose=0.5)
    xy = car.outline()                       # (n, 2) closed polygon, car lengths
    solid = car.mask(nx=320, ny=112, length_cells=64, x_nose=77)   # (nx, ny) bool, for LBM / FNO inputs
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# design space used by the lab experiments (car_lbm / car_surrogate)
DESIGN_SPACE = {
    "slant_deg": (5.0, 40.0),        # rear-window / fastback slant from the roof
    "windshield_deg": (20.0, 55.0),  # windshield rake from the horizontal
    "hood": (0.45, 0.75),            # hood height as a fraction of the body height
    "clearance": (0.04, 0.12),       # ground clearance, car lengths
    "diffuser_deg": (0.0, 10.0),     # underbody diffuser angle at the rear
    "nose": (0.0, 1.0),              # nose rounding, 0 = blunt, 1 = round
}


@dataclass
class CarProfile:
    slant_deg: float = 25.0
    windshield_deg: float = 35.0
    hood: float = 0.62
    clearance: float = 0.08
    diffuser_deg: float = 4.0
    nose: float = 0.5
    height: float = 0.30             # body height above the clearance, car lengths
    slant_length: float = 0.28       # length of the rear slant, car lengths

    def params(self) -> dict[str, float]:
        return asdict(self)

    def outline(self, smooth: int = 4) -> np.ndarray:
        H, c = self.height, self.clearance
        top = c + H
        hh = c + self.hood * H
        xw0 = 0.28
        xw1 = min(0.6, xw0 + (top - hh) / math.tan(math.radians(self.windshield_deg)))
        xs0 = 1.0 - self.slant_length * math.cos(math.radians(self.slant_deg)) - 0.02
        ys1 = max(top - self.slant_length * math.sin(math.radians(self.slant_deg)), c + 0.35 * H)
        xd0 = 0.78
        yd1 = c + (1.0 - xd0) * math.tan(math.radians(self.diffuser_deg))
        rn = 0.04 + 0.10 * self.nose
        pts = [(0.0, c + 0.45 * H), (rn * 0.4, c + 0.05 * H), (rn, c), (xd0, c), (1.0, yd1), (1.0, ys1),
               (max(xs0, xw1 + 0.02), top), (xw1, top), (xw0, hh), (rn, hh), (rn * 0.25, c + 0.75 * H)]
        P = np.array(pts + [pts[0]], dtype=float)
        for _ in range(smooth):
            Q = 0.75 * P[:-1] + 0.25 * P[1:]
            R = 0.25 * P[:-1] + 0.75 * P[1:]
            P = np.empty((2 * len(Q) + 1, 2))
            P[0:-1:2], P[1:-1:2] = Q, R
            P[-1] = P[0]
        return P

    def frontal_height(self) -> float:
        P = self.outline()
        return float(P[:, 1].max() - P[:, 1].min())

    def mask(self, nx: int, ny: int, length_cells: float, x_nose: float) -> np.ndarray:
        """Cells (cell centres) inside the body on an (nx, ny) lattice with the road at y = 0."""
        from matplotlib.path import Path
        xs = (np.arange(nx) + 0.5 - x_nose) / length_cells
        ys = (np.arange(ny) + 0.5) / length_cells
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return Path(self.outline()).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(nx, ny)

    def sdf(self, nx: int, ny: int, length_cells: float, x_nose: float) -> np.ndarray:
        """Signed distance to the outline in cells (negative inside), from the rasterised mask."""
        from scipy.ndimage import distance_transform_edt
        m = self.mask(nx, ny, length_cells, x_nose)
        return (distance_transform_edt(~m) - distance_transform_edt(m)).astype(np.float32)


def sample_designs(n: int, seed: int = 0) -> list[dict[str, float]]:
    """Latin-hypercube designs over ``DESIGN_SPACE``."""
    rng = np.random.default_rng(seed)
    keys = list(DESIGN_SPACE)
    u = (np.argsort(rng.random((len(keys), n)), axis=1).T + rng.random((n, len(keys)))) / n
    return [{k: float(lo + (hi - lo) * u[i, j]) for j, (k, (lo, hi)) in enumerate(DESIGN_SPACE.items())}
            for i in range(n)]
