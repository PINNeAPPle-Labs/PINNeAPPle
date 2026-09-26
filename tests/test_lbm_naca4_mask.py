"""naca4_mask: cambered NACA 4-digit geometry with the aerodynamic angle-of-attack convention."""
import math

import numpy as np

from pinneapple_simulation.numerical_solvers.lbm import airfoil_naca_mask, naca4_mask


def test_naca4412_thickness_camber_and_chord():
    c = 200
    m = naca4_mask(400, 200, "4412", c, 0.0, x_le=100, y_le=100).numpy()
    cols = np.nonzero(m.any(1))[0]
    assert cols.min() == 100 and abs(cols.max() - 300) <= 1
    assert abs(max(m[i].sum() for i in range(100, 301)) / c - 0.12) < 0.01  # max thickness 12 %
    ys = np.nonzero(m[100 + int(0.4 * c)])[0]
    assert abs(((ys.min() + ys.max()) / 2 - 100) / c - 0.04) < 0.005  # max camber 4 % at 40 % chord


def test_positive_aoa_lifts_the_nose():
    c, aoa = 200, 15.0
    m = naca4_mask(400, 200, "0012", c, aoa, x_le=100, y_le=100).numpy()
    te = np.nonzero(m.any(1))[0].max()
    y_te = np.nonzero(m[te])[0].mean() - 100
    assert abs(y_te + c * math.sin(math.radians(aoa))) < 2  # trailing edge below the leading edge


def test_symmetric_section_matches_the_legacy_mask_at_zero_aoa():
    new = naca4_mask(200, 100, "0012", 80, 0.0, x_le=50, y_le=50).numpy()
    old = airfoil_naca_mask(200, 100, 80, 0.0, "0012").numpy()
    assert abs(int(new.sum()) - int(old.sum())) <= 0.05 * old.sum()
